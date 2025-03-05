#!/usr/bin/env python
# coding: utf-8

"""
Utilities for Retrieval-Augmented Generation (RAG) with Azure AI Search and OpenAI.
"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.docstore.document import Document
from openai import AzureOpenAI
import os
import logging
import json
import re
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
logger.info(f"Loaded environment variables in rag_utils: VECTOR_SEARCH_KEY={os.getenv('VECTOR_SEARCH_KEY')[:8]}...")

def split_into_pages(text: str) -> List[Tuple[str, str]]:
    """
    Splits document content into pages based on markers like 'Page X'.

    Args:
        text (str): Raw document content.

    Returns:
        List[Tuple[str, str]]: List of tuples containing (page_number, page_content).
    """
    page_pattern = r'\nPage (\d+)\n'
    parts = re.split(page_pattern, text)
    pages = []
    for i in range(1, len(parts), 2):
        page_num = parts[i]
        page_content = parts[i+1].strip()
        pages.append((page_num, page_content))
    if not pages:
        pages.append(('1', text.strip()))  # Default to one page if no markers
    return pages

def create_extracted_chunks(extracted_data: Dict) -> List[str]:
    """
    Creates context-rich chunks from extracted structured data.

    Args:
        extracted_data (Dict): Structured data extracted from the document.

    Returns:
        List[str]: List of chunk strings for master account and sub-accounts.
    """
    chunks = []
    if 'master_account' in extracted_data:
        master = extracted_data['master_account']
        master_chunk = (
            f"Master Account {master.get('account_number', 'N/A')}: "
            f"Total Due: ${master.get('total_due', 'N/A')}, "
            f"Due Date: {master.get('due_date', 'N/A')}, "
            f"Vendor: {master.get('vendor_name', 'N/A')}"
        )
        chunks.append(master_chunk)

    if 'sub_accounts' in extracted_data:
        for sub in extracted_data['sub_accounts']:
            sub_chunk = (
                f"Sub-Account {sub.get('sub_account_number', 'N/A')}: "
                f"Location: {sub.get('location', 'N/A')}, "
                f"Total Due: ${sub.get('total_due', 'N/A')}, "
                f"Line Items: "
            )
            line_items = sub.get('line_items', [])
            if line_items:
                sub_chunk += ", ".join(
                    f"{item.get('description', 'N/A')} "
                    f"({item.get('date_range', 'N/A')}, ${item.get('total', 'N/A')})"
                    for item in line_items
                )
            else:
                sub_chunk += "None"
            chunks.append(sub_chunk)
    return chunks

def chunk_and_index_document(
    document_content: str,
    analysis_result_layout: Optional[Dict] = None,
    extracted_data: Optional[Dict] = None,
    document_name: str = "bill_document"
) -> Optional[AzureSearch]:
    """
    Chunks document content using extracted data and page-based splitting, then indexes it in Azure AI Search.

    Args:
        document_content (str): Raw document text.
        analysis_result_layout (Optional[Dict]): Layout analysis result (currently unused).
        extracted_data (Optional[Dict]): Extracted structured data.
        document_name (str): Document name for metadata (default: "bill_document").

    Returns:
        Optional[AzureSearch]: Vector store with indexed chunks, or None if no chunks are created.
    """
    logger.info(f"Starting chunk_and_index_document for {document_name}")

    chunks = []

    # Add chunks from extracted data
    if extracted_data:
        logger.info("Creating chunks from extracted data")
        extracted_chunks = create_extracted_chunks(extracted_data)
        chunks.extend(extracted_chunks)

    # Add page-based chunks from raw content
    logger.info("Creating page-based chunks from document content")
    pages = split_into_pages(document_content)
    for page_num, content in pages:
        page_chunk = f"Page {page_num}: {content}"
        chunks.append(page_chunk)

    chunked_docs = [
        Document(page_content=chunk, metadata={"source": document_name, "chunk_id": i})
        for i, chunk in enumerate(chunks) if chunk.strip()
    ]
    logger.info(f"Created {len(chunked_docs)} chunks")

    if not chunked_docs:
        logger.warning("No valid chunks; returning None")
        return None

    logger.info("Initializing AzureOpenAIEmbeddings")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-12-01-preview"
    )

    logger.info(f"Setting up AzureSearch with endpoint={os.getenv('VECTOR_SEARCH_ENDPOINT')}")
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("VECTOR_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("VECTOR_SEARCH_KEY"),
        index_name="bill-document-index",
        embedding_function=embeddings.embed_query
    )

    logger.info(f"Adding {len(chunked_docs)} documents to vector store")
    try:
        vector_store.add_documents(documents=chunked_docs)
        logger.info("Documents added successfully")
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise

    return vector_store

def rag_query(query: str, vector_store: AzureSearch, top_k: int = 15) -> Dict:
    """
    Queries the vector store and generates an answer using OpenAI.

    Args:
        query (str): User's question.
        vector_store (AzureSearch): Indexed document chunks.
        top_k (int): Number of chunks to retrieve (default: 15).

    Returns:
        Dict: A dictionary with keys "found" (bool) and "value" (str or None).
    """
    logger.info(f"Executing rag_query: '{query}' with top_k={top_k}")

    # Check if vector store is valid
    if not vector_store:
        logger.warning("Vector store is None")
        return {"found": False, "value": None}

    # Perform similarity search
    logger.info("Performing similarity search")
    try:
        results = vector_store.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in results])
        logger.info(f"Retrieved {len(results)} results. Context preview: {context[:200]}...")
    except Exception as e:
        logger.error(f"Similarity search error: {str(e)}")
        return {"found": False, "value": None}

    # Construct prompt with explicit JSON instructions
    prompt = (
        f"Using the following document excerpts, answer the question: {query}\n"
        f"If multiple chunks are relevant, analyze them together to provide a complete answer.\n"
        f"Return only the following JSON object with double quotes and no additional text: "
        f'{{"found": true, "value": "your_answer"}} if the answer is found, or '
        f'{{"found": false, "value": null}} if not.\n\n'
        f"Excerpts:\n{context}"
    )

    # Initialize Azure OpenAI client
    logger.info("Initializing AzureOpenAI client")
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    # Request chat completion
    logger.info("Requesting chat completion")
    try:
        response = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_response = response.choices[0].message.content.strip()
        logger.info(f"Raw response: {raw_response}")

        # Clean up code block markers if present
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        
        # Parse the response as JSON
        result = json.loads(raw_response)
        logger.info(f"Query result: {result}")
        return result
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return {"found": False, "value": None}
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        return {"found": False, "value": None}