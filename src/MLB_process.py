#!/usr/bin/env python
# coding: utf-8

# Imports
import os
import json
import time
import logging
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from langchain.docstore.document import Document
from openai import AzureOpenAI
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")

# Function definitions
def document_layout_analysis(document_path, di_endpoint=DOC_INTELLIGENCE_ENDPOINT, di_key=DOC_INTELLIGENCE_KEY):
    """Analyze document layout using Azure Document Intelligence."""
    logger.info(f"Starting document analysis for {document_path}")
    start_time = time.time()
    doc_intelligence_client = DocumentIntelligenceClient(
        endpoint=di_endpoint,
        credential=AzureKeyCredential(di_key)
    )
    features = [DocumentAnalysisFeature.STYLE_FONT] if document_path.endswith('.pdf') else []
    
    with open(document_path, "rb") as f:
        poller = doc_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            f,
            content_type="application/octet-stream",
            output_content_format="markdown",
            features=features
        )
    result = poller.result()
    end_time = time.time()
    logger.info(f"Document analysis completed in {end_time - start_time:.2f} seconds")
    return result

def apply_tags_to_content(content, styles):
    """Apply bold tags to content based on style information."""
    logger.info("Applying style tags to content")
    start_time = time.time()
    tagged_content = content
    for style in styles:
        for span in style.get('spans', []):
            offset, length = span['offset'], span['length']
            if style.get('fontWeight') == 'bold':
                tagged_content = (
                    tagged_content[:offset] +
                    f"<b>{tagged_content[offset:offset + length]}</b>" +
                    tagged_content[offset + length:]
                )
    end_time = time.time()
    logger.info(f"Style tags applied in {end_time - start_time:.2f} seconds")
    return tagged_content

def semantic_chunking(document, file_name):
    """Chunk document content based on 'Service Location X of Y' headers."""
    logger.info("Starting semantic chunking")
    start_time = time.time()
    content = apply_tags_to_content(document['content'], document.get('styles', [])) if 'styles' in document else document['content']
    # Find all service location headers
    headers = list(re.finditer(r'Service Location \d+ of \d+', content))
    chunks = []
    if headers:
        # First chunk: content before the first header
        if headers[0].start() > 0:
            chunks.append(Document(page_content=content[:headers[0].start()].strip(), metadata={"source": f"{file_name}.md"}))
        # Chunks for each service location
        for i in range(len(headers)):
            start = headers[i].start()
            end = headers[i+1].start() if i+1 < len(headers) else len(content)
            chunk_content = content[start:end].strip()
            chunks.append(Document(page_content=chunk_content, metadata={"source": f"{file_name}.md"}))
    else:
        # If no headers found, use the entire content
        chunks.append(Document(page_content=content.strip(), metadata={"source": f"{file_name}.md"}))
    end_time = time.time()
    logger.info(f"Semantic chunking completed in {end_time - start_time:.2f} seconds. Number of chunks: {len(chunks)}")
    return chunks

def get_aoai_response(query, force_json=False, prompt_file="prompt.json"):
    """Get response from Azure OpenAI model."""
    logger.info("Requesting response from Azure OpenAI")
    start_time = time.time()
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    
    with open(prompt_file, "r", encoding="utf-8") as file:
        prompt_base = file.read().strip()
    
    prompt = prompt_base + "\n\nDocument chunk: " + query
    
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"} if force_json else None
    )
    end_time = time.time()
    logger.info(f"Azure OpenAI response received in {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content

def clean_llm_response(response, force_json=False):
    """Clean the LLM response, extracting JSON if not forced."""
    if force_json:
        return response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    return json_match.group(0) if json_match else response

def get_llm_response(query, force_json=False, prompt_file="prompt.json"):
    """Wrapper function to get and clean LLM response."""
    response = get_aoai_response(query, force_json, prompt_file)
    return clean_llm_response(response, force_json)

def main():
    # Create necessary directories
    os.makedirs(os.path.join("data", "documents"), exist_ok=True)
    
    # Main processing
    document_name = "test_bill_3.pdf"
    document_path = os.path.join("data", "documents", document_name)
    logger.info(f"Processing document: {document_name}")
    results = document_layout_analysis(document_path)
    chunks = semantic_chunking(results, document_name)

    prompt_file = "telecom_prompt.txt"
    force_json = True

    master_account = {}
    sub_accounts = []
    for i, chunk in enumerate(tqdm(chunks)):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        content = chunk.page_content
        response = get_llm_response(content, force_json, prompt_file)
        data = json.loads(response)
        # Take the first master_account with an account_number
        if data.get("master_account", {}).get("account_number") and not master_account:
            master_account = data["master_account"]
        # Collect all sub-accounts
        if "sub_accounts" in data:
            sub_accounts.extend(data["sub_accounts"])

    # Add total_due to each sub-account if not present
    logger.info("Calculating total_due for sub-accounts where necessary")
    for sub_account in sub_accounts:
        if 'total_due' not in sub_account or not sub_account['total_due'].strip():
            if 'line_items' in sub_account and sub_account['line_items']:
                total_due = sum(
                    float(line_item['total'].replace('$', '').replace(',', '')) 
                    for line_item in sub_account['line_items'] 
                    if line_item.get('total') and line_item['total'].strip()
                )
                sub_account['total_due'] = f"${total_due:.2f}"
            else:
                sub_account['total_due'] = "$0.00"

    output = {"master_account": master_account, "sub_accounts": sub_accounts}

    # Save output to JSON file
    output_file = os.path.join("data", "telecom_output.json")
    logger.info(f"Saving output to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print a summary to the terminal
    logger.info("Processing complete. Output saved to telecom_output.json")
    print(f"Master account: {master_account}")
    print(f"Number of sub-accounts: {len(sub_accounts)}")

if __name__ == "__main__":
    main() 