#!/usr/bin/env python
# coding: utf-8

"""
Module for processing Multi-Location Bills (MLB).
"""

import os
import json
import logging
import time
import re
from typing import Dict, Any, List, Optional
from langchain.docstore.document import Document
from openai import AzureOpenAI
from tqdm import tqdm
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_tags_to_content(content: str, styles: List[Dict[str, Any]]) -> str:
    """
    Apply bold tags to content based on style information.

    Args:
        content: The document content.
        styles: List of style information.

    Returns:
        Content with bold tags applied.
    """
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

def semantic_chunking(document: Dict[str, Any], file_name: str) -> List[Document]:
    """
    Chunk document content dynamically based on potential headers or content clustering,
    excluding master account summary sections.

    Args:
        document: The document analysis result.
        file_name: The name of the document file.

    Returns:
        List of document chunks.
    """
    logger.info("Starting dynamic chunking")
    start_time = time.time()

    content = apply_tags_to_content(document['content'], document.get('styles', [])) if 'styles' in document else document['content']

    # Define potential header patterns for different bill formats
    header_patterns = [
        r'Service Location \d+ of \d+',  # Spectrum format
        r'Location:',                    # Comcast format
        r'Site \d+',                     # Other potential format
        r'Account \d+',                  # Another potential format
        r'Location Summary',             # For summary tables
    ]
    header_regex = re.compile('|'.join(header_patterns))

    # Define pattern for master summary lines (e.g., "Subtotal | ... | $754.18")
    master_summary_pattern = r'^\| Subtotal \|.*\|\s*[\$]?[-]?\d+\.\d{2}\s*\|.*\|\s*[\$]?[-]?\d+\.\d{2}\s*\|.*$'

    headers = list(header_regex.finditer(content))
    chunks = []

    if headers:
        if headers[0].start() > 0:
            chunks.append(Document(page_content=content[:headers[0].start()].strip(), metadata={"source": f"{file_name}.md"}))
        for i in range(len(headers)):
            start = headers[i].start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            chunk_content = content[start:end].strip()

            # Truncate chunk before master summary
            summary_match = re.search(master_summary_pattern, chunk_content, re.MULTILINE)
            if summary_match:
                chunk_content = chunk_content[:summary_match.start()].strip()

            chunks.append(Document(page_content=chunk_content, metadata={"source": f"{file_name}.md"}))
    else:
        logger.warning("No headers found, falling back to content clustering")
        potential_chunks = re.split(r'\n\n', content)
        for chunk in potential_chunks:
            if chunk.strip():
                chunks.append(Document(page_content=chunk.strip(), metadata={"source": f"{file_name}.md"}))

    end_time = time.time()
    logger.info(f"Dynamic chunking completed in {end_time - start_time:.2f} seconds. Number of chunks: {len(chunks)}")
    return chunks

def preprocess_chunk(content: str) -> str:
    """
    Preprocess a document chunk to enhance sub-account number identification.

    Args:
        content: The document chunk content.

    Returns:
        Preprocessed content with enhanced sub-account numbers.
    """
    enhanced_content = content

    # Bold 9-digit sub-account numbers (common in telecom bills)
    account_number_pattern = r'\b\d{9}\b'
    matches = re.finditer(account_number_pattern, enhanced_content)
    for match in matches:
        account_num = match.group(0)
        enhanced_content = enhanced_content.replace(account_num, f"<b>{account_num}</b>")

    # Bold labeled account numbers (e.g., "Account #: 12345")
    account_patterns = [
        r'(?i)(account\s*(?:#|number|no\.?)\s*[:\s]?\s*)([a-zA-Z0-9\s\-]+)'
    ]
    for pattern in account_patterns:
        matches = re.finditer(pattern, enhanced_content)
        for match in matches:
            prefix = match.group(1)
            account_num = match.group(2).strip()
            if (re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', account_num) or    # Skip dates
                re.match(r'\d{5}(?:-\d{4})?$', account_num) or          # Skip ZIP codes
                re.match(r'\d{3}-\d{3}-\d{4}', account_num)):           # Skip phone numbers
                continue
            enhanced_content = enhanced_content.replace(match.group(0), f"{prefix}<b>{account_num}</b>")

    return enhanced_content

def get_llm_response(
    query: str,
    prompt_file: str = "telecom_prompt.txt",
    force_json: bool = True,
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> str:
    """
    Get response from Azure OpenAI model with a generalized prompt, enhanced to focus
    on sub-account-specific totals.

    Args:
        query: The query to send to the model.
        prompt_file: Path to the prompt file.
        force_json: Whether to force JSON output.
        openai_endpoint: Azure OpenAI endpoint.
        openai_api_key: Azure OpenAI API key.
        deployment_name: Azure OpenAI deployment name.

    Returns:
        The model's response.
    """
    try:
        logger.info("Requesting response from Azure OpenAI")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        openai_endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        openai_api_key = openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        deployment_name = deployment_name or os.getenv("DEPLOYMENT_NAME")

        if not all([openai_endpoint, openai_api_key, deployment_name]):
            raise ValueError("Missing required OpenAI credentials")

        client = AzureOpenAI(
            azure_endpoint=openai_endpoint,
            api_key=openai_api_key,
            api_version="2024-02-01"
        )

        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_base = file.read().strip()

        # Append instructions for precise sub-account total extraction
        prompt_base += (
            "\n\nExtract the total due for the sub-account from its specific \"Subtotal\" line within the chunk. "
            "This is typically a single line showing the total for that sub-account, not the master account's total. "
            "Ignore any lines that appear to be summaries for the entire bill, such as \"CURRENT CHARGES SUBTOTAL\" or \"BALANCE DUE\"."
        )

        prompt = f"{prompt_base}\n\nDocument chunk: {query}"
        start_time = time.time()

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"} if force_json else None
        )

        end_time = time.time()
        logger.info(f"Azure OpenAI response received in {end_time - start_time:.2f} seconds")
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        raise

def extract_fallback_sub_account(chunk_content: str) -> str:
    """
    Fallback to extract the first 9-digit number in the chunk.

    Args:
        chunk_content: The document chunk content.

    Returns:
        The first 9-digit number found, or "Unknown".
    """
    account_number_pattern = r'\b\d{9}\b'
    match = re.search(account_number_pattern, chunk_content)
    return match.group(0) if match else "Unknown"

def serialize_for_logging(obj):
    """
    Recursively convert date objects to strings for JSON serialization in logging.
    """
    if isinstance(obj, dict):
        return {k: serialize_for_logging(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_logging(item) for item in obj]
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()  # Converts date to ISO 8601 string (e.g., "2024-10-01")
    else:
        return obj

def process_mlb(
    analysis_result: Dict[str, Any],
    master_account: Dict[str, Any],
    document_name: str,
    prompt_file: str = "telecom_prompt.txt",
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a Multi-Location Bill (MLB) using OpenAI, with validation for sub-account totals.

    Args:
        analysis_result: The result from document analysis (expects prebuilt-layout model).
        master_account: Pre-extracted master account data from prebuilt-invoice analysis.
        document_name: The name of the document file.
        prompt_file: Path to the MLB prompt file.
        openai_endpoint: Azure OpenAI endpoint.
        openai_api_key: Azure OpenAI API key.
        deployment_name: Azure OpenAI deployment name.

    Returns:
        Dict with extracted MLB data.
    """
    try:
        logger.info(f"Processing Multi-Location Bill (MLB): {document_name}")

        # Serialize master_account for logging purposes only
        logger.info(f"Received master account data: {json.dumps(serialize_for_logging(master_account), indent=2)}")

        # Chunk the document dynamically for sub-accounts
        chunks = semantic_chunking(analysis_result, document_name)
        sub_accounts = []

        for i, chunk in enumerate(tqdm(chunks)):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            content = chunk.page_content
            enhanced_content = preprocess_chunk(content)

            # Debugging: Save preprocessed chunk
            debug_dir = "data/debug"
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, f"{document_name}_chunk_{i+1}.txt"), "w") as f:
                f.write(enhanced_content)

            response = get_llm_response(
                enhanced_content,
                prompt_file=prompt_file,
                force_json=True,
                openai_endpoint=openai_endpoint,
                openai_api_key=openai_api_key,
                deployment_name=deployment_name
            )

            data = json.loads(response)

            if "sub_accounts" in data:
                for sub_account in data["sub_accounts"]:
                    if not sub_account.get('sub_account_number') or sub_account['sub_account_number'] == "Unknown":
                        sub_account['sub_account_number'] = extract_fallback_sub_account(enhanced_content)
                sub_accounts.extend(data["sub_accounts"])

        # Validate and correct sub-account totals
        logger.info("Validating and correcting sub-account totals")
        for sub_account in sub_accounts:
            if 'line_items' in sub_account and sub_account['line_items']:
                calculated_total = 0.0
                for line_item in sub_account['line_items']:
                    if 'total' in line_item:
                        try:
                            charge = float(line_item['total'].replace('$', '').replace(',', ''))
                            calculated_total += charge
                        except ValueError:
                            logger.warning(f"Invalid total value in line item: {line_item['total']}")

                if 'total_due' not in sub_account or not sub_account['total_due'].strip():
                    sub_account['total_due'] = f"${calculated_total:.2f}"
                else:
                    try:
                        extracted_total = float(sub_account['total_due'].replace('$', '').replace(',', ''))
                        if abs(calculated_total - extracted_total) > 0.01:
                            logger.warning(
                                f"Total due mismatch for sub-account {sub_account.get('sub_account_number', 'Unknown')}: "
                                f"Extracted ${extracted_total}, but line items sum to ${calculated_total}"
                            )
                            sub_account['total_due'] = f"${calculated_total:.2f}"
                    except ValueError:
                        logger.warning(f"Invalid total_due value: {sub_account['total_due']}")
                        sub_account['total_due'] = f"${calculated_total:.2f}"
            elif 'total_due' not in sub_account or not sub_account['total_due'].strip():
                sub_account['total_due'] = "$0.00"

        output = {
            "master_account": master_account,
            "sub_accounts": sub_accounts
        }

        logger.info(f"MLB processing complete. Master account: {master_account.get('account_number', 'N/A')}")
        logger.info(f"Number of sub-accounts: {len(sub_accounts)}")
        return output

    except Exception as e:
        logger.error(f"Error processing MLB: {str(e)}")
        raise

if __name__ == "__main__":
    from src.analyze import analyze_document
    from dotenv import load_dotenv
    load_dotenv()

    document_path = os.path.join("data", "documents", "test_bill_3.pdf")
    try:
        raw_result, analysis_result = analyze_document(document_path, model="prebuilt-layout")
        document_name = os.path.basename(document_path)
        # Provide a mock master_account for testing
        master_account = {
            "account_number": "123456789",
            "total_due": "$100.00",
            "due_date": "2023-06-15",
            "vendor_name": "Sample Vendor"
        }
        mlb_data = process_mlb(analysis_result, master_account, document_name)
        output_file = os.path.join("data", "output", f"{os.path.splitext(document_name)[0]}_output.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(mlb_data, f, indent=2)
        print(f"MLB data saved to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")