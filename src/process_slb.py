#!/usr/bin/env python
# coding: utf-8

"""
Module for processing Single-Location Bills (SLB).
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_slb(
    analysis_result: Dict[str, Any],
    prompt_file: str = "prompts/slb_prompt.txt",
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a Single-Location Bill (SLB) using OpenAI.
    
    Args:
        analysis_result: The result from document analysis.
        prompt_file: Path to the SLB prompt file.
        openai_endpoint: Azure OpenAI endpoint. If None, uses environment variable.
        openai_api_key: Azure OpenAI API key. If None, uses environment variable.
        deployment_name: Azure OpenAI deployment name. If None, uses environment variable.
        
    Returns:
        Dict with extracted SLB data.
        
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If OpenAI credentials are missing.
        Exception: For other errors during processing.
    """
    try:
        logger.info("Processing Single-Location Bill (SLB)")
        
        # Check if prompt file exists
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        # Use environment variables if not provided
        if openai_endpoint is None:
            openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not openai_endpoint:
                raise ValueError("OpenAI endpoint not provided and not found in environment variables")
        
        if openai_api_key is None:
            openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        if deployment_name is None:
            deployment_name = os.getenv("DEPLOYMENT_NAME")
            if not deployment_name:
                raise ValueError("OpenAI deployment name not provided and not found in environment variables")
        
        # Initialize the OpenAI client
        client = AzureOpenAI(
            azure_endpoint=openai_endpoint,
            api_key=openai_api_key,
            api_version="2024-02-01"
        )
        
        # Read the SLB prompt
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_base = file.read().strip()
        
        # Get the document content
        content = analysis_result.get('content', '')
        
        # Prepare the prompt with document content
        prompt = f"{prompt_base}\n\nDocument content:\n{content}"
        
        # Request processing from OpenAI
        logger.info("Requesting processing from Azure OpenAI")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        end_time = time.time()
        logger.info(f"Processing response received in {end_time - start_time:.2f} seconds")
        
        # Parse the response
        extracted_data = json.loads(response.choices[0].message.content)
        
        # Log extraction results
        account = extracted_data.get('account', {})
        line_items = extracted_data.get('line_items', [])
        logger.info(f"Extracted account number: {account.get('account_number', 'N/A')}")
        logger.info(f"Extracted invoice date: {account.get('invoice_date', 'N/A')}")
        logger.info(f"Extracted total due: {account.get('total_due', 'N/A')}")
        logger.info(f"Extracted {len(line_items)} line items")
        
        return extracted_data
    
    except Exception as e:
        logger.error(f"Error processing SLB: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    from src.analyze import analyze_document
    from dotenv import load_dotenv
    load_dotenv()
    
    document_path = os.path.join("data", "documents", "test_bill_3.pdf")
    try:
        analysis_result = analyze_document(document_path)
        slb_data = process_slb(analysis_result)
        print(json.dumps(slb_data, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}") 