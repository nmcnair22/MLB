#!/usr/bin/env python
# coding: utf-8

"""
Module for analyzing PDF documents using Azure Document Intelligence.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_field_value(field) -> Any:
    """
    Extract the value from a DocumentField based on its type.
    
    Args:
        field: DocumentField object from Azure Document Intelligence.
        
    Returns:
        The extracted value, type-dependent (str, float, list, dict, etc.).
    """
    if not field or not hasattr(field, 'type'):
        return None
    
    field_type = field.type
    if field_type == 'string':
        return field.value_string
    elif field_type == 'number':
        return field.value_number
    elif field_type == 'date':
        return field.value_date
    elif field_type == 'currency':
        return field.value_currency.amount if field.value_currency else None
    elif field_type == 'array':
        return [extract_field_value(item) for item in field.value_array] if field.value_array else []
    elif field_type == 'object':
        return {key: extract_field_value(value) for key, value in field.value_object.items()} if field.value_object else {}
    elif field_type == 'phoneNumber':
        return field.value_phone_number
    elif field_type == 'address':
        return {
            "houseNumber": field.value_address.house_number,
            "road": field.value_address.road,
            "postalCode": field.value_address.postal_code,
            "city": field.value_address.city,
            "state": field.value_address.state,
            "streetAddress": field.value_address.street_address,
            "unit": field.value_address.unit
        } if field.value_address else None
    else:
        logger.warning(f"Unknown field type '{field_type}' encountered")
        return field.content

def analyze_document(
    document_path: str,
    model: str = "prebuilt-invoice",
    di_endpoint: Optional[str] = None,
    di_key: Optional[str] = None
) -> Tuple[AnalyzeResult, Dict[str, Any]]:
    """
    Analyze a PDF document using Azure Document Intelligence with the specified model.
    
    Args:
        document_path: Path to the PDF document.
        model: Model to use ("prebuilt-invoice" or "prebuilt-layout"), defaults to "prebuilt-invoice".
        di_endpoint: Azure Document Intelligence endpoint. If None, uses environment variable.
        di_key: Azure Document Intelligence API key. If None, uses environment variable.
        
    Returns:
        Tuple of (raw_result: AnalyzeResult, parsed_result: Dict[str, Any]) containing raw API response and parsed data.
        
    Raises:
        FileNotFoundError: If the document does not exist.
        ValueError: If the document is not a PDF, is empty, or exceeds size limit.
        Exception: For other errors during analysis.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    if not document_path.lower().endswith('.pdf'):
        raise ValueError(f"Document must be a PDF file: {document_path}")
    
    file_size = os.path.getsize(document_path)
    if file_size == 0:
        raise ValueError(f"Document is empty: {document_path}")
    if file_size > 50 * 1024 * 1024:  # 50 MB limit
        raise ValueError(f"Document exceeds 50MB limit: {file_size} bytes")

    di_endpoint = di_endpoint or os.getenv("DOC_INTELLIGENCE_ENDPOINT")
    di_key = di_key or os.getenv("DOC_INTELLIGENCE_KEY")
    
    if not di_endpoint or not di_key:
        raise ValueError("Missing Azure Document Intelligence endpoint or key")

    try:
        logger.info(f"Starting document analysis for {document_path} with model {model}")
        start_time = time.time()
        
        client = DocumentIntelligenceClient(
            endpoint=di_endpoint,
            credential=AzureKeyCredential(di_key)
        )
        
        features = [DocumentAnalysisFeature.STYLE_FONT] if model == "prebuilt-layout" else []
        
        with open(document_path, "rb") as f:
            poller = client.begin_analyze_document(
                model,
                f,
                content_type="application/octet-stream",
                output_content_format="markdown",
                features=features
            )
        
        result = poller.result()
        
        # Parsed result
        content = result.content
        fields = {}
        if model == "prebuilt-invoice" and result.documents and len(result.documents) > 0:
            document = result.documents[0]
            if document.fields:
                for name, field in document.fields.items():
                    fields[name] = extract_field_value(field)
        
        end_time = time.time()
        logger.info(f"Document analysis with {model} completed in {end_time - start_time:.2f} seconds")
        
        parsed_result = {
            "content": content,
            "fields": fields if model == "prebuilt-invoice" else {},
            "styles": [style.__dict__ for style in result.styles] if model == "prebuilt-layout" and result.styles else []
        }
        
        return result, parsed_result
    
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    document_path = os.path.join("data", "documents", "test_bill_3.pdf")
    try:
        raw_result, parsed_result = analyze_document(document_path, model="prebuilt-invoice")
        print(f"Raw Result Type: {type(raw_result)}")
        print(f"Content length: {len(parsed_result['content'])}")
        print(f"Fields extracted: {len(parsed_result['fields'])}")
        print(f"Sample field - CustomerId: {parsed_result['fields'].get('CustomerId', 'Not found')}")
    except Exception as e:
        print(f"Error: {str(e)}")