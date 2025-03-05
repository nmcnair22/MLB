#!/usr/bin/env python
# coding: utf-8

"""
Module for archiving bills and extracted data.
"""

import os
import json
import shutil
import logging
from typing import Dict, Any, Optional
from .utils import serialize_dates  # Import from utils.py instead of main.py

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def archive_bill(
    document_path: str,
    data: Dict[str, Any],
    validation_result: Dict[str, Any],
    bill_type: Optional[str],
    archive_dir: str = "data/archive",
    audit_dir: str = "data/audit",
    output_dir: str = "data/output"
) -> Dict[str, str]:
    """
    Archive a bill and its extracted data based on validation results.
    
    Args:
        document_path: Path to the original PDF document.
        data: The extracted bill data.
        validation_result: The validation result.
        bill_type: The type of bill (SLB, MLB, or None if unknown).
        archive_dir: Directory for successfully processed bills.
        audit_dir: Directory for bills failing validation.
        output_dir: Directory for extracted JSON data.
        
    Returns:
        Dict with paths to the archived files.
        
    Raises:
        FileNotFoundError: If the document does not exist.
        Exception: For other errors during archiving.
    """
    try:
        logger.info(f"Archiving {bill_type or 'unknown'} bill: {document_path}")
        
        # Check if document exists
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Create directories if they don't exist
        os.makedirs(archive_dir, exist_ok=True)
        os.makedirs(audit_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the document filename
        document_name = os.path.basename(document_path)
        document_base_name = os.path.splitext(document_name)[0]
        
        # Determine target directory based on validation result
        is_valid = validation_result.get('valid', False)
        target_dir = archive_dir if is_valid else audit_dir
        
        # Create target paths
        target_document_path = os.path.join(target_dir, document_name)
        output_json_path = os.path.join(output_dir, f"{document_base_name}_output.json")
        
        # Copy the document to the target directory
        shutil.copy2(document_path, target_document_path)
        logger.info(f"Document copied to {target_document_path}")
        
        # Prepare combined data for JSON output
        archive_data = {
            "extracted_data": data,
            "validation_result": validation_result,
            "bill_type": bill_type
        }
        
        # Save the combined data to the output directory with serialization
        with open(output_json_path, 'w') as f:
            json.dump(serialize_dates(archive_data), f, indent=2)
        logger.info(f"Data saved to {output_json_path}")
        
        # Remove the original document if successfully archived
        if os.path.exists(target_document_path):
            os.remove(document_path)
            logger.info(f"Original document removed: {document_path}")
        
        return {
            'document': target_document_path,
            'data': output_json_path
        }
    
    except Exception as e:
        logger.error(f"Error archiving bill: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Sample data and validation result for testing
    sample_data = {
        "account": {
            "account_number": "123456789",
            "invoice_date": "2023-05-15",
            "total_due": "$150.25"
        },
        "line_items": [
            {
                "description": "Internet Service",
                "date_range": "May 1-31, 2023",
                "recurring_charges": "$99.99",
                "taxes_fees": "$10.26",
                "total": "$110.25"
            }
        ]
    }
    
    sample_validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Create a test document
    test_document_path = os.path.join("data", "documents", "test_archive.txt")
    with open(test_document_path, 'w') as f:
        f.write("Test document for archiving")
    
    try:
        archive_result = archive_bill(
            test_document_path,
            sample_data,
            sample_validation,
            "SLB"
        )
        print(f"Document archived to: {archive_result['document']}")
        print(f"Data saved to: {archive_result['data']}")
    except Exception as e:
        print(f"Error: {str(e)}")