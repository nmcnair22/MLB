#!/usr/bin/env python
# coding: utf-8

"""
Main module for orchestrating the bill processing pipeline with detailed API response debugging.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required functions
from .analyze import analyze_document
from .bill_type import determine_bill_type
from .process_slb import process_slb
from .process_mlb import process_mlb
from .validate import validate_data, perform_basic_validation
from .archive import archive_bill
from .rag_utils import rag_query, chunk_and_index_document
from .utils import serialize_dates

def list_documents(documents_dir: str = "data/documents") -> List[str]:
    """Lists all PDF documents in the specified directory."""
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir, exist_ok=True)
        logger.info(f"Created documents directory: {documents_dir}")
        return []
    documents = [os.path.join(documents_dir, file) for file in os.listdir(documents_dir) if file.lower().endswith('.pdf')]
    return documents

def print_extracted_fields(analysis_result):
    """Prints extracted fields from the invoice analysis."""
    print("\n=== Extracted Fields from Document Analysis ===")
    fields = analysis_result.get('fields', {})
    for key, value in fields.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            print(f"{key}:")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    print(f"  Item {i+1}:")
                    for sub_key, sub_value in item.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  Item {i+1}: {item}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)

def print_slb_data(extracted_data):
    """Prints extracted data for Single Line Bills (SLB)."""
    print("\n=== Extracted SLB Data ===")
    account = extracted_data.get('account', {})
    line_items = extracted_data.get('line_items', [])
    print("Account Information:")
    for key, value in account.items():
        print(f"  {key}: {value}")
    print("\nLine Items:")
    for i, item in enumerate(line_items):
        print(f"  Item {i+1}:")
        for key, value in item.items():
            print(f"    {key}: {value}")
    print("=" * 50)

def print_mlb_data(extracted_data):
    """Prints extracted data for Multi-Line Bills (MLB)."""
    print("\n=== Extracted MLB Data ===")
    master_account = extracted_data.get('master_account', {})
    sub_accounts = extracted_data.get('sub_accounts', [])
    print("Master Account Information:")
    for key, value in master_account.items():
        print(f"  {key}: {value}")
    print(f"\nSub-Accounts ({len(sub_accounts)}):")
    for i, account in enumerate(sub_accounts):
        print(f"\n  Sub-Account {i+1}:")
        for key, value in account.items():
            if key != 'line_items':
                print(f"    {key}: {value}")
        line_items = account.get('line_items', [])
        if line_items:
            print(f"    Line Items ({len(line_items)}):")
            for j, item in enumerate(line_items):
                print(f"      Item {j+1}:")
                for key, value in item.items():
                    print(f"        {key}: {value}")
    print("=" * 50)

def process_document(document_path: str) -> Dict[str, Any]:
    """Processes a document through analysis, extraction, validation, and archiving."""
    logger.info(f"Processing document: {document_path}")
    document_name = os.path.basename(document_path)
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print("\nStep 1: Analyzing document with prebuilt-invoice")
    raw_result_invoice, analysis_result_invoice = analyze_document(document_path, model="prebuilt-invoice")
    print("Step 1: Completed")
    print_extracted_fields(analysis_result_invoice)
    
    analysis_raw_file = os.path.join(output_dir, f"{document_name}_analysis_invoice_raw.json")
    with open(analysis_raw_file, 'w') as f:
        json.dump(serialize_dates(analysis_result_invoice), f, indent=2)
    print(f"Step 1 Raw API Response: Written to {analysis_raw_file}")

    fields = analysis_result_invoice.get('fields', {})
    master_account = {
        'account_number': fields.get('CustomerId'),
        'total_due': str(fields.get('AmountDue')),
        'due_date': fields.get('DueDate'),
        'vendor_name': fields.get('VendorName'),
    }
    document_content = analysis_result_invoice.get('content', '')

    print("\nStep 2: Determining bill type")
    bill_type_result = determine_bill_type(analysis_result_invoice)
    bill_type, status = bill_type_result['bill_type'], bill_type_result['status']
    print(f"Step 2: Completed - Bill Type: {bill_type}, Status: {status}")
    print(f"Step 2 Result: {json.dumps(bill_type_result, indent=2)}")

    if status == "audit":
        print("Step 2: Flagged for audit")
        validation_result = {"valid": False, "errors": [{"field": "bill_type", "error": "Unknown account number"}]}
        archive_result = archive_bill(document_path, {"error": "Unknown account number"}, validation_result, None)
        print(f"Step 5: Archiving bill - Completed (Audit)")
        print(f"Step 5 Result: Archived to {archive_result['document']}, Data saved to {archive_result['data']}")
        return {
            'document_path': document_path,
            'bill_type': None,
            'status': 'audit',
            'validation': validation_result,
            'archive': archive_result,
            'document_content': document_content,
            'analysis_result_layout': None
        }

    print(f"\nStep 3: Processing {bill_type} bill")
    if bill_type == "SLB":
        extracted_data = process_slb(analysis_result_invoice)
        print("Step 3: Completed")
        print_slb_data(extracted_data)
        extracted_file = os.path.join(output_dir, f"{document_name}_extracted_raw.json")
        with open(extracted_file, 'w') as f:
            json.dump(serialize_dates(extracted_data), f, indent=2)
        print(f"Step 3 Raw API Response: Written to {extracted_file}")
        analysis_result_layout = None
    else:  # MLB
        print("\nStep 3: Analyzing document with prebuilt-layout for MLB")
        raw_result_layout, analysis_result_layout = analyze_document(document_path, model="prebuilt-layout")
        print("Step 3: Completed layout analysis")
        # Save raw layout analysis for debugging
        layout_raw_file = os.path.join(output_dir, f"{document_name}_analysis_layout_raw.json")
        with open(layout_raw_file, 'w') as f:
            json.dump(serialize_dates(analysis_result_layout), f, indent=2)
        print(f"Step 3 Layout Raw API Response: Written to {layout_raw_file}")
        
        extracted_data = process_mlb(
            analysis_result_layout,
            master_account,
            document_name,
            prompt_file="telecom_prompt.txt"
        )
        print("Step 3: Completed MLB processing")
        print_mlb_data(extracted_data)
        extracted_file = os.path.join(output_dir, f"{document_name}_extracted.json")
        with open(extracted_file, 'w') as f:
            json.dump(serialize_dates(extracted_data), f, indent=2)
        print(f"Step 3 Parsed Result: Written to {extracted_file}")

    print("\nStep 4: Validating extracted data")
    basic_validation = perform_basic_validation(extracted_data, bill_type)
    validation_result = validate_data(extracted_data, bill_type, document_content=document_content)
    print(f"Step 4: Completed - Validation: {'Valid' if validation_result['valid'] else 'Invalid'}")
    print("\n=== Validation Results ===")
    print(f"Valid: {validation_result['valid']}")
    if not validation_result['valid'] and 'errors' in validation_result:
        print("\nErrors:")
        for error in validation_result['errors']:
            print(f"  Field: {error.get('field', 'Unknown')}")
            print(f"  Error: {error.get('error', 'Unknown error')}")
    if 'notes' in validation_result and validation_result['notes']:
        print("\nNotes:")
        for note in validation_result['notes']:
            print(f"  Field: {note.get('field', 'Unknown')}")
            print(f"  Note: {note.get('note', 'Unknown note')}")
    print("=" * 50)
    validation_raw_file = os.path.join(output_dir, f"{document_name}_validation_raw.json")
    with open(validation_raw_file, 'w') as f:
        json.dump(serialize_dates(validation_result), f, indent=2)
    print(f"Step 4 Raw API Response: Written to {validation_raw_file}")

    print("\nStep 5: Archiving bill")
    archive_result = archive_bill(document_path, extracted_data, validation_result, bill_type)
    print("Step 5: Completed")
    print(f"Step 5 Result: Archived to {archive_result['document']}, Data saved to {archive_result['data']}")

    return {
        'document_path': document_path,
        'bill_type': bill_type,
        'status': status,
        'extracted_data': extracted_data,
        'validation': validation_result,
        'archive': archive_result,
        'document_content': document_content,
        'analysis_result_layout': analysis_result_layout
    }

def main():
    """Main function to run the bill processing pipeline."""
    try:
        load_dotenv()
        print(f"Loaded VECTOR_SEARCH_KEY: {os.getenv('VECTOR_SEARCH_KEY')[:8]}...")
        required_vars = [
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "DEPLOYMENT_NAME",
            "DOC_INTELLIGENCE_ENDPOINT", "DOC_INTELLIGENCE_KEY",
            "VECTOR_SEARCH_ENDPOINT", "VECTOR_SEARCH_KEY"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
            sys.exit(1)
        
        documents = list_documents()
        if not documents:
            logger.warning("No PDF documents found in data/documents directory")
            print("No PDF documents found in data/documents directory")
            sys.exit(0)
        
        print("\nAvailable documents:")
        for i, doc in enumerate(documents):
            print(f"{i+1}. {os.path.basename(doc)}")
        
        while True:
            try:
                selection = input("\nEnter the number of the document to process (or 'q' to quit): ").strip()
                if selection.lower() == 'q':
                    print("Exiting...")
                    sys.exit(0)
                index = int(selection) - 1
                if 0 <= index < len(documents):
                    selected_document = documents[index]
                    break
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(documents)}")
            except ValueError:
                print("Invalid input. Please enter a number or 'q'")
        
        print(f"\nProcessing document: {os.path.basename(selected_document)}")
        result = process_document(selected_document)
        
        extracted_data = result.get('extracted_data', {})
        validation_result = result.get('validation', {})
        bill_type = result.get('bill_type')
        document_content = result.get('document_content', '')
        analysis_result_layout = result.get('analysis_result_layout')
        
        print("\nProcessing complete!")
        print(f"Bill type: {bill_type}")
        print(f"Status: {result['status']}")
        print(f"Validation: {'Passed' if validation_result['valid'] else 'Invalid'}")
        print(f"Document archived to: {result['archive']['document']}")
        print(f"Data saved to: {result['archive']['data']}")
        
        # Set up RAG vector store using raw data (layout for MLB, invoice content for SLB)
        vector_store = chunk_and_index_document(
            document_content,
            analysis_result_layout
        ) if document_content else None
        
        while True:
            print("\nScript completed. Choose an option:")
            print("1. Exit")
            print("2. Ask a question about the bill")
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                logger.info("User chose to exit")
                print("Exiting script.")
                break
            elif choice == "2":
                if not vector_store:
                    print("Error: No document content available for querying.")
                    continue
                question = input("Enter your question about the bill: ").strip()
                logger.info(f"User asked: {question}")
                response = rag_query(question, vector_store)
                answer = response['value'] if response['found'] else 'Not found in the document'
                print(f"Answer: {answer}")
                logger.info(f"RAG response: {answer}")
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()