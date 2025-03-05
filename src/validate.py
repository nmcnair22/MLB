#!/usr/bin/env python
# coding: utf-8

"""
Module for validating extracted bill data using OpenAI and RAG-based corrections.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from openai import AzureOpenAI
from .rag_utils import chunk_and_index_document, rag_query

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_mlb_totals(data: Dict[str, Any]) -> Tuple[bool, float, float, List[Dict[str, str]]]:
    """
    Validate MLB totals by comparing master account total with sum of sub-account totals.
    
    Args:
        data: The extracted MLB data.
        
    Returns:
        Tuple of (is_valid: bool, master_total: float, sub_total: float, errors: List[Dict[str, str]])
    """
    logger.info("Starting validate_mlb_totals")
    errors = []
    try:
        # Get master account total
        master_total_str = data.get('master_account', {}).get('total_due', '0')
        master_total = float(master_total_str.replace('$', '').replace(',', ''))
        logger.info(f"Master total: {master_total}")
        
        # Sum sub-account totals
        sub_total = 0.0
        for sub_account in data.get('sub_accounts', []):
            try:
                sub_total += float(sub_account.get('total_due', '0').replace('$', '').replace(',', ''))
            except ValueError as e:
                errors.append({
                    'field': f'sub_accounts[{sub_account.get("sub_account_number", "Unknown")}].total_due',
                    'error': f'Invalid sub-account total: {sub_account.get("total_due")}'
                })
        logger.info(f"Sub-account total: {sub_total}")
        
        # Check if totals match within $0.02 tolerance
        is_valid = abs(master_total - sub_total) <= 0.02
        if not is_valid:
            errors.append({
                'field': 'master_account.total_due',
                'error': f'Master total (${master_total:.2f}) does not match sum of sub-account totals (${sub_total:.2f})'
            })
        
        return is_valid, master_total, sub_total, errors
    
    except ValueError as e:
        logger.error(f"Error parsing master total: {e}")
        errors.append({
            'field': 'master_account.total_due',
            'error': f'Invalid master account total: {master_total_str}'
        })
        return False, 0.0, 0.0, errors

def validate_master_due_date(master_account: Dict[str, Any], vector_store: Any, max_retries: int = 2) -> Tuple[bool, str]:
    """
    Validate that the master account has a due date; use RAG to correct if missing.
    
    Args:
        master_account: Extracted master account data.
        vector_store: Indexed document chunks for RAG queries.
        max_retries: Maximum correction attempts.
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info("Starting validate_master_due_date")
    if not vector_store:
        logger.warning("No vector store provided; skipping RAG correction")
        return bool(master_account.get("due_date")), "Due date check without RAG"

    for attempt in range(max_retries + 1):
        logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
        if "due_date" not in master_account or not master_account["due_date"]:
            query = "What is the due date listed on the bill?"
            logger.info(f"Querying RAG: {query}")
            response = rag_query(query, vector_store)
            logger.info(f"RAG response: {response}")
            if response["found"] and response["value"]:
                master_account["due_date"] = response["value"]
                message = f"Due date updated to {response['value']} after attempt {attempt + 1}."
                logger.info(message)
                return True, message
            else:
                message = f"Attempt {attempt + 1}: Due date not found in document."
                logger.info(message)
                if attempt == max_retries:
                    return False, message
        else:
            logger.info("Due date present; validation successful")
            return True, "Due date present"
    logger.error("Unexpected exit from retry loop")
    return False, "Unexpected loop termination"

def validate_sub_account_totals(master_account: Dict[str, Any], sub_accounts: List[Dict[str, Any]], vector_store: Any, max_retries: int = 2) -> Tuple[bool, str]:
    """
    Validate that sub-account totals match the master total; use RAG if inconsistent.
    
    Args:
        master_account: Extracted master account data.
        sub_accounts: List of sub-account data.
        vector_store: Indexed document chunks for RAG queries.
        max_retries: Maximum correction attempts.
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    logger.info("Starting validate_sub_account_totals")
    if not vector_store:
        logger.warning("No vector store provided; skipping RAG correction")
        return True, "Totals check skipped without RAG"

    for attempt in range(max_retries + 1):
        logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
        try:
            master_total = float(str(master_account.get('total_due', '0')).replace('$', '').replace(',', ''))
            sub_total = sum(float(str(sub.get('total_due', '0')).replace('$', '').replace(',', '')) for sub in sub_accounts)
            logger.info(f"Master total: {master_total}, Sub total: {sub_total}")
        except ValueError as e:
            logger.error(f"Error parsing totals: {e}")
            return False, "Invalid total due format in master or sub-accounts"

        if abs(master_total - sub_total) > 0.02:  # Allow for rounding differences
            query = "What is the total due amount listed on the bill?"
            logger.info(f"Querying RAG: {query}")
            response = rag_query(query, vector_store)
            logger.info(f"RAG response: {response}")
            if response["found"] and response["value"]:
                master_account["total_due"] = response["value"]
                message = f"Master total updated to {response['value']} after attempt {attempt + 1}."
                logger.info(message)
                return True, message
            else:
                message = f"Attempt {attempt + 1}: Could not reconcile totals (Master: {master_total}, Sub: {sub_total})."
                logger.info(message)
                if attempt == max_retries:
                    return False, message
        else:
            logger.info("Totals match; validation successful")
            return True, "Sub-account totals match master total"
    logger.error("Unexpected exit from retry loop")
    return False, "Unexpected loop termination"

def validate_data(
    data: Dict[str, Any],
    bill_type: str,
    document_content: str = "",
    prompt_file: str = "prompts/validation_prompt.txt",
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate extracted bill data focusing on essential fields and data quality with RAG corrections.
    
    Args:
        data: The extracted bill data to validate.
        bill_type: The type of bill (SLB or MLB).
        document_content: Raw text content of the bill document for RAG queries.
        prompt_file: Path to the validation prompt file (unused in this version).
        openai_endpoint: Azure OpenAI endpoint (optional).
        openai_api_key: Azure OpenAI API key (optional).
        deployment_name: Azure OpenAI deployment name (optional).
        
    Returns:
        Dict with validation results including 'valid' flag, errors, and notes.
    """
    logger.info(f"Starting validate_data for {bill_type} bill")
    errors = []
    notes = []
    
    try:
        if bill_type == "MLB":
            master_account = data.get('master_account', {})
            sub_accounts = data.get('sub_accounts', [])
            
            # Index the document for RAG queries if content is provided
            logger.info("Initializing vector store for RAG")
            vector_store = chunk_and_index_document(document_content) if document_content else None
            logger.info("Vector store initialization complete")
            
            # Phase 1: Validate master due date with RAG correction
            success, message = validate_master_due_date(master_account, vector_store)
            if not success:
                errors.append({'field': 'master_account.due_date', 'error': message})
            else:
                notes.append({'field': 'master_account.due_date', 'note': message})
            
            # Phase 2: Validate sub-account totals with RAG correction
            if sub_accounts:
                success, message = validate_sub_account_totals(master_account, sub_accounts, vector_store)
                if not success:
                    errors.append({'field': 'totals', 'error': message})
                else:
                    notes.append({'field': 'totals', 'note': message})
            
            # Validate master account required fields
            required_master_fields = {
                'account_number': 'Account number',
                'total_due': 'Amount due',
                'vendor_name': 'Vendor name'
            }
            logger.info("Checking required master account fields")
            for field, display_name in required_master_fields.items():
                if not master_account.get(field):
                    errors.append({
                        'field': f'master_account.{field}',
                        'error': f'Missing required field: {display_name}'
                    })
            
            # Validate data types and formats
            if master_account.get('total_due'):
                try:
                    amount = float(str(master_account['total_due']).replace('$', '').replace(',', ''))
                    if amount < 0:
                        notes.append({
                            'field': 'master_account.total_due',
                            'note': f'Negative total due amount: ${amount:.2f}'
                        })
                except ValueError:
                    errors.append({
                        'field': 'master_account.total_due',
                        'error': f'Invalid amount format: {master_account["total_due"]}'
                    })
            
            if master_account.get('due_date'):
                try:
                    from dateutil import parser
                    parser.parse(str(master_account['due_date']))
                except ValueError:
                    errors.append({
                        'field': 'master_account.due_date',
                        'error': f'Invalid date format: {master_account["due_date"]}'
                    })
            
            # Validate sub-accounts
            logger.info("Validating sub-accounts")
            if not sub_accounts:
                errors.append({
                    'field': 'sub_accounts',
                    'error': 'No sub-accounts found'
                })
            else:
                for i, sub_account in enumerate(sub_accounts):
                    if not sub_account.get('sub_account_number'):
                        errors.append({
                            'field': f'sub_accounts[{i}].sub_account_number',
                            'error': 'Missing sub-account number'
                        })
                    
                    if not sub_account.get('total_due'):
                        errors.append({
                            'field': f'sub_accounts[{i}].total_due',
                            'error': 'Missing total due amount'
                        })
                    else:
                        try:
                            float(str(sub_account['total_due']).replace('$', '').replace(',', ''))
                        except ValueError:
                            errors.append({
                                'field': f'sub_accounts[{i}].total_due',
                                'error': f'Invalid amount format: {sub_account["total_due"]}'
                            })
        
        elif bill_type == "SLB":
            logger.info("Performing basic validation for SLB")
            return perform_basic_validation(data, bill_type)
        
        # Return validation results
        result = {
            'valid': len(errors) == 0,
            'errors': errors,
            'notes': notes
        }
        logger.info(f"Validation completed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        errors.append({
            'field': 'general',
            'error': f'Validation error: {str(e)}'
        })
        return {
            'valid': False,
            'errors': errors,
            'notes': notes
        }

def perform_basic_validation(data: Dict[str, Any], bill_type: str) -> Dict[str, Any]:
    """
    Perform basic validation checks without using OpenAI or RAG.
    
    Args:
        data: The extracted bill data to validate.
        bill_type: The type of bill (SLB or MLB).
        
    Returns:
        Dict with validation results including 'valid' flag and any errors.
    """
    logger.info(f"Starting perform_basic_validation for {bill_type}")
    errors = []
    warnings = []
    
    try:
        if bill_type == "SLB":
            # Validate SLB data
            account = data.get('account', {})
            
            # Check account number
            if not account.get('account_number'):
                errors.append({
                    'field': 'account.account_number',
                    'error': 'Account number is missing'
                })
            
            # Check invoice date
            if not account.get('invoice_date'):
                errors.append({
                    'field': 'account.invoice_date',
                    'error': 'Invoice date is missing'
                })
            
            # Check total due
            if not account.get('total_due'):
                errors.append({
                    'field': 'account.total_due',
                    'error': 'Total due is missing'
                })
            
            # Check line items
            line_items = data.get('line_items', [])
            if not line_items:
                warnings.append({
                    'field': 'line_items',
                    'warning': 'No line items found'
                })
            
        elif bill_type == "MLB":
            # Validate MLB data
            master_account = data.get('master_account', {})
            
            # Check master account number
            if not master_account.get('account_number'):
                errors.append({
                    'field': 'master_account.account_number',
                    'error': 'Master account number is missing'
                })
            
            # Check invoice date
            if not master_account.get('invoice_date'):
                errors.append({
                    'field': 'master_account.invoice_date',
                    'error': 'Invoice date is missing'
                })
            
            # Check total due
            if not master_account.get('total_due'):
                errors.append({
                    'field': 'master_account.total_due',
                    'error': 'Total due is missing'
                })
            
            # Check sub-accounts
            sub_accounts = data.get('sub_accounts', [])
            if not sub_accounts:
                errors.append({
                    'field': 'sub_accounts',
                    'error': 'No sub-accounts found'
                })
            
            # Check each sub-account
            for i, sub_account in enumerate(sub_accounts):
                if not sub_account.get('sub_account_number'):
                    errors.append({
                        'field': f'sub_accounts[{i}].sub_account_number',
                        'error': 'Sub-account number is missing'
                    })
                
                if not sub_account.get('location'):
                    warnings.append({
                        'field': f'sub_accounts[{i}].location',
                        'warning': 'Location is missing'
                    })
                
                # Check line items
                line_items = sub_account.get('line_items', [])
                if not line_items:
                    warnings.append({
                        'field': f'sub_accounts[{i}].line_items',
                        'warning': 'No line items found'
                    })
    
    except Exception as e:
        logger.error(f"Error in basic validation: {str(e)}")
        errors.append({
            'field': 'general',
            'error': f'Validation error: {str(e)}'
        })
    
    result = {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
    logger.info(f"Basic validation completed: {result}")
    return result

if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Sample data for testing (MLB example)
    sample_data_mlb = {
        "master_account": {
            "account_number": "123456789",
            "total_due": "$500.00",
            # "due_date": "2023-06-15",  # Intentionally missing for test
            "vendor_name": "Sample Vendor"
        },
        "sub_accounts": [
            {
                "sub_account_number": "987654321",
                "total_due": "$200.00",
                "line_items": [{"description": "Service A", "total": "$100.00"}]
            },
            {
                "sub_account_number": "456789123",
                "total_due": "$300.00",
                "line_items": [{"description": "Service B", "total": "$150.00"}]
            }
        ]
    }
    
    # Simulate document content (normally from analyze.py)
    document_content = "Due Date: 2023-06-15\nTotal Due: $500.00\nSub-Account 1: $200.00\nSub-Account 2: $300.00"
    
    try:
        # Perform validation with RAG for MLB
        validation_result = validate_data(sample_data_mlb, "MLB", document_content)
        print(f"Validation result: {'Valid' if validation_result['valid'] else 'Invalid'}")
        
        if not validation_result['valid']:
            print("Errors:")
            for error in validation_result['errors']:
                print(f"  - {error['field']}: {error['error']}")
        
        if validation_result['notes']:
            print("Notes:")
            for note in validation_result['notes']:
                print(f"  - {note['field']}: {note['note']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")