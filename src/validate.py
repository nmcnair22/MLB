#!/usr/bin/env python
# coding: utf-8

"""
Module for validating extracted bill data using OpenAI.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from openai import AzureOpenAI

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
    errors = []
    try:
        # Get master account total
        master_total = float(data.get('master_account', {}).get('total_due', '0').replace('$', '').replace(',', ''))
        
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
        
        # Check if totals match within $0.02 tolerance
        is_valid = abs(master_total - sub_total) <= 0.02
        
        if not is_valid:
            errors.append({
                'field': 'master_account.total_due',
                'error': f'Master total (${master_total:.2f}) does not match sum of sub-account totals (${sub_total:.2f})'
            })
        
        return is_valid, master_total, sub_total, errors
    
    except ValueError as e:
        errors.append({
            'field': 'master_account.total_due',
            'error': f'Invalid master account total: {data.get("master_account", {}).get("total_due")}'
        })
        return False, 0.0, 0.0, errors

def validate_data(
    data: Dict[str, Any],
    bill_type: str,
    prompt_file: str = "prompts/validation_prompt.txt",
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate extracted bill data focusing on essential fields and data quality.
    
    Args:
        data: The extracted bill data to validate.
        bill_type: The type of bill (SLB or MLB).
        prompt_file: Path to the validation prompt file.
        openai_endpoint: Azure OpenAI endpoint.
        openai_api_key: Azure OpenAI API key.
        deployment_name: Azure OpenAI deployment name.
        
    Returns:
        Dict with validation results including 'valid' flag, errors, and notes.
    """
    errors = []
    notes = []
    
    try:
        if bill_type == "MLB":
            # Validate master account required fields
            master_account = data.get('master_account', {})
            required_master_fields = {
                'account_number': 'Account number',
                'total_due': 'Amount due',
                'due_date': 'Due date',
                'vendor_name': 'Vendor name'
            }
            
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
                    # Try parsing the date in various formats
                    from dateutil import parser
                    parser.parse(str(master_account['due_date']))
                except ValueError:
                    errors.append({
                        'field': 'master_account.due_date',
                        'error': f'Invalid date format: {master_account["due_date"]}'
                    })
            
            # Validate sub-accounts
            sub_accounts = data.get('sub_accounts', [])
            if not sub_accounts:
                errors.append({
                    'field': 'sub_accounts',
                    'error': 'No sub-accounts found'
                })
            else:
                for i, sub_account in enumerate(sub_accounts):
                    # Check required sub-account fields
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
            
            # Calculate and verify totals
            if not errors:
                master_total = float(str(master_account['total_due']).replace('$', '').replace(',', ''))
                sub_total = sum(
                    float(str(sub['total_due']).replace('$', '').replace(',', ''))
                    for sub in sub_accounts
                )
                
                if abs(master_total - sub_total) > 0.02:  # Allow for rounding differences
                    notes.append({
                        'field': 'totals',
                        'note': f'Total mismatch: Master total ${master_total:.2f} vs. Sub-accounts total ${sub_total:.2f}'
                    })
        
        # Return validation results
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'notes': notes
        }
    
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
    Perform basic validation checks without using OpenAI.
    
    Args:
        data: The extracted bill data to validate.
        bill_type: The type of bill (SLB or MLB).
        
    Returns:
        Dict with validation results including 'valid' flag and any errors.
    """
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
        errors.append({
            'field': 'general',
            'error': f'Validation error: {str(e)}'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Sample data for testing
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
            },
            {
                "description": "Phone Service",
                "date_range": "May 1-31, 2023",
                "recurring_charges": "$35.00",
                "taxes_fees": "$5.00",
                "total": "$40.00"
            }
        ]
    }
    
    try:
        # First perform basic validation
        basic_result = perform_basic_validation(sample_data, "SLB")
        print(f"Basic validation result: {'Valid' if basic_result['valid'] else 'Invalid'}")
        
        if not basic_result['valid']:
            print("Errors:")
            for error in basic_result['errors']:
                print(f"  - {error['field']}: {error['error']}")
        
        if basic_result['warnings']:
            print("Warnings:")
            for warning in basic_result['warnings']:
                print(f"  - {warning['field']}: {warning['warning']}")
        
        # Then perform OpenAI validation
        openai_result = validate_data(sample_data, "SLB")
        print(f"OpenAI validation result: {'Valid' if openai_result['valid'] else 'Invalid'}")
        
        if not openai_result['valid']:
            print("Errors:")
            for error in openai_result['errors']:
                print(f"  - {error['field']}: {error['error']}")
        
        if openai_result.get('warnings', []):
            print("Warnings:")
            for warning in openai_result['warnings']:
                print(f"  - {warning['field']}: {warning['warning']}")
    
    except Exception as e:
        print(f"Error: {str(e)}") 