#!/usr/bin/env python
# coding: utf-8

"""
Module for validating extracted bill data using OpenAI.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data(
    data: Dict[str, Any],
    bill_type: str,
    prompt_file: str = "prompts/validation_prompt.txt",
    openai_endpoint: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    deployment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate extracted bill data using OpenAI.
    
    Args:
        data: The extracted bill data to validate.
        bill_type: The type of bill (SLB or MLB).
        prompt_file: Path to the validation prompt file.
        openai_endpoint: Azure OpenAI endpoint. If None, uses environment variable.
        openai_api_key: Azure OpenAI API key. If None, uses environment variable.
        deployment_name: Azure OpenAI deployment name. If None, uses environment variable.
        
    Returns:
        Dict with validation results including 'valid' flag and any errors.
        
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If OpenAI credentials are missing.
        Exception: For other errors during validation.
    """
    try:
        logger.info(f"Validating {bill_type} bill data")
        
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
        
        # Read the validation prompt
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt_base = file.read().strip()
        
        # Prepare the prompt with bill type and data
        prompt = f"{prompt_base}\n\nBill Type: {bill_type}\n\nData to validate:\n{json.dumps(data, indent=2)}"
        
        # Request validation from OpenAI
        logger.info("Requesting validation from Azure OpenAI")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        end_time = time.time()
        logger.info(f"Validation response received in {end_time - start_time:.2f} seconds")
        
        # Parse the response
        validation_result = json.loads(response.choices[0].message.content)
        
        # Log validation result
        if validation_result.get('valid', False):
            logger.info("Validation passed")
        else:
            errors = validation_result.get('errors', [])
            logger.warning(f"Validation failed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"Error in {error.get('field')}: {error.get('error')}")
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise

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