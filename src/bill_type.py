#!/usr/bin/env python
# coding: utf-8

"""
Module for determining bill type (SLB or MLB) based on account number using a live database lookup.
"""

import logging
import re
from typing import Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def clean_account_number(account_number: str) -> str:
    """
    Clean the account number by removing non-alphanumeric characters.
    
    Args:
        account_number: The raw account number string.
        
    Returns:
        Cleaned account number.
    """
    return re.sub(r'[^a-zA-Z0-9]', '', account_number)

def get_db_connection():
    """
    Establish a connection to the production database.
    
    Returns:
        MySQL connection object.
        
    Raises:
        Error: If connection fails.
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "cissdm.cis.local"),
            database=os.getenv("DB_NAME", "tem"),
            user=os.getenv("DB_USER", "view"),
            password=os.getenv("DB_PASS", "Eastw00d")
        )
        if connection.is_connected():
            logger.info("Successfully connected to the database")
            return connection
    except Error as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def determine_bill_type(
    analysis_result: Dict[str, Any]
) -> Dict[str, str]:
    """
    Determine the bill type (SLB or MLB) based on the account number via live database lookup.
    
    Args:
        analysis_result: The result from document analysis containing 'fields' and 'content'.
        
    Returns:
        Dict with 'bill_type' (SLB or MLB) and 'status' (ok or audit).
        
    Raises:
        ValueError: If no account number is found in the document.
        Exception: For database connection or query errors.
    """
    try:
        logger.info("Determining bill type")
        
        # Extract account number from analysis result
        account_number = None
        
        # Try fields first (CustomerId or InvoiceId might be used as account number)
        fields = analysis_result.get('fields', {})
        if 'CustomerId' in fields:
            account_number = fields['CustomerId']
        elif 'InvoiceId' in fields:
            account_number = fields['InvoiceId']
        
        # Fallback to content if not in fields
        if not account_number:
            content = analysis_result.get('content', '')
            account_match = re.search(r'Account\s*(?:#|Number|No\.?)\s*[:\s]?\s*([a-zA-Z0-9\s\-]+)', content, re.IGNORECASE)
            if account_match:
                account_number = account_match.group(1).strip()
        
        if not account_number:
            logger.warning("No account number found in document, flagging for audit")
            return {
                'bill_type': None,
                'status': 'audit'
            }
        
        # Clean the account number
        cleaned_account_number = clean_account_number(str(account_number))
        logger.info(f"Found account number: {cleaned_account_number}")
        
        # Connect to the database
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Query temMasterViewUpdated
        query = "SELECT multipleLocations FROM temMasterViewUpdated WHERE accountNumber = %s"
        cursor.execute(query, (cleaned_account_number,))
        result = cursor.fetchone()
        
        # Close the connection
        cursor.close()
        connection.close()
        
        if result:
            logger.info(f"Database record found for account {cleaned_account_number}")
            multiple_locations = result['multipleLocations']
            bill_type = 'MLB' if multiple_locations == 1 else 'SLB'
            logger.info(f"Bill type determined: {bill_type} based on multipleLocations: {multiple_locations}")
            return {
                'bill_type': bill_type,
                'status': 'ok'
            }
        else:
            logger.warning(f"Account number {cleaned_account_number} not found in database, flagging for audit")
            return {
                'bill_type': None,
                'status': 'audit'
            }
    
    except Error as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error determining bill type: {str(e)}")
        raise

if __name__ == "__main__":
    from src.analyze import analyze_document
    load_dotenv()
    
    document_path = os.path.join("data", "documents", "test_bill_2.pdf")
    try:
        analysis_result = analyze_document(document_path)
        bill_type_result = determine_bill_type(analysis_result)
        print(f"Bill type: {bill_type_result['bill_type']}")
        print(f"Status: {bill_type_result['status']}")
    except Exception as e:
        print(f"Error: {str(e)}")