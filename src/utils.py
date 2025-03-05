#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for the bill processing pipeline.
"""

import json
import datetime

def serialize_dates(obj):
    """
    Serialize date, datetime objects, and custom Azure objects to JSON-serializable formats.
    
    Args:
        obj: The object to serialize (dict, list, date, etc.).
        
    Returns:
        A JSON-serializable version of the object.
    """
    if isinstance(obj, dict):
        return {k: serialize_dates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_dates(item) for item in obj]
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return serialize_dates(obj.__dict__)
    elif hasattr(obj, 'as_dict') and callable(getattr(obj, 'as_dict')):
        return serialize_dates(obj.as_dict())
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)