# Telecom Bill Processing System

This project provides an automated system for processing telecom bills, with specific handling for Single-Location Bills (SLB) and Multi-Location Bills (MLB). It uses Azure Document Intelligence for document analysis and Azure OpenAI for extracting structured data from telecom bills.

## Features

- Automatic bill type detection (SLB vs MLB)
- Document layout analysis using Azure Document Intelligence
- Intelligent data extraction using Azure OpenAI
- Validation of extracted data
- Archiving of processed bills
- Detailed logging and debugging output

## Project Structure

- `src/`: Source code for the bill processing pipeline
  - `main.py`: Main orchestration module
  - `analyze.py`: Document analysis using Azure Document Intelligence
  - `bill_type.py`: Bill type detection
  - `process_slb.py`: Processing for Single-Location Bills
  - `process_mlb.py`: Processing for Multi-Location Bills
  - `validate.py`: Data validation
  - `archive.py`: Bill archiving
- `data/`: Data directories
  - `documents/`: Input PDF documents
  - `output/`: Extracted data output
  - `debug/`: Debug information
  - `audit/`: Bills flagged for audit
  - `archive/`: Archived bills
- `prompts/`: Prompt templates for Azure OpenAI
  - `telecom_prompt.txt`: Main prompt for telecom bill processing
  - `mlb_prompt.txt`: Specific prompt for MLB processing
  - `slb_prompt.txt`: Specific prompt for SLB processing
  - `validation_prompt.txt`: Prompt for data validation
- `notebooks/`: Jupyter notebooks for development and testing
- `tests/`: Test files

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your Azure OpenAI and Document Intelligence credentials:
```
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_KEY=your_openai_api_key
DEPLOYMENT_NAME=your_deployment_name
DOC_INTELLIGENCE_ENDPOINT=your_doc_intelligence_endpoint
DOC_INTELLIGENCE_KEY=your_doc_intelligence_key
```

## Usage

1. Place telecom bill PDFs in the `data/documents/` directory

2. Run the main processing script:
```bash
python -m src.main
```

3. Follow the interactive prompts to select a document for processing

4. Review the extracted data in the `data/output/` directory

## Processing Pipeline

1. **Document Analysis**: The system analyzes the document using Azure Document Intelligence
2. **Bill Type Detection**: Determines if the bill is SLB or MLB
3. **Data Extraction**: Extracts structured data using Azure OpenAI with specialized prompts
4. **Validation**: Validates the extracted data for completeness and consistency
5. **Archiving**: Archives the processed bill and extracted data

## Debugging

- Check the `data/debug/` directory for preprocessed document chunks
- Review the console output for detailed processing information
- Examine the extracted JSON files in `data/output/` directory

## License

[Specify your license here] 