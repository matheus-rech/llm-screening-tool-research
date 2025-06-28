#!/usr/bin/env python3
"""Test script to process the user's citation data file."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_user_citation_data():
    """Test processing the user's citation data file."""
    file_path = '/home/ubuntu/attachments/28109698-2406-427d-8382-7281fdae2739/records+3.ris'
    
    try:
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File size: {len(content)} characters")
        print(f"File lines: {len(content.splitlines())} lines")
        
        from app.services.utils.file_parser import load_studies
        
        studies = load_studies(content, 'records+3.ris')
        print(f"Successfully parsed {len(studies)} studies")
        
        if studies:
            print(f"\nFirst study details:")
            first_study = studies[0]
            print(f"  Title: {first_study.get('title', 'No title')}")
            print(f"  Authors: {first_study.get('authors', 'No authors')}")
            print(f"  Journal: {first_study.get('journal_name', 'No journal')}")
            print(f"  Year: {first_study.get('year', 'No year')}")
            print(f"  Abstract length: {len(first_study.get('abstract', ''))} characters")
            print(f"  DOI: {first_study.get('doi', 'No DOI')}")
            print(f"  PMID: {first_study.get('pmid', 'No PMID')}")
            
            if len(studies) > 1:
                print(f"\nSecond study title: {studies[1].get('title', 'No title')}")
            if len(studies) > 2:
                print(f"Third study title: {studies[2].get('title', 'No title')}")
        
        print(f"\nFile parsing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_user_citation_data()
    sys.exit(0 if success else 1)
