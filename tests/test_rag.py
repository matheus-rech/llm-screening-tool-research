import pytest
from unittest.mock import patch, mock_open, MagicMock
from app.services.utils.file_parser import (
    parse_ris_file, parse_bibtex_file, parse_csv_file
)
from app.services.screening.dual_llm_screener import DualProviderScreeningOrchestrator

class TestFileParsers:
    """Test file parsing functions."""
    
    def test_parse_ris_file(self, temp_file, sample_ris_content):
        """Test RIS file parsing."""
        with open(temp_file, 'w') as f:
            f.write(sample_ris_content)
        
        articles = parse_ris_file(sample_ris_content)
        assert len(articles) == 1
        assert articles[0]['title'] == 'Test Article'
        assert articles[0]['abstract'] == 'This is a test abstract for screening.'
    
    def test_parse_csv_file(self, temp_file):
        """Test CSV file parsing."""
        csv_content = "title,abstract\nTest Title,Test Abstract\n"
        with open(temp_file, 'w') as f:
            f.write(csv_content)
        
        articles = parse_csv_file(temp_file)
        assert len(articles) == 1
        assert articles[0]['title'] == 'Test Title'
        assert articles[0]['abstract'] == 'Test Abstract'
    
    def test_pmid_parsing(self):
        """Test PMID parsing functionality."""
        from app.services.utils.file_parser import parse_pmid_file_and_fetch
        assert parse_pmid_file_and_fetch is not None

class TestScreening:
    """Test article screening functions."""
    
    def test_dual_provider_screening_orchestrator_initialization(self):
        """Test DualProviderScreeningOrchestrator can be initialized."""
        orchestrator = DualProviderScreeningOrchestrator("test-openai-key", "test-anthropic-key")
        assert orchestrator is not None
        assert orchestrator.openai_provider is not None
        assert orchestrator.anthropic_provider is not None
