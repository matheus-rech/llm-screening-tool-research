import pytest
import json
import tempfile
from pathlib import Path
from app.services.utils.config_manager import (
    ConfigurationManager, ProjectConfiguration, PICOCriteria, 
    APIConfiguration, ProcessingConfiguration
)
from app.services.utils.exceptions import ConfigurationError, ValidationError

class TestConfigurationManager:
    """Test enhanced configuration management."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create configuration manager with temp directory."""
        return ConfigurationManager(temp_config_dir)
    
    @pytest.fixture
    def valid_config_data(self):
        """Valid configuration data for testing."""
        return {
            'name': 'Test Project',
            'description': 'Test project description',
            'pico_data': {
                'population': 'Adults with diabetes',
                'intervention': 'Exercise therapy',
                'comparison': 'Standard care',
                'outcomes': 'Blood glucose control',
                'time_frame': '6 months',
                'study_types': 'RCT'
            },
            'api_data': {
                'api_key': 'test-api-key',
                'conservative_model': 'gpt-4o-mini',
                'liberal_model': 'gpt-3.5-turbo'
            },
            'processing_data': {
                'batch_size': 10,
                'max_workers': 5
            }
        }
    
    def test_create_valid_configuration(self, config_manager, valid_config_data):
        """Test creating a valid configuration."""
        config = config_manager.create_configuration(**valid_config_data)
        
        assert config.name == 'Test Project'
        assert config.pico.population == 'Adults with diabetes'
        assert config.api.api_key == 'test-api-key'
        assert config.processing.batch_size == 10
    
    def test_create_invalid_configuration(self, config_manager):
        """Test creating invalid configuration raises error."""
        invalid_data = {
            'name': '',  # Empty name
            'description': 'Test',
            'pico_data': {
                'population': '',  # Empty population
                'intervention': 'Test',
                'comparison': 'Test',
                'outcomes': 'Test',
                'time_frame': '',
                'study_types': ''
            },
            'api_data': {
                'api_key': ''  # Empty API key
            }
        }
        
        with pytest.raises(ValidationError):
            config_manager.create_configuration(**invalid_data)
    
    def test_save_and_load_template(self, config_manager, valid_config_data):
        """Test saving and loading configuration templates."""
        config = config_manager.create_configuration(**valid_config_data)
        
        # Save template
        template_path = config_manager.save_template(config, 'test_template')
        assert Path(template_path).exists()
        
        # Load template
        loaded_template = config_manager.load_template('test_template')
        
        assert loaded_template['name'] == 'Test Project'
        assert loaded_template['pico']['population'] == 'Adults with diabetes'
        # API key should not be saved in template
        assert 'api_key' not in loaded_template['api']
    
    def test_list_templates(self, config_manager, valid_config_data):
        """Test listing available templates."""
        config = config_manager.create_configuration(**valid_config_data)
        
        # Initially no templates
        assert config_manager.list_templates() == []
        
        # Save some templates
        config_manager.save_template(config, 'template1')
        config_manager.save_template(config, 'template2')
        
        templates = config_manager.list_templates()
        assert 'template1' in templates
        assert 'template2' in templates
    
    def test_delete_template(self, config_manager, valid_config_data):
        """Test deleting templates."""
        config = config_manager.create_configuration(**valid_config_data)
        config_manager.save_template(config, 'to_delete')
        
        assert 'to_delete' in config_manager.list_templates()
        
        # Delete template
        success = config_manager.delete_template('to_delete')
        assert success
        assert 'to_delete' not in config_manager.list_templates()
        
        # Try to delete non-existent template
        success = config_manager.delete_template('non_existent')
        assert not success
    
    def test_validate_configuration_dict(self, config_manager):
        """Test validating configuration dictionaries."""
        valid_dict = {
            'name': 'Test',
            'description': 'Test description',
            'pico': {
                'population': 'Adults',
                'intervention': 'Drug',
                'comparison': 'Placebo',
                'outcomes': 'Recovery',
                'time_frame': '6 months',
                'study_types': 'RCT'
            },
            'api': {
                'api_key': 'test-key'
            }
        }
        
        errors = config_manager.validate_configuration_dict(valid_dict)
        assert len(errors) == 0
        
        # Test invalid dict
        invalid_dict = {
            'name': '',  # Empty name
            'pico': {
                'population': '',  # Empty population
                'intervention': 'Drug',
                'comparison': 'Placebo',
                'outcomes': 'Recovery'
            },
            'api': {}  # Missing api_key
        }
        
        errors = config_manager.validate_configuration_dict(invalid_dict)
        assert len(errors) > 0
    
    def test_merge_with_template(self, config_manager, valid_config_data):
        """Test merging templates with overrides."""
        config = config_manager.create_configuration(**valid_config_data)
        config_manager.save_template(config, 'base_template')
        
        overrides = {
            'name': 'Modified Project',
            'api': {
                'api_key': 'new-api-key',
                'max_retries': 5
            }
        }
        
        merged = config_manager.merge_with_template('base_template', overrides)
        
        assert merged['name'] == 'Modified Project'
        assert merged['api']['api_key'] == 'new-api-key'
        assert merged['api']['max_retries'] == 5
        # Original values should be preserved
        assert merged['pico']['population'] == 'Adults with diabetes'

class TestPICOCriteria:
    """Test PICO criteria validation."""
    
    def test_valid_pico(self):
        """Test valid PICO criteria."""
        pico = PICOCriteria(
            population='Adults',
            intervention='Drug A',
            comparison='Placebo',
            outcomes='Recovery',
            time_frame='6 months',
            study_types='RCT'
        )
        
        errors = pico.validate()
        assert len(errors) == 0
    
    def test_invalid_pico(self):
        """Test invalid PICO criteria."""
        pico = PICOCriteria(
            population='',  # Empty
            intervention='Drug A',
            comparison='',  # Empty
            outcomes='Recovery',
            time_frame='6 months',
            study_types='RCT'
        )
        
        errors = pico.validate()
        assert len(errors) == 2  # population and comparison are empty
