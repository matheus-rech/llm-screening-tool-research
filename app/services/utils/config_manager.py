"""Enhanced configuration management with validation and templates."""

import json
import os
import copy
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import jsonschema
from datetime import datetime
from app.services.utils.exceptions import ConfigurationError, ValidationError

@dataclass
class PICOCriteria:
    """PICO criteria configuration."""
    population: str
    intervention: str
    comparison: str
    outcomes: str
    time_frame: str = ""
    study_types: str = ""
    
    def validate(self) -> List[str]:
        """Validate PICO criteria and return list of errors."""
        errors = []
        
        required_fields = ['population', 'intervention', 'comparison', 'outcomes']
        for field in required_fields:
            value = getattr(self, field)
            if not value or not value.strip():
                errors.append(f"{field.capitalize()} is required")
        
        return errors

@dataclass
class APIConfiguration:
    """API configuration settings."""
    api_key: str
    conservative_model: str = "gpt-4o-mini"
    liberal_model: str = "gpt-3.5-turbo"
    resolver_model: str = "gpt-4"
    max_retries: int = 3
    timeout: int = 30
    max_tokens: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate API configuration."""
        errors = []
        
        if not self.api_key or not self.api_key.strip():
            errors.append("API key is required")
        
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")
            
        if self.timeout <= 0:
            errors.append("Timeout must be positive")
            
        return errors

@dataclass
class ProcessingConfiguration:
    """Processing configuration settings."""
    batch_size: int = 10
    max_workers: int = 5
    rate_limit_per_minute: int = 60
    enable_async: bool = False
    save_intermediate_results: bool = True
    
    def validate(self) -> List[str]:
        """Validate processing configuration."""
        errors = []
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
            
        if self.max_workers <= 0:
            errors.append("Max workers must be positive")
            
        if self.rate_limit_per_minute <= 0:
            errors.append("Rate limit must be positive")
            
        return errors

@dataclass
class ProjectConfiguration:
    """Complete project configuration."""
    name: str
    description: str
    pico: PICOCriteria
    api: APIConfiguration
    processing: ProcessingConfiguration
    created_at: str
    modified_at: str
    version: str = "1.0"
    
    def validate(self) -> List[str]:
        """Validate entire configuration."""
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("Project name is required")
            
        errors.extend(self.pico.validate())
        errors.extend(self.api.validate())
        errors.extend(self.processing.validate())
        
        return errors

class ConfigurationManager:
    """Enhanced configuration management system."""
    
    # Default values for PICO criteria fields
    PICO_DEFAULTS = {
        'population': '',
        'intervention': '',
        'comparison': '',
        'outcomes': '',
        'time_frame': '',
        'study_types': ''
    }
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "version": {"type": "string"},
            "created_at": {"type": "string"},
            "modified_at": {"type": "string"},
            "pico": {
                "type": "object",
                "properties": {
                    "population": {"type": "string", "minLength": 1},
                    "intervention": {"type": "string", "minLength": 1},
                    "comparison": {"type": "string", "minLength": 1},
                    "outcomes": {"type": "string", "minLength": 1},
                    "time_frame": {"type": "string"},
                    "study_types": {"type": "string"}
                },
                "required": ["population", "intervention", "comparison", "outcomes"]
            },
            "api": {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "minLength": 1},
                    "conservative_model": {"type": "string"},
                    "liberal_model": {"type": "string"},
                    "resolver_model": {"type": "string"},
                    "max_retries": {"type": "integer", "minimum": 0},
                    "timeout": {"type": "integer", "minimum": 1},
                    "max_tokens": {"type": ["integer", "null"], "minimum": 1}
                }
                # api_key optional for templates; validation ensures requirement
            },
            "processing": {
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "minimum": 1},
                    "max_workers": {"type": "integer", "minimum": 1},
                    "rate_limit_per_minute": {"type": "integer", "minimum": 1},
                    "enable_async": {"type": "boolean"},
                    "save_intermediate_results": {"type": "boolean"}
                }
            }
        },
        "required": ["name", "pico", "api"]
    }
    
    def __init__(self, template_dir: str = "config_templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
    
    def create_configuration(self, 
                           name: str,
                           description: str,
                           pico_data: Dict[str, str],
                           api_data: Dict[str, Any],
                           processing_data: Optional[Dict[str, Any]] = None) -> ProjectConfiguration:
        """Create a new configuration with validation."""
        
        if processing_data is None:
            processing_data = {}
        
        now = datetime.now().isoformat()
        
        try:
            pico_kwargs = {**self.PICO_DEFAULTS, **pico_data}
            pico = PICOCriteria(**pico_kwargs)

            api = APIConfiguration(**api_data)
            processing = ProcessingConfiguration(**processing_data)
            
            config = ProjectConfiguration(
                name=name,
                description=description,
                pico=pico,
                api=api,
                processing=processing,
                created_at=now,
                modified_at=now
            )
            
            # Validate configuration
            errors = config.validate()
            if errors:
                raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
            
            return config
            
        except TypeError as e:
Research
            raise ValidationError(f"Invalid configuration data: {e}")
    
    def save_template(self, config: ProjectConfiguration, template_name: str) -> str:
        """Save configuration as a template."""
        
        template_path = self.template_dir / f"{template_name}.json"
        
        # Convert to dict and remove instance-specific data
        config_dict = asdict(config)
        config_dict.pop('created_at', None)
        config_dict.pop('modified_at', None)
        config_dict['api'].pop('api_key', None)  # Don't save API keys in templates
        
        try:
            with open(template_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            return str(template_path)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save template: {e}")
    
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a configuration template."""
        
        template_path = self.template_dir / f"{template_name}.json"
        
        if not template_path.exists():
            raise ConfigurationError(f"Template '{template_name}' not found")
        
        try:
            with open(template_path, 'r') as f:
                template_data = json.load(f)

            # Validate against schema (API key optional for templates)
            template_schema = copy.deepcopy(self.SCHEMA)
            template_schema["properties"]["api"]["required"] = []
            jsonschema.validate(template_data, template_schema)
            return template_data
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in template: {e}")
        except jsonschema.ValidationError as e:
            raise ValidationError(f"Template validation failed: {e.message}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load template: {e}")
    
    def list_templates(self) -> List[str]:
        """List available configuration templates."""
        
        templates = []
        for file_path in self.template_dir.glob("*.json"):
            templates.append(file_path.stem)
        
        return sorted(templates)
    
    def delete_template(self, template_name: str) -> bool:
        """Delete a configuration template."""
        
        template_path = self.template_dir / f"{template_name}.json"
        
        if template_path.exists():
            template_path.unlink()
            return True
        
        return False
    
    def validate_configuration_dict(self, config_dict: Dict[str, Any]) -> List[str]:
        """Validate a configuration dictionary."""
        
        errors = []
        
        try:
            jsonschema.validate(config_dict, self.SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        
        # Additional custom validation
        try:
            pico_data = {**self.PICO_DEFAULTS, **config_dict.get('pico', {})}
            pico = PICOCriteria(**pico_data)
            errors.extend(pico.validate())
        except TypeError as e:
            errors.append(f"Invalid PICO data: {e}")
        
        try:
            api = APIConfiguration(**config_dict.get('api', {}))
            errors.extend(api.validate())
        except TypeError as e:
            errors.append(f"Invalid API data: {e}")
        
        try:
            processing = ProcessingConfiguration(**config_dict.get('processing', {}))
            errors.extend(processing.validate())
        except TypeError as e:
            errors.append(f"Invalid processing data: {e}")
        
        return errors
    
    def merge_with_template(self, template_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge template with override values."""
        
        template_data = self.load_template(template_name)
        
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Deep merge two dictionaries."""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_config = deep_merge(template_data, overrides)
        
        # Validate merged configuration
        errors = self.validate_configuration_dict(merged_config)
        if errors:
            raise ValidationError(f"Merged configuration is invalid: {'; '.join(errors)}")
        
        return merged_config