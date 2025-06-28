#!/usr/bin/env python3
"""
Deployment health check script to verify all components are working correctly.
This script helps identify deployment issues and ensures proper environment setup.
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment_variables():
    """Check if required environment variables are set"""
    logger.info("Checking environment variables...")
    
    required_vars = ['FLASK_ENV', 'SECRET_KEY']
    optional_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'ENTREZ_EMAIL']
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
        else:
            logger.info(f"✅ {var} is set")
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
        else:
            logger.info(f"✅ {var} is set")
    
    if missing_required:
        logger.error(f"❌ Missing required environment variables: {missing_required}")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️ Missing optional environment variables: {missing_optional}")
    
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("Checking dependencies...")
    
    required_modules = [
        'flask', 'flask_sqlalchemy', 'flask_migrate',
        'openai', 'anthropic', 'pandas', 'pydantic',
        'rispy', 'bibtexparser', 'lxml', 'biopython',
        'openpyxl', 'tiktoken', 'aiohttp'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} available")
        except ImportError:
            logger.error(f"❌ {module} missing")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

def check_app_creation():
    """Check if Flask app can be created successfully"""
    logger.info("Checking Flask app creation...")
    
    try:
        from app import create_app
        app = create_app()
        logger.info("✅ Flask app created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Flask app creation failed: {e}")
        return False

def check_database_models():
    """Check if database models can be imported"""
    logger.info("Checking database models...")
    
    try:
        from app.models.screening_models import Project, Article
        logger.info("✅ Database models imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Database models import failed: {e}")
        return False

def check_llm_providers():
    """Check if LLM provider classes can be imported and instantiated"""
    logger.info("Checking LLM providers...")
    
    try:
        from app.services.screening.dual_llm_screener import ModelConfig, DualProviderScreeningOrchestrator
        
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.7
        )
        logger.info("✅ ModelConfig created successfully")
        
        from app.services.utils.dual_llm_comparison_exporter import DualLLMComparisonExporter
        logger.info("✅ DualLLMComparisonExporter imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ LLM provider check failed: {e}")
        return False

def main():
    """Run all health checks"""
    logger.info("=== Deployment Health Check ===")
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("Dependencies", check_dependencies),
        ("Flask App Creation", check_app_creation),
        ("Database Models", check_database_models),
        ("LLM Providers", check_llm_providers)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {check_name} check failed with exception: {e}")
            all_passed = False
    
    logger.info(f"\n=== Health Check Results ===")
    if all_passed:
        logger.info("✅ All health checks passed - deployment should be successful")
        sys.exit(0)
    else:
        logger.error("❌ Some health checks failed - deployment may have issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
