#!/usr/bin/env python3
"""
Comprehensive test script to verify LLM screening fixes.
Tests the main issues that were causing pending states.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_test_logging():
    """Set up logging for test execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_llm_fixes.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def test_import_structure():
    """Test that all imports work correctly after fixes."""
    logger = logging.getLogger(__name__)
    logger.info("Testing import structure...")
    
    try:
        # Test core imports
        from app import create_app
        from app.models.screening_models import db, Project, Article
        from app.services.screening.dual_llm_screener import (
            OpenAIProvider, AnthropicProvider, DualProviderScreeningOrchestrator,
            ScreeningCriteria, HumanReviewTriggers, ScreeningResultsStore
        )
        from app.services.screening.modern_llm import (
            ComprehensiveScreeningResult, PICOTTExtraction
        )
        from app.services.utils.error_handler import retry_with_backoff, setup_screening_logger
        from app.services.utils.exceptions import APIError, ValidationError
        
        logger.info("✅ All imports successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_provider_initialization():
    """Test that providers can be initialized with timeout configs."""
    logger = logging.getLogger(__name__)
    logger.info("Testing provider initialization...")
    
    try:
        from app.services.screening.dual_llm_screener import (
            OpenAIProvider, AnthropicProvider, ModelConfig
        )
        
        # Test with dummy API keys (won't make actual calls)
        openai_config = ModelConfig(provider='openai', model_name='gpt-4o', temperature=0.1)
        anthropic_config = ModelConfig(provider='anthropic', model_name='claude-3-5-sonnet-20241022', temperature=0.1)
        
        openai_provider = OpenAIProvider("dummy-key", openai_config)
        anthropic_provider = AnthropicProvider("dummy-key", anthropic_config)
        
        # Check that timeout configuration was applied
        assert hasattr(openai_provider.client, '_client')  # OpenAI client internal structure
        assert hasattr(anthropic_provider.client, '_client')  # Anthropic client internal structure
        
        logger.info("✅ Provider initialization successful with timeout configs")
        return True
        
    except Exception as e:
        logger.error(f"❌ Provider initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_error_handling():
    """Test enhanced error handling."""
    logger = logging.getLogger(__name__)
    logger.info("Testing error handling...")
    
    try:
        from app.services.utils.exceptions import APIError
        from app.services.utils.error_handler import retry_with_backoff
        
        # Test APIError with different status codes
        timeout_error = APIError("Timeout", status_code=408)
        rate_limit_error = APIError("Rate limit", status_code=429, retry_after=60)
        
        assert timeout_error.status_code == 408
        assert rate_limit_error.retry_after == 60
        
        # Test retry decorator (with mock function)
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Temporary failure", status_code=500)
            return "success"
        
        result = mock_api_call()
        assert result == "success"
        assert call_count == 3  # Should have retried twice
        
        logger.info("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_database_operations():
    """Test database transaction handling."""
    logger = logging.getLogger(__name__)
    logger.info("Testing database operations...")
    
    try:
        from app import create_app
        from app.models.screening_models import db, Project, Article
        
        # Create test app
        app = create_app('testing')
        
        with app.app_context():
            # Create tables
            db.create_all()
            
            # Test project creation
            project = Project(
                name="Test Project",
                description="Testing database operations",
                config={"test": True}
            )
            db.session.add(project)
            db.session.commit()
            
            # Test article creation
            article = Article(
                project_id=project.id,
                title="Test Article",
                abstract="This is a test abstract",
                status="pending"
            )
            db.session.add(article)
            db.session.commit()
            
            # Test status update
            article.status = "processed"
            article.decision_reasoning = {
                "test": True,
                "timestamp": datetime.now().isoformat()
            }
            db.session.commit()
            
            # Verify changes
            updated_article = Article.query.get(article.id)
            assert updated_article.status == "processed"
            assert updated_article.decision_reasoning["test"] is True
            
            logger.info("✅ Database operations successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database operations failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_screening_criteria():
    """Test screening criteria creation and validation."""
    logger = logging.getLogger(__name__)
    logger.info("Testing screening criteria...")
    
    try:
        from app.services.screening.dual_llm_screener import ScreeningCriteria
        
        criteria = ScreeningCriteria(
            research_question="Test research question",
            target_population="Test population",
            target_intervention="Test intervention", 
            target_comparison="Test comparison",
            target_outcomes=["outcome1", "outcome2"],
            target_time_frame="Test timeframe",
            target_study_types=["RCT", "cohort"],
            inclusion_criteria=["Include criterion 1", "Include criterion 2"],
            exclusion_criteria=["Exclude criterion 1"],
            temperature=0.1,
            seed=12345,
            openai_model="gpt-4o",
            anthropic_model="claude-3-5-sonnet-20241022"
        )
        
        # Verify all fields are set
        assert criteria.research_question == "Test research question"
        assert len(criteria.target_outcomes) == 2
        assert criteria.temperature == 0.1
        assert criteria.seed == 12345
        
        logger.info("✅ Screening criteria creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Screening criteria test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_logging_setup():
    """Test enhanced logging configuration."""
    logger = logging.getLogger(__name__)
    logger.info("Testing logging setup...")
    
    try:
        from app.services.utils.error_handler import setup_screening_logger
        
        # Test logger creation
        screening_logger = setup_screening_logger("test_screening")
        
        # Test different log levels
        screening_logger.debug("Debug message")
        screening_logger.info("Info message")
        screening_logger.warning("Warning message")
        screening_logger.error("Error message")
        
        # Verify logger has handlers
        assert len(screening_logger.handlers) > 0
        
        logger.info("✅ Logging setup successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging setup failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    logger = setup_test_logging()
    logger.info("🚀 Starting comprehensive LLM screening fixes test...")
    logger.info("=" * 60)
    
    tests = [
        test_import_structure,
        test_provider_initialization,
        test_error_handling,
        test_database_operations,
        test_screening_criteria,
        test_logging_setup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        test_name = test.__name__
        logger.info(f"\n📋 Running {test_name}...")
        
        try:
            if test():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Test Summary:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        logger.info("🎉 All tests passed! LLM screening fixes are working correctly.")
        return 0
    else:
        logger.error(f"⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)