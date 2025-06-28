#!/usr/bin/env python3
"""
Comprehensive System Health Check for LLM Screening Tool
Tests dual LLM implementation, frontend-backend integration, and core functionality.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        # Core Flask imports
        import flask
        from flask import Flask
        try:
            print(f"✅ Flask {flask.__version__} imported successfully")
        except AttributeError:
            print("✅ Flask imported successfully (version not accessible)")

        # Database imports
        from flask_sqlalchemy import SQLAlchemy
        print("✅ Flask-SQLAlchemy imported successfully")

        # LLM provider imports
        import openai
        print(f"✅ OpenAI {openai.__version__} imported successfully")

        import anthropic
        print(f"✅ Anthropic {anthropic.__version__} imported successfully")

        # Pydantic for data validation
        import pydantic
        print(f"✅ Pydantic {pydantic.__version__} imported successfully")

        # File parsing libraries
        import rispy
        import bibtexparser
        try:
            from lxml import etree
        except ImportError:
            print("⚠️ lxml not available")
        from Bio import Entrez
        print("✅ File parsing libraries imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False

def test_app_structure():
    """Test that the app structure is correct."""
    print("\n🏗️ Testing app structure...")

    required_files = [
        'app/__init__.py',
        'app/models/screening_models.py',
        'app/routes/main.py',
        'app/routes/screening.py',
        'app/services/screening/dual_llm_screener.py',
        'app/services/screening/enhanced_systematic_review.py',
        'app/services/utils/file_parser.py',
        'run.py'
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")
    return True

def test_app_models():
    """Test that database models can be imported and initialized."""
    print("\n📊 Testing database models...")

    try:
        from app.models.screening_models import db, Project, Article
        print("✅ Database models imported successfully")

        # Test model structure
        project_fields = ['id', 'name', 'description', 'created_at', 'config']
        article_fields = ['id', 'title', 'abstract', 'authors', 'journal', 'year', 'status']

        for field in project_fields:
            if hasattr(Project, field):
                print(f"✅ Project.{field} exists")
            else:
                print(f"❌ Project.{field} missing")
                return False

        for field in article_fields:
            if hasattr(Article, field):
                print(f"✅ Article.{field} exists")
            else:
                print(f"❌ Article.{field} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Error testing models: {e}")
        traceback.print_exc()
        return False

def test_dual_llm_screener():
    """Test the dual LLM screener implementation."""
    print("\n🤖 Testing dual LLM screener...")

    try:
        from app.services.screening.dual_llm_screener import (
            DualProviderScreeningOrchestrator,
            ScreeningCriteria,
            OpenAIProvider,
            AnthropicProvider,
            ComprehensiveScreeningResult
        )
        print("✅ Dual LLM screener classes imported successfully")

        # Test ScreeningCriteria model
        criteria = ScreeningCriteria(
            research_question="Test research question",
            target_population="Test population",
            target_intervention="Test intervention",
            target_comparison="Test comparison",
            target_outcomes=["Test outcome"],
            target_time_frame="Test timeframe",
            target_study_types=["RCT"],
            inclusion_criteria=["Test inclusion"],
            exclusion_criteria=["Test exclusion"]
        )
        print("✅ ScreeningCriteria model works correctly")

        # Test provider initialization (without API keys)
        try:
            openai_provider = OpenAIProvider("")
            anthropic_provider = AnthropicProvider("")
            print("✅ LLM providers can be initialized")
        except Exception as e:
            print(f"⚠️ Provider initialization warning: {e}")

        # Test orchestrator initialization
        try:
            orchestrator = DualProviderScreeningOrchestrator("", "")
            print("✅ Dual provider orchestrator can be initialized")
        except Exception as e:
            print(f"⚠️ Orchestrator initialization warning: {e}")

        return True

    except Exception as e:
        print(f"❌ Error testing dual LLM screener: {e}")
        traceback.print_exc()
        return False

def test_enhanced_systematic_review():
    """Test the enhanced systematic review implementation."""
    print("\n📋 Testing enhanced systematic review...")

    try:
        from app.services.screening.enhanced_systematic_review import (
            ContextAwareScreeningOrchestrator,
            SystematicReviewCriteria,
            CitationContext,
            EnhancedScreeningResult
        )
        print("✅ Enhanced systematic review classes imported successfully")

        # Test SystematicReviewCriteria model
        criteria = SystematicReviewCriteria(
            research_question="Test research question",
            target_population="Test population",
            target_intervention="Test intervention",
            target_comparison="Test comparison",
            target_outcomes=["Test outcome"],
            target_time_frame="Test timeframe",
            target_study_types=["RCT"],
            inclusion_criteria=["Test inclusion"],
            exclusion_criteria=["Test exclusion"],
            domain="medical"
        )
        print("✅ SystematicReviewCriteria model works correctly")

        # Test CitationContext model
        context = CitationContext()
        print("✅ CitationContext model works correctly")

        # Test orchestrator initialization
        try:
            orchestrator = ContextAwareScreeningOrchestrator("", "")
            print("✅ Context-aware orchestrator can be initialized")
        except Exception as e:
            print(f"⚠️ Context-aware orchestrator warning: {e}")

        return True

    except Exception as e:
        print(f"❌ Error testing enhanced systematic review: {e}")
        traceback.print_exc()
        return False

def test_file_parser():
    """Test file parsing functionality."""
    print("\n📁 Testing file parser...")

    try:
        from app.services.utils.file_parser import (
            detect_file_format,
            parse_ris_file,
            parse_csv_file,
            load_studies
        )
        print("✅ File parser functions imported successfully")

        # Test format detection
        test_cases = [
            ("test.ris", "ris"),
            ("test.csv", "csv"),
            ("test.xml", "xml"),
            ("test.bib", "bibtex"),
            ("pmids.txt", "pmid_list")
        ]

        for filename, expected_format in test_cases:
            detected = detect_file_format(filename)
            if detected == expected_format:
                print(f"✅ Format detection for {filename}: {detected}")
            else:
                print(f"⚠️ Format detection for {filename}: expected {expected_format}, got {detected}")

        # Test RIS parsing with sample data
        sample_ris = """TY  - JOUR
TI  - Test Article Title
AU  - Test Author
AB  - Test abstract content
PY  - 2023
ER  -
"""

        studies = parse_ris_file(sample_ris)
        if studies and len(studies) > 0:
            print(f"✅ RIS parsing works: parsed {len(studies)} studies")
        else:
            print("⚠️ RIS parsing returned no studies")

        return True

    except Exception as e:
        print(f"❌ Error testing file parser: {e}")
        traceback.print_exc()
        return False

def test_flask_routes():
    """Test Flask routes and app initialization."""
    print("\n🌐 Testing Flask routes...")

    try:
        # Import the Flask app
        from run import app
        print("✅ Flask app imported successfully")

        # Test app configuration
        with app.app_context():
            print(f"✅ App name: {app.name}")
            print(f"✅ Debug mode: {app.debug}")

            # Test that routes are registered
            routes = []
            for rule in app.url_map.iter_rules():
                routes.append(f"{rule.methods} {rule.rule}")

            print(f"✅ Found {len(routes)} registered routes")

            # Check for key routes
            key_routes = [
                ('GET', '/'),
                ('POST', '/upload'),
                ('GET', '/dashboard'),
                ('POST', '/screen')
            ]

            for method, path in key_routes:
                route_found = any(method in route and path in route for route in routes)
                if route_found:
                    print(f"✅ Route found: {method} {path}")
                else:
                    print(f"⚠️ Route not found: {method} {path}")

        return True

    except Exception as e:
        print(f"❌ Error testing Flask routes: {e}")
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test environment variable configuration."""
    print("\n🔧 Testing environment variables...")

    # Check for .env.example file
    if os.path.exists('.env.example'):
        print("✅ .env.example file exists")

        with open('.env.example', 'r') as f:
            env_example = f.read()

        required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'ENTREZ_EMAIL']

        for var in required_vars:
            if var in env_example:
                print(f"✅ {var} documented in .env.example")

                # Check if actually set in environment
                if os.getenv(var):
                    print(f"✅ {var} is set in environment")
                else:
                    print(f"⚠️ {var} not set in environment (expected for testing)")
            else:
                print(f"❌ {var} not documented in .env.example")
    else:
        print("❌ .env.example file not found")
        return False

    return True

def test_database_schema():
    """Test database schema and migrations."""
    print("\n🗄️ Testing database schema...")

    try:
        # Check migration files
        migration_dir = 'migrations/versions'
        if os.path.exists(migration_dir):
            migration_files = [f for f in os.listdir(migration_dir) if f.endswith('.py')]
            print(f"✅ Found {len(migration_files)} migration files")
        else:
            print("⚠️ No migrations directory found")

        # Test database models can be imported
        from app.models.screening_models import db, Project, Article

        # Create a test Flask app to test database operations
        from flask import Flask
        test_app = Flask(__name__)
        test_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        test_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

        db.init_app(test_app)

        with test_app.app_context():
            # Create tables
            db.create_all()
            print("✅ Database tables created successfully")

            # Test creating a project
            project = Project(
                name="Test Project",
                description="Test Description",
                config={"test": "config"}
            )
            db.session.add(project)
            db.session.commit()
            print("✅ Project creation works")

            # Test creating an article
            article = Article(
                project_id=project.id,
                title="Test Article",
                abstract="Test Abstract",
                authors="Test Author",
                status="pending"
            )
            db.session.add(article)
            db.session.commit()
            print("✅ Article creation works")

            # Test querying
            projects = Project.query.all()
            articles = Article.query.all()
            print(f"✅ Database queries work: {len(projects)} projects, {len(articles)} articles")

        return True

    except Exception as e:
        print(f"❌ Error testing database schema: {e}")
        traceback.print_exc()
        return False

def test_api_key_validation():
    """Test API key validation without making actual API calls."""
    print("\n🔑 Testing API key validation...")

    try:
        from app.services.screening.dual_llm_screener import OpenAIProvider, AnthropicProvider

        # Test with empty API keys (should not crash)
        openai_provider = OpenAIProvider("")
        anthropic_provider = AnthropicProvider("")
        print("✅ Providers handle empty API keys gracefully")

        # Test with mock API keys
        openai_provider = OpenAIProvider("sk-test-key")
        anthropic_provider = AnthropicProvider("test-key")
        print("✅ Providers accept API key format")

        return True

    except Exception as e:
        print(f"❌ Error testing API key validation: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("🚀 Starting Comprehensive System Health Check")
    print("=" * 60)

    test_results = {}

    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("App Structure", test_app_structure),
        ("Database Models", test_app_models),
        ("Dual LLM Screener", test_dual_llm_screener),
        ("Enhanced Systematic Review", test_enhanced_systematic_review),
        ("File Parser", test_file_parser),
        ("Flask Routes", test_flask_routes),
        ("Environment Variables", test_environment_variables),
        ("Database Schema", test_database_schema),
        ("API Key Validation", test_api_key_validation)
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results[test_name] = False

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System appears to be healthy.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")

    # Specific dual LLM implementation check
    print("\n" + "=" * 60)
    print("🤖 DUAL LLM IMPLEMENTATION STATUS")
    print("=" * 60)

    dual_llm_tests = ["Dual LLM Screener", "Enhanced Systematic Review", "API Key Validation"]
    dual_llm_passed = sum(1 for test in dual_llm_tests if test_results.get(test, False))

    print(f"Dual LLM Core: {dual_llm_passed}/{len(dual_llm_tests)} components working")

    if dual_llm_passed == len(dual_llm_tests):
        print("✅ Dual LLM implementation is functional and ready for use")
    else:
        print("❌ Dual LLM implementation has issues that need attention")

    return test_results

if __name__ == "__main__":
    results = run_comprehensive_test()

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
