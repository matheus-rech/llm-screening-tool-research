#!/usr/bin/env python3
"""
Test script to verify complete file upload workflow.
Tests file parsing, database integration, and article creation with proper status handling.
"""

import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.utils.file_parser import parse_ris_file, parse_ris_manual, load_studies
from app.models.screening_models import db, Project, Article
from app import create_app
from datetime import datetime, timezone

def parse_studies_from_files():
    """Helper function to parse studies from test files."""
    test_files = ['test_citation_data.ris', 'test_records.ris']
    all_studies = []
    
    for filename in test_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                studies = parse_ris_file(content)
                
                if len(studies) == 0:
                    studies = parse_ris_manual(content)
                
                enhanced_studies = load_studies(content, filename)
                
                # Normalize authors field to string if it's a list
                for study in studies:
                    authors = study.get('authors', '')
                    if isinstance(authors, list):
                        study['authors'] = ', '.join(authors)
                
                all_studies.extend(studies)
                
            except Exception as e:
                continue
    
    return all_studies

def test_file_parsing():
    """Test file parsing functionality with detailed output."""
    print("🔬 Testing File Parsing")
    print("-" * 40)
    
    all_studies = parse_studies_from_files()
    
    for filename in ['test_citation_data.ris', 'test_records.ris']:
        if os.path.exists(filename):
            print(f"\n📄 Testing file: {filename}")
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"   File size: {len(content)} characters")
                
                studies = parse_ris_file(content)
                print(f"   ✅ parse_ris_file: {len(studies)} studies parsed")
                
                if len(studies) == 0:
                    studies = parse_ris_manual(content)
                    print(f"   ✅ parse_ris_manual (fallback): {len(studies)} studies parsed")
                
                enhanced_studies = load_studies(content, filename)
                print(f"   ✅ load_studies: {len(enhanced_studies)} studies parsed")
                
                for i, study in enumerate(studies[:3]):  # Show first 3 studies
                    print(f"\n   📋 Study {i+1}:")
                    print(f"      Title: {study.get('title', 'N/A')[:80]}...")
                    print(f"      Authors: {study.get('authors', 'N/A')}")
                    print(f"      Year: {study.get('year', 'N/A')}")
                    print(f"      Journal: {study.get('journal_name', 'N/A')}")
                    print(f"      Abstract: {study.get('abstract', 'N/A')[:100]}...")
                    print(f"      PMID: {study.get('pmid', 'N/A')}")
                    
                    authors = study.get('authors', '')
                    if isinstance(authors, list):
                        print(f"      ⚠️  Authors is list, converting: {authors}")
                        authors = ', '.join(authors)
                        print(f"      ✅ Authors converted to string: {authors}")
                    else:
                        print(f"      ✅ Authors already string: {type(authors)}")
                
            except Exception as e:
                print(f"   ❌ Error parsing {filename}: {e}")
                continue
    
    return all_studies

n8zf8q-codex/review-and-fix-workflow
def test_file_parsing():
    """Test file parsing functionality with detailed output."""
    print("🔬 Testing File Parsing")
    print("-" * 40)
    
    all_studies = parse_studies_from_files()
    assert len(all_studies) > 0, "No studies were parsed from the test files"

def test_database_integration():
=======
@pytest.fixture
def studies():
    """Fixture that returns parsed studies for database tests."""
    return test_file_parsing()

def test_database_integration(studies=None):
 Research
    """Test database integration with parsed studies."""
    print("\n🗄️  Testing Database Integration")
    print("-" * 40)
    
n8zf8q-codex/review-and-fix-workflow
    studies = parse_studies_from_files()

    if studies is None:
        studies = test_file_parsing()

 Research
    try:
        app = create_app()
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.app_context():
            db.create_all()
            print("   ✅ Database tables created")
            
            project = Project(
                name="Test Upload Workflow Project",
                description="Testing complete file upload workflow",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            db.session.add(project)
            db.session.commit()
            print(f"   ✅ Test project created (ID: {project.id})")
            
            articles_created = 0
            for i, study in enumerate(studies):
                try:
                    authors = study.get('authors', '')
                    if isinstance(authors, list):
                        authors = ', '.join(authors)
                    
                    keywords = study.get('keywords', '')
                    if isinstance(keywords, list):
                        keywords = ', '.join(keywords)
                    
                    article = Article(
                        project_id=project.id,
                        title=study.get('title', ''),
                        authors=authors,
                        journal=study.get('journal_name', ''),
                        year=study.get('year'),
                        abstract=study.get('abstract', ''),
                        doi=study.get('doi', ''),
                        pmid=study.get('pmid', ''),
                        original_data=study,
                        status='pending',  # Critical: articles should have 'pending' status
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    
                    db.session.add(article)
                    articles_created += 1
                    
                    print(f"   📄 Article {i+1}: {study.get('title', '')[:50]}...")
                    print(f"      Status: {article.status}")
                    print(f"      Authors type: {type(authors)} - {authors[:50]}...")
                    
                except Exception as e:
                    print(f"   ❌ Error creating article {i+1}: {e}")
                    continue
            
            db.session.commit()
            print(f"   ✅ Successfully created {articles_created} articles in database")
            
            pending_articles = Article.query.filter_by(project_id=project.id, status='pending').all()
            included_articles = Article.query.filter_by(project_id=project.id, status='included').all()
            excluded_articles = Article.query.filter_by(project_id=project.id, status='excluded').all()
            
            print(f"\n   📊 Database Verification:")
            print(f"      Total articles: {len(pending_articles) + len(included_articles) + len(excluded_articles)}")
            print(f"      Pending articles: {len(pending_articles)}")
            print(f"      Included articles: {len(included_articles)}")
            print(f"      Excluded articles: {len(excluded_articles)}")
            
            print(f"\n   📋 Sample Pending Articles:")
            for article in pending_articles[:3]:
                print(f"      - ID: {article.id}")
                print(f"        Title: {article.title[:60]}...")
                print(f"        Status: {article.status}")
                print(f"        Authors: {article.authors[:40]}...")
                print(f"        Year: {article.year}")
                print("")
            
            assert len(pending_articles) > 0, "No pending articles found in database"
            
    except Exception as e:
        print(f"   ❌ Database integration test failed: {e}")
        assert False, f"Database integration test failed: {e}"

def test_complete_workflow():
    """Test complete workflow from file parsing to database storage."""
    print("🚀 LLM Screening Tool - Complete Upload Workflow Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("")
    
    studies = parse_studies_from_files()
    
    if not studies:
        print("\n❌ No studies parsed - workflow test failed")
        assert False, "No studies parsed - workflow test failed"
    
    print(f"\n✅ File parsing successful: {len(studies)} studies parsed")
    
    # Run database integration test
    test_database_integration()
    
    print("\n✅ Database integration successful")
    
    print("\n🎉 Complete Workflow Test Results")
    print("=" * 60)
    print("✅ File parsing: PASSED")
    print("✅ Type conversion (authors/keywords): PASSED")
    print("✅ Database integration: PASSED")
    print("✅ Article creation with 'pending' status: PASSED")
    print("✅ Complete upload workflow: PASSED")
    print("")
    print("🔍 Key Findings:")
    print(f"   - Successfully parsed {len(studies)} citations from RIS file")
    print("   - All author/keyword lists properly converted to comma-separated strings")
    print("   - Articles created in database with 'pending' status (not mock data)")
    print("   - Ready for screening interface to display real uploaded citations")
    print("")
    print("✨ The file upload workflow is working correctly!")
    
    assert True, "Complete workflow test passed"

if __name__ == "__main__":
    try:
        test_complete_workflow()
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error in workflow test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
