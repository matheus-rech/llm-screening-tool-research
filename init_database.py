#!/usr/bin/env python3
"""
Initialize database tables for the LLM Screening Tool.
"""

import os
import logging
from app import create_app, db

def init_database():
    """Initialize database tables."""
    if not os.environ.get('ANTHROPIC_API_KEY'):
        logging.error("❌ ANTHROPIC_API_KEY environment variable not set")
        return False
    if not os.environ.get('OPENAI_API_KEY'):
        logging.error("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    config_name = os.environ.get('FLASK_CONFIG', 'development')
    app = create_app(config_name)
    
    with app.app_context():
        try:
            db.create_all()
            logging.info("✅ Database tables created successfully")
            
            from app.models.screening_models import Project, Article
            projects = Project.query.all()
            articles = Article.query.all()
            
            logging.info(f"📊 Current database state:")
            logging.info(f"   Projects: {len(projects)}")
            logging.info(f"   Articles: {len(articles)}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Database initialization failed: {e}")
            return False

if __name__ == "__main__":
    init_database()
