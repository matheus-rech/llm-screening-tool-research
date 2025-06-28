#!/usr/bin/env python3
"""
Script to clean mock articles from the database.
This removes the specific mock articles that are contaminating the screening interface.
"""

from app import create_app, db
from app.models.screening_models import Article

def clean_mock_articles():
    """Remove mock articles from the database."""
    app = create_app('development')
    
    with app.app_context():
        mock_titles = [
            'The effect of exercise on depression in adults',
            'Cognitive behavioral therapy for anxiety in children', 
            'The impact of diet on cardiovascular health: a cohort study'
        ]
        
        deleted_count = 0
        for title in mock_titles:
            articles = Article.query.filter_by(title=title).all()
            print(f'Found {len(articles)} articles with title: {title}')
            for article in articles:
                print(f'  Deleting article ID {article.id} from project {article.project_id}')
                db.session.delete(article)
                deleted_count += 1
        
        db.session.commit()
        print(f'Successfully deleted {deleted_count} mock articles from database')
        
        remaining_articles = Article.query.filter_by(project_id=1).all()
        print(f'Remaining articles in project 1: {len(remaining_articles)}')
        
        print('First 3 remaining articles:')
        for i, article in enumerate(remaining_articles[:3]):
            print(f'  {i+1}. {article.title[:60]}...')

if __name__ == "__main__":
    clean_mock_articles()
