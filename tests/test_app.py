import pytest
import json
from app.models.screening_models import Project, Article
from app import create_app, db

class TestRoutes:
    """Test Flask application routes."""
    
    def test_index_route(self, client):
        """Test the main index route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'LLM Screening Tool' in response.data
    
    def test_create_project_route(self, client):
        """Test project creation route."""
        response = client.get('/create_project')
        assert response.status_code == 200
        assert b'Create New Project' in response.data
    
    def test_project_creation(self, client):
        """Test actual project creation."""
        project_data = {
            'name': 'Test Project',
            'description': 'Test Description',
            'population': 'Adults',
            'intervention': 'Drug A',
            'comparison': 'Placebo',
            'outcomes': 'Blood pressure',
            'time_frame': '3 months',
            'study_types': 'RCT'
        }
        
        response = client.post('/create_project', data=project_data)
        assert response.status_code == 302  # Redirect after creation
        
        # Verify project was created
        project = Project.query.filter_by(name='Test Project').first()
        assert project is not None
        assert project.config is not None
        assert project.config.get('pico', {}).get('population') == 'Adults'

class TestModels:
    """Test database models."""
    
    def test_project_model(self, client):
        """Test Project model creation and attributes."""
        with client.application.app_context():
            project = Project(
                name="Test Project",
                description="Test Description",
                config={
                    'pico': {
                        'population': 'Test Population',
                        'intervention': 'Test Intervention',
                        'comparison': 'Test Comparison',
                        'outcomes': 'Test Outcomes',
                        'time_frame': 'Test Time',
                        'study_types': 'Test Types'
                    }
                }
            )
            db.session.add(project)
            db.session.commit()
            
            assert project.id is not None
            assert project.name == "Test Project"
            assert project.config['pico']['population'] == "Test Population"
    
    def test_article_model(self, app):
        """Test Article model creation and relationships."""
        with app.app_context():
            project = Project(
                name="Test Project for Article",
                description="Test Description",
                config={'pico': {'population': 'Test Population'}}
            )
            db.session.add(project)
            db.session.commit()
            
            article = Article(
                title="Test Article",
                abstract="Test Abstract",
                project_id=project.id,
                status="pending"
            )
            db.session.add(article)
            db.session.commit()
            
            assert article.id is not None
            assert article.project_id == project.id
            assert article.status == "pending"
