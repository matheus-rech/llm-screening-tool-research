import pytest
import tempfile
import os
from app import create_app, db
from app.models.screening_models import Project, Article

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    db_fd, db_path = tempfile.mkstemp()
    app = create_app({
        'TESTING': True, 
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'WTF_CSRF_ENABLED': False
    })
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()
    
    os.close(db_fd)
    os.unlink(db_path)

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test CLI runner for the app."""
    return app.test_cli_runner()

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    os.unlink(path)

@pytest.fixture
def sample_ris_content():
    """Sample RIS content for testing."""
    return """TY  - JOUR
AU  - Smith, John
TI  - Test Article
JO  - Test Journal
PY  - 2023
AB  - This is a test abstract for screening.
ER  - 

"""

@pytest.fixture
def sample_project(app):
    """Create a sample project for testing."""
    with app.app_context():
        project = Project(
            name="Test Project",
            description="A test project for systematic review",
            config={
                'pico': {
                    'population': 'Adults with diabetes',
                    'intervention': 'Exercise therapy',
                    'comparison': 'Standard care',
                    'outcomes': 'Blood glucose levels',
                    'time_frame': '6 months',
                    'study_types': 'RCT, cohort studies'
                }
            }
        )
        db.session.add(project)
        db.session.commit()
        return project

@pytest.fixture
def sample_article(app, sample_project):
    """Create a sample article for testing."""
    with app.app_context():
        article = Article(
            title="Test Article Title",
            abstract="This is a test abstract for screening purposes.",
            authors="Smith, J.; Doe, A.",
            journal="Test Journal",
            year="2023",
            project_id=sample_project.id
        )
        db.session.add(article)
        db.session.commit()
        return article
