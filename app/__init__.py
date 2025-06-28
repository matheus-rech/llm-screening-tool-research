"""
Flask Application Factory
Creates and configures the Flask application instance.
"""

import os
import logging
from flask import Flask
from flask_migrate import Migrate

# Import db from models to ensure single instance
from app.models.screening_models import db
migrate = Migrate()

def create_app(config_name=None):
    """Create Flask application using factory pattern."""

    app = Flask(__name__)

    # Configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    if config_name == 'development':
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///screening_projects.db'
        app.config['DEBUG'] = True
    elif config_name == 'production':
        database_url = os.getenv('DATABASE_URL', 'sqlite:///screening_projects.db')
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
        app.config['DEBUG'] = False
    else:  # testing
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['TESTING'] = True

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    app.config['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

    # Configure logging for debugging
    if config_name == 'development':
        logging.basicConfig(level=logging.INFO)
        app.logger.setLevel(logging.INFO)

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)

    # Import models to ensure they're registered with SQLAlchemy
    from app.models import screening_models

    # Register blueprints - MVP ONLY CORE FUNCTIONALITY
    from app.routes.main import main_bp, configure_template_directory
    from app.routes.screening import modern_screening_bp

    # Configure template directory
    configure_template_directory(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(modern_screening_bp)

    # TODO: Re-enable for full version
    # from app.routes.analytics import analytics_bp
    # from app.routes.enhanced import enhanced_bp
    # app.register_blueprint(analytics_bp)
    # app.register_blueprint(enhanced_bp)

    return app
