#!/usr/bin/env python3
"""
Application Entry Point
Runs the Flask application using the factory pattern.
"""

import os
from app import create_app, db

# Create application instance with development config for local testing
app = create_app('development')

# Initialize database tables
with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

if __name__ == '__main__':
    # Run the application
    debug = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('PORT', 5000))
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )
