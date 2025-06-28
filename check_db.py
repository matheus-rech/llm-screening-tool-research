import sqlite3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    conn = sqlite3.connect('instance/screening_projects.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Available tables:", [table[0] for table in tables])
    
    if any('project' in table[0].lower() for table in tables):
        cursor.execute("SELECT COUNT(*) FROM project;")
        project_count = cursor.fetchone()[0]
        print(f"Number of projects: {project_count}")
        
        if project_count > 0:
            cursor.execute("SELECT id, name, description FROM project LIMIT 5;")
            projects = cursor.fetchall()
            print("Sample projects:")
            for project in projects:
                print(f"  ID: {project[0]}, Name: {project[1]}, Description: {project[2]}")
    
    if any('article' in table[0].lower() for table in tables):
        cursor.execute("SELECT COUNT(*) FROM article;")
        article_count = cursor.fetchone()[0]
        print(f"Number of articles: {article_count}")
        
        if article_count > 0:
            cursor.execute("SELECT id, project_id, title, decision_reasoning FROM article WHERE decision_reasoning IS NOT NULL LIMIT 3;")
            articles = cursor.fetchall()
            print("Sample articles with screening results:")
            for article in articles:
                print(f"  ID: {article[0]}, Project: {article[1]}, Title: {article[2][:50]}...")
                print(f"    Has decision_reasoning: {'Yes' if article[3] else 'No'}")
    
    conn.close()
    
except Exception as e:
    print(f"Error checking database: {e}")
