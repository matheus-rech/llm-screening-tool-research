from app import create_app, db
from app.models.screening_models import Project

app = create_app()
with app.app_context():
    projects = Project.query.all()
    print(f'Total projects in database: {len(projects)}')
    
    for project in projects:
        print(f'Project {project.id}: "{project.name}" - Status: {project.status}')
        print(f'  Description: {project.description}')
        print(f'  Created: {project.created_at}')
        print(f'  Updated: {project.updated_at}')
        print()
    
    project_4 = Project.query.get(4)
    if project_4:
        print(f'Project 4 exists: {project_4.name}')
    else:
        print('Project 4 does not exist in database')
