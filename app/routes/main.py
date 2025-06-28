"""
Main Application Routes
Contains the primary Flask routes from the original app.py
"""

import json
import os
from datetime import datetime
from io import StringIO
import csv
import logging

import pandas as pd
import rispy
import bibtexparser
from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, Response, stream_with_context, jsonify, send_file, session
from werkzeug.utils import secure_filename

from app.models.screening_models import db, Project, Article
from app.services.utils.file_parser import load_studies
from app.services.screening.dual_llm_screener import ModelConfig, DualProviderScreeningOrchestrator

logger = logging.getLogger(__name__)

# Create Blueprint for main routes
main_bp = Blueprint('main', __name__)

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('config_templates', exist_ok=True)

def configure_template_directory(app):
    """Configure the CONFIG_TEMPLATES_DIR for the Flask app."""
    app.config['CONFIG_TEMPLATES_DIR'] = os.path.abspath('config_templates')

@main_bp.route('/')
def dashboard():
    """Main dashboard showing all projects."""
    from sqlalchemy.orm import joinedload
    projects = Project.query.options(joinedload(Project.articles)).order_by(Project.updated_at.desc()).all()
    return render_template('dashboard.html', projects=projects)

@main_bp.route('/create_project', methods=['GET', 'POST'])
def create_project():
    """Create a new screening project."""
    if request.method == 'POST':
        name = request.form['name']
        description = request.form.get('description', '')
        
        pico_config = {
            'population': request.form.get('population', ''),
            'intervention': request.form.get('intervention', ''),
            'comparison': request.form.get('comparison', ''),
            'outcomes': request.form.get('outcomes', ''),
            'time_frame': request.form.get('time_frame', ''),
            'study_types': request.form.get('study_types', '')
        }
        
        project = Project(
            name=name,
            description=description,
            status='active',
            config={'pico': pico_config}
        )
        
        db.session.add(project)
        db.session.commit()
        
        return redirect(url_for('main.project_detail', project_id=project.id))
    
    return render_template('create_project.html')

@main_bp.route('/project/<int:project_id>')
def project_detail(project_id):
    """Show project details and articles."""
    project = Project.query.get_or_404(project_id)
    articles = Article.query.filter_by(project_id=project_id).order_by(Article.created_at.desc()).all()
    
    # Calculate statistics
    total_articles = len(articles)
    processed_articles = len([a for a in articles if a.status != 'pending'])
    included_count = len([a for a in articles if a.status == 'included'])
    excluded_count = len([a for a in articles if a.status == 'excluded'])
    
    stats = {
        'total': total_articles,
        'processed': processed_articles,
        'included': included_count,
        'excluded': excluded_count,
        'pending': total_articles - processed_articles
    }
    
    return render_template('dashboard.html', project=project, articles=articles, stats=stats)

@main_bp.route('/project/<int:project_id>/upload', methods=['POST'])
def upload_file(project_id):
    """Upload and parse reference files with database source tracking."""
    from app.models.screening_models import PublicationSource
    from app.services.utils.file_parser import load_studies_with_source_tracking
    
    project = Project.query.get_or_404(project_id)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    database_source = request.form.get('database_source', 'Auto-detect')
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        try:
            # Read file content and parse with source tracking
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            studies, detected_source = load_studies_with_source_tracking(
                file_content, 
                filename, 
                entrez_email=os.getenv('ENTREZ_EMAIL', '')
            )
            
            final_source = detected_source if database_source == 'Auto-detect' else database_source
            
            # Create articles in database
            articles_created = 0
            for study in studies:
                # Convert authors list to string if it's a list
                authors = study.get('authors', '')
                if isinstance(authors, list):
                    authors = ', '.join(authors)
                
                article = Article(
                    project_id=project_id,
                    title=study.get('title', ''),
                    authors=authors,
                    journal=study.get('journal_name', ''),
                    year=study.get('year'),
                    abstract=study.get('abstract', ''),
                    doi=study.get('doi', ''),
                    pmid=study.get('pmid', ''),
                    original_data=study,
                    status='pending'
                )
                db.session.add(article)
                db.session.flush()  # Get article ID
                
                pub_source = PublicationSource(
                    article_id=article.id,
                    source_database=final_source,
                    source_id=study.get('pmid') or study.get('doi') or '',
                    import_date=datetime.utcnow()
                )
                db.session.add(pub_source)
                articles_created += 1
            
            # Update project
            project.original_filename = filename
            project.file_path = filepath
            project.total_articles = articles_created
            project.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded and parsed {articles_created} articles from {final_source}',
                'articles_count': articles_created,
                'database_source': final_source
            })
            
        except Exception as e:
            logger.error(f"Error parsing file: {str(e)}")
            return jsonify({'error': f'Error parsing file: {str(e)}'}), 500

@main_bp.route('/project/<int:project_id>/export/dual-llm-comparison')
def export_dual_llm_comparison(project_id):
    """Export dual-LLM comparison spreadsheet."""
    project = Project.query.get_or_404(project_id)
    articles = Article.query.filter_by(project_id=project_id).all()
    
    articles_with_results = [a for a in articles if a.decision_reasoning]
    
    if not articles_with_results:
        return jsonify({'error': 'No dual-LLM screening results found'}), 400
    
    from app.services.utils.dual_llm_comparison_exporter import DualLLMComparisonExporter
    exporter = DualLLMComparisonExporter()
    
    try:
        filepath = exporter.generate_comparison_spreadsheet(articles_with_results, project.name)
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f'{project.name}_dual_llm_comparison.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error(f"Error generating comparison spreadsheet: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@main_bp.route('/project/<int:project_id>/export/<format>')
def export_results(project_id, format):
    """Export screening results in various formats."""
    project = Project.query.get_or_404(project_id)
    articles = Article.query.filter_by(project_id=project_id).all()
    
    if format == 'csv':
        return export_csv(project, articles)
    elif format == 'ris':
        return export_ris(project, articles)
    elif format == 'bibtex':
        return export_bibtex(project, articles)
    else:
        return jsonify({'error': 'Unsupported format'}), 400

def export_csv(project, articles):
    """Export results as CSV."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Title', 'Authors', 'Journal', 'Year', 'Abstract', 'DOI', 'PMID', 'Status', 'Decision_Reasoning'])
    
    # Write articles
    for article in articles:
        writer.writerow([
            article.title or '',
            article.authors or '',
            article.journal or '',
            article.year or '',
            article.abstract or '',
            article.doi or '',
            article.pmid or '',
            article.status or '',
            str(article.decision_reasoning) if article.decision_reasoning else ''
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={project.name}_results.csv'}
    )

def export_ris(project, articles):
    """Export results as RIS format."""
    ris_entries = []
    
    for article in articles:
        entry = {
            'TY': 'JOUR',  # Journal article
            'TI': article.title or '',
            'AU': [author.strip() for author in (article.authors or '').split(',') if author.strip()],
            'JO': article.journal or '',
            'PY': str(article.year) if article.year else '',
            'AB': article.abstract or '',
            'DO': article.doi or '',
            'N1': f"Status: {article.status}"
        }
        
        # Remove empty fields
        entry = {k: v for k, v in entry.items() if v}
        ris_entries.append(entry)
    
    # Convert to RIS format
    ris_output = rispy.dumps(ris_entries)
    
    return Response(
        ris_output,
        mimetype='application/x-research-info-systems',
        headers={'Content-Disposition': f'attachment; filename={project.name}_results.ris'}
    )

def export_bibtex(project, articles):
    """Export results as BibTeX format."""
    bib_db = bibtexparser.bibdatabase.BibDatabase()
    
    for i, article in enumerate(articles):
        entry = {
            'ENTRYTYPE': 'article',
            'ID': f'article_{i+1}',
            'title': article.title or '',
            'author': article.authors or '',
            'journal': article.journal or '',
            'year': str(article.year) if article.year else '',
            'abstract': article.abstract or '',
            'doi': article.doi or '',
            'note': f"Status: {article.status}"
        }
        
        # Remove empty fields
        entry = {k: v for k, v in entry.items() if v}
        bib_db.entries.append(entry)
    
    # Convert to BibTeX format
    writer = bibtexparser.bwriter.BibTexWriter()
    bibtex_output = writer.write(bib_db)
    
    return Response(
        bibtex_output,
        mimetype='text/plain',
        headers={'Content-Disposition': f'attachment; filename={project.name}_results.bib'}
    )

@main_bp.route('/project/<int:project_id>/screening')
def screening_interface(project_id):
    """Show the screening interface."""
    project = Project.query.get_or_404(project_id)
    return render_template('screening/modern_screening_interface.html', project=project)

@main_bp.route('/project/<int:project_id>/analytics')
def analytics_dashboard(project_id):
    """Show the analytics dashboard."""
    project = Project.query.get_or_404(project_id)
    return render_template('analytics/analytics_dashboard.html', project=project)

# Legacy routes for compatibility
@main_bp.route('/screening_tool')
def legacy_screening_tool():
    """Legacy screening tool route."""
    return render_template('screening_tool.html')

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory('uploads', filename)

@main_bp.route('/results/<filename>')
def result_file(filename):
    """Serve result files."""
    return send_from_directory('results', filename)


@main_bp.route('/api/model-config', methods=['GET', 'POST'])
def model_configuration():
    """Get or update model configuration."""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': {
                'temperature': 0.1,
                'seed': None,
                'openai_model': 'gpt-4o',
                'anthropic_model': 'claude-3-5-sonnet-20241022'
            }
        })
    
    elif request.method == 'POST':
        try:
            config = request.get_json()
            
            if not config:
                return jsonify({'success': False, 'error': 'No configuration data provided'}), 400
            
            temperature = config.get('temperature')
            if temperature is not None:
                try:
                    temperature = float(temperature)
                    if not (0.0 <= temperature <= 2.0):
                        return jsonify({'success': False, 'error': 'Temperature must be between 0.0 and 2.0'}), 400
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'error': 'Temperature must be a valid number'}), 400
            
            seed = config.get('seed')
            if seed is not None and seed != '':
                try:
                    seed = int(seed)
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'error': 'Seed must be a valid integer or null'}), 400
            else:
                seed = None
            
            openai_model = config.get('openai_model', 'gpt-4o')
            anthropic_model = config.get('anthropic_model', 'claude-3-5-sonnet-20241022')
            
            if not isinstance(openai_model, str) or not openai_model.strip():
                return jsonify({'success': False, 'error': 'OpenAI model must be a non-empty string'}), 400
            
            if not isinstance(anthropic_model, str) or not anthropic_model.strip():
                return jsonify({'success': False, 'error': 'Anthropic model must be a non-empty string'}), 400
            
            from flask import session
            session['model_config'] = {
                'temperature': temperature,
                'seed': seed,
                'openai_model': openai_model.strip(),
                'anthropic_model': anthropic_model.strip()
            }
            
            return jsonify({
                'success': True,
                'message': 'Model configuration updated successfully',
                'config': session['model_config']
            })
            
        except Exception as e:
            logger.error(f"Error updating model configuration: {str(e)}")
            return jsonify({'success': False, 'error': f'Configuration update failed: {str(e)}'}), 500

@main_bp.route('/api/model-config/project/<int:project_id>', methods=['GET', 'POST'])
def project_model_configuration(project_id):
    """Get or update model configuration for a specific project."""
    project = Project.query.get_or_404(project_id)
    
    if request.method == 'GET':
        from flask import session
        project_config_key = f'model_config_project_{project_id}'
        config = session.get(project_config_key, {
            'temperature': 0.1,
            'seed': None,
            'openai_model': 'gpt-4o',
            'anthropic_model': 'claude-3-5-sonnet-20241022'
        })
        
        return jsonify({
            'success': True,
            'project_id': project_id,
            'project_name': project.name,
            'config': config
        })
    
    elif request.method == 'POST':
        try:
            config = request.get_json()
            
            if not config:
                return jsonify({'success': False, 'error': 'No configuration data provided'}), 400
            
            temperature = config.get('temperature', 0.1)
            if temperature is not None:
                try:
                    temperature = float(temperature)
                    if not (0.0 <= temperature <= 2.0):
                        return jsonify({'success': False, 'error': 'Temperature must be between 0.0 and 2.0'}), 400
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'error': 'Temperature must be a valid number'}), 400
            
            seed = config.get('seed')
            if seed is not None and seed != '':
                try:
                    seed = int(seed)
                except (ValueError, TypeError):
                    return jsonify({'success': False, 'error': 'Seed must be a valid integer or null'}), 400
            else:
                seed = None
            
            openai_model = config.get('openai_model', 'gpt-4o')
            anthropic_model = config.get('anthropic_model', 'claude-3-5-sonnet-20241022')
            
            if not isinstance(openai_model, str) or not openai_model.strip():
                return jsonify({'success': False, 'error': 'OpenAI model must be a non-empty string'}), 400
            
            if not isinstance(anthropic_model, str) or not anthropic_model.strip():
                return jsonify({'success': False, 'error': 'Anthropic model must be a non-empty string'}), 400
            
            from flask import session
            project_config_key = f'model_config_project_{project_id}'
            session[project_config_key] = {
                'temperature': temperature,
                'seed': seed,
                'openai_model': openai_model.strip(),
                'anthropic_model': anthropic_model.strip()
            }
            
            return jsonify({
                'success': True,
                'message': f'Model configuration updated for project "{project.name}"',
                'project_id': project_id,
                'config': session[project_config_key]
            })
            
        except Exception as e:
            logger.error(f"Error updating project model configuration: {str(e)}")
            return jsonify({'success': False, 'error': f'Configuration update failed: {str(e)}'}), 500
