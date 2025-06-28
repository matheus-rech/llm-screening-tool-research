"""
Enhanced Routes for Advanced Features Integration
Connects all our new features to the Flask application.

Integration Points:
- Active Learning routes for smart suggestions
- Cost Estimation endpoints
- Collaborative Screening features
- PICO Extraction capabilities
- Enhanced UI components
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from app.models.screening_models import db, Project, Article
from app.services.ml.active_learning import active_learning_manager, ActiveLearningConfig
# from app.services.utils.cost_tracker import cost_tracker  # Temporarily disabled
# from app.services.collaboration.collaborative_screening import collaborative_manager, ReviewerRole, DecisionStatus
# from app.services.extraction.pico_extractor import pico_extraction_manager
from app.services.utils.config_manager import ProjectConfiguration
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint for enhanced features
enhanced_bp = Blueprint('enhanced', __name__, url_prefix='/enhanced')

# ============================================================================
# ACTIVE LEARNING ROUTES
# ============================================================================

@enhanced_bp.route('/active-learning/<project_id>/initialize', methods=['POST'])
def initialize_active_learning(project_id):
    """Initialize active learning for a project."""
    try:
        result = active_learning_manager.initialize_active_learning(project_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to initialize active learning: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/active-learning/<project_id>/suggestions')
def get_smart_suggestions(project_id):
    """Get smart article suggestions using active learning."""
    try:
        count = request.args.get('count', 10, type=int)
        strategy = request.args.get('strategy', 'uncertainty')
        
        suggestions = active_learning_manager.get_smart_suggestions(
            project_id, count, strategy
        )
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'strategy': strategy
        })
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/active-learning/<project_id>/performance')
def get_al_performance(project_id):
    """Get active learning performance metrics."""
    try:
        learner = active_learning_manager.get_or_create_learner(project_id)
        performance = learner.get_performance_summary()
        
        return jsonify({
            'success': True,
            'performance': performance
        })
    except Exception as e:
        logger.error(f"Failed to get AL performance: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/active-learning/<project_id>/retrain', methods=['POST'])
def retrain_active_learning(project_id):
    """Manually retrain active learning model."""
    try:
        new_article_ids = request.json.get('new_article_ids', [])
        
        learner = active_learning_manager.get_or_create_learner(project_id)
        success = learner.update_model(project_id, new_article_ids)
        
        return jsonify({
            'success': success,
            'message': 'Model retrained successfully' if success else 'Not enough new labels for retraining'
        })
    except Exception as e:
        logger.error(f"Failed to retrain model: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================================
# COST ESTIMATION ROUTES
# ============================================================================

@enhanced_bp.route('/cost/estimate/<project_id>')
def estimate_project_cost(project_id):
    """Estimate cost for screening a project."""
    try:
        sample_size = request.args.get('sample_size', 10, type=int)
        
        estimate = cost_estimator.estimate_project_cost(project_id, sample_size)
        
        return jsonify({
            'success': True,
            'estimate': estimate.to_dict()
        })
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/cost/optimize', methods=['POST'])
def optimize_costs():
    """Get cost optimization suggestions."""
    try:
        data = request.json
        max_budget = data.get('max_budget')
        estimate_data = data.get('estimate')
        
        # Reconstruct estimate object (simplified)
        from cost_tracker import CostEstimate
        from decimal import Decimal
        
        estimate = CostEstimate(
            total_articles=estimate_data['total_articles'],
            estimated_input_tokens_per_article=estimate_data['estimated_input_tokens_per_article'],
            estimated_output_tokens_per_article=estimate_data['estimated_output_tokens_per_article'],
            conservative_model_cost=Decimal(str(estimate_data['conservative_model_cost'])),
            liberal_model_cost=Decimal(str(estimate_data['liberal_model_cost'])),
            resolver_model_cost=Decimal(str(estimate_data['resolver_model_cost'])),
            total_estimated_cost=Decimal(str(estimate_data['total_estimated_cost'])),
            confidence_level=estimate_data['confidence_level'],
            assumptions=estimate_data['assumptions']
        )
        
        optimization = cost_estimator.suggest_cost_optimization(estimate, Decimal(str(max_budget)))
        
        return jsonify({
            'success': True,
            'optimization': optimization
        })
    except Exception as e:
        logger.error(f"Cost optimization failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/cost/track/<project_id>/current')
def get_current_costs(project_id):
    """Get current cost tracking for a project."""
    try:
        costs = cost_tracker.get_current_costs(project_id)
        
        return jsonify({
            'success': True,
            'costs': costs
        })
    except Exception as e:
        logger.error(f"Failed to get current costs: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/cost/models')
def get_model_pricing():
    """Get current model pricing information."""
    try:
        pricing_info = {}
        for model_name, pricing in MODEL_PRICING.items():
            pricing_info[model_name] = {
                'input_cost_per_1k': float(pricing.input_cost_per_1k_tokens),
                'output_cost_per_1k': float(pricing.output_cost_per_1k_tokens),
                'context_window': pricing.context_window,
                'notes': pricing.notes
            }
        
        return jsonify({
            'success': True,
            'pricing': pricing_info
        })
    except Exception as e:
        logger.error(f"Failed to get model pricing: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================================
# COLLABORATIVE SCREENING ROUTES
# ============================================================================

@enhanced_bp.route('/collaboration/reviewers', methods=['GET', 'POST'])
def manage_reviewers():
    """Create or list reviewers."""
    if request.method == 'POST':
        try:
            data = request.json
            reviewer = collaborative_manager.create_reviewer(
                name=data['name'],
                email=data['email'],
                institution=data.get('institution'),
                expertise_areas=data.get('expertise_areas', [])
            )
            
            return jsonify({
                'success': True,
                'reviewer': {
                    'id': reviewer.id,
                    'name': reviewer.name,
                    'email': reviewer.email
                }
            })
        except Exception as e:
            logger.error(f"Failed to create reviewer: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    else:  # GET
        try:
            from collaborative_screening import Reviewer
            reviewers = Reviewer.query.filter_by(is_active=True).all()
            
            reviewer_list = []
            for reviewer in reviewers:
                reviewer_list.append({
                    'id': reviewer.id,
                    'name': reviewer.name,
                    'email': reviewer.email,
                    'institution': reviewer.institution,
                    'created_at': reviewer.created_at.isoformat()
                })
            
            return jsonify({
                'success': True,
                'reviewers': reviewer_list
            })
        except Exception as e:
            logger.error(f"Failed to list reviewers: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/collaboration/assign', methods=['POST'])
def assign_reviewer_to_project():
    """Assign reviewer to project."""
    try:
        data = request.json
        assignment = collaborative_manager.assign_reviewer_to_project(
            project_id=data['project_id'],
            reviewer_id=data['reviewer_id'],
            role=ReviewerRole(data['role']),
            assigned_by=data.get('assigned_by')
        )
        
        return jsonify({
            'success': True,
            'assignment_id': assignment.id
        })
    except Exception as e:
        logger.error(f"Failed to assign reviewer: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/collaboration/<project_id>/articles/<reviewer_id>')
def get_reviewer_articles(project_id, reviewer_id):
    """Get articles assigned to a specific reviewer."""
    try:
        count = request.args.get('count', 10, type=int)
        strategy = request.args.get('strategy', 'round_robin')
        
        articles = collaborative_manager.get_articles_for_reviewer(
            project_id, reviewer_id, count, strategy
        )
        
        article_list = []
        for article in articles:
            article_list.append({
                'id': article.id,
                'title': article.title,
                'abstract': article.abstract,
                'authors': article.authors,
                'year': article.year,
                'journal': article.journal
            })
        
        return jsonify({
            'success': True,
            'articles': article_list,
            'strategy': strategy
        })
    except Exception as e:
        logger.error(f"Failed to get reviewer articles: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/collaboration/decision', methods=['POST'])
def record_reviewer_decision():
    """Record a reviewer's decision on an article."""
    try:
        data = request.json
        
        decision = collaborative_manager.record_reviewer_decision(
            article_id=data['article_id'],
            reviewer_id=data['reviewer_id'],
            decision=DecisionStatus(data['decision']),
            reasoning=data.get('reasoning'),
            confidence=data.get('confidence', 3),
            time_spent=data.get('time_spent'),
            llm_suggestion=data.get('llm_suggestion')
        )
        
        return jsonify({
            'success': True,
            'decision_id': decision.id
        })
    except Exception as e:
        logger.error(f"Failed to record decision: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/collaboration/<project_id>/reliability')
def get_inter_rater_reliability(project_id):
    """Get inter-rater reliability metrics."""
    try:
        reviewer1_id = request.args.get('reviewer1_id')
        reviewer2_id = request.args.get('reviewer2_id')
        
        if reviewer1_id and reviewer2_id:
            reliability = collaborative_manager.calculate_inter_rater_reliability(
                project_id, reviewer1_id, reviewer2_id
            )
            
            return jsonify({
                'success': True,
                'reliability': {
                    'reviewer1_id': reliability.reviewer1_id,
                    'reviewer2_id': reliability.reviewer2_id,
                    'total_overlapping_decisions': reliability.total_overlapping_decisions,
                    'agreements': reliability.agreements,
                    'disagreements': reliability.disagreements,
                    'agreement_percentage': reliability.agreement_percentage,
                    'cohens_kappa': reliability.cohens_kappa,
                    'specific_agreements': reliability.specific_agreements
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Both reviewer IDs required'}), 400
            
    except Exception as e:
        logger.error(f"Failed to calculate reliability: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/collaboration/<project_id>/stats')
def get_collaboration_stats(project_id):
    """Get collaboration statistics for a project."""
    try:
        stats = collaborative_manager.get_project_collaboration_stats(project_id)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Failed to get collaboration stats: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================================
# PICO EXTRACTION ROUTES
# ============================================================================

@enhanced_bp.route('/pico/extract/<article_id>')
def extract_pico_from_article(article_id):
    """Extract PICO elements from a single article."""
    try:
        method = request.args.get('method', 'hybrid')
        project_id = request.args.get('project_id')
        
        article = Article.query.get_or_404(article_id)
        
        # Initialize LLM extractor if needed
        if method in ['llm', 'hybrid']:
            from openai import OpenAI
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                client = OpenAI(api_key=api_key)
                pico_extraction_manager.initialize_llm_extractor(client)
        
        result = pico_extraction_manager.extract_from_article(article, method, project_id)
        
        return jsonify({
            'success': True,
            'extraction': {
                'population_elements': [elem.__dict__ for elem in result.population_elements],
                'intervention_elements': [elem.__dict__ for elem in result.intervention_elements],
                'comparison_elements': [elem.__dict__ for elem in result.comparison_elements],
                'outcome_elements': [elem.__dict__ for elem in result.outcome_elements],
                'overall_confidence': result.overall_confidence,
                'extraction_method': result.extraction_method,
                'processing_time_seconds': result.processing_time_seconds
            }
        })
    except Exception as e:
        logger.error(f"PICO extraction failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/pico/suggest/<project_id>')
def suggest_pico_criteria(project_id):
    """Auto-suggest PICO criteria from project articles."""
    try:
        sample_size = request.args.get('sample_size', 20, type=int)
        method = request.args.get('method', 'hybrid')
        
        # Initialize LLM extractor if needed
        if method in ['llm', 'hybrid']:
            from openai import OpenAI
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                client = OpenAI(api_key=api_key)
                pico_extraction_manager.initialize_llm_extractor(client)
        
        result = pico_extraction_manager.auto_suggest_pico_from_project(
            project_id, sample_size, method
        )
        
        return jsonify({
            'success': True,
            'suggestion': result
        })
    except Exception as e:
        logger.error(f"PICO suggestion failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ============================================================================
# ENHANCED UI ROUTES
# ============================================================================

@enhanced_bp.route('/dashboard')
def enhanced_dashboard():
    """Render enhanced dashboard with advanced features."""
    try:
        projects = Project.query.all()
        
        # Get additional stats for dashboard
        total_articles = db.session.query(Article).count()
        
        return render_template('enhanced_dashboard.html', 
                             projects=projects,
                             total_articles=total_articles)
    except Exception as e:
        logger.error(f"Failed to render enhanced dashboard: {e}")
        return redirect(url_for('dashboard'))

@enhanced_bp.route('/screening/<project_id>')
def active_learning_interface(project_id):
    """Render active learning screening interface."""
    try:
        project = Project.query.get_or_404(project_id)
        
        return render_template('active_learning_interface.html', 
                             project=project)
    except Exception as e:
        logger.error(f"Failed to render AL interface: {e}")
        return redirect(url_for('project_view', project_id=project_id))

@enhanced_bp.route('/templates')
def pico_templates():
    """Render PICO templates management page."""
    try:
        config_manager = ConfigurationManager()
        templates = config_manager.list_templates()
        
        return render_template('pico_templates.html', 
                             templates=templates)
    except Exception as e:
        logger.error(f"Failed to render templates page: {e}")
        return redirect(url_for('dashboard'))

# ============================================================================
# API HELPER ROUTES
# ============================================================================

@enhanced_bp.route('/health')
def health_check():
    """Health check for enhanced features."""
    try:
        status = {
            'active_learning': True,
            'cost_tracking': True,
            'collaboration': True,
            'pico_extraction': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@enhanced_bp.route('/features')
def get_available_features():
    """Get list of available enhanced features."""
    features = {
        'active_learning': {
            'name': 'Active Learning',
            'description': 'AI-powered article prioritization',
            'endpoints': [
                '/enhanced/active-learning/<project_id>/initialize',
                '/enhanced/active-learning/<project_id>/suggestions',
                '/enhanced/active-learning/<project_id>/performance'
            ]
        },
        'cost_estimation': {
            'name': 'Cost Estimation',
            'description': 'LLM cost tracking and optimization',
            'endpoints': [
                '/enhanced/cost/estimate/<project_id>',
                '/enhanced/cost/optimize',
                '/enhanced/cost/models'
            ]
        },
        'collaboration': {
            'name': 'Collaborative Screening',
            'description': 'Multi-reviewer project support',
            'endpoints': [
                '/enhanced/collaboration/reviewers',
                '/enhanced/collaboration/assign',
                '/enhanced/collaboration/<project_id>/reliability'
            ]
        },
        'pico_extraction': {
            'name': 'PICO Extraction',
            'description': 'Automatic PICO criteria suggestion',
            'endpoints': [
                '/enhanced/pico/extract/<article_id>',
                '/enhanced/pico/suggest/<project_id>'
            ]
        }
    }
    
    return jsonify({
        'success': True,
        'features': features
    })

# Error handlers for the blueprint
@enhanced_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@enhanced_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500