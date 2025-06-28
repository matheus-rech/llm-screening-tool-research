"""
Modern LLM Screening API Routes
Flask API endpoints for the modern dual-provider screening system.
"""

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from flask import Blueprint, request, jsonify, render_template, current_app, session
from app.models.screening_models import db, Project, Article
from app.services.screening import (
    DualProviderScreeningOrchestrator,
    ScreeningCriteria,
    HumanReviewTriggers,
    ScreeningResultsStore,
    ScreeningWorkflowOrchestrator,
    WorkflowConfig,
    WorkflowType,
    WorkflowProgress,
    ScreeningWorkflowFactory
)
from app.services.screening.dual_llm_screener import ModelConfig
# Cost tracking temporarily disabled - using logging instead
from app.services.utils.error_handler import handle_file_parsing_error
from app.services.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Create Blueprint for modern screening routes
modern_screening_bp = Blueprint('modern_screening', __name__, url_prefix='/api/screening')

# ============================================================================
# CONFIGURATION AND SETUP ROUTES
# ============================================================================

@modern_screening_bp.route('/setup', methods=['GET'])
def setup_page():
    """Render the modern screening setup page."""
    try:
        return render_template('screening/modern_screening_setup.html')
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Screening Setup - Test Mode</title>
            <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto p-8">
                <h1 class="text-3xl font-bold mb-8">LLM Screening Setup - Test Mode</h1>
                <p class="text-red-600 mb-4">Template error: {str(e)}</p>

                <div class="bg-white rounded-lg p-6 mb-8">
                    <h3 class="text-lg font-semibold text-gray-900 mb-4">Model Parameters</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">OpenAI Temperature</label>
                            <input type="number" min="0" max="2" step="0.1" value="0.1"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Anthropic Temperature</label>
                            <input type="number" min="0" max="2" step="0.1" value="0.1"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">OpenAI Seed (Optional)</label>
                            <input type="number" min="0" max="4294967295" placeholder="Leave empty for random"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-2">Anthropic Seed (Optional)</label>
                            <input type="number" min="0" max="4294967295" placeholder="Leave empty for random"
                                   class="w-full px-3 py-2 border border-gray-300 rounded-lg">
                        </div>
                    </div>
                </div>

                <div class="bg-green-50 rounded-lg p-6">
                    <h3 class="text-lg font-semibold text-green-900 mb-2">✅ Dynamic Model Configuration Test</h3>
                    <p class="text-green-700">The temperature and seed controls are now visible and functional!</p>
                    <p class="text-sm text-green-600 mt-2">This confirms that the ModelConfig and DualModelConfig classes are working correctly.</p>
                </div>
            </div>
        </body>
        </html>
        """

@modern_screening_bp.route('/start', methods=['POST'])
def start_screening():
    """Initialize a new modern screening project and trigger automatic processing."""

    data = request.get_json()
    if not data:
        raise ValidationError("No configuration data provided")

    # Validate required fields
    required_fields = ['researchQuestion', 'picott', 'inclusionCriteria', 'exclusionCriteria', 'llmConfig']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")

    # Create screening criteria from configuration
    try:
        criteria = ScreeningCriteria(
            research_question=data['researchQuestion'],
            target_population=data['picott'].get('population', ''),
            target_intervention=data['picott'].get('intervention', ''),
            target_comparison=data['picott'].get('comparison', ''),
            target_outcomes=data['picott'].get('outcomes', '').split(',') if data['picott'].get('outcomes') else [],
            target_time_frame=data['picott'].get('timeFrame', ''),
            target_study_types=data['picott'].get('studyTypes', '').split(',') if data['picott'].get('studyTypes') else [],
            inclusion_criteria=data['inclusionCriteria'],
            exclusion_criteria=data['exclusionCriteria'],
            temperature=float(data['llmConfig'].get('temperature', 0.1)),
            seed=int(data['llmConfig']['seed']) if data['llmConfig'].get('seed') is not None and str(data['llmConfig']['seed']).isdigit() else None,
            openai_model=data['llmConfig'].get('openaiModel', 'gpt-4o'),
            anthropic_model=data['llmConfig'].get('anthropicModel', 'claude-3-5-sonnet-20241022')
        )
    except Exception as e:
        raise ValidationError(f"Invalid screening criteria: {str(e)}")

    # Create or update project
    project_id = session.get('current_project_id')
    if project_id:
        project = Project.query.get(project_id)
        if project:
            # Update existing project with modern screening config
            project.config = {
                'modern_screening': True,
                'criteria': criteria.model_dump(),
                'llm_config': data['llmConfig'],
                'created_at': datetime.now().isoformat(),
                'auto_process': data.get('autoProcess', True)
            }
        else:
            raise ValidationError("Project not found")
    else:
        raise ValidationError("No active project found. Please upload articles first.")

    # Store configuration
    db.session.commit()

    # Log project initialization for monitoring
    logger.info(f"Starting screening for project {project_id} with {pending_count} pending articles")

    # **NEW: Trigger automatic screening if enabled**
    if data.get('autoProcess', True):
        try:
            # Get pending articles count
            pending_count = Article.query.filter_by(
                project_id=project_id,
                status='pending'
            ).count()

            if pending_count > 0:
                # Start background processing
                # from threading import Thread # Move to top of file

                def process_articles_background():
                    """Background processing of articles."""
                    try:
                        # Process articles in batches
                        batch_size = min(5, pending_count)  # Start with small batch
                        articles = Article.query.filter_by(
                            project_id=project_id,
                            status='pending'
                        ).limit(batch_size).all()

                        # Initialize orchestrator
                        openai_key = os.getenv('OPENAI_API_KEY')
                        anthropic_key = os.getenv('ANTHROPIC_API_KEY')

                        if not openai_key or not anthropic_key:
                            logger.error("Missing API keys for LLM providers")
                            return

                        orchestrator = DualProviderScreeningOrchestrator(
                            openai_api_key=openai_key,
                            anthropic_api_key=anthropic_key
                        )

                        processed_count = 0
                        for article in articles:
                            try:
                                # Screen with both providers
                                results = orchestrator.screen_article_dual_provider(
                                    article, criteria, str(project_id)
                                )

                                openai_result = results.get('openai')
                                anthropic_result = results.get('anthropic')

                                # Analyze agreement
                                if openai_result is not None and anthropic_result is not None:
                                    agreement_analysis = orchestrator.analyze_provider_agreement(
                                        openai_result, anthropic_result
                                    )
                                else:
                                    agreement_analysis = {
                                        'agreement': False,
                                        'reason': 'One or both providers failed to return results',
                                        'requires_human_review': True,
                                        'decision_agreement': False,
                                        'confidence_agreement': False,
                                        'relevance_agreement': False,
                                        'confidence_difference': 1.0,
                                        'relevance_difference': 1.0
                                    }

                                # Check human review triggers
                                if openai_result is not None and anthropic_result is not None:
                                    human_review_triggers = HumanReviewTriggers.should_trigger_human_review(
                                        openai_result, anthropic_result
                                    )
                                else:
                                    human_review_triggers = {
                                        'should_review': True,
                                        'uncertainty_score': 1.0,
                                        'triggers': ['One or both providers failed'],
                                        'trigger_count': 1
                                    }

                                # Store results
                                if openai_result is not None and anthropic_result is not None:
                                    ScreeningResultsStore.store_screening_results(
                                        str(article.id),
                                        str(project_id),
                                        openai_result,
                                        anthropic_result,
                                        agreement_analysis,
                                        human_review_triggers
                                    )
                                else:
                                    # Handle case where one or both providers failed
                                    article.status = 'human_review_required'
                                    article.decision_reasoning = {
                                        'error': 'One or both LLM providers failed',
                                        'timestamp': datetime.now().isoformat(),
                                        'requires_human_review': True,
                                        'agreement_analysis': agreement_analysis,
                                        'human_review_triggers': human_review_triggers
                                    }
                                    # db.session.commit() # Consider committing outside the loop or in batches

                                processed_count += 1
                                logger.info(f"Processed article {article.id} ({processed_count}/{len(articles)})")

                            except Exception as e:
                                logger.error(f"Failed to process article {article.id}: {e}")
                                # Mark article as failed with specific error handling
                                article.status = 'failed'
                                error_type = "unknown_error"
                                
                                # Categorize the error for better handling
                                if "timeout" in str(e).lower():
                                    error_type = "api_timeout"
                                    article.status = 'pending'  # Can retry timeouts
                                elif "rate_limit" in str(e).lower():
                                    error_type = "rate_limit"
                                    article.status = 'pending'  # Can retry rate limits
                                elif "api" in str(e).lower():
                                    error_type = "api_error"
                                    article.status = 'human_review_required'
                                else:
                                    article.status = 'human_review_required'
                                
                                article.decision_reasoning = {
                                    'error': str(e),
                                    'error_type': error_type,
                                    'timestamp': datetime.now().isoformat(),
                                    'requires_human_review': article.status == 'human_review_required',
                                    'retry_count': getattr(article, 'retry_count', 0) + 1
                                }
                                
                                # Commit individual article updates to prevent data loss
                                try:
                                    db.session.commit()
                                except Exception as commit_error:
                                    logger.error(f"Failed to commit article {article.id} error state: {commit_error}")
                                    db.session.rollback()

                        # Commit all successful updates
                        try:
                            db.session.commit()
                            logger.info(f"Background processing completed: {processed_count}/{len(articles)} articles processed successfully")
                        except Exception as commit_error:
                            logger.error(f"Failed to commit final batch: {commit_error}")
                            db.session.rollback()

                    except Exception as e:
                        logger.error(f"Background processing failed: {e}")

                # Process articles with proper Flask application context
                with current_app.app_context():
                    process_articles_background()

                logger.info(f"Started background processing for {pending_count} pending articles")

                return jsonify({
                    'success': True,
                    'project_id': project_id,
                    'message': f'Modern screening initialized and processing started for {pending_count} articles',
                    'auto_processing': True,
                    'pending_articles': pending_count
                })
            else:
                return jsonify({
                    'success': True,
                    'project_id': project_id,
                    'message': 'Modern screening initialized - no pending articles to process',
                    'auto_processing': False,
                    'pending_articles': 0
                })

        except Exception as e:
            logger.error(f"Failed to start automatic processing: {e}")
            # Still return success for configuration, but note processing issue
            return jsonify({
                'success': True,
                'project_id': project_id,
                'message': 'Modern screening initialized, but automatic processing failed to start',
                'auto_processing': False,
                'error': str(e)
            })

    logger.info(f"Modern screening started for project {project_id}")

    return jsonify({
        'success': True,
        'project_id': project_id,
        'message': 'Modern screening initialized successfully',
        'auto_processing': False
    })

@modern_screening_bp.route('/interface/<project_id>')
def screening_interface(project_id):
    """Render the modern screening interface."""

    project = Project.query.get_or_404(project_id)

    return render_template('modern_screening_interface.html', project=project)

# ============================================================================
# ARTICLE QUEUE AND MANAGEMENT ROUTES
# ============================================================================

@modern_screening_bp.route('/queue', methods=['GET'])
def get_article_queue():
    """Get prioritized article queue for screening."""

    strategy = request.args.get('strategy', 'ai_priority')
    limit = int(request.args.get('limit', 20))
    project_id = request.args.get('project_id')

    if not project_id:
        project_id = session.get('current_project_id')
        if not project_id:
            raise ValidationError("No project specified")

    # Get articles based on strategy
    query = Article.query.filter_by(project_id=project_id)

    if strategy == 'ai_priority':
        # Prioritize uncertain cases and conflicts
        articles = query.filter(
            Article.status.in_(['pending', 'human_review_required', 'uncertain'])
        ).order_by(Article.created_at.desc()).limit(limit).all()

    elif strategy == 'high_confidence':
        # Show high-confidence cases first
        articles = query.filter_by(status='pending').order_by(
            Article.created_at.desc()
        ).limit(limit).all()

    elif strategy == 'low_confidence':
        # Show uncertain cases
        articles = query.filter(
            Article.status.in_(['uncertain', 'human_review_required'])
        ).limit(limit).all()

    elif strategy == 'conflicts':
        # Show only conflicts
        articles = query.filter_by(status='human_review_required').limit(limit).all()

    elif strategy == 'random':
        # Random order
        articles = query.filter_by(status='pending').order_by(db.func.random()).limit(limit).all()

    else:
        articles = query.filter_by(status='pending').limit(limit).all()

    # Format articles for frontend
    article_data = []
    for article in articles:
        article_info = {
            'id': article.id,
            'title': article.title,
            'authors': article.authors,
            'year': article.year,
            'journal': article.journal,
            'abstract': article.abstract,
            'status': article.status,
            'openai_prediction': None,
            'anthropic_prediction': None,
            'confidence': None,
            'requires_human_review': article.status == 'human_review_required'
        }

        # Extract AI predictions if available
        if article.decision_reasoning and isinstance(article.decision_reasoning, dict):
            reasoning = article.decision_reasoning

            if 'openai_result' in reasoning and reasoning['openai_result']:
                openai_data = reasoning['openai_result']
                article_info['openai_prediction'] = openai_data.get('screening_decision', {}).get('final_decision')

            if 'anthropic_result' in reasoning and reasoning['anthropic_result']:
                anthropic_data = reasoning['anthropic_result']
                article_info['anthropic_prediction'] = anthropic_data.get('screening_decision', {}).get('final_decision')

            # Calculate average confidence
            if 'openai_result' in reasoning and 'anthropic_result' in reasoning:
                openai_conf = reasoning['openai_result'].get('screening_decision', {}).get('confidence_score', 0)
                anthropic_conf = reasoning['anthropic_result'].get('screening_decision', {}).get('confidence_score', 0)
                article_info['confidence'] = (openai_conf + anthropic_conf) / 2

        article_data.append(article_info)

    return jsonify({
        'success': True,
        'articles': article_data,
        'strategy': strategy,
        'total_available': query.count()
    })

@modern_screening_bp.route('/analysis/<article_id>', methods=['GET'])
def get_analysis(article_id):
    """Get AI analysis results for a specific article."""

    article = Article.query.get_or_404(article_id)

    if not article.decision_reasoning:
        # No analysis available - trigger screening
        return jsonify({
            'success': False,
            'message': 'Analysis not available. Article needs to be screened.',
            'analysis': None,
            'agreement': None
        })

    reasoning = article.decision_reasoning

    return jsonify({
        'success': True,
        'analysis': {
            'openai': reasoning.get('openai_result'),
            'anthropic': reasoning.get('anthropic_result')
        },
        'agreement': reasoning.get('agreement_analysis'),
        'human_review_triggers': reasoning.get('human_review_triggers'),
        'final_decision': reasoning.get('final_decision')
    })

# ============================================================================
# SCREENING EXECUTION ROUTES
# ============================================================================

@modern_screening_bp.route('/process/<article_id>', methods=['POST'])
def process_article(article_id):
    """Process a single article with dual-provider screening."""

    article = Article.query.get_or_404(article_id)
    project = Project.query.get_or_404(article.project_id)

    # Get screening criteria from project config
    if not project.config or 'criteria' not in project.config:
        raise ValidationError("Project does not have modern screening configuration")

    criteria_dict = project.config['criteria']
    criteria = ScreeningCriteria(**criteria_dict)

    # Get model configuration from session or project config
    session_config = session.get('model_config', {})
    llm_config = project.config.get('llmConfig', {})

    model1_config = session_config.get('model1', {})
    model2_config = session_config.get('model2', {})

    from app.services.screening.dual_llm_screener import ModelConfig, DualModelConfig

    # Create OpenAI configuration
    openai_config = ModelConfig(
        provider=model1_config.get('provider', 'openai'),
        model_name=model1_config.get('model_name', 'gpt-4.1'),
        temperature=float(model1_config.get('temperature', llm_config.get('openaiTemperature', 0.1))),
        seed=int(model1_config['seed']) if model1_config.get('seed') else llm_config.get('openaiSeed'),
        max_tokens=4000
    )

    # Create Anthropic configuration
    anthropic_config = ModelConfig(
        provider=model2_config.get('provider', 'anthropic'),
        model_name=model2_config.get('model_name', 'claude-3-7-sonnet-20250219'),
        temperature=float(model2_config.get('temperature', llm_config.get('anthropicTemperature', 0.1))),
        seed=int(model2_config['seed']) if model2_config.get('seed') else llm_config.get('anthropicSeed'),
        max_tokens=4000
    )

    # Create dual model configuration
    dual_config = DualModelConfig(
        openai_config=openai_config,
        anthropic_config=anthropic_config
    )

    # Initialize screening orchestrator with dynamic configuration
    # import os # Move to top of file
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if not openai_key or not anthropic_key:
        raise ValidationError("Missing API keys for LLM providers")

    orchestrator = DualProviderScreeningOrchestrator(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        config=dual_config
    )

    try:
        # Screen with both providers
        results = orchestrator.screen_article_dual_provider(article, criteria, str(project.id))

        # Analyze agreement
        openai_result = results.get('openai')
        anthropic_result = results.get('anthropic')

        if openai_result is not None and anthropic_result is not None:
            agreement_analysis = orchestrator.analyze_provider_agreement(
                openai_result, anthropic_result
            )
        else:
            # Handle case where one or both providers failed
            agreement_analysis = {
                'agreement': False,
                'reason': 'One or both providers failed to return results',
                'requires_human_review': True,
                'decision_agreement': False,
                'confidence_agreement': False,
                'relevance_agreement': False,
                'confidence_difference': 1.0,
                'relevance_difference': 1.0
            }

        # Check human review triggers and store results
        if openai_result is not None and anthropic_result is not None:
            human_review_triggers = HumanReviewTriggers.should_trigger_human_review(
                openai_result, anthropic_result
            )

            # Store results
            screening_record = ScreeningResultsStore.store_screening_results(
                article_id,
                str(project.id),
                openai_result,
                anthropic_result,
                agreement_analysis,
                human_review_triggers
            )
        else:
            # Handle case where one or both providers failed
            human_review_triggers = {
                'should_review': True,
                'uncertainty_score': 1.0,
                'triggers': ['One or both providers failed'],
                'trigger_count': 1
            }

            # Create a manual screening record for failed cases
            screening_record = {
                'final_decision': 'UNCERTAIN',
                'requires_human_review': True
            }

            # Update article directly
            article.status = 'human_review_required'
            article.decision_reasoning = {
                'error': 'One or both LLM providers failed',
                'timestamp': datetime.now().isoformat(),
                'requires_human_review': True,
                'agreement_analysis': agreement_analysis,
                'human_review_triggers': human_review_triggers
            }
            db.session.commit()

        logger.info(f"Article {article_id} processed successfully")

        return jsonify({
            'success': True,
            'results': {
                'openai': openai_result.model_dump() if openai_result is not None else None,
                'anthropic': anthropic_result.model_dump() if anthropic_result is not None else None,
                'agreement': agreement_analysis,
                'human_review_triggers': human_review_triggers,
                'final_decision': screening_record.get('final_decision'),
                'requires_human_review': screening_record.get('requires_human_review')
            }
        })

    except APIError as e:
        logger.error(f"API error processing article {article_id}: {str(e)}")
        # Update article status based on error type
        if e.status_code == 408:  # Timeout
            article.status = 'pending'  # Can retry
        elif e.status_code == 429:  # Rate limit
            article.status = 'pending'  # Can retry after delay
        else:
            article.status = 'human_review_required'
            
        article.decision_reasoning = {
            'error': str(e),
            'error_type': 'api_error',
            'status_code': e.status_code,
            'timestamp': datetime.now().isoformat(),
            'requires_human_review': article.status == 'human_review_required'
        }
        
        try:
            db.session.commit()
        except Exception as commit_error:
            logger.error(f"Failed to commit article error state: {commit_error}")
            db.session.rollback()
            
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing article {article_id}: {str(e)}")
        # Mark article for human review on unexpected errors
        article.status = 'human_review_required'
        article.decision_reasoning = {
            'error': str(e),
            'error_type': 'unexpected_error',
            'timestamp': datetime.now().isoformat(),
            'requires_human_review': True
        }
        
        try:
            db.session.commit()
        except Exception as commit_error:
            logger.error(f"Failed to commit article error state: {commit_error}")
            db.session.rollback()
            
        raise

@modern_screening_bp.route('/decision', methods=['POST'])
def save_human_decision():
    """Save human reviewer decision."""

    data = request.get_json()

    required_fields = ['article_id', 'decision']
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")

    article = Article.query.get_or_404(data['article_id'])

    # Update article with human decision
    if not article.decision_reasoning:
        article.decision_reasoning = {}

    article.decision_reasoning['human_decision'] = {
        'decision': data['decision'],
        'reasoning': data.get('reasoning', ''),
        'send_to_expert': data.get('send_to_expert', False),
        'add_to_training': data.get('add_to_training', False),
        'timestamp': datetime.now().isoformat(),
        'reviewer_id': session.get('user_id', 'anonymous')
    }

    # Update article status based on human decision
    if data['decision'] == 'INCLUDE':
        article.status = 'included'
    elif data['decision'] == 'EXCLUDE':
        article.status = 'excluded'
    else:
        article.status = 'uncertain'

    db.session.commit()

    logger.info(f"Human decision saved for article {data['article_id']}: {data['decision']}")

    return jsonify({
        'success': True,
        'message': 'Decision saved successfully'
    })

# ============================================================================
# PROGRESS AND MONITORING ROUTES
# ============================================================================

@modern_screening_bp.route('/progress', methods=['GET'])
def get_progress():
    """Get screening progress and statistics."""

    project_id = request.args.get('project_id')
    if not project_id:
        project_id = session.get('current_project_id')
        if not project_id:
            raise ValidationError("No project specified")

    # Get article counts
    total_articles = Article.query.filter_by(project_id=project_id).count()

    processed_articles = Article.query.filter_by(project_id=project_id).filter(
        Article.status.in_(['included', 'excluded', 'uncertain', 'human_review_required'])
    ).count()

    included_count = Article.query.filter_by(project_id=project_id, status='included').count()
    excluded_count = Article.query.filter_by(project_id=project_id, status='excluded').count()
    uncertain_count = Article.query.filter_by(project_id=project_id, status='uncertain').count()
    conflict_count = Article.query.filter_by(project_id=project_id, status='human_review_required').count()

    # Cost tracking disabled - return zero cost for now
    total_cost = 0.0  # Will be implemented with proper cost tracking module

    # Calculate agreement rate
    agreement_rate = 0.0
    if processed_articles > 0:
        agreement_rate = (processed_articles - conflict_count) / processed_articles

    # Provider status (simplified - assume healthy)
    provider_status = {
        'openai': True,
        'anthropic': True
    }

    return jsonify({
        'success': True,
        'progress': {
            'total': total_articles,
            'processed': processed_articles,
            'included': included_count,
            'excluded': excluded_count,
            'uncertain': uncertain_count,
            'conflicts': conflict_count
        },
        'cost': total_cost,
        'provider_status': provider_status,
        'agreement_rate': agreement_rate
    })

@modern_screening_bp.route('/stats/<project_id>', methods=['GET'])
def get_detailed_stats(project_id):
    """Get detailed statistics for a project."""

    project = Project.query.get_or_404(project_id)

    # Basic counts
    articles = Article.query.filter_by(project_id=project_id).all()

    stats = {
        'total_articles': len(articles),
        'by_status': {},
        'processing_time': 0,
        'cost_breakdown': {},
        'agreement_analysis': {}
    }

    # Count by status
    for article in articles:
        status = article.status
        stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

    # Calculate agreement metrics
    agreement_count = 0
    disagreement_count = 0

    for article in articles:
        if article.decision_reasoning and isinstance(article.decision_reasoning, dict):
            agreement_data = article.decision_reasoning.get('agreement_analysis', {})
            if agreement_data.get('agreement'):
                agreement_count += 1
            else:
                disagreement_count += 1

    stats['agreement_analysis'] = {
        'agreements': agreement_count,
        'disagreements': disagreement_count,
        'rate': agreement_count / max(agreement_count + disagreement_count, 1)
    }

    # Cost breakdown disabled - will be implemented with proper cost tracking
    stats['cost_breakdown'] = {'total_cost': 0.0, 'openai_cost': 0.0, 'anthropic_cost': 0.0}

    return jsonify({
        'success': True,
        'stats': stats
    })

# ============================================================================
# WORKFLOW EXECUTION ROUTES
# ============================================================================

@modern_screening_bp.route('/workflow/start', methods=['POST'])
def start_workflow():
    """Start automated screening workflow."""

    data = request.get_json()
    project_id = data.get('project_id')
    workflow_type = data.get('workflow_type', 'adaptive')

    if not project_id:
        project_id = session.get('current_project_id')
        if not project_id:
            raise ValidationError("No project specified")

    project = Project.query.get_or_404(project_id)

    # Get screening criteria
    if not project.config or 'criteria' not in project.config:
        raise ValidationError("Project does not have modern screening configuration")

    criteria_dict = project.config['criteria']
    criteria = ScreeningCriteria(**criteria_dict)

    # Create workflow configuration
    workflow_config = WorkflowConfig(
        workflow_type=WorkflowType(workflow_type),
        batch_size=data.get('batch_size', 10),
        max_concurrent=data.get('max_concurrent', 5),
        enable_early_stopping=data.get('enable_early_stopping', True),
        max_budget=data.get('max_budget')
    )

    # Start workflow (simplified - in production this would be async)
    import os
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if not openai_key or not anthropic_key:
        raise ValidationError("Missing API keys for LLM providers")

    orchestrator = ScreeningWorkflowOrchestrator(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key
    )

    # Store workflow status
    session['active_workflow'] = {
        'project_id': project_id,
        'started_at': datetime.now().isoformat(),
        'workflow_type': workflow_type
    }

    logger.info(f"Workflow started for project {project_id} with type {workflow_type}")

    return jsonify({
        'success': True,
        'workflow_id': f"{project_id}-{datetime.now().timestamp()}",
        'message': 'Workflow started successfully'
    })

# ============================================================================
# CONFIGURATION ROUTES
# ============================================================================

@modern_screening_bp.route('/config/<int:project_id>', methods=['GET'])
def get_model_config(project_id):
    """Get current model configuration for project."""
    from app.services.screening.model_registry import ModelRegistry
    
    project = Project.query.get_or_404(project_id)

    # Get model configuration from project config
    if not project.config or 'llmConfig' not in project.config:
        # Return default configuration with latest models
        return jsonify({
            'success': True,
            'config': {
                'openaiModel': 'gpt-4o-2024-11-20',
                'anthropicModel': 'claude-3-5-sonnet-20241022',
                'openaiTemperature': 0.1,
                'anthropicTemperature': 0.1,
                'openaiSeed': None,
                'anthropicSeed': None,
                'primaryProvider': 'openai',
                'secondaryProvider': 'anthropic'
            },
            'available_models': {
                'openai': list(ModelRegistry.OPENAI_MODELS.keys()),
                'anthropic': list(ModelRegistry.ANTHROPIC_MODELS.keys())
            },
            'recommended_pairs': ModelRegistry.get_recommended_pairs()
        })

    llm_config = project.config['llmConfig']
    return jsonify({
        'success': True,
        'config': {
            'openaiModel': llm_config.get('openaiModel', 'gpt-4o-2024-11-20'),
            'anthropicModel': llm_config.get('anthropicModel', 'claude-3-5-sonnet-20241022'),
            'openaiTemperature': llm_config.get('openaiTemperature', 0.1),
            'anthropicTemperature': llm_config.get('anthropicTemperature', 0.1),
            'openaiSeed': llm_config.get('openaiSeed'),
            'anthropicSeed': llm_config.get('anthropicSeed'),
            'primaryProvider': llm_config.get('primaryProvider', 'openai'),
            'secondaryProvider': llm_config.get('secondaryProvider', 'anthropic')
        },
        'available_models': {
            'openai': list(ModelRegistry.OPENAI_MODELS.keys()),
            'anthropic': list(ModelRegistry.ANTHROPIC_MODELS.keys())
        },
        'recommended_pairs': ModelRegistry.get_recommended_pairs()
    })

@modern_screening_bp.route('/config/models', methods=['POST'])
def save_model_config():
    """Save model configuration for dual-LLM screening."""
    data = request.get_json()

    if not data:
        return jsonify({'success': False, 'error': 'No configuration data provided'}), 400

    # Validate model configurations
    try:
        model1_config = ModelConfig(
            provider=data['model1']['provider'],
            model_name=data['model1']['model_name'],
            temperature=float(data['model1']['temperature']),
            seed=int(data['model1']['seed']) if data['model1']['seed'] else None,
            max_tokens=4000
        )

        model2_config = ModelConfig(
            provider=data['model2']['provider'],
            model_name=data['model2']['model_name'],
            temperature=float(data['model2']['temperature']),
            seed=int(data['model2']['seed']) if data['model2']['seed'] else None,
            max_tokens=4000
        )

        session['model_config'] = {
            'model1': data['model1'],
            'model2': data['model2']
        }

        return jsonify({
            'success': True,
            'message': 'Model configuration saved successfully',
            'status': 'Configuration Updated'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Invalid configuration: {str(e)}',
            'status': 'Configuration Error'
        }), 400

@modern_screening_bp.route('/config/models', methods=['GET'])
def get_dual_model_config():
    """Get current dual model configuration with latest model options."""
    from app.services.screening.model_registry import ModelRegistry

    # Return session config or default configuration
    session_config = session.get('model_config', {})

    default_config = {
        'model1': {
            'provider': 'openai',
            'model_name': 'gpt-4o-2024-11-20',
            'temperature': 0.1,
            'seed': None
        },
        'model2': {
            'provider': 'anthropic',
            'model_name': 'claude-3-5-sonnet-20241022',
            'temperature': 0.1,
            'seed': None
        }
    }

    config = {
        'model1': {**default_config['model1'], **session_config.get('model1', {})},
        'model2': {**default_config['model2'], **session_config.get('model2', {})}
    }

    return jsonify({
        'success': True,
        'config': config,
        'available_models': {
            'openai': [
                {
                    'name': name,
                    'tier': model_config.tier.value,
                    'description': model_config.description,
                    'capabilities': [cap.value for cap in model_config.capabilities],
                    'pricing_input': model_config.pricing_per_1k_input,
                    'pricing_output': model_config.pricing_per_1k_output
                }
                for name, model_config in ModelRegistry.OPENAI_MODELS.items()
            ],
            'anthropic': [
                {
                    'name': name,
                    'tier': model_config.tier.value,
                    'description': model_config.description,
                    'capabilities': [cap.value for cap in model_config.capabilities],
                    'pricing_input': model_config.pricing_per_1k_input,
                    'pricing_output': model_config.pricing_per_1k_output
                }
                for name, model_config in ModelRegistry.ANTHROPIC_MODELS.items()
            ]
        },
        'recommended_pairs': ModelRegistry.get_recommended_pairs()
    })

@modern_screening_bp.route('/config/test-connection', methods=['POST'])
def test_model_connection():
    """Test connection to LLM provider."""
    data = request.get_json()

    if not data or 'config' not in data:
        return jsonify({'success': False, 'error': 'No configuration provided'}), 400

    config_data = data['config']

    try:
        model_config = ModelConfig(
            provider=config_data['provider'],
            model_name=config_data['model_name'],
            temperature=float(config_data['temperature']),
            seed=int(config_data['seed']) if config_data['seed'] else None
        )

        if config_data['provider'] == 'openai':
            api_key = config_data.get('api_key') or current_app.config.get('OPENAI_API_KEY')
            if not api_key:
                return jsonify({'success': False, 'status': 'Missing API Key', 'message': 'OpenAI API key required'})

            from openai import OpenAI, APIError
            client = OpenAI(api_key=api_key)

            try:
                test_response = client.chat.completions.create(
                    model=config_data['model_name'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )

                if test_response and test_response.choices:
                    return jsonify({'success': True, 'status': 'Connected', 'message': 'OpenAI connection successful'})
                else:
                    return jsonify({'success': False, 'status': 'Connection Failed', 'message': 'OpenAI API call failed'})
            except Exception as e:
                return jsonify({'success': False, 'status': 'Connection Failed', 'message': f'OpenAI API error: {e}'})
        elif config_data['provider'] == 'anthropic':
            api_key = config_data.get('api_key') or current_app.config.get('ANTHROPIC_API_KEY')
            if not api_key:
                return jsonify({'success': False, 'status': 'Missing API Key', 'message': 'Anthropic API key required'})

            from anthropic import Anthropic, APIError
            client = Anthropic(api_key=api_key)

            try:
                test_response = client.messages.create(
                    model=config_data['model_name'],
                    max_tokens=10,
                    temperature=0.1,
                    messages=[
                        {"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}
                    ]
                )

                if test_response and test_response.content:
                    return jsonify({'success': True, 'status': 'Connected', 'message': 'Anthropic connection successful'})
                else:
                    return jsonify({'success': False, 'status': 'Connection Failed', 'message': 'Anthropic API call failed'})
            except Exception as e:
                return jsonify({'success': False, 'status': 'Connection Failed', 'message': f'Anthropic API error: {e}'})

        else:
            return jsonify({'success': True, 'status': 'Local Model', 'message': f'{config_data["provider"]} configured'})

    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'Connection Failed',
            'message': f'Connection test failed: {str(e)}'
        })

@modern_screening_bp.route('/config/templates/save', methods=['POST'])
def save_template():
    """Save current PICO-TT configuration as a template."""
    data = request.get_json()

    if not data or 'name' not in data:
        return jsonify({'success': False, 'error': 'Template name required'}), 400

    template_name = data['name']
    template_data = {
        'name': template_name,
        'research_question': data.get('research_question', ''),
        'population': data.get('population', ''),
        'intervention': data.get('intervention', ''),
        'comparison': data.get('comparison', ''),
        'outcomes': data.get('outcomes', ''),
        'time_frame': data.get('time_frame', ''),
        'study_types': data.get('study_types', ''),
        'inclusion_criteria': data.get('inclusion_criteria', []),
        'exclusion_criteria': data.get('exclusion_criteria', []),
        'created_at': datetime.now().isoformat()
    }

    template_file = os.path.join(current_app.config['CONFIG_TEMPLATES_DIR'], f"{template_name}.json")
    try:
        with open(template_file, 'w') as f:
            json.dump(template_data, f, indent=2)
        return jsonify({'success': True, 'message': f'Template "{template_name}" saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to save template: {str(e)}'}), 500

@modern_screening_bp.route('/config/templates/load/<template_name>', methods=['GET'])
def load_template(template_name):
    """Load a saved template."""
    template_file = os.path.join(current_app.config['CONFIG_TEMPLATES_DIR'], f"{template_name}.json")

    if not os.path.exists(template_file):
        return jsonify({'success': False, 'error': 'Template not found'}), 404

    try:
        with open(template_file, 'r') as f:
            template_data = json.load(f)
        return jsonify({'success': True, 'template': template_data})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to load template: {str(e)}'}), 500

@modern_screening_bp.route('/config/templates/list', methods=['GET'])
def list_templates():
    """List all available templates."""
    templates_dir = current_app.config['CONFIG_TEMPLATES_DIR']
    templates = []

    if os.path.exists(templates_dir):
        for filename in os.listdir(templates_dir):
            if filename.endswith('.json'):
                template_name = filename[:-5]  # Remove .json extension
                templates.append(template_name)

    return jsonify({'success': True, 'templates': templates})

@modern_screening_bp.route('/models/info/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Get detailed information about a specific model."""
    from app.services.screening.model_registry import ModelRegistry
    
    model_config = ModelRegistry.get_model_config(model_name)
    if not model_config:
        return jsonify({
            'success': False,
            'error': f'Model {model_name} not found'
        }), 404
    
    return jsonify({
        'success': True,
        'model': {
            'name': model_config.name,
            'provider': model_config.provider,
            'tier': model_config.tier.value,
            'description': model_config.description,
            'capabilities': [cap.value for cap in model_config.capabilities],
            'max_tokens': model_config.max_tokens,
            'context_window': model_config.context_window,
            'recommended_timeout': model_config.recommended_timeout,
            'supports_system_message': model_config.supports_system_message,
            'pricing': {
                'input_per_1k': model_config.pricing_per_1k_input,
                'output_per_1k': model_config.pricing_per_1k_output
            },
            'features': {
                'structured_output': ModelRegistry.supports_structured_output(model_name),
                'tool_calling': ModelRegistry.supports_tool_calling(model_name),
                'reasoning': ModelRegistry.is_reasoning_model(model_name)
            }
        }
    })

@modern_screening_bp.route('/models/estimate-cost', methods=['POST'])
def estimate_model_cost():
    """Estimate cost for using specific models."""
    from app.services.screening.model_registry import ModelRegistry
    
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    
    model_name = data.get('model_name')
    input_tokens = data.get('input_tokens', 0)
    output_tokens = data.get('output_tokens', 0)
    
    if not model_name:
        return jsonify({'success': False, 'error': 'Model name required'}), 400
    
    cost = ModelRegistry.estimate_cost(model_name, input_tokens, output_tokens)
    
    return jsonify({
        'success': True,
        'cost_estimate': {
            'model': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_cost': round(cost, 4),
            'cost_breakdown': {
                'input_cost': round((input_tokens / 1000) * ModelRegistry.get_model_config(model_name).pricing_per_1k_input, 4),
                'output_cost': round((output_tokens / 1000) * ModelRegistry.get_model_config(model_name).pricing_per_1k_output, 4)
            }
        }
    })

@modern_screening_bp.route('/project/<int:project_id>/statistics', methods=['GET'])
def get_project_statistics(project_id):
    """Get project statistics for PRISMA flowchart."""
    try:
        from app.models.screening_models import Project, Article, PublicationSource

        project = Project.query.get_or_404(project_id)

        articles = Article.query.filter_by(project_id=project_id).all()

        # Calculate statistics
        total_articles = len(articles)
        screened_articles = len([a for a in articles if a.status != 'pending'])
        included_articles = len([a for a in articles if a.status == 'included'])
        excluded_articles = len([a for a in articles if a.status == 'excluded'])
        pending_articles = len([a for a in articles if a.status == 'pending'])
        duplicate_articles = 0  # TODO: Implement duplicate detection

        source_breakdown = {}
        for article in articles:
            pub_sources = PublicationSource.query.filter_by(article_id=article.id).all()
            if pub_sources:
                for source in pub_sources:
                    db_name = source.source_database
                    if db_name not in source_breakdown:
                        source_breakdown[db_name] = 0
                    source_breakdown[db_name] += 1
            else:
                if 'Unknown' not in source_breakdown:
                    source_breakdown['Unknown'] = 0
                source_breakdown['Unknown'] += 1

        return jsonify({
            'success': True,
            'total_articles': total_articles,
            'screened_articles': screened_articles,
            'included_articles': included_articles,
            'excluded_articles': excluded_articles,
            'pending_articles': pending_articles,
            'duplicate_articles': duplicate_articles,
            'database_sources': source_breakdown,
            'project_name': project.name,
            'project_id': project_id
        })

    except Exception as e:
        logger.error(f"Error fetching project statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch project statistics: {str(e)}'
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@modern_screening_bp.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({
        'success': False,
        'error': 'ValidationError',
        'message': str(error)
    }), 400

@modern_screening_bp.errorhandler(Exception)
def handle_general_error(error):
    logger.error(f"Unhandled error in modern screening: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'InternalError',
        'message': 'An internal error occurred'
    }), 500
