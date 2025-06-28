"""
Analytics API Routes
Flask routes for accessing screening analytics and metrics.
"""

import logging
from flask import Blueprint, request, jsonify, send_file
from datetime import datetime
import os
import tempfile

from app.services.screening.analytics import ScreeningAnalyticsAgent, AnalyticsFactory
from app.services.utils.error_handler import handle_file_parsing_error
from app.services.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Create Blueprint for analytics routes
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# ============================================================================
# COMPREHENSIVE ANALYTICS ROUTES
# ============================================================================

@analytics_bp.route('/project/<project_id>/dashboard', methods=['GET'])
def get_analytics_dashboard(project_id):
    """Get comprehensive analytics dashboard data."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        return jsonify({
            'success': True,
            'analytics': analytics,
            'generated_at': analytics['generated_at']
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/summary', methods=['GET'])
def get_project_summary(project_id):
    """Get quick project summary for dashboard cards."""

    try:
        summary = AnalyticsFactory.get_project_summary(project_id)

        return jsonify({
            'success': True,
            'summary': summary
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

# ============================================================================
# SPECIFIC METRICS ROUTES
# ============================================================================

@analytics_bp.route('/project/<project_id>/decisions', methods=['GET'])
def get_decision_distribution(project_id):
    """Get decision distribution metrics."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        return jsonify({
            'success': True,
            'decision_distribution': analytics['decision_distribution'],
            'trends': analytics['trend_analysis']
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/agreement', methods=['GET'])
def get_agreement_metrics(project_id):
    """Get provider agreement analysis."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        return jsonify({
            'success': True,
            'agreement_metrics': analytics['agreement_metrics'],
            'provider_comparison': analytics['provider_comparison']
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/quality', methods=['GET'])
def get_quality_metrics(project_id):
    """Get quality assessment metrics."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        return jsonify({
            'success': True,
            'quality_metrics': analytics['quality_metrics'],
            'confidence_analysis': analytics['confidence_analysis'],
            'pico_analysis': analytics['pico_extraction_analysis']
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/performance', methods=['GET'])
def get_performance_metrics(project_id):
    """Get performance and efficiency metrics."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        return jsonify({
            'success': True,
            'performance_metrics': analytics['performance_metrics'],
            'cost_analysis': analytics['cost_analysis']
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/human-review', methods=['GET'])
def get_human_review_analysis(project_id):
    """Get human review patterns and efficiency."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        return jsonify({
            'success': True,
            'human_review_analysis': analytics['human_review_analysis'],
            'recommendations': [r for r in analytics['recommendations'] if r['type'] in ['agreement', 'confidence']]
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

# ============================================================================
# EXPORT AND REPORTING ROUTES
# ============================================================================

@analytics_bp.route('/project/<project_id>/export', methods=['POST'])
def export_analytics_report(project_id):
    """Export comprehensive analytics report to JSON."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            filename = agent.export_comprehensive_report(tmp_file.name)

        # Send file and clean up
        def remove_file(response):
            try:
                os.unlink(filename)
            except Exception as e:
                # Import logging to log exceptions
                import logging
                logging.exception("Error while removing temporary file", exc_info=True)
            return response

        return send_file(
            filename,
            as_attachment=True,
            download_name=f'screening_analytics_{project_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mimetype='application/json'
        )

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/export/csv', methods=['POST'])
def export_csv_report(project_id):
    """Export screening results to CSV format."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        # Convert to CSV format
        import pandas as pd

        articles_data = analytics['export_data']['articles']

        # Flatten the data for CSV
        csv_data = []
        for article in articles_data:
            row = {
                'article_id': article['id'],
                'title': article['title'],
                'authors': article['authors'],
                'journal': article['journal'],
                'year': article['year'],
                'final_status': article['final_status'],
                'created_at': article['created_at'],
                'updated_at': article['updated_at']
            }

            # Add AI decision data if available
            if article['decision_reasoning']:
                reasoning = article['decision_reasoning']

                # OpenAI data
                openai_result = reasoning.get('openai_result', {})
                if openai_result:
                    openai_decision = openai_result.get('screening_decision', {})
                    row.update({
                        'openai_decision': openai_decision.get('final_decision'),
                        'openai_confidence': openai_decision.get('confidence_score'),
                        'openai_reasoning': openai_decision.get('primary_reason', '')[:200]  # Truncate
                    })

                # Anthropic data
                anthropic_result = reasoning.get('anthropic_result', {})
                if anthropic_result:
                    anthropic_decision = anthropic_result.get('screening_decision', {})
                    row.update({
                        'anthropic_decision': anthropic_decision.get('final_decision'),
                        'anthropic_confidence': anthropic_decision.get('confidence_score'),
                        'anthropic_reasoning': anthropic_decision.get('primary_reason', '')[:200]  # Truncate
                    })

                # Agreement data
                agreement = reasoning.get('agreement_analysis', {})
                row.update({
                    'providers_agree': agreement.get('agreement', False),
                    'confidence_difference': agreement.get('confidence_difference', 0),
                    'requires_human_review': reasoning.get('requires_human_review', False)
                })

                # Human decision if available
                human_decision = reasoning.get('human_decision')
                if human_decision:
                    row.update({
                        'human_decision': human_decision.get('decision'),
                        'human_reasoning': human_decision.get('reasoning', '')[:200]  # Truncate
                    })

            csv_data.append(row)

        # Create DataFrame and export to CSV
        df = pd.DataFrame(csv_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            csv_filename = tmp_file.name

        return send_file(
            csv_filename,
            as_attachment=True,
            download_name=f'screening_results_{project_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")
    except Exception as e:
        logger.error(f"CSV export failed: {str(e)}")
        raise ValidationError(f"Export failed: {str(e)}")

# ============================================================================
# REAL-TIME METRICS ROUTES (like original script)
# ============================================================================

@analytics_bp.route('/project/<project_id>/live-stats', methods=['GET'])
def get_live_statistics(project_id):
    """Get real-time statistics (similar to original script summary)."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        # Format like original RealDualLLMEvaluator summary
        live_stats = {
            'total_evaluations': analytics['performance_metrics']['total_articles_processed'],
            'decision_distribution': {
                'include': analytics['decision_distribution']['include_count'],
                'exclude': analytics['decision_distribution']['exclude_count'],
                'conflicts': analytics['decision_distribution']['conflict_count'],
                'uncertain': analytics['decision_distribution']['uncertain_count']
            },
            'quality_metrics': {
                'conflict_rate': analytics['agreement_metrics']['conflict_rate'],
                'human_review_rate': analytics['human_review_analysis']['human_review_rate'],
                'average_confidence': (
                    analytics['agreement_metrics']['provider_bias_analysis']['openai_avg_confidence'] +
                    analytics['agreement_metrics']['provider_bias_analysis']['anthropic_avg_confidence']
                ) / 2,
                'average_agreement': analytics['agreement_metrics']['decision_agreement_rate']
            },
            'api_performance': {
                'gpt4_errors': 0,  # Would track from error logs
                'claude_errors': 0,  # Would track from error logs
                'average_response_time': analytics['performance_metrics']['average_response_time'],
                'api_error_rate': analytics['performance_metrics']['api_error_rate']
            },
            'cost_summary': {
                'total_cost': analytics['cost_analysis']['total_cost'],
                'cost_per_article': analytics['cost_analysis']['efficiency_metrics']['cost_per_article']
            }
        }

        return jsonify({
            'success': True,
            'live_stats': live_stats,
            'last_updated': analytics['generated_at']
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

@analytics_bp.route('/project/<project_id>/provider-comparison', methods=['GET'])
def get_detailed_provider_comparison(project_id):
    """Get detailed provider comparison (enhanced version of original script)."""

    try:
        agent = AnalyticsFactory.create_agent(project_id)
        analytics = agent.get_comprehensive_analytics()

        provider_data = analytics['provider_comparison']

        # Enhanced comparison with recommendations
        comparison = {
            'openai_analysis': {
                'model_name': 'GPT-4o',
                'total_evaluations': provider_data['openai_stats']['total_evaluations'],
                'inclusion_rate': provider_data['openai_stats']['inclusion_rate'],
                'average_confidence': provider_data['openai_stats']['average_confidence'],
                'performance_score': provider_data['openai_stats']['average_confidence'] * (1 - analytics['performance_metrics']['api_error_rate'] / 100)
            },
            'anthropic_analysis': {
                'model_name': 'Claude-3.5-Sonnet',
                'total_evaluations': provider_data['anthropic_stats']['total_evaluations'],
                'inclusion_rate': provider_data['anthropic_stats']['inclusion_rate'],
                'average_confidence': provider_data['anthropic_stats']['average_confidence'],
                'performance_score': provider_data['anthropic_stats']['average_confidence'] * (1 - analytics['performance_metrics']['api_error_rate'] / 100)
            },
            'agreement_analysis': provider_data['agreement_analysis'],
            'bias_detection': provider_data['bias_detection'],
            'recommendations': [
                r for r in analytics['recommendations']
                if r['type'] in ['agreement', 'performance']
            ]
        }

        return jsonify({
            'success': True,
            'provider_comparison': comparison
        })

    except ValueError as e:
        raise ValidationError(f"Invalid project: {str(e)}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@analytics_bp.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({
        'success': False,
        'error': 'ValidationError',
        'message': str(error)
    }), 400

@analytics_bp.errorhandler(Exception)
def handle_general_error(error):
    logger.error(f"Unhandled error in analytics: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'InternalError',
        'message': 'An internal error occurred'
    }), 500
