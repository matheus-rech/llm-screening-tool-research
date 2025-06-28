"""
Advanced Screening Analytics & Metrics Agent
Comprehensive scoring, metrics tracking, and performance analysis for systematic review screening.
Inspired by RealDualLLMEvaluator.py but enhanced for production use.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pandas as pd
from sqlalchemy import func

from app.models.screening_models import db, Project, Article
from .modern_llm import ComprehensiveScreeningResult
# from app.services.utils.cost_tracker import cost_tracker  # Temporarily disabled

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS DATA MODELS
# ============================================================================

@dataclass
class AgreementMetrics:
    """Agreement analysis between LLM providers."""
    decision_agreement_rate: float
    confidence_correlation: float
    average_confidence_delta: float
    conflict_rate: float
    low_confidence_rate: float
    agreement_score: float
    provider_bias_analysis: Dict[str, float]

@dataclass
class QualityMetrics:
    """Quality assessment metrics for screening performance."""
    extraction_completeness_avg: float
    reasoning_quality_avg: float
    consistency_score: float
    human_ai_agreement_rate: float
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Performance and efficiency metrics."""
    total_articles_processed: int
    processing_speed_per_hour: float
    average_response_time: float
    api_error_rate: float
    cost_per_article: float
    total_cost: float
    human_review_efficiency: float

@dataclass
class DecisionDistribution:
    """Distribution of screening decisions."""
    include_count: int
    exclude_count: int
    uncertain_count: int
    conflict_count: int
    human_review_count: int
    
    @property
    def total(self) -> int:
        return self.include_count + self.exclude_count + self.uncertain_count + self.conflict_count
    
    @property
    def inclusion_rate(self) -> float:
        return self.include_count / max(self.total, 1) * 100
    
    @property
    def exclusion_rate(self) -> float:
        return self.exclude_count / max(self.total, 1) * 100

@dataclass
class ProviderComparison:
    """Detailed comparison between LLM providers."""
    openai_stats: Dict[str, Any]
    anthropic_stats: Dict[str, Any]
    agreement_analysis: Dict[str, Any]
    bias_detection: Dict[str, Any]
    performance_comparison: Dict[str, Any]

# ============================================================================
# SCREENING ANALYTICS AGENT
# ============================================================================

class ScreeningAnalyticsAgent:
    """
    Advanced analytics agent for systematic review screening.
    Tracks metrics, scores, and performance across all screening activities.
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project = Project.query.get(project_id)
        if not self.project:
            raise ValueError(f"Project {project_id} not found")
        
        # Initialize metric caches
        self._metrics_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
    def get_comprehensive_analytics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get complete analytics dashboard for the project.
        """
        
        if not force_refresh and self._is_cache_valid():
            return self._metrics_cache
        
        logger.info(f"Generating comprehensive analytics for project {self.project_id}")
        
        # Get all articles with screening data
        articles = self._get_screened_articles()
        
        if not articles:
            return self._empty_analytics()
        
        # Calculate all metrics
        analytics = {
            'project_info': self._get_project_info(),
            'decision_distribution': self._calculate_decision_distribution(articles),
            'agreement_metrics': self._calculate_agreement_metrics(articles),
            'quality_metrics': self._calculate_quality_metrics(articles),
            'performance_metrics': self._calculate_performance_metrics(articles),
            'provider_comparison': self._calculate_provider_comparison(articles),
            'trend_analysis': self._calculate_trend_analysis(articles),
            'confidence_analysis': self._calculate_confidence_analysis(articles),
            'pico_extraction_analysis': self._calculate_pico_analysis(articles),
            'human_review_analysis': self._calculate_human_review_analysis(articles),
            'cost_analysis': self._calculate_cost_analysis(),
            'recommendations': self._generate_recommendations(articles),
            'export_data': self._prepare_export_data(articles),
            'generated_at': datetime.now().isoformat()
        }
        
        # Cache results
        self._metrics_cache = analytics
        self._cache_timestamp = datetime.now()
        
        return analytics
    
    def _get_screened_articles(self) -> List[Article]:
        """Get articles with screening decision data."""
        return Article.query.filter(
            Article.project_id == self.project_id,
            Article.decision_reasoning.isnot(None),
            Article.status.in_(['included', 'excluded', 'uncertain', 'human_review_required'])
        ).all()
    
    def _calculate_decision_distribution(self, articles: List[Article]) -> DecisionDistribution:
        """Calculate distribution of screening decisions."""
        
        decisions = []
        for article in articles:
            if article.decision_reasoning and isinstance(article.decision_reasoning, dict):
                final_decision = article.decision_reasoning.get('final_decision', 'UNCERTAIN')
                decisions.append(final_decision)
            else:
                # Fallback to article status
                if article.status == 'included':
                    decisions.append('INCLUDE')
                elif article.status == 'excluded':
                    decisions.append('EXCLUDE')
                elif article.status == 'human_review_required':
                    decisions.append('CONFLICT')
                else:
                    decisions.append('UNCERTAIN')
        
        return DecisionDistribution(
            include_count=decisions.count('INCLUDE'),
            exclude_count=decisions.count('EXCLUDE'),
            uncertain_count=decisions.count('UNCERTAIN'),
            conflict_count=decisions.count('CONFLICT'),
            human_review_count=sum(1 for a in articles if a.status == 'human_review_required')
        )
    
    def _calculate_agreement_metrics(self, articles: List[Article]) -> AgreementMetrics:
        """Calculate agreement metrics between LLM providers."""
        
        agreements = []
        confidence_deltas = []
        openai_confidences = []
        anthropic_confidences = []
        openai_decisions = []
        anthropic_decisions = []
        
        for article in articles:
            if not article.decision_reasoning or not isinstance(article.decision_reasoning, dict):
                continue
                
            reasoning = article.decision_reasoning
            openai_result = reasoning.get('openai_result')
            anthropic_result = reasoning.get('anthropic_result')
            
            if not openai_result or not anthropic_result:
                continue
            
            # Extract decisions and confidences
            openai_decision = openai_result.get('screening_decision', {}).get('final_decision')
            anthropic_decision = anthropic_result.get('screening_decision', {}).get('final_decision')
            openai_conf = openai_result.get('screening_decision', {}).get('confidence_score', 0)
            anthropic_conf = anthropic_result.get('screening_decision', {}).get('confidence_score', 0)
            
            if openai_decision and anthropic_decision:
                # Decision agreement
                agreements.append(openai_decision == anthropic_decision)
                openai_decisions.append(openai_decision)
                anthropic_decisions.append(anthropic_decision)
                
                # Confidence analysis
                confidence_deltas.append(abs(openai_conf - anthropic_conf))
                openai_confidences.append(openai_conf)
                anthropic_confidences.append(anthropic_conf)
        
        if not agreements:
            return AgreementMetrics(0, 0, 0, 0, 0, 0, {})
        
        # Calculate metrics
        decision_agreement_rate = sum(agreements) / len(agreements) * 100
        confidence_correlation = np.corrcoef(openai_confidences, anthropic_confidences)[0, 1] if len(openai_confidences) > 1 else 0
        average_confidence_delta = np.mean(confidence_deltas) if confidence_deltas else 0
        conflict_rate = (len(agreements) - sum(agreements)) / len(agreements) * 100
        
        # Low confidence rate (either provider < 70%)
        low_confidence_count = sum(1 for oc, ac in zip(openai_confidences, anthropic_confidences) if min(oc, ac) < 0.7)
        low_confidence_rate = low_confidence_count / len(agreements) * 100
        
        # Overall agreement score
        agreement_score = (decision_agreement_rate / 100) * (1 - average_confidence_delta)
        
        # Provider bias analysis
        openai_include_rate = openai_decisions.count('INCLUDE') / len(openai_decisions) * 100 if openai_decisions else 0
        anthropic_include_rate = anthropic_decisions.count('INCLUDE') / len(anthropic_decisions) * 100 if anthropic_decisions else 0
        
        provider_bias_analysis = {
            'openai_inclusion_bias': openai_include_rate - 50,  # Deviation from neutral
            'anthropic_inclusion_bias': anthropic_include_rate - 50,
            'bias_difference': abs(openai_include_rate - anthropic_include_rate),
            'openai_avg_confidence': np.mean(openai_confidences) if openai_confidences else 0,
            'anthropic_avg_confidence': np.mean(anthropic_confidences) if anthropic_confidences else 0
        }
        
        return AgreementMetrics(
            decision_agreement_rate=decision_agreement_rate,
            confidence_correlation=confidence_correlation,
            average_confidence_delta=average_confidence_delta,
            conflict_rate=conflict_rate,
            low_confidence_rate=low_confidence_rate,
            agreement_score=agreement_score,
            provider_bias_analysis=provider_bias_analysis
        )
    
    def _calculate_quality_metrics(self, articles: List[Article]) -> QualityMetrics:
        """Calculate quality assessment metrics."""
        
        extraction_completeness_scores = []
        reasoning_quality_scores = []
        human_ai_agreements = []
        
        for article in articles:
            if not article.decision_reasoning or not isinstance(article.decision_reasoning, dict):
                continue
                
            reasoning = article.decision_reasoning
            
            # Extract quality scores from AI results
            for provider in ['openai_result', 'anthropic_result']:
                result = reasoning.get(provider)
                if result:
                    extraction_completeness_scores.append(result.get('extraction_completeness', 0))
                    reasoning_quality_scores.append(result.get('reasoning_quality', 0))
            
            # Human-AI agreement (if human decision exists)
            human_decision = reasoning.get('human_decision')
            ai_decision = reasoning.get('final_decision')
            if human_decision and ai_decision:
                human_ai_agreements.append(human_decision.get('decision') == ai_decision)
        
        # Calculate consistency score based on provider agreement
        agreement_data = self._calculate_agreement_metrics(articles)
        consistency_score = agreement_data.agreement_score
        
        return QualityMetrics(
            extraction_completeness_avg=np.mean(extraction_completeness_scores) if extraction_completeness_scores else 0,
            reasoning_quality_avg=np.mean(reasoning_quality_scores) if reasoning_quality_scores else 0,
            consistency_score=consistency_score,
            human_ai_agreement_rate=np.mean(human_ai_agreements) * 100 if human_ai_agreements else None
        )
    
    def _calculate_performance_metrics(self, articles: List[Article]) -> PerformanceMetrics:
        """Calculate performance and efficiency metrics."""
        
        total_articles = len(articles)
        
        # Calculate processing timeline
        if articles:
            start_time = min(a.created_at for a in articles if a.created_at)
            end_time = max(a.updated_at for a in articles if a.updated_at)
            processing_duration = (end_time - start_time).total_seconds() / 3600  # hours
            processing_speed = total_articles / max(processing_duration, 0.1)
        else:
            processing_speed = 0
        
        # Calculate average response time (simplified)
        response_times = []
        api_errors = 0
        
        for article in articles:
            if article.decision_reasoning and isinstance(article.decision_reasoning, dict):
                reasoning = article.decision_reasoning
                
                # Check for API errors
                openai_result = reasoning.get('openai_result')
                anthropic_result = reasoning.get('anthropic_result')
                
                if not openai_result:
                    api_errors += 1
                if not anthropic_result:
                    api_errors += 1
                
                # Estimate response time (would be better if stored)
                response_times.append(2.5)  # Average estimated time per API call
        
        api_error_rate = (api_errors / (total_articles * 2)) * 100 if total_articles > 0 else 0
        average_response_time = np.mean(response_times) if response_times else 0
        
        # Cost analysis
        cost_data = cost_tracker.get_current_costs(self.project_id)
        total_cost = cost_data.get('total_cost', 0) if cost_data else 0
        cost_per_article = total_cost / max(total_articles, 1)
        
        # Human review efficiency
        human_reviews_needed = sum(1 for a in articles if a.status == 'human_review_required')
        human_review_efficiency = (1 - human_reviews_needed / max(total_articles, 1)) * 100
        
        return PerformanceMetrics(
            total_articles_processed=total_articles,
            processing_speed_per_hour=processing_speed,
            average_response_time=average_response_time,
            api_error_rate=api_error_rate,
            cost_per_article=cost_per_article,
            total_cost=total_cost,
            human_review_efficiency=human_review_efficiency
        )
    
    def _calculate_provider_comparison(self, articles: List[Article]) -> ProviderComparison:
        """Detailed comparison between LLM providers."""
        
        openai_decisions = []
        anthropic_decisions = []
        openai_confidences = []
        anthropic_confidences = []
        openai_response_times = []
        anthropic_response_times = []
        
        for article in articles:
            if not article.decision_reasoning or not isinstance(article.decision_reasoning, dict):
                continue
                
            reasoning = article.decision_reasoning
            openai_result = reasoning.get('openai_result')
            anthropic_result = reasoning.get('anthropic_result')
            
            if openai_result:
                openai_decisions.append(openai_result.get('screening_decision', {}).get('final_decision'))
                openai_confidences.append(openai_result.get('screening_decision', {}).get('confidence_score', 0))
                openai_response_times.append(2.0)  # Estimated
                
            if anthropic_result:
                anthropic_decisions.append(anthropic_result.get('screening_decision', {}).get('final_decision'))
                anthropic_confidences.append(anthropic_result.get('screening_decision', {}).get('confidence_score', 0))
                anthropic_response_times.append(3.0)  # Estimated
        
        # OpenAI stats
        openai_stats = {
            'total_evaluations': len(openai_decisions),
            'inclusion_rate': openai_decisions.count('INCLUDE') / max(len(openai_decisions), 1) * 100,
            'exclusion_rate': openai_decisions.count('EXCLUDE') / max(len(openai_decisions), 1) * 100,
            'average_confidence': np.mean(openai_confidences) if openai_confidences else 0,
            'confidence_std': np.std(openai_confidences) if openai_confidences else 0,
            'average_response_time': np.mean(openai_response_times) if openai_response_times else 0
        }
        
        # Anthropic stats
        anthropic_stats = {
            'total_evaluations': len(anthropic_decisions),
            'inclusion_rate': anthropic_decisions.count('INCLUDE') / max(len(anthropic_decisions), 1) * 100,
            'exclusion_rate': anthropic_decisions.count('EXCLUDE') / max(len(anthropic_decisions), 1) * 100,
            'average_confidence': np.mean(anthropic_confidences) if anthropic_confidences else 0,
            'confidence_std': np.std(anthropic_confidences) if anthropic_confidences else 0,
            'average_response_time': np.mean(anthropic_response_times) if anthropic_response_times else 0
        }
        
        # Agreement analysis
        agreements = [od == ad for od, ad in zip(openai_decisions, anthropic_decisions) if od and ad]
        agreement_analysis = {
            'decision_agreement_rate': sum(agreements) / max(len(agreements), 1) * 100,
            'inclusion_bias_difference': abs(openai_stats['inclusion_rate'] - anthropic_stats['inclusion_rate']),
            'confidence_correlation': np.corrcoef(openai_confidences, anthropic_confidences)[0, 1] if len(openai_confidences) > 1 else 0
        }
        
        return ProviderComparison(
            openai_stats=openai_stats,
            anthropic_stats=anthropic_stats,
            agreement_analysis=agreement_analysis,
            bias_detection={
                'openai_conservative_bias': 50 - openai_stats['inclusion_rate'],
                'anthropic_conservative_bias': 50 - anthropic_stats['inclusion_rate']
            },
            performance_comparison={
                'speed_advantage': 'openai' if openai_stats['average_response_time'] < anthropic_stats['average_response_time'] else 'anthropic',
                'confidence_advantage': 'openai' if openai_stats['average_confidence'] > anthropic_stats['average_confidence'] else 'anthropic'
            }
        )
    
    def _calculate_trend_analysis(self, articles: List[Article]) -> Dict[str, Any]:
        """Calculate trends over time."""
        
        # Group articles by date
        daily_stats = defaultdict(lambda: {'include': 0, 'exclude': 0, 'total': 0})
        
        for article in articles:
            if article.updated_at:
                date_key = article.updated_at.date().isoformat()
                daily_stats[date_key]['total'] += 1
                
                if article.status == 'included':
                    daily_stats[date_key]['include'] += 1
                elif article.status == 'excluded':
                    daily_stats[date_key]['exclude'] += 1
        
        # Calculate daily inclusion rates
        trend_data = []
        for date, stats in sorted(daily_stats.items()):
            inclusion_rate = stats['include'] / max(stats['total'], 1) * 100
            trend_data.append({
                'date': date,
                'inclusion_rate': inclusion_rate,
                'total_processed': stats['total']
            })
        
        return {
            'daily_trends': trend_data,
            'trend_direction': self._calculate_trend_direction(trend_data),
            'processing_velocity': len(articles) / max(len(daily_stats), 1)  # articles per day
        }
    
    def _calculate_confidence_analysis(self, articles: List[Article]) -> Dict[str, Any]:
        """Analyze confidence distributions and patterns."""
        
        all_confidences = []
        low_confidence_articles = []
        high_confidence_articles = []
        
        for article in articles:
            if not article.decision_reasoning or not isinstance(article.decision_reasoning, dict):
                continue
                
            reasoning = article.decision_reasoning
            
            for provider in ['openai_result', 'anthropic_result']:
                result = reasoning.get(provider)
                if result:
                    confidence = result.get('screening_decision', {}).get('confidence_score', 0)
                    all_confidences.append(confidence)
                    
                    if confidence < 0.7:
                        low_confidence_articles.append(article.id)
                    elif confidence > 0.9:
                        high_confidence_articles.append(article.id)
        
        return {
            'confidence_distribution': {
                'mean': np.mean(all_confidences) if all_confidences else 0,
                'median': np.median(all_confidences) if all_confidences else 0,
                'std': np.std(all_confidences) if all_confidences else 0,
                'min': np.min(all_confidences) if all_confidences else 0,
                'max': np.max(all_confidences) if all_confidences else 0
            },
            'confidence_categories': {
                'high_confidence_count': len(set(high_confidence_articles)),
                'low_confidence_count': len(set(low_confidence_articles)),
                'high_confidence_rate': len(set(high_confidence_articles)) / max(len(articles), 1) * 100,
                'low_confidence_rate': len(set(low_confidence_articles)) / max(len(articles), 1) * 100
            }
        }
    
    def _calculate_pico_analysis(self, articles: List[Article]) -> Dict[str, Any]:
        """Analyze PICO extraction quality and completeness."""
        
        pico_completeness = {
            'population': [],
            'intervention': [],
            'comparison': [],
            'outcomes': [],
            'time_frame': [],
            'study_types': []
        }
        
        for article in articles:
            if not article.decision_reasoning or not isinstance(article.decision_reasoning, dict):
                continue
                
            reasoning = article.decision_reasoning
            
            for provider in ['openai_result', 'anthropic_result']:
                result = reasoning.get(provider)
                if result and 'picott_extraction' in result:
                    pico_data = result['picott_extraction']
                    
                    for component in pico_completeness.keys():
                        value = pico_data.get(component)
                        if value and str(value).strip() and str(value).lower() != 'not identified':
                            pico_completeness[component].append(1)
                        else:
                            pico_completeness[component].append(0)
        
        # Calculate completion rates
        pico_completion_rates = {}
        for component, values in pico_completeness.items():
            pico_completion_rates[component] = np.mean(values) * 100 if values else 0
        
        return {
            'pico_completion_rates': pico_completion_rates,
            'overall_pico_completeness': np.mean(list(pico_completion_rates.values())),
            'most_extracted_component': max(pico_completion_rates, key=pico_completion_rates.get) if pico_completion_rates else None,
            'least_extracted_component': min(pico_completion_rates, key=pico_completion_rates.get) if pico_completion_rates else None
        }
    
    def _calculate_human_review_analysis(self, articles: List[Article]) -> Dict[str, Any]:
        """Analyze human review patterns and efficiency."""
        
        human_reviews = []
        ai_human_agreements = []
        review_reasons = []
        
        for article in articles:
            if not article.decision_reasoning or not isinstance(article.decision_reasoning, dict):
                continue
                
            reasoning = article.decision_reasoning
            human_decision = reasoning.get('human_decision')
            
            if human_decision:
                human_reviews.append(human_decision)
                
                # Check AI-human agreement
                ai_decision = reasoning.get('final_decision')
                if ai_decision:
                    ai_human_agreements.append(human_decision.get('decision') == ai_decision)
                
                # Collect review triggers
                triggers = reasoning.get('human_review_triggers', {})
                if triggers.get('triggers'):
                    review_reasons.extend(triggers['triggers'])
        
        # Analyze review reasons
        reason_counts = Counter(review_reasons)
        
        return {
            'total_human_reviews': len(human_reviews),
            'human_review_rate': len(human_reviews) / max(len(articles), 1) * 100,
            'ai_human_agreement_rate': np.mean(ai_human_agreements) * 100 if ai_human_agreements else None,
            'common_review_triggers': dict(reason_counts.most_common(5)),
            'review_efficiency_score': (1 - len(human_reviews) / max(len(articles), 1)) * 100
        }
    
    def _calculate_cost_analysis(self) -> Dict[str, Any]:
        """Analyze cost patterns and efficiency."""
        
        cost_data = cost_tracker.get_current_costs(self.project_id)
        
        if not cost_data:
            return {'total_cost': 0, 'cost_breakdown': {}, 'efficiency_metrics': {}}
        
        return {
            'total_cost': cost_data.get('total_cost', 0),
            'cost_breakdown': cost_data.get('breakdown', {}),
            'efficiency_metrics': {
                'cost_per_article': cost_data.get('cost_per_article', 0),
                'cost_per_inclusion': cost_data.get('cost_per_inclusion', 0),
                'projected_total_cost': cost_data.get('projected_cost', 0)
            },
            'cost_optimization_suggestions': self._generate_cost_suggestions(cost_data)
        }
    
    def _generate_recommendations(self, articles: List[Article]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analytics."""
        
        recommendations = []
        
        # Analyze current metrics
        agreement_metrics = self._calculate_agreement_metrics(articles)
        performance_metrics = self._calculate_performance_metrics(articles)
        
        # Agreement-based recommendations
        if agreement_metrics.conflict_rate > 20:
            recommendations.append({
                'type': 'agreement',
                'priority': 'high',
                'title': 'High Conflict Rate Detected',
                'description': f'Providers disagree on {agreement_metrics.conflict_rate:.1f}% of articles. Consider refining inclusion criteria or adding domain-specific prompts.',
                'action': 'Review and refine screening criteria'
            })
        
        # Confidence-based recommendations
        if agreement_metrics.low_confidence_rate > 30:
            recommendations.append({
                'type': 'confidence',
                'priority': 'medium',
                'title': 'Low Confidence Patterns',
                'description': f'{agreement_metrics.low_confidence_rate:.1f}% of decisions have low confidence. Consider providing more specific criteria.',
                'action': 'Enhance prompt specificity'
            })
        
        # Cost-based recommendations
        if performance_metrics.cost_per_article > 0.01:
            recommendations.append({
                'type': 'cost',
                'priority': 'medium',
                'title': 'Cost Optimization Opportunity',
                'description': f'Cost per article is ${performance_metrics.cost_per_article:.3f}. Consider batch processing or model optimization.',
                'action': 'Implement cost optimization strategies'
            })
        
        # Performance recommendations
        if performance_metrics.api_error_rate > 5:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'API Reliability Issues',
                'description': f'API error rate is {performance_metrics.api_error_rate:.1f}%. Implement better error handling and retry logic.',
                'action': 'Improve API reliability'
            })
        
        return recommendations
    
    def _generate_cost_suggestions(self, cost_data: Dict) -> List[str]:
        """Generate cost optimization suggestions."""
        
        suggestions = []
        total_cost = cost_data.get('total_cost', 0)
        
        if total_cost > 50:
            suggestions.append("Consider implementing early stopping for obvious exclusions")
        
        if cost_data.get('breakdown', {}).get('gpt4', 0) > cost_data.get('breakdown', {}).get('claude', 0):
            suggestions.append("OpenAI costs are higher - consider using Claude for initial screening")
        
        return suggestions
    
    def _prepare_export_data(self, articles: List[Article]) -> Dict[str, Any]:
        """Prepare data for export (similar to original RealDualLLMEvaluator)."""
        
        export_articles = []
        
        for article in articles:
            article_data = {
                'id': article.id,
                'title': article.title,
                'abstract': article.abstract,
                'authors': article.authors,
                'journal': article.journal,
                'year': article.year,
                'final_status': article.status,
                'decision_reasoning': article.decision_reasoning,
                'created_at': article.created_at.isoformat() if article.created_at else None,
                'updated_at': article.updated_at.isoformat() if article.updated_at else None
            }
            export_articles.append(article_data)
        
        return {
            'articles': export_articles,
            'summary_statistics': asdict(self._calculate_decision_distribution(articles)),
            'agreement_metrics': asdict(self._calculate_agreement_metrics(articles)),
            'quality_metrics': asdict(self._calculate_quality_metrics(articles))
        }
    
    def export_comprehensive_report(self, filename: str) -> str:
        """Export comprehensive analytics report to JSON (like original script)."""
        
        analytics = self.get_comprehensive_analytics(force_refresh=True)
        
        # Add metadata
        export_data = {
            'project_info': analytics['project_info'],
            'screening_summary': analytics['decision_distribution'],
            'quality_metrics': analytics['quality_metrics'],
            'agreement_analysis': analytics['agreement_metrics'],
            'provider_comparison': analytics['provider_comparison'],
            'performance_metrics': analytics['performance_metrics'],
            'cost_analysis': analytics['cost_analysis'],
            'recommendations': analytics['recommendations'],
            'detailed_results': analytics['export_data'],
            'export_metadata': {
                'exported_at': datetime.now().isoformat(),
                'project_id': self.project_id,
                'total_articles_analyzed': analytics['performance_metrics']['total_articles_processed'],
                'analytics_version': '2.0'
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Comprehensive analytics report exported to {filename}")
        return filename
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _is_cache_valid(self) -> bool:
        """Check if metrics cache is still valid."""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            'project_info': self._get_project_info(),
            'message': 'No screening data available',
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_project_info(self) -> Dict[str, Any]:
        """Get basic project information."""
        return {
            'id': self.project.id,
            'name': self.project.name,
            'created_at': self.project.created_at.isoformat() if self.project.created_at else None,
            'total_articles': Article.query.filter_by(project_id=self.project_id).count(),
            'config': self.project.config or {}
        }
    
    def _calculate_trend_direction(self, trend_data: List[Dict]) -> str:
        """Calculate overall trend direction."""
        if len(trend_data) < 2:
            return 'insufficient_data'
        
        rates = [t['inclusion_rate'] for t in trend_data]
        if rates[-1] > rates[0]:
            return 'increasing'
        elif rates[-1] < rates[0]:
            return 'decreasing'
        else:
            return 'stable'

# ============================================================================
# ANALYTICS FACTORY
# ============================================================================

class AnalyticsFactory:
    """Factory for creating analytics agents."""
    
    @staticmethod
    def create_agent(project_id: str) -> ScreeningAnalyticsAgent:
        """Create analytics agent for a project."""
        return ScreeningAnalyticsAgent(project_id)
    
    @staticmethod
    def get_project_summary(project_id: str) -> Dict[str, Any]:
        """Get quick project summary."""
        agent = ScreeningAnalyticsAgent(project_id)
        analytics = agent.get_comprehensive_analytics()
        
        return {
            'project_id': project_id,
            'total_processed': analytics['performance_metrics']['total_articles_processed'],
            'inclusion_rate': analytics['decision_distribution']['inclusion_rate'],
            'agreement_rate': analytics['agreement_metrics']['decision_agreement_rate'],
            'cost': analytics['cost_analysis']['total_cost'],
            'status': 'active' if analytics['performance_metrics']['total_articles_processed'] > 0 else 'inactive'
        }