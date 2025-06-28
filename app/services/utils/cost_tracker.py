"""
Cost Tracking and Estimation System for LLM Screening
Based on AIScreenR patterns but enhanced with our architecture.

Integration Points:
- Uses our config_manager.py for cost configuration
- Integrates with our error_handler.py for robust cost calculations
- Connects to our database models for tracking actual usage
- Works with our concurrent_processor.py for batch cost optimization
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import tiktoken
from decimal import Decimal, ROUND_UP
import threading
from collections import defaultdict

from app.models.screening_models import db, Project, Article
from app.services.utils.config_manager import ProjectConfiguration
from app.services.utils.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class ModelPricing:
    """Pricing information for LLM models."""
    model_name: str
    input_cost_per_1k_tokens: Decimal
    output_cost_per_1k_tokens: Decimal
    context_window: int
    notes: str = ""
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate cost for given token usage."""
        input_cost = (Decimal(input_tokens) / 1000) * self.input_cost_per_1k_tokens
        output_cost = (Decimal(output_tokens) / 1000) * self.output_cost_per_1k_tokens
        return input_cost + output_cost

# Current pricing as of 2024 (update regularly!)
MODEL_PRICING = {
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        input_cost_per_1k_tokens=Decimal("0.00015"),
        output_cost_per_1k_tokens=Decimal("0.0006"),
        context_window=128000,
        notes="Most cost-effective for screening"
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        input_cost_per_1k_tokens=Decimal("0.0005"),
        output_cost_per_1k_tokens=Decimal("0.0015"),
        context_window=16000,
        notes="Balanced cost and performance"
    ),
    "gpt-4": ModelPricing(
        model_name="gpt-4",
        input_cost_per_1k_tokens=Decimal("0.03"),
        output_cost_per_1k_tokens=Decimal("0.06"),
        context_window=8000,
        notes="Highest quality but expensive"
    ),
    "gpt-4-turbo": ModelPricing(
        model_name="gpt-4-turbo",
        input_cost_per_1k_tokens=Decimal("0.01"),
        output_cost_per_1k_tokens=Decimal("0.03"),
        context_window=128000,
        notes="Good balance for complex screening"
    ),
    "claude-3-haiku": ModelPricing(
        model_name="claude-3-haiku",
        input_cost_per_1k_tokens=Decimal("0.00025"),
        output_cost_per_1k_tokens=Decimal("0.00125"),
        context_window=200000,
        notes="Anthropic's cost-effective option"
    ),
    "claude-3-sonnet": ModelPricing(
        model_name="claude-3-sonnet",
        input_cost_per_1k_tokens=Decimal("0.003"),
        output_cost_per_1k_tokens=Decimal("0.015"),
        context_window=200000,
        notes="Anthropic's balanced option"
    ),
    "gemini-pro": ModelPricing(
        model_name="gemini-pro",
        input_cost_per_1k_tokens=Decimal("0.0005"),
        output_cost_per_1k_tokens=Decimal("0.0015"),
        context_window=32000,
        notes="Google's Gemini Pro model"
    ),
    "gemini-pro-vision": ModelPricing(
        model_name="gemini-pro-vision",
        input_cost_per_1k_tokens=Decimal("0.00025"),
        output_cost_per_1k_tokens=Decimal("0.0005"),
        context_window=16000,
        notes="Google's multimodal model"
    ),
    "llama2-7b": ModelPricing(
        model_name="llama2-7b",
        input_cost_per_1k_tokens=Decimal("0.0001"),
        output_cost_per_1k_tokens=Decimal("0.0002"),
        context_window=4000,
        notes="Ollama local Llama2 7B (estimated cost)"
    ),
    "llama2-13b": ModelPricing(
        model_name="llama2-13b",
        input_cost_per_1k_tokens=Decimal("0.0002"),
        output_cost_per_1k_tokens=Decimal("0.0004"),
        context_window=4000,
        notes="Ollama local Llama2 13B (estimated cost)"
    )
}

@dataclass
class CostEstimate:
    """Cost estimation for screening project."""
    total_articles: int
    estimated_input_tokens_per_article: int
    estimated_output_tokens_per_article: int
    conservative_model_cost: Decimal
    liberal_model_cost: Decimal
    resolver_model_cost: Decimal
    total_estimated_cost: Decimal
    confidence_level: str  # "high", "medium", "low"
    assumptions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_articles': self.total_articles,
            'estimated_input_tokens_per_article': self.estimated_input_tokens_per_article,
            'estimated_output_tokens_per_article': self.estimated_output_tokens_per_article,
            'conservative_model_cost': float(self.conservative_model_cost),
            'liberal_model_cost': float(self.liberal_model_cost),
            'resolver_model_cost': float(self.resolver_model_cost),
            'total_estimated_cost': float(self.total_estimated_cost),
            'confidence_level': self.confidence_level,
            'assumptions': self.assumptions
        }

@dataclass
class ActualCostTracking:
    """Track actual costs during screening."""
    project_id: str
    start_time: datetime
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: Decimal = Decimal('0.00')
    articles_processed: int = 0
    model_usage: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}))
    errors_encountered: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_usage(self, model_name: str, input_tokens: int, output_tokens: int):
        """Add usage for a specific model."""
        self.model_usage[model_name]['input_tokens'] += input_tokens
        self.model_usage[model_name]['output_tokens'] += output_tokens
        self.model_usage[model_name]['calls'] += 1
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost if pricing available
        if model_name in MODEL_PRICING:
            model_cost = MODEL_PRICING[model_name].calculate_cost(input_tokens, output_tokens)
            self.total_cost += model_cost
        
        self.last_updated = datetime.now()
    
    def get_cost_breakdown(self) -> Dict:
        """Get detailed cost breakdown by model."""
        breakdown = {}
        for model_name, usage in self.model_usage.items():
            if model_name in MODEL_PRICING:
                model_cost = MODEL_PRICING[model_name].calculate_cost(
                    usage['input_tokens'], 
                    usage['output_tokens']
                )
                breakdown[model_name] = {
                    'input_tokens': usage['input_tokens'],
                    'output_tokens': usage['output_tokens'],
                    'calls': usage['calls'],
                    'cost': float(model_cost)
                }
        return breakdown

class TokenEstimator:
    """Estimate token usage for screening operations."""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        self._cache = {}
        self._lock = threading.Lock()
    
    def estimate_tokens(self, text: str, use_cache: bool = True) -> int:
        """Estimate token count for text."""
        if use_cache:
            text_hash = hash(text)
            with self._lock:
                if text_hash in self._cache:
                    return self._cache[text_hash]
        
        try:
            token_count = len(self.encoding.encode(text))
            
            if use_cache:
                with self._lock:
                    self._cache[text_hash] = token_count
            
            return token_count
        except Exception as e:
            logger.warning(f"Token estimation failed, using approximation: {e}")
            # Fallback: rough approximation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def estimate_screening_prompt_tokens(self, pico_criteria: Dict, article_title: str, article_abstract: str) -> int:
        """Estimate tokens for a complete screening prompt."""
        # Build approximate prompt
        prompt_template = f"""
        Based on the following PICO criteria, please classify the abstract as "Include", "Exclude", or "Uncertain".
        
        PICO Criteria:
        - Population: {pico_criteria.get('population', 'Not specified')}
        - Intervention: {pico_criteria.get('intervention', 'Not specified')}
        - Comparison: {pico_criteria.get('comparison', 'Not specified')}
        - Outcome: {pico_criteria.get('outcomes', 'Not specified')}
        - Timeframe: {pico_criteria.get('time_frame', 'Not specified')}
        - Study Type: {pico_criteria.get('study_types', 'Not specified')}
        
        Abstract to analyze:
        - Title: {article_title}
        - Abstract: {article_abstract}
        
        Your response must be in JSON format with 'decision' and 'reasoning' fields.
        """
        
        return self.estimate_tokens(prompt_template)
    
    def estimate_response_tokens(self) -> int:
        """Estimate typical response tokens for screening decision."""
        # Based on observed patterns: decision + reasoning
        typical_response = """
        {
            "decision": "Include",
            "reasoning": "This study meets the PICO criteria as it involves the target population with the specified intervention compared to the control, measuring the relevant outcomes within the timeframe. The study design is appropriate and the methodology is sound for inclusion in the systematic review."
        }
        """
        return self.estimate_tokens(typical_response)
    
    def estimate_conflict_resolution_tokens(self, article_title: str, article_abstract: str, conservative_reasoning: str, liberal_reasoning: str) -> Tuple[int, int]:
        """Estimate tokens for conflict resolution."""
        conflict_prompt = f"""
        TWO AI ASSISTANTS DISAGREE ON A STUDY'S INCLUSION. RESOLVE THE CONFLICT.

        STUDY:
        - Title: {article_title}
        - Abstract: {article_abstract}

        ASSISTANT 1 (Conservative):
        - Decision: Exclude
        - Reasoning: {conservative_reasoning}

        ASSISTANT 2 (Liberal):
        - Decision: Include
        - Reasoning: {liberal_reasoning}

        YOUR TASK:
        Analyze both arguments and provide a final, detailed "Conflict Report" in JSON format.
        """
        
        input_tokens = self.estimate_tokens(conflict_prompt)
        
        # Conflict resolution responses are typically longer
        typical_conflict_response = """
        {
            "conflict_report": "The disagreement stems from different interpretations of the population criteria. The conservative assistant focused on the strict age requirements while the liberal assistant considered the broader population definition. Given the specific inclusion criteria, the conservative interpretation appears more appropriate. However, the study design is methodologically sound and measures relevant outcomes. Recommendation: Include with note about population boundary conditions."
        }
        """
        output_tokens = self.estimate_tokens(typical_conflict_response)
        
        return input_tokens, output_tokens

class CostEstimator:
    """Estimate costs for screening projects."""
    
    def __init__(self):
        self.token_estimator = TokenEstimator()
    
    def estimate_project_cost(self, project_id: str, sample_size: int = 10) -> CostEstimate:
        """Estimate total cost for screening a project."""
        project = Project.query.get(project_id)
        if not project:
            raise ValidationError(f"Project {project_id} not found")
        
        # Get article sample for estimation
        sample_articles = db.session.query(Article).filter(
            Article.project_id == project_id
        ).limit(sample_size).all()
        
        if not sample_articles:
            raise ValidationError("No articles found for cost estimation")
        
        total_articles = db.session.query(Article).filter(
            Article.project_id == project_id
        ).count()
        
        # Estimate tokens per article
        input_tokens_samples = []
        for article in sample_articles:
            prompt_tokens = self.token_estimator.estimate_screening_prompt_tokens(
                project.config.get('pico', {}),
                article.title or '',
                article.abstract or ''
            )
            input_tokens_samples.append(prompt_tokens)
        
        avg_input_tokens = int(sum(input_tokens_samples) / len(input_tokens_samples))
        avg_output_tokens = self.token_estimator.estimate_response_tokens()
        
        # Get model names from config
        config = project.config
        conservative_model = config.get('conservative_model', 'gpt-4o-mini')
        liberal_model = config.get('liberal_model', 'gpt-3.5-turbo')
        resolver_model = config.get('resolver_model', 'gpt-4')
        
        # Calculate costs
        conservative_pricing = MODEL_PRICING.get(conservative_model)
        liberal_pricing = MODEL_PRICING.get(liberal_model)
        resolver_pricing = MODEL_PRICING.get(resolver_model)
        
        if not all([conservative_pricing, liberal_pricing, resolver_pricing]):
            raise ConfigurationError("Unknown model in pricing configuration")
        
        # Conservative model cost (all articles)
        conservative_cost = conservative_pricing.calculate_cost(
            avg_input_tokens * total_articles,
            avg_output_tokens * total_articles
        )
        
        # Liberal model cost (all articles)
        liberal_cost = liberal_pricing.calculate_cost(
            avg_input_tokens * total_articles,
            avg_output_tokens * total_articles
        )
        
        # Resolver model cost (estimated 20% conflicts)
        estimated_conflicts = int(total_articles * 0.2)
        conflict_input_tokens, conflict_output_tokens = self.token_estimator.estimate_conflict_resolution_tokens(
            "Sample title", "Sample abstract", "Sample reasoning", "Sample reasoning"
        )
        
        resolver_cost = resolver_pricing.calculate_cost(
            conflict_input_tokens * estimated_conflicts,
            conflict_output_tokens * estimated_conflicts
        )
        
        total_cost = conservative_cost + liberal_cost + resolver_cost
        
        # Determine confidence level
        confidence = "high" if sample_size >= 20 else "medium" if sample_size >= 10 else "low"
        
        assumptions = [
            f"Based on {sample_size} sample articles",
            f"Assumes {estimated_conflicts} conflicts ({estimated_conflicts/total_articles*100:.1f}%)",
            f"Token estimation accuracy ±20%",
            f"Pricing current as of {datetime.now().strftime('%Y-%m-%d')}"
        ]
        
        return CostEstimate(
            total_articles=total_articles,
            estimated_input_tokens_per_article=avg_input_tokens,
            estimated_output_tokens_per_article=avg_output_tokens,
            conservative_model_cost=conservative_cost,
            liberal_model_cost=liberal_cost,
            resolver_model_cost=resolver_cost,
            total_estimated_cost=total_cost,
            confidence_level=confidence,
            assumptions=assumptions
        )
    
    def suggest_cost_optimization(self, estimate: CostEstimate, max_budget: Decimal) -> Dict:
        """Suggest optimizations to stay within budget."""
        if estimate.total_estimated_cost <= max_budget:
            return {
                'optimization_needed': False,
                'message': 'Estimated cost is within budget',
                'current_estimate': float(estimate.total_estimated_cost),
                'budget': float(max_budget)
            }
        
        suggestions = []
        potential_savings = Decimal('0.00')
        
        # Suggest cheaper models
        conservative_savings = self._calculate_model_savings('gpt-4o-mini', estimate)
        if conservative_savings > 0:
            suggestions.append({
                'type': 'model_change',
                'description': 'Switch conservative model to gpt-4o-mini',
                'savings': float(conservative_savings)
            })
            potential_savings += conservative_savings
        
        liberal_savings = self._calculate_model_savings('gpt-4o-mini', estimate, model_type='liberal')
        if liberal_savings > 0:
            suggestions.append({
                'type': 'model_change',
                'description': 'Switch liberal model to gpt-4o-mini',
                'savings': float(liberal_savings)
            })
            potential_savings += liberal_savings
        
        # Suggest active learning to reduce articles
        if estimate.total_articles > 100:
            al_savings = estimate.total_estimated_cost * Decimal('0.3')  # Assume 30% reduction
            suggestions.append({
                'type': 'active_learning',
                'description': 'Use active learning to reduce screening load by ~30%',
                'savings': float(al_savings)
            })
            potential_savings += al_savings
        
        return {
            'optimization_needed': True,
            'budget_exceeded_by': float(estimate.total_estimated_cost - max_budget),
            'potential_savings': float(potential_savings),
            'suggestions': suggestions,
            'optimized_estimate': float(estimate.total_estimated_cost - potential_savings)
        }
    
    def _calculate_model_savings(self, cheaper_model: str, estimate: CostEstimate, model_type: str = 'conservative') -> Decimal:
        """Calculate savings from switching to cheaper model."""
        if cheaper_model not in MODEL_PRICING:
            return Decimal('0.00')
        
        cheaper_pricing = MODEL_PRICING[cheaper_model]
        new_cost = cheaper_pricing.calculate_cost(
            estimate.estimated_input_tokens_per_article * estimate.total_articles,
            estimate.estimated_output_tokens_per_article * estimate.total_articles
        )
        
        if model_type == 'conservative':
            current_cost = estimate.conservative_model_cost
        else:
            current_cost = estimate.liberal_model_cost
        
        return max(Decimal('0.00'), current_cost - new_cost)

class CostTracker:
    """Track actual costs during screening operations."""
    
    def __init__(self):
        self.active_tracking = {}  # project_id -> ActualCostTracking
        self._lock = threading.Lock()
    
    def start_tracking(self, project_id: str) -> ActualCostTracking:
        """Start cost tracking for a project."""
        with self._lock:
            tracking = ActualCostTracking(project_id=project_id, start_time=datetime.now())
            self.active_tracking[project_id] = tracking
            return tracking
    
    def record_api_call(self, project_id: str, model_name: str, input_tokens: int, output_tokens: int):
        """Record an API call for cost tracking."""
        with self._lock:
            if project_id in self.active_tracking:
                self.active_tracking[project_id].add_usage(model_name, input_tokens, output_tokens)
                logger.debug(f"Recorded API call: {model_name}, {input_tokens} in, {output_tokens} out")
    
    def record_article_completion(self, project_id: str):
        """Record completion of article screening."""
        with self._lock:
            if project_id in self.active_tracking:
                self.active_tracking[project_id].articles_processed += 1
    
    def record_error(self, project_id: str):
        """Record an error during screening."""
        with self._lock:
            if project_id in self.active_tracking:
                self.active_tracking[project_id].errors_encountered += 1
    
    def get_current_costs(self, project_id: str) -> Optional[Dict]:
        """Get current cost information for a project."""
        with self._lock:
            if project_id not in self.active_tracking:
                return None
            
            tracking = self.active_tracking[project_id]
            elapsed_time = datetime.now() - tracking.start_time
            
            return {
                'project_id': project_id,
                'elapsed_time_minutes': elapsed_time.total_seconds() / 60,
                'articles_processed': tracking.articles_processed,
                'total_cost': float(tracking.total_cost),
                'total_input_tokens': tracking.total_input_tokens,
                'total_output_tokens': tracking.total_output_tokens,
                'cost_breakdown': tracking.get_cost_breakdown(),
                'errors_encountered': tracking.errors_encountered,
                'average_cost_per_article': float(tracking.total_cost / max(1, tracking.articles_processed))
            }
    
    def finalize_tracking(self, project_id: str) -> Optional[Dict]:
        """Finalize tracking and return summary."""
        with self._lock:
            if project_id not in self.active_tracking:
                return None
            
            final_summary = self.get_current_costs(project_id)
            del self.active_tracking[project_id]
            
            # Save to database for historical tracking
            try:
                project = Project.query.get(project_id)
                if project:
                    cost_history = project.config.get('cost_history', [])
                    cost_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'final_summary': final_summary
                    })
                    project.config['cost_history'] = cost_history
                    db.session.commit()
            except Exception as e:
                logger.error(f"Failed to save cost history: {e}")
            
            return final_summary

# Global instances
cost_estimator = CostEstimator()
cost_tracker = CostTracker()
