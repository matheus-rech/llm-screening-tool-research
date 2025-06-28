"""
Enhanced Systematic Review Screening Service
Implements advanced context-aware dual-LLM screening with systematic review best practices.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from collections import defaultdict

from pydantic import BaseModel, Field
from app.models.screening_models import db, Article, Project
from .dual_llm_screener import (
    DualProviderScreeningOrchestrator,
    ScreeningCriteria,
    ComprehensiveScreeningResult,
    ScreeningResultsStore
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED SYSTEMATIC REVIEW MODELS
# ============================================================================

class SystematicReviewCriteria(BaseModel):
    """Enhanced criteria for systematic review screening with domain knowledge."""

    # Core PICO-TT criteria
    research_question: str
    target_population: str
    target_intervention: str
    target_comparison: str
    target_outcomes: List[str]
    target_time_frame: str
    target_study_types: List[str]

    # Enhanced systematic review criteria
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]

    # Domain-specific knowledge
    domain: str = Field(default="medical", description="Research domain (medical, social, engineering, etc.)")
    specialty_keywords: List[str] = Field(default_factory=list, description="Domain-specific keywords")
    methodology_requirements: List[str] = Field(default_factory=list, description="Required methodological features")

    # Quality assessment criteria
    minimum_sample_size: Optional[int] = None
    required_study_duration: Optional[str] = None
    language_restrictions: List[str] = Field(default_factory=list)
    publication_date_range: Optional[Tuple[int, int]] = None

    # Screening parameters
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    agreement_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    uncertainty_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # LLM configuration
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    seed: Optional[int] = None
    openai_model: str = Field(default="gpt-4o")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")

class CitationContext(BaseModel):
    """Context information for citation relationships and domain knowledge."""

    # Citation network analysis
    citing_articles: List[str] = Field(default_factory=list, description="Articles that cite this one")
    cited_articles: List[str] = Field(default_factory=list, description="Articles cited by this one")
    co_citation_strength: float = Field(default=0.0, description="Co-citation network strength")

    # Author network
    author_collaboration_score: float = Field(default=0.0, description="Author collaboration network score")
    author_expertise_domains: List[str] = Field(default_factory=list, description="Author expertise areas")

    # Journal and venue context
    journal_impact_factor: Optional[float] = None
    journal_domain_relevance: float = Field(default=0.0, description="Journal relevance to research domain")
    venue_type: str = Field(default="journal", description="Publication venue type")

    # Content similarity
    similar_articles: List[Dict] = Field(default_factory=list, description="Similar articles with scores")
    topic_clusters: List[str] = Field(default_factory=list, description="Topic cluster assignments")

    # Historical screening context
    similar_decisions: List[Dict] = Field(default_factory=list, description="Previous decisions on similar articles")
    decision_patterns: Dict[str, float] = Field(default_factory=dict, description="Historical decision patterns")

class EnhancedScreeningResult(BaseModel):
    """Enhanced screening result with systematic review context."""

    # Core screening result
    base_result: ComprehensiveScreeningResult

    # Context analysis
    citation_context: CitationContext

    # Enhanced decision metrics
    systematic_review_quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality for systematic review")
    methodology_adequacy_score: float = Field(ge=0.0, le=1.0, description="Methodological adequacy score")
    domain_relevance_score: float = Field(ge=0.0, le=1.0, description="Domain-specific relevance score")

    # Risk of bias assessment
    risk_of_bias_indicators: List[str] = Field(default_factory=list, description="Potential bias indicators")
    study_quality_concerns: List[str] = Field(default_factory=list, description="Quality concerns identified")

    # Systematic review specific flags
    duplicate_likelihood: float = Field(default=0.0, ge=0.0, le=1.0, description="Likelihood of being a duplicate")
    grey_literature_flag: bool = Field(default=False, description="Identified as grey literature")
    language_barrier_flag: bool = Field(default=False, description="Language accessibility concerns")

    # Decision support
    human_review_priority: str = Field(default="normal", description="Priority level for human review")
    reviewer_expertise_needed: List[str] = Field(default_factory=list, description="Required reviewer expertise")

    # Metadata
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    context_analysis_version: str = Field(default="1.0")

# ============================================================================
# CONTEXT-AWARE SCREENING ORCHESTRATOR
# ============================================================================

class ContextAwareScreeningOrchestrator:
    """Enhanced orchestrator with context awareness and systematic review intelligence."""

    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.dual_screener = DualProviderScreeningOrchestrator(openai_api_key, anthropic_api_key)
        self.context_analyzer = CitationContextAnalyzer()
        self.decision_intelligence = DecisionIntelligenceEngine()
        self.quality_assessor = StudyQualityAssessor()

    def screen_article_with_context(self,
                                   article: Article,
                                   criteria: SystematicReviewCriteria,
                                   project_id: str,
                                   project_context: Optional[Dict] = None) -> EnhancedScreeningResult:
        """Screen article with full context awareness and systematic review intelligence."""

        # Step 1: Analyze citation context
        citation_context = self.context_analyzer.analyze_citation_context(
            article, criteria, project_context or {}
        )

        # Step 2: Perform dual-LLM screening with enhanced prompts
        base_screening_criteria = self._convert_to_base_criteria(criteria)
        dual_results = self.dual_screener.screen_article_dual_provider(
            article, base_screening_criteria, project_id
        )

        # Step 3: Apply decision intelligence
        enhanced_decision = self.decision_intelligence.enhance_decision(
            dual_results, citation_context, criteria
        )

        # Step 4: Assess study quality
        quality_assessment = self.quality_assessor.assess_study_quality(
            article, criteria, citation_context
        )

        # Step 5: Combine results into enhanced screening result
        enhanced_result = EnhancedScreeningResult(
            base_result=enhanced_decision,
            citation_context=citation_context,
            systematic_review_quality_score=quality_assessment['overall_quality'],
            methodology_adequacy_score=quality_assessment['methodology_score'],
            domain_relevance_score=quality_assessment['domain_relevance'],
            risk_of_bias_indicators=quality_assessment['bias_indicators'],
            study_quality_concerns=quality_assessment['quality_concerns'],
            duplicate_likelihood=citation_context.similar_articles[0]['similarity'] if citation_context.similar_articles else 0.0,
            human_review_priority=self._determine_review_priority(enhanced_decision, quality_assessment),
            reviewer_expertise_needed=self._determine_required_expertise(criteria, quality_assessment)
        )

        return enhanced_result

    def _convert_to_base_criteria(self, enhanced_criteria: SystematicReviewCriteria) -> ScreeningCriteria:
        """Convert enhanced criteria to base screening criteria."""
        return ScreeningCriteria(
            research_question=enhanced_criteria.research_question,
            target_population=enhanced_criteria.target_population,
            target_intervention=enhanced_criteria.target_intervention,
            target_comparison=enhanced_criteria.target_comparison,
            target_outcomes=enhanced_criteria.target_outcomes,
            target_time_frame=enhanced_criteria.target_time_frame,
            target_study_types=enhanced_criteria.target_study_types,
            inclusion_criteria=enhanced_criteria.inclusion_criteria,
            exclusion_criteria=enhanced_criteria.exclusion_criteria,
            minimum_relevance_score=enhanced_criteria.confidence_threshold,
            minimum_confidence_score=enhanced_criteria.agreement_threshold,
            require_human_review_threshold=enhanced_criteria.uncertainty_threshold,
            temperature=enhanced_criteria.temperature,
            seed=enhanced_criteria.seed,
            openai_model=enhanced_criteria.openai_model,
            anthropic_model=enhanced_criteria.anthropic_model
        )

    def _determine_review_priority(self, decision: ComprehensiveScreeningResult, quality: Dict) -> str:
        """Determine human review priority based on decision and quality metrics."""
        if decision.screening_decision.requires_human_review:
            if quality['overall_quality'] < 0.3:
                return "low"
            elif quality['overall_quality'] > 0.8 and decision.screening_decision.confidence_score < 0.6:
                return "high"
            else:
                return "normal"
        return "none"

    def _determine_required_expertise(self, criteria: SystematicReviewCriteria, quality: Dict) -> List[str]:
        """Determine required reviewer expertise based on criteria and quality assessment."""
        expertise_needed = []

        if criteria.domain:
            expertise_needed.append(f"{criteria.domain}_specialist")

        if quality.get('methodology_concerns'):
            expertise_needed.append("methodology_expert")

        if quality.get('statistical_concerns'):
            expertise_needed.append("statistician")

        return expertise_needed

# ============================================================================
# CITATION CONTEXT ANALYZER
# ============================================================================

class CitationContextAnalyzer:
    """Analyzes citation context and relationships for enhanced screening."""

    def analyze_citation_context(self,
                                article: Article,
                                criteria: SystematicReviewCriteria,
                                project_context: Dict) -> CitationContext:
        """Analyze comprehensive citation context for an article."""

        context = CitationContext()

        # Analyze citation network
        context.citing_articles = self._find_citing_articles(article)
        context.cited_articles = self._extract_cited_articles(article)
        context.co_citation_strength = self._calculate_co_citation_strength(article, project_context)

        # Analyze author network
        context.author_collaboration_score = self._calculate_author_collaboration(article, project_context)
        context.author_expertise_domains = self._identify_author_expertise(article)

        # Analyze journal context
        context.journal_impact_factor = self._get_journal_impact_factor(article.journal)
        context.journal_domain_relevance = self._calculate_journal_domain_relevance(article.journal, criteria.domain)
        context.venue_type = self._classify_venue_type(article.journal)

        # Find similar articles
        context.similar_articles = self._find_similar_articles(article, project_context)
        context.topic_clusters = self._assign_topic_clusters(article, criteria)

        # Analyze historical decisions
        context.similar_decisions = self._find_similar_historical_decisions(article, project_context)
        context.decision_patterns = self._analyze_decision_patterns(article, project_context)

        return context

    def _find_citing_articles(self, article: Article) -> List[str]:
        """Find articles that cite this article."""
        # Simplified implementation - in practice, would use citation databases
        return []

    def _extract_cited_articles(self, article: Article) -> List[str]:
        """Extract articles cited by this article."""
        # Would parse references from full text or use citation databases
        return []

    def _calculate_co_citation_strength(self, article: Article, project_context: Dict) -> float:
        """Calculate co-citation network strength."""
        # Simplified implementation
        return 0.0

    def _calculate_author_collaboration(self, article: Article, project_context: Dict) -> float:
        """Calculate author collaboration network score."""
        if not article.authors:
            return 0.0

        # Count author overlaps with other articles in project
        project_authors = set()
        for other_article in project_context.get('articles', []):
            if other_article.authors:
                project_authors.update(other_article.authors.split(', '))

        article_authors = set(article.authors.split(', '))
        overlap = len(article_authors.intersection(project_authors))

        return min(overlap / len(article_authors), 1.0) if article_authors else 0.0

    def _identify_author_expertise(self, article: Article) -> List[str]:
        """Identify author expertise domains."""
        # Simplified implementation - would use author databases
        return []

    def _get_journal_impact_factor(self, journal: Optional[str]) -> Optional[float]:
        """Get journal impact factor."""
        # Would integrate with journal databases
        return None

    def _calculate_journal_domain_relevance(self, journal: Optional[str], domain: str) -> float:
        """Calculate journal relevance to research domain."""
        if not journal:
            return 0.0

        # Simplified domain matching
        domain_keywords = {
            'medical': ['medicine', 'health', 'clinical', 'medical', 'therapy', 'treatment'],
            'social': ['social', 'psychology', 'sociology', 'behavior', 'society'],
            'engineering': ['engineering', 'technology', 'technical', 'systems', 'design']
        }

        journal_lower = journal.lower()
        keywords = domain_keywords.get(domain, [])

        matches = sum(1 for keyword in keywords if keyword in journal_lower)
        return min(matches / len(keywords), 1.0) if keywords else 0.0

    def _classify_venue_type(self, journal: Optional[str]) -> str:
        """Classify publication venue type."""
        if not journal:
            return "unknown"

        journal_lower = journal.lower()

        if any(word in journal_lower for word in ['conference', 'proceedings', 'symposium']):
            return "conference"
        elif any(word in journal_lower for word in ['preprint', 'arxiv', 'biorxiv']):
            return "preprint"
        elif any(word in journal_lower for word in ['thesis', 'dissertation']):
            return "thesis"
        else:
            return "journal"

    def _find_similar_articles(self, article: Article, project_context: Dict) -> List[Dict]:
        """Find similar articles with similarity scores."""
        similar = []

        if not article.abstract:
            return similar

        # Simple text similarity with other articles in project
        for other_article in project_context.get('articles', []):
            if other_article.id == article.id or not other_article.abstract:
                continue

            similarity = self._calculate_text_similarity(article.abstract, other_article.abstract)
            if similarity > 0.3:  # Threshold for similarity
                similar.append({
                    'article_id': other_article.id,
                    'title': other_article.title,
                    'similarity': similarity
                })

        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:5]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two texts."""
        # Simplified implementation using word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _assign_topic_clusters(self, article: Article, criteria: SystematicReviewCriteria) -> List[str]:
        """Assign article to topic clusters."""
        clusters = []

        if not article.abstract:
            return clusters

        abstract_lower = article.abstract.lower()

        # Simple keyword-based clustering
        if any(keyword.lower() in abstract_lower for keyword in criteria.specialty_keywords):
            clusters.append("specialty_relevant")

        if any(outcome.lower() in abstract_lower for outcome in criteria.target_outcomes):
            clusters.append("outcome_relevant")

        if criteria.target_intervention.lower() in abstract_lower:
            clusters.append("intervention_relevant")

        return clusters

    def _find_similar_historical_decisions(self, article: Article, project_context: Dict) -> List[Dict]:
        """Find similar historical screening decisions."""
        # Would query database for similar articles and their decisions
        return []

    def _analyze_decision_patterns(self, article: Article, project_context: Dict) -> Dict[str, float]:
        """Analyze historical decision patterns for similar articles."""
        # Would analyze patterns in historical decisions
        return {}

# ============================================================================
# DECISION INTELLIGENCE ENGINE
# ============================================================================

class DecisionIntelligenceEngine:
    """Advanced decision intelligence for systematic review screening."""

    def enhance_decision(self,
                        dual_results: Dict[str, ComprehensiveScreeningResult],
                        citation_context: CitationContext,
                        criteria: SystematicReviewCriteria) -> ComprehensiveScreeningResult:
        """Enhance screening decision with context and intelligence."""

        openai_result = dual_results.get('openai')
        anthropic_result = dual_results.get('anthropic')

        if not openai_result and not anthropic_result:
            raise ValueError("No valid screening results from either provider")

        if not openai_result:
            return anthropic_result  # type: ignore

        if not anthropic_result:
            return openai_result

        # Calculate weighted consensus
        enhanced_decision = self._calculate_weighted_consensus(
            openai_result, anthropic_result, citation_context, criteria
        )

        # Apply context-based adjustments
        enhanced_decision = self._apply_context_adjustments(
            enhanced_decision, citation_context, criteria
        )

        # Update confidence based on context
        enhanced_decision = self._update_confidence_with_context(
            enhanced_decision, citation_context
        )

        return enhanced_decision

    def _calculate_weighted_consensus(self,
                                    openai_result: ComprehensiveScreeningResult,
                                    anthropic_result: ComprehensiveScreeningResult,
                                    citation_context: CitationContext,
                                    criteria: SystematicReviewCriteria) -> ComprehensiveScreeningResult:
        """Calculate weighted consensus between providers."""

        # Base weights
        openai_weight = 0.5
        anthropic_weight = 0.5

        # Adjust weights based on context
        if citation_context.journal_domain_relevance > 0.8:
            # High domain relevance - trust more confident model
            if openai_result.screening_decision.confidence_score > anthropic_result.screening_decision.confidence_score:
                openai_weight = 0.6
                anthropic_weight = 0.4
            else:
                openai_weight = 0.4
                anthropic_weight = 0.6

        # Calculate weighted scores
        weighted_confidence = (
            openai_result.screening_decision.confidence_score * openai_weight +
            anthropic_result.screening_decision.confidence_score * anthropic_weight
        )

        weighted_relevance = (
            openai_result.research_relevance.relevance_score * openai_weight +
            anthropic_result.research_relevance.relevance_score * anthropic_weight
        )

        # Determine consensus decision
        if openai_result.screening_decision.final_decision == anthropic_result.screening_decision.final_decision:
            consensus_decision = openai_result.screening_decision.final_decision
        else:
            # Use weighted confidence to break ties
            if openai_result.screening_decision.confidence_score * openai_weight > anthropic_result.screening_decision.confidence_score * anthropic_weight:
                consensus_decision = openai_result.screening_decision.final_decision
            else:
                consensus_decision = anthropic_result.screening_decision.final_decision

        # Create enhanced result (using openai_result as base)
        enhanced_result = openai_result.model_copy()
        enhanced_result.screening_decision.final_decision = consensus_decision
        enhanced_result.screening_decision.confidence_score = weighted_confidence
        enhanced_result.research_relevance.relevance_score = weighted_relevance
        enhanced_result.screening_decision.detailed_reasoning = f"Weighted consensus: OpenAI ({openai_weight:.1f}) + Anthropic ({anthropic_weight:.1f}). {enhanced_result.screening_decision.detailed_reasoning}"

        return enhanced_result

    def _apply_context_adjustments(self,
                                 decision: ComprehensiveScreeningResult,
                                 citation_context: CitationContext,
                                 criteria: SystematicReviewCriteria) -> ComprehensiveScreeningResult:
        """Apply context-based adjustments to decision."""

        # Adjust based on similar articles
        if citation_context.similar_articles:
            high_similarity_count = sum(1 for article in citation_context.similar_articles if article['similarity'] > 0.8)
            if high_similarity_count > 0:
                decision.screening_decision.requires_human_review = True
                decision.screening_decision.human_review_reason = f"High similarity to {high_similarity_count} other articles - potential duplicate"

        # Adjust based on journal relevance
        if citation_context.journal_domain_relevance < 0.3:
            decision.research_relevance.relevance_score *= 0.8  # Reduce relevance for low-relevance journals
            decision.screening_decision.detailed_reasoning += " Note: Low journal domain relevance."

        # Adjust based on venue type
        if citation_context.venue_type == "preprint":
            decision.screening_decision.requires_human_review = True
            decision.screening_decision.human_review_reason = "Preprint - requires quality assessment"

        return decision

    def _update_confidence_with_context(self,
                                      decision: ComprehensiveScreeningResult,
                                      citation_context: CitationContext) -> ComprehensiveScreeningResult:
        """Update confidence score based on context."""

        context_confidence_boost = 0.0

        # Boost confidence for high journal relevance
        if citation_context.journal_domain_relevance > 0.8:
            context_confidence_boost += 0.1

        # Boost confidence for strong author collaboration
        if citation_context.author_collaboration_score > 0.5:
            context_confidence_boost += 0.05

        # Reduce confidence for potential duplicates
        if citation_context.similar_articles and citation_context.similar_articles[0]['similarity'] > 0.9:
            context_confidence_boost -= 0.2

        # Apply boost (with bounds)
        new_confidence = decision.screening_decision.confidence_score + context_confidence_boost
        decision.screening_decision.confidence_score = max(0.0, min(1.0, new_confidence))

        return decision

# ============================================================================
# STUDY QUALITY ASSESSOR
# ============================================================================

class StudyQualityAssessor:
    """Assesses study quality for systematic review inclusion."""

    def assess_study_quality(self,
                           article: Article,
                           criteria: SystematicReviewCriteria,
                           citation_context: CitationContext) -> Dict[str, Any]:
        """Comprehensive study quality assessment."""

        assessment = {
            'overall_quality': 0.0,
            'methodology_score': 0.0,
            'domain_relevance': 0.0,
            'bias_indicators': [],
            'quality_concerns': [],
            'methodology_concerns': [],
            'statistical_concerns': []
        }

        # Assess methodology adequacy
        methodology_score = self._assess_methodology(article, criteria)
        assessment['methodology_score'] = methodology_score

        # Assess domain relevance
        domain_relevance = self._assess_domain_relevance(article, criteria, citation_context)
        assessment['domain_relevance'] = domain_relevance

        # Identify bias indicators
        bias_indicators = self._identify_bias_indicators(article, criteria)
        assessment['bias_indicators'] = bias_indicators

        # Identify quality concerns
        quality_concerns = self._identify_quality_concerns(article, criteria, citation_context)
        assessment['quality_concerns'] = quality_concerns

        # Calculate overall quality score
        assessment['overall_quality'] = self._calculate_overall_quality(
            methodology_score, domain_relevance, len(bias_indicators), len(quality_concerns)
        )

        return assessment

    def _assess_methodology(self, article: Article, criteria: SystematicReviewCriteria) -> float:
        """Assess methodological adequacy."""
        if not article.abstract:
            return 0.0

        abstract_lower = article.abstract.lower()
        methodology_score = 0.0

        # Check for study design mentions
        study_designs = ['randomized', 'controlled', 'cohort', 'case-control', 'cross-sectional', 'systematic review', 'meta-analysis']
        design_mentions = sum(1 for design in study_designs if design in abstract_lower)
        methodology_score += min(design_mentions * 0.2, 0.4)

        # Check for sample size mentions
        if any(word in abstract_lower for word in ['participants', 'subjects', 'patients', 'n=']):
            methodology_score += 0.2

        # Check for statistical analysis mentions
        if any(word in abstract_lower for word in ['statistical', 'analysis', 'significant', 'p-value', 'confidence interval']):
            methodology_score += 0.2

        # Check for outcome measures
        if any(outcome.lower() in abstract_lower for outcome in criteria.target_outcomes):
            methodology_score += 0.2

        return min(methodology_score, 1.0)

    def _assess_domain_relevance(self,
                               article: Article,
                               criteria: SystematicReviewCriteria,
                               citation_context: CitationContext) -> float:
        """Assess domain-specific relevance."""
        relevance_score = 0.0

        # Journal domain relevance
        relevance_score += citation_context.journal_domain_relevance * 0.4

        # Keyword relevance
        if article.abstract and criteria.specialty_keywords:
            abstract_lower = article.abstract.lower()
            keyword_matches = sum(1 for keyword in criteria.specialty_keywords if keyword.lower() in abstract_lower)
            keyword_relevance = min(keyword_matches / len(criteria.specialty_keywords), 1.0)
            relevance_score += keyword_relevance * 0.3

        # Topic cluster relevance
        if citation_context.topic_clusters:
            cluster_relevance = len(citation_context.topic_clusters) * 0.1
            relevance_score += min(cluster_relevance, 0.3)

        return min(relevance_score, 1.0)

    def _identify_bias_indicators(self, article: Article, criteria: SystematicReviewCriteria) -> List[str]:
        """Identify potential bias indicators."""
        bias_indicators = []

        if not article.abstract:
            return bias_indicators

        abstract_lower = article.abstract.lower()

        # Selection bias indicators
        if 'convenience sample' in abstract_lower or 'volunteer' in abstract_lower:
            bias_indicators.append("Selection bias: convenience sampling")

        # Publication bias indicators
        if 'pilot study' in abstract_lower or 'preliminary' in abstract_lower:
            bias_indicators.append("Publication bias: preliminary study")

        # Funding bias indicators
        if any(word in abstract_lower for word in ['sponsored', 'funded by', 'grant from']):
            bias_indicators.append("Potential funding bias")

        # Small sample bias
        if 'small sample' in abstract_lower or any(f'n={i}' in abstract_lower for i in range(1, 30)):
            bias_indicators.append("Small sample size bias")

        return bias_indicators

    def _identify_quality_concerns(self,
                                 article: Article,
                                 criteria: SystematicReviewCriteria,
                                 citation_context: CitationContext) -> List[str]:
        """Identify study quality concerns."""
        concerns = []

        # Missing abstract
        if not article.abstract or len(article.abstract.strip()) < 50:
            concerns.append("Insufficient abstract information")

        # Missing key information
        if not article.authors:
            concerns.append("Missing author information")

        if not article.year:
            concerns.append("Missing publication year")

        # Venue concerns
        if citation_context.venue_type == "preprint":
            concerns.append("Preprint - not peer reviewed")

        # Language concerns
        if criteria.language_restrictions and article.abstract:
            # Simple language detection (would use proper language detection in practice)
            if not self._is_acceptable_language(article.abstract, criteria.language_restrictions):
                concerns.append("Language barrier")

        # Date range concerns
        if criteria.publication_date_range and article.year:
            min_year, max_year = criteria.publication_date_range
            if article.year < min_year or article.year > max_year:
                concerns.append(f"Outside publication date range ({min_year}-{max_year})")

        return concerns

    def _is_acceptable_language(self, text: str, acceptable_languages: List[str]) -> bool:
        """Check if text is in acceptable language."""
        # Simplified implementation - would use proper language detection
        if 'english' in [lang.lower() for lang in acceptable_languages]:
            return True  # Assume English for now
        return False

    def _calculate_overall_quality(self,
                                 methodology_score: float,
                                 domain_relevance: float,
                                 bias_count: int,
                                 concern_count: int) -> float:
        """Calculate overall quality score."""

        # Base score from methodology and relevance
        base_score = (methodology_score * 0.6) + (domain_relevance * 0.4)

        # Penalties for bias and concerns
        bias_penalty = min(bias_count * 0.1, 0.3)
        concern_penalty = min(concern_count * 0.05, 0.2)

        # Calculate final score with penalties
        final_score = base_score - bias_penalty - concern_penalty

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
