"""
Modern LLM Screening System with Dual Provider Reasoning
Uses latest Anthropic + OpenAI models with Pydantic structured outputs.

Key Features:
- Single abstract exposure per LLM
- Two provider approach (Anthropic Claude + OpenAI GPT)
- PICOTT extraction with inclusion/exclusion criteria evaluation
- Pydantic structured outputs for reliable data extraction
- Mathematical triggers for human review
- Comprehensive metrics and scoring
"""

import logging
from typing import List, Dict, Optional, Union, Literal
from dataclasses import dataclass
from datetime import datetime
import json
from decimal import Decimal

from pydantic import BaseModel, Field, validator
import openai
import anthropic
from openai import OpenAI
from anthropic import Anthropic

from app.models.screening_models import db, Project, Article
# from app.services.utils.cost_tracker import cost_tracker  # Temporarily disabled
from app.services.utils.error_handler import retry_with_backoff
from app.services.utils.exceptions import APIError, ValidationError

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class PICOTTExtraction(BaseModel):
    """Extracted PICOTT elements from abstract."""
    population: Optional[str] = Field(None, description="Target population described in the study")
    intervention: Optional[str] = Field(None, description="Intervention or exposure being studied")
    comparison: Optional[str] = Field(None, description="Control or comparison group")
    outcomes: List[str] = Field(default_factory=list, description="Outcomes measured in the study")
    time_frame: Optional[str] = Field(None, description="Time frame or follow-up period")
    study_types: List[str] = Field(default_factory=list, description="Type of study (RCT, cohort, etc.)")
    
    @validator('outcomes', 'study_types', pre=True)
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v or []

class CriteriaEvaluation(BaseModel):
    """Evaluation against inclusion/exclusion criteria."""
    meets_inclusion_criteria: bool = Field(description="Whether study meets inclusion criteria")
    inclusion_reasoning: str = Field(description="Detailed reasoning for inclusion criteria evaluation")
    
    violates_exclusion_criteria: bool = Field(description="Whether study violates exclusion criteria")
    exclusion_reasoning: str = Field(description="Detailed reasoning for exclusion criteria evaluation")
    
    inclusion_criteria_matched: List[str] = Field(default_factory=list, description="Specific inclusion criteria that were matched")
    exclusion_criteria_violated: List[str] = Field(default_factory=list, description="Specific exclusion criteria that were violated")

class ResearchQuestionRelevance(BaseModel):
    """Evaluation of relevance to research question."""
    can_answer_research_question: bool = Field(description="Whether this study can contribute to answering the research question")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score from 0 to 1")
    relevance_reasoning: str = Field(description="Detailed explanation of relevance assessment")
    key_findings_relevant: List[str] = Field(default_factory=list, description="Key findings that are relevant to research question")

class ScreeningDecision(BaseModel):
    """Final screening decision with reasoning."""
    final_decision: Literal["INCLUDE", "EXCLUDE", "UNCERTAIN"] = Field(description="Final inclusion decision")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the decision from 0 to 1")
    
    primary_reason: str = Field(description="Primary reason for the decision")
    detailed_reasoning: str = Field(description="Comprehensive reasoning for the decision")
    
    requires_human_review: bool = Field(description="Whether this decision requires human review")
    human_review_reason: Optional[str] = Field(None, description="Reason why human review is needed")

class ComprehensiveScreeningResult(BaseModel):
    """Complete structured output from LLM screening."""
    
    # Article identification
    article_title: str = Field(description="Title of the article being screened")
    
    # PICOTT extraction
    picott_extraction: PICOTTExtraction = Field(description="Extracted PICOTT elements")
    
    # Criteria evaluation
    criteria_evaluation: CriteriaEvaluation = Field(description="Evaluation against inclusion/exclusion criteria")
    
    # Research question relevance
    research_relevance: ResearchQuestionRelevance = Field(description="Relevance to research question")
    
    # Final decision
    screening_decision: ScreeningDecision = Field(description="Final screening decision")
    
    # Metadata
    llm_provider: str = Field(description="LLM provider used (openai/anthropic)")
    model_name: str = Field(description="Specific model used")
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Quality indicators
    extraction_completeness: float = Field(ge=0.0, le=1.0, description="How complete was the PICOTT extraction")
    reasoning_quality: float = Field(ge=0.0, le=1.0, description="Quality of reasoning provided")

# ============================================================================
# SCREENING CRITERIA CONFIGURATION
# ============================================================================

@dataclass
class ScreeningCriteria:
    """Configuration for screening criteria."""
    research_question: str
    
    # PICOTT criteria
    target_population: str
    target_intervention: str
    target_comparison: str
    target_outcomes: List[str]
    target_time_frame: str
    target_study_types: List[str]
    
    # Inclusion criteria
    inclusion_criteria: List[str]
    
    # Exclusion criteria
    exclusion_criteria: List[str]
    
    # Quality thresholds
    minimum_relevance_score: float = 0.3
    minimum_confidence_score: float = 0.7
    require_human_review_threshold: float = 0.5

# ============================================================================
# LLM PROVIDER INTERFACES
# ============================================================================

class OpenAIProvider:
    """OpenAI provider using latest GPT models."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            timeout=120.0,  # 2 minute timeout
            max_retries=3
        )
        self.model_name = "gpt-4o"  # Latest reasoning model
        self.provider_name = "openai"
    
    @retry_with_backoff(max_retries=3)
    def screen_abstract(self, 
                       abstract: str, 
                       title: str, 
                       criteria: ScreeningCriteria,
                       project_id: str) -> ComprehensiveScreeningResult:
        """Screen abstract using OpenAI GPT with structured output."""
        
        prompt = self._build_screening_prompt(abstract, title, criteria)
        
        try:
            # Handle different model types with appropriate API methods
            timeout = 120.0  # Default 2 minutes
            
            if "o1" in self.model_name.lower() or "o3" in self.model_name.lower():
                # Reasoning models (o1-preview, o1-mini, o3-mini) - no system message, no structured output
                timeout = 300.0  # 5 minutes for reasoning models
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user", 
                            "content": f"You are an expert systematic review researcher. {prompt}\n\nPlease respond with valid JSON only, matching the comprehensive screening results schema."
                        }
                    ],
                    timeout=timeout
                )
                
                # Parse response manually for reasoning models
                response_content = response.choices[0].message.content
                try:
                    result_dict = json.loads(response_content)
                except json.JSONDecodeError:
                    # Extract JSON from markdown if present
                    import re
                    json_match = re.search(r'```json\s*({.*?})\s*```', response_content, re.DOTALL)
                    if json_match:
                        result_dict = json.loads(json_match.group(1))
                    else:
                        raise ValueError("Could not extract valid JSON from reasoning model response")
                
                result = ComprehensiveScreeningResult(**result_dict)
                
            elif "gpt-4o" in self.model_name.lower():
                # GPT-4o models - use structured outputs (Structured Outputs API)
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert systematic review researcher. Analyze the provided abstract against the given criteria and provide a comprehensive structured assessment."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    response_format=ComprehensiveScreeningResult,
                    temperature=0.1,  # Low temperature for consistency
                    timeout=timeout
                )
                
                result = response.choices[0].message.parsed
                
            else:
                # Other models (gpt-4-turbo, gpt-3.5-turbo) - standard completion
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert systematic review researcher. Analyze the provided abstract against the given criteria and provide a comprehensive structured assessment. Respond with valid JSON only."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    timeout=timeout
                )
                
                # Parse response manually
                response_content = response.choices[0].message.content
                result_dict = json.loads(response_content)
                result = ComprehensiveScreeningResult(**result_dict)
            
            # Log usage information for monitoring
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"OpenAI API usage - Model: {self.model_name}, "
                           f"Prompt tokens: {response.usage.prompt_tokens}, "
                           f"Completion tokens: {response.usage.completion_tokens}, "
                           f"Total tokens: {response.usage.total_tokens}")
            
            # Add provider metadata
            result.llm_provider = self.provider_name
            result.model_name = self.model_name
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI screening failed for project {project_id}: {e}")
            # Add more specific error handling
            if "timeout" in str(e).lower():
                raise APIError(f"OpenAI API timeout after {timeout}s: {e}", status_code=408) from e
            elif "rate_limit" in str(e).lower():
                raise APIError(f"OpenAI rate limit exceeded: {e}", status_code=429, retry_after=60) from e
            elif "invalid_request" in str(e).lower():
                raise APIError(f"OpenAI invalid request: {e}", status_code=400) from e
            else:
                raise APIError(f"OpenAI screening failed: {e}") from e
    
    def _build_screening_prompt(self, abstract: str, title: str, criteria: ScreeningCriteria) -> str:
        """Build comprehensive screening prompt."""
        return f"""
        Please conduct a comprehensive systematic review screening of this research article.

        RESEARCH QUESTION: {criteria.research_question}

        TARGET PICOTT CRITERIA:
        - Population: {criteria.target_population}
        - Intervention: {criteria.target_intervention}
        - Comparison: {criteria.target_comparison}
        - Outcomes: {', '.join(criteria.target_outcomes)}
        - Time Frame: {criteria.target_time_frame}
        - Study Types: {', '.join(criteria.target_study_types)}

        INCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in criteria.inclusion_criteria)}

        EXCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in criteria.exclusion_criteria)}

        ARTICLE TO ANALYZE:
        Title: {title}
        Abstract: {abstract}

        INSTRUCTIONS:
        1. Extract PICOTT elements present in the abstract
        2. Evaluate against inclusion criteria (provide specific reasoning)
        3. Evaluate against exclusion criteria (provide specific reasoning)
        4. Assess relevance to research question with confidence score
        5. Make final decision (INCLUDE/EXCLUDE/UNCERTAIN) with detailed reasoning
        6. Determine if human review is needed based on uncertainty or complexity

        Provide your analysis in the structured format requested.
        """

class AnthropicProvider:
    """Anthropic provider using latest Claude models."""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(
            api_key=api_key,
            timeout=120.0,  # 2 minute timeout
            max_retries=3
        )
        self.model_name = "claude-3-5-sonnet-20241022"  # Latest Claude model
        self.provider_name = "anthropic"
    
    @retry_with_backoff(max_retries=3)
    def screen_abstract(self, 
                       abstract: str, 
                       title: str, 
                       criteria: ScreeningCriteria,
                       project_id: str) -> ComprehensiveScreeningResult:
        """Screen abstract using Anthropic Claude with structured output."""
        
        prompt = self._build_screening_prompt(abstract, title, criteria)
        
        try:
            # Handle different Claude model versions with appropriate methods
            timeout = 120.0  # 2 minutes default
            
            if "claude-3-5" in self.model_name.lower():
                # Claude 3.5 models - use tool calling for structured output
                timeout = 90.0
                
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.1,
                    timeout=timeout,
                    system="You are an expert systematic review researcher with extensive experience in evidence-based medicine and research methodology.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    tools=[
                        {
                            "name": "screening_analysis",
                            "description": "Provide structured screening analysis for systematic review",
                            "input_schema": ComprehensiveScreeningResult.model_json_schema()
                        }
                    ],
                    tool_choice={"type": "tool", "name": "screening_analysis"}
                )
                
                # Parse tool use response
                if response.content and len(response.content) > 0:
                    content_block = response.content[0]
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        result_dict = content_block.input
                        result = ComprehensiveScreeningResult(**result_dict)
                    else:
                        raise ValueError("Expected tool use response from Claude 3.5")
                else:
                    raise ValueError("No content in Claude 3.5 response")
                    
            elif "claude-3" in self.model_name.lower():
                # Claude 3 models - standard text completion with JSON
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.1,
                    timeout=timeout,
                    system="You are an expert systematic review researcher. Analyze the provided abstract against the given criteria and provide a comprehensive structured assessment. Respond ONLY with valid JSON matching the required schema.",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                # Parse text response
                response_text = response.content[0].text
                result_dict = json.loads(response_text)
                result = ComprehensiveScreeningResult(**result_dict)
                
            else:
                # Fallback for other Claude models
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.1,
                    timeout=timeout,
                    system="You are an expert systematic review researcher. Respond with valid JSON only.",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nProvide your analysis as valid JSON matching the comprehensive screening results schema."
                        }
                    ]
                )
                
                # Parse response
                response_text = response.content[0].text
                try:
                    result_dict = json.loads(response_text)
                except json.JSONDecodeError:
                    # Extract JSON from markdown if present
                    import re
                    json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
                    if json_match:
                        result_dict = json.loads(json_match.group(1))
                    else:
                        raise ValueError("Could not extract valid JSON from Claude response")
                
                result = ComprehensiveScreeningResult(**result_dict)
            
            # Log usage information for monitoring
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Anthropic API usage - Model: {self.model_name}, "
                           f"Input tokens: {response.usage.input_tokens}, "
                           f"Output tokens: {response.usage.output_tokens}")
            
            # Result parsing handled in model-specific blocks above
            
            # Add provider metadata
            result.llm_provider = self.provider_name
            result.model_name = self.model_name
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic screening failed for project {project_id}: {e}")
            # Add more specific error handling
            if "timeout" in str(e).lower():
                raise APIError(f"Anthropic API timeout after {timeout}s: {e}", status_code=408) from e
            elif "rate_limit" in str(e).lower() or "429" in str(e):
                raise APIError(f"Anthropic rate limit exceeded: {e}", status_code=429, retry_after=60) from e
            elif "invalid_request" in str(e).lower() or "400" in str(e):
                raise APIError(f"Anthropic invalid request: {e}", status_code=400) from e
            else:
                raise APIError(f"Anthropic screening failed: {e}") from e
    
    def _build_screening_prompt(self, abstract: str, title: str, criteria: ScreeningCriteria) -> str:
        """Build comprehensive screening prompt with JSON schema."""
        
        # Get JSON schema for structured output
        schema = ComprehensiveScreeningResult.model_json_schema()
        
        return f"""
        Please conduct a comprehensive systematic review screening of this research article.

        RESEARCH QUESTION: {criteria.research_question}

        TARGET PICOTT CRITERIA:
        - Population: {criteria.target_population}
        - Intervention: {criteria.target_intervention}
        - Comparison: {criteria.target_comparison}
        - Outcomes: {', '.join(criteria.target_outcomes)}
        - Time Frame: {criteria.target_time_frame}
        - Study Types: {', '.join(criteria.target_study_types)}

        INCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in criteria.inclusion_criteria)}

        EXCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in criteria.exclusion_criteria)}

        ARTICLE TO ANALYZE:
        Title: {title}
        Abstract: {abstract}

        INSTRUCTIONS:
        1. Extract PICOTT elements present in the abstract
        2. Evaluate against inclusion criteria (provide specific reasoning)
        3. Evaluate against exclusion criteria (provide specific reasoning)
        4. Assess relevance to research question with confidence score
        5. Make final decision (INCLUDE/EXCLUDE/UNCERTAIN) with detailed reasoning
        6. Determine if human review is needed based on uncertainty or complexity

        Respond with valid JSON matching this exact schema:
        {json.dumps(schema, indent=2)}

        Your response must be valid JSON only, no additional text.
        """

# ============================================================================
# DUAL PROVIDER SCREENING ORCHESTRATOR
# ============================================================================

class DualProviderScreeningOrchestrator:
    """Orchestrates screening using both OpenAI and Anthropic providers."""
    
    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.openai_provider = OpenAIProvider(openai_api_key)
        self.anthropic_provider = AnthropicProvider(anthropic_api_key)
        
    def screen_article_dual_provider(self, 
                                   article: Article, 
                                   criteria: ScreeningCriteria,
                                   project_id: str) -> Dict[str, ComprehensiveScreeningResult]:
        """Screen article using both providers and return both results."""
        
        results = {}
        
        # Screen with OpenAI
        try:
            results['openai'] = self.openai_provider.screen_abstract(
                article.abstract or "",
                article.title or "",
                criteria,
                project_id
            )
            logger.info(f"OpenAI screening completed for article {article.id}")
        except Exception as e:
            logger.error(f"OpenAI screening failed for article {article.id}: {e}")
            results['openai'] = None
        
        # Screen with Anthropic
        try:
            results['anthropic'] = self.anthropic_provider.screen_abstract(
                article.abstract or "",
                article.title or "",
                criteria,
                project_id
            )
            logger.info(f"Anthropic screening completed for article {article.id}")
        except Exception as e:
            logger.error(f"Anthropic screening failed for article {article.id}: {e}")
            results['anthropic'] = None
        
        return results
    
    def analyze_provider_agreement(self, 
                                 openai_result: ComprehensiveScreeningResult,
                                 anthropic_result: ComprehensiveScreeningResult) -> Dict:
        """Analyze agreement between two provider results."""
        
        if not openai_result or not anthropic_result:
            return {
                'agreement': False,
                'reason': 'One or both providers failed',
                'requires_human_review': True
            }
        
        # Decision agreement
        decision_agreement = (
            openai_result.screening_decision.final_decision == 
            anthropic_result.screening_decision.final_decision
        )
        
        # Confidence agreement (within 0.2 threshold)
        confidence_diff = abs(
            openai_result.screening_decision.confidence_score - 
            anthropic_result.screening_decision.confidence_score
        )
        confidence_agreement = confidence_diff <= 0.2
        
        # Relevance agreement (within 0.2 threshold)
        relevance_diff = abs(
            openai_result.research_relevance.relevance_score - 
            anthropic_result.research_relevance.relevance_score
        )
        relevance_agreement = relevance_diff <= 0.2
        
        overall_agreement = decision_agreement and confidence_agreement
        
        return {
            'agreement': overall_agreement,
            'decision_agreement': decision_agreement,
            'confidence_agreement': confidence_agreement,
            'relevance_agreement': relevance_agreement,
            'confidence_difference': confidence_diff,
            'relevance_difference': relevance_diff,
            'requires_human_review': (
                not overall_agreement or 
                openai_result.screening_decision.requires_human_review or
                anthropic_result.screening_decision.requires_human_review
            )
        }

# ============================================================================
# MATHEMATICAL TRIGGERS FOR HUMAN REVIEW
# ============================================================================

class HumanReviewTriggers:
    """Mathematical formulas to determine when human review is needed."""
    
    @staticmethod
    def calculate_uncertainty_score(openai_result: ComprehensiveScreeningResult,
                                  anthropic_result: ComprehensiveScreeningResult) -> float:
        """Calculate uncertainty score based on provider disagreement."""
        
        if not openai_result or not anthropic_result:
            return 1.0  # Maximum uncertainty if one failed
        
        # Decision disagreement weight
        decision_weight = 0.4
        if openai_result.screening_decision.final_decision != anthropic_result.screening_decision.final_decision:
            decision_uncertainty = 1.0
        else:
            decision_uncertainty = 0.0
        
        # Confidence disagreement weight
        confidence_weight = 0.3
        confidence_diff = abs(
            openai_result.screening_decision.confidence_score - 
            anthropic_result.screening_decision.confidence_score
        )
        confidence_uncertainty = min(confidence_diff * 2, 1.0)  # Scale to 0-1
        
        # Individual low confidence weight
        low_confidence_weight = 0.3
        min_confidence = min(
            openai_result.screening_decision.confidence_score,
            anthropic_result.screening_decision.confidence_score
        )
        low_confidence_uncertainty = 1.0 - min_confidence
        
        total_uncertainty = (
            decision_uncertainty * decision_weight +
            confidence_uncertainty * confidence_weight +
            low_confidence_uncertainty * low_confidence_weight
        )
        
        return min(total_uncertainty, 1.0)
    
    @staticmethod
    def should_trigger_human_review(openai_result: ComprehensiveScreeningResult,
                                  anthropic_result: ComprehensiveScreeningResult,
                                  uncertainty_threshold: float = 0.6) -> Dict:
        """Determine if human review should be triggered."""
        
        triggers = []
        
        # High uncertainty trigger
        uncertainty_score = HumanReviewTriggers.calculate_uncertainty_score(
            openai_result, anthropic_result
        )
        
        if uncertainty_score >= uncertainty_threshold:
            triggers.append(f"High uncertainty score: {uncertainty_score:.2f}")
        
        # Provider disagreement trigger
        if openai_result and anthropic_result:
            if openai_result.screening_decision.final_decision != anthropic_result.screening_decision.final_decision:
                triggers.append("Provider decision disagreement")
        
        # Low confidence trigger
        if openai_result and openai_result.screening_decision.confidence_score < 0.5:
            triggers.append(f"Low OpenAI confidence: {openai_result.screening_decision.confidence_score:.2f}")
        
        if anthropic_result and anthropic_result.screening_decision.confidence_score < 0.5:
            triggers.append(f"Low Anthropic confidence: {anthropic_result.screening_decision.confidence_score:.2f}")
        
        # Explicit human review requests
        if openai_result and openai_result.screening_decision.requires_human_review:
            triggers.append(f"OpenAI requested review: {openai_result.screening_decision.human_review_reason}")
        
        if anthropic_result and anthropic_result.screening_decision.requires_human_review:
            triggers.append(f"Anthropic requested review: {anthropic_result.screening_decision.human_review_reason}")
        
        return {
            'should_review': len(triggers) > 0,
            'uncertainty_score': uncertainty_score,
            'triggers': triggers,
            'trigger_count': len(triggers)
        }

# ============================================================================
# SCREENING RESULTS STORAGE
# ============================================================================

class ScreeningResultsStore:
    """Store and manage screening results in database."""
    
    @staticmethod
    def store_screening_results(article_id: str,
                              project_id: str,
                              openai_result: ComprehensiveScreeningResult,
                              anthropic_result: ComprehensiveScreeningResult,
                              agreement_analysis: Dict,
                              human_review_triggers: Dict) -> str:
        """Store comprehensive screening results."""
        
        # Create comprehensive record
        screening_record = {
            'article_id': article_id,
            'project_id': project_id,
            'timestamp': datetime.now().isoformat(),
            
            # Provider results
            'openai_result': openai_result.model_dump() if openai_result else None,
            'anthropic_result': anthropic_result.model_dump() if anthropic_result else None,
            
            # Analysis
            'agreement_analysis': agreement_analysis,
            'human_review_triggers': human_review_triggers,
            
            # Final consensus
            'final_decision': ScreeningResultsStore._determine_final_decision(
                openai_result, anthropic_result, agreement_analysis
            ),
            'requires_human_review': human_review_triggers['should_review']
        }
        
        # Update article in database
        article = Article.query.get(article_id)
        if article:
            article.decision_reasoning = screening_record
            
            # Set article status based on consensus
            if screening_record['requires_human_review']:
                article.status = 'human_review_required'
            else:
                final_decision = screening_record['final_decision']
                if final_decision == 'INCLUDE':
                    article.status = 'included'
                elif final_decision == 'EXCLUDE':
                    article.status = 'excluded'
                else:
                    article.status = 'uncertain'
            
            db.session.commit()
        
        return screening_record
    
    @staticmethod
    def _determine_final_decision(openai_result: ComprehensiveScreeningResult,
                                anthropic_result: ComprehensiveScreeningResult,
                                agreement_analysis: Dict) -> str:
        """Determine final consensus decision."""
        
        if not openai_result or not anthropic_result:
            return 'UNCERTAIN'
        
        openai_decision = openai_result.screening_decision.final_decision
        anthropic_decision = anthropic_result.screening_decision.final_decision
        
        if agreement_analysis['decision_agreement']:
            return openai_decision  # Both agree
        
        # Disagreement resolution logic
        if 'INCLUDE' in [openai_decision, anthropic_decision] and 'EXCLUDE' in [openai_decision, anthropic_decision]:
            # Include vs Exclude disagreement - requires human review
            return 'UNCERTAIN'
        
        # If one is uncertain, defer to the more confident one
        if openai_decision == 'UNCERTAIN':
            return anthropic_decision
        if anthropic_decision == 'UNCERTAIN':
            return openai_decision
        
        return 'UNCERTAIN'  # Default for unclear cases