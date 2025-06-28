"""
Dual LLM Screening Service
Implements the core dual-provider screening logic using OpenAI and Anthropic.
"""

import logging
from typing import List, Dict, Optional, Union, Literal, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os

from pydantic import BaseModel, Field, validator
import openai
import anthropic
from openai import OpenAI
from anthropic import Anthropic

from app.models.screening_models import db, Article
# from app.services.utils.cost_tracker import cost_tracker  # Temporarily disabled
from app.services.utils.error_handler import retry_with_backoff
from app.services.utils.exceptions import APIError, ValidationError
from app.services.screening.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# ============================================================================
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for LLM providers."""
    provider: str  # 'openai' or 'anthropic'
    model_name: str
    temperature: float = 0.1
    seed: Optional[int] = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.seed is not None and (self.seed < 0 or self.seed > 2**32):
            raise ValueError("Seed must be between 0 and 2^32")

@dataclass
class MultiModelConfig:
    """Configuration for multiple LLM providers."""
    providers: List[ModelConfig]
    consensus_strategy: str = "weighted_voting"  # weighted_voting, majority, unanimous
    uncertainty_threshold: float = 0.6

    def __post_init__(self):
        if len(self.providers) < 2:
            raise ValueError("At least 2 providers required for multi-model screening")

        provider_names = [p.provider for p in self.providers]
        if len(set(provider_names)) != len(provider_names):
            raise ValueError("Duplicate providers not allowed")

@dataclass
class DualModelConfig:
    """Configuration for both LLM providers."""
    openai_config: ModelConfig
    anthropic_config: ModelConfig

    def to_multi_config(self) -> MultiModelConfig:
        """Convert to MultiModelConfig for unified processing."""
        return MultiModelConfig(providers=[self.openai_config, self.anthropic_config])

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

class ScreeningCriteria(BaseModel):
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

    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for LLM generation (0.0-2.0)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible results")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model to use")

# ============================================================================
# LLM PROVIDER INTERFACES
# ============================================================================

class OpenAIProvider:
    """OpenAI provider with configurable parameters and latest API best practices."""

    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(provider='openai', model_name='gpt-4o-2024-11-20')
        
        # Get model-specific configuration from registry
        model_config = ModelRegistry.get_model_config(self.config.model_name)
        timeout = model_config.recommended_timeout if model_config else 120.0
        
        # Configure client with timeout settings
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=3
        )
        self.model_name = self.config.model_name
        self.provider_name = "openai"

    @retry_with_backoff(max_retries=3)
    def screen_abstract(self,
                       abstract: str,
                       title: str,
                       criteria: ScreeningCriteria,
                       project_id: str) -> ComprehensiveScreeningResult:
        """Screen abstract using OpenAI GPT with structured output and latest API practices."""

        prompt = self._build_screening_prompt(abstract, title, criteria)

        try:
            # Use the latest structured outputs approach
            request_params = {
                "model": criteria.openai_model,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert systematic review researcher with extensive experience in evidence-based medicine and research methodology.

Your task is to conduct a comprehensive screening of research abstracts for systematic reviews. You must:
1. Carefully extract PICOTT elements from the abstract
2. Systematically evaluate against inclusion/exclusion criteria
3. Assess relevance to the research question with quantified confidence
4. Make evidence-based decisions with detailed reasoning
5. Identify cases requiring human expert review

Be thorough, precise, and conservative in your assessments. When in doubt, err on the side of caution and recommend human review."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.config.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "screening_result",
                        "schema": ComprehensiveScreeningResult.model_json_schema(),
                        "strict": True
                    }
                }
            }

            # Add optional parameters
            if self.config.max_tokens is not None:
                request_params["max_tokens"] = self.config.max_tokens

            if self.config.seed is not None:
                request_params["seed"] = self.config.seed

            # Handle different model types with specific configurations
            if "o1" in criteria.openai_model.lower() or "o3" in criteria.openai_model.lower():
                # For reasoning models (o1-preview, o1-mini, o3-mini, etc.)
                request_params.pop("temperature", None)  # Remove temperature
                request_params.pop("response_format", None)  # Remove structured output for o1/o3
                timeout = 300.0  # 5 minutes for reasoning models
                
                # Use standard completion for reasoning models
                response = self.client.chat.completions.create(
                    **request_params,
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
                
            elif "gpt-4o" in criteria.openai_model.lower():
                # For GPT-4o models, use structured outputs
                timeout = 120.0  # 2 minutes for standard models
                response = self.client.chat.completions.create(
                    **request_params,
                    timeout=timeout
                )
                
                # Parse the structured response
                response_content = response.choices[0].message.content
                try:
                    result_dict = json.loads(response_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OpenAI response as JSON: {e}\nResponse content: {response_content}")
                    raise ValueError("Invalid JSON in OpenAI response") from e
                result = ComprehensiveScreeningResult(**result_dict)
                
            else:
                # For other models (gpt-4-turbo, gpt-3.5-turbo, etc.)
                timeout = 120.0  # 2 minutes for standard models
                request_params.pop("response_format", None)  # Remove structured output
                
                response = self.client.chat.completions.create(
                    **request_params,
                    timeout=timeout
                )
                
                # Parse response manually
                response_content = response.choices[0].message.content
                try:
                    result_dict = json.loads(response_content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'{.*}', response_content, re.DOTALL)
                    if json_match:
                        result_dict = json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not extract valid JSON from model response")
                
                result = ComprehensiveScreeningResult(**result_dict)

            # Log usage information for monitoring
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"OpenAI API usage - Model: {criteria.openai_model}, "
                           f"Prompt tokens: {response.usage.prompt_tokens}, "
                           f"Completion tokens: {response.usage.completion_tokens}, "
                           f"Total tokens: {response.usage.total_tokens}")

            # Parse the structured response
            response_content = response.choices[0].message.content
            try:
                result_dict = json.loads(response_content)
            except json.JSONDecodeError as e:
                # Handle invalid JSON gracefully (log, raise, or return a default/failure result)
                logger.error(f"Failed to parse OpenAI response as JSON: {e}\nResponse content: {response_content}")
                raise ValueError("Invalid JSON in OpenAI response") from e
            result = ComprehensiveScreeningResult(**result_dict)

            # Add provider metadata
            result.llm_provider = self.provider_name
            result.model_name = criteria.openai_model

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
    """Anthropic provider with configurable parameters and latest API best practices."""

    def __init__(self, api_key: str, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(provider='anthropic', model_name='claude-3-5-sonnet-20241022')
        
        # Get model-specific configuration from registry
        model_config = ModelRegistry.get_model_config(self.config.model_name)
        timeout = model_config.recommended_timeout if model_config else 120.0
        
        # Configure client with timeout settings
        self.client = Anthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=3
        )
        self.model_name = self.config.model_name
        self.provider_name = "anthropic"

    @retry_with_backoff(max_retries=3)
    def screen_abstract(self,
                       abstract: str,
                       title: str,
                       criteria: ScreeningCriteria,
                       project_id: str) -> ComprehensiveScreeningResult:
        """Screen abstract using Anthropic Claude with structured output and latest API practices."""

        prompt = self._build_screening_prompt(abstract, title, criteria)

        try:
            # Enhanced system prompt for better reasoning
            system_prompt = """You are an expert systematic review researcher with extensive experience in evidence-based medicine, research methodology, and clinical research evaluation.

Your expertise includes:
- PICOTT framework analysis (Population, Intervention, Comparison, Outcomes, Time frame, Study Types)
- Systematic review methodology and PRISMA guidelines
- Critical appraisal of research studies
- Evidence synthesis and meta-analysis principles

Your task is to conduct a comprehensive screening of research abstracts for systematic reviews. You must:
1. Carefully extract PICOTT elements from the abstract with high precision
2. Systematically evaluate against inclusion/exclusion criteria with detailed reasoning
3. Assess relevance to the research question with quantified confidence scores
4. Make evidence-based decisions with comprehensive reasoning
5. Identify cases requiring human expert review based on uncertainty or complexity

Be thorough, precise, and conservative in your assessments. When in doubt, err on the side of caution and recommend human review. Provide detailed reasoning for all decisions to ensure transparency and reproducibility."""

            request_params = {
                "model": criteria.anthropic_model,
                "max_tokens": self.config.max_tokens or 8000,  # Increased for comprehensive responses
                "temperature": self.config.temperature,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            # Add tool use for structured output if supported by model
            if "claude-3-5" in criteria.anthropic_model.lower() or "claude-3" in criteria.anthropic_model.lower():
                # Use the latest structured output approach for Claude 3.5
                request_params["tools"] = [
                    {
                        "name": "screening_analysis",
                        "description": "Provide structured screening analysis for systematic review",
                        "input_schema": ComprehensiveScreeningResult.model_json_schema()
                    }
                ]
                request_params["tool_choice"] = {"type": "tool", "name": "screening_analysis"}

            # Set timeout based on model type
            timeout = 120.0  # 2 minutes default
            if "claude-3-5" in criteria.anthropic_model:
                timeout = 90.0  # Slightly shorter for Claude 3.5
            
            response = self.client.messages.create(
                **request_params,
                timeout=timeout
            )

            # Log usage information for monitoring
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"Anthropic API usage - Model: {criteria.anthropic_model}, "
                           f"Input tokens: {response.usage.input_tokens}, "
                           f"Output tokens: {response.usage.output_tokens}")

            # Parse response based on format
            if hasattr(response, 'content') and response.content:
                if len(response.content) > 0 and hasattr(response.content[0], 'type'):
                    # Handle tool use response
                    if response.content[0].type == 'tool_use':
                        result_dict = response.content[0].input
                        result = ComprehensiveScreeningResult(**result_dict)
                    else:
                        # Handle text response
                        response_text = response.content[0].text
                        # Extract JSON from response if wrapped in markdown or other text
                        if "```json" in response_text:
                            json_start = response_text.find("```json") + 7
                            json_end = response_text.find("```", json_start)
                            if json_start != -1 and json_end != -1:
                                response_text = response_text[json_start:json_end].strip()
                        elif "```" in response_text:
                            json_start = response_text.find("```") + 3
                            json_end = response_text.find("```", json_start)
                            if json_start != -1 and json_end != -1:
                                response_text = response_text[json_start:json_end].strip()

                        result_dict = json.loads(response_text)
                        result = ComprehensiveScreeningResult(**result_dict)
                else:
                    # Fallback for simple text response
                    response_text = response.content[0].text
                    result_dict = json.loads(response_text)
                    result = ComprehensiveScreeningResult(**result_dict)
            else:
                raise APIError("No content in Anthropic response")

            # Add provider metadata
            result.llm_provider = self.provider_name
            result.model_name = criteria.anthropic_model

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Anthropic JSON parsing failed for project {project_id}: {e}")
            raise APIError(f"Anthropic response parsing failed: {e}", status_code=422) from e
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

class GoogleProvider:
    """Google Gemini provider for screening."""

    def __init__(self, config: ModelConfig):
        self.config = config

    async def screen_abstract(self, criteria: ScreeningCriteria, title: str, abstract: str) -> ComprehensiveScreeningResult:
        """Screen abstract using Google Gemini."""
        prompt = self._build_screening_prompt(criteria, title, abstract)
        raise NotImplementedError("Google Gemini provider not yet implemented")

    def _build_screening_prompt(self, criteria: ScreeningCriteria, title: str, abstract: str) -> str:
        """Build screening prompt for Google Gemini."""
        return f"""
        You are an expert systematic review researcher. Analyze the following abstract against the specified PICO-TT criteria and provide a comprehensive screening decision.

        RESEARCH QUESTION: {criteria.research_question}

        PICO-TT CRITERIA:
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

        ABSTRACT TO ANALYZE:
        Title: {title}
        Abstract: {abstract}

        Provide your analysis in the following JSON format with the same structure as other providers.
        """

class OllamaProvider:
    """Ollama local provider for screening."""

    def __init__(self, config: ModelConfig):
        self.config = config

    async def screen_abstract(self, criteria: ScreeningCriteria, title: str, abstract: str) -> ComprehensiveScreeningResult:
        """Screen abstract using Ollama local models."""
        raise NotImplementedError("Ollama provider not yet implemented")

    def _build_screening_prompt(self, criteria: ScreeningCriteria, title: str, abstract: str) -> str:
        """Build screening prompt for Ollama."""
        return f"""
        You are an expert systematic review researcher. Analyze the following abstract against the specified PICO-TT criteria and provide a comprehensive screening decision.

        RESEARCH QUESTION: {criteria.research_question}

        PICO-TT CRITERIA:
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

        ABSTRACT TO ANALYZE:
        Title: {title}
        Abstract: {abstract}

        Provide your analysis in the following JSON format with the same structure as other providers.
        """

class MultiProviderScreeningOrchestrator:
    """Orchestrate screening across multiple LLM providers."""

    def __init__(self, config: MultiModelConfig):
        self.config = config
        self.providers = self._initialize_providers()

    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize provider instances based on configuration."""
        providers = {}
        for provider_config in self.config.providers:
            if provider_config.provider == 'openai':
                providers['openai'] = OpenAIProvider("", provider_config)
            elif provider_config.provider == 'anthropic':
                providers['anthropic'] = AnthropicProvider("", provider_config)
            elif provider_config.provider == 'google':
                providers['google'] = GoogleProvider(provider_config)
            elif provider_config.provider == 'ollama':
                providers['ollama'] = OllamaProvider(provider_config)
            else:
                raise ValueError(f"Unsupported provider: {provider_config.provider}")
        return providers

    async def screen_article_multi_provider(self, criteria: ScreeningCriteria, title: str, abstract: str) -> Dict:
        """Screen article using multiple providers."""
        results = {}

        # Screen with all providers
        for provider_name, provider in self.providers.items():
            try:
                result = await provider.screen_abstract(criteria, title, abstract)
                results[provider_name] = result
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = None

        # Analyze consensus
        consensus_analysis = self.analyze_multi_provider_consensus(results)

        review_triggers = MultiProviderReviewTriggers.should_trigger_human_review(
            results, self.config.uncertainty_threshold
        )

        return {
            'provider_results': results,
            'consensus_analysis': consensus_analysis,
            'human_review_triggers': review_triggers,
            'final_decision': self._determine_consensus_decision(results, consensus_analysis)
        }

    def analyze_multi_provider_consensus(self, results: Dict) -> Dict:
        """Analyze consensus across multiple providers."""
        valid_results = {k: v for k, v in results.items() if v is not None}

        if len(valid_results) < 2:
            return {
                'consensus_reached': False,
                'reason': 'Insufficient valid provider results',
                'agreement_score': 0.0
            }

        # Decision consensus
        decisions = [r.screening_decision.final_decision for r in valid_results.values()]
        decision_counts = {d: decisions.count(d) for d in set(decisions)}
        majority_decision = max(decision_counts.keys(), key=lambda k: decision_counts[k])
        decision_consensus = decision_counts[majority_decision] / len(decisions)

        # Confidence consensus
        confidences = [r.screening_decision.confidence_score for r in valid_results.values()]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)

        agreement_score = decision_consensus * (1 - confidence_variance)

        return {
            'consensus_reached': agreement_score >= 0.7,
            'agreement_score': agreement_score,
            'majority_decision': majority_decision,
            'decision_distribution': decision_counts,
            'average_confidence': avg_confidence,
            'confidence_variance': confidence_variance,
            'provider_count': len(valid_results)
        }

    def _determine_consensus_decision(self, results: Dict, consensus_analysis: Dict) -> str:
        """Determine final consensus decision using configured strategy."""
        valid_results = {k: v for k, v in results.items() if v is not None}

        if not valid_results:
            return 'UNCERTAIN'

        if self.config.consensus_strategy == "majority":
            return consensus_analysis.get('majority_decision', 'UNCERTAIN')

        elif self.config.consensus_strategy == "weighted_voting":
            weighted_votes = {}
            for provider_name, result in valid_results.items():
                decision = result.screening_decision.final_decision
                confidence = result.screening_decision.confidence_score
                weighted_votes[decision] = weighted_votes.get(decision, 0) + confidence

            return max(weighted_votes.keys(), key=lambda k: weighted_votes[k]) if weighted_votes else 'UNCERTAIN'

        elif self.config.consensus_strategy == "unanimous":
            decisions = [r.screening_decision.final_decision for r in valid_results.values()]
            return decisions[0] if len(set(decisions)) == 1 else 'UNCERTAIN'

        return 'UNCERTAIN'

# ============================================================================
# DUAL PROVIDER SCREENING ORCHESTRATOR
# ============================================================================

class DualProviderScreeningOrchestrator:
    """Orchestrates screening using both OpenAI and Anthropic providers."""

    def __init__(self, openai_api_key: str, anthropic_api_key: str, config: Optional[DualModelConfig] = None):
        if config:
            self.openai_provider = OpenAIProvider(openai_api_key, config.openai_config)
            self.anthropic_provider = AnthropicProvider(anthropic_api_key, config.anthropic_config)
        else:
            # Use latest available models as defaults
            default_openai_config = ModelConfig(provider="openai", model_name="gpt-4o-2024-11-20")
            default_anthropic_config = ModelConfig(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
            self.openai_provider = OpenAIProvider(openai_api_key, default_openai_config)
            self.anthropic_provider = AnthropicProvider(anthropic_api_key, default_anthropic_config)

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

class MultiProviderReviewTriggers:
    """Enhanced review triggers for multiple providers."""

    @staticmethod
    def should_trigger_human_review(results: Dict, uncertainty_threshold: float = 0.6) -> Dict:
        """Determine if human review should be triggered for multi-provider results."""
        valid_results = {k: v for k, v in results.items() if v is not None}
        triggers = []

        if len(valid_results) < 2:
            triggers.append("Insufficient provider responses")
            return {
                'should_review': True,
                'triggers': triggers,
                'uncertainty_score': 1.0
            }

        uncertainty_score = MultiProviderReviewTriggers._calculate_multi_uncertainty(valid_results)

        if uncertainty_score >= uncertainty_threshold:
            triggers.append(f"High multi-provider uncertainty: {uncertainty_score:.2f}")

        decisions = [r.screening_decision.final_decision for r in valid_results.values()]
        unique_decisions = set(decisions)
        if len(unique_decisions) > 1:
            triggers.append(f"Provider decision disagreement: {unique_decisions}")

        low_confidence_providers = []
        for provider_name, result in valid_results.items():
            if result.screening_decision.confidence_score < 0.5:
                low_confidence_providers.append(f"{provider_name}: {result.screening_decision.confidence_score:.2f}")

        if low_confidence_providers:
            triggers.append(f"Low confidence providers: {', '.join(low_confidence_providers)}")

        # Check for explicit review requests
        review_requests = []
        for provider_name, result in valid_results.items():
            if result.screening_decision.requires_human_review:
                review_requests.append(f"{provider_name}: {result.screening_decision.human_review_reason}")

        if review_requests:
            triggers.extend(review_requests)

        return {
            'should_review': len(triggers) > 0,
            'uncertainty_score': uncertainty_score,
            'triggers': triggers,
            'trigger_count': len(triggers),
            'provider_count': len(valid_results)
        }

    @staticmethod
    def _calculate_multi_uncertainty(results: Dict) -> float:
        """Calculate uncertainty score for multiple providers."""
        if len(results) < 2:
            return 1.0

        # Decision variance
        decisions = [r.screening_decision.final_decision for r in results.values()]
        decision_entropy = len(set(decisions)) / len(decisions)

        # Confidence variance
        confidences = [r.screening_decision.confidence_score for r in results.values()]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)

        uncertainty = (decision_entropy * 0.6) + (confidence_variance * 0.4)
        return min(uncertainty, 1.0)

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
                              human_review_triggers: Dict) -> Dict:
        """Store comprehensive screening results with proper transaction handling."""

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

        # Update article in database with transaction handling
        try:
            article = Article.query.get(article_id)
            if not article:
                logger.error(f"Article {article_id} not found in database")
                raise ValueError(f"Article {article_id} not found")
                
            # Store previous status for rollback if needed
            previous_status = article.status
            previous_reasoning = article.decision_reasoning
            
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

            # Commit the transaction
            db.session.commit()
            logger.info(f"Successfully stored screening results for article {article_id}, status: {article.status}")
            
        except Exception as e:
            # Rollback on any error
            db.session.rollback()
            logger.error(f"Failed to store screening results for article {article_id}: {e}")
            
            # Try to restore previous state
            try:
                article = Article.query.get(article_id)
                if article:
                    article.status = previous_status
                    article.decision_reasoning = previous_reasoning
                    db.session.commit()
                    logger.info(f"Restored previous state for article {article_id}")
            except Exception as restore_error:
                logger.error(f"Failed to restore previous state for article {article_id}: {restore_error}")
                db.session.rollback()
            
            raise

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
