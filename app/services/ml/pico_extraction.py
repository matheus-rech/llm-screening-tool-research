"""
PICO Extraction System for Automatic Criteria Suggestion
Based on PICOtron and research patterns but enhanced with LLM capabilities.

Integration Points:
- Uses our config_manager.py for PICO validation and templates
- Integrates with our error_handler.py for robust NLP operations  
- Works with our cost_tracker.py for LLM-based extraction costs
- Enhances our existing PICO configuration system
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import json

from app import db, Article, Project
from config_manager import PICOCriteria
from cost_tracker import cost_tracker
from error_handler import retry_with_backoff, safe_json_parse
from exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class ExtractedPICOElement:
    """Extracted PICO element with confidence and context."""
    element_type: str  # 'population', 'intervention', 'comparison', 'outcome'
    text: str
    confidence: float  # 0.0 to 1.0
    source_sentence: str
    start_pos: int
    end_pos: int
    supporting_evidence: List[str] = None

@dataclass
class PICOExtractionResult:
    """Result of PICO extraction from text."""
    population_elements: List[ExtractedPICOElement]
    intervention_elements: List[ExtractedPICOElement]
    comparison_elements: List[ExtractedPICOElement]
    outcome_elements: List[ExtractedPICOElement]
    overall_confidence: float
    extraction_method: str
    processing_time_seconds: float

class RegexPICOExtractor:
    """Rule-based PICO extraction using regex patterns."""
    
    def __init__(self):
        self.population_patterns = [
            r'(?i)participants?\s+(?:were|included|comprised|consisted\s+of)?\s*([^.]{1,100})',
            r'(?i)patients?\s+with\s+([^.]{1,50})',
            r'(?i)adults?\s+(?:aged|with|having)\s+([^.]{1,50})',
            r'(?i)children\s+(?:aged|with|having)\s+([^.]{1,50})',
            r'(?i)subjects?\s+(?:aged|with|having)\s+([^.]{1,50})',
            r'(?i)population[:\s]+([^.]{1,100})',
            r'(?i)(?:men|women|males?|females?)\s+(?:aged|with)\s+([^.]{1,50})'
        ]
        
        self.intervention_patterns = [
            r'(?i)(?:received|given|administered|treated\s+with|underwent)\s+([^.]{1,100})',
            r'(?i)intervention[:\s]+([^.]{1,100})',
            r'(?i)treatment[:\s]+([^.]{1,100})',
            r'(?i)therapy[:\s]+([^.]{1,100})',
            r'(?i)drug[:\s]+([^.]{1,50})',
            r'(?i)medication[:\s]+([^.]{1,50})',
            r'(?i)procedure[:\s]+([^.]{1,100})',
            r'(?i)surgery[:\s]+([^.]{1,100})'
        ]
        
        self.comparison_patterns = [
            r'(?i)compared\s+(?:to|with|against)\s+([^.]{1,100})',
            r'(?i)control[:\s]+([^.]{1,100})',
            r'(?i)placebo[:\s]*([^.]{1,50})',
            r'(?i)versus\s+([^.]{1,100})',
            r'(?i)vs\.?\s+([^.]{1,50})',
            r'(?i)standard\s+care[:\s]*([^.]{1,50})',
            r'(?i)no\s+treatment[:\s]*([^.]{1,50})'
        ]
        
        self.outcome_patterns = [
            r'(?i)outcome[:\s]+([^.]{1,100})',
            r'(?i)endpoint[:\s]+([^.]{1,100})',
            r'(?i)measured\s+([^.]{1,100})',
            r'(?i)assessed\s+([^.]{1,100})',
            r'(?i)primary\s+outcome[:\s]+([^.]{1,100})',
            r'(?i)secondary\s+outcome[:\s]+([^.]{1,100})',
            r'(?i)improvement\s+in\s+([^.]{1,100})',
            r'(?i)reduction\s+in\s+([^.]{1,100})',
            r'(?i)mortality[:\s]*([^.]{1,50})',
            r'(?i)survival[:\s]*([^.]{1,50})'
        ]
    
    def extract_from_text(self, text: str, title: str = "") -> PICOExtractionResult:
        """Extract PICO elements using regex patterns."""
        import time
        start_time = time.time()
        
        # Combine title and abstract for extraction
        full_text = f"{title} {text}".strip()
        sentences = self._split_into_sentences(full_text)
        
        population_elements = self._extract_elements(sentences, self.population_patterns, 'population')
        intervention_elements = self._extract_elements(sentences, self.intervention_patterns, 'intervention')
        comparison_elements = self._extract_elements(sentences, self.comparison_patterns, 'comparison')
        outcome_elements = self._extract_elements(sentences, self.outcome_patterns, 'outcome')
        
        # Calculate overall confidence based on number of elements found
        total_elements = len(population_elements) + len(intervention_elements) + len(comparison_elements) + len(outcome_elements)
        overall_confidence = min(1.0, total_elements / 8.0)  # Assume 8 elements for high confidence
        
        processing_time = time.time() - start_time
        
        return PICOExtractionResult(
            population_elements=population_elements,
            intervention_elements=intervention_elements,
            comparison_elements=comparison_elements,
            outcome_elements=outcome_elements,
            overall_confidence=overall_confidence,
            extraction_method="regex",
            processing_time_seconds=processing_time
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_elements(self, sentences: List[str], patterns: List[str], element_type: str) -> List[ExtractedPICOElement]:
        """Extract elements of specific type from sentences."""
        elements = []
        
        for sentence in sentences:
            for pattern in patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    if match.groups():
                        extracted_text = match.group(1).strip()
                        if len(extracted_text) > 5:  # Minimum length filter
                            elements.append(ExtractedPICOElement(
                                element_type=element_type,
                                text=extracted_text,
                                confidence=0.6,  # Medium confidence for regex
                                source_sentence=sentence,
                                start_pos=match.start(1),
                                end_pos=match.end(1)
                            ))
        
        return elements

class LLMPICOExtractor:
    """LLM-based PICO extraction using our existing LLM infrastructure."""
    
    def __init__(self, openai_client=None):
        self.client = openai_client
    
    @retry_with_backoff(max_retries=3)
    def extract_from_text(self, text: str, title: str = "", project_id: str = None) -> PICOExtractionResult:
        """Extract PICO elements using LLM."""
        import time
        from openai import OpenAI
        
        start_time = time.time()
        
        if not self.client:
            raise ConfigurationError("OpenAI client not configured for PICO extraction")
        
        # Track costs if project provided
        if project_id:
            cost_tracker.start_tracking(f"{project_id}_pico_extraction")
        
        prompt = f"""
        You are an expert in systematic review methodology. Extract PICO elements from the following research abstract.
        
        Title: {title}
        Abstract: {text}
        
        For each PICO element, identify specific text spans and rate your confidence (0.0-1.0):
        
        PICO Elements to extract:
        - Population: Who are the study participants?
        - Intervention: What treatment/intervention was applied?
        - Comparison: What was the control or comparison group?
        - Outcomes: What outcomes were measured?
        
        Return your response in JSON format:
        {{
            "population": [
                {{"text": "extracted text", "confidence": 0.9, "source_sentence": "full sentence"}}
            ],
            "intervention": [...],
            "comparison": [...],
            "outcomes": [...],
            "overall_confidence": 0.8,
            "extraction_reasoning": "Brief explanation of extraction quality"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for extraction
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a PICO extraction expert. Respond only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for consistency
            )
            
            # Track token usage
            if project_id and hasattr(response, 'usage'):
                cost_tracker.record_api_call(
                    f"{project_id}_pico_extraction",
                    "gpt-4o-mini",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            result_data = safe_json_parse(response.choices[0].message.content)
            
            # Convert to our format
            population_elements = [
                ExtractedPICOElement(
                    element_type='population',
                    text=elem['text'],
                    confidence=elem.get('confidence', 0.7),
                    source_sentence=elem.get('source_sentence', ''),
                    start_pos=0,
                    end_pos=len(elem['text'])
                ) for elem in result_data.get('population', [])
            ]
            
            intervention_elements = [
                ExtractedPICOElement(
                    element_type='intervention',
                    text=elem['text'],
                    confidence=elem.get('confidence', 0.7),
                    source_sentence=elem.get('source_sentence', ''),
                    start_pos=0,
                    end_pos=len(elem['text'])
                ) for elem in result_data.get('intervention', [])
            ]
            
            comparison_elements = [
                ExtractedPICOElement(
                    element_type='comparison',
                    text=elem['text'],
                    confidence=elem.get('confidence', 0.7),
                    source_sentence=elem.get('source_sentence', ''),
                    start_pos=0,
                    end_pos=len(elem['text'])
                ) for elem in result_data.get('comparison', [])
            ]
            
            outcome_elements = [
                ExtractedPICOElement(
                    element_type='outcomes',
                    text=elem['text'],
                    confidence=elem.get('confidence', 0.7),
                    source_sentence=elem.get('source_sentence', ''),
                    start_pos=0,
                    end_pos=len(elem['text'])
                ) for elem in result_data.get('outcomes', [])
            ]
            
            processing_time = time.time() - start_time
            
            return PICOExtractionResult(
                population_elements=population_elements,
                intervention_elements=intervention_elements,
                comparison_elements=comparison_elements,
                outcome_elements=outcome_elements,
                overall_confidence=result_data.get('overall_confidence', 0.7),
                extraction_method="llm",
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM PICO extraction failed: {e}")
            raise ValidationError(f"PICO extraction failed: {e}")

class PICOAggregator:
    """Aggregate PICO elements from multiple articles to suggest criteria."""
    
    def __init__(self):
        self.element_frequency = defaultdict(Counter)
        self.confidence_weights = defaultdict(list)
    
    def aggregate_extractions(self, extractions: List[PICOExtractionResult]) -> Dict[str, List[Tuple[str, float]]]:
        """Aggregate PICO elements from multiple extractions."""
        all_elements = {
            'population': [],
            'intervention': [],
            'comparison': [],
            'outcomes': []
        }
        
        for extraction in extractions:
            # Collect all elements
            all_elements['population'].extend(extraction.population_elements)
            all_elements['intervention'].extend(extraction.intervention_elements)
            all_elements['comparison'].extend(extraction.comparison_elements)
            all_elements['outcomes'].extend(extraction.outcome_elements)
        
        # Aggregate and rank by frequency and confidence
        aggregated = {}
        
        for element_type, elements in all_elements.items():
            # Normalize and count similar texts
            normalized_elements = defaultdict(list)
            
            for elem in elements:
                # Simple normalization - could be enhanced with similarity matching
                normalized_text = self._normalize_text(elem.text)
                normalized_elements[normalized_text].append(elem)
            
            # Calculate aggregate scores
            scored_elements = []
            for normalized_text, elem_list in normalized_elements.items():
                frequency = len(elem_list)
                avg_confidence = sum(e.confidence for e in elem_list) / len(elem_list)
                aggregate_score = frequency * avg_confidence
                
                # Use the highest confidence version as representative text
                best_elem = max(elem_list, key=lambda x: x.confidence)
                
                scored_elements.append((best_elem.text, aggregate_score))
            
            # Sort by aggregate score and take top results
            scored_elements.sort(key=lambda x: x[1], reverse=True)
            aggregated[element_type] = scored_elements[:5]  # Top 5 for each element
        
        return aggregated
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Simple normalization - could be enhanced with stemming, lemmatization
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def suggest_pico_criteria(self, aggregated_elements: Dict[str, List[Tuple[str, float]]], min_confidence: float = 0.3) -> PICOCriteria:
        """Suggest PICO criteria from aggregated elements."""
        
        def get_top_suggestion(elements: List[Tuple[str, float]]) -> str:
            """Get top suggestion above confidence threshold."""
            if not elements:
                return ""
            
            top_elements = [elem for elem, score in elements if score >= min_confidence]
            if top_elements:
                return top_elements[0]
            return elements[0][0] if elements else ""
        
        # Build suggested criteria
        population = get_top_suggestion(aggregated_elements.get('population', []))
        intervention = get_top_suggestion(aggregated_elements.get('intervention', []))
        comparison = get_top_suggestion(aggregated_elements.get('comparison', []))
        outcomes = get_top_suggestion(aggregated_elements.get('outcomes', []))
        
        return PICOCriteria(
            population=population,
            intervention=intervention,
            comparison=comparison,
            outcomes=outcomes,
            time_frame="",  # Not typically extracted from abstracts
            study_types=""  # Would need separate analysis
        )

class PICOExtractionManager:
    """High-level manager for PICO extraction operations."""
    
    def __init__(self):
        self.regex_extractor = RegexPICOExtractor()
        self.llm_extractor = None
        self.aggregator = PICOAggregator()
    
    def initialize_llm_extractor(self, openai_client):
        """Initialize LLM extractor with client."""
        self.llm_extractor = LLMPICOExtractor(openai_client)
    
    def extract_from_article(self, article: Article, method: str = "hybrid", project_id: str = None) -> PICOExtractionResult:
        """Extract PICO elements from a single article."""
        
        if method == "regex":
            return self.regex_extractor.extract_from_text(article.abstract or "", article.title or "")
        
        elif method == "llm":
            if not self.llm_extractor:
                raise ConfigurationError("LLM extractor not initialized")
            return self.llm_extractor.extract_from_text(article.abstract or "", article.title or "", project_id)
        
        elif method == "hybrid":
            # Use both methods and combine results
            regex_result = self.regex_extractor.extract_from_text(article.abstract or "", article.title or "")
            
            if self.llm_extractor:
                try:
                    llm_result = self.llm_extractor.extract_from_text(article.abstract or "", article.title or "", project_id)
                    # Combine results (simplified - could be more sophisticated)
                    return self._combine_extractions(regex_result, llm_result)
                except Exception as e:
                    logger.warning(f"LLM extraction failed, using regex only: {e}")
                    return regex_result
            else:
                return regex_result
        
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _combine_extractions(self, regex_result: PICOExtractionResult, llm_result: PICOExtractionResult) -> PICOExtractionResult:
        """Combine regex and LLM extraction results."""
        # Simple combination - prefer LLM results but include regex if LLM missed elements
        
        combined_population = llm_result.population_elements or regex_result.population_elements
        combined_intervention = llm_result.intervention_elements or regex_result.intervention_elements
        combined_comparison = llm_result.comparison_elements or regex_result.comparison_elements
        combined_outcomes = llm_result.outcome_elements or regex_result.outcome_elements
        
        return PICOExtractionResult(
            population_elements=combined_population,
            intervention_elements=combined_intervention,
            comparison_elements=combined_comparison,
            outcome_elements=combined_outcomes,
            overall_confidence=max(regex_result.overall_confidence, llm_result.overall_confidence),
            extraction_method="hybrid",
            processing_time_seconds=regex_result.processing_time_seconds + llm_result.processing_time_seconds
        )
    
    def auto_suggest_pico_from_project(self, project_id: str, sample_size: int = 20, method: str = "hybrid") -> Dict:
        """Auto-suggest PICO criteria from project articles."""
        
        # Get sample articles (prefer already labeled as 'include' if available)
        included_articles = db.session.query(Article).filter(
            Article.project_id == project_id,
            Article.status.in_(['include', 'included'])
        ).limit(sample_size).all()
        
        if len(included_articles) < 5:
            # Fall back to any articles if not enough labeled
            articles = db.session.query(Article).filter(
                Article.project_id == project_id
            ).limit(sample_size).all()
        else:
            articles = included_articles
        
        if not articles:
            raise ValidationError("No articles found for PICO extraction")
        
        logger.info(f"Extracting PICO from {len(articles)} articles using {method} method")
        
        # Extract PICO from all articles
        extractions = []
        successful_extractions = 0
        
        for article in articles:
            try:
                extraction = self.extract_from_article(article, method, project_id)
                extractions.append(extraction)
                successful_extractions += 1
            except Exception as e:
                logger.warning(f"Failed to extract PICO from article {article.id}: {e}")
        
        if not extractions:
            raise ValidationError("Failed to extract PICO from any articles")
        
        # Aggregate results
        aggregated = self.aggregator.aggregate_extractions(extractions)
        suggested_criteria = self.aggregator.suggest_pico_criteria(aggregated)
        
        # Calculate confidence metrics
        avg_confidence = sum(e.overall_confidence for e in extractions) / len(extractions)
        extraction_success_rate = successful_extractions / len(articles)
        
        return {
            'suggested_pico': {
                'population': suggested_criteria.population,
                'intervention': suggested_criteria.intervention,
                'comparison': suggested_criteria.comparison,
                'outcomes': suggested_criteria.outcomes,
                'time_frame': suggested_criteria.time_frame,
                'study_types': suggested_criteria.study_types
            },
            'extraction_stats': {
                'articles_processed': len(articles),
                'successful_extractions': successful_extractions,
                'extraction_success_rate': extraction_success_rate,
                'average_confidence': avg_confidence,
                'method_used': method
            },
            'detailed_elements': aggregated,
            'validation_errors': suggested_criteria.validate()
        }

# Global manager instance
pico_extraction_manager = PICOExtractionManager()