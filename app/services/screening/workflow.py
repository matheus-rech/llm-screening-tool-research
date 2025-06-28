"""
Screening Workflow Orchestrator
Handles batch, loop, and chain processing for modern LLM screening.

Workflow Options:
1. BATCH: Process multiple articles simultaneously
2. LOOP: Process articles one by one with real-time updates  
3. CHAIN: Process articles in dependency chains (e.g., similar articles together)
"""

import logging
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

from app.models.screening_models import db, Project, Article
from .modern_llm import (
    DualProviderScreeningOrchestrator, 
    ScreeningCriteria, 
    HumanReviewTriggers,
    ScreeningResultsStore,
    ComprehensiveScreeningResult
)
from app.services.utils.concurrent_processor import OptimizedProcessor
# from app.services.utils.cost_tracker import cost_tracker  # Temporarily disabled

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Available workflow processing types."""
    BATCH = "batch"           # Process in batches simultaneously
    LOOP = "loop"             # Process one by one with updates
    CHAIN = "chain"           # Process in dependency chains
    ADAPTIVE = "adaptive"     # Adapt strategy based on results

@dataclass
class WorkflowConfig:
    """Configuration for screening workflow."""
    workflow_type: WorkflowType
    batch_size: int = 10
    max_concurrent: int = 5
    enable_early_stopping: bool = True
    early_stop_threshold: float = 0.95  # Stop if 95% excluded
    progress_callback: Optional[Callable] = None
    enable_cost_optimization: bool = True
    max_budget: Optional[float] = None

@dataclass
class WorkflowProgress:
    """Progress tracking for workflow execution."""
    total_articles: int
    processed_articles: int
    included_count: int
    excluded_count: int
    uncertain_count: int
    human_review_required: int
    total_cost: float
    estimated_time_remaining: float
    current_status: str
    
    @property
    def completion_percentage(self) -> float:
        if self.total_articles == 0:
            return 0.0
        return (self.processed_articles / self.total_articles) * 100

class ScreeningWorkflowOrchestrator:
    """Main orchestrator for screening workflows."""
    
    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.dual_screener = DualProviderScreeningOrchestrator(openai_api_key, anthropic_api_key)
        self.processor = OptimizedProcessor(max_workers=5, max_api_calls_per_minute=120)
        self.results_store = ScreeningResultsStore()
        
    async def execute_screening_workflow(self, 
                                       project_id: str, 
                                       criteria: ScreeningCriteria,
                                       config: WorkflowConfig) -> AsyncGenerator[WorkflowProgress, None]:
        """Execute screening workflow with real-time progress updates."""
        
        # Get articles to process
        articles = self._get_articles_for_processing(project_id)
        
        if not articles:
            raise ValueError("No articles found for processing")
        
        # Initialize progress tracking
        progress = WorkflowProgress(
            total_articles=len(articles),
            processed_articles=0,
            included_count=0,
            excluded_count=0,
            uncertain_count=0,
            human_review_required=0,
            total_cost=0.0,
            estimated_time_remaining=0.0,
            current_status="Initializing screening workflow"
        )
        
        # Start cost tracking
        cost_tracker.start_tracking(project_id)
        
        try:
            # Execute based on workflow type
            if config.workflow_type == WorkflowType.BATCH:
                async for update in self._execute_batch_workflow(articles, criteria, project_id, config, progress):
                    yield update
                    
            elif config.workflow_type == WorkflowType.LOOP:
                async for update in self._execute_loop_workflow(articles, criteria, project_id, config, progress):
                    yield update
                    
            elif config.workflow_type == WorkflowType.CHAIN:
                async for update in self._execute_chain_workflow(articles, criteria, project_id, config, progress):
                    yield update
                    
            elif config.workflow_type == WorkflowType.ADAPTIVE:
                async for update in self._execute_adaptive_workflow(articles, criteria, project_id, config, progress):
                    yield update
            
            # Finalize workflow
            progress.current_status = "Workflow completed successfully"
            yield progress
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            progress.current_status = f"Workflow failed: {str(e)}"
            yield progress
            raise
        
        finally:
            # Finalize cost tracking
            cost_tracker.finalize_tracking(project_id)
    
    async def _execute_batch_workflow(self, 
                                    articles: List[Article], 
                                    criteria: ScreeningCriteria,
                                    project_id: str,
                                    config: WorkflowConfig,
                                    progress: WorkflowProgress) -> AsyncGenerator[WorkflowProgress, None]:
        """Execute batch processing workflow."""
        
        progress.current_status = "Processing articles in batches"
        yield progress
        
        # Process in batches
        for i in range(0, len(articles), config.batch_size):
            batch = articles[i:i + config.batch_size]
            
            progress.current_status = f"Processing batch {i//config.batch_size + 1}"
            yield progress
            
            # Process batch concurrently
            batch_tasks = []
            for article in batch:
                task = self._screen_single_article(article, criteria, project_id)
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Update progress with batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Article screening failed: {result}")
                    continue
                    
                progress.processed_articles += 1
                self._update_progress_with_result(progress, result)
            
            # Check for early stopping
            if config.enable_early_stopping and self._should_stop_early(progress, config):
                progress.current_status = "Early stopping triggered"
                yield progress
                break
            
            yield progress
    
    async def _execute_loop_workflow(self, 
                                   articles: List[Article], 
                                   criteria: ScreeningCriteria,
                                   project_id: str,
                                   config: WorkflowConfig,
                                   progress: WorkflowProgress) -> AsyncGenerator[WorkflowProgress, None]:
        """Execute loop processing workflow with real-time updates."""
        
        progress.current_status = "Processing articles sequentially"
        yield progress
        
        for i, article in enumerate(articles):
            progress.current_status = f"Processing article {i+1}: {article.title[:50]}..."
            yield progress
            
            try:
                # Screen single article
                result = await self._screen_single_article(article, criteria, project_id)
                
                # Update progress
                progress.processed_articles += 1
                self._update_progress_with_result(progress, result)
                
                # Check for early stopping
                if config.enable_early_stopping and self._should_stop_early(progress, config):
                    progress.current_status = "Early stopping triggered"
                    yield progress
                    break
                    
            except Exception as e:
                logger.error(f"Failed to process article {article.id}: {e}")
                progress.processed_articles += 1
            
            yield progress
    
    async def _execute_chain_workflow(self, 
                                    articles: List[Article], 
                                    criteria: ScreeningCriteria,
                                    project_id: str,
                                    config: WorkflowConfig,
                                    progress: WorkflowProgress) -> AsyncGenerator[WorkflowProgress, None]:
        """Execute chain processing workflow (similar articles together)."""
        
        progress.current_status = "Organizing articles into similarity chains"
        yield progress
        
        # Group articles by similarity (simplified - could use ML clustering)
        article_chains = self._group_articles_by_similarity(articles)
        
        for chain_index, chain in enumerate(article_chains):
            progress.current_status = f"Processing chain {chain_index + 1} of {len(article_chains)}"
            yield progress
            
            # Process articles in chain
            for article in chain:
                try:
                    result = await self._screen_single_article(article, criteria, project_id)
                    progress.processed_articles += 1
                    self._update_progress_with_result(progress, result)
                    
                except Exception as e:
                    logger.error(f"Failed to process article {article.id}: {e}")
                    progress.processed_articles += 1
                
                yield progress
    
    async def _execute_adaptive_workflow(self, 
                                       articles: List[Article], 
                                       criteria: ScreeningCriteria,
                                       project_id: str,
                                       config: WorkflowConfig,
                                       progress: WorkflowProgress) -> AsyncGenerator[WorkflowProgress, None]:
        """Execute adaptive workflow that changes strategy based on results."""
        
        progress.current_status = "Starting adaptive screening workflow"
        yield progress
        
        # Start with loop processing for first 10 articles to assess patterns
        initial_batch = articles[:10]
        remaining_articles = articles[10:]
        
        inclusion_rate = 0.0
        
        # Process initial batch
        for article in initial_batch:
            try:
                result = await self._screen_single_article(article, criteria, project_id)
                progress.processed_articles += 1
                self._update_progress_with_result(progress, result)
                
                # Calculate inclusion rate
                if progress.processed_articles > 0:
                    inclusion_rate = progress.included_count / progress.processed_articles
                    
            except Exception as e:
                logger.error(f"Failed to process article {article.id}: {e}")
                progress.processed_articles += 1
            
            yield progress
        
        # Adapt strategy based on initial results
        if inclusion_rate > 0.3:  # High inclusion rate - use batch processing
            progress.current_status = "Switching to batch processing (high inclusion rate detected)"
            yield progress
            
            config.workflow_type = WorkflowType.BATCH
            async for update in self._execute_batch_workflow(remaining_articles, criteria, project_id, config, progress):
                yield update
                
        elif inclusion_rate < 0.1:  # Low inclusion rate - use early stopping
            progress.current_status = "Switching to early stopping mode (low inclusion rate detected)"
            yield progress
            
            config.enable_early_stopping = True
            config.early_stop_threshold = 0.9
            async for update in self._execute_loop_workflow(remaining_articles, criteria, project_id, config, progress):
                yield update
        
        else:  # Medium inclusion rate - continue with loop
            progress.current_status = "Continuing with sequential processing"
            yield progress
            
            async for update in self._execute_loop_workflow(remaining_articles, criteria, project_id, config, progress):
                yield update
    
    async def _screen_single_article(self, 
                                   article: Article, 
                                   criteria: ScreeningCriteria,
                                   project_id: str) -> Dict:
        """Screen a single article using dual provider approach."""
        
        # Screen with both providers
        results = self.dual_screener.screen_article_dual_provider(article, criteria, project_id)
        
        openai_result = results.get('openai')
        anthropic_result = results.get('anthropic')
        
        # Analyze agreement
        agreement_analysis = self.dual_screener.analyze_provider_agreement(openai_result, anthropic_result)
        
        # Check human review triggers
        human_review_triggers = HumanReviewTriggers.should_trigger_human_review(
            openai_result, anthropic_result
        )
        
        # Store results
        screening_record = self.results_store.store_screening_results(
            article.id,
            project_id,
            openai_result,
            anthropic_result,
            agreement_analysis,
            human_review_triggers
        )
        
        return screening_record
    
    def _update_progress_with_result(self, progress: WorkflowProgress, result: Dict):
        """Update progress tracking with screening result."""
        
        final_decision = result.get('final_decision', 'UNCERTAIN')
        
        if final_decision == 'INCLUDE':
            progress.included_count += 1
        elif final_decision == 'EXCLUDE':
            progress.excluded_count += 1
        else:
            progress.uncertain_count += 1
        
        if result.get('requires_human_review', False):
            progress.human_review_required += 1
        
        # Update cost (simplified)
        costs = cost_tracker.get_current_costs(result['project_id'])
        if costs:
            progress.total_cost = costs.get('total_cost', 0.0)
    
    def _should_stop_early(self, progress: WorkflowProgress, config: WorkflowConfig) -> bool:
        """Determine if early stopping should be triggered."""
        
        if progress.processed_articles < 20:  # Need minimum sample
            return False
        
        exclusion_rate = progress.excluded_count / progress.processed_articles
        return exclusion_rate >= config.early_stop_threshold
    
    def _get_articles_for_processing(self, project_id: str) -> List[Article]:
        """Get articles that need processing for the project."""
        
        return db.session.query(Article).filter(
            Article.project_id == project_id,
            Article.status == 'pending'
        ).all()
    
    def _group_articles_by_similarity(self, articles: List[Article]) -> List[List[Article]]:
        """Group articles by similarity (simplified implementation)."""
        
        # Simplified grouping by journal or year
        # In practice, you'd use ML clustering based on abstracts
        
        groups = {}
        
        for article in articles:
            # Group by journal as a simple similarity measure
            key = article.journal or "unknown"
            if key not in groups:
                groups[key] = []
            groups[key].append(article)
        
        return list(groups.values())

# ============================================================================
# WORKFLOW FACTORY
# ============================================================================

class ScreeningWorkflowFactory:
    """Factory for creating configured screening workflows."""
    
    @staticmethod
    def create_fast_screening_workflow(project_id: str, 
                                     max_budget: Optional[float] = None) -> tuple[ScreeningWorkflowOrchestrator, WorkflowConfig]:
        """Create workflow optimized for speed."""
        
        import os
        orchestrator = ScreeningWorkflowOrchestrator(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        config = WorkflowConfig(
            workflow_type=WorkflowType.BATCH,
            batch_size=20,
            max_concurrent=10,
            enable_early_stopping=True,
            early_stop_threshold=0.9,
            max_budget=max_budget
        )
        
        return orchestrator, config
    
    @staticmethod
    def create_accurate_screening_workflow(project_id: str) -> tuple[ScreeningWorkflowOrchestrator, WorkflowConfig]:
        """Create workflow optimized for accuracy."""
        
        import os
        orchestrator = ScreeningWorkflowOrchestrator(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        config = WorkflowConfig(
            workflow_type=WorkflowType.LOOP,
            batch_size=1,
            max_concurrent=1,
            enable_early_stopping=False
        )
        
        return orchestrator, config
    
    @staticmethod
    def create_balanced_screening_workflow(project_id: str) -> tuple[ScreeningWorkflowOrchestrator, WorkflowConfig]:
        """Create workflow balanced between speed and accuracy."""
        
        import os
        orchestrator = ScreeningWorkflowOrchestrator(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        
        config = WorkflowConfig(
            workflow_type=WorkflowType.ADAPTIVE,
            batch_size=10,
            max_concurrent=5,
            enable_early_stopping=True,
            early_stop_threshold=0.95
        )
        
        return orchestrator, config