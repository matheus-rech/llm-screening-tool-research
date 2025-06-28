"""Optimized concurrent processing for article screening."""

import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from threading import Lock, Semaphore
import time
from queue import Queue, Empty
from threading import Thread

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of processing an article."""
    article_id: str
    success: bool
    result: Dict[Any, Any]
    error: Optional[str] = None
    processing_time: float = 0.0

class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = Lock()
    
    def acquire(self) -> bool:
        """Try to acquire permission for an API call."""
        with self._lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False
    
    def time_until_available(self) -> float:
        """Get time until next call is available."""
        with self._lock:
            if len(self.calls) < self.max_calls:
                return 0.0
            
            oldest_call = min(self.calls)
            return self.time_window - (time.time() - oldest_call)

class OptimizedProcessor:
    """Optimized processor for concurrent article screening."""
    
    def __init__(self, 
                 max_workers: int = 10, 
                 max_api_calls_per_minute: int = 60,
                 batch_size: int = 5):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.rate_limiter = RateLimiter(max_api_calls_per_minute, 60.0)
        self.semaphore = Semaphore(max_workers)
        self.results_queue = Queue()
        self.stats = {
            'processed': 0,
            'errors': 0,
            'total_time': 0.0,
            'api_calls': 0
        }
    
    def process_batch_optimized(self, 
                               articles: List[Dict], 
                               processing_func: Callable,
                               **kwargs) -> List[ProcessingResult]:
        """Process a batch of articles with optimized concurrency."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_article = {}
            
            for article in articles:
                future = executor.submit(self._process_single_with_rate_limit, 
                                       article, processing_func, **kwargs)
                future_to_article[future] = article
            
            # Collect results as they complete
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.stats['processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
                    error_result = ProcessingResult(
                        article_id=article.get('id', 'unknown'),
                        success=False,
                        result={},
                        error=str(e)
                    )
                    results.append(error_result)
                    self.stats['errors'] += 1
        
        return results
    
    def _process_single_with_rate_limit(self, 
                                       article: Dict, 
                                       processing_func: Callable,
                                       **kwargs) -> ProcessingResult:
        """Process a single article with rate limiting."""
        
        start_time = time.time()
        
        # Wait for rate limit if necessary
        while not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.time_until_available()
            if wait_time > 0:
                time.sleep(min(wait_time + 0.1, 1.0))  # Small buffer and max 1s wait
        
        try:
            with self.semaphore:  # Limit concurrent executions
                result = processing_func(article, **kwargs)
                self.stats['api_calls'] += 1
                
                processing_time = time.time() - start_time
                self.stats['total_time'] += processing_time
                
                return ProcessingResult(
                    article_id=article.get('id', str(hash(str(article)))),
                    success=True,
                    result=result,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing article: {e}")
            return ProcessingResult(
                article_id=article.get('id', str(hash(str(article)))),
                success=False,
                result={},
                error=str(e),
                processing_time=processing_time
            )
    
    def process_articles_streaming(self, 
                                  articles: List[Dict],
                                  processing_func: Callable,
                                  progress_callback: Optional[Callable] = None,
                                  **kwargs) -> None:
        """Process articles with streaming results and progress updates."""
        
        total_articles = len(articles)
        processed_count = 0
        
        # Process in batches
        for i in range(0, len(articles), self.batch_size):
            batch = articles[i:i + self.batch_size]
            batch_results = self.process_batch_optimized(batch, processing_func, **kwargs)
            
            for result in batch_results:
                processed_count += 1
                
                if progress_callback:
                    progress_callback({
                        'processed': processed_count,
                        'total': total_articles,
                        'current_result': result,
                        'stats': self.get_stats()
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats['processed'] > 0:
            stats['avg_processing_time'] = stats['total_time'] / stats['processed']
            stats['success_rate'] = (stats['processed'] - stats['errors']) / stats['processed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats

class AsyncProcessor:
    """Async processor for even better performance with aiohttp."""
    
    def __init__(self, max_concurrent: int = 20):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_articles_async(self, 
                                   articles: List[Dict],
                                   async_processing_func: Callable,
                                   **kwargs) -> List[ProcessingResult]:
        """Process articles asynchronously."""
        
        async def process_single(article):
            async with self.semaphore:
                start_time = time.time()
                try:
                    result = await async_processing_func(article, **kwargs)
                    return ProcessingResult(
                        article_id=article.get('id', str(hash(str(article)))),
                        success=True,
                        result=result,
                        processing_time=time.time() - start_time
                    )
                except Exception as e:
                    return ProcessingResult(
                        article_id=article.get('id', str(hash(str(article)))),
                        success=False,
                        result={},
                        error=str(e),
                        processing_time=time.time() - start_time
                    )
        
        tasks = [process_single(article) for article in articles]
        return await asyncio.gather(*tasks)