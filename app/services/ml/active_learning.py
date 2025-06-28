"""
Active Learning System for Systematic Review Screening
Based on ASReview patterns but integrated with our LLM dual-screening approach.

Integration Points:
- Uses our error_handler.py for robust ML operations
- Integrates with concurrent_processor.py for batch operations
- Uses our config_manager.py for ML configuration
- Connects to our database models (Article, Project)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import joblib
import os

from app.models.screening_models import db, Article, Project
from error_handler import retry_with_backoff, safe_json_parse
from exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class ActiveLearningConfig:
    """Configuration for active learning system."""
    model_type: str = "logistic_regression"  # or "random_forest"
    feature_type: str = "tfidf"  # or "bert_embeddings"
    min_training_samples: int = 20
    max_training_samples: int = 1000
    uncertainty_threshold: float = 0.7
    batch_size: int = 10
    retrain_frequency: int = 50  # Retrain every N new labels
    save_models: bool = True
    model_path: str = "models/active_learning"

class FeatureExtractor:
    """Feature extraction for articles."""
    
    def __init__(self, feature_type: str = "tfidf"):
        self.feature_type = feature_type
        self.vectorizer = None
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit feature extractor and transform texts."""
        if self.feature_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            features = self.vectorizer.fit_transform(texts)
        else:
            raise ValueError(f"Feature type {self.feature_type} not supported")
        
        self.is_fitted = True
        return features.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted extractor."""
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted")
        
        if self.feature_type == "tfidf":
            features = self.vectorizer.transform(texts)
            return features.toarray()
        else:
            raise ValueError(f"Feature type {self.feature_type} not supported")

class ActiveLearner:
    """Active learning system for systematic review screening."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config.feature_type)
        self.classifier = self._create_classifier()
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        
        # Create model directory
        os.makedirs(config.model_path, exist_ok=True)
    
    def _create_classifier(self):
        """Create classifier based on config."""
        if self.config.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Model type {self.config.model_type} not supported")
    
    @retry_with_backoff(max_retries=3)
    def train_initial_model(self, project_id: str) -> Dict[str, float]:
        """Train initial model from labeled articles."""
        logger.info(f"Training initial active learning model for project {project_id}")
        
        # Get labeled articles from database
        labeled_articles = db.session.query(Article).filter(
            Article.project_id == project_id,
            Article.status.in_(['include', 'exclude', 'included', 'excluded'])
        ).all()
        
        if len(labeled_articles) < self.config.min_training_samples:
            raise ValidationError(
                f"Need at least {self.config.min_training_samples} labeled articles, "
                f"got {len(labeled_articles)}"
            )
        
        # Prepare training data
        texts = []
        labels = []
        
        for article in labeled_articles:
            text = f"{article.title or ''} {article.abstract or ''}"
            texts.append(text.strip())
            
            # Convert status to binary label
            label = 1 if article.status in ['include', 'included'] else 0
            labels.append(label)
        
        # Extract features and train
        X = self.feature_extractor.fit_transform(texts)
        y = np.array(labels)
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        # Calculate performance metrics
        cv_scores = cross_val_score(self.classifier, X, y, cv=5, scoring='f1')
        self.performance_metrics = {
            'f1_score_mean': cv_scores.mean(),
            'f1_score_std': cv_scores.std(),
            'training_samples': len(labeled_articles),
            'positive_samples': sum(labels),
            'negative_samples': len(labels) - sum(labels),
            'trained_at': datetime.now().isoformat()
        }
        
        # Save model if configured
        if self.config.save_models:
            self.save_model(project_id)
        
        logger.info(f"Model trained with F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return self.performance_metrics
    
    def predict_relevance_scores(self, articles: List[Article]) -> List[Tuple[str, float, float]]:
        """Predict relevance scores for articles.
        
        Returns:
            List of (article_id, relevance_probability, uncertainty_score)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_initial_model first.")
        
        if not articles:
            return []
        
        # Prepare texts
        texts = []
        article_ids = []
        
        for article in articles:
            text = f"{article.title or ''} {article.abstract or ''}"
            texts.append(text.strip())
            article_ids.append(article.id)
        
        # Extract features and predict
        X = self.feature_extractor.transform(texts)
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(X)
        
        results = []
        for i, article_id in enumerate(article_ids):
            # Probability of being relevant (class 1)
            relevance_prob = probabilities[i][1]
            
            # Uncertainty score (closer to 0.5 = more uncertain)
            uncertainty = 1 - abs(relevance_prob - 0.5) * 2
            
            results.append((article_id, relevance_prob, uncertainty))
        
        return results
    
    def suggest_next_articles(self, project_id: str, n: int = 10, strategy: str = "uncertainty") -> List[Dict]:
        """Suggest next articles to review based on active learning strategy."""
        
        # Get pending articles
        pending_articles = db.session.query(Article).filter(
            Article.project_id == project_id,
            Article.status == 'pending'
        ).all()
        
        if not pending_articles:
            return []
        
        # Get predictions
        predictions = self.predict_relevance_scores(pending_articles)
        
        # Apply strategy
        if strategy == "uncertainty":
            # Sort by uncertainty (highest uncertainty first)
            sorted_predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
        elif strategy == "relevance":
            # Sort by relevance probability (highest relevance first)
            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        elif strategy == "mixed":
            # Combine uncertainty and relevance
            mixed_scores = [(aid, prob, unc, prob * 0.7 + unc * 0.3) 
                          for aid, prob, unc in predictions]
            sorted_predictions = sorted(mixed_scores, key=lambda x: x[3], reverse=True)
            sorted_predictions = [(x[0], x[1], x[2]) for x in sorted_predictions]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Get top N suggestions
        top_predictions = sorted_predictions[:n]
        
        # Build result with article details
        suggestions = []
        for article_id, relevance_prob, uncertainty in top_predictions:
            article = db.session.query(Article).filter(Article.id == article_id).first()
            if article:
                suggestions.append({
                    'article_id': article_id,
                    'title': article.title,
                    'abstract': article.abstract[:300] + "..." if len(article.abstract or "") > 300 else article.abstract,
                    'relevance_probability': float(relevance_prob),
                    'uncertainty_score': float(uncertainty),
                    'recommendation_reason': self._get_recommendation_reason(relevance_prob, uncertainty, strategy)
                })
        
        return suggestions
    
    def _get_recommendation_reason(self, prob: float, uncertainty: float, strategy: str) -> str:
        """Generate human-readable recommendation reason."""
        if strategy == "uncertainty":
            if uncertainty > 0.8:
                return "High uncertainty - model needs your input on this type of paper"
            elif uncertainty > 0.6:
                return "Moderate uncertainty - could go either way"
            else:
                return "Lower uncertainty but still valuable for training"
        elif strategy == "relevance":
            if prob > 0.8:
                return "Very likely relevant - high priority for inclusion"
            elif prob > 0.6:
                return "Likely relevant - worth reviewing"
            else:
                return "Possibly relevant - model suggests checking"
        else:  # mixed
            return "Balanced selection considering both relevance and uncertainty"
    
    def update_model(self, project_id: str, new_article_ids: List[str]) -> bool:
        """Update model with new labeled articles."""
        if len(new_article_ids) < self.config.retrain_frequency:
            return False  # Not enough new labels to retrain
        
        logger.info(f"Updating active learning model with {len(new_article_ids)} new labels")
        
        try:
            # Retrain with all labeled data
            metrics = self.train_initial_model(project_id)
            
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'new_labels': len(new_article_ids),
                'metrics': metrics
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return False
    
    def save_model(self, project_id: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_file = os.path.join(self.config.model_path, f"model_{project_id}.joblib")
        features_file = os.path.join(self.config.model_path, f"features_{project_id}.joblib")
        
        joblib.dump(self.classifier, model_file)
        joblib.dump(self.feature_extractor, features_file)
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, project_id: str) -> bool:
        """Load trained model from disk."""
        model_file = os.path.join(self.config.model_path, f"model_{project_id}.joblib")
        features_file = os.path.join(self.config.model_path, f"features_{project_id}.joblib")
        
        if not (os.path.exists(model_file) and os.path.exists(features_file)):
            return False
        
        try:
            self.classifier = joblib.load(model_file)
            self.feature_extractor = joblib.load(features_file)
            self.is_trained = True
            logger.info(f"Model loaded from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary including history."""
        return {
            'current_metrics': self.performance_metrics,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'config': {
                'model_type': self.config.model_type,
                'feature_type': self.config.feature_type,
                'min_training_samples': self.config.min_training_samples
            }
        }

class ActiveLearningManager:
    """High-level manager for active learning integration."""
    
    def __init__(self):
        self.learners = {}  # project_id -> ActiveLearner
    
    def get_or_create_learner(self, project_id: str, config: Optional[ActiveLearningConfig] = None) -> ActiveLearner:
        """Get existing learner or create new one."""
        if project_id not in self.learners:
            if config is None:
                config = ActiveLearningConfig()
            
            learner = ActiveLearner(config)
            # Try to load existing model
            if learner.load_model(project_id):
                logger.info(f"Loaded existing model for project {project_id}")
            
            self.learners[project_id] = learner
        
        return self.learners[project_id]
    
    def initialize_active_learning(self, project_id: str) -> Dict:
        """Initialize active learning for a project."""
        try:
            learner = self.get_or_create_learner(project_id)
            
            if not learner.is_trained:
                metrics = learner.train_initial_model(project_id)
                return {
                    'success': True,
                    'message': 'Active learning initialized successfully',
                    'metrics': metrics
                }
            else:
                return {
                    'success': True,
                    'message': 'Active learning already initialized',
                    'metrics': learner.performance_metrics
                }
        
        except ValidationError as e:
            return {
                'success': False,
                'message': str(e),
                'suggestion': 'Please label more articles manually before enabling active learning'
            }
        except Exception as e:
            logger.error(f"Failed to initialize active learning: {e}")
            return {
                'success': False,
                'message': f'Failed to initialize: {str(e)}'
            }
    
    def get_smart_suggestions(self, project_id: str, count: int = 10, strategy: str = "uncertainty") -> List[Dict]:
        """Get smart article suggestions using active learning."""
        try:
            learner = self.get_or_create_learner(project_id)
            
            if not learner.is_trained:
                # Fall back to random selection with warning
                pending_articles = db.session.query(Article).filter(
                    Article.project_id == project_id,
                    Article.status == 'pending'
                ).limit(count).all()
                
                return [{
                    'article_id': article.id,
                    'title': article.title,
                    'abstract': (article.abstract[:300] + "...") if len(article.abstract or "") > 300 else article.abstract,
                    'relevance_probability': 0.5,  # Unknown
                    'uncertainty_score': 1.0,     # Maximum uncertainty
                    'recommendation_reason': 'Random selection - active learning not yet trained'
                } for article in pending_articles]
            
            return learner.suggest_next_articles(project_id, count, strategy)
        
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []

# Global manager instance
active_learning_manager = ActiveLearningManager()