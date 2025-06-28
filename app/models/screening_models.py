"""
Database Models for Systematic Review Screening
Contains SQLAlchemy models for projects, articles, and related entities.
"""

from datetime import datetime
import json
import logging
from typing import List, Dict
from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger(__name__)

# This db instance will be imported and used by app/__init__.py
db = SQLAlchemy()

class Project(db.Model):
    """Project model for systematic review projects."""

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Configuration stored as JSON
    config = db.Column(db.JSON)

    # Status tracking
    status = db.Column(db.String(50), default='active')  # active, completed, archived

    # File information
    original_filename = db.Column(db.String(255))
    file_path = db.Column(db.String(500))

    # Statistics
    total_articles = db.Column(db.Integer, default=0)
    processed_articles = db.Column(db.Integer, default=0)

    # Relationships
    articles = db.relationship('Article', backref='project_ref', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        """String representation of Project."""
        return f'<Project {self.name}>'

    @property
    def has_screening_results(self):
        """Check if project has articles with decision_reasoning."""
        return any(article.decision_reasoning is not None for article in self.articles)

    def to_dict(self):
        """Convert project to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'config': self.config,
            'status': self.status,
            'total_articles': self.total_articles,
            'processed_articles': self.processed_articles
        }

class Article(db.Model):
    """Article model for individual research papers."""

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)

    # Article metadata
    title = db.Column(db.Text)
    authors = db.Column(db.Text)
    journal = db.Column(db.String(500))
    year = db.Column(db.Integer)
    abstract = db.Column(db.Text)
    doi = db.Column(db.String(200))
    pmid = db.Column(db.String(50))

    # Original citation data (stored as JSON for flexibility)
    original_data = db.Column(db.JSON)

    # Screening status
    status = db.Column(db.String(50), default='pending')  # pending, included, excluded, uncertain, human_review_required

    # Decision reasoning (stores AI analysis and human decisions)
    decision_reasoning = db.Column(db.JSON)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Screening metadata
    screened_by = db.Column(db.String(100))  # user or AI model identifier
    screening_date = db.Column(db.DateTime)

    # Priority score (for active learning)
    priority_score = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f'<Article {self.title[:50]}...>'

    def to_dict(self):
        """Convert article to dictionary."""
        return {
            'id': self.id,
            'project_id': self.project_id,
            'title': self.title,
            'authors': self.authors,
            'journal': self.journal,
            'year': self.year,
            'abstract': self.abstract,
            'doi': self.doi,
            'pmid': self.pmid,
            'status': self.status,
            'decision_reasoning': self.decision_reasoning,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'screened_by': self.screened_by,
            'screening_date': self.screening_date.isoformat() if self.screening_date else None,
            'priority_score': self.priority_score
        }

# Additional models for enhanced features

class Reviewer(db.Model):
    """Model for collaborative screening reviewers."""

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    role = db.Column(db.String(50), default='reviewer')  # reviewer, expert, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Reviewer {self.name}>'

class ReviewAssignment(db.Model):
    """Model for tracking reviewer assignments to projects."""

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('reviewer.id'), nullable=False)
    role = db.Column(db.String(50), default='reviewer')  # reviewer, adjudicator, expert
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Performance tracking
    articles_reviewed = db.Column(db.Integer, default=0)
    agreement_rate = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f'<ReviewAssignment P{self.project_id} R{self.reviewer_id}>'

class ScreeningDecision(db.Model):
    """Model for individual screening decisions (supports multiple reviewers)."""

    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey('article.id'), nullable=False)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('reviewer.id'), nullable=True)  # None for AI decisions

    # Decision details
    decision = db.Column(db.String(20), nullable=False)  # INCLUDE, EXCLUDE, UNCERTAIN
    confidence = db.Column(db.Float)
    reasoning = db.Column(db.Text)

    # Decision metadata
    decision_type = db.Column(db.String(20), default='human')  # human, ai_openai, ai_anthropic
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Additional data (stored as JSON)
    decision_metadata = db.Column(db.JSON)

    def __repr__(self):
        return f'<ScreeningDecision A{self.article_id} {self.decision}>'

class ConflictResolution(db.Model):
    """Model for tracking conflict resolution between reviewers."""

    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey('article.id'), nullable=False)

    # Conflict details
    reviewer1_decision = db.Column(db.String(20))
    reviewer2_decision = db.Column(db.String(20))
    final_decision = db.Column(db.String(20))

    # Resolution metadata
    resolved_by = db.Column(db.Integer, db.ForeignKey('reviewer.id'))
    resolution_method = db.Column(db.String(50))  # discussion, adjudication, ai_assistance
    resolution_notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

    def __repr__(self):
        return f'<ConflictResolution A{self.article_id}>'

class PublicationSource(db.Model):
    """Track publication sources for articles."""

    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey('article.id'), nullable=False)
    source_database = db.Column(db.String(50), nullable=False)  # PubMed, Scopus, etc.
    source_id = db.Column(db.String(200))  # Original ID from source
    import_date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PublicationSource {self.source_database}:{self.source_id}>'

class DuplicateDetection:
    """Service for detecting and managing duplicate articles."""

    @staticmethod
    def find_duplicates(project_id: int) -> List[Dict]:
        """Find potential duplicates in a project."""
        articles = Article.query.filter_by(project_id=project_id).all()
        duplicates = []

        doi_groups = {}
        for article in articles:
            if article.doi:
                clean_doi = article.doi.strip().lower()
                if clean_doi in doi_groups:
                    doi_groups[clean_doi].append(article)
                else:
                    doi_groups[clean_doi] = [article]

        for doi, group in doi_groups.items():
            if len(group) > 1:
                duplicates.append({
                    'type': 'doi_match',
                    'identifier': doi,
                    'articles': [a.id for a in group],
                    'confidence': 1.0
                })

        pmid_groups = {}
        for article in articles:
            if article.pmid:
                clean_pmid = article.pmid.strip()
                if clean_pmid in pmid_groups:
                    pmid_groups[clean_pmid].append(article)
                else:
                    pmid_groups[clean_pmid] = [article]

        for pmid, group in pmid_groups.items():
            if len(group) > 1:
                duplicates.append({
                    'type': 'pmid_match',
                    'identifier': pmid,
                    'articles': [a.id for a in group],
                    'confidence': 1.0
                })

        title_duplicates = DuplicateDetection._find_title_duplicates(articles)
        duplicates.extend(title_duplicates)

        return duplicates

    @staticmethod
    def _find_title_duplicates(articles: List[Article], threshold: float = 0.9) -> List[Dict]:
        """Find duplicates based on title similarity."""
        from difflib import SequenceMatcher

        duplicates = []
        processed = set()

        for i, article1 in enumerate(articles):
            if i in processed or not article1.title:
                continue

            similar_articles = [article1]

            for j, article2 in enumerate(articles[i+1:], i+1):
                if j in processed or not article2.title:
                    continue

                similarity = SequenceMatcher(None,
                    article1.title.lower().strip(),
                    article2.title.lower().strip()
                ).ratio()

                if similarity >= threshold:
                    similar_articles.append(article2)
                    processed.add(j)

            if len(similar_articles) > 1:
                duplicates.append({
                    'type': 'title_similarity',
                    'identifier': article1.title[:50] + "...",
                    'articles': [a.id for a in similar_articles],
                    'confidence': max(SequenceMatcher(None,
                        similar_articles[0].title.lower(),
                        a.title.lower()
                    ).ratio() for a in similar_articles[1:])
                })
                processed.add(i)

        return duplicates

    @staticmethod
    def merge_duplicates(article_ids: List[int], keep_article_id: int) -> bool:
        """Merge duplicate articles, keeping one as primary."""
        try:
            keep_article = Article.query.get(keep_article_id)
            if not keep_article or keep_article_id not in article_ids:
                return False

            merge_articles = Article.query.filter(
                Article.id.in_([aid for aid in article_ids if aid != keep_article_id])
            ).all()

            for article in merge_articles:
                sources = PublicationSource.query.filter_by(article_id=article.id).all()
                for source in sources:
                    source.article_id = keep_article_id

                db.session.delete(article)

            db.session.commit()
            return True

        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to merge duplicates: {e}")
            return False
