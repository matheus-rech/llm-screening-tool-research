"""
Collaborative Screening System for Multi-Reviewer Projects
Based on Abstrackr patterns but enhanced with our LLM dual-screening.

Integration Points:
- Extends our database models (Article, Project) with reviewer support
- Uses our error_handler.py for robust multi-user operations
- Integrates with our config_manager.py for reviewer configuration
- Works with our active_learning.py for team-based ML training
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from sqlalchemy import and_, or_, func
import hashlib
from flask import session

from app import db, Project, Article
from exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)

class ReviewerRole(Enum):
    """Roles for collaborative screening."""
    ADMIN = "admin"           # Full project control
    REVIEWER = "reviewer"     # Can screen and resolve conflicts
    SCREENER = "screener"     # Can only do initial screening
    OBSERVER = "observer"     # Read-only access

class DecisionStatus(Enum):
    """Decision status for collaborative review."""
    PENDING = "pending"
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"
    CONFLICT = "conflict"
    RESOLVED = "resolved"

# Extended database models for collaboration
class Reviewer(db.Model):
    """Reviewer model for collaborative screening."""
    __tablename__ = 'reviewers'
    
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    institution = db.Column(db.String(200))
    expertise_areas = db.Column(db.Text)  # JSON list
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    project_assignments = db.relationship('ProjectReviewer', back_populates='reviewer', cascade='all, delete-orphan')
    decisions = db.relationship('ReviewerDecision', back_populates='reviewer', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Reviewer {self.name} ({self.email})>'

class ProjectReviewer(db.Model):
    """Association between projects and reviewers."""
    __tablename__ = 'project_reviewers'
    
    id = db.Column(db.String(36), primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey('project.id'), nullable=False)
    reviewer_id = db.Column(db.String(36), db.ForeignKey('reviewers.id'), nullable=False)
    role = db.Column(db.Enum(ReviewerRole), nullable=False, default=ReviewerRole.SCREENER)
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    assigned_by = db.Column(db.String(36))  # ID of admin who assigned
    is_active = db.Column(db.Boolean, default=True)
    
    # Screening configuration
    screening_batch_size = db.Column(db.Integer, default=10)
    auto_assign_articles = db.Column(db.Boolean, default=True)
    
    # Relationships
    project = db.relationship('Project', backref='reviewer_assignments')
    reviewer = db.relationship('Reviewer', back_populates='project_assignments')
    
    __table_args__ = (db.UniqueConstraint('project_id', 'reviewer_id', name='unique_project_reviewer'),)

class ReviewerDecision(db.Model):
    """Individual reviewer decisions on articles."""
    __tablename__ = 'reviewer_decisions'
    
    id = db.Column(db.String(36), primary_key=True)
    article_id = db.Column(db.String(36), db.ForeignKey('article.id'), nullable=False)
    reviewer_id = db.Column(db.String(36), db.ForeignKey('reviewers.id'), nullable=False)
    project_id = db.Column(db.String(36), db.ForeignKey('project.id'), nullable=False)
    
    # Decision details
    decision = db.Column(db.Enum(DecisionStatus), nullable=False)
    reasoning = db.Column(db.Text)
    confidence_level = db.Column(db.Integer, default=3)  # 1-5 scale
    time_spent_seconds = db.Column(db.Integer)
    
    # LLM assistance info
    used_llm_suggestion = db.Column(db.Boolean, default=False)
    llm_suggestion_decision = db.Column(db.String(20))
    llm_agreement = db.Column(db.Boolean)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ip_address = db.Column(db.String(45))  # For audit trail
    user_agent = db.Column(db.Text)
    
    # Relationships
    article = db.relationship('Article', backref='reviewer_decisions')
    reviewer = db.relationship('Reviewer', back_populates='decisions')
    project = db.relationship('Project')
    
    __table_args__ = (db.UniqueConstraint('article_id', 'reviewer_id', name='unique_article_reviewer_decision'),)

class ConflictResolution(db.Model):
    """Track conflict resolution between reviewers."""
    __tablename__ = 'conflict_resolutions'
    
    id = db.Column(db.String(36), primary_key=True)
    article_id = db.Column(db.String(36), db.ForeignKey('article.id'), nullable=False)
    project_id = db.Column(db.String(36), db.ForeignKey('project.id'), nullable=False)
    
    # Conflicting decisions
    decision1_id = db.Column(db.String(36), db.ForeignKey('reviewer_decisions.id'))
    decision2_id = db.Column(db.String(36), db.ForeignKey('reviewer_decisions.id'))
    
    # Resolution details
    resolved_by = db.Column(db.String(36), db.ForeignKey('reviewers.id'))  # Reviewer who resolved
    final_decision = db.Column(db.Enum(DecisionStatus), nullable=False)
    resolution_reasoning = db.Column(db.Text)
    resolution_method = db.Column(db.String(50))  # "manual", "llm_assisted", "discussion"
    
    # LLM assistance in resolution
    llm_resolution_suggestion = db.Column(db.Text)  # JSON of LLM suggestion
    
    # Metadata
    conflict_detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)
    
    # Relationships
    article = db.relationship('Article')
    project = db.relationship('Project')
    resolver = db.relationship('Reviewer', foreign_keys=[resolved_by])

@dataclass
class InterRaterReliability:
    """Inter-rater reliability metrics."""
    reviewer1_id: str
    reviewer2_id: str
    total_overlapping_decisions: int
    agreements: int
    disagreements: int
    agreement_percentage: float
    cohens_kappa: float
    specific_agreements: Dict[str, int]  # by decision type
    
class CollaborativeScreeningManager:
    """Manager for collaborative screening operations."""
    
    def __init__(self):
        self.session_reviewers = {}  # Track current session reviewers
    
    def create_reviewer(self, name: str, email: str, institution: str = None, expertise_areas: List[str] = None) -> Reviewer:
        """Create a new reviewer."""
        # Check if reviewer already exists
        existing = Reviewer.query.filter_by(email=email).first()
        if existing:
            raise ValidationError(f"Reviewer with email {email} already exists")
        
        reviewer = Reviewer(
            id=self._generate_id(),
            name=name,
            email=email,
            institution=institution,
            expertise_areas=expertise_areas or []
        )
        
        db.session.add(reviewer)
        db.session.commit()
        
        logger.info(f"Created reviewer: {name} ({email})")
        return reviewer
    
    def assign_reviewer_to_project(self, project_id: str, reviewer_id: str, role: ReviewerRole, assigned_by: str = None) -> ProjectReviewer:
        """Assign a reviewer to a project with specific role."""
        # Validate project and reviewer exist
        project = Project.query.get(project_id)
        reviewer = Reviewer.query.get(reviewer_id)
        
        if not project:
            raise ValidationError(f"Project {project_id} not found")
        if not reviewer:
            raise ValidationError(f"Reviewer {reviewer_id} not found")
        
        # Check if already assigned
        existing = ProjectReviewer.query.filter_by(
            project_id=project_id,
            reviewer_id=reviewer_id,
            is_active=True
        ).first()
        
        if existing:
            existing.role = role
            existing.assigned_by = assigned_by
            db.session.commit()
            return existing
        
        assignment = ProjectReviewer(
            id=self._generate_id(),
            project_id=project_id,
            reviewer_id=reviewer_id,
            role=role,
            assigned_by=assigned_by
        )
        
        db.session.add(assignment)
        db.session.commit()
        
        logger.info(f"Assigned reviewer {reviewer.name} to project {project.name} as {role.value}")
        return assignment
    
    def get_articles_for_reviewer(self, project_id: str, reviewer_id: str, count: int = 10, strategy: str = "round_robin") -> List[Article]:
        """Get articles assigned to a specific reviewer."""
        # Get reviewer assignment
        assignment = ProjectReviewer.query.filter_by(
            project_id=project_id,
            reviewer_id=reviewer_id,
            is_active=True
        ).first()
        
        if not assignment:
            raise ValidationError(f"Reviewer not assigned to project")
        
        if strategy == "round_robin":
            # Get articles not yet reviewed by this reviewer
            already_reviewed = db.session.query(ReviewerDecision.article_id).filter_by(
                reviewer_id=reviewer_id,
                project_id=project_id
            ).subquery()
            
            articles = db.session.query(Article).filter(
                Article.project_id == project_id,
                ~Article.id.in_(already_reviewed)
            ).limit(count).all()
            
        elif strategy == "conflict_resolution":
            # Get articles with conflicts that need resolution
            conflicts = db.session.query(ConflictResolution.article_id).filter(
                ConflictResolution.project_id == project_id,
                ConflictResolution.resolved_at.is_(None)
            ).subquery()
            
            articles = db.session.query(Article).filter(
                Article.id.in_(conflicts)
            ).limit(count).all()
            
        elif strategy == "uncertain":
            # Get articles marked as uncertain by other reviewers
            uncertain_decisions = db.session.query(ReviewerDecision.article_id).filter(
                ReviewerDecision.project_id == project_id,
                ReviewerDecision.decision == DecisionStatus.UNCERTAIN,
                ReviewerDecision.reviewer_id != reviewer_id
            ).subquery()
            
            articles = db.session.query(Article).filter(
                Article.id.in_(uncertain_decisions)
            ).limit(count).all()
            
        else:
            raise ValueError(f"Unknown assignment strategy: {strategy}")
        
        return articles
    
    def record_reviewer_decision(self, article_id: str, reviewer_id: str, decision: DecisionStatus, 
                               reasoning: str = None, confidence: int = 3, time_spent: int = None,
                               llm_suggestion: Dict = None) -> ReviewerDecision:
        """Record a reviewer's decision on an article."""
        
        # Get article and validate
        article = Article.query.get(article_id)
        if not article:
            raise ValidationError(f"Article {article_id} not found")
        
        # Check if decision already exists
        existing = ReviewerDecision.query.filter_by(
            article_id=article_id,
            reviewer_id=reviewer_id
        ).first()
        
        if existing:
            # Update existing decision
            existing.decision = decision
            existing.reasoning = reasoning
            existing.confidence_level = confidence
            existing.time_spent_seconds = time_spent
            existing.updated_at = datetime.utcnow()
            
            if llm_suggestion:
                existing.used_llm_suggestion = True
                existing.llm_suggestion_decision = llm_suggestion.get('decision')
                existing.llm_agreement = (decision.value == llm_suggestion.get('decision'))
            
            db.session.commit()
            decision_record = existing
        else:
            # Create new decision
            decision_record = ReviewerDecision(
                id=self._generate_id(),
                article_id=article_id,
                reviewer_id=reviewer_id,
                project_id=article.project_id,
                decision=decision,
                reasoning=reasoning,
                confidence_level=confidence,
                time_spent_seconds=time_spent
            )
            
            if llm_suggestion:
                decision_record.used_llm_suggestion = True
                decision_record.llm_suggestion_decision = llm_suggestion.get('decision')
                decision_record.llm_agreement = (decision.value == llm_suggestion.get('decision'))
            
            db.session.add(decision_record)
            db.session.commit()
        
        # Check for conflicts with other reviewers
        self._check_and_create_conflicts(article_id, article.project_id)
        
        logger.info(f"Recorded decision: {decision.value} for article {article_id} by reviewer {reviewer_id}")
        return decision_record
    
    def _check_and_create_conflicts(self, article_id: str, project_id: str):
        """Check for conflicts and create conflict records."""
        # Get all decisions for this article
        decisions = ReviewerDecision.query.filter_by(
            article_id=article_id,
            project_id=project_id
        ).all()
        
        if len(decisions) < 2:
            return  # Need at least 2 decisions to have conflict
        
        # Check for disagreements
        decision_values = [d.decision for d in decisions]
        unique_decisions = set(decision_values)
        
        if len(unique_decisions) > 1 and DecisionStatus.UNCERTAIN not in unique_decisions:
            # We have a conflict (excluding uncertain decisions)
            include_exclude_conflict = (
                DecisionStatus.INCLUDE in unique_decisions and 
                DecisionStatus.EXCLUDE in unique_decisions
            )
            
            if include_exclude_conflict:
                # Check if conflict resolution already exists
                existing_conflict = ConflictResolution.query.filter_by(
                    article_id=article_id,
                    project_id=project_id,
                    resolved_at=None
                ).first()
                
                if not existing_conflict:
                    # Create conflict resolution record
                    conflict = ConflictResolution(
                        id=self._generate_id(),
                        article_id=article_id,
                        project_id=project_id,
                        decision1_id=decisions[0].id,
                        decision2_id=decisions[1].id
                    )
                    
                    db.session.add(conflict)
                    
                    # Update article status to conflict
                    article = Article.query.get(article_id)
                    article.status = 'conflict'
                    
                    db.session.commit()
                    
                    logger.info(f"Created conflict resolution for article {article_id}")
    
    def resolve_conflict(self, conflict_id: str, resolver_id: str, final_decision: DecisionStatus, 
                        reasoning: str, method: str = "manual", llm_suggestion: Dict = None) -> ConflictResolution:
        """Resolve a conflict between reviewers."""
        conflict = ConflictResolution.query.get(conflict_id)
        if not conflict:
            raise ValidationError(f"Conflict {conflict_id} not found")
        
        if conflict.resolved_at:
            raise ValidationError("Conflict already resolved")
        
        # Update conflict resolution
        conflict.resolved_by = resolver_id
        conflict.final_decision = final_decision
        conflict.resolution_reasoning = reasoning
        conflict.resolution_method = method
        conflict.resolved_at = datetime.utcnow()
        
        if llm_suggestion:
            conflict.llm_resolution_suggestion = llm_suggestion
        
        # Update article status
        article = Article.query.get(conflict.article_id)
        article.status = final_decision.value
        
        db.session.commit()
        
        logger.info(f"Resolved conflict {conflict_id} with decision: {final_decision.value}")
        return conflict
    
    def calculate_inter_rater_reliability(self, project_id: str, reviewer1_id: str, reviewer2_id: str) -> InterRaterReliability:
        """Calculate inter-rater reliability between two reviewers."""
        # Get overlapping decisions
        decisions1 = db.session.query(ReviewerDecision).filter_by(
            project_id=project_id,
            reviewer_id=reviewer1_id
        ).all()
        
        decisions2 = db.session.query(ReviewerDecision).filter_by(
            project_id=project_id,
            reviewer_id=reviewer2_id
        ).all()
        
        # Find overlapping articles
        articles1 = {d.article_id: d.decision for d in decisions1}
        articles2 = {d.article_id: d.decision for d in decisions2}
        
        overlapping_articles = set(articles1.keys()) & set(articles2.keys())
        
        if len(overlapping_articles) == 0:
            return InterRaterReliability(
                reviewer1_id=reviewer1_id,
                reviewer2_id=reviewer2_id,
                total_overlapping_decisions=0,
                agreements=0,
                disagreements=0,
                agreement_percentage=0.0,
                cohens_kappa=0.0,
                specific_agreements={}
            )
        
        # Calculate agreements
        agreements = 0
        specific_agreements = {}
        
        for article_id in overlapping_articles:
            decision1 = articles1[article_id]
            decision2 = articles2[article_id]
            
            if decision1 == decision2:
                agreements += 1
                decision_key = decision1.value
                specific_agreements[decision_key] = specific_agreements.get(decision_key, 0) + 1
        
        disagreements = len(overlapping_articles) - agreements
        agreement_percentage = (agreements / len(overlapping_articles)) * 100
        
        # Calculate Cohen's Kappa (simplified version)
        cohens_kappa = self._calculate_cohens_kappa(overlapping_articles, articles1, articles2)
        
        return InterRaterReliability(
            reviewer1_id=reviewer1_id,
            reviewer2_id=reviewer2_id,
            total_overlapping_decisions=len(overlapping_articles),
            agreements=agreements,
            disagreements=disagreements,
            agreement_percentage=agreement_percentage,
            cohens_kappa=cohens_kappa,
            specific_agreements=specific_agreements
        )
    
    def _calculate_cohens_kappa(self, overlapping_articles: set, decisions1: Dict, decisions2: Dict) -> float:
        """Calculate Cohen's Kappa coefficient."""
        # Simplified calculation - in practice, you'd want a more robust implementation
        try:
            from sklearn.metrics import cohen_kappa_score
            
            labels1 = [decisions1[aid].value for aid in overlapping_articles]
            labels2 = [decisions2[aid].value for aid in overlapping_articles]
            
            return cohen_kappa_score(labels1, labels2)
        except ImportError:
            # Fallback to simple agreement rate if sklearn not available
            agreements = sum(1 for aid in overlapping_articles if decisions1[aid] == decisions2[aid])
            return agreements / len(overlapping_articles) if overlapping_articles else 0.0
    
    def get_project_collaboration_stats(self, project_id: str) -> Dict:
        """Get collaboration statistics for a project."""
        # Get all reviewers for project
        reviewers = db.session.query(Reviewer).join(ProjectReviewer).filter(
            ProjectReviewer.project_id == project_id,
            ProjectReviewer.is_active == True
        ).all()
        
        # Get decision counts
        decision_stats = {}
        for reviewer in reviewers:
            decisions = ReviewerDecision.query.filter_by(
                project_id=project_id,
                reviewer_id=reviewer.id
            ).all()
            
            decision_stats[reviewer.id] = {
                'reviewer_name': reviewer.name,
                'total_decisions': len(decisions),
                'decisions_by_type': {
                    'include': len([d for d in decisions if d.decision == DecisionStatus.INCLUDE]),
                    'exclude': len([d for d in decisions if d.decision == DecisionStatus.EXCLUDE]),
                    'uncertain': len([d for d in decisions if d.decision == DecisionStatus.UNCERTAIN])
                },
                'avg_confidence': sum(d.confidence_level for d in decisions) / len(decisions) if decisions else 0,
                'avg_time_per_decision': sum(d.time_spent_seconds or 0 for d in decisions) / len(decisions) if decisions else 0
            }
        
        # Get conflict stats
        total_conflicts = ConflictResolution.query.filter_by(project_id=project_id).count()
        resolved_conflicts = ConflictResolution.query.filter(
            ConflictResolution.project_id == project_id,
            ConflictResolution.resolved_at.isnot(None)
        ).count()
        
        return {
            'total_reviewers': len(reviewers),
            'reviewer_stats': decision_stats,
            'conflicts': {
                'total': total_conflicts,
                'resolved': resolved_conflicts,
                'pending': total_conflicts - resolved_conflicts
            }
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID for database records."""
        import uuid
        return str(uuid.uuid4())

# Global manager instance
collaborative_manager = CollaborativeScreeningManager()