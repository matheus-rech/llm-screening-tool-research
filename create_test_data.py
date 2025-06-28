import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app import create_app, db
from app.models.screening_models import Project, Article

def create_test_project_with_data():
    """Create a test project with sample articles and dual-LLM screening results."""
    
    app = create_app()
    
    with app.app_context():
        project = Project(
            name="Test Dual-LLM Comparison",
            description="Test project for dual-LLM comparison export functionality",
            created_at=datetime.now()
        )
        db.session.add(project)
        db.session.commit()
        
        print(f"Created project: {project.name} (ID: {project.id})")
        
        sample_decision_reasoning = {
            "openai_result": {
                "picott_extraction": {
                    "population": ["adults with type 2 diabetes", "newly diagnosed patients"],
                    "intervention": ["metformin therapy", "oral medication"],
                    "comparison": ["placebo", "standard care"],
                    "outcomes": ["HbA1c reduction", "glycemic control"],
                    "time_frame": "24 months",
                    "study_type": "randomized controlled trial"
                },
                "criteria_evaluation": {
                    "meets_inclusion_criteria": True,
                    "inclusion_reasoning": "Study population matches target demographics and intervention is relevant",
                    "violates_exclusion_criteria": False,
                    "exclusion_reasoning": "No exclusion criteria violations identified"
                },
                "research_relevance": {
                    "relevance_score": 0.95,
                    "relevance_reasoning": "Highly relevant to diabetes management research"
                },
                "screening_decision": {
                    "final_decision": "include",
                    "confidence_score": 0.92,
                    "detailed_reasoning": "Strong match with PICO-TT criteria, high-quality RCT design"
                }
            },
            "anthropic_result": {
                "picott_extraction": {
                    "population": ["type 2 diabetes patients", "adult population"],
                    "intervention": ["metformin treatment"],
                    "comparison": ["placebo control"],
                    "outcomes": ["HbA1c levels", "metabolic outcomes"],
                    "time_frame": "2-year follow-up",
                    "study_type": "RCT"
                },
                "criteria_evaluation": {
                    "meets_inclusion_criteria": True,
                    "inclusion_reasoning": "Population and intervention align with inclusion criteria",
                    "violates_exclusion_criteria": False,
                    "exclusion_reasoning": "No exclusion criteria met"
                },
                "research_relevance": {
                    "relevance_score": 0.88,
                    "relevance_reasoning": "Relevant to systematic review objectives"
                },
                "screening_decision": {
                    "final_decision": "include",
                    "confidence_score": 0.89,
                    "detailed_reasoning": "Meets inclusion criteria with good study design"
                }
            },
            "agreement_analysis": {
                "decision_agreement": True,
                "confidence_difference": 0.03,
                "needs_human_review": False
            }
        }
        
        articles_data = [
            {
                "title": "Efficacy and safety of metformin in type 2 diabetes: A randomized controlled trial",
                "authors": "Johnson, A., Smith, B., Wilson, C.",
                "year": 2021,
                "abstract": "This randomized controlled trial investigates the long-term efficacy and safety of metformin compared to placebo in 500 adults with newly diagnosed type 2 diabetes over a 24-month period.",
                "status": "included",
                "decision_reasoning": json.dumps(sample_decision_reasoning)
            },
            {
                "title": "Short-term effects of GLP-1 agonists in pre-diabetic adults",
                "authors": "Chen, C., Davis, D., Miller, E.",
                "year": 2022,
                "abstract": "This pilot study examines the effects of GLP-1 agonists on glycemic control in adults over a 3-month period.",
                "status": "excluded",
                "decision_reasoning": json.dumps({
                    "openai_result": {
                        "picott_extraction": {
                            "population": ["pre-diabetic adults"],
                            "intervention": ["GLP-1 agonists"],
                            "comparison": ["not found"],
                            "outcomes": ["glycemic control"],
                            "time_frame": "3 months",
                            "study_type": "pilot study"
                        },
                        "criteria_evaluation": {
                            "meets_inclusion_criteria": False,
                            "inclusion_reasoning": "Population is pre-diabetic, not type 2 diabetes",
                            "violates_exclusion_criteria": True,
                            "exclusion_reasoning": "Study duration too short (3 months vs required 12+ months)"
                        },
                        "research_relevance": {
                            "relevance_score": 0.45,
                            "relevance_reasoning": "Limited relevance due to population and duration mismatch"
                        },
                        "screening_decision": {
                            "final_decision": "exclude",
                            "confidence_score": 0.85,
                            "detailed_reasoning": "Does not meet inclusion criteria for population and study duration"
                        }
                    },
                    "anthropic_result": {
                        "picott_extraction": {
                            "population": ["pre-diabetic individuals"],
                            "intervention": ["GLP-1 receptor agonists"],
                            "comparison": ["not found"],
                            "outcomes": ["glucose control"],
                            "time_frame": "3-month period",
                            "study_type": "pilot study"
                        },
                        "criteria_evaluation": {
                            "meets_inclusion_criteria": False,
                            "inclusion_reasoning": "Wrong population (pre-diabetic vs type 2 diabetes)",
                            "violates_exclusion_criteria": True,
                            "exclusion_reasoning": "Insufficient study duration"
                        },
                        "research_relevance": {
                            "relevance_score": 0.40,
                            "relevance_reasoning": "Poor match with review criteria"
                        },
                        "screening_decision": {
                            "final_decision": "exclude",
                            "confidence_score": 0.90,
                            "detailed_reasoning": "Clear exclusion based on population and duration criteria"
                        }
                    },
                    "agreement_analysis": {
                        "decision_agreement": True,
                        "confidence_difference": 0.05,
                        "needs_human_review": False
                    }
                })
            },
            {
                "title": "Conflicting evidence on insulin therapy effectiveness",
                "authors": "Brown, F., Taylor, G., Anderson, H.",
                "year": 2020,
                "abstract": "Mixed results study on insulin therapy in type 2 diabetes patients with conflicting outcomes.",
                "status": "human_review_required",
                "decision_reasoning": json.dumps({
                    "openai_result": {
                        "picott_extraction": {
                            "population": ["type 2 diabetes patients"],
                            "intervention": ["insulin therapy"],
                            "comparison": ["standard care"],
                            "outcomes": ["mixed results"],
                            "time_frame": "12 months",
                            "study_type": "clinical trial"
                        },
                        "criteria_evaluation": {
                            "meets_inclusion_criteria": True,
                            "inclusion_reasoning": "Population and intervention match criteria",
                            "violates_exclusion_criteria": False,
                            "exclusion_reasoning": "No clear exclusion criteria violations"
                        },
                        "research_relevance": {
                            "relevance_score": 0.70,
                            "relevance_reasoning": "Relevant but with methodological concerns"
                        },
                        "screening_decision": {
                            "final_decision": "include",
                            "confidence_score": 0.65,
                            "detailed_reasoning": "Meets basic criteria but has quality concerns"
                        }
                    },
                    "anthropic_result": {
                        "picott_extraction": {
                            "population": ["diabetes patients"],
                            "intervention": ["insulin treatment"],
                            "comparison": ["control group"],
                            "outcomes": ["conflicting outcomes"],
                            "time_frame": "1 year",
                            "study_type": "trial"
                        },
                        "criteria_evaluation": {
                            "meets_inclusion_criteria": False,
                            "inclusion_reasoning": "Methodological issues affect reliability",
                            "violates_exclusion_criteria": True,
                            "exclusion_reasoning": "Poor study quality and conflicting results"
                        },
                        "research_relevance": {
                            "relevance_score": 0.55,
                            "relevance_reasoning": "Questionable reliability due to conflicting results"
                        },
                        "screening_decision": {
                            "final_decision": "exclude",
                            "confidence_score": 0.75,
                            "detailed_reasoning": "Quality concerns outweigh potential relevance"
                        }
                    },
                    "agreement_analysis": {
                        "decision_agreement": False,
                        "confidence_difference": 0.10,
                        "needs_human_review": True
                    }
                })
            }
        ]
        
        for article_data in articles_data:
            article = Article(
                project_id=project.id,
                title=article_data["title"],
                authors=article_data["authors"],
                year=article_data["year"],
                abstract=article_data["abstract"],
                status=article_data["status"],
                decision_reasoning=article_data["decision_reasoning"]
            )
            db.session.add(article)
        
        db.session.commit()
        print(f"Created {len(articles_data)} test articles with dual-LLM screening results")
        
        return project.id

if __name__ == "__main__":
    project_id = create_test_project_with_data()
    print(f"Test data created successfully! Project ID: {project_id}")
    print(f"You can now test the export functionality at: http://localhost:5000/project/{project_id}/export/dual-llm-comparison")
