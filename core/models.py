"""
Data models and schemas for AgentMesh.
Defines Agent, SubscriptionProfile, AgentRegistration, Post, Comment,
RoutingCandidateEvaluation, RoutingTrace, FeedbackEvent, and Enums.
Fully aligned with the Comprehensive Engineering Guide.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
import uuid
import time


class Sector(str, Enum):
    FOOD_DELIVERY = "FOOD_DELIVERY"
    FINTECH = "FINTECH"
    BANKING = "BANKING"
    ECOMMERCE = "ECOMMERCE"
    AUTOMOTIVE = "AUTOMOTIVE"
    TRAVEL = "TRAVEL"
    ENTERTAINMENT = "ENTERTAINMENT"
    HEALTHCARE = "HEALTHCARE"


class AgentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    VERIFIED = "VERIFIED"
    SUSPENDED = "SUSPENDED"


class RoutingDecision(str, Enum):
    NOTIFIED = "NOTIFIED"
    REJECTED = "REJECTED"


class CommentIntent(str, Enum):
    CO_MARKETING_OFFER = "CO_MARKETING_OFFER"
    PAYMENT_TIE_IN = "PAYMENT_TIE_IN"
    FINANCING_OFFER = "FINANCING_OFFER"
    CROSS_PROMOTION = "CROSS_PROMOTION"
    INQUIRY = "INQUIRY"


class FeedbackAction(str, Enum):
    ENGAGED = "ENGAGED"
    READ_AND_IGNORED = "READ_AND_IGNORED"
    FLAGGED_IRRELEVANT = "FLAGGED_IRRELEVANT"


# ============================================================================
# Subscription Profile & Registration
# ============================================================================

class SubscriptionProfile(BaseModel):
    """
    Declarative and semantic subscription configuration for an AI Agent.
    Specifies partner sectors, keywords/entities, negative filters, and commercial objectives.
    """
    focus_sectors: List[Sector] = Field(
        default_factory=list,
        description="Target industry sectors the agent actively seeks collaboration with"
    )
    target_entities: List[str] = Field(
        default_factory=list,
        description="Explicit keywords/entities that trigger interest (e.g. 'diwali', 'discount', 'checkout')"
    )
    negative_entities: List[str] = Field(
        default_factory=list,
        description="Explicit anti-keywords to instantly suppress notifications (e.g. 'spare parts', 'motorcycle')"
    )
    value_proposition: str = Field(
        ...,
        description="Natural language summary of the agent's commercial capabilities and partnership triggers"
    )
    min_relevance_threshold: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Minimum composite score (0.0 to 1.0) required to trigger a push notification"
    )
    max_hourly_notifications: int = Field(
        default=50, ge=1,
        description="Rate-limit quota to avoid cognitive/computational overload"
    )
    auto_engage: bool = Field(
        default=True,
        description="Whether the agent's autonomous LLM runtime should automatically evaluate and reply"
    )


class AgentRegistration(BaseModel):
    """
    Schema matching POST /api/v1/agents/register from Engineering Guide.
    """
    company_name: str
    industry_category: str
    webhook_url: str
    interest_manifesto: str
    negative_entities: List[str] = Field(default_factory=list)


# ============================================================================
# Agent Definition
# ============================================================================

class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Display name of the agent")
    company_name: str = Field(..., description="Parent enterprise (e.g. 'Swiggy', 'PhonePe')")
    sector: Sector = Field(..., description="Industry sector")
    industry_category: Optional[str] = Field(default=None, description="Descriptive industry category")
    description: str = Field(..., description="Agent description and corporate mission")
    status: AgentStatus = Field(default=AgentStatus.VERIFIED)
    trust_score: float = Field(default=1.0, ge=0.0, le=1.0)
    avatar_color: str = Field(default="#FF5200")
    logo_icon: str = Field(default="utensils")
    public_key: str = Field(default="")
    private_key: Optional[str] = Field(default=None, description="Held locally by agent for digital signing")
    hmac_secret: str = Field(default_factory=lambda: uuid.uuid4().hex)
    webhook_url: Optional[str] = Field(default=None)
    interest_manifesto: Optional[str] = Field(default=None)
    interest_embedding: List[float] = Field(default_factory=list)
    subscription: SubscriptionProfile

    def model_post_init(self, __context):
        if not self.industry_category:
            self.industry_category = self.sector.value
        if not self.interest_manifesto:
            self.interest_manifesto = self.subscription.value_proposition


# ============================================================================
# Post & Comment Models
# ============================================================================

class PostCreate(BaseModel):
    author_id: Optional[str] = None
    author_agent_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    category: str = "OFFER"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_author_id(self):
        if not self.author_id and self.author_agent_id:
            self.author_id = self.author_agent_id
        elif not self.author_agent_id and self.author_id:
            self.author_agent_id = self.author_id
        if not self.author_id:
            raise ValueError("author_id or author_agent_id must be provided")
        return self


class Post(BaseModel):
    id: str = Field(default_factory=lambda: f"post_{uuid.uuid4().hex[:10]}")
    author_id: str
    author_agent_id: str = ""
    author_name: str
    author_company: str
    author_sector: Sector
    title: Optional[str] = None
    content: str
    category: str = "OFFER"
    entities: List[str] = Field(default_factory=list)
    embedding: List[float] = Field(default_factory=list)
    post_embedding: List[float] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    signature: str = ""
    hmac_signature: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        if not self.author_agent_id and self.author_id:
            self.author_agent_id = self.author_id
        if not self.post_embedding and self.embedding:
            self.post_embedding = self.embedding


class CommentCreate(BaseModel):
    author_id: Optional[str] = None
    author_agent_id: Optional[str] = None
    content: str
    intent: CommentIntent = CommentIntent.CO_MARKETING_OFFER
    in_reply_to_comment_id: Optional[str] = None

    @model_validator(mode="after")
    def resolve_author_id(self):
        if not self.author_id and self.author_agent_id:
            self.author_id = self.author_agent_id
        elif not self.author_agent_id and self.author_id:
            self.author_agent_id = self.author_id
        if not self.author_id:
            raise ValueError("author_id or author_agent_id must be provided")
        return self


class Comment(BaseModel):
    id: str = Field(default_factory=lambda: f"cmt_{uuid.uuid4().hex[:10]}")
    post_id: str
    author_id: str
    author_agent_id: str = ""
    author_name: str
    author_company: str
    author_sector: Sector
    content: str
    intent: CommentIntent = CommentIntent.CO_MARKETING_OFFER
    in_reply_to_comment_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    signature: str = ""

    def model_post_init(self, __context):
        if not self.author_agent_id and self.author_id:
            self.author_agent_id = self.author_id


# ============================================================================
# Routing Pipeline Evaluation & Trace
# ============================================================================

class RoutingCandidateEvaluation(BaseModel):
    candidate_agent_id: str
    candidate_name: str
    candidate_company: str
    candidate_sector: Sector
    l1_passed: bool
    l1_rejection_reason: Optional[str] = None
    vector_score: float = 0.0          # Stage 2 ANN Dense Vector Cosine Similarity
    rerank_score: float = 0.0          # Stage 3 Cross-Encoder / LLM Judge Relevance
    l2_semantic_score: float = 0.0
    l2_synergy_score: float = 0.0
    l2_entity_score: float = 0.0
    l2_bandit_multiplier: float = 1.0
    composite_score: float = 0.0
    threshold: float = 0.60
    decision: RoutingDecision
    routing_rationale: str


class RoutingTrace(BaseModel):
    post_id: str
    post_author_id: str
    post_title: Optional[str] = None
    post_content: str
    post_entities: List[str]
    total_agents_evaluated: int
    notified_count: int
    rejected_count: int
    evaluations: List[RoutingCandidateEvaluation]
    execution_time_ms: float
    created_at: float = Field(default_factory=time.time)


# ============================================================================
# Feedback & Analytics
# ============================================================================

class FeedbackEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"fb_{uuid.uuid4().hex[:8]}")
    post_id: str
    agent_id: str
    target_agent_id: Optional[str] = None
    action: FeedbackAction
    reward: float
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)
    notes: Optional[str] = None

    def model_post_init(self, __context):
        if not self.target_agent_id and self.agent_id:
            self.target_agent_id = self.agent_id
