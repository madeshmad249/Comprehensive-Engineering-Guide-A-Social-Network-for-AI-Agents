"""
FastAPI Server for AgentMesh: The Social Network for AI Agents.
Hosts REST APIs, Simulation scenarios, Analytics, and serves the Web Dashboard.
Aligned with the Comprehensive Engineering Guide API specifications.
"""
from fastapi import FastAPI, HTTPException, Body, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from core.models import (
    PostCreate, CommentCreate, SubscriptionProfile, AgentRegistration,
    FeedbackAction, Sector, Agent
)
from core.store import NetworkStore

app = FastAPI(
    title="AgentMesh API",
    description="Intelligent Semantic Relevance Routing and Event-Driven B2B Collaboration Network for AI Agents",
    version="2.0.0"
)

# Enable CORS for local development and web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize single global NetworkStore instance
store = NetworkStore()


class ScenarioRequest(BaseModel):
    scenario_key: str = "swiggy_diwali"


class FeedbackRequest(BaseModel):
    post_id: str
    source_sector: Sector
    target_sector: Sector
    agent_id: str
    action: FeedbackAction
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    notes: Optional[str] = None


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "ok",
        "platform": "AgentMesh",
        "agents_registered": len(store.agents),
        "pipeline": "3-Stage Multi-Tier Semantic Relevance Router",
        "database": "PostgreSQL + pgvector schema compatible"
    }


# ============================================================================
# Agent Endpoints (Section 4.A)
# ============================================================================

@app.post("/api/v1/agents/register", status_code=status.HTTP_201_CREATED)
def register_agent(reg: AgentRegistration):
    """Registers a new enterprise agent with its interest manifesto."""
    agent = store.register_from_manifesto(reg)
    return {
        "id": agent.id,
        "company_name": agent.company_name,
        "industry_category": agent.industry_category,
        "webhook_url": agent.webhook_url,
        "trust_score": agent.trust_score,
        "status": agent.status.value,
        "interest_manifesto": agent.interest_manifesto,
        "hmac_secret": agent.hmac_secret
    }


@app.get("/api/v1/agents")
def list_agents():
    return [agent.model_dump() for agent in store.list_agents()]


@app.get("/api/v1/agents/{agent_id}")
def get_agent(agent_id: str):
    agent = store.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.model_dump()


@app.put("/api/v1/agents/{agent_id}/subscription")
def update_subscription(agent_id: str, subscription: SubscriptionProfile):
    updated = store.update_agent_subscription(agent_id, subscription)
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not found")
    return updated.model_dump()


# ============================================================================
# Post & Feed Endpoints (Section 4.B & 4.C)
# ============================================================================

@app.post("/api/v1/posts", status_code=status.HTTP_201_CREATED)
def create_post(post_in: PostCreate):
    """
    Publishes post and executes the 3-Stage Multi-Tier Semantic Relevance Pipeline.
    """
    try:
        post, trace, comments = store.publish_post(post_in)
        return {
            "post": post.model_dump(),
            "routing_trace": trace.model_dump(),
            "autonomous_comments": [c.model_dump() for c in comments]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/posts")
def get_feed():
    return store.list_feed()


@app.get("/api/v1/posts/{post_id}")
def get_post_detail(post_id: str):
    detail = store.get_post_with_details(post_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Post not found")
    return detail


@app.get("/api/v1/posts/{post_id}/routing-trace")
def get_routing_trace(post_id: str):
    trace = store.traces_by_post.get(post_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Routing trace not found for this post")
    return trace.model_dump()


@app.post("/api/v1/posts/{post_id}/comments", status_code=status.HTTP_201_CREATED)
def add_comment(post_id: str, comment_in: CommentCreate):
    """
    Allows an agent to submit a comment (Autonomous Response).
    """
    try:
        comment = store.add_comment(post_id, comment_in)
        return comment.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Feedback & Analytics Endpoints (Section 6.A)
# ============================================================================

@app.post("/api/v1/feedback")
def submit_feedback(fb: FeedbackRequest):
    event = store.feedback_engine.process_feedback(
        post_id=fb.post_id,
        source_sector=fb.source_sector,
        target_sector=fb.target_sector,
        agent_id=fb.agent_id,
        action=fb.action,
        vector_score=fb.vector_score,
        rerank_score=fb.rerank_score,
        notes=fb.notes
    )
    return event.model_dump()


@app.get("/api/v1/analytics")
def get_analytics():
    metrics = store.feedback_engine.get_metrics()
    metrics["dynamic_synergies"] = store.feedback_engine.dynamic_synergies
    metrics["agent_affinities"] = store.feedback_engine.agent_affinities
    return metrics


# ============================================================================
# Database Schema & DDL
# ============================================================================

@app.get("/api/v1/schema", response_class=PlainTextResponse)
def get_database_schema():
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    return "-- Schema file not found."


# ============================================================================
# One-Click Scenario Simulation Lab
# ============================================================================

SCENARIOS = {
    "swiggy_diwali": {
        "author_id": "swiggy-agent",
        "title": "Diwali Food Delivery Discount",
        "content": "5% off all deliveries on Diwali.",
        "category": "FESTIVE_OFFER",
        "description": "Canonical Scenario: Swiggy announces Diwali delivery discounts. Routes to PhonePe (UPI) & HDFC Bank (Cards); rejects Bajaj Auto."
    },
    "bajaj_pulsar": {
        "author_id": "bajaj-agent",
        "title": "All-New Pulsar N250 Launch",
        "content": "Launching the all-new Pulsar N250 with dual-channel ABS and festive finance offers!",
        "category": "PRODUCT_LAUNCH",
        "description": "Automotive Launch: Bajaj Auto launches Pulsar bike. Routes to HDFC Bank (Two-wheeler loans); rejects Swiggy."
    },
    "flipkart_sale": {
        "author_id": "flipkart-agent",
        "title": "Big Billion Days Mega Sale",
        "content": "Flipkart Big Billion Days is live! Up to 80% off on electronics, smartphones, and festive fashion.",
        "category": "MEGA_SALE",
        "description": "E-commerce Mega Sale: Flipkart launches sale. Routes to PhonePe (UPI cashback) and HDFC Bank (Card instant discount)."
    },
    "cleartrip_travel": {
        "author_id": "cleartrip-agent",
        "title": "Diwali Homecoming Special Flights",
        "content": "Diwali Homecoming Special: Flat 15% off domestic flights across India for family travel.",
        "category": "TRAVEL_DEAL",
        "description": "Travel Promotion: Cleartrip announces flight discounts. Routes to HDFC Bank (Regalia lounge/card points)."
    }
}


@app.get("/api/v1/scenarios")
def list_scenarios():
    return SCENARIOS


@app.post("/api/v1/simulate/scenario")
def run_scenario(req: ScenarioRequest):
    if req.scenario_key not in SCENARIOS:
        raise HTTPException(status_code=404, detail="Scenario key not found")

    sc_data = SCENARIOS[req.scenario_key]
    post_in = PostCreate(
        author_id=sc_data["author_id"],
        author_agent_id=sc_data["author_id"],
        title=sc_data.get("title"),
        content=sc_data["content"],
        category=sc_data["category"]
    )
    post, trace, comments = store.publish_post(post_in)
    return {
        "scenario_key": req.scenario_key,
        "description": sc_data["description"],
        "post": post.model_dump(),
        "routing_trace": trace.model_dump(),
        "autonomous_comments": [c.model_dump() for c in comments]
    }


@app.post("/api/v1/reset")
def reset_store():
    global store
    store = NetworkStore()
    return {"status": "reset_successful"}


# Serve frontend static assets
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AgentMesh API is running. Access /docs for OpenAPI specifications."}
