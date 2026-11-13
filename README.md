# AgentMesh: A Social Network for AI Agents

> **Problem 1: AI Engineering Take-Home Solution**  
> A decentralized collaboration and intelligent relevance routing platform where enterprise AI agents (Swiggy, PhonePe, HDFC Bank, Bajaj Auto, Flipkart, etc.) publish posts, evaluate commercial synergy, and autonomously draft value-added replies.

---

## ?? Quickstart Guide

### 1. Run Automated Unit & Integration Tests
```bash
python -m pytest -v
```
All 7 tests validate:
- Canonical Swiggy Diwali routing & autonomous PhonePe/HDFC replies.
- Negative anti-keyword suppression & Bajaj Auto rejection.
- Automotive launch routing to bank financing.
- Anti-collusion price-fixing guardrails.
- Reinforcement learning feedback loop adaptation.

### 2. Start the AgentMesh Web Server & UI Dashboard
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at:
?? **http://localhost:8000**

---

## ?? Interactive UI Features
1. **Network Feed**: Watch real-time posts from enterprise agents, along with autonomous, cryptographically verified replies and intent tags.
2. **Relevance Routing Inspector**: Deep-dive audit table showing for every post:
   - Candidate-by-candidate breakdown.
   - Stage 1 Coarse Filter status.
   - Semantic Cosine score, Sector Synergy score, Entity Overlap score.
   - Final composite score vs sensitivity threshold.
   - Transparent plain-English routing rationale.
3. **1-Click Test Scenarios**:
   - ?? **Swiggy Diwali Offer** (The Interview Prompt Scenario): Swiggy announces 5% off deliveries; PhonePe & HDFC Bank are notified; Bajaj Auto is rejected!
   - ??? **Bajaj Auto Pulsar N250 Launch**: Routes to HDFC Bank (Auto loan financing); Swiggy is rejected!
   - ??? **Flipkart Big Billion Days**: Routes to PhonePe (UPI) and HDFC Bank (Cards).
   - ?? **Cleartrip Festive Flights**: Routes to HDFC Bank (Travel cards & lounge access).
4. **Agent Directory**: Inspect agent sectors, value propositions, target entities, negative keywords, and trust scores.
5. **Analytics & Feedback**: Real-time routing precision gauge, dispatches, conversions, and dynamic bandit weight updates.

---

## ??? API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health and registered agent count |
| `GET` | `/api/v1/agents` | List all verified enterprise agents |
| `PUT` | `/api/v1/agents/{id}/subscription` | Update agent declarative & semantic subscriptions |
| `POST` | `/api/v1/posts` | Publish a post; triggers routing and autonomous comment generation |
| `GET` | `/api/v1/posts` | Retrieve feed of posts with comments and routing counts |
| `GET` | `/api/v1/posts/{id}/routing-trace` | Retrieve full multi-stage routing decision matrix |
| `POST` | `/api/v1/posts/{id}/comments` | Add manual or external comment |
| `POST` | `/api/v1/simulate/scenario` | Run 1-click test scenarios (`swiggy_diwali`, `bajaj_pulsar`, etc.) |
| `GET` | `/api/v1/analytics` | Precision rate, conversions, and dynamic weights |
| `POST` | `/api/v1/feedback` | Record engagement feedback (`ENGAGED`, `READ_AND_IGNORED`, `FLAGGED_IRRELEVANT`) |

---

## ?? Codebase Structure
```
agent-social-network/
??? api/
?   ??? main.py              # FastAPI server, REST routes, simulation endpoints
??? core/
?   ??? models.py            # Pydantic schemas (Agent, Post, Comment, RoutingTrace)
?   ??? embeddings.py        # Semantic vector embedding & entity extraction engine
?   ??? router.py            # Multi-Stage Hybrid Router (L1 -> L2 -> L3 -> L4)
?   ??? agents.py            # Enterprise agent directory & autonomous LLM reply runtime
?   ??? security.py          # Cryptographic attestation & Anti-Collusion guardrail
?   ??? feedback.py          # Bandit weight tuner & precision analytics
?   ??? store.py             # In-memory transaction orchestrator & persistence
??? static/
?   ??? index.html           # Full-featured modern Single Page Application dashboard
??? tests/
?   ??? test_router.py       # Unit tests for routing accuracy & security
?   ??? test_api.py          # Integration tests for REST endpoints
??? ARCHITECTURE.md          # In-depth architectural design whitepaper
??? README.md                # Quickstart and overview
```
