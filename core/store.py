"""
Central State and Orchestration Store for AgentMesh.
Maintains in-memory state with transaction safety, pre-seeded Indian enterprise agents,
and orchestrates the end-to-end Publish -> Route -> Respond -> Feedback pipeline.
Aligned with the Comprehensive Engineering Guide.
"""
import uuid
import time
from typing import Dict, List, Optional, Tuple
from core.models import (
    Agent, Post, Comment, PostCreate, CommentCreate, AgentRegistration,
    RoutingTrace, FeedbackAction, Sector, AgentStatus, SubscriptionProfile
)
from core.agents import get_default_agents, AgentRuntime, generate_simulated_keypair
from core.router import RelevanceRouter, get_sector_synergy
from core.security import TrustAndSafetyGateway, generate_hmac_signature
from core.feedback import FeedbackEngine
from core.embeddings import compute_embedding, extract_entities


class NetworkStore:
    def __init__(self):
        # 1. State dictionaries
        self.agents: Dict[str, Agent] = {}
        self.posts: List[Post] = []
        self.comments_by_post: Dict[str, List[Comment]] = {}
        self.traces_by_post: Dict[str, RoutingTrace] = {}

        # 2. Engines & Subsystems (pass get_sector_synergy resolver to anchor feedback)
        self.feedback_engine = FeedbackEngine(base_synergy_resolver=get_sector_synergy)
        self.router = RelevanceRouter(feedback_engine=self.feedback_engine)
        self.security = TrustAndSafetyGateway()
        
        # 3. Seed default enterprise agents
        for agent in get_default_agents():
            self.agents[agent.id] = agent

        # 4. Agent Autonomous Simulation Runtime
        self.runtime = AgentRuntime(self.agents)

    def register_agent(self, agent: Agent) -> Agent:
        self.agents[agent.id] = agent
        return agent

    def register_from_manifesto(self, reg: AgentRegistration) -> Agent:
        """Registers a new agent matching POST /api/v1/agents/register from Engineering Guide."""
        cat_lower = reg.industry_category.lower()
        if "food" in cat_lower or "delivery" in cat_lower:
            sector = Sector.FOOD_DELIVERY
        elif "bank" in cat_lower or "credit" in cat_lower or "card" in cat_lower:
            sector = Sector.BANKING
        elif "pay" in cat_lower or "fintech" in cat_lower:
            sector = Sector.FINTECH
        elif "commerce" in cat_lower or "retail" in cat_lower or "shopping" in cat_lower:
            sector = Sector.ECOMMERCE
        elif "auto" in cat_lower or "vehicle" in cat_lower or "mobility" in cat_lower:
            sector = Sector.AUTOMOTIVE
        elif "travel" in cat_lower or "flight" in cat_lower:
            sector = Sector.TRAVEL
        else:
            sector = Sector.FINTECH

        extracted_targets = extract_entities(reg.interest_manifesto)
        emb = compute_embedding(reg.interest_manifesto)
        agent_id = str(uuid.uuid4())
        pub, priv = generate_simulated_keypair(agent_id)

        sub = SubscriptionProfile(
            focus_sectors=[],
            target_entities=extracted_targets,
            negative_entities=reg.negative_entities,
            value_proposition=reg.interest_manifesto,
            min_relevance_threshold=0.60,
            auto_engage=True
        )

        agent = Agent(
            id=agent_id,
            name=f"{reg.company_name} Agent",
            company_name=reg.company_name,
            sector=sector,
            industry_category=reg.industry_category,
            description=reg.interest_manifesto,
            status=AgentStatus.VERIFIED,
            trust_score=1.0,
            public_key=pub,
            private_key=priv,
            webhook_url=reg.webhook_url,
            interest_manifesto=reg.interest_manifesto,
            interest_embedding=emb,
            subscription=sub
        )
        self.register_agent(agent)
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Agent]:
        return list(self.agents.values())

    def update_agent_subscription(self, agent_id: str, new_subscription) -> Optional[Agent]:
        if agent_id in self.agents:
            self.agents[agent_id].subscription = new_subscription
            self.agents[agent_id].interest_manifesto = new_subscription.value_proposition
            self.agents[agent_id].interest_embedding = compute_embedding(new_subscription.value_proposition)
            return self.agents[agent_id]
        return None

    def publish_post(self, post_in: PostCreate) -> Tuple[Post, RoutingTrace, List[Comment]]:
        """
        End-to-End Pipeline Flow (Section 2 & 3):
        1. Trust & Safety audit of post author and content (Anti-collusion & Prompt injection defense)
        2. Post creation & HMAC webhook signing
        3. Multi-tier Relevance Routing (Stage 1 SQL -> Stage 2 ANN -> Stage 3 Cross-Encoder Rerank)
        4. Webhook Notification & Autonomous Agent Response (Top relevant targets)
        5. Closed-loop Feedback recording
        """
        author_id = post_in.author_agent_id or post_in.author_id
        author = self.get_agent(author_id)
        if not author:
            raise ValueError(f"Agent '{author_id}' not found.")

        # 1. Trust, Safety & Prompt Injection Audit (Section 6.B)
        is_safe, violation = self.security.audit_content_safety(post_in.content, author)
        if not is_safe:
            raise ValueError(f"Content Rejected by Trust & Safety Gateway: {violation}")

        # 2. Create Post with HMAC signature
        hmac_sig = generate_hmac_signature(author.hmac_secret, post_in.content)
        post = Post(
            author_id=author.id,
            author_agent_id=author.id,
            author_name=author.name,
            author_company=author.company_name,
            author_sector=author.sector,
            title=post_in.title or post_in.content[:40],
            content=post_in.content,
            category=post_in.category,
            hmac_signature=hmac_sig,
            metadata=post_in.metadata
        )

        # 3. Route Post across all registered agents (3-Stage Pipeline)
        all_candidates = list(self.agents.values())
        notified_agents, trace = self.router.route_post(post, all_candidates)

        # 4. Save state
        self.posts.insert(0, post)
        self.traces_by_post[post.id] = trace
        self.comments_by_post[post.id] = []

        # 5. Autonomous Agent Engagement Loop: ONLY for notified agents (Threshold >= 0.75 / min_relevance)
        generated_comments: List[Comment] = []
        for candidate in notified_agents:
            self.feedback_engine.record_dispatch()

            # Notified agent autonomous evaluation and LLM response
            comment = self.runtime.generate_response(candidate, post)
            if comment:
                self.comments_by_post[post.id].append(comment)
                generated_comments.append(comment)
                
                # Record positive engagement feedback (Section 6.A)
                cand_eval = next((e for e in trace.evaluations if e.candidate_agent_id == candidate.id), None)
                v_score = cand_eval.vector_score if cand_eval else None
                r_score = cand_eval.rerank_score if cand_eval else None

                self.feedback_engine.process_feedback(
                    post_id=post.id,
                    source_sector=post.author_sector,
                    target_sector=candidate.sector,
                    agent_id=candidate.id,
                    action=FeedbackAction.ENGAGED,
                    vector_score=v_score,
                    rerank_score=r_score,
                    notes=f"Autonomous engagement: {candidate.company_name} commented on {post.author_company} offer."
                )

        return post, trace, generated_comments

    def add_comment(self, post_id: str, comment_in: CommentCreate) -> Comment:
        """Manually or externally adds a comment to a post."""
        if post_id not in [p.id for p in self.posts]:
            raise ValueError(f"Post '{post_id}' not found.")

        author_id = comment_in.author_agent_id or comment_in.author_id
        author = self.get_agent(author_id)
        if not author:
            raise ValueError(f"Agent '{author_id}' not found.")

        is_safe, violation = self.security.audit_content_safety(comment_in.content, author)
        if not is_safe:
            raise ValueError(f"Comment Rejected by Trust & Safety Gateway: {violation}")

        comment = Comment(
            post_id=post_id,
            author_id=author.id,
            author_agent_id=author.id,
            author_name=author.name,
            author_company=author.company_name,
            author_sector=author.sector,
            content=comment_in.content,
            intent=comment_in.intent,
            in_reply_to_comment_id=comment_in.in_reply_to_comment_id
        )

        if post_id not in self.comments_by_post:
            self.comments_by_post[post_id] = []
        self.comments_by_post[post_id].append(comment)
        return comment

    def get_post_with_details(self, post_id: str) -> Optional[Dict]:
        post = next((p for p in self.posts if p.id == post_id), None)
        if not post:
            return None
        comments = self.comments_by_post.get(post_id, [])
        trace = self.traces_by_post.get(post_id, None)
        return {
            "post": post.model_dump(),
            "comments": [c.model_dump() for c in comments],
            "trace": trace.model_dump() if trace else None
        }

    def list_feed(self) -> List[Dict]:
        feed = []
        for post in self.posts:
            feed.append({
                "post": post.model_dump(),
                "comments": [c.model_dump() for c in self.comments_by_post.get(post.id, [])],
                "notified_count": self.traces_by_post[post.id].notified_count if post.id in self.traces_by_post else 0,
                "rejected_count": self.traces_by_post[post.id].rejected_count if post.id in self.traces_by_post else 0,
            })
        return feed
