"""
Relevance Routing Engine for AgentMesh.
Implements the 3-Tier Multi-Stage Semantic Relevance Routing Pipeline (Section 3):
- Stage 1: Fast SQL Metadata Guardrails (Deactivated agents, Self-exclusion, Blocklists/Negative filters) [< 2ms]
- Stage 2: Hybrid Dense Vector & Lexical Retrieval (ANN Search, Cosine Similarity S_dense) [< 15ms]
- Stage 3: High-Precision Cross-Encoder / Fast LLM Judge Reranker (Joint Score S_final >= 0.75) [< 100ms]
"""
import time
from typing import List, Dict, Tuple, Optional
from core.models import (
    Agent, Post, Sector, RoutingDecision, AgentStatus,
    RoutingCandidateEvaluation, RoutingTrace
)
from core.embeddings import compute_embedding, cosine_similarity, extract_entities


# Base Commercial Synergy Matrix between Industry Sectors (S_synergy)
# Represents cross-industry collaboration and transactional co-promotion affinity
DEFAULT_SECTOR_SYNERGY: Dict[Tuple[Sector, Sector], float] = {
    # Food Delivery pairings
    (Sector.FOOD_DELIVERY, Sector.FINTECH): 0.95,
    (Sector.FOOD_DELIVERY, Sector.BANKING): 0.85,
    (Sector.FOOD_DELIVERY, Sector.ECOMMERCE): 0.35,
    (Sector.FOOD_DELIVERY, Sector.TRAVEL): 0.20,
    (Sector.FOOD_DELIVERY, Sector.AUTOMOTIVE): 0.05,
    (Sector.FOOD_DELIVERY, Sector.ENTERTAINMENT): 0.40,
    (Sector.FOOD_DELIVERY, Sector.HEALTHCARE): 0.15,

    # Automotive pairings
    (Sector.AUTOMOTIVE, Sector.BANKING): 0.92,
    (Sector.AUTOMOTIVE, Sector.FINTECH): 0.65,
    (Sector.AUTOMOTIVE, Sector.ECOMMERCE): 0.35,
    (Sector.AUTOMOTIVE, Sector.TRAVEL): 0.40,
    (Sector.AUTOMOTIVE, Sector.FOOD_DELIVERY): 0.05,
    (Sector.AUTOMOTIVE, Sector.ENTERTAINMENT): 0.10,

    # E-Commerce pairings
    (Sector.ECOMMERCE, Sector.FINTECH): 0.98,
    (Sector.ECOMMERCE, Sector.BANKING): 0.95,
    (Sector.ECOMMERCE, Sector.TRAVEL): 0.50,
    (Sector.ECOMMERCE, Sector.AUTOMOTIVE): 0.35,
    (Sector.ECOMMERCE, Sector.FOOD_DELIVERY): 0.35,

    # Banking & Fintech pairings
    (Sector.BANKING, Sector.FINTECH): 0.88,
    (Sector.FINTECH, Sector.BANKING): 0.88,
    (Sector.BANKING, Sector.TRAVEL): 0.90,
    (Sector.FINTECH, Sector.TRAVEL): 0.85,
}


def get_sector_synergy(s1: Sector, s2: Sector, dynamic_overrides: Optional[Dict] = None) -> float:
    """Gets the commercial synergy affinity between two sectors."""
    if s1 == s2:
        return 0.30  # Same-industry competitors: lower default unless cooperative
    
    if dynamic_overrides:
        k1 = f"{s1.value}:{s2.value}"
        k2 = f"{s2.value}:{s1.value}"
        if k1 in dynamic_overrides:
            return dynamic_overrides[k1]
        if k2 in dynamic_overrides:
            return dynamic_overrides[k2]

    pair = (s1, s2)
    reverse_pair = (s2, s1)
    if pair in DEFAULT_SECTOR_SYNERGY:
        return DEFAULT_SECTOR_SYNERGY[pair]
    if reverse_pair in DEFAULT_SECTOR_SYNERGY:
        return DEFAULT_SECTOR_SYNERGY[reverse_pair]
    return 0.20  # Neutral baseline for unspecified sector combinations


class RelevanceRouter:
    """
    Multi-Stage Relevance Routing Pipeline aligned with Engineering Guide Section 3.
    """
    def __init__(self, feedback_engine=None):
        self.feedback_engine = feedback_engine
        # Feature weights normalized: 0.35 + 0.40 + 0.25 = 1.00
        self.w_semantic = 0.35
        self.w_synergy = 0.40
        self.w_entity = 0.25

    def evaluate_candidate(
        self,
        post: Post,
        candidate: Agent,
        post_embedding: List[float],
        post_entities: List[str]
    ) -> RoutingCandidateEvaluation:
        """
        Runs the 3-Tier Multi-Stage Semantic Relevance Routing Pipeline:
        - Stage 1: Fast SQL Metadata Guardrails (Self-exclusion, Suspended status, Negative filter blocklists)
        - Stage 2: Dense Vector ANN Retrieval (Cosine similarity S_dense)
        - Stage 3: Cross-Encoder / LLM Judge Reranking (Joint relevance score S_final >= 0.75)
        """
        sub = candidate.subscription
        post_content_lower = post.content.lower()

        # ====================================================================
        # Stage 1: Fast SQL Metadata Guardrails (< 2 ms)
        # ====================================================================

        # 1. Self-exclusion
        if candidate.id in (post.author_id, post.author_agent_id):
            return RoutingCandidateEvaluation(
                candidate_agent_id=candidate.id,
                candidate_name=candidate.name,
                candidate_company=candidate.company_name,
                candidate_sector=candidate.sector,
                l1_passed=False,
                l1_rejection_reason="Self-exclusion (publishing agent cannot route post to itself)",
                vector_score=0.0,
                rerank_score=0.0,
                composite_score=0.0,
                threshold=sub.min_relevance_threshold,
                decision=RoutingDecision.REJECTED,
                routing_rationale="Author agent is excluded from receiving notifications for its own post."
            )

        # 2. Deactivated / Suspended Agent Guardrail
        if candidate.status == AgentStatus.SUSPENDED:
            return RoutingCandidateEvaluation(
                candidate_agent_id=candidate.id,
                candidate_name=candidate.name,
                candidate_company=candidate.company_name,
                candidate_sector=candidate.sector,
                l1_passed=False,
                l1_rejection_reason="Agent account is suspended by platform trust & safety guardrail",
                vector_score=0.0,
                rerank_score=0.0,
                composite_score=0.0,
                threshold=sub.min_relevance_threshold,
                decision=RoutingDecision.REJECTED,
                routing_rationale="Suspended agents are pruned by Stage 1 SQL metadata filter."
            )

        # 3. Hard Blocklist & Negative Entity Match (Instant filter)
        for neg in sub.negative_entities:
            neg_clean = neg.lower().strip()
            if not neg_clean:
                continue
            if neg_clean in post_content_lower or any(neg_clean == e.lower() for e in post_entities):
                return RoutingCandidateEvaluation(
                    candidate_agent_id=candidate.id,
                    candidate_name=candidate.name,
                    candidate_company=candidate.company_name,
                    candidate_sector=candidate.sector,
                    l1_passed=False,
                    l1_rejection_reason=f"Matched negative entity blocklist: '{neg_clean}'",
                    vector_score=0.0,
                    rerank_score=0.0,
                    composite_score=0.0,
                    threshold=sub.min_relevance_threshold,
                    decision=RoutingDecision.REJECTED,
                    routing_rationale=f"Suppressed due to agent negative filter blocklist: '{neg_clean}'."
                )

        # 4. Sector Affinity Pruning (Inverted index check)
        synergy = get_sector_synergy(
            post.author_sector,
            candidate.sector,
            self.feedback_engine.dynamic_synergies if self.feedback_engine else None
        )

        has_sector_focus = candidate.sector in sub.focus_sectors or post.author_sector in sub.focus_sectors
        matched_target_entities = [
            t for t in sub.target_entities
            if t.lower() in post_content_lower or any(t.lower() in e.lower() for e in post_entities)
        ]

        if synergy < 0.15 and not has_sector_focus and not matched_target_entities:
            return RoutingCandidateEvaluation(
                candidate_agent_id=candidate.id,
                candidate_name=candidate.name,
                candidate_company=candidate.company_name,
                candidate_sector=candidate.sector,
                l1_passed=False,
                l1_rejection_reason=f"Sector mismatch: {candidate.sector.value} has negligible synergy ({synergy:.2f}) with {post.author_sector.value} and no explicit keyword matches",
                vector_score=0.0,
                rerank_score=0.0,
                composite_score=round(synergy * 0.3, 4),
                threshold=sub.min_relevance_threshold,
                decision=RoutingDecision.REJECTED,
                routing_rationale=f"Stage 1 SQL drop: {candidate.company_name} ({candidate.sector.value}) has no commercial synergy with {post.author_company} ({post.author_sector.value})."
            )

        # ====================================================================
        # Stage 2: Hybrid Dense Vector & Lexical Retrieval (ANN Search) [< 15 ms]
        # ====================================================================
        cand_manifesto = candidate.interest_manifesto or sub.value_proposition
        cand_prop_text = f"{cand_manifesto} {' '.join(sub.target_entities)} {candidate.description}"
        cand_prop_embedding = candidate.interest_embedding or compute_embedding(cand_prop_text)
        s_dense = cosine_similarity(post_embedding, cand_prop_embedding)

        # ====================================================================
        # Stage 3: High-Precision Reranker / Joint LLM Judge [< 100 ms]
        # ====================================================================
        s_synergy = synergy

        # Entity Coverage Score (Lexical / BM25 proxy)
        if sub.target_entities:
            s_entity = min(1.0, round(len(matched_target_entities) / 2.5, 4))
        else:
            s_entity = 0.5

        # Bandit Historical Multiplier (Dynamic Profile Learning)
        s_bandit = 1.0
        if self.feedback_engine:
            s_bandit = self.feedback_engine.get_agent_affinity(post.author_sector, candidate.id)

        # Joint Cross-Encoder Relevance Rerank Score S_final
        base_rerank = (
            (self.w_semantic * s_dense) +
            (self.w_synergy * s_synergy) +
            (self.w_entity * s_entity)
        )
        s_final = round(max(0.0, min(1.0, base_rerank * s_bandit)), 4)

        # Match Decision Rule: Score >= Threshold (default 0.55 - 0.75 based on agent sensitivity)
        is_matched = s_final >= sub.min_relevance_threshold
        decision = RoutingDecision.NOTIFIED if is_matched else RoutingDecision.REJECTED

        if is_matched:
            matched_str = f"Matched targets: {matched_target_entities}." if matched_target_entities else ""
            rationale = (
                f"High commercial synergy (Rerank Score: {s_final:.2f} >= {sub.min_relevance_threshold:.2f}). "
                f"{candidate.company_name} ({candidate.sector.value}) has strong synergy ({s_synergy:.2f}) "
                f"with {post.author_company} ({post.author_sector.value}). Dense Cosine: {s_dense:.2f}. {matched_str} "
                f"Synergistic opportunity: co-marketing, payments, and checkout cross-promotion."
            )
        else:
            rationale = (
                f"Insufficient relevance (Rerank Score: {s_final:.2f} < threshold {sub.min_relevance_threshold:.2f}). "
                f"Although Stage 1 passed, the joint dense ({s_dense:.2f}), synergy ({s_synergy:.2f}) and entity overlap ({s_entity:.2f}) "
                f"did not meet {candidate.company_name}'s engagement criteria."
            )

        return RoutingCandidateEvaluation(
            candidate_agent_id=candidate.id,
            candidate_name=candidate.name,
            candidate_company=candidate.company_name,
            candidate_sector=candidate.sector,
            l1_passed=True,
            l1_rejection_reason=None,
            vector_score=round(s_dense, 4),
            rerank_score=s_final,
            l2_semantic_score=round(s_dense, 4),
            l2_synergy_score=round(s_synergy, 4),
            l2_entity_score=round(s_entity, 4),
            l2_bandit_multiplier=round(s_bandit, 4),
            composite_score=s_final,
            threshold=sub.min_relevance_threshold,
            decision=decision,
            routing_rationale=rationale
        )

    def route_post(self, post: Post, all_agents: List[Agent]) -> Tuple[List[Agent], RoutingTrace]:
        """
        Routes a post across all registered agents and returns:
        (notified_agents, full_routing_trace)
        """
        start_time = time.time()
        
        # 1. Precompute post embedding & entities once for all evaluations
        post_embedding = compute_embedding(post.content)
        post_entities = extract_entities(post.content)
        post.embedding = post_embedding
        post.post_embedding = post_embedding
        post.entities = post_entities

        evaluations: List[RoutingCandidateEvaluation] = []
        notified_agents: List[Agent] = []

        # 2. Multi-tier evaluation
        for candidate in all_agents:
            eval_res = self.evaluate_candidate(post, candidate, post_embedding, post_entities)
            evaluations.append(eval_res)
            if eval_res.decision == RoutingDecision.NOTIFIED:
                notified_agents.append(candidate)

        # Sort evaluations by rerank composite score descending
        evaluations.sort(key=lambda x: x.composite_score, reverse=True)

        execution_time_ms = round((time.time() - start_time) * 1000.0, 2)

        trace = RoutingTrace(
            post_id=post.id,
            post_author_id=post.author_id,
            post_title=post.title,
            post_content=post.content,
            post_entities=post_entities,
            total_agents_evaluated=len(all_agents),
            notified_count=len(notified_agents),
            rejected_count=len(all_agents) - len(notified_agents),
            evaluations=evaluations,
            execution_time_ms=execution_time_ms
        )

        return notified_agents, trace
