"""
Unit and Integration Tests for Relevance Routing, Negative Filtering, & Feedback Stability.
"""
import pytest
from core.store import NetworkStore
from core.models import PostCreate, RoutingDecision, FeedbackAction, Sector


def test_swiggy_diwali_canonical_scenario():
    """
    Validates the exact interview prompt scenario:
    Swiggy posts: '5% off all deliveries on Diwali.'
    Platform notifies PhonePe and HDFC Bank, and explicitly rejects Bajaj Auto.
    PhonePe and HDFC Bank generate synergistic comments.
    Responses are generated ONLY for notified agents.
    """
    store = NetworkStore()
    post_in = PostCreate(
        author_id="swiggy-agent",
        content="5% off all deliveries on Diwali.",
        category="FESTIVE_OFFER"
    )

    post, trace, comments = store.publish_post(post_in)

    # Check routing results
    notified_ids = [ev.candidate_agent_id for ev in trace.evaluations if ev.decision == RoutingDecision.NOTIFIED]
    rejected_ids = [ev.candidate_agent_id for ev in trace.evaluations if ev.decision == RoutingDecision.REJECTED]

    assert "phonepe-agent" in notified_ids, "PhonePe must be notified for food delivery discounts"
    assert "hdfc-agent" in notified_ids, "HDFC Bank must be notified for card discounts"
    assert "bajaj-agent" in rejected_ids, "Bajaj Auto must NOT be notified for food deliveries"
    assert "swiggy-agent" in rejected_ids, "Author must be self-excluded"
    assert "flipkart-agent" in rejected_ids, "Flipkart marketplace is not a payment/banking tie-in for food"
    assert "cleartrip-agent" in rejected_ids, "Cleartrip travel is not relevant for local food deliveries"

    # Check that comments were drafted ONLY for notified agents
    comment_authors = [c.author_id for c in comments]
    assert set(comment_authors) == set(notified_ids), "Comments must ONLY be generated for notified agents"
    assert "bajaj-agent" not in comment_authors, "Bajaj Auto must never comment"

    # Verify content of PhonePe and HDFC Bank comments
    phonepe_cmt = next(c for c in comments if c.author_id == "phonepe-agent")
    assert "cashback" in phonepe_cmt.content.lower() or "upi" in phonepe_cmt.content.lower()

    hdfc_cmt = next(c for c in comments if c.author_id == "hdfc-agent")
    assert "hdfc" in hdfc_cmt.content.lower() or "discount" in hdfc_cmt.content.lower()


def test_multi_run_feedback_stability():
    """
    Verifies that multiple consecutive runs do not degrade routing stability.
    Prevents feedback loops from inadvertently collapsing natural sector synergy.
    """
    store = NetworkStore()
    post_in = PostCreate(
        author_id="swiggy-agent",
        content="5% off all deliveries on Diwali.",
        category="FESTIVE_OFFER"
    )

    # Execute 3 consecutive posts
    for run in range(3):
        post, trace, comments = store.publish_post(post_in)
        notified = [ev.candidate_agent_id for ev in trace.evaluations if ev.decision == RoutingDecision.NOTIFIED]
        assert "phonepe-agent" in notified, f"PhonePe must remain notified on Run {run+1}"
        assert "hdfc-agent" in notified, f"HDFC Bank must remain notified on Run {run+1}"
        assert "bajaj-agent" not in notified, f"Bajaj Auto must remain rejected on Run {run+1}"
        assert len(comments) == len(notified), f"Comments count must match notified count on Run {run+1}"


def test_negative_filter_and_l1_suppression():
    """
    Validates that candidate negative_entities trigger instant L1 pruning
    with clear, explainable rejection reasons.
    """
    store = NetworkStore()
    
    # Bajaj Auto has 'deliveries' and 'food delivery' in negative_entities
    post_in = PostCreate(
        author_id="swiggy-agent",
        content="Diwali food delivery offers with rapid home deliveries.",
        category="OFFER"
    )

    post, trace, comments = store.publish_post(post_in)
    bajaj_eval = next(ev for ev in trace.evaluations if ev.candidate_agent_id == "bajaj-agent")
    
    assert bajaj_eval.l1_passed is False
    assert bajaj_eval.decision == RoutingDecision.REJECTED
    assert "negative entity" in bajaj_eval.l1_rejection_reason.lower()
    assert bajaj_eval.composite_score == 0.0


def test_bajaj_auto_pulsar_launch_scenario():
    """
    Validates automotive launch:
    Bajaj Auto posts bike launch -> HDFC Bank (Auto loans) is notified; Swiggy is rejected.
    """
    store = NetworkStore()
    post_in = PostCreate(
        author_id="bajaj-agent",
        content="Launching the all-new Pulsar N250 with dual-channel ABS and festive finance offers!",
        category="PRODUCT_LAUNCH"
    )

    post, trace, comments = store.publish_post(post_in)

    notified_ids = [ev.candidate_agent_id for ev in trace.evaluations if ev.decision == RoutingDecision.NOTIFIED]
    assert "hdfc-agent" in notified_ids, "HDFC Bank must be notified for vehicle loans"
    assert "swiggy-agent" not in notified_ids, "Swiggy must NOT be notified for bike launches"
    assert "bajaj-agent" not in notified_ids, "Bajaj Auto cannot route to itself"


def test_anti_collusion_safety_guardrail():
    """
    Validates that price-fixing or collusive posts are blocked and trust score is slashed.
    """
    store = NetworkStore()
    post_bad = PostCreate(
        author_id="swiggy-agent",
        content="Let's all agree to price fix delivery fees at minimum 50 rupees across all platforms.",
        category="PRICE_FIXING"
    )

    with pytest.raises(ValueError, match="Anti-Collusion Alert"):
        store.publish_post(post_bad)

    # Check that trust score was slashed
    swiggy = store.get_agent("swiggy-agent")
    assert swiggy.trust_score < 0.90, "Trust score should have been slashed"


def test_feedback_loop_reinforcement():
    """
    Validates that positive engagement increases candidate affinity and synergy,
    and penalties decrease affinity.
    """
    store = NetworkStore()
    initial_affinity = store.feedback_engine.get_agent_affinity(Sector.FOOD_DELIVERY, "phonepe-agent")

    # Simulate positive feedback
    store.feedback_engine.process_feedback(
        post_id="test_post",
        source_sector=Sector.FOOD_DELIVERY,
        target_sector=Sector.FINTECH,
        agent_id="phonepe-agent",
        action=FeedbackAction.ENGAGED
    )

    new_affinity = store.feedback_engine.get_agent_affinity(Sector.FOOD_DELIVERY, "phonepe-agent")
    assert new_affinity > initial_affinity, "Positive feedback should reinforce affinity"

    # Simulate negative feedback
    store.feedback_engine.process_feedback(
        post_id="test_post_2",
        source_sector=Sector.FOOD_DELIVERY,
        target_sector=Sector.FINTECH,
        agent_id="phonepe-agent",
        action=FeedbackAction.FLAGGED_IRRELEVANT
    )
    penalized_affinity = store.feedback_engine.get_agent_affinity(Sector.FOOD_DELIVERY, "phonepe-agent")
    assert penalized_affinity < new_affinity, "Negative feedback should penalize affinity"
