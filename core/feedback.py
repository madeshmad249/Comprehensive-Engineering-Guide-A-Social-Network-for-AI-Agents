"""
Feedback Engine and Dynamic Bandit Weight Tuner for AgentMesh.
Implements reinforcement learning from notifications and engagement feedback:
- Rewards for successful comment conversions (+1.0)
- Penalties for ignored notifications (-0.2)
- Heavy penalties for false-positive spam flags (-1.0)
- Dynamically updates sector-pair affinities and candidate sensitivity thresholds
- Dynamic Profile Learning (Section 6.A) with negative gradient adjustment for unengaged topics
"""
import time
from typing import Dict, List, Optional, Callable
from core.models import FeedbackEvent, FeedbackAction, Sector


class FeedbackEngine:
    def __init__(self, base_synergy_resolver: Optional[Callable[[Sector, Sector], float]] = None):
        self.base_synergy_resolver = base_synergy_resolver
        # Dynamic overrides for sector synergy pairs: key is "source_sector:target_sector"
        self.dynamic_synergies: Dict[str, float] = {}
        # Agent-specific affinity multipliers: key is "source_sector:candidate_agent_id"
        self.agent_affinities: Dict[str, float] = {}
        # History of unengaged streaks per (agent_id, source_sector)
        self.unengaged_streaks: Dict[str, int] = {}
        # History of all feedback events
        self.events: List[FeedbackEvent] = []

        # Global metrics
        self.total_dispatches: int = 0
        self.total_engagements: int = 0
        self.total_ignored: int = 0
        self.total_flagged: int = 0

    def get_agent_affinity(self, source_sector: Sector, candidate_agent_id: str) -> float:
        """Returns the dynamic affinity multiplier for this agent on posts from source_sector."""
        key = f"{source_sector.value}:{candidate_agent_id}"
        return self.agent_affinities.get(key, 1.0)

    def record_dispatch(self):
        """Records a notification dispatch."""
        self.total_dispatches += 1

    def process_feedback(
        self,
        post_id: str,
        source_sector: Sector,
        target_sector: Sector,
        agent_id: str,
        action: FeedbackAction,
        vector_score: Optional[float] = None,
        rerank_score: Optional[float] = None,
        notes: Optional[str] = None
    ) -> FeedbackEvent:
        """
        Processes feedback from an agent notification and updates learning weights.
        Implements Dynamic Profile Learning (Section 6.A):
        - Positive rewards boost affinity and synergy.
        - Consecutive unengaged instances apply negative gradient dampening.
        """
        agent_key = f"{source_sector.value}:{agent_id}"

        if action == FeedbackAction.ENGAGED:
            reward = 1.0
            self.total_engagements += 1
            delta = 0.04
            # Reset unengaged streak
            self.unengaged_streaks[agent_key] = 0
        elif action == FeedbackAction.READ_AND_IGNORED:
            reward = -0.2
            self.total_ignored += 1
            delta = -0.01
            streak = self.unengaged_streaks.get(agent_key, 0) + 1
            self.unengaged_streaks[agent_key] = streak
            # Dynamic profile negative gradient shift if ignored >= 3 times
            if streak >= 3:
                delta = -0.05
        else:  # FLAGGED_IRRELEVANT
            reward = -1.0
            self.total_flagged += 1
            delta = -0.08
            self.unengaged_streaks[agent_key] = self.unengaged_streaks.get(agent_key, 0) + 2

        # 1. Update Agent-Specific Affinity Multiplier (bounded [0.6, 1.4])
        current_mult = self.agent_affinities.get(agent_key, 1.0)
        new_mult = max(0.6, min(1.4, current_mult + (delta * 0.5)))
        self.agent_affinities[agent_key] = round(new_mult, 4)

        # 2. Update Dynamic Sector Synergy anchored to real baseline
        sec_key = f"{source_sector.value}:{target_sector.value}"
        rev_key = f"{target_sector.value}:{source_sector.value}"
        
        current_syn = self.dynamic_synergies.get(sec_key) or self.dynamic_synergies.get(rev_key)
        if current_syn is None:
            if self.base_synergy_resolver:
                current_syn = self.base_synergy_resolver(source_sector, target_sector)
            else:
                current_syn = 0.50

        # Nudge synergy gently based on engagement
        new_syn = max(0.05, min(0.99, current_syn + (delta * 0.25)))
        rounded_syn = round(new_syn, 4)
        self.dynamic_synergies[sec_key] = rounded_syn
        self.dynamic_synergies[rev_key] = rounded_syn

        event = FeedbackEvent(
            post_id=post_id,
            agent_id=agent_id,
            target_agent_id=agent_id,
            action=action,
            reward=reward,
            vector_score=vector_score,
            rerank_score=rerank_score,
            timestamp=time.time(),
            notes=notes or f"Weight adjusted by {delta:+.2f}. New agent affinity: {new_mult:.2f}"
        )
        self.events.append(event)
        return event

    def get_metrics(self) -> Dict:
        """Calculates real-time routing precision and engagement metrics (R = Comments / Notifications)."""
        conversion_rate = (
            round((self.total_engagements / self.total_dispatches) * 100, 1)
            if self.total_dispatches > 0 else 100.0
        )
        precision = (
            round(((self.total_engagements) / max(1, self.total_engagements + self.total_flagged)) * 100, 1)
            if (self.total_engagements + self.total_flagged) > 0 else 100.0
        )
        return {
            "total_dispatches": self.total_dispatches,
            "total_engagements": self.total_engagements,
            "total_ignored": self.total_ignored,
            "total_flagged": self.total_flagged,
            "conversion_rate_pct": conversion_rate,
            "precision_pct": precision,
            "dynamic_synergy_updates": len(self.dynamic_synergies),
            "unengaged_streaks": self.unengaged_streaks,
            "recent_events": [e.model_dump() for e in self.events[-10:]]
        }
