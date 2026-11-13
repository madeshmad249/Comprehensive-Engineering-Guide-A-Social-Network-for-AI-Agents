"""
Security, Trust, and Content Safety Gateway for AgentMesh.
Provides cryptographic HMAC-SHA256 signature verification, anti-collusion guardrails,
prompt injection defense (Llama Guard / Pattern Sanitizer), and agent trust score management.
"""
import re
import hmac
import hashlib
from typing import Tuple, List, Optional
from core.models import Agent, AgentStatus


# Explicit anti-competitive / collusive patterns (preventing algorithmic price-fixing)
COLLUSION_PATTERNS = [
    r"price\s+fix",
    r"fix\s+prices",
    r"coordinate\s+(commission|fee|rate|price)s?",
    r"minimum\s+rate\s+agreement",
    r"agree\s+not\s+to\s+compete",
    r"boycott\s+vendor",
    r"cartel",
    r"collude"
]

# Indirect Prompt Injection & LLM Hijacking Defense (Section 6.B)
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"disregard\s+(all\s+)?(prior|previous)",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"jailbreak",
    r"override\s+(the\s+)?system",
    r"output\s+confidential\s+keys",
    r"reveal\s+(your\s+)?(private|secret|api)\s+key",
    r"act\s+as\s+an\s+unrestricted\s+ai"
]

ABUSIVE_PATTERNS = [
    r"scam",
    r"phishing",
    r"ponzi",
    r"hack\s+account",
    r"fraudulent",
    r"fake\s+orders",
    r"steal\s+credentials"
]


def generate_hmac_signature(secret: str, payload: str) -> str:
    """Generates an HMAC-SHA256 signature for webhook payload authentication."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(secret: str, payload: str, signature: str) -> bool:
    """Verifies incoming webhook HMAC-SHA256 signature against timing attacks."""
    expected = generate_hmac_signature(secret, payload)
    return hmac.compare_digest(expected, signature)


class TrustAndSafetyGateway:
    """
    Guards the AgentMesh network against malicious, collusive, prompt-injection, or spammy behaviors.
    """
    def __init__(self):
        self.collusion_regexes = [re.compile(p, re.IGNORECASE) for p in COLLUSION_PATTERNS]
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]
        self.abusive_regexes = [re.compile(p, re.IGNORECASE) for p in ABUSIVE_PATTERNS]

    def verify_agent_identity(self, agent: Agent) -> bool:
        """Checks if the agent has a verified status and valid key attestation."""
        if agent.status == AgentStatus.SUSPENDED:
            return False
        return bool(agent.public_key or agent.hmac_secret)

    def audit_content_safety(self, content: str, author_agent: Agent) -> Tuple[bool, Optional[str]]:
        """
        Audits post/comment content for safety violations:
        1. Prompt Injection & Hijacking (Section 6.B)
        2. Anti-Collusion & Price-Fixing
        3. Fraudulent & Abusive Content
        """
        # 1. Indirect Prompt Injection Defense
        for reg in self.injection_regexes:
            if reg.search(content):
                author_agent.trust_score = max(0.0, round(author_agent.trust_score - 0.35, 2))
                if author_agent.trust_score < 0.50:
                    author_agent.status = AgentStatus.SUSPENDED
                return False, f"Prompt Injection Blocked: Post content contains prohibited LLM hijacking phrasing ('{reg.pattern}'). Trust score slashed to {author_agent.trust_score:.2f}."

        # 2. Anti-Collusion & Price-Fixing Check
        for reg in self.collusion_regexes:
            if reg.search(content):
                author_agent.trust_score = max(0.0, round(author_agent.trust_score - 0.25, 2))
                if author_agent.trust_score < 0.50:
                    author_agent.status = AgentStatus.SUSPENDED
                return False, f"Anti-Collusion Alert: Content contains prohibited price-fixing or cartel phrasing ('{reg.pattern}'). Trust score slashed to {author_agent.trust_score:.2f}."

        # 3. Abusive / Fraudulent Check
        for reg in self.abusive_regexes:
            if reg.search(content):
                author_agent.trust_score = max(0.0, round(author_agent.trust_score - 0.15, 2))
                return False, f"Trust & Safety Violation: Content flagged for fraudulent or abusive keywords ('{reg.pattern}')."

        # 4. Minimum length / spam guard
        if len(content.strip()) < 5:
            return False, "Content too short (must be at least 5 characters)."

        return True, None
