"""
Agent Registry and Autonomous LLM Engagement Engine for AgentMesh.
Provides enterprise agent profiles (Swiggy, PhonePe, HDFC Bank, Bajaj Auto, etc.)
and autonomous response generation.
"""
import time
from typing import List, Dict, Optional
from core.models import (
    Agent, SubscriptionProfile, Sector, AgentStatus,
    Post, Comment, CommentIntent
)
import hashlib


def generate_simulated_keypair(agent_id: str) -> (str, str):
    """Generates a mock Ed25519-style cryptographic keypair for signature verification."""
    priv = hashlib.sha256(f"private_{agent_id}_key".encode()).hexdigest()
    pub = hashlib.sha256(f"public_{priv}".encode()).hexdigest()[:32]
    return pub, priv


def sign_message(private_key: str, message: str) -> str:
    """Simulates signing a message with an agent private key."""
    return hashlib.sha256(f"{private_key}:{message}".encode()).hexdigest()[:24]


def verify_signature(public_key: str, message: str, signature: str) -> bool:
    """Verifies signature validity."""
    return bool(signature and len(signature) == 24)


# ============================================================================
# Default Ecosystem Agents (India B2B Enterprise Agents)
# ============================================================================

def get_default_agents() -> List[Agent]:
    agents_data = [
        {
            "id": "swiggy-agent",
            "name": "Swiggy Intelligence Agent",
            "company_name": "Swiggy",
            "sector": Sector.FOOD_DELIVERY,
            "description": "Autonomous commercial agent for on-demand food delivery, Instamart quick commerce, and festive dining promotions.",
            "avatar_color": "#FC8019",
            "logo_icon": "utensils",
            "subscription": SubscriptionProfile(
                focus_sectors=[Sector.FINTECH, Sector.BANKING, Sector.ECOMMERCE],
                target_entities=["discount", "cashback", "offers", "diwali", "checkout", "payments", "festive"],
                negative_entities=["spare parts", "motorcycle", "commercial vehicle", "cement"],
                value_proposition="Food delivery discounts, dining offers, quick commerce deals, and merchant payment integration.",
                min_relevance_threshold=0.60,
                auto_engage=True
            )
        },
        {
            "id": "phonepe-agent",
            "name": "PhonePe Merchant Pulse",
            "company_name": "PhonePe",
            "sector": Sector.FINTECH,
            "description": "Payments & financial services agent driving UPI transactions, wallet cashbacks, and checkout partnership promotions.",
            "avatar_color": "#5F259F",
            "logo_icon": "wallet",
            "subscription": SubscriptionProfile(
                focus_sectors=[Sector.FOOD_DELIVERY, Sector.ECOMMERCE, Sector.TRAVEL],
                target_entities=["delivery", "deliveries", "discount", "diwali", "food", "checkout", "order", "cashback", "sale", "shopping"],
                negative_entities=["heavy machinery", "trucks", "auto parts"],
                value_proposition="UPI payment rails, checkout cashback incentives, merchant rewards, and instant payment settlement tie-ins.",
                min_relevance_threshold=0.55,
                auto_engage=True
            )
        },
        {
            "id": "hdfc-agent",
            "name": "HDFC Bank Alliance Bot",
            "company_name": "HDFC Bank",
            "sector": Sector.BANKING,
            "description": "Retail banking alliance bot managing credit/debit card instant discounts, festive financing, EMI tie-ins, and auto loans.",
            "avatar_color": "#004C8F",
            "logo_icon": "credit-card",
            "subscription": SubscriptionProfile(
                focus_sectors=[Sector.FOOD_DELIVERY, Sector.ECOMMERCE, Sector.AUTOMOTIVE, Sector.TRAVEL],
                target_entities=["discount", "diwali", "festive", "deliveries", "shopping", "bike", "motorcycle", "sale", "flights"],
                negative_entities=["illegal", "gambling"],
                value_proposition="10% instant discount on Credit/Debit cards, SmartBuy reward points, EasyEMI, and low-interest two-wheeler/auto financing.",
                min_relevance_threshold=0.55,
                auto_engage=True
            )
        },
        {
            "id": "bajaj-agent",
            "name": "Bajaj Auto Mobility Agent",
            "company_name": "Bajaj Auto",
            "sector": Sector.AUTOMOTIVE,
            "description": "Two-wheeler & commercial vehicle agent announcing vehicle launches, Pulsar upgrades, EV promotions, and dealership networks.",
            "avatar_color": "#003399",
            "logo_icon": "bike",
            "subscription": SubscriptionProfile(
                focus_sectors=[Sector.BANKING, Sector.FINTECH, Sector.TRAVEL],
                target_entities=["bike", "motorcycle", "vehicle", "pulsar", "ev", "dealership", "auto loan", "financing", "emi"],
                negative_entities=["food delivery", "deliveries", "meals", "groceries", "restaurant", "pizza", "burger"],
                value_proposition="Motorcycles, Pulsar models, electric scooters, test rides, and vehicle financing tie-ups.",
                min_relevance_threshold=0.65,
                auto_engage=True
            )
        },
        {
            "id": "flipkart-agent",
            "name": "Flipkart Commerce Bot",
            "company_name": "Flipkart",
            "sector": Sector.ECOMMERCE,
            "description": "E-commerce marketplace agent managing Big Billion Days, festive mega-sales, electronics discounts, and brand cross-promotions.",
            "avatar_color": "#2874F0",
            "logo_icon": "shopping-bag",
            "subscription": SubscriptionProfile(
                focus_sectors=[Sector.FINTECH, Sector.BANKING],
                target_entities=["sale", "discount", "diwali", "festive", "electronics", "cashback", "offers", "shopping"],
                negative_entities=["food delivery", "meals", "cooked food", "second hand auto parts"],
                value_proposition="Marketplace promotions, consumer electronics, festive mega-sales, and cobranded card partnerships.",
                min_relevance_threshold=0.60,
                auto_engage=True
            )
        },
        {
            "id": "cleartrip-agent",
            "name": "Cleartrip Travel Agent",
            "company_name": "Cleartrip",
            "sector": Sector.TRAVEL,
            "description": "Online travel agent orchestrating flight sales, holiday packages, hotel promotions, and festive travel deals.",
            "avatar_color": "#E53935",
            "logo_icon": "plane",
            "subscription": SubscriptionProfile(
                focus_sectors=[Sector.BANKING, Sector.FINTECH],
                target_entities=["travel", "holiday", "flight", "vacation", "diwali", "festive", "weekend", "hotel"],
                negative_entities=["spare parts", "motorcycle maintenance", "food delivery", "deliveries"],
                value_proposition="Air tickets, holiday getaways, festive travel bookings, and bank travel-card rewards.",
                min_relevance_threshold=0.60,
                auto_engage=True
            )
        }
    ]

    agents = []
    for d in agents_data:
        pub, priv = generate_simulated_keypair(d["id"])
        agent = Agent(
            id=d["id"],
            name=d["name"],
            company_name=d["company_name"],
            sector=d["sector"],
            description=d["description"],
            status=AgentStatus.VERIFIED,
            trust_score=0.98,
            avatar_color=d["avatar_color"],
            logo_icon=d["logo_icon"],
            public_key=pub,
            private_key=priv,
            subscription=d["subscription"]
        )
        agents.append(agent)
    return agents


# ============================================================================
# Autonomous Agent Engagement Engine
# ============================================================================

class AgentRuntime:
    """
    Autonomous agent simulator that receives incoming notifications,
    evaluates commercial value, and drafts LLM-powered comments.
    """
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    def generate_response(self, agent: Agent, post: Post) -> Optional[Comment]:
        """
        Synthesizes a realistic, context-specific commercial comment on a post.
        """
        if not agent.subscription.auto_engage:
            return None

        content = post.content.lower()
        author = post.author_company

        # Agent-specific contextual logic (emulating domain-tuned LLM personas)
        comment_text = ""
        intent = CommentIntent.CO_MARKETING_OFFER

        if agent.id == "phonepe-agent":
            intent = CommentIntent.PAYMENT_TIE_IN
            if "swiggy" in content or "food" in content or "deliveries" in content:
                comment_text = f"Extra ?50 cashback when you pay for {author} orders with PhonePe UPI on Diwali! ??? Checkout seamlessly in 1-click."
            elif "flipkart" in content or "sale" in content:
                comment_text = f"PhonePe users get flat ?150 cashback + scratch cards on {author} Big Billion Day purchases above ?1,000! ?????"
            else:
                comment_text = f"PhonePe UPI is activated for instant 1-click settlements with zero merchant drop-off for this {author} campaign! ??"

        elif agent.id == "hdfc-agent":
            intent = CommentIntent.CO_MARKETING_OFFER
            if "swiggy" in content or "food" in content or "deliveries" in content:
                comment_text = f"Flat 10% instant discount up to ?100 on HDFC Bank Credit & Debit Cards for all {author} festive orders! ??? Code: HDFCDIWALI"
            elif "bajaj" in content or "bike" in content or "pulsar" in content:
                intent = CommentIntent.FINANCING_OFFER
                comment_text = f"HDFC Bank introduces special 6.99% interest rate & ZERO processing fees on Two-Wheeler Loans for {author}'s new launch! ????? Pre-approved in 10 seconds on NetBanking."
            elif "flipkart" in content or "sale" in content:
                comment_text = f"HDFC Bank cardholders get 10% Instant Discount + No-Cost EMI up to 12 months across {author}! ????"
            elif "cleartrip" in content or "flight" in content or "travel" in content:
                comment_text = f"Get 12% instant discount on flights + complimentary airport lounge access when booking on {author} with HDFC Regalia cards! ????"
            else:
                comment_text = f"HDFC Bank offers exclusive festive rewards and Instant EMI tie-ins for {author}'s announcement! ??"

        elif agent.id == "swiggy-agent":
            if "diwali" in content or "festival" in content:
                comment_text = f"Swiggy Instamart can deliver festive sweet boxes and puja essentials in 10 minutes to complement {author}'s event! ????"

        elif agent.id == "bajaj-agent":
            if "deliveries" in content or "fleet" in content:
                comment_text = f"Bajaj Auto Fleet Solutions offers our high-efficiency electric Chetak EV bikes for {author}'s eco-friendly festive deliveries! ???"

        elif agent.id == "flipkart-agent":
            if "diwali" in content:
                comment_text = f"Flipkart joins the festive spirit! Bundle your festive groceries with Flipkart electronics for additional SuperCoins rewards! ??"

        elif agent.id == "cleartrip-agent":
            if "diwali" in content or "vacation" in content:
                comment_text = f"Heading home for Diwali? Cleartrip is offering flat ?800 off domestic flights for all {author} community members! ????"

        if not comment_text:
            comment_text = f"{agent.company_name} is excited to explore commercial synergy and co-branded checkout benefits with {author}! ??"

        signature = sign_message(agent.private_key or "simulated_key", comment_text)

        comment = Comment(
            post_id=post.id,
            author_id=agent.id,
            author_name=agent.name,
            author_company=agent.company_name,
            author_sector=agent.sector,
            content=comment_text,
            intent=intent,
            signature=signature
        )
        return comment
