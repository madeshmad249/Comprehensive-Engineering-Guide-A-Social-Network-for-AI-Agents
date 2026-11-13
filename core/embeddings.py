"""
Semantic Embedding and Entity Extraction Engine for AgentMesh.
Provides ultra-fast, zero-dependency semantic vector embeddings, cosine similarity,
domain-aware entity extraction, and linguistic normalization.
"""
import re
import math
import hashlib
from typing import List, Set, Dict

EMBEDDING_DIM = 64

# Canonical domain ontologies to seed semantic clusters in vector space
DOMAIN_CLUSTERS = {
    "commerce_discount": [
        "offer", "discount", "deal", "save", "saving", "sale", "festive",
        "diwali", "cashback", "voucher", "coupon", "rewards", "exclusive"
    ],
    "food_grocery": [
        "delivery", "food", "restaurant", "dining", "order", "deliveries",
        "grocery", "meals", "swiggy", "zomato", "instamart", "blinkit", "kitchen"
    ],
    "payments_fintech": [
        "payment", "payments", "upi", "wallet", "cashback", "checkout",
        "phonepe", "paytm", "gpay", "qr", "transaction", "transfer", "instant"
    ],
    "banking_credit": [
        "bank", "banking", "hdfc", "card", "cards", "credit", "debit",
        "loan", "emi", "interest", "account", "financing", "lending"
    ],
    "ecommerce_retail": [
        "flipkart", "amazon", "shopping", "electronics", "fashion", "cart",
        "marketplace", "products", "big billion", "orders", "shipping"
    ],
    "automotive_mobility": [
        "bike", "motorcycle", "pulsar", "vehicle", "scooter", "bajaj",
        "car", "auto", "dealership", "engine", "two wheeler", "ev", "rides"
    ],
    "travel_hospitality": [
        "flight", "travel", "hotel", "ticket", "vacation", "cleartrip",
        "booking", "tourism", "trip", "airline"
    ]
}

LEMMA_MAP = {
    "deliveries": "delivery", "delivering": "delivery", "delivered": "delivery",
    "discounts": "discount", "discounted": "discount",
    "offers": "offer", "offering": "offer", "offered": "offer",
    "orders": "order", "ordering": "order", "ordered": "order",
    "payments": "payment", "paying": "payment", "paid": "payment",
    "cards": "card",
    "loans": "loan",
    "bikes": "bike",
    "motorcycles": "motorcycle",
    "vehicles": "vehicle",
    "flights": "flight",
    "hotels": "hotel",
    "sales": "sale",
    "stores": "store",
}

INTENT_SYNONYMS = {
    "off": ["discount", "offer"],
    "save": ["discount", "offer"],
    "savings": ["discount", "offer"],
    "deal": ["discount", "offer"],
    "deals": ["discount", "offer"],
    "diwali": ["festive", "diwali"],
    "festive": ["festive", "diwali"],
    "festival": ["festive", "diwali"],
    "checkout": ["payment", "checkout", "upi"],
    "cashbacks": ["cashback"],
}


def tokenize(text: str) -> List[str]:
    """Extract lowercase alpha-numeric tokens from text."""
    return re.findall(r'[a-z0-9]+', text.lower())


def extract_entities(text: str) -> List[str]:
    """
    Extract relevant business & commerce entities from post content
    with morphological normalization and intent expansion.
    """
    tokens = tokenize(text)
    stop_words = {
        "a", "an", "the", "and", "or", "on", "in", "at", "to", "for",
        "of", "with", "by", "all", "is", "are", "when", "you", "pay",
        "this", "that", "it", "from", "extra", "flat"
    }
    raw_entities = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    # Recognize multi-word phrases
    full_text = text.lower()
    known_multiwords = [
        "food delivery", "credit card", "debit card", "big billion",
        "bajaj auto", "hdfc bank", "phonepe upi", "two wheeler", "auto loan"
    ]
    for mw in known_multiwords:
        if mw in full_text:
            raw_entities.append(mw.replace(" ", "_"))

    # Intent & Lemma expansion
    expanded = set()
    for token in raw_entities:
        expanded.add(token)
        if token in LEMMA_MAP:
            expanded.add(LEMMA_MAP[token])
        if token in INTENT_SYNONYMS:
            for syn in INTENT_SYNONYMS[token]:
                expanded.add(syn)

    # Percentage off detection (e.g. "5% off" -> "discount", "offer")
    if "%" in text or "percent" in full_text or "off" in tokens:
        expanded.add("discount")
        expanded.add("offer")

    return sorted(list(expanded))


def compute_embedding(text: str) -> List[float]:
    """Computes a normalized dense vector embedding (length 64) for text."""
    vector = [0.0] * EMBEDDING_DIM
    tokens = tokenize(text)
    if not tokens:
        return vector

    # 1. Domain ontology projection
    cluster_idx = 0
    for cluster_name, keywords in DOMAIN_CLUSTERS.items():
        base_dim = cluster_idx * 5
        score = 0.0
        for token in tokens:
            for kw in keywords:
                if token == kw or (len(token) > 3 and token in kw) or (len(kw) > 3 and kw in token):
                    score += 1.0
        if score > 0:
            for d in range(5):
                weight = score * (1.0 / (d + 1))
                vector[base_dim + d] += weight
        cluster_idx += 1

    # 2. Sub-word & n-gram hashing
    hash_base = 35
    hash_dims = EMBEDDING_DIM - hash_base
    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        dim = hash_base + (h % hash_dims)
        vector[dim] += 1.0
        if len(token) >= 4:
            for i in range(len(token) - 2):
                sub = token[i:i+3]
                h_sub = int(hashlib.md5(sub.encode('utf-8')).hexdigest(), 16)
                dim_sub = hash_base + (h_sub % hash_dims)
                vector[dim_sub] += 0.4

    # 3. L2 Normalization
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude > 1e-9:
        vector = [round(x / magnitude, 5) for x in vector]
    else:
        vector = [0.0] * EMBEDDING_DIM

    return vector


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two unit vectors (range: 0.0 to 1.0)."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, round(dot, 4)))
