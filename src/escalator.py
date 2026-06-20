import json

def generate_handoff_summary(user_query: str, persona: str, context_chunks: list, trigger_reason: str) -> dict:
    """Compiles a clean, highly structured JSON handoff report for support dispatching."""
    return {
        "persona": persona,
        "detected_issue": user_query[:120] + "..." if len(user_query) > 120 else user_query,
        "escalation_reason": trigger_reason,
        "retrieved_sources": list(set([c["source"] for c in context_chunks])) if context_chunks else [],
        "confidence_score": max([c["score"] for c in context_chunks]) if context_chunks else 0.0,
        "recommended_action": "Route ticket immediately to specialized representative for audit adjustment."
    }

def check_escalation_criteria(user_query: str, best_score: float, confidence_threshold: float) -> tuple[bool, str]:
    """Evaluates whether automation rules warrant a live agent transition."""
    sensitive_keywords = ["refund", "chargeback", "dispute", "duplicate charge", "legal"]
    
    if any(word in user_query.lower() for word in sensitive_keywords):
        return True, "Sensitive operational topic detected (Billing/Refund Integrity)."
        
    if best_score < confidence_threshold:
        return True, f"Retrieval confidence score ({best_score}) dropped below runtime threshold ({confidence_threshold})."
        
    return False, ""