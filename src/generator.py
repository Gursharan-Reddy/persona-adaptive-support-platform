import os
from google import genai
from google.genai import types

def build_adaptive_response(user_query: str, persona: str, context_chunks: list) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

    # Assign contextual personas matching requirement objectives
    if persona == "Technical Expert":
        persona_instructions = (
            "You are a Senior Systems Engineer. Provide clear root-cause analysis, "
            "configuration specifications, and precise API pathways or code blocks. "
            "Keep technical descriptions exact, explicit, and structured."
        )
    elif persona == "Frustrated User":
        persona_instructions = (
            "You are a deeply empathetic, reassuring Customer Care Specialist. "
            "Begin with a warm, genuine validation of their difficulty. Use straightforward, "
            "reassuring, and simple action-oriented bullet steps. Avoid complex engineering jargon."
        )
    else:  # Business Executive
        persona_instructions = (
            "You are a concise Client Relations Director. Focus on direct business outcomes, "
            "impact summaries, and timelines for resolution. Keep responses extremely "
            "brief, professional, and skip unnecessary structural config details."
        )

    context_text = "\n\n".join([f"Source [{c['source']}]: {c['text']}" for c in context_chunks])

    full_system_prompt = (
        f"{persona_instructions}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Formulate responses strictly utilizing the verified facts present inside the document context.\n"
        "- If the answer is unmentioned, clearly report that the instruction details are unavailable.\n\n"
        f"VERIFIED HISTORICAL DATA CONTEXT:\n{context_text}"
    )

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            temperature=0.2
        )
    )
    return response.text