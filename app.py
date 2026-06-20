import streamlit as st
from dotenv import load_dotenv
import json

# Load environment configuration variables
load_dotenv()

from src.classifier import classify_customer_persona
from src.rag_pipeline_v2 import LocalRAGPipeline
from src.escalator import check_escalation_criteria, generate_handoff_summary
from src.generator import build_adaptive_response
from src.config import COSMIC_CONFIDENCE_THRESHOLD

st.set_page_config(page_title="Persona Adaptive AI Support", layout="wide")

# Persistent Application Initialization State
if "rag_pipeline" not in st.session_state:
    with st.spinner("Initializing Database System Context Chunks..."):
        pipeline = LocalRAGPipeline()
        pipeline.ingest_data_directory()
        st.session_state["rag_pipeline"] = pipeline

st.title("🤖 Persona-Adaptive Customer Support Platform")
st.caption("A dynamic support loop matching responses directly to user characteristics.")
st.write("---")

user_input = st.text_area("Input Customer Message Scenario Here:", height=100, 
                          placeholder="Type your troubleshooting issue, system request, or escalation demand...")

if st.button("Execute Pipeline Lifecycle"):
    if not user_input.strip():
        st.warning("Please enter a valid support query text.")
    else:
        # 1. Classification Phase
        with st.spinner("Analyzing communication persona style..."):
            classification = classify_customer_persona(user_input)
            detected_persona = classification.get("persona", "Frustrated User")
            classification_reason = classification.get("reasoning", "")
        
        # 2. Vector Retrieval Lookup
        with st.spinner("Querying vector index metadata collections..."):
            retrieved_chunks = st.session_state["rag_pipeline"].retrieve_context(user_input, top_k=2)
            best_score = max([c["score"] for c in retrieved_chunks]) if retrieved_chunks else 0.0

        # 3. Automation Escalation Guard Validation
        is_escalated, escalation_reason = check_escalation_criteria(user_input, best_score, COSMIC_CONFIDENCE_THRESHOLD)

        # UI Diagnostic Columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📋 Pipeline Analysis Metrics")
            st.metric(label="Identified Persona Class", value=detected_persona)
            st.write(f"**Classification Reason:** {classification_reason}")
            
            st.markdown("---")
            st.write("**Top Relevant Reference Sources Retrieved:**")
            if retrieved_chunks:
                for chunk in retrieved_chunks:
                    st.info(f"📄 **Source:** {chunk['source']} (Match Score: {chunk['score']})\n\n{chunk['text']}")
            else:
                st.warning("No context components extracted from storage.")

        with col2:
            st.subheader("⚡ Generation Engine Response Layout")
            
            if is_escalated:
                st.error(f"🚨 **System Escalation Triggered!**\n\n*Reason:* {escalation_reason}")
                
                # Formulate structural human ticket data
                handoff_report = generate_handoff_summary(user_input, detected_persona, retrieved_chunks, escalation_reason)
                
                st.write("### 📄 Formatted Human Agent Ticket Handoff Payload (JSON)")
                st.code(json.dumps(handoff_report, indent=2), language="json")
            else:
                st.success("✅ **Query Processed Automatically Within System Guidelines**")
                
                # 4. Generate Adaptive Response
                with st.spinner("Compiling tailored text generation structures..."):
                    agent_response = build_adaptive_response(user_input, detected_persona, retrieved_chunks)
                
                st.write("### 💬 Personal-Adapted Bot Response:")
                st.markdown(agent_response)