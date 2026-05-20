import os
import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(
    page_title="Reserving & Pricing Actuarial Agent", 
    page_icon="📊", 
    layout="wide"
)

# 2. Secure API Key Retrieval
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.sidebar.warning("⚠️ GEMINI_API_KEY missing from environment setup.")
    api_key = st.sidebar.text_input("Enter your Gemini API Key manually:", type="password")
    if not api_key:
        st.info("Please provide your API key to get started.")
        st.stop()

# Initialize Gen AI Client
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# 3. Sidebar - Persona Settings & Frameworks
st.sidebar.header("🛡️ Actuarial Agent Controls")
st.sidebar.markdown(
    "This agent is specialized in **P&C Actuarial Science**, including "
    "loss reserving, pricing, and stochastic data frameworks."
)

# Actuarial System Instructions
ACTUARIAL_SYSTEM_INSTRUCTION = (
    "You are an expert, credentialed Property & Casualty (P&C) Actuarial AI Agent. "
    "Your expertise includes traditional loss reserving methodologies (Chain Ladder, "
    "Bornhuetter-Ferguson, Cape Cod), advanced predictive modeling (Generalized Linear Models - GLMs), "
    "and stochastic reserving frameworks. "
    "Provide rigorous, technically accurate, and professional explanations. "
    "Where mathematical formulas or structural derivations are requested, present them clearly "
    "using standard notation. Maintain a professional, peer-level consulting tone."
)

# 4. Layout: Main Interface Split into Tabs
st.title("📊 Reserving & Pricing Actuarial Agent")
tab1, tab2 = st.tabs(["💬 Actuarial Consultation Chat", "📐 Quick Reference Guide"])

# --- TAB 1: Conversational Chat Interface ---
with tab1:
    # Initialize Chat Session State
    if "actuarial_chat_history" not in st.session_state:
        st.session_state.actuarial_chat_history = []

    # Display Past Conversation
    for message in st.session_state.actuarial_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input and Logic
    if user_prompt := st.chat_input("Ask about loss reserving, premium pricing, GLMs..."):
        # Render user message
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        st.session_state.actuarial_chat_history.append({"role": "user", "content": user_prompt})

        # Generate Streaming Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Reconstruct chat log history
                formatted_contents = []
                for msg in st.session_state.actuarial_chat_history:
                    formatted_contents.append(f"{msg['role'].capitalize()}: {msg['content']}")
                
                # Request a streaming response incorporating System Instructions
                response_stream = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=ACTUARIAL_SYSTEM_INSTRUCTION,
                        temperature=0.2, # Lower temperature for more accurate, analytical outputs
                    )
                )
                
                for chunk in response_stream:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                    
                response_placeholder.markdown(full_response)
                st.session_state.actuarial_chat_history.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# --- TAB 2: Quick Reference Guide ---
with tab2:
    st.header("Core Actuarial Methodologies")
    st.markdown(
        "Use these prompt templates in the consultation tab to test the agent's analytical depth:"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reserving & Development")
        st.info(
            "**Example Prompt:**\n"
            "\"Explain the mechanical differences between the Chain Ladder and Bornhuetter-Ferguson "
            "reserving methods. Under what circumstances would an actuary put 100% weight on the BF method?\""
        )
    with col2:
        st.subheader("Predictive Modeling & Pricing")
        st.info(
            "**Example Prompt:**\n"
            "\"Draft a clean Python blueprint using `statsmodels` to fit a Generalized Linear Model (GLM) "
            "for insurance claim frequency. Assume a Poisson distribution with a log link function, and "
            "include an exposure offset variable.\""
        )