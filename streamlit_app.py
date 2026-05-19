import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Gemini AI Assistant")
st.caption("Powered by `gemini-2.5-flash` via the Google Gen AI SDK")

# 2. Secure API Key Retrieval
# First checks Streamlit Secrets (for production cloud), then fallback to local environment variables
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    # Fallback option allowing users to enter a key directly in the sidebar if missing
    st.sidebar.warning("⚠️ GEMINI_API_KEY missing from environment setup.")
    api_key = st.sidebar.text_input("Enter your Gemini API Key manually:", type="password")
    if not api_key:
        st.info("Please provide your API key in the sidebar to get started.")
        st.stop()

# 3. Initialize the Gen AI Client
@st.cache_resource
def get_genai_client(key):
    return genai.Client(api_key=key)

client = get_genai_client(api_key)

# 4. Initialize Chat Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 5. Display Past Conversation
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input and Logic
if user_prompt := st.chat_input("Ask me anything..."):
    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Generate streaming response from Gemini 2.5 Flash
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Reconstruct entire session context to keep the "Agent" context-aware
            # Transforming history into standard format for the API
            formatted_contents = []
            for msg in st.session_state.chat_history:
                formatted_contents.append(f"{msg['role'].capitalize()}: {msg['content']}")
            
            # Request a streaming response
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=formatted_contents
            )
            
            for chunk in response_stream:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")