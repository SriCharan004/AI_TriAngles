import os
import sqlite3
import ast
import streamlit as st
from google import genai
from google.genai import types

# --- 1. PAGE SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="Actuarator: Token-Optimized Risk Pipeline",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. PRIVATE CREDENTIALING SECRETS ---
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.sidebar.error("❌ GEMINI_API_KEY is completely missing from your configuration workspace.")
    st.stop()

@st.cache_resource
def initialize_gemini_client(key):
    return genai.Client(api_key=key)

client = initialize_gemini_client(api_key)

# --- 3. SEEDING THE DETERMINISTIC ACTUARIAL DATABASE ---
DB_NAME = "actuarial_firm.db"

@st.cache_resource
def build_and_seed_sqlite_db():
    """Initializes local database tables exactly following your notebook's logic."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS loan_portfolios (
            loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            principal_amount REAL,
            annual_rate REAL,
            tenure_years INTEGER,
            risk_score TEXT
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM loan_portfolios")
    if cursor.fetchone()[0] == 0:
        # Seed core fallback evaluation values to test system sanity bounds
        mock_records = [
            ("Peter Farrell PhD", 987000.0, 0.120, 10, "Critical"),
            ("Kenneth Walls", 4020000.0, 0.130, 5, "High"),
            ("Victoria Hanna", 4562000.0, 0.139, 10, "Critical"),
            ("Carl Perez", 2796000.0, 12.50, 5, "High")  # High-yield/catastrophic boundary asset
        ]
        cursor.executemany('''
            INSERT INTO loan_portfolios (customer_name, principal_amount, annual_rate, tenure_years, risk_score)
            VALUES (?, ?, ?, ?, ?)
        ''', mock_records)
        conn.commit()
    conn.close()

build_and_seed_sqlite_db()

# --- 4. STREAMLINED, TOKEN-EFFICIENT ACTUARIAL AGENT ACTIONS (No LLM Overhead) ---
def native_portfolio_extractor(loan_id: int) -> dict:
    """TOKEN-SAVER: Pulls structured metadata natively from SQLite without an intermediate LLM."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT customer_name, principal_amount, annual_rate, tenure_years, risk_score FROM loan_portfolios WHERE loan_id = ?", (loan_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "id": loan_id, "name": row[0], "principal": row[1], 
        "rate": row[2], "tenure": row[3], "risk": row[4]
    }

# --- 5. STREAMLIT APP VIEW LAYOUT ---
st.title("⚖️ Actuarator: Credit Asset Risk & Compliance Pipeline")
st.markdown("Automated inter-reporting compliance orchestration engine optimized to reduce token burn rates.")

# Sidebar Configuration Control Panel
st.sidebar.header("🛠️ Pipeline Parameters")
target_id = st.sidebar.number_input("Target Credit Asset Loan ID:", min_value=1, max_value=4, value=4, step=1)
run_pipeline = st.sidebar.button("Trigger Downstream Pipeline Execution")

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Token Optimization Active:** Direct relational memory extraction completely circumvents intermediate parsing layers.")

# Execution Sandbox Area
if run_pipeline:
    with st.spinner("Extracting parameters and coordinating multi-agent risk calculations..."):
        
        # STEP 1: Direct System Memory Extraction
        asset = native_portfolio_extractor(target_id)
        
        if not asset:
            st.error(f"Loan Asset ID {target_id} could not be successfully isolated inside current database schemas.")
        else:
            # Render Clean UI representation of the extracted metadata layer
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Policyholder Name", value=asset["name"])
            with col2:
                st.metric(label="Total Contract Principal", value=f"${asset['principal']:,.2f}")
            with col3:
                st.metric(label="Stated Yield Rate", value=f"{asset['rate']*100:.2f}%")
                
            st.markdown("---")
            
            # --- AGENT STAGE 2: SYSTEM WORKER PRICING EVALUATION ---
            st.subheader("⚡ Section 1: Actuarial Risk Underwriting Audit")
            
            # Clean compressed context framework string pass to Gemini
            pricing_prompt = (
                f"Analyze this credit asset underwriting contract:\n"
                f"- Client: {asset['name']}\n"
                f"- Principal Size: ${asset['principal']:,.2f}\n"
                f"- Horizon Term: {asset['tenure']} Years\n"
                f"- Stated Interest Rate: {asset['rate']*100:.2f}%\n"
                f"- Risk Classification: {asset['risk']}\n\n"
                f"Verify if the Stated Interest Rate logically aligns with underwriting targets "
                f"(Targets: Low >5%, Medium >8%, High >12%, Critical >18%). Evaluate structural risk "
                f"implications given principal constraints. Keep output tightly synthesized."
            )
            
            # Low temperature enforces precise, non-drift deterministic completions
            pricing_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=pricing_prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            pricing_text = pricing_response.text
            st.info(pricing_text)
            
            # --- AGENT STAGE 3: FINANCIAL SOLVENCY BUFFER MODELER ---
            st.subheader("🛡️ Section 2: Solvency Buffer Reserve Modeling")
            
            capital_prompt = (
                f"Review the downstream actuarial critique:\n\"{pricing_text}\"\n\n"
                f"Enforce Capital Allocation Baseline Directive:\n"
                f"- Mandate a high-risk 15% Economic Capital Solvency Buffer ONLY if explicit 'High', "
                f"'Critical', 'catastrophic', or 'unviable' default exposures are declared.\n"
                f"- Otherwise, default to a standard baseline cushion of 8%.\n\n"
                f"State the allocated capital percentage and state the exact pricing parameter logic."
            )
            
            capital_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=capital_prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            capital_text = capital_response.text
            st.warning(capital_text)
            
            # --- AGENT STAGE 4: CHIEF RISK EXECUTIVE SYNTHESIS ---
            st.subheader("📊 Section 3: Executive Board Risk Summary")
            
            # Synthesis layer combines structural notes without leaking conversation history loops
            executive_synthesis_prompt = (
                f"You are the Chief Actuary. Synthesize these active calculations into an actionable "
                f"three-sentence executive risk disclosure block tailored for Board of Directors review:\n"
                f"1. Asset Identity: Loan ID {asset['id']}, Policyholder {asset['name']}.\n"
                f"2. Audit Findings: {pricing_text}\n"
                f"3. Capital Buffer Decisions: {capital_text}"
            )
            
            final_synthesis = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=executive_synthesis_prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            
            st.success(final_synthesis.text)
else:
    st.info("👈 Use the dashboard control rail to select an active asset ID and coordinate the risk pipeline processing chain.")