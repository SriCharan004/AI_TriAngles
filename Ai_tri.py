import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from google.genai import types
import json

# =====================================================================
# 1. PAGE CONFIGURATION & LAYOUT SPECIFICATIONS
# =====================================================================
st.set_page_config(
    page_title="LLM-AARS Reserving Diagnostics",
    page_icon="📐",
    layout="wide"
)

# Custom structural CSS to maximize table scannability and layout typography
st.markdown("""
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .stAlert {margin-top: 0.5rem; margin-bottom: 0.5rem;}
        hr {margin-top: 1rem; margin-bottom: 1rem;}
        div[data-testid="stDataFrame"] table { font-family: monospace !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 LLM-AARS: Live Multi-Agent Chain Ladder Diagnostics")
st.markdown("---")

# Initialize the cloud database connection using configurations in .streamlit/secrets.toml
conn = st.connection("postgresql", type="sql")

# =====================================================================
# 2. SECURE API KEY DISCOVERY ENGINE (With Fallback Logic)
# =====================================================================
api_key = None

# Check 1: Try reading from Streamlit Secrets Management Cloud Engine (Standard Uppercase)
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
# Check 2: Try falling back to a nested configuration block if specified differently
elif "connections" in st.secrets and "gemini_api_key" in st.secrets["connections"]:
    api_key = st.secrets["connections"]["gemini_api_key"]
elif "connections" in st.secrets and "GEMINI_API_KEY" in st.secrets["connections"]:
    api_key = st.secrets["connections"]["GEMINI_API_KEY"]

# If all secure lookups fail, halt gracefully with explicit setup instructions
if not api_key:
    st.error("""
    ❌ **Critical Configuration Missing: 'GEMINI_API_KEY' is unreadable.**
    
    **How to fix this error on Streamlit Cloud:**
    1. Go to your **Streamlit Community Cloud Dashboard**.
    2. Click the three vertical dots (**...**) next to your running app and select **Settings** ➡️ **Secrets**.
    3. Ensure your panel text environment matches this exact case-sensitive layout:
    ```toml
    GEMINI_API_KEY = "your_actual_api_key_here"
    ```
    4. Click **Save** and wait for the app to auto-reload.
    """)
    st.stop()

# Initialize the live Google GenAI Client securely
ai_client = genai.Client(api_key=api_key)

try:
    # =====================================================================
    # 3. DYNAMIC QUANTITATIVE DATA EXTRACTION ENGINE
    # =====================================================================
    tx_query = "SELECT * FROM claim_transactions;"
    tx_data = conn.query(tx_query, ttl=0)
    
    if tx_data.empty:
        st.warning("⚠️ Database table 'claim_transactions' is currently empty. Please execute your database seeding script.")
        st.stop()
        
    # Standardize schema mappings to lower-case characters safely
    tx_data.columns = [col.lower() for col in tx_data.columns]
    
    # Auto-detect target column identifying payment volume
    loss_column = None
    for possible_name in ['incremental_paid_loss', 'incremental_loss', 'paid_loss', 'incremental_paid']:
        if possible_name in tx_data.columns:
            loss_column = possible_name
            break
            
    if not loss_column:
        loss_column = tx_data.select_dtypes(include=[np.number]).columns[-1]

    # --- INTROSPECTION LAYER: AUTOMATICALLY EXTRACT DIMENSIONS ---
    all_years = sorted(tx_data['origin_year'].unique().tolist())
    all_devs = sorted(tx_data['development_months'].unique().tolist())
    target_headers = [f"{all_devs[i]}-{all_devs[i+1]} Mo" for i in range(len(all_devs)-1)]

    # Pivot incremental financials and roll up into cumulative actuarial triangles
    inc_pivot = tx_data.pivot_table(index='origin_year', columns='development_months', values=loss_column, aggfunc='sum')
    inc_tri = inc_pivot.reindex(index=all_years, columns=all_devs)
    cum_tri = inc_tri.cumsum(axis=1)
    
    # Generate true historical Age-to-Age link ratio matrix
    ldf_tri = pd.DataFrame(index=all_years, columns=target_headers)
    for i in range(len(all_devs)-1):
        ldf_tri[target_headers[i]] = cum_tri[all_devs[i+1]] / cum_tri[all_devs[i]]

    # --- ACTUARIAL AGENT: VOLUME-WEIGHTED CHAIN LADDER BENCHMARKS ---
    cl_benchmarks = {}
    for i, col in enumerate(target_headers):
        t_prev = all_devs[i]
        t_curr = all_devs[i+1]
        mask = cum_tri[t_curr].notna() & cum_tri[t_prev].notna()
        cl_benchmarks[col] = cum_tri.loc[mask, t_curr].sum() / cum_tri.loc[mask, t_prev].sum()

    # =====================================================================
    # 4. SIDEBAR INTERFACE COORDINATOR
    # =====================================================================
    st.sidebar.header("🔍 Deep Dive Coordinator")
    selected_oy = st.sidebar.selectbox("Select Origin Year for Analysis Row:", all_years, index=len(all_years)-2 if len(all_years) > 1 else 0)
    selected_col = st.sidebar.selectbox("Select Dev Period Column for SHAP Breakdown:", target_headers, index=0)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Force Multi-Agent Recalculation", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # =====================================================================
    # 5. LIVE COGNITIVE AI AGENT COURIER LAYER (Structured JSON Schema Engine)
    # =====================================================================
    notes_all_query = "SELECT * FROM claim_notes;"
    notes_all_df = conn.query(notes_all_query, ttl=0)
    if not notes_all_df.empty:
        notes_all_df.columns = [c.lower() for c in notes_all_df.columns]

    llm_suggestions = {}
    llm_reasons = {}
    agent_shap_contributions = {} 

    # Implement progress indicator wrapper across live API generations
    with st.spinner("🤖 Multi-Agent Engine is mining claims text logs via Gemini Pro..."):
        for i, col in enumerate(target_headers):
            dev_month_target = all_devs[i+1]
            col_baseline = cl_benchmarks[col]
            
            # Isolate text segment to match chosen cell intersection (Row & Column matching)
            if not notes_all_df.empty:
                cell_notes = notes_all_df[
                    (notes_all_df['development_months'] == dev_month_target) & 
                    (notes_all_df['origin_year'] == selected_oy)
                ]
                combined_text = " ".join(cell_notes['note_text'].astype(str)).strip() if not cell_notes.empty else ""
            else:
                combined_text = ""

            # Standard defaults if no unstructured notes exist for the cell coordinate intersection
            agent_delta = 0.0
            reason_string = "Stable trend: Development matches historical patterns; standard parameters appropriate."

            # If matching text diaries exist, prompt Gemini to calculate cognitive risk metrics
            if combined_text:
                prompt = f"""
                You are an expert Casualty Actuarial Pricing and Reserving AI Agent. 
                Analyze these unstructured adjuster diary logs for Origin Year {selected_oy} during the {col} maturity development step.
                
                RAW NOTES FROM ADJUSTERS:
                "{combined_text}"
                
                YOUR TASK:
                1. Determine the systemic risk impact of this text on our standard Link Ratio calculation.
                2. Calculate an appropriate 'risk_loading_coefficient' (a floating point decimal shift value).
                   - For inflation, catastrophe, or late re-opened claims, return a positive number between +0.02 and +0.35 depending on severity.
                   - For subrogation recovery cash inflows or process settlement acceleration, return a negative number between -0.02 and -0.25.
                3. Write a professional, concise 'audit_rationale' sentence summarizing your decision.
                
                You must return your response inside a valid JSON object matching this schema:
                {{
                  "risk_loading_coefficient": float,
                  "audit_rationale": "string"
                }}
                """
                
                try:
                    # Execute structured json instruction using the official 2.0 SDK client
                    response = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    
                    # Parse token outputs safely
                    result_data = json.loads(response.text)
                    agent_delta = float(result_data.get("risk_loading_coefficient", 0.0))
                    reason_string = f"AI Agent Selection (+{agent_delta:+.2f}): " + str(result_data.get("audit_rationale", ""))
                except Exception as ai_err:
                    reason_string = f"AI Agent Evaluation Bypass: Fallback applied. Trace: {ai_err}"
                    agent_delta = 0.0

            llm_suggestions[col] = col_baseline + agent_delta
            llm_reasons[col] = reason_string
            agent_shap_contributions[col] = agent_delta

    # =====================================================================
    # 6. SUMMARY MATRIX COMPILATION LAYER
    # =====================================================================
    summary_matrix = ldf_tri.copy().map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    summary_matrix.index = summary_matrix.index.astype(str)
    
    # --- CHRONOLOGICAL ROLLING N-YEAR AVERAGE ENG COMPILATION ---
    max_averages_needed = 3
    for n in range(1, max_averages_needed + 1):
        rolling_row_values = []
        for col in target_headers:
            valid_factors = ldf_tri[col].dropna().tolist()
            if len(valid_factors) >= n:
                rolling_row_values.append(f"{np.mean(valid_factors[-n:]):.4f}")
            else:
                rolling_row_values.append("-")
        summary_matrix.loc[f"{n} year"] = rolling_row_values

    # Merge dynamic baselines and calculations into a single table
    summary_matrix.loc["LDF"] = [f"{cl_benchmarks[c]:.2f}" for c in target_headers]
    summary_matrix.loc["LLM Suggested"] = [f"{llm_suggestions[c]:.2f}" for c in target_headers]
    summary_matrix.loc["Reason"] = [llm_reasons[c] for c in target_headers]

    st.subheader(f"📊 Live Loss Development Grid Matrix (Active Focus Row: Origin Year {selected_oy})")
    st.dataframe(summary_matrix, use_container_width=True)

    # =====================================================================
    # 7. SHAP WATERFALL ATTRIBUTION VISUALIZER
    # =====================================================================
    st.write("---")
    st.subheader(f"🤖 Agent SHAP Explanation Summary for Profile Column: {selected_col}")
    
    base_anchor = cl_benchmarks[selected_col]
    final_target = llm_suggestions[selected_col]
    ai_calculated_delta = agent_shap_contributions[selected_col]
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown(f"### 🛡️ Defensibility Narrative: **{selected_col} Window**")
        st.info(f"**Standard Chain Ladder Baseline Average:** `{base_anchor:.4f}`")
        st.success(f"**Final LLM Suggested Selection:** `{final_target:.4f}`")
        
        st.markdown(f"""
        **AI Decision Logic Decomposition:**
        * **Actuarial Baseline Vector:** `{base_anchor:.4f}`
        * **AI Agent Extracted Delta:** `{ai_calculated_delta:+.4f}`
        * **Mathematical Final Selection:** `{final_target:.4f}`
        
        **Audit Trail Defensibility Narrative:**
        > \"*{llm_reasons[selected_col]}*\"
        """)
        
    with col_right:
        shap_labels = ["Chain Ladder Baseline"]
        shap_deltas = [base_anchor]
        
        if abs(ai_calculated_delta) > 0.0001:
            shap_labels.append("AI Agent Calculated Delta")
            shap_deltas.append(ai_calculated_delta)
            
        graph_x = shap_labels + ["Final Selected LDF"]
        graph_y = shap_deltas + [final_target]
        measures = ["relative"] * len(shap_deltas) + ["total"]
        
        fig = go.Figure(go.Waterfall(
            orientation="v", 
            measure=measures, 
            x=graph_x, 
            y=graph_y,
            textposition="outside",
            text=[f"{d:+.2f}" if m == "relative" and i > 0 else f"{d:.2f}" for i, (d, m) in enumerate(zip(graph_y, measures))],
            connector={"line": {"color": "rgb(63, 63, 63)", "dash": "dot"}},
            decreasing={"marker": {"color": "#2ca02c"}}, 
            increasing={"marker": {"color": "#d62728"}}, 
            totals={"marker": {"color": "#1f77b4"}}       
        ))
        
        fig.update_layout(
            yaxis_title="LDF Factor Scale", 
            margin=dict(t=15, b=15, l=15, r=15), 
            height=380, 
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Operational Interruption: {e}")
