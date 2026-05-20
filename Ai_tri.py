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

st.markdown("""
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .stAlert {margin-top: 0.5rem; margin-bottom: 0.5rem;}
        hr {margin-top: 1rem; margin-bottom: 1rem;}
        div[data-testid="stDataFrame"] table { font-family: monospace !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 LLM-AARS: Portfolio-Wide Multi-Agent Chain Ladder Diagnostics")
st.markdown("---")

conn = st.connection("postgresql", type="sql")

# SECURE API KEY DISCOVERY ENGINE
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "connections" in st.secrets and "gemini_api_key" in st.secrets["connections"]:
    api_key = st.secrets["connections"]["gemini_api_key"]

if not api_key:
    st.error("❌ Critical Configuration Missing: 'GEMINI_API_KEY' is unreadable inside your Secrets dashboard.")
    st.stop()

ai_client = genai.Client(api_key=api_key)

try:
    # =====================================================================
    # 2. DYNAMIC QUANTITATIVE DATA EXTRACTION ENGINE
    # =====================================================================
    tx_query = "SELECT * FROM claim_transactions;"
    tx_data = conn.query(tx_query, ttl=0)
    
    if tx_data.empty:
        st.warning("⚠️ Database table empty. Please seed your data tables via SQL.")
        st.stop()
        
    tx_data.columns = [col.lower() for col in tx_data.columns]
    loss_column = 'incremental_paid_loss' if 'incremental_paid_loss' in tx_data.columns else tx_data.select_dtypes(include=[np.number]).columns[-1]

    all_years = sorted(tx_data['origin_year'].unique().tolist())
    all_devs = sorted(tx_data['development_months'].unique().tolist())
    target_headers = [f"{all_devs[i]}-{all_devs[i+1]} Mo" for i in range(len(all_devs)-1)]

    inc_pivot = tx_data.pivot_table(index='origin_year', columns='development_months', values=loss_column, aggfunc='sum')
    inc_tri = inc_pivot.reindex(index=all_years, columns=all_devs)
    cum_tri = inc_tri.cumsum(axis=1)
    
    ldf_tri = pd.DataFrame(index=all_years, columns=target_headers)
    for i in range(len(all_devs)-1):
        ldf_tri[target_headers[i]] = cum_tri[all_devs[i+1]] / cum_tri[all_devs[i]]

    # --- ACTUARIAL AGENT: VOLUME-WEIGHTED BASELINE LDF (OVERALL ANCHOR) ---
    cl_benchmarks = {}
    for i, col in enumerate(target_headers):
        t_prev = all_devs[i]
        t_curr = all_devs[i+1]
        mask = cum_tri[t_curr].notna() & cum_tri[t_prev].notna()
        cl_benchmarks[col] = cum_tri.loc[mask, t_curr].sum() / cum_tri.loc[mask, t_prev].sum()

    # =====================================================================
    # 3. SIDEBAR INTERFACE COORDINATOR (Controls the SHAP Deep Dive view)
    # =====================================================================
    st.sidebar.header("🔍 Deep Dive Coordinator")
    selected_col = st.sidebar.selectbox("Select Dev Period Column for SHAP Breakdown:", target_headers, index=0)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Force Multi-Agent Recalculation", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # =====================================================================
    # 4. OVERALL PORTFOLIO AI AGENT PARSER (Synthesizing Full Columns)
    # =====================================================================
    notes_all_query = "SELECT * FROM claim_notes;"
    notes_all_df = conn.query(notes_all_query, ttl=0)
    if not notes_all_df.empty:
        notes_all_df.columns = [c.lower() for c in notes_all_df.columns]

    llm_suggestions = {}
    llm_reasons = {}
    agent_shap_contributions = {} 

    with st.spinner("🤖 AI Reserving Agent is synthesizing portfolio logs across all historical years..."):
        for i, col in enumerate(target_headers):
            dev_month_target = all_devs[i+1]
            col_baseline = cl_benchmarks[col]
            
            # THE SYNTHESIS FIX: Extract notes for ALL years in this development slice
            if not notes_all_df.empty:
                column_notes = notes_all_df[notes_all_df['development_months'] == dev_month_target]
                
                # Format each historical entry so Gemini knows which year it came from
                formatted_notes_list = []
                for _, row in column_notes.iterrows():
                    formatted_notes_list.append(f"[Year {row['origin_year']}]: {row['note_text']}")
                combined_text = "\n".join(formatted_notes_list).strip()
            else:
                combined_text = ""

            agent_delta = 0.0
            reason_string = "Stable trend: Development matches historical patterns; standard parameters appropriate."

            if combined_text:
                prompt = f"""
                You are an expert Casualty Actuarial Reserving AI Agent reviewing an entire portfolio.
                Analyze these unstructured adjuster diaries collected across ALL historical origin years for the {col} maturity window.
                
                HISTORICAL NOTES LOG:
                {combined_text}
                
                YOUR TASK:
                1. Look across all historical years to spot systemic drifts, inflation issues, or operational processing shifts.
                2. Determine an overall 'risk_loading_coefficient' (a positive or negative float decimal shift) to adjust the baseline average:
                   - For systemic cost inflation, recurring severe weather/fire patterns, or a high volume of late-reopened files, return a positive factor shift (+0.04 to +0.35).
                   - For consistent subrogation recoveries or processing settlement speed-ups across multiple years, return a negative factor shift (-0.04 to -0.25).
                   - If anomalies cancel out or are purely isolated events, return 0.00.
                3. Provide a professional 'audit_rationale' summarizing the overall trend, the metric shift, and your justification.
                
                You must return your response inside a valid JSON object matching this schema:
                {{
                  "risk_loading_coefficient": float,
                  "audit_rationale": "string"
                }}
                """
                
                try:
                    response = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    
                    result_data = json.loads(response.text)
                    agent_delta = float(result_data.get("risk_loading_coefficient", 0.0))
                    reason_string = f"AI Selection ({agent_delta:+.2f}): " + str(result_data.get("audit_rationale", ""))
                except Exception as ai_err:
                    reason_string = f"AI Agent Bypass: Fallback applied. Trace: {ai_err}"
                    agent_delta = 0.0

            # Apply modification directly over the overall volume-weighted benchmark
            llm_suggestions[col] = col_baseline + agent_delta
            llm_reasons[col] = reason_string
            agent_shap_contributions[col] = agent_delta

    # =====================================================================
    # 5. SUMMARY MATRIX COMPILATION LAYER
    # =====================================================================
    summary_matrix = ldf_tri.copy().map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    summary_matrix.index = summary_matrix.index.astype(str)
    
    # Chronological Rolling N-Year Averages
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

    # Row 1: The standard actuarial baseline anchor calculation
    summary_matrix.loc["LDF"] = [f"{cl_benchmarks[c]:.2f}" for c in target_headers]
    # Row 2: The overall target calculated dynamically by the synthesized AI Agent
    summary_matrix.loc["LLM Suggested"] = [f"{llm_suggestions[c]:.2f}" for c in target_headers]
    # Row 3: The complete audit trail rationale detailing the drift type and justification
    summary_matrix.loc["Reason"] = [llm_reasons[c] for c in target_headers]

    st.subheader("📊 Portfolio-Wide Loss Development Grid Matrix (Synthesized Agent Engine)")
    st.dataframe(summary_matrix, use_container_width=True)

    # =====================================================================
    # 6. SHAP WATERFALL ATTRIBUTION VISUALIZER
    # =====================================================================
    st.write("---")
    st.subheader(f"🤖 Portfolio-Wide SHAP Explanation Summary: {selected_col} Window")
    
    base_anchor = cl_benchmarks[selected_col]
    final_target = llm_suggestions[selected_col]
    ai_calculated_delta = agent_shap_contributions[selected_col]
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown(f"### 🛡️ Defensibility Audit: **{selected_col} Overall**")
        st.info(f"**Portfolio Volume-Weighted LDF Baseline:** `{base_anchor:.4f}`")
        st.success(f"**Final Synthesized LLM Selection:** `{final_target:.4f}`")
        
        st.markdown(f"""
        **AI Decision Logic Decomposition:**
        * **Actuarial Baseline Average:** `{base_anchor:.4f}`
        * **Synthesized Portfolio Delta:** `{ai_calculated_delta:+.4f}`
        * **Final Defensible LDF Target:** `{final_target:.4f}`
        
        **Synthesized Justification & Strategic Audit Trail:**
        > \"*{llm_reasons[selected_col]}*\"
        """)
        
    with col_right:
        shap_labels = ["Portfolio Baseline LDF"]
        shap_deltas = [base_anchor]
        
        if abs(ai_calculated_delta) > 0.0001:
            shap_labels.append("Synthesized Agent Delta")
            shap_deltas.append(ai_calculated_delta)
            
        graph_x = shap_labels + ["Final Suggested Target"]
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
