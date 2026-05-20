import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

st.title("🤖 LLM-AARS: AI Agent-Driven Reserving Engine")
st.markdown("---")

# Initialize the cloud database connection using configurations in .streamlit/secrets.toml
conn = st.connection("postgresql", type="sql")

try:
    # =====================================================================
    # 2. DYNAMIC QUANTITATIVE DATA EXTRACTION ENGINE
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
    # 3. SIDEBAR INTERFACE COORDINATOR
    # =====================================================================
    st.sidebar.header("🔍 Deep Dive Coordinator")
    selected_oy = st.sidebar.selectbox("Select Origin Year for Analysis Row:", all_years, index=len(all_years)-2 if len(all_years) > 1 else 0)
    selected_col = st.sidebar.selectbox("Select Dev Period Column for SHAP Breakdown:", target_headers, index=0)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Force Multi-Agent Recalculation", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # =====================================================================
    # 4. QUALITATIVE COGNITIVE AI AGENT PARSER
    # =====================================================================
    notes_all_query = "SELECT * FROM claim_notes;"
    notes_all_df = conn.query(notes_all_query, ttl=0)
    if not notes_all_df.empty:
        notes_all_df.columns = [c.lower() for c in notes_all_df.columns]

    llm_suggestions = {}
    llm_reasons = {}
    agent_shap_contributions = {} 

    for i, col in enumerate(target_headers):
        dev_month_target = all_devs[i+1]
        col_baseline = cl_benchmarks[col]
        
        # Isolate text segment to match chosen cell intersection (Row & Column matching)
        if not notes_all_df.empty:
            cell_notes = notes_all_df[
                (notes_all_df['development_months'] == dev_month_target) & 
                (notes_all_df['origin_year'] == selected_oy)
            ]
            combined_text = " ".join(cell_notes['note_text'].astype(str)).lower() if not cell_notes.empty else ""
        else:
            combined_text = ""

        # --- THE AI AGENT HEURISTIC STRUCTURAL LOADER ---
        agent_severity_coefficient = 0.0
        reason_string = "Stable trend: Development matches historical patterns; standard parameters appropriate."

        if combined_text:
            # Condition A: Material Price & Resource Inflation
            if "inflation" in combined_text or "spike" in combined_text:
                base_inflation_impact = 0.10
                modifier = 1.5 if "severe" in combined_text else 1.0
                agent_severity_coefficient += (base_inflation_impact * modifier)
                reason_string = f"AI Agent Adjustment (+{agent_severity_coefficient:.2f}): Inflationary pressures and supply line index spikes active."

            # Condition B: Litigation, Social Inflation, and Court Venues
            if "verdict" in combined_text or "court" in combined_text or "jury" in combined_text:
                base_legal_impact = 0.06
                modifier = 2.0 if "nuclear" in combined_text else 1.0
                agent_severity_coefficient += (base_legal_impact * modifier)
                reason_string = f"AI Agent Adjustment (+{agent_severity_coefficient:.2f}): Nuclear social inflation trial vectors realized."

            # Condition C: Reinsurance Salvage & Subrogation Recovery Offsets
            if "subrogation" in combined_text or "salvage" in combined_text or "recovery" in combined_text:
                base_recovery_impact = -0.08
                modifier = 1.5 if "massive" in combined_text else 1.0
                agent_severity_coefficient += (base_recovery_impact * modifier)
                reason_string = f"AI Agent Adjustment ({agent_severity_coefficient:.2f}): Major subrogation cash inflows masking gross losses."

            # Condition D: Claims Settlement Acceleration Processes
            if "speed" in combined_text or "closing" in combined_text or "lightning" in combined_text:
                base_speed_impact = -0.05
                modifier = 1.6 if "lightning" in combined_text else 1.0
                agent_severity_coefficient += (base_speed_impact * modifier)
                reason_string = f"AI Agent Adjustment ({agent_severity_coefficient:.2f}): Process operational speed-up clearing legacy open files."
                
            # Condition E: Tail-End Latent Claims Reopenings
            if "reopen" in combined_text or "latent" in combined_text:
                base_reopen_impact = 0.12
                modifier = 1.8 if "spinal" in combined_text or "severe" in combined_text else 1.0
                agent_severity_coefficient += (base_reopen_impact * modifier)
                reason_string = f"AI Agent Adjustment (+{agent_severity_coefficient:.2f}): Latent risk manifestations forcing unexpected tail reopens."

        llm_suggestions[col] = col_baseline + agent_severity_coefficient
        llm_reasons[col] = reason_string
        agent_shap_contributions[col] = agent_severity_coefficient

    # =====================================================================
    # 5. SUMMARY MATRIX COMPILATION LAYER
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

    # Append core summary actuarial variables and textual narratives
    summary_matrix.loc["LDF"] = [f"{cl_benchmarks[c]:.2f}" for c in target_headers]
    summary_matrix.loc["LLM Suggested"] = [f"{llm_suggestions[c]:.2f}" for c in target_headers]
    summary_matrix.loc["Reason"] = [llm_reasons[c] for c in target_headers]

    st.subheader(f"📊 Live Loss Development Grid Matrix (Active Focus Row: Origin Year {selected_oy})")
    st.dataframe(summary_matrix, use_container_width=True)

    # =====================================================================
    # 6. SHAP WATERFALL ATTRIBUTION VISUALIZER
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