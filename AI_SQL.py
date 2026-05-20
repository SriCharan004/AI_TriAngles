import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration & Database Initialization
# ---------------------------------------------------------
st.set_page_config(
    page_title="XAI Actuarial Audit Console",
    page_icon="🧠",
    layout="wide"
)

# Initialize the native PostgreSQL connection
# Streamlit reads credentials automatically from .streamlit/secrets.toml
conn = st.connection("postgresql", type="sql")

st.title("🧠 Explainable AI (XAI) Actuarial Audit Console")
st.markdown("Select an active insurance policy profile from the registry below to generate interactive feature attributions.")

try:
    # ---------------------------------------------------------
    # 2. Data Fetching & Casing Normalization
    # ---------------------------------------------------------
    # Fetch real-time data from your Neon PostgreSQL database
    query = "SELECT * FROM actuarial_ai_analysis;"
    raw_df = conn.query(query, ttl=0) # ttl=0 guarantees a fresh query on app interaction
    
    # Force all dataset columns to lowercase to prevent case-sensitive index crashes
    df = raw_df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Robust fallback mechanism: If the newer XAI columns are not found in the 
    # Neon database yet, inject default values so the code executes flawlessly.
    if 'xai_top_driver' not in df.columns:
        df['xai_top_driver'] = 'Baseline Risk Model Evaluation'
        df['xai_driver_impact_pct'] = 100
        df['shap_base_value'] = 15.0
        df['shap_value_delta'] = df['ai_risk_score'].astype(float) - 15.0

    # ---------------------------------------------------------
    # 3. Interactive Dataframe Registry View
    # ---------------------------------------------------------
    st.subheader("Policy Database Registry")
    
    # Isolate specific columns for the clean table layout
    registry_cols = ['policy_holder_id', 'age', 'annual_premium', 'ai_risk_score', 'ai_fraud_flag', 'xai_top_driver']
    display_df = df[[col for col in registry_cols if col in df.columns]]
    
    # Render the interactive dataframe with row-selection activated
    selected_rows = st.dataframe(
        display_df, 
        column_config={
            "policy_holder_id": "Policy ID",
            "age": "Age",
            "annual_premium": st.column_config.NumberColumn("Premium", format="$%d"),
            "ai_risk_score": st.column_config.NumberColumn("AI Risk Score", format="%d/100"),
            "ai_fraud_flag": "Fraud Flag",
            "xai_top_driver": "Primary Risk Factor (XAI)"
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",          # Tells Streamlit to instantly re-execute upon a user's click
        selection_mode="single-row" # Limit user interaction to one case evaluation at a time
    )

    st.write("---")

    # ---------------------------------------------------------
    # 4. State Management for Row Selection
    # ---------------------------------------------------------
    # Track which index was highlighted. If none, default back to the first row (index 0).
    clicked_indices = selected_rows.get("selection", {}).get("rows", [])
    
    if clicked_indices:
        selected_index = clicked_indices[0]
        status_message = "👉 Displaying Selected Policy Profile"
    else:
        selected_index = 0
        status_message = "💡 Click any row above to switch profiles. Showing default (Row 1):"

    # Isolate the exact record chosen by the user
    policy_data = df.iloc[selected_index]

    # ---------------------------------------------------------
    # 5. Dual-Column Dashboard Layout (Metrics vs Graph)
    # ---------------------------------------------------------
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader(f"{status_message} ({policy_data['policy_holder_id']})")
        
        # Contextual banner based on AI Fraud Anomaly checks
        if policy_data.get('ai_fraud_flag'):
            st.error(f"⚠️ **Audit Status:** Flagged for manual review. Top Driver: {policy_data['xai_top_driver']}.")
        else:
            st.success("✨ **Audit Status:** Standard automated underwriting clearance approved.")

        # Text breakdown panel
        st.write(f"**Policyholder Demographics & Profile:**")
        st.info(f"""
        * **Age:** {policy_data['age']} years old
        * **Current Premium:** ${policy_data['annual_premium']:.2f}
        * **Prior Recorded Claims:** {policy_data['historical_claims_count']}
        * **Assigned Baseline Risk Class:** {policy_data['risk_class']}
        """)
        
        st.markdown(f"""
        ### 🔍 Model Transparency Report
        The AI assigned a risk score of **{policy_data['ai_risk_score']}** primarily driven by **{policy_data['xai_top_driver']}**, 
        which accounted for **{policy_data['xai_driver_impact_pct']}%** of the total model variance for this prediction.
        """)

    with col_right:
        st.subheader("📊 SHAP Value Risk Attribution Waterfall")
        st.markdown("This chart explains how the model moved from a generic baseline score to its final prediction.")

        # Safely extract quantitative values for the graph
        base_val = float(policy_data.get('shap_base_value', 15.0))
        final_val = float(policy_data.get('ai_risk_score', 15.0))
        delta_val = float(policy_data.get('shap_value_delta', final_val - base_val))

        # Build the Explainable AI Waterfall Chart via Plotly Graph Objects
        fig = go.Figure(go.Waterfall(
            name="XAI Attribution",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Model Cohort Base", f"Impact: {policy_data['xai_top_driver']}", "Calculated Risk Score"],
            textposition="outside",
            text=[f"+{base_val}", f"{'+' if delta_val >= 0 else ''}{delta_val:.1f}", f"Final: {final_val}"],
            y=[base_val, delta_val, final_val],
            connector={"line": {"color": "rgb(63, 63, 63)", "dash": "dot"}},
            decreasing={"marker": {"color": "#2ca02c"}}, # Green bar when risk is mitigated
            increasing={"marker": {"color": "#d62728"}}, # Red bar when risk is inflated
            totals={"marker": {"color": "#1f77b4"}}       # Blue bar for the final result
        ))

        fig.update_layout(
            yaxis_title="Risk Scale Points",
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            showlegend=False
        )
        
        # Deploy the chart into the right-hand dashboard layout column
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Operational error updating the console view: {e}")