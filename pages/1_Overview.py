import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from analytics import track_page_view

track_page_view(page_title="ChurnSentinel - Overview Dashboard", page_path="/overview")

st.markdown("<style>[data-testid='stSidebarNav'] { display: none !important; }</style>", unsafe_allow_html=True)

# Re-Instantiate components maps
home_page = st.Page("app.py")
overview_page = st.Page("pages/1_Overview.py")
individual_page = st.Page("pages/2_Individual.py")
bulk_page = st.Page("pages/3_Bulk.py")
help_page = st.Page("pages/4_Help.py")

assets = pickle.load(open('model.pkl', 'rb'))
metrics = assets['metrics']

st.header("📈 Executive Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model AUC-ROC", f"{metrics['auc']:.2f}", "Stability High")
m2.metric("Catch Rate (Recall)", f"{metrics['recall']*100:.1f}%", "Churn Caught")
m3.metric("System Uptime", "99.9%", "Live")
m4.metric("Avg. Sentiment", "0.12", "-0.04")

st.divider()
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("⚠️ Top Churn Drivers")
    importance_df = pd.DataFrame({
        'Driver': ['Sentiment Score', 'Contract Type', 'Tenure', 'Monthly Charges'],
        'Impact Score': [45, 30, 15, 10]
    }).sort_values(by='Impact Score', ascending=True)
    
    fig_importance = px.bar(importance_df, x='Impact Score', y='Driver', orientation='h', title="What's Driving Churn?", color_discrete_sequence=['#EF4444'])
    fig_importance.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_importance, use_container_width=True)

with col_right:
    st.subheader("📋 Retention Funnel")
    funnel_data = pd.DataFrame({"Stage": ["Total Users", "Monitored", "At Risk", "Critical"], "Value": [1000, 850, 240, 65]})
    fig_funnel = px.funnel(funnel_data, x='Value', y='Stage', color_discrete_sequence=['#1E3A8A'])
    fig_funnel.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_funnel, use_container_width=True)

st.markdown("---")
st.markdown("#### 🚩 Current Business Alerts")
st.info("**Insight:** Negative sentiment in 'Month-to-Month' contracts has increased by 15% this week. Focus retention efforts on this segment.")
