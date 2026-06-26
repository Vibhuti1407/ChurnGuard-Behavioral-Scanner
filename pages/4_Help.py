import streamlit as st
from analytics import track_page_view

track_page_view(page_title="ChurnSentinel - Help Documentation")
st.markdown("<style>[data-testid='stSidebarNav'] { display: none !important; }</style>", unsafe_allow_html=True)

home_page = st.Page("app.py")
overview_page = st.Page("pages/1_Overview.py")
individual_page = st.Page("pages/2_Individual.py")
bulk_page = st.Page("pages/3_Bulk.py")
help_page = st.Page("pages/4_Help.py")

st.header("❓ System Help & Documentation")
with st.expander("How is the Churn Risk calculated?"):
        st.write("""
            The system uses a **Hybrid XGBoost Model**. It weighs two types of data:
            1. **Behavioral:** Tenure, Contract Type, and Monthly Charges.
            2. **Emotional:** Real-time sentiment analysis (NLP) of the customer's last support ticket.
        """)
        
with st.expander("What do the Risk Levels mean?"):
    st.markdown("""
    - <span style='color:#10B981'>**Stable (0-30%):**</span> Customer is healthy. No action needed.
    - <span style='color:#F59E0B'>**Warning (31-60%):**</span> Showing signs of churn. Send automated re-engagement email.
    - <span style='color:#EF4444'>**Critical (61-100%):**</span> High probability of leaving. Requires human intervention.
    """, unsafe_allow_html=True)

with st.expander("CSV Requirements for Batch Upload"):
    st.write("Your CSV must contain the following columns:")
    st.code("tenure, MonthlyCharges, Contract, Ticket_Text")
    st.caption("Note: Contract values should be 0 (Month-to-month), 1 (One year), or 2 (Two year).")
