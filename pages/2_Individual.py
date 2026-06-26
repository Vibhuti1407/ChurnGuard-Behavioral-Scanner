import streamlit as st
import pandas as pd
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from analytics import inject_ga

inject_ga(page_title="ChurnSentinel - Individual Risk Checker", page_path="/individual-check")
st.markdown("<style>[data-testid='stSidebarNav'] { display: none !important; }</style>", unsafe_allow_html=True)

home_page = st.Page("app.py")
overview_page = st.Page("pages/1_Overview.py")
individual_page = st.Page("pages/2_Individual.py")
bulk_page = st.Page("pages/3_Bulk.py")
help_page = st.Page("pages/4_Help.py")

assets = pickle.load(open('model.pkl', 'rb'))
model = assets['model']
sid = SentimentIntensityAnalyzer()

@st.fragment
def strategy_simulator_fragment(original_risk, charges, tenure, contract_val, sentiment):
    st.markdown("---")
    st.subheader("📊 Strategy Simulator")
    discount = st.slider("Apply Retention Discount ($)", 0, 100, 0, key="sim_slider")
    sim_charges = float(charges - discount)
    sim_features = pd.DataFrame([[float(tenure), sim_charges, int(contract_val), float(sentiment)]], columns=['tenure', 'MonthlyCharges', 'Contract', 'sentiment_score'])
    new_prob = model.predict_proba(sim_features)[0][1]
    diff = new_prob - original_risk
    c1, c2 = st.columns(2)
    c1.metric("Simulated Monthly Bill", f"${sim_charges:.2f}")
    c2.metric("New Risk Score", f"{new_prob*100:.1f}%", delta=f"{diff*100:.1f}%", delta_color="inverse")

def get_advanced_playbook(prob, charges, sentiment, contract):
    """Categorizes churn drivers and returns a specific business playbook."""
    if prob > 0.6:
        if sentiment < -0.3:
            return {
                "title": "🚨 Service Recovery Protocol",
                "driver": "Negative Sentiment",
                "action": "Immediate 24h retention callback by Support Lead.",
                "offer": "Issue a 1-month service credit + technical audit."
            }
        elif contract == 0:
            return {
                "title": "🚨 Contract Migration Strategy",
                "driver": "Contract Flexibility",
                "action": "Flag for Sales 'Lock-in' campaign.",
                "offer": "Offer 20% discount on a 1-year commitment."
            }
        else:
            return {
                "title": "🚨 High-Value Save Plan",
                "driver": "Pricing/General",
                "action": "Manual account review required.",
                "offer": "Propose a lower-tier plan down-sell."
            }
    elif 0.3 < prob <= 0.6:
        return {
            "title": "⚠️ Engagement Campaign",
            "driver": "Moderate Risk",
            "action": "Automated re-engagement email sequence.",
            "offer": "Send educational 'Feature Spotlight' content."
        }
    return {
        "title": "✅ Loyalty Maintenance",
        "driver": "Healthy Account",
        "action": "Standard communication cycle.",
        "offer": "Target for Referral Program invitation."
    }

st.header("👤 Customer Risk Intelligence")
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    with st.container(border=True):
        cid = st.text_input("Customer ID", "CUST-4492")
        tenure = st.slider("Tenure (Months)", 0, 72, 24)
        charges = st.number_input("Monthly Charges ($)", 20, 150, 65)
        contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        ticket = st.text_area("Latest Support Ticket", height=150)
        predict_btn = st.button("Generate Risk Score", use_container_width=True, type="primary")

with col_result:
    if predict_btn:       
        # Calculation
        sentiment = sid.polarity_scores(ticket)['compound']
        contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
        features = pd.DataFrame([[tenure, charges, contract_map[contract], sentiment]], 
                                columns=['tenure', 'MonthlyCharges', 'Contract', 'sentiment_score'])
        risk_proba = model.predict_proba(features)[0][1]
        
        # Save to state so the fragment can access it without losing it on rerun
        st.session_state.last_risk = risk_proba
        
        rev_at_risk = charges * risk_proba

        # Visual Display
        st.subheader("Analysis Summary")
        
        # Risk Gauge Simulation
        color = "#EF4444" if risk_proba > 0.6 else "#F59E0B" if risk_proba > 0.3 else "#10B981"
        st.markdown(f"""
            <div style="background-color: #111827; padding: 25px; border-radius: 12px; border-left: 10px solid {color};">
                <h1 style="margin:0; color:{color}; font-size:45px;">{risk_proba*100:.1f}%</h1>
                <p style="margin:0; color:white; font-size:20px; font-weight:bold;">Probability of Churn</p>
                <p style="color:#9CA3AF; margin-top:10px;"><b>Revenue at Risk:</b> ${rev_at_risk:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Playbook Suggestions
        playbook = get_advanced_playbook(risk_proba, charges, sentiment, contract_map[contract])
        
        st.write("")
        with st.expander(f"🛠️ Playbook: {playbook['title']}", expanded=True):
            st.write(f"**Primary Driver:** {playbook['driver']}")
            st.info(f"**Recommended Action:** {playbook['action']}")
            st.success(f"**Retention Offer:** {playbook['offer']}")
            
            if risk_proba > 0.6:
                st.button("📧 Send Pre-approved Offer Now")
                # Fragment-based simulator
                strategy_simulator_fragment(risk_proba, charges, tenure, contract_map[contract], sentiment)
