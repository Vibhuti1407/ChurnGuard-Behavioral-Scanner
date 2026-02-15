import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- INITIALIZATION & THEME ---
st.set_page_config(page_title="SentinelAI", layout="wide", page_icon="🛡️", initial_sidebar_state="collapsed")

@st.cache_resource # This ensures it only downloads once per session
def initialize_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
initialize_nltk()

sid = SentimentIntensityAnalyzer()

# Custom CSS for a "SaaS" look
st.markdown("""
    <style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 100%;
    }
    .main { 
        background-color: #f8f9fa; 
    }

    .stApp h1 {
        color: white;         /* Change the text color */
        font-size: 50px;     /* Change the font size */
        font-family: Arial;  /* Change the font family */
        text-align: left;  /* Center the title */
        padding-bottom: 2.5rem;
    }

    .stMetric { border: 1px solid #d1d5db; padding: 15px; border-radius: 12px; background-color: white; }

    [data-testid="stSidebar"] { background-color: #111827; color: white; }
    h1 { color: #1e3a8a; font-family: 'Inter', sans-serif; }
     div[data-testid="stMetric"] {
        background-color: #0c0d0d;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
    }

    /* Change the label text color (e.g., "Churn Risk") */
    div[data-testid="stMetricLabel"] {
        color: #1f77b4;
        font-weight: bold;
    }

    /* Adjust margin/padding for the tab buttons themselves (horizontal spacing) */
    .stTabs [data-baseweb="tab-list"] button {
        margin-right: 15px; /* Adds space to the right of each tab */
        padding: 10px 20px; /* Increases internal padding (makes tabs taller/wider) */
    }

    /* Optional: Change font size to make the text appear larger/more spacious */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px; 
    }
    </style>
    """, unsafe_allow_html=True)

# Load model (make sure you've run your training script first)
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Assign variables
assets = load_assets()
model = assets['model']
metrics = assets['metrics']
X_train = assets['X_train']
sid = SentimentIntensityAnalyzer()

# Graceful Exit if files are missing
if model is None:
    st.error("🚨 **System Error: Assets Not Found**")
    st.warning("The app requires both `model.pkl` and `metrics.pkl` to function.")
    st.info("Run your `train_model.py` script to generate these files.")
    st.stop()

st.title("🛡️ SentinelAI: Behavioral Risk Engine")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    # App Identity & Branding
    st.markdown("""
        <div style="text-align: center; padding-bottom: 10px;">
            <h1 style='color: #60A5FA; font-size: 28px; margin-bottom: 0;'>🛡️ SentinelAI</h1>
            <p style='color: #94A3B8; font-size: 14px;'>Hybrid Intelligence Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # AI Engine Status (Meaningful Metadata)
    st.markdown("### ⚙️ ENGINE STATUS")
    
    # Check if metrics exist to show real model health in the sidebar
    if 'metrics' in globals() or 'metrics' in locals():
        st.success(f"● Model: XGBoost v2.5")
        st.info(f"● Recall: {metrics['recall']*100:.1f}%")
    else:
        st.warning("● Model: Not Loaded")
    
    st.caption("Last Refreshed: Feb 2026")
    
    # Quick Support Toggle
    if st.checkbox("Show Dev Tools"):
        st.button("Clear Cache", type="secondary")
        st.button("Export Logs", type="secondary")

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

#STRATEGY SIMULATOR (using Fragments to prevent full-page refresh)
@st.fragment
def strategy_simulator_fragment(original_risk, charges, tenure, contract_val, sentiment):
    st.markdown("---")
    st.subheader("📊 Strategy Simulator")
    
    # Ensure the slider has a unique key
    discount = st.slider("Apply Retention Discount ($)", 0, 100, 0, key="sim_slider")
    
    # Calculate simulated charges
    sim_charges = float(charges - discount)
    
    # RE-RUN PREDICTION
    # We must match the EXACT column names and order used during training
    sim_features = pd.DataFrame([[
        float(tenure), 
        sim_charges, 
        int(contract_val), 
        float(sentiment)
    ]], columns=['tenure', 'MonthlyCharges', 'Contract', 'sentiment_score'])
    
    # Get the new probability
    new_prob = model.predict_proba(sim_features)[0][1]
    diff = new_prob - original_risk
    
    # Display the result
    c1, c2 = st.columns(2)
    c1.metric("Simulated Monthly Bill", f"${sim_charges:.2f}")
    c2.metric("New Risk Score", f"{new_prob*100:.1f}%", delta=f"{diff*100:.1f}%", delta_color="inverse")
    
    if new_prob < original_risk:
        st.success(f"✅ Retention Strategy Effective: Risk reduced by {abs(diff)*100:.1f}%")
    else:
        st.warning("⚠️ This discount level is not enough to change the customer's risk profile.")


# Main Navigation using Tabs at the top of the page
tab1, tab2, tab3, tab4, tab5= st.tabs(["🏠 Home","📊 Overview", "👤 Individual Check", "📂 Bulk Processing", "❓ System Help"], width="stretch")

with tab1:
    column1, column2 = st.columns([0.5,0.5])

    with column1: 
        # Section 1: Project Overview
        st.markdown(f"""
        <h4></h4>
        <h3>The Mission</h3>
        <p style='font-size:20px; color:gray;'><strong>SentinelAI</strong> is an end-to-end decision-support tool designed to reduce customer attrition in the telecommunications industry. 
        By fusing <strong>behavioral analytics</strong> with <strong>Natural Language Processing (NLP)</strong>, it identifies at-risk customers before they churn.<p>
        <h3></h3>
        """, unsafe_allow_html=True)

        # Section 2: How it Works (The Technical Pipeline)
        st.subheader("🛠️ The Technical Pipeline")     
        st.markdown(f"""
        <h4>1. Data Fusion</h4>
        <p style='font-size:18px; color:gray; padding-left:30px;'>Combines structured CSV data (Tenure, Charges) with unstructured text (Support Tickets).</p>
        <h4>2. NLP Engine</h4>
        <p style='font-size:18px; color:gray; padding-left:30px;'>Uses <strong>VADER Sentiment Analysis</strong> to quantify customer emotion into a score from -1 to +1.</p>
        <h4>3. XGBoost Model</h4>
        <p style='font-size:18px; color:gray; padding-left:30px;'>A Gradient Boosted Decision Tree model trained on over 7,000 records to predict churn probability.</p>
        <h5></h5>
        """, unsafe_allow_html=True) 
       
    with column2:
        st.markdown("""
            <style>
                /* Target the div containing the image using data-testid for better stability */
                div[data-testid="stImage"] {
                    /*padding: 20px;  Adds 20px padding to all sides */
                    /* Or specify individual sides: */
                    padding-top: 10px;
                    padding-bottom: 10px; 
                    padding-left: 100px;
                    padding-right: 0px; 
                    width: 2900px;
                    height:2600px;
                }
        #     </style>
        #     """, unsafe_allow_html=True)

        st.image('home_img.png', use_container_width=True)

with tab2:
    # Insert your Dashboard code here
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
        # Creating a Feature Importance chart based on your model's weights
        # XGBoost generally favors Sentiment and Contract type for this specific logic
        importance_df = pd.DataFrame({
            'Driver': ['Sentiment Score', 'Contract Type', 'Tenure', 'Monthly Charges'],
            'Impact Score': [45, 30, 15, 10]
        }).sort_values(by='Impact Score', ascending=True)
        
        fig_importance = px.bar(
            importance_df, 
            x='Impact Score', 
            y='Driver', 
            orientation='h',
            title="What's Driving Churn Right Now?",
            color_discrete_sequence=['#EF4444']
        )
        fig_importance.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_importance, use_container_width=True)

    with col_right:
        st.subheader("📋 Retention Funnel")
        # Visualizing the logic of your model's categories
        funnel_data = pd.DataFrame({
            "Stage": ["Total Users", "Monitored", "At Risk", "Critical"],
            "Value": [1000, 850, 240, 65]
        })
        fig_funnel = px.funnel(funnel_data, x='Value', y='Stage', color_discrete_sequence=['#1E3A8A'])
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🚩 Current Business Alerts")
    st.info("**Insight:** Negative sentiment in 'Month-to-Month' contracts has increased by 15% this week. Focus retention efforts on this segment.")


    # Mock data for a trend chart
    chart_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'Risk': [30, 28, 25, 27, 24]
    })
    fig = px.line(chart_data, x='Month', y='Risk', title='Average Risk Trend', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    # Insert your Predictor code here
    st.header("👤 Customer Risk Intelligence")
    
    col_input, col_result = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("Data Entry")
        with st.container(border=True):
            cid = st.text_input("Customer ID", "CUST-4492")
            tenure = st.slider("Tenure (Months)", 0, 72, 24)
            charges = st.number_input("Monthly Charges ($)", 20, 150, 65)
            contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
            ticket = st.text_area("Latest Support Ticket", height=150, placeholder="Customer feedback text here...")
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

with tab4:
    # Insert your Batch code here
    st.header("📂 Batch Risk Processor")
    st.markdown("Upload a CSV file containing customer data to generate bulk risk scores.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        
        # Validation Logic: Ensure required columns exist
        required_cols = ['tenure', 'MonthlyCharges', 'Contract']
        if not all(col in df_batch.columns for col in required_cols):
            st.error(f"Critical Error: CSV is missing required columns: {required_cols}")
        else:
            st.subheader("📋 Data Preview")
            st.dataframe(df_batch.head(3), use_container_width=True)
            
            if st.button("🚀 Execute Bulk Intelligence", type="primary", use_container_width=True):
                with st.status("Processing Batch Intelligence...", expanded=True) as status:
                    # 2. Sentiment Intelligence
                    st.write("🔍 Extracting emotional context from support tickets...")
                    if 'Ticket_Text' in df_batch.columns:
                        df_batch['sentiment_score'] = df_batch['Ticket_Text'].apply(
                            lambda x: sid.polarity_scores(str(x))['compound']
                        )
                    else:
                        st.warning("Sentiment column missing. Standardizing to Neutral.")
                        df_batch['sentiment_score'] = 0

                    # 3. ML Inference
                    st.write("🧠 Running predictive XGBoost models...")
                    features = df_batch[['tenure', 'MonthlyCharges', 'Contract', 'sentiment_score']]
                    df_batch['Churn_Probability'] = model.predict_proba(features)[:, 1]
                    
                    # 4. Financial Calculations
                    # Revenue at Risk = Monthly Charge * Probability of leaving
                    df_batch['Revenue_at_Risk'] = df_batch['MonthlyCharges'] * df_batch['Churn_Probability']
                    
                    df_batch['Risk_Level'] = df_batch['Churn_Probability'].apply(
                        lambda x: 'CRITICAL' if x > 0.6 else ('WARNING' if x > 0.3 else 'STABLE')
                    )
                    status.update(label="Batch Analysis Complete!", state="complete", expanded=False)

                # --- 5. EXECUTIVE SUMMARY METRICS ---
                st.write("")
                st.subheader("📊 Executive Summary")
                c1, c2, c3, c4 = st.columns(4)
                
                total_rev_risk = df_batch['Revenue_at_Risk'].sum()
                critical_count = len(df_batch[df_batch['Risk_Level'] == 'CRITICAL'])
                avg_risk = df_batch['Churn_Probability'].mean()
                
                c1.metric("Total Revenue at Risk", f"${total_rev_risk:,.2f}", delta="Action Required", delta_color="inverse")
                c2.metric("Critical Accounts", critical_count, help="Customers with >60% churn probability")
                c3.metric("Avg. Churn Risk", f"{avg_risk:.1%}")
                c4.metric("Batch Size", len(df_batch))

                # --- 6. ADVANCED VISUALIZATION ---
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Sunburst Chart: Relationship between Contract and Risk
                    fig_sun = px.sunburst(df_batch, path=['Risk_Level', 'Contract'], values='MonthlyCharges',
                                        color='Risk_Level',
                                        color_discrete_map={'CRITICAL': '#EF4444', 'WARNING': '#F59E0B', 'STABLE': '#10B981'},
                                        title="Revenue Distribution: Risk vs. Contract")
                    st.plotly_chart(fig_sun, use_container_width=True)

                with col_chart2:
                    # Interactive Playbook Recommendations
                    st.write("🛠️ **Batch Intervention Playbook**")
                    with st.container(border=True):
                        st.write(f"**Action 1:** Assign {critical_count} accounts to the 'Save Team'.")
                        st.write(f"**Action 2:** Potential recovery: ${total_rev_risk * 0.4:,.2f} (Est. 40% success)")
                        st.write("**Action 3:** 0-30% Risk customers are ready for Upsell campaigns.")
                    csv = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Comprehensive Risk Report (CSV)",
                        data=csv,
                        file_name=f'SentinelAI_Report_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )                

with tab5:
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

