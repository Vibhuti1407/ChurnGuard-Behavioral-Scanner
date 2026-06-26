import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from analytics import track_page_view

track_page_view(page_title="ChurnSentinel - Bulk Batch Processing", page_path="/bulk-processing")
st.markdown("<style>[data-testid='stSidebarNav'] { display: none !important; }</style>", unsafe_allow_html=True)

home_page = st.Page("app.py")
overview_page = st.Page("pages/1_Overview.py")
individual_page = st.Page("pages/2_Individual.py")
bulk_page = st.Page("pages/3_Bulk.py")
help_page = st.Page("pages/4_Help.py")

assets = pickle.load(open('model.pkl', 'rb'))
model = assets['model']
sid = SentimentIntensityAnalyzer()

st.header("📂 Batch Risk Processor")
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
                    file_name=f'ChurnSentinel_Report_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )                
