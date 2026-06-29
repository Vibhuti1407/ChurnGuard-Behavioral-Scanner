import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from analytics import track_page_view  

# --- INITIALIZATION & THEME ---
st.set_page_config(page_title="ChurnSentinel", layout="wide", page_icon="🛡️", initial_sidebar_state="collapsed")

@st.cache_resource 
def initialize_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
initialize_nltk()

# Custom CSS for a "SaaS" look and hiding default sidebar
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stHeader"] {
        background-color: #0E1117 !important;
        color: white;
    }
    
    /* Responsive Desktop Layout Base Margins */
    .block-container { 
        padding-top: 1.7rem; 
        padding-bottom: 1rem; 
        padding-left: 4rem; 
        padding-right: 4rem; 
        max-width: 100%; 
    }
    .main { background-color: #f8f9fa; }
    .stApp {
        background-color: #0E1117;  
        color: #FAFAFA;
    }
    .stApp h1 { 
        color: white; 
        font-size: 50px; 
        font-family: Arial; 
        text-align: left; 
        padding-bottom: 2.5rem; 
    }
    h2, [data-testid="stHeader"] h2 {
        font-size: 33px !important;     
        margin-top: 3px !important;
        margin-bottom: 3px !important;
    }    
    h3 {
        font-size: 25px !important;
        margin-top: 3px !important;
        margin-bottom: 3px !important;
    }
    .stMetric { 
        border: 1px solid #d1d5db; 
        padding: 15px; 
        border-radius: 12px; 
        background-color: white; 
    }
    [data-testid="stSidebar"] { 
        background-color: #111827; 
        color: white; 
    }
    h1 { 
        color: #1e3a8a; 
        font-family: 'Inter', sans-serif; 
    }
    div[data-testid="stMetric"] { 
        background-color: #0c0d0d; 
        border: 1px solid #e6e9ef; 
        padding: 15px; 
        border-radius: 10px;
    }     
    hr { 
        margin-top: 0.1rem !important; 
        margin-bottom: 0.8rem !important; 
    }
    div[data-testid="stMainBlockContainer"] h2 {
        margin-top: -15px !important;
        padding-top: 0px !important;
    }
    div.stPageLink a {
        color: #FFFFFF !important; 
        font-weight: 500 !important;
        text-decoration: none !important;
    }
    div.stPageLink a p {
        color: #FFFFFF !important;
        font-size: 16px !important;
    }

    /* =========================================================================
       📱 MOBILE DEVICE OPTIMIZATIONS (Screen size less than 768px wide)
       ========================================================================= */
    @media (max-width: 768px) {
        /* Reduce heavy page indentation on small phone viewports */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        /* Prevent your Top Title Text from wrapping aggressively */
        .stApp h1 {
            font-size: 28px !important;
            padding-bottom: 1rem !important;
        }

        h2, [data-testid="stHeader"] h2 {
            font-size: 22px !important;
        }

        h3 {
            font-size: 18px !important;
        }

        /* Force your 5-column top navigation to become full-width stackable buttons */
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 8px !important;
        }

        /* Clean up metric alignments */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Load model assets
@st.cache_resource
def load_assets():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

assets = load_assets()
if assets is None:
    st.error("🚨 **System Error: Assets Not Found**")
    st.stop()

metrics = assets['metrics']

st.title("🛡️ ChurnSentinel: Behavioral Risk Engine")

# --- DEFINING PAGES FOR NAVIGATION ---
def render_home():
    track_page_view(page_title="ChurnSentinel - Home", page_path="/home")
   
    column1, column2 = st.columns([0.5,0.5])
    with column1: 
        st.markdown("""
        <p></p>
        <h2>The Mission</h2>
        <p style='font-size:20px; color:gray;'><strong>ChurnSentinel</strong> is an end-to-end decision-support tool designed to reduce customer attrition in the telecommunications industry. 
        By fusing <strong>behavioral analytics</strong> with <strong>Natural Language Processing (NLP)</strong>, it identifies at-risk customers before they churn.<p><h4></h4>
        """, unsafe_allow_html=True)

        st.subheader("🛠️ The Technical Pipeline")     
        st.markdown("""
        <h4>1. Data Fusion</h4>
        <p style='font-size:18px; color:gray; padding-left:30px;'>Combines structured CSV data (Tenure, Charges) with unstructured text (Support Tickets).</p>
        <h4>2. NLP Engine</h4>
        <p style='font-size:18px; color:gray; padding-left:30px;'>Uses <strong>VADER Sentiment Analysis</strong> to quantify customer emotion into a score from -1 to +1.</p>
        <h4>3. XGBoost Model</h4>
        <p style='font-size:18px; color:gray; padding-left:30px;'>A Gradient Boosted Decision Tree model trained on over 7,000 records to predict churn probability.</p>
        """, unsafe_allow_html=True) 
       
    with column2:
        st.markdown("""
            <style>
                div[data-testid="stImage"] {
                    padding-top: 10px;
                    padding-bottom: 10px; 
                    padding-left: 0px; /* Centered layout for responsive adjustments */
                    margin: 0 auto;
                    width: 100% !important; /* Scale dynamically */
                    height: auto !important;
                }
        #     </style>
        #     """, unsafe_allow_html=True)
        st.image('home_img.png', use_container_width=True)
        
# Map out your internal sub-page files accurately
home_page = st.Page(render_home, title="Home", icon="🏠", default=True)
overview_page = st.Page("pages/1_Overview.py", title="Overview", icon="📊")
individual_page = st.Page("pages/2_Individual.py", title="Individual Check", icon="👤")
bulk_page = st.Page("pages/3_Bulk.py", title="Bulk Processing", icon="📂")
help_page = st.Page("pages/4_Help.py", title="System Help", icon="❓")

# Initialize routing configuration
pg = st.navigation([home_page, overview_page, individual_page, bulk_page, help_page], position="hidden")

# --- CLICKABLE TOP NAVIGATION MENU BAR ---
nav1, nav2, nav3, nav4, nav5 = st.columns(5)
with nav1: st.page_link(home_page, label="Home", use_container_width=True)
with nav2: st.page_link(overview_page, label="Overview", use_container_width=True)
with nav3: st.page_link(individual_page, label="Individual Check", use_container_width=True)
with nav4: st.page_link(bulk_page, label="Bulk Processing", use_container_width=True)
with nav5: st.page_link(help_page, label="System Help", use_container_width=True)
st.divider()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding-bottom: 10px;">
            <h1 style='color: #60A5FA; font-size: 28px; margin-bottom: 0;'>🛡️ ChurnSentinel</h1>
            <p style='color: #94A3B8; font-size: 14px;'>Hybrid Intelligence Engine</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("### ⚙️ ENGINE STATUS")
    st.success("● Model: XGBoost v2.5")
    st.info(f"● Recall: {metrics['recall']*100:.1f}%")
    st.caption("Last Refreshed: Feb 2026")

# --- EXECUTE ACTIVE ROUTE ---
# This command handles displaying the selected subpage without stacking content!
pg.run()
