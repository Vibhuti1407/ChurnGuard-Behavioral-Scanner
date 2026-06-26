import streamlit as st
import uuid
import requests
import time

def track_page_view(page_title: str):
    api_secret = "A2DX9eAyS6eSSFGR1I8ZQQ" 
    measurement_id = "G-PB79XNJY9X"
    
    # Track unique user sessions locally using streamlit session state
    if "ga_client_id" not in st.session_state:
        st.session_state.ga_client_id = str(uuid.uuid4())

    if "ga_session_id" not in st.session_state:
        st.session_state.ga_session_id = str(int(time.time()))
        
    # Track the active viewed tab in state to ensure we only send one hit per tab change click
    if "last_tracked_page" not in st.session_state:
        st.session_state.last_tracked_page = None

    # Trigger tracking call ONLY when a new page tab is actually rendered or switched
    if st.session_state.last_tracked_page != page_title:
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={measurement_id}&api_secret={api_secret}"
        
        payload = {
            "client_id": st.session_state.ga_client_id,
            "events": [{
                "name": "page_view",
                "params": {
                    "page_title": page_title,
                    "session_id": st.session_state.ga_session_id,
                    "engagement_time_msec": "100"
                }
            }]
        }
        
        try:
            requests.post(url, json=payload, timeout=2)
            st.session_state.last_tracked_page = page_title  # Mark this page as tracked
        except Exception:
            pass # Silently pass if there's a network glitch
