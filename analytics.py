import streamlit as st
import uuid
import requests
import time

def track_page_view(page_title: str, page_path: str):
    api_secret = "A2DX9eAyS6eSSFGR1I8ZQQ" 
    measurement_id = "G-PB79XNJY9X"
    base_url = "https://churnsentinel.streamlit.app"
    
    # 1. INITIAL APP LAUNCH / NEW USER DETECTION
    # If ga_client_id doesn't exist, this is a brand new user session
    is_new_session = "ga_client_id" not in st.session_state

    if is_new_session:
        st.session_state.ga_client_id = str(uuid.uuid4())
        st.session_state.ga_session_id = str(int(time.time()))
        st.session_state.last_tracked_page = None

    # 2. TRIGGER TRACKING (If it's a new session OR they actually switched pages)
    if is_new_session or st.session_state.last_tracked_page != page_title:
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={measurement_id}&api_secret={api_secret}"
        
        # Build events list dynamically
        events_to_send = []

        # If it's their very first load, append a session_start event
        if is_new_session:
            events_to_send.append({
                "name": "session_start",
                "params": {
                    "session_id": st.session_state.ga_session_id
                }
            })

        # Always append the page_view event
        events_to_send.append({
            "name": "page_view",
            "params": {
                "page_title": page_title,
                "page_path": page_path,
                "page_location": f"{base_url}{page_path}",
                "session_id": st.session_state.ga_session_id,
                "engagement_time_msec": "100"
            }
        })
        
        # Construct final payload
        payload = {
            "client_id": st.session_state.ga_client_id,
            "events": events_to_send
        }
        
        try:
            requests.post(url, json=payload, timeout=2)
            st.session_state.last_tracked_page = page_title  # Mark this page as tracked
        except Exception:
            pass # Silently pass if there's a network glitch
