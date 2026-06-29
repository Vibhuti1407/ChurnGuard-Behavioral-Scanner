import streamlit as st
import uuid
import requests
import time

def track_page_view(page_title: str, page_path: str):
    """
    Tracks application usage by sending page views, session starts, 
    and new user events to Google Analytics 4 via the Measurement Protocol.
    """
    # --- CONFIGURATION ---
    # Replace these with your actual GA4 credentials if they differ
    api_secret = "A2DX9eAyS6eSSFGR1I8ZQQ" 
    measurement_id = "G-PB79XNJY9X"
    base_url = "https://churnguard-behavioral-scanner.streamlit.app/"
    
    # --- 1. INITIAL APP LAUNCH / NEW USER DETECTION ---
    # If ga_client_id doesn't exist in the current session state, this is a brand new visit
    is_new_session = "ga_client_id" not in st.session_state

    if is_new_session:
        st.session_state.ga_client_id = str(uuid.uuid4())
        st.session_state.ga_session_id = str(int(time.time()))
        st.session_state.last_tracked_page = None

    # --- 2. TRIGGER TRACKING LOGIC ---
    # Fire the tracker if it's a completely new session OR if the user switched tabs/pages
    if is_new_session or st.session_state.last_tracked_page != page_title:
        url = f"https://www.google-analytics.com/debug/mp/collect?measurement_id={measurement_id}&api_secret={api_secret}"
        
        # We build our array of events dynamically
        events_to_send = []

        if is_new_session:
            # Explicitly force GA4 to log a "New User" in your dashboard overview
            events_to_send.append({
                "name": "first_visit",
                "params": {
                    "session_id": st.session_state.ga_session_id
                }
            })
            
            # Log the official start of this user session
            events_to_send.append({
                "name": "session_start",
                "params": {
                    "session_id": st.session_state.ga_session_id
                }
            })

        # Always record the page view details
        events_to_send.append({
            "name": "page_view",
            "params": {
                "page_title": page_title,
                "page_path": page_path,
                "page_location": f"{base_url}{page_path}",
                "session_id": st.session_state.ga_session_id,
                "engagement_time_msec": 100
            }
        })
        
        # Wrap everything neatly inside the GA4 payload envelope
        payload = {
            "client_id": st.session_state.ga_client_id,
            "events": events_to_send
        }
        
        # --- 3. EXECUTE NETWORK REQUEST ---
        try:
            requests.post(url, json=payload, timeout=2)
            # Lock in this page title so subsequent script reruns don't duplicate logs
            st.session_state.last_tracked_page = page_title  
        except Exception:
            pass # Fail silently so your user's experience isn't broken by network blips
