import streamlit as st
import uuid
import requests
import time

# --- CONFIGURATION ---
MEASUREMENT_ID = "G-PB79XNJY9X"
API_SECRET = "A2DX9eAyS6eSSFGR1I8ZQQ"
BASE_URL = "https://churnguard-behavioral-scanner.streamlit.app"

def initialize_analytics():
    """
    Initializes unique user tracking IDs in the Streamlit session state 
    if they do not already exist for the current visitor.
    """
    if "ga_client_id" not in st.session_state:
        st.session_state.ga_client_id = str(uuid.uuid4())
        st.session_state.ga_session_id = str(int(time.time()))
        st.session_state.last_tracked_page = None


def track_page_view(page_title: str, page_path: str):
    """
    Automatically tracks application page views, session starts, 
    and new user events to Google Analytics 4.
    """
    # Ensure tracking tokens exist
    initialize_analytics()

    # Only fire tracking if it's a brand new session or the user switched pages/tabs
    if st.session_state.last_tracked_page != page_title:
        
        # Live production endpoint (Corrected URL path structure)
        url = f"https://www.google-analytics.com/debug/mp/collect?measurement_id={MEASUREMENT_ID}&api_secret={API_SECRET}"        

        response = requests.post(url, json=payload, timeout=2)
        print("GA4 Validation Response:", response.text)

        events_to_send = []

        # If last_tracked_page is None, this is the first execution of this session
        if st.session_state.last_tracked_page is None:
            # Logs a "New User" baseline in the GA4 dashboard overview
            events_to_send.append({
                "name": "first_visit",
                "params": {
                    "session_id": st.session_state.ga_session_id
                }
            })
            
            # Logs the structural start of this user session
            events_to_send.append({
                "name": "session_start",
                "params": {
                    "session_id": st.session_state.ga_session_id
                }
            })

        # Always record the active page view information
        events_to_send.append({
            "name": "page_view",
            "params": {
                "page_title": page_title,
                "page_path": page_path,
                "page_location": f"{BASE_URL}{page_path}",
                "session_id": st.session_state.ga_session_id,
                "engagement_time_msec": 100  # Fixed: Integer representation
            }
        })
        
        payload = {
            "client_id": st.session_state.ga_client_id,
            "events": events_to_send
        }
        
        try:
            requests.post(url, json=payload, timeout=2)
            # Cache the page title to prevent repetitive triggering during script reruns
            st.session_state.last_tracked_page = page_title  
        except Exception:
            pass  # Fail silently to avoid interrupting the user interface


def send_ga4_event(event_name: str, params: dict = None):
    """
    Utility function to send a custom standalone backend event to GA4.
    Automatically binds the event to the user's active Streamlit session.
    """
    # Ensure tracking tokens exist
    initialize_analytics()
    
    # Live production endpoint
    url = f"https://www.google-analytics.com/mp/collect?measurement_id={MEASUREMENT_ID}&api_secret={API_SECRET}"
    
    # Attach session id to user parameters if not explicitly provided
    event_params = params or {}
    if "session_id" not in event_params:
        event_params["session_id"] = st.session_state.ga_session_id

    payload = {
        "client_id": st.session_state.ga_client_id,
        "events": [{
            "name": event_name,
            "params": event_params
        }]
    }
    
    try:
        requests.post(url, json=payload, timeout=2)
    except Exception:
        pass  # Fail silently
