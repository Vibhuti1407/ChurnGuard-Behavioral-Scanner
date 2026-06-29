# analytics.py
import streamlit as st
import streamlit.components.v1 as components

def track_page_view(page_title: str, page_path: str):
    """Injects the GA4 tracking script safely into the browser layout container with explicit paths."""
    measurement_id = "G-PB79XNJY9X"
    base_url = "https://churnsentinel.streamlit.app"
    
    html_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      
      // Explicitly tell GA4 what the parent page URL metrics are
      gtag('config', '{measurement_id}', {{
        'page_title': '{page_title}',
        'page_path': '{page_path}',
        'page_location': '{base_url}{page_path}'
      }});
    </script>
    """
    
    with st.container():
        element_key = f"ga_tag_{page_title.replace(' ', '_').lower()}"
        components.html(html_code, height=0, width=0, key=element_key)
