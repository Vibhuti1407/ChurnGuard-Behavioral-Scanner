# analytics.py
import streamlit as st
import streamlit.components.v1 as components

def track_page_view(page_title: str):
    """Injects the GA4 tracking script to execute directly in the browser window."""
    measurement_id = "G-PB79XNJY9X"
    
    html_code = f"""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{measurement_id}', {{
        'page_title': '{page_title}'
      }});
    </script>
    """
    # Embed a hidden tracking iframe into the page container
    components.html(html_code, height=0, width=0)
