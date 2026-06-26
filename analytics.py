import streamlit as st
import streamlit.components.v1 as components

def inject_ga(page_title, page_path):
    GA_MEASUREMENT_ID = "G-PB79XNJY9X" 
    
    ga_js = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());

        // Explicitly set the page path and title for Streamlit's SPA layout
        gtag('config', '{GA_MEASUREMENT_ID}', {{
            'page_title': '{page_title}',
            'page_path': '{page_path}'
        }});
    </script>
    """
    # Injects an invisible snippet to trigger GA without breaking the layout
    components.html(ga_js, height=0, width=0)
