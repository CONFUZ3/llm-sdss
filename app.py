"""Main Streamlit application entry point."""

import streamlit as st
from config import load_custom_css
from ui_components import UIComponents

# Set page config for initial load
st.set_page_config(
    page_title="LLM-SDSS", 
    page_icon="🏙️", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/llm-sdss',
        'Report a bug': 'https://github.com/yourusername/llm-sdss/issues',
        'About': "# LLM-Driven Spatial Decision Support System\nThis application helps urban planners identify optimal locations based on multiple criteria using AI-powered analysis."
    }
)


def main():
    """Main application entry point."""
    # Load custom CSS
    st.markdown(load_custom_css(), unsafe_allow_html=True)
    
    # App title
    st.title("LLM-Driven Spatial Decision Support System 🏙️")
    st.markdown(
        "Find optimal locations for development based on multiple spatial criteria, "
        "enhanced with AI-powered analysis."
    )
    
    # Render sidebar
    services = UIComponents.render_sidebar()
    db_service = services.get('db_service')
    gemini_service = services.get('gemini_service')
    
    # Create tabs
    analysis_tab, results_tab = st.tabs(["Analysis", "Results"])
    
    # Render each tab
    with analysis_tab:
        UIComponents.render_analysis_tab(db_service, gemini_service)
    
    with results_tab:
        UIComponents.render_results_tab()


if __name__ == "__main__":
    main()

