"""UI components for the Streamlit application."""

import os
import io
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
from typing import Optional
from models import DatabaseConfig, AnalysisCriteria
from database import DatabaseService
from llm_service import GeminiService, process_analysis_query, create_assistant
from suitability_analysis import SuitabilityAnalysisService
from visualization import VisualizationService
from pdf_report import PDFReportService


class UIComponents:
    """UI components for the Streamlit application."""
    
    @staticmethod
    def render_sidebar():
        """Render the sidebar UI."""
        st.sidebar.header("Configuration")
        
        # Database connection section
        st.sidebar.subheader("Database Connection")
        
        # Use session state to persist form values
        if 'db_host' not in st.session_state:
            st.session_state.db_host = "localhost"
        if 'db_port' not in st.session_state:
            st.session_state.db_port = "5432"
        if 'db_name' not in st.session_state:
            st.session_state.db_name = "gis"
        if 'db_user' not in st.session_state:
            st.session_state.db_user = "postgres"
        if 'db_password' not in st.session_state:
            st.session_state.db_password = ""
        
        db_host = st.sidebar.text_input("Host", value=st.session_state.db_host, key="db_host_input")
        db_port = st.sidebar.text_input("Port", value=st.session_state.db_port, key="db_port_input")
        db_name = st.sidebar.text_input("Database", value=st.session_state.db_name, key="db_name_input")
        db_user = st.sidebar.text_input("Username", value=st.session_state.db_user, key="db_user_input")
        db_password = st.sidebar.text_input("Password", type="password", value=st.session_state.db_password, key="db_password_input")
        
        # Gemini API configuration
        st.sidebar.subheader("Gemini API Configuration")
        
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = ""
        
        gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, key="gemini_api_key_input")
        
        # Connect button
        connect_pressed = st.sidebar.button("Connect", type="primary")
        
        if connect_pressed:
            # Update session state
            st.session_state.db_host = db_host
            st.session_state.db_port = db_port
            st.session_state.db_name = db_name
            st.session_state.db_user = db_user
            st.session_state.db_password = db_password
            st.session_state.gemini_api_key = gemini_api_key
            
            # Create database config and connect
            db_config = DatabaseConfig(db_host, db_port, db_name, db_user, db_password)
            
            if 'db_service' not in st.session_state:
                st.session_state.db_service = DatabaseService(db_config)
            else:
                st.session_state.db_service.config = db_config
            
            if st.session_state.db_service.connect():
                st.sidebar.success("Connected to database!")
                
                # Setup Gemini if API key provided
                if gemini_api_key:
                    try:
                        if 'gemini_service' not in st.session_state:
                            st.session_state.gemini_service = GeminiService(gemini_api_key)
                        else:
                            st.session_state.gemini_service.api_key = gemini_api_key
                        
                        if st.session_state.gemini_service.connect():
                            st.sidebar.success("Gemini API configured successfully!")
                    except Exception as e:
                        st.sidebar.error(f"Error setting up Gemini API: {e}")
                else:
                    st.sidebar.warning("Gemini API key not provided. Some features will be disabled.")
        
        # Advanced options
        with st.sidebar.expander("Advanced Options", expanded=False):
            if 'cell_size' not in st.session_state:
                st.session_state.cell_size = 30
            
            st.session_state.cell_size = st.number_input(
                "Cell size (meters)",
                min_value=10,
                max_value=100,
                value=st.session_state.cell_size,
                step=5,
                help="Size of the grid cells in meters. Smaller values provide more detailed analysis but increase computation time."
            )
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            "LLM-Driven Spatial Decision Support System helps urban planners and developers "
            "identify optimal locations for development based on multiple spatial criteria. "
            "Created using PyLUSAT, Streamlit, and Google's Gemini AI."
        )
        
        # Return the services
        return {
            'db_service': st.session_state.get('db_service'),
            'gemini_service': st.session_state.get('gemini_service')
        }
    
    @staticmethod
    def render_analysis_tab(db_service: Optional[DatabaseService] = None, 
                           gemini_service: Optional[GeminiService] = None):
        """Render the analysis tab UI."""
        st.header("Site Suitability Analysis", divider="blue")
        
        if not db_service or not db_service.engine:
            st.warning("Please connect to the database first using the sidebar.")
            return
        
        st.subheader("Define Suitability Criteria")
        
        # Get available layers
        available_layers = db_service.get_available_layers()
        
        # Two methods for defining criteria: Gemini AI or manual
        criteria_tab1, criteria_tab2 = st.tabs(["Define with AI", "Define Manually"])
        
        with criteria_tab1:
            if not gemini_service or not gemini_service.model:
                st.warning("Gemini API not configured. Please add an API key in the sidebar.")
            else:
                st.markdown('''
                Describe what you're looking for in plain English, and the AI will convert your description into specific criteria.
                
                **Example:** "I need to find a suitable location for a new residential area that should be within 2km of schools, 5km of healthcare facilities, and 1km of roads. It should also be at least 500m away from railways for safety. Proximity to schools is most important, followed by healthcare."
                ''')
                
                criteria_text = st.text_area(
                    "Describe your site suitability criteria in natural language",
                    value="I need to find a suitable location for a new residential area that should be within 2km of schools, 5km of healthcare facilities, and 1km of roads. It should also be at least 500m away from railways for safety. Proximity to schools is most important, followed by healthcare.",
                    height=150
                )
                
                if st.button("Analyze Criteria with AI", type="primary", key="analyze_criteria_btn"):
                    with st.spinner("Analyzing criteria with AI..."):
                        criteria = gemini_service.analyze_criteria(criteria_text, available_layers=available_layers)
                        
                        if criteria:
                            st.session_state.criteria = criteria
                            st.success("Criteria analyzed successfully!")
                            
                            # Display the identified criteria
                            st.subheader("Identified Criteria")
                            st.markdown(f"**Objective:** {criteria.objective}")
                            
                            # Display layers, distance requirements, and weights
                            criteria_df = pd.DataFrame({
                                "Layer": criteria.layers,
                                "Distance Requirement (m)": [criteria.distance_requirements.get(layer, "-") for layer in criteria.layers],
                                "Weight (1-100)": [criteria.weights.get(layer, "-") for layer in criteria.layers],
                                "Avoid?": [layer in criteria.avoid for layer in criteria.layers]
                            })
                            
                            st.dataframe(criteria_df, use_container_width=True)
                        else:
                            st.error("Failed to analyze criteria. Please try again with a more specific description.")
        
        with criteria_tab2:
            st.markdown("Manually select layers and define criteria for each layer.")
            
            # Display available layers
            st.info(f"Available layers: {', '.join(available_layers)}")
            
            # Manual layer selection
            selected_layers = st.multiselect("Select layers to include", available_layers)
            
            if selected_layers:
                st.subheader("Distance Requirements and Weights")
                
                manual_distance_reqs = {}
                manual_weights = {}
                manual_avoid_layers = []
                
                # Create columns for each criterion
                for layer in selected_layers:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        distance = st.number_input(f"Distance (m) for {layer}", min_value=100, max_value=10000, value=1000, step=100)
                        manual_distance_reqs[layer] = distance
                    
                    with col2:
                        weight = st.slider(f"Weight for {layer}", min_value=1, max_value=10, value=5)
                        manual_weights[layer] = weight
                    
                    with col3:
                        avoid = st.checkbox(f"Avoid {layer} (distance is minimum)")
                        if avoid:
                            manual_avoid_layers.append(layer)
                
                # Objective field
                objective = st.text_area("Analysis Objective", value="Manual site suitability analysis")
                
                # Create criteria button
                if st.button("Set Manual Criteria", type="primary"):
                    st.session_state.criteria = AnalysisCriteria(
                        layers=selected_layers,
                        distance_requirements=manual_distance_reqs,
                        weights=manual_weights,
                        avoid=manual_avoid_layers,
                        objective=objective
                    )
                    
                    st.success("Manual criteria set!")
                    
                    # Show the criteria as a table
                    criteria_df = pd.DataFrame({
                        "Layer": selected_layers,
                        "Distance Requirement (m)": [manual_distance_reqs.get(layer, "-") for layer in selected_layers],
                        "Weight (1-100)": [manual_weights.get(layer, "-") for layer in selected_layers],
                        "Avoid?": [layer in manual_avoid_layers for layer in selected_layers]
                    })
                    
                    st.dataframe(criteria_df, use_container_width=True)
        
        # Run analysis button
        st.markdown("---")
        
        if 'criteria' in st.session_state:
            st.success("Criteria are defined and ready for analysis.")
            
            if st.button("Run Suitability Analysis", type="primary", key="run_analysis_btn"):
                with st.spinner("Running analysis..."):
                    # Create analysis service
                    analysis_service = SuitabilityAnalysisService(db_service, gemini_service)
                    
                    # Run analysis
                    result = analysis_service.analyze(st.session_state.criteria, st.session_state.cell_size)
                    
                    if result:
                        st.session_state.analysis_result = result
                        
                        # Try to load boundary
                        try:
                            boundary = db_service.load_boundary("islamabad", result.grid.utm_crs)
                            st.session_state.boundary = boundary
                        except Exception as e:
                            st.warning(f"Could not load boundary: {e}")
                            st.session_state.boundary = None
                        
                        st.success("Analysis completed successfully!")
                        st.balloons()
                    else:
                        st.error("Analysis failed. Please check the logs for details.")
        else:
            st.info("Please define criteria before running the analysis.")
    
    @staticmethod
    def render_results_tab():
        """Render the results tab UI."""
        st.header("Analysis Results", divider="blue")
        
        if 'analysis_result' not in st.session_state:
            st.info("No analysis results available. Please run an analysis first.")
            return
        
        result = st.session_state.analysis_result
        boundary = st.session_state.get('boundary')
        
        # Overview section
        st.subheader("Overview")
        
        # Display basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Objective:** {result.criteria.objective}")
            st.markdown(f"**Analysis Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M')}")
            st.markdown(f"**Number of Suitable Locations:** {len(result.locations)}")
        
        with col2:
            st.markdown(f"**Number of Criteria Layers:** {len(result.criteria.layers)}")
            layers_text = ", ".join(result.criteria.layers)
            st.markdown(f"**Layers Used:** {layers_text}")
        
        # Map section
        st.subheader("Suitability Map")
        
        # Create and display the visualization
        fig = VisualizationService.visualize_raster(
            result.suitability_raster,
            result.grid.transform,
            "Spatial Suitability Analysis",
            boundary_gdf=boundary
        )
        
        st.pyplot(fig)
        
        # Add map download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="Download Map Image",
            data=buf,
            file_name="suitability_map.png",
            mime="image/png"
        )
        
        # Locations section
        st.subheader("Suitable Locations")
        
        # Create a DataFrame for easier display
        locations_data = []
        for loc in result.locations:
            locations_data.append({
                "ID": loc.location_id,
                "Area (ha)": loc.area_hectares,
                "Suitability Score": loc.suitability_score,
                "Latitude": loc.latitude,
                "Longitude": loc.longitude,
                "Neighborhood": loc.neighborhood or "N/A",
                "City": loc.city or "N/A"
            })
        
        locations_df = pd.DataFrame(locations_data)
        
        # Display the locations table
        st.dataframe(locations_df, use_container_width=True)
        
        # Add download buttons for locations data
        csv = locations_df.to_csv(index=False)
        st.download_button(
            label="Download Locations CSV",
            data=csv,
            file_name="suitable_locations.csv",
            mime="text/csv"
        )
        
        # Interactive map of locations
        st.subheader("Interactive Map")
        
        # Get top location IDs from Gemini analysis if available
        top_location_ids = None
        if result.gemini_analysis and 'top_locations' in result.gemini_analysis:
            top_location_ids = result.gemini_analysis['top_locations']
        
        try:
            # Create the map
            m = VisualizationService.create_locations_map(
                result.locations,
                top_location_ids,
                boundary
            )
            
            # Save to a temporary file to avoid JSON serialization issues
            if m:
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp:
                        m.save(temp.name)
                        temp_path = temp.name
                    
                    # Read the HTML file directly instead of using _repr_html_()
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        map_html = f.read()
                    
                    # Display the map
                    st.components.v1.html(map_html, height=500)
                    
                    # Create a download button
                    st.download_button(
                        label="Download Interactive Map",
                        data=map_html,
                        file_name="suitability_map.html",
                        mime="text/html"
                    )
                finally:
                    # Clean up the temporary file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            else:
                st.warning("Could not create interactive map.")
        except Exception as e:
            st.error(f"Error displaying interactive map: {e}")
            st.warning("Try running the analysis again with updated data.")
        
        # Add Gemini Analysis section if available
        if result.gemini_analysis and 'overall_summary' in result.gemini_analysis:
            st.subheader("AI Analysis")
            
            # Display overall summary
            st.markdown("### Overall Summary")
            st.write(result.gemini_analysis['overall_summary'])
            
            # Display comparison if available
            if 'comparison' in result.gemini_analysis:
                with st.expander("Comparative Analysis", expanded=True):
                    st.write(result.gemini_analysis['comparison'])
            
            # Display top locations with details
            st.markdown("### Top Recommended Locations")
            
            top_ids = result.gemini_analysis['top_locations']
            top_locations = [loc for loc in result.locations if loc.location_id in top_ids]
            
            for loc in top_locations:
                loc_id = str(loc.location_id)
                
                with st.expander(f"Location {loc.location_id}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Area:** {loc.area_hectares:.2f} hectares")
                        st.markdown(f"**Suitability Score:** {loc.suitability_score:.2f}")
                        st.markdown(f"**Coordinates:** {loc.latitude:.6f}, {loc.longitude:.6f}")
                        
                        if loc.address:
                            st.markdown(f"**Address:** {loc.address}")
                        
                        if loc.neighborhood:
                            st.markdown(f"**Neighborhood:** {loc.neighborhood}")
                        
                        if loc.city:
                            st.markdown(f"**City:** {loc.city}")
                    
                    with col2:
                        if loc_id in result.gemini_analysis['explanations']:
                            st.markdown("#### Analysis")
                            st.write(result.gemini_analysis['explanations'][loc_id])
                        
                        if loc_id in result.gemini_analysis['considerations']:
                            st.markdown("#### Development Considerations")
                            st.write(result.gemini_analysis['considerations'][loc_id])
                    
                    # Display nearby features if available
                    if loc.nearby_features:
                        st.markdown("#### Nearby Features")
                        
                        for layer, features in loc.nearby_features.items():
                            if features:
                                st.markdown(f"**{layer.title()}:**")
                                
                                try:
                                    # Create a small DataFrame for the features
                                    features_df = pd.DataFrame([
                                        {'Name': str(f.get('name', '')), 'Distance (m)': float(f.get('distance', 0))} 
                                        for f in features
                                    ])
                                    
                                    if not features_df.empty:
                                        features_df['Distance (m)'] = features_df['Distance (m)'].astype(float).round(0).astype(int)
                                        st.dataframe(features_df, use_container_width=True)
                                except Exception as e:
                                    # Fallback to simple text display
                                    for f in features[:5]:
                                        st.write(f"{str(f.get('name', 'Unknown'))} - {float(f.get('distance', 0)):.0f}m")
        
        # PDF Report section
        st.subheader("Generate Report")
        
        st.markdown(
            "Generate a comprehensive PDF report containing all analysis results, "
            "suitability maps, and location recommendations."
        )
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                # Create PDF service
                pdf_service = PDFReportService(result, boundary)
                
                # Create a temporary file for the PDF
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        temp_path = tmp.name
                        
                    # Generate the report
                    success = pdf_service.generate_report(temp_path)
                    
                    if success and os.path.exists(temp_path):
                        # Read the file
                        with open(temp_path, 'rb') as f:
                            pdf_data = f.read()
                        
                        # Create download button
                        st.success("PDF report generated successfully!")
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"suitability_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Failed to generate PDF report.")
                finally:
                    # Clean up the temporary file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

        # Add chatbot UI
        add_simple_chatbot_ui(result, st.session_state.get('gemini_api_key'))


def add_simple_chatbot_ui(analysis_result, gemini_api_key):
    """Add a chatbot UI to the results tab with comprehensive analysis context
    and GIS/urban planning expertise."""
    if analysis_result:
        st.subheader("Ask the GIS & Urban Planning Expert")
        
        expert_info = """
        This AI assistant combines analysis of your results with expertise in:
        - GIS methodology and spatial analysis
        - Urban planning and site development
        - Infrastructure and accessibility planning
        - Environmental and social considerations
        - Local context and regulations
        
        Ask about your results or seek professional insights about urban development.
        """
        
        with st.expander("About the Expert Assistant", expanded=False):
            st.markdown(expert_info)
        
        # Create assistant if not already in session state
        if 'assistant' not in st.session_state and gemini_api_key:
            st.session_state.assistant = create_assistant(gemini_api_key)
            
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Example questions to help users
        if len(st.session_state.chat_history) == 0:
            with st.expander("Example questions to ask", expanded=False):
                st.markdown("""
                - What additional analysis would improve the results?
                - What environmental factors should be considered for Location 1?
                - How would public transportation access impact these locations?
                - What zoning regulations might affect development at the top location?
                - What community engagement strategies would you recommend?
                - How does the proximity to schools impact the development potential?
                - What infrastructure challenges might these locations face?
                - Which location has the best development potential for affordable housing?
                - How would climate change factors impact these locations?
                - What GIS methodology would you recommend for refining this analysis?
                """)
        
        # Chat input
        query = st.chat_input("Ask a question about the analysis or urban planning:")
        
        if query and 'assistant' in st.session_state:
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
                
            # Add to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = process_analysis_query(st.session_state.assistant, query, analysis_result)
                    st.markdown(response)
                    
            # Add to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        elif query:
            st.warning("Please provide a Gemini API key in the sidebar to enable the expert assistant.")
        
        # Add a button to clear chat history
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

