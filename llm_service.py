"""LLM service for interacting with Google's Gemini API."""

import json
import streamlit as st
from typing import Dict, List, Optional
import google.generativeai as genai
from google.generativeai import GenerativeModel
from models import AnalysisCriteria, SuitableLocation, AnalysisResult


class GeminiService:
    """Service for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.model = None
    
    def connect(self) -> bool:
        """Setup Gemini API connection."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            return True
        except Exception as e:
            st.error(f"Error setting up Gemini API: {e}")
            return False
    
    def analyze_criteria(self, user_input: str, available_layers: List[str] = None) -> Optional[AnalysisCriteria]:
        """Analyze user input to extract suitability criteria.
        
        Args:
            user_input: Natural language description of criteria
            available_layers: List of available layer names from database. If None, uses default list.
        """
        if not self.model:
            st.error("Gemini API not configured.")
            return None
        
        # Use provided layers or default list
        if available_layers is None or len(available_layers) == 0:
            available_layers = ['masjids', 'railway', 'roads', 'healthcare', 'bank', 'schools', 
                              'builtup', 'water', 'bareland', 'bus_stops', 'crops']
        
        # Format layers list for prompt
        layers_list = "\n".join([f"        - {layer}" for layer in available_layers])
        
        prompt = f'''
        Analyze the following user criteria for site suitability analysis:
        "{user_input}"
        
        Using ONLY the following available GIS layers:
{layers_list}
        Pay special attention to any terms indicating avoidance such as:
        - "avoid"
         - "away from"
         - "not near"
         - "far from"
         - "excluding"
         - "outside of"
         - "not in"
         - "stay clear of"
         - "not on"
         - at least X metres from etc
    
    For builtup areas specifically, if the user indicates any preference to avoid built-up areas, populated areas, 
    developed areas, or urban areas, make sure to include "builtup" in the avoid list (if "builtup" is available in the layers list).
        
        Identify and return a JSON structure with:
        1. The specific layers needed from the list above. Check for similar keywords. For example, call schools layer, if it says educational institutes, call school etc
        2. Distance requirements for each layer (e.g., if 10m you output 10, if 1km you output 1000 etc)
        3. Weights for each criteria (on a scale of 1-100)
        4. If there are any layers that should be avoided (e.g., away from railways)
        5. The overall objective of the analysis
        
        Format your response as valid JSON without markdown or code formatting.
        Just return the raw JSON object like this:
        {{"layers": [], "distance_requirements": {{}}, "weights": {{}}, "avoid": [], "objective": ""}}
        '''

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to ensure it's valid JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.replace("```", "", 1)
                
            response_text = response_text.strip()
            
            # Parse the JSON
            criteria_dict = json.loads(response_text)
            
            # Create and return an AnalysisCriteria object
            return AnalysisCriteria.from_dict(criteria_dict)
        except Exception as e:
            st.error(f"Error analyzing criteria with Gemini: {e}")
            return None
    
    def analyze_locations(self, 
                         locations: List[SuitableLocation], 
                         criteria: AnalysisCriteria, 
                         analysis_summary: str) -> Optional[Dict]:
        """Analyze suitable locations and provide recommendations."""
        if not self.model:
            st.error("Gemini API not configured.")
            return None
        
        try:
            # Prepare the data for Gemini
            location_data = []
            for location in locations[:10]:  # Send top 10 candidates to Gemini
                location_info = location.to_dict()
                
                # Add nearby features if available
                if location.nearby_features:
                    location_info['nearby_features'] = location.nearby_features
                
                location_data.append(location_info)
            
            # Create an enhanced prompt for Gemini
            prompt = f'''
            I've conducted a site suitability analysis with the following criteria:
            
            Objective: {criteria.objective}
            
            Layers used:
            {json.dumps(criteria.layers, indent=2)}
            
            Distance requirements:
            {json.dumps(criteria.distance_requirements, indent=2)}
            
            Weights:
            {json.dumps(criteria.weights, indent=2)}
            
            Analysis summary:
            {analysis_summary}
            
            Here are the top candidate locations from my analysis, including real-world context and nearby features:
            {json.dumps(location_data, indent=2)}
            
            Based on this information, please:
            1. Select the top 3 most viable locations
            2. Provide a detailed explanation for each selection, referencing the specific neighborhood/area and nearby features
            3. Suggest potential development considerations for each location based on the surrounding context
            4. Compare the top 3 locations, highlighting the strengths and weaknesses of each
            5. Be specific about the neighborhoods or areas based on the provided location data (coordinates, addresses, city information) and use your knowledge of the local/regional context
            Format your response as a JSON object with these keys:
            - top_locations: Array of the 3 selected location IDs
            - explanations: Object with location IDs as keys and explanation text as values that references the real-world context
            - considerations: Object with location IDs as keys and development considerations as values
            - comparison: A comparative analysis of the three locations
            - overall_summary: Text summarizing your analysis that mentions the specific neighborhoods/areas
            
            The JSON should look like:
            {{
              "top_locations": [1, 5, 3],
              "explanations": {{
                "1": "Explanation for location 1 in [neighborhood/city]...",
                "5": "Explanation for location 5 in [neighborhood/city]...",
                "3": "Explanation for location 3 in [neighborhood/city]..."
              }},
              "considerations": {{
                "1": "Considerations for location 1...",
                "5": "Considerations for location 5...",
                "3": "Considerations for location 3..."
              }},
              "comparison": "Comparative analysis of the three locations...",
              "overall_summary": "Overall summary mentioning specific neighborhoods..."
            }}
            
            Return ONLY this JSON without additional text, markdown formatting, or explanations.
            '''
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up the response to ensure it's valid JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "", 1)
            if response_text.endswith("```"):
                response_text = response_text.replace("```", "", 1)
                
            response_text = response_text.strip()
            
            # Parse the JSON response
            try:
                results = json.loads(response_text)
                return results
            except json.JSONDecodeError:
                st.error("Could not parse Gemini's response as JSON.")
                return {"error": "JSON parsing failed", "raw_response": response_text}
            
        except Exception as e:
            st.error(f"Error analyzing locations with Gemini: {e}")
            return {"error": str(e)}


def process_analysis_query(model, query, analysis_result):
    """Process a query about the analysis result with comprehensive context, 
    with Gemini acting as both a GIS expert and urban planner."""
    
    # Create a detailed context about the analysis
    context = {
        "objective": analysis_result.criteria.objective,
        "timestamp": analysis_result.timestamp.isoformat(),
        "total_locations": len(analysis_result.locations),
        "criteria_layers": analysis_result.criteria.layers,
        "distance_requirements": analysis_result.criteria.distance_requirements,
        "weights": analysis_result.criteria.weights,
        "avoid_layers": analysis_result.criteria.avoid
    }
    
    # Add top locations data
    top_locations = []
    for loc in analysis_result.get_top_locations(5):  # Get top 5 locations
        location_data = {
            "id": loc.location_id,
            "score": loc.suitability_score,
            "area_hectares": loc.area_hectares,
            "coordinates": [loc.latitude, loc.longitude],
            "address": loc.address if hasattr(loc, 'address') and loc.address else "Not available",
            "neighborhood": loc.neighborhood if hasattr(loc, 'neighborhood') and loc.neighborhood else "Not available",
            "city": loc.city if hasattr(loc, 'city') and loc.city else "Not available"
        }
        
        # Add nearby features if available
        if hasattr(loc, 'nearby_features') and loc.nearby_features:
            nearby = {}
            for layer, features in loc.nearby_features.items():
                feature_list = []
                for f in features[:3]:  # Take top 3 features per layer
                    feature_list.append({
                        "name": f.get('name', 'Unknown'),
                        "distance": f.get('distance', 0)
                    })
                nearby[layer] = feature_list
            location_data["nearby_features"] = nearby
            
        # Add explanations if available
        if hasattr(loc, 'explanation') and loc.explanation:
            location_data["explanation"] = loc.explanation
            
        if hasattr(loc, 'considerations') and loc.considerations:
            location_data["considerations"] = loc.considerations
            
        top_locations.append(location_data)
    
    context["top_locations"] = top_locations
    
    # Add Gemini analysis if available
    if hasattr(analysis_result, 'gemini_analysis') and analysis_result.gemini_analysis:
        context["gemini_analysis"] = {
            "overall_summary": analysis_result.gemini_analysis.get('overall_summary', ''),
            "comparison": analysis_result.gemini_analysis.get('comparison', '')
        }
    
    # Create a comprehensive prompt
    prompt = f"""
    You are an expert GIS developer and urban planner with extensive experience in land suitability analysis, 
    site selection, and urban development planning. You have deep knowledge of:
    
    1. GIS methodologies and spatial analysis techniques
    2. Urban planning principles and best practices
    3. Land use regulations and zoning considerations
    4. Infrastructure development requirements
    5. Environmental impact assessments
    6. Sustainable development practices
    7. Community engagement in urban planning
    8. Real estate development feasibility
    9. Transportation planning and accessibility
    10. Public utilities and services planning
    
    You're helping analyze a land suitability study. Here's the key information about the analysis:
    
    OBJECTIVE: {context['objective']}
    
    ANALYSIS DETAILS:
    - Analysis date: {context['timestamp']}
    - Found {context['total_locations']} suitable locations
    - Criteria layers used: {', '.join(context['criteria_layers'])}
    
    CRITERIA DETAILS:
    - Distance requirements: {context['distance_requirements']}
    - Weights used: {context['weights']}
    - Layers to avoid: {context['avoid_layers']}
    
    TOP LOCATIONS:
    """
    
    # Add top locations to the prompt
    for loc in context['top_locations']:
        prompt += f"""
        LOCATION {loc['id']} (Score: {loc['score']:.2f}, Area: {loc['area_hectares']:.2f} ha)
        - Coordinates: {loc['coordinates'][0]:.6f}, {loc['coordinates'][1]:.6f}
        - Address: {loc['address']}
        - Neighborhood: {loc['neighborhood']}
        - City: {loc['city']}
        """
        
        # Add nearby features if available
        if 'nearby_features' in loc:
            prompt += "- Nearby features:\n"
            for layer, features in loc['nearby_features'].items():
                feature_text = f"  * {layer.title()}: "
                feature_details = []
                for f in features:
                    feature_details.append(f"{f['name']} ({f['distance']:.0f}m)")
                prompt += feature_text + ", ".join(feature_details) + "\n"
        
        # Add explanations and considerations
        if 'explanation' in loc:
            prompt += f"- Analysis: {loc['explanation']}\n"
            
        if 'considerations' in loc:
            prompt += f"- Considerations: {loc['considerations']}\n"
    
    # Add Gemini analysis if available
    if 'gemini_analysis' in context:
        prompt += f"""
        OVERALL ANALYSIS:
        {context['gemini_analysis']['overall_summary']}
        
        COMPARATIVE ANALYSIS:
        {context['gemini_analysis']['comparison']}
        """
    
    # Expert knowledge context
    prompt += """
    As a GIS expert and urban planner, you should:
    
    1. Apply professional urban planning principles when answering questions
    2. Reference standard GIS methodologies and best practices
    3. Consider environmental, social, and economic sustainability factors
    4. Make connections to real-world development considerations beyond what's in the data
    5. Suggest additional analyses or data layers that could improve the study when relevant
    6. Provide technical explanations about GIS methods when appropriate
    7. Consider local/regional context based on the provided location data (coordinates, addresses, city information) when making recommendations
    8. Relate site characteristics to potential development constraints or opportunities
    9. Comment on accessibility, transportation, and infrastructure implications
    10. Consider local regulations and planning frameworks relevant to the region/area in your responses
    11. Don't undermine the analysis by the app.
    You should respond not just as a data analyst, but as a seasoned professional who understands 
    the practical challenges and opportunities in urban development and site selection across different geographic contexts.
    """
    
    # Add user query
    prompt += f"""
    
    Based on the land suitability analysis and your expertise, please answer the following question:
    {query}
    
    Frame your answer in terms of professional GIS analysis and urban planning expertise. Don't talk about limitations too much.
    """
    
    # Generate response
    response = model.generate_content(prompt)
    return response.text


def create_assistant(gemini_api_key):
    """Creates a simple conversational assistant."""
    genai.configure(api_key=gemini_api_key)
    return GenerativeModel('gemini-1.5-pro')

