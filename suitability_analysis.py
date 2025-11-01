"""Main suitability analysis service that orchestrates the complete MCDA process."""

from datetime import datetime
import streamlit as st
import numpy as np
from typing import Optional
from models import AnalysisCriteria, AnalysisResult
from database import DatabaseService
from geocoding import GeoCodingService
from spatial_analysis import SpatialAnalysisService
from llm_service import GeminiService


class SuitabilityAnalysisService:
    """Main service for performing spatial suitability analysis in the LLM-Driven Spatial Decision Support System."""
    
    def __init__(self, db_service: DatabaseService, gemini_service: Optional[GeminiService] = None):
        """Initialize with required services."""
        self.db_service = db_service
        self.gemini_service = gemini_service
        self.geocoding_service = GeoCodingService()
    
    def analyze(self, criteria: AnalysisCriteria, cell_size: float = 30) -> Optional[AnalysisResult]:
        """Perform complete suitability analysis based on criteria."""
        try:
            # Fetch layers from database
            layers = self.db_service.fetch_layers(criteria.layers)
            
            if not layers:
                st.error("No layers could be loaded from the database.")
                return None
            
            # Create analysis grid
            grid = SpatialAnalysisService.create_analysis_grid(layers, cell_size)
            
            if grid is None:
                st.error("Could not create analysis grid.")
                return None
            
            # Process each layer
            processed_rasters = []
            raster_weights = []
            # Store named reclassified rasters for reporting
            named_rasters = {}
            
            for layer_name in criteria.layers:
                if layer_name not in layers:
                    st.warning(f"Layer {layer_name} not available, skipping.")
                    continue
                    
                # Get distance requirement and weight
                distance_req = criteria.distance_requirements.get(layer_name, 1000)
                weight = criteria.weights.get(layer_name, 5)
                avoid = layer_name in criteria.avoid
                
                # Project layer to UTM
                layer_utm = layers[layer_name].to_crs(grid.utm_crs)
                
                # Calculate distance raster
                distance_raster = SpatialAnalysisService.calculate_distance_raster(
                    layer_utm,
                    grid,
                    cell_size
                )
                
                if distance_raster is None:
                    continue
                
                # Reclassify the distance raster based on criteria
                if avoid:
                    # For layers to avoid (higher distance is better)
                    reclass_dict = {
                        (0, distance_req): 0,         # Too close (unsuitable)
                        (distance_req, float('inf')): 100  # Far enough (suitable)
                    }
                else:
                    # For layers to be near (lower distance is better)
                    reclass_dict = {
                        (0, distance_req): 100,        # Within range (suitable)
                        (distance_req, float('inf')): 0   # Too far (unsuitable)
                    }
                    
                reclassified = SpatialAnalysisService.reclassify_raster(distance_raster, reclass_dict)
                
                # Store the reclassified raster with its name
                named_rasters[layer_name] = {
                    'raster': reclassified,
                    'weight': weight,
                    'avoid': avoid,
                    'distance_req': distance_req
                }
                
                # Add to collection for overlay
                processed_rasters.append(reclassified)
                raster_weights.append(weight)
            
            # Perform weighted overlay
            if not processed_rasters:
                st.error("No rasters could be processed for analysis.")
                return None
                
            final_suitability = SpatialAnalysisService.weighted_overlay(processed_rasters, raster_weights)
            
            # Rescale to 0-100
            valid_mask = (final_suitability != -9999)
            if np.any(valid_mask):
                min_val = np.min(final_suitability[valid_mask])
                max_val = np.max(final_suitability[valid_mask])
                
                if max_val > min_val:
                    final_suitability[valid_mask] = (
                        (final_suitability[valid_mask] - min_val) / (max_val - min_val) * 100
                    )
            
            # Extract suitable locations
            locations = SpatialAnalysisService.extract_suitable_locations(
                final_suitability, 
                grid, 
                threshold_percentile=95
            )
            
            if not locations:
                st.warning("No suitable locations found meeting the criteria.")
                return None
            
            # Add geocoding information
            self.geocoding_service.reverse_geocode_locations(locations[:10])  # Only geocode top 10
            
            # Add nearby features information
            nearby_features = self.db_service.get_nearby_features(locations[:10])
            for location in locations[:10]:
                location_id = str(location.location_id)
                if location_id in nearby_features:
                    location.nearby_features = nearby_features[location_id]
            
            # Run Gemini analysis if available
            gemini_analysis = None
            if self.gemini_service:
                analysis_summary = f"Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M')}. Found {len(locations)} potential sites using {len(processed_rasters)} criteria layers."
                gemini_analysis = self.gemini_service.analyze_locations(locations, criteria, analysis_summary)
                
                # Update location objects with Gemini analysis
                if gemini_analysis and 'explanations' in gemini_analysis and 'considerations' in gemini_analysis:
                    for location in locations:
                        location_id = str(location.location_id)
                        if location_id in gemini_analysis['explanations']:
                            location.explanation = gemini_analysis['explanations'][location_id]
                        if location_id in gemini_analysis['considerations']:
                            location.considerations = gemini_analysis['considerations'][location_id]
            
            # Update AnalysisResult to include named rasters
            result = AnalysisResult(
                suitability_raster=final_suitability,
                grid=grid,
                locations=locations,
                criteria=criteria,
                gemini_analysis=gemini_analysis
            )
            
            # Add named rasters to a custom attribute
            result.named_rasters = named_rasters
            
            return result
            
        except Exception as e:
            st.error(f"Error in suitability analysis: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

