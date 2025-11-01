"""Visualization service for creating maps and charts."""

from datetime import date, datetime
import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
import folium
from typing import List, Optional
from models import SuitableLocation


class VisualizationService:
    """Service for data visualization."""
    
    @staticmethod
    def visualize_raster(raster_array: np.ndarray, 
                        transform, 
                        title: str, 
                        cmap: str = 'viridis', 
                        nodata: float = -9999, 
                        boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> plt.Figure:
        """Create a visualization of a raster."""
        # Create a masked array to handle nodata values
        masked_array = np.ma.masked_where(raster_array == nodata, raster_array)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Creating a custom colormap that goes from red to green
        colors = [(0.8, 0, 0), (1, 1, 0), (0, 0.8, 0)]  # Red to Yellow to Green
        custom_cmap = LinearSegmentedColormap.from_list('custom_suitability', colors)
        
        # Plot the raster
        img = ax.imshow(masked_array, cmap=custom_cmap)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Suitability Score')
        
        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Remove axes
        ax.set_axis_off()
        
        # Overlay boundary if provided
        if boundary_gdf is not None:
            for geom in boundary_gdf.geometry:
                if geom.type == 'Polygon':
                    mpl_poly = MplPolygon(np.array(geom.exterior.coords), fill=False, edgecolor='blue', linewidth=2)
                    ax.add_patch(mpl_poly)
                elif geom.type == 'MultiPolygon':
                    for part in geom.geoms:
                        mpl_poly = MplPolygon(np.array(part.exterior.coords), fill=False, edgecolor='blue', linewidth=2)
                        ax.add_patch(mpl_poly)
        
        return fig
    
    @staticmethod
    def create_locations_map(locations: List[SuitableLocation], 
                            top_location_ids: List[int] = None, 
                            boundary_gdf: Optional[gpd.GeoDataFrame] = None) -> folium.Map:
        """Create an interactive folium map with the suitable locations."""
        # Filter for top locations if specified
        if top_location_ids:
            top_locations = [loc for loc in locations if loc.location_id in top_location_ids]
        else:
            top_locations = locations[:3]  # Default to top 3
        
        if not top_locations:
            return None
        
        # Calculate map center
        center_lat = sum(loc.latitude for loc in top_locations) / len(top_locations)
        center_lon = sum(loc.longitude for loc in top_locations) / len(top_locations)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, 
                       tiles='CartoDB positron', control_scale=True)
        
        # Add boundary if provided
        if boundary_gdf is not None:
            # Convert any date fields to strings to avoid JSON serialization issues
            boundary_gdf_wgs84 = boundary_gdf.to_crs(epsg=4326).copy()
            
            # Convert date columns to strings
            for col in boundary_gdf_wgs84.columns:
                if col != boundary_gdf_wgs84._geometry_column_name:
                    # Check if column contains dates
                    if boundary_gdf_wgs84[col].dtype.kind == 'M' or \
                       (len(boundary_gdf_wgs84) > 0 and any(isinstance(x, (date, datetime)) 
                                                         for x in boundary_gdf_wgs84[col].dropna().head(1))):
                        boundary_gdf_wgs84[col] = boundary_gdf_wgs84[col].apply(
                            lambda x: str(x) if x is not None else None
                        )
            
            folium.GeoJson(
                data=boundary_gdf_wgs84.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'blue',
                    'weight': 2,
                    'dashArray': '5, 5'
                },
                name='Boundary'
            ).add_to(m)
        
        # Add top locations with enhanced information
        for location in top_locations:
            # Create simplified HTML for popup (avoiding complex data structures)
            html = f'''
            <div style="font-family: Arial; max-width: 350px;">
                <h3 style="color: #4A90E2;">Location {location.location_id}</h3>
                <div style="margin-bottom: 10px;">
                    <strong>Area:</strong> {location.area_hectares:.2f} hectares<br>
                    <strong>Suitability Score:</strong> {location.suitability_score:.2f}<br>
                    <strong>Coordinates:</strong> {location.latitude:.6f}, {location.longitude:.6f}
                </div>
            '''
            
            # Add address information if available - ensure strings are used
            if location.address:
                html += f'''
                <div style="margin-bottom: 10px;">
                    <strong>Address:</strong> {str(location.address)}<br>
                    <strong>Neighborhood:</strong> {str(location.neighborhood or "Not available")}<br>
                    <strong>City:</strong> {str(location.city or "Not available")}<br>
                </div>
                '''
            
            # Add nearby features if available - ensure all values are properly stringified
            if location.nearby_features:
                html += "<div style='margin-bottom: 10px;'><h4 style='margin-bottom: 5px;'>Nearby Features:</h4><ul style='padding-left: 20px; margin-top: 5px;'>"
                for layer, features in location.nearby_features.items():
                    if features:
                        html += f"<li><strong>{str(layer.title())}</strong>: "
                        feature_list = []
                        for f in features[:3]:
                            # Ensure we're using primitive types that are JSON serializable
                            name = str(f.get('name', ''))
                            distance = float(f.get('distance', 0))
                            feature_list.append(f"{name} ({distance:.0f}m)")
                        html += ", ".join(feature_list)
                        html += "</li>"
                html += "</ul></div>"
            
            # Add explanation and considerations if available - ensure they're strings
            if hasattr(location, 'explanation') and location.explanation:
                html += f'''
                <div style="margin-bottom: 10px;">
                    <h4 style="margin-bottom: 5px;">Analysis:</h4>
                    <p style="margin-top: 5px;">{str(location.explanation)}</p>
                </div>
                '''
            
            if hasattr(location, 'considerations') and location.considerations:
                html += f'''
                <div>
                    <h4 style="margin-bottom: 5px;">Development Considerations:</h4>
                    <p style="margin-top: 5px;">{str(location.considerations)}</p>
                </div>
                '''
            
            html += "</div>"
            
            iframe = folium.IFrame(html=html, width=370, height=400)
            popup = folium.Popup(iframe)
            
            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=popup,
                icon=folium.Icon(color='red', icon='star'),
                tooltip=f"Location {location.location_id} - Score: {location.suitability_score:.2f}"
            ).add_to(m)
        
        # Add all other locations with simpler markers
        other_locations = [loc for loc in locations if loc.location_id not in [l.location_id for l in top_locations]]
        for location in other_locations[:20]:  # Limit to 20 to avoid clutter
            folium.CircleMarker(
                location=[location.latitude, location.longitude],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                tooltip=f"Location {location.location_id} - Score: {location.suitability_score:.2f}"
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

