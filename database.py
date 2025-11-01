"""Database service for interacting with PostgreSQL spatial database."""

from typing import Dict, List, Optional
import streamlit as st
import geopandas as gpd
from sqlalchemy import create_engine, text
from models import DatabaseConfig, SuitableLocation


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize with database configuration."""
        self.config = config
        self.engine = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            conn_string = self.config.get_connection_string()
            self.engine = create_engine(conn_string)
            
            # Test the connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
        except Exception as e:
            st.error(f"Database connection error: {e}")
            st.warning("Please check your database credentials and ensure PostgreSQL server is running.")
            return False
    
    def get_available_layers(self) -> List[str]:
        """Get list of available tables in the database."""
        if not self.engine:
            return []
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
                return [row[0] for row in result]
        except Exception as e:
            st.error(f"Error checking available tables: {e}")
            return []
    
    def fetch_layer(self, layer_name: str) -> Optional[gpd.GeoDataFrame]:
        """Fetch a specific layer from the database."""
        if not self.engine:
            st.error("No database connection available.")
            return None
        
        try:
            query = f"SELECT * FROM {layer_name}"
            gdf = gpd.read_postgis(query, self.engine, geom_col='geom')
            
            # Check if the layer has any rows
            if len(gdf) == 0:
                st.warning(f"Layer {layer_name} exists but contains no data.")
                return None
            
            # Ensure the geometry column is properly set
            if 'geom' in gdf.columns:
                gdf = gdf.set_geometry('geom')
            
            # Check CRS
            if gdf.crs is None:
                st.warning(f"Layer {layer_name} has no CRS information. Assuming WGS84 (EPSG:4326)")
                gdf.set_crs(epsg=4326, inplace=True)
            
            return gdf
        except Exception as e:
            st.error(f"Could not fetch layer {layer_name}: {e}")
            return None
    
    def fetch_layers(self, layer_names: List[str]) -> Dict[str, gpd.GeoDataFrame]:
        """Fetch multiple layers from the database."""
        layers = {}
        available_tables = self.get_available_layers()
        
        for layer_name in layer_names:
            if layer_name not in available_tables:
                st.warning(f"Layer {layer_name} does not exist in the database.")
                continue
            
            gdf = self.fetch_layer(layer_name)
            if gdf is not None:
                layers[layer_name] = gdf
                st.success(f"Successfully loaded {layer_name} with {len(gdf)} features.")
        
        return layers
    
    def get_nearby_features(self, locations: List[SuitableLocation], radius_meters: float = 1000) -> Dict[str, Dict]:
        """Find nearby features from database layers for locations."""
        if not self.engine:
            return {}
        
        result = {}
        available_layers = self.get_available_layers()
        
        # Layers to check (exclude any that aren't in the database)
        layers_to_check = ['schools', 'healthcare', 'bank', 'roads', 'railway', 'masjids']
        valid_layers = [layer for layer in layers_to_check if layer in available_layers]
        
        for location in locations:
            location_id = str(location.location_id)
            lat = location.latitude
            lon = location.longitude
            
            location_features = {}
            
            for layer in valid_layers:
                try:
                    # SQL query to find features within radius
                    query = f"""
                    SELECT name, ST_Distance(
                        ST_Transform(geom, 3857),
                        ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857)
                    ) as distance
                    FROM {layer}
                    WHERE ST_DWithin(
                        ST_Transform(geom, 3857),
                        ST_Transform(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326), 3857),
                        {radius_meters}
                    )
                    ORDER BY distance
                    LIMIT 5;
                    """
                    
                    # Execute query
                    with self.engine.connect() as conn:
                        query_result = conn.execute(text(query))
                        features = [{"name": row[0], "distance": row[1]} for row in query_result]
                    
                    location_features[layer] = features
                except Exception as e:
                    st.warning(f"Error querying {layer} for location {location_id}: {e}")
            
            result[location_id] = location_features
        
        return result
    
    def load_boundary(self, boundary_name: str = "islamabad", target_crs: str = "EPSG:4326") -> Optional[gpd.GeoDataFrame]:
        """Load a boundary from the database."""
        if not self.engine:
            return None
            
        try:
            gdf = self.fetch_layer(boundary_name)
            
            if gdf is None:
                return None
                
            # Project to the target CRS if needed
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
                
            return gdf
        except Exception as e:
            st.warning(f"Could not load boundary {boundary_name}: {e}")
            return None

