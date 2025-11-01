"""Spatial analysis service for GIS operations including grid creation, distance calculations, and raster processing."""

import re
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, Point
from typing import Dict, List, Optional
from pylusat import distance
from models import AnalysisGrid, SuitableLocation


class SpatialAnalysisService:
    """Service for spatial analysis operations."""
    
    @staticmethod
    def create_analysis_grid(layers: Dict[str, gpd.GeoDataFrame], cell_size: float = 30) -> Optional[AnalysisGrid]:
        """Create a consistent analysis grid for all layers."""
        # First, ensure all layers are in WGS84 for consistent processing
        for name, gdf in layers.items():
            if gdf.crs is None:
                st.warning(f"Layer {name} has no CRS. Assuming WGS84.")
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
                layers[name] = gdf
        
        # Get study area bounds in WGS84
        all_bounds = []
        for gdf in layers.values():
            if not gdf.empty:
                bounds = gdf.total_bounds
                all_bounds.append(bounds)
        
        if not all_bounds:
            st.error("No valid bounds found in any layer")
            return None
        
        # Calculate overall bounds in WGS84
        bounds_array = np.array(all_bounds)
        xmin = bounds_array[:, 0].min()
        ymin = bounds_array[:, 1].min()
        xmax = bounds_array[:, 2].max()
        ymax = bounds_array[:, 3].max()
        
        # Calculate centroid for UTM zone selection
        center_lon = (xmin + xmax) / 2
        center_lat = (ymin + ymax) / 2
        
        # Determine UTM zone
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone}" if center_lat >= 0 else f"EPSG:327{utm_zone}"
        
        # Create a temporary GeoDataFrame with the extent polygon in WGS84
        extent_poly = box(xmin, ymin, xmax, ymax)
        extent_gdf = gpd.GeoDataFrame(geometry=[extent_poly], crs="EPSG:4326")
        
        # Project the extent to UTM
        extent_gdf_utm = extent_gdf.to_crs(utm_crs)
        bounds_utm = extent_gdf_utm.total_bounds
        
        # Add buffer to ensure coverage
        buffer_size = cell_size * 2
        xmin_utm = bounds_utm[0] - buffer_size
        ymin_utm = bounds_utm[1] - buffer_size
        xmax_utm = bounds_utm[2] + buffer_size
        ymax_utm = bounds_utm[3] + buffer_size
        
        # Calculate grid dimensions
        width = int(np.ceil((xmax_utm - xmin_utm) / cell_size))
        height = int(np.ceil((ymax_utm - ymin_utm) / cell_size))
        
        # Create transform (from upper left corner)
        transform = rasterio.transform.from_origin(xmin_utm, ymax_utm, cell_size, cell_size)
        
        # Print debug information
        st.write("Analysis Grid Information:")
        st.write(f"WGS84 Bounds: {xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}")
        st.write(f"UTM Zone: {utm_zone}")
        st.write(f"UTM Bounds: {xmin_utm:.1f}, {ymin_utm:.1f}, {xmax_utm:.1f}, {ymax_utm:.1f}")
        st.write(f"Grid Dimensions: {width} x {height} cells")
        st.write(f"Cell Size: {cell_size} meters")
        
        return AnalysisGrid(transform, width, height, utm_crs, (xmin_utm, ymin_utm, xmax_utm, ymax_utm))
    
    @staticmethod
    def calculate_distance_raster(gdf: gpd.GeoDataFrame, grid: AnalysisGrid, cell_size: float = 30) -> Optional[np.ndarray]:
        """Calculate distance raster from features in the GeoDataFrame."""
        try:
            # Get geometry types present in the layer
            geom_types = set(gdf.geometry.type)
            
            # Create grid points for the raster
            xs = np.linspace(grid.transform.c, grid.transform.c + (grid.width - 1) * grid.transform.a, grid.width)
            ys = np.linspace(grid.transform.f, grid.transform.f + (grid.height - 1) * grid.transform.e, grid.height)
            grid_x, grid_y = np.meshgrid(xs, ys)
            grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            
            # Create GeoDataFrame with grid points
            grid_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), 
                crs=gdf.crs
            )

            # For point geometries
            if geom_types.issubset({'Point', 'MultiPoint'}):
                # Handle MultiPoint geometries by converting to Point if needed
                if 'MultiPoint' in geom_types:
                    gdf = gdf.copy()
                    mask = gdf.geometry.type == 'MultiPoint'
                    gdf.loc[mask, 'geometry'] = gdf.loc[mask, 'geometry'].apply(lambda g: g.centroid)
                
                try:
                    # Use PyLUSAT's to_point function
                    distances = distance.to_point(
                        input_gdf=grid_gdf,
                        point_gdf=gdf,
                        method='euclidean'
                    )
                except Exception as e:
                    st.warning(f"PyLUSAT to_point failed: {e}. Using direct calculation.")
                    
                    # Fallback to direct calculation
                    distances_array = np.full(len(grid_gdf), np.inf)
                    
                    # Extract coordinates from grid points and target points
                    grid_coords = np.array([(p.x, p.y) for p in grid_gdf.geometry])
                    target_coords = np.array([(p.x, p.y) for p in gdf.geometry])
                    
                    # Use vectorized operations for better performance
                    for i in range(len(grid_coords)):
                        # Calculate squared distances to all points at once
                        squared_dists = (target_coords[:, 0] - grid_coords[i, 0])**2 + (target_coords[:, 1] - grid_coords[i, 1])**2
                        # Get minimum distance
                        distances_array[i] = np.sqrt(np.min(squared_dists)) if len(squared_dists) > 0 else np.inf
                    
                    # Convert to pandas Series to match PyLUSAT's output format
                    distances = pd.Series(distances_array)
            
            # For line geometries
            elif geom_types.issubset({'LineString', 'MultiLineString'}):
                # Use pylusat's to_line to calculate distances
                try:
                    distances = distance.to_line(grid_gdf, gdf, cellsize=cell_size, method='euclidean')
                except Exception as e:
                    st.warning(f"PyLUSAT to_line failed: {e}. Using alternative method.")
                    
                    # Rasterize the line geometries
                    rasterized = np.zeros((grid.height, grid.width), dtype=np.uint8)
                    shapes = [(geom, 1) for geom in gdf.geometry]
                    rasterized = rasterize(
                        shapes=shapes,
                        out=rasterized,
                        transform=grid.transform,
                        fill=0,
                        all_touched=True
                    )
                    
                    # Calculate distance transform
                    binary_image = (rasterized == 0).astype(np.uint8)
                    dist_transform = distance_transform_edt(binary_image) * cell_size
                    
                    # Flatten the distances to match the grid points
                    distances = pd.Series(dist_transform.flatten())
            
            # For polygon geometries or mixed types, rasterize and use distance transform
            else:
                # Rasterize the geometries
                rasterized = np.zeros((grid.height, grid.width), dtype=np.uint8)
                shapes = [(geom, 1) for geom in gdf.geometry]
                rasterized = rasterize(
                    shapes=shapes,
                    out=rasterized,
                    transform=grid.transform,
                    fill=0,
                    all_touched=True
                )
                
                # Calculate distance transform (distance to nearest non-zero cell)
                binary_image = (rasterized == 0).astype(np.uint8)
                dist_transform = distance_transform_edt(binary_image) * cell_size
                
                # Flatten the distances to match the grid points
                distances = pd.Series(dist_transform.flatten())
            
            # Reshape the distances to a 2D array (raster)
            distance_raster = distances.values.reshape((grid.height, grid.width))
            
            # Verify there are no NaN values in the raster
            if np.isnan(distance_raster).any():
                st.warning("NaN values detected in the distance raster, replacing with max distance")
                max_dist = np.nanmax(distance_raster)
                distance_raster = np.nan_to_num(distance_raster, nan=max_dist)

            return distance_raster
        
        except Exception as e:
            st.error(f"Error in distance calculation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    @staticmethod
    def reclassify_raster(raster_array: np.ndarray, reclassify_dict: Dict, nodata_value: float = -9999) -> np.ndarray:
        """Reclassify raster based on value ranges with continuous gradient."""
        output = np.full_like(raster_array, nodata_value, dtype=float)
        
        # Create a mask for valid data
        valid_mask = (raster_array != nodata_value)
        
        # For continuous gradient approach, we'll use this flag to determine if we're using the old method
        use_continuous_scale = True
        for (min_val, max_val) in reclassify_dict.keys():
            if isinstance(min_val, str) or isinstance(max_val, str):
                use_continuous_scale = False  # Fall back to old method if string values
                break
        
        if use_continuous_scale:
            # Extract distance requirement from reclassify_dict for continuous scaling
            try:
                # We'll assume there are typically two ranges in the dict: (0, dist_req) and (dist_req, inf)
                items = list(reclassify_dict.items())
                
                # Determine if this is an "avoid" layer (where higher distance is better)
                if len(items) >= 2:
                    is_avoid = items[0][1] < items[1][1]  # Check if first item value < second item value
                    distance_req = items[0][0][1]  # Get the distance requirement
                    
                    # Create continuous gradient based on distance
                    if is_avoid:  # For layers to avoid (higher distance is better)
                        # Start at 0 for d=0, increase to 100 as d approaches 2*distance_req
                        max_distance = 2 * distance_req
                        # Apply sigmoid-like scaling: slow start, rapid middle, slow end
                        output[valid_mask] = np.clip(100 * (raster_array[valid_mask] / max_distance)**1.5, 0, 100)
                    else:  # For layers to be near (lower distance is better)
                        # Start at 100 for d=0, decrease to 0 as d approaches distance_req
                        # Apply exponential decay formula: 100 * e^(-3d/distance_req)
                        output[valid_mask] = 100 * np.exp(-3 * raster_array[valid_mask] / distance_req)
                        
                    return output
                    
            except (IndexError, TypeError) as e:
                # Fall back to old method if any issue occurs
                st.warning(f"Using standard reclassification due to: {e}")
                use_continuous_scale = False
        
        # Fall back to original discrete classification approach
        if not use_continuous_scale:
            # Apply reclassification only to valid data
            for (min_val, max_val), new_val in reclassify_dict.items():
                try:
                    # Convert min_val to float, handling string inputs
                    if isinstance(min_val, str):
                        # Extract numerical value using regex
                        numbers = re.findall(r'\d+\.?\d*', min_val)
                        if numbers:
                            min_val_float = float(numbers[0])
                        else:
                            st.warning(f"Could not extract numerical value from '{min_val}', using 0")
                            min_val_float = 0.0
                    
                    else:
                        min_val_float = float(min_val)
                    
                    # Handle infinity specially
                    if max_val == float('inf') or max_val == 'inf':
                        mask = (raster_array >= min_val_float) & valid_mask
                    else:
                        # Convert max_val to float, handling string inputs
                        if isinstance(max_val, str):
                            if max_val.lower() == 'inf' or max_val.lower() == 'infinity':
                                max_val_float = float('inf')
                            else:
                                # Extract numerical value using regex
                                numbers = re.findall(r'\d+\.?\d*', max_val)
                                if numbers:
                                    max_val_float = float(numbers[0])
                                else:
                                    st.warning(f"Could not extract numerical value from '{max_val}', using infinity")
                                    max_val_float = float('inf')
                        else:
                            max_val_float = float(max_val)
                        
                        mask = (raster_array >= min_val_float) & (raster_array <= max_val_float) & valid_mask
                        
                    output[mask] = new_val
                    
                except Exception as e:
                    st.error(f"Error in reclassification: {e} for values min={min_val}, max={max_val}")
                    # Continue with next reclassification rule
                    continue
        
        return output
    
    @staticmethod
    def weighted_overlay(rasters: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
        """Perform weighted overlay of multiple rasters."""
        # Validate inputs
        if not rasters or not weights:
            st.error("No rasters or weights provided for overlay")
            return None
        
        # Check if we have the same number of rasters and weights
        if len(rasters) != len(weights):
            st.error(f"Number of rasters ({len(rasters)}) doesn't match number of weights ({len(weights)})")
            return None
        
        # Check if all rasters have the same shape
        shapes = [r.shape for r in rasters]
        unique_shapes = set(shapes)
        
        if len(unique_shapes) > 1:
            st.warning(f"Rasters have different dimensions: {unique_shapes}")
            
            # Find the smallest dimensions that will fit all rasters
            min_height = min(shape[0] for shape in shapes)
            min_width = min(shape[1] for shape in shapes)
            
            st.info(f"Resizing all rasters to {min_height} x {min_width}")
            
            # Resize all rasters to the smallest dimension
            resized_rasters = []
            for i, raster in enumerate(rasters):
                if raster.shape != (min_height, min_width):
                    # Simple resize by slicing (faster than resampling for this purpose)
                    resized = raster[:min_height, :min_width]
                    resized_rasters.append(resized)
                else:
                    resized_rasters.append(raster)
            
            rasters = resized_rasters
        
        # Normalize weights to sum to 1
        weights = np.array(weights, dtype=float)
        weight_sum = weights.sum()
        
        if weight_sum == 0:
            st.error("Sum of weights is zero, cannot perform weighted overlay")
            return None
            
        weights = weights / weight_sum
        
        # Create output raster with the same shape as the input rasters
        output = np.zeros_like(rasters[0], dtype=float)
        
        # Create a mask to track valid data cells (cells where at least one raster has data)
        valid_mask = np.zeros_like(output, dtype=bool)
        
        # Apply weights
        for raster, weight in zip(rasters, weights):
            # Skip if weight is zero
            if weight == 0:
                continue
                
            # Create mask for nodata values
            raster_valid = (raster != -9999)
            
            # Update the overall valid mask
            valid_mask = valid_mask | raster_valid
            
            # Apply weight only to valid data
            output[raster_valid] += raster[raster_valid] * weight
        
        # Set nodata values in the output
        output[~valid_mask] = -9999
        
        return output
    
    @staticmethod
    def extract_suitable_locations(suitability_raster: np.ndarray, 
                                  grid: AnalysisGrid, 
                                  threshold_percentile: float = 95) -> List[SuitableLocation]:
        """Extract top suitable locations from the suitability raster."""
        try:
            # Create a mask for the highest suitability areas
            masked_raster = np.ma.masked_equal(suitability_raster, -9999)
            threshold = np.percentile(masked_raster.compressed(), threshold_percentile)
            high_suitability_mask = (masked_raster >= threshold) & (~np.ma.getmaskarray(masked_raster))
            
            # Find clusters of high suitability cells
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(high_suitability_mask)
            
            # Calculate size and average suitability of each cluster
            locations = []
            for i in range(1, num_features + 1):
                cluster_mask = labeled_array == i
                cluster_size = np.sum(cluster_mask)
                avg_suitability = np.mean(masked_raster[cluster_mask])
                
                # Find centroid of cluster
                rows, cols = np.where(cluster_mask)
                centroid_row = np.mean(rows)
                centroid_col = np.mean(cols)
                
                # Convert to UTM coordinates
                x = grid.transform.c + centroid_col * grid.transform.a
                y = grid.transform.f + centroid_row * grid.transform.e
                
                # Calculate area in hectares
                area_hectares = cluster_size * (30 ** 2) / 10000  # Assuming 30m cell size, convert to hectares
                
                # Create Point geometry
                point = Point(x, y)
                
                # Create a GeoDataFrame with the point in UTM coords
                point_gdf = gpd.GeoDataFrame(geometry=[point], crs=grid.utm_crs)
                
                # Convert to WGS84 for lat/lon
                point_wgs84 = point_gdf.to_crs("EPSG:4326")
                lon = point_wgs84.geometry.x[0]
                lat = point_wgs84.geometry.y[0]
                
                # Create a SuitableLocation object
                location = SuitableLocation(
                    location_id=i,
                    geometry=point,
                    area_hectares=area_hectares,
                    suitability_score=float(avg_suitability),
                    latitude=lat,
                    longitude=lon
                )
                
                locations.append(location)
            
            # Sort locations by suitability score
            locations.sort(key=lambda x: x.suitability_score, reverse=True)
            
            return locations
            
        except Exception as e:
            st.error(f"Error extracting locations: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return []

