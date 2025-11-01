"""Data models for the LLM-Driven Spatial Decision Support System."""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
from shapely.geometry import Point


class DatabaseConfig:
    """Database connection configuration."""
    def __init__(self, host: str, port: str, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class AnalysisGrid:
    """Represents the spatial grid for analysis."""
    def __init__(self, transform, width: int, height: int, utm_crs: str, bounds: Tuple):
        self.transform = transform
        self.width = width
        self.height = height
        self.utm_crs = utm_crs
        self.bounds = bounds  # (xmin, ymin, xmax, ymax)
    
    def get_extent(self) -> List[float]:
        """Get extent in format suitable for visualization."""
        xmin, ymin, xmax, ymax = self.bounds
        return [xmin, xmax, ymin, ymax]


class AnalysisCriteria:
    """Stores criteria for spatial suitability analysis."""
    def __init__(self, 
                 layers: List[str], 
                 distance_requirements: Dict[str, float], 
                 weights: Dict[str, int], 
                 avoid: List[str], 
                 objective: str):
        self.layers = layers
        self.distance_requirements = distance_requirements
        self.weights = weights
        self.avoid = avoid
        self.objective = objective
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "layers": self.layers,
            "distance_requirements": self.distance_requirements,
            "weights": self.weights,
            "avoid": self.avoid,
            "objective": self.objective
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisCriteria':
        """Create AnalysisCriteria from dictionary."""
        return cls(
            layers=data.get("layers", []),
            distance_requirements=data.get("distance_requirements", {}),
            weights=data.get("weights", {}),
            avoid=data.get("avoid", []),
            objective=data.get("objective", "")
        )


class SuitableLocation:
    """Represents a suitable location found by analysis."""
    def __init__(self, 
                 location_id: int,
                 geometry: Point,
                 area_hectares: float, 
                 suitability_score: float,
                 latitude: float,
                 longitude: float,
                 address: Optional[str] = None,
                 neighborhood: Optional[str] = None,
                 city: Optional[str] = None,
                 nearby_features: Optional[Dict] = None):
        self.location_id = location_id
        self.geometry = geometry
        self.area_hectares = area_hectares
        self.suitability_score = suitability_score
        self.latitude = latitude
        self.longitude = longitude
        self.address = address
        self.neighborhood = neighborhood
        self.city = city
        self.nearby_features = nearby_features or {}
        self.explanation = ""
        self.considerations = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.location_id,
            "area_hectares": float(self.area_hectares),
            "suitability_score": float(self.suitability_score),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "address": str(self.address) if self.address else None,
            "neighborhood": str(self.neighborhood) if self.neighborhood else None,
            "city": str(self.city) if self.city else None
        }


class AnalysisResult:
    """Stores the results of a spatial suitability analysis."""
    def __init__(self, 
                 suitability_raster,
                 grid: AnalysisGrid,
                 locations: List[SuitableLocation],
                 criteria: AnalysisCriteria,
                 timestamp: datetime = None,
                 gemini_analysis: Optional[Dict] = None):
        self.suitability_raster = suitability_raster
        self.grid = grid
        self.locations = locations
        self.criteria = criteria
        self.timestamp = timestamp or datetime.now()
        self.gemini_analysis = gemini_analysis or {}
    
    def get_top_locations(self, n: int = 3) -> List[SuitableLocation]:
        """Get top n locations by suitability score."""
        return sorted(self.locations, key=lambda x: x.suitability_score, reverse=True)[:n]
        
    def to_json_serializable(self):
        """Convert to JSON serializable format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'criteria': self.criteria.to_dict(),
            'locations_count': len(self.locations),
            'grid_info': {
                'width': self.grid.width,
                'height': self.grid.height,
                'utm_crs': str(self.grid.utm_crs),
                'bounds': list(self.grid.bounds)
            },
            'gemini_analysis': self.gemini_analysis
        }

