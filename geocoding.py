"""Geocoding service for reverse geocoding operations."""

from typing import Dict, List
from datetime import date, datetime
import streamlit as st
from geopy.geocoders import Nominatim
from models import SuitableLocation


class GeoCodingService:
    """Service for geocoding operations."""
    
    def __init__(self, user_agent: str = "land_suitability_analysis_app"):
        """Initialize with user agent."""
        self.geolocator = Nominatim(user_agent=user_agent)
    
    def reverse_geocode(self, location: SuitableLocation) -> Dict:
        """Reverse geocode a location to get address information."""
        try:
            result = self.geolocator.reverse(f"{location.latitude}, {location.longitude}", exactly_one=True)
            
            if not result:
                return {
                    "address": "Address not found",
                    "neighborhood": "",
                    "suburb": "",
                    "city": "",
                    "state": ""
                }
            
            address = result.raw['address']
            
            # Extract useful information from address
            neighborhood = address.get('neighbourhood', '')
            suburb = address.get('suburb', '')
            city_district = address.get('city_district', '')
            city = address.get('city', address.get('town', ''))
            state = address.get('state', '')
            
            # Format a readable location description
            location_description = ", ".join(filter(None, [neighborhood, suburb, city_district, city, state]))
            
            return {
                "address": result.address,
                "location_description": location_description if location_description else "Location details not available",
                "neighborhood": neighborhood,
                "suburb": suburb,
                "city": city,
                "state": state
            }
        except Exception as e:
            st.warning(f"Error in reverse geocoding: {e}")
            return {
                "address": "Could not retrieve address",
                "neighborhood": "",
                "suburb": "",
                "city": "",
                "state": ""
            }
    
    def reverse_geocode_locations(self, locations: List[SuitableLocation]) -> None:
        """Reverse geocode a list of locations and update their attributes."""
        for location in locations:
            geocode_result = self.reverse_geocode(location)
            location.address = geocode_result["address"]
            location.neighborhood = geocode_result["neighborhood"]
            location.city = geocode_result["city"]

