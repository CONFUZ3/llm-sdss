"""PDF report generation service."""

import os
import tempfile
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from typing import Optional
from models import AnalysisResult
from visualization import VisualizationService


class PDFReportService:
    """Service for generating PDF reports of analysis results."""
    
    def __init__(self, result: AnalysisResult, boundary_gdf: Optional[gpd.GeoDataFrame] = None):
        """Initialize with analysis results."""
        self.result = result
        self.boundary_gdf = boundary_gdf
        self.styles = getSampleStyleSheet()
        self.initialize_styles()
    
    def initialize_styles(self):
        """Initialize custom styles for the PDF report."""
        # Add custom styles - with checks to avoid duplicates
        style_definitions = [
            ('ReportTitle', {
                'parent': self.styles['Heading1'],
                'fontSize': 20,
                'alignment': TA_CENTER,
                'spaceAfter': 12
            }),
            ('Subtitle', {
                'parent': self.styles['Heading2'],
                'fontSize': 16,
                'spaceBefore': 12,
                'spaceAfter': 6
            }),
            ('Body', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'spaceBefore': 6,
                'spaceAfter': 6
            }),
            ('Location', {
                'parent': self.styles['Normal'],
                'fontSize': 12,
                'textColor': colors.navy,
                'spaceBefore': 6,
                'spaceAfter': 6
            })
        ]
        
        # Add styles if they don't already exist
        for style_name, style_props in style_definitions:
            if style_name not in self.styles:
                self.styles.add(ParagraphStyle(name=style_name, **style_props))
    
    def generate_report(self, output_path: str) -> bool:
        """Generate a PDF report of the analysis results."""
        # List to keep track of temporary files
        temp_files = []
        
        try:
            # Create a PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=1*cm,
                leftMargin=1*cm,
                topMargin=1.5*cm,
                bottomMargin=1.5*cm
            )
            
            # Build the content
            content = []
            
            # Add title
            title = Paragraph("LLM-Driven Spatial Decision Support System - Analysis Report", self.styles['ReportTitle'])
            content.append(title)
            
            # Add date - using string format to avoid JSON serialization issues
            date_text = Paragraph(f"Generated on: {self.result.timestamp.strftime('%Y-%m-%d %H:%M')}", self.styles['Body'])
            content.append(date_text)
            content.append(Spacer(1, 0.5*cm))
            
            # Add objective
            objective = Paragraph("Analysis Objective", self.styles['Subtitle'])
            content.append(objective)
            objective_text = Paragraph(self.result.criteria.objective, self.styles['Body'])
            content.append(objective_text)
            content.append(Spacer(1, 0.5*cm))
            
            # Add criteria summary
            criteria = Paragraph("Analysis Criteria", self.styles['Subtitle'])
            content.append(criteria)
            
            # Layers and weights table
            layers_data = [['Layer', 'Distance Requirement', 'Weight', 'Avoid?']]
            for layer in self.result.criteria.layers:
                distance = self.result.criteria.distance_requirements.get(layer, '')
                weight = self.result.criteria.weights.get(layer, '')
                avoid = "Yes" if layer in self.result.criteria.avoid else "No"
                layers_data.append([layer, f"{distance} meters", weight, avoid])
            
            layers_table = Table(layers_data, colWidths=[4*cm, 4*cm, 2*cm, 2*cm])
            layers_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            content.append(layers_table)
            content.append(Spacer(1, 0.5*cm))
            
            # Add individual layer rasters
            if hasattr(self.result, 'named_rasters') and self.result.named_rasters:
                layer_rasters_title = Paragraph("Individual Layer Analysis", self.styles['Subtitle'])
                content.append(layer_rasters_title)
                content.append(Paragraph("Each layer was analyzed individually to create suitability surfaces before combining them.", self.styles['Body']))
                content.append(Spacer(1, 0.3*cm))
                
                # Create a plot for each raster
                for layer_name, raster_info in self.result.named_rasters.items():
                    # Create a subtitle for the layer
                    avoid_text = "avoid" if raster_info['avoid'] else "proximity to"
                    layer_title = Paragraph(f"{layer_name.title()} ({avoid_text})", self.styles['Location'])
                    content.append(layer_title)
                    
                    # Add layer details
                    distance_req = raster_info['distance_req']
                    weight = raster_info['weight']
                    layer_details = Paragraph(
                        f"Distance requirement: {distance_req}m | Weight: {weight}/10", 
                        self.styles['Body']
                    )
                    content.append(layer_details)
                    
                    # Generate visualization for this layer's raster
                    layer_map_title = f"{layer_name.title()} Suitability"
                    if raster_info['avoid']:
                        layer_map_title += " (higher values = further away)"
                    else:
                        layer_map_title += " (higher values = closer)"
                    
                    fig = VisualizationService.visualize_raster(
                        raster_info['raster'],
                        self.result.grid.transform,
                        layer_map_title,
                        boundary_gdf=self.boundary_gdf
                    )
                    
                    # Save figure to a temporary file
                    fd, temp_map_path = tempfile.mkstemp(suffix='.png')
                    os.close(fd)
                    temp_files.append(temp_map_path)
                    
                    try:
                        fig.savefig(temp_map_path, format='png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        
                        if os.path.exists(temp_map_path) and os.path.getsize(temp_map_path) > 0:
                            img = Image(temp_map_path, width=16*cm, height=10*cm)
                            content.append(img)
                        else:
                            content.append(Paragraph(f"Could not generate map for {layer_name}.", self.styles['Body']))
                    except Exception as e:
                        content.append(Paragraph(f"Error generating map for {layer_name}: {str(e)}", self.styles['Body']))
                    
                    content.append(Spacer(1, 0.5*cm))
            
            # Add suitability map
            map_title = Paragraph("Final Suitability Map", self.styles['Subtitle'])
            content.append(map_title)
            
            # Generate map image
            fig = VisualizationService.visualize_raster(
                self.result.suitability_raster,
                self.result.grid.transform,
                "Spatial Suitability Analysis",
                boundary_gdf=self.boundary_gdf
            )
            
            # Save figure to a temporary file
            fd, temp_map_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            temp_files.append(temp_map_path)
            
            try:
                fig.savefig(temp_map_path, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Check if file exists and has content
                if os.path.exists(temp_map_path) and os.path.getsize(temp_map_path) > 0:
                    img = Image(temp_map_path, width=16*cm, height=12*cm)
                    content.append(img)
                else:
                    st.warning(f"Map image file is empty or not accessible: {temp_map_path}")
                    map_note = Paragraph(
                        "Note: The suitability map could not be included in this PDF. Please refer to the web application.",
                        self.styles['Body']
                    )
                    content.append(map_note)
            except Exception as e:
                st.error(f"Error saving map image: {e}")
                map_note = Paragraph(
                    "Note: The suitability map could not be included in this PDF. Please refer to the web application.",
                    self.styles['Body']
                )
                content.append(map_note)
            
            content.append(Spacer(1, 0.5*cm))
            
            # Add top locations section
            locations_title = Paragraph("Top Suitable Locations", self.styles['Subtitle'])
            content.append(locations_title)
            
            # Get top 3 locations
            top_locations = self.result.get_top_locations(3)
            
            # If we have Gemini analysis, use its top locations instead
            if self.result.gemini_analysis and 'top_locations' in self.result.gemini_analysis:
                top_ids = self.result.gemini_analysis['top_locations']
                top_locations = [loc for loc in self.result.locations if loc.location_id in top_ids]
            
            # Add a note about the interactive map
            map_note = Paragraph(
                "Note: An interactive map is available in the web application.",
                self.styles['Body']
            )
            content.append(map_note)
            content.append(Spacer(1, 0.5*cm))
            
            # Add details for each top location
            for location in top_locations:
                location_title = Paragraph(f"Location {location.location_id}", self.styles['Location'])
                content.append(location_title)
                
                details = [
                    f"Area: {location.area_hectares:.2f} hectares",
                    f"Suitability Score: {location.suitability_score:.2f}",
                    f"Coordinates: {location.latitude:.6f}, {location.longitude:.6f}"
                ]
                
                if location.address:
                    details.append(f"Address: {str(location.address)}")
                
                if location.neighborhood:
                    details.append(f"Neighborhood: {str(location.neighborhood)}")
                
                if location.city:
                    details.append(f"City: {str(location.city)}")
                
                # Convert details to paragraphs
                for detail in details:
                    p = Paragraph(detail, self.styles['Body'])
                    content.append(p)
                
                # Add explanation if available from Gemini
                if location.explanation:
                    explanation_title = Paragraph("Analysis:", self.styles['Body'])
                    content.append(explanation_title)
                    explanation = Paragraph(str(location.explanation), self.styles['Body'])
                    content.append(explanation)
                
                # Add considerations if available from Gemini
                if location.considerations:
                    consid_title = Paragraph("Development Considerations:", self.styles['Body'])
                    content.append(consid_title)
                    consid = Paragraph(str(location.considerations), self.styles['Body'])
                    content.append(consid)
                
                # Add nearby features if available
                if location.nearby_features:
                    nearby_title = Paragraph("Nearby Features:", self.styles['Body'])
                    content.append(nearby_title)
                    
                    for layer, features in location.nearby_features.items():
                        if features:
                            feature_text = f"<b>{str(layer).title()}:</b> "
                            feature_details = []
                            for f in features[:3]:
                                name = str(f.get('name', ''))
                                distance = float(f.get('distance', 0))
                                feature_details.append(f"{name} ({distance:.0f}m)")
                            feature_text += ", ".join(feature_details)
                            feature_para = Paragraph(feature_text, self.styles['Body'])
                            content.append(feature_para)
                
                content.append(Spacer(1, 0.5*cm))
            
            # Add comparison section if available from Gemini
            if self.result.gemini_analysis and 'comparison' in self.result.gemini_analysis:
                comparison_title = Paragraph("Comparative Analysis of Top Locations", self.styles['Subtitle'])
                content.append(comparison_title)
                
                comparison_text = Paragraph(str(self.result.gemini_analysis['comparison']), self.styles['Body'])
                content.append(comparison_text)
                content.append(Spacer(1, 0.5*cm))
            
            # Add overall summary if available from Gemini
            if self.result.gemini_analysis and 'overall_summary' in self.result.gemini_analysis:
                summary_title = Paragraph("Overall Summary", self.styles['Subtitle'])
                content.append(summary_title)
                
                summary_text = Paragraph(str(self.result.gemini_analysis['overall_summary']), self.styles['Body'])
                content.append(summary_text)
            
            # Build the document
            doc.build(content)
            
            # Success - return true
            return True
            
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as cleanup_error:
                    st.warning(f"Could not delete temporary file {temp_file}: {cleanup_error}")

