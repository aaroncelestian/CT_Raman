import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import cv2
from skimage import filters, morphology, measure, exposure
from scipy import ndimage
import pickle
import io
import base64

# Configure page
st.set_page_config(
    page_title="CT-Raman Stone Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .colormap-preview {
        height: 30px;
        border-radius: 4px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class StoneAnalysisApp:
    def __init__(self):
        """Initialize the Stone Analysis App"""
        if 'initialized' not in st.session_state:
            self.initialize_session_state()
        
        # Load CT image if not already loaded
        if 'ct_image' not in st.session_state:
            self.load_ct_image()
    
    def initialize_session_state(self):
        """Initialize session state with default parameters"""
        # Optimized parameters
        st.session_state.dog_sigma1 = 5.2
        st.session_state.dog_sigma2 = 3.0
        st.session_state.stone_threshold = 0.50
        st.session_state.min_stone_size = 50000
        st.session_state.hole_fill_size = 5858
        
        # Composition thresholds
        st.session_state.bacteria_threshold = 24
        st.session_state.bacteria_rich_threshold = 44
        st.session_state.intergrowth_threshold = 58
        st.session_state.whewellite_rich_threshold = 75
        
        # Visual settings
        st.session_state.overlay_alpha = 0.5
        st.session_state.anti_aliasing = False
        st.session_state.current_colormap = 'original'
        
        # Available colormaps
        st.session_state.colormaps = {
            'original': ['#000000', '#8B0000', '#FF0000', '#32CD32', '#FFA500', '#FFD700'],
            'scientific': ['#000080', '#0066CC', '#00AA00', '#FFAA00', '#FF6600', '#CC0000'],
            'viridis': ['#440154', '#31688e', '#35b779', '#fde725', '#ff7f00', '#dc143c'],
            'plasma': ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
            'cool': ['#000080', '#0080ff', '#00ffff', '#80ff80', '#ffff00', '#ff8000']
        }
        
        # Line scan data
        st.session_state.line_scan_data = None
        st.session_state.line_coordinates = None
        
        st.session_state.initialized = True
    
    def load_ct_image(self):
        """Load the CT image"""
        try:
            ct_image = np.array(Image.open('slice_1092.tif'))
            st.session_state.ct_image = ct_image
            st.session_state.ct_shape = ct_image.shape
            st.session_state.ct_min = ct_image.min()
            st.session_state.ct_max = ct_image.max()
        except FileNotFoundError:
            st.error("Could not find slice_1092.tif. Please ensure the file is in the working directory.")
            st.stop()
    
    def apply_dog_filter(self):
        """Apply DoG filter with current parameters"""
        ct_image = st.session_state.ct_image
        ct_norm = (ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())
        
        gaussian1 = filters.gaussian(ct_norm, sigma=st.session_state.dog_sigma1)
        gaussian2 = filters.gaussian(ct_norm, sigma=st.session_state.dog_sigma2)
        dog_result = gaussian1 - gaussian2
        
        return (dog_result - dog_result.min()) / (dog_result.max() - dog_result.min())
    
    def create_stone_mask(self, dog_enhanced):
        """Create stone mask with current parameters"""
        ct_image = st.session_state.ct_image
        ct_norm = (ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())
        
        # Combined approach
        inverted_dog = 1.0 - dog_enhanced
        stone_score = 0.7 * ct_norm + 0.3 * inverted_dog
        
        # Apply threshold
        stone_candidates = stone_score > st.session_state.stone_threshold
        stone_candidates = morphology.remove_small_objects(
            stone_candidates, min_size=st.session_state.min_stone_size
        )
        
        if np.sum(stone_candidates) > 0:
            labeled_candidates = measure.label(stone_candidates)
            regions = measure.regionprops(labeled_candidates)
            largest_region = max(regions, key=lambda r: r.area)
            stone_mask = labeled_candidates == largest_region.label
        else:
            stone_mask = np.zeros_like(ct_image, dtype=bool)
        
        # Morphological cleanup
        kernel = morphology.disk(3)
        stone_mask = morphology.binary_opening(stone_mask, kernel)
        stone_mask = morphology.binary_closing(stone_mask, kernel)
        
        # Fill holes
        if st.session_state.hole_fill_size > 0:
            filled_mask = ndimage.binary_fill_holes(stone_mask)
            holes = filled_mask & ~stone_mask
            
            if np.sum(holes) > 0:
                labeled_holes = measure.label(holes)
                hole_regions = measure.regionprops(labeled_holes)
                
                for region in hole_regions:
                    if region.area <= st.session_state.hole_fill_size:
                        for coord in region.coords:
                            stone_mask[coord[0], coord[1]] = True
        
        return stone_mask
    
    def create_composition_map(self, stone_mask):
        """Create composition map with current thresholds"""
        ct_image = st.session_state.ct_image
        
        if np.sum(stone_mask) == 0:
            return np.zeros_like(ct_image, dtype=np.uint8)
        
        stone_intensities = ct_image[stone_mask]
        min_intensity = np.min(stone_intensities)
        max_intensity = np.max(stone_intensities)
        
        # Create composition zones
        composition_zones = np.zeros_like(ct_image, dtype=np.uint8)
        
        # Calculate threshold values
        range_span = max_intensity - min_intensity
        bacteria_thresh = min_intensity + (st.session_state.bacteria_threshold / 100.0) * range_span
        bacteria_rich_thresh = min_intensity + (st.session_state.bacteria_rich_threshold / 100.0) * range_span
        intergrowth_thresh = min_intensity + (st.session_state.intergrowth_threshold / 100.0) * range_span
        whewellite_rich_thresh = min_intensity + (st.session_state.whewellite_rich_threshold / 100.0) * range_span
        
        # Assign zones
        composition_zones[stone_mask] = 3  # Default to intergrowth
        
        # Override with specific zones
        pure_bacteria_mask = stone_mask & (ct_image <= bacteria_thresh)
        bacteria_rich_mask = stone_mask & (ct_image > bacteria_thresh) & (ct_image <= bacteria_rich_thresh)
        intergrowth_mask = stone_mask & (ct_image > bacteria_rich_thresh) & (ct_image <= intergrowth_thresh)
        whewellite_rich_mask = stone_mask & (ct_image > intergrowth_thresh) & (ct_image <= whewellite_rich_thresh)
        pure_whewellite_mask = stone_mask & (ct_image > whewellite_rich_thresh)
        
        composition_zones[pure_bacteria_mask] = 1
        composition_zones[bacteria_rich_mask] = 2
        composition_zones[intergrowth_mask] = 3
        composition_zones[whewellite_rich_mask] = 4
        composition_zones[pure_whewellite_mask] = 5
        
        return composition_zones
    
    def apply_anti_aliasing(self, image, factor=2):
        """Apply anti-aliasing to composition map"""
        if not st.session_state.anti_aliasing:
            return image
        
        h, w = image.shape
        upscaled = ndimage.zoom(image, factor, order=0)
        smoothed = ndimage.gaussian_filter(upscaled.astype(float), sigma=0.5)
        downscaled = ndimage.zoom(smoothed, 1/factor, order=1)
        
        return downscaled
    
    def create_plotly_image(self, image_data, title, colorscale='gray', show_colorbar=False):
        """Create a plotly figure for image display"""
        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(image_data),
            colorscale=colorscale,
            showscale=show_colorbar,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            width=400,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    def create_composition_figure(self, composition_map, stone_mask, title_suffix=""):
        """Create composition figure with custom colormap"""
        display_map = self.apply_anti_aliasing(composition_map.astype(float))
        colors = st.session_state.colormaps[st.session_state.current_colormap]
        
        fig = go.Figure(data=go.Heatmap(
            z=np.flipud(display_map),
            colorscale=[[i/5, colors[i]] for i in range(6)],
            zmin=0, zmax=5,
            showscale=True,
            colorbar=dict(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Zone: %{z}<extra></extra>'
        ))
        
        # Calculate statistics
        if np.sum(stone_mask) > 0:
            zone_counts = [np.sum(composition_map == i) for i in range(1, 6)]
            total_stone = sum(zone_counts)
            zone_percentages = [count/total_stone*100 if total_stone > 0 else 0 for count in zone_counts]
            
            stats_text = f"B:{zone_percentages[0]:.1f}% BR:{zone_percentages[1]:.1f}% I:{zone_percentages[2]:.1f}% WR:{zone_percentages[3]:.1f}% W:{zone_percentages[4]:.1f}%"
            title = f"Composition Zones ({st.session_state.current_colormap}){title_suffix}<br><sub>{stats_text}</sub>"
        else:
            title = f"Composition Zones ({st.session_state.current_colormap}){title_suffix}"
        
        fig.update_layout(
            title=title,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            width=500,
            height=500,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        return fig
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("🔬 Stone Analysis Controls")
        
        # DoG Parameters
        st.sidebar.subheader("DoG Filter Parameters")
        st.session_state.dog_sigma1 = st.sidebar.slider(
            "DoG σ1 (↑ reduces holes)", 0.1, 20.0, st.session_state.dog_sigma1, 0.1
        )
        st.session_state.dog_sigma2 = st.sidebar.slider(
            "DoG σ2 (↓ reduces holes)", 0.1, 10.0, st.session_state.dog_sigma2, 0.1
        )
        
        # Stone Detection
        st.sidebar.subheader("Stone Detection")
        st.session_state.stone_threshold = st.sidebar.slider(
            "Stone Threshold", 0.1, 0.9, st.session_state.stone_threshold, 0.01
        )
        st.session_state.min_stone_size = st.sidebar.number_input(
            "Min Stone Size (pixels)", 10000, 200000, st.session_state.min_stone_size, 1000
        )
        st.session_state.hole_fill_size = st.sidebar.number_input(
            "Hole Fill Size (pixels)", 0, 10000, st.session_state.hole_fill_size, 100
        )
        
        # Composition Thresholds
        st.sidebar.subheader("Composition Thresholds (%)")
        st.session_state.bacteria_threshold = st.sidebar.slider(
            "Pure Bacteria (0-X%)", 5, 35, st.session_state.bacteria_threshold, 1
        )
        st.session_state.bacteria_rich_threshold = st.sidebar.slider(
            "Bacteria-Rich (X-Y%)", 25, 55, st.session_state.bacteria_rich_threshold, 1
        )
        st.session_state.intergrowth_threshold = st.sidebar.slider(
            "Intergrowth (Y-Z%)", 45, 75, st.session_state.intergrowth_threshold, 1
        )
        st.session_state.whewellite_rich_threshold = st.sidebar.slider(
            "Whewellite-Rich (Z-W%)", 65, 95, st.session_state.whewellite_rich_threshold, 1
        )
        
        # Visual Settings
        st.sidebar.subheader("Visual Settings")
        st.session_state.overlay_alpha = st.sidebar.slider(
            "Overlay Transparency", 0.0, 1.0, st.session_state.overlay_alpha, 0.05
        )
        st.session_state.anti_aliasing = st.sidebar.checkbox(
            "Anti-Aliasing", st.session_state.anti_aliasing
        )
        
        # Colormap selection with preview
        st.sidebar.write("**Colormap:**")
        for name, colors in st.session_state.colormaps.items():
            col1, col2 = st.sidebar.columns([1, 3])
            with col1:
                if st.radio("", [name], key=f"radio_{name}", label_visibility="collapsed"):
                    st.session_state.current_colormap = name
            with col2:
                # Create colormap preview
                gradient_html = f"""
                <div class="colormap-preview" style="background: linear-gradient(to right, {', '.join(colors)});"></div>
                """
                st.markdown(gradient_html, unsafe_allow_html=True)
        
        # Action buttons
        st.sidebar.subheader("Actions")
        if st.sidebar.button("🔄 Recalculate", type="primary"):
            self.recalculate_analysis()
        
        if st.sidebar.button("💾 Save Settings"):
            self.save_settings()
        
        if st.sidebar.button("📤 Export Results"):
            self.export_results()
    
    def recalculate_analysis(self):
        """Recalculate all analysis components"""
        with st.spinner("Recalculating analysis..."):
            # Store results in session state
            st.session_state.dog_enhanced = self.apply_dog_filter()
            st.session_state.stone_mask = self.create_stone_mask(st.session_state.dog_enhanced)
            st.session_state.composition_map = self.create_composition_map(st.session_state.stone_mask)
            
        st.success("Analysis updated!")
    
    def save_settings(self):
        """Save current settings to file"""
        settings = {
            'dog_sigma1': st.session_state.dog_sigma1,
            'dog_sigma2': st.session_state.dog_sigma2,
            'stone_threshold': st.session_state.stone_threshold,
            'hole_fill_size': st.session_state.hole_fill_size,
            'min_stone_size': st.session_state.min_stone_size,
            'overlay_alpha': st.session_state.overlay_alpha,
            'anti_aliasing': st.session_state.anti_aliasing,
            'current_colormap': st.session_state.current_colormap,
            'bacteria_threshold': st.session_state.bacteria_threshold,
            'bacteria_rich_threshold': st.session_state.bacteria_rich_threshold,
            'intergrowth_threshold': st.session_state.intergrowth_threshold,
            'whewellite_rich_threshold': st.session_state.whewellite_rich_threshold
        }
        
        with open('streamlit_stone_settings.pkl', 'wb') as f:
            pickle.dump(settings, f)
        
        st.sidebar.success("Settings saved!")
    
    def export_results(self):
        """Export analysis results"""
        # Implementation for exporting results
        st.sidebar.info("Export functionality coming soon!")
    
    def run(self):
        """Main application runner"""
        # Sidebar
        self.render_sidebar()
        
        # Main title
        st.title("🔬 CT-Raman Kidney Stone Analysis")
        st.markdown("Interactive analysis tool for correlating CT imaging with Raman spectroscopy data")
        
        # Initialize analysis if not done
        if 'dog_enhanced' not in st.session_state:
            self.recalculate_analysis()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Analysis", "📈 Line Scan", "📋 Export"])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_analysis_tab()
        
        with tab3:
            self.render_line_scan_tab()
        
        with tab4:
            self.render_export_tab()
    
    def render_overview_tab(self):
        """Render overview tab with key metrics and images"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics
        stone_coverage = np.sum(st.session_state.stone_mask) / st.session_state.stone_mask.size * 100
        
        with col1:
            st.metric("Stone Coverage", f"{stone_coverage:.1f}%")
        
        with col2:
            st.metric("Image Size", f"{st.session_state.ct_shape[0]}×{st.session_state.ct_shape[1]}")
        
        with col3:
            st.metric("Intensity Range", f"{st.session_state.ct_min}-{st.session_state.ct_max}")
        
        with col4:
            aa_status = "Enabled" if st.session_state.anti_aliasing else "Disabled"
            st.metric("Anti-Aliasing", aa_status)
        
        # Image grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original CT Image")
            fig_orig = self.create_plotly_image(st.session_state.ct_image, "Original CT")
            st.plotly_chart(fig_orig, use_container_width=True)
        
        with col2:
            st.subheader("Stone Mask")
            fig_mask = self.create_plotly_image(st.session_state.stone_mask, "Stone Mask")
            st.plotly_chart(fig_mask, use_container_width=True)
    
    def render_analysis_tab(self):
        """Render detailed analysis tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("DoG Enhanced")
            fig_dog = self.create_plotly_image(st.session_state.dog_enhanced, "DoG Enhanced")
            st.plotly_chart(fig_dog, use_container_width=True)
            
            st.subheader("Composition Zones")
            fig_comp = self.create_composition_figure(
                st.session_state.composition_map, 
                st.session_state.stone_mask
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            # CLAHE enhanced CT
            ct_norm = (st.session_state.ct_image - st.session_state.ct_image.min()) / (st.session_state.ct_image.max() - st.session_state.ct_image.min())
            ct_uint8 = (ct_norm * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            ct_clahe = clahe.apply(ct_uint8)
            
            st.subheader("CLAHE Enhanced")
            fig_clahe = self.create_plotly_image(ct_clahe, "CLAHE Enhanced")
            st.plotly_chart(fig_clahe, use_container_width=True)
            
            # Composition statistics
            st.subheader("Composition Statistics")
            if np.sum(st.session_state.stone_mask) > 0:
                zone_counts = [np.sum(st.session_state.composition_map == i) for i in range(1, 6)]
                total_stone = sum(zone_counts)
                
                zone_names = ['Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
                zone_percentages = [count/total_stone*100 if total_stone > 0 else 0 for count in zone_counts]
                
                df = pd.DataFrame({
                    'Zone': zone_names,
                    'Pixels': zone_counts,
                    'Percentage': zone_percentages
                })
                
                st.dataframe(df, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(df, values='Percentage', names='Zone', 
                               title="Composition Distribution",
                               color_discrete_sequence=st.session_state.colormaps[st.session_state.current_colormap][1:])
                st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_line_scan_tab(self):
        """Render line scan analysis tab"""
        st.subheader("📈 Line Scan Analysis")
        st.markdown("Click on the composition map to select line scan coordinates, then click 'Extract Line Scan'")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive composition map for line selection
            fig_comp = self.create_composition_figure(
                st.session_state.composition_map, 
                st.session_state.stone_mask,
                " - Click to Select Line"
            )
            
            # Get click events (this is a simplified version - in practice, you'd use plotly callbacks)
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with col2:
            st.subheader("Line Selection")
            
            # Manual coordinate input
            st.write("**Start Point:**")
            start_x = st.number_input("Start X", 0, st.session_state.ct_shape[1]-1, 500)
            start_y = st.number_input("Start Y", 0, st.session_state.ct_shape[0]-1, 500)
            
            st.write("**End Point:**")
            end_x = st.number_input("End X", 0, st.session_state.ct_shape[1]-1, 1000)
            end_y = st.number_input("End Y", 0, st.session_state.ct_shape[0]-1, 1000)
            
            if st.button("📊 Extract Line Scan", type="primary"):
                self.extract_line_scan(start_y, start_x, end_y, end_x)
        
        # Display line scan results
        if st.session_state.line_scan_data is not None:
            st.subheader("Line Scan Results")
            
            # Create line scan plot
            x_pixels = np.arange(len(st.session_state.line_scan_data))
            colors = st.session_state.colormaps[st.session_state.current_colormap]
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=x_pixels, 
                y=st.session_state.line_scan_data,
                mode='lines+markers',
                name='Composition',
                line=dict(width=3),
                marker=dict(size=4)
            ))
            
            fig_line.update_layout(
                title=f"Line Scan Profile ({len(st.session_state.line_scan_data)} pixels)",
                xaxis_title="Pixel Position Along Line",
                yaxis_title="Composition Type",
                yaxis=dict(
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
                ),
                height=400
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Line scan statistics
            composition_names = ['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
            unique_values, counts = np.unique(st.session_state.line_scan_data, return_counts=True)
            
            line_stats = []
            for val, count in zip(unique_values, counts):
                if val < len(composition_names):
                    line_stats.append({
                        'Zone': composition_names[int(val)],
                        'Pixels': count,
                        'Percentage': count/len(st.session_state.line_scan_data)*100
                    })
            
            df_line = pd.DataFrame(line_stats)
            st.dataframe(df_line, use_container_width=True)
    
    def extract_line_scan(self, start_y, start_x, end_y, end_x):
        """Extract line scan data"""
        line_profile = measure.profile_line(
            st.session_state.composition_map,
            (start_y, start_x),
            (end_y, end_x),
            linewidth=1,
            mode='constant'
        )
        
        st.session_state.line_scan_data = line_profile
        st.session_state.line_coordinates = [(start_y, start_x), (end_y, end_x)]
        
        st.success(f"Line scan extracted: {len(line_profile)} pixels")
    
    def render_export_tab(self):
        """Render export tab"""
        st.subheader("📋 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Options")
            
            export_format = st.selectbox("Export Format", ["PNG", "SVG", "PDF", "Data (CSV)"])
            include_stats = st.checkbox("Include Statistics", True)
            high_dpi = st.checkbox("High DPI (300)", True)
            
            if st.button("📤 Export Analysis", type="primary"):
                st.info("Export functionality will be implemented based on selected options")
        
        with col2:
            st.subheader("Current Settings Summary")
            
            settings_summary = f"""
            **DoG Parameters:**
            - σ1: {st.session_state.dog_sigma1}
            - σ2: {st.session_state.dog_sigma2}
            
            **Stone Detection:**
            - Threshold: {st.session_state.stone_threshold}
            - Min Size: {st.session_state.min_stone_size:,}
            - Hole Fill: {st.session_state.hole_fill_size:,}
            
            **Visual:**
            - Colormap: {st.session_state.current_colormap}
            - Anti-aliasing: {'On' if st.session_state.anti_aliasing else 'Off'}
            - Overlay Alpha: {st.session_state.overlay_alpha}
            """
            
            st.markdown(settings_summary)

# Main app execution
if __name__ == "__main__":
    app = StoneAnalysisApp()
    app.run() 