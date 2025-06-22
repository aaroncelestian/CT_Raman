#!/usr/bin/env python3
"""
CT-Raman Kidney Stone Analysis Widget Interface
Standalone version that can be run directly or imported into Jupyter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display, clear_output
import cv2
from PIL import Image
from skimage import filters, morphology, measure, exposure
from scipy import ndimage
import pickle
import pandas as pd
from matplotlib.patches import Circle

class StoneAnalysisWidget:
    def __init__(self):
        """Initialize the Stone Analysis Widget Interface"""
        # Load CT image
        try:
            self.ct_image = np.array(Image.open('slice_1092.tif'))
            self.ct_shape = self.ct_image.shape
            print(f"✅ CT image loaded: {self.ct_shape}")
        except FileNotFoundError:
            print("❌ Error: Could not find 'slice_1092.tif'")
            print("Please ensure the CT image file is in the current directory.")
            return
        
        # Optimized default parameters
        self.dog_sigma1 = 5.2
        self.dog_sigma2 = 3.0
        self.stone_threshold = 0.50
        self.min_stone_size = 50000
        self.hole_fill_size = 5858
        self.bacteria_threshold = 24
        self.bacteria_rich_threshold = 44
        self.intergrowth_threshold = 58
        self.whewellite_rich_threshold = 75
        self.overlay_alpha = 0.5
        self.anti_aliasing = False
        self.current_colormap = 'original'
        
        # Available colormaps
        self.colormaps = {
            'original': ['#000000', '#8B0000', '#FF0000', '#32CD32', '#FFA500', '#FFD700'],
            'scientific': ['#000080', '#0066CC', '#00AA00', '#FFAA00', '#FF6600', '#CC0000'],
            'viridis': ['#440154', '#31688e', '#35b779', '#fde725', '#ff7f00', '#dc143c'],
            'plasma': ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
            'cool': ['#000080', '#0080ff', '#00ffff', '#80ff80', '#ffff00', '#ff8000']
        }
        
        # Line scan data
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.click_count = 0
        
        # Analysis results
        self.dog_enhanced = None
        self.stone_mask = None
        self.composition_map = None
        
        # Create widgets
        self.create_widgets()
        self.create_layout()
        
        # Initial calculation
        print("🔄 Running initial analysis...")
        self.recalculate_analysis()
        print("✅ Interface initialized!")
    
    def create_widgets(self):
        """Create all UI widgets"""
        # DoG parameters
        self.sigma1_slider = widgets.FloatSlider(
            value=self.dog_sigma1, min=0.1, max=20.0, step=0.1,
            description='DoG σ1:', style={'description_width': '100px'}
        )
        self.sigma2_slider = widgets.FloatSlider(
            value=self.dog_sigma2, min=0.1, max=10.0, step=0.1,
            description='DoG σ2:', style={'description_width': '100px'}
        )
        
        # Stone detection
        self.threshold_slider = widgets.FloatSlider(
            value=self.stone_threshold, min=0.1, max=0.9, step=0.01,
            description='Stone Thresh:', style={'description_width': '100px'}
        )
        self.min_size_input = widgets.IntText(
            value=self.min_stone_size, description='Min Size:', style={'description_width': '100px'}
        )
        self.hole_fill_input = widgets.IntText(
            value=self.hole_fill_size, description='Hole Fill:', style={'description_width': '100px'}
        )
        
        # Composition thresholds
        self.bacteria_slider = widgets.IntSlider(
            value=self.bacteria_threshold, min=5, max=35, step=1,
            description='Bacteria %:', style={'description_width': '100px'}
        )
        self.bacteria_rich_slider = widgets.IntSlider(
            value=self.bacteria_rich_threshold, min=25, max=55, step=1,
            description='B-Rich %:', style={'description_width': '100px'}
        )
        self.intergrowth_slider = widgets.IntSlider(
            value=self.intergrowth_threshold, min=45, max=75, step=1,
            description='Intergrowth %:', style={'description_width': '100px'}
        )
        self.whewellite_rich_slider = widgets.IntSlider(
            value=self.whewellite_rich_threshold, min=65, max=95, step=1,
            description='W-Rich %:', style={'description_width': '100px'}
        )
        
        # Visual settings
        self.alpha_slider = widgets.FloatSlider(
            value=self.overlay_alpha, min=0.0, max=1.0, step=0.05,
            description='Overlay α:', style={'description_width': '100px'}
        )
        self.aa_checkbox = widgets.Checkbox(
            value=self.anti_aliasing, description='Anti-Aliasing'
        )
        self.colormap_dropdown = widgets.Dropdown(
            options=list(self.colormaps.keys()), value=self.current_colormap,
            description='Colormap:', style={'description_width': '100px'}
        )
        
        # Action buttons
        self.recalc_button = widgets.Button(
            description='🔄 Recalculate', button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.save_button = widgets.Button(
            description='💾 Save Settings', button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.clear_line_button = widgets.Button(
            description='🗑️ Clear Line', button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # Output areas
        self.output_main = widgets.Output()
        self.output_line_scan = widgets.Output()
        self.output_stats = widgets.Output()
        
        # Connect event handlers
        self.recalc_button.on_click(self.on_recalculate)
        self.save_button.on_click(self.on_save)
        self.clear_line_button.on_click(self.on_clear_line)
    
    def create_layout(self):
        """Create the widget layout"""
        # Control panels
        dog_controls = widgets.VBox([
            widgets.HTML('<b>🎛️ DoG Filter Parameters</b>'),
            self.sigma1_slider,
            self.sigma2_slider
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px'))
        
        stone_controls = widgets.VBox([
            widgets.HTML('<b>🎯 Stone Detection</b>'),
            self.threshold_slider,
            self.min_size_input,
            self.hole_fill_input
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px'))
        
        comp_controls = widgets.VBox([
            widgets.HTML('<b>🧪 Composition Thresholds</b>'),
            self.bacteria_slider,
            self.bacteria_rich_slider,
            self.intergrowth_slider,
            self.whewellite_rich_slider
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px'))
        
        visual_controls = widgets.VBox([
            widgets.HTML('<b>🎨 Visual Settings</b>'),
            self.alpha_slider,
            self.aa_checkbox,
            self.colormap_dropdown
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px'))
        
        action_controls = widgets.VBox([
            widgets.HTML('<b>⚡ Actions</b>'),
            self.recalc_button,
            self.save_button,
            self.clear_line_button
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='5px'))
        
        # Left sidebar with controls
        self.sidebar = widgets.VBox([
            dog_controls,
            stone_controls,
            comp_controls,
            visual_controls,
            action_controls
        ], layout=widgets.Layout(width='350px', overflow='auto'))
        
        # Main content area
        self.main_content = widgets.VBox([
            widgets.HTML('<h2>📊 Stone Analysis Results</h2>'),
            self.output_main,
            widgets.HTML('<h3>📈 Line Scan Analysis</h3>'),
            widgets.HTML('<p><i>Click two points on the composition map to create a line scan</i></p>'),
            self.output_line_scan,
            widgets.HTML('<h3>📋 Statistics</h3>'),
            self.output_stats
        ], layout=widgets.Layout(flex='1', padding='10px'))
        
        # Main layout
        self.main_layout = widgets.HBox([
            self.sidebar,
            self.main_content
        ], layout=widgets.Layout(height='800px'))
    
    def display(self):
        """Display the interface"""
        display(self.main_layout)
    
    def get_current_parameters(self):
        """Get current parameter values from widgets"""
        return {
            'dog_sigma1': self.sigma1_slider.value,
            'dog_sigma2': self.sigma2_slider.value,
            'stone_threshold': self.threshold_slider.value,
            'min_stone_size': self.min_size_input.value,
            'hole_fill_size': self.hole_fill_input.value,
            'bacteria_threshold': self.bacteria_slider.value,
            'bacteria_rich_threshold': self.bacteria_rich_slider.value,
            'intergrowth_threshold': self.intergrowth_slider.value,
            'whewellite_rich_threshold': self.whewellite_rich_slider.value,
            'overlay_alpha': self.alpha_slider.value,
            'anti_aliasing': self.aa_checkbox.value,
            'current_colormap': self.colormap_dropdown.value
        }
    
    def apply_dog_filter(self, params):
        """Apply DoG filter"""
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        gaussian1 = filters.gaussian(ct_norm, sigma=params['dog_sigma1'])
        gaussian2 = filters.gaussian(ct_norm, sigma=params['dog_sigma2'])
        dog_result = gaussian1 - gaussian2
        return (dog_result - dog_result.min()) / (dog_result.max() - dog_result.min())
    
    def create_stone_mask(self, dog_enhanced, params):
        """Create stone mask"""
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        inverted_dog = 1.0 - dog_enhanced
        stone_score = 0.7 * ct_norm + 0.3 * inverted_dog
        
        stone_candidates = stone_score > params['stone_threshold']
        stone_candidates = morphology.remove_small_objects(
            stone_candidates, min_size=params['min_stone_size']
        )
        
        if np.sum(stone_candidates) > 0:
            labeled_candidates = measure.label(stone_candidates)
            regions = measure.regionprops(labeled_candidates)
            largest_region = max(regions, key=lambda r: r.area)
            stone_mask = labeled_candidates == largest_region.label
        else:
            stone_mask = np.zeros_like(self.ct_image, dtype=bool)
        
        # Morphological cleanup
        kernel = morphology.disk(3)
        stone_mask = morphology.binary_opening(stone_mask, kernel)
        stone_mask = morphology.binary_closing(stone_mask, kernel)
        
        # Fill holes
        if params['hole_fill_size'] > 0:
            filled_mask = ndimage.binary_fill_holes(stone_mask)
            holes = filled_mask & ~stone_mask
            
            if np.sum(holes) > 0:
                labeled_holes = measure.label(holes)
                hole_regions = measure.regionprops(labeled_holes)
                
                for region in hole_regions:
                    if region.area <= params['hole_fill_size']:
                        for coord in region.coords:
                            stone_mask[coord[0], coord[1]] = True
        
        return stone_mask
    
    def create_composition_map(self, stone_mask, params):
        """Create composition map"""
        if np.sum(stone_mask) == 0:
            return np.zeros_like(self.ct_image, dtype=np.uint8)
        
        stone_intensities = self.ct_image[stone_mask]
        min_intensity = np.min(stone_intensities)
        max_intensity = np.max(stone_intensities)
        
        composition_zones = np.zeros_like(self.ct_image, dtype=np.uint8)
        
        # Calculate threshold values
        range_span = max_intensity - min_intensity
        bacteria_thresh = min_intensity + (params['bacteria_threshold'] / 100.0) * range_span
        bacteria_rich_thresh = min_intensity + (params['bacteria_rich_threshold'] / 100.0) * range_span
        intergrowth_thresh = min_intensity + (params['intergrowth_threshold'] / 100.0) * range_span
        whewellite_rich_thresh = min_intensity + (params['whewellite_rich_threshold'] / 100.0) * range_span
        
        # Assign zones
        composition_zones[stone_mask] = 3
        
        pure_bacteria_mask = stone_mask & (self.ct_image <= bacteria_thresh)
        bacteria_rich_mask = stone_mask & (self.ct_image > bacteria_thresh) & (self.ct_image <= bacteria_rich_thresh)
        intergrowth_mask = stone_mask & (self.ct_image > bacteria_rich_thresh) & (self.ct_image <= intergrowth_thresh)
        whewellite_rich_mask = stone_mask & (self.ct_image > intergrowth_thresh) & (self.ct_image <= whewellite_rich_thresh)
        pure_whewellite_mask = stone_mask & (self.ct_image > whewellite_rich_thresh)
        
        composition_zones[pure_bacteria_mask] = 1
        composition_zones[bacteria_rich_mask] = 2
        composition_zones[intergrowth_mask] = 3
        composition_zones[whewellite_rich_mask] = 4
        composition_zones[pure_whewellite_mask] = 5
        
        return composition_zones
    
    def apply_anti_aliasing(self, image, enable_aa, factor=2):
        """Apply anti-aliasing"""
        if not enable_aa:
            return image
        
        h, w = image.shape
        upscaled = ndimage.zoom(image, factor, order=0)
        smoothed = ndimage.gaussian_filter(upscaled.astype(float), sigma=0.5)
        downscaled = ndimage.zoom(smoothed, 1/factor, order=1)
        return downscaled
    
    def recalculate_analysis(self):
        """Recalculate analysis with current parameters"""
        params = self.get_current_parameters()
        
        # Apply algorithms
        self.dog_enhanced = self.apply_dog_filter(params)
        self.stone_mask = self.create_stone_mask(self.dog_enhanced, params)
        self.composition_map = self.create_composition_map(self.stone_mask, params)
        
        # Update displays
        self.update_main_display()
        self.update_statistics()
    
    def update_main_display(self):
        """Update main analysis display"""
        with self.output_main:
            clear_output(wait=True)
            
            params = self.get_current_parameters()
            colors = self.colormaps[params['current_colormap']]
            cmap_zones = ListedColormap(colors)
            
            # Create CLAHE enhanced CT
            ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
            ct_uint8 = (ct_norm * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            ct_clahe = clahe.apply(ct_uint8)
            
            # Apply anti-aliasing to composition map
            display_comp = self.apply_anti_aliasing(
                self.composition_map.astype(float), params['anti_aliasing']
            )
            
            # Create figure with fixed size
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('CT-Raman Stone Analysis Results', fontsize=16, fontweight='bold')
            
            # Original CT
            axes[0, 0].imshow(self.ct_image, cmap='gray')
            axes[0, 0].set_title('Original CT')
            axes[0, 0].axis('off')
            
            # DoG Enhanced
            axes[0, 1].imshow(self.dog_enhanced, cmap='gray')
            axes[0, 1].set_title('DoG Enhanced')
            axes[0, 1].axis('off')
            
            # CLAHE Enhanced
            axes[0, 2].imshow(ct_clahe, cmap='gray')
            axes[0, 2].set_title('CLAHE Enhanced')
            axes[0, 2].axis('off')
            
            # Stone Mask
            axes[1, 0].imshow(self.stone_mask, cmap='gray')
            stone_coverage = np.sum(self.stone_mask) / self.stone_mask.size * 100
            axes[1, 0].set_title(f'Stone Mask ({stone_coverage:.1f}%)')
            axes[1, 0].axis('off')
            
            # Composition Zones (clickable for line scan)
            self.comp_ax = axes[1, 1]
            interpolation = 'bilinear' if params['anti_aliasing'] else 'nearest'
            self.comp_ax.imshow(display_comp, cmap=cmap_zones, vmin=0, vmax=5, interpolation=interpolation)
            self.comp_ax.set_title(f'Composition Zones ({params["current_colormap"]})')
            self.comp_ax.axis('off')
            
            # Add click handler for line scan
            self.comp_ax.figure.canvas.mpl_connect('button_press_event', self.on_composition_click)
            
            # Overlay
            axes[1, 2].imshow(ct_clahe, cmap='gray', alpha=0.7)
            overlay_map = np.where(self.stone_mask, display_comp, 0)
            axes[1, 2].imshow(overlay_map, cmap=cmap_zones, vmin=0, vmax=5, 
                             alpha=params['overlay_alpha'], interpolation=interpolation)
            aa_status = "AA On" if params['anti_aliasing'] else "AA Off"
            axes[1, 2].set_title(f'CLAHE + Composition ({aa_status}, α={params["overlay_alpha"]:.2f})')
            axes[1, 2].axis('off')
            
            # Draw existing line if present
            if self.line_start is not None and self.line_end is not None:
                self.comp_ax.plot([self.line_start[1], self.line_end[1]], 
                                 [self.line_start[0], self.line_end[0]], 
                                 'white', linewidth=3, alpha=0.8)
                self.comp_ax.plot([self.line_start[1], self.line_end[1]], 
                                 [self.line_start[0], self.line_end[0]], 
                                 'black', linewidth=1.5, alpha=0.8)
                
                # Mark points
                self.comp_ax.plot(self.line_start[1], self.line_start[0], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
                self.comp_ax.plot(self.line_end[1], self.line_end[0], 'bo', markersize=8, markeredgecolor='white', markeredgewidth=2)
            
            plt.tight_layout()
            plt.show()
    
    def on_composition_click(self, event):
        """Handle clicks on composition map for line scan"""
        if event.inaxes != self.comp_ax:
            return
        
        if event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            
            if self.click_count == 0:
                # First click - start point
                self.line_start = (y, x)
                self.click_count = 1
                print(f"✅ Line start point set: ({x}, {y})")
                
            elif self.click_count == 1:
                # Second click - end point
                self.line_end = (y, x)
                self.click_count = 0
                print(f"✅ Line end point set: ({x}, {y})")
                
                # Extract and display line scan
                self.extract_line_scan()
                self.update_main_display()  # Redraw with line
    
    def extract_line_scan(self):
        """Extract composition values along the selected line"""
        if self.line_start is None or self.line_end is None:
            return
        
        line_profile = measure.profile_line(
            self.composition_map, self.line_start, self.line_end,
            linewidth=1, mode='constant'
        )
        
        self.line_scan_data = line_profile
        self.update_line_scan_display()
        
        print(f"📊 Line scan extracted: {len(line_profile)} pixels")
    
    def update_line_scan_display(self):
        """Update line scan visualization"""
        if self.line_scan_data is None:
            return
        
        with self.output_line_scan:
            clear_output(wait=True)
            
            params = self.get_current_parameters()
            colors = self.colormaps[params['current_colormap']]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Line scan profile
            x_pixels = np.arange(len(self.line_scan_data))
            ax1.plot(x_pixels, self.line_scan_data, 'o-', linewidth=2, markersize=4, color='blue')
            
            # Color background according to composition
            for i in range(len(self.line_scan_data)):
                comp_val = int(self.line_scan_data[i])
                if comp_val < len(colors):
                    ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color=colors[comp_val])
            
            ax1.set_xlabel('Pixel Position Along Line')
            ax1.set_ylabel('Composition Type')
            ax1.set_ylim(0.5, 5.5)
            ax1.set_yticks([1, 2, 3, 4, 5])
            ax1.set_yticklabels(['Pure\nBacteria', 'Bacteria-\nRich', 'Intergrowth', 'Whewellite-\nRich', 'Pure\nWhewellite'])
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'Line Scan Profile ({len(self.line_scan_data)} pixels)')
            
            # Statistics pie chart
            composition_names = ['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
            unique_values, counts = np.unique(self.line_scan_data, return_counts=True)
            
            labels = []
            sizes = []
            colors_pie = []
            
            for val, count in zip(unique_values, counts):
                if val < len(composition_names) and count > 0:
                    labels.append(f'{composition_names[int(val)]}\n({count/len(self.line_scan_data)*100:.1f}%)')
                    sizes.append(count)
                    colors_pie.append(colors[int(val)])
            
            if sizes:
                ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Line Scan Composition Distribution')
            
            plt.tight_layout()
            plt.show()
    
    def update_statistics(self):
        """Update statistics display"""
        with self.output_stats:
            clear_output(wait=True)
            
            if np.sum(self.stone_mask) > 0:
                zone_counts = [np.sum(self.composition_map == i) for i in range(1, 6)]
                total_stone = sum(zone_counts)
                
                zone_names = ['Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
                zone_percentages = [count/total_stone*100 if total_stone > 0 else 0 for count in zone_counts]
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Zone': zone_names,
                    'Pixels': zone_counts,
                    'Percentage': [f'{p:.2f}%' for p in zone_percentages]
                })
                
                # Display metrics
                stone_coverage = np.sum(self.stone_mask) / self.stone_mask.size * 100
                
                print(f"📊 Analysis Statistics")
                print(f"Stone Coverage: {stone_coverage:.1f}% ({np.sum(self.stone_mask):,} pixels)")
                print(f"Image Size: {self.ct_shape[0]}×{self.ct_shape[1]}")
                print(f"Intensity Range: {self.ct_image.min()}-{self.ct_image.max()}")
                print("\n📋 Composition Distribution:")
                print(df.to_string(index=False))
                
                if self.line_scan_data is not None:
                    print(f"\n📈 Current Line Scan: {len(self.line_scan_data)} pixels")
                    print(f"Line: ({self.line_start[1]}, {self.line_start[0]}) → ({self.line_end[1]}, {self.line_end[0]})")
    
    def on_recalculate(self, button):
        """Handle recalculate button click"""
        print("🔄 Recalculating analysis...")
        self.recalculate_analysis()
        print("✅ Analysis complete!")
    
    def on_save(self, button):
        """Handle save button click"""
        params = self.get_current_parameters()
        
        with open('jupyter_stone_settings.pkl', 'wb') as f:
            pickle.dump(params, f)
        
        print("💾 Settings saved to 'jupyter_stone_settings.pkl'")
        print(f"Stone coverage: {np.sum(self.stone_mask)/self.stone_mask.size*100:.1f}%")
        print(f"Anti-aliasing: {'On' if params['anti_aliasing'] else 'Off'}")
        print(f"Colormap: {params['current_colormap']}")
    
    def on_clear_line(self, button):
        """Handle clear line button click"""
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.click_count = 0
        
        with self.output_line_scan:
            clear_output(wait=True)
            print("Click two points on the composition map to create a line scan")
        
        self.update_main_display()
        self.update_statistics()
        print("🗑️ Line scan cleared")


def main():
    """Main function to run the standalone interface"""
    print("🔬 CT-Raman Stone Analysis Interface")
    print("=====================================")
    
    # Create and display the interface
    try:
        stone_widget = StoneAnalysisWidget()
        stone_widget.display()
        return stone_widget
    except Exception as e:
        print(f"❌ Error creating interface: {e}")
        return None


if __name__ == "__main__":
    # Run the interface
    widget = main() 