import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from PIL import Image
import cv2
from skimage import filters, morphology, measure, exposure
from scipy import ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle
import os

class InteractiveStoneThresholdTuner:
    def __init__(self, ct_image_path):
        """
        Interactive tool for tuning stone boundary and composition thresholds
        
        Args:
            ct_image_path: Path to the CT image
        """
        self.ct_image_path = ct_image_path
        self.ct_image = None
        self.dog_enhanced = None
        self.stone_mask = None
        self.composition_map = None
        
        # Optimized parameters from previous tuning session
        self.dog_sigma1 = 5.2
        self.dog_sigma2 = 3.0
        self.stone_threshold = 0.50
        self.min_stone_size = 50000
        self.hole_fill_size = 5858
        
        # Optimized composition thresholds (as percentiles of stone intensity range)
        self.bacteria_threshold = 24  # 0-24% whewellite
        self.bacteria_rich_threshold = 44  # 24-44% whewellite
        self.intergrowth_threshold = 58  # 44-58% whewellite
        self.whewellite_rich_threshold = 75  # 58-75% whewellite
        # 75-100% pure whewellite
        
        # Overlay transparency control
        self.overlay_alpha = 0.5
        
        # Visual enhancement options
        self.anti_aliasing = False
        self.current_colormap = 'original'
        self.available_colormaps = {
            'original': ['black', 'darkred', 'red', 'lime', 'orange', 'gold'],
            'scientific': ['#000080', '#0066CC', '#00AA00', '#FFAA00', '#FF6600', '#CC0000'],
            'viridis': ['#440154', '#31688e', '#35b779', '#fde725', '#ff7f00', '#dc143c'],
            'plasma': ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
            'cool': ['#000080', '#0080ff', '#00ffff', '#80ff80', '#ffff00', '#ff8000']
        }
        
        # Line scan variables
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.is_selecting_line = False
        
        self.load_ct_image()
        self.setup_interactive_interface()
    
    def load_ct_image(self):
        """Load the CT image"""
        if self.ct_image_path.endswith('.tif'):
            self.ct_image = np.array(Image.open(self.ct_image_path))
        else:
            self.ct_image = cv2.imread(self.ct_image_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"CT Image loaded: Shape {self.ct_image.shape}")
        print(f"Intensity range: {self.ct_image.min()} - {self.ct_image.max()}")
    
    def apply_dog_filter(self):
        """Apply DoG filter with current parameters"""
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        gaussian1 = filters.gaussian(ct_norm, sigma=self.dog_sigma1)
        gaussian2 = filters.gaussian(ct_norm, sigma=self.dog_sigma2)
        dog_result = gaussian1 - gaussian2
        self.dog_enhanced = (dog_result - dog_result.min()) / (dog_result.max() - dog_result.min())
    
    def create_stone_mask(self):
        """Create stone mask with current parameters"""
        if self.dog_enhanced is None:
            self.apply_dog_filter()
        
        # Use combined approach like in the main script
        inverted_dog = 1.0 - self.dog_enhanced
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Combined score
        stone_score = 0.7 * ct_norm + 0.3 * inverted_dog
        
        # Apply threshold
        stone_candidates = stone_score > self.stone_threshold
        stone_candidates = morphology.remove_small_objects(stone_candidates, min_size=self.min_stone_size)
        
        if np.sum(stone_candidates) > 0:
            labeled_candidates = measure.label(stone_candidates)
            regions = measure.regionprops(labeled_candidates)
            largest_region = max(regions, key=lambda r: r.area)
            self.stone_mask = labeled_candidates == largest_region.label
        else:
            self.stone_mask = np.zeros_like(self.ct_image, dtype=bool)
        
        # Clean up
        kernel = morphology.disk(3)
        self.stone_mask = morphology.binary_opening(self.stone_mask, kernel)
        self.stone_mask = morphology.binary_closing(self.stone_mask, kernel)
        
        # Fill holes to eliminate black areas within stone
        if self.hole_fill_size > 0:
            # Fill holes smaller than hole_fill_size
            filled_mask = ndimage.binary_fill_holes(self.stone_mask)
            
            # Only keep filled areas that are smaller than hole_fill_size
            holes = filled_mask & ~self.stone_mask
            if np.sum(holes) > 0:
                labeled_holes = measure.label(holes)
                hole_regions = measure.regionprops(labeled_holes)
                
                for region in hole_regions:
                    if region.area <= self.hole_fill_size:
                        hole_coords = region.coords
                        for coord in hole_coords:
                            self.stone_mask[coord[0], coord[1]] = True
        
        print(f"Stone mask: {np.sum(self.stone_mask):,} pixels ({np.sum(self.stone_mask)/self.stone_mask.size*100:.1f}%)")
    
    def create_composition_map(self):
        """Create composition map with current thresholds"""
        if self.stone_mask is None:
            self.create_stone_mask()
        
        # Get intensity range within stone
        if np.sum(self.stone_mask) == 0:
            self.composition_map = np.zeros_like(self.ct_image, dtype=np.uint8)
            return
        
        stone_intensities = self.ct_image[self.stone_mask]
        min_intensity = np.min(stone_intensities)
        max_intensity = np.max(stone_intensities)
        
        # Create composition zones
        composition_zones = np.zeros_like(self.ct_image, dtype=np.uint8)
        
        # Calculate threshold values
        range_span = max_intensity - min_intensity
        bacteria_thresh = min_intensity + (self.bacteria_threshold / 100.0) * range_span
        bacteria_rich_thresh = min_intensity + (self.bacteria_rich_threshold / 100.0) * range_span
        intergrowth_thresh = min_intensity + (self.intergrowth_threshold / 100.0) * range_span
        whewellite_rich_thresh = min_intensity + (self.whewellite_rich_threshold / 100.0) * range_span
        
        # Assign all stone pixels to intergrowth first
        composition_zones[self.stone_mask] = 3
        
        # Override with specific zones
        pure_bacteria_mask = self.stone_mask & (self.ct_image <= bacteria_thresh)
        bacteria_rich_mask = self.stone_mask & (self.ct_image > bacteria_thresh) & (self.ct_image <= bacteria_rich_thresh)
        intergrowth_mask = self.stone_mask & (self.ct_image > bacteria_rich_thresh) & (self.ct_image <= intergrowth_thresh)
        whewellite_rich_mask = self.stone_mask & (self.ct_image > intergrowth_thresh) & (self.ct_image <= whewellite_rich_thresh)
        pure_whewellite_mask = self.stone_mask & (self.ct_image > whewellite_rich_thresh)
        
        composition_zones[pure_bacteria_mask] = 1
        composition_zones[bacteria_rich_mask] = 2
        composition_zones[intergrowth_mask] = 3
        composition_zones[whewellite_rich_mask] = 4
        composition_zones[pure_whewellite_mask] = 5
        
        self.composition_map = composition_zones
    
    def update_display(self):
        """Update all displays with current parameters"""
        # Recalculate everything
        self.apply_dog_filter()
        self.create_stone_mask()
        self.create_composition_map()
        
        # Update DoG enhanced display
        self.ax_dog.clear()
        self.ax_dog.imshow(self.dog_enhanced, cmap='gray')
        self.ax_dog.set_title('DoG Enhanced')
        self.ax_dog.axis('off')
        
        # Update stone mask display
        self.ax_mask.clear()
        self.ax_mask.imshow(self.stone_mask, cmap='gray')
        stone_percentage = np.sum(self.stone_mask) / self.stone_mask.size * 100
        self.ax_mask.set_title(f'Stone Mask ({stone_percentage:.1f}%)')
        self.ax_mask.axis('off')
        
        # Update composition display with anti-aliasing and custom colormap
        self.ax_comp.clear()
        if self.composition_map is not None:
            # Apply anti-aliasing if enabled
            display_map = self.apply_anti_aliasing(self.composition_map.astype(float))
            
            # Get current colormap
            cmap_zones = self.get_current_colormap()
            
            # Display with interpolation for smoother look
            interpolation = 'bilinear' if self.anti_aliasing else 'nearest'
            self.ax_comp.imshow(display_map, cmap=cmap_zones, vmin=0, vmax=5, interpolation=interpolation)
            
            # Calculate statistics
            if np.sum(self.stone_mask) > 0:
                zone_counts = [np.sum(self.composition_map == i) for i in range(1, 6)]
                total_stone = sum(zone_counts)
                zone_percentages = [count/total_stone*100 if total_stone > 0 else 0 for count in zone_counts]
                
                title = f'Composition Zones ({self.current_colormap})\n'
                title += f'B:{zone_percentages[0]:.1f}% BR:{zone_percentages[1]:.1f}% I:{zone_percentages[2]:.1f}% WR:{zone_percentages[3]:.1f}% W:{zone_percentages[4]:.1f}%'
                self.ax_comp.set_title(title, fontsize=9)
        
        self.ax_comp.axis('off')
        
        # Update overlay display with interactive transparency and CLAHE enhancement
        self.ax_overlay.clear()
        if self.composition_map is not None:
            # Apply CLAHE to enhance contrast in the CT image
            ct_uint8 = ((self.ct_image - self.ct_image.min()) / 
                       (self.ct_image.max() - self.ct_image.min()) * 255).astype(np.uint8)
            
            # Create CLAHE object (clip limit=3.0, tile grid size=8x8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            ct_clahe = clahe.apply(ct_uint8)
            
            # Show CLAHE-enhanced CT image as background
            self.ax_overlay.imshow(ct_clahe, cmap='gray', alpha=0.7)
            
            # Apply anti-aliasing to overlay composition map
            overlay_map = np.where(self.stone_mask, self.composition_map, 0)
            display_overlay = self.apply_anti_aliasing(overlay_map.astype(float))
            
            # Overlay composition map with adjustable transparency
            cmap_zones = self.get_current_colormap()
            interpolation = 'bilinear' if self.anti_aliasing else 'nearest'
            
            self.ax_overlay.imshow(display_overlay, cmap=cmap_zones, vmin=0, vmax=5, 
                                 alpha=self.overlay_alpha, interpolation=interpolation)
            
            aa_status = "AA On" if self.anti_aliasing else "AA Off"
            self.ax_overlay.set_title(f'CLAHE + Composition ({aa_status}, α={self.overlay_alpha:.2f})')
        
        self.ax_overlay.axis('off')
        
        # Update parameter text
        param_text = f"""Optimized Parameters:
σ1: {self.dog_sigma1:.1f} (↑ reduces holes)
σ2: {self.dog_sigma2:.1f} (↓ reduces holes)
Stone Threshold: {self.stone_threshold:.2f}
Hole Fill Size: {self.hole_fill_size:,}
Min Size: {self.min_stone_size:,}
Overlay Alpha: {self.overlay_alpha:.2f}
Anti-Aliasing: {'On' if self.anti_aliasing else 'Off'}
Colormap: {self.current_colormap.title()}

Composition Thresholds (%):
Pure Bacteria: 0-{self.bacteria_threshold}%
Bacteria-Rich: {self.bacteria_threshold}-{self.bacteria_rich_threshold}%
Intergrowth: {self.bacteria_rich_threshold}-{self.intergrowth_threshold}%
Whewellite-Rich: {self.intergrowth_threshold}-{self.whewellite_rich_threshold}%
Pure Whewellite: {self.whewellite_rich_threshold}-100%

Tips:
• Increase σ1 to fill holes
• Decrease σ2 to reduce holes  
• Hole Fill: max hole size to fill
• Alpha: overlay transparency
• Anti-aliasing: smooth edges
• Colormap: visual appearance"""
        
        self.ax_params.clear()
        self.ax_params.text(0.05, 0.95, param_text, transform=self.ax_params.transAxes, 
                           fontsize=7, verticalalignment='top', fontfamily='monospace')
        self.ax_params.axis('off')
        
        self.fig.canvas.draw()
    
    def setup_interactive_interface(self):
        """Setup the interactive matplotlib interface"""
        self.fig = plt.figure(figsize=(24, 15))
        
        # Create subplots - now including line scan display
        self.ax_orig = plt.subplot(3, 5, 1)
        self.ax_dog = plt.subplot(3, 5, 2)
        self.ax_mask = plt.subplot(3, 5, 3)
        self.ax_comp = plt.subplot(3, 5, 4)
        self.ax_overlay = plt.subplot(3, 5, 5)
        self.ax_params = plt.subplot(3, 5, 6)
        self.ax_line_scan = plt.subplot(3, 5, (12, 15))  # Span bottom row
        
        # Original image
        self.ax_orig.imshow(self.ct_image, cmap='gray')
        self.ax_orig.set_title('Original CT Image')
        self.ax_orig.axis('off')
        
        # Set up line scan plot
        self.ax_line_scan.set_title('Line Scan Profile\n(Click two points on Composition Zones to create line)')
        self.ax_line_scan.set_xlabel('Pixel Position Along Line')
        self.ax_line_scan.set_ylabel('Composition Type')
        self.ax_line_scan.grid(True, alpha=0.3)
        
        # Connect mouse click events
        self.fig.canvas.mpl_connect('button_press_event', self.on_composition_click)
        
        # Create sliders
        slider_height = 0.02
        slider_spacing = 0.03
        slider_left = 0.15
        slider_width = 0.25
        
        # DoG parameters
        ax_sigma1 = plt.axes([slider_left, 0.35, slider_width, slider_height])
        self.slider_sigma1 = Slider(ax_sigma1, 'DoG σ1', 0.1, 20.0, valinit=self.dog_sigma1, valfmt='%.1f')
        
        ax_sigma2 = plt.axes([slider_left, 0.32, slider_width, slider_height])
        self.slider_sigma2 = Slider(ax_sigma2, 'DoG σ2', 0.1, 10.0, valinit=self.dog_sigma2, valfmt='%.1f')
        
        ax_stone_thresh = plt.axes([slider_left, 0.29, slider_width, slider_height])
        self.slider_stone_thresh = Slider(ax_stone_thresh, 'Stone Thresh', 0.1, 0.9, valinit=self.stone_threshold, valfmt='%.2f')
        
        ax_hole_fill = plt.axes([slider_left, 0.26, slider_width, slider_height])
        self.slider_hole_fill = Slider(ax_hole_fill, 'Hole Fill Size', 0, 10000, valinit=self.hole_fill_size, valfmt='%d')
        
        ax_min_size = plt.axes([slider_left, 0.23, slider_width, slider_height])
        self.slider_min_size = Slider(ax_min_size, 'Min Size', 10000, 200000, valinit=self.min_stone_size, valfmt='%d')
        
        # Overlay transparency slider
        ax_alpha = plt.axes([slider_left, 0.20, slider_width, slider_height])
        self.slider_alpha = Slider(ax_alpha, 'Overlay Alpha', 0.0, 1.0, valinit=self.overlay_alpha, valfmt='%.2f')
        
        # Composition thresholds - move to right side
        slider_right = 0.55
        
        ax_bacteria = plt.axes([slider_right, 0.35, slider_width, slider_height])
        self.slider_bacteria = Slider(ax_bacteria, 'Bacteria %', 5, 35, valinit=self.bacteria_threshold, valfmt='%d')
        
        ax_bacteria_rich = plt.axes([slider_right, 0.32, slider_width, slider_height])
        self.slider_bacteria_rich = Slider(ax_bacteria_rich, 'Bacteria-Rich %', 25, 55, valinit=self.bacteria_rich_threshold, valfmt='%d')
        
        ax_intergrowth = plt.axes([slider_right, 0.29, slider_width, slider_height])
        self.slider_intergrowth = Slider(ax_intergrowth, 'Intergrowth %', 45, 75, valinit=self.intergrowth_threshold, valfmt='%d')
        
        ax_whewellite_rich = plt.axes([slider_right, 0.26, slider_width, slider_height])
        self.slider_whewellite_rich = Slider(ax_whewellite_rich, 'Whewellite-Rich %', 65, 95, valinit=self.whewellite_rich_threshold, valfmt='%d')
        
        # Visual controls - Anti-aliasing toggle
        ax_aa_toggle = plt.axes([0.85, 0.35, 0.08, 0.08])
        self.radio_aa = RadioButtons(ax_aa_toggle, ['AA Off', 'AA On'], active=0)
        self.radio_aa.on_clicked(self.on_aa_change)
        
        # Colormap selection
        ax_colormap = plt.axes([0.85, 0.20, 0.12, 0.12])
        colormap_labels = list(self.available_colormaps.keys())
        self.radio_colormap = RadioButtons(ax_colormap, colormap_labels, active=0)
        self.radio_colormap.on_clicked(self.on_colormap_change)
        
        # Connect sliders to update function
        self.slider_sigma1.on_changed(self.on_slider_change)
        self.slider_sigma2.on_changed(self.on_slider_change)
        self.slider_stone_thresh.on_changed(self.on_slider_change)
        self.slider_hole_fill.on_changed(self.on_slider_change)
        self.slider_min_size.on_changed(self.on_slider_change)
        self.slider_alpha.on_changed(self.on_slider_change)
        self.slider_bacteria.on_changed(self.on_slider_change)
        self.slider_bacteria_rich.on_changed(self.on_slider_change)
        self.slider_intergrowth.on_changed(self.on_slider_change)
        self.slider_whewellite_rich.on_changed(self.on_slider_change)
        
        # Save button
        ax_save = plt.axes([0.82, 0.05, 0.08, 0.04])
        self.button_save = Button(ax_save, 'Save Settings')
        self.button_save.on_clicked(self.save_settings)
        
        # Reset button
        ax_reset = plt.axes([0.82, 0.01, 0.08, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_settings)
        
        # Clear line button
        ax_clear_line = plt.axes([0.70, 0.05, 0.08, 0.04])
        self.button_clear_line = Button(ax_clear_line, 'Clear Line')
        self.button_clear_line.on_clicked(self.clear_line)
        
        # Initial update
        self.update_display()
        
        plt.tight_layout()
        plt.show()
    
    def on_slider_change(self, val):
        """Handle slider changes"""
        self.dog_sigma1 = self.slider_sigma1.val
        self.dog_sigma2 = self.slider_sigma2.val
        self.stone_threshold = self.slider_stone_thresh.val
        self.hole_fill_size = int(self.slider_hole_fill.val)
        self.min_stone_size = int(self.slider_min_size.val)
        self.overlay_alpha = self.slider_alpha.val
        self.bacteria_threshold = int(self.slider_bacteria.val)
        self.bacteria_rich_threshold = int(self.slider_bacteria_rich.val)
        self.intergrowth_threshold = int(self.slider_intergrowth.val)
        self.whewellite_rich_threshold = int(self.slider_whewellite_rich.val)
        
        self.update_display()
    
    def save_settings(self, event):
        """Save current settings to file"""
        settings = {
            'dog_sigma1': self.dog_sigma1,
            'dog_sigma2': self.dog_sigma2,
            'stone_threshold': self.stone_threshold,
            'hole_fill_size': self.hole_fill_size,
            'min_stone_size': self.min_stone_size,
            'overlay_alpha': self.overlay_alpha,
            'anti_aliasing': self.anti_aliasing,
            'current_colormap': self.current_colormap,
            'bacteria_threshold': self.bacteria_threshold,
            'bacteria_rich_threshold': self.bacteria_rich_threshold,
            'intergrowth_threshold': self.intergrowth_threshold,
            'whewellite_rich_threshold': self.whewellite_rich_threshold
        }
        
        with open('optimized_stone_settings_v2.pkl', 'wb') as f:
            pickle.dump(settings, f)
        
        print("Settings saved to 'optimized_stone_settings_v2.pkl'")
        print(f"Stone coverage: {np.sum(self.stone_mask)/self.stone_mask.size*100:.1f}%")
        print(f"Anti-aliasing: {'On' if self.anti_aliasing else 'Off'}")
        print(f"Colormap: {self.current_colormap}")
        
        # Prepare various enhancements for comparison figure
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # 1. Original CT (normalized for display)
        original_ct = ct_norm
        
        # 2. CLAHE-enhanced
        ct_uint8 = (ct_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        ct_clahe = clahe.apply(ct_uint8) / 255.0
        
        # 3. Morphological Enhanced (using original technique: white_tophat + black_tophat)
        kernel = morphology.disk(2)
        tophat = morphology.white_tophat(ct_norm, kernel)
        blackhat = morphology.black_tophat(ct_norm, kernel)
        morpho_enhanced = ct_norm + tophat - blackhat
        # Apply intensity rescaling and convert to uint8 (like original ct_enhancement.py)
        morphological_enhanced = (exposure.rescale_intensity(morpho_enhanced) * 255).astype(np.uint8)
        # Convert back to float for display consistency
        morphological_enhanced = morphological_enhanced.astype(np.float64) / 255.0
        
        # 4. Density map (composition overlay on CLAHE background)
        cmap_zones = self.get_current_colormap()
        overlay_map = np.where(self.stone_mask, self.composition_map, 0)
        display_overlay = self.apply_anti_aliasing(overlay_map.astype(float))
        
        # Create comprehensive enhancement comparison figure (4 panels)
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(original_ct, cmap='gray')
        plt.title('Original CT')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(ct_clahe, cmap='gray')
        plt.title('CLAHE Enhanced')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(morphological_enhanced, cmap='gray')
        plt.title('Morphological Enhanced')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        # Density map with CLAHE background and composition overlay
        plt.imshow(ct_clahe, cmap='gray', alpha=0.7)
        interpolation = 'bilinear' if self.anti_aliasing else 'nearest'
        plt.imshow(display_overlay, cmap=cmap_zones, vmin=0, vmax=5, alpha=self.overlay_alpha, interpolation=interpolation)
        aa_status = "AA On" if self.anti_aliasing else "AA Off"
        plt.title(f'Density Map ({self.current_colormap}, {aa_status}, α={self.overlay_alpha:.2f})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_stone_enhancement_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Also save individual composition overlay for reference
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        display_comp = self.apply_anti_aliasing(self.composition_map.astype(float))
        plt.imshow(display_comp, cmap=cmap_zones, vmin=0, vmax=5, interpolation=interpolation)
        plt.title(f'Pure Composition Zones\n({self.current_colormap}, {aa_status})')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(ct_clahe, cmap='gray', alpha=0.7)
        plt.imshow(display_overlay, cmap=cmap_zones, vmin=0, vmax=5, alpha=self.overlay_alpha, interpolation=interpolation)
        plt.title(f'CLAHE + Composition Overlay\n({aa_status}, α={self.overlay_alpha:.2f})')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Show colormap legend
        legend_data = np.arange(6).reshape(6, 1)
        plt.imshow(legend_data, cmap=cmap_zones, vmin=0, vmax=5, aspect='auto')
        plt.yticks([0, 1, 2, 3, 4, 5], ['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite'])
        plt.xticks([])
        plt.title(f'{self.current_colormap.title()} Colormap')
        
        plt.tight_layout()
        plt.savefig('final_composition_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def reset_settings(self, event):
        """Reset to optimized default settings"""
        # Reset to the optimized values, not original defaults
        self.slider_sigma1.reset()
        self.slider_sigma2.reset()
        self.slider_stone_thresh.reset()
        self.slider_hole_fill.reset()
        self.slider_min_size.reset()
        self.slider_alpha.reset()
        self.slider_bacteria.reset()
        self.slider_bacteria_rich.reset()
        self.slider_intergrowth.reset()
        self.slider_whewellite_rich.reset()
    
    def on_composition_click(self, event):
        """Handle mouse clicks on composition map for line selection"""
        if event.inaxes != self.ax_comp:
            return
        
        if event.button == 1:  # Left click
            if not self.is_selecting_line:
                # Start line selection
                self.line_start = (int(event.ydata), int(event.xdata))  # (row, col)
                self.is_selecting_line = True
                print(f"Line start: {self.line_start}")
            else:
                # End line selection
                self.line_end = (int(event.ydata), int(event.xdata))  # (row, col)
                self.is_selecting_line = False
                print(f"Line end: {self.line_end}")
                self.extract_line_scan()
                self.update_line_scan_display()
    
    def extract_line_scan(self):
        """Extract composition values along the selected line"""
        if self.line_start is None or self.line_end is None or self.composition_map is None:
            return
        
        # Use measure.profile_line to extract values along the line
        line_profile = measure.profile_line(
            self.composition_map, 
            self.line_start, 
            self.line_end,
            linewidth=1,
            mode='constant'
        )
        
        self.line_scan_data = line_profile
        
        # Print some statistics
        composition_names = ['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite']
        unique_values, counts = np.unique(line_profile, return_counts=True)
        
        print(f"\nLine scan results ({len(line_profile)} pixels):")
        for val, count in zip(unique_values, counts):
            if val < len(composition_names):
                print(f"  {composition_names[int(val)]}: {count} pixels ({count/len(line_profile)*100:.1f}%)")
    
    def update_line_scan_display(self):
        """Update the line scan plot"""
        if self.line_scan_data is None:
            return
        
        # Clear and update line scan plot
        self.ax_line_scan.clear()
        
        # Create x-axis (pixel positions)
        x_pixels = np.arange(len(self.line_scan_data))
        
        # Plot the composition line scan
        colors = self.available_colormaps[self.current_colormap]
        y_values = self.line_scan_data
        
        # Create a step plot to show discrete composition zones
        self.ax_line_scan.plot(x_pixels, y_values, 'o-', linewidth=2, markersize=3, color='blue')
        
        # Color the background according to composition
        for i in range(len(y_values)):
            comp_val = int(y_values[i])
            if comp_val < len(colors):
                self.ax_line_scan.axvspan(i-0.5, i+0.5, alpha=0.3, color=colors[comp_val])
        
        self.ax_line_scan.set_xlabel('Pixel Position Along Line')
        self.ax_line_scan.set_ylabel('Composition Type')
        self.ax_line_scan.set_ylim(0.5, 5.5)
        self.ax_line_scan.set_yticks([1, 2, 3, 4, 5])
        self.ax_line_scan.set_yticklabels(['Pure\nBacteria', 'Bacteria-\nRich', 'Intergrowth', 'Whewellite-\nRich', 'Pure\nWhewellite'])
        self.ax_line_scan.grid(True, alpha=0.3)
        self.ax_line_scan.set_title(f'Line Scan Profile ({len(self.line_scan_data)} pixels, {self.current_colormap} colormap)')
        
        # Draw the line on the composition map
        if self.line_start is not None and self.line_end is not None:
            self.ax_comp.plot([self.line_start[1], self.line_end[1]], 
                             [self.line_start[0], self.line_end[0]], 
                             'white', linewidth=2, alpha=0.8)
            self.ax_comp.plot([self.line_start[1], self.line_end[1]], 
                             [self.line_start[0], self.line_end[0]], 
                             'black', linewidth=1, alpha=0.8)
        
        self.fig.canvas.draw()
    
    def clear_line(self, event):
        """Clear the current line selection"""
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.is_selecting_line = False
        
        # Clear line scan plot
        self.ax_line_scan.clear()
        self.ax_line_scan.set_title('Line Scan Profile\n(Click two points on Composition Zones to create line)')
        self.ax_line_scan.set_xlabel('Pixel Position Along Line')
        self.ax_line_scan.set_ylabel('Composition Type')
        self.ax_line_scan.grid(True, alpha=0.3)
        
        # Redraw composition map without line
        self.update_display()
        
        print("Line selection cleared.")

    def apply_anti_aliasing(self, image, factor=2):
        """Apply anti-aliasing to composition map"""
        if not self.anti_aliasing:
            return image
        
        # Upscale using nearest neighbor to preserve discrete values
        h, w = image.shape
        upscaled = ndimage.zoom(image, factor, order=0)
        
        # Apply slight gaussian smoothing to edges only
        smoothed = ndimage.gaussian_filter(upscaled.astype(float), sigma=0.5)
        
        # Downscale back to original size
        downscaled = ndimage.zoom(smoothed, 1/factor, order=1)
        
        # Round back to discrete values while preserving smoothing
        return downscaled
    
    def get_current_colormap(self):
        """Get the current colormap as ListedColormap"""
        colors = self.available_colormaps[self.current_colormap]
        return ListedColormap(colors)

    def on_aa_change(self, label):
        """Handle anti-aliasing toggle"""
        self.anti_aliasing = label == 'AA On'
        print(f"Anti-aliasing: {'On' if self.anti_aliasing else 'Off'}")
        self.update_display()
    
    def on_colormap_change(self, label):
        """Handle colormap selection"""
        self.current_colormap = label
        print(f"Colormap changed to: {label}")
        # Update line scan display if it exists
        if self.line_scan_data is not None:
            self.update_line_scan_display()
        self.update_display()

def main():
    """Run the interactive threshold tuner with optimized starting values"""
    tuner = InteractiveStoneThresholdTuner('slice_1092.tif')
    return tuner

if __name__ == "__main__":
    tuner = main() 