#!/usr/bin/env python3
"""
Standalone CT-Raman Kidney Stone Analysis Application
Professional desktop application with separate windows and density calibration
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for separate windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
import cv2
from PIL import Image
from skimage import filters, morphology, measure, exposure
from scipy import ndimage, interpolate
import pickle
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from matplotlib.patches import Rectangle

class StoneAnalysisApp:
    def __init__(self):
        """Initialize the Stone Analysis Application"""
        # Load CT image
        try:
            self.ct_image = np.array(Image.open('slice_1092.tif'))
            self.ct_shape = self.ct_image.shape
            print(f"✅ CT image loaded: {self.ct_shape}")
        except FileNotFoundError:
            print("❌ Error: Could not find 'slice_1092.tif'")
            print("Please ensure the CT image file is in the current directory.")
            return
        
        # Raman density calibration (g/cm³)
        self.setup_density_calibration()
        
        # Optimized default parameters - more restrictive for bacteria, more generous for whewellite
        self.dog_sigma1 = 5.2
        self.dog_sigma2 = 3.0
        self.stone_threshold = 0.50
        self.min_stone_size = 50000
        self.hole_fill_size = 5858
        # Using 10% binned boundaries (0-10%, 10-20%, ..., 90-100%)
        self.bacteria_threshold = 5           # Even more restrictive
        self.bacteria_rich_threshold = 15     # Reduced further
        self.intergrowth_threshold = 35       # Reduced significantly  
        self.whewellite_rich_threshold = 60   # More whewellite-rich regions
        self.current_colormap = 'morphological'  # Default to high contrast
        
        # Available colormaps (restored to 10 colors for fine visual detail)
        self.colormaps = {
            'viridis': ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21908c', '#22a884', '#44bf70', '#7ad151', '#bddf26', '#f0f921', '#000000'],
            'original': ['#000000', '#8B0000', '#B22222', '#DC143C', '#FF4500', '#FF6347', '#FFA500', '#FFD700', '#ADFF2F', '#7FFF00', '#00FF00', '#FFFFFF'],
            'scientific': ['#000080', '#1a0099', '#3300bb', '#4d00dd', '#6600ff', '#8000ff', '#9900ff', '#b300ff', '#cc00ff', '#e600ff', '#ff00ff', '#FFFFFF'],
            'plasma': ['#0d0887', '#4b0c6b', '#6a00a8', '#8b0aa6', '#a53582', '#bc5090', '#d1719b', '#e594a7', '#f2b9b2', '#fcddbf', '#f0f921', '#000000'],
            'cool': ['#000080', '#1a1a9a', '#3333b4', '#4d4dce', '#6666e8', '#8080ff', '#9999ff', '#b3b3ff', '#ccccff', '#e6e6ff', '#ffffff', '#000000'],
            'morphological': self.create_high_contrast_colormap()
        }
        
        # Line scan data
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.line_scan_densities = None
        self.click_count = 0
        
        # Analysis results
        self.dog_enhanced = None
        self.stone_mask = None
        self.composition_map = None
        
        # Component densities for 10-zone visual system (fine detail)
        self.component_densities = {
            '100% Bacteria': 0.95,      # Zone 1: Pure bacteria
            '90% Bacteria': 1.05,       # Zone 2: 90% bacteria, 10% whewellite
            '80% Bacteria': 1.15,       # Zone 3: 80% bacteria, 20% whewellite
            '70% Bacteria': 1.25,       # Zone 4: 70% bacteria, 30% whewellite
            '60% Bacteria': 1.35,       # Zone 5: 60% bacteria, 40% whewellite
            '50/50 Mix': 1.45,          # Zone 6: 50% bacteria, 50% whewellite
            '40% Bacteria': 1.55,       # Zone 7: 40% bacteria, 60% whewellite
            '30% Bacteria': 1.65,       # Zone 8: 30% bacteria, 70% whewellite
            '20% Bacteria': 1.75,       # Zone 9: 20% bacteria, 80% whewellite
            '10% Bacteria': 1.85,       # Zone 10: 10% bacteria, 90% whewellite
            'Pure Whewellite': 2.23,    # Zone 11: 0% bacteria, 100% whewellite
            'Holes/Voids': 0.001       # Zone 12: Air density
        }
        
        # Simplified categories for line profile analysis (20% intervals)
        self.line_profile_categories = {
            'Pure Bacteria': {'zones': [1, 2], 'density': 0.95, 'description': '100-80% Bacteria'},
            'Bacteria-Rich': {'zones': [3, 4], 'density': 1.15, 'description': '80-60% Bacteria'}, 
            'Intergrowth': {'zones': [5, 6], 'density': 1.45, 'description': '60-40% Bacteria'},
            'Whewellite-Rich': {'zones': [7, 8], 'density': 1.75, 'description': '40-20% Bacteria'},
            'Pure Whewellite': {'zones': [9, 10, 11], 'density': 2.23, 'description': '20-0% Bacteria'},
            'Holes/Voids': {'zones': [12], 'density': 0.001, 'description': 'Air/Voids'}
        }
        
        # Perform initial analysis
        print("🔄 Running initial analysis...")
        self.recalculate_analysis()
        print("✅ Analysis complete!")
        
        # Create main window
        self.create_main_window()
    
    def setup_density_calibration(self):
        """Setup density calibration from Raman data"""
        # Create intensity to density mapping
        self.setup_intensity_density_mapping()
    
    def setup_intensity_density_mapping(self):
        """Create mapping from CT intensity to density"""
        # First, determine background CT intensity from areas outside the stone
        # Use initial stone mask to identify background areas
        temp_stone_mask = self.create_initial_stone_mask()
        background_pixels = self.ct_image[~temp_stone_mask]
        
        if len(background_pixels) > 0:
            # Use median of background pixels as reference
            self.background_ct_intensity = np.median(background_pixels)
        else:
            # Fallback to image minimum
            self.background_ct_intensity = self.ct_image.min()
        
        # Get CT intensity ranges for stone areas only
        stone_intensities = self.ct_image[temp_stone_mask] if np.sum(temp_stone_mask) > 0 else self.ct_image
        stone_min = np.min(stone_intensities)
        stone_max = np.max(stone_intensities)
        
        # Create calibration curve with very restrictive bacteria zones to target 1.7 g/cm³ mean
        intensity_points = np.array([
            self.background_ct_intensity,  # Background/air
            stone_min,  # Start of stone = 100% bacteria
            stone_min + 0.005*(stone_max-stone_min),  # 0.5% = 100% bacteria threshold (very restrictive)
            stone_min + 0.015*(stone_max-stone_min),  # 1.5% = 90% bacteria (very restrictive)
            stone_min + 0.035*(stone_max-stone_min),  # 3.5% = 80% bacteria (restrictive)
            stone_min + 0.07*(stone_max-stone_min),   # 7% = 70% bacteria
            stone_min + 0.12*(stone_max-stone_min),   # 12% = 60% bacteria
            stone_min + 0.20*(stone_max-stone_min),   # 20% = 50/50 mix
            stone_min + 0.30*(stone_max-stone_min),   # 30% = 40% bacteria
            stone_min + 0.42*(stone_max-stone_min),   # 42% = 30% bacteria
            stone_min + 0.55*(stone_max-stone_min),   # 55% = 20% bacteria
            stone_min + 0.70*(stone_max-stone_min),   # 70% = 10% bacteria
            stone_max  # Pure whewellite (top 30% of intensities for 2.23 g/cm³)
        ])
        
        density_points = np.array([
            0.001,  # Air density
            0.95,   # 100% bacteria
            0.95,   # 100% bacteria (up to 2% threshold)
            1.05,   # 90% bacteria, 10% whewellite
            1.15,   # 80% bacteria, 20% whewellite
            1.25,   # 70% bacteria, 30% whewellite
            1.35,   # 60% bacteria, 40% whewellite
            1.45,   # 50% bacteria, 50% whewellite
            1.55,   # 40% bacteria, 60% whewellite
            1.65,   # 30% bacteria, 70% whewellite
            1.75,   # 20% bacteria, 80% whewellite
            1.85,   # 10% bacteria, 90% whewellite
            2.23    # Pure whewellite (0% bacteria, 100% whewellite)
        ])
        
        # Create interpolation function
        self.intensity_to_density = interpolate.interp1d(
            intensity_points, density_points, 
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        print(f"📊 Background CT intensity: {self.background_ct_intensity:.0f} HU -> {density_points[0]:.3f} g/cm³ (air)")
        print(f"📊 Stone CT range: {stone_min:.0f} - {stone_max:.0f} HU")
        print(f"📊 Stone density range: {density_points[1]:.2f} - {density_points[-1]:.2f} g/cm³")
        
        # Debug: Print the intensity-to-density mapping points
        print(f"\n🔍 Intensity-to-Density Mapping:")
        comp_names = ['Air', 'Stone Min (Bacteria)', '0.5% Bacteria', '1.5% Bacteria', '3.5% Bacteria', '7% Bacteria', '12% Bacteria', '20% Bacteria', '30% Bacteria', '42% Bacteria', '55% Bacteria', '70% Bacteria', 'Stone Max (Whewellite)']
        for i, (intensity, density) in enumerate(zip(intensity_points, density_points)):
            print(f"  {comp_names[i]:20s}: {intensity:7.0f} HU -> {density:.3f} g/cm³")
    
    def create_initial_stone_mask(self):
        """Create initial rough stone mask for background identification"""
        # Simple threshold-based approach for initial background identification
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Use Otsu's method for initial thresholding
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(ct_norm)
        initial_mask = ct_norm > thresh
        
        # Clean up with morphological operations
        kernel = morphology.disk(5)
        initial_mask = morphology.binary_opening(initial_mask, kernel)
        initial_mask = morphology.remove_small_objects(initial_mask, min_size=10000)
        
        return initial_mask
    
    def recalculate_analysis(self):
        """Recalculate analysis with current parameters"""
        # Apply algorithms
        self.dog_enhanced = self.apply_dog_filter()
        self.stone_mask = self.create_stone_mask(self.dog_enhanced)
        self.composition_map = self.create_composition_map(self.stone_mask)
        
        # Print statistics
        self.print_statistics()
    
    def apply_dog_filter(self):
        """Apply DoG filter"""
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        gaussian1 = filters.gaussian(ct_norm, sigma=self.dog_sigma1)
        gaussian2 = filters.gaussian(ct_norm, sigma=self.dog_sigma2)
        dog_result = gaussian1 - gaussian2
        return (dog_result - dog_result.min()) / (dog_result.max() - dog_result.min())
    
    def create_stone_mask(self, dog_enhanced):
        """Create stone mask"""
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        inverted_dog = 1.0 - dog_enhanced
        stone_score = 0.7 * ct_norm + 0.3 * inverted_dog
        
        stone_candidates = stone_score > self.stone_threshold
        stone_candidates = morphology.remove_small_objects(
            stone_candidates, min_size=self.min_stone_size
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
        if self.hole_fill_size > 0:
            filled_mask = ndimage.binary_fill_holes(stone_mask)
            holes = filled_mask & ~stone_mask
            
            if np.sum(holes) > 0:
                labeled_holes = measure.label(holes)
                hole_regions = measure.regionprops(labeled_holes)
                
                for region in hole_regions:
                    if region.area <= self.hole_fill_size:
                        for coord in region.coords:
                            stone_mask[coord[0], coord[1]] = True
        
        return stone_mask
    
    def create_composition_map(self, stone_mask):
        """Create composition map with 10-zone system for fine visual detail"""
        if np.sum(stone_mask) == 0:
            return np.zeros_like(self.ct_image, dtype=np.uint8)
        
        # Identify holes within the stone (CT intensity ≤ background)
        hole_mask = stone_mask & (self.ct_image <= self.background_ct_intensity)
        
        # Get stone intensities excluding holes
        stone_only_mask = stone_mask & ~hole_mask
        
        if np.sum(stone_only_mask) == 0:
            composition_zones = np.zeros_like(self.ct_image, dtype=np.uint8)
            composition_zones[hole_mask] = 12  # Assign holes to zone 12
            return composition_zones
        
        stone_intensities = self.ct_image[stone_only_mask]
        min_intensity = np.min(stone_intensities)
        max_intensity = np.max(stone_intensities)
        
        composition_zones = np.zeros_like(self.ct_image, dtype=np.uint8)
        
        # Calculate threshold values matching the very restrictive intensity-to-density mapping
        range_span = max_intensity - min_intensity
        
        thresholds = [
            min_intensity + 0.005 * range_span,  # 0.5% = 100% bacteria threshold (very restrictive)
            min_intensity + 0.015 * range_span,  # 1.5% = 90% bacteria (very restrictive)
            min_intensity + 0.035 * range_span,  # 3.5% = 80% bacteria (restrictive)
            min_intensity + 0.07 * range_span,   # 7% = 70% bacteria
            min_intensity + 0.12 * range_span,   # 12% = 60% bacteria
            min_intensity + 0.20 * range_span,   # 20% = 50/50 mix
            min_intensity + 0.30 * range_span,   # 30% = 40% bacteria
            min_intensity + 0.42 * range_span,   # 42% = 30% bacteria
            min_intensity + 0.55 * range_span,   # 55% = 20% bacteria
            min_intensity + 0.70 * range_span    # 70% = 10% bacteria
        ]
        
        # Assign zones for stone areas (zones 1-11) - higher intensities = more whewellite
        composition_zones[stone_only_mask] = 6  # Default to 50/50 mix (zone 6)
        
        # Zone assignments based on intensity ranges
        composition_zones[stone_only_mask & (self.ct_image <= thresholds[0])] = 1  # 100% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[0]) & (self.ct_image <= thresholds[1])] = 2  # 90% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[1]) & (self.ct_image <= thresholds[2])] = 3  # 80% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[2]) & (self.ct_image <= thresholds[3])] = 4  # 70% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[3]) & (self.ct_image <= thresholds[4])] = 5  # 60% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[4]) & (self.ct_image <= thresholds[5])] = 6  # 50/50 mix
        composition_zones[stone_only_mask & (self.ct_image > thresholds[5]) & (self.ct_image <= thresholds[6])] = 7  # 40% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[6]) & (self.ct_image <= thresholds[7])] = 8  # 30% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[7]) & (self.ct_image <= thresholds[8])] = 9  # 20% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[8]) & (self.ct_image <= thresholds[9])] = 10 # 10% bacteria
        composition_zones[stone_only_mask & (self.ct_image > thresholds[9])] = 11  # Pure whewellite (top 15%)
        
        # Assign holes to zone 12
        composition_zones[hole_mask] = 12
        
        return composition_zones
    
    def print_statistics(self):
        """Print analysis statistics for 10-zone system"""
        if np.sum(self.stone_mask) > 0:
            zone_counts = [np.sum(self.composition_map == i) for i in range(1, 13)]  # Include zones 1-12
            total_stone = sum(zone_counts)
            
            zone_names = ['100% Bacteria', '90% Bacteria', '80% Bacteria', '70% Bacteria', '60% Bacteria', 
                         '50/50 Mix', '40% Bacteria', '30% Bacteria', '20% Bacteria', '10% Bacteria', 'Pure Whewellite', 'Holes/Voids']
            zone_percentages = [count/total_stone*100 if total_stone > 0 else 0 for count in zone_counts]
            
            stone_coverage = np.sum(self.stone_mask) / self.stone_mask.size * 100
            hole_count = zone_counts[11]  # Holes are in zone 12 (index 11)
            
            print(f"\n📊 Stone Analysis Statistics")
            print(f"Stone Coverage: {stone_coverage:.1f}% ({np.sum(self.stone_mask):,} pixels)")
            print(f"Background CT Intensity: {self.background_ct_intensity:.0f} HU")
            print(f"Image Size: {self.ct_shape[0]}×{self.ct_shape[1]}")
            print(f"Intensity Range: {self.ct_image.min()}-{self.ct_image.max()} HU")
            
            print("\n📋 Composition Distribution:")
            for i, (name, count, pct) in enumerate(zip(zone_names, zone_counts, zone_percentages)):
                if name == 'Holes/Voids':
                    density_text = "~0.001 g/cm³ (air)"
                else:
                    density = self.component_densities[name]
                    density_text = f"{density:.2f} g/cm³"
                print(f"  {name:15s}: {count:7,} pixels ({pct:5.1f}%) - {density_text}")
    
    def create_main_window(self):
        """Create the main analysis window"""
        # Create main figure with more space for sliders
        self.fig = plt.figure(figsize=(20, 14))
        
        # Create grid layout for plots and sliders
        gs = self.fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 0.8])
        
        # Main title
        self.fig.suptitle('CT-Raman Stone Analysis - Click Composition Zones for Line Scan', 
                         fontsize=16, fontweight='bold')
        
        # Create analysis plots
        self.axes = []
        self.axes.append(self.fig.add_subplot(gs[0, 0]))  # Original CT
        self.axes.append(self.fig.add_subplot(gs[0, 1]))  # DoG Enhanced
        self.axes.append(self.fig.add_subplot(gs[0, 2]))  # CLAHE Enhanced
        self.axes.append(self.fig.add_subplot(gs[1, 0]))  # Stone Mask
        self.axes.append(self.fig.add_subplot(gs[1, 1]))  # Composition Zones
        self.axes.append(self.fig.add_subplot(gs[1, 2]))  # Overlay
        
        # Create CLAHE enhanced CT
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        ct_uint8 = (ct_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        ct_clahe = clahe.apply(ct_uint8)
        
        # Plot images with proper interpolation
        self.axes[0].imshow(self.ct_image, cmap='gray', interpolation='nearest')
        self.axes[0].set_title('Original CT')
        self.axes[0].axis('off')
        
        self.axes[1].imshow(self.dog_enhanced, cmap='gray', interpolation='nearest')
        self.axes[1].set_title('DoG Enhanced')
        self.axes[1].axis('off')
        
        self.axes[2].imshow(ct_clahe, cmap='gray', interpolation='nearest')
        self.axes[2].set_title('CLAHE Enhanced')
        self.axes[2].axis('off')
        
        self.axes[3].imshow(self.stone_mask, cmap='gray', interpolation='nearest')
        stone_coverage = np.sum(self.stone_mask) / self.stone_mask.size * 100
        self.axes[3].set_title(f'Stone Mask ({stone_coverage:.1f}%)')
        self.axes[3].axis('off')
        
        # Composition zones (clickable) - ENHANCED CONTRAST with morphological enhancement
        colors = self.colormaps[self.current_colormap]
        cmap_zones = ListedColormap(colors)
        self.comp_ax = self.axes[4]
        
        # Apply morphological enhancement for better contrast
        if self.current_colormap == 'morphological':
            enhanced_comp_map = self.enhance_composition_contrast(self.composition_map)
            self.comp_ax.imshow(enhanced_comp_map, cmap=cmap_zones, vmin=0, vmax=11, interpolation='bilinear')
        else:
            self.comp_ax.imshow(self.composition_map, cmap=cmap_zones, vmin=0, vmax=11, interpolation='nearest')
        
        self.comp_ax.set_title('Composition Zones - Enhanced Contrast (Click for Line Scan)')
        self.comp_ax.axis('off')
        
        # Overlay - Enhanced contrast overlay
        self.axes[5].imshow(ct_clahe, cmap='gray', alpha=0.7, interpolation='nearest')
        overlay_map = np.where(self.stone_mask, self.composition_map, 0)
        
        # Apply enhancement to overlay if using morphological colormap
        if self.current_colormap == 'morphological':
            enhanced_overlay = self.enhance_composition_contrast(overlay_map)
            self.axes[5].imshow(enhanced_overlay, cmap=cmap_zones, vmin=0, vmax=11, alpha=0.8, interpolation='bilinear')
        else:
            self.axes[5].imshow(overlay_map, cmap=cmap_zones, vmin=0, vmax=11, alpha=0.6, interpolation='nearest')
        
        self.axes[5].set_title('CLAHE + Enhanced Composition Overlay')
        self.axes[5].axis('off')
        
        # Connect click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add parameter adjustment sliders
        self.add_parameter_sliders(gs)
        
        # Add control buttons
        self.add_control_buttons()
        
        plt.tight_layout()
        plt.show()
    
    def add_parameter_sliders(self, gs):
        """Add interactive sliders for parameter adjustment"""
        # Create slider panel
        slider_ax = self.fig.add_subplot(gs[:, 3])
        slider_ax.set_title('Parameter Adjustment', fontsize=12, fontweight='bold')
        slider_ax.axis('off')
        
        # Define slider positions (relative to slider_ax)
        slider_height = 0.03
        slider_spacing = 0.08
        slider_left = 0.1
        slider_width = 0.8
        
        # DoG Sigma 1
        ax_sigma1 = plt.axes([0.82, 0.85, 0.15, slider_height])
        self.slider_sigma1 = Slider(ax_sigma1, 'DoG σ1', 1.0, 10.0, valinit=self.dog_sigma1, valstep=0.1)
        self.slider_sigma1.on_changed(self.update_sigma1)
        
        # DoG Sigma 2
        ax_sigma2 = plt.axes([0.82, 0.80, 0.15, slider_height])
        self.slider_sigma2 = Slider(ax_sigma2, 'DoG σ2', 0.5, 8.0, valinit=self.dog_sigma2, valstep=0.1)
        self.slider_sigma2.on_changed(self.update_sigma2)
        
        # Stone Threshold
        ax_stone_thresh = plt.axes([0.82, 0.75, 0.15, slider_height])
        self.slider_stone_thresh = Slider(ax_stone_thresh, 'Stone Thresh', 0.1, 0.9, valinit=self.stone_threshold, valstep=0.01)
        self.slider_stone_thresh.on_changed(self.update_stone_threshold)
        
        # Bacteria Threshold
        ax_bacteria = plt.axes([0.82, 0.70, 0.15, slider_height])
        self.slider_bacteria = Slider(ax_bacteria, 'Bacteria %', 5, 50, valinit=self.bacteria_threshold, valstep=1)
        self.slider_bacteria.on_changed(self.update_bacteria_threshold)
        
        # Bacteria Rich Threshold
        ax_bacteria_rich = plt.axes([0.82, 0.65, 0.15, slider_height])
        self.slider_bacteria_rich = Slider(ax_bacteria_rich, 'Bact-Rich %', 20, 70, valinit=self.bacteria_rich_threshold, valstep=1)
        self.slider_bacteria_rich.on_changed(self.update_bacteria_rich_threshold)
        
        # Intergrowth Threshold
        ax_intergrowth = plt.axes([0.82, 0.60, 0.15, slider_height])
        self.slider_intergrowth = Slider(ax_intergrowth, 'Intergrowth %', 40, 80, valinit=self.intergrowth_threshold, valstep=1)
        self.slider_intergrowth.on_changed(self.update_intergrowth_threshold)
        
        # Whewellite Rich Threshold
        ax_whewellite_rich = plt.axes([0.82, 0.55, 0.15, slider_height])
        self.slider_whewellite_rich = Slider(ax_whewellite_rich, 'Whew-Rich %', 60, 90, valinit=self.whewellite_rich_threshold, valstep=1)
        self.slider_whewellite_rich.on_changed(self.update_whewellite_rich_threshold)
        
        # Min Stone Size (log scale)
        ax_min_size = plt.axes([0.82, 0.50, 0.15, slider_height])
        self.slider_min_size = Slider(ax_min_size, 'Min Size', 10000, 100000, valinit=self.min_stone_size, valstep=5000)
        self.slider_min_size.on_changed(self.update_min_stone_size)
        
        # Hole Fill Size
        ax_hole_fill = plt.axes([0.82, 0.45, 0.15, slider_height])
        self.slider_hole_fill = Slider(ax_hole_fill, 'Hole Fill', 1000, 20000, valinit=self.hole_fill_size, valstep=500)
        self.slider_hole_fill.on_changed(self.update_hole_fill_size)
        
        # Reset button
        ax_reset = plt.axes([0.82, 0.35, 0.15, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset Params')
        self.btn_reset.on_clicked(self.reset_parameters)
        
        # Colormap selector
        ax_colormap = plt.axes([0.82, 0.25, 0.15, 0.04])
        self.btn_colormap = Button(ax_colormap, 'Next Colormap')
        self.btn_colormap.on_clicked(self.cycle_colormap)
        
        # Add parameter info text
        param_text = f"""Current Settings:
DoG σ1: {self.dog_sigma1:.1f}
DoG σ2: {self.dog_sigma2:.1f}
Stone: {self.stone_threshold:.2f}
Bacteria: {self.bacteria_threshold}%
B-Rich: {self.bacteria_rich_threshold}%
Inter: {self.intergrowth_threshold}%
W-Rich: {self.whewellite_rich_threshold}%
Min Size: {self.min_stone_size:,}
Hole Fill: {self.hole_fill_size:,}
Colormap: {self.current_colormap}"""
        
        self.param_text = slider_ax.text(0.05, 0.15, param_text, fontsize=9, 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                                        verticalalignment='top')
    
    # Slider update methods
    def update_sigma1(self, val):
        self.dog_sigma1 = val
        self.recalculate_and_update()
    
    def update_sigma2(self, val):
        self.dog_sigma2 = val
        self.recalculate_and_update()
    
    def update_stone_threshold(self, val):
        self.stone_threshold = val
        self.recalculate_and_update()
    
    def update_bacteria_threshold(self, val):
        self.bacteria_threshold = int(val)
        self.recalculate_and_update()
    
    def update_bacteria_rich_threshold(self, val):
        self.bacteria_rich_threshold = int(val)
        self.recalculate_and_update()
    
    def update_intergrowth_threshold(self, val):
        self.intergrowth_threshold = int(val)
        self.recalculate_and_update()
    
    def update_whewellite_rich_threshold(self, val):
        self.whewellite_rich_threshold = int(val)
        self.recalculate_and_update()
    
    def update_min_stone_size(self, val):
        self.min_stone_size = int(val)
        self.recalculate_and_update()
    
    def update_hole_fill_size(self, val):
        self.hole_fill_size = int(val)
        self.recalculate_and_update()
    
    def reset_parameters(self, event):
        """Reset all parameters to optimized defaults"""
        self.dog_sigma1 = 5.2
        self.dog_sigma2 = 3.0
        self.stone_threshold = 0.50
        self.min_stone_size = 50000
        self.hole_fill_size = 5858
        self.bacteria_threshold = 5
        self.bacteria_rich_threshold = 15
        self.intergrowth_threshold = 35
        self.whewellite_rich_threshold = 60
        
        # Update slider positions
        self.slider_sigma1.reset()
        self.slider_sigma2.reset()
        self.slider_stone_thresh.reset()
        self.slider_bacteria.reset()
        self.slider_bacteria_rich.reset()
        self.slider_intergrowth.reset()
        self.slider_whewellite_rich.reset()
        self.slider_min_size.reset()
        self.slider_hole_fill.reset()
        
        self.recalculate_and_update()
    
    def cycle_colormap(self, event):
        """Cycle through available colormaps"""
        colormap_list = list(self.colormaps.keys())
        current_index = colormap_list.index(self.current_colormap)
        next_index = (current_index + 1) % len(colormap_list)
        self.current_colormap = colormap_list[next_index]
        
        self.update_displays()
        print(f"🎨 Colormap changed to: {self.current_colormap}")
        if self.current_colormap == 'morphological':
            print("   ✨ Morphological enhancement active - Beautiful contrast enabled!")
    
    def recalculate_and_update(self):
        """Recalculate analysis and update displays"""
        # Clear line scan data when parameters change
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.line_scan_densities = None
        self.click_count = 0
        
        # Recalculate analysis
        self.recalculate_analysis()
        
        # Update displays
        self.update_displays()
    
    def update_displays(self):
        """Update all display elements"""
        # Update composition map and overlays
        colors = self.colormaps[self.current_colormap]
        cmap_zones = ListedColormap(colors)
        
        # Update stone mask
        stone_coverage = np.sum(self.stone_mask) / self.stone_mask.size * 100
        self.axes[3].clear()
        self.axes[3].imshow(self.stone_mask, cmap='gray', interpolation='nearest')
        self.axes[3].set_title(f'Stone Mask ({stone_coverage:.1f}%)')
        self.axes[3].axis('off')
        
        # Update composition zones with enhanced contrast
        self.comp_ax.clear()
        
        # Apply morphological enhancement for better contrast
        if self.current_colormap == 'morphological':
            enhanced_comp_map = self.enhance_composition_contrast(self.composition_map)
            self.comp_ax.imshow(enhanced_comp_map, cmap=cmap_zones, vmin=0, vmax=11, interpolation='bilinear')
        else:
            self.comp_ax.imshow(self.composition_map, cmap=cmap_zones, vmin=0, vmax=11, interpolation='nearest')
        
        self.comp_ax.set_title('Composition Zones - Enhanced Contrast (Click for Line Scan)')
        self.comp_ax.axis('off')
        
        # Update overlay
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        ct_uint8 = (ct_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        ct_clahe = clahe.apply(ct_uint8)
        
        self.axes[5].clear()
        self.axes[5].imshow(ct_clahe, cmap='gray', alpha=0.7, interpolation='nearest')
        overlay_map = np.where(self.stone_mask, self.composition_map, 0)
        
        # Apply enhancement to overlay if using morphological colormap
        if self.current_colormap == 'morphological':
            enhanced_overlay = self.enhance_composition_contrast(overlay_map)
            self.axes[5].imshow(enhanced_overlay, cmap=cmap_zones, vmin=0, vmax=11, alpha=0.8, interpolation='bilinear')
        else:
            self.axes[5].imshow(overlay_map, cmap=cmap_zones, vmin=0, vmax=11, alpha=0.6, interpolation='nearest')
        
        self.axes[5].set_title('CLAHE + Enhanced Composition Overlay')
        self.axes[5].axis('off')
        
        # Update parameter text
        param_text = f"""Current Settings:
DoG σ1: {self.dog_sigma1:.1f}
DoG σ2: {self.dog_sigma2:.1f}
Stone: {self.stone_threshold:.2f}
Bacteria: {self.bacteria_threshold}%
B-Rich: {self.bacteria_rich_threshold}%
Inter: {self.intergrowth_threshold}%
W-Rich: {self.whewellite_rich_threshold}%
Min Size: {self.min_stone_size:,}
Hole Fill: {self.hole_fill_size:,}
Colormap: {self.current_colormap}"""
        
        self.param_text.set_text(param_text)
        
        # Redraw canvas
        self.fig.canvas.draw()
    
    def on_click(self, event):
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
                
                # Draw start point
                self.comp_ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                self.fig.canvas.draw()
                
            elif self.click_count == 1:
                # Second click - end point
                self.line_end = (y, x)
                self.click_count = 0
                print(f"✅ Line end point set: ({x}, {y})")
                
                # Draw end point and line
                self.comp_ax.plot(x, y, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
                self.comp_ax.plot([self.line_start[1], self.line_end[1]], 
                                [self.line_start[0], self.line_end[0]], 
                                'white', linewidth=4, alpha=0.8)
                self.comp_ax.plot([self.line_start[1], self.line_end[1]], 
                                [self.line_start[0], self.line_end[0]], 
                                'black', linewidth=2, alpha=0.8)
                self.fig.canvas.draw()
                
                # Extract and display line scan in new window
                self.extract_and_display_line_scan()
    
    def extract_and_display_line_scan(self):
        """Extract line scan data and open in new window with grouped analysis"""
        if self.line_start is None or self.line_end is None:
            return
        
        # Extract line profiles (fine detail for composition visualization)
        composition_profile = measure.profile_line(
            self.composition_map, self.line_start, self.line_end,
            linewidth=1, mode='constant'
        )
        
        intensity_profile = measure.profile_line(
            self.ct_image, self.line_start, self.line_end,
            linewidth=1, mode='constant'
        )
        
        # Convert intensities to densities using fine mapping
        density_profile = self.intensity_to_density(intensity_profile)
        # Clamp negative densities to zero
        density_profile = np.maximum(density_profile, 0)
        
        # Group zones for cleaner line profile analysis
        grouped_profile, grouped_densities = self.group_zones_for_line_profile(composition_profile)
        
        self.line_scan_data = composition_profile  # Keep fine detail for composition plot
        self.line_scan_densities = density_profile  # Use actual density profile calculated from intensity
        
        print(f"📊 Line scan extracted: {len(composition_profile)} pixels")
        print(f"📊 Fine zones grouped into 5 categories for analysis")
        
        # Create new window for line scan
        self.create_line_scan_window()
    
    def create_line_scan_window(self):
        """Create separate window for line scan analysis"""
        # Create new figure
        fig_line = plt.figure(figsize=(16, 10))
        fig_line.suptitle(f'Line Scan Analysis: ({self.line_start[1]}, {self.line_start[0]}) → ({self.line_end[1]}, {self.line_end[0]})', 
                         fontsize=14, fontweight='bold')
        
        # Create subplots - adjusted layout to make room for density controls
        gs = fig_line.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[2, 2, 1, 0.3])
        
        ax1 = fig_line.add_subplot(gs[0, :2])  # Composition profile
        ax2 = fig_line.add_subplot(gs[1, :2])  # Density profile
        ax3 = fig_line.add_subplot(gs[2, :2])  # CT intensity profile
        ax4 = fig_line.add_subplot(gs[0:3, 2])   # Statistics horizontal bar chart (full height)
        
        x_pixels = np.arange(len(self.line_scan_data))
        colors = self.colormaps[self.current_colormap]
        
        # 1. Composition profile (fine detail - 10% zones)
        ax1.plot(x_pixels, self.line_scan_data, 'o-', linewidth=2, markersize=3, color='blue')
        
        # Color background according to fine composition zones
        for i in range(len(self.line_scan_data)):
            comp_val = int(self.line_scan_data[i])
            if comp_val < len(colors):
                ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color=colors[comp_val])
        
        ax1.set_ylabel('% Bacteria')
        ax1.set_ylim(0.5, 11.5)  # Exclude holes (zone 12)
        ax1.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ax1.set_yticklabels(['100', '', '80', '', '60', 
                           '', '40', '', '20', '', '0'])
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Composition Profile - Fine Detail ({len(self.line_scan_data)} pixels)')
        
        # Add color legend under composition plot
        legend_elements = []
        composition_labels = ['Background', '100% Bacteria', '90% Bacteria', '80% Bacteria', '70% Bacteria', '60% Bacteria', 
                            '50/50 Mix', '40% Bacteria', '30% Bacteria', '20% Bacteria', '10% Bacteria', 'Pure Whewellite', 'Holes/Voids']
        
        for i, (color, label) in enumerate(zip(colors, composition_labels)):
            legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=label))
        
        # Create legend below the composition plot (adjust ncol for 13 items total)
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                  ncol=5, fontsize=7, frameon=True, fancybox=True, shadow=True)
        
        # 2. Enhanced Density profile with grouped categories (cleaner analysis)
        # Define density thresholds for grouped composition regions
        air_density = 0.001
        bacteria_density = 0.95
        bacteria_rich_density = 1.15
        intergrowth_density = 1.45
        whewellite_rich_density = 1.75
        pure_whewellite_density = 2.23
        
        # Background shading for grouped composition regions
        ax2.axhspan(air_density, bacteria_density, color='#e3f2fd', alpha=0.3, label='Pure Bacteria (Zones 1-2)')
        ax2.axhspan(bacteria_density, bacteria_rich_density, color='#e8f5e8', alpha=0.3, label='Bacteria-Rich (Zones 3-4)')
        ax2.axhspan(bacteria_rich_density, intergrowth_density, color='#fff3e0', alpha=0.3, label='Intergrowth (Zones 5-6)')
        ax2.axhspan(intergrowth_density, whewellite_rich_density, color='#fce4ec', alpha=0.3, label='Whewellite-Rich (Zones 7-8)')
        ax2.axhspan(whewellite_rich_density, pure_whewellite_density + 0.1, color='#f3e5f5', alpha=0.3, label='Pure Whewellite (Zones 9-11)')
        
        # Horizontal threshold lines
        threshold_densities = [bacteria_density, bacteria_rich_density, intergrowth_density, whewellite_rich_density, pure_whewellite_density]
        
        for density in threshold_densities:
            ax2.axhline(density, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Plot grouped density profile (cleaner, using 5 categories)
        ax2.plot(x_pixels, self.line_scan_densities, 'o-', linewidth=2, markersize=3, color='red')
        ax2.fill_between(x_pixels, self.line_scan_densities, alpha=0.3, color='red')
        
        # Calculate statistics
        density_min, density_max = self.line_scan_densities.min(), self.line_scan_densities.max()
        density_mean = self.line_scan_densities.mean()
        ax2.axhline(density_mean, color='black', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean: {density_mean:.2f} g/cm³')
        
        ax2.set_ylabel('Density (g/cm³)')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Density Profile - Calculated from CT Intensity')
        
        # Set y-axis limits starting at 0.5
        ax2.set_ylim(0.5, pure_whewellite_density + 0.1)
        
        ax2.legend(fontsize=8)
        
        # 3. CT intensity profile
        intensity_profile = measure.profile_line(
            self.ct_image, self.line_start, self.line_end,
            linewidth=1, mode='constant'
        )
        
        ax3.plot(x_pixels, intensity_profile, 'o-', linewidth=2, markersize=3, color='green')
        ax3.fill_between(x_pixels, intensity_profile, alpha=0.3, color='green')
        ax3.set_xlabel('Pixel Position Along Line')
        ax3.set_ylabel('CT Intensity (HU)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('CT Intensity Profile')
        
        # Set tight y-axis limits for CT intensity to show variations
        intensity_min, intensity_max = intensity_profile.min(), intensity_profile.max()
        intensity_range = intensity_max - intensity_min
        intensity_margin = max(intensity_range * 0.05, 50)  # 5% margin or at least 50 HU
        ax3.set_ylim(intensity_min - intensity_margin, intensity_max + intensity_margin)
        
        # 4. Statistics horizontal bar chart - Show grouped categories for cleaner analysis
        grouped_profile, _ = self.group_zones_for_line_profile(self.line_scan_data)
        composition_names = ['Background', 'Pure Bacteria (Zones 1-2)', 'Bacteria-Rich (Zones 3-4)', 'Intergrowth (Zones 5-6)', 'Whewellite-Rich (Zones 7-8)', 'Pure Whewellite (Zones 9-11)', 'Holes/Voids']
        unique_values, counts = np.unique(grouped_profile, return_counts=True)
        
        # Filter out background (value 0) from chart but include holes/voids (value 6)
        labels = []
        percentages = []
        colors_bar = []
        
        total_non_background = 0
        for val, count in zip(unique_values, counts):
            if val > 0 and val < len(composition_names) and count > 0:  # Exclude background (val=0)
                total_non_background += count
        
        # Map grouped categories to appropriate colors
        grouped_colors = ['#440154', '#2a788e', '#22a884', '#7ad151', '#f0f921', '#000000']  # Representative colors for 5 groups + holes
        
        for val, count in zip(unique_values, counts):
            if val > 0 and val < len(composition_names) and count > 0:  # Exclude background (val=0)
                name = composition_names[int(val)]
                # Get density from line profile categories
                category_densities = [0.95, 1.15, 1.45, 1.75, 2.23, 0.001]  # Grouped densities
                density = category_densities[int(val)-1] if val <= 6 else 0.001
                pct = count/total_non_background*100 if total_non_background > 0 else 0
                if name == 'Holes/Voids':
                    labels.append(f'{name}\n({density:.3f} g/cm³)')
                else:
                    labels.append(f'{name}\n({density:.2f} g/cm³)')
                percentages.append(pct)
                colors_bar.append(grouped_colors[int(val)-1] if val <= 6 else '#000000')
        
        if percentages:
            # Create horizontal bar chart
            y_pos = np.arange(len(labels))
            bars = ax4.barh(y_pos, percentages, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Customize the chart
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(labels, fontsize=9)
            ax4.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
            ax4.set_title('Stone Composition\n(Excluding Background)', fontsize=12, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add percentage labels on bars
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                width = bar.get_width()
                ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{pct:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)
            
            # Set x-axis limits to accommodate labels
            ax4.set_xlim(0, max(percentages) * 1.2)
            
            # Invert y-axis to show highest percentage at top
            ax4.invert_yaxis()
            
        else:
            ax4.text(0.5, 0.5, 'No stone data\nin line scan', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Stone Composition')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        self.print_line_scan_statistics()
    
    def print_line_scan_statistics(self):
        """Print detailed line scan statistics"""
        if self.line_scan_densities is None:
            return
        
        print(f"\n📈 Line Scan Analysis Results")
        print(f"═" * 50)
        print(f"Line coordinates: ({self.line_start[1]}, {self.line_start[0]}) → ({self.line_end[1]}, {self.line_end[0]})")
        print(f"Profile length: {len(self.line_scan_data)} pixels")
        print(f"Density range: {self.line_scan_densities.min():.3f} - {self.line_scan_densities.max():.3f} g/cm³")
        print(f"Mean density: {self.line_scan_densities.mean():.3f} ± {self.line_scan_densities.std():.3f} g/cm³")
        print(f"Median density: {np.median(self.line_scan_densities):.3f} g/cm³")
        
        # Composition statistics (excluding background)
        composition_names = ['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite', 'Holes/Voids']
        unique_values, counts = np.unique(self.line_scan_data, return_counts=True)
        
        # Calculate totals excluding background
        total_pixels = len(self.line_scan_data)
        background_count = np.sum(self.line_scan_data == 0)
        stone_count = total_pixels - background_count
        
        print(f"\nComposition breakdown:")
        print(f"  {'Background':15s}: {background_count:4d} pixels ({background_count/total_pixels*100:5.1f}%) - excluded from analysis")
        
        for val, count in zip(unique_values, counts):
            if val > 0 and val < len(composition_names) and count > 0:  # Exclude background
                name = composition_names[int(val)]
                pct_total = count/total_pixels*100
                pct_stone = count/stone_count*100 if stone_count > 0 else 0
                if name in self.component_densities:
                    density = self.component_densities[name]
                    if name == 'Holes/Voids':
                        print(f"  {name:15s}: {count:4d} pixels ({pct_total:5.1f}% total, {pct_stone:5.1f}% of stone) - {density:.3f} g/cm³ (air)")
                    else:
                        print(f"  {name:15s}: {count:4d} pixels ({pct_total:5.1f}% total, {pct_stone:5.1f}% of stone) - {density:.2f} g/cm³")
    
    def add_control_buttons(self):
        """Add control buttons to the main window"""
        # Create button axes
        ax_clear = plt.axes([0.02, 0.02, 0.1, 0.04])
        ax_export = plt.axes([0.14, 0.02, 0.1, 0.04])
        ax_save = plt.axes([0.26, 0.02, 0.1, 0.04])
        
        # Create buttons
        self.btn_clear = Button(ax_clear, 'Clear Line')
        self.btn_export = Button(ax_export, 'Export Data')
        self.btn_save = Button(ax_save, 'Save Settings')
        
        # Connect button callbacks
        self.btn_clear.on_clicked(self.clear_line)
        self.btn_export.on_clicked(self.export_data)
        self.btn_save.on_clicked(self.save_settings)
    
    def clear_line(self, event):
        """Clear current line scan"""
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.line_scan_densities = None
        self.click_count = 0
        
        # Redraw composition plot
        colors = self.colormaps[self.current_colormap]
        cmap_zones = ListedColormap(colors)
        self.comp_ax.clear()
        self.comp_ax.imshow(self.composition_map, cmap=cmap_zones, vmin=0, vmax=11)
        self.comp_ax.set_title('Composition Zones (Click for Line Scan)')
        self.comp_ax.axis('off')
        self.fig.canvas.draw()
        
        print("🗑️ Line scan cleared")
    
    def export_data(self, event):
        """Export line scan data to CSV"""
        if self.line_scan_data is None:
            print("❌ No line scan data to export. Create a line scan first.")
            return
        
        # Create export data
        x_pixels = np.arange(len(self.line_scan_data))
        intensity_profile = measure.profile_line(
            self.ct_image, self.line_start, self.line_end,
            linewidth=1, mode='constant'
        )
        
        # Add composition names for easier interpretation
        composition_names = ['Background', 'Pure Bacteria', 'Bacteria-Rich', 'Intergrowth', 'Whewellite-Rich', 'Pure Whewellite', 'Holes/Voids']
        composition_labels = [composition_names[int(val)] if int(val) < len(composition_names) else 'Unknown' 
                            for val in self.line_scan_data]
        
        export_df = pd.DataFrame({
            'Pixel_Position': x_pixels,
            'CT_Intensity_HU': intensity_profile,
            'Density_g_cm3': self.line_scan_densities,
            'Composition_Zone': self.line_scan_data,
            'Composition_Name': composition_labels
        })
        
        filename = f'line_scan_{self.line_start[1]}_{self.line_start[0]}_to_{self.line_end[1]}_{self.line_end[0]}.csv'
        export_df.to_csv(filename, index=False)
        
        # Calculate and add summary statistics
        stone_pixels = np.sum(self.line_scan_data > 0)
        background_pixels = np.sum(self.line_scan_data == 0)
        
        print(f"\n💾 Line scan data exported to: {filename}")
        print(f"   Columns: {list(export_df.columns)}")
        print(f"   Total rows: {len(export_df)}")
        print(f"   Stone pixels: {stone_pixels} ({stone_pixels/len(export_df)*100:.1f}%)")
        print(f"   Background pixels: {background_pixels} ({background_pixels/len(export_df)*100:.1f}%)")
        print(f"   Density range: {self.line_scan_densities.min():.3f} - {self.line_scan_densities.max():.3f} g/cm³")
        print(f"   Mean density: {self.line_scan_densities.mean():.3f} g/cm³")
        
        # Also save metadata
        metadata = {
            'line_start': self.line_start,
            'line_end': self.line_end,
            'total_pixels': len(self.line_scan_data),
            'stone_pixels': int(stone_pixels),
            'background_pixels': int(background_pixels),
            'density_min': float(self.line_scan_densities.min()),
            'density_max': float(self.line_scan_densities.max()),
            'density_mean': float(self.line_scan_densities.mean()),
            'density_std': float(self.line_scan_densities.std()),
            'analysis_parameters': {
                'dog_sigma1': self.dog_sigma1,
                'dog_sigma2': self.dog_sigma2,
                'stone_threshold': self.stone_threshold,
                'bacteria_threshold': self.bacteria_threshold,
                'bacteria_rich_threshold': self.bacteria_rich_threshold,
                'intergrowth_threshold': self.intergrowth_threshold,
                'whewellite_rich_threshold': self.whewellite_rich_threshold
            }
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.pkl')
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"   Metadata saved to: {metadata_filename}")
    
    def save_settings(self, event):
        """Save current analysis settings"""
        settings = {
            'dog_sigma1': self.dog_sigma1,
            'dog_sigma2': self.dog_sigma2,
            'stone_threshold': self.stone_threshold,
            'min_stone_size': self.min_stone_size,
            'hole_fill_size': self.hole_fill_size,
            'bacteria_threshold': self.bacteria_threshold,
            'bacteria_rich_threshold': self.bacteria_rich_threshold,
            'intergrowth_threshold': self.intergrowth_threshold,
            'whewellite_rich_threshold': self.whewellite_rich_threshold,
            'current_colormap': self.current_colormap,
            'component_densities': self.component_densities
        }
        
        with open('stone_analysis_settings.pkl', 'wb') as f:
            pickle.dump(settings, f)
        
        print("💾 Settings saved to 'stone_analysis_settings.pkl'")

    def group_zones_for_line_profile(self, composition_profile):
        """Group fine composition zones into broader categories for line profile analysis"""
        grouped_profile = np.zeros_like(composition_profile)
        grouped_densities = np.zeros_like(composition_profile, dtype=float)
        
        for i, zone in enumerate(composition_profile):
            zone = int(zone)
            # Map fine zones to grouped categories
            if zone in [1, 2]:  # Pure Bacteria group
                grouped_profile[i] = 1
                grouped_densities[i] = 0.95
            elif zone in [3, 4]:  # Bacteria-Rich group
                grouped_profile[i] = 2
                grouped_densities[i] = 1.15
            elif zone in [5, 6]:  # Intergrowth group
                grouped_profile[i] = 3
                grouped_densities[i] = 1.45
            elif zone in [7, 8]:  # Whewellite-Rich group
                grouped_profile[i] = 4
                grouped_densities[i] = 1.75
            elif zone in [9, 10, 11]:  # Pure Whewellite group
                grouped_profile[i] = 5
                grouped_densities[i] = 2.23
            elif zone == 12:  # Holes/Voids
                grouped_profile[i] = 6
                grouped_densities[i] = 0.001
            else:  # Background
                grouped_profile[i] = 0
                grouped_densities[i] = 0.001
        
        return grouped_profile, grouped_densities

    def enhance_composition_contrast(self, composition_map):
        """Apply morphological enhancement to composition zones for better contrast"""
        # Create enhanced composition map
        enhanced_map = composition_map.copy().astype(np.float32)
        
        # Apply morphological operations for edge enhancement
        from skimage import morphology, filters
        
        # Create a structuring element
        selem = morphology.disk(2)
        
        # Apply morphological gradient to enhance boundaries between zones
        morph_gradient = morphology.dilation(enhanced_map, selem) - morphology.erosion(enhanced_map, selem)
        
        # Enhance edges using a small Gaussian filter
        edges = filters.gaussian(morph_gradient, sigma=0.5)
        
        # Combine original with edge enhancement
        enhanced_map = enhanced_map + 0.3 * edges
        
        # Apply adaptive contrast enhancement
        for zone in range(1, 12):  # For each composition zone
            zone_mask = (composition_map == zone)
            if np.sum(zone_mask) > 0:
                # Apply local contrast enhancement
                zone_values = enhanced_map[zone_mask]
                # Stretch contrast within each zone
                zone_min, zone_max = np.percentile(zone_values, [5, 95])
                if zone_max > zone_min:
                    enhanced_map[zone_mask] = zone + 0.4 * (zone_values - zone_min) / (zone_max - zone_min)
        
        return enhanced_map
        
    def create_high_contrast_colormap(self):
        """Create high-contrast colormap similar to morphological enhanced images"""
        # High contrast colormap with distinct boundaries
        high_contrast_colors = [
            '#000000',  # Background - Black
            '#4A0E4E',  # Zone 1 - Deep Purple (100% Bacteria)
            '#7209B7',  # Zone 2 - Purple (90% Bacteria)  
            '#2E86AB',  # Zone 3 - Deep Blue (80% Bacteria)
            '#A23B72',  # Zone 4 - Magenta (70% Bacteria)
            '#F18F01',  # Zone 5 - Orange (60% Bacteria)
            '#C73E1D',  # Zone 6 - Red (50/50 Mix)
            '#F4A261',  # Zone 7 - Light Orange (40% Bacteria)
            '#E9C46A',  # Zone 8 - Yellow (30% Bacteria)
            '#2A9D8F',  # Zone 9 - Teal (20% Bacteria)
            '#264653',  # Zone 10 - Dark Green (10% Bacteria)
            '#FFE66D',  # Zone 11 - Bright Yellow (Pure Whewellite)
            '#FFFFFF'   # Zone 12 - White (Holes/Voids)
        ]
        return high_contrast_colors

def main():
    """Main function to run the standalone application"""
    print("🔬 CT-Raman Stone Analysis - Standalone Application")
    print("=" * 60)
    print("Features:")
    print("• Click-based line scan creation")
    print("• Separate windows for line scan plots")
    print("• Raman-calibrated density values (g/cm³)")
    print("• Professional data export (CSV)")
    print("=" * 60)
    
    try:
        app = StoneAnalysisApp()
        print("\n🎯 Instructions:")
        print("1. Click first point on composition zones image")
        print("2. Click second point to create line scan")
        print("3. New window will open with density profile")
        print("4. Use buttons for export and settings")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 