#!/usr/bin/env python3
"""
Modern CT-Raman Kidney Stone Analysis Application (PySide6)
Fast, responsive GUI with Qt widgets and embedded matplotlib canvases
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image
from skimage import filters, morphology, measure, exposure
from scipy import ndimage, interpolate
import pickle
import pandas as pd

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QSlider, QComboBox,
                               QGroupBox, QGridLayout, QSplitter, QFileDialog, QMessageBox,
                               QTextEdit, QScrollArea, QSpinBox)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

class StoneAnalysisApp(QMainWindow):
    def __init__(self, ct_image_path=None):
        super().__init__()
        self.setWindowTitle("CT-Raman Stone Analysis - Modern Interface")
        self.setGeometry(50, 50, 1600, 900)
        
        # Load CT image
        if ct_image_path is None:
            ct_image_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select CT Image File",
                "",
                "TIFF files (*.tif *.tiff);;All image files (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*.*)"
            )
            
            if not ct_image_path:
                QMessageBox.warning(self, "No Image", "No image file selected. Exiting.")
                sys.exit(0)
        
        try:
            self.ct_image = np.array(Image.open(ct_image_path))
            self.ct_shape = self.ct_image.shape
            print(f"✅ CT image loaded: {ct_image_path}")
            print(f"   Shape: {self.ct_shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {e}")
            sys.exit(1)
        
        # Setup density calibration
        self.setup_density_calibration()
        
        # Default parameters (matching classic version)
        self.dog_sigma1 = 5.2
        self.dog_sigma2 = 3.0
        self.stone_threshold = 0.50
        self.min_stone_size = 50000
        self.hole_fill_size = 5858
        self.bacteria_threshold = 5
        self.bacteria_rich_threshold = 15
        self.intergrowth_threshold = 35
        self.whewellite_rich_threshold = 60
        
        # Colormaps (13 colors for zones 0-12)
        self.colormaps = {
            'viridis': ['#000000', '#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21908c', '#22a884', '#44bf70', '#7ad151', '#bddf26', '#f0f921', '#FFFFFF'],
            'original': ['#000000', '#8B0000', '#B22222', '#DC143C', '#FF4500', '#FF6347', '#FFA500', '#FFD700', '#ADFF2F', '#7FFF00', '#00FF00', '#90EE90', '#FFFFFF'],
            'plasma': ['#000000', '#0d0887', '#4b0c6b', '#6a00a8', '#8b0aa6', '#a53582', '#bc5090', '#d1719b', '#e594a7', '#f2b9b2', '#fcddbf', '#f0f921', '#FFFFFF'],
            'morphological': self.create_high_contrast_colormap(),
            'thermal': ['#000000', '#000080', '#000099', '#0000FF', '#0066FF', '#00CCFF', '#00FFCC', '#66FF66', '#CCFF00', '#FFCC00', '#FF6600', '#FF0000', '#FFFFFF'],
            'geological': ['#000000', '#2C1810', '#4A2C2A', '#6B4423', '#8B5A2B', '#CD853F', '#DAA520', '#F0E68C', '#FFFACD', '#F5F5DC', '#FFFFF0', '#FFFEF0', '#FFFFFF'],
        }
        self.current_colormap = 'morphological'
        
        # Line scan data
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.line_scan_data_raw = None
        self.line_scan_densities_raw = None
        self.click_count = 0
        self.bin_size = 10  # Default bin size for line scan smoothing
        
        # Analysis results
        self.dog_enhanced = None
        self.stone_mask = None
        self.density_map = None
        self.composition_map = None
        
        # Update timer for debouncing slider changes
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.recalculate_analysis)
        
        # Setup UI
        self.setup_ui()
        
        # Initial analysis
        self.recalculate_analysis()
        
    def setup_density_calibration(self):
        """Setup Raman-based density calibration"""
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
        
        # Determine background CT intensity
        temp_stone_mask = self.create_initial_stone_mask()
        background_pixels = self.ct_image[~temp_stone_mask]
        
        if len(background_pixels) > 0:
            self.background_ct_intensity = np.median(background_pixels)
        else:
            self.background_ct_intensity = self.ct_image.min()
        
        self.ct_min = float(self.ct_image.min())
        self.ct_max = float(self.ct_image.max())
        self.density_min = 0.001
        self.density_max = 2.23
        
    def create_initial_stone_mask(self):
        """Create initial rough stone mask for background identification"""
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(ct_norm)
        initial_mask = ct_norm > thresh
        
        kernel = morphology.disk(5)
        initial_mask = morphology.binary_opening(initial_mask, kernel)
        initial_mask = morphology.remove_small_objects(initial_mask, min_size=10000)
        
        return initial_mask
    
    def create_high_contrast_colormap(self):
        """Create high contrast colormap with 13 colors for zones 0-12"""
        return ['#000000', '#8B0000', '#DC143C', '#FF4500', '#FFA500', '#FFD700', '#ADFF2F', '#00FF00', '#00FFFF', '#0080FF', '#0000FF', '#FF00FF', '#FFFFFF']
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        # Right panel - Visualizations
        viz_panel = self.create_visualization_panel()
        main_layout.addWidget(viz_panel, stretch=4)
        
    def create_control_panel(self):
        """Create the control panel with sliders"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Parameter Controls")
        title.setFont(QFont("Helvetica", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # DoG Parameters
        dog_group = QGroupBox("DoG Filtering")
        dog_layout = QVBoxLayout()
        
        self.sigma1_slider = self.create_slider("DoG σ1", 1.0, 10.0, self.dog_sigma1, 0.1, dog_layout)
        self.sigma2_slider = self.create_slider("DoG σ2", 0.5, 8.0, self.dog_sigma2, 0.1, dog_layout)
        
        dog_group.setLayout(dog_layout)
        layout.addWidget(dog_group)
        
        # Stone Isolation
        stone_group = QGroupBox("Stone Isolation")
        stone_layout = QVBoxLayout()
        
        self.stone_thresh_slider = self.create_slider("Stone Threshold", 0.1, 0.9, self.stone_threshold, 0.01, stone_layout)
        self.min_size_slider = self.create_slider("Min Size", 10000, 100000, self.min_stone_size, 5000, stone_layout)
        self.hole_fill_slider = self.create_slider("Hole Fill", 1000, 20000, self.hole_fill_size, 500, stone_layout)
        
        stone_group.setLayout(stone_layout)
        layout.addWidget(stone_group)
        
        # Composition Thresholds
        comp_group = QGroupBox("Composition Classification")
        comp_layout = QVBoxLayout()
        
        self.bacteria_slider = self.create_slider("Bacteria %", 5, 50, self.bacteria_threshold, 1, comp_layout)
        self.bacteria_rich_slider = self.create_slider("Bacteria-Rich %", 10, 70, self.bacteria_rich_threshold, 1, comp_layout)
        self.intergrowth_slider = self.create_slider("Intergrowth %", 20, 80, self.intergrowth_threshold, 1, comp_layout)
        self.whewellite_rich_slider = self.create_slider("Whewellite-Rich %", 50, 90, self.whewellite_rich_threshold, 1, comp_layout)
        
        comp_group.setLayout(comp_layout)
        layout.addWidget(comp_group)
        
        # Colormap selector
        colormap_layout = QHBoxLayout()
        colormap_label = QLabel("Colormap:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(self.colormaps.keys()))
        self.colormap_combo.setCurrentText(self.current_colormap)
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_combo)
        layout.addLayout(colormap_layout)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        reset_btn = QPushButton("Reset Parameters")
        reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(reset_btn)
        
        clear_btn = QPushButton("Clear Line Scan")
        clear_btn.clicked.connect(self.clear_line)
        button_layout.addWidget(clear_btn)
        
        # Binning controls
        bin_layout = QHBoxLayout()
        bin_label = QLabel("Bin Size:")
        self.bin_spinbox = QSpinBox()
        self.bin_spinbox.setMinimum(1)
        self.bin_spinbox.setMaximum(100)
        self.bin_spinbox.setValue(self.bin_size)
        self.bin_spinbox.setSuffix(" px")
        bin_layout.addWidget(bin_label)
        bin_layout.addWidget(self.bin_spinbox)
        button_layout.addLayout(bin_layout)
        
        update_bin_btn = QPushButton("Update Binning")
        update_bin_btn.clicked.connect(self.update_binning)
        button_layout.addWidget(update_bin_btn)
        
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        button_layout.addWidget(export_btn)
        
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setStyleSheet("background-color: #f0f0f0; font-family: monospace; font-size: 9pt;")
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)
        
        layout.addStretch()
        
        return panel
        
    def create_slider(self, label, min_val, max_val, default_val, step, parent_layout):
        """Create a labeled slider"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label with value
        label_widget = QLabel(f"{label}: {default_val:.2f}")
        label_widget.setFont(QFont("Helvetica", 10))
        layout.addWidget(label_widget)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        
        # Convert to integer range for slider
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setValue(int(default_val / step))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((max_val - min_val) / step / 10))
        
        # Store metadata
        slider.step_size = step
        slider.label_widget = label_widget
        slider.label_text = label
        
        # Connect signal
        slider.valueChanged.connect(lambda v: self.on_slider_change(slider, v))
        
        layout.addWidget(slider)
        parent_layout.addWidget(container)
        
        return slider
        
    def on_slider_change(self, slider, value):
        """Handle slider value change"""
        real_value = value * slider.step_size
        slider.label_widget.setText(f"{slider.label_text}: {real_value:.2f}")
        
        # Update parameter
        if slider == self.sigma1_slider:
            self.dog_sigma1 = real_value
        elif slider == self.sigma2_slider:
            self.dog_sigma2 = real_value
        elif slider == self.stone_thresh_slider:
            self.stone_threshold = real_value
        elif slider == self.min_size_slider:
            self.min_stone_size = int(real_value)
        elif slider == self.hole_fill_slider:
            self.hole_fill_size = int(real_value)
        elif slider == self.bacteria_slider:
            self.bacteria_threshold = int(real_value)
        elif slider == self.bacteria_rich_slider:
            self.bacteria_rich_threshold = int(real_value)
        elif slider == self.intergrowth_slider:
            self.intergrowth_threshold = int(real_value)
        elif slider == self.whewellite_rich_slider:
            self.whewellite_rich_threshold = int(real_value)
        
        # Debounce updates - wait 300ms after last change
        self.update_timer.start(300)
        
    def create_visualization_panel(self):
        """Create the visualization panel with matplotlib canvases"""
        panel = QWidget()
        layout = QGridLayout(panel)
        
        # Create 6 canvases in 2x3 grid
        self.canvas_ct = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_dog = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_mask = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_density = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_comp = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_line = MplCanvas(self, width=5, height=4, dpi=100)
        
        # Add to grid
        layout.addWidget(QLabel("Original CT"), 0, 0)
        layout.addWidget(self.canvas_ct, 1, 0)
        
        layout.addWidget(QLabel("DoG Enhanced"), 0, 1)
        layout.addWidget(self.canvas_dog, 1, 1)
        
        layout.addWidget(QLabel("Stone Mask"), 0, 2)
        layout.addWidget(self.canvas_mask, 1, 2)
        
        layout.addWidget(QLabel("Density Map"), 2, 0)
        layout.addWidget(self.canvas_density, 3, 0)
        
        layout.addWidget(QLabel("Composition Zones"), 2, 1)
        layout.addWidget(self.canvas_comp, 3, 1)
        
        layout.addWidget(QLabel("Line Scan"), 2, 2)
        layout.addWidget(self.canvas_line, 3, 2)
        
        # Connect click handler to composition canvas
        self.canvas_comp.mpl_connect('button_press_event', self.on_canvas_click)
        
        return panel
        
    def recalculate_analysis(self):
        """Recalculate the entire analysis pipeline"""
        self.log_status("Recalculating analysis...")
        
        # Apply DoG filter
        self.apply_dog_filter()
        
        # Create stone mask
        self.create_stone_mask()
        
        # Calculate density map
        self.calculate_density_map()
        
        # Classify composition
        self.classify_composition()
        
        # Update all visualizations
        self.update_visualizations()
        
        self.log_status("Analysis complete!")
        
    def apply_dog_filter(self):
        """Apply Difference of Gaussians filter"""
        # Normalize CT image first (critical for consistent thresholding)
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        gaussian1 = filters.gaussian(ct_norm, sigma=self.dog_sigma1)
        gaussian2 = filters.gaussian(ct_norm, sigma=self.dog_sigma2)
        dog_result = gaussian1 - gaussian2
        # Normalize DoG result to 0-1 range
        self.dog_enhanced = (dog_result - dog_result.min()) / (dog_result.max() - dog_result.min())
        
    def create_stone_mask(self):
        """Create binary mask of stone region"""
        # Normalize CT image
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Normalize and invert DoG (stone interior has low DoG values)
        dog_norm = (self.dog_enhanced - self.dog_enhanced.min()) / (self.dog_enhanced.max() - self.dog_enhanced.min())
        inverted_dog = 1.0 - dog_norm
        
        # Combined score: 70% CT intensity + 30% inverted DoG
        stone_score = 0.7 * ct_norm + 0.3 * inverted_dog
        
        # Threshold to get candidates
        stone_candidates = stone_score > self.stone_threshold
        stone_candidates = morphology.remove_small_objects(stone_candidates, min_size=self.min_stone_size)
        
        # Select largest connected region
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
        
        self.stone_mask = stone_mask
        
    def calculate_density_map(self):
        """Calculate density map from CT intensities"""
        self.density_map = np.zeros_like(self.ct_image, dtype=float)
        self.density_map[self.stone_mask] = self.density_min + \
            (self.ct_image[self.stone_mask] - self.ct_min) / (self.ct_max - self.ct_min) * \
            (self.density_max - self.density_min)
            
    def classify_composition(self):
        """Classify composition into 10-zone system for fine visual detail"""
        self.composition_map = np.zeros_like(self.ct_image, dtype=np.uint8)
        
        if not np.any(self.stone_mask):
            return
        
        # Identify holes within the stone (CT intensity ≤ background)
        hole_mask = self.stone_mask & (self.ct_image <= self.background_ct_intensity)
        
        # Get stone intensities excluding holes
        stone_only_mask = self.stone_mask & ~hole_mask
        
        if np.sum(stone_only_mask) == 0:
            self.composition_map[hole_mask] = 12  # Assign holes to zone 12
            return
        
        stone_intensities = self.ct_image[stone_only_mask]
        min_intensity = np.min(stone_intensities)
        max_intensity = np.max(stone_intensities)
        range_span = max_intensity - min_intensity
        
        # Calculate threshold values for 10-zone system
        thresholds = [
            min_intensity + 0.005 * range_span,  # 0.5% = 100% bacteria threshold
            min_intensity + 0.015 * range_span,  # 1.5% = 90% bacteria
            min_intensity + 0.035 * range_span,  # 3.5% = 80% bacteria
            min_intensity + 0.07 * range_span,   # 7% = 70% bacteria
            min_intensity + 0.12 * range_span,   # 12% = 60% bacteria
            min_intensity + 0.20 * range_span,   # 20% = 50/50 mix
            min_intensity + 0.30 * range_span,   # 30% = 40% bacteria
            min_intensity + 0.42 * range_span,   # 42% = 30% bacteria
            min_intensity + 0.55 * range_span,   # 55% = 20% bacteria
            min_intensity + 0.70 * range_span    # 70% = 10% bacteria
        ]
        
        # Assign zones for stone areas (zones 1-11)
        self.composition_map[stone_only_mask] = 6  # Default to 50/50 mix
        
        # Zone assignments based on intensity ranges
        self.composition_map[stone_only_mask & (self.ct_image <= thresholds[0])] = 1  # 100% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[0]) & (self.ct_image <= thresholds[1])] = 2  # 90% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[1]) & (self.ct_image <= thresholds[2])] = 3  # 80% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[2]) & (self.ct_image <= thresholds[3])] = 4  # 70% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[3]) & (self.ct_image <= thresholds[4])] = 5  # 60% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[4]) & (self.ct_image <= thresholds[5])] = 6  # 50/50 mix
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[5]) & (self.ct_image <= thresholds[6])] = 7  # 40% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[6]) & (self.ct_image <= thresholds[7])] = 8  # 30% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[7]) & (self.ct_image <= thresholds[8])] = 9  # 20% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[8]) & (self.ct_image <= thresholds[9])] = 10 # 10% bacteria
        self.composition_map[stone_only_mask & (self.ct_image > thresholds[9])] = 11  # Pure whewellite
        
        # Assign holes to zone 12
        self.composition_map[hole_mask] = 12
        
    def update_visualizations(self):
        """Update all visualization canvases"""
        # Original CT
        self.canvas_ct.axes.clear()
        self.canvas_ct.axes.imshow(self.ct_image, cmap='gray')
        self.canvas_ct.axes.set_title('Original CT Image')
        self.canvas_ct.axes.axis('off')
        self.canvas_ct.draw()
        
        # DoG Enhanced
        self.canvas_dog.axes.clear()
        self.canvas_dog.axes.imshow(self.dog_enhanced, cmap='gray')
        self.canvas_dog.axes.set_title('DoG Enhanced')
        self.canvas_dog.axes.axis('off')
        self.canvas_dog.draw()
        
        # Stone Mask
        self.canvas_mask.axes.clear()
        self.canvas_mask.axes.imshow(self.stone_mask, cmap='gray')
        stone_pct = np.sum(self.stone_mask) / self.stone_mask.size * 100
        self.canvas_mask.axes.set_title(f'Stone Mask ({stone_pct:.1f}%)')
        self.canvas_mask.axes.axis('off')
        self.canvas_mask.draw()
        
        # Density Map
        self.canvas_density.fig.clear()
        self.canvas_density.axes = self.canvas_density.fig.add_subplot(111)
        im = self.canvas_density.axes.imshow(self.density_map, cmap='viridis', vmin=self.density_min, vmax=self.density_max)
        self.canvas_density.axes.set_title('Density Map (g/cm³)')
        self.canvas_density.axes.axis('off')
        self.canvas_density.fig.colorbar(im, ax=self.canvas_density.axes, fraction=0.046)
        self.canvas_density.draw()
        
        # Composition Map
        self.canvas_comp.axes.clear()
        colors = self.colormaps[self.current_colormap]
        cmap = ListedColormap(colors)
        self.canvas_comp.axes.imshow(self.composition_map, cmap=cmap, vmin=0, vmax=12)
        self.canvas_comp.axes.set_title('Composition Zones (click 2 points for line scan)')
        self.canvas_comp.axes.axis('off')
        
        # Draw line if exists
        if self.line_start and self.line_end:
            y0, x0 = self.line_start
            y1, x1 = self.line_end
            self.canvas_comp.axes.plot([x0, x1], [y0, y1], 'r-', linewidth=2)
            self.canvas_comp.axes.plot(x0, y0, 'ro', markersize=8)
            self.canvas_comp.axes.plot(x1, y1, 'ro', markersize=8)
        
        self.canvas_comp.draw()
        
        # Update line scan if available
        if self.line_scan_data is not None:
            self.update_line_scan_plot()
            
    def on_canvas_click(self, event):
        """Handle click on composition canvas for line scan"""
        if event.inaxes != self.canvas_comp.axes:
            return
            
        if event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            
            if self.click_count == 0:
                self.line_start = (y, x)
                self.click_count = 1
                self.log_status(f"Line start: ({x}, {y})")
                self.update_visualizations()
            elif self.click_count == 1:
                self.line_end = (y, x)
                self.click_count = 0
                self.log_status(f"Line end: ({x}, {y})")
                self.extract_line_scan()
                self.update_visualizations()
    
    def extract_line_scan(self):
        """Extract line scan data between two points"""
        if self.line_start is None or self.line_end is None:
            return
            
        y0, x0 = self.line_start
        y1, x1 = self.line_end
        
        length = int(np.hypot(x1 - x0, y1 - y0))
        x_coords = np.linspace(x0, x1, length)
        y_coords = np.linspace(y0, y1, length)
        
        # Extract raw data
        self.line_scan_data_raw = ndimage.map_coordinates(self.composition_map, [y_coords, x_coords], order=0)
        self.line_scan_densities_raw = ndimage.map_coordinates(self.density_map, [y_coords, x_coords], order=1)
        
        # Apply binning
        self.apply_binning()
        
        self.log_status(f"Line scan extracted: {length} points, binned to {len(self.line_scan_data)} points")
        
    def update_line_scan_plot(self):
        """Update the line scan visualization with composition and density profiles"""
        self.canvas_line.fig.clear()
        
        if self.line_scan_densities is None:
            ax = self.canvas_line.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Click 2 points on\nComposition Zones', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            self.canvas_line.draw()
            return
        
        # Create two subplots
        ax1 = self.canvas_line.fig.add_subplot(211)
        ax2 = self.canvas_line.fig.add_subplot(212)
        
        x = np.arange(len(self.line_scan_densities))
        colors = self.colormaps[self.current_colormap]
        
        # Top plot: Composition profile with colored background
        ax1.plot(x, self.line_scan_data, 'o-', linewidth=2, markersize=4, color='blue')
        
        # Color background according to composition zones
        for i in range(len(self.line_scan_data)):
            comp_val = int(self.line_scan_data[i])
            if comp_val < len(colors):
                ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color=colors[comp_val])
        
        ax1.set_ylabel('Zone')
        ax1.set_ylim(0.5, 12.5)
        ax1.set_yticks([1, 3, 5, 7, 9, 11])
        ax1.set_yticklabels(['100%\nBact', '80%\nBact', '60%\nBact', '40%\nBact', '20%\nBact', 'Pure\nWhew'])
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_title(f'Composition Profile (Bin={self.bin_size}px, {len(self.line_scan_data)} bins)')
        ax1.set_xticklabels([])
        
        # Bottom plot: Density profile
        ax2.plot(x, self.line_scan_densities, 'o-', linewidth=2, markersize=4, color='red')
        ax2.fill_between(x, self.line_scan_densities, alpha=0.3, color='red')
        
        # Add density threshold lines
        ax2.axhline(0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(1.15, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(1.45, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(1.75, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(2.23, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Position along line (bins)')
        ax2.set_ylabel('Density (g/cm³)')
        ax2.set_title('Density Profile')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.8, 2.4)
        
        self.canvas_line.fig.tight_layout()
        self.canvas_line.draw()
        
    def change_colormap(self, colormap_name):
        """Change the colormap"""
        self.current_colormap = colormap_name
        self.update_visualizations()
        
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.dog_sigma1 = 5.2
        self.dog_sigma2 = 3.0
        self.stone_threshold = 0.50
        self.min_stone_size = 50000
        self.hole_fill_size = 5858
        self.bacteria_threshold = 5
        self.bacteria_rich_threshold = 15
        self.intergrowth_threshold = 35
        self.whewellite_rich_threshold = 60
        
        # Update sliders
        self.sigma1_slider.setValue(int(self.dog_sigma1 / self.sigma1_slider.step_size))
        self.sigma2_slider.setValue(int(self.dog_sigma2 / self.sigma2_slider.step_size))
        self.stone_thresh_slider.setValue(int(self.stone_threshold / self.stone_thresh_slider.step_size))
        self.min_size_slider.setValue(int(self.min_stone_size / self.min_size_slider.step_size))
        self.hole_fill_slider.setValue(int(self.hole_fill_size / self.hole_fill_slider.step_size))
        self.bacteria_slider.setValue(int(self.bacteria_threshold / self.bacteria_slider.step_size))
        self.bacteria_rich_slider.setValue(int(self.bacteria_rich_threshold / self.bacteria_rich_slider.step_size))
        self.intergrowth_slider.setValue(int(self.intergrowth_threshold / self.intergrowth_slider.step_size))
        self.whewellite_rich_slider.setValue(int(self.whewellite_rich_threshold / self.whewellite_rich_slider.step_size))
        
        self.recalculate_analysis()
        self.log_status("Parameters reset to defaults")
        
    def apply_binning(self):
        """Apply binning to raw line scan data"""
        if self.line_scan_data_raw is None:
            return
        
        bin_size = self.bin_size
        n_points = len(self.line_scan_data_raw)
        
        if bin_size >= n_points:
            # If bin size is larger than data, just average everything
            self.line_scan_data = np.array([np.median(self.line_scan_data_raw)])
            self.line_scan_densities = np.array([np.mean(self.line_scan_densities_raw)])
            return
        
        # Calculate number of bins
        n_bins = n_points // bin_size
        
        # Bin the data
        binned_composition = []
        binned_density = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size
            
            # Use mode for composition (most common zone)
            bin_comp = self.line_scan_data_raw[start_idx:end_idx]
            mode_comp = np.median(bin_comp)  # Use median as approximation of mode
            binned_composition.append(mode_comp)
            
            # Use mean for density
            bin_dens = self.line_scan_densities_raw[start_idx:end_idx]
            binned_density.append(np.mean(bin_dens))
        
        # Handle remaining points
        if n_points % bin_size != 0:
            start_idx = n_bins * bin_size
            bin_comp = self.line_scan_data_raw[start_idx:]
            bin_dens = self.line_scan_densities_raw[start_idx:]
            binned_composition.append(np.median(bin_comp))
            binned_density.append(np.mean(bin_dens))
        
        self.line_scan_data = np.array(binned_composition)
        self.line_scan_densities = np.array(binned_density)
    
    def update_binning(self):
        """Update binning from spinbox value"""
        self.bin_size = self.bin_spinbox.value()
        
        if self.line_scan_data_raw is not None:
            self.apply_binning()
            self.update_line_scan_plot()
            self.log_status(f"Binning updated: {self.bin_size} pixels -> {len(self.line_scan_data)} bins")
        else:
            self.log_status("No line scan data to bin. Draw a line first.")
    
    def clear_line(self):
        """Clear the line scan"""
        self.line_start = None
        self.line_end = None
        self.line_scan_data = None
        self.line_scan_densities = None
        self.line_scan_data_raw = None
        self.line_scan_densities_raw = None
        self.click_count = 0
        self.update_visualizations()
        self.log_status("Line scan cleared")
        
    def export_data(self):
        """Export line scan data"""
        if self.line_scan_data is None:
            QMessageBox.warning(self, "No Data", "No line scan data to export. Click two points first.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Save Line Scan Data", "line_scan_data.csv", "CSV files (*.csv)")
        
        if filename:
            df = pd.DataFrame({
                'Position': np.arange(len(self.line_scan_densities)),
                'Density_g_cm3': self.line_scan_densities,
                'Composition_Zone': self.line_scan_data
            })
            df.to_csv(filename, index=False)
            self.log_status(f"Data exported to {filename}")
            QMessageBox.information(self, "Export Complete", f"Data saved to:\n{filename}")
            
    def save_settings(self):
        """Save current settings"""
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
            'colormap': self.current_colormap
        }
        
        with open('stone_analysis_settings.pkl', 'wb') as f:
            pickle.dump(settings, f)
            
        self.log_status("Settings saved to stone_analysis_settings.pkl")
        QMessageBox.information(self, "Settings Saved", "Settings saved successfully!")
        
    def log_status(self, message):
        """Log a status message"""
        self.status_text.append(message)
        print(message)

def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    print("🔬 CT-Raman Stone Analysis - Modern Interface")
    print("=" * 60)
    print("Instructions:")
    print("1. Adjust parameters using sliders (updates automatically)")
    print("2. Click two points on Composition Zones for line scan")
    print("3. Use buttons to export data and save settings")
    
    window = StoneAnalysisApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
