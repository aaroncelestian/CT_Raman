import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import filters, morphology, segmentation, measure
from scipy import ndimage, interpolate
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd

class StoneLayerAnalyzer:
    def __init__(self, ct_analyzer, raman_reference_path):
        """
        Initialize Stone Layer Analyzer for compositional analysis
        
        Args:
            ct_analyzer: CTImageAnalyzer instance with loaded CT data
            raman_reference_path: Path to Raman density measurement image (ground truth)
        """
        self.ct_analyzer = ct_analyzer
        self.raman_reference_path = raman_reference_path
        self.raman_density_map = None
        self.composition_map = None
        self.layer_analysis = {}
        self.calibration_params = {}
        
    def extract_raman_density_signatures(self):
        """Extract density signatures from Raman reference data"""
        # Load Raman density measurement
        raman_img = plt.imread(self.raman_reference_path)
        
        # Convert to grayscale if needed
        if len(raman_img.shape) == 3:
            raman_gray = np.dot(raman_img[...,:3], [0.299, 0.587, 0.114])
        else:
            raman_gray = raman_img
        
        # Normalize to 0-1 range
        raman_norm = (raman_gray - raman_gray.min()) / (raman_gray.max() - raman_gray.min())
        
        # Use Gaussian Mixture Model to identify distinct density populations
        # Reshape for GMM
        data = raman_norm.flatten().reshape(-1, 1)
        
        # Fit GMM with 2 components (whewellite vs bacteria)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)
        
        # Get component parameters
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_
        
        # Assign labels based on density (higher density = whewellite)
        if means[0] > means[1]:
            whewellite_idx, bacteria_idx = 0, 1
        else:
            whewellite_idx, bacteria_idx = 1, 0
        
        self.calibration_params = {
            'whewellite_density_mean': means[whewellite_idx],
            'whewellite_density_std': stds[whewellite_idx],
            'bacteria_density_mean': means[bacteria_idx], 
            'bacteria_density_std': stds[bacteria_idx],
            'whewellite_weight': weights[whewellite_idx],
            'bacteria_weight': weights[bacteria_idx],
            'threshold': (means[0] + means[1]) / 2  # Midpoint threshold
        }
        
        # Create reference density map
        labels = gmm.predict(data)
        self.raman_density_map = labels.reshape(raman_norm.shape)
        
        print("Raman Density Calibration:")
        print(f"Whewellite - Mean: {self.calibration_params['whewellite_density_mean']:.3f}, "
              f"Std: {self.calibration_params['whewellite_density_std']:.3f}")
        print(f"Bacteria - Mean: {self.calibration_params['bacteria_density_mean']:.3f}, "
              f"Std: {self.calibration_params['bacteria_density_std']:.3f}")
        print(f"Threshold: {self.calibration_params['threshold']:.3f}")
        
        return self.calibration_params
    
    def apply_raman_calibration_to_ct(self):
        """Apply Raman-derived calibration to the full CT image"""
        if self.ct_analyzer.ct_image is None:
            self.ct_analyzer.load_ct_image()
        
        # First, segment the stone from background
        if not hasattr(self, 'stone_mask'):
            self.segment_stone_from_background()
        
        # Normalize CT image
        ct_norm = (self.ct_analyzer.ct_image - self.ct_analyzer.ct_image.min()) / \
                  (self.ct_analyzer.ct_image.max() - self.ct_analyzer.ct_image.min())
        
        # Create composition map initialized to background (0)
        composition_map = np.zeros_like(ct_norm)
        
        # Only analyze pixels within the stone
        stone_pixels = ct_norm[self.stone_mask]
        
        # Use calibrated threshold to classify regions within the stone
        threshold = self.calibration_params['threshold']
        
        # Adjust threshold based on CT characteristics if needed
        # This accounts for potential differences between Raman and CT intensity scales
        ct_threshold = self.adaptive_threshold_adjustment(stone_pixels, threshold)
        
        # Create binary classification only within stone
        whewellite_mask = (ct_norm >= ct_threshold) & self.stone_mask
        bacteria_mask = (ct_norm < ct_threshold) & self.stone_mask
        
        # Assign composition labels
        composition_map[whewellite_mask] = 2  # Whewellite-dominant
        composition_map[bacteria_mask] = 1   # Bacteria-dominant
        # Background remains 0
        
        # Clean up using morphological operations (only within stone)
        composition_map = self.clean_composition_map(composition_map)
        
        self.composition_map = composition_map
        return composition_map
    
    def adaptive_threshold_adjustment(self, stone_pixels, raman_threshold):
        """Adaptively adjust threshold based on CT image characteristics within stone"""
        # Analyze stone pixel histogram to find equivalent threshold
        hist, bins = np.histogram(stone_pixels.flatten(), bins=100)
        
        # Find peaks in histogram (bimodal distribution)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=np.max(hist)*0.1)
        
        if len(peaks) >= 2:
            # Use the valley between peaks as threshold
            valley_idx = np.argmin(hist[peaks[0]:peaks[-1]]) + peaks[0]
            ct_threshold = bins[valley_idx]
        else:
            # Use Otsu's method as fallback on stone pixels only
            ct_threshold = filters.threshold_otsu(stone_pixels)
        
        print(f"Adaptive threshold adjustment: Raman {raman_threshold:.3f} -> CT {ct_threshold:.3f}")
        return ct_threshold
    
    def clean_composition_map(self, composition_map):
        """Clean up composition map using morphological operations"""
        # Remove small isolated regions
        cleaned_map = np.copy(composition_map)
        
        for label in [1, 2]:  # bacteria, whewellite
            mask = composition_map == label
            # Remove small objects
            mask_cleaned = morphology.remove_small_objects(mask, min_size=100)
            # Fill small holes
            mask_cleaned = morphology.remove_small_holes(mask_cleaned, area_threshold=50)
            
            # Update cleaned map
            cleaned_map[composition_map == label] = 0
            cleaned_map[mask_cleaned] = label
        
        return cleaned_map
    
    def analyze_stone_layers(self, method='radial'):
        """Analyze compositional layers in the kidney stone"""
        if self.composition_map is None:
            self.apply_raman_calibration_to_ct()
        
        if method == 'radial':
            return self.radial_layer_analysis()
        elif method == 'contour':
            return self.contour_layer_analysis()
        elif method == 'gradient':
            return self.gradient_layer_analysis()
    
    def radial_layer_analysis(self):
        """Analyze composition as a function of distance from center"""
        # Find stone center (centroid of stone regions only)
        mask = self.stone_mask  # Use stone mask instead of composition > 0
        coords = np.where(mask)
        center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
        
        # Create distance map from center
        y, x = np.ogrid[:self.composition_map.shape[0], :self.composition_map.shape[1]]
        distance_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Analyze composition vs distance (only within stone)
        max_distance = np.max(distance_map[mask])
        radial_bins = np.linspace(0, max_distance, 20)
        
        radial_analysis = {
            'distances': [],
            'whewellite_fraction': [],
            'bacteria_fraction': [],
            'total_pixels': []
        }
        
        for i in range(len(radial_bins)-1):
            r_min, r_max = radial_bins[i], radial_bins[i+1]
            ring_mask = (distance_map >= r_min) & (distance_map < r_max) & mask
            
            if np.sum(ring_mask) > 0:
                ring_composition = self.composition_map[ring_mask]
                whewellite_count = np.sum(ring_composition == 2)
                bacteria_count = np.sum(ring_composition == 1)
                total_count = len(ring_composition)
                
                radial_analysis['distances'].append((r_min + r_max) / 2)
                radial_analysis['whewellite_fraction'].append(whewellite_count / total_count)
                radial_analysis['bacteria_fraction'].append(bacteria_count / total_count)
                radial_analysis['total_pixels'].append(total_count)
        
        self.layer_analysis['radial'] = radial_analysis
        return radial_analysis
    
    def contour_layer_analysis(self):
        """Analyze composition using contour-based layers"""
        # Create distance transform from stone boundary
        mask = self.stone_mask  # Use stone mask instead of composition > 0
        
        # Distance transform from boundary
        dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Create contour layers
        max_dist = np.max(dist_transform)
        n_layers = 10
        layer_thresholds = np.linspace(0, max_dist, n_layers+1)
        
        contour_analysis = {
            'layer_depths': [],
            'whewellite_fraction': [],
            'bacteria_fraction': [],
            'layer_area': []
        }
        
        for i in range(n_layers):
            depth_min, depth_max = layer_thresholds[i], layer_thresholds[i+1]
            layer_mask = (dist_transform >= depth_min) & (dist_transform < depth_max) & mask
            
            if np.sum(layer_mask) > 0:
                layer_composition = self.composition_map[layer_mask]
                whewellite_count = np.sum(layer_composition == 2)
                bacteria_count = np.sum(layer_composition == 1)
                total_count = len(layer_composition)
                
                contour_analysis['layer_depths'].append((depth_min + depth_max) / 2)
                contour_analysis['whewellite_fraction'].append(whewellite_count / total_count)
                contour_analysis['bacteria_fraction'].append(bacteria_count / total_count)
                contour_analysis['layer_area'].append(total_count)
        
        self.layer_analysis['contour'] = contour_analysis
        return contour_analysis
    
    def gradient_layer_analysis(self):
        """Analyze compositional gradients"""
        # Calculate local composition gradients
        whewellite_mask = (self.composition_map == 2).astype(float)
        
        # Apply Gaussian smoothing for gradient calculation
        smoothed = filters.gaussian(whewellite_mask, sigma=2)
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(smoothed)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Identify high-gradient regions (transition zones) within stone only
        threshold = np.percentile(gradient_magnitude[self.stone_mask], 80)
        transition_zones = (gradient_magnitude > threshold) & self.stone_mask
        
        gradient_analysis = {
            'gradient_magnitude': gradient_magnitude,
            'transition_zones': transition_zones,
            'mean_gradient': np.mean(gradient_magnitude[self.stone_mask]),
            'transition_area_fraction': np.sum(transition_zones) / np.sum(self.stone_mask)
        }
        
        self.layer_analysis['gradient'] = gradient_analysis
        return gradient_analysis
    
    def visualize_layer_analysis(self):
        """Comprehensive visualization of layer analysis"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Original CT image
        axes[0, 0].imshow(self.ct_analyzer.ct_image, cmap='gray')
        axes[0, 0].set_title('Original CT Image')
        axes[0, 0].axis('off')
        
        # Stone boundary mask
        if hasattr(self, 'stone_mask'):
            axes[0, 1].imshow(self.stone_mask, cmap='gray')
            axes[0, 1].set_title('Stone Segmentation\n(White=Stone, Black=Background)')
            axes[0, 1].axis('off')
        
        # Composition map
        if self.composition_map is not None:
            colors = ['black', 'red', 'gold']  # background, bacteria, whewellite
            cmap = ListedColormap(colors)
            im = axes[0, 2].imshow(self.composition_map, cmap=cmap, vmin=0, vmax=2)
            axes[0, 2].set_title('Composition Map\n(Black=Background, Red=Bacteria, Gold=Whewellite)')
            axes[0, 2].axis('off')
            
            # Add colorbar
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='black', label='Background'),
                             Patch(facecolor='red', label='Bacteria-dominant'),
                             Patch(facecolor='gold', label='Whewellite-dominant')]
            axes[0, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Radial analysis plot
        if 'radial' in self.layer_analysis:
            radial = self.layer_analysis['radial']
            axes[1, 0].plot(radial['distances'], radial['whewellite_fraction'], 'o-', 
                          color='gold', linewidth=2, label='Whewellite')
            axes[1, 0].plot(radial['distances'], radial['bacteria_fraction'], 'o-', 
                          color='red', linewidth=2, label='Bacteria')
            axes[1, 0].set_xlabel('Distance from Center (pixels)')
            axes[1, 0].set_ylabel('Composition Fraction')
            axes[1, 0].set_title('Radial Composition Profile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Contour analysis plot
        if 'contour' in self.layer_analysis:
            contour = self.layer_analysis['contour']
            axes[1, 1].plot(contour['layer_depths'], contour['whewellite_fraction'], 's-', 
                          color='gold', linewidth=2, label='Whewellite')
            axes[1, 1].plot(contour['layer_depths'], contour['bacteria_fraction'], 's-', 
                          color='red', linewidth=2, label='Bacteria')
            axes[1, 1].set_xlabel('Depth from Surface (pixels)')
            axes[1, 1].set_ylabel('Composition Fraction')
            axes[1, 1].set_title('Layer Depth Profile')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Gradient analysis
        if 'gradient' in self.layer_analysis:
            gradient = self.layer_analysis['gradient']
            im_grad = axes[1, 2].imshow(gradient['gradient_magnitude'], cmap='hot')
            axes[1, 2].set_title('Compositional Gradients\n(Transition Zones)')
            axes[1, 2].axis('off')
            plt.colorbar(im_grad, ax=axes[1, 2], label='Gradient Magnitude')
        
        # Composition statistics
        if self.composition_map is not None:
            whewellite_pixels = np.sum(self.composition_map == 2)
            bacteria_pixels = np.sum(self.composition_map == 1)
            total_stone_pixels = whewellite_pixels + bacteria_pixels  # Only stone pixels
            
            if total_stone_pixels > 0:
                fractions = [bacteria_pixels/total_stone_pixels, whewellite_pixels/total_stone_pixels]
                labels = ['Bacteria-dominant', 'Whewellite-dominant']
                colors = ['red', 'gold']
                
                axes[2, 0].pie(fractions, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[2, 0].set_title('Stone Composition\n(Background Excluded)')
        
        # Layer thickness analysis
        if 'contour' in self.layer_analysis:
            contour = self.layer_analysis['contour']
            axes[2, 1].bar(range(len(contour['layer_depths'])), 
                         [w + b for w, b in zip(contour['whewellite_fraction'], contour['bacteria_fraction'])],
                         color='lightgray', alpha=0.5, label='Total')
            axes[2, 1].bar(range(len(contour['layer_depths'])), contour['whewellite_fraction'], 
                         color='gold', alpha=0.8, label='Whewellite')
            axes[2, 1].bar(range(len(contour['layer_depths'])), contour['bacteria_fraction'], 
                         bottom=contour['whewellite_fraction'], color='red', alpha=0.8, label='Bacteria')
            axes[2, 1].set_xlabel('Layer (Surface to Core)')
            axes[2, 1].set_ylabel('Composition Fraction')
            axes[2, 1].set_title('Layer-by-Layer Composition')
            axes[2, 1].legend()
        
        # Summary statistics text
        axes[2, 2].axis('off')
        if hasattr(self, 'calibration_params') and self.composition_map is not None:
            stats_text = f"""Stone Composition Analysis Summary:

Calibration (from Raman):
• Whewellite density: {self.calibration_params['whewellite_density_mean']:.3f} ± {self.calibration_params['whewellite_density_std']:.3f}
• Bacteria density: {self.calibration_params['bacteria_density_mean']:.3f} ± {self.calibration_params['bacteria_density_std']:.3f}

Overall Composition:
• Whewellite-dominant: {whewellite_pixels/total_stone_pixels*100:.1f}%
• Bacteria-dominant: {bacteria_pixels/total_stone_pixels*100:.1f}%

Total analyzed pixels: {total_stone_pixels:,}
"""
            axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('stone_layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_layer_report(self):
        """Generate comprehensive layer analysis report"""
        report = []
        report.append("Kidney Stone Layer Composition Analysis")
        report.append("=" * 50)
        
        # Calibration summary
        if hasattr(self, 'calibration_params'):
            report.append("\nRaman Calibration Parameters:")
            report.append("-" * 30)
            report.append(f"Whewellite density mean: {self.calibration_params['whewellite_density_mean']:.4f}")
            report.append(f"Whewellite density std:  {self.calibration_params['whewellite_density_std']:.4f}")
            report.append(f"Bacteria density mean:   {self.calibration_params['bacteria_density_mean']:.4f}")
            report.append(f"Bacteria density std:    {self.calibration_params['bacteria_density_std']:.4f}")
            report.append(f"Classification threshold: {self.calibration_params['threshold']:.4f}")
        
        # Overall composition
        if self.composition_map is not None:
            whewellite_pixels = np.sum(self.composition_map == 2)
            bacteria_pixels = np.sum(self.composition_map == 1)
            total_stone_pixels = whewellite_pixels + bacteria_pixels  # Only stone pixels
            
            report.append("\nOverall Stone Composition:")
            report.append("-" * 30)
            report.append(f"Whewellite-dominant regions: {whewellite_pixels/total_stone_pixels*100:.2f}% ({whewellite_pixels:,} pixels)")
            report.append(f"Bacteria-dominant regions:   {bacteria_pixels/total_stone_pixels*100:.2f}% ({bacteria_pixels:,} pixels)")
            report.append(f"Total analyzed area:         {total_stone_pixels:,} pixels")
        
        # Radial analysis
        if 'radial' in self.layer_analysis:
            radial = self.layer_analysis['radial']
            report.append("\nRadial Composition Analysis:")
            report.append("-" * 30)
            # Find where composition changes most dramatically
            whew_fractions = np.array(radial['whewellite_fraction'])
            max_change_idx = np.argmax(np.abs(np.diff(whew_fractions)))
            
            report.append(f"Core composition (center):     {whew_fractions[0]*100:.1f}% whewellite")
            report.append(f"Surface composition (edge):    {whew_fractions[-1]*100:.1f}% whewellite")
            report.append(f"Maximum change at distance:    {radial['distances'][max_change_idx]:.1f} pixels")
        
        # Save report
        with open('stone_layer_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report

    def segment_stone_from_background(self):
        """Segment the actual stone from air/background"""
        if self.ct_analyzer.ct_image is None:
            self.ct_analyzer.load_ct_image()
        
        # Normalize CT image
        ct_norm = (self.ct_analyzer.ct_image - self.ct_analyzer.ct_image.min()) / \
                  (self.ct_analyzer.ct_image.max() - self.ct_analyzer.ct_image.min())
        
        # Use multiple methods to identify stone boundary
        
        # Method 1: Otsu thresholding to separate stone from air
        thresh_otsu = filters.threshold_otsu(ct_norm)
        stone_mask_otsu = ct_norm > thresh_otsu
        
        # Method 2: Use a more conservative threshold (air is much darker than stone)
        # Air should be very low density, stone material should be higher
        air_threshold = np.percentile(ct_norm.flatten(), 20)  # Bottom 20% is likely air
        stone_mask_conservative = ct_norm > air_threshold
        
        # Method 3: Use morphological operations to clean up
        stone_mask = stone_mask_conservative
        
        # Fill holes and remove small objects
        stone_mask = morphology.remove_small_holes(stone_mask, area_threshold=1000)
        stone_mask = morphology.remove_small_objects(stone_mask, min_size=5000)
        
        # Get the largest connected component (should be the main stone)
        labeled_mask = measure.label(stone_mask)
        regions = measure.regionprops(labeled_mask)
        
        if regions:
            # Find largest region
            largest_region = max(regions, key=lambda r: r.area)
            stone_mask = labeled_mask == largest_region.label
        
        # Apply morphological closing to smooth boundaries
        kernel = morphology.disk(3)
        stone_mask = morphology.binary_closing(stone_mask, kernel)
        
        # Store the stone mask
        self.stone_mask = stone_mask
        
        print(f"Stone segmentation complete:")
        print(f"Stone area: {np.sum(stone_mask):,} pixels")
        print(f"Background area: {np.sum(~stone_mask):,} pixels")
        print(f"Stone fraction of image: {np.sum(stone_mask)/stone_mask.size*100:.1f}%")
        
        return stone_mask

def main():
    # Import the CT analyzer
    from ct_enhancement import CTImageAnalyzer
    
    # Initialize CT analyzer
    print("Initializing CT analyzer...")
    ct_analyzer = CTImageAnalyzer('slice_1092.tif', 'DensityMeasure.png')
    ct_analyzer.load_ct_image()
    
    # Initialize layer analyzer
    print("Initializing layer analyzer...")
    layer_analyzer = StoneLayerAnalyzer(ct_analyzer, 'DensityMeasure.png')
    
    # Extract Raman calibration
    print("Extracting Raman density signatures...")
    calibration = layer_analyzer.extract_raman_density_signatures()
    
    # Apply calibration to full CT image
    print("Applying Raman calibration to CT image...")
    composition_map = layer_analyzer.apply_raman_calibration_to_ct()
    
    # Analyze layers
    print("Analyzing stone layers...")
    layer_analyzer.analyze_stone_layers('radial')
    layer_analyzer.analyze_stone_layers('contour') 
    layer_analyzer.analyze_stone_layers('gradient')
    
    # Visualize results
    print("Generating visualizations...")
    layer_analyzer.visualize_layer_analysis()
    
    # Generate report
    print("Generating layer analysis report...")
    layer_analyzer.generate_layer_report()
    
    print("\nLayer analysis complete!")

if __name__ == "__main__":
    main() 