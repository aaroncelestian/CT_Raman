import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import filters, morphology, segmentation, measure
from scipy import ndimage
from sklearn.mixture import GaussianMixture
import seaborn as sns
from matplotlib.colors import ListedColormap
import pickle
import os

class EnhancedStoneAnalyzer:
    def __init__(self, ct_image_path, raman_reference_path=None, annotation_file=None):
        """
        Enhanced Stone Analyzer with automatic air removal and composition analysis
        
        Args:
            ct_image_path: Path to the CT image
            raman_reference_path: Path to Raman calibration data (optional)
            annotation_file: Path to annotation file for air removal thresholds
        """
        self.ct_image_path = ct_image_path
        self.raman_reference_path = raman_reference_path
        self.annotation_file = annotation_file
        self.ct_image = None
        self.stone_mask = None
        self.composition_map = None
        self.air_threshold = None
        self.composition_threshold = None
        self.analysis_results = {}
        
    def load_ct_image(self):
        """Load the CT image"""
        if self.ct_image_path.endswith('.tif'):
            self.ct_image = np.array(Image.open(self.ct_image_path))
        else:
            self.ct_image = cv2.imread(self.ct_image_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"CT Image loaded: Shape {self.ct_image.shape}, dtype: {self.ct_image.dtype}")
        print(f"Intensity range: {self.ct_image.min()} - {self.ct_image.max()}")
        return self.ct_image
    
    def load_annotation_thresholds(self):
        """Load thresholds from annotation file or calculate defaults"""
        # Try to load existing annotation file
        annotation_files = ['stone_annotations.pkl', 'annotation_based_report.txt']
        
        if self.annotation_file and os.path.exists(self.annotation_file):
            # Use specified annotation file
            if self.annotation_file.endswith('.pkl'):
                self._load_thresholds_from_pickle(self.annotation_file)
            else:
                self._load_thresholds_from_report(self.annotation_file)
        else:
            # Look for existing annotation files
            found_file = None
            for file in annotation_files:
                if os.path.exists(file):
                    found_file = file
                    break
            
            if found_file:
                if found_file.endswith('.pkl'):
                    self._load_thresholds_from_pickle(found_file)
                else:
                    self._load_thresholds_from_report(found_file)
            else:
                # Use default thresholds based on typical CT values
                self._calculate_default_thresholds()
        
        print(f"Air removal threshold: {self.air_threshold}")
        if self.composition_threshold:
            print(f"Composition threshold: {self.composition_threshold}")
    
    def _load_thresholds_from_pickle(self, pickle_file):
        """Load thresholds from pickle annotation file"""
        try:
            with open(pickle_file, 'rb') as f:
                annotations = pickle.load(f)
            
            # Calculate thresholds from annotations
            air_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'air']
            bacteria_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'bacteria']
            whewellite_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'whewellite']
            
            if air_intensities and (bacteria_intensities or whewellite_intensities):
                air_max = np.mean(air_intensities) + 2 * np.std(air_intensities)
                stone_min = min(np.mean(bacteria_intensities) if bacteria_intensities else float('inf'),
                               np.mean(whewellite_intensities) if whewellite_intensities else float('inf'))
                stone_min -= 2 * (np.std(bacteria_intensities) if bacteria_intensities else 0)
                
                self.air_threshold = (air_max + stone_min) / 2
                
                # Calculate composition threshold if both bacteria and whewellite present
                if bacteria_intensities and whewellite_intensities:
                    bacteria_max = np.mean(bacteria_intensities) + np.std(bacteria_intensities)
                    whewellite_min = np.mean(whewellite_intensities) - np.std(whewellite_intensities)
                    self.composition_threshold = (bacteria_max + whewellite_min) / 2
                    
                print(f"Loaded thresholds from annotations: {len(annotations)} points")
            else:
                self._calculate_default_thresholds()
                
        except Exception as e:
            print(f"Error loading annotation file: {e}")
            self._calculate_default_thresholds()
    
    def _load_thresholds_from_report(self, report_file):
        """Load thresholds from text report file"""
        try:
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Parse thresholds from report
            if 'air_stone:' in content:
                line = [l for l in content.split('\n') if 'air_stone:' in l][0]
                self.air_threshold = float(line.split(':')[1].strip())
            
            if 'bacteria_whewellite:' in content:
                line = [l for l in content.split('\n') if 'bacteria_whewellite:' in l][0]
                self.composition_threshold = float(line.split(':')[1].strip())
                
            print(f"Loaded thresholds from report file")
            
        except Exception as e:
            print(f"Error loading report file: {e}")
            self._calculate_default_thresholds()
    
    def _calculate_default_thresholds(self):
        """Calculate default thresholds using image statistics"""
        if self.ct_image is None:
            self.load_ct_image()
        
        # Normalize image
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Use percentile-based thresholds
        # Air is typically the lowest ~20% of intensities
        self.air_threshold = np.percentile(self.ct_image.flatten(), 25)
        
        # For composition, use Otsu on stone pixels
        stone_pixels = self.ct_image[self.ct_image > self.air_threshold]
        if len(stone_pixels) > 0:
            stone_norm = (stone_pixels - stone_pixels.min()) / (stone_pixels.max() - stone_pixels.min())
            self.composition_threshold = filters.threshold_otsu(stone_norm) * (stone_pixels.max() - stone_pixels.min()) + stone_pixels.min()
        
        print("Using default thresholds based on image statistics")
    
    def remove_air_preprocessing(self):
        """Preprocess image to remove air and create stone mask"""
        if self.ct_image is None:
            self.load_ct_image()
        
        if self.air_threshold is None:
            self.load_annotation_thresholds()
        
        # Create initial stone mask (non-air regions)
        stone_mask = self.ct_image > self.air_threshold
        initial_stone_pixels = np.sum(stone_mask)
        initial_air_pixels = np.sum(~stone_mask)
        
        print(f"Initial threshold result:")
        print(f"  Stone pixels: {initial_stone_pixels:,}")
        print(f"  Air pixels: {initial_air_pixels:,} ({initial_air_pixels/stone_mask.size*100:.1f}%)")
        
        # Clean up the mask using morphological operations (but preserve small air regions)
        # Remove small stone objects (noise) but keep air
        stone_mask_cleaned = morphology.remove_small_objects(stone_mask, min_size=500)  # Reduced from 1000
        
        after_small_objects = np.sum(stone_mask_cleaned)
        print(f"After removing small stone objects:")
        print(f"  Stone pixels: {after_small_objects:,}")
        print(f"  Air pixels: {np.sum(~stone_mask_cleaned):,}")
        
        # Fill small holes in stone (but don't fill large air regions)
        stone_mask_filled = morphology.remove_small_holes(stone_mask_cleaned, area_threshold=25)  # Much smaller - only fill tiny holes
        
        after_fill_holes = np.sum(stone_mask_filled)
        print(f"After filling small holes:")
        print(f"  Stone pixels: {after_fill_holes:,}")
        print(f"  Air pixels: {np.sum(~stone_mask_filled):,}")
        
        # Get largest connected component (main stone) - this might be removing air regions!
        labeled_mask = measure.label(stone_mask_filled)
        if labeled_mask.max() > 0:
            regions = measure.regionprops(labeled_mask)
            # Instead of taking only the largest, take all regions above a certain size
            min_region_size = stone_mask.size * 0.001  # Keep regions > 0.1% of image
            
            final_mask = np.zeros_like(stone_mask, dtype=bool)
            for region in regions:
                if region.area > min_region_size:
                    final_mask[labeled_mask == region.label] = True
            
            stone_mask = final_mask
        else:
            stone_mask = stone_mask_filled
        
        after_components = np.sum(stone_mask)
        print(f"After connected component analysis:")
        print(f"  Stone pixels: {after_components:,}")
        print(f"  Air pixels: {np.sum(~stone_mask):,}")
        
        # Apply light morphological closing to smooth boundaries (smaller kernel)
        kernel = morphology.disk(2)  # Reduced from 3
        stone_mask = morphology.binary_closing(stone_mask, kernel)
        
        final_stone_pixels = np.sum(stone_mask)
        final_air_pixels = np.sum(~stone_mask)
        
        print(f"After morphological closing:")
        print(f"  Stone pixels: {final_stone_pixels:,}")
        print(f"  Air pixels: {final_air_pixels:,}")
        
        self.stone_mask = stone_mask
        
        # Calculate statistics
        stone_area = np.sum(stone_mask)
        total_area = stone_mask.size
        air_area = total_area - stone_area
        
        print(f"\nAir removal preprocessing complete:")
        print(f"Stone area: {stone_area:,} pixels ({stone_area/total_area*100:.1f}%)")
        print(f"Air area: {air_area:,} pixels ({air_area/total_area*100:.1f}%)")
        print(f"Air threshold used: {self.air_threshold:.1f}")
        
        return stone_mask
    
    def analyze_stone_composition(self):
        """Analyze composition within the stone (non-air) regions"""
        if self.stone_mask is None:
            self.remove_air_preprocessing()
        
        # Create composition map
        composition_map = np.zeros_like(self.ct_image, dtype=np.uint8)
        
        if self.composition_threshold is not None:
            # Use annotation-based threshold
            whewellite_mask = (self.ct_image > self.composition_threshold) & self.stone_mask
            bacteria_mask = (self.ct_image <= self.composition_threshold) & self.stone_mask
            
            composition_map[bacteria_mask] = 1    # Bacteria-dominant
            composition_map[whewellite_mask] = 2  # Whewellite-dominant
            
            print(f"Composition threshold used: {self.composition_threshold:.1f}")
            
        else:
            # Use Raman calibration if available
            if self.raman_reference_path and os.path.exists(self.raman_reference_path):
                composition_map = self._analyze_with_raman_calibration()
            else:
                # Use adaptive thresholding on stone pixels only
                stone_pixels = self.ct_image[self.stone_mask]
                if len(stone_pixels) > 0:
                    adaptive_thresh = filters.threshold_otsu(stone_pixels)
                    
                    whewellite_mask = (self.ct_image > adaptive_thresh) & self.stone_mask
                    bacteria_mask = (self.ct_image <= adaptive_thresh) & self.stone_mask
                    
                    composition_map[bacteria_mask] = 1
                    composition_map[whewellite_mask] = 2
                    
                    print(f"Adaptive threshold used: {adaptive_thresh:.1f}")
        
        # Clean up composition map
        composition_map = self._clean_composition_map(composition_map)
        
        self.composition_map = composition_map
        
        # Calculate composition statistics
        whewellite_pixels = np.sum(composition_map == 2)
        bacteria_pixels = np.sum(composition_map == 1)
        stone_pixels = whewellite_pixels + bacteria_pixels
        
        if stone_pixels > 0:
            self.analysis_results = {
                'whewellite_pixels': whewellite_pixels,
                'bacteria_pixels': bacteria_pixels,
                'stone_pixels': stone_pixels,
                'whewellite_percentage': whewellite_pixels / stone_pixels * 100,
                'bacteria_percentage': bacteria_pixels / stone_pixels * 100,
                'air_pixels': np.sum(~self.stone_mask),
                'total_pixels': self.stone_mask.size
            }
            
            print(f"\nStone Composition Analysis:")
            print(f"Whewellite-dominant: {self.analysis_results['whewellite_percentage']:.1f}% ({whewellite_pixels:,} pixels)")
            print(f"Bacteria-dominant: {self.analysis_results['bacteria_percentage']:.1f}% ({bacteria_pixels:,} pixels)")
            print(f"Total stone area: {stone_pixels:,} pixels")
        
        return composition_map
    
    def _analyze_with_raman_calibration(self):
        """Analyze composition using Raman calibration data"""
        # Load Raman data
        raman_img = plt.imread(self.raman_reference_path)
        
        if len(raman_img.shape) == 3:
            raman_gray = np.dot(raman_img[...,:3], [0.299, 0.587, 0.114])
        else:
            raman_gray = raman_img
        
        # Normalize Raman data
        raman_norm = (raman_gray - raman_gray.min()) / (raman_gray.max() - raman_gray.min())
        
        # Use GMM to find density signatures
        data = raman_norm.flatten().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)
        
        means = gmm.means_.flatten()
        threshold = (means[0] + means[1]) / 2
        
        # Apply to CT data (normalized)
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Map threshold to CT intensity space
        stone_pixels = ct_norm[self.stone_mask]
        ct_threshold = np.percentile(stone_pixels, threshold * 100)
        
        # Create composition map
        composition_map = np.zeros_like(self.ct_image, dtype=np.uint8)
        whewellite_mask = (ct_norm > ct_threshold) & self.stone_mask
        bacteria_mask = (ct_norm <= ct_threshold) & self.stone_mask
        
        composition_map[bacteria_mask] = 1
        composition_map[whewellite_mask] = 2
        
        print(f"Raman-calibrated threshold: {ct_threshold:.3f} (normalized)")
        
        return composition_map
    
    def _clean_composition_map(self, composition_map):
        """Clean up composition map using morphological operations"""
        cleaned_map = np.copy(composition_map)
        
        for label in [1, 2]:  # bacteria, whewellite
            mask = composition_map == label
            # Remove small objects
            mask_cleaned = morphology.remove_small_objects(mask, min_size=50)
            # Fill small holes
            mask_cleaned = morphology.remove_small_holes(mask_cleaned, area_threshold=25)
            
            # Update cleaned map
            cleaned_map[composition_map == label] = 0
            cleaned_map[mask_cleaned] = label
        
        return cleaned_map
    
    def create_radial_profile(self):
        """Create radial composition profile from stone center"""
        if self.composition_map is None:
            self.analyze_stone_composition()
        
        # Find stone center
        coords = np.where(self.stone_mask)
        center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
        
        # Create distance map
        y, x = np.ogrid[:self.composition_map.shape[0], :self.composition_map.shape[1]]
        distance_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Analyze composition vs distance
        max_distance = np.max(distance_map[self.stone_mask])
        radial_bins = np.linspace(0, max_distance, 20)
        
        distances = []
        whewellite_fractions = []
        bacteria_fractions = []
        
        for i in range(len(radial_bins)-1):
            r_min, r_max = radial_bins[i], radial_bins[i+1]
            ring_mask = (distance_map >= r_min) & (distance_map < r_max) & self.stone_mask
            
            if np.sum(ring_mask) > 0:
                ring_composition = self.composition_map[ring_mask]
                whewellite_count = np.sum(ring_composition == 2)
                bacteria_count = np.sum(ring_composition == 1)
                total_count = whewellite_count + bacteria_count
                
                if total_count > 0:
                    distances.append((r_min + r_max) / 2)
                    whewellite_fractions.append(whewellite_count / total_count)
                    bacteria_fractions.append(bacteria_count / total_count)
        
        return distances, whewellite_fractions, bacteria_fractions
    
    def visualize_results(self, save_path=None):
        """Comprehensive visualization of analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original CT image
        axes[0, 0].imshow(self.ct_image, cmap='gray')
        axes[0, 0].set_title('Original CT Image')
        axes[0, 0].axis('off')
        
        # Air removal result
        if self.stone_mask is not None:
            # Show preprocessed image (stone only)
            preprocessed = np.copy(self.ct_image).astype(float)
            preprocessed[~self.stone_mask] = np.nan
            
            axes[0, 1].imshow(preprocessed, cmap='gray')
            axes[0, 1].set_title(f'Preprocessed (Air Removed)\nThreshold: {self.air_threshold:.0f}')
            axes[0, 1].axis('off')
        
        # Composition map
        if self.composition_map is not None:
            colors = ['black', 'red', 'gold']  # background, bacteria, whewellite
            cmap = ListedColormap(colors)
            
            axes[0, 2].imshow(self.composition_map, cmap=cmap, vmin=0, vmax=2)
            axes[0, 2].set_title('Stone Composition\n(Red=Bacteria, Gold=Whewellite)')
            axes[0, 2].axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='Bacteria-dominant'),
                             Patch(facecolor='gold', label='Whewellite-dominant')]
            axes[0, 2].legend(handles=legend_elements, loc='upper right')
        
        # Radial profile
        try:
            distances, whew_fractions, bact_fractions = self.create_radial_profile()
            axes[1, 0].plot(distances, whew_fractions, 'o-', color='gold', linewidth=2, label='Whewellite')
            axes[1, 0].plot(distances, bact_fractions, 'o-', color='red', linewidth=2, label='Bacteria')
            axes[1, 0].set_xlabel('Distance from Center (pixels)')
            axes[1, 0].set_ylabel('Composition Fraction')
            axes[1, 0].set_title('Radial Composition Profile')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].text(0.5, 0.5, 'Radial profile\nnot available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Radial Composition Profile')
        
        # Composition statistics
        if hasattr(self, 'analysis_results') and self.analysis_results:
            results = self.analysis_results
            fractions = [results['bacteria_percentage'], results['whewellite_percentage']]
            labels = ['Bacteria-dominant', 'Whewellite-dominant']
            colors_pie = ['red', 'gold']
            
            axes[1, 1].pie(fractions, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Stone Composition Distribution')
        
        # Analysis summary
        axes[1, 2].axis('off')
        if hasattr(self, 'analysis_results') and self.analysis_results:
            summary_text = f"""Analysis Summary:

Air Removal:
• Air threshold: {self.air_threshold:.0f}
• Air removed: {self.analysis_results['air_pixels']:,} pixels

Stone Composition:
• Whewellite: {self.analysis_results['whewellite_percentage']:.1f}%
• Bacteria: {self.analysis_results['bacteria_percentage']:.1f}%
• Stone area: {self.analysis_results['stone_pixels']:,} pixels

Method: {'Annotation-based' if self.composition_threshold else 'Adaptive'}
"""
            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = []
        report.append("Enhanced Stone Composition Analysis Report")
        report.append("=" * 50)
        
        # Preprocessing summary
        report.append(f"\nPreprocessing (Air Removal):")
        report.append("-" * 30)
        report.append(f"Air threshold used: {self.air_threshold:.1f}")
        
        if hasattr(self, 'analysis_results'):
            results = self.analysis_results
            report.append(f"Air pixels removed: {results['air_pixels']:,}")
            report.append(f"Stone pixels retained: {results['stone_pixels']:,}")
            report.append(f"Air removal efficiency: {results['air_pixels']/results['total_pixels']*100:.1f}%")
        
        # Composition analysis
        if hasattr(self, 'analysis_results'):
            report.append(f"\nStone Composition Analysis:")
            report.append("-" * 30)
            report.append(f"Whewellite-dominant regions: {results['whewellite_percentage']:.2f}% ({results['whewellite_pixels']:,} pixels)")
            report.append(f"Bacteria-dominant regions: {results['bacteria_percentage']:.2f}% ({results['bacteria_pixels']:,} pixels)")
            report.append(f"Total stone area analyzed: {results['stone_pixels']:,} pixels")
            
            if self.composition_threshold:
                report.append(f"Composition threshold: {self.composition_threshold:.1f} (annotation-based)")
            else:
                report.append("Composition threshold: Adaptive (image-based)")
        
        # Method summary
        report.append(f"\nMethodology:")
        report.append("-" * 15)
        report.append("1. Automatic air removal using intensity thresholding")
        report.append("2. Stone boundary identification and cleanup")
        report.append("3. Compositional analysis within stone regions only")
        if self.composition_threshold:
            report.append("4. Annotation-based calibration for whewellite/bacteria classification")
        else:
            report.append("4. Adaptive thresholding for whewellite/bacteria classification")
        
        # Save report
        with open('enhanced_stone_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Enhanced Stone Analysis Pipeline...")
        print("=" * 50)
        
        # Step 1: Load image
        self.load_ct_image()
        
        # Step 2: Load thresholds
        self.load_annotation_thresholds()
        
        # Step 3: Remove air (preprocessing)
        print("\nStep 1: Removing air/background...")
        self.remove_air_preprocessing()
        
        # Step 4: Analyze composition
        print("\nStep 2: Analyzing stone composition...")
        self.analyze_stone_composition()
        
        # Step 5: Visualize results
        print("\nStep 3: Generating visualizations...")
        self.visualize_results('enhanced_stone_analysis.png')
        
        # Step 6: Generate report
        print("\nStep 4: Generating report...")
        self.generate_report()
        
        print("\n" + "=" * 50)
        print("Enhanced Stone Analysis Complete!")
        
        return self.analysis_results


def main():
    """Main function to run enhanced stone analysis"""
    # Initialize analyzer
    analyzer = EnhancedStoneAnalyzer(
        ct_image_path='slice_1092.tif',
        raman_reference_path='DensityMeasure.png'
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main() 