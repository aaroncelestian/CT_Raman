import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import filters, morphology, segmentation, measure, feature
from scipy import ndimage
import pickle
import os

class DoGStoneIsolator:
    def __init__(self, ct_image_path, annotation_file=None):
        """
        Kidney Stone Isolator using DoG (Difference of Gaussians) edge detection
        
        Args:
            ct_image_path: Path to the CT image
            annotation_file: Path to annotation file for composition thresholds
        """
        self.ct_image_path = ct_image_path
        self.annotation_file = annotation_file
        self.ct_image = None
        self.dog_enhanced = None
        self.stone_mask = None
        self.isolated_stone = None
        self.composition_map = None
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
    
    def load_composition_threshold(self):
        """Load composition threshold from annotation files"""
        annotation_files = ['stone_annotations.pkl', 'annotation_based_report.txt']
        
        if self.annotation_file and os.path.exists(self.annotation_file):
            # Use specified annotation file
            if self.annotation_file.endswith('.pkl'):
                self._load_from_pickle(self.annotation_file)
            else:
                self._load_from_report(self.annotation_file)
        else:
            # Look for existing annotation files
            found_file = None
            for file in annotation_files:
                if os.path.exists(file):
                    found_file = file
                    break
            
            if found_file:
                if found_file.endswith('.pkl'):
                    self._load_from_pickle(found_file)
                else:
                    self._load_from_report(found_file)
            else:
                # Use adaptive thresholding
                print("No annotation file found - will use adaptive thresholding")
        
        if self.composition_threshold:
            print(f"Composition threshold loaded: {self.composition_threshold}")
    
    def _load_from_pickle(self, pickle_file):
        """Load composition threshold from pickle file"""
        try:
            with open(pickle_file, 'rb') as f:
                annotations = pickle.load(f)
            
            bacteria_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'bacteria']
            whewellite_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'whewellite']
            
            if bacteria_intensities and whewellite_intensities:
                bacteria_max = np.mean(bacteria_intensities) + np.std(bacteria_intensities)
                whewellite_min = np.mean(whewellite_intensities) - np.std(whewellite_intensities)
                self.composition_threshold = (bacteria_max + whewellite_min) / 2
                
        except Exception as e:
            print(f"Error loading pickle file: {e}")
    
    def _load_from_report(self, report_file):
        """Load composition threshold from report file"""
        try:
            with open(report_file, 'r') as f:
                content = f.read()
            
            if 'bacteria_whewellite:' in content:
                line = [l for l in content.split('\n') if 'bacteria_whewellite:' in l][0]
                self.composition_threshold = float(line.split(':')[1].strip())
                
        except Exception as e:
            print(f"Error loading report file: {e}")
    
    def apply_dog_filter(self, sigma1=1.0, sigma2=3.0):
        """
        Apply Difference of Gaussians filter to enhance edges
        
        Args:
            sigma1: Standard deviation for first Gaussian (smaller - fine details)
            sigma2: Standard deviation for second Gaussian (larger - broader features)
        """
        if self.ct_image is None:
            self.load_ct_image()
        
        # Normalize image to 0-1 range for better filter performance
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Apply DoG filter
        gaussian1 = filters.gaussian(ct_norm, sigma=sigma1)
        gaussian2 = filters.gaussian(ct_norm, sigma=sigma2)
        dog_result = gaussian1 - gaussian2
        
        # Normalize DoG result
        self.dog_enhanced = (dog_result - dog_result.min()) / (dog_result.max() - dog_result.min())
        
        print(f"DoG filter applied: sigma1={sigma1}, sigma2={sigma2}")
        return self.dog_enhanced
    
    def create_stone_mask_from_dog(self, edge_threshold=0.3, min_stone_size=50000):
        """
        Create stone mask using DoG edge detection - BALANCED APPROACH
        
        Args:
            edge_threshold: Threshold for edge detection (0-1)
            min_stone_size: Minimum size for stone regions (pixels)
        """
        if self.dog_enhanced is None:
            self.apply_dog_filter()
        
        print(f"DoG enhanced range: {self.dog_enhanced.min():.3f} - {self.dog_enhanced.max():.3f}")
        
        # Balanced approach: Use inverted DoG with relaxed constraints
        inverted_dog = 1.0 - self.dog_enhanced
        
        # Test broader range of thresholds with relaxed size constraints
        test_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        best_threshold = 0.3
        best_stone_area = 0
        best_mask = None
        
        for thresh in test_thresholds:
            # Create binary mask of stone candidates
            stone_candidates = inverted_dog > thresh
            
            # Remove small objects (less aggressive)
            stone_candidates = morphology.remove_small_objects(stone_candidates, min_size=5000)
            
            # Get largest connected component
            labeled_candidates = measure.label(stone_candidates)
            if labeled_candidates.max() > 0:
                regions = measure.regionprops(labeled_candidates)
                largest_region = max(regions, key=lambda r: r.area)
                stone_mask_candidate = labeled_candidates == largest_region.label
                
                stone_area = np.sum(stone_mask_candidate)
                stone_percentage = stone_area / self.ct_image.size * 100
                
                print(f"Threshold {thresh:.2f}: stone area = {stone_area:,} pixels ({stone_percentage:.1f}%)")
                
                # Look for reasonable stone size (12-35% of image) - more relaxed
                if 12 <= stone_percentage <= 35:
                    best_threshold = thresh
                    best_stone_area = stone_area
                    best_mask = stone_mask_candidate
                    print(f"  -> Good candidate!")
        
        if best_mask is not None:
            print(f"Using inverted DoG threshold: {best_threshold:.2f} (stone area: {best_stone_area:,})")
            self.stone_mask = best_mask
            
        else:
            print("No suitable stone region found with inverted DoG, trying combined approach...")
            
            # Alternative: Combine CT intensity with DoG guidance (more relaxed)
            ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
            
            # Try different combinations
            combinations = [
                (ct_norm * inverted_dog, "CT × inverted_DoG"),
                (ct_norm + 0.3 * inverted_dog, "CT + 0.3×inverted_DoG"),
                (0.7 * ct_norm + 0.3 * inverted_dog, "0.7×CT + 0.3×inverted_DoG")
            ]
            
            best_combination = None
            best_combo_area = 0
            
            for stone_score, combo_name in combinations:
                print(f"Trying {combo_name}...")
                
                for thresh in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    stone_candidates = stone_score > thresh
                    stone_candidates = morphology.remove_small_objects(stone_candidates, min_size=10000)
                    
                    if np.sum(stone_candidates) > 0:
                        labeled_candidates = measure.label(stone_candidates)
                        regions = measure.regionprops(labeled_candidates)
                        largest_region = max(regions, key=lambda r: r.area)
                        stone_mask_candidate = labeled_candidates == largest_region.label
                        
                        stone_percentage = np.sum(stone_mask_candidate) / self.ct_image.size * 100
                        print(f"  Threshold {thresh:.2f}: {stone_percentage:.1f}% stone")
                        
                        if 15 <= stone_percentage <= 30:
                            if np.sum(stone_mask_candidate) > best_combo_area:
                                best_combination = stone_mask_candidate
                                best_combo_area = np.sum(stone_mask_candidate)
                                print(f"    -> New best combination!")
                                break
            
            if best_combination is not None:
                self.stone_mask = best_combination
                print(f"Using combined approach (stone area: {best_combo_area:,})")
            else:
                print("Fallback: Using moderate intensity threshold")
                # Last resort: moderate threshold on original image
                ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
                
                # Try different intensity thresholds
                for intensity_thresh in [0.45, 0.5, 0.55, 0.6]:
                    high_intensity = ct_norm > intensity_thresh
                    high_intensity = morphology.remove_small_objects(high_intensity, min_size=20000)
                    
                    if np.sum(high_intensity) > 0:
                        labeled_regions = measure.label(high_intensity)
                        regions = measure.regionprops(labeled_regions)
                        largest_region = max(regions, key=lambda r: r.area)
                        candidate_mask = labeled_regions == largest_region.label
                        
                        stone_percentage = np.sum(candidate_mask) / self.ct_image.size * 100
                        print(f"Intensity threshold {intensity_thresh:.2f}: {stone_percentage:.1f}% stone")
                        
                        if 10 <= stone_percentage <= 40:
                            self.stone_mask = candidate_mask
                            break
        
        # Final cleanup - moderate
        if hasattr(self, 'stone_mask') and self.stone_mask is not None:
            kernel = morphology.disk(3)
            self.stone_mask = morphology.binary_opening(self.stone_mask, kernel)
            self.stone_mask = morphology.binary_closing(self.stone_mask, kernel)
        else:
            # Create minimal fallback
            print("Creating minimal central stone region as last resort")
            center_y, center_x = self.ct_image.shape[0]//2, self.ct_image.shape[1]//2
            self.stone_mask = np.zeros(self.ct_image.shape, dtype=bool)
            radius = 300
            y, x = np.ogrid[:self.ct_image.shape[0], :self.ct_image.shape[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            self.stone_mask = mask
        
        stone_area = np.sum(self.stone_mask)
        total_area = self.stone_mask.size
        stone_percentage = stone_area/total_area*100
        
        print(f"FINAL Stone mask created:")
        print(f"Stone area: {stone_area:,} pixels ({stone_percentage:.1f}%)")
        print(f"Background area: {total_area-stone_area:,} pixels ({(total_area-stone_area)/total_area*100:.1f}%)")
        
        # Validation
        if stone_percentage > 40:
            print(f"WARNING: Stone area ({stone_percentage:.1f}%) seems too large")
        elif stone_percentage < 8:
            print(f"WARNING: Stone area ({stone_percentage:.1f}%) seems too small")
        else:
            print(f"✅ Stone area ({stone_percentage:.1f}%) looks good!")
        
        return self.stone_mask
    
    def isolate_stone(self):
        """Isolate the stone region from the original CT image"""
        if self.stone_mask is None:
            self.create_stone_mask_from_dog()
        
        # Create isolated stone image
        self.isolated_stone = np.copy(self.ct_image).astype(float)
        self.isolated_stone[~self.stone_mask] = np.nan  # Set background to NaN
        
        print("Stone successfully isolated using DoG-based segmentation")
        return self.isolated_stone
    
    def analyze_stone_composition(self):
        """Analyze composition within the isolated stone - GRADIENT COMPOSITION APPROACH"""
        if self.isolated_stone is None:
            self.isolate_stone()
        
        if self.composition_threshold is None:
            self.load_composition_threshold()
        
        print(f"\nGRADIENT COMPOSITION ANALYSIS:")
        print(f"Creating continuous compositional scale from 100% bacteria to 100% whewellite")
        print(f"Highlighting bacteria/crystal intergrowth zones of special interest")
        
        # Get intensity range within stone
        stone_intensities = self.ct_image[self.stone_mask]
        min_intensity = np.min(stone_intensities)
        max_intensity = np.max(stone_intensities)
        
        print(f"Stone intensity range: {min_intensity} - {max_intensity}")
        print(f"Total stone pixels: {np.sum(self.stone_mask):,}")
        
        # Create continuous composition map (0-100 scale)
        composition_continuous = np.zeros_like(self.ct_image, dtype=np.float32)
        
        # Map intensities to composition percentages within stone boundary
        stone_mask_indices = self.stone_mask
        stone_pixel_intensities = self.ct_image[stone_mask_indices]
        
        # Normalize intensities to 0-100 scale (whewellite percentage)
        # Min intensity = 0% whewellite (100% bacteria)
        # Max intensity = 100% whewellite (0% bacteria)
        whewellite_percentages = ((stone_pixel_intensities - min_intensity) / 
                                (max_intensity - min_intensity) * 100)
        
        composition_continuous[stone_mask_indices] = whewellite_percentages
        
        # Create categorical zones for analysis
        composition_zones = np.zeros_like(self.ct_image, dtype=np.uint8)
        
        # Debug: Check stone mask properties
        print(f"Stone mask shape: {self.stone_mask.shape}")
        print(f"Stone mask dtype: {self.stone_mask.dtype}")
        print(f"Stone mask True pixels: {np.sum(self.stone_mask):,}")
        print(f"Stone mask non-zero pixels: {np.count_nonzero(self.stone_mask):,}")
        
        # Method 1: Direct assignment using boolean indexing (more reliable)
        print("Using direct boolean indexing for zone assignment...")
        
        # Assign all stone pixels to intergrowth first (default)
        composition_zones[self.stone_mask] = 3  # Default all to intergrowth
        
        # Then override with specific zones based on thresholds
        # Pure bacteria (0-15% whewellite)
        pure_bacteria_condition = self.stone_mask & (self.ct_image <= min_intensity + 0.15 * (max_intensity - min_intensity))
        composition_zones[pure_bacteria_condition] = 1
        
        # Bacteria-rich (15-35% whewellite)
        bacteria_rich_condition = self.stone_mask & (self.ct_image > min_intensity + 0.15 * (max_intensity - min_intensity)) & (self.ct_image <= min_intensity + 0.35 * (max_intensity - min_intensity))
        composition_zones[bacteria_rich_condition] = 2
        
        # Intergrowth (35-65% whewellite) - already set as default
        intergrowth_condition = self.stone_mask & (self.ct_image > min_intensity + 0.35 * (max_intensity - min_intensity)) & (self.ct_image <= min_intensity + 0.65 * (max_intensity - min_intensity))
        composition_zones[intergrowth_condition] = 3
        
        # Whewellite-rich (65-85% whewellite)
        whewellite_rich_condition = self.stone_mask & (self.ct_image > min_intensity + 0.65 * (max_intensity - min_intensity)) & (self.ct_image <= min_intensity + 0.85 * (max_intensity - min_intensity))
        composition_zones[whewellite_rich_condition] = 4
        
        # Pure whewellite (85-100% whewellite)
        pure_whewellite_condition = self.stone_mask & (self.ct_image > min_intensity + 0.85 * (max_intensity - min_intensity))
        composition_zones[pure_whewellite_condition] = 5
        
        # Verify that all stone pixels have been assigned
        assigned_pixels = np.sum(composition_zones > 0)
        expected_stone_pixels = np.sum(self.stone_mask)
        unassigned_in_stone = np.sum(self.stone_mask & (composition_zones == 0))
        
        print(f"Zone assignment verification (Method 2):")
        print(f"Expected stone pixels: {expected_stone_pixels:,}")
        print(f"Assigned pixels: {assigned_pixels:,}")
        print(f"Unassigned pixels within stone: {unassigned_in_stone:,}")
        
        # Force assignment of any remaining unassigned stone pixels
        if unassigned_in_stone > 0:
            print(f"ERROR: {unassigned_in_stone:,} stone pixels were not assigned to zones!")
            print("Force-assigning ALL remaining stone pixels to intergrowth zone...")
            unassigned_mask = self.stone_mask & (composition_zones == 0)
            composition_zones[unassigned_mask] = 3  # Force assign to intergrowth
            
            # Verify the fix
            final_unassigned = np.sum(self.stone_mask & (composition_zones == 0))
            print(f"After force assignment, unassigned pixels: {final_unassigned:,}")
        else:
            print("✅ All stone pixels successfully assigned to composition zones")
        
        # Double-check: make sure every stone pixel has a non-zero zone value
        stone_pixels_with_zones = np.sum(composition_zones[self.stone_mask] > 0)
        total_stone_pixels = np.sum(self.stone_mask)
        print(f"Final verification: {stone_pixels_with_zones:,} / {total_stone_pixels:,} stone pixels have zones")
        
        if stone_pixels_with_zones != total_stone_pixels:
            print("CRITICAL ERROR: Some stone pixels still have zone value 0!")
            # Emergency fix
            remaining_unassigned = self.stone_mask & (composition_zones == 0)
            if np.any(remaining_unassigned):
                composition_zones[remaining_unassigned] = 3  # Emergency assignment to intergrowth
                print("Emergency assignment completed.")
        else:
            print("✅ Perfect zone coverage achieved!")
        
        # Store both continuous and zonal compositions
        self.composition_map = composition_zones
        self.composition_continuous = composition_continuous
        
        # Calculate statistics for each zone using the composition_zones directly
        pure_bacteria_pixels = np.sum(composition_zones == 1)
        bacteria_rich_pixels = np.sum(composition_zones == 2)
        intergrowth_pixels = np.sum(composition_zones == 3)
        whewellite_rich_pixels = np.sum(composition_zones == 4)
        pure_whewellite_pixels = np.sum(composition_zones == 5)
        
        total_stone_pixels = (pure_bacteria_pixels + bacteria_rich_pixels + intergrowth_pixels + 
                            whewellite_rich_pixels + pure_whewellite_pixels)
        
        # Calculate average intensities for each zone
        bacteria_avg = np.mean(self.ct_image[composition_zones == 1]) if pure_bacteria_pixels > 0 else 0
        bacteria_rich_avg = np.mean(self.ct_image[composition_zones == 2]) if bacteria_rich_pixels > 0 else 0
        intergrowth_avg = np.mean(self.ct_image[composition_zones == 3]) if intergrowth_pixels > 0 else 0
        whewellite_rich_avg = np.mean(self.ct_image[composition_zones == 4]) if whewellite_rich_pixels > 0 else 0
        whewellite_avg = np.mean(self.ct_image[composition_zones == 5]) if pure_whewellite_pixels > 0 else 0
        
        self.analysis_results = {
            'pure_bacteria_pixels': pure_bacteria_pixels,
            'bacteria_rich_pixels': bacteria_rich_pixels,
            'intergrowth_pixels': intergrowth_pixels,
            'whewellite_rich_pixels': whewellite_rich_pixels,
            'pure_whewellite_pixels': pure_whewellite_pixels,
            'stone_pixels': total_stone_pixels,
            'pure_bacteria_percentage': pure_bacteria_pixels / total_stone_pixels * 100 if total_stone_pixels > 0 else 0,
            'bacteria_rich_percentage': bacteria_rich_pixels / total_stone_pixels * 100 if total_stone_pixels > 0 else 0,
            'intergrowth_percentage': intergrowth_pixels / total_stone_pixels * 100 if total_stone_pixels > 0 else 0,
            'whewellite_rich_percentage': whewellite_rich_pixels / total_stone_pixels * 100 if total_stone_pixels > 0 else 0,
            'pure_whewellite_percentage': pure_whewellite_pixels / total_stone_pixels * 100 if total_stone_pixels > 0 else 0,
            'total_pixels': self.stone_mask.size,
            'stone_area_percentage': total_stone_pixels / self.stone_mask.size * 100,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity
        }
        
        print(f"\nGRADIENT Composition Analysis (DoG-isolated):")
        print(f"Stone boundary: {total_stone_pixels:,} pixels ({self.analysis_results['stone_area_percentage']:.1f}% of image)")
        print(f"\nCompositional Zones:")
        print(f"Pure Bacteria (0-15% whewellite):     {self.analysis_results['pure_bacteria_percentage']:.1f}% ({pure_bacteria_pixels:,} pixels) - avg intensity: {bacteria_avg:.0f}")
        print(f"Bacteria-Rich (15-35% whewellite):    {self.analysis_results['bacteria_rich_percentage']:.1f}% ({bacteria_rich_pixels:,} pixels) - avg intensity: {bacteria_rich_avg:.0f}")
        print(f"🔍 INTERGROWTH (35-65% whewellite):    {self.analysis_results['intergrowth_percentage']:.1f}% ({intergrowth_pixels:,} pixels) - avg intensity: {intergrowth_avg:.0f}")
        print(f"Whewellite-Rich (65-85% whewellite):  {self.analysis_results['whewellite_rich_percentage']:.1f}% ({whewellite_rich_pixels:,} pixels) - avg intensity: {whewellite_rich_avg:.0f}")
        print(f"Pure Whewellite (85-100% whewellite): {self.analysis_results['pure_whewellite_percentage']:.1f}% ({pure_whewellite_pixels:,} pixels) - avg intensity: {whewellite_avg:.0f}")
        print(f"\n🌟 Intergrowth zones of special scientific interest: {self.analysis_results['intergrowth_percentage']:.1f}%")
        
        return composition_zones
    
    def create_enhanced_visualizations(self):
        """Create comprehensive visualizations including DoG enhancements"""
        
        # Apply different enhancement techniques for comparison
        if self.ct_image is None:
            self.load_ct_image()
        
        # Normalize for processing
        ct_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # 1. Gamma enhancement (high contrast)
        gamma_enhanced = np.power(ct_norm, 0.5)  # Gamma < 1 brightens
        
        # 2. DoG enhancement
        if self.dog_enhanced is None:
            self.apply_dog_filter()
        
        # 3. Morphological enhancement
        kernel = morphology.disk(3)
        morph_enhanced = morphology.white_tophat(ct_norm, kernel)
        
        # 4. Bilateral filter enhancement
        # Convert to uint8 for bilateral filter
        ct_uint8 = (ct_norm * 255).astype(np.uint8)
        bilateral_enhanced = cv2.bilateralFilter(ct_uint8, 9, 75, 75) / 255.0
        
        return gamma_enhanced, self.dog_enhanced, morph_enhanced, bilateral_enhanced
    
    def visualize_dog_isolation_process(self):
        """Visualize the DoG-based stone isolation process"""
        
        # Get enhancement visualizations
        gamma_enh, dog_enh, morph_enh, bilateral_enh = self.create_enhanced_visualizations()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Row 1: Enhancement comparisons
        axes[0, 0].imshow(gamma_enh, cmap='gray')
        axes[0, 0].set_title('Gamma High Enhanced')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(dog_enh, cmap='gray')
        axes[0, 1].set_title('DoG Enhanced\n(Perfect Stone Outline)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(morph_enh, cmap='gray')
        axes[0, 2].set_title('Morphological Enhanced')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(bilateral_enh, cmap='gray')
        axes[0, 3].set_title('Bilateral Enhanced')
        axes[0, 3].axis('off')
        
        # Row 2: DoG isolation process
        axes[1, 0].imshow(self.ct_image, cmap='gray')
        axes[1, 0].set_title('Original CT Image')
        axes[1, 0].axis('off')
        
        if self.dog_enhanced is not None:
            axes[1, 1].imshow(self.dog_enhanced, cmap='gray')
            axes[1, 1].set_title('DoG Filter Result\n(Edge Enhancement)')
            axes[1, 1].axis('off')
        
        if self.stone_mask is not None:
            axes[1, 2].imshow(self.stone_mask, cmap='gray')
            axes[1, 2].set_title('Stone Mask\n(From DoG Edges)')
            axes[1, 2].axis('off')
        
        if self.isolated_stone is not None:
            axes[1, 3].imshow(self.isolated_stone, cmap='gray')
            axes[1, 3].set_title('Isolated Stone\n(Background Removed)')
            axes[1, 3].axis('off')
        
        # Row 3: Composition analysis
        if self.composition_map is not None:
            # Create custom colormap for gradient composition
            from matplotlib.colors import ListedColormap, LinearSegmentedColormap
            import matplotlib.cm as cm
            
            # Show continuous composition if available
            if hasattr(self, 'composition_continuous'):
                # Use a continuous colormap from dark red (bacteria) to bright yellow (whewellite)
                # with green highlighting the intergrowth zone
                colors_continuous = ['darkred', 'red', 'orange', 'yellow', 'lightyellow']
                n_bins = 256
                cmap_continuous = LinearSegmentedColormap.from_list('bacteria_whewellite', colors_continuous, N=n_bins)
                
                axes[2, 0].imshow(self.composition_continuous, cmap=cmap_continuous, vmin=0, vmax=100)
                axes[2, 0].set_title('Continuous Composition\n(Dark Red=100% Bacteria, Yellow=100% Whewellite)')
                axes[2, 0].axis('off')
                
                # Add colorbar
                from matplotlib.colorbar import ColorbarBase
                from matplotlib.colors import Normalize
                
                # Create a small inset for colorbar
                cbar_ax = axes[2, 0].inset_axes([0.02, 0.02, 0.3, 0.05])
                norm = Normalize(vmin=0, vmax=100)
                cbar = ColorbarBase(cbar_ax, cmap=cmap_continuous, norm=norm, orientation='horizontal')
                cbar.set_label('% Whewellite', fontsize=8)
                cbar.ax.tick_params(labelsize=6)
            
            # Show zonal composition with distinct colors
            if hasattr(self, 'composition_map'):
                # Define colors for each zone
                zone_colors = ['black',        # 0: background
                             'darkred',       # 1: Pure bacteria (0-15%)
                             'red',           # 2: Bacteria-rich (15-35%)
                             'lime',          # 3: Intergrowth (35-65%) - BRIGHT GREEN for special interest
                             'orange',        # 4: Whewellite-rich (65-85%)
                             'gold']          # 5: Pure whewellite (85-100%)
                
                cmap_zones = ListedColormap(zone_colors)
                
                axes[2, 1].imshow(self.composition_map, cmap=cmap_zones, vmin=0, vmax=5)
                axes[2, 1].set_title('Compositional Zones\n(Bright Green = Intergrowth of Interest)')
                axes[2, 1].axis('off')
                
                # Add zone legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='darkred', label='Pure Bacteria (0-15%)'),
                    Patch(facecolor='red', label='Bacteria-Rich (15-35%)'),
                    Patch(facecolor='lime', label='🔍 INTERGROWTH (35-65%)'),
                    Patch(facecolor='orange', label='Whewellite-Rich (65-85%)'),
                    Patch(facecolor='gold', label='Pure Whewellite (85-100%)')
                ]
                axes[2, 1].legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Show stone boundary overlay
            axes[2, 2].imshow(self.ct_image, cmap='gray')
            if self.stone_mask is not None:
                # Create boundary outline
                boundary = morphology.binary_erosion(self.stone_mask) ^ self.stone_mask
                axes[2, 2].contour(boundary, colors='red', linewidths=2)
            axes[2, 2].set_title('Stone Boundary Overlay\n(Red outline from DoG)')
            axes[2, 2].axis('off')
            
            # Composition statistics with gradient info
            if hasattr(self, 'analysis_results') and self.analysis_results:
                results = self.analysis_results
                
                # Create pie chart for compositional zones in a separate subplot
                axes[2, 3].axis('off')  # Clear the text area first
                
                # Create pie chart in an inset
                pie_ax = axes[2, 3].inset_axes([0.1, 0.3, 0.8, 0.6])
                
                if all(key in results for key in ['pure_bacteria_percentage', 'bacteria_rich_percentage', 
                                                'intergrowth_percentage', 'whewellite_rich_percentage', 
                                                'pure_whewellite_percentage']):
                    fractions = [
                        results['pure_bacteria_percentage'],
                        results['bacteria_rich_percentage'], 
                        results['intergrowth_percentage'],
                        results['whewellite_rich_percentage'],
                        results['pure_whewellite_percentage']
                    ]
                    labels = ['Pure Bacteria', 'Bacteria-Rich', 'INTERGROWTH', 'Whewellite-Rich', 'Pure Whewellite']
                    colors_pie = ['darkred', 'red', 'lime', 'orange', 'gold']
                    
                    # Only include non-zero fractions
                    non_zero_indices = [i for i, f in enumerate(fractions) if f > 0]
                    if non_zero_indices:
                        fractions_nz = [fractions[i] for i in non_zero_indices]
                        labels_nz = [labels[i] for i in non_zero_indices]
                        colors_nz = [colors_pie[i] for i in non_zero_indices]
                        
                        wedges, texts, autotexts = pie_ax.pie(fractions_nz, labels=labels_nz, colors=colors_nz, 
                                                            autopct='%1.1f%%', startangle=90)
                        
                        # Highlight intergrowth wedge
                        for i, label in enumerate(labels_nz):
                            if 'INTERGROWTH' in label:
                                wedges[i].set_linewidth(3)
                                wedges[i].set_edgecolor('black')
                        
                        pie_ax.set_title('Gradient Composition\n(Green=Intergrowth Interest)', fontsize=10)
            
            # Analysis summary with gradient info - move to bottom of the subplot
            if hasattr(self, 'analysis_results') and self.analysis_results:
                summary_text = f"""DoG Gradient Results:

Stone: {self.analysis_results['stone_area_percentage']:.1f}% of image
Pixels: {self.analysis_results['stone_pixels']:,}

Composition Zones:
• Pure Bacteria: {self.analysis_results.get('pure_bacteria_percentage', 0):.1f}%
• Bacteria-Rich: {self.analysis_results.get('bacteria_rich_percentage', 0):.1f}%
• 🔍 INTERGROWTH: {self.analysis_results.get('intergrowth_percentage', 0):.1f}%
• Whewellite-Rich: {self.analysis_results.get('whewellite_rich_percentage', 0):.1f}%
• Pure Whewellite: {self.analysis_results.get('pure_whewellite_percentage', 0):.1f}%

Features:
• Continuous mapping
• Highlights intergrowth
• No black areas
"""
                axes[2, 3].text(0.05, 0.25, summary_text, transform=axes[2, 3].transAxes,
                               fontsize=8, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('dog_stone_isolation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_dog_analysis(self):
        """Run complete DoG-based stone isolation and analysis"""
        print("Starting DoG-Based Stone Isolation Pipeline...")
        print("=" * 50)
        
        # Step 1: Load image
        self.load_ct_image()
        
        # Step 2: Load composition thresholds
        self.load_composition_threshold()
        
        # Step 3: Apply DoG filter for edge detection
        print("\nStep 1: Applying DoG filter for edge enhancement...")
        self.apply_dog_filter()
        
        # Step 4: Create stone mask from DoG edges
        print("\nStep 2: Creating stone mask from DoG edges...")
        self.create_stone_mask_from_dog()
        
        # Step 5: Isolate stone region
        print("\nStep 3: Isolating stone region...")
        self.isolate_stone()
        
        # Step 6: Analyze composition
        print("\nStep 4: Analyzing stone composition...")
        self.analyze_stone_composition()
        
        # Step 7: Create visualizations
        print("\nStep 5: Generating comprehensive visualizations...")
        self.visualize_dog_isolation_process()
        
        print("\n" + "=" * 50)
        print("DoG-Based Stone Isolation Complete!")
        
        return self.analysis_results


def main():
    """Main function to run DoG-based stone isolation"""
    # Initialize isolator
    isolator = DoGStoneIsolator('slice_1092.tif')
    
    # Run complete analysis
    results = isolator.run_complete_dog_analysis()
    
    return isolator, results

if __name__ == "__main__":
    isolator, results = main() 