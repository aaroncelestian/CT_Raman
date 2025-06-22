import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import filters, segmentation, measure, morphology, exposure
from skimage.feature import local_binary_pattern
from scipy import ndimage
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class CTImageAnalyzer:
    def __init__(self, ct_image_path, raman_image_path=None):
        """
        Initialize the CT Image Analyzer for kidney stone analysis
        
        Args:
            ct_image_path: Path to the micro CT slice image
            raman_image_path: Path to the Raman spectroscopy density measurement image
        """
        self.ct_image_path = ct_image_path
        self.raman_image_path = raman_image_path
        self.ct_image = None
        self.enhanced_images = {}
        
    def load_ct_image(self):
        """Load and preprocess the CT image"""
        # Load the CT image
        if self.ct_image_path.endswith('.tif'):
            self.ct_image = np.array(Image.open(self.ct_image_path))
        else:
            self.ct_image = cv2.imread(self.ct_image_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"CT Image loaded: Shape {self.ct_image.shape}, dtype: {self.ct_image.dtype}")
        print(f"Intensity range: {self.ct_image.min()} - {self.ct_image.max()}")
        return self.ct_image
    
    def enhance_density_differences(self):
        """Apply multiple enhancement techniques to highlight density differences"""
        if self.ct_image is None:
            self.load_ct_image()
        
        # Normalize to 0-1 range for processing
        img_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_clahe = clahe.apply((img_norm * 255).astype(np.uint8))
        self.enhanced_images['CLAHE'] = enhanced_clahe
        
        # 2. Gamma correction for highlighting different density ranges
        # Gamma < 1 highlights dark regions (bacteria), Gamma > 1 highlights bright regions (crystals)
        gamma_low = 0.5  # Enhance dark regions (bacteria)
        gamma_high = 1.8  # Enhance bright regions (crystals)
        
        enhanced_gamma_low = exposure.adjust_gamma(img_norm, gamma_low)
        enhanced_gamma_high = exposure.adjust_gamma(img_norm, gamma_high)
        
        self.enhanced_images['Gamma_Low'] = (enhanced_gamma_low * 255).astype(np.uint8)
        self.enhanced_images['Gamma_High'] = (enhanced_gamma_high * 255).astype(np.uint8)
        
        # 3. Multi-scale enhancement using Difference of Gaussians
        sigma1, sigma2 = 1, 3
        dog = filters.gaussian(img_norm, sigma1) - filters.gaussian(img_norm, sigma2)
        dog_enhanced = exposure.rescale_intensity(dog)
        self.enhanced_images['DoG'] = (dog_enhanced * 255).astype(np.uint8)
        
        # 4. Local Binary Pattern for texture enhancement
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img_norm, n_points, radius, method='uniform')
        self.enhanced_images['LBP'] = (exposure.rescale_intensity(lbp) * 255).astype(np.uint8)
        
        # 5. Morphological enhancement
        kernel = morphology.disk(2)
        tophat = morphology.white_tophat(img_norm, kernel)
        blackhat = morphology.black_tophat(img_norm, kernel)
        morpho_enhanced = img_norm + tophat - blackhat
        self.enhanced_images['Morphological'] = (exposure.rescale_intensity(morpho_enhanced) * 255).astype(np.uint8)
        
        # 6. Edge-preserving smoothing with bilateral filter
        bilateral = cv2.bilateralFilter((img_norm * 255).astype(np.uint8), 9, 75, 75)
        self.enhanced_images['Bilateral'] = bilateral
        
        return self.enhanced_images
    
    def create_density_map(self):
        """Create a density-based segmentation map"""
        if self.ct_image is None:
            self.load_ct_image()
        
        # Normalize image
        img_norm = (self.ct_image - self.ct_image.min()) / (self.ct_image.max() - self.ct_image.min())
        
        # Use Otsu's method for initial segmentation
        thresh_otsu = filters.threshold_otsu(img_norm)
        
        # Multi-level thresholding for density classification
        # Define thresholds for different density regions
        low_density_thresh = thresh_otsu * 0.7   # Bacteria-rich regions
        high_density_thresh = thresh_otsu * 1.3  # Crystal-rich regions
        
        # Create density map
        density_map = np.zeros_like(img_norm)
        density_map[img_norm < low_density_thresh] = 1    # Low density (bacteria)
        density_map[(img_norm >= low_density_thresh) & (img_norm < high_density_thresh)] = 2  # Medium density
        density_map[img_norm >= high_density_thresh] = 3  # High density (crystals)
        
        # Apply morphological operations to clean up the segmentation
        for i in range(1, 4):
            mask = density_map == i
            mask_cleaned = morphology.remove_small_objects(mask, min_size=50)
            mask_cleaned = morphology.remove_small_holes(mask_cleaned, area_threshold=30)
            density_map[density_map == i] = 0
            density_map[mask_cleaned] = i
        
        self.density_map = density_map
        return density_map
    
    def create_custom_colormap(self):
        """Create a custom colormap for density visualization"""
        # Define colors for different density regions
        colors = ['#000080', '#0080FF', '#80FF80', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)
        return cmap
    
    def visualize_enhancements(self, save_path=None):
        """Visualize all enhancement techniques"""
        n_images = len(self.enhanced_images) + 2  # +2 for original and density map
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        # Original image
        plt.subplot(rows, cols, 1)
        plt.imshow(self.ct_image, cmap='gray')
        plt.title('Original CT Image')
        plt.axis('off')
        
        # Enhanced images
        for i, (name, img) in enumerate(self.enhanced_images.items(), 2):
            plt.subplot(rows, cols, i)
            plt.imshow(img, cmap='gray')
            plt.title(f'{name} Enhanced')
            plt.axis('off')
        
        # Density map
        if hasattr(self, 'density_map'):
            plt.subplot(rows, cols, len(self.enhanced_images) + 2)
            custom_cmap = self.create_custom_colormap()
            plt.imshow(self.density_map, cmap=custom_cmap)
            plt.title('Density Map')
            plt.colorbar(label='Density Level')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_density_distribution(self):
        """Analyze the distribution of density values"""
        if self.ct_image is None:
            self.load_ct_image()
        
        plt.figure(figsize=(15, 5))
        
        # Histogram of original intensities
        plt.subplot(1, 3, 1)
        plt.hist(self.ct_image.flatten(), bins=100, alpha=0.7, edgecolor='black')
        plt.title('Original Intensity Distribution')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        
        # Histogram after enhancement (using CLAHE as example)
        if 'CLAHE' in self.enhanced_images:
            plt.subplot(1, 3, 2)
            plt.hist(self.enhanced_images['CLAHE'].flatten(), bins=100, alpha=0.7, edgecolor='black', color='orange')
            plt.title('Enhanced Intensity Distribution (CLAHE)')
            plt.xlabel('Intensity')
            plt.ylabel('Frequency')
        
        # Density map distribution
        if hasattr(self, 'density_map'):
            plt.subplot(1, 3, 3)
            unique, counts = np.unique(self.density_map, return_counts=True)
            labels = ['Background', 'Low Density\n(Bacteria)', 'Medium Density', 'High Density\n(Crystals)']
            plt.bar(unique, counts, color=['black', 'blue', 'green', 'red'])
            plt.title('Density Region Distribution')
            plt.xlabel('Density Level')
            plt.ylabel('Pixel Count')
            plt.xticks(unique, [labels[int(i)] if i < len(labels) else f'Level {int(i)}' for i in unique])
        
        plt.tight_layout()
        plt.show()
    
    def load_raman_data(self):
        """Load and display Raman spectroscopy data for correlation"""
        if self.raman_image_path:
            raman_img = plt.imread(self.raman_image_path)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(raman_img)
            plt.title('Raman Spectroscopy Density Measure')
            plt.axis('off')
            
            # If it's a plot image, we can try to extract some information
            plt.subplot(1, 2, 2)
            if len(raman_img.shape) == 3:
                # Convert to grayscale for analysis
                raman_gray = np.dot(raman_img[...,:3], [0.299, 0.587, 0.114])
                plt.imshow(raman_gray, cmap='viridis')
                plt.title('Raman Data (Grayscale)')
                plt.colorbar(label='Intensity')
            else:
                plt.imshow(raman_img, cmap='viridis')
                plt.title('Raman Data')
                plt.colorbar(label='Intensity')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def correlate_ct_raman(self):
        """Suggest methods for correlating CT and Raman data"""
        print("CT-Raman Correlation Strategies:")
        print("=" * 40)
        print("1. Spatial Registration:")
        print("   - Use fiducial markers or anatomical landmarks")
        print("   - Apply image registration algorithms (e.g., rigid, affine, or elastic)")
        print("   - Account for different spatial resolutions")
        
        print("\n2. Density Correlation:")
        print("   - Map CT Hounsfield units to Raman peak intensities")
        print("   - Correlate whewellite crystal peaks with high-density CT regions")
        print("   - Correlate bacterial biofilm signatures with low-density CT regions")
        
        print("\n3. Statistical Analysis:")
        print("   - ROI-based correlation analysis")
        print("   - Pixel-wise correlation coefficients")
        print("   - Machine learning approaches for pattern recognition")
        
        print("\n4. Validation:")
        print("   - Cross-validation with histological data")
        print("   - Quantitative assessment of correlation accuracy")

def main():
    # Initialize the analyzer
    analyzer = CTImageAnalyzer('slice_1092.tif', 'DensityMeasure.png')
    
    # Load and analyze the CT image
    print("Loading CT image...")
    analyzer.load_ct_image()
    
    # Enhance the image to highlight density differences
    print("Applying enhancement techniques...")
    enhanced_images = analyzer.enhance_density_differences()
    
    # Create density map
    print("Creating density map...")
    density_map = analyzer.create_density_map()
    
    # Visualize results
    print("Visualizing enhancements...")
    analyzer.visualize_enhancements('enhanced_ct_results.png')
    
    # Analyze density distribution
    print("Analyzing density distribution...")
    analyzer.analyze_density_distribution()
    
    # Load and display Raman data
    print("Loading Raman data...")
    analyzer.load_raman_data()
    
    # Provide correlation strategies
    analyzer.correlate_ct_raman()
    
    print("\nAnalysis complete! Enhanced images and analysis saved.")

if __name__ == "__main__":
    main() 