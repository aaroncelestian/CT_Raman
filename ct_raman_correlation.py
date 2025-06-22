import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize, stats
from skimage import transform, feature, filters
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from skimage.morphology import disk

class CTRamanCorrelator:
    def __init__(self, ct_analyzer):
        """
        Initialize CT-Raman correlator
        
        Args:
            ct_analyzer: CTImageAnalyzer instance with loaded and processed data
        """
        self.ct_analyzer = ct_analyzer
        self.registration_params = {}
        self.correlation_results = {}
        
    def extract_ct_features(self):
        """Extract quantitative features from CT data"""
        if self.ct_analyzer.ct_image is None:
            self.ct_analyzer.load_ct_image()
        
        ct_norm = (self.ct_analyzer.ct_image - self.ct_analyzer.ct_image.min()) / \
                  (self.ct_analyzer.ct_image.max() - self.ct_analyzer.ct_image.min())
        
        features = {}
        
        # 1. Statistical features
        features['mean_intensity'] = np.mean(ct_norm)
        features['std_intensity'] = np.std(ct_norm)
        features['skewness'] = stats.skew(ct_norm.flatten())
        features['kurtosis'] = stats.kurtosis(ct_norm.flatten())
        
        # 2. Texture features using Gray Level Co-occurrence Matrix
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert to appropriate data type for GLCM
        ct_uint8 = (ct_norm * 255).astype(np.uint8)
        
        # Calculate GLCM for different angles and distances
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(ct_uint8, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract texture properties
        features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
        features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
        features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
        features['energy'] = np.mean(graycoprops(glcm, 'energy'))
        features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
        
        # 3. Gradient features
        grad_x = cv2.Sobel(ct_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(ct_norm, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # 4. Morphological features
        if hasattr(self.ct_analyzer, 'density_map'):
            unique, counts = np.unique(self.ct_analyzer.density_map, return_counts=True)
            total_pixels = np.sum(counts[1:])  # Exclude background
            
            if total_pixels > 0:
                features['bacteria_fraction'] = counts[1] / total_pixels if len(counts) > 1 else 0
                features['crystal_fraction'] = counts[-1] / total_pixels if len(counts) > 2 else 0
        
        self.ct_features = features
        return features
    
    def create_density_profile(self, method='radial'):
        """Create density profiles for correlation analysis"""
        if self.ct_analyzer.ct_image is None:
            self.ct_analyzer.load_ct_image()
        
        ct_norm = (self.ct_analyzer.ct_image - self.ct_analyzer.ct_image.min()) / \
                  (self.ct_analyzer.ct_image.max() - self.ct_analyzer.ct_image.min())
        
        if method == 'radial':
            # Create radial profile from center
            center = np.array(ct_norm.shape) // 2
            y, x = np.ogrid[:ct_norm.shape[0], :ct_norm.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Create radial bins
            max_radius = int(np.max(r))
            radial_profile = []
            radii = []
            
            for radius in range(0, max_radius, 5):  # 5-pixel bins
                mask = (r >= radius) & (r < radius + 5)
                if np.any(mask):
                    radial_profile.append(np.mean(ct_norm[mask]))
                    radii.append(radius + 2.5)  # Center of bin
            
            return np.array(radii), np.array(radial_profile)
        
        elif method == 'linear':
            # Create linear profiles along major axes
            center = np.array(ct_norm.shape) // 2
            
            # Horizontal profile
            h_profile = ct_norm[center[0], :]
            # Vertical profile
            v_profile = ct_norm[:, center[1]]
            
            return {'horizontal': h_profile, 'vertical': v_profile}
    
    def segment_regions_for_correlation(self, n_clusters=3):
        """Segment CT image into regions for targeted correlation"""
        if self.ct_analyzer.ct_image is None:
            self.ct_analyzer.load_ct_image()
        
        # Extract features for clustering
        ct_norm = (self.ct_analyzer.ct_image - self.ct_analyzer.ct_image.min()) / \
                  (self.ct_analyzer.ct_image.max() - self.ct_analyzer.ct_image.min())
        
        # Create feature vectors for each pixel
        features = []
        
        # Intensity
        features.append(ct_norm.flatten())
        
        # Gradient magnitude
        grad_x = cv2.Sobel(ct_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(ct_norm, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        features.append(gradient_mag.flatten())
        
        # Local standard deviation using convolution
        kernel = np.ones((5, 5)) / 25  # 5x5 averaging kernel
        local_mean = ndimage.convolve(ct_norm, kernel, mode='constant')
        local_mean_sq = ndimage.convolve(ct_norm**2, kernel, mode='constant')
        local_variance = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))  # Ensure non-negative
        features.append(local_std.flatten())
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Reshape back to image
        segmented_image = cluster_labels.reshape(ct_norm.shape)
        
        self.segmented_regions = segmented_image
        return segmented_image
    
    def correlate_with_raman_data(self, raman_data_path=None):
        """Correlate CT features with Raman spectroscopy data"""
        if raman_data_path is None:
            raman_data_path = self.ct_analyzer.raman_image_path
        
        if raman_data_path is None:
            print("No Raman data path provided")
            return None
        
        # Load Raman data
        raman_img = plt.imread(raman_data_path)
        
        # If it's a plot image, try to extract data
        if len(raman_img.shape) == 3:
            # Convert to grayscale
            raman_gray = np.dot(raman_img[...,:3], [0.299, 0.587, 0.114])
        else:
            raman_gray = raman_img
        
        # Extract CT features
        ct_features = self.extract_ct_features()
        
        # For demonstration, create synthetic correlation
        # In practice, you would extract quantitative data from Raman spectra
        correlation_analysis = {
            'ct_features': ct_features,
            'raman_summary': {
                'mean_intensity': np.mean(raman_gray),
                'std_intensity': np.std(raman_gray),
                'peak_intensity': np.max(raman_gray),
                'background_level': np.percentile(raman_gray, 10)
            }
        }
        
        # Calculate correlation coefficients (example)
        if hasattr(self, 'segmented_regions'):
            region_correlations = {}
            for region in np.unique(self.segmented_regions):
                mask = self.segmented_regions == region
                if np.sum(mask) > 10:  # Minimum region size
                    ct_region_mean = np.mean(self.ct_analyzer.ct_image[mask])
                    # This would be replaced with actual Raman measurements
                    region_correlations[f'region_{region}'] = {
                        'ct_mean': ct_region_mean,
                        'pixel_count': np.sum(mask)
                    }
            
            correlation_analysis['region_analysis'] = region_correlations
        
        self.correlation_results = correlation_analysis
        return correlation_analysis
    
    def visualize_correlation_results(self):
        """Visualize correlation analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original CT image
        axes[0, 0].imshow(self.ct_analyzer.ct_image, cmap='gray')
        axes[0, 0].set_title('Original CT Image')
        axes[0, 0].axis('off')
        
        # Segmented regions
        if hasattr(self, 'segmented_regions'):
            im1 = axes[0, 1].imshow(self.segmented_regions, cmap='tab10')
            axes[0, 1].set_title('Segmented Regions')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
        
        # Density map
        if hasattr(self.ct_analyzer, 'density_map'):
            im2 = axes[0, 2].imshow(self.ct_analyzer.density_map, cmap='viridis')
            axes[0, 2].set_title('Density Map')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2])
        
        # Radial profile
        radii, profile = self.create_density_profile('radial')
        axes[1, 0].plot(radii, profile, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Distance from Center (pixels)')
        axes[1, 0].set_ylabel('Normalized Intensity')
        axes[1, 0].set_title('Radial Density Profile')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance
        if hasattr(self, 'ct_features'):
            features = list(self.ct_features.keys())
            values = list(self.ct_features.values())
            
            axes[1, 1].barh(features, values)
            axes[1, 1].set_title('CT Feature Values')
            axes[1, 1].set_xlabel('Feature Value')
        
        # Correlation summary
        if self.correlation_results:
            # Create a simple correlation visualization
            axes[1, 2].text(0.1, 0.9, 'Correlation Summary:', fontsize=12, fontweight='bold', 
                           transform=axes[1, 2].transAxes)
            
            y_pos = 0.8
            for key, value in self.correlation_results.get('ct_features', {}).items():
                if isinstance(value, (int, float)):
                    axes[1, 2].text(0.1, y_pos, f'{key}: {value:.3f}', 
                                   transform=axes[1, 2].transAxes)
                    y_pos -= 0.1
            
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('ct_raman_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_correlation_report(self):
        """Generate a comprehensive correlation report"""
        report = []
        report.append("CT-Raman Correlation Analysis Report")
        report.append("=" * 50)
        
        if hasattr(self, 'ct_features'):
            report.append("\nCT Image Features:")
            report.append("-" * 20)
            for feature, value in self.ct_features.items():
                if isinstance(value, (int, float)):
                    report.append(f"{feature:.<25} {value:.4f}")
        
        if self.correlation_results:
            report.append("\nRaman Data Summary:")
            report.append("-" * 20)
            raman_summary = self.correlation_results.get('raman_summary', {})
            for feature, value in raman_summary.items():
                report.append(f"{feature:.<25} {value:.4f}")
        
        if 'region_analysis' in self.correlation_results:
            report.append("\nRegion-wise Analysis:")
            report.append("-" * 20)
            for region, data in self.correlation_results['region_analysis'].items():
                report.append(f"{region}:")
                report.append(f"  CT Mean Intensity: {data['ct_mean']:.4f}")
                report.append(f"  Pixel Count: {data['pixel_count']}")
        
        report.append("\nRecommendations for CT-Raman Correlation:")
        report.append("-" * 40)
        report.append("1. Spatial Registration:")
        report.append("   - Align CT and Raman coordinate systems")
        report.append("   - Use fiducial markers if available")
        
        report.append("\n2. Quantitative Analysis:")
        report.append("   - Extract specific Raman peak intensities")
        report.append("   - Correlate whewellite peaks with high-density CT regions")
        report.append("   - Correlate bacterial signatures with low-density regions")
        
        report.append("\n3. Statistical Validation:")
        report.append("   - Perform cross-correlation analysis")
        report.append("   - Calculate correlation coefficients")
        report.append("   - Apply significance testing")
        
        # Save report
        with open('ct_raman_correlation_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report

def main():
    # Import the CT analyzer
    from ct_enhancement import CTImageAnalyzer
    
    # Initialize analyzer and correlator
    ct_analyzer = CTImageAnalyzer('slice_1092.tif', 'DensityMeasure.png')
    ct_analyzer.load_ct_image()
    ct_analyzer.enhance_density_differences()
    ct_analyzer.create_density_map()
    
    correlator = CTRamanCorrelator(ct_analyzer)
    
    # Perform correlation analysis
    print("Extracting CT features...")
    correlator.extract_ct_features()
    
    print("Segmenting regions...")
    correlator.segment_regions_for_correlation()
    
    print("Correlating with Raman data...")
    correlator.correlate_with_raman_data()
    
    print("Visualizing results...")
    correlator.visualize_correlation_results()
    
    print("Generating report...")
    correlator.generate_correlation_report()
    
    print("\nCorrelation analysis complete!")

if __name__ == "__main__":
    main() 