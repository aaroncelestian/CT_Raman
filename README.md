# CT-Raman Kidney Stone Analysis

This project provides tools for enhancing micro CT images of kidney stones and correlating density differences with Raman spectroscopy measurements. The analysis focuses on highlighting differences between crystalline whewellite (light areas) and bacterial regions (dark areas).

## Features

### CT Image Enhancement (`ct_enhancement.py`)
- **Multiple Enhancement Techniques**: CLAHE, gamma correction, morphological operations, bilateral filtering
- **Density-based Segmentation**: Automated classification of regions based on CT density
- **Visualization Tools**: Comprehensive plotting and analysis of enhancement results
- **Quantitative Analysis**: Statistical analysis of density distributions

### CT-Raman Correlation (`ct_raman_correlation.py`)
- **Feature Extraction**: Quantitative texture and morphological features from CT data
- **Region Segmentation**: Machine learning-based clustering for targeted analysis
- **Correlation Analysis**: Framework for correlating CT and Raman measurements
- **Reporting Tools**: Automated generation of analysis reports

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic CT Enhancement

```python
from ct_enhancement import CTImageAnalyzer

# Initialize analyzer
analyzer = CTImageAnalyzer('slice_1092.tif', 'DensityMeasure.png')

# Load and enhance CT image
analyzer.load_ct_image()
enhanced_images = analyzer.enhance_density_differences()

# Create density map
density_map = analyzer.create_density_map()

# Visualize results
analyzer.visualize_enhancements('enhanced_results.png')
analyzer.analyze_density_distribution()
```

### Advanced Correlation Analysis

```python
from ct_raman_correlation import CTRamanCorrelator

# Initialize correlator with CT analyzer
correlator = CTRamanCorrelator(analyzer)

# Extract features and segment regions
correlator.extract_ct_features()
correlator.segment_regions_for_correlation()

# Correlate with Raman data
correlator.correlate_with_raman_data()

# Generate comprehensive report
correlator.visualize_correlation_results()
correlator.generate_correlation_report()
```

### Quick Start - Run Complete Analysis

```bash
python ct_enhancement.py
python ct_raman_correlation.py
```

## Enhancement Techniques

### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhances local contrast while preserving overall structure
- Ideal for highlighting subtle density differences

### 2. Gamma Correction
- **Low Gamma (0.5)**: Enhances dark regions (bacterial areas)
- **High Gamma (1.8)**: Enhances bright regions (crystalline areas)

### 3. Difference of Gaussians (DoG)
- Multi-scale enhancement for edge detection
- Highlights boundaries between different density regions

### 4. Local Binary Pattern (LBP)
- Texture-based enhancement
- Useful for identifying bacterial biofilm patterns

### 5. Morphological Operations
- Top-hat and black-hat transformations
- Enhances small-scale density variations

### 6. Bilateral Filtering
- Edge-preserving smoothing
- Reduces noise while maintaining boundaries

## Density Classification

The system automatically classifies regions into:
- **Low Density (Blue)**: Bacterial regions
- **Medium Density (Green)**: Transition zones
- **High Density (Red)**: Crystalline whewellite regions

## Correlation Strategies

### 1. Spatial Registration
- Align CT and Raman coordinate systems
- Account for different spatial resolutions
- Use fiducial markers when available

### 2. Quantitative Correlation
- Map CT Hounsfield units to Raman peak intensities
- Correlate whewellite crystal peaks (specific wavenumbers) with high-density CT regions
- Correlate bacterial biofilm signatures with low-density CT regions

### 3. Statistical Analysis
- Region of Interest (ROI) based correlation
- Pixel-wise correlation coefficients
- Machine learning approaches for pattern recognition

## Output Files

- `enhanced_ct_results.png`: Visualization of all enhancement techniques
- `ct_raman_correlation_analysis.png`: Comprehensive correlation analysis
- `ct_raman_correlation_report.txt`: Detailed text report

## Key Features for Kidney Stone Analysis

### Whewellite Detection
- High-density regions in CT correlate with crystalline structures
- Gamma enhancement (>1.0) optimized for crystal visualization
- Texture analysis identifies crystalline patterns

### Bacterial Region Identification
- Low-density regions identified through adaptive thresholding
- Gamma enhancement (<1.0) optimized for bacterial visualization
- Local Binary Pattern analysis reveals biofilm textures

### Density Profiling
- Radial density profiles from stone center
- Linear profiles along major axes
- Quantitative metrics for correlation with Raman data

## Customization

### Adjusting Enhancement Parameters

```python
# Modify gamma values for different emphasis
gamma_low = 0.3   # More aggressive bacterial enhancement
gamma_high = 2.0  # More aggressive crystal enhancement

# Adjust CLAHE parameters
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))

# Modify segmentation thresholds
low_density_thresh = thresh_otsu * 0.6   # More sensitive bacterial detection
high_density_thresh = thresh_otsu * 1.5  # More selective crystal detection
```

### Custom Correlation Analysis

```python
# Add custom features for correlation
def extract_custom_features(self, ct_image):
    # Implement domain-specific feature extraction
    # e.g., specific texture patterns, shape metrics
    pass

# Implement spatial registration
def register_ct_raman(self, ct_image, raman_data):
    # Implement image registration algorithms
    # Account for scale, rotation, translation differences
    pass
```

## Expected Results

The analysis will help you:

1. **Enhance CT Images**: Clearly distinguish between crystalline and bacterial regions
2. **Quantify Density Differences**: Obtain numerical metrics for correlation
3. **Correlate with Raman Data**: Framework for linking CT density to Raman spectroscopy
4. **Generate Reports**: Comprehensive documentation of analysis results

## Next Steps for Implementation

1. **Spatial Registration**: Implement precise alignment between CT and Raman coordinate systems
2. **Raman Peak Extraction**: Develop methods to extract quantitative data from Raman spectra
3. **Statistical Validation**: Apply correlation analysis with significance testing
4. **Machine Learning**: Train models to automatically classify regions based on combined CT-Raman features

## Notes

- The current implementation provides a framework for correlation analysis
- Actual correlation requires precise spatial registration between CT and Raman measurements
- The system is designed to be extensible for adding new enhancement techniques and correlation methods
- Results should be validated against histological or other ground truth data when available

## Contact

For questions about the analysis or suggestions for improvements, please refer to the documentation in the code or modify the parameters according to your specific research needs. 