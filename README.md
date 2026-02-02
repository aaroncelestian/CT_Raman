# CT-Raman Kidney Stone Analysis

This project provides tools for enhancing micro CT images of kidney stones and correlating density differences with Raman spectroscopy measurements. The analysis focuses on highlighting differences between crystalline whewellite (light areas) and bacterial regions (dark areas).

## Project Structure

```
CT_Raman/
├── src/                          # Source code
│   ├── analysis/                # Core analysis modules
│   │   ├── ct_enhancement.py
│   │   ├── ct_raman_correlation.py
│   │   ├── dog_stone_isolation.py
│   │   ├── enhanced_stone_analysis.py
│   │   └── stone_layer_analysis.py
│   ├── interactive/             # Interactive tools & applications
│   │   ├── interactive_annotation.py
│   │   ├── interactive_stone_tuning.py
│   │   ├── stone_analysis_app.py
│   │   ├── stone_analysis_standalone.py
│   │   └── stone_analysis_widget.py
│   └── utils/                   # Utilities & diagnostics
│       ├── compare_isolation_methods.py
│       └── threshold_diagnostic.py
├── data/                        # Data files (CSVs, pickles)
│   ├── line_scan_*.csv
│   ├── *_metadata.pkl
│   └── optimized_stone_settings*.pkl
├── notebooks/                   # Jupyter notebooks
├── reports/                     # Generated analysis reports
│   ├── annotation_based_report.txt
│   ├── ct_raman_correlation_report.txt
│   ├── enhanced_stone_report.txt
│   └── stone_layer_report.txt
├── tests/                       # Test scripts
│   ├── run_stone_analysis.py
│   └── simple_test.py
├── docs/                        # Documentation
│   └── README_widget.md
├── images/                      # Image outputs
├── requirements.txt             # Main dependencies
├── requirements_jupyter.txt     # Jupyter-specific dependencies
└── requirements_standalone.txt  # Standalone app dependencies
```

## Quick Start

The easiest way to run any tool in this project is using the **GUI launcher**:

```bash
python launch.py
```

This will open a graphical interface with:
- **Tabbed categories** - Analysis Tools, Interactive Tools, Utilities, and Tests
- **Easy selection** - Click or double-click to launch any tool
- **Real-time output log** - See script output directly in the launcher
- **Professional interface** - Clean, organized, and user-friendly

Simply select a tool from the list and click "Launch Selected" or double-click the tool name.

## Main Workflow

### Overview
This application analyzes kidney stone CT images to identify and quantify different compositional regions, correlating CT density with material composition (bacterial biofilm vs. crystalline whewellite).

### Typical Analysis Pipeline

1. **Load CT Image** → The application loads a micro-CT slice of a kidney stone
2. **Stone Isolation** → Uses Difference of Gaussians (DoG) filtering to isolate the stone from background
3. **Density Mapping** → Maps CT intensity (Hounsfield Units) to physical density (g/cm³)
4. **Composition Classification** → Classifies regions into categories:
   - **Pure Bacteria** (0.95 g/cm³) - Dark regions, low density
   - **Bacteria-Rich** (1.15 g/cm³) - Bacterial biofilm dominant
   - **Intergrowth** (1.45 g/cm³) - Mixed bacterial/crystalline
   - **Whewellite-Rich** (1.75 g/cm³) - Crystalline dominant
   - **Pure Whewellite** (2.23 g/cm³) - Bright regions, high density
5. **Visualization** → Displays composition maps with customizable colormaps
6. **Line Scan Analysis** → Interactive tool to analyze density profiles along user-defined lines
7. **Export Results** → Save composition maps, line scans, and statistical reports

### Key Features
- **Interactive parameter tuning** - Adjust thresholds and filters in real-time
- **Density calibration** - Automatic calibration from CT intensity to physical density
- **Multiple visualization modes** - 12+ colormaps for different analysis needs
- **Quantitative analysis** - Statistical summaries of compositional percentages
- **Line profile analysis** - Detailed density profiles along custom paths
- **Raman correlation** - Framework for correlating with Raman spectroscopy data

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
from src.analysis.ct_enhancement import CTImageAnalyzer

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
from src.analysis.ct_raman_correlation import CTRamanCorrelator

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
python src/analysis/ct_enhancement.py
python src/analysis/ct_raman_correlation.py
```

### Interactive Tools

```bash
# Run the standalone analysis application
python src/interactive/stone_analysis_standalone.py

# Run the interactive widget
python src/interactive/stone_analysis_widget.py
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