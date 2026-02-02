# CT-Raman Analysis Workflow Guide

## Overview
This guide explains the recommended workflow for analyzing kidney stone CT images, organized in the order you should use the tools.

---

## 📋 Recommended Workflow Order

### 1️⃣ Initial Analysis (START HERE)

#### **CT Enhancement**
- **Purpose**: Test different image enhancement methods to determine the best approach for your data
- **What it does**: 
  - Applies multiple enhancement techniques (CLAHE, gamma correction, DoG, etc.)
  - Compares results side-by-side
  - Helps you understand which method highlights bacteria vs. whewellite best
- **When to use**: First step with any new CT image
- **Output**: Comparison visualizations showing different enhancement methods

---

### 2️⃣ Main Analysis

#### **Stone Analysis Standalone** ⭐ MAIN TOOL
- **Purpose**: Complete interactive analysis with all features
- **What it does**:
  - Loads CT image via file dialog
  - Applies DoG filtering for stone isolation
  - Maps CT intensity to physical density (g/cm³)
  - Classifies composition into 5 categories (Pure Bacteria → Pure Whewellite)
  - Interactive line scan analysis
  - Real-time parameter tuning
  - Multiple colormap options
- **When to use**: Primary analysis tool for most workflows
- **Output**: 
  - Composition maps
  - Line scan profiles
  - Statistical summaries
  - Interactive visualizations

#### **Enhanced Stone Analysis**
- **Purpose**: Advanced analysis using annotation-based thresholds
- **What it does**:
  - Uses manually annotated regions for training
  - More precise threshold detection
  - Detailed composition analysis
  - Generates comprehensive reports
- **When to use**: When you need more precise control or have training annotations
- **Output**: 
  - Enhanced composition maps
  - Detailed analysis reports
  - Annotation-based statistics

#### **Stone Layer Analysis**
- **Purpose**: Layer-by-layer composition analysis
- **What it does**:
  - Analyzes stone in concentric layers from center to edge
  - Tracks compositional changes radially
  - Useful for understanding stone growth patterns
- **When to use**: When studying stone formation history or layering
- **Output**: 
  - Layer-wise composition profiles
  - Radial density gradients
  - Growth pattern analysis

---

### 3️⃣ Correlation & Export

#### **CT-Raman Correlation**
- **Purpose**: Correlate CT features with Raman spectroscopy data
- **What it does**:
  - Extracts quantitative features from CT data
  - Performs texture analysis (GLCM)
  - Segments regions for correlation
  - Framework for linking CT density to Raman peaks
- **When to use**: When you have Raman spectroscopy data to correlate
- **Output**: 
  - Correlation analysis plots
  - Feature extraction results
  - Statistical correlation reports

---

## 🔧 Training Data (Optional)

### **Interactive Annotation**
- **Purpose**: Manually annotate regions for training Enhanced Stone Analysis
- **What it does**: Click-based interface to mark whewellite, bacteria, and air regions
- **When to use**: To create custom training data (`stone_annotations.pkl`) for Enhanced Stone Analysis
- **Output**: Saves annotations that improve threshold detection accuracy

---

## 🔍 Diagnostics (Troubleshooting)

### **Compare Isolation Methods**
- **Purpose**: Compare different stone isolation algorithms
- **What it does**: Runs multiple isolation methods side-by-side
- **When to use**: When stone isolation isn't working well

### **Threshold Diagnostic**
- **Purpose**: Debug threshold detection issues
- **What it does**: Analyzes intensity distributions and threshold calculations
- **When to use**: When composition classification seems incorrect

---

## 🗑️ Hidden Tools (Redundant)

These tools are hidden from the launcher because their functionality is already in Stone Analysis Standalone:

### **Dog Stone Isolation**
- **Replacement**: Use "Enhanced Stone Analysis" instead

### **Interactive Stone Tuning**
- **Replacement**: Use "Stone Analysis Standalone" (has all tuning built-in)

### **Stone Analysis Widget**
- **Replacement**: Use "Stone Analysis Standalone" (Jupyter version not needed)

### **Stone Analysis App**
- **Replacement**: Use "Stone Analysis Standalone" (Streamlit version not needed)

---

## 💡 Quick Start Guide

### For First-Time Users:
1. **CT Enhancement** → Test different methods to understand your data
2. **Stone Analysis Standalone** ⭐ → Complete interactive analysis (DoG tuning, thresholds, line scans)
3. **CT-Raman Correlation** → (Optional) If you have Raman spectroscopy data

### For Advanced Users:
1. **Interactive Annotation** → Create custom training data
2. **Enhanced Stone Analysis** → Use annotation-based thresholds
3. **Stone Layer Analysis** → Detailed radial layer analysis
4. **CT-Raman Correlation** → Correlate with spectroscopy

### For Troubleshooting:
1. **Stone Analysis Standalone** → Has built-in parameter tuning (no separate tool needed)
2. **Compare Isolation Methods** → Test different isolation algorithms
3. **Threshold Diagnostic** → Debug classification issues

---

## 📊 Understanding the Output

### Composition Categories (by density):
- **Pure Bacteria** (0.95 g/cm³) - Dark regions, low density biofilm
- **Bacteria-Rich** (1.15 g/cm³) - Bacterial biofilm dominant
- **Intergrowth** (1.45 g/cm³) - Mixed bacterial/crystalline regions
- **Whewellite-Rich** (1.75 g/cm³) - Crystalline calcium oxalate dominant
- **Pure Whewellite** (2.23 g/cm³) - Bright regions, high density crystals

### Key Metrics:
- **Stone Area**: Total pixels classified as stone (vs. background)
- **Composition Percentages**: % of stone that is each category
- **Density Map**: Physical density in g/cm³ for each pixel
- **Line Scan**: Density profile along a user-defined path

---

## 🎯 Tips for Best Results

1. **Image Quality**: Higher resolution CT scans give better results
2. **Calibration**: Ensure CT scanner is properly calibrated for Hounsfield Units
3. **Parameter Tuning**: Default parameters work for most cases, but tuning may be needed
4. **Multiple Views**: Analyze multiple slices to get a complete picture
5. **Validation**: Cross-reference with Raman spectroscopy when available

---

## 📝 Notes

- All tools now prompt for image file selection (no hardcoded paths)
- Tools can be run in any order, but the recommended order gives best results
- Interactive tools provide real-time feedback for parameter optimization
- Reports are automatically saved to the `reports/` directory
- Line scan data is saved to the `data/` directory
