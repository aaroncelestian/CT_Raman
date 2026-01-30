# 🔬 CT-Raman Stone Analysis Widget Interface

A professional Jupyter widget interface for analyzing kidney stone CT images and correlating with Raman spectroscopy data.

## 🚀 Quick Start

### Option 1: New Jupyter Notebook (Recommended)

1. **Start JupyterLab:**
   ```bash
   jupyter lab
   ```

2. **Create a new notebook** and paste this code in the first cell:
   ```python
   # Configure matplotlib for interactive widgets
   %matplotlib widget
   import matplotlib.pyplot as plt
   plt.style.use('default')
   
   # Import and launch the widget
   from stone_analysis_widget import StoneAnalysisWidget
   
   print("🚀 Launching CT-Raman Stone Analysis Interface...")
   stone_widget = StoneAnalysisWidget()
   stone_widget.display()
   ```

3. **Run the cell** and the interface will appear below

### Option 2: Direct Python Script

1. **Run the standalone script:**
   ```bash
   python stone_analysis_widget.py
   ```

## 📋 Prerequisites

Make sure you have installed the required packages:
```bash
pip install -r requirements_jupyter.txt
```

Required files in your working directory:
- `slice_1092.tif` (your CT image)
- `stone_analysis_widget.py` (the interface code)

## 🎯 Interface Features

### **📊 Main Display (6-Panel Analysis):**
- **Original CT**: Raw kidney stone slice
- **DoG Enhanced**: Edge-enhanced for boundary detection  
- **CLAHE Enhanced**: Contrast-enhanced CT
- **Stone Mask**: Detected stone boundary with coverage %
- **Composition Zones**: Color-coded mineral phases (**clickable for line scans**)
- **Overlay**: CLAHE + composition overlay

### **🎛️ Controls (Left Sidebar):**

#### **DoG Filter Parameters:**
- **σ1**: Larger Gaussian (↑ reduces holes)
- **σ2**: Smaller Gaussian (↓ reduces holes)

#### **Stone Detection:**
- **Stone Threshold**: Boundary detection sensitivity
- **Min Size**: Minimum stone size in pixels
- **Hole Fill**: Maximum hole size to fill

#### **Composition Thresholds (%):**
- **Bacteria %**: Pure bacteria cutoff (0-X%)
- **B-Rich %**: Bacteria-rich zone (X-Y%)  
- **Intergrowth %**: Mixed zone (Y-Z%)
- **W-Rich %**: Whewellite-rich zone (Z-W%)
- Pure whewellite is everything above W-Rich %

#### **Visual Settings:**
- **Overlay α**: Transparency of composition overlay
- **Anti-Aliasing**: Smooth edges for publication quality
- **Colormap**: 5 scientific color schemes

#### **Action Buttons:**
- **🔄 Recalculate**: Apply current parameters
- **💾 Save Settings**: Export parameters to file
- **🗑️ Clear Line**: Remove current line scan

### **📈 Line Scan Analysis:**

1. **Click first point** on "Composition Zones" image (red dot appears)
2. **Click second point** to complete line (blue dot + line appears)
3. **View results**: 
   - Line profile showing composition along path
   - Pie chart with percentages
   - Detailed statistics

### **📋 Statistics Panel:**
- Stone coverage percentage
- Pixel counts for each composition zone
- Image dimensions and intensity ranges
- Current line scan information

## 🎨 Available Colormaps

- **Original**: Traditional black→red→green→orange→yellow
- **Scientific**: Blue→cyan→green→orange→red sequence
- **Viridis**: Perceptually uniform purple→green→yellow
- **Plasma**: High-contrast purple→pink→yellow
- **Cool**: Blue→cyan→green→yellow→orange gradient

## 🔧 Optimized Default Parameters

The interface loads with parameters optimized for typical kidney stone analysis:
- DoG σ1: 5.2, σ2: 3.0 (good edge enhancement)
- Stone threshold: 0.50 (balanced boundary detection)
- Min size: 50,000 pixels (filters noise)
- Hole fill: 5,858 pixels (fills small gaps)
- Composition thresholds: 24%, 44%, 58%, 75% (empirically derived)

## 💾 Data Export

### **Settings:**
- Saved to `jupyter_stone_settings.pkl`
- Contains all current parameter values
- Can be loaded in future sessions

### **Analysis Results:**
- Stone mask and composition maps available as numpy arrays
- Line scan data stored in widget object
- Statistics printed to console for copy/paste

## 🚨 Troubleshooting

### **Notebook Won't Load:**
- Delete any corrupted .ipynb files
- Create a fresh notebook
- Copy the launch code above

### **Widget Not Displaying:**
- Ensure `%matplotlib widget` is run first
- Install ipympl: `pip install ipympl`
- Restart kernel and try again

### **CT Image Not Found:**
- Verify `slice_1092.tif` is in current directory
- Check file permissions
- Try absolute path if needed

### **Import Errors:**
- Install requirements: `pip install -r requirements_jupyter.txt`
- Restart Jupyter kernel
- Check Python environment

### **Click Detection Not Working:**
- Make sure matplotlib backend is 'widget'
- Try clicking directly on composition zones image
- Restart interface if needed

## 🔬 Scientific Workflow

1. **Load interface** with default parameters
2. **Adjust stone detection** to capture full boundary
3. **Fine-tune composition thresholds** based on known mineralogy
4. **Select colormaps** appropriate for publication
5. **Create line scans** across regions of interest
6. **Export settings** for reproducible analysis
7. **Document results** from statistics panel

## 📚 Technical Details

### **Algorithms Used:**
- **DoG Filter**: Difference of Gaussians for edge enhancement
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Morphological Operations**: Opening, closing, hole filling
- **Connected Components**: Largest region selection
- **Anti-aliasing**: Gaussian smoothing with interpolation

### **Performance:**
- Real-time parameter updates
- Fixed-size plots prevent distortion
- Efficient numpy operations
- Interactive matplotlib integration

This interface provides a professional, research-grade tool for CT-Raman correlation analysis with publication-quality visualizations and quantitative line scan capabilities. 