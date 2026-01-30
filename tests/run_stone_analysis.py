#!/usr/bin/env python3
"""
Quick Launcher for CT-Raman Stone Analysis
Simplified entry point for the standalone application
"""

import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('skimage', 'scikit-image'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("   pip install -r requirements_standalone.txt")
        return False
    
    return True

def check_ct_image():
    """Check if CT image file exists"""
    if not os.path.exists('slice_1092.tif'):
        print("❌ CT image file 'slice_1092.tif' not found!")
        print("   Please ensure the CT image is in the current directory.")
        return False
    return True

def main():
    """Main launcher function"""
    print("🔬 CT-Raman Stone Analysis Launcher")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return False
    
    print("✅ All packages installed")
    
    # Check CT image
    print("🔍 Checking CT image...")
    if not check_ct_image():
        return False
    
    print("✅ CT image found")
    
    # Run the application
    print("\n🚀 Starting Stone Analysis Application...")
    print("\n📖 Quick Guide:")
    print("1. Main window will show 6 analysis panels with interactive sliders")
    print("2. Click first point on 'Composition Zones' image")
    print("3. Click second point to create line scan")
    print("4. New window opens with density profile in g/cm³")
    print("5. Use 'Export Data' button to save line scan to CSV")
    print("6. Horizontal bar chart excludes background and shows stone composition")
    print("7. Adjust parameters with sliders - analysis updates in real-time")
    print("8. Use 'Next Colormap' button to cycle through color schemes")
    print("=" * 50)
    
    try:
        from stone_analysis_standalone import main as run_analysis
        run_analysis()
        return True
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1) 