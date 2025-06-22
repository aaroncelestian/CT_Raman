import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

def diagnose_thresholds():
    """Diagnostic tool to investigate threshold issues"""
    
    # Load CT image
    ct_image = np.array(Image.open('slice_1092.tif'))
    print(f"CT Image: {ct_image.shape}, range: {ct_image.min()} - {ct_image.max()}")
    
    annotations = None
    air_intensities = []
    bacteria_intensities = []
    whewellite_intensities = []
    current_threshold = None
    
    # Load annotations if available
    if os.path.exists('stone_annotations.pkl'):
        with open('stone_annotations.pkl', 'rb') as f:
            annotations = pickle.load(f)
        
        # Analyze annotation intensities
        air_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'air']
        bacteria_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'bacteria']
        whewellite_intensities = [ann['intensity'] for ann in annotations if ann['label'] == 'whewellite']
        
        print(f"\nManual Annotation Analysis (from pickle):")
        print(f"Air points: {len(air_intensities)}, mean: {np.mean(air_intensities):.1f}, range: {np.min(air_intensities)}-{np.max(air_intensities)}")
        print(f"Bacteria points: {len(bacteria_intensities)}, mean: {np.mean(bacteria_intensities):.1f}, range: {np.min(bacteria_intensities)}-{np.max(bacteria_intensities)}")
        print(f"Whewellite points: {len(whewellite_intensities)}, mean: {np.mean(whewellite_intensities):.1f}, range: {np.min(whewellite_intensities)}-{np.max(whewellite_intensities)}")
        
    elif os.path.exists('annotation_based_report.txt'):
        # Load from report file
        with open('annotation_based_report.txt', 'r') as f:
            content = f.read()
        
        print(f"\nLoading annotation data from report file...")
        
        # Parse the report to get statistics
        lines = content.split('\n')
        stats = {}
        
        # Extract means from report (reconstruct annotation intensities for visualization)
        if 'Air:' in content and 'Mean intensity:' in content:
            for i, line in enumerate(lines):
                if 'Mean intensity:' in line:
                    # Get the section header
                    if i > 0:
                        section = lines[i-1].strip().rstrip(':')
                        mean_val = float(line.split(':')[1].strip())
                        
                        # Find std and range
                        std_val = float(lines[i+1].split(':')[1].strip()) if i+1 < len(lines) else 0
                        range_line = lines[i+2].split(':')[1].strip() if i+2 < len(lines) else "0-0"
                        min_val, max_val = map(float, range_line.split('-'))
                        count = int(lines[i+3].split(':')[1].strip()) if i+3 < len(lines) else 0
                        
                        stats[section.lower()] = {
                            'mean': mean_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'count': count
                        }
        
        # Reconstruct approximate annotation intensities for visualization
        if 'air' in stats:
            # Generate points around the mean for visualization
            np.random.seed(42)  # For reproducible results
            air_intensities = np.random.normal(stats['air']['mean'], stats['air']['std'], stats['air']['count']).tolist()
            bacteria_intensities = np.random.normal(stats['bacteria']['mean'], stats['bacteria']['std'], stats['bacteria']['count']).tolist()
            whewellite_intensities = np.random.normal(stats['whewellite']['mean'], stats['whewellite']['std'], stats['whewellite']['count']).tolist()
            
            print(f"Air: mean={stats['air']['mean']:.1f}, std={stats['air']['std']:.1f}, range={stats['air']['min']:.0f}-{stats['air']['max']:.0f}, n={stats['air']['count']}")
            print(f"Bacteria: mean={stats['bacteria']['mean']:.1f}, std={stats['bacteria']['std']:.1f}, range={stats['bacteria']['min']:.0f}-{stats['bacteria']['max']:.0f}, n={stats['bacteria']['count']}")
            print(f"Whewellite: mean={stats['whewellite']['mean']:.1f}, std={stats['whewellite']['std']:.1f}, range={stats['whewellite']['min']:.0f}-{stats['whewellite']['max']:.0f}, n={stats['whewellite']['count']}")
            
            # Get thresholds from report
            if 'air_stone:' in content:
                current_threshold = float([l for l in content.split('\n') if 'air_stone:' in l][0].split(':')[1].strip())
        
    if current_threshold is None and air_intensities and (bacteria_intensities or whewellite_intensities):
        # Calculate threshold
        air_max = np.mean(air_intensities) + 2 * np.std(air_intensities)
        stone_min = min(np.mean(bacteria_intensities) if bacteria_intensities else float('inf'),
                       np.mean(whewellite_intensities) if whewellite_intensities else float('inf'))
        stone_min -= 2 * (np.std(bacteria_intensities) if bacteria_intensities else 0)
        
        current_threshold = (air_max + stone_min) / 2
        
    if current_threshold is not None:
        print(f"\nCurrent threshold calculation:")
        air_max = np.mean(air_intensities) + 2 * np.std(air_intensities)
        stone_min = min(np.mean(bacteria_intensities) if bacteria_intensities else float('inf'),
                       np.mean(whewellite_intensities) if whewellite_intensities else float('inf'))
        stone_min -= 2 * (np.std(bacteria_intensities) if bacteria_intensities else 0)
        print(f"Air max (mean + 2*std): {air_max:.1f}")
        print(f"Stone min (mean - 2*std): {stone_min:.1f}")
        print(f"Calculated air threshold: {current_threshold:.1f}")
        
        # Check how many pixels this would classify
        air_pixels = np.sum(ct_image <= current_threshold)
        stone_pixels = np.sum(ct_image > current_threshold)
        total_pixels = ct_image.size
        
        print(f"\nPixel classification with current threshold:")
        print(f"Air pixels: {air_pixels:,} ({air_pixels/total_pixels*100:.1f}%)")
        print(f"Stone pixels: {stone_pixels:,} ({stone_pixels/total_pixels*100:.1f}%)")
    
    # Image intensity statistics
    print(f"\nImage Intensity Statistics:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(ct_image.flatten(), p)
        print(f"{p:2d}th percentile: {value:.0f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(ct_image, cmap='gray')
    axes[0, 0].set_title('Original CT Image')
    axes[0, 0].axis('off')
    
    # Image histogram
    axes[0, 1].hist(ct_image.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Full Image Histogram')
    axes[0, 1].set_yscale('log')
    
    # Add current threshold line if available
    if current_threshold is not None:
        axes[0, 1].axvline(current_threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Air Threshold: {current_threshold:.0f}')
        axes[0, 1].legend()
    
    # Annotation analysis if available
    if air_intensities and bacteria_intensities and whewellite_intensities:
        axes[0, 2].hist(air_intensities, alpha=0.7, label='Air', color='green', bins=20, range=(5000, 20000))
        axes[0, 2].hist(bacteria_intensities, alpha=0.7, label='Bacteria', color='blue', bins=20, range=(5000, 20000))
        axes[0, 2].hist(whewellite_intensities, alpha=0.7, label='Whewellite', color='red', bins=20, range=(5000, 20000))
        if current_threshold is not None:
            axes[0, 2].axvline(current_threshold, color='black', linestyle='--', linewidth=2, 
                              label=f'Air Threshold: {current_threshold:.0f}')
        axes[0, 2].set_xlabel('Intensity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Manual Annotation Histograms')
        axes[0, 2].legend()
        
        # Test different thresholds
        test_thresholds = [6000, 7000, 8000, 9000, 10000]
        air_fractions = []
        
        for thresh in test_thresholds:
            air_count = np.sum(ct_image <= thresh)
            air_fraction = air_count / ct_image.size * 100
            air_fractions.append(air_fraction)
        
        axes[1, 0].plot(test_thresholds, air_fractions, 'o-', linewidth=2, markersize=8)
        axes[1, 0].axhline(5, color='red', linestyle='--', alpha=0.7, label='5% (typical air)')
        axes[1, 0].axhline(20, color='orange', linestyle='--', alpha=0.7, label='20% (high air)')
        if current_threshold is not None:
            # Find air fraction for current threshold
            current_air_fraction = np.sum(ct_image <= current_threshold) / ct_image.size * 100
            axes[1, 0].axvline(current_threshold, color='black', linestyle='--', alpha=0.7, 
                              label=f'Current: {current_threshold:.0f} ({current_air_fraction:.1f}%)')
        axes[1, 0].set_xlabel('Threshold Value')
        axes[1, 0].set_ylabel('Air Fraction (%)')
        axes[1, 0].set_title('Air Fraction vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Alternative threshold suggestions
        axes[1, 1].axis('off')
        
        # Suggest alternative thresholds
        suggestions = []
        
        # Option 1: Use minimum of all annotation types
        min_all = min(min(air_intensities), min(bacteria_intensities), min(whewellite_intensities))
        suggestions.append(f"Min of all annotations: {min_all:.0f}")
        
        # Option 2: Use 5th percentile
        thresh_5pct = np.percentile(ct_image.flatten(), 5)
        suggestions.append(f"5th percentile: {thresh_5pct:.0f}")
        
        # Option 3: Use mean of air minus 1 std
        air_low = np.mean(air_intensities) - np.std(air_intensities)
        suggestions.append(f"Air mean - 1 std: {air_low:.0f}")
        
        # Option 4: Manual inspection value
        if current_threshold is not None:
            suggestions.append(f"Current calc: {current_threshold:.0f}")
        
        suggestion_text = "Alternative Thresholds:\n\n" + "\n".join(suggestions)
        suggestion_text += f"\n\nIssue: Air mean ({np.mean(air_intensities):.0f}) > Bacteria mean ({np.mean(bacteria_intensities):.0f})"
        suggestion_text += "\nThis suggests air regions may be\ndenser than expected, or\nannotations need review."
        
        axes[1, 1].text(0.05, 0.95, suggestion_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Show threshold effect visualization
        if current_threshold is not None:
            # Create a mask showing what would be classified as air
            air_mask = ct_image <= current_threshold
            visualization = np.zeros((*ct_image.shape, 3))  # RGB image
            
            # Show air in green, stone in grayscale
            visualization[air_mask] = [0, 1, 0]  # Green for air
            visualization[~air_mask] = np.stack([ct_image[~air_mask]/ct_image.max()]*3, axis=-1)  # Grayscale for stone
            
            axes[1, 2].imshow(visualization)
            axes[1, 2].set_title(f'Air Detection Result\n(Green = Air, Threshold = {current_threshold:.0f})')
            axes[1, 2].axis('off')
    else:
        # No annotation data available
        axes[0, 2].text(0.5, 0.5, 'No annotation data\navailable', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Annotation Data')
        
        axes[1, 0].text(0.5, 0.5, 'No annotation data\navailable', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Threshold Analysis')
        
        axes[1, 1].text(0.5, 0.5, 'Run interactive_annotation.py\nto create manual annotations', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Recommendations')
        
        axes[1, 2].text(0.5, 0.5, 'No threshold\navailable', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Air Detection Result')
    
    plt.tight_layout()
    plt.savefig('threshold_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Suggest corrective action
    print(f"\n" + "="*50)
    print("DIAGNOSTIC SUMMARY:")
    print("="*50)
    
    if air_intensities:
        print(f"Issue identified: Air mean intensity ({np.mean(air_intensities):.0f}) > Bacteria mean ({np.mean(bacteria_intensities):.0f})")
        print("This is unusual and suggests:")
        print("1. Air regions might be imaging artifacts or dense material")
        print("2. Manual annotations may need review")
        print("3. Alternative thresholding approach needed")
        if current_threshold is not None:
            air_pixels = np.sum(ct_image <= current_threshold)
            print(f"\nWith current threshold {current_threshold:.0f}:")
            print(f"Air pixels detected: {air_pixels:,} ({air_pixels/ct_image.size*100:.1f}%)")
            if air_pixels == 0:
                print("*** NO AIR PIXELS DETECTED - THRESHOLD TOO LOW ***")
                print(f"Try threshold around {np.min(air_intensities):.0f} or higher")
    else:
        print("No annotation data found.")
        print("Run 'python interactive_annotation.py' to create manual annotations.")
    
    return ct_image, annotations

if __name__ == "__main__":
    ct_image, annotations = diagnose_thresholds() 