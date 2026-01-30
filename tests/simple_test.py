#!/usr/bin/env python3
"""
Simple test of the stone analysis without complex widgets
Run this to verify everything is working correctly
"""

import numpy as np
import matplotlib.pyplot as plt
from stone_analysis_widget import StoneAnalysisWidget

def test_analysis():
    """Test the stone analysis with simple plots"""
    print("🔬 Testing CT-Raman Stone Analysis...")
    
    # Create widget (this does the analysis automatically)
    widget = StoneAnalysisWidget()
    
    # Create simple plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CT-Raman Stone Analysis Results', fontsize=16, fontweight='bold')
    
    # Get current analysis results
    ct_image = widget.ct_image
    dog_enhanced = widget.dog_enhanced
    stone_mask = widget.stone_mask
    composition_map = widget.composition_map
    
    # Plot results
    axes[0, 0].imshow(ct_image, cmap='gray')
    axes[0, 0].set_title('Original CT')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(dog_enhanced, cmap='gray')
    axes[0, 1].set_title('DoG Enhanced')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(stone_mask, cmap='gray')
    stone_coverage = np.sum(stone_mask) / stone_mask.size * 100
    axes[0, 2].set_title(f'Stone Mask ({stone_coverage:.1f}%)')
    axes[0, 2].axis('off')
    
    # Composition map with colormap
    colors = widget.colormaps['original']
    from matplotlib.colors import ListedColormap
    cmap_zones = ListedColormap(colors)
    
    axes[1, 0].imshow(composition_map, cmap=cmap_zones, vmin=0, vmax=5)
    axes[1, 0].set_title('Composition Zones')
    axes[1, 0].axis('off')
    
    # Statistics plot
    if np.sum(stone_mask) > 0:
        zone_counts = [np.sum(composition_map == i) for i in range(1, 6)]
        zone_names = ['Pure\nBacteria', 'Bacteria-\nRich', 'Intergrowth', 'Whewellite-\nRich', 'Pure\nWhewellite']
        total_stone = sum(zone_counts)
        zone_percentages = [count/total_stone*100 if total_stone > 0 else 0 for count in zone_counts]
        
        axes[1, 1].pie(zone_percentages, labels=zone_names, autopct='%1.1f%%', 
                      colors=colors[1:6], startangle=90)
        axes[1, 1].set_title('Composition Distribution')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Analysis complete!")
    print("If you can see the plots above, everything is working correctly.")
    print("You can now try the full widget interface in Jupyter!")

if __name__ == "__main__":
    test_analysis() 