import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import pickle
from PIL import Image
import cv2

class InteractiveAnnotator:
    def __init__(self, ct_image_path):
        """
        Interactive annotation tool for kidney stone CT images
        
        Args:
            ct_image_path: Path to the CT image
        """
        self.ct_image_path = ct_image_path
        self.ct_image = None
        self.annotations = []
        self.current_label = 'whewellite'
        self.load_image()
        self.setup_interface()
        
    def load_image(self):
        """Load the CT image"""
        if self.ct_image_path.endswith('.tif'):
            self.ct_image = np.array(Image.open(self.ct_image_path))
        else:
            self.ct_image = cv2.imread(self.ct_image_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"Image loaded: {self.ct_image.shape}")
        
    def setup_interface(self):
        """Setup the interactive interface"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Display the CT image
        self.im = self.ax.imshow(self.ct_image, cmap='gray', interpolation='nearest')
        self.ax.set_title('Click to Annotate Regions\n(Red=Whewellite, Blue=Bacteria, Green=Air)')
        self.ax.axis('off')
        
        # Add radio buttons for label selection
        rax = plt.axes([0.02, 0.7, 0.15, 0.15])
        self.radio = RadioButtons(rax, ('whewellite', 'bacteria', 'air'))
        self.radio.on_clicked(self.set_label)
        
        # Add control buttons
        ax_save = plt.axes([0.02, 0.5, 0.1, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save_annotations)
        
        ax_load = plt.axes([0.02, 0.45, 0.1, 0.04])
        self.btn_load = Button(ax_load, 'Load')
        self.btn_load.on_clicked(self.load_annotations)
        
        ax_clear = plt.axes([0.02, 0.4, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear_annotations)
        
        ax_analyze = plt.axes([0.02, 0.35, 0.1, 0.04])
        self.btn_analyze = Button(ax_analyze, 'Analyze')
        self.btn_analyze.on_clicked(self.run_analysis)
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add instructions
        instructions = """
Instructions:
1. Select label type (radio buttons)
2. Click on image regions
3. Red dots = Whewellite
4. Blue dots = Bacteria  
5. Green dots = Air
6. Save when done
        """
        
        self.ax.text(0.02, 0.25, instructions, transform=self.fig.transFigure, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
    def set_label(self, label):
        """Set the current annotation label"""
        self.current_label = label
        print(f"Current label: {label}")
        
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Check if click is within image bounds
        if 0 <= x < self.ct_image.shape[1] and 0 <= y < self.ct_image.shape[0]:
            # Get intensity value at clicked point
            intensity = self.ct_image[y, x]
            
            # Store annotation
            annotation = {
                'x': x,
                'y': y,
                'label': self.current_label,
                'intensity': intensity
            }
            self.annotations.append(annotation)
            
            # Plot the annotation point
            colors = {'whewellite': 'red', 'bacteria': 'blue', 'air': 'green'}
            self.ax.plot(x, y, 'o', color=colors[self.current_label], 
                        markersize=8, markeredgecolor='white', markeredgewidth=1)
            
            self.fig.canvas.draw()
            
            print(f"Annotated: ({x}, {y}) as {self.current_label}, intensity: {intensity}")
            print(f"Total annotations: {len(self.annotations)}")
    
    def save_annotations(self, event):
        """Save annotations to file"""
        filename = 'stone_annotations.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.annotations, f)
        print(f"Saved {len(self.annotations)} annotations to {filename}")
        
        # Also save as text file for readability
        text_filename = 'stone_annotations.txt'
        with open(text_filename, 'w') as f:
            f.write("x\ty\tlabel\tintensity\n")
            for ann in self.annotations:
                f.write(f"{ann['x']}\t{ann['y']}\t{ann['label']}\t{ann['intensity']}\n")
        print(f"Also saved as text file: {text_filename}")
    
    def load_annotations(self, event):
        """Load annotations from file"""
        try:
            filename = 'stone_annotations.pkl'
            with open(filename, 'rb') as f:
                self.annotations = pickle.load(f)
            
            # Redraw all annotations
            self.ax.clear()
            self.im = self.ax.imshow(self.ct_image, cmap='gray', interpolation='nearest')
            self.ax.set_title('Click to Annotate Regions\n(Red=Whewellite, Blue=Bacteria, Green=Air)')
            self.ax.axis('off')
            
            colors = {'whewellite': 'red', 'bacteria': 'blue', 'air': 'green'}
            for ann in self.annotations:
                self.ax.plot(ann['x'], ann['y'], 'o', color=colors[ann['label']], 
                           markersize=8, markeredgecolor='white', markeredgewidth=1)
            
            self.fig.canvas.draw()
            print(f"Loaded {len(self.annotations)} annotations from {filename}")
            
        except FileNotFoundError:
            print("No annotation file found")
    
    def clear_annotations(self, event):
        """Clear all annotations"""
        self.annotations = []
        
        # Redraw image without annotations
        self.ax.clear()
        self.im = self.ax.imshow(self.ct_image, cmap='gray', interpolation='nearest')
        self.ax.set_title('Click to Annotate Regions\n(Red=Whewellite, Blue=Bacteria, Green=Air)')
        self.ax.axis('off')
        self.fig.canvas.draw()
        
        print("Cleared all annotations")
    
    def run_analysis(self, event):
        """Run analysis using the manual annotations"""
        if len(self.annotations) < 3:
            print("Need annotations for all three categories (whewellite, bacteria, air)")
            return
            
        # Check if we have all three types
        labels_present = set(ann['label'] for ann in self.annotations)
        required_labels = {'whewellite', 'bacteria', 'air'}
        
        if not required_labels.issubset(labels_present):
            missing = required_labels - labels_present
            print(f"Missing annotations for: {missing}")
            return
        
        print("Running analysis with manual annotations...")
        analyzer = AnnotationBasedAnalyzer(self.ct_image, self.annotations)
        analyzer.run_analysis()
    
    def show(self):
        """Display the annotation interface"""
        plt.show()
    
    def get_annotation_summary(self):
        """Get summary of current annotations"""
        if not self.annotations:
            return "No annotations yet"
        
        label_counts = {}
        for ann in self.annotations:
            label = ann['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        summary = "Annotation Summary:\n"
        for label, count in label_counts.items():
            summary += f"  {label}: {count} points\n"
        
        return summary


class AnnotationBasedAnalyzer:
    def __init__(self, ct_image, annotations):
        """
        Analyzer that uses manual annotations for calibration
        
        Args:
            ct_image: The CT image array
            annotations: List of annotation dictionaries
        """
        self.ct_image = ct_image
        self.annotations = annotations
        self.thresholds = {}
        
    def calculate_thresholds(self):
        """Calculate intensity thresholds based on annotations"""
        # Group annotations by label
        intensities = {'whewellite': [], 'bacteria': [], 'air': []}
        
        for ann in self.annotations:
            intensities[ann['label']].append(ann['intensity'])
        
        # Calculate statistics for each group
        stats = {}
        for label, values in intensities.items():
            if values:
                stats[label] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Calculate thresholds (midpoints between groups)
        # Sort by mean intensity: air < bacteria < whewellite (typically)
        sorted_labels = sorted(stats.keys(), key=lambda x: stats[x]['mean'])
        
        self.thresholds = {}
        if len(sorted_labels) >= 2:
            # Threshold between air and stone material
            if 'air' in stats and len(sorted_labels) > 1:
                air_max = stats['air']['mean'] + 2 * stats['air']['std']
                next_label = [l for l in sorted_labels if l != 'air'][0]
                stone_min = stats[next_label]['mean'] - 2 * stats[next_label]['std']
                self.thresholds['air_stone'] = (air_max + stone_min) / 2
            
            # Threshold between bacteria and whewellite
            if 'bacteria' in stats and 'whewellite' in stats:
                bacteria_max = stats['bacteria']['mean'] + stats['bacteria']['std']
                whewellite_min = stats['whewellite']['mean'] - stats['whewellite']['std']
                self.thresholds['bacteria_whewellite'] = (bacteria_max + whewellite_min) / 2
        
        print("Annotation-based Statistics:")
        for label, stat in stats.items():
            print(f"{label}: mean={stat['mean']:.1f}, std={stat['std']:.1f}, "
                  f"range=[{stat['min']}-{stat['max']}], n={stat['count']}")
        
        print("\nCalculated Thresholds:")
        for thresh_name, value in self.thresholds.items():
            print(f"{thresh_name}: {value:.1f}")
        
        return stats, self.thresholds
    
    def create_classification_map(self):
        """Create classification map using annotation-based thresholds"""
        # Calculate thresholds first
        stats, thresholds = self.calculate_thresholds()
        
        # Create classification map
        classification_map = np.zeros_like(self.ct_image)
        
        # Apply thresholds
        if 'air_stone' in thresholds:
            # Separate air from stone
            stone_mask = self.ct_image > thresholds['air_stone']
            
            if 'bacteria_whewellite' in thresholds:
                # Within stone, separate bacteria from whewellite
                whewellite_mask = (self.ct_image > thresholds['bacteria_whewellite']) & stone_mask
                bacteria_mask = (self.ct_image <= thresholds['bacteria_whewellite']) & stone_mask
                
                classification_map[whewellite_mask] = 3  # Whewellite
                classification_map[bacteria_mask] = 2    # Bacteria
                classification_map[~stone_mask] = 1      # Air
            else:
                # Only stone vs air
                classification_map[stone_mask] = 2
                classification_map[~stone_mask] = 1
        
        return classification_map
    
    def run_analysis(self):
        """Run complete analysis with annotation-based calibration"""
        # Calculate thresholds
        stats, thresholds = self.calculate_thresholds()
        
        # Create classification map
        classification_map = self.create_classification_map()
        
        # Visualize results
        self.visualize_results(classification_map, stats)
        
        # Generate report
        self.generate_report(classification_map, stats, thresholds)
    
    def visualize_results(self, classification_map, stats):
        """Visualize annotation-based analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with annotations
        axes[0, 0].imshow(self.ct_image, cmap='gray')
        colors = {'whewellite': 'red', 'bacteria': 'blue', 'air': 'green'}
        for ann in self.annotations:
            axes[0, 0].plot(ann['x'], ann['y'], 'o', color=colors[ann['label']], 
                          markersize=6, markeredgecolor='white', markeredgewidth=1)
        axes[0, 0].set_title('Original Image with Annotations')
        axes[0, 0].axis('off')
        
        # Classification map
        from matplotlib.colors import ListedColormap
        cmap_colors = ['black', 'green', 'red', 'gold']  # background, air, bacteria, whewellite
        custom_cmap = ListedColormap(cmap_colors)
        
        im = axes[0, 1].imshow(classification_map, cmap=custom_cmap, vmin=0, vmax=3)
        axes[0, 1].set_title('Classification Map\n(Green=Air, Red=Bacteria, Gold=Whewellite)')
        axes[0, 1].axis('off')
        
        # Intensity histograms by annotation
        axes[1, 0].hist([ann['intensity'] for ann in self.annotations if ann['label'] == 'air'], 
                       alpha=0.7, label='Air', color='green', bins=20)
        axes[1, 0].hist([ann['intensity'] for ann in self.annotations if ann['label'] == 'bacteria'], 
                       alpha=0.7, label='Bacteria', color='red', bins=20)
        axes[1, 0].hist([ann['intensity'] for ann in self.annotations if ann['label'] == 'whewellite'], 
                       alpha=0.7, label='Whewellite', color='gold', bins=20)
        axes[1, 0].set_xlabel('CT Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Intensity Distribution by Annotation')
        axes[1, 0].legend()
        
        # Classification statistics
        unique, counts = np.unique(classification_map, return_counts=True)
        labels = ['Background', 'Air', 'Bacteria', 'Whewellite']
        
        # Only show categories that exist
        existing_labels = []
        existing_counts = []
        colors_pie = []
        color_map = {0: 'black', 1: 'green', 2: 'red', 3: 'gold'}
        
        for i, count in zip(unique, counts):
            if i < len(labels) and count > 0:
                existing_labels.append(labels[i])
                existing_counts.append(count)
                colors_pie.append(color_map[i])
        
        if existing_counts:
            axes[1, 1].pie(existing_counts, labels=existing_labels, colors=colors_pie, 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Classification Distribution')
        
        plt.tight_layout()
        plt.savefig('annotation_based_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, classification_map, stats, thresholds):
        """Generate analysis report"""
        report = []
        report.append("Annotation-Based Stone Analysis Report")
        report.append("=" * 45)
        
        # Annotation statistics
        report.append("\nManual Annotation Statistics:")
        report.append("-" * 30)
        for label, stat in stats.items():
            report.append(f"{label.capitalize()}:")
            report.append(f"  Mean intensity: {stat['mean']:.1f}")
            report.append(f"  Std deviation:  {stat['std']:.1f}")
            report.append(f"  Range:         {stat['min']}-{stat['max']}")
            report.append(f"  Sample count:   {stat['count']}")
        
        # Calculated thresholds
        report.append("\nCalculated Thresholds:")
        report.append("-" * 20)
        for thresh_name, value in thresholds.items():
            report.append(f"{thresh_name}: {value:.1f}")
        
        # Classification results
        unique, counts = np.unique(classification_map, return_counts=True)
        total_pixels = np.sum(counts)
        
        report.append("\nClassification Results:")
        report.append("-" * 25)
        labels = ['Background', 'Air', 'Bacteria', 'Whewellite']
        for i, count in zip(unique, counts):
            if i < len(labels):
                percentage = count / total_pixels * 100
                report.append(f"{labels[i]}: {count:,} pixels ({percentage:.1f}%)")
        
        # Save report
        with open('annotation_based_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        return report


def main():
    """Main function to run interactive annotation"""
    annotator = InteractiveAnnotator('slice_1092.tif')
    
    print("Interactive Annotation Tool")
    print("=" * 30)
    print("Instructions:")
    print("1. Select the type of region using radio buttons")
    print("2. Click on representative areas in the image")
    print("3. Red dots = Whewellite regions")
    print("4. Blue dots = Bacteria regions") 
    print("5. Green dots = Air/background regions")
    print("6. Click 'Save' to save your annotations")
    print("7. Click 'Analyze' to run calibrated analysis")
    print("\nClick on at least 3-5 points for each category for best results!")
    
    annotator.show()

if __name__ == "__main__":
    main() 