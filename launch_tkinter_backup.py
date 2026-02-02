#!/usr/bin/env python3
"""
CT-Raman Analysis Launcher (GUI)
Main entry point for running various analysis tools and applications.
"""

import sys
import os
import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QListWidget, 
                               QTextEdit, QTabWidget, QFrame, QSplitter, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, QProcess
from PySide6.QtGui import QFont, QColor, QPalette

class ProcessRunner(QThread):
    """Thread for running scripts without blocking the GUI."""
    output_signal = Signal(str)
    finished_signal = Signal(int, str)
    
    def __init__(self, script_path, project_root):
        super().__init__()
        self.script_path = script_path
        self.project_root = project_root
        
    def run(self):
        """Run the script in a subprocess."""
        try:
            process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.project_root)
            )
            
            for line in process.stdout:
                self.output_signal.emit(line.rstrip())
            
            process.wait()
            self.finished_signal.emit(process.returncode, str(self.script_path))
        except Exception as e:
            self.output_signal.emit(f"Error: {e}")
            self.finished_signal.emit(-1, str(self.script_path))

class LauncherGUI(QMainWindow):
    def __init__(self, root):
        self.root = root
        self.root.title("CT-Raman Analysis Launcher")
        self.root.geometry("1100x700")
        self.root.resizable(True, True)
        
        # Get the project root directory
        self.project_root = Path(__file__).parent
        
        # Define script paths and categories
        self.tools = {
            "Analysis Tools": [
                ("CT Enhancement", "src/analysis/ct_enhancement.py"),
                ("CT-Raman Correlation", "src/analysis/ct_raman_correlation.py"),
                ("Dog Stone Isolation", "src/analysis/dog_stone_isolation.py"),
                ("Enhanced Stone Analysis", "src/analysis/enhanced_stone_analysis.py"),
                ("Stone Layer Analysis", "src/analysis/stone_layer_analysis.py"),
            ],
            "Interactive Tools": [
                ("Stone Analysis Standalone", "src/interactive/stone_analysis_standalone.py"),
                ("Stone Analysis Widget", "src/interactive/stone_analysis_widget.py"),
                ("Stone Analysis App", "src/interactive/stone_analysis_app.py"),
                ("Interactive Annotation", "src/interactive/interactive_annotation.py"),
                ("Interactive Stone Tuning", "src/interactive/interactive_stone_tuning.py"),
            ],
            "Utilities": [
                ("Compare Isolation Methods", "src/utils/compare_isolation_methods.py"),
                ("Threshold Diagnostic", "src/utils/threshold_diagnostic.py"),
            ],
            "Tests": [
                ("Run Stone Analysis", "tests/run_stone_analysis.py"),
                ("Simple Test", "tests/simple_test.py"),
            ],
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="CT-Raman Kidney Stone Analysis",
            font=("Helvetica", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Workflow sidebar
        workflow_frame = tk.Frame(self.root, bg="#ecf0f1", width=250)
        workflow_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        workflow_frame.pack_propagate(False)
        
        self.create_workflow_sidebar(workflow_frame)
        
        # Main content area
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Tool selection
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(
            left_frame,
            text="Select a tool to launch:",
            font=("Helvetica", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Create notebook for categories
        notebook = ttk.Notebook(left_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add tabs for each category
        for category, tools in self.tools.items():
            tab_frame = tk.Frame(notebook)
            notebook.add(tab_frame, text=category)
            
            # Create listbox for tools in this category
            listbox = tk.Listbox(
                tab_frame,
                font=("Helvetica", 11),
                selectmode=tk.SINGLE,
                activestyle='dotbox'
            )
            listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add tools to listbox
            for tool_name, script_path in tools:
                listbox.insert(tk.END, tool_name)
            
            # Bind double-click to launch
            listbox.bind('<Double-Button-1>', 
                        lambda e, cat=category, lb=listbox: self.launch_selected(cat, lb))
            
            # Add launch button
            btn_frame = tk.Frame(tab_frame)
            btn_frame.pack(fill=tk.X, padx=5, pady=5)
            
            tk.Button(
                btn_frame,
                text="Launch Selected",
                command=lambda cat=category, lb=listbox: self.launch_selected(cat, lb),
                bg="#3498db",
                fg="black",
                font=("Helvetica", 10, "bold"),
                padx=20,
                pady=5
            ).pack(side=tk.RIGHT)
        
        # Right panel - Output log
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(
            right_frame,
            text="Output Log:",
            font=("Helvetica", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(
            right_frame,
            font=("Courier", 9),
            bg="#f8f9fa",
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom button frame
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            bottom_frame,
            text="Clear Log",
            command=self.clear_log,
            bg="#95a5a6",
            fg="black",
            font=("Helvetica", 10),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT)
        
        tk.Button(
            bottom_frame,
            text="Exit",
            command=self.root.quit,
            bg="#e74c3c",
            fg="black",
            font=("Helvetica", 10, "bold"),
            padx=20,
            pady=5
        ).pack(side=tk.RIGHT)
        
        # Initial log message
        self.log("CT-Raman Analysis Launcher initialized.")
        self.log("Select a tool from the tabs and click 'Launch Selected' or double-click.")
    
    def create_workflow_sidebar(self, parent):
        """Create the workflow sidebar showing the analysis pipeline."""
        # Title
        title = tk.Label(
            parent,
            text="Analysis Workflow",
            font=("Helvetica", 14, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        title.pack(pady=(10, 15))
        
        # Workflow steps
        workflow_steps = [
            ("1", "Load CT Image", "#3498db"),
            ("2", "Stone Isolation", "#9b59b6"),
            ("3", "Density Mapping", "#e67e22"),
            ("4", "Composition Classification", "#27ae60"),
            ("5", "Visualization", "#f39c12"),
            ("6", "Line Scan Analysis", "#16a085"),
            ("7", "Export Results", "#c0392b"),
        ]
        
        for step_num, step_name, color in workflow_steps:
            step_frame = tk.Frame(parent, bg="#ecf0f1")
            step_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Step number circle
            num_label = tk.Label(
                step_frame,
                text=step_num,
                font=("Helvetica", 12, "bold"),
                bg=color,
                fg="white",
                width=3,
                height=1,
                relief=tk.RAISED,
                borderwidth=2
            )
            num_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Step name
            name_label = tk.Label(
                step_frame,
                text=step_name,
                font=("Helvetica", 10),
                bg="#ecf0f1",
                fg="#2c3e50",
                anchor=tk.W
            )
            name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Separator
        separator = tk.Frame(parent, bg="#bdc3c7", height=2)
        separator.pack(fill=tk.X, padx=10, pady=20)
        
        # Composition categories info
        info_title = tk.Label(
            parent,
            text="Composition Types",
            font=("Helvetica", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        info_title.pack(pady=(0, 10))
        
        compositions = [
            ("Pure Bacteria", "0.95 g/cm³", "#8B0000"),
            ("Bacteria-Rich", "1.15 g/cm³", "#DC143C"),
            ("Intergrowth", "1.45 g/cm³", "#FFA500"),
            ("Whewellite-Rich", "1.75 g/cm³", "#ADFF2F"),
            ("Pure Whewellite", "2.23 g/cm³", "#00FF00"),
        ]
        
        for comp_name, density, color in compositions:
            comp_frame = tk.Frame(parent, bg="#ecf0f1")
            comp_frame.pack(fill=tk.X, padx=10, pady=3)
            
            # Color indicator
            color_box = tk.Label(
                comp_frame,
                text="  ",
                bg=color,
                width=2,
                relief=tk.SOLID,
                borderwidth=1
            )
            color_box.pack(side=tk.LEFT, padx=(0, 8))
            
            # Composition info
            info_label = tk.Label(
                comp_frame,
                text=f"{comp_name}\n{density}",
                font=("Helvetica", 8),
                bg="#ecf0f1",
                fg="#2c3e50",
                anchor=tk.W,
                justify=tk.LEFT
            )
            info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def log(self, message):
        """Add a message to the log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()
        
    def clear_log(self):
        """Clear the log text."""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared.")
        
    def launch_selected(self, category, listbox):
        """Launch the selected tool."""
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a tool to launch.")
            return
        
        index = selection[0]
        tool_name, script_path = self.tools[category][index]
        full_path = self.project_root / script_path
        
        if not full_path.exists():
            self.log(f"❌ Error: Script not found at {full_path}")
            messagebox.showerror("File Not Found", f"Script not found:\n{full_path}")
            return
        
        self.log(f"\n{'='*60}")
        self.log(f"🚀 Launching: {tool_name}")
        self.log(f"   Path: {script_path}")
        self.log(f"{'='*60}")
        
        # Launch in a separate thread to avoid blocking the GUI
        thread = threading.Thread(target=self.run_script, args=(full_path, tool_name))
        thread.daemon = True
        thread.start()
        
    def run_script(self, script_path, tool_name):
        """Run a Python script in a subprocess."""
        try:
            # Run the script as a subprocess
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.project_root)
            )
            
            # Read output in real-time
            for line in process.stdout:
                self.log(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                self.log(f"✅ {tool_name} completed successfully.")
            else:
                self.log(f"❌ {tool_name} exited with code {process.returncode}")
                
        except Exception as e:
            self.log(f"❌ Error running {tool_name}: {e}")
            messagebox.showerror("Execution Error", f"Error running {tool_name}:\n{e}")

def main():
    """Main launcher function."""
    root = tk.Tk()
    app = LauncherGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
