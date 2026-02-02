#!/usr/bin/env python3
"""
CT-Raman Analysis Launcher (PySide6 GUI)
Main entry point for running various analysis tools and applications.
"""

import sys
import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QListWidget, 
                               QTextEdit, QTabWidget, QFrame, QSplitter, QMessageBox,
                               QListWidgetItem)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT-Raman Analysis Launcher")
        self.setGeometry(100, 100, 1200, 750)
        
        # Get the project root directory
        self.project_root = Path(__file__).parent
        
        # Define script paths and categories in workflow order
        self.tools = {
            "🔬 Main Analysis": [
                ("Stone Analysis Standalone (Modern)", "src/interactive/stone_analysis_standalone_qt.py",
                 "FAST Qt interface - DoG/threshold tuning, bacteria vs whewellite, line scans, export"),
                ("Stone Analysis Standalone (Classic)", "src/interactive/stone_analysis_standalone.py",
                 "Original matplotlib interface (slower but stable)"),
            ],
            "🎯 Training Data (Optional)": [
                ("Interactive Annotation", "src/interactive/interactive_annotation.py",
                 "Create custom training annotations (bacteria, whewellite, air) to improve threshold detection"),
            ],
        }
        
        # Tools hidden from launcher (all functionality in Stone Analysis Standalone)
        self.hidden_tools = {
            # Analysis tools - functionality in standalone
            "CT Enhancement": "src/analysis/ct_enhancement.py",
            "Enhanced Stone Analysis": "src/analysis/enhanced_stone_analysis.py",
            "Stone Layer Analysis": "src/analysis/stone_layer_analysis.py",
            "CT-Raman Correlation": "src/analysis/ct_raman_correlation.py",
            "Dog Stone Isolation": "src/analysis/dog_stone_isolation.py",
            # Interactive tools - redundant versions
            "Interactive Stone Tuning": "src/interactive/interactive_stone_tuning.py",
            "Stone Analysis Widget": "src/interactive/stone_analysis_widget.py",
            "Stone Analysis App": "src/interactive/stone_analysis_app.py",
            # Diagnostic tools
            "Compare Isolation Methods": "src/utils/compare_isolation_methods.py",
            "Threshold Diagnostic": "src/utils/threshold_diagnostic.py",
        }
        
        self.current_runner = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Content area with splitter
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Workflow sidebar
        workflow_sidebar = self.create_workflow_sidebar()
        content_splitter.addWidget(workflow_sidebar)
        
        # Main content (tools and log)
        main_content = self.create_main_content()
        content_splitter.addWidget(main_content)
        
        # Set splitter proportions
        content_splitter.setStretchFactor(0, 1)  # Sidebar
        content_splitter.setStretchFactor(1, 4)  # Main content
        
        main_layout.addWidget(content_splitter)
        
        # Bottom buttons
        bottom_frame = self.create_bottom_buttons()
        main_layout.addWidget(bottom_frame)
        
    def create_header(self):
        """Create the header section."""
        header = QFrame()
        header.setStyleSheet("background-color: #2c3e50; color: white;")
        header.setFixedHeight(80)
        
        layout = QVBoxLayout(header)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("CT-Raman Kidney Stone Analysis")
        title.setFont(QFont("Helvetica", 20, QFont.Bold))
        title.setStyleSheet("color: white;")
        title.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        return header
        
    def create_workflow_sidebar(self):
        """Create the workflow sidebar."""
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #ecf0f1;")
        sidebar.setFixedWidth(280)
        
        title = QLabel(
            sidebar,
            text="Analysis Workflow",
            font=QFont("Helvetica", 14, QFont.Bold),
        )
        title.setStyleSheet("color: #2c3e50; background-color: #ecf0f1;")
        title.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout(sidebar)
        layout.addWidget(title)
        layout.addSpacing(15)
        
        # Workflow steps - simplified
        workflow_steps = [
            ("1", "Load CT Image", "#3498db"),
            ("2", "DoG Filtering", "#9b59b6"),
            ("3", "Stone Isolation", "#e67e22"),
            ("4", "Density Mapping", "#27ae60"),
            ("5", "Composition Tuning", "#f39c12"),
            ("6", "Line Scan Analysis", "#16a085"),
            ("7", "Export Results", "#c0392b"),
        ]
        
        for num, name, color in workflow_steps:
            step_widget = self.create_workflow_step(num, name, color)
            layout.addWidget(step_widget)
            layout.addSpacing(5)
        
        layout.addSpacing(15)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #bdc3c7;")
        separator.setFixedHeight(2)
        layout.addWidget(separator)
        layout.addSpacing(15)
        
        # Composition types
        comp_title = QLabel("Composition Types")
        comp_title.setFont(QFont("Helvetica", 14, QFont.Bold))
        comp_title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(comp_title)
        layout.addSpacing(10)
        
        compositions = [
            ("Pure Bacteria", "0.95 g/cm³", "#8B0000"),
            ("Bacteria-Rich", "1.15 g/cm³", "#DC143C"),
            ("Intergrowth", "1.45 g/cm³", "#FFA500"),
            ("Whewellite-Rich", "1.75 g/cm³", "#ADFF2F"),
            ("Pure Whewellite", "2.23 g/cm³", "#00FF00"),
        ]
        
        for name, density, color in compositions:
            comp_widget = self.create_composition_item(name, density, color)
            layout.addWidget(comp_widget)
            layout.addSpacing(3)
        
        layout.addStretch()
        return sidebar
        
    def create_workflow_step(self, num, name, color):
        """Create a workflow step widget."""
        widget = QFrame()
        widget.setStyleSheet("background-color: transparent;")
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Number badge
        badge = QLabel(num)
        badge.setFixedSize(35, 35)
        badge.setAlignment(Qt.AlignCenter)
        badge.setFont(QFont("Helvetica", 12, QFont.Bold))
        badge.setStyleSheet(f"""
            background-color: {color};
            color: white;
            border-radius: 17px;
            border: 2px solid {color};
        """)
        
        # Step name
        label = QLabel(name)
        label.setFont(QFont("Helvetica", 11))
        label.setStyleSheet("color: #2c3e50;")
        
        layout.addWidget(badge)
        layout.addSpacing(10)
        layout.addWidget(label)
        layout.addStretch()
        
        return widget
        
    def create_composition_item(self, name, density, color):
        """Create a composition type item."""
        widget = QFrame()
        widget.setStyleSheet("background-color: transparent;")
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Color box
        color_box = QLabel()
        color_box.setFixedSize(20, 20)
        color_box.setStyleSheet(f"""
            background-color: {color};
            border: 1px solid #7f8c8d;
        """)
        
        # Info label
        info = QLabel(f"{name}\n{density}")
        info.setFont(QFont("Helvetica", 9))
        info.setStyleSheet("color: #2c3e50;")
        
        layout.addWidget(color_box)
        layout.addSpacing(8)
        layout.addWidget(info)
        layout.addStretch()
        
        return widget
        
    def create_main_content(self):
        """Create the main content area with tools and log."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Left: Tool selection
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        tools_label = QLabel("Select a tool to launch:")
        tools_label.setFont(QFont("Helvetica", 13, QFont.Bold))
        left_layout.addWidget(tools_label)
        
        # Tab widget for categories
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                background: white;
            }
            QTabBar::tab {
                background: #ecf0f1;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
        """)
        
        self.list_widgets = {}
        for category, tools in self.tools.items():
            list_widget = QListWidget()
            list_widget.setFont(QFont("Helvetica", 11))
            list_widget.setStyleSheet("""
                QListWidget {
                    border: none;
                    background: white;
                }
                QListWidget::item {
                    padding: 8px;
                }
                QListWidget::item:selected {
                    background: #3498db;
                    color: white;
                }
                QListWidget::item:hover {
                    background: #ecf0f1;
                }
            """)
            
            for tool_info in tools:
                tool_name = tool_info[0]
                tool_desc = tool_info[2] if len(tool_info) > 2 else ""
                item = QListWidgetItem(tool_name)
                if tool_desc:
                    item.setToolTip(tool_desc)
                list_widget.addItem(item)
            
            list_widget.itemDoubleClicked.connect(lambda item, cat=category: self.launch_selected(cat))
            self.list_widgets[category] = list_widget
            
            # Container for list and button
            tab_container = QWidget()
            tab_layout = QVBoxLayout(tab_container)
            tab_layout.setContentsMargins(5, 5, 5, 5)
            
            tab_layout.addWidget(list_widget)
            
            # Launch button
            launch_btn = QPushButton("Launch Selected")
            launch_btn.setFont(QFont("Helvetica", 11, QFont.Bold))
            launch_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: black;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                }
            """)
            launch_btn.clicked.connect(lambda checked, cat=category: self.launch_selected(cat))
            tab_layout.addWidget(launch_btn, alignment=Qt.AlignRight)
            
            self.tab_widget.addTab(tab_container, category)
        
        left_layout.addWidget(self.tab_widget)
        
        # Right: Output log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        log_label = QLabel("Output Log:")
        log_label.setFont(QFont("Helvetica", 13, QFont.Bold))
        right_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Use system monospace font
        mono_font = QFont()
        mono_font.setStyleHint(QFont.Monospace)
        mono_font.setFamily("Monaco")  # macOS default monospace
        mono_font.setPointSize(9)
        self.log_text.setFont(mono_font)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
            }
        """)
        right_layout.addWidget(self.log_text)
        
        # Add to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        # Initial log message
        self.log("CT-Raman Analysis Launcher initialized.")
        self.log("Select a tool from the tabs and click 'Launch Selected' or double-click.")
        
        return widget
        
    def create_bottom_buttons(self):
        """Create bottom button bar."""
        frame = QFrame()
        frame.setStyleSheet("background-color: #ecf0f1;")
        frame.setFixedHeight(60)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Clear log button
        clear_btn = QPushButton("Clear Log")
        clear_btn.setFont(QFont("Helvetica", 11))
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: black;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        clear_btn.clicked.connect(self.clear_log)
        
        # Exit button
        exit_btn = QPushButton("Exit")
        exit_btn.setFont(QFont("Helvetica", 11, QFont.Bold))
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: black;
                border: none;
                padding: 10px 25px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        exit_btn.clicked.connect(self.close)
        
        layout.addWidget(clear_btn)
        layout.addStretch()
        layout.addWidget(exit_btn)
        
        return frame
        
    def launch_selected(self, category):
        """Launch the selected tool."""
        list_widget = self.list_widgets[category]
        current_item = list_widget.currentItem()
        
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a tool to launch.")
            return
        
        tool_name = current_item.text()
        script_path = None
        
        for tool_info in self.tools[category]:
            name = tool_info[0]
            path = tool_info[1]
            if name == tool_name:
                script_path = self.project_root / path
                break
        
        if not script_path or not script_path.exists():
            self.log(f"❌ Error: Script not found at {script_path}")
            QMessageBox.critical(self, "File Not Found", f"Script not found:\n{script_path}")
            return
        
        self.log(f"\n{'='*60}")
        self.log(f"🚀 Launching: {tool_name}")
        self.log(f"   Path: {script_path.relative_to(self.project_root)}")
        self.log(f"{'='*60}")
        
        # Run in thread
        self.current_runner = ProcessRunner(script_path, self.project_root)
        self.current_runner.output_signal.connect(self.log)
        self.current_runner.finished_signal.connect(self.on_process_finished)
        self.current_runner.start()
        
    def on_process_finished(self, return_code, script_path):
        """Handle process completion."""
        if return_code == 0:
            self.log(f"✅ Process completed successfully.")
        else:
            self.log(f"❌ Process exited with code {return_code}")
        
    def log(self, message):
        """Add a message to the log."""
        self.log_text.append(message)
        
    def clear_log(self):
        """Clear the log."""
        self.log_text.clear()
        self.log("Log cleared.")
    
    def closeEvent(self, event):
        """Handle window close event and cleanup threads."""
        if self.current_runner and self.current_runner.isRunning():
            self.current_runner.terminate()
            self.current_runner.wait()
        event.accept()

def main():
    """Main launcher function."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = LauncherGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
