import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit, QHBoxLayout, QProgressBar
)
from PyQt5.QtCore import Qt
import threading
from backend import CADBackend


class CADAnalyzerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏨 CAD Analyzer Pro - Desktop Edition")
        self.setGeometry(100, 100, 1200, 800)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.file_upload_tab = QWidget()
        self.analysis_tab = QWidget()
        self.ilot_tab = QWidget()
        self.corridor_tab = QWidget()
        self.results_tab = QWidget()

        self.tabs.addTab(self.file_upload_tab, "📁 File Upload")
        self.tabs.addTab(self.analysis_tab, "🔍 Analysis")
        self.tabs.addTab(self.ilot_tab, "🏗️ Îlot Placement")
        self.tabs.addTab(self.corridor_tab, "🛤️ Corridor Generation")
        self.tabs.addTab(self.results_tab, "📊 Results & Export")

        self.backend = CADBackend()
        self.analysis_results = None

        self.init_file_upload_tab()
        self.init_analysis_tab()
        self.init_ilot_placement_tab()
        self.init_corridor_generation_tab()
        self.init_results_export_tab()

    def init_file_upload_tab(self):
        layout = QVBoxLayout()
        self.upload_label = QLabel("Upload your floor plan (DXF, DWG, PDF, Image)")
        self.upload_label.setAlignment(Qt.AlignCenter)
        self.upload_btn = QPushButton("Select File")
        self.upload_btn.clicked.connect(self.open_file_dialog)
        self.file_info = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.upload_label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.file_info)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_text)
        self.file_upload_tab.setLayout(layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Floor Plan", "", "CAD Files (*.dxf *.dwg *.pdf);;Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.file_info.setText(f"Selected: {file_path}")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.result_text.setText("")
            # Start async parsing
            def on_parsing_done(results):
                self.progress_bar.setVisible(False)
                if results.get('success'):
                    self.analysis_results = results
                    summary = self.format_results_summary(results)
                    self.result_text.setText(summary)
                    self.update_analysis_tab(results)
                else:
                    self.result_text.setText(f"Error: {results.get('error', 'Unknown error')}")
                    self.update_analysis_tab(None)
            threading.Thread(target=lambda: self.backend.parse_file_async(file_path, on_parsing_done)).start()

    def format_results_summary(self, results):
        lines = []
        lines.append(f"Entities: {results.get('entity_count', 0)}")
        lines.append(f"Walls: {results.get('wall_count', 0)}")
        lines.append(f"Restricted: {results.get('restricted_count', 0)}")
        lines.append(f"Entrances: {results.get('entrance_count', 0)}")
        if 'bounds' in results:
            b = results['bounds']
            lines.append(f"Bounds: X({b['min_x']} - {b['max_x']}), Y({b['min_y']} - {b['max_y']})")
        return '\n'.join(lines)

    def init_analysis_tab(self):
        layout = QVBoxLayout()

        # Title
        self.analysis_title = QLabel("<b>🔍 Floor Plan Analysis</b>")
        self.analysis_title.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.analysis_title)

        # Results area (raw JSON for debug)
        self.analysis_results_text = QTextEdit()
        self.analysis_results_text.setReadOnly(True)
        layout.addWidget(self.analysis_results_text)

        # Unique layers
        self.layers_label = QLabel()
        layout.addWidget(self.layers_label)

        # Validation summary
        self.validation_label = QLabel()
        layout.addWidget(self.validation_label)

        # Plan bounds
        self.bounds_label = QLabel()
        layout.addWidget(self.bounds_label)

        # Visualization (matplotlib)
        self.analysis_canvas = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self.analysis_fig = Figure(figsize=(6, 4))
            self.analysis_canvas = FigureCanvas(self.analysis_fig)
            layout.addWidget(self.analysis_canvas)
        except ImportError:
            layout.addWidget(QLabel("Matplotlib not installed. Visualization unavailable."))

        self.analysis_tab.setLayout(layout)

    def update_analysis_tab(self, results):
        """Update the Analysis tab with parsed results"""
        if not results or not results.get('success'):
            self.analysis_results_text.setText("No analysis results. Please upload and process a floor plan first.")
            self.layers_label.setText("")
            self.validation_label.setText("")
            self.bounds_label.setText("")
            if self.analysis_canvas:
                self.analysis_fig.clear()
                self.analysis_canvas.draw()
            return

        # Show raw results (for debug)
        import json
        self.analysis_results_text.setText(json.dumps({k: v for k, v in results.items() if k != 'entities'}, indent=2))

        # Unique layers (from entities)
        entities = results.get('entities', [])
        layers = sorted({e.get('layer', 'Default') for e in entities if e.get('layer', None) is not None})
        self.layers_label.setText(f"<b>Unique DXF Layers:</b> {', '.join(layers) if layers else 'N/A'}")

        # Zone validation (simple)
        wall_count = results.get('wall_count', 0)
        restricted_count = results.get('restricted_count', 0)
        entrance_count = results.get('entrance_count', 0)
        validation_msgs = []
        validation_msgs.append(f"<b>Walls detected:</b> {'✅' if wall_count else '❌'} ({wall_count})")
        validation_msgs.append(f"<b>Restricted areas detected:</b> {'✅' if restricted_count else '❌'} ({restricted_count})")
        validation_msgs.append(f"<b>Entrances detected:</b> {'✅' if entrance_count else '❌'} ({entrance_count})")
        self.validation_label.setText('<br>'.join(validation_msgs))

        # Plan bounds
        bounds = results.get('bounds', {})
        if bounds:
            w = bounds.get('max_x', 0) - bounds.get('min_x', 0)
            h = bounds.get('max_y', 0) - bounds.get('min_y', 0)
            self.bounds_label.setText(f"<b>Plan Dimensions:</b> Width: {w:.1f} m, Height: {h:.1f} m, Min X: {bounds.get('min_x', 0):.1f}, Max Y: {bounds.get('max_y', 0):.1f}")
        else:
            self.bounds_label.setText("")

        # Visualization (matplotlib)
        if self.analysis_canvas:
            self.analysis_fig.clear()
            ax = self.analysis_fig.add_subplot(111)
            # Draw plan bounds
            if bounds:
                rect = [
                    [bounds['min_x'], bounds['min_y']],
                    [bounds['max_x'], bounds['min_y']],
                    [bounds['max_x'], bounds['max_y']],
                    [bounds['min_x'], bounds['max_y']],
                    [bounds['min_x'], bounds['min_y']]
                ]
                xs, ys = zip(*rect)
                ax.plot(xs, ys, 'k-', label='Plan Bounds')
            # Draw walls
            for wall in results.get('walls', []):
                xs, ys = zip(*wall)
                ax.plot(xs, ys, 'gray', linewidth=2, label='Wall')
            # Draw restricted areas
            for area in results.get('restricted_areas', []):
                xs, ys = zip(*area)
                ax.fill(xs, ys, color='lightblue', alpha=0.5, label='Restricted')
            # Draw entrances
            for ent in results.get('entrances', []):
                xs, ys = zip(*ent)
                ax.fill(xs, ys, color='red', alpha=0.3, label='Entrance')
            ax.set_title('Floor Plan Analysis')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize='small')
            self.analysis_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CADAnalyzerMainWindow()
    window.show()
    sys.exit(app.exec_())
