import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, 
    QFileDialog, QTextEdit, QHBoxLayout, QProgressBar
)
from PyQt5.QtCore import Qt
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import CADBackend

class ProductionCADAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏨 CAD Analyzer Pro - Desktop Production Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize tabs
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

        self.init_all_tabs()

    def init_all_tabs(self):
        self.init_file_upload_tab()
        self.init_analysis_tab()
        self.init_ilot_placement_tab()
        self.init_corridor_generation_tab()
        self.init_results_export_tab()

    def init_file_upload_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("<h2>📁 Upload Floor Plan</h2>")
        layout.addWidget(title)
        
        info = QLabel("Supported: DXF, DWG, PNG, JPG files with real data extraction")
        layout.addWidget(info)
        
        self.upload_btn = QPushButton("Select CAD File")
        self.upload_btn.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.upload_btn)
        
        self.file_info = QLabel("")
        layout.addWidget(self.file_info)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        
        self.file_upload_tab.setLayout(layout)

    def open_file_dialog(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Floor Plan", "", 
                "CAD Files (*.dxf *.dwg);;Images (*.png *.jpg *.jpeg)"
            )
            if file_path:
                self.file_info.setText(f"Selected: {file_path}")
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)
                self.result_text.setText("Processing file...")
                
                def on_parsing_done(results):
                    try:
                        self.progress_bar.setVisible(False)
                        if results and results.get('success'):
                            self.analysis_results = results
                            summary = self.format_results_summary(results)
                            self.result_text.setText(f"✅ File processed successfully!\n\n{summary}")
                            self.update_analysis_tab(results)
                        else:
                            error_msg = results.get('error', 'Unknown error') if results else 'Processing failed'
                            self.result_text.setText(f"❌ Error: {error_msg}")
                            self.update_analysis_tab(None)
                    except Exception as e:
                        self.result_text.setText(f"❌ Display error: {str(e)}")
                
                def safe_parse():
                    try:
                        self.backend.parse_file_async(file_path, on_parsing_done)
                    except Exception as e:
                        on_parsing_done({'success': False, 'error': f'Parse error: {str(e)}'})
                
                threading.Thread(target=safe_parse, daemon=True).start()
        except Exception as e:
            self.result_text.setText(f"❌ File dialog error: {str(e)}")

    def format_results_summary(self, results):
        lines = []
        lines.append(f"Method: {results.get('method', 'unknown')}")
        lines.append(f"Entities: {results.get('entity_count', 0)}")
        lines.append(f"Walls: {results.get('wall_count', 0)}")
        lines.append(f"Restricted Areas: {results.get('restricted_count', 0)}")
        lines.append(f"Entrances: {results.get('entrance_count', 0)}")
        if 'bounds' in results:
            b = results['bounds']
            lines.append(f"Dimensions: {b['max_x']-b['min_x']:.1f} x {b['max_y']-b['min_y']:.1f} m")
        return '\n'.join(lines)

    def init_analysis_tab(self):
        layout = QVBoxLayout()
        
        self.analysis_title = QLabel("<h2>🔍 Floor Plan Analysis</h2>")
        layout.addWidget(self.analysis_title)
        
        self.analysis_results_text = QTextEdit()
        self.analysis_results_text.setReadOnly(True)
        layout.addWidget(self.analysis_results_text)
        
        # Visualization
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self.analysis_fig = Figure(figsize=(10, 6))
            self.analysis_canvas = FigureCanvas(self.analysis_fig)
            layout.addWidget(self.analysis_canvas)
        except ImportError:
            layout.addWidget(QLabel("Matplotlib not installed. Visualization unavailable."))
            self.analysis_canvas = None
        
        self.analysis_tab.setLayout(layout)

    def update_analysis_tab(self, results):
        if not results or not results.get('success'):
            self.analysis_results_text.setText("No analysis results available.")
            if self.analysis_canvas:
                self.analysis_fig.clear()
                self.analysis_canvas.draw()
            return

        # Display analysis info
        info = f"Analysis Results:\n"
        info += f"- Method: {results.get('method', 'unknown')}\n"
        info += f"- Walls: {results.get('wall_count', 0)}\n"
        info += f"- Restricted Areas: {results.get('restricted_count', 0)}\n"
        info += f"- Entrances: {results.get('entrance_count', 0)}\n"
        
        bounds = results.get('bounds', {})
        if bounds:
            w = bounds.get('max_x', 0) - bounds.get('min_x', 0)
            h = bounds.get('max_y', 0) - bounds.get('min_y', 0)
            info += f"- Dimensions: {w:.1f} x {h:.1f} m\n"
        
        self.analysis_results_text.setText(info)

        # Visualization
        if self.analysis_canvas and bounds:
            self.analysis_fig.clear()
            ax = self.analysis_fig.add_subplot(111)
            
            # Floor bounds
            rect = [
                [bounds['min_x'], bounds['min_y']],
                [bounds['max_x'], bounds['min_y']],
                [bounds['max_x'], bounds['max_y']],
                [bounds['min_x'], bounds['max_y']],
                [bounds['min_x'], bounds['min_y']]
            ]
            xs, ys = zip(*rect)
            ax.plot(xs, ys, 'k-', linewidth=2, label='Floor Bounds')
            
            # Walls
            for wall in results.get('walls', []):
                if len(wall) >= 2:
                    xs, ys = zip(*wall)
                    ax.plot(xs, ys, 'gray', linewidth=2, label='Wall')
            
            ax.set_title('Floor Plan Analysis')
            ax.set_aspect('equal')
            ax.legend()
            self.analysis_canvas.draw()

    def init_ilot_placement_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("<h2>🏗️ Îlot Placement</h2>")
        layout.addWidget(title)
        
        config_label = QLabel("Size Distribution: Small(10%) | Medium(25%) | Large(30%) | XL(35%)")
        layout.addWidget(config_label)
        
        self.place_ilots_btn = QPushButton("Generate Îlot Placement")
        self.place_ilots_btn.clicked.connect(self.place_ilots)
        layout.addWidget(self.place_ilots_btn)
        
        self.ilot_progress = QProgressBar()
        self.ilot_progress.setVisible(False)
        layout.addWidget(self.ilot_progress)
        
        self.ilot_results = QTextEdit()
        self.ilot_results.setReadOnly(True)
        layout.addWidget(self.ilot_results)
        
        # Visualization
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self.ilot_fig = Figure(figsize=(10, 6))
            self.ilot_canvas = FigureCanvas(self.ilot_fig)
            layout.addWidget(self.ilot_canvas)
        except ImportError:
            self.ilot_canvas = None
        
        self.ilot_tab.setLayout(layout)

    def place_ilots(self):
        try:
            if not self.analysis_results or not self.analysis_results.get('success'):
                self.ilot_results.setText("❌ No analysis results. Please upload and analyze a floor plan first.")
                return
            
            self.place_ilots_btn.setEnabled(False)
            self.ilot_progress.setVisible(True)
            self.ilot_progress.setRange(0, 0)
            self.ilot_results.setText("Placing îlots...")
            
            def on_placement_done(result):
                try:
                    self.ilot_progress.setVisible(False)
                    self.place_ilots_btn.setEnabled(True)
                    
                    if result and result.get('success'):
                        ilots = result.get('ilots', [])
                        metrics = result.get('metrics', {})
                        distribution = result.get('distribution', {})
                        
                        results_text = f"✅ Successfully placed {len(ilots)} îlots\n\n"
                        results_text += f"Metrics:\n"
                        results_text += f"- Space Utilization: {metrics.get('space_utilization', 0)*100:.1f}%\n"
                        results_text += f"- Efficiency Score: {metrics.get('efficiency_score', 0)*100:.1f}%\n\n"
                        results_text += f"Distribution:\n"
                        results_text += f"- Small: {distribution.get('size_0_1', 0)} îlots\n"
                        results_text += f"- Medium: {distribution.get('size_1_3', 0)} îlots\n"
                        results_text += f"- Large: {distribution.get('size_3_5', 0)} îlots\n"
                        results_text += f"- XL: {distribution.get('size_5_10', 0)} îlots"
                        
                        self.ilot_results.setText(results_text)
                        
                        if self.ilot_canvas:
                            self.update_ilot_visualization(ilots)
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'Placement failed'
                        self.ilot_results.setText(f"❌ Placement failed: {error_msg}")
                except Exception as e:
                    self.ilot_results.setText(f"❌ Display error: {str(e)}")
                    self.place_ilots_btn.setEnabled(True)
                    self.ilot_progress.setVisible(False)
            
            def safe_placement():
                try:
                    self.backend.place_ilots_async(self.analysis_results, None, on_placement_done)
                except Exception as e:
                    on_placement_done({'success': False, 'error': f'Placement error: {str(e)}'})
            
            threading.Thread(target=safe_placement, daemon=True).start()
        except Exception as e:
            self.ilot_results.setText(f"❌ Placement error: {str(e)}")
            self.place_ilots_btn.setEnabled(True)
            self.ilot_progress.setVisible(False)

    def update_ilot_visualization(self, ilots):
        if not self.ilot_canvas:
            return
        
        self.ilot_fig.clear()
        ax = self.ilot_fig.add_subplot(111)
        
        # Floor bounds
        bounds = self.analysis_results.get('bounds', {})
        if bounds:
            rect = [
                [bounds['min_x'], bounds['min_y']],
                [bounds['max_x'], bounds['min_y']],
                [bounds['max_x'], bounds['max_y']],
                [bounds['min_x'], bounds['max_y']],
                [bounds['min_x'], bounds['min_y']]
            ]
            xs, ys = zip(*rect)
            ax.plot(xs, ys, 'k-', linewidth=2)
        
        # Walls
        for wall in self.analysis_results.get('walls', []):
            if len(wall) >= 2:
                xs, ys = zip(*wall)
                ax.plot(xs, ys, 'gray', linewidth=1, alpha=0.7)
        
        # Îlots
        colors = {'size_0_1': 'yellow', 'size_1_3': 'orange', 'size_3_5': 'green', 'size_5_10': 'purple'}
        
        for ilot in ilots:
            x, y = ilot['x'], ilot['y']
            w, h = ilot['width'], ilot['height']
            color = colors.get(ilot['size_category'], 'gray')
            
            rect_x = [x-w/2, x+w/2, x+w/2, x-w/2, x-w/2]
            rect_y = [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2]
            ax.fill(rect_x, rect_y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title('Îlot Placement Results')
        ax.set_aspect('equal')
        self.ilot_canvas.draw()

    def init_corridor_generation_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("<h2>🛤️ Corridor Generation</h2>")
        layout.addWidget(title)
        
        config_label = QLabel("Configuration: Width: 1.5m | Min Spacing: 1.0m | Wall Clearance: 0.5m")
        layout.addWidget(config_label)
        
        self.generate_corridors_btn = QPushButton("Generate Mandatory Corridors")
        self.generate_corridors_btn.clicked.connect(self.generate_corridors)
        layout.addWidget(self.generate_corridors_btn)
        
        self.corridor_progress = QProgressBar()
        self.corridor_progress.setVisible(False)
        layout.addWidget(self.corridor_progress)
        
        self.corridor_results = QTextEdit()
        self.corridor_results.setReadOnly(True)
        layout.addWidget(self.corridor_results)
        
        # Visualization
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self.corridor_fig = Figure(figsize=(10, 6))
            self.corridor_canvas = FigureCanvas(self.corridor_fig)
            layout.addWidget(self.corridor_canvas)
        except ImportError:
            self.corridor_canvas = None
        
        self.corridor_tab.setLayout(layout)

    def generate_corridors(self):
        if not self.backend.placed_ilots:
            self.corridor_results.setText("❌ No îlots available. Please complete îlot placement first.")
            return
        
        self.generate_corridors_btn.setEnabled(False)
        self.corridor_progress.setVisible(True)
        self.corridor_progress.setRange(0, 0)
        self.corridor_results.setText("Generating corridors...")
        
        def on_corridors_done(result):
            self.corridor_progress.setVisible(False)
            self.generate_corridors_btn.setEnabled(True)
            
            if result.get('success'):
                corridors = result.get('corridors', [])
                
                results_text = f"✅ Successfully generated {len(corridors)} corridors\n\n"
                results_text += "Corridor Details:\n"
                
                for i, corridor in enumerate(corridors[:5]):
                    results_text += f"- Corridor {i+1}: {corridor.get('type', 'unknown')} "
                    results_text += f"(Width: {corridor.get('width', 0):.1f}m)\n"
                
                if len(corridors) > 5:
                    results_text += f"... and {len(corridors) - 5} more corridors"
                
                self.corridor_results.setText(results_text)
                
                if self.corridor_canvas:
                    self.update_corridor_visualization(corridors)
            else:
                self.corridor_results.setText(f"❌ Corridor generation failed: {result.get('error', 'Unknown error')}")
        
        self.backend.generate_corridors_async(on_corridors_done)

    def update_corridor_visualization(self, corridors):
        if not self.corridor_canvas:
            return
        
        self.corridor_fig.clear()
        ax = self.corridor_fig.add_subplot(111)
        
        # Floor bounds
        bounds = self.analysis_results.get('bounds', {})
        if bounds:
            rect = [
                [bounds['min_x'], bounds['min_y']],
                [bounds['max_x'], bounds['min_y']],
                [bounds['max_x'], bounds['max_y']],
                [bounds['min_x'], bounds['max_y']],
                [bounds['min_x'], bounds['min_y']]
            ]
            xs, ys = zip(*rect)
            ax.plot(xs, ys, 'k-', linewidth=2)
        
        # Îlots
        colors = {'size_0_1': 'yellow', 'size_1_3': 'orange', 'size_3_5': 'green', 'size_5_10': 'purple'}
        for ilot in self.backend.placed_ilots:
            x, y = ilot['x'], ilot['y']
            w, h = ilot['width'], ilot['height']
            color = colors.get(ilot['size_category'], 'gray')
            
            rect_x = [x-w/2, x+w/2, x+w/2, x-w/2, x-w/2]
            rect_y = [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2]
            ax.fill(rect_x, rect_y, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
        
        # Corridors
        for corridor in corridors:
            path_points = corridor.get('path_points', [])
            if len(path_points) >= 2:
                xs, ys = zip(*path_points)
                ax.plot(xs, ys, 'blue', linewidth=corridor.get('width', 1.5)*2, alpha=0.8)
        
        ax.set_title('Corridor Network')
        ax.set_aspect('equal')
        self.corridor_canvas.draw()

    def init_results_export_tab(self):
        layout = QVBoxLayout()
        
        title = QLabel("<h2>📊 Results & Export</h2>")
        layout.addWidget(title)
        
        # Final visualization
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            self.final_fig = Figure(figsize=(12, 8))
            self.final_canvas = FigureCanvas(self.final_fig)
            layout.addWidget(self.final_canvas)
        except ImportError:
            self.final_canvas = None
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_json_btn = QPushButton("📊 Export JSON")
        self.export_json_btn.clicked.connect(self.export_json)
        export_layout.addWidget(self.export_json_btn)
        
        self.export_image_btn = QPushButton("🖼️ Export Image")
        self.export_image_btn.clicked.connect(self.export_image)
        export_layout.addWidget(self.export_image_btn)
        
        self.refresh_btn = QPushButton("🔄 Refresh View")
        self.refresh_btn.clicked.connect(self.refresh_final_view)
        export_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(export_layout)
        
        self.export_status = QLabel("")
        layout.addWidget(self.export_status)
        
        self.results_tab.setLayout(layout)

    def refresh_final_view(self):
        if not self.final_canvas:
            return
        
        self.final_fig.clear()
        ax = self.final_fig.add_subplot(111)
        
        # Floor bounds
        if self.analysis_results:
            bounds = self.analysis_results.get('bounds', {})
            if bounds:
                rect = [
                    [bounds['min_x'], bounds['min_y']],
                    [bounds['max_x'], bounds['min_y']],
                    [bounds['max_x'], bounds['max_y']],
                    [bounds['min_x'], bounds['max_y']],
                    [bounds['min_x'], bounds['min_y']]
                ]
                xs, ys = zip(*rect)
                ax.plot(xs, ys, 'k-', linewidth=3)
            
            # Walls
            for wall in self.analysis_results.get('walls', []):
                if len(wall) >= 2:
                    xs, ys = zip(*wall)
                    ax.plot(xs, ys, 'gray', linewidth=1, alpha=0.7)
        
        # Îlots
        colors = {'size_0_1': 'yellow', 'size_1_3': 'orange', 'size_3_5': 'green', 'size_5_10': 'purple'}
        for ilot in self.backend.placed_ilots:
            x, y = ilot['x'], ilot['y']
            w, h = ilot['width'], ilot['height']
            color = colors.get(ilot['size_category'], 'gray')
            
            rect_x = [x-w/2, x+w/2, x+w/2, x-w/2, x-w/2]
            rect_y = [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2]
            ax.fill(rect_x, rect_y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Corridors
        for corridor in self.backend.corridors:
            path_points = corridor.get('path_points', [])
            if len(path_points) >= 2:
                xs, ys = zip(*path_points)
                ax.plot(xs, ys, 'blue', linewidth=corridor.get('width', 1.5)*2, alpha=0.8)
        
        ax.set_title('Final Layout - Complete Analysis')
        ax.set_aspect('equal')
        self.final_canvas.draw()
        
        status = f"Îlots: {len(self.backend.placed_ilots)} | Corridors: {len(self.backend.corridors)}"
        self.export_status.setText(status)

    def export_json(self):
        json_data = self.backend.export_results('json')
        if json_data:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON Export", "cad_analysis_results.json", "JSON Files (*.json)"
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_data)
                self.export_status.setText(f"✅ JSON exported to {file_path}")
        else:
            self.export_status.setText("❌ No data to export")

    def export_image(self):
        if self.final_canvas:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image Export", "cad_analysis_layout.png", "PNG Files (*.png)"
            )
            if file_path:
                self.final_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.export_status.setText(f"✅ Image exported to {file_path}")
        else:
            self.export_status.setText("❌ No visualization to export")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProductionCADAnalyzer()
    window.show()
    sys.exit(app.exec_())