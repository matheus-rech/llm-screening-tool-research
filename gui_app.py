#!/usr/bin/env python3
"""
Qt GUI Application for LLM Screening Tool
Simple desktop interface for the systematic review screening tool.
"""

import sys
import os
import json
import threading
from pathlib import Path

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, 
                                QProgressBar, QTabWidget, QFormLayout, QLineEdit, 
                                QComboBox, QSpinBox, QMessageBox, QScrollArea, QFrame)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QFont, QPalette, QColor
except ImportError:
    print("PyQt6 not found. Installing...")
    os.system("pip install PyQt6")
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, 
                                QProgressBar, QTabWidget, QFormLayout, QLineEdit, 
                                QComboBox, QSpinBox, QMessageBox, QScrollArea, QFrame)
    from PyQt6.QtCore import QThread, pyqtSignal, Qt
    from PyQt6.QtGui import QFont, QPalette, QColor

# Import your screening modules
try:
    from app.services.utils.file_parser import load_studies
    from app.services.screening.modern_llm import ModernLLMScreener
    from app.services.utils.config_manager import ConfigManager
except ImportError:
    print("Warning: Could not import screening modules. Some features may not work.")

class ScreeningWorker(QThread):
    """Worker thread for running screening process"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path, config):
        super().__init__()
        self.file_path = file_path
        self.config = config
        
    def run(self):
        try:
            self.status.emit("Loading studies...")
            studies = load_studies(self.file_path)
            self.progress.emit(20)
            
            self.status.emit("Initializing screening...")
            screener = ModernLLMScreener(self.config)
            self.progress.emit(40)
            
            results = []
            total = len(studies)
            
            for i, study in enumerate(studies):
                self.status.emit(f"Screening study {i+1}/{total}...")
                try:
                    result = screener.screen_study(study)
                    results.append(result)
                except Exception as e:
                    self.error.emit(f"Error screening study {i+1}: {str(e)}")
                    continue
                
                progress = 40 + int((i + 1) / total * 60)
                self.progress.emit(progress)
            
            self.status.emit("Screening complete!")
            self.progress.emit(100)
            self.result.emit({"studies": results, "total": len(results)})
            
        except Exception as e:
            self.error.emit(f"Screening failed: {str(e)}")

class LLMScreeningGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Screening Tool - Desktop")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            QTabWidget::pane {
                border: 1px solid #4a5568;
                background-color: #2d3748;
            }
            QTabBar::tab {
                background-color: #4a5568;
                color: #e2e8f0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2d3748;
                border-bottom: 2px solid #3182ce;
            }
            QPushButton {
                background-color: #3182ce;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2c5282;
            }
            QPushButton:disabled {
                background-color: #4a5568;
                color: #a0aec0;
            }
            QTextEdit, QLineEdit {
                background-color: #2d3748;
                border: 1px solid #4a5568;
                color: #e2e8f0;
                padding: 4px;
                border-radius: 4px;
            }
            QLabel {
                color: #e2e8f0;
            }
            QProgressBar {
                border: 1px solid #4a5568;
                border-radius: 4px;
                text-align: center;
                background-color: #2d3748;
                color: #e2e8f0;
            }
            QProgressBar::chunk {
                background-color: #3182ce;
                border-radius: 3px;
            }
        """)
        
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("LLM Screening Tool")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # File Selection Tab
        self.init_file_tab()
        
        # Configuration Tab
        self.init_config_tab()
        
        # Screening Tab
        self.init_screening_tab()
        
        # Results Tab
        self.init_results_tab()
        
    def init_file_tab(self):
        file_widget = QWidget()
        layout = QVBoxLayout(file_widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #a0aec0;")
        file_layout.addWidget(self.file_path_label)
        
        self.select_file_btn = QPushButton("Select File")
        self.select_file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_file_btn)
        
        layout.addLayout(file_layout)
        
        # File info
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setMaximumHeight(200)
        layout.addWidget(QLabel("File Information:"))
        layout.addWidget(self.file_info)
        
        layout.addStretch()
        self.tabs.addTab(file_widget, "📁 File Selection")
        
    def init_config_tab(self):
        config_widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(config_widget)
        scroll.setWidgetResizable(True)
        
        layout = QFormLayout(config_widget)
        
        # API Configuration
        api_frame = QFrame()
        api_frame.setStyleSheet("QFrame { border: 1px solid #4a5568; border-radius: 4px; padding: 10px; }")
        api_layout = QFormLayout(api_frame)
        
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setPlaceholderText("sk-...")
        api_layout.addRow("OpenAI API Key:", self.openai_key)
        
        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.anthropic_key.setPlaceholderText("sk-ant-...")
        api_layout.addRow("Anthropic API Key:", self.anthropic_key)
        
        self.entrez_email = QLineEdit()
        self.entrez_email.setPlaceholderText("your-email@domain.com")
        api_layout.addRow("Entrez Email:", self.entrez_email)
        
        layout.addRow("API Configuration", api_frame)
        
        # PICO Configuration
        pico_frame = QFrame()
        pico_frame.setStyleSheet("QFrame { border: 1px solid #4a5568; border-radius: 4px; padding: 10px; }")
        pico_layout = QFormLayout(pico_frame)
        
        self.research_question = QTextEdit()
        self.research_question.setMaximumHeight(60)
        self.research_question.setPlaceholderText("Enter your research question...")
        pico_layout.addRow("Research Question:", self.research_question)
        
        self.population = QTextEdit()
        self.population.setMaximumHeight(60)
        self.population.setPlaceholderText("Describe target population...")
        pico_layout.addRow("Population:", self.population)
        
        self.intervention = QTextEdit()
        self.intervention.setMaximumHeight(60)
        self.intervention.setPlaceholderText("Describe intervention...")
        pico_layout.addRow("Intervention:", self.intervention)
        
        self.comparison = QTextEdit()
        self.comparison.setMaximumHeight(60)
        self.comparison.setPlaceholderText("Describe comparison...")
        pico_layout.addRow("Comparison:", self.comparison)
        
        self.outcomes = QTextEdit()
        self.outcomes.setMaximumHeight(60)
        self.outcomes.setPlaceholderText("Describe outcomes...")
        pico_layout.addRow("Outcomes:", self.outcomes)
        
        layout.addRow("PICO Configuration", pico_frame)
        
        # Screening Settings
        settings_frame = QFrame()
        settings_frame.setStyleSheet("QFrame { border: 1px solid #4a5568; border-radius: 4px; padding: 10px; }")
        settings_layout = QFormLayout(settings_frame)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100)
        self.batch_size.setValue(10)
        settings_layout.addRow("Batch Size:", self.batch_size)
        
        self.processing_strategy = QComboBox()
        self.processing_strategy.addItems(["ADAPTIVE", "BATCH", "LOOP", "CHAIN"])
        settings_layout.addRow("Processing Strategy:", self.processing_strategy)
        
        layout.addRow("Screening Settings", settings_frame)
        
        # Save/Load buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_config)
        load_btn = QPushButton("Load Configuration")
        load_btn.clicked.connect(self.load_config)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(load_btn)
        layout.addRow(btn_layout)
        
        self.tabs.addTab(scroll, "⚙️ Configuration")
        
    def init_screening_tab(self):
        screening_widget = QWidget()
        layout = QVBoxLayout(screening_widget)
        
        # Start button
        self.start_btn = QPushButton("Start Screening")
        self.start_btn.clicked.connect(self.start_screening)
        self.start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 12px; }")
        layout.addWidget(self.start_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to start screening...")
        layout.addWidget(self.status_label)
        
        # Log
        layout.addWidget(QLabel("Screening Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.tabs.addTab(screening_widget, "🔬 Screening")
        
    def init_results_tab(self):
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Summary
        self.summary_label = QLabel("No results yet...")
        layout.addWidget(self.summary_label)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        self.tabs.addTab(results_widget, "📊 Results")
        
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Reference File",
            "",
            "All Supported (*.ris *.bib *.csv *.tsv *.xml *.txt);;RIS Files (*.ris);;BibTeX Files (*.bib);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.file_path_label.setText(f"Selected: {Path(file_path).name}")
            self.file_path_label.setStyleSheet("color: #68d391;")
            self.selected_file = file_path
            
            # Try to load and show file info
            try:
                studies = load_studies(file_path)
                info = f"File loaded successfully!\n\n"
                info += f"Total studies: {len(studies)}\n"
                info += f"File type: {Path(file_path).suffix}\n"
                info += f"File size: {Path(file_path).stat().st_size / 1024:.1f} KB\n\n"
                
                if studies:
                    sample = studies[0]
                    info += "Sample study:\n"
                    info += f"Title: {sample.get('title', 'N/A')[:100]}...\n"
                    info += f"Authors: {sample.get('authors', 'N/A')}\n"
                    info += f"Year: {sample.get('year', 'N/A')}\n"
                
                self.file_info.setText(info)
                
            except Exception as e:
                self.file_info.setText(f"Error loading file: {str(e)}")
                
    def save_config(self):
        config = {
            "openai_api_key": self.openai_key.text(),
            "anthropic_api_key": self.anthropic_key.text(),
            "entrez_email": self.entrez_email.text(),
            "research_question": self.research_question.toPlainText(),
            "pico": {
                "population": self.population.toPlainText(),
                "intervention": self.intervention.toPlainText(),
                "comparison": self.comparison.toPlainText(),
                "outcomes": self.outcomes.toPlainText()
            },
            "batch_size": self.batch_size.value(),
            "processing_strategy": self.processing_strategy.currentText()
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "config.json", "JSON Files (*.json)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", "Configuration saved successfully!")
            
    def load_config(self):
        # Try to load from default .env file first
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            self.openai_key.setText(os.getenv('OPENAI_API_KEY', ''))
            self.anthropic_key.setText(os.getenv('ANTHROPIC_API_KEY', ''))
            self.entrez_email.setText(os.getenv('ENTREZ_EMAIL', ''))
            
        except ImportError:
            pass
        
        # Option to load from JSON file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                self.openai_key.setText(config.get('openai_api_key', ''))
                self.anthropic_key.setText(config.get('anthropic_api_key', ''))
                self.entrez_email.setText(config.get('entrez_email', ''))
                self.research_question.setText(config.get('research_question', ''))
                
                pico = config.get('pico', {})
                self.population.setText(pico.get('population', ''))
                self.intervention.setText(pico.get('intervention', ''))
                self.comparison.setText(pico.get('comparison', ''))
                self.outcomes.setText(pico.get('outcomes', ''))
                
                self.batch_size.setValue(config.get('batch_size', 10))
                
                strategy = config.get('processing_strategy', 'ADAPTIVE')
                index = self.processing_strategy.findText(strategy)
                if index >= 0:
                    self.processing_strategy.setCurrentIndex(index)
                
                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration: {str(e)}")
                
    def start_screening(self):
        if not hasattr(self, 'selected_file'):
            QMessageBox.warning(self, "Error", "Please select a file first!")
            return
            
        if not self.openai_key.text() or not self.anthropic_key.text():
            QMessageBox.warning(self, "Error", "Please enter API keys!")
            return
            
        # Prepare configuration
        config = {
            "openai_api_key": self.openai_key.text(),
            "anthropic_api_key": self.anthropic_key.text(),
            "entrez_email": self.entrez_email.text(),
            "research_question": self.research_question.toPlainText(),
            "pico": {
                "population": self.population.toPlainText(),
                "intervention": self.intervention.toPlainText(),
                "comparison": self.comparison.toPlainText(),
                "outcomes": self.outcomes.toPlainText()
            },
            "batch_size": self.batch_size.value(),
            "processing_strategy": self.processing_strategy.currentText()
        }
        
        # Disable start button
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Screening in progress...")
        
        # Start worker thread
        self.worker = ScreeningWorker(self.selected_file, config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.update_status)
        self.worker.result.connect(self.screening_complete)
        self.worker.error.connect(self.screening_error)
        self.worker.start()
        
        # Switch to screening tab
        self.tabs.setCurrentIndex(2)
        
    def update_status(self, status):
        self.status_label.setText(status)
        self.log_text.append(f"[{threading.current_thread().name}] {status}")
        
    def screening_complete(self, results):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Screening")
        self.export_btn.setEnabled(True)
        
        # Show summary
        total = results["total"]
        self.summary_label.setText(f"Screening complete! Processed {total} studies.")
        
        # Show results
        results_text = f"Screening Results ({total} studies)\n"
        results_text += "=" * 50 + "\n\n"
        
        for i, study in enumerate(results["studies"][:10], 1):  # Show first 10
            results_text += f"{i}. {study.get('title', 'No title')[:100]}...\n"
            results_text += f"   Decision: {study.get('decision', 'Unknown')}\n"
            results_text += f"   Confidence: {study.get('confidence', 'N/A')}\n\n"
            
        if total > 10:
            results_text += f"... and {total - 10} more studies\n"
            
        self.results_text.setText(results_text)
        self.screening_results = results
        
        # Switch to results tab
        self.tabs.setCurrentIndex(3)
        
    def screening_error(self, error):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Screening")
        QMessageBox.critical(self, "Screening Error", error)
        
    def export_results(self):
        if not hasattr(self, 'screening_results'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "screening_results.json", 
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.screening_results, f, indent=2)
                elif file_path.endswith('.csv'):
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Title', 'Decision', 'Confidence', 'Reasoning'])
                        for study in self.screening_results["studies"]:
                            writer.writerow([
                                study.get('title', ''),
                                study.get('decision', ''),
                                study.get('confidence', ''),
                                study.get('reasoning', '')
                            ])
                            
                QMessageBox.information(self, "Success", "Results exported successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("LLM Screening Tool")
    app.setApplicationVersion("1.0")
    
    window = LLMScreeningGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()