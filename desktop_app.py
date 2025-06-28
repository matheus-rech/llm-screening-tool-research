#!/usr/bin/env python3
"""
Desktop GUI for LLM Screening Tool
Professional interface with full functionality access.
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
except ImportError:
    print("Installing PyQt6...")
    os.system("pip install PyQt6")
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *

# Import screening modules
try:
    from app.services.utils.file_parser import load_studies
    from app.services.screening.modern_llm import ModernLLMScreener
    from app.services.utils.config_manager import ConfigManager
except ImportError:
    print("Warning: Screening modules not found. Running in demo mode.")
    ModernLLMScreener = None

class ScreeningWorker(QThread):
    """Background worker for screening process"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    article_processed = pyqtSignal(dict)
    screening_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, config):
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.is_cancelled = False
        
    def cancel(self):
        self.is_cancelled = True
        
    def run(self):
        try:
            self.status_updated.emit("Loading studies from file...")
            studies = load_studies(self.file_path)
            total_studies = len(studies)
            
            if total_studies == 0:
                self.error_occurred.emit("No studies found in file")
                return
                
            self.status_updated.emit(f"Found {total_studies} studies. Initializing screening...")
            self.progress_updated.emit(5)
            
            if ModernLLMScreener:
                screener = ModernLLMScreener(self.config)
            else:
                # Demo mode
                self.run_demo_screening(studies)
                return
                
            results = []
            
            for i, study in enumerate(studies):
                if self.is_cancelled:
                    self.status_updated.emit("Screening cancelled")
                    return
                    
                self.status_updated.emit(f"Screening article {i+1}/{total_studies}: {study.get('title', 'Untitled')[:50]}...")
                
                try:
                    result = screener.screen_study(study)
                    results.append(result)
                    self.article_processed.emit(result)
                    
                except Exception as e:
                    error_result = {
                        **study,
                        'decision': 'ERROR',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
                    self.article_processed.emit(error_result)
                
                progress = 5 + int((i + 1) / total_studies * 90)
                self.progress_updated.emit(progress)
                
            self.progress_updated.emit(100)
            self.status_updated.emit(f"Screening complete! Processed {len(results)} articles.")
            self.screening_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"Screening failed: {str(e)}")
            
    def run_demo_screening(self, studies):
        """Demo mode when screening modules aren't available"""
        results = []
        decisions = ['INCLUDE', 'EXCLUDE', 'CONFLICT']
        
        for i, study in enumerate(studies[:10]):  # Limit to 10 for demo
            if self.is_cancelled:
                return
                
            self.status_updated.emit(f"Demo screening {i+1}/10: {study.get('title', 'Untitled')[:50]}...")
            
            # Simulate processing time
            time.sleep(0.5)
            
            import random
            decision = random.choice(decisions)
            confidence = random.randint(70, 95)
            
            result = {
                **study,
                'decision': decision,
                'confidence': confidence,
                'reasoning': f"Demo screening result for testing purposes",
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            self.article_processed.emit(result)
            
            progress = 5 + int((i + 1) / 10 * 90)
            self.progress_updated.emit(progress)
            
        self.progress_updated.emit(100)
        self.status_updated.emit("Demo screening complete!")
        self.screening_complete.emit(results)

class LLMScreeningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Screening Tool - Desktop Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Application state
        self.selected_file = None
        self.screening_results = []
        self.config_manager = ConfigManager() if 'ConfigManager' in globals() else None
        
        self.setup_styles()
        self.init_ui()
        self.load_default_config()
        
    def setup_styles(self):
        """Setup modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            
            QTabWidget::pane {
                border: 2px solid #3a3a3a;
                background-color: #2a2a2a;
                border-radius: 8px;
            }
            
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #ffffff;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            
            QTabBar::tab:selected {
                background-color: #0d7377;
                color: #ffffff;
            }
            
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
            
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            
            QPushButton:hover {
                background-color: #14a085;
            }
            
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            
            QPushButton:disabled {
                background-color: #5a5a5a;
                color: #aaaaaa;
            }
            
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #3a3a3a;
                border: 2px solid #5a5a5a;
                color: #ffffff;
                padding: 8px;
                border-radius: 6px;
                font-size: 12px;
            }
            
            QLineEdit:focus, QTextEdit:focus {
                border-color: #0d7377;
            }
            
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            
            QGroupBox {
                color: #ffffff;
                border: 2px solid #5a5a5a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #0d7377;
            }
            
            QProgressBar {
                border: 2px solid #5a5a5a;
                border-radius: 6px;
                text-align: center;
                background-color: #3a3a3a;
                color: #ffffff;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 4px;
            }
            
            QTreeWidget {
                background-color: #2a2a2a;
                border: 2px solid #5a5a5a;
                border-radius: 6px;
                color: #ffffff;
                selection-background-color: #0d7377;
            }
            
            QTreeWidget::item {
                padding: 4px;
                border-bottom: 1px solid #3a3a3a;
            }
            
            QTreeWidget::item:selected {
                background-color: #0d7377;
            }
            
            QSplitter::handle {
                background-color: #5a5a5a;
                width: 3px;
                border-radius: 1px;
            }
        """)
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        self.create_header(main_layout)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_project_tab()
        self.create_config_tab()
        self.create_screening_tab()
        self.create_results_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready - Select a file to begin")
        self.statusBar().setStyleSheet("background-color: #2a2a2a; color: #ffffff; padding: 4px;")
        
    def create_header(self, layout):
        """Create application header"""
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet("background-color: #0d7377; border-bottom: 2px solid #3a3a3a;")
        
        header_layout = QHBoxLayout(header)
        
        # Title and logo
        title_layout = QVBoxLayout()
        title = QLabel("🔬 LLM Screening Tool")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; background: transparent;")
        
        subtitle = QLabel("Dual-LLM Systematic Review Screening")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #b0b0b0; background: transparent;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        
        header_layout.addStretch()
        
        # Quick actions
        quick_layout = QVBoxLayout()
        
        self.quick_start_btn = QPushButton("Quick Start")
        self.quick_start_btn.clicked.connect(self.quick_start)
        
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)
        
        quick_layout.addWidget(self.quick_start_btn)
        quick_layout.addWidget(self.help_btn)
        header_layout.addLayout(quick_layout)
        
        layout.addWidget(header)
        
    def create_project_tab(self):
        """Create project management tab"""
        project_widget = QWidget()
        layout = QVBoxLayout(project_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # File selection group
        file_group = QGroupBox("Reference File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # File selection row
        file_select_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        file_select_layout.addWidget(self.file_path_label)
        
        file_select_layout.addStretch()
        
        self.select_file_btn = QPushButton("📁 Browse Files")
        self.select_file_btn.clicked.connect(self.select_file)
        file_select_layout.addWidget(self.select_file_btn)
        
        file_layout.addLayout(file_select_layout)
        
        # File info
        self.file_info_text = QTextEdit()
        self.file_info_text.setMaximumHeight(120)
        self.file_info_text.setReadOnly(True)
        self.file_info_text.setPlaceholderText("File information will appear here after selection...")
        file_layout.addWidget(self.file_info_text)
        
        layout.addWidget(file_group)
        
        # Project info group
        project_info_group = QGroupBox("Project Information")
        project_info_layout = QFormLayout(project_info_group)
        
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("Enter project name...")
        project_info_layout.addRow("Project Name:", self.project_name)
        
        self.project_description = QTextEdit()
        self.project_description.setMaximumHeight(80)
        self.project_description.setPlaceholderText("Optional project description...")
        project_info_layout.addRow("Description:", self.project_description)
        
        layout.addWidget(project_info_group)
        
        layout.addStretch()
        
        self.tabs.addTab(project_widget, "📂 Project")
        
    def create_config_tab(self):
        """Create configuration tab"""
        config_widget = QWidget()
        
        # Create splitter for side-by-side layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20, 20, 10, 20)
        
        # API Configuration
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout(api_group)
        
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setPlaceholderText("sk-proj-...")
        api_layout.addRow("OpenAI API Key:", self.openai_key)
        
        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.anthropic_key.setPlaceholderText("sk-ant-...")
        api_layout.addRow("Anthropic API Key:", self.anthropic_key)
        
        self.entrez_email = QLineEdit()
        self.entrez_email.setPlaceholderText("your-email@domain.com")
        api_layout.addRow("Entrez Email:", self.entrez_email)
        
        left_layout.addWidget(api_group)
        
        # PICO Configuration
        pico_group = QGroupBox("PICO-T Framework")
        pico_layout = QFormLayout(pico_group)
        
        self.research_question = QTextEdit()
        self.research_question.setMaximumHeight(60)
        self.research_question.setPlaceholderText("Enter your research question...")
        pico_layout.addRow("Research Question:", self.research_question)
        
        self.population = QTextEdit()
        self.population.setMaximumHeight(50)
        self.population.setPlaceholderText("Target population...")
        pico_layout.addRow("Population (P):", self.population)
        
        self.intervention = QTextEdit()
        self.intervention.setMaximumHeight(50)
        self.intervention.setPlaceholderText("Intervention...")
        pico_layout.addRow("Intervention (I):", self.intervention)
        
        self.comparison = QTextEdit()
        self.comparison.setMaximumHeight(50)
        self.comparison.setPlaceholderText("Comparison...")
        pico_layout.addRow("Comparison (C):", self.comparison)
        
        self.outcomes = QTextEdit()
        self.outcomes.setMaximumHeight(50)
        self.outcomes.setPlaceholderText("Outcomes...")
        pico_layout.addRow("Outcomes (O):", self.outcomes)
        
        left_layout.addWidget(pico_group)
        
        # Screening Settings
        settings_group = QGroupBox("Screening Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100)
        self.batch_size.setValue(10)
        settings_layout.addRow("Batch Size:", self.batch_size)
        
        self.processing_strategy = QComboBox()
        self.processing_strategy.addItems(["ADAPTIVE", "BATCH", "LOOP", "CHAIN"])
        settings_layout.addRow("Processing Strategy:", self.processing_strategy)
        
        self.conservative_model = QComboBox()
        self.conservative_model.addItems(["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        settings_layout.addRow("Conservative Model:", self.conservative_model)
        
        self.liberal_model = QComboBox()
        self.liberal_model.addItems(["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"])
        settings_layout.addRow("Liberal Model:", self.liberal_model)
        
        left_layout.addWidget(settings_group)
        
        left_layout.addStretch()
        
        # Configuration actions
        config_actions_layout = QHBoxLayout()
        
        save_config_btn = QPushButton("💾 Save Config")
        save_config_btn.clicked.connect(self.save_configuration)
        
        load_config_btn = QPushButton("📂 Load Config")
        load_config_btn.clicked.connect(self.load_configuration)
        
        reset_config_btn = QPushButton("🔄 Reset")
        reset_config_btn.clicked.connect(self.reset_configuration)
        
        config_actions_layout.addWidget(save_config_btn)
        config_actions_layout.addWidget(load_config_btn)
        config_actions_layout.addWidget(reset_config_btn)
        config_actions_layout.addStretch()
        
        left_layout.addLayout(config_actions_layout)
        
        # Right side - Configuration Templates
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 20, 20, 20)
        
        templates_group = QGroupBox("Configuration Templates")
        templates_layout = QVBoxLayout(templates_group)
        
        self.templates_list = QTreeWidget()
        self.templates_list.setHeaderLabels(["Template Name", "Type", "Modified"])
        self.templates_list.setRootIsDecorated(False)
        templates_layout.addWidget(self.templates_list)
        
        template_actions_layout = QHBoxLayout()
        
        load_template_btn = QPushButton("Load Template")
        load_template_btn.clicked.connect(self.load_template)
        
        save_template_btn = QPushButton("Save as Template")
        save_template_btn.clicked.connect(self.save_template)
        
        template_actions_layout.addWidget(load_template_btn)
        template_actions_layout.addWidget(save_template_btn)
        template_actions_layout.addStretch()
        
        templates_layout.addLayout(template_actions_layout)
        right_layout.addWidget(templates_group)
        
        # Config preview
        preview_group = QGroupBox("Configuration Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.config_preview = QTextEdit()
        self.config_preview.setReadOnly(True)
        self.config_preview.setMaximumHeight(200)
        preview_layout.addWidget(self.config_preview)
        
        right_layout.addWidget(preview_group)
        
        right_layout.addStretch()
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
        
        # Main layout
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(splitter)
        
        self.tabs.addTab(config_widget, "⚙️ Configuration")
        
    def create_screening_tab(self):
        """Create screening execution tab"""
        screening_widget = QWidget()
        layout = QVBoxLayout(screening_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Control panel
        control_group = QGroupBox("Screening Control")
        control_layout = QVBoxLayout(control_group)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        
        self.start_screening_btn = QPushButton("🚀 Start Screening")
        self.start_screening_btn.clicked.connect(self.start_screening)
        self.start_screening_btn.setStyleSheet("QPushButton { padding: 15px 30px; font-size: 14px; }")
        
        self.stop_screening_btn = QPushButton("⏹️ Stop")
        self.stop_screening_btn.clicked.connect(self.stop_screening)
        self.stop_screening_btn.setEnabled(False)
        
        self.pause_screening_btn = QPushButton("⏸️ Pause")
        self.pause_screening_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_screening_btn)
        button_layout.addWidget(self.stop_screening_btn)
        button_layout.addWidget(self.pause_screening_btn)
        button_layout.addStretch()
        
        control_layout.addLayout(button_layout)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(QLabel("Progress:"))
        progress_layout.addWidget(self.progress_bar)
        
        control_layout.addLayout(progress_layout)
        
        # Status
        self.status_label = QLabel("Ready to start screening...")
        self.status_label.setStyleSheet("color: #0d7377; font-weight: bold;")
        control_layout.addWidget(self.status_label)
        
        layout.addWidget(control_group)
        
        # Real-time results
        results_group = QGroupBox("Live Screening Results")
        results_layout = QVBoxLayout(results_group)
        
        self.live_results = QTreeWidget()
        self.live_results.setHeaderLabels(["Title", "Decision", "Confidence", "Status"])
        self.live_results.setAlternatingRowColors(True)
        results_layout.addWidget(self.live_results)
        
        layout.addWidget(results_group)
        
        # Statistics panel
        stats_layout = QHBoxLayout()
        
        self.stats_included = QLabel("Included: 0")
        self.stats_excluded = QLabel("Excluded: 0")
        self.stats_conflicts = QLabel("Conflicts: 0")
        self.stats_errors = QLabel("Errors: 0")
        
        for stat in [self.stats_included, self.stats_excluded, self.stats_conflicts, self.stats_errors]:
            stat.setStyleSheet("background-color: #3a3a3a; padding: 10px; border-radius: 6px; font-weight: bold;")
            stats_layout.addWidget(stat)
        
        layout.addLayout(stats_layout)
        
        self.tabs.addTab(screening_widget, "🔬 Screening")
        
    def create_results_tab(self):
        """Create results analysis tab"""
        results_widget = QWidget()
        
        # Splitter for results and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Results list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20, 20, 10, 20)
        
        # Filters
        filters_group = QGroupBox("Filter Results")
        filters_layout = QHBoxLayout(filters_group)
        
        self.filter_decision = QComboBox()
        self.filter_decision.addItems(["All", "INCLUDE", "EXCLUDE", "CONFLICT", "ERROR"])
        self.filter_decision.currentTextChanged.connect(self.filter_results)
        
        self.filter_confidence = QSpinBox()
        self.filter_confidence.setRange(0, 100)
        self.filter_confidence.setValue(0)
        self.filter_confidence.setSuffix("% min confidence")
        
        filters_layout.addWidget(QLabel("Decision:"))
        filters_layout.addWidget(self.filter_decision)
        filters_layout.addWidget(QLabel("Confidence:"))
        filters_layout.addWidget(self.filter_confidence)
        filters_layout.addStretch()
        
        left_layout.addWidget(filters_group)
        
        # Results table
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Title", "Authors", "Decision", "Confidence", "Reasoning"])
        self.results_tree.itemClicked.connect(self.show_result_details)
        left_layout.addWidget(self.results_tree)
        
        # Export options
        export_layout = QHBoxLayout()
        
        export_csv_btn = QPushButton("📊 Export CSV")
        export_csv_btn.clicked.connect(lambda: self.export_results('csv'))
        
        export_json_btn = QPushButton("📄 Export JSON")
        export_json_btn.clicked.connect(lambda: self.export_results('json'))
        
        export_ris_btn = QPushButton("📚 Export RIS")
        export_ris_btn.clicked.connect(lambda: self.export_results('ris'))
        
        export_layout.addWidget(export_csv_btn)
        export_layout.addWidget(export_json_btn)
        export_layout.addWidget(export_ris_btn)
        export_layout.addStretch()
        
        left_layout.addLayout(export_layout)
        
        # Right side - Result details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 20, 20, 20)
        
        # Summary statistics
        summary_group = QGroupBox("Summary Statistics")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        summary_layout.addWidget(self.summary_text)
        
        right_layout.addWidget(summary_group)
        
        # Selected result details
        details_group = QGroupBox("Article Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        right_layout.addWidget(details_group)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 500])
        
        # Main layout
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addWidget(splitter)
        
        self.tabs.addTab(results_widget, "📊 Results")
        
    def select_file(self):
        """Handle file selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference File",
            "",
            "Reference Files (*.ris *.bib *.csv *.tsv *.xml *.txt);;RIS Files (*.ris);;BibTeX Files (*.bib);;CSV Files (*.csv *.tsv);;XML Files (*.xml);;Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            self.selected_file = file_path
            file_name = Path(file_path).name
            self.file_path_label.setText(f"📄 {file_name}")
            self.file_path_label.setStyleSheet("color: #0d7377; font-weight: bold;")
            
            # Load file info
            self.load_file_info(file_path)
            
            # Auto-set project name if empty
            if not self.project_name.text():
                name = Path(file_path).stem
                self.project_name.setText(f"Screening Project - {name}")
                
    def load_file_info(self, file_path):
        """Load and display file information"""
        try:
            self.statusBar().showMessage("Loading file information...")
            
            if load_studies:
                studies = load_studies(file_path)
                total_studies = len(studies)
                
                info = f"✅ File loaded successfully!\n\n"
                info += f"📊 Total studies: {total_studies:,}\n"
                info += f"📁 File type: {Path(file_path).suffix.upper()}\n"
                info += f"💾 File size: {Path(file_path).stat().st_size / 1024:.1f} KB\n"
                info += f"📅 Modified: {datetime.fromtimestamp(Path(file_path).stat().st_mtime).strftime('%Y-%m-%d %H:%M')}\n\n"
                
                if studies and total_studies > 0:
                    sample = studies[0]
                    info += "📖 Sample study:\n"
                    info += f"  Title: {sample.get('title', 'N/A')[:80]}{'...' if len(sample.get('title', '')) > 80 else ''}\n"
                    info += f"  Authors: {sample.get('authors', 'N/A')}\n"
                    info += f"  Year: {sample.get('year', 'N/A')}\n"
                    info += f"  Journal: {sample.get('journal', 'N/A')}\n"
                    
                    # Check for abstracts
                    abstracts_count = sum(1 for s in studies if s.get('abstract'))
                    info += f"\n📝 Studies with abstracts: {abstracts_count:,} ({abstracts_count/total_studies*100:.1f}%)"
                    
            else:
                info = f"📄 File selected: {Path(file_path).name}\n"
                info += f"💾 Size: {Path(file_path).stat().st_size / 1024:.1f} KB\n"
                info += "\n⚠️ Running in demo mode - actual file parsing not available"
                
            self.file_info_text.setText(info)
            self.statusBar().showMessage(f"File loaded: {total_studies if 'total_studies' in locals() else '?'} studies found")
            
        except Exception as e:
            error_info = f"❌ Error loading file:\n\n{str(e)}\n\n"
            error_info += "Please check that the file format is supported and the file is not corrupted."
            self.file_info_text.setText(error_info)
            self.statusBar().showMessage("Error loading file")
            
    def load_default_config(self):
        """Load default configuration"""
        try:
            # Try to load from environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            self.openai_key.setText(os.getenv('OPENAI_API_KEY', ''))
            self.anthropic_key.setText(os.getenv('ANTHROPIC_API_KEY', ''))
            self.entrez_email.setText(os.getenv('ENTREZ_EMAIL', ''))
            
        except ImportError:
            pass
            
        # Update preview
        self.update_config_preview()
        
    def update_config_preview(self):
        """Update configuration preview"""
        config = self.get_current_config()
        
        # Create readable preview
        preview = "Current Configuration:\n\n"
        preview += f"🔑 APIs Configured: "
        apis = []
        if config.get('openai_api_key'): apis.append("OpenAI")
        if config.get('anthropic_api_key'): apis.append("Anthropic")
        preview += ", ".join(apis) if apis else "None"
        preview += "\n\n"
        
        preview += f"❓ Research Question: {config.get('research_question', 'Not set')[:100]}{'...' if len(config.get('research_question', '')) > 100 else ''}\n\n"
        
        pico = config.get('pico', {})
        preview += "🎯 PICO Framework:\n"
        preview += f"  P: {pico.get('population', 'Not set')[:50]}{'...' if len(pico.get('population', '')) > 50 else ''}\n"
        preview += f"  I: {pico.get('intervention', 'Not set')[:50]}{'...' if len(pico.get('intervention', '')) > 50 else ''}\n"
        preview += f"  C: {pico.get('comparison', 'Not set')[:50]}{'...' if len(pico.get('comparison', '')) > 50 else ''}\n"
        preview += f"  O: {pico.get('outcomes', 'Not set')[:50]}{'...' if len(pico.get('outcomes', '')) > 50 else ''}\n\n"
        
        preview += f"⚙️ Batch Size: {config.get('batch_size', 10)}\n"
        preview += f"🔄 Strategy: {config.get('processing_strategy', 'ADAPTIVE')}\n"
        preview += f"🤖 Models: {config.get('conservative_model', 'gpt-4o-mini')} / {config.get('liberal_model', 'claude-3-5-sonnet-20241022')}"
        
        self.config_preview.setText(preview)
        
    def get_current_config(self):
        """Get current configuration as dictionary"""
        return {
            'openai_api_key': self.openai_key.text(),
            'anthropic_api_key': self.anthropic_key.text(),
            'entrez_email': self.entrez_email.text(),
            'research_question': self.research_question.toPlainText(),
            'pico': {
                'population': self.population.toPlainText(),
                'intervention': self.intervention.toPlainText(),
                'comparison': self.comparison.toPlainText(),
                'outcomes': self.outcomes.toPlainText()
            },
            'batch_size': self.batch_size.value(),
            'processing_strategy': self.processing_strategy.currentText(),
            'conservative_model': self.conservative_model.currentText(),
            'liberal_model': self.liberal_model.currentText()
        }
        
    def start_screening(self):
        """Start the screening process"""
        if not self.selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a reference file first!")
            self.tabs.setCurrentIndex(0)  # Switch to project tab
            return
            
        config = self.get_current_config()
        
        if not config['openai_api_key']:
            QMessageBox.warning(self, "Missing API Key", "Please enter your OpenAI API key!")
            self.tabs.setCurrentIndex(1)  # Switch to config tab
            return
            
        if not config['research_question'].strip():
            QMessageBox.warning(self, "Missing Research Question", "Please enter your research question!")
            self.tabs.setCurrentIndex(1)  # Switch to config tab
            return
            
        # Confirm start
        reply = QMessageBox.question(
            self, 
            "Start Screening", 
            f"Start screening with the selected file?\n\nFile: {Path(self.selected_file).name}\nThis process may take some time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        # Prepare UI for screening
        self.start_screening_btn.setEnabled(False)
        self.stop_screening_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.live_results.clear()
        self.screening_results = []
        
        # Reset statistics
        self.update_statistics()
        
        # Start worker thread
        self.screening_worker = ScreeningWorker(self.selected_file, config)
        self.screening_worker.progress_updated.connect(self.progress_bar.setValue)
        self.screening_worker.status_updated.connect(self.status_label.setText)
        self.screening_worker.article_processed.connect(self.add_live_result)
        self.screening_worker.screening_complete.connect(self.screening_finished)
        self.screening_worker.error_occurred.connect(self.screening_error)
        
        self.screening_worker.start()
        
        # Switch to screening tab
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage("Screening in progress...")
        
    def stop_screening(self):
        """Stop the screening process"""
        if hasattr(self, 'screening_worker') and self.screening_worker.isRunning():
            self.screening_worker.cancel()
            self.screening_worker.wait(3000)  # Wait up to 3 seconds
            
        self.start_screening_btn.setEnabled(True)
        self.stop_screening_btn.setEnabled(False)
        self.status_label.setText("Screening stopped by user")
        self.statusBar().showMessage("Screening stopped")
        
    def add_live_result(self, result):
        """Add result to live results display"""
        item = QTreeWidgetItem(self.live_results)
        
        title = result.get('title', 'Untitled')[:60] + ('...' if len(result.get('title', '')) > 60 else '')
        decision = result.get('decision', 'Unknown')
        confidence = f"{result.get('confidence', 0)}%" if 'confidence' in result else 'N/A'
        status = "✅" if decision in ['INCLUDE', 'EXCLUDE'] else "⚠️" if decision == 'CONFLICT' else "❌"
        
        item.setText(0, title)
        item.setText(1, decision)
        item.setText(2, confidence)
        item.setText(3, status)
        
        # Color coding
        if decision == 'INCLUDE':
            item.setBackground(1, QColor(45, 123, 45, 100))
        elif decision == 'EXCLUDE':
            item.setBackground(1, QColor(123, 45, 45, 100))
        elif decision == 'CONFLICT':
            item.setBackground(1, QColor(123, 123, 45, 100))
        else:  # ERROR
            item.setBackground(1, QColor(100, 100, 100, 100))
            
        self.screening_results.append(result)
        self.update_statistics()
        
        # Auto-scroll to latest
        self.live_results.scrollToBottom()
        
    def update_statistics(self):
        """Update screening statistics"""
        if not self.screening_results:
            self.stats_included.setText("Included: 0")
            self.stats_excluded.setText("Excluded: 0")
            self.stats_conflicts.setText("Conflicts: 0")
            self.stats_errors.setText("Errors: 0")
            return
            
        included = len([r for r in self.screening_results if r.get('decision') == 'INCLUDE'])
        excluded = len([r for r in self.screening_results if r.get('decision') == 'EXCLUDE'])
        conflicts = len([r for r in self.screening_results if r.get('decision') == 'CONFLICT'])
        errors = len([r for r in self.screening_results if r.get('decision') == 'ERROR'])
        
        self.stats_included.setText(f"Included: {included}")
        self.stats_excluded.setText(f"Excluded: {excluded}")
        self.stats_conflicts.setText(f"Conflicts: {conflicts}")
        self.stats_errors.setText(f"Errors: {errors}")
        
    def screening_finished(self, results):
        """Handle screening completion"""
        self.start_screening_btn.setEnabled(True)
        self.stop_screening_btn.setEnabled(False)
        
        total = len(results)
        included = len([r for r in results if r.get('decision') == 'INCLUDE'])
        excluded = len([r for r in results if r.get('decision') == 'EXCLUDE'])
        conflicts = len([r for r in results if r.get('decision') == 'CONFLICT'])
        
        self.status_label.setText(f"✅ Screening complete! {total} articles processed.")
        self.statusBar().showMessage(f"Screening complete: {included} included, {excluded} excluded, {conflicts} conflicts")
        
        # Update results tab
        self.populate_results_tab(results)
        
        # Show completion message
        QMessageBox.information(
            self,
            "Screening Complete",
            f"Screening completed successfully!\n\n"
            f"Total articles: {total}\n"
            f"Included: {included}\n"
            f"Excluded: {excluded}\n"
            f"Conflicts: {conflicts}\n\n"
            f"Results are available in the Results tab."
        )
        
    def screening_error(self, error_message):
        """Handle screening error"""
        self.start_screening_btn.setEnabled(True)
        self.stop_screening_btn.setEnabled(False)
        self.status_label.setText(f"❌ Screening failed: {error_message}")
        self.statusBar().showMessage("Screening failed")
        
        QMessageBox.critical(
            self,
            "Screening Error",
            f"Screening failed with error:\n\n{error_message}\n\n"
            f"Please check your configuration and try again."
        )
        
    def populate_results_tab(self, results):
        """Populate the results tab with screening results"""
        self.results_tree.clear()
        
        for result in results:
            item = QTreeWidgetItem(self.results_tree)
            
            title = result.get('title', 'Untitled')
            authors = result.get('authors', 'Unknown')[:30] + ('...' if len(result.get('authors', '')) > 30 else '')
            decision = result.get('decision', 'Unknown')
            confidence = f"{result.get('confidence', 0)}%" if 'confidence' in result else 'N/A'
            reasoning = result.get('reasoning', 'No reasoning provided')[:50] + ('...' if len(result.get('reasoning', '')) > 50 else '')
            
            item.setText(0, title)
            item.setText(1, authors)
            item.setText(2, decision)
            item.setText(3, confidence)
            item.setText(4, reasoning)
            
            # Store full result in item data
            item.setData(0, Qt.ItemDataRole.UserRole, result)
            
            # Color coding
            if decision == 'INCLUDE':
                item.setBackground(2, QColor(45, 123, 45, 100))
            elif decision == 'EXCLUDE':
                item.setBackground(2, QColor(123, 45, 45, 100))
            elif decision == 'CONFLICT':
                item.setBackground(2, QColor(123, 123, 45, 100))
            else:  # ERROR
                item.setBackground(2, QColor(100, 100, 100, 100))
                
        # Update summary
        self.update_results_summary(results)
        
        # Switch to results tab
        self.tabs.setCurrentIndex(3)
        
    def update_results_summary(self, results):
        """Update results summary statistics"""
        total = len(results)
        if total == 0:
            self.summary_text.setText("No results to display.")
            return
            
        included = len([r for r in results if r.get('decision') == 'INCLUDE'])
        excluded = len([r for r in results if r.get('decision') == 'EXCLUDE'])
        conflicts = len([r for r in results if r.get('decision') == 'CONFLICT'])
        errors = len([r for r in results if r.get('decision') == 'ERROR'])
        
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in results if 'confidence' in r and isinstance(r.get('confidence'), (int, float))]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        summary = f"""📊 Screening Summary
        
Total Articles: {total:,}
✅ Included: {included:,} ({included/total*100:.1f}%)
❌ Excluded: {excluded:,} ({excluded/total*100:.1f}%)
⚠️ Conflicts: {conflicts:,} ({conflicts/total*100:.1f}%)
❌ Errors: {errors:,} ({errors/total*100:.1f}%)

📈 Average Confidence: {avg_confidence:.1f}%
🎯 Decision Rate: {(included+excluded)/total*100:.1f}%
⚡ Review Rate: {conflicts/total*100:.1f}%"""

        self.summary_text.setText(summary)
        
    def show_result_details(self, item):
        """Show detailed information for selected result"""
        result = item.data(0, Qt.ItemDataRole.UserRole)
        if not result:
            return
            
        details = f"""📖 Article Details

Title: {result.get('title', 'N/A')}

Authors: {result.get('authors', 'N/A')}

Journal: {result.get('journal', 'N/A')}

Year: {result.get('year', 'N/A')}

Abstract: {result.get('abstract', 'N/A')[:500]}{'...' if len(result.get('abstract', '')) > 500 else ''}

🤖 Screening Results:
Decision: {result.get('decision', 'N/A')}
Confidence: {result.get('confidence', 'N/A')}%

Reasoning: {result.get('reasoning', 'N/A')}

Timestamp: {result.get('timestamp', 'N/A')}"""

        self.details_text.setText(details)
        
    def filter_results(self):
        """Filter results based on selected criteria"""
        decision_filter = self.filter_decision.currentText()
        confidence_filter = self.filter_confidence.value()
        
        for i in range(self.results_tree.topLevelItemCount()):
            item = self.results_tree.topLevelItem(i)
            result = item.data(0, Qt.ItemDataRole.UserRole)
            
            show = True
            
            # Decision filter
            if decision_filter != "All" and result.get('decision') != decision_filter:
                show = False
                
            # Confidence filter
            confidence = result.get('confidence', 0)
            if isinstance(confidence, (int, float)) and confidence < confidence_filter:
                show = False
                
            item.setHidden(not show)
            
    def export_results(self, format_type):
        """Export results in specified format"""
        if not self.screening_results:
            QMessageBox.warning(self, "No Results", "No screening results to export!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Results as {format_type.upper()}",
            f"screening_results.{format_type}",
            f"{format_type.upper()} Files (*.{format_type})"
        )
        
        if not file_path:
            return
            
        try:
            if format_type == 'csv':
                self.export_csv(file_path)
            elif format_type == 'json':
                self.export_json(file_path)
            elif format_type == 'ris':
                self.export_ris(file_path)
                
            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export results:\n{str(e)}")
            
    def export_csv(self, file_path):
        """Export results as CSV"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Authors', 'Journal', 'Year', 'Abstract', 'Decision', 'Confidence', 'Reasoning', 'Timestamp'])
            
            for result in self.screening_results:
                writer.writerow([
                    result.get('title', ''),
                    result.get('authors', ''),
                    result.get('journal', ''),
                    result.get('year', ''),
                    result.get('abstract', ''),
                    result.get('decision', ''),
                    result.get('confidence', ''),
                    result.get('reasoning', ''),
                    result.get('timestamp', '')
                ])
                
    def export_json(self, file_path):
        """Export results as JSON"""
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_results': len(self.screening_results),
                'project_name': self.project_name.text(),
                'source_file': self.selected_file
            },
            'configuration': self.get_current_config(),
            'results': self.screening_results
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
    def export_ris(self, file_path):
        """Export results as RIS format"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for result in self.screening_results:
                f.write("TY  - JOUR\n")
                f.write(f"TI  - {result.get('title', '')}\n")
                if result.get('authors'):
                    for author in result.get('authors', '').split(', '):
                        if author.strip():
                            f.write(f"AU  - {author.strip()}\n")
                f.write(f"JO  - {result.get('journal', '')}\n")
                f.write(f"PY  - {result.get('year', '')}\n")
                f.write(f"AB  - {result.get('abstract', '')}\n")
                f.write(f"N1  - Decision: {result.get('decision', '')} (Confidence: {result.get('confidence', '')}%)\n")
                f.write(f"N1  - Reasoning: {result.get('reasoning', '')}\n")
                f.write("ER  - \n\n")
                
    def save_configuration(self):
        """Save current configuration"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "screening_config.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                config = self.get_current_config()
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")
                
    def load_configuration(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    
                self.set_configuration(config)
                QMessageBox.information(self, "Success", "Configuration loaded successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")
                
    def set_configuration(self, config):
        """Set UI elements from configuration dictionary"""
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
        
        # Set combo boxes
        strategy = config.get('processing_strategy', 'ADAPTIVE')
        index = self.processing_strategy.findText(strategy)
        if index >= 0:
            self.processing_strategy.setCurrentIndex(index)
            
        conservative = config.get('conservative_model', 'gpt-4o-mini')
        index = self.conservative_model.findText(conservative)
        if index >= 0:
            self.conservative_model.setCurrentIndex(index)
            
        liberal = config.get('liberal_model', 'claude-3-5-sonnet-20241022')
        index = self.liberal_model.findText(liberal)
        if index >= 0:
            self.liberal_model.setCurrentIndex(index)
            
        self.update_config_preview()
        
    def reset_configuration(self):
        """Reset configuration to defaults"""
        self.openai_key.clear()
        self.anthropic_key.clear()
        self.entrez_email.clear()
        self.research_question.clear()
        self.population.clear()
        self.intervention.clear()
        self.comparison.clear()
        self.outcomes.clear()
        self.batch_size.setValue(10)
        self.processing_strategy.setCurrentIndex(0)
        self.conservative_model.setCurrentIndex(0)
        self.liberal_model.setCurrentIndex(0)
        
        self.update_config_preview()
        
    def load_template(self):
        """Load configuration template"""
        # Placeholder for template loading
        QMessageBox.information(self, "Templates", "Template loading will be implemented in future version.")
        
    def save_template(self):
        """Save current configuration as template"""
        # Placeholder for template saving
        QMessageBox.information(self, "Templates", "Template saving will be implemented in future version.")
        
    def quick_start(self):
        """Quick start wizard"""
        QMessageBox.information(
            self,
            "Quick Start",
            "Quick Start Guide:\n\n"
            "1. Select a reference file (RIS, BibTeX, CSV, etc.)\n"
            "2. Configure your API keys and research question\n"
            "3. Set up PICO criteria\n"
            "4. Start screening process\n"
            "5. Review results and export\n\n"
            "For detailed help, click the Help button."
        )
        
    def show_help(self):
        """Show help information"""
        help_text = """
🔬 LLM Screening Tool - Help

📂 PROJECT TAB:
• Select reference files (RIS, BibTeX, CSV, XML, TXT)
• Set project name and description
• View file information and statistics

⚙️ CONFIGURATION TAB:
• Enter API keys for OpenAI and Anthropic
• Define research question and PICO criteria
• Adjust screening settings and model selection
• Save/load configuration profiles

🔬 SCREENING TAB:
• Start/stop screening process
• Monitor progress in real-time
• View live results and statistics
• Cancel screening if needed

📊 RESULTS TAB:
• Browse and filter screening results
• View detailed article information
• Export results in multiple formats
• Analyze screening statistics

🔑 REQUIRED:
• OpenAI API key (for conservative screening)
• Research question and PICO criteria
• Reference file in supported format

💡 TIP: Use the demo mode to test the interface
without API keys by selecting a file and starting
screening with empty API keys.
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Help - LLM Screening Tool")
        msg.setText(help_text)
        msg.setStyleSheet("QMessageBox { background-color: #2a2a2a; }")
        msg.exec()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("LLM Screening Tool")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Academic Research Tools")
    
    # Create and show main window
    window = LLMScreeningApp()
    window.show()
    
    # Connect configuration changes to preview updates
    for widget in [window.openai_key, window.anthropic_key, window.entrez_email, 
                   window.research_question, window.population, window.intervention,
                   window.comparison, window.outcomes]:
        if hasattr(widget, 'textChanged'):
            widget.textChanged.connect(window.update_config_preview)
        elif hasattr(widget, 'valueChanged'):
            widget.valueChanged.connect(window.update_config_preview)
        elif hasattr(widget, 'currentTextChanged'):
            widget.currentTextChanged.connect(window.update_config_preview)
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()