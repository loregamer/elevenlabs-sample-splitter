import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                            QProgressBar, QComboBox, QSpinBox, QMessageBox,
                            QListWidget, QListWidgetItem, QGroupBox, QScrollArea, QSizePolicy,
                            QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import math

class AudioDurationThread(QThread):
    """Thread for calculating total duration of audio files"""
    duration_calculated = pyqtSignal(int)
    
    def __init__(self, input_files):
        super().__init__()
        self.input_files = input_files
        
    def run(self):
        try:
            total_duration = 0
            
            # Process files in batches to avoid memory issues
            for file_path in self.input_files:
                try:
                    # Get audio duration without loading entire file into memory
                    segment = AudioSegment.from_file(file_path)
                    total_duration += len(segment)
                    
                    # Clear memory
                    del segment
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            
            self.duration_calculated.emit(total_duration)
        except Exception as e:
            print(f"Error in duration thread: {e}")


class AudioSplitterThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    completed = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_files, output_dir, output_format, max_length):
        super().__init__()
        self.input_files = input_files
        self.output_dir = output_dir
        self.output_format = output_format
        self.max_length = max_length  # in minutes
        
    def run(self):
        try:
            # Convert max_length from minutes to milliseconds
            max_length_ms = self.max_length * 60 * 1000
            
            # Process files in chunks to manage memory better
            self.status_updated.emit("Processing input files...")
            
            # If we have too many files, process them in batches
            batch_size = 20  # Process 20 files at a time to avoid memory issues
            total_files = len(self.input_files)
            total_batches = math.ceil(total_files / batch_size)
            
            if total_files > batch_size:                
                # For large file sets, process in batches
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_files)
                    
                    # Process this batch
                    self.status_updated.emit(f"Processing batch {batch_idx+1}/{total_batches} (files {start_idx+1}-{end_idx})")
                    batch_files = self.input_files[start_idx:end_idx]
                    
                    # Process this batch
                    self._process_audio_batch(batch_files, start_idx, batch_idx, total_batches)
                    
                    # Update progress based on completed batches
                    self.progress_updated.emit(int((batch_idx+1) / total_batches * 100))
                    
                self.status_updated.emit(f"Completed! {total_batches} batches processed.")
                self.completed.emit([])  # No single list of files, as we processed in batches
            else:
                # For smaller file sets, process normally
                merged_audio = AudioSegment.empty()
                
                for i, file_path in enumerate(self.input_files):
                    self.status_updated.emit(f"Loading file {i+1}/{len(self.input_files)}: {os.path.basename(file_path)}")
                    audio = AudioSegment.from_file(file_path)
                    merged_audio += audio
                    self.progress_updated.emit(int((i+1) / len(self.input_files) * 30))  # First 30% for loading
                
                # Process the merged audio
                output_files = self._process_merged_audio(merged_audio)
                self.status_updated.emit(f"Completed! {len(output_files)} files created.")
                self.completed.emit(output_files)
                
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
            
    def _process_audio_batch(self, batch_files, start_idx, batch_idx, total_batches):
        """Process a batch of audio files"""
        try:
            # Convert max_length from minutes to milliseconds
            max_length_ms = self.max_length * 60 * 1000
            
            # Merge files in this batch
            merged_audio = AudioSegment.empty()
            
            for i, file_path in enumerate(batch_files):
                self.status_updated.emit(f"Loading file {start_idx+i+1}/{len(self.input_files)}: {os.path.basename(file_path)}")
                audio = AudioSegment.from_file(file_path)
                merged_audio += audio
                # Clear memory
                del audio
            
            # Process the merged audio for this batch
            batch_output_dir = os.path.join(self.output_dir, f"batch_{batch_idx+1}")
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # Process this batch's merged audio
            self._process_merged_audio(merged_audio, batch_output_dir, batch_idx)
            
            # Clear memory
            del merged_audio
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing batch {batch_idx+1}: {str(e)}")
    
    def _process_merged_audio(self, merged_audio, output_dir=None, batch_idx=None):
        """Process a merged audio segment"""
        try:
            if output_dir is None:
                output_dir = self.output_dir
                
            # Convert max_length from minutes to milliseconds
            max_length_ms = self.max_length * 60 * 1000
            
            total_length_ms = len(merged_audio)
            if batch_idx is not None:
                self.status_updated.emit(f"Batch {batch_idx+1} audio length: {self._format_time(total_length_ms)}")
            else:
                self.status_updated.emit(f"Total audio length: {self._format_time(total_length_ms)}")
            
            # Calculate estimated number of output files
            min_segments = math.ceil(total_length_ms / max_length_ms)
            self.status_updated.emit(f"Estimated segments: {min_segments}")
            
            # Find non-silent sections to avoid cutting during speech
            self.status_updated.emit("Analyzing audio for speech segments...")
            non_silent_ranges = detect_nonsilent(
                merged_audio, 
                min_silence_len=500,  # Minimum silence length in ms
                silence_thresh=-40     # Silence threshold in dB
            )
            
            # Split audio at silent points
            self.status_updated.emit("Splitting audio at silent points...")
            output_files = []
            
            if not non_silent_ranges:
                self.error_occurred.emit("No speech detected in the audio.")
                return []
            
            # Determine split points based on max_length and silence
            split_points = [0]  # Start with the beginning of the audio
            current_position = 0
            
            while current_position < total_length_ms:
                target_position = current_position + max_length_ms
                
                if target_position >= total_length_ms:
                    break  # We've reached the end
                
                # Find the best silence point before target_position
                best_split_point = None
                
                # Look for silence AFTER the last speech segment that ends before target_position
                for i, (start, end) in enumerate(non_silent_ranges):
                    if end < target_position:
                        # If there's another speech segment after this one
                        if i + 1 < len(non_silent_ranges):
                            next_start = non_silent_ranges[i + 1][0]
                            # If there's silence between this and the next segment
                            if next_start - end > 500:  # At least 500ms of silence
                                best_split_point = end + 250  # Mid-point of silence
                        else:
                            # This is the last speech segment
                            best_split_point = end + 250
                    elif start > target_position:
                        # We've gone past the target position
                        break
                
                # If no good silence point found, use exactly max_length
                if best_split_point is None:
                    best_split_point = target_position
                
                split_points.append(best_split_point)
                current_position = best_split_point
            
            split_points.append(total_length_ms)  # End with the end of the audio
            
            # Create the output files
            prefix = f"batch{batch_idx+1}_" if batch_idx is not None else ""
            
            for i in range(len(split_points) - 1):
                start_time = split_points[i]
                end_time = split_points[i + 1]
                
                segment = merged_audio[start_time:end_time]
                output_filename = os.path.join(
                    output_dir, 
                    f"{prefix}segment_{i+1:03d}.{self.output_format}"
                )
                
                self.status_updated.emit(f"Exporting segment {i+1}/{len(split_points)-1}: {os.path.basename(output_filename)}")
                segment.export(output_filename, format=self.output_format)
                output_files.append(output_filename)
                
                # Update progress if this is not a batch process
                if batch_idx is None:
                    # Update progress (remaining 70% of progress bar)
                    self.progress_updated.emit(30 + int((i+1) / (len(split_points)-1) * 70))
            
            return output_files
                
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
            return []
    
    def _format_time(self, milliseconds):
        """Format milliseconds to a readable time string."""
        seconds = milliseconds / 1000
        minutes = seconds // 60
        seconds %= 60
        hours = minutes // 60
        minutes %= 60
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{seconds:.1f}s"


class ElevenLabsSampleSplitter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.input_files = []
        self.output_dir = ""
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("ElevenLabs Sample Splitter")
        self.setMinimumSize(600, 500)
        
        # Create central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Input Files Group
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)
        
        # List of selected files
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.files_list.setUniformItemSizes(True)  # Performance optimization
        input_layout.addWidget(self.files_list)
        
        # File count label
        self.file_count_label = QLabel("0 files selected")
        input_layout.addWidget(self.file_count_label)
        
        files_buttons_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.clicked.connect(self.add_files)
        self.remove_files_btn = QPushButton("Remove Selected")
        self.remove_files_btn.clicked.connect(self.remove_files)
        self.clear_files_btn = QPushButton("Clear All")
        self.clear_files_btn.clicked.connect(self.clear_files)
        
        files_buttons_layout.addWidget(self.add_files_btn)
        files_buttons_layout.addWidget(self.remove_files_btn)
        files_buttons_layout.addWidget(self.clear_files_btn)
        input_layout.addLayout(files_buttons_layout)
        
        # Output Settings Group
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        
        # Replace label with text box
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Enter output directory path")
        self.output_dir_edit.textChanged.connect(self.update_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        
        self.select_output_btn = QPushButton("Browse...")
        self.select_output_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.select_output_btn)
        output_layout.addLayout(output_dir_layout)
        
        # Output format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp3", "wav", "ogg", "flac"])
        format_layout.addWidget(self.format_combo)
        output_layout.addLayout(format_layout)
        
        # Max segment length
        max_length_layout = QHBoxLayout()
        max_length_layout.addWidget(QLabel("Maximum Segment Length (minutes):"))
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1, 30)
        self.max_length_spin.setValue(5)  # Default to 5 minutes
        max_length_layout.addWidget(self.max_length_spin)
        output_layout.addLayout(max_length_layout)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar)
        
        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.setVisible(False)
        status_layout.addWidget(self.warning_label)
        
        # Process button
        self.process_btn = QPushButton("Split Audio")
        self.process_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.process_btn.clicked.connect(self.process_audio)
        
        # Add groups to main layout
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(status_group)
        main_layout.addWidget(self.process_btn)
        
        # Set layout for central widget
        central_widget.setLayout(main_layout)
        
        # Thread for processing
        self.splitter_thread = None
        
    def add_files(self):
        file_dialog = QFileDialog()
        files, _ = file_dialog.getOpenFileNames(
            self, 
            "Select Audio Files", 
            "", 
            "Audio Files (*.mp3 *.wav *.ogg *.flac);;All Files (*)"
        )
        
        if files:
            self.input_files.extend(files)
            self.update_files_list()
            
            # Start duration analysis in a separate thread
            if hasattr(self, 'duration_thread') and self.duration_thread is not None and self.duration_thread.isRunning():
                # If a thread is already running, we'll update it later when it finishes
                pass
            else:
                self.start_duration_analysis()
    
    def remove_files(self):
        selected_items = self.files_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path in self.input_files:
                self.input_files.remove(file_path)
        
        self.update_files_list()
    
    def clear_files(self):
        self.input_files = []
        self.update_files_list()
    
    def update_files_list(self):
        # Batch update for better performance
        self.files_list.setUpdatesEnabled(False)
        self.files_list.clear()
        
        # Use a chunk size to avoid UI freezing with thousands of files
        chunk_size = 100
        total_files = len(self.input_files)
        
        # Update file count label
        self.file_count_label.setText(f"{total_files} files selected")
        
        # Add first chunk immediately
        end_idx = min(chunk_size, total_files)
        for i in range(end_idx):
            item = QListWidgetItem(os.path.basename(self.input_files[i]))
            item.setData(Qt.ItemDataRole.UserRole, self.input_files[i])
            self.files_list.addItem(item)
        
        self.files_list.setUpdatesEnabled(True)
        
        # If we have more files, add them incrementally to prevent UI freezing
        if total_files > chunk_size:
            self._current_batch_index = chunk_size
            QTimer.singleShot(10, self._add_next_files_batch)
    
    def update_output_dir(self, text):
        """Updates the output directory when the text changes"""
        self.output_dir = text
        
        # Trigger duration analysis if we have files and a valid directory
        if self.input_files and os.path.isdir(self.output_dir) and not (
            hasattr(self, 'duration_thread') and 
            self.duration_thread is not None and 
            self.duration_thread.isRunning()):
            self.start_duration_analysis()
    
    def select_output_dir(self):
        """Opens a file dialog to select output directory"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir_edit.setText(folder)
    
    def check_output_file_count(self):
        """Legacy method kept for compatibility, now redirects to start_duration_analysis"""
        if self.input_files and not (hasattr(self, 'duration_thread') and self.duration_thread is not None and self.duration_thread.isRunning()):
            self.start_duration_analysis()

    def _add_next_files_batch(self):
        """Add next batch of files to the list widget"""
        if not hasattr(self, '_current_batch_index'):
            return
            
        chunk_size = 100
        start_idx = self._current_batch_index
        end_idx = min(start_idx + chunk_size, len(self.input_files))
        
        self.files_list.setUpdatesEnabled(False)
        for i in range(start_idx, end_idx):
            item = QListWidgetItem(os.path.basename(self.input_files[i]))
            item.setData(Qt.ItemDataRole.UserRole, self.input_files[i])
            self.files_list.addItem(item)
        self.files_list.setUpdatesEnabled(True)
        
        if end_idx < len(self.input_files):
            self._current_batch_index = end_idx
            QTimer.singleShot(10, self._add_next_files_batch)
    
    def start_duration_analysis(self):
        """Start a thread to analyze audio durations"""
        self.duration_thread = AudioDurationThread(self.input_files)
        self.duration_thread.duration_calculated.connect(self.update_duration_warning)
        self.duration_thread.start()
    
    def update_duration_warning(self, total_duration):
        """Update warning based on calculated duration"""
        try:
            # Convert max_length from minutes to milliseconds
            max_length_ms = self.max_length_spin.value() * 60 * 1000
            
            # Estimate number of segments
            estimated_segments = math.ceil(total_duration / max_length_ms)
            
            if estimated_segments > 25:
                self.warning_label.setText(f"Warning: This will create approximately {estimated_segments} files!")
                self.warning_label.setVisible(True)
            else:
                self.warning_label.setVisible(False)
        except Exception as e:
            print(f"Error calculating file count: {e}")
            self.warning_label.setVisible(False)
    
    def process_audio(self):
        if not self.input_files:
            QMessageBox.warning(self, "Error", "No input files selected.")
            return
            
        if not self.output_dir:
            QMessageBox.warning(self, "Error", "No output directory selected.")
            return
        
        # Create and start the processing thread
        self.splitter_thread = AudioSplitterThread(
            self.input_files,
            self.output_dir,
            self.format_combo.currentText(),
            self.max_length_spin.value()
        )
        
        # Connect signals
        self.splitter_thread.progress_updated.connect(self.update_progress)
        self.splitter_thread.status_updated.connect(self.update_status)
        self.splitter_thread.completed.connect(self.processing_completed)
        self.splitter_thread.error_occurred.connect(self.processing_error)
        
        # Disable UI during processing
        self.set_ui_enabled(False)
        
        # Start processing
        self.splitter_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def processing_completed(self, output_files):
        self.set_ui_enabled(True)
        
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Successfully created {len(output_files)} audio segments."
        )
    
    def processing_error(self, error_message):
        self.set_ui_enabled(True)
        
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during processing:\n{error_message}"
        )
    
    def set_ui_enabled(self, enabled):
        self.add_files_btn.setEnabled(enabled)
        self.remove_files_btn.setEnabled(enabled)
        self.clear_files_btn.setEnabled(enabled)
        self.select_output_btn.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.max_length_spin.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.files_list.setEnabled(enabled)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ElevenLabsSampleSplitter()
    window.show()
    sys.exit(app.exec())