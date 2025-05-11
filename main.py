import sys
import os
import math
import tempfile
import concurrent.futures
import gc
import logging
import time
import json
import numpy as np
from functools import partial
from pathlib import Path

# Import theme
from theme import get_theme

# Import QtAwesome for icons
try:
    import qtawesome as qta
except ImportError:
    print("QtAwesome not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qtawesome"])
    import qtawesome as qta

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QFileDialog, QLabel,
                            QProgressBar, QComboBox, QSpinBox, QMessageBox,
                            QListWidget, QListWidgetItem, QGroupBox, QScrollArea, QSizePolicy,
                            QLineEdit, QTabWidget, QCheckBox, QSlider, QDoubleSpinBox,
                            QRadioButton, QButtonGroup, QMenu, QToolButton, QDialog,
                            QDialogButtonBox, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QRunnable, QThreadPool
from PyQt6.QtGui import QIcon, QAction, QFileSystemModel

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.effects import normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.log'))
    ]
)
logger = logging.getLogger('ElevenLabsSplitter')

# Signal dispatcher for worker threads
class WorkerSignals(QObject):
    """Defines the signals available from worker threads."""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


# Worker class for parallel processing
class Worker(QRunnable):
    """Worker thread for parallel processing tasks"""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.kwargs['progress_callback'] = self.signals.progress
        self.kwargs['status_callback'] = self.signals.status

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}", exc_info=True)
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


# File processing utilities
class AudioUtils:
    """Utility functions for audio processing"""
    @staticmethod
    def get_audio_duration(file_path, progress_callback=None, status_callback=None):
        """Get duration of an audio file without loading the entire file"""
        try:
            if status_callback:
                status_callback(f"Checking duration: {os.path.basename(file_path)}")
                
            segment = AudioSegment.from_file(file_path)
            duration = len(segment)
            
            # Clear memory
            del segment
            gc.collect()
            
            return duration
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {str(e)}")
            return 0
    
    @staticmethod
    def process_chunk(file_paths, temp_dir=None, progress_callback=None, status_callback=None):
        """Process a chunk of audio files and combine them"""
        try:
            if not file_paths:
                return None, 0
                
            if not temp_dir:
                temp_dir = tempfile.mkdtemp()
                
            combined = AudioSegment.empty()
            total_files = len(file_paths)
            
            for i, file_path in enumerate(file_paths):
                if status_callback:
                    status_callback(f"Loading {i+1}/{total_files}: {os.path.basename(file_path)}")
                    
                try:
                    audio = AudioSegment.from_file(file_path)
                    combined += audio
                    
                    # Clear memory
                    del audio
                    gc.collect()
                    
                    if progress_callback:
                        progress_callback(int((i+1) / total_files * 100))
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    
            # Save temporary file
            if len(combined) > 0:
                temp_file = os.path.join(temp_dir, f"temp_chunk_{time.time()}.wav")
                combined.export(temp_file, format="wav")
                
                # Clear memory
                duration = len(combined)
                del combined
                gc.collect()
                
                return temp_file, duration
            else:
                return None, 0
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return None, 0


class AudioDurationThread(QThread):
    """Thread for calculating total duration of audio files with parallel processing"""
    duration_calculated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, input_files):
        super().__init__()
        self.input_files = input_files
        self.thread_pool = QThreadPool()
        
    def run(self):
        try:
            if not self.input_files:
                self.duration_calculated.emit(0)
                return
                
            self.status_updated.emit("Calculating total audio duration...")
            total_duration = 0
            
            # Use multiprocessing for faster duration calculation
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                # Submit all files for processing
                futures = [executor.submit(AudioUtils.get_audio_duration, file_path) 
                          for file_path in self.input_files]
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        duration = future.result()
                        total_duration += duration
                        progress = int((i+1) / len(futures) * 100)
                        self.status_updated.emit(f"Duration calculation: {progress}%")
                    except Exception as e:
                        logger.error(f"Error in duration calculation: {str(e)}")
            
            self.duration_calculated.emit(total_duration)
        except Exception as e:
            logger.error(f"Error in duration thread: {str(e)}", exc_info=True)
            self.duration_calculated.emit(0)


class AudioSplitterThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    completed = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    merged_file_created = pyqtSignal(str)
    
    def __init__(self, input_files, output_dir, settings):
        super().__init__()
        self.input_files = input_files
        self.output_dir = output_dir
        self.settings = settings
        self.temp_dir = None
        self.merged_file = None
        self.temp_files = []
        
    def run(self):
        try:
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp()
            
            # Configure processing parameters
            max_length_ms = self.settings['max_length'] * 60 * 1000
            output_format = self.settings['output_format']
            bitrate = self.settings.get('bitrate', '192k')
            sample_rate = self.settings.get('sample_rate', 44100)
            channels = self.settings.get('channels', 2)
            normalize_audio = self.settings.get('normalize', False)
            fade_in = self.settings.get('fade_in', 0)
            fade_out = self.settings.get('fade_out', 0)
            min_silence_len = self.settings.get('min_silence_len', 500)
            silence_thresh = self.settings.get('silence_thresh', -40)
            save_merged = self.settings.get('save_merged', False)
            naming_pattern = self.settings.get('naming_pattern', 'segment_{num:03d}')
            
            # Process files in parallel chunks to optimize memory usage
            self.status_updated.emit("Preparing to combine audio files...")
            total_files = len(self.input_files)
            
            if total_files == 0:
                self.error_occurred.emit("No input files selected.")
                return
            
            # Determine optimal chunk size based on available memory and CPU cores
            # For most systems, processing 4-10 files at once is a good balance
            num_cores = os.cpu_count() or 4
            chunk_size = min(max(4, num_cores), 10, total_files)
            num_chunks = math.ceil(total_files / chunk_size)
            
            # Process in chunks using multiprocessing
            chunk_results = []
            
            self.status_updated.emit(f"Processing files in {num_chunks} chunks...")
            
            # Create chunks of files
            file_chunks = [self.input_files[i:i+chunk_size] for i in range(0, total_files, chunk_size)]
            
            # Process each chunk in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
                # Submit all chunks for processing
                future_to_chunk = {executor.submit(AudioUtils.process_chunk, chunk, self.temp_dir): i 
                                 for i, chunk in enumerate(file_chunks)}
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                    try:
                        chunk_idx = future_to_chunk[future]
                        temp_file, duration = future.result()
                        
                        if temp_file:
                            chunk_results.append(temp_file)
                            self.temp_files.append(temp_file)  # Keep track for cleanup
                        
                        # Update progress (first 50% for combining)
                        progress = int((i+1) / len(file_chunks) * 50)
                        self.progress_updated.emit(progress)
                        self.status_updated.emit(f"Processed chunk {i+1}/{len(file_chunks)}")
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        self.error_occurred.emit(f"Error processing chunk: {str(e)}")
                        return
            
            # Now combine the chunk results
            self.status_updated.emit("Combining processed chunks...")
            
            # If we only have one chunk, use it directly
            if len(chunk_results) == 1:
                merged_audio_file = chunk_results[0]
            else:
                # Combine all chunks into final merged file
                merged_audio = AudioSegment.empty()
                
                for i, temp_file in enumerate(chunk_results):
                    try:
                        self.status_updated.emit(f"Combining chunk {i+1}/{len(chunk_results)}...")
                        chunk_audio = AudioSegment.from_file(temp_file)
                        merged_audio += chunk_audio
                        
                        # Clear memory
                        del chunk_audio
                        gc.collect()
                        
                        # Update progress (from 50% to 60%)
                        progress = 50 + int((i+1) / len(chunk_results) * 10)
                        self.progress_updated.emit(progress)
                    except Exception as e:
                        logger.error(f"Error combining chunk {i+1}: {str(e)}")
                        self.error_occurred.emit(f"Error combining audio chunks: {str(e)}")
                        return
                
                # Apply audio processing to merged file if needed
                if normalize_audio:
                    self.status_updated.emit("Normalizing audio...")
                    merged_audio = normalize(merged_audio)
                
                # Save merged file to temp directory
                merged_audio_file = os.path.join(self.temp_dir, f"merged_audio_{time.time()}.wav")
                merged_audio.export(merged_audio_file, format="wav")
                self.temp_files.append(merged_audio_file)  # Keep track for cleanup
                
                # Clear memory
                del merged_audio
                gc.collect()
            
            # Save merged file if requested
            if save_merged:
                merged_output = os.path.join(self.output_dir, f"merged.{output_format}")
                
                # Load merged file
                merged_audio = AudioSegment.from_file(merged_audio_file)
                
                # Export with specified settings
                export_params = {
                    'format': output_format,
                    'bitrate': bitrate,
                    'parameters': ['-ac', str(channels), '-ar', str(sample_rate)]
                }
                
                # Apply effects if needed
                if fade_in > 0:
                    merged_audio = merged_audio.fade_in(fade_in)
                if fade_out > 0:
                    merged_audio = merged_audio.fade_out(fade_out)
                
                self.status_updated.emit(f"Saving merged file: {os.path.basename(merged_output)}")
                merged_audio.export(merged_output, **export_params)
                
                # Notify about merged file creation
                self.merged_file_created.emit(merged_output)
                self.merged_file = merged_output
                
                # Clear memory
                del merged_audio
                gc.collect()
            
            # Process the merged audio file
            self.status_updated.emit("Processing merged audio for splitting...")
            output_files = self._process_merged_audio(
                merged_audio_file, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                max_length_ms=max_length_ms,
                output_format=output_format,
                bitrate=bitrate,
                sample_rate=sample_rate,
                channels=channels,
                fade_in=fade_in,
                fade_out=fade_out,
                naming_pattern=naming_pattern
            )
            
            if output_files:
                self.status_updated.emit(f"Completed! {len(output_files)} files created.")
                self.completed.emit(output_files)
            else:
                self.error_occurred.emit("No output files were created.")
                
        except Exception as e:
            logger.error(f"Error in splitter thread: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Error: {str(e)}")
        finally:
            # Clean up temp files
            self._cleanup_temp_files()
            
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            # Remove temp files
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_file}: {str(e)}")
            
            # Remove temp directory if it exists and is empty
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    os.rmdir(self.temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove temp directory {self.temp_dir}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    def _process_merged_audio(self, merged_audio_file, min_silence_len=500, silence_thresh=-40, 
                             max_length_ms=300000, output_format="mp3", bitrate="192k", 
                             sample_rate=44100, channels=2, fade_in=0, fade_out=0,
                             naming_pattern="segment_{num:03d}"):
        """Process a merged audio file with optimized silence detection"""
        try:
            # Load the merged audio
            self.status_updated.emit("Loading merged audio...")
            merged_audio = AudioSegment.from_file(merged_audio_file)
            
            total_length_ms = len(merged_audio)
            self.status_updated.emit(f"Total audio length: {self._format_time(total_length_ms)}")
            
            # Calculate estimated number of output files
            min_segments = math.ceil(total_length_ms / max_length_ms)
            self.status_updated.emit(f"Estimated segments: {min_segments}")
            
            # Find non-silent sections using optimized algorithm
            self.status_updated.emit("Analyzing audio for speech segments...")
            
            # Use numpy for faster processing of large audio files
            # Convert audio to numpy array for faster processing
            samples = np.array(merged_audio.get_array_of_samples())
            
            # Detect non-silent ranges using pydub
            non_silent_ranges = detect_nonsilent(
                merged_audio, 
                min_silence_len=min_silence_len,  
                silence_thresh=silence_thresh
            )
            
            # Split audio at silent points
            self.status_updated.emit("Finding optimal split points...")
            output_files = []
            
            if not non_silent_ranges:
                self.error_occurred.emit("No speech detected in the audio.")
                return []
            
            # Optimize split points calculation
            split_points = self._calculate_split_points(
                non_silent_ranges, total_length_ms, max_length_ms, min_silence_len)
            
            # Create the output files
            segment_count = len(split_points) - 1
            
            # Export settings
            export_params = {
                'format': output_format,
                'bitrate': bitrate,
                'parameters': ['-ac', str(channels), '-ar', str(sample_rate)]
            }
            
            # Use multiprocessing for parallel export
            self.status_updated.emit("Exporting audio segments...")
            
            # Create segments
            segments = []
            for i in range(segment_count):
                start_time = split_points[i]
                end_time = split_points[i + 1]
                
                # Get the segment
                segment = merged_audio[start_time:end_time]
                
                # Apply effects if needed
                if fade_in > 0:
                    segment = segment.fade_in(fade_in)
                if fade_out > 0:
                    segment = segment.fade_out(fade_out)
                
                # Generate filename using pattern
                segment_num = i + 1
                filename = naming_pattern.format(num=segment_num, total=segment_count)
                output_filename = os.path.join(self.output_dir, f"{filename}.{output_format}")
                
                segments.append((segment, output_filename, export_params, i, segment_count))
                
                # Update progress (from 60% to 70%)
                self.progress_updated.emit(60 + int((i+1) / segment_count * 10))
            
            # Clear original audio to free memory
            del merged_audio
            gc.collect()
            
            # Export segments in parallel
            exported_files = []
            
            # Define export function for parallel processing
            def export_segment(segment_data, progress_callback=None, status_callback=None):
                segment, output_filename, params, idx, total = segment_data
                
                try:
                    if status_callback:
                        status_callback(f"Exporting segment {idx+1}/{total}: {os.path.basename(output_filename)}")
                    
                    # Export the segment
                    segment.export(output_filename, **params)
                    
                    # Clear memory
                    del segment
                    gc.collect()
                    
                    if progress_callback:
                        progress_callback(int((idx+1) / total * 100))
                    
                    return output_filename
                except Exception as e:
                    logger.error(f"Error exporting segment {idx+1}: {str(e)}")
                    return None
            
            # Use multithreading for exports
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                future_to_segment = {executor.submit(export_segment, segment_data): i 
                                   for i, segment_data in enumerate(segments)}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_segment)):
                    try:
                        output_file = future.result()
                        if output_file:
                            exported_files.append(output_file)
                        
                        # Update progress (from 70% to 100%)
                        progress = 70 + int((i+1) / len(segments) * 30)
                        self.progress_updated.emit(progress)
                    except Exception as e:
                        logger.error(f"Error in export process: {str(e)}")
            
            return exported_files
                
        except Exception as e:
            logger.error(f"Error processing merged audio: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Error processing audio: {str(e)}")
            return []
    
    def _calculate_split_points(self, non_silent_ranges, total_length_ms, max_length_ms, min_silence_len):
        """Calculate optimal split points based on silence detection"""
        # Start with the beginning of the audio
        split_points = [0]
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
                        if next_start - end > min_silence_len:
                            best_split_point = end + min_silence_len // 2  # Mid-point of silence
                    else:
                        # This is the last speech segment
                        best_split_point = end + min_silence_len // 2
                elif start > target_position:
                    # We've gone past the target position
                    break
            
            # If no good silence point found, use exactly max_length
            if best_split_point is None:
                # Look ahead for the next silence after target_position
                for i, (start, end) in enumerate(non_silent_ranges):
                    if start > target_position:
                        # Found a speech segment that starts after our target
                        # If there's a previous segment with a gap, use that
                        if i > 0:
                            prev_end = non_silent_ranges[i-1][1]
                            if start - prev_end > min_silence_len:
                                best_split_point = prev_end + min_silence_len // 2
                                break
                
                # If still no good point, use target position
                if best_split_point is None:
                    best_split_point = target_position
            
            split_points.append(best_split_point)
            current_position = best_split_point
        
        split_points.append(total_length_ms)  # End with the end of the audio
        return split_points
    
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


class SettingsDialog(QDialog):
    """Dialog for advanced settings"""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setMinimumWidth(500)
        
        # Get theme from parent
        if parent is not None and hasattr(parent, 'theme'):
            self.theme = parent.theme
        else:
            self.theme = get_theme("discord")
            
        # Apply dialog theme
        self.setStyleSheet(self.theme["dialog"])
        
        if settings is None:
            self.settings = {}
        else:
            self.settings = settings.copy()
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs for settings categories
        tabs = QTabWidget()
        tabs.setStyleSheet(self.theme["tab_widget"])
        
        # Audio Format tab
        format_tab = QWidget()
        format_layout = QVBoxLayout(format_tab)
        
        # Sample rate
        sample_rate_layout = QHBoxLayout()
        sample_rate_label = QLabel("Sample Rate (Hz):")
        sample_rate_label.setStyleSheet(self.theme["label"])
        sample_rate_layout.addWidget(sample_rate_label)
        
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.setStyleSheet(self.theme["input"])
        self.sample_rate_combo.addItems(["8000", "16000", "22050", "44100", "48000", "96000"])
        self.sample_rate_combo.setCurrentText("44100")
        sample_rate_layout.addWidget(self.sample_rate_combo)
        format_layout.addLayout(sample_rate_layout)
        
        # Bit rate (for compressed formats)
        bitrate_layout = QHBoxLayout()
        bitrate_label = QLabel("Bit Rate:")
        bitrate_label.setStyleSheet(self.theme["label"])
        bitrate_layout.addWidget(bitrate_label)
        
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setStyleSheet(self.theme["input"])
        self.bitrate_combo.addItems(["64k", "128k", "192k", "256k", "320k"])
        self.bitrate_combo.setCurrentText("192k")
        bitrate_layout.addWidget(self.bitrate_combo)
        format_layout.addLayout(bitrate_layout)
        
        # Channels
        channels_layout = QHBoxLayout()
        channels_label = QLabel("Channels:")
        channels_label.setStyleSheet(self.theme["label"])
        channels_layout.addWidget(channels_label)
        
        self.channels_group = QButtonGroup(self)
        self.mono_radio = QRadioButton("Mono (1)")
        self.mono_radio.setStyleSheet(self.theme["radio_button"])
        self.stereo_radio = QRadioButton("Stereo (2)")
        self.stereo_radio.setStyleSheet(self.theme["radio_button"])
        self.stereo_radio.setChecked(True)  # Default to stereo
        self.channels_group.addButton(self.mono_radio, 1)
        self.channels_group.addButton(self.stereo_radio, 2)
        channels_layout.addWidget(self.mono_radio)
        channels_layout.addWidget(self.stereo_radio)
        format_layout.addLayout(channels_layout)
        
        # Audio Processing tab
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        
        # Normalization
        self.normalize_check = QCheckBox("Normalize Audio")
        self.normalize_check.setStyleSheet(self.theme["check_box"])
        self.normalize_check.setToolTip("Normalize audio volume across all segments")
        processing_layout.addWidget(self.normalize_check)
        
        # Fade in/out
        fade_layout = QGridLayout()
        
        fade_in_label = QLabel("Fade In (ms):")
        fade_in_label.setStyleSheet(self.theme["label"])
        fade_layout.addWidget(fade_in_label, 0, 0)
        
        self.fade_in_spin = QSpinBox()
        self.fade_in_spin.setStyleSheet(self.theme["input"])
        self.fade_in_spin.setRange(0, 5000)
        self.fade_in_spin.setSingleStep(100)
        self.fade_in_spin.setValue(0)
        fade_layout.addWidget(self.fade_in_spin, 0, 1)
        
        fade_out_label = QLabel("Fade Out (ms):")
        fade_out_label.setStyleSheet(self.theme["label"])
        fade_layout.addWidget(fade_out_label, 1, 0)
        
        self.fade_out_spin = QSpinBox()
        self.fade_out_spin.setStyleSheet(self.theme["input"])
        self.fade_out_spin.setRange(0, 5000)
        self.fade_out_spin.setSingleStep(100)
        self.fade_out_spin.setValue(0)
        fade_layout.addWidget(self.fade_out_spin, 1, 1)
        processing_layout.addLayout(fade_layout)
        
        # Save merged file option
        self.save_merged_check = QCheckBox("Save Merged File Before Splitting")
        self.save_merged_check.setStyleSheet(self.theme["check_box"])
        self.save_merged_check.setToolTip("Save the combined audio file before splitting it into segments")
        processing_layout.addWidget(self.save_merged_check)
        
        # Silence Detection tab
        silence_tab = QWidget()
        silence_layout = QVBoxLayout(silence_tab)
        
        # Min silence length
        min_silence_layout = QHBoxLayout()
        min_silence_label = QLabel("Minimum Silence Length (ms):")
        min_silence_label.setStyleSheet(self.theme["label"])
        min_silence_layout.addWidget(min_silence_label)
        
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setStyleSheet(self.theme["input"])
        self.min_silence_spin.setRange(100, 2000)
        self.min_silence_spin.setSingleStep(50)
        self.min_silence_spin.setValue(500)
        min_silence_layout.addWidget(self.min_silence_spin)
        silence_layout.addLayout(min_silence_layout)
        
        # Silence threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Silence Threshold (dB):")
        threshold_label.setStyleSheet(self.theme["label"])
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setStyleSheet(self.theme["input"])
        self.threshold_spin.setRange(-80, -10)
        self.threshold_spin.setValue(-40)
        threshold_layout.addWidget(self.threshold_spin)
        silence_layout.addLayout(threshold_layout)
        
        # Output Files tab
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        
        # Naming pattern
        naming_layout = QHBoxLayout()
        naming_label = QLabel("Naming Pattern:")
        naming_label.setStyleSheet(self.theme["label"])
        naming_layout.addWidget(naming_label)
        
        self.naming_edit = QLineEdit("segment_{num:03d}")
        self.naming_edit.setStyleSheet(self.theme["input"])
        self.naming_edit.setToolTip("Use {num} for segment number and {total} for total segments")
        naming_layout.addWidget(self.naming_edit)
        output_layout.addLayout(naming_layout)
        
        # Add tabs to the tab widget
        tabs.addTab(format_tab, "Audio Format")
        tabs.addTab(processing_tab, "Processing")
        tabs.addTab(silence_tab, "Silence Detection")
        tabs.addTab(output_tab, "Output Files")
        layout.addWidget(tabs)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # Apply theme to buttons
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        
        if ok_button:
            ok_button.setStyleSheet(self.theme["button"])
            ok_button.setIcon(qta.icon('fa5s.check', color=self.theme["text_heading"]))
        
        if cancel_button:
            cancel_button.setStyleSheet(self.theme["secondary_button"])
            cancel_button.setIcon(qta.icon('fa5s.times', color=self.theme["text_heading"]))
            
        layout.addWidget(buttons)
    
    def load_settings(self):
        # Load settings into UI widgets
        self.sample_rate_combo.setCurrentText(str(self.settings.get('sample_rate', 44100)))
        self.bitrate_combo.setCurrentText(self.settings.get('bitrate', '192k'))
        
        channels = self.settings.get('channels', 2)
        if channels == 1:
            self.mono_radio.setChecked(True)
        else:
            self.stereo_radio.setChecked(True)
        
        self.normalize_check.setChecked(self.settings.get('normalize', False))
        self.fade_in_spin.setValue(self.settings.get('fade_in', 0))
        self.fade_out_spin.setValue(self.settings.get('fade_out', 0))
        self.save_merged_check.setChecked(self.settings.get('save_merged', False))
        
        self.min_silence_spin.setValue(self.settings.get('min_silence_len', 500))
        self.threshold_spin.setValue(self.settings.get('silence_thresh', -40))
        
        self.naming_edit.setText(self.settings.get('naming_pattern', 'segment_{num:03d}'))
    
    def get_settings(self):
        # Update settings from UI widgets
        self.settings['sample_rate'] = int(self.sample_rate_combo.currentText())
        self.settings['bitrate'] = self.bitrate_combo.currentText()
        self.settings['channels'] = 1 if self.mono_radio.isChecked() else 2
        
        self.settings['normalize'] = self.normalize_check.isChecked()
        self.settings['fade_in'] = self.fade_in_spin.value()
        self.settings['fade_out'] = self.fade_out_spin.value()
        self.settings['save_merged'] = self.save_merged_check.isChecked()
        
        self.settings['min_silence_len'] = self.min_silence_spin.value()
        self.settings['silence_thresh'] = self.threshold_spin.value()
        
        self.settings['naming_pattern'] = self.naming_edit.text()
        
        return self.settings


class ElevenLabsSampleSplitter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.input_files = []
        self.output_dir = ""
        self.settings = {}
        self.init_default_settings()
        
        # Load theme
        self.theme = get_theme("discord")
        self.apply_theme()
        
        self.init_ui()
        
    def init_default_settings(self):
        """Initialize default settings"""
        self.settings = {
            'output_format': 'mp3',
            'max_length': 5,  # minutes
            'bitrate': '192k',
            'sample_rate': 44100,
            'channels': 2,
            'normalize': False,
            'fade_in': 0,
            'fade_out': 0,
            'min_silence_len': 500,
            'silence_thresh': -40,
            'save_merged': False,
            'naming_pattern': 'segment_{num:03d}'
        }
        
    def apply_theme(self):
        """Apply the selected theme to the application"""
        # Apply window style
        self.setStyleSheet(self.theme["window"])
        
        # Set application icon
        app_icon = qta.icon('fa5s.wave-square', color=self.theme["primary"])
        self.setWindowIcon(app_icon)
        
    def init_ui(self):
        self.setWindowTitle("ElevenLabs Sample Splitter")
        self.setMinimumSize(700, 600)
        
        # Create central widget with tabs
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Input Files Group
        input_group = QGroupBox("Input Files")
        input_group.setStyleSheet(self.theme["group_box"])
        input_layout = QVBoxLayout(input_group)
        
        # List of selected files
        self.files_list = QListWidget()
        self.files_list.setStyleSheet(self.theme["list_widget"])
        self.files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.files_list.setUniformItemSizes(True)  # Performance optimization
        input_layout.addWidget(self.files_list)
        
        # File count label
        self.file_count_label = QLabel("0 files selected")
        self.file_count_label.setStyleSheet(self.theme["label"])
        input_layout.addWidget(self.file_count_label)
        
        files_buttons_layout = QHBoxLayout()
        
        # Add Files button with icon
        self.add_files_btn = QPushButton(" Add Files")
        self.add_files_btn.setIcon(qta.icon('fa5s.file-audio', color=self.theme["text_heading"]))
        self.add_files_btn.setStyleSheet(self.theme["button"])
        self.add_files_btn.clicked.connect(self.add_files)
        
        # Remove Selected button with icon
        self.remove_files_btn = QPushButton(" Remove Selected")
        self.remove_files_btn.setIcon(qta.icon('fa5s.trash-alt', color=self.theme["text_heading"]))
        self.remove_files_btn.setStyleSheet(self.theme["secondary_button"])
        self.remove_files_btn.clicked.connect(self.remove_files)
        
        # Clear All button with icon
        self.clear_files_btn = QPushButton(" Clear All")
        self.clear_files_btn.setIcon(qta.icon('fa5s.times-circle', color=self.theme["text_heading"]))
        self.clear_files_btn.setStyleSheet(self.theme["danger_button"])
        self.clear_files_btn.clicked.connect(self.clear_files)
        
        files_buttons_layout.addWidget(self.add_files_btn)
        files_buttons_layout.addWidget(self.remove_files_btn)
        files_buttons_layout.addWidget(self.clear_files_btn)
        input_layout.addLayout(files_buttons_layout)
        
        # Output Settings Group
        output_group = QGroupBox("Output Settings")
        output_group.setStyleSheet(self.theme["group_box"])
        output_layout = QVBoxLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("Output Directory:")
        output_dir_label.setStyleSheet(self.theme["label"])
        output_dir_layout.addWidget(output_dir_label)
        
        # Text box for output directory
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setStyleSheet(self.theme["input"])
        self.output_dir_edit.setPlaceholderText("Enter output directory path")
        self.output_dir_edit.textChanged.connect(self.update_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        
        self.select_output_btn = QPushButton(" Browse...")
        self.select_output_btn.setIcon(qta.icon('fa5s.folder-open', color=self.theme["text_heading"]))
        self.select_output_btn.setStyleSheet(self.theme["secondary_button"])
        self.select_output_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.select_output_btn)
        output_layout.addLayout(output_dir_layout)
        
        # Basic output settings
        basic_settings_layout = QGridLayout()
        
        # Output format
        format_label = QLabel("Output Format:")
        format_label.setStyleSheet(self.theme["label"])
        basic_settings_layout.addWidget(format_label, 0, 0)
        
        self.format_combo = QComboBox()
        self.format_combo.setStyleSheet(self.theme["input"])
        self.format_combo.addItems(["mp3", "wav", "ogg", "flac"])
        self.format_combo.currentTextChanged.connect(self.update_format_setting)
        basic_settings_layout.addWidget(self.format_combo, 0, 1)
        
        # Max segment length
        length_label = QLabel("Maximum Segment Length (minutes):")
        length_label.setStyleSheet(self.theme["label"])
        basic_settings_layout.addWidget(length_label, 1, 0)
        
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setStyleSheet(self.theme["input"])
        self.max_length_spin.setRange(1, 60)
        self.max_length_spin.setValue(self.settings['max_length'])
        self.max_length_spin.valueChanged.connect(self.update_max_length_setting)
        basic_settings_layout.addWidget(self.max_length_spin, 1, 1)
        
        # Advanced settings button
        self.advanced_settings_btn = QPushButton(" Advanced Settings")
        self.advanced_settings_btn.setIcon(qta.icon('fa5s.cogs', color=self.theme["text_heading"]))
        self.advanced_settings_btn.setStyleSheet(self.theme["secondary_button"])
        self.advanced_settings_btn.clicked.connect(self.show_advanced_settings)
        basic_settings_layout.addWidget(self.advanced_settings_btn, 2, 0, 1, 2)
        
        output_layout.addLayout(basic_settings_layout)
        
        # Status display
        status_group = QGroupBox("Status")
        status_group.setStyleSheet(self.theme["group_box"])
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(self.theme["status_label"])
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(self.theme["progress_bar"])
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar)
        
        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet(self.theme["warning_label"])
        self.warning_label.setVisible(False)
        status_layout.addWidget(self.warning_label)
        
        # Process button
        self.process_btn = QPushButton(" Split Audio")
        self.process_btn.setIcon(qta.icon('fa5s.cut', color=self.theme["text_heading"]))
        self.process_btn.setStyleSheet(self.theme["button"] + "font-weight: bold; padding: 12px; font-size: 14px;")
        self.process_btn.clicked.connect(self.process_audio)
        
        # Add groups to main layout
        main_layout.addWidget(input_group, 3)  # Give input group more space
        main_layout.addWidget(output_group, 1)
        main_layout.addWidget(status_group, 1)
        main_layout.addWidget(self.process_btn)
        
        # Set layout for central widget
        central_widget.setLayout(main_layout)
        
        # Thread for processing
        self.splitter_thread = None
        self.duration_thread = None
        
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
    
    def remove_files(self):
        selected_items = self.files_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path in self.input_files:
                self.input_files.remove(file_path)
        
        self.update_files_list()
        
        # Update duration analysis
        if self.input_files and not (
            hasattr(self, 'duration_thread') and 
            self.duration_thread is not None and 
            self.duration_thread.isRunning()):
            self.start_duration_analysis()
    
    def clear_files(self):
        self.input_files = []
        self.update_files_list()
        self.warning_label.setVisible(False)
    
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
        
        # Enable/disable split button based on output directory
        has_valid_dir = bool(text.strip()) and os.path.isdir(text)
        self.process_btn.setEnabled(has_valid_dir)
    
    def select_output_dir(self):
        """Opens a file dialog to select output directory"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir_edit.setText(folder)
    
    def update_format_setting(self, format_text):
        """Update the output format setting"""
        self.settings['output_format'] = format_text
        
    def update_max_length_setting(self, value):
        """Update the max length setting"""
        self.settings['max_length'] = value
        
        # Trigger duration analysis if we have files
        if self.input_files and not (
            hasattr(self, 'duration_thread') and 
            self.duration_thread is not None and 
            self.duration_thread.isRunning()):
            self.start_duration_analysis()
    
    def show_advanced_settings(self):
        """Show the advanced settings dialog"""
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec():
            # Update settings from dialog
            self.settings = dialog.get_settings()
            
            # Update UI to reflect settings
            self.format_combo.setCurrentText(self.settings['output_format'])
            self.max_length_spin.setValue(self.settings['max_length'])
            
            # Update duration warning if we have files
            if self.input_files and not (
                hasattr(self, 'duration_thread') and 
                self.duration_thread is not None and 
                self.duration_thread.isRunning()):
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
        # Cancel any existing thread
        if hasattr(self, 'duration_thread') and self.duration_thread is not None and self.duration_thread.isRunning():
            try:
                self.duration_thread.quit()
                self.duration_thread.wait(100)
            except Exception as e:
                logger.warning(f"Error stopping duration thread: {str(e)}")
        
        # Start new thread
        self.duration_thread = AudioDurationThread(self.input_files)
        self.duration_thread.duration_calculated.connect(self.update_duration_warning)
        self.duration_thread.status_updated.connect(self.update_status)
        self.duration_thread.start()
    
    def update_duration_warning(self, total_duration):
        """Update warning based on calculated duration"""
        try:
            if total_duration <= 0:
                self.warning_label.setVisible(False)
                return
                
            # Convert max_length from minutes to milliseconds
            max_length_ms = self.settings['max_length'] * 60 * 1000
            
            # Estimate number of segments
            estimated_segments = math.ceil(total_duration / max_length_ms)
            
            # Format duration for display
            duration_str = self._format_time(total_duration)
            
            if estimated_segments > 25:
                self.warning_label.setText(
                    f"Warning: Total audio duration is {duration_str}. " 
                    f"This will create approximately {estimated_segments} files!"
                )
                self.warning_label.setVisible(True)
            else:
                self.warning_label.setText(f"Total audio duration: {duration_str}. Estimated segments: {estimated_segments}")
                self.warning_label.setVisible(True)
                self.warning_label.setStyleSheet("color: blue;")
        except Exception as e:
            logger.error(f"Error calculating file count: {str(e)}")
            self.warning_label.setVisible(False)
    
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
    
    def process_audio(self):
        if not self.input_files:
            QMessageBox.warning(self, "Error", "No input files selected.")
            return
            
        if not self.output_dir:
            QMessageBox.warning(self, "Error", "No output directory selected.")
            return
            
        if not os.path.isdir(self.output_dir):
            QMessageBox.warning(self, "Error", "Invalid output directory.")
            return
        
        # Calculate total duration first
        self.duration_thread = AudioDurationThread(self.input_files)
        self.duration_thread.duration_calculated.connect(self.update_duration_warning)
        self.duration_thread.status_updated.connect(self.update_status)
        self.duration_thread.finished.connect(self._start_processing)
        self.duration_thread.start()
        
        # Disable UI during processing
        self.set_ui_enabled(False)
    
    def _start_processing(self):
        # Update settings from UI
        self.settings['output_format'] = self.format_combo.currentText()
        self.settings['max_length'] = self.max_length_spin.value()
        
        # Create and start the processing thread
        self._start_processing()
        
    def _start_processing(self):
        """Start the actual audio processing after duration calculation"""
        # Create the processing thread
        self.splitter_thread = AudioSplitterThread(
            self.input_files,
            self.output_dir,
            self.settings
        )
        
        # Connect signals
        self.splitter_thread.progress_updated.connect(self.update_progress)
        self.splitter_thread.status_updated.connect(self.update_status)
        self.splitter_thread.completed.connect(self.processing_completed)
        self.splitter_thread.error_occurred.connect(self.processing_error)
        self.splitter_thread.merged_file_created.connect(self.merged_file_created)
        
        # Start processing
        self.splitter_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def processing_completed(self, output_files):
        self.set_ui_enabled(True)
        
        if not output_files:
            QMessageBox.warning(
                self,
                "Processing Result",
                "No output files were created."
            )
            return
        
        # If we have a merged file, include it in the message
        merged_msg = ""
        if hasattr(self.splitter_thread, 'merged_file') and self.splitter_thread.merged_file:
            merged_msg = f"\n\nMerged file saved to:\n{self.splitter_thread.merged_file}"
        
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Successfully created {len(output_files)} audio segments.{merged_msg}"
        )
        
        # Offer to open output directory
        reply = QMessageBox.question(
            self,
            "Open Output Directory",
            "Would you like to open the output directory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Open the directory in the system file explorer
                if sys.platform == 'win32':
                    os.startfile(self.output_dir)
                elif sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.call(['open', self.output_dir])
                else:  # Linux
                    import subprocess
                    subprocess.call(['xdg-open', self.output_dir])
            except Exception as e:
                logger.error(f"Error opening output directory: {str(e)}")
    
    def merged_file_created(self, file_path):
        """Handle notification that merged file was created"""
        self.update_status(f"Merged file created: {os.path.basename(file_path)}")
    
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
        self.advanced_settings_btn.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
        self.files_list.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled)


if __name__ == "__main__":
    # Create application
    app = QApplication(sys.argv)
    
    # Set application name and organization
    app.setApplicationName("ElevenLabs Sample Splitter")
    app.setOrganizationName("ElevenLabs Tools")
    
    # Create and show main window
    window = ElevenLabsSampleSplitter()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())