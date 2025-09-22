import sys
import numpy as np
import parselmouth
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, Qt
import pyqtgraph as pg
from scipy import signal
import colorsys

# --- スタイリング (変更なし) ---
STYLESHEET = """
    QWidget {
        background-color: #0a0a0a;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    #controls, #status, #debugInfo, #formantPlot {
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
        padding: 15px;
    }
    #formantPlot {
        border: 2px solid #00ff88;
    }
    QLabel {
        background-color: transparent;
    }
    h2 {
        color: #00ff88;
        font-size: 18px;
        font-weight: bold;
    }
    .formant-info {
        color: #88ff88;
        font-size: 12px;
    }
    QPushButton {
        background-color: #00ff88;
        color: black;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #00cc6a;
    }
    QPushButton.recording {
        background-color: #ff4444;
        color: white;
    }
    .vowel-btn {
        padding: 5px 8px;
        font-size: 11px;
        min-width: 40px;
    }
    .plot-title {
        font-size: 12px;
        font-weight: bold;
    }
"""

# --- 音声処理部分 (変更なし) ---
VOWELS = {
    'i': {'name': 'い', 'color': '#FF0000', 'f1': 324.0, 'f2': 2426.3},
    'e': {'name': 'え', 'color': '#00FF00', 'f1': 501.7, 'f2': 2064.6},
    'a': {'name': 'あ', 'color': '#0000FF', 'f1': 633.9, 'f2': 1087.5},
    'o': {'name': 'お', 'color': '#FFFF00', 'f1': 445.1, 'f2': 854.2},
    'u': {'name': 'う', 'color': '#FF00FF', 'f1': 343.8, 'f2': 1281.0}
}
SILENCE_THRESHOLD = 0.01

def extract_formants_praat(audio_data, sample_rate):
    if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
        return {'f1': 0, 'f2': 0, 'error': 'Silence detected'}
    try:
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(time_step=0.01, maximum_formant=5000)
        f1_values = [formant.get_value_at_time(1, t) for t in formant.ts() if not np.isnan(formant.get_value_at_time(1, t))]
        f2_values = [formant.get_value_at_time(2, t) for t in formant.ts() if not np.isnan(formant.get_value_at_time(2, t))]
        if not f1_values or not f2_values: return {'f1': 0, 'f2': 0, 'error': 'No formants found'}
        f1, f2 = np.median(f1_values), np.median(f2_values)
        return {'f1': f1, 'f2': f2}
    except Exception as e:
        return {'f1': 0, 'f2': 0, 'error': str(e)}

def classify_vowel(f1, f2):
    if f1 <= 0 or f2 <= 0: return None, 0
    best_match, min_dist = None, float('inf')
    for vowel, data in VOWELS.items():
        dist = np.sqrt((f1 - data['f1'])**2 + (f2 - data['f2'])**2)
        if dist < min_dist: min_dist, best_match = dist, vowel
    threshold = 350
    confidence = max(0, 100 * (1 - min_dist / (threshold * 2)))
    if min_dist > threshold: return best_match, round(confidence)
    return best_match, round(confidence)

class AudioWorker(QObject):
    data_updated = pyqtSignal(dict)
    
    def __init__(self, sample_rate=44100, chunk_size=2048, max_freq=4000):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_freq = max_freq
        self.is_running = False
        self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        audio_level = np.sqrt(np.mean(indata**2))
        win = signal.get_window('hann', self.chunk_size)
        fft_data = np.fft.rfft(indata[:, 0] * win)
        spectrum = np.abs(fft_data)
        freqs = np.fft.rfftfreq(self.chunk_size, 1/self.sample_rate)
        
        mask = freqs <= self.max_freq
        freqs = freqs[mask]
        spectrum = spectrum[mask]
        
        peak_freq = freqs[np.argmax(spectrum)] if len(freqs) > 0 else 0
        formant_data = extract_formants_praat(indata[:, 0], self.sample_rate)
        f1, f2 = formant_data.get('f1', 0), formant_data.get('f2', 0)
        
        result = {
            'spectrum': spectrum, 'audio_level': audio_level, 
            'peak_freq': peak_freq, 'f1': f1, 'f2': f2
        }
        self.data_updated.emit(result)

    @pyqtSlot()
    def start(self):
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, blocksize=self.chunk_size,
            channels=1, callback=self._audio_callback, dtype='float32'
        )
        self.stream.start()

    @pyqtSlot()
    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()
        self.is_running = False

import colorsys
import numpy as np
import pyqtgraph as pg

def create_chrome_music_lab_colormap():
    """色の差がはっきりわかる高コントラストなカラーマップ (黄緑→シアン→濃紫)"""
    positions = np.linspace(0.0, 1.0, 256)
    colors = []
    
    
    for pos in positions:
        # hue (色相): 弱い音(黄緑 0.3)から強い音(濃紫 0.7)へ大きく変化させます
        hue = 0.3 + 0.4 * pos
        
        # saturation (彩度): 弱い音は控えめに、強くなるほど鮮やかにします
        saturation = 0.4 + 0.6 * pos
        
        # value (明度): 弱い音は非常に明るく、強い音は暗くしてコントラストを強調します
        value = 1.0 - 0.6 * (pos ** 2)
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

        # RGB値を0-255の範囲に変換
        colors.append((int(r*255), int(g*255), int(b*255)))
    
    colors[0] = (0, 0, 0)
    return pg.ColorMap(positions, colors)

class SpectrogramApp(QMainWindow):
    def __init__(self, sample_rate=44100, chunk_size=2048):
        super().__init__()
        self.setWindowTitle("Spectrogram Visualizer")
        self.setGeometry(100, 100, 1400, 900)
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_freq = 4000
        
        self.target_vowel = 'i'
        self.is_recording = False
        self.current_f1, self.current_f2 = 0, 0
        
        freqs = np.fft.rfftfreq(self.chunk_size, 1/self.sample_rate)
        self.num_freq_bins = len(freqs[freqs <= self.max_freq])
        self.spectrogram_data = np.zeros((200, self.num_freq_bins)) 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self._create_main_layout()

        self.audio_thread = QThread()
        self.audio_worker = AudioWorker(sample_rate=self.sample_rate, chunk_size=self.chunk_size, max_freq=self.max_freq)
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_worker.data_updated.connect(self._update_ui)
        self.audio_thread.started.connect(self.audio_worker.start)

    def _create_plots(self):
        self.spectrogram_widget = pg.GraphicsLayoutWidget()
        p1 = self.spectrogram_widget.addPlot(title="Spectrogram")
        p1.setLabel('left', 'Frequency', units='Hz')
        
        p1.getAxis('bottom').hide()
        p1.getViewBox().setBackgroundColor('#0a0a0a')
        
        p1.invertY(False) 
        
        self.spec_image = pg.ImageItem()
        p1.addItem(self.spec_image)
        
        p1.setRange(yRange=(0, self.max_freq))
        
        self.spec_image.setImage(self.spectrogram_data)
        self.spec_image.setRect(0, 0, 1000, self.max_freq)
        
        self.spec_image.setColorMap(create_chrome_music_lab_colormap())
        
        self.target_f1_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#00ff88", width=2, style=Qt.PenStyle.DashLine))
        self.target_f2_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#00ff88', width=2, style=Qt.PenStyle.DashLine))
        p1.addItem(self.target_f1_line)
        p1.addItem(self.target_f2_line)

        self.target_f1_label = pg.TextItem(color='#00ff88', anchor=(0, 0))
        self.target_f2_label = pg.TextItem(color='#00ff88', anchor=(0, 0))
        p1.addItem(self.target_f1_label)
        p1.addItem(self.target_f2_label)
        
        self.measured_f1_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ff4444', width=3, style=Qt.PenStyle.DotLine))
        p1.addItem(self.measured_f1_line)
        self.measured_f1_label = pg.TextItem(color='#ff4444', anchor=(1, 0))
        p1.addItem(self.measured_f1_label)
        
        self.measured_f2_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ffff44', width=3, style=Qt.PenStyle.DotLine))
        p1.addItem(self.measured_f2_line)
        self.measured_f2_label = pg.TextItem(color='#ffff44', anchor=(1, 0))
        p1.addItem(self.measured_f2_label)

        self.formant_plot_widget = pg.GraphicsLayoutWidget()
        p2 = self.formant_plot_widget.addPlot(title="Formant Space")
        
        # --- 修正点 ---
        # 標準的な母音図の配置に合わせる
        # Y軸 (F1): 上が低周波 (狭母音)、下が高周波 (広母音) -> invertY(True)
        # X軸 (F2): 左が高周波 (前舌母音)、右が低周波 (後舌母音) -> invertX(True)
        p2.getViewBox().invertY(True)
        p2.getViewBox().invertX(True)
        
        # 背景色を設定
        p2.getViewBox().setBackgroundColor('#0a0a0a')
        
        # 軸ラベルを設定
        p2.setLabel('left', 'F1 (Hz)')
        p2.setLabel('bottom', 'F2 (Hz)')

        # 描画範囲を母音の分布に合わせて設定
        # xRange=(F2 max, F2 min), yRange=(F1 max, F1 min)
        # yRangeの(900, 200)は、下端が900Hz, 上端が200Hzを意味する
        p2.setRange(xRange=(2800, 600), yRange=(900, 200), padding=0.1)
        
        self.vowel_plots = {}
        for key, val in VOWELS.items():
            plot = pg.ScatterPlotItem(x=[val['f2']], y=[val['f1']], size=15, brush=pg.mkBrush(color=val['color']), name=val['name'])
            p2.addItem(plot)
            self.vowel_plots[key] = plot
            v_label = pg.TextItem(text=val['name'], color='white', anchor=(0.5, 1.5))
            v_label.setPos(val['f2'], val['f1'])
            p2.addItem(v_label)

        self.current_pos_plot = pg.ScatterPlotItem(size=20, brush=pg.mkBrush('r'), pen=pg.mkPen('w', width=2))
        p2.addItem(self.current_pos_plot)

    def _set_target_vowel(self, vowel):
        self.target_vowel = vowel
        self.target_vowel_text.setText(f"{vowel} ({VOWELS[vowel]['name']})")
        
        target_f1 = VOWELS[vowel]['f1']
        target_f2 = VOWELS[vowel]['f2']
        
        self.target_f1_line.setPos(target_f1)
        self.target_f2_line.setPos(target_f2)

        self.target_f1_label.setPos(10, target_f1 - 5)
        self.target_f1_label.setText(f"Target F1: {int(target_f1)}Hz")
        self.target_f2_label.setPos(10, target_f2 - 5)
        self.target_f2_label.setText(f"Target F2: {int(target_f2)}Hz")

        self._highlight_target_vowel()

    @pyqtSlot(dict)
    def _update_ui(self, data):
        spectrum = data['spectrum']
        min_log_val = np.log10(1e-12)
        max_log_val = np.log10(np.max(spectrum) + 1e-12) if np.max(spectrum) > 0 else min_log_val + 1
        
        if max_log_val - min_log_val == 0:
             normalized_spectrum = np.zeros_like(spectrum)
        else:
             normalized_spectrum = (np.log10(spectrum + 1e-12) - min_log_val) / (max_log_val - min_log_val)

        self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=0)
        self.spectrogram_data[-1, :] = normalized_spectrum
        
        self.spec_image.setImage(self.spectrogram_data, autoLevels=False, levels=(0,1))
        
        self.audio_level_text.setText(f"{data['audio_level']:.3f}")
        self.peak_freq_text.setText(f"{int(data['peak_freq'])} Hz")
        
        f1, f2 = data['f1'], data['f2']
        if f1 > 0 and f2 > 0 and f2 > f1:
            alpha = 0.6
            self.current_f1 = alpha * f1 + (1 - alpha) * self.current_f1
            self.current_f2 = alpha * f2 + (1 - alpha) * self.current_f2
            
            self.formant_text.setText(f"F1={int(self.current_f1)}Hz, F2={int(self.current_f2)}Hz")
            self.current_pos_plot.setData(x=[self.current_f2], y=[self.current_f1])
            
            self.measured_f1_line.setPos(self.current_f1)
            self.measured_f2_line.setPos(self.current_f2)

            self.measured_f1_label.setPos(self.spectrogram_widget.width() - 10, self.current_f1 - 5)
            self.measured_f1_label.setText(f"F1: {int(self.current_f1)}Hz")
            self.measured_f2_label.setPos(self.spectrogram_widget.width() - 10, self.current_f2 - 5)
            self.measured_f2_label.setText(f"F2: {int(self.current_f2)}Hz")

            vowel, conf = classify_vowel(self.current_f1, self.current_f2)
            if vowel:
                is_match = "✅" if vowel == self.target_vowel else ""
                self.detected_vowel_text.setText(f"{vowel} ({VOWELS[vowel]['name']}) {is_match} {conf}%")
            else:
                 self.detected_vowel_text.setText("-")
        else:
            self.formant_text.setText("F1=-Hz, F2=-Hz")
            self.current_pos_plot.setData([], [])
            self.measured_f1_line.setPos(-1)
            self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText("")
            self.measured_f2_label.setText("")

    def _create_main_layout(self):
        main_layout = QGridLayout(self.central_widget)
        self._create_plots()
        self._create_status_panel()
        self._create_controls()
        self._create_debug_panel()
        
        main_layout.addWidget(self.spectrogram_widget, 0, 0)
        main_layout.addWidget(self.formant_plot_widget, 1, 0)
        
        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(self.controls_widget)
        right_panel_layout.addWidget(self.status_widget)
        right_panel_layout.addWidget(self.debug_widget)
        right_panel_layout.addStretch()
        main_layout.addLayout(right_panel_layout, 0, 1, 2, 1)

        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 1)

    def _create_controls(self):
        self.controls_widget = QWidget(); self.controls_widget.setObjectName("controls")
        layout = QVBoxLayout(self.controls_widget)
        title = QLabel("<h2>🎯 Audio Spectrogram</h2>"); layout.addWidget(title)
        self.record_btn = QPushButton("🎙️ Start Recording"); self.record_btn.clicked.connect(self._toggle_recording)
        reset_btn = QPushButton("Reset"); reset_btn.clicked.connect(self._reset_view)
        btn_layout = QHBoxLayout(); btn_layout.addWidget(self.record_btn); btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Target Vowel:"))
        vowel_layout = QGridLayout(); buttons = {}
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        for (key, val), pos in zip(VOWELS.items(), positions):
            btn = QPushButton(f"{val['name']} ({key})"); btn.setObjectName("vowel-btn"); btn.setProperty("class", "vowel-btn")
            btn.clicked.connect(lambda _, v=key: self._set_target_vowel(v))
            buttons[key] = btn; vowel_layout.addWidget(btn, pos[0], pos[1])
        layout.addLayout(vowel_layout); self.vowel_buttons = buttons
        self._set_target_vowel(self.target_vowel)

    def _create_status_panel(self):
        self.status_widget = QWidget(); self.status_widget.setObjectName("status")
        layout = QGridLayout(self.status_widget)
        self.status_text = self._add_status_row(layout, 0, "Status:", "Click Start Recording")
        self.target_vowel_text = self._add_status_row(layout, 1, "Target:", "i (い)")
        self.audio_level_text = self._add_status_row(layout, 2, "Audio Level:", "0", is_formant=True)
        self.formant_text = self._add_status_row(layout, 3, "Formants:", "F1=-Hz, F2=-Hz", is_formant=True)
        self.peak_freq_text = self._add_status_row(layout, 4, "Frequency Peak:", "-Hz", is_formant=True)
        self.detected_vowel_text = self._add_status_row(layout, 5, "Detected:", "-", is_formant=True)

    def _add_status_row(self, layout, row, label_text, value_text, is_formant=False):
        label = QLabel(f"<strong>{label_text}</strong>"); value = QLabel(value_text)
        if is_formant: value.setProperty("class", "formant-info")
        layout.addWidget(label, row, 0); layout.addWidget(value, row, 1)
        return value

    def _create_debug_panel(self):
        self.debug_widget = QWidget(); self.debug_widget.setObjectName("debugInfo")
        layout = QVBoxLayout(self.debug_widget); self.debug_text = QLabel("Debug: Ready"); layout.addWidget(self.debug_text)

    def _toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.setText("⏹️ Stop Recording"); self.record_btn.setProperty("class", "recording")
            self.status_text.setText("Recording..."); self.audio_thread.start()
            self._set_target_vowel(self.target_vowel)
            self.measured_f1_line.setPos(-1); self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText(""); self.measured_f2_label.setText("")
        else:
            self.audio_worker.stop(); self.audio_thread.quit(); self.audio_thread.wait()
            self.record_btn.setText("🎙️ Start Recording"); self.record_btn.setProperty("class", "")
            self.status_text.setText("Stopped")
            self.measured_f1_line.setPos(-1); self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText(""); self.measured_f2_label.setText("")
        self.record_btn.style().unpolish(self.record_btn); self.record_btn.style().polish(self.record_btn)

    def _reset_view(self):
        self.spectrogram_data.fill(0)
        self.spec_image.setImage(self.spectrogram_data, autoLevels=False, levels=(0,1))
        self.current_f1, self.current_f2 = 0, 0
        self.current_pos_plot.setData([], [])
        self.measured_f1_line.setPos(-1); self.measured_f2_line.setPos(-1)
        self.measured_f1_label.setText(""); self.measured_f2_label.setText("")
        
    def _highlight_target_vowel(self):
        for key, btn in self.vowel_buttons.items():
            is_target = (key == self.target_vowel)
            style = f"background-color: {'#00ff88' if is_target else '#555'}; color: {'black' if is_target else 'white'};"
            btn.setStyleSheet(style)
        for key, plot in self.vowel_plots.items():
            is_target = (key == self.target_vowel)
            plot.setSize(25 if is_target else 15); plot.setPen('w' if is_target else None, width=2)

    def closeEvent(self, event):
        if self.is_recording: self.audio_worker.stop(); self.audio_thread.quit(); self.audio_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = SpectrogramApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()