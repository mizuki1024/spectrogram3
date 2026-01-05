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

import asyncio
import threading
import websockets
import json

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

# --- 音声処理部分 ---
# 'name'キーは内部的には残しますが、UI表示には使用しません
VOWELS = {
    'a': {'name': 'ah',   'color': '#FF66CC', 'f1': 634, 'f2': 1088},
    'i': {'name': 'heed', 'color': '#FF5733', 'f1': 324, 'f2': 2426},
    'u': {'name': 'who\'d','color': '#5833FF', 'f1': 344, 'f2': 1281},
    'e': {'name': 'eh',   'color': '#66FF33', 'f1': 502, 'f2': 2065}, 
    'o': {'name': 'oh',   'color': '#33CCFF', 'f1': 445, 'f2': 854},
    'ɪ': {'name': 'hid',  'color': '#FF8D33', 'f1': 390, 'f2': 1990},
    'ɛ': {'name': 'head', 'color': '#FFC300', 'f1': 530, 'f2': 1840},
    'æ': {'name': 'had',  'color': '#DAF7A6', 'f1': 660, 'f2': 1720},
    'ɑ': {'name': 'hod',  'color': '#33FF57', 'f1': 730, 'f2': 1090},
    'ʊ': {'name': 'hood', 'color': '#33A5FF', 'f1': 440, 'f2': 1020},
    'ʌ': {'name': 'bud',  'color': '#C70039', 'f1': 640, 'f2': 1190},
    'ə': {'name': 'sofa', 'color': '#900C3F', 'f1': 500, 'f2': 1500}, 
    'iː': {'name': 'ee',    'color': '#FF33AA', 'f1': 270, 'f2': 2290},
    'ɑː': {'name': 'ahh',   'color': '#33FF99', 'f1': 750, 'f2': 1200}, 
    'ɜː': {'name': 'er',    'color': '#FF9933', 'f1': 550, 'f2': 1600}, 
    'ɔː': {'name': 'hawed','color': '#33FFCE', 'f1': 570, 'f2': 840},
    'uː': {'name': 'oo',    'color': '#3366FF', 'f1': 300, 'f2': 870},
}
SILENCE_THRESHOLD = 0.01
TIME_STEP = 0.03
MAX_FORMANT = 5000

def extract_formants_praat(audio_data, sample_rate):
    if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
        return {'f1': 0, 'f2': 0, 'error': 'Silence detected'}
    try:
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(time_step=TIME_STEP, maximum_formant=MAX_FORMANT)
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
        dist = np.sqrt((np.log(f1) - np.log(data['f1']))**2 + (np.log(f2) - np.log(data['f2']))**2)
        if dist < min_dist: min_dist, best_match = dist, vowel

    threshold = 0.4
    confidence = max(0, 100 * (1 - min_dist / (threshold * 2)))
    if min_dist > threshold: return best_match, round(confidence)
    return best_match, round(confidence)

def get_pronunciation_advice(current_f1, current_f2, target_vowel):
    """F1とF2の位置に基づいて発音アドバイスを生成"""
    if current_f1 <= 0 or current_f2 <= 0:
        return "マイクに向かって話してください"

    target_data = VOWELS[target_vowel]
    target_f1 = target_data['f1']
    target_f2 = target_data['f2']

    # F1とF2の差を計算（許容範囲: 10%）
    f1_diff_percent = ((current_f1 - target_f1) / target_f1) * 100
    f2_diff_percent = ((current_f2 - target_f2) / target_f2) * 100

    advice = []

    # F1のアドバイス（口の開き具合）
    if abs(f1_diff_percent) > 10:
        if f1_diff_percent > 0:
            advice.append("🔽 口を少し閉じてください (F1が高すぎます)")
        else:
            advice.append("🔼 口をもっと開けてください (F1が低すぎます)")

    # F2のアドバイス（舌の前後位置）
    if abs(f2_diff_percent) > 10:
        if f2_diff_percent > 0:
            advice.append("👈 舌を後ろに引いてください (F2が高すぎます)")
        else:
            advice.append("👉 舌を前に出してください (F2が低すぎます)")

    if not advice:
        return "✅ 完璧です！この発音を維持してください！"

    return " | ".join(advice)

class AudioWorker(QObject):
    data_updated = pyqtSignal(dict)
    
    def __init__(self, sample_rate=44100, chunk_size=2048, max_freq=5500):
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

def create_chrome_music_lab_colormap():
    """色の差がはっきりわかる高コントラストなカラーマップ (黄緑→シアン→濃紫)"""
    positions = np.linspace(0.0, 1.0, 256)
    colors = []
    
    for pos in positions:
        hue = 0.3 + 0.4 * pos
        saturation = 0.4 + 0.6 * pos
        value = 1.0 - 0.6 * (pos ** 2)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r*255), int(g*255), int(b*255)))
        
    # 最も値が低い色（＝無音）を黒に設定する
    colors[0] = (0, 0, 0)
        
    return pg.ColorMap(positions, colors)

class SpectrogramApp(QMainWindow):
    def __init__(self, sample_rate=44100, chunk_size=2048):
        super().__init__()
        self.setWindowTitle("English Vowel Spectrogram Visualizer")
        self.setGeometry(100, 100, 1400, 900)
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_freq = 5500
        
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

        # WebSocket

        self.ws_server = F1WebSocketServer() 
        self.ws_server.start()

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
        p1.addItem(self.target_f1_line); p1.addItem(self.target_f2_line)
        self.target_f1_label = pg.TextItem(color='#00ff88', anchor=(0, 0))
        self.target_f2_label = pg.TextItem(color='#00ff88', anchor=(0, 0))
        p1.addItem(self.target_f1_label); p1.addItem(self.target_f2_label)
        
        self.measured_f1_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ff4444', width=3, style=Qt.PenStyle.DotLine))
        self.measured_f2_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ffff44', width=3, style=Qt.PenStyle.DotLine))
        p1.addItem(self.measured_f1_line); p1.addItem(self.measured_f2_line)
        self.measured_f1_label = pg.TextItem(color='#ff4444', anchor=(1, 0))
        self.measured_f2_label = pg.TextItem(color='#ffff44', anchor=(1, 0))
        p1.addItem(self.measured_f1_label); p1.addItem(self.measured_f2_label)

        self.formant_plot_widget = pg.GraphicsLayoutWidget()
        p2 = self.formant_plot_widget.addPlot(title="Formant Space (Vowel Chart)")
        p2.setLabel('bottom', 'F2 (Hz)')
        p2.setLabel('left', 'F1 (Hz)')
        p2.getViewBox().invertX(True) 
        p2.getViewBox().invertY(True)
        p2.getViewBox().setBackgroundColor('#0a0a0a')
        
        p2.setRange(xRange=(2500, 700), yRange=(900, 200), padding=0.1)
        
        self.vowel_plots = {}
        for key, val in VOWELS.items():
            plot = pg.ScatterPlotItem(x=[val['f2']], y=[val['f1']], size=15, brush=pg.mkBrush(color=val['color']), name=val['name'])
            p2.addItem(plot)
            self.vowel_plots[key] = plot
            v_label = pg.TextItem(text=f"{key}", color='white', anchor=(0.5, 1.5))
            v_label.setPos(val['f2'], val['f1'])
            p2.addItem(v_label)

        self.current_pos_plot = pg.ScatterPlotItem(size=20, brush=pg.mkBrush('r'), pen=pg.mkPen('w', width=2))
        p2.addItem(self.current_pos_plot)
        
    def _set_target_vowel(self, vowel):
        self.target_vowel = vowel
        self.target_vowel_text.setText(f"{vowel}")

        target_f1 = VOWELS[vowel]['f1']
        target_f2 = VOWELS[vowel]['f2']

        self.target_f1_line.setPos(target_f1)
        self.target_f2_line.setPos(target_f2)

        self.target_f1_label.setText(f"Target F1: {int(target_f1)}Hz")
        self.target_f2_label.setText(f"Target F2: {int(target_f2)}Hz")
        self.target_f1_label.setPos(10, target_f1 + 5)
        self.target_f2_label.setPos(10, target_f2 + 5)

        self._highlight_target_vowel()

        # 🔽 追加：録音中でなくても target を送る
        if hasattr(self, "ws_server") and self.ws_server:
            self.ws_server.send_formant(
                self.current_f1 if self.is_recording else 100,
                self.target_vowel
        )


    @pyqtSlot(dict)
    def _update_ui(self, data):
        spectrum = data['spectrum']
        min_log_val = np.log10(1e-12)
        max_log_val = np.log10(np.max(spectrum) + 1e-12) if np.max(spectrum) > 0 else min_log_val + 1
        
        normalized_spectrum = np.zeros_like(spectrum)
        if max_log_val > min_log_val:
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
            
            # if self.is_recording:
            #     self.ws_server.send_f1(self.current_f1)

            if self.is_recording:
                vowel, conf = classify_vowel(self.current_f1, self.current_f2)
                self.ws_server.send_formant(
                    self.current_f1,
                    # vowel,                 # 現在の母音
                    self.target_vowel      # 目標の母音
                )
                # print(f"Sending formant: {self.current_f1},{self.target_vowel}")

            self.formant_text.setText(f"F1={int(self.current_f1)}Hz, F2={int(self.current_f2)}Hz")
            self.current_pos_plot.setData(x=[self.current_f2], y=[self.current_f1])
            
            self.measured_f1_line.setPos(self.current_f1)
            self.measured_f2_line.setPos(self.current_f2)
            self.measured_f1_label.setText(f"F1: {int(self.current_f1)}Hz")
            self.measured_f2_label.setText(f"F2: {int(self.current_f2)}Hz")
            self.measured_f1_label.setPos(self.spectrogram_widget.width() - 10, self.current_f1 + 5)
            self.measured_f2_label.setPos(self.spectrogram_widget.width() - 10, self.current_f2 + 5)

            vowel, conf = classify_vowel(self.current_f1, self.current_f2)
            if vowel:
                is_match = "✅ Match!" if vowel == self.target_vowel else ""
                self.detected_vowel_text.setText(f"{vowel} {is_match} ({conf}%)")
            else:
                 self.detected_vowel_text.setText(" - ")

            # 発音アドバイスを更新
            advice = get_pronunciation_advice(self.current_f1, self.current_f2, self.target_vowel)
            self.advice_text.setText(advice)
        else:
            self.formant_text.setText("F1= - Hz, F2= - Hz")
            self.current_pos_plot.setData([], [])
            self.measured_f1_line.setPos(-1)
            self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText("")
            self.measured_f2_label.setText("")
            self.advice_text.setText("マイクに向かって話してください")

    def _create_main_layout(self):
        main_layout = QGridLayout(self.central_widget)
        self._create_plots()
        self._create_status_panel()
        self._create_controls()
        
        main_layout.addWidget(self.spectrogram_widget, 0, 0)
        main_layout.addWidget(self.formant_plot_widget, 1, 0)
        
        right_panel_layout = QVBoxLayout()
        right_panel_layout.addWidget(self.controls_widget)
        right_panel_layout.addWidget(self.status_widget)
        right_panel_layout.addStretch()
        main_layout.addLayout(right_panel_layout, 0, 1, 2, 1)

        main_layout.setColumnStretch(0, 3); main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(0, 1); main_layout.setRowStretch(1, 1)

    def _create_controls(self):
        self.controls_widget = QWidget(); self.controls_widget.setObjectName("controls")
        layout = QVBoxLayout(self.controls_widget)
        title = QLabel("<h2>🎯 Vowel Formant Visualizer</h2>"); layout.addWidget(title)
        self.record_btn = QPushButton("🎙️ Start Recording"); self.record_btn.clicked.connect(self._toggle_recording)
        reset_btn = QPushButton("🔄 Reset View"); reset_btn.clicked.connect(self._reset_view)
        btn_layout = QHBoxLayout(); btn_layout.addWidget(self.record_btn); btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("<strong>Target Vowel:</strong>"))
        
        vowel_layout = QGridLayout(); buttons = {}
        vowel_items = list(VOWELS.items())
        positions = [(i, j) for i in range(4) for j in range(5)]
        
        for i, (key, val) in enumerate(vowel_items):
            pos = positions[i]
            btn = QPushButton(f"{key}"); 
            btn.setObjectName("vowel-btn"); btn.setProperty("class", "vowel-btn")
            btn.clicked.connect(lambda _, v=key: self._set_target_vowel(v))
            buttons[key] = btn; vowel_layout.addWidget(btn, pos[0], pos[1])
            
        layout.addLayout(vowel_layout); self.vowel_buttons = buttons
        self._set_target_vowel(self.target_vowel)

    def _create_status_panel(self):
        self.status_widget = QWidget(); self.status_widget.setObjectName("status")
        layout = QGridLayout(self.status_widget)
        self.status_text = self._add_status_row(layout, 0, "Status:", "Click 'Start Recording'")
        self.target_vowel_text = self._add_status_row(layout, 1, "Target:", "i")
        self.audio_level_text = self._add_status_row(layout, 2, "Audio Level:", "0.000", is_formant=True)
        self.peak_freq_text = self._add_status_row(layout, 3, "Peak Frequency:", "- Hz", is_formant=True)
        self.formant_text = self._add_status_row(layout, 4, "Measured Formants:", "F1=-, F2=-", is_formant=True)
        self.detected_vowel_text = self._add_status_row(layout, 5, "Detected Vowel:", "-", is_formant=True)

        # アドバイスセクションを追加
        advice_label = QLabel("<h2>💡 発音アドバイス</h2>")
        layout.addWidget(advice_label, 6, 0, 1, 2)
        self.advice_text = QLabel("マイクに向かって話してください")
        self.advice_text.setWordWrap(True)
        self.advice_text.setStyleSheet("color: #ffff44; font-size: 13px; font-weight: bold; padding: 10px; background-color: rgba(50, 50, 50, 0.5); border-radius: 5px;")
        layout.addWidget(self.advice_text, 7, 0, 1, 2)

    def _add_status_row(self, layout, row, label_text, value_text, is_formant=False):
        label = QLabel(f"<strong>{label_text}</strong>"); value = QLabel(value_text)
        if is_formant: value.setProperty("class", "formant-info")
        layout.addWidget(label, row, 0); layout.addWidget(value, row, 1)
        return value

    def _toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.setText("⏹️ Stop Recording"); self.record_btn.setProperty("class", "recording")
            self.status_text.setText("🔴 Recording..."); self.audio_thread.start()
            self._set_target_vowel(self.target_vowel)
            self.measured_f1_line.setPos(-1); self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText(""); self.measured_f2_label.setText("")
        else:
            self.audio_worker.stop(); self.audio_thread.quit(); self.audio_thread.wait()
            self.record_btn.setText("🎙️ Start Recording"); self.record_btn.setProperty("class", "")
            self.status_text.setText("Stopped.")
            self.measured_f1_line.setPos(-1); self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText(""); self.measured_f2_label.setText("")

            # self.ws_server.send_f1(100)

            self.ws_server.send_formant(
                100,
                # None,
                self.target_vowel
            )

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
            style = f"background-color: {'#00ff88' if is_target else '#444'}; color: {'black' if is_target else 'white'};"
            btn.setStyleSheet(style)
        for key, plot in self.vowel_plots.items():
            is_target = (key == self.target_vowel)
            plot.setSize(25 if is_target else 15); plot.setPen('w' if is_target else None, width=2)

    def closeEvent(self, event):
        if self.is_recording: self.audio_worker.stop(); self.audio_thread.quit(); self.audio_thread.wait()
        event.accept()

class F1WebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()

        self.loop = None
        self.server = None
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            print("WebSocket server already running")
            return

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start_server())
        self.loop.run_forever()

    async def _start_server(self):
        self.server = await websockets.serve(
            self._handler, self.host, self.port
        )
        print(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    def send_formant(self, f1, target_vowel):
        if not self.loop or not self.loop.is_running():
            return
        if not self.clients:
            return

        data = json.dumps({
            "f1": float(f1),
            "target_vowel": target_vowel
        })

        asyncio.run_coroutine_threadsafe(
            self._broadcast(data), self.loop
        )



    async def _broadcast(self, message):
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )

    def stop(self):
        pass


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = SpectrogramApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
