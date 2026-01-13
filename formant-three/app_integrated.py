import sys
import numpy as np
import parselmouth
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGridLayout, QTabWidget, QLineEdit, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, Qt, QTimer
import pyqtgraph as pg
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import colorsys

import asyncio
import threading
import websockets
import json
import os
import time

import librosa
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- スタイリング ---
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
    QTabWidget::pane {
        border: 1px solid #00ff88;
        background-color: #0a0a0a;
    }
    QTabBar::tab {
        background-color: #1a1a1a;
        color: white;
        padding: 10px 20px;
        border: 1px solid #00ff88;
        border-bottom: none;
    }
    QTabBar::tab:selected {
        background-color: #00ff88;
        color: black;
    }
"""

# --- 音声処理共通関数 ---
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
    return best_match, round(confidence)

def get_pronunciation_advice(current_f1, current_f2, target_vowel):
    if current_f1 <= 0 or current_f2 <= 0: return "マイクに向かって話してください"
    target_data = VOWELS[target_vowel]
    f1_diff = ((current_f1 - target_data['f1']) / target_data['f1']) * 100
    f2_diff = ((current_f2 - target_data['f2']) / target_data['f2']) * 100
    advice = []
    if abs(f1_diff) > 10: advice.append("🔽 口を少し閉じて" if f1_diff > 0 else "🔼 口をもっと開けて")
    if abs(f2_diff) > 10: advice.append("👈 舌を後ろに" if f2_diff > 0 else "👉 舌を前に")
    return " | ".join(advice) if advice else "✅ 完璧です！"

def generate_reference_audio(word):
    if not word: return None
    ref_path = "reference.wav"
    os.system(f"say -v Samantha -o {ref_path} --data-format=LEF32@16000 {word}")
    return ref_path
def extract_stress_contour(audio_path, sr=16000):
    try:
        # --- Praat (Parselmouth) を使用 ---
        
        # 1. 音声を読み込む
        sound = parselmouth.Sound(audio_path)
        
        # 2. ピッチを抽出 (Praatの "To Pitch" 機能)
        # time_step: 計測間隔(秒), pitch_floor: 最低音(Hz), pitch_ceiling: 最高音(Hz)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        
        # 3. 数値配列を取り出す
        # Praatは音程がない部分(無声音)を 0 として返します
        pitch_values = pitch.selected_array['frequency']
        
        # 時間軸を取り出す
        times = pitch.xs()

        # 4. 0.0〜1.0 に正規化
        # (0Hzの部分は除外して最大値を計算しないと、変な比率になるため注意)
        max_pitch = np.max(pitch_values)
        if max_pitch > 0:
            stress_curve = pitch_values / max_pitch
        else:
            stress_curve = pitch_values # 全部0の場合

        # 5. 滑らかにする (ガウシアンフィルタ)
        stress_curve_smooth = gaussian_filter1d(stress_curve, sigma=2.0)

        return times, stress_curve_smooth

    except Exception as e:
        print(f"Error: {e}")
        return None, None
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
        spectrum = np.abs(np.fft.rfft(indata[:, 0] * win))
        freqs = np.fft.rfftfreq(self.chunk_size, 1/self.sample_rate)
        mask = freqs <= self.max_freq
        peak_freq = freqs[mask][np.argmax(spectrum[mask])] if len(freqs[mask]) > 0 else 0
        formant = extract_formants_praat(indata[:, 0], self.sample_rate)
        self.data_updated.emit({'spectrum': spectrum[mask], 'audio_level': audio_level, 'peak_freq': peak_freq, **formant})
    @pyqtSlot()
    def start(self):
        self.is_running = True
        self.stream = sd.InputStream(samplerate=self.sample_rate, blocksize=self.chunk_size, channels=1, callback=self._audio_callback, dtype='float32')
        self.stream.start()
    @pyqtSlot()
    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()
        self.is_running = False

def create_chrome_music_lab_colormap():
    positions = np.linspace(0.0, 1.0, 256)
    colors = []
    for pos in positions:
        r, g, b = colorsys.hsv_to_rgb(0.3 + 0.4 * pos, 0.4 + 0.6 * pos, 1.0 - 0.6 * (pos ** 2))
        colors.append((int(r*255), int(g*255), int(b*255)))
    colors[0] = (0, 0, 0)
    return pg.ColorMap(positions, colors)

class F1WebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host, self.port, self.clients = host, port, set()
        self.loop, self.thread = None, None
        self.server_started = threading.Event()
        self.server = None

    def start(self):
        if self.thread and self.thread.is_alive():
            print("⚠️ WebSocketサーバーは既に起動しています")
            return
        print(f"🔄 WebSocketサーバーを起動中... (ポート: {self.port})")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        # サーバーが起動するまで最大5秒待つ
        if self.server_started.wait(timeout=5):
            print(f"✅ WebSocketサーバーが正常に起動しました: ws://{self.host}:{self.port}")
        else:
            print(f"❌ WebSocketサーバーの起動がタイムアウトしました")

    def _run(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def start_server():
                self.server = await websockets.serve(self._handler, self.host, self.port)
                print(f"🔌 WebSocketサーバーがポート {self.port} でリッスンを開始しました")
                self.server_started.set()

            self.loop.run_until_complete(start_server())
            self.loop.run_forever()
        except Exception as e:
            print(f"❌ WebSocketサーバーの起動に失敗しました: {e}")
            import traceback
            traceback.print_exc()
            self.server_started.set()  # エラーでもブロック解除

    async def _handler(self, websocket):
        client_addr = websocket.remote_address
        print(f"🔗 クライアントが接続しました: {client_addr}")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                pass  # クライアントからのメッセージを待つ
        except Exception as e:
            print(f"⚠️ WebSocket接続エラー: {e}")
        finally:
            self.clients.discard(websocket)
            print(f"🔌 クライアントが切断しました: {client_addr}")

    def send_formant(self, f1, f2, target_vowel):
        if self.loop and self.loop.is_running() and self.clients:
            data = json.dumps({"f1": float(f1),  "f2": float(f2), "target_vowel": target_vowel})
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)

    async def _broadcast(self, message):
        if self.clients:
            await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.patch.set_facecolor('#0a0a0a')
        self.axes.set_facecolor('#0a0a0a')
        self.axes.tick_params(colors='white')
        for spine in self.axes.spines.values(): spine.set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        super(MplCanvas, self).__init__(fig)

class IntegratedApp(QMainWindow):
    def __init__(self, sample_rate=44100, chunk_size=2048):
        super().__init__()
        self.setWindowTitle("English Pronunciation Coach - Integrated")
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

        print("=" * 60)
        print("🚀 統合アプリケーションを初期化中...")
        print("=" * 60)
        self.ws_server = F1WebSocketServer()
        self.ws_server.start()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tab1 = self._create_formant_tab()
        self.tabs.addTab(self.tab1, "母音分析")
        self.tab2 = self._create_stress_tab()
        self.tabs.addTab(self.tab2, "強弱分析")
        self.audio_thread = QThread()
        self.audio_worker = AudioWorker(sample_rate=self.sample_rate, chunk_size=self.chunk_size, max_freq=self.max_freq)
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_worker.data_updated.connect(self._update_ui)
        self.audio_thread.started.connect(self.audio_worker.start)

        # ★追加: ドラッグ操作用の変数
        self.dragging_line = None # 'start' or 'end' or None
        self.trim_start_val = 0.0
        self.trim_end_val = 0.0

    # --- Tab 1 (そのままでOK) ---
    def _create_formant_tab(self):
        widget = QWidget()
        main_layout = QGridLayout(widget)
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
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 1)
        return widget

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
        self.measured_f2_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ffff44', width=3, style=Qt.PenStyle.DotLine))
        p1.addItem(self.measured_f1_line)
        p1.addItem(self.measured_f2_line)
        self.measured_f1_label = pg.TextItem(color='#ff4444', anchor=(1, 0))
        self.measured_f2_label = pg.TextItem(color='#ffff44', anchor=(1, 0))
        p1.addItem(self.measured_f1_label)
        p1.addItem(self.measured_f2_label)
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

    def _create_controls(self):
        self.controls_widget = QWidget()
        self.controls_widget.setObjectName("controls")
        layout = QVBoxLayout(self.controls_widget)
        title = QLabel("<h2>🎯 Vowel Formant Visualizer</h2>")
        layout.addWidget(title)
        self.record_btn = QPushButton("🎙️ Start Recording")
        self.record_btn.clicked.connect(self._toggle_recording)
        reset_btn = QPushButton("🔄 Reset View")
        reset_btn.clicked.connect(self._reset_view)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.record_btn)
        btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("<strong>Target Vowel:</strong>"))
        vowel_layout = QGridLayout()
        buttons = {}
        vowel_items = list(VOWELS.items())
        positions = [(i, j) for i in range(4) for j in range(5)]
        for i, (key, val) in enumerate(vowel_items):
            pos = positions[i]
            btn = QPushButton(f"{key}")
            btn.setObjectName("vowel-btn")
            btn.setProperty("class", "vowel-btn")
            btn.clicked.connect(lambda _, v=key: self._set_target_vowel(v))
            buttons[key] = btn
            vowel_layout.addWidget(btn, pos[0], pos[1])
        layout.addLayout(vowel_layout)
        self.vowel_buttons = buttons
        self._set_target_vowel(self.target_vowel)

    def _create_status_panel(self):
        self.status_widget = QWidget()
        self.status_widget.setObjectName("status")
        layout = QGridLayout(self.status_widget)
        self.status_text = self._add_status_row(layout, 0, "Status:", "Click 'Start Recording'")
        self.target_vowel_text = self._add_status_row(layout, 1, "Target:", "i")
        self.audio_level_text = self._add_status_row(layout, 2, "Audio Level:", "0.000", is_formant=True)
        self.peak_freq_text = self._add_status_row(layout, 3, "Peak Frequency:", "- Hz", is_formant=True)
        self.formant_text = self._add_status_row(layout, 4, "Measured Formants:", "F1=-, F2=-", is_formant=True)
        self.detected_vowel_text = self._add_status_row(layout, 5, "Detected Vowel:", "-", is_formant=True)
        advice_label = QLabel("<h2>💡 発音アドバイス</h2>")
        layout.addWidget(advice_label, 6, 0, 1, 2)
        self.advice_text = QLabel("マイクに向かって話してください")
        self.advice_text.setWordWrap(True)
        self.advice_text.setStyleSheet("color: #ffff44; font-size: 13px; font-weight: bold; padding: 10px; background-color: rgba(50, 50, 50, 0.5); border-radius: 5px;")
        layout.addWidget(self.advice_text, 7, 0, 1, 2)

    def _add_status_row(self, layout, row, label_text, value_text, is_formant=False):
        label = QLabel(f"<strong>{label_text}</strong>")
        value = QLabel(value_text)
        if is_formant: value.setProperty("class", "formant-info")
        layout.addWidget(label, row, 0)
        layout.addWidget(value, row, 1)
        return value

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
        # if hasattr(self, "ws_server") and self.ws_server:
        #     self.ws_server.send_formant(self.current_f1 if self.is_recording else 100, self.target_vowel)
        # 🔽 追加：録音中でなくても target を送る
        if hasattr(self, "ws_server") and self.ws_server:
            self.ws_server.send_formant(
                self.current_f1 if self.is_recording else 100,
                self.current_f2 if self.is_recording else 2000,
                self.target_vowel
        )
    @pyqtSlot(dict)
    def _update_ui(self, data):
        spectrum = data['spectrum']
        min_log = np.log10(1e-12)
        max_log = np.log10(np.max(spectrum) + 1e-12) if np.max(spectrum) > 0 else min_log + 1
        normalized_spectrum = np.zeros_like(spectrum)
        if max_log > min_log: normalized_spectrum = (np.log10(spectrum + 1e-12) - min_log) / (max_log - min_log)
        self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=0)
        self.spectrogram_data[-1, :] = normalized_spectrum
        self.spec_image.setImage(self.spectrogram_data, autoLevels=False, levels=(0,1))
        self.audio_level_text.setText(f"{data['audio_level']:.3f}")
        self.peak_freq_text.setText(f"{int(data['peak_freq'])} Hz")
        f1, f2 = data['f1'], data['f2']
        if f1 > 0 and f2 > 0 and f2 > f1:
            self.current_f1 = 0.6 * f1 + 0.4 * self.current_f1
            self.current_f2 = 0.6 * f2 + 0.4 * self.current_f2
            if self.is_recording:
                vowel, conf = classify_vowel(self.current_f1, self.current_f2)
                self.ws_server.send_formant(self.current_f1, self.current_f2, self.target_vowel)
            self.formant_text.setText(f"F1={int(self.current_f1)}Hz, F2={int(self.current_f2)}Hz")
            self.current_pos_plot.setData(x=[self.current_f2], y=[self.current_f1])
            self.measured_f1_line.setPos(self.current_f1)
            self.measured_f2_line.setPos(self.current_f2)
            self.measured_f1_label.setText(f"F1: {int(self.current_f1)}Hz")
            self.measured_f2_label.setText(f"F2: {int(self.current_f2)}Hz")
            self.measured_f1_label.setPos(self.spectrogram_widget.width() - 10, self.current_f1 + 5)
            self.measured_f2_label.setPos(self.spectrogram_widget.width() - 10, self.current_f2 + 5)
            vowel, conf = classify_vowel(self.current_f1, self.current_f2)
            self.detected_vowel_text.setText(f"{vowel} {'✅ Match!' if vowel == self.target_vowel else ''} ({conf}%)" if vowel else " - ")
            self.advice_text.setText(get_pronunciation_advice(self.current_f1, self.current_f2, self.target_vowel))
        else:
            self.formant_text.setText("F1= - Hz, F2= - Hz")
            self.current_pos_plot.setData([], [])
            self.measured_f1_line.setPos(-1)
            self.measured_f2_line.setPos(-1)
            self.detected_vowel_text.setText(" - ")
            self.advice_text.setText("マイクに向かって話してください")

    def _toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.setText("⏹️ Stop Recording")
            self.record_btn.setProperty("class", "recording")
            self.status_text.setText("🔴 Recording...")
            self.audio_thread.start()
            self._set_target_vowel(self.target_vowel)
        else:
            self.audio_worker.stop()
            self.audio_thread.quit()
            self.audio_thread.wait()
            self.record_btn.setText("🎙️ Start Recording")
            self.record_btn.setProperty("class", "")
            self.status_text.setText("Stopped.")
            self.ws_server.send_formant(100, 2000, self.target_vowel)
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)

    def _reset_view(self):
        self.spectrogram_data.fill(0)
        self.spec_image.setImage(self.spectrogram_data, autoLevels=False, levels=(0,1))
        self.current_f1, self.current_f2 = 0, 0
        self.current_pos_plot.setData([], [])
        self.measured_f1_line.setPos(-1)
        self.measured_f2_line.setPos(-1)

    def _highlight_target_vowel(self):
        for key, btn in self.vowel_buttons.items():
            is_target = (key == self.target_vowel)
            btn.setStyleSheet(f"background-color: {'#00ff88' if is_target else '#444'}; color: {'black' if is_target else 'white'};")
        for key, plot in self.vowel_plots.items():
            is_target = (key == self.target_vowel)
            plot.setSize(25 if is_target else 15)
            plot.setPen('w' if is_target else None, width=2)

    # --- Tab 2: Stress Analysis (Modified) ---

    def _create_stress_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        header = QLabel("<h2>🗣️ 英語発音 強弱コーチ AI (Direct Drag)</h2>")
        header.setMaximumHeight(40)
        layout.addWidget(header)
        
        instruction = QLabel("<b>ステップ1：</b>単語を入力してお手本を聞く。<br><b>ステップ2：</b>録音後、<b>赤線（開始）</b>と<b>青線（終了）</b>をマウスで掴んでトリミング。<br><b>ステップ3：</b>波形のカーブを重ねてリズムを確認。")
        instruction.setStyleSheet("font-size: 14px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;")
        instruction.setMaximumHeight(80)
        layout.addWidget(instruction)

        input_layout = QHBoxLayout()
        word_label = QLabel("練習したい単語:")
        word_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        input_layout.addWidget(word_label)
        self.word_input = QLineEdit("potato")
        self.word_input.setStyleSheet("padding: 8px; font-size: 14px; background-color: #1a1a1a; color: white; border: 2px solid #00ff88; border-radius: 5px;")
        input_layout.addWidget(self.word_input)
        self.play_ref_btn = QPushButton("🔈 お手本を聞く")
        self.play_ref_btn.clicked.connect(self._play_reference)
        input_layout.addWidget(self.play_ref_btn)
        layout.addLayout(input_layout)

        record_layout = QHBoxLayout()
        self.record_stress_btn = QPushButton("🎤 録音開始")
        self.record_stress_btn.clicked.connect(self._start_stress_recording)
        record_layout.addWidget(self.record_stress_btn)
        self.analyze_btn = QPushButton("診断スタート")
        self.analyze_btn.clicked.connect(self._analyze_stress)
        self.analyze_btn.setEnabled(False)
        record_layout.addWidget(self.analyze_btn)
        layout.addLayout(record_layout)

        # Canvas (Mouse events connected here)
        self.stress_canvas = MplCanvas(self, width=12, height=6, dpi=100)
        
        # ★ マウスイベントの接続
        self.stress_canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.stress_canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.stress_canvas.mpl_connect('button_release_event', self.on_canvas_release)
        
        layout.addWidget(self.stress_canvas, 1) 
        self._show_initial_graph()

        # Result Controls
        self.result_controls = QWidget()
        res_layout = QHBoxLayout(self.result_controls)
        res_layout.setContentsMargins(0, 5, 0, 5)
        self.btn_play_native_res = QPushButton("🔵 お手本のみ")
        self.btn_play_native_res.setStyleSheet("background-color: #2E86DE; color: white;")
        self.btn_play_native_res.clicked.connect(self._play_native_result)
        res_layout.addWidget(self.btn_play_native_res)
        self.btn_play_user_res = QPushButton("🔴 自分のみ")
        self.btn_play_user_res.setStyleSheet("background-color: #EE5A6F; color: white;")
        self.btn_play_user_res.clicked.connect(self._play_user_result)
        res_layout.addWidget(self.btn_play_user_res)
        self.btn_play_both_res = QPushButton("🔊 同時に再生")
        self.btn_play_both_res.setStyleSheet("background-color: #A355EE; color: white; font-weight: bold;")
        self.btn_play_both_res.clicked.connect(self._play_both_result)
        res_layout.addWidget(self.btn_play_both_res)
        self.btn_retrim = QPushButton("✂️ トリミングし直し")
        self.btn_retrim.setStyleSheet("background-color: #555; color: white; margin-left: 20px;")
        self.btn_retrim.clicked.connect(self._retrim_recording)
        res_layout.addWidget(self.btn_retrim)
        self.result_controls.setVisible(False)
        layout.addWidget(self.result_controls, 0)

        # Trim Info (No Slider, just text)
        self.trim_widget = QWidget()
        trim_layout = QVBoxLayout(self.trim_widget)
        trim_layout.setContentsMargins(0, 5, 0, 5)
        
        trim_label = QLabel("<b>📐 トリミング:</b> グラフ上の赤線(Start)と青線(End)を直接ドラッグしてください")
        trim_label.setStyleSheet("font-size: 13px; color: #00ff88;")
        trim_layout.addWidget(trim_label)
        
        info_layout = QHBoxLayout()
        self.trim_start_label = QLabel("Start: 0.00s")
        self.trim_start_label.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 14px;")
        info_layout.addWidget(self.trim_start_label)
        self.trim_end_label = QLabel("End: 0.00s")
        self.trim_end_label.setStyleSheet("color: #3366ff; font-weight: bold; font-size: 14px;")
        info_layout.addWidget(self.trim_end_label)
        trim_layout.addLayout(info_layout)
        
        trim_btn_layout = QHBoxLayout()
        self.play_trim_btn = QPushButton("🔊 選択範囲を再生")
        self.play_trim_btn.clicked.connect(self._play_trimmed_range)
        self.play_trim_btn.setStyleSheet("background-color: #3366ff; color: white;")
        trim_btn_layout.addWidget(self.play_trim_btn)
        self.apply_trim_btn = QPushButton("✂️ この範囲を使用")
        self.apply_trim_btn.clicked.connect(self._apply_trim)
        trim_btn_layout.addWidget(self.apply_trim_btn)
        self.reset_trim_btn = QPushButton("🔄 リセット")
        self.reset_trim_btn.clicked.connect(self._reset_trim)
        trim_btn_layout.addWidget(self.reset_trim_btn)
        trim_layout.addLayout(trim_btn_layout)
        
        self.trim_widget.setVisible(False)
        layout.addWidget(self.trim_widget, 0)

        self.stress_message = QTextEdit()
        self.stress_message.setReadOnly(True)
        self.stress_message.setMaximumHeight(100)
        self.stress_message.setStyleSheet("background-color: #1a1a1a; padding: 10px; font-size: 13px;")
        self.stress_message.setText("👆 まず「お手本を聞く」ボタンを押して、ネイティブの発音を確認してください。")
        layout.addWidget(self.stress_message, 0)

        # Variables
        self.stress_audio_data = None
        self.stress_recording = False
        self.original_audio = None
        self.audio_duration = 0
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback_position)
        self.playback_start_time = 0
        self.playback_duration = 0
        self.is_playing = False
        self.playback_position_line = None
        
        return widget

    # --- Mouse Event Handlers for Dragging Lines ---

    def on_canvas_press(self, event):
        """マウスをクリックした瞬間の処理"""
        if event.inaxes != self.stress_canvas.axes or not self.trim_widget.isVisible(): return
        if event.button != 1: return # 左クリックのみ

        # クリック位置
        click_x = event.xdata
        if click_x is None: return

        # 許容誤差（秒）- 近くをクリックすれば掴めるように
        tolerance = self.audio_duration * 0.05 # 5%の範囲

        dist_start = abs(click_x - self.trim_start_val)
        dist_end = abs(click_x - self.trim_end_val)

        if dist_start < tolerance and dist_start < dist_end:
            self.dragging_line = 'start'
        elif dist_end < tolerance:
            self.dragging_line = 'end'
        else:
            self.dragging_line = None

    def on_canvas_motion(self, event):
        """マウスを動かしている間の処理"""
        if self.dragging_line is None or event.xdata is None or event.inaxes != self.stress_canvas.axes:
            return

        new_x = event.xdata

        # 範囲制限
        if self.dragging_line == 'start':
            # 0以上、Endより手前
            new_x = max(0, min(new_x, self.trim_end_val - 0.05))
            self.trim_start_val = new_x
            self.trim_start_line.set_xdata([new_x, new_x])
            self.trim_start_label.setText(f"Start: {new_x:.2f}s")
        elif self.dragging_line == 'end':
            # Startより後ろ、最大時間以下
            new_x = max(self.trim_start_val + 0.05, min(new_x, self.audio_duration))
            self.trim_end_val = new_x
            self.trim_end_line.set_xdata([new_x, new_x])
            self.trim_end_label.setText(f"End: {new_x:.2f}s")

        self.stress_canvas.draw()

    def on_canvas_release(self, event):
        """マウスを離した時の処理"""
        self.dragging_line = None

    # --- Other Methods (Same logic, slightly updated variables) ---

    def _play_native_result(self):
        if self.is_playing: self._stop_playback()
        ref_path = "reference.wav"
        if os.path.exists(ref_path):
            data, fs = librosa.load(ref_path, sr=16000)
            sd.play(data, fs)

    def _play_user_result(self):
        if self.is_playing: self._stop_playback()
        if hasattr(self, 'user_audio_path') and os.path.exists(self.user_audio_path):
            data, fs = librosa.load(self.user_audio_path, sr=16000)
            sd.play(data, fs)

    def _play_both_result(self):
        if self.is_playing: self._stop_playback()
        ref_path = "reference.wav"
        user_path = getattr(self, 'user_audio_path', None)
        if not (os.path.exists(ref_path) and user_path and os.path.exists(user_path)): return
        try:
            ref_data, _ = librosa.load(ref_path, sr=16000)
            user_data, _ = librosa.load(user_path, sr=16000)
            max_len = max(len(ref_data), len(user_data))
            if len(ref_data) < max_len: ref_data = np.pad(ref_data, (0, max_len - len(ref_data)))
            if len(user_data) < max_len: user_data = np.pad(user_data, (0, max_len - len(user_data)))
            mixed_audio = (ref_data * 0.5) + (user_data * 0.5)
            sd.play(mixed_audio, samplerate=16000)
        except Exception as e:
            self.stress_message.setText(f"再生エラー: {str(e)}")

    def _retrim_recording(self):
        if self.is_playing: self._stop_playback()
        self.result_controls.setVisible(False)
        if self.original_audio is not None:
            self._show_waveform_and_trim_controls()
            self.stress_message.setText("📏 グラフ上の線をドラッグして調整し、「この範囲を使用」を押してください。")
            self.analyze_btn.setEnabled(False)
        else:
            self.stress_message.setText("❌ 元の録音データがありません。")

    def _play_reference(self):
        word = self.word_input.text().strip()
        if not word:
            self.stress_message.setText("❌ 単語を入力してください。")
            return
        self.stress_message.setText("🔄 お手本音声を生成中...")
        QApplication.processEvents()
        ref_path = generate_reference_audio(word)
        if ref_path and os.path.exists(ref_path):
            self.stress_message.setText("🔊 お手本音声を再生中...")
            QApplication.processEvents()
            os.system(f"afplay {ref_path}")
            self.stress_message.setText(f"✅ お手本を再生しました。\n次に「録音開始」ボタンを押して、真似して発音してください。")
        else:
            self.stress_message.setText("❌ お手本音声の生成に失敗しました。")

    def _start_stress_recording(self):
        if not self.stress_recording:
            self.stress_recording = True
            self.record_stress_btn.setText("⏹️ 録音停止")
            self.record_stress_btn.setStyleSheet("background-color: #ff4444; color: white;")
            self.stress_message.setText("🔴 録音中... 単語を発音してください。終わったら「録音停止」ボタンを押してください。")
            self.stress_audio_data = []
            self.stress_stream = sd.InputStream(samplerate=16000, channels=1, callback=self._stress_audio_callback, dtype='float32')
            self.stress_stream.start()
            self.result_controls.setVisible(False)
            self.trim_widget.setVisible(False)
            self._show_initial_graph()
        else:
            self.stress_recording = False
            self.stress_stream.stop()
            self.stress_stream.close()
            self.record_stress_btn.setText("🎤 録音開始")
            self.record_stress_btn.setStyleSheet("background-color: #00ff88; color: black;")
            if self.stress_audio_data:
                audio_array = np.concatenate(self.stress_audio_data, axis=0)
                if audio_array.ndim > 1: audio_array = audio_array.flatten()
                audio_array = audio_array.astype(np.float32)
                audio_array = np.ascontiguousarray(audio_array)
                self.original_audio = audio_array.copy()
                self.audio_duration = len(audio_array) / 16000
                self.user_audio_path = "user_recording.wav"
                import scipy.io.wavfile as wav
                wav.write(self.user_audio_path, 16000, audio_array)
                
                # 初期値設定
                self.trim_start_val = 0.0
                self.trim_end_val = self.audio_duration
                
                self._show_waveform_and_trim_controls()
                self.stress_message.setText(f"✅ 録音完了！\n赤線と青線をドラッグして調整 → 🔊「再生」で確認 → ✂️「使用」で決定")
                self.analyze_btn.setEnabled(False)
            else:
                self.stress_message.setText("❌ 録音データがありません。")

    def _stress_audio_callback(self, indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        self.stress_audio_data.append(indata.flatten().copy())

    def _show_waveform_and_trim_controls(self):
        self.stress_canvas.axes.clear()
        time_axis = np.linspace(0, self.audio_duration, len(self.original_audio))
        self.stress_canvas.axes.plot(time_axis, self.original_audio, color='#00ff88', linewidth=0.5)
        self.stress_canvas.axes.set_xlabel('Time (seconds)', fontsize=12, color='white')
        self.stress_canvas.axes.set_ylabel('Amplitude', fontsize=12, color='white')
        self.stress_canvas.axes.set_title('Drag Red/Blue Lines to Trim', fontsize=14, color='#00ff88')
        self.stress_canvas.axes.grid(True, linestyle='--', alpha=0.3, color='white')
        
        # 線を描画し、参照を保持
        self.trim_start_line = self.stress_canvas.axes.axvline(self.trim_start_val, color='red', linewidth=3, linestyle='--', label='Start')
        self.trim_end_line = self.stress_canvas.axes.axvline(self.trim_end_val, color='blue', linewidth=3, linestyle='--', label='End')
        
        self.trim_start_label.setText(f"Start: {self.trim_start_val:.2f}s")
        self.trim_end_label.setText(f"End: {self.trim_end_val:.2f}s")
        
        self.stress_canvas.axes.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        self.stress_canvas.draw()
        self.trim_widget.setVisible(True)

    def _play_trimmed_range(self):
        if self.is_playing:
            self._stop_playback()
            return
        if self.original_audio is None: return
        
        # スライダーではなく変数値を使用
        start_time = self.trim_start_val
        end_time = self.trim_end_val
        
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        
        # 範囲外チェック
        if start_sample >= end_sample: return

        trimmed_audio = self.original_audio[start_sample:end_sample]
        if trimmed_audio.ndim > 1: trimmed_audio = trimmed_audio.flatten()
        trimmed_audio = trimmed_audio.astype(np.float32)
        max_val = np.max(np.abs(trimmed_audio))
        if max_val > 0: trimmed_audio = trimmed_audio / max_val * 0.5
        trimmed_audio = np.ascontiguousarray(trimmed_audio)
        if self.playback_position_line is None:
            self.playback_position_line = self.stress_canvas.axes.axvline(start_time, color='orange', linewidth=3, linestyle='-', alpha=0.8)
        else:
            self.playback_position_line.set_xdata([start_time, start_time])
            self.playback_position_line.set_visible(True)
        self.stress_canvas.draw()
        self.play_trim_btn.setText("⏸️ 停止")
        self.play_trim_btn.setStyleSheet("background-color: #ff6600; color: white;")
        self.is_playing = True
        self.playback_start_time = start_time
        self.playback_duration = end_time - start_time
        try:
            sd.play(trimmed_audio, samplerate=16000, blocksize=2048)
            self.playback_actual_start = time.time()
            self.playback_timer.start(50)
        except Exception as e:
            print(f"Play Error: {e}")
            self._stop_playback()

    def _update_playback_position(self):
        if not self.is_playing: return
        elapsed = time.time() - self.playback_actual_start
        if elapsed >= self.playback_duration:
            self._stop_playback()
            return
        current_position = self.playback_start_time + elapsed
        if self.playback_position_line:
            self.playback_position_line.set_xdata([current_position, current_position])
            self.stress_canvas.figure.canvas.draw_idle()

    def _stop_playback(self):
        self.is_playing = False
        self.playback_timer.stop()
        sd.stop()
        if self.playback_position_line:
            self.playback_position_line.set_visible(False)
            self.stress_canvas.draw()
        self.play_trim_btn.setText("🔊 選択範囲を再生")
        self.play_trim_btn.setStyleSheet("background-color: #3366ff; color: white;")

    def _apply_trim(self):
        if self.is_playing: self._stop_playback()
        
        # 変数値を使用
        start_time = self.trim_start_val
        end_time = self.trim_end_val
        
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        trimmed_audio = self.original_audio[start_sample:end_sample]
        import scipy.io.wavfile as wav
        wav.write(self.user_audio_path, 16000, trimmed_audio)
        self.analyze_btn.setEnabled(True)
        trimmed_duration = (end_time - start_time)
        self.stress_message.setText(f"✂️ 範囲を設定しました（{trimmed_duration:.2f}秒）。\n👉 調整するか、そのまま「診断スタート」を押してください。")

    def _reset_trim(self):
        if self.is_playing: self._stop_playback()
        self.original_audio = None
        self.audio_duration = 0
        self.stress_audio_data = None
        if hasattr(self, 'user_audio_path') and os.path.exists(self.user_audio_path):
            os.remove(self.user_audio_path)
        self.trim_widget.setVisible(False)
        self.result_controls.setVisible(False)
        self._show_initial_graph()
        self.analyze_btn.setEnabled(False)
        self.stress_message.setText("🔄 リセットしました。「録音開始」からやり直してください。")

    def _show_initial_graph(self):
        self.stress_canvas.axes.clear()
        self.stress_canvas.axes.text(0.5, 0.5, '単語を入力して「お手本を聞く」→「録音」→「診断スタート」\nの順に操作してください',
            horizontalalignment='center', verticalalignment='center', transform=self.stress_canvas.axes.transAxes, fontsize=16, color='#00ff88', weight='bold')
        self.stress_canvas.axes.set_xlim(0, 1)
        self.stress_canvas.axes.set_ylim(0, 1)
        self.stress_canvas.axes.axis('off')
        self.stress_canvas.draw()

    def _analyze_stress(self):
        word = self.word_input.text().strip()
        if not word:
            self.stress_message.setText("❌ 単語を入力してください。")
            return
        if not hasattr(self, 'user_audio_path'):
            self.stress_message.setText("❌ 先に録音してください。")
            return
        self.stress_message.setText("🔄 解析中... 波形を生成しています。")
        QApplication.processEvents()
        ref_audio_path = generate_reference_audio(word)
        ref_time, ref_curve = extract_stress_contour(ref_audio_path)
        user_time, user_curve = extract_stress_contour(self.user_audio_path)
        if ref_time is None or user_time is None:
            self.stress_message.setText("❌ 解析に失敗しました。録音データを確認してください。")
            return
        self.stress_canvas.axes.clear()
        self.stress_canvas.axes.plot(ref_time, ref_curve, color='#2E86DE', linewidth=2, label='Native Speaker (Ref)')
        self.stress_canvas.axes.fill_between(ref_time, ref_curve, color='#2E86DE', alpha=0.1)
        self.stress_canvas.axes.plot(user_time, user_curve, color='#EE5A6F', linewidth=2, label='Your Pronunciation')
        self.stress_canvas.axes.set_xlabel('Time (seconds)', fontsize=12, color='white')
        self.stress_canvas.axes.set_ylabel('Stress Level (Pitch )', fontsize=12, color='white')
        self.stress_canvas.axes.set_title(f'Rhythm & Intonation: "{word.upper()}"', fontsize=16, fontweight='bold', color='#00ff88')
        self.stress_canvas.axes.grid(True, linestyle='--', alpha=0.3, color='white')
        self.stress_canvas.axes.legend(fontsize=10, facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        self.stress_canvas.draw()
        self.trim_widget.setVisible(False)
        self.result_controls.setVisible(True)
        msg = "✅ 解析完了！\n"
        ref_duration = ref_time[-1]
        user_duration = user_time[-1]
        if ref_duration > 0:
            ratio = user_duration / ref_duration
            if ratio > 1.3:
                msg += f"🐢 ネイティブより {ratio:.1f}倍 ゆっくり喋っています。\n"
                msg += "👉 「同時再生」でリズムの違いを聞き比べてみましょう。"
            elif ratio < 0.7:
                msg += f"🐇 ネイティブより {1/ratio:.1f}倍 早口です。\n"
                msg += "👉 もう少し母音を長く、粘り強く発音してみましょう。"
            else:
                msg += "👏 リズム（速さ）はバッチリです！\n"
                msg += "👉 赤い線と青い線の「山の位置」が重なるように練習しましょう。"
        self.stress_message.setText(msg)

    def closeEvent(self, event):
        if self.is_recording:
            self.audio_worker.stop()
            self.audio_thread.quit()
            self.audio_thread.wait()
        if hasattr(self, 'stress_recording') and self.stress_recording:
            self.stress_stream.stop()
            self.stress_stream.close()
        if hasattr(self, 'is_playing') and self.is_playing:
            self._stop_playback()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = IntegratedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
