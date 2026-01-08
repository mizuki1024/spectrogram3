import sys
import numpy as np
import parselmouth
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGridLayout, QTabWidget, QLineEdit, QTextEdit)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, Qt
import pyqtgraph as pg
from scipy import signal
import colorsys

import asyncio
import threading
import websockets
import json
import os

# === app_prototype.pyからの追加インポート ===
import torch
import torchaudio
import librosa
import matplotlib
matplotlib.use('Qt5Agg')  # PyQt6で使用するバックエンド
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 日本語フォント設定（macOS用）
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

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

# --- 音声処理部分（formant_en.pyから） ---
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
    if current_f1 <= 0 or current_f2 <= 0:
        return "マイクに向かって話してください"

    target_data = VOWELS[target_vowel]
    target_f1 = target_data['f1']
    target_f2 = target_data['f2']

    f1_diff_percent = ((current_f1 - target_f1) / target_f1) * 100
    f2_diff_percent = ((current_f2 - target_f2) / target_f2) * 100

    advice = []

    if abs(f1_diff_percent) > 10:
        if f1_diff_percent > 0:
            advice.append("🔽 口を少し閉じてください (F1が高すぎます)")
        else:
            advice.append("🔼 口をもっと開けてください (F1が低すぎます)")

    if abs(f2_diff_percent) > 10:
        if f2_diff_percent > 0:
            advice.append("👈 舌を後ろに引いてください (F2が高すぎます)")
        else:
            advice.append("👉 舌を前に出してください (F2が低すぎます)")

    if not advice:
        return "✅ 完璧です！この発音を維持してください！"

    return " | ".join(advice)

# === app_prototype.pyからの強弱分析関数 ===

# Wav2Vec2モデルのロード（遅延ロード）
class Wav2Vec2Models:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("Wav2Vec2モデルをロード中...")
        self.device = torch.device("cpu")
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.labels = self.bundle.get_labels()
        print("Wav2Vec2モデルのロード完了")

def robust_align(waveform, emission, labels, sample_rate):
    probs = torch.softmax(emission, dim=-1)
    predicted_indices = torch.argmax(probs, dim=-1)
    segments = []
    ratio = waveform.size(1) / emission.size(0) / sample_rate
    current_char = None
    start_frame = 0

    for t, char_idx in enumerate(predicted_indices):
        char = labels[char_idx]
        if char != current_char:
            if current_char is not None and current_char != "-":
                segments.append({
                    "char": current_char,
                    "start": start_frame * ratio,
                    "end": t * ratio
                })
            current_char = char
            start_frame = t

    return segments

def get_phoneme_intensities(audio_path, target_word):
    models = Wav2Vec2Models.get_instance()

    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except:
        return []

    waveform = torch.from_numpy(y).float().unsqueeze(0).to(models.device)

    with torch.inference_mode():
        emissions, _ = models.model(waveform)
        emissions = emissions[0]

    raw_segments = robust_align(waveform, emissions, models.labels, models.bundle.sample_rate)

    target_chars = list(target_word.upper())
    filtered_segments = []
    search_idx = 0

    for seg in raw_segments:
        if search_idx < len(target_chars) and seg['char'] == target_chars[search_idx]:
            filtered_segments.append(seg)
            search_idx += 1

    if not filtered_segments:
        return []

    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    results = []
    global_rms = np.sqrt(np.mean(y**2))
    global_f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 1.0
    all_durations = [seg['end'] - seg['start'] for seg in filtered_segments]
    global_duration_mean = np.mean(all_durations) if all_durations else 0.1

    for seg in filtered_segments:
        char = seg['char']
        start_time = seg['start']
        end_time = seg['end']

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        if end_sample - start_sample < 512:
            end_sample = start_sample + 512

        if end_sample > len(y):
            end_sample = len(y)
        if start_sample >= len(y):
            start_sample = max(0, len(y) - 512)

        chunk = y[start_sample:end_sample]

        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
        rms_normalized = rms / global_rms if global_rms > 0 else 0

        hop_length = 512
        start_frame = int(start_time * sr / hop_length)
        end_frame = int(end_time * sr / hop_length)

        if start_frame < len(f0) and end_frame <= len(f0) and end_frame > start_frame:
            f0_chunk = f0[start_frame:end_frame]
            f0_voiced = f0_chunk[f0_chunk > 0]
            mean_f0 = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
            f0_normalized = mean_f0 / global_f0_mean if global_f0_mean > 0 else 0
        else:
            f0_normalized = 0

        duration = end_time - start_time
        duration_normalized = duration / global_duration_mean if global_duration_mean > 0 else 0

        stress_score = (
            0.40 * f0_normalized +
            0.35 * rms_normalized +
            0.25 * duration_normalized
        )

        results.append({
            "char": char,
            "strength": stress_score,
            "rms": rms_normalized,
            "pitch": f0_normalized,
            "duration": duration_normalized
        })

    return results

def generate_reference_audio(word):
    if not word:
        return None
    ref_path = "reference.wav"
    os.system(f"say -v Samantha -o {ref_path} --data-format=LEF32@22050 {word}")
    return ref_path

# === AudioWorkerクラス（formant_en.pyから） ===
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
    positions = np.linspace(0.0, 1.0, 256)
    colors = []

    for pos in positions:
        hue = 0.3 + 0.4 * pos
        saturation = 0.4 + 0.6 * pos
        value = 1.0 - 0.6 * (pos ** 2)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r*255), int(g*255), int(b*255)))

    colors[0] = (0, 0, 0)

    return pg.ColorMap(positions, colors)

# === WebSocketサーバークラス ===
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

# === Matplotlibキャンバスクラス ===
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.patch.set_facecolor('#0a0a0a')
        self.axes.set_facecolor('#0a0a0a')
        self.axes.tick_params(colors='white')
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.spines['right'].set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        super(MplCanvas, self).__init__(fig)

# === メインアプリケーション ===
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

        # WebSocketサーバー
        self.ws_server = F1WebSocketServer()
        self.ws_server.start()

        # タブウィジェット作成
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # タブ1: リアルタイム母音分析
        self.tab1 = self._create_formant_tab()
        self.tabs.addTab(self.tab1, "母音分析")

        # タブ2: 単語強弱分析
        self.tab2 = self._create_stress_tab()
        self.tabs.addTab(self.tab2, "強弱分析")

        # AudioWorkerスレッド
        self.audio_thread = QThread()
        self.audio_worker = AudioWorker(sample_rate=self.sample_rate, chunk_size=self.chunk_size, max_freq=self.max_freq)
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_worker.data_updated.connect(self._update_ui)
        self.audio_thread.started.connect(self.audio_worker.start)

    def _create_formant_tab(self):
        """タブ1: リアルタイム母音分析（既存のformant_en.py機能）"""
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

    def _create_stress_tab(self):
        """タブ2: 単語強弱分析（app_prototype.py機能）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ヘッダー
        header = QLabel("<h2>🗣️ 英語発音 強弱コーチ AI</h2>")
        layout.addWidget(header)

        instruction = QLabel("<b>ステップ1：</b>単語を入力してお手本を聞く。<br><b>ステップ2：</b>真似して録音し、グラフで確認する。")
        instruction.setStyleSheet("font-size: 14px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;")
        layout.addWidget(instruction)

        # 単語入力エリア
        input_layout = QHBoxLayout()
        word_label = QLabel("練習したい単語:")
        word_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        input_layout.addWidget(word_label)

        self.word_input = QLineEdit("potato")
        self.word_input.setStyleSheet("""
            padding: 8px;
            font-size: 14px;
            background-color: #1a1a1a;
            color: white;
            border: 2px solid #00ff88;
            border-radius: 5px;
        """)
        self.word_input.setPlaceholderText("例: potato, apple, banana")
        input_layout.addWidget(self.word_input)

        self.play_ref_btn = QPushButton("🔈 お手本を聞く")
        self.play_ref_btn.clicked.connect(self._play_reference)
        input_layout.addWidget(self.play_ref_btn)

        layout.addLayout(input_layout)

        # 録音エリア
        record_layout = QHBoxLayout()
        self.record_stress_btn = QPushButton("🎤 録音開始")
        self.record_stress_btn.clicked.connect(self._start_stress_recording)
        record_layout.addWidget(self.record_stress_btn)

        self.analyze_btn = QPushButton("診断スタート")
        self.analyze_btn.clicked.connect(self._analyze_stress)
        self.analyze_btn.setEnabled(False)
        record_layout.addWidget(self.analyze_btn)

        layout.addLayout(record_layout)

        # グラフエリア
        self.stress_canvas = MplCanvas(self, width=12, height=6, dpi=100)
        layout.addWidget(self.stress_canvas)

        # 初期メッセージをグラフに表示
        self._show_initial_graph()

        # メッセージエリア
        self.stress_message = QTextEdit()
        self.stress_message.setReadOnly(True)
        self.stress_message.setMaximumHeight(100)
        self.stress_message.setStyleSheet("background-color: #1a1a1a; padding: 10px; font-size: 13px;")
        self.stress_message.setText("👆 まず「お手本を聞く」ボタンを押して、ネイティブの発音を確認してください。")
        layout.addWidget(self.stress_message)

        # 録音用の変数
        self.stress_audio_data = None
        self.stress_recording = False

        return widget

    def _show_initial_graph(self):
        """初期グラフに説明を表示"""
        self.stress_canvas.axes.clear()
        self.stress_canvas.axes.text(0.5, 0.5,
            '単語を入力して「お手本を聞く」→「録音」→「診断スタート」\nの順に操作してください',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.stress_canvas.axes.transAxes,
            fontsize=16,
            color='#00ff88',
            weight='bold')
        self.stress_canvas.axes.set_xlim(0, 1)
        self.stress_canvas.axes.set_ylim(0, 1)
        self.stress_canvas.axes.axis('off')
        self.stress_canvas.draw()

    def _create_plots(self):
        """スペクトログラムとフォルマント空間のプロット作成"""
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
        """コントロールパネル作成"""
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
        """ステータスパネル作成"""
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
        if is_formant:
            value.setProperty("class", "formant-info")
        layout.addWidget(label, row, 0)
        layout.addWidget(value, row, 1)
        return value

    def _set_target_vowel(self, vowel):
        """目標母音を設定"""
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

        if hasattr(self, "ws_server") and self.ws_server:
            self.ws_server.send_formant(
                self.current_f1 if self.is_recording else 100,
                self.target_vowel
            )

    @pyqtSlot(dict)
    def _update_ui(self, data):
        """UIを更新"""
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

            if self.is_recording:
                vowel, conf = classify_vowel(self.current_f1, self.current_f2)
                self.ws_server.send_formant(
                    self.current_f1,
                    self.target_vowel
                )

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

    def _toggle_recording(self):
        """録音のトグル"""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.setText("⏹️ Stop Recording")
            self.record_btn.setProperty("class", "recording")
            self.status_text.setText("🔴 Recording...")
            self.audio_thread.start()
            self._set_target_vowel(self.target_vowel)
            self.measured_f1_line.setPos(-1)
            self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText("")
            self.measured_f2_label.setText("")
        else:
            self.audio_worker.stop()
            self.audio_thread.quit()
            self.audio_thread.wait()
            self.record_btn.setText("🎙️ Start Recording")
            self.record_btn.setProperty("class", "")
            self.status_text.setText("Stopped.")
            self.measured_f1_line.setPos(-1)
            self.measured_f2_line.setPos(-1)
            self.measured_f1_label.setText("")
            self.measured_f2_label.setText("")

            self.ws_server.send_formant(
                100,
                self.target_vowel
            )

        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)

    def _reset_view(self):
        """ビューをリセット"""
        self.spectrogram_data.fill(0)
        self.spec_image.setImage(self.spectrogram_data, autoLevels=False, levels=(0,1))
        self.current_f1, self.current_f2 = 0, 0
        self.current_pos_plot.setData([], [])
        self.measured_f1_line.setPos(-1)
        self.measured_f2_line.setPos(-1)
        self.measured_f1_label.setText("")
        self.measured_f2_label.setText("")

    def _highlight_target_vowel(self):
        """目標母音をハイライト"""
        for key, btn in self.vowel_buttons.items():
            is_target = (key == self.target_vowel)
            style = f"background-color: {'#00ff88' if is_target else '#444'}; color: {'black' if is_target else 'white'};"
            btn.setStyleSheet(style)
        for key, plot in self.vowel_plots.items():
            is_target = (key == self.target_vowel)
            plot.setSize(25 if is_target else 15)
            plot.setPen('w' if is_target else None, width=2)

    # === タブ2: 強弱分析の機能 ===

    def _play_reference(self):
        """お手本音声を再生"""
        word = self.word_input.text().strip()
        if not word:
            self.stress_message.setText("❌ 単語を入力してください。")
            return

        self.stress_message.setText("🔄 お手本音声を生成中...")
        QApplication.processEvents()  # UIを更新

        ref_path = generate_reference_audio(word)
        if ref_path and os.path.exists(ref_path):
            self.stress_message.setText("🔊 お手本音声を再生中...")
            QApplication.processEvents()

            # macOSのafplayコマンドで音声を再生
            os.system(f"afplay {ref_path}")

            self.stress_message.setText(f"✅ お手本を再生しました。\n次に「録音開始」ボタンを押して、真似して発音してください。")
        else:
            self.stress_message.setText("❌ お手本音声の生成に失敗しました。")

    def _start_stress_recording(self):
        """強弱分析用の録音開始"""
        if not self.stress_recording:
            self.stress_recording = True
            self.record_stress_btn.setText("⏹️ 録音停止")
            self.record_stress_btn.setStyleSheet("background-color: #ff4444; color: white;")
            self.stress_message.setText("🔴 録音中... 単語を発音してください。終わったら「録音停止」ボタンを押してください。")

            # 録音開始
            self.stress_audio_data = []
            self.stress_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                callback=self._stress_audio_callback,
                dtype='float32'
            )
            self.stress_stream.start()
        else:
            # 録音停止
            self.stress_recording = False
            self.stress_stream.stop()
            self.stress_stream.close()
            self.record_stress_btn.setText("🎤 録音開始")
            self.record_stress_btn.setStyleSheet("")  # スタイルをリセット

            # 録音データを保存
            if self.stress_audio_data:
                audio_array = np.concatenate(self.stress_audio_data, axis=0)
                self.user_audio_path = "user_recording.wav"
                import scipy.io.wavfile as wav
                wav.write(self.user_audio_path, 16000, audio_array)
                self.stress_message.setText(f"✅ 録音完了しました！\n👉 次に「診断スタート」ボタンを押して分析してください。")
                self.analyze_btn.setEnabled(True)
            else:
                self.stress_message.setText("❌ 録音データがありません。もう一度お試しください。")

    def _stress_audio_callback(self, indata, frames, time, status):
        """強弱分析用の録音コールバック"""
        if status:
            print(status, file=sys.stderr)
        self.stress_audio_data.append(indata.copy())

    def _analyze_stress(self):
        """強弱分析を実行"""
        word = self.word_input.text().strip()
        if not word:
            self.stress_message.setText("❌ 単語を入力してください。")
            return

        if not hasattr(self, 'user_audio_path'):
            self.stress_message.setText("❌ 先に録音してください。")
            return

        self.stress_message.setText("🔄 AIで解析中... しばらくお待ちください。")
        QApplication.processEvents()  # UIを更新

        # お手本音声を生成
        ref_audio_path = generate_reference_audio(word)

        # ユーザーの音声とお手本の音声を解析
        user_data = get_phoneme_intensities(self.user_audio_path, word)
        ref_data = get_phoneme_intensities(ref_audio_path, word)

        if not user_data:
            self.stress_message.setText("⚠️ 音声認識に失敗しました。もう少しはっきり発音してみてください。\n（単語全体を発音していますか？）")
            return

        # グラフデータの準備
        labels = [d['char'] for d in ref_data]
        ref_values = [d['strength'] for d in ref_data]

        user_values = []
        ref_idx = 0
        for u_item in user_data:
            if ref_idx < len(ref_values):
                user_values.append(u_item['strength'])
                ref_idx += 1

        while len(user_values) < len(ref_values):
            user_values.append(0)

        # グラフを描画
        self.stress_canvas.axes.clear()
        x = np.arange(len(labels))

        self.stress_canvas.axes.plot(x, ref_values,
                marker='o', linestyle='-', color='#2E86DE',
                label='Native Speaker', linewidth=3, markersize=10, alpha=0.7)

        self.stress_canvas.axes.plot(x, user_values,
                marker='s', linestyle='-', color='#EE5A6F',
                label='Your Pronunciation', linewidth=3, markersize=10)

        self.stress_canvas.axes.set_xticks(x)
        self.stress_canvas.axes.set_xticklabels(labels, fontsize=16, fontweight='bold', color='white')
        self.stress_canvas.axes.set_ylabel('Stress Score', fontsize=14, fontweight='bold', color='white')
        self.stress_canvas.axes.set_xlabel('Phonemes', fontsize=14, fontweight='bold', color='white')
        self.stress_canvas.axes.set_title(f'Pronunciation Analysis: "{word.upper()}"',
                     fontsize=18, fontweight='bold', pad=20, color='#00ff88')

        # 主強勢の位置を強調（凡例より先に描画）
        max_ref_idx = np.argmax(ref_values)
        self.stress_canvas.axes.axvspan(max_ref_idx - 0.3, max_ref_idx + 0.3,
                   alpha=0.15, color='yellow', label='Primary Stress (主強勢)')

        # 凡例の設定
        legend = self.stress_canvas.axes.legend(fontsize=12, loc='upper right', facecolor='#1a1a1a', edgecolor='white')
        for text in legend.get_texts():
            text.set_color('white')

        self.stress_canvas.axes.grid(True, linestyle='--', alpha=0.3, color='white')
        self.stress_canvas.axes.set_ylim(bottom=0)

        self.stress_canvas.draw()

        # フィードバックメッセージ
        stress_position = labels[max_ref_idx] if max_ref_idx < len(labels) else "?"
        message = f"✅ 解析完了！\n\n"
        message += f"📍 この単語の主強勢: 「{stress_position}」の位置（黄色の帯）\n\n"
        message += f"💡 青い線（お手本）の形に近づけましょう\n"
        message += f"　 ・青い線が高い位置 = 強く発音\n"
        message += f"　 ・青い線が低い位置 = 弱く発音\n\n"
        message += f"🎯 改善のコツ: ピッチ（音の高さ）を意識すると良くなります"
        self.stress_message.setText(message)

    def closeEvent(self, event):
        """アプリ終了時の処理"""
        if self.is_recording:
            self.audio_worker.stop()
            self.audio_thread.quit()
            self.audio_thread.wait()
        if hasattr(self, 'stress_recording') and self.stress_recording:
            self.stress_stream.stop()
            self.stress_stream.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = IntegratedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
