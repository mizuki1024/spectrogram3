# === 必要なライブラリをインポート ===
import gradio as gr              # Webアプリの画面を作るライブラリ
import torch                     # PyTorch: AIモデルを動かすためのフレームワーク
import torchaudio                # 音声処理用のPyTorchライブラリ（Wav2Vec2モデルを使う）
import librosa                   # 音声解析ライブラリ（ピッチ検出などに使用）
import numpy as np               # 数値計算ライブラリ（配列操作、統計計算）
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import os                        # OSコマンド実行用（macOSの音声合成を使う）

# === 1. 初期設定: AIモデルのロード ===
print("AIモデルをロード中...")  # ユーザーに進行状況を表示

# 計算デバイスを設定（GPU が使えない場合は CPU を使用）
device = torch.device("cpu")

# Wav2Vec2 の事前学習済みモデルをロード
# WAV2VEC2_ASR_BASE_960H = 960時間の英語音声で学習されたモデル
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

# モデル本体を取得して、指定したデバイス（CPU）に配置
model = bundle.get_model().to(device)

# モデルが認識できる文字のリスト（A〜Z + 特殊記号）を取得
# labels = ['-', '|', 'E', 'T', 'A', ...] のような29個の文字リスト
labels = bundle.get_labels()

# === 2. 解析関数群 ===

def robust_align(waveform, emission):
    """
    AIモデルの出力（emission）を時間軸付きの音素セグメントに変換する関数

    入力:
        waveform: 音声波形データ（torch.Tensor）
        emission: AIモデルの予測結果（各時間フレーム×各文字の確率）
    出力:
        segments: [{"char": "H", "start": 0.0, "end": 0.06}, ...] のリスト
    """
    # AIの生スコアを確率に変換（各時間フレームで合計が1.0になるように正規化）
    probs = torch.softmax(emission, dim=-1)

    # 各時間フレームで最も確率が高い文字のインデックス番号を取得
    # 例: [8, 8, 8, 2, 2, ...] のような数字の配列
    predicted_indices = torch.argmax(probs, dim=-1)

    # 結果を保存するリスト（最終的に返すデータ）
    segments = []

    # フレーム番号を実時間（秒）に変換する係数を計算
    # waveform.size(1) = 音声の総サンプル数
    # emission.size(0) = AIの出力フレーム数
    # bundle.sample_rate = サンプリングレート（16000Hz）
    ratio = waveform.size(1) / emission.size(0) / bundle.sample_rate

    # 現在処理中の文字（最初は何もない）
    current_char = None
    # 現在の文字グループの開始フレーム番号
    start_frame = 0

    # 各時間フレームをループ処理
    # t = 時間フレーム番号（0, 1, 2, ...）
    # char_idx = そのフレームで予測された文字のインデックス番号
    for t, char_idx in enumerate(predicted_indices):
        # インデックス番号を実際の文字に変換（8 → 'H' など）
        char = labels[char_idx]

        # 文字が変わったかチェック
        if char != current_char:
            # 前の文字のセグメントを保存（最初のループと特殊記号"-"は除外）
            if current_char is not None and current_char != "-":
                segments.append({
                    "char": current_char,              # 文字
                    "start": start_frame * ratio,      # 開始時刻（秒）
                    "end": t * ratio                   # 終了時刻（秒）
                })

            # 新しい文字グループの処理を開始
            current_char = char
            start_frame = t

    # 完成したセグメントリストを返す
    return segments

def get_phoneme_intensities(audio_path, target_word):
    """
    音声ファイルから各音素の強勢特徴を抽出する改良版

    この関数が行うこと:
    1. 音声ファイルを読み込む
    2. AIで音素（音の単位）を検出
    3. 各音素の強勢を3つの指標で測定:
       - 音量（RMS）
       - ピッチ（F0）
       - 持続時間
    4. それらを統合して総合的な強勢スコアを計算

    入力:
        audio_path: 音声ファイルのパス
        target_word: 分析したい単語（例: "potato"）
    出力:
        各音素の情報を含む辞書のリスト
    """
    # === ステップ1: 音声ファイルを読み込む ===
    # librosa.load() で音声を読み込み、16000Hzにリサンプリング
    # y = 音声データ（numpy配列）
    # sr = サンプリングレート（16000）
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except:
        # ファイル読み込み失敗時は空リストを返す
        return []

    # === ステップ2: AIモデルで音素を検出 ===
    # numpy配列をPyTorchのテンソルに変換
    # .float() = 浮動小数点型に変換
    # .unsqueeze(0) = バッチ次元を追加（[samples] → [1, samples]）
    # .to(device) = CPUに配置
    waveform = torch.from_numpy(y).float().unsqueeze(0).to(device)

    # AIモデルを推論モードで実行（学習しないので高速）
    with torch.inference_mode():
        # model() を実行してemissionを取得
        # emissions = 各時間フレーム×各文字の予測スコア
        # _ = 使わない2番目の返り値を無視
        emissions, _ = model(waveform)
        # バッチ次元を削除（[1, frames, chars] → [frames, chars]）
        emissions = emissions[0]

    # robust_align() で時間軸付きの音素セグメントに変換
    # 例: [{"char": "P", "start": 0.0, "end": 0.1}, ...]
    raw_segments = robust_align(waveform, emissions)

    # === ステップ3: ターゲット単語の音素だけを抽出 ===
    # 単語を大文字に変換して文字リストに分解
    # "potato" → ['P', 'O', 'T', 'A', 'T', 'O']
    target_chars = list(target_word.upper())

    # ターゲット単語に該当する音素だけをフィルタリング
    filtered_segments = []
    search_idx = 0  # 現在探している文字の位置

    # AIが検出した全セグメントをループ
    for seg in raw_segments:
        # まだ探す文字が残っていて、かつ現在のセグメントがその文字と一致するか？
        if search_idx < len(target_chars) and seg['char'] == target_chars[search_idx]:
            # 一致したらリストに追加
            filtered_segments.append(seg)
            # 次の文字を探す
            search_idx += 1

    # ターゲット単語の音素が1つも見つからなかった場合は空リストを返す
    if not filtered_segments:
        return []

    # === 1. ピッチ（F0）解析 ===
    # librosa.pyin() = 音声からピッチ（音の高さ）を検出する関数
    # 返り値: f0（ピッチの時系列データ）、voiced_flag（無視）、voiced_probs（無視）
    f0, _, _ = librosa.pyin(
        y,                                  # 解析対象の音声データ（16000Hzのnumpy配列）
        fmin=librosa.note_to_hz('C2'),      # 最低周波数: 65Hz（人間の声の下限。低いドの音）
        fmax=librosa.note_to_hz('C7'),      # 最高周波数: 2093Hz（人間の声の上限。高いドの音）
        sr=sr,                              # サンプリングレート（16000Hz）
        frame_length=2048                   # 1フレームあたりのサンプル数（ピッチ検出の精度に影響）
    )
    # librosa.pyinは無声音（ささやき声など）の部分でNaN（数値なし）を返すので、0に置き換える
    # こうすることで、後の計算でエラーが出ないようにする
    f0 = np.nan_to_num(f0, nan=0.0)

    # === 2. 各音素の特徴量を計算 ===
    # 最終的な結果を保存するリスト（各音素ごとに1つのデータが入る）
    results = []

    # === グローバル統計（正規化用の基準値） ===
    # 音素ごとの値を比較可能にするため、音声全体の平均値を計算する

    # 音声全体の音量（RMS = Root Mean Square）を計算
    # y**2 = 各サンプルを2乗、mean = 平均、sqrt = 平方根
    # これで音声全体の「平均的な音の大きさ」がわかる
    global_rms = np.sqrt(np.mean(y**2))

    # 音声全体の平均ピッチ（F0）を計算
    # f0[f0 > 0] = ピッチが検出された部分だけを抽出（0は無声音なので除外）
    # np.any(f0 > 0) = ピッチが1つでも検出されたかチェック
    # 検出されていれば平均を計算、されていなければ1.0を使う（ゼロ除算を防ぐ）
    global_f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 1.0

    # 各音素の持続時間（長さ）をリストで取得
    # seg['end'] - seg['start'] = 終了時刻 - 開始時刻 = 持続時間（秒）
    # すべてのセグメントに対してこの計算を実行
    all_durations = [seg['end'] - seg['start'] for seg in filtered_segments]

    # 音素の平均持続時間を計算（発話速度の基準になる）
    # all_durations が空の場合は0.1秒をデフォルト値として使う
    global_duration_mean = np.mean(all_durations) if all_durations else 0.1

    # === ステップ4: 各音素の特徴量を計算 ===
    # filtered_segments の各セグメントに対してループ処理
    for seg in filtered_segments:
        # セグメント情報を取得
        char = seg['char']          # 文字（例: 'P'）
        start_time = seg['start']   # 開始時刻（秒）
        end_time = seg['end']       # 終了時刻（秒）

        # --- 時間（秒）をサンプル位置（整数インデックス）に変換 ---
        # 時刻 × サンプリングレート = サンプル位置
        # 例: 0.1秒 × 16000Hz = 1600サンプル目
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 最小長を確保（短すぎるとRMS計算が不安定になるため）
        # 512サンプル = 約0.032秒（16000Hzの場合）
        if end_sample - start_sample < 512:
            end_sample = start_sample + 512

        # 範囲チェック（音声データの範囲を超えないように）
        if end_sample > len(y):
            end_sample = len(y)
        if start_sample >= len(y):
            # 開始位置が音声の外にある場合、最後の512サンプルを使う
            start_sample = max(0, len(y) - 512)

        # この音素に該当する音声データの一部（チャンク）を切り出す
        chunk = y[start_sample:end_sample]

        # === 特徴量1: 音量（RMS） ===
        # RMS = Root Mean Square（二乗平均平方根）= 音の大きさ
        # chunk**2 = 各サンプルを2乗
        # mean = 平均を計算
        # sqrt = 平方根を取る
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0

        # グローバルRMSで割って正規化（全体の平均を1.0とする）
        # これで「この音素は平均の何倍の音量か？」がわかる
        rms_normalized = rms / global_rms if global_rms > 0 else 0

        # === 特徴量2: ピッチ（F0の平均） ===
        # F0データはフレーム単位なので、時間をフレームインデックスに変換する必要がある

        # hop_length = フレーム間の間隔（サンプル数）
        # librosa.pyinのデフォルトは512サンプル
        hop_length = 512

        # 時間（秒）をF0のフレームインデックスに変換
        # 計算式: 時刻（秒） × サンプリングレート / hop_length = フレーム番号
        start_frame = int(start_time * sr / hop_length)
        end_frame = int(end_time * sr / hop_length)

        # フレーム範囲が有効かチェック
        if start_frame < len(f0) and end_frame <= len(f0) and end_frame > start_frame:
            # この音素に該当するF0データを切り出す
            f0_chunk = f0[start_frame:end_frame]

            # 有声音（声帯が振動している音）のF0だけを抽出
            # 0 は無声音なので除外する
            f0_voiced = f0_chunk[f0_chunk > 0]

            # 有声音のF0の平均を計算（この音素の平均ピッチ）
            mean_f0 = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0

            # グローバル平均ピッチで割って正規化
            # これで「この音素は平均の何倍の高さか？」がわかる
            f0_normalized = mean_f0 / global_f0_mean if global_f0_mean > 0 else 0
        else:
            # フレーム範囲が無効な場合は0
            f0_normalized = 0

        # === 特徴量3: 持続時間 ===
        # 終了時刻 - 開始時刻 = この音素の長さ（秒）
        duration = end_time - start_time

        # グローバル平均持続時間で割って正規化
        # これで「この音素は平均の何倍の長さか？」がわかる
        duration_normalized = duration / global_duration_mean if global_duration_mean > 0 else 0

        # === 総合ストレススコア ===
        # 英語の強勢（stress）は複数の要素で決まる:
        # - ピッチ（高い音）: 40%の重み
        # - 音量（大きい音）: 35%の重み
        # - 持続時間（長い音）: 25%の重み
        # この重み付けは言語学的研究に基づいている
        stress_score = (
            0.40 * f0_normalized +      # ピッチの寄与
            0.35 * rms_normalized +     # 音量の寄与
            0.25 * duration_normalized  # 持続時間の寄与
        )

        # この音素の全情報を辞書としてリストに追加
        results.append({
            "char": char,                      # 文字
            "strength": stress_score,          # 総合強勢スコア
            "rms": rms_normalized,             # 正規化された音量
            "pitch": f0_normalized,            # 正規化されたピッチ
            "duration": duration_normalized    # 正規化された持続時間
        })

    # 全音素の解析結果を返す
    return results

# === 3. アプリ用機能関数 ===

def generate_reference_audio(word):
    """
    macOSの音声合成機能を使ってお手本音声を生成する関数

    入力:
        word: 発音させたい単語（例: "potato"）
    出力:
        生成された音声ファイルのパス（"reference.wav"）
    """
    # 単語が空の場合はNoneを返す
    if not word:
        return None

    # 出力ファイルのパス
    ref_path = "reference.wav"

    # macOSの "say" コマンドを使って音声合成
    # -v Samantha: 女性のネイティブ英語音声を使用
    # -o {ref_path}: 出力ファイルパスを指定
    # --data-format=LEF32@22050: 32bit float, 22050Hzで出力
    # {word}: 発音する単語
    os.system(f"say -v Samantha -o {ref_path} --data-format=LEF32@22050 {word}")

    # 生成されたファイルパスを返す
    return ref_path

def analyze_and_plot(user_audio, target_word):
    """
    ユーザーの録音音声を解析してお手本と比較するグラフを生成する関数

    入力:
        user_audio: ユーザーが録音した音声ファイルのパス
        target_word: 分析対象の単語（例: "potato"）
    出力:
        fig: matplotlibのグラフオブジェクト
        message: ユーザーへのフィードバックメッセージ
    """
    # === ステップ1: 入力チェック ===
    # 録音がない場合はエラーメッセージを返す
    if user_audio is None:
        return None, "⚠️ まず録音してください。"

    # === ステップ2: お手本音声を生成 ===
    # 毎回新しいお手本を生成することで、単語が変更された場合にも対応
    ref_audio_path = generate_reference_audio(target_word)

    # === ステップ3: 音声解析 ===
    # ユーザーの音声とお手本の音声を両方解析
    user_data = get_phoneme_intensities(user_audio, target_word)
    ref_data = get_phoneme_intensities(ref_audio_path, target_word)

    # ユーザーの音声が認識できなかった場合はエラーメッセージを返す
    if not user_data:
        return None, "⚠️ 音声認識に失敗しました。もう少しはっきり発音してみてください。"

    # === ステップ4: グラフデータの準備 ===
    # お手本データから文字ラベルと強勢値を抽出
    # labels = ['P', 'O', 'T', 'A', 'T', 'O']
    labels = [d['char'] for d in ref_data]
    # ref_values = [0.5, 1.2, 0.8, 1.5, 0.9, 0.7] のような強勢スコアのリスト
    ref_values = [d['strength'] for d in ref_data]

    # ユーザーデータをお手本データに合わせて整列
    # （ユーザーが一部の音素を発音しなかった場合に対応）
    user_values = []
    ref_idx = 0

    # ユーザーが発音した各音素を順番に処理
    for u_item in user_data:
        # まだお手本の音素が残っている場合
        if ref_idx < len(ref_values):
            # その音素の強勢スコアを追加
            user_values.append(u_item['strength'])
            ref_idx += 1

    # ユーザーが発音しなかった音素があれば、0で埋める
    # これでグラフの横軸の長さが揃う
    while len(user_values) < len(ref_values):
        user_values.append(0)

    # === ステップ5: グラフの描画 ===
    # 12インチ×6インチのグラフを作成
    fig, ax = plt.subplots(figsize=(12, 6))

    # X軸の位置（0, 1, 2, ... len(labels)-1）
    x = np.arange(len(labels))

    # お手本の線を描画（青色、丸マーカー）
    ax.plot(x, ref_values,
            marker='o',              # 丸マーカー
            linestyle='-',           # 実線
            color='#2E86DE',         # 青色
            label='Native Speaker',  # 凡例のラベル
            linewidth=3,             # 線の太さ
            markersize=10,           # マーカーのサイズ
            alpha=0.7)               # 透明度（70%）

    # ユーザーの線を描画（赤色、四角マーカー）
    ax.plot(x, user_values,
            marker='s',                    # 四角マーカー
            linestyle='-',                 # 実線
            color='#EE5A6F',               # 赤色
            label='Your Pronunciation',    # 凡例のラベル
            linewidth=3,                   # 線の太さ
            markersize=10)                 # マーカーのサイズ

    # X軸の設定
    ax.set_xticks(x)  # 目盛りの位置
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')  # ラベル（P, O, T, ...）

    # Y軸のラベル
    ax.set_ylabel('Stress Score', fontsize=14, fontweight='bold')

    # グラフのタイトル（単語名を含む）
    ax.set_title(f'Pronunciation Analysis: "{target_word.upper()}"',
                 fontsize=18, fontweight='bold', pad=20)

    # 凡例を右上に表示
    ax.legend(fontsize=12, loc='upper right')

    # グリッド線を表示（点線、薄く）
    ax.grid(True, linestyle='--', alpha=0.3)

    # Y軸の下限を0に設定（負の値を表示しない）
    ax.set_ylim(bottom=0)

    # === 主強勢の位置を黄色で強調表示 ===
    # ref_values の中で最大値のインデックスを取得
    max_ref_idx = np.argmax(ref_values)

    # その位置に薄い黄色の縦帯を描画
    ax.axvspan(max_ref_idx - 0.3,    # 帯の左端
               max_ref_idx + 0.3,     # 帯の右端
               alpha=0.15,             # 透明度（15%）
               color='yellow',         # 黄色
               label='Primary Stress') # 凡例のラベル

    # レイアウトを自動調整（タイトルやラベルが切れないように）
    plt.tight_layout()

    # === ステップ6: フィードバックメッセージを生成 ===
    # 主強勢がある文字を取得
    stress_position = labels[max_ref_idx] if max_ref_idx < len(labels) else "?"

    # ユーザーに表示するメッセージを組み立てる
    message = f"✅ 解析完了！\n"
    message += f"📍 この単語の主強勢: 「{stress_position}」の位置\n"
    message += f"💡 青い線（お手本）の形に近づけましょう\n"
    message += f"🎯 特にピッチ（音の高さ）を意識すると改善します"

    # グラフとメッセージを返す
    return fig, message

# === 4. 画面レイアウト構築（Gradio Blocks） ===

# Gradio Blocks API を使ってカスタムレイアウトを作成
# title = ブラウザのタブに表示されるタイトル
with gr.Blocks(title="Pronunciation Coach") as demo:

    # === ヘッダー部分 ===
    # Markdownで見出しと説明を表示
    gr.Markdown("# 🗣️ 英語発音 強弱コーチ AI")
    gr.Markdown("ステップ1：単語を入力してお手本を聞く。 ステップ2：真似して録音し、グラフで確認する。")

    # === 上段：単語入力とお手本再生ボタン ===
    # gr.Row() = 横並びのレイアウト
    with gr.Row():
        # 左側（幅の2/3）: 単語入力欄
        with gr.Column(scale=2):
            txt_word = gr.Textbox(
                value="potato",                      # 初期値
                label="練習したい単語 (Word)",       # ラベル
                lines=1                               # 1行のテキストボックス
            )
        # 右側（幅の1/3）: お手本再生ボタン
        with gr.Column(scale=1):
            btn_play_ref = gr.Button(
                "🔈 お手本を聞く (Play Native Audio)",  # ボタンのテキスト
                variant="secondary"                      # スタイル（灰色）
            )

    # お手本音声のプレイヤー（音声ファイルを再生するためのコンポーネント）
    # type="filepath" = ファイルパスを受け取る
    # interactive=False = ユーザーが編集できない（再生のみ）
    audio_ref = gr.Audio(label="お手本音声", type="filepath", interactive=False)

    # 区切り線を表示
    gr.Markdown("---")

    # === 中段：録音エリア ===
    with gr.Row():
        # マイク録音コンポーネント
        audio_mic = gr.Audio(
            sources=["microphone"],              # マイクからの入力のみを許可
            type="filepath",                     # ファイルパスとして処理
            label="🎤 あなたの発音を録音"       # ラベル
        )
        # 診断開始ボタン
        btn_analyze = gr.Button(
            "診断スタート (Analyze)",            # ボタンのテキスト
            variant="primary"                     # スタイル（青色で強調）
        )

    # === 下段：結果表示エリア ===
    with gr.Row():
        # グラフを表示するコンポーネント
        plot_output = gr.Plot(label="強弱比較グラフ")

    # システムメッセージを表示するテキストボックス
    txt_msg = gr.Textbox(label="システムメッセージ")

    # === イベント設定（ボタンのクリック時の動作を定義） ===

    # 「お手本を聞く」ボタンがクリックされたときの処理
    btn_play_ref.click(
        fn=generate_reference_audio,  # 実行する関数
        inputs=[txt_word],             # 入力: 単語テキストボックスの値
        outputs=[audio_ref]            # 出力: お手本音声プレイヤーに音声をセット
    )

    # 「診断スタート」ボタンがクリックされたときの処理
    btn_analyze.click(
        fn=analyze_and_plot,           # 実行する関数
        inputs=[audio_mic, txt_word],  # 入力: 録音音声と単語
        outputs=[plot_output, txt_msg] # 出力: グラフとメッセージ
    )

# === 5. アプリケーションの起動 ===
# このスクリプトが直接実行された場合のみ、Webサーバーを起動
if __name__ == "__main__":
    demo.launch()  # Gradioアプリを起動（ブラウザで http://localhost:7860 が開く）