import gradio as gr
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 初期設定 (変更なし) ---
print("AIモデルをロード中...")
device = torch.device("cpu")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()

# --- 2. 解析関数群 (変更なし) ---
def robust_align(waveform, emission):
    probs = torch.softmax(emission, dim=-1)
    predicted_indices = torch.argmax(probs, dim=-1)
    segments = []
    ratio = waveform.size(1) / emission.size(0) / bundle.sample_rate
    current_char = None
    start_frame = 0
    for t, char_idx in enumerate(predicted_indices):
        char = labels[char_idx]
        if char != current_char:
            if current_char is not None and current_char != "-":
                segments.append({"char": current_char, "start": start_frame * ratio, "end": t * ratio})
            current_char = char
            start_frame = t
    return segments

def get_phoneme_intensities(audio_path, target_word):
    # 音声ロード & AI解析
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except:
        return []
        
    waveform = torch.from_numpy(y).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = emissions[0]
    
    raw_segments = robust_align(waveform, emissions)
    
    # ターゲット文字抽出
    target_chars = list(target_word.upper())
    filtered_segments = []
    search_idx = 0
    for seg in raw_segments:
        if search_idx < len(target_chars) and seg['char'] == target_chars[search_idx]:
            filtered_segments.append(seg)
            search_idx += 1
            
    # 強度計算
    y_full, _ = librosa.load(audio_path, sr=None)
    global_rms = np.sqrt(np.mean(y_full**2))
    
    results = []
    for seg in filtered_segments:
        start = int(seg['start'] * sr)
        end = int(seg['end'] * sr)
        if end - start < 512: end = start + 512
        chunk = y_full[start:end]
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
        strength = rms / global_rms if global_rms > 0 else 0
        results.append({"char": seg['char'], "strength": strength})
    return results

# --- 3. アプリ用機能関数 ---

def generate_reference_audio(word):
    """お手本音声を生成して返すだけの関数"""
    if not word: return None
    ref_path = "reference.wav"
    os.system(f"say -v Samantha -o {ref_path} --data-format=LEF32@22050 {word}")
    return ref_path

def analyze_and_plot(user_audio, target_word):
    """録音データを解析してグラフを返す関数"""
    if user_audio is None:
        return None, "⚠️ まず録音してください。"

    # 念のためリファレンスも再生成（整合性確保）
    ref_audio_path = generate_reference_audio(target_word)
    
    # 解析
    user_data = get_phoneme_intensities(user_audio, target_word)
    ref_data = get_phoneme_intensities(ref_audio_path, target_word)
    
    if not user_data:
        return None, "⚠️ 音声認識に失敗しました。もう少しはっきり発音してみてください。"

    # グラフデータ準備
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

    # 描画
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    
    ax.plot(x, ref_values, marker='o', linestyle='-', color='blue', label='Native', linewidth=2, alpha=0.6)
    ax.plot(x, user_values, marker='o', linestyle='-', color='red', label='You', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Stress Level')
    ax.set_title(f"Pronunciation Check: {target_word}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig, "✅ 解析完了！青い線（お手本）の形を真似しましょう。"

# --- 4. 画面レイアウト構築 (Blocks) ---

with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown("# 🗣️ 英語発音 強弱コーチ AI")
    gr.Markdown("ステップ1：単語を入力してお手本を聞く。 ステップ2：真似して録音し、グラフで確認する。")
    
    # 上段：単語入力とお手本再生
    with gr.Row():
        with gr.Column(scale=2):
            txt_word = gr.Textbox(value="potato", label="練習したい単語 (Word)", lines=1)
        with gr.Column(scale=1):
            btn_play_ref = gr.Button("🔈 お手本を聞く (Play Native Audio)", variant="secondary")
    
    # お手本プレイヤー（最初は隠れていても良いが、ここでは常時表示）
    audio_ref = gr.Audio(label="お手本音声", type="filepath", interactive=False)

    gr.Markdown("---") # 区切り線

    # 中段：録音エリア
    with gr.Row():
        audio_mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤 あなたの発音を録音")
        btn_analyze = gr.Button("診断スタート (Analyze)", variant="primary")

    # 下段：結果表示
    with gr.Row():
        plot_output = gr.Plot(label="強弱比較グラフ")
    
    txt_msg = gr.Textbox(label="システムメッセージ")

    # --- イベント設定 ---
    
    # 「お手本を聞く」ボタンが押されたら -> 音声を生成してプレイヤーにセット
    btn_play_ref.click(
        fn=generate_reference_audio,
        inputs=[txt_word],
        outputs=[audio_ref]
    )

    # 「診断スタート」ボタンが押されたら -> 録音音声と単語を使って解析
    btn_analyze.click(
        fn=analyze_and_plot,
        inputs=[audio_mic, txt_word],
        outputs=[plot_output, txt_msg]
    )

if __name__ == "__main__":
    demo.launch()