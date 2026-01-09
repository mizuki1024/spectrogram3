# formant-three

## 概要
本フォルダは、Python によるフォルマント解析と、
JavaScript（three.js）による 3D 口形状可視化を連動させた
アプリケーションを含む。

Python 側アプリケーションが処理開始を制御し、
抽出された第1フォルマント（F1）周波数に基づいて、
three.js 側で口形状をリアルタイムに制御する。

---

## ディレクトリ構成
formant-three/
├─ formant_en.py(不要)
└─ vite-project/
└─ app_integrated.py
---

## 実行環境

### Python
- Python 3.9 以上

### JavaScript / Web
- Node.js 18 以上
- npm

---

## 使用方法

### 1. Python 側アプリケーションの起動
フォルマント解析および通信制御を行う Python アプリケーションを起動する。

```bash
python formant-three/app_integrated.py

別のターミナルで Web アプリケーションを起動する。

cd vite-project
npm install
npm run dev

ブラウザで表示されたページを開いたままにしておく。

⸻

3. アプリケーションの開始

Python アプリケーション上の Start ボタンを押す。

Start ボタンを押すと、音声入力の取得とフォルマント解析が開始され、
抽出された第1フォルマント（F1）値が WebSocket を介して
three.js 側にリアルタイム送信される。

three.js 側では、受信した F1 値に応じて
3D 顔モデルの口形状が連動して動作する。

⸻

注意事項
	•	Python 側アプリケーションを先に起動してください。



