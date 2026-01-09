import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/* =======================
   VOWELS（Pythonと一致）
======================= */
const VOWELS = {
  'a':  { name:'ah',   color:'#FF66CC', f1:634, f2:1088 },
  'i':  { name:'heed', color:'#FF5733', f1:324, f2:2426 },
  'u':  { name:"who'd",color:'#5833FF', f1:344, f2:1281 },
  'e':  { name:'eh',   color:'#66FF33', f1:502, f2:2065 },
  'o':  { name:'oh',   color:'#33CCFF', f1:445, f2:854 },
  'ɪ':  { name:'hid',  color:'#FF8D33', f1:390, f2:1990 },
  'ɛ':  { name:'head', color:'#FFC300', f1:530, f2:1840 },
  'æ':  { name:'had',  color:'#DAF7A6', f1:660, f2:1720 },
  'ɑ':  { name:'hod',  color:'#33FF57', f1:730, f2:1090 },
  'ʊ':  { name:'hood', color:'#33A5FF', f1:440, f2:1020 },
  'ʌ':  { name:'bud',  color:'#C70039', f1:640, f2:1190 },
  'ə':  { name:'sofa', color:'#900C3F', f1:500, f2:1500 },
  'iː': { name:'ee',   color:'#FF33AA', f1:270, f2:2290 },
  'ɑː': { name:'ahh',  color:'#33FF99', f1:750, f2:1200 },
  'ɜː': { name:'er',   color:'#FF9933', f1:550, f2:1600 },
  'ɔː': { name:'hawed',color:'#33FFCE', f1:570, f2:840 },
  'uː': { name:'oo',   color:'#3366FF', f1:300, f2:870 },
};

/* =======================
   WebSocket
======================= */
const MAXF1 = 900;
const MINF1 = 100;
const MAXF2 = 2500;
// const MINF2 = 800;

let f1Value = MINF1;
let f2Value = 2000;
let value = 0;
let value2 = 0;

const JAW_MIN = 145;
const JAW_MAX = 180;
const LIP_MIN = 4;
const LIP_MAX = 15;
const ROUND_MIN = 10;
const ROUND_MAX = 40;
const HEAD_MIN = 65;
const HEAD_MAX = 68;
const F2_ROUND_START = 1300;

let targetData = { f1: 100, f2: 0 };

const socket = new WebSocket("ws://localhost:8765");

socket.onopen = () => {
  console.log("✅ WebSocket接続成功！サーバーと連携できています。");
};

socket.onerror = (error) => {
  console.error("❌ WebSocket接続エラー:", error);
  console.error("Pythonアプリケーション（app_integrated.py）が起動しているか確認してください。");
};

socket.onclose = () => {
  console.log("⚠️ WebSocket接続が閉じられました。");
};

socket.onmessage = (event) => {
  console.log("RAW:", event.data);
  const data = JSON.parse(event.data);

  // 現在のf1 F1
  if (data.f1 !== undefined) {
    f1Value = Number(data.f1);
    console.log("F1 updated:", f1Value);
  }

  // 現在のf2 F2
  if (data.f2 !== undefined) {
    f2Value = Number(data.f2);
    console.log("F2 updated:", f2Value);
  }

  // 目標母音
  if (data.target_vowel && VOWELS[data.target_vowel]) {
    targetData.f1 = VOWELS[data.target_vowel].f1;
    targetData.f2 = VOWELS[data.target_vowel].f2;
    console.log("Target vowel updated:", data.target_vowel, targetData);
  }
};

/* =======================
   Scene
======================= */
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  60, window.innerWidth / window.innerHeight, 0.1, 1000
);
camera.position.set(0, 2, 21);

const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 1));

/* =======================
   Mouth models
======================= */
let mouthUnder = null;
let mouthOver = null;
let mouth_L = null;
let mouth_R = null;
let head = null;
let mouthUnder2 = null;
let mouthOver2 = null;
let mouth_L2 = null;
let mouth_R2 = null;
let head2 = null;

const loader = new GLTFLoader();

let position = 6;
// ===== 現在の口（右）=====
loader.load("/model.glb", (gltf) => {
  const model = gltf.scene;
  model.position.x = position;
  scene.add(model);

  mouthUnder = model.getObjectByName("mouth_under");
  mouthOver = model.getObjectByName("mouth_over");
  mouth_L = model.getObjectByName("mouth_L");
  mouth_R = model.getObjectByName("mouth_R");
  head = model.getObjectByName("head");

  console.log("current mouth:", mouthUnder, mouthOver, mouth_L, mouth_R, head);
});

// ===== 目標の口（左）=====
loader.load("/model2.glb", (gltf) => {
  const model = gltf.scene;
  model.position.x = -position;
  scene.add(model);

  mouthUnder2 = model.getObjectByName("mouth_under_2");
  mouthOver2 = model.getObjectByName("mouth_over_2");
  mouth_L2 = model.getObjectByName("mouth_L_2");
  mouth_R2 = model.getObjectByName("mouth_R_2");
  head2 = model.getObjectByName("head_2");

  console.log("target mouth:", mouthUnder2, mouthOver2, mouth_L2, mouth_R2, head2);
});

/* =======================
   Animation
======================= */
function animate() {
  requestAnimationFrame(animate);

  if (mouthUnder && mouthOver && mouth_L && mouth_R && head && mouthUnder2 && mouthOver2 && mouth_L2 && mouth_R2 && head2) {

    // ① 下顎（回転）
    let a = (JAW_MAX - JAW_MIN) / (MAXF1 - MINF1);
    let b = JAW_MAX - a * MAXF1;
    mouthUnder.rotation.x = (f1Value * a + b) * Math.PI / 180;

    // ② 上唇（Y移動）
    let c = (LIP_MAX - LIP_MIN) / (MAXF1 - MINF1);
    let d = LIP_MAX - c * MAXF1;
    mouthOver.position.y = (f1Value * c + d) * 0.001;

    // ③ 口の丸め（F2制限あり）
    let e = (ROUND_MAX - ROUND_MIN) / (MAXF2 - F2_ROUND_START);
    let f = ROUND_MAX - e * MAXF2;

    if (f2Value > MAXF2) {
      value = MAXF2;
    } else if (f2Value < F2_ROUND_START) {
      value = F2_ROUND_START;
    } else {
      value = f2Value;
    }

    let g = (value * e + f) * 0.001;

    mouth_L.position.x =  g;
    mouth_R.position.x = -g;

    // ④ 頭
    let h = (-HEAD_MAX + HEAD_MIN) / (MAXF1 - MINF1);
    let i = -HEAD_MAX - h * MAXF1;
    head.rotation.x = (f1Value * h + i) * Math.PI / 180;


    // 目標の口

    // ① 下顎（回転）
    let a2 = (JAW_MAX - JAW_MIN) / (MAXF1 - MINF1);
    let b2 = JAW_MAX - a2 * MAXF1;
    mouthUnder2.rotation.x = (targetData.f1 * a2 + b2) * Math.PI / 180;

    // ② 上唇（Y移動）
    let c2 = (LIP_MAX - LIP_MIN) / (MAXF1 - MINF1);
    let d2 = LIP_MAX - c2 * MAXF1;
    mouthOver2.position.y = (targetData.f1 * c2 + d2) * 0.001;

    // ③ 口の丸め（F2制限あり）
    let e2 = (ROUND_MAX - ROUND_MIN) / (MAXF2 - F2_ROUND_START);
    let f2 = ROUND_MAX - e2 * MAXF2;

    if (targetData.f2 > MAXF2) {
      value2 = MAXF2;
    } else if (targetData.f2 < F2_ROUND_START) {
      value2 = F2_ROUND_START;
    } else {
      value2 = targetData.f2;
    }

    let g2 = (value2 * e2 + f2) * 0.001;

    mouth_L2.position.x =  g2;
    mouth_R2.position.x = -g2;

    // ④ 頭
    let h2 = (-HEAD_MAX + HEAD_MIN) / (MAXF1 - MINF1);
    let i2 = -HEAD_MAX - h2 * MAXF1;
    head2.rotation.x = (targetData.f1 * h2 + i2) * Math.PI / 180;
  
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

/* =======================
   Resize
======================= */
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
