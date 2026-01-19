import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8";

/* =======================
   Scene / Camera / Renderer
======================= */
const scene = new THREE.Scene();

//カメラの設定
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
// camera.position.set(0, 2, 21);

const CAMERA_PRESET = {
  1: { x: 0, y: 2, z: 21 }, // 1モデル
  2: { x: 0, y: 2, z: 26 }, // 2モデル
  3: { x: 0, y: 2, z: 32 }, // 3モデル
};

//レンダーの設定
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
//canvasをHTMLのbodyに追加
document.body.appendChild(renderer.domElement);

//マウスで回転や移動などの操作
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 1));

/* =======================
   表示用 Group（4要素）
======================= */
const currentModelGroup = new THREE.Group();
const targetModelGroup  = new THREE.Group();
const halfModelGroup    = new THREE.Group();
const cameraGroup       = new THREE.Group();

scene.add(currentModelGroup);
scene.add(targetModelGroup);
scene.add(halfModelGroup);
scene.add(cameraGroup);

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
  }
};


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
let mouthUnder_half = null;
let mouthOver_half = null;
let mouth_L_half = null;
let mouth_R_half = null;
let head_half = null;
let tongue00 = null;
let tongue01 = null;
let tongue02 = null;
let tongue03 = null;
let tongue04 = null;

/* =======================
   GLTF Models
======================= */
const loader = new GLTFLoader();
// const position = 6;

// 現在モデル
loader.load("/model.glb", (gltf) => {
  const model = gltf.scene;
  // model.position.x = position;
  currentModelGroup.add(model);

  mouthUnder = model.getObjectByName("mouth_under");
  mouthOver = model.getObjectByName("mouth_over");
  mouth_L = model.getObjectByName("mouth_L");
  mouth_R = model.getObjectByName("mouth_R");
  head = model.getObjectByName("head");

  console.log("current mouth:", mouthUnder, mouthOver, mouth_L, mouth_R, head);

});

// 目標モデル
loader.load("/model2.glb", (gltf) => {
  const model = gltf.scene;
  // model.position.x = -position;
  targetModelGroup.add(model);

  mouthUnder2 = model.getObjectByName("mouth_under_2");
  mouthOver2 = model.getObjectByName("mouth_over_2");
  mouth_L2 = model.getObjectByName("mouth_L_2");
  mouth_R2 = model.getObjectByName("mouth_R_2");
  head2 = model.getObjectByName("head_2");

  console.log("target mouth:", mouthUnder2, mouthOver2, mouth_L2, mouth_R2, head2);
});

//モデル(半身)
loader.load("/model_half.glb", (gltf) => {
  const model = gltf.scene;
  // model.position.x = position-5;
  model.rotation.y = -Math.PI / 2;
  halfModelGroup.add(model);

  mouthUnder_half = model.getObjectByName("mouth_under_half");
  mouthOver_half = model.getObjectByName("mouth_over_half");
  mouth_L_half = model.getObjectByName("mouth_L_half");
  mouth_R_half = model.getObjectByName("mouth_R_half");
  head_half = model.getObjectByName("head_half");
  tongue00 = model.getObjectByName("tounge00_half");
  tongue01 = model.getObjectByName("tounge01_half");
  tongue02 = model.getObjectByName("tounge02_half");
  tongue03 = model.getObjectByName("tounge03_half");
  tongue04 = model.getObjectByName("tounge04_half");

  console.log("current mouth:", mouthUnder_half, mouthOver_half, mouth_L_half, mouth_R_half, head_half);

});

/* =======================
   Web Camera Plane
======================= */
const videoEl = document.createElement("video");
videoEl.autoplay = true;
videoEl.muted = true;
videoEl.playsInline = true;

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    videoEl.srcObject = stream;
    videoEl.onloadedmetadata = () => videoEl.play();
  })
  .catch(err => console.error("Camera error:", err));

const videoTexture = new THREE.VideoTexture(videoEl);
videoTexture.minFilter = THREE.LinearFilter;
videoTexture.magFilter = THREE.LinearFilter;
videoTexture.colorSpace = THREE.SRGBColorSpace;

const CAMERA_PLANE_W = 12;
const CAMERA_PLANE_H = 15;

//映像を貼るメッシュの生成
const cameraPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(CAMERA_PLANE_W, CAMERA_PLANE_H, 1, 1),
  new THREE.MeshBasicMaterial({ map: videoTexture })
);
// cameraPlane.position.set(0, 2, 6);
//表示するためのシーンに追加
cameraGroup.add(cameraPlane);


currentModelGroup.visible = true;
targetModelGroup.visible = false;
halfModelGroup.visible = false;
cameraGroup.visible = false;

/* =======================
   MediaPipe FaceLandmarker
======================= */
let faceLandmarker;
let lastVideoTime = -1;

// 使用している顔のランドマーク
const TARGET_POINTS = [13, 14, 61, 291, 152, 10];

async function initFaceLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });
}

initFaceLandmarker()
  .then(() => {
    console.log("FaceLandmarker ready");
  })
  .catch(err => {
    console.error("FaceLandmarker init error:", err);
  });

/* =======================
   Landmark Visualization
======================= */

//ランドマークを表示するためのグループ
const landmarkGroup = new THREE.Group();
cameraPlane.add(landmarkGroup);

//ランドマークを表示するためのメッシュ
const landmarkMeshes = {};
const pointGeometry = new THREE.SphereGeometry(0.08, 16, 16);
const pointMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });

//各ポイントにランドマークを表示
TARGET_POINTS.forEach(i => {
  const mesh = new THREE.Mesh(pointGeometry, pointMaterial);
  landmarkGroup.add(mesh);
  landmarkMeshes[i] = mesh;
});

/* =======================
   Landmark Update
======================= */

function dist(a, b) {
  if (!a || !b) return 0; // aまたはbがundefinedなら0を返す
  return a.position.distanceTo(b.position);
}

const infoDiv = document.getElementById("info");

// =======================
// 安全に距離を計算する関数
// =======================
function safeDist(a, b) {
  if (!a || !b) return 0;  // a または b が undefined なら 0
  return a.position.distanceTo(b.position);
}

// =======================
// ★ 平滑化用（追加）
// =======================
let smoothOpen  = 0;
let smoothWidth = 0;
let smoothwidthAngleDeg_1 = 0;
let smoothwidthAngleDeg_2 = 0;
const alpha = 0.2;   // 0.1〜0.3 推奨

function updateLandmarks(faceLandmarks) {
  if (!faceLandmarks) return; // undefinedなら何もしない

  //カメラのサイズ
  const w = CAMERA_PLANE_W;
  const h = CAMERA_PLANE_H;

  //ランドマークの座標を更新
  TARGET_POINTS.forEach(i => {
    const lm = faceLandmarks[i];
    if (!lm) return;

    const x = (lm.x - 0.5) * w;
    const y = -(lm.y - 0.5) * h;
    const z = 0.01;

    landmarkMeshes[i].position.set(x, y, z);
  });


  //顔の高さの計算
  const faceHeight = faceLandmarks[152].y - faceLandmarks[10].y;
  if (faceHeight === 0) return; // まだ正しい顔サイズが取れていない

  //正規化
  const openNorm  = (faceLandmarks[14].y  - faceLandmarks[13].y) / faceHeight;
  const widthNorm = (faceLandmarks[61].x - faceLandmarks[291].x) / faceHeight;


  const p61  = faceLandmarks[61];
  const p291 = faceLandmarks[291];
  const p13  = faceLandmarks[13];
  const p14  = faceLandmarks[14];
  
  // ===== 基準ベクトル（61 → 291）=====
  const baseX = p291.x - p61.x;
  const baseY = p291.y - p61.y;
  const baseAngle = Math.atan2(baseY, baseX); // これが 0° 扱い
  
  // ===== 61 → 13 =====
  const v13x = p13.x - p61.x;
  const v13y = p13.y - p61.y;
  const angle13 = Math.atan2(v13y, v13x) - baseAngle;
  
  // ===== 61 → 14 =====
  const v14x = p14.x - p61.x;
  const v14y = p14.y - p61.y;
  const angle14 = Math.atan2(v14y, v14x) - baseAngle;
  
  function normalizeAngle(rad) {
    if (rad > Math.PI)  rad -= 2 * Math.PI;
    if (rad < -Math.PI) rad += 2 * Math.PI;
    return rad;
  }
  
  const normAngle13 = normalizeAngle(angle13);
  const normAngle14 = normalizeAngle(angle14);

  const deg13 = -normAngle13 * 180 / Math.PI;
  const deg14 = normAngle14 * 180 / Math.PI;

  

  // ===== 距離計算 =====
  //前回値と今回値を線形補間
  smoothOpen  = alpha * openNorm  + (1 - alpha) * smoothOpen;
  smoothWidth = alpha * widthNorm + (1 - alpha) * smoothWidth;
  smoothwidthAngleDeg_1 = alpha * deg13 + (1 - alpha) * smoothwidthAngleDeg_1;
  smoothwidthAngleDeg_2 = alpha * deg14 + (1 - alpha) * smoothwidthAngleDeg_2;


  // infoDiv.innerText = `open: ${smoothOpen.toFixed(3)} | width: ${smoothWidth.toFixed(3)}`;
  if (infoDiv) {
    infoDiv.innerText =
      `open: ${smoothOpen.toFixed(3)} | widthDeg_1: ${smoothwidthAngleDeg_1.toFixed(3)} | widthDeg_2: ${smoothwidthAngleDeg_2.toFixed(3)}`;
  }
  
}


//毎フレーム顔を検出
async function detectFace() {
  if (
    videoEl.videoWidth === 0 ||
    videoEl.videoHeight === 0
  ) {
    return; // まだカメラ準備中
  }
  if (!faceLandmarker) return;

  if (videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;

    const result = await faceLandmarker.detectForVideo(videoEl, performance.now());

    if (result.faceLandmarks.length > 0) {
      updateLandmarks(result.faceLandmarks[0]);
    }
  }
}

// /* =======================
//    チェックボックス制御
// ======================= */

/* =======================
   チェックボックス制御
======================= */
const chkCurrent = document.getElementById("chkCurrent");
const chkTarget  = document.getElementById("chkTarget");
const chkHalf    = document.getElementById("chkHalf");
const chkCamera  = document.getElementById("chkCamera");

const items = [
  { chk: chkCurrent, group: currentModelGroup },
  { chk: chkTarget,  group: targetModelGroup },
  { chk: chkHalf,    group: halfModelGroup },
  { chk: chkCamera,  group: cameraGroup }
];

/* ===== スロット定義 ===== */
const SLOT_MAP = {
  1: [  0 ],
  2: [ -7,  7 ],
  3: [ -13,  0,  13 ],
};

const MODEL_PRIORITY = new Map([
  [targetModelGroup,  0],
  [halfModelGroup,    1],
  [currentModelGroup, 2],
  [cameraGroup,       3],
]);

const layoutItems = [
  { chk: chkTarget,  group: targetModelGroup,  type: "model" },
  { chk: chkHalf,    group: halfModelGroup,    type: "model" },
  { chk: chkCurrent, group: currentModelGroup, type: "model" },
  { chk: chkCamera,  group: cameraGroup,       type: "camera" },
];

function updateLayout() {

  // /* ===== 初期化 ===== */
  layoutItems.forEach(item => {
    item.group.visible = false;
  });

  /* ===== camera 表示 ===== */
  landmarkGroup.visible = chkCamera.checked;

  /* ===== モデル抽出 ===== */

  const activeItems = layoutItems
  .filter(item => item.chk.checked)
  .sort((a, b) =>
    (MODEL_PRIORITY.get(a.group) ?? 99) -
    (MODEL_PRIORITY.get(b.group) ?? 99)
  );

  const visibleItems = activeItems.slice(0, 3);
  const count = visibleItems.length;

  if (count === 0) {
    controls.enabled = false;
    return;
  }

  /* ===== スロット ===== */
  const slots = SLOT_MAP[count];

  visibleItems.forEach((item, index) => {
    item.group.visible = true;
    item.group.position.set(slots[index], 0, 0);
  });

  /* ===== カメラ位置 ===== */
  const cam = CAMERA_PRESET[count];
  camera.position.set(cam.x, cam.y, cam.z);
  camera.lookAt(0, 0, 0);

  /* ===== OrbitControls ===== */
  controls.enabled = true;
}

// イベント登録
// items.forEach(item => {
//   item.chk.addEventListener("change", updateLayout);
// });
function getCheckedCount() {
  return items.filter(item => item.chk.checked).length;
}

items.forEach(item => {
  item.chk.addEventListener("change", (e) => {

    const checkedCount = getCheckedCount();

    // ❌ 0個は禁止
    if (checkedCount === 0) {
      e.target.checked = true;
      return;
    }

    // ❌ 4個以上は禁止
    if (checkedCount > 3) {
      e.target.checked = false;
      return;
    }

    // ✅ 条件OKならレイアウト更新
    updateLayout();
  });
});

// ★ 初期状態を反映（重要）
updateLayout({ target: chkCurrent });

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

  if (chkCamera.checked) {
    detectFace();
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
