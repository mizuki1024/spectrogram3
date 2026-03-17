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

// vite-project/public/model_half.glbは、舌のモデル未作成

const VOWELS = {
  'a':  { name:'ah',   color:'#FF66CC', f1:634, f2:1088, lip_open:0.1189, lip_wide:0.5435 , model: "/model_half.glb"},
  'i':  { name:'heed', color:'#FF5733', f1:324, f2:2426, lip_open:0.0412, lip_wide:0.7208 , model: "/model_half.glb"},
  'u':  { name:"who'd",color:'#5833FF', f1:344, f2:1281, lip_open:0.0701, lip_wide:0.5292 , model: "/model_half.glb"},
  'e':  { name:'eh',   color:'#66FF33', f1:502, f2:2065, lip_open:0.1223, lip_wide:0.6871 , model: "/model_half.glb"},
  'o':  { name:'oh',   color:'#33CCFF', f1:445, f2:854, lip_open:0.1036, lip_wide:0.5199  , model: "/model_half.glb"},
  'ɪ':  { name:'hid',  color:'#FF8D33', f1:390, f2:1990, lip_open:0.0887, lip_wide:0.7121 , model: "/model_half_i.glb"},
  'ɛ':  { name:'head', color:'#FFC300', f1:530, f2:1840, lip_open:0.1315, lip_wide:0.6947 , model: "/model_half.glb"},
  'æ':  { name:'had',  color:'#DAF7A6', f1:660, f2:1720, lip_open:0.1292, lip_wide:0.6863 , model: "/model_half_ae.glb"},
  'ɑ':  { name:'hod',  color:'#33FF57', f1:730, f2:1090, lip_open:0.1616, lip_wide:0.5615 , model: "/model_half_a.glb"},
  'ʊ':  { name:'hood', color:'#33A5FF', f1:440, f2:1020, lip_open:0.1142, lip_wide:0.5624 , model: "/model_half_u.glb"},
  'ʌ':  { name:'bud',  color:'#C70039', f1:640, f2:1190, lip_open:0.1391, lip_wide:0.5578 , model: "/model_half_v.glb"},
  'ə':  { name:'sofa', color:'#900C3F', f1:500, f2:1500, lip_open:0.1352, lip_wide:0.5653 , model: "/model_half.glb"},
  'iː': { name:'ee',   color:'#FF33AA', f1:270, f2:2290, lip_open:0.0275, lip_wide:0.7028 , model: "/model_half_i_.glb"},
  'ɑː': { name:'ahh',  color:'#33FF99', f1:750, f2:1200, lip_open:0.1692, lip_wide:0.5454 , model: "/model_half_a_.glb"},
  'ɜː': { name:'er',   color:'#FF9933', f1:550, f2:1600, lip_open:0.1319, lip_wide:0.6744 , model: "/model_half.glb"},
  'ɔː': { name:'hawed',color:'#33FFCE', f1:570, f2:840, lip_open:0.1380, lip_wide:0.5534  , model: "/model_half_c_.glb"},
  'uː': { name:'oo',   color:'#3366FF', f1:300, f2:870, lip_open:0.0587, lip_wide:0.5441  , model: "/model_half_u_.glb"},
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
const ROUND_MIN = 15;
const ROUND_MAX = 40;
const HEAD_MIN = 65;
const HEAD_MAX = 68;
const F2_ROUND_START = 600;

let targetData = { f1: 100, f2: 0 };

const socket = new WebSocket("ws://localhost:8765");

socket.onmessage = (event) => {
  console.log("RAW:", event.data);
  const data = JSON.parse(event.data);

  let f1Updated = false;
  let f2Updated = false;

  if (data.f1 !== undefined) {
    f1Value = Number(data.f1);
    f1Updated = true;
  }

  if (data.f2 !== undefined) {
    f2Value = Number(data.f2);
    f2Updated = true;
  }


  // 目標母音
  if (data.target_vowel && VOWELS[data.target_vowel]) {
    targetData.f1 = VOWELS[data.target_vowel].f1;
    targetData.f2 = VOWELS[data.target_vowel].f2;
  }

  console.log("Target:", targetData.f1, targetData.f2);

  if (f1Updated || f2Updated) {
    updateModel();
  }
};

/* =======================
   目標母音の判定
======================= */

// 目標母音の誤差
const F_EPS = 1e-6;

function getTargetVowelByExactMatch(f1, f2) {
  for (const key in VOWELS) {
    const v = VOWELS[key];
    if (
      Math.abs(v.f1 - f1) < F_EPS &&
      Math.abs(v.f2 - f2) < F_EPS
    ) {
      return key;
    }
  }
  return null;
}

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
// let mouthUnder_half = null;
// let mouthOver_half = null;
// let mouth_L_half = null;
// let mouth_R_half = null;
// let head_half = null;
// let tongue00 = null;
// let tongue01 = null;
// let tongue02 = null;
// let tongue03 = null;
// let tongue04 = null;

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

let currentModelPath = null;

const modelCache = {};

function updateModel() {
  console.log("🔥 updateModel called", targetData.f1, targetData.f2);
  const vowel = getTargetVowelByExactMatch(targetData.f1, targetData.f2);
  console.log("👉 vowel:", vowel);
  if (!vowel) return;

  const modelPath = VOWELS[vowel].model;
  console.log("👉 modelPath:", modelPath);

  if (modelPath === currentModelPath) return;
  currentModelPath = modelPath;

  halfModelGroup.clear();

  // 👇キャッシュあれば再利用
  if (modelCache[modelPath]) {
    halfModelGroup.add(modelCache[modelPath].clone(true));
    return;
  }

  loader.load(modelPath, (gltf) => {
    console.log("✅ loaded:", modelPath);
    console.log("children:", gltf.scene.children);

    const model = gltf.scene;

    model.rotation.y = -Math.PI / 2;

    modelCache[modelPath] = model;
    halfModelGroup.add(model);

    // 必要なら再取得
    // mouthUnder_half = model.getObjectByName("mouth_under_half");
    // mouthOver_half  = model.getObjectByName("mouth_over_half");
    // mouth_L_half    = model.getObjectByName("mouth_L_half");
    // mouth_R_half    = model.getObjectByName("mouth_R_half");
    // head_half       = model.getObjectByName("head_half");
  });

}

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
const CAMERA_PLANE_H = 12;

//映像を貼るメッシュの生成
const cameraPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(CAMERA_PLANE_W, CAMERA_PLANE_H, 1, 1),
  new THREE.MeshBasicMaterial({ map: videoTexture })
);

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
// const TARGET_POINTS = [13, 14, 61, 291, 152, 10, 33, 263];
const TARGET_POINTS = [13, 14, 61, 291];

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
// let smoothwidthAngleDeg_1 = 0;
// let smoothwidthAngleDeg_2 = 0;
const alpha = 0.2;   // 0.1〜0.3 推奨

function updateLandmarks(faceLandmarks) {

const p10  = faceLandmarks[10];
const p152 = faceLandmarks[152];
const p33  = faceLandmarks[33];
const p263 = faceLandmarks[263];
const p61  = faceLandmarks[61];
const p291 = faceLandmarks[291];
const p13  = faceLandmarks[13];
const p14  = faceLandmarks[14];

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
  const faceHeight = Math.hypot(
    p152.x - p10.x,
    p152.y - p10.y
  );
  if (faceHeight === 0) return;

  // 顔の幅(目の外側の距離)
  const faceWidth = Math.hypot(
    p263.x - p33.x,
    p263.y - p33.y
  );

  const mouthWidth = Math.hypot(
    p291.x - p61.x,
    p291.y - p61.y
  );

  const mouthOpen = Math.hypot(
    p14.x - p13.x,
    p14.y - p13.y
  );

    // 正規化（横基準）
    const widthNorm = mouthWidth / faceWidth;

    // 正規化（縦基準）
    const openNorm = mouthOpen / faceHeight;

  // ===== 距離計算 =====
  //前回値と今回値を線形補間
  smoothOpen  = alpha * openNorm  + (1 - alpha) * smoothOpen;
  smoothWidth = alpha * widthNorm + (1 - alpha) * smoothWidth;

  if (infoDiv) {
    infoDiv.innerText =
      `open: ${smoothOpen.toFixed(3)} | width: ${smoothWidth.toFixed(3)}`;
  }
  
}

const OPEN_EPS = 0.015;   // 縦の許容誤差
const WIDE_EPS = 0.02;    // 横の許容誤差

function judgeMouthInstruction() {

  if (!targetData || targetData.f2 === 0) return "";

  // 目標母音
  const targetKey = getTargetVowelByExactMatch(
    targetData.f1,
    targetData.f2
  );

  if (!targetKey) return "";

  const t = VOWELS[targetKey];

  const openDiff  = smoothOpen  - t.lip_open;
  const wideDiff  = smoothWidth - t.lip_wide;

  const openOK = Math.abs(openDiff) < OPEN_EPS;
  const wideOK = Math.abs(wideDiff) < WIDE_EPS;

  if (openOK && wideOK) {
    return "完璧！";
  }

  let msgs = [];

  // 横
  if (!wideOK) {
    msgs.push(wideDiff < 0
      ? "もっと口を横に開けて"
      : "横はそのままで"
    );
  } else {
    msgs.push("横はそのままで");
  }

  // 縦
  if (!openOK) {
    msgs.push(openDiff < 0
      ? "もっと口を縦に開けて"
      : "縦はそのままで"
    );
  } else {
    msgs.push("縦はそのままで");
  }

  return msgs.join("、");
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

targetData = { f1: 634, f2: 1088 };
updateModel();

const instructionDiv = document.getElementById("instruction");

let lastInstruction = "";

/* =======================
   Animation
======================= */
function animate() {
  requestAnimationFrame(animate);

  if (mouthUnder && mouthOver && mouth_L && mouth_R && head && mouthUnder2 && mouthOver2 && mouth_L2 && mouth_R2 && head2) {

    // ① 下顎（回転）
    let a = (JAW_MAX - JAW_MIN) / (MAXF1 - MINF1);
    let b = JAW_MAX - a * MAXF1;

    let f1ForJaw = f1Value;

    if(f1Value >= 450 && f1Value <= 680) {
      f1ForJaw = 500;
    } else if(f1Value > 680 ) {
      f1ForJaw = 5/3 * f1Value - 1900/3;
    }

    mouthUnder.rotation.x = (f1ForJaw * a + b) * Math.PI / 180;

    // ② 上唇（Y移動）
    let c = (LIP_MAX - LIP_MIN) / (MAXF1 - MINF1);
    let d = LIP_MAX - c * MAXF1;
    mouthOver.position.y = (f1ForJaw * c + d) * 0.001;

    // ③ 口の丸め（F2制限あり）
    let e = (ROUND_MAX - ROUND_MIN) / (MAXF2 - F2_ROUND_START);
    let f = ROUND_MAX - e * MAXF2;

    value = f2Value;

    if (f2Value >= 1600) {
      value = 0.36 * f2Value + 1680;
      // console.log(value);
      
    } else if (f2Value < 1600) {
      value = 2/7 * f2Value + 4000/7;
      // console.log(value);
    }

    let g = (value * e + f) * 0.001;

    mouth_L.position.x =  g;
    mouth_R.position.x = -g;

    // ④ 頭
    let h = (-HEAD_MAX + HEAD_MIN) / (MAXF1 - MINF1);
    let i = -HEAD_MAX - h * MAXF1;
    head.rotation.x = (f1ForJaw * h + i) * Math.PI / 180;


    // 目標の口

    // ① 下顎（回転）
    let a2 = (JAW_MAX - JAW_MIN) / (MAXF1 - MINF1);
    let b2 = JAW_MAX - a2 * MAXF1;

    let f1ForJaw2 = targetData.f1;

    if(targetData.f1 >= 450 && targetData.f1 <= 680) {
      f1ForJaw2 = 500;
    } else if(targetData.f1 > 680 ) {
      f1ForJaw2 = 5/3 * targetData.f1 - 1900/3;
    }
    let aa = (f1ForJaw2 * a2 + b2) * Math.PI / 180;
    mouthUnder2.rotation.x = aa;
    // console.log(aa);

    // ② 上唇（Y移動）
    let c2 = (LIP_MAX - LIP_MIN) / (MAXF1 - MINF1);
    let d2 = LIP_MAX - c2 * MAXF1;
    mouthOver2.position.y = (f1ForJaw2 * c2 + d2) * 0.001;

    // ③ 口の丸め（F2制限あり）
    let e2 = (ROUND_MAX - ROUND_MIN) / (MAXF2 - F2_ROUND_START);
    let f2 = ROUND_MAX - e2 * MAXF2;

    value2 = targetData.f2;

    if (targetData.f2 >= 1600) {
      value2 = 0.36 * targetData.f2 + 1680;
      // console.log(value);
      
    } else if (targetData.f2 < 1600) {
      value2 = 2/7 * targetData.f2 + 4000/7;
      // console.log(value);
    }

    // console.log(value2);

    let g2 = (value2 * e2 + f2) * 0.001;

    mouth_L2.position.x =  g2;
    mouth_R2.position.x = -g2;

    // console.log(g2);

    // ④ 頭
    let h2 = (-HEAD_MAX + HEAD_MIN) / (MAXF1 - MINF1);
    let i2 = -HEAD_MAX - h2 * MAXF1;
    head2.rotation.x = (f1ForJaw2 * h2 + i2) * Math.PI / 180;
  
  }

  if (chkCamera.checked) {
    detectFace();

    // ★ カメラONのときだけ判定
    const inst = judgeMouthInstruction();

    if (inst !== lastInstruction) {
      instructionDiv.innerText = inst;
      lastInstruction = inst;
    }

  } else {
    // ★ カメラOFFなら指示を消す
    instructionDiv.innerText = "";
    lastInstruction = "";
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
