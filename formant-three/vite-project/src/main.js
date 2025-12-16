import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// =======================
// WebSocket (F1受信)
// =======================
let f1Value = 100; // ← スライダー初期値と同じ

const socket = new WebSocket("ws://localhost:8765");

socket.onmessage = (event) => {
  console.log("RAW:", event.data);   // ← まず生データ確認

  let data;
  try {
    data = JSON.parse(event.data);
  } catch (e) {
    console.error("JSON parse error:", e);
    return;
  }

  console.log("PARSED:", data);      // ← パース後確認

  if (data.F1 !== undefined) {
    f1Value = Number(data.F1);
    console.log("F1 received:", f1Value);
  } else {
    console.warn("F1 not found in data");
  }
};

// ----- Scene -----
const scene = new THREE.Scene();

// Camera
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 2, 20);

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);

// Light
scene.add(new THREE.AmbientLight(0xffffff, 1));

// ===== Mouth Bones =====
let mouthUnder = null;
let mouthOver = null;

// ----- GLB load -----
const loader = new GLTFLoader();
loader.load(
  "/model.glb",
  (gltf) => {
    const model = gltf.scene;
    scene.add(model);

    mouthUnder = model.getObjectByName("mouth_under");
    mouthOver  = model.getObjectByName("mouth_over");
  }
);

// Animation
function animate() {
  requestAnimationFrame(animate);

  if (mouthUnder && mouthOver) {

    // 🔴 計算式はそのまま
    const move = (f1Value * 1 / 140 - 145 / 7) * 0.001;
    mouthOver.position.y = move;

    const rad = ((f1Value * 13 / 350 + 632 / 7) / 180) * Math.PI;
    mouthUnder.rotation.x = rad;
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

// Resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});