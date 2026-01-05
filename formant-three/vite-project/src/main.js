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
let f1Value = 100;
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

  // 目標母音
  if (data.target_vowel && VOWELS[data.target_vowel]) {
    targetData.f1 = VOWELS[data.target_vowel].f1;
    targetData.f2 = VOWELS[data.target_vowel].f2;
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
let mouthUnder2 = null;
let mouthOver2 = null;


const loader = new GLTFLoader();

let position = 6;
// ===== 現在の口（右）=====
loader.load("/model.glb", (gltf) => {
  const model = gltf.scene;
  model.position.x = position;
  scene.add(model);

  mouthUnder = model.getObjectByName("mouth_under");
  mouthOver = model.getObjectByName("mouth_over");

  console.log("current mouth:", mouthUnder, mouthOver);
});

// ===== 目標の口（左）=====
loader.load("/model2.glb", (gltf) => {
  const model = gltf.scene;
  model.position.x = -position;
  scene.add(model);

  mouthUnder2 = model.getObjectByName("mouth_under_2");
  mouthOver2 = model.getObjectByName("mouth_over_2");

  console.log("target mouth:", mouthUnder2, mouthOver2);
});

/* =======================
   Animation
======================= */
function animate() {
  requestAnimationFrame(animate);

  if (mouthUnder && mouthOver && mouthUnder2 && mouthOver2) {

    // 現在の口
    const move = (f1Value * 1/140 - 145/7) * 0.001;
    mouthOver.position.y = move;
    mouthUnder.rotation.x = ((f1Value * 13/350 + 632/7) / 180) * Math.PI;

    // 目標の口
    const move2 = (targetData.f1 * 1/140 - 145/7) * 0.001;
    mouthOver2.position.y = move2;
    mouthUnder2.rotation.x =((targetData.f1 * 13/350 + 632/7) / 180) * Math.PI;
  }

  controls.update();
  renderer.render(scene, camera);

  // console.log(
  //   mouthCurrent,
  //   mouthOverCurrent,
  //   mouthTarget,
  //   mouthOverTarget
  // );
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
