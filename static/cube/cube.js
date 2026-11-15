/* Rubik's Cipher front end: Three.js cube, move animation, fixed-camera export, encode/decode UI. */
(() => {
  const CFG = window.CUBE_CONFIG;
  const FACES = "URFDLB";
  // face -> [axis (0=x,1=y,2=z), layer coordinate, quarter turns for a clockwise move]  (same as cube.py)
  const AXIS = { U: [1, 1, -1], D: [1, -1, 1], R: [0, 1, -1], L: [0, -1, 1], F: [2, 1, -1], B: [2, -1, 1] };

  // facelet index -> {p: grid position, n: outward normal}   (same formula as cube.py)
  function faceletGeometry() {
    const geo = [];
    for (const face of FACES) for (let k = 0; k < 9; k++) {
      const row = Math.floor(k / 3), col = k % 3; let p, n;
      switch (face) {
        case "U": p = [col - 1, 1, row - 1]; n = [0, 1, 0]; break;
        case "D": p = [col - 1, -1, 1 - row]; n = [0, -1, 0]; break;
        case "F": p = [col - 1, 1 - row, 1]; n = [0, 0, 1]; break;
        case "B": p = [1 - col, 1 - row, -1]; n = [0, 0, -1]; break;
        case "R": p = [1, 1 - row, 1 - col]; n = [1, 0, 0]; break;
        default:  p = [-1, 1 - row, col - 1]; n = [-1, 0, 0];
      }
      geo.push({ p, n });
    }
    return geo;
  }
  const GEO = faceletGeometry();
  const GEO_INDEX = new Map(GEO.map((g, i) => [g.p.join() + "|" + g.n.join(), i]));

  // ------------------------------------------------------------------ scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(CFG.background);
  const viewGroup = new THREE.Group();   // drag-to-rotate
  const cubeGroup = new THREE.Group();   // the 26 cubies
  viewGroup.add(cubeGroup); scene.add(viewGroup);

  const S = CFG.cubie, T = CFG.sticker;
  const bodyGeo = new THREE.BoxGeometry(S, S, S);
  const bodyMat = new THREE.MeshBasicMaterial({ color: 0x161616 });
  const shape = new THREE.Shape(); { // rounded square sticker
    const h = T / 2, r = T * 0.12;
    shape.moveTo(-h + r, -h); shape.lineTo(h - r, -h); shape.quadraticCurveTo(h, -h, h, -h + r);
    shape.lineTo(h, h - r); shape.quadraticCurveTo(h, h, h - r, h); shape.lineTo(-h + r, h);
    shape.quadraticCurveTo(-h, h, -h, h - r); shape.lineTo(-h, -h + r); shape.quadraticCurveTo(-h, -h, -h + r, -h);
  }
  const stickerGeo = new THREE.ShapeGeometry(shape, 6);
  const NORMALS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const cubies = [], stickers = new Array(54);
  for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
    if (!x && !y && !z) continue;
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.set(x * S, y * S, z * S);
    body.userData.home = body.position.clone();
    for (const n of NORMALS) {
      if ([x, y, z][n.findIndex(v => v)] !== n.find(v => v)) continue;   // outer face only
      const st = new THREE.Mesh(stickerGeo, new THREE.MeshBasicMaterial({ color: 0xeeeeee }));
      st.position.set(n[0], n[1], n[2]).multiplyScalar(S / 2 + 0.002);
      if (n[0] === 1) st.rotation.y = Math.PI / 2; else if (n[0] === -1) st.rotation.y = -Math.PI / 2;
      else if (n[1] === 1) st.rotation.x = -Math.PI / 2; else if (n[1] === -1) st.rotation.x = Math.PI / 2;
      else if (n[2] === -1) st.rotation.y = Math.PI;
      stickers[GEO_INDEX.get([x, y, z].join() + "|" + n.join())] = st;
      body.add(st);
    }
    cubies.push(body); cubeGroup.add(body);
  }

  // soft shadow (hidden in exports)
  const shadow = (() => {
    const c = document.createElement("canvas"); c.width = c.height = 256;
    const g = c.getContext("2d"), grd = g.createRadialGradient(128, 128, 10, 128, 128, 128);
    grd.addColorStop(0, "rgba(0,0,0,0.16)"); grd.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grd; g.fillRect(0, 0, 256, 256);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(5.6, 5.6), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
    m.rotation.x = -Math.PI / 2; m.position.y = -2.2; m.scale.set(1, 0.55, 1); scene.add(m); return m;
  })();

  let busy = false;
  function setFacelets(f) { for (let i = 0; i < 54; i++) stickers[i].material.color.set(CFG.palette[f[i]]); }
  function resetCube() { cubies.forEach(c => { cubeGroup.attach(c); c.position.copy(c.userData.home); c.quaternion.identity(); }); }
  function snap(c) {
    c.updateMatrix(); const e = c.matrix.elements; for (let i = 0; i < 16; i++) e[i] = Math.round(e[i]);
    c.matrix.decompose(c.position, c.quaternion, c.scale); c.scale.set(1, 1, 1);
  }
  function doMove(move, ms) {
    return new Promise(res => {
      const [axis, layer, q] = AXIS[move[0]];
      let turns = q; if (move.endsWith("2")) turns *= 2; else if (move.endsWith("'")) turns *= -1;
      const angle = turns * Math.PI / 2, ax = new THREE.Vector3(+(axis === 0), +(axis === 1), +(axis === 2));
      const sel = cubies.filter(c => Math.round(c.position.getComponent(axis) / S) === layer);
      const pivot = new THREE.Group(); cubeGroup.add(pivot); sel.forEach(c => pivot.attach(c));
      const t0 = performance.now();
      (function step(now) {
        const t = Math.min(1, (now - t0) / ms), e = t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
        pivot.setRotationFromAxisAngle(ax, angle * e);
        if (t < 1) return requestAnimationFrame(step);
        pivot.updateMatrixWorld(true); sel.forEach(c => { cubeGroup.attach(c); snap(c); }); cubeGroup.remove(pivot); res();
      })(t0);
    });
  }
  let playToken = 0;
  async function play(moves, ms, onStep) {
    const token = ++playToken;
    busy = true;
    $("status").classList.add("live");                // bigger move-by-move line while the cube turns
    for (let i = 0; i < moves.length; i++) {
      await doMove(moves[i], ms);
      if (token !== playToken) return false;          // cancelled by New words
      if (onStep) onStep(i + 1, moves.length);
    }
    $("status").classList.remove("live");
    busy = false;
    return true;
  }
  function cancelPlay() { playToken++; busy = false; $("status").classList.remove("live"); }

  // fixed-camera export (must match cube_vision.py)
  const exportRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  exportRenderer.setPixelRatio(1); exportRenderer.setSize(CFG.size, CFG.size);
  function renderView(view) {
    const d = CFG.views[view], h = CFG.half;
    const cam = new THREE.OrthographicCamera(-h, h, h, -h, 0.1, 100);
    cam.position.set(d[0] * 10, d[1] * 10, d[2] * 10); cam.up.set(0, 1, 0); cam.lookAt(0, 0, 0); cam.updateMatrixWorld(true);
    const saved = viewGroup.quaternion.clone(); viewGroup.quaternion.identity(); shadow.visible = false; viewGroup.updateMatrixWorld(true);
    exportRenderer.render(scene, cam);
    const url = exportRenderer.domElement.toDataURL("image/png");
    viewGroup.quaternion.copy(saved); shadow.visible = true;
    return url;
  }

  // ------------------------------------------------------------------ render-only test harness
  if (window.RENDER_ONLY) {
    setFacelets(window.RENDER_ONLY.facelets);
    document.getElementById("render-only").appendChild(exportRenderer.domElement);
    renderView(window.RENDER_ONLY.view);
    return;
  }

  // ------------------------------------------------------------------ interactive view
  const canvas = document.getElementById("cube");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(5.6, 4.8, 7.1); camera.lookAt(0, -0.3, 0);
  function resize() { const w = canvas.clientWidth || 560; renderer.setSize(w, w, false); }
  window.addEventListener("resize", resize); resize();
  (function loop() { requestAnimationFrame(loop); renderer.render(scene, camera); })();

  let drag = null;
  canvas.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    flipAnim = null;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY];
    const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dx * 0.008);
    const qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), dy * 0.008);
    viewGroup.quaternion.premultiply(qx.multiply(qy));
  });
  canvas.addEventListener("pointerup", () => drag = null); canvas.addEventListener("pointercancel", () => drag = null);
  canvas.addEventListener("dblclick", () => flipTo(false));

  // "Other side": rotate the cube so the interactive camera sees what a camera on the opposite side would see
  const FLIP_Q = (() => {
    const up = new THREE.Vector3(0, 1, 0), target = new THREE.Vector3(0, -0.3, 0);
    const a = new THREE.Matrix4().lookAt(camera.position, target, up);
    const b = new THREE.Matrix4().lookAt(camera.position.clone().negate(), target.clone().negate(), up);
    return new THREE.Quaternion().setFromRotationMatrix(a).multiply(new THREE.Quaternion().setFromRotationMatrix(b).invert());
  })();
  let flipped = false, flipAnim = null;
  function flipTo(back) {
    flipped = back;
    const from = viewGroup.quaternion.clone(), to = back ? FLIP_Q.clone() : new THREE.Quaternion();
    const t0 = performance.now(), ms = 450, token = flipAnim = {};
    (function step(now) {
      if (flipAnim !== token) return;
      const t = Math.min(1, (now - t0) / ms), e = t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      THREE.Quaternion.slerp(from, to, viewGroup.quaternion, e);
      if (t < 1) requestAnimationFrame(step);
    })(t0);
    const btn = document.getElementById("flip"); if (btn) btn.textContent = back ? "Front side" : "Other side";
  }
  document.getElementById("flip").addEventListener("click", () => flipTo(!flipped));

  // ------------------------------------------------------------------ UI
  const $ = id => document.getElementById(id);
  const wordEls = [...document.querySelectorAll("#words span")];
  const MOVE_MS = 260;
  let state = { scramble: null, pngs: null };
  const status = (msg, err) => { const live = $("status").classList.contains("live"); $("status").textContent = msg; $("status").className = "hint" + (err ? " err" : "") + (live && !err ? " live" : ""); };
  const detail = msg => { $("detail").textContent = msg; };
  const clearWords = () => wordEls.forEach(el => { el.textContent = ""; el.className = ""; });
  const showWord = (i, w, tier) => {
    wordEls[i].textContent = w;
    if (tier) { const s = document.createElement("small"); s.textContent = tier; wordEls[i].appendChild(s); }
    wordEls[i].style.background = ["#f0a9aa", "#eee59d", "#96c6e9"][i]; requestAnimationFrame(() => wordEls[i].className = "show");
  };
  const tierLine = (words, tiers) => words.map((w, i) => `${w} · ${tiers[i]}`).join("   ");

  const has = id => !!$(id);
  const on = (id, ev, fn) => { if (has(id)) $(id).addEventListener(ev, fn); };
  function setMode(mode) {
    if (has("tab-encode")) { $("tab-encode").setAttribute("aria-pressed", mode === "encode"); $("tab-decode").setAttribute("aria-pressed", mode === "decode"); }
    if (has("panel-encode")) $("panel-encode").hidden = mode !== "encode";
    if (has("panel-decode")) $("panel-decode").hidden = mode !== "decode";
    clearWords(); detail("");
    status(mode === "encode" ? "type any three English words (the list comes first, then a 370k-word dictionary), or press Random" : "load the two exported images, then Decode");
  }
  on("tab-encode", "click", () => setMode("encode")); on("tab-decode", "click", () => setMode("decode"));

  on("random", "click", async () => { const r = await (await fetch("/random")).json(); ["w1", "w2", "w3"].forEach((id, i) => $(id).value = r.words[i]); });
  ["w1", "w2", "w3"].forEach(id => on(id, "keydown", e => { if (e.key === "Enter") $("encode").click(); }));

  on("encode", "click", async () => {
    if (busy) return;
    const words = ["w1", "w2", "w3"].map(id => $(id).value.trim());
    clearWords(); detail(""); $("thumbs").innerHTML = ""; $("export").disabled = $("replay").disabled = true;
    const r = await fetch("/encode", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ words }) });
    const res = await r.json();
    if (!r.ok) return status(res.error, true);
    state.scramble = res.scramble; state.facelets = res.facelets;
    resetCube(); setFacelets(CFG.solved);
    status(`scrambling: ${res.scramble.length} moves`);
    if (!await play(res.scramble, MOVE_MS, (i, n) => status(`scrambling: ${res.scramble.slice(0, i).join(" ")}`))) return;
    status(`${res.scramble.length} moves. The cube now says "${words.join(" ")}". Export it.`);
    detail(`${tierLine(res.words, res.tiers)}\n${CFG.data_bits} bits: ${res.bits}\nscramble (${res.scramble.length} moves): ${res.scramble.join(" ")}`);
    $("export").disabled = $("replay").disabled = false;
  });

  on("clear", "click", () => {
    cancelPlay();
    state = { scramble: null, pngs: null };
    ["w1", "w2", "w3"].forEach(id => $(id).value = "");
    resetCube(); setFacelets(CFG.solved); flipTo(false);
    clearWords(); detail(""); $("thumbs").innerHTML = "";
    $("export").disabled = $("replay").disabled = true;
    status("type three new words, or press Random");
    $("w1").focus();
  });

  on("replay", "click", async () => {
    if (busy || !state.scramble) return;
    resetCube(); setFacelets(CFG.solved);
    if (!await play(state.scramble, MOVE_MS, (i) => status(`scrambling: ${state.scramble.slice(0, i).join(" ")}`))) return;
    status(`${state.scramble.length} moves. Export it.`);
  });

  on("export", "click", () => {
    if (busy) return;
    const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
    const pngs = ["A", "B"].map(v => ({ view: v, name: `cube_${stamp}_${v}.png`, url: renderView(v) }));
    $("thumbs").innerHTML = pngs.map(p => `<figure><img src="${p.url}" alt="view ${p.view}"><a class="btn" download="${p.name}" href="${p.url}">Save ${p.view}</a></figure>`).join("");
    pngs.forEach((p, i) => setTimeout(() => { const a = document.createElement("a"); a.href = p.url; a.download = p.name; document.body.appendChild(a); a.click(); a.remove(); }, i * 400));
    status("two images saved. Only the pictures travel; the words are in the stickers.");
  });

  let chosen = [];
  on("files", "change", e => {
    chosen = [...e.target.files].slice(0, 2);
    $("thumbs-in").innerHTML = "";
    chosen.forEach(f => { const img = document.createElement("img"); img.src = URL.createObjectURL(f); const fig = document.createElement("figure"); fig.appendChild(img); fig.append(f.name); $("thumbs-in").appendChild(fig); });
    $("decode").disabled = chosen.length !== 2;
    status(chosen.length === 2 ? "ready to decode" : "choose both images (A and B)");
  });

  async function playResult(res) {
    resetCube(); setFacelets(res.facelets);
    const n = res.solve.length, marks = [1, 2, 3].map(k => Math.ceil(n * k / 3));
    status(`valid cube. solving: ${n} moves`);
    if (n === 0) res.words.forEach((w, i) => showWord(i, w, res.tiers[i]));
    await play(res.solve, MOVE_MS, (i) => {
      status(`solving: ${res.solve.slice(0, i).join(" ")}`);
      marks.forEach((m, k) => { if (i === m && !wordEls[k].textContent) showWord(k, res.words[k], res.tiers[k]); });
    });
    status(`solved. "${res.words.join(" ")}"`);
    detail(`${tierLine(res.words, res.tiers)}\n${CFG.data_bits} bits: ${res.bits}`);
  }

  on("decode", "click", async () => {
    if (busy || chosen.length !== 2) return;
    clearWords(); detail("");
    const fd = new FormData(); chosen.forEach((f, i) => fd.append("img" + i, f));
    status("reading stickers…");
    const r = await fetch("/decode", { method: "POST", body: fd });
    const res = await r.json();
    if (!r.ok) return status(res.error, true);
    await playResult(res);
  });

  // ---- phone camera: photograph the encoder's screen (front view, then Other side)
  const video = $("video"), overlay = $("overlay");
  let stream = null, camOn = false, shots = [];
  async function startCam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return status("the camera needs HTTPS: open the https:// address", true);
    try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 1280 } }, audio: false }); }
    catch (e) { return status("camera not available: " + e.message, true); }
    video.srcObject = stream; await video.play();
    $("cam").hidden = false; $("cambar").hidden = false; camOn = true; shots = []; renderShots(); clearWords(); detail("");
    status("Point the phone at the encoder screen. The shutter unlocks when the cube is detected.");
    detectLoop();
  }
  function stopCam() { camOn = false; if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; $("cam").hidden = true; $("cambar").hidden = true; }
  function grab(maxW, q) {
    const vw = video.videoWidth || 1280, vh = video.videoHeight || 960;
    const c = document.createElement("canvas"), s = Math.min(1, maxW / vw);
    c.width = Math.round(vw * s); c.height = Math.round(vh * s);
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    return new Promise((resolve, reject) => {
      try { c.toBlob(b => b ? resolve({ blob: b, w: c.width, h: c.height }) : reject(new Error("empty frame")), "image/jpeg", q); }
      catch (e) { reject(e); }
    });
  }
  function drawHex(hex, w, h) {
    const cw = overlay.clientWidth, ch = overlay.clientHeight; overlay.width = cw; overlay.height = ch;
    const g = overlay.getContext("2d"); g.clearRect(0, 0, cw, ch);
    if (!hex) return;
    const s = Math.min(cw / w, ch / h), ox = (cw - w * s) / 2, oy = (ch - h * s) / 2;
    g.beginPath(); hex.forEach((p, i) => g[i ? "lineTo" : "moveTo"](ox + p[0] * s, oy + p[1] * s)); g.closePath();
    g.lineWidth = 4; g.strokeStyle = "#4cd964"; g.stroke();
  }
  async function detectLoop() {
    while (camOn) {
      try {
        const { blob, w, h } = await grab(640, 0.7);
        const r = await fetch("/detect", { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
        const res = await r.json();
        drawHex(res.hex, w, h);
        $("camstate").textContent = res.hex ? "Cube detected" : "Looking for the cube…";
        $("camstate").classList.toggle("ok", !!res.hex);
        $("shot").disabled = !res.hex || shots.length >= 2;
      } catch (e) { /* keep looking */ }
      await new Promise(r => setTimeout(r, 250));
    }
  }
  function renderShots() {
    $("camdecode").disabled = shots.length !== 2;
    $("shot").disabled = shots.length >= 2;
    $("shot").textContent = shots.length >= 2 ? "2 photos taken" : `Take photo ${shots.length + 1} of 2`;
    try {
      $("shots").innerHTML = "";
      shots.forEach((b, i) => { const img = document.createElement("img"); img.alt = `photo ${i + 1}`; img.src = URL.createObjectURL(b); $("shots").appendChild(img); });
    } catch (e) { /* thumbnails are optional */ }
  }
  async function decodeShots() {
    if (busy || shots.length !== 2) return;
    const fd = new FormData(); shots.forEach((b, i) => fd.append("photo" + i, b, `photo${i}.jpg`));
    status("reading the cube from the photos…");
    let res, ok;
    try { const r = await fetch("/photo_decode", { method: "POST", body: fd }); ok = r.ok; res = await r.json(); }
    catch (e) { shots = []; renderShots(); return status("upload failed: " + e.message + " (try again)", true); }
    if (!ok) { shots = []; renderShots(); return status(res.error + " (try again)", true); }
    stopCam(); await playResult(res);
  }
  on("restart", "click", () => {
    cancelPlay(); stopCam();
    shots = []; renderShots();
    chosen = []; $("files").value = ""; $("thumbs-in").innerHTML = ""; $("decode").disabled = true;
    resetCube(); setFacelets(CFG.solved); flipTo(false);
    clearWords(); detail("");
    status("load the two exported images, or press Use camera");
  });
  on("usecam", "click", startCam);
  on("camclose", "click", () => { stopCam(); status("camera closed"); });
  on("shot", "click", async () => {
    if (shots.length >= 2) return;
    let shot;
    try { shot = await grab(1600, 0.92); }
    catch (e) { return status("could not capture a frame: " + e.message, true); }
    shots.push(shot.blob); renderShots();
    if (shots.length === 1) status("Got photo 1. Press Other side on the encoder, then take photo 2.");
    else { status("Two photos taken. Decoding…"); decodeShots(); }
  });
  on("camdecode", "click", decodeShots);

  resetCube(); setFacelets(CFG.solved); setMode(CFG.mode === "decode" ? "decode" : "encode");
  { // test hooks: #f=<facelets>&back  -> show a state, optionally jump straight to the other side
    const hp = new URLSearchParams(location.hash.slice(1));
    if (hp.get("f") && hp.get("f").length === 54) setFacelets(hp.get("f"));
    if (hp.has("back")) { flipTo(true); flipAnim = null; viewGroup.quaternion.copy(FLIP_Q); }
  }
})();
