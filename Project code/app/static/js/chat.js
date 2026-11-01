/* AI Chat — sign to compose, local Ollama to reply. No typing required. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var cam = $("cam"), snap = $("snap"), previewBox = $("previewBox"),
      compose = $("compose"), signHint = $("signHint"), thread = $("thread"),
      textForm = $("textForm"), textIn = $("textIn"), btnAsk = $("btnAsk");
  var history = [];
  var lastAskedText = "";

  fetch("/api/health", { cache: "no-store" })
    .then(function (r) { return r.json(); })
    .then(function (h) {
      if (h.model_exists) {
        $("healthPill").classList.add("ready");
        $("healthText").textContent = "Model ready · " + h.classes.length + " signs";
      } else {
        $("healthPill").classList.add("attention");
        $("healthText").textContent = "Model not trained";
      }
    })
    .catch(function () {
      $("healthPill").classList.add("attention");
      $("healthText").textContent = "API unreachable";
    });

  /* Trained word-signs that render as control characters, not literal text. */
  function toText(label) { return label === "SPACE" ? " " : label; }

  function bubble(role, text) {
    var d = document.createElement("div");
    d.className = "bubble " + role;
    d.textContent = text;
    thread.appendChild(d);
    thread.scrollTop = thread.scrollHeight;
    return d;
  }

  /* ---- camera mirror correction (same rule as main console) ----
     Flips preview + capture together so the model always sees
     exactly what you see. Enable ONLY if the preview looks mirrored vs
     the training window. Persisted in localStorage. */
  var mirrorFix = false;
  try { mirrorFix = localStorage.getItem("mirrorFix") === "1"; } catch (e) {}
  function applyMirror() {
    cam.style.transform = mirrorFix ? "scaleX(-1)" : "none";
    var b = $("btnMirror");
    if (b) {
      b.textContent = "Fix mirrored camera: " + (mirrorFix ? "On" : "Off");
      b.setAttribute("aria-pressed", mirrorFix ? "true" : "false");
    }
    try { localStorage.setItem("mirrorFix", mirrorFix ? "1" : "0"); } catch (e) {}
  }
  function drawCamToSnap() {
    var ctx = snap.getContext("2d");
    ctx.save();
    if (mirrorFix) { ctx.translate(snap.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(cam, 0, 0);
    ctx.restore();
  }
  var btnMirror = $("btnMirror");
  if (btnMirror) btnMirror.addEventListener("click", function () {
    mirrorFix = !mirrorFix;
    applyMirror();
  });
  applyMirror();

  $("btnStart").addEventListener("click", function () {
    var btn = this;
    btn.disabled = true;
    btn.textContent = "Requesting…";
    navigator.mediaDevices.getUserMedia({ video: { width: 640 } })
      .then(function (stream) {
        cam.srcObject = stream;
        applyMirror(); /* scaleX(-1) only when mirrorFix is on */
        previewBox.classList.add("live");
        btn.textContent = "Camera on";
      })
      .catch(function () {
        btn.disabled = false;
        btn.textContent = "Start camera";
        signHint.textContent = "Camera permission was denied by the browser.";
      });
  });

  $("btnSign").addEventListener("click", function () {
    if (!cam.srcObject) { signHint.textContent = "Start the camera first."; return; }
    var btn = this;
    btn.disabled = true;
    signHint.textContent = "Reading sign…";
    snap.width = cam.videoWidth;
    snap.height = cam.videoHeight;
    drawCamToSnap(); /* mirror-corrected capture: server sees what you see */
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: snap.toDataURL("image/jpeg", 0.85) })
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        btn.disabled = false;
        if (!res.ok) {
          signHint.textContent = res.body.error || "No hand detected — hold the sign steady.";
          return;
        }
        if (res.body.confidence >= 0.25) {
          if (res.body.label === "SEND") { if (btnAsk) btnAsk.click(); }
          else { compose.textContent += toText(res.body.label); }
          signHint.textContent = "Last capture: " + res.body.label +
            " (" + Math.round(res.body.confidence * 100) + "%)";
        } else {
          signHint.textContent = "Unsure (" + res.body.label + ") — hold the sign steady and retry.";
        }
      })
      .catch(function () {
        btn.disabled = false;
        signHint.textContent = "Request failed. Is the server running?";
      });
  });

  function doSpace() { compose.textContent += " "; }
  function doClear() {
    compose.textContent = "";
    signHint.textContent = "Last capture: —";
  }
  function bindBtn(id, fn) {
    var el = $(id);
    if (el) el.addEventListener("click", fn);
  }
  bindBtn("btnSpace", doSpace);
  bindBtn("btnClear", doClear);

  /* ---- live predict (letters commit when held steady) ---- */
  var btnLive = $("btnLive"), liveInd = $("liveInd");
  var liveOn = false, liveBusy = false;
  var pendingLabel = null, pendingCount = 0, missCount = 0, lastCommit = "", lastCommitTime = 0;
  var NEED = 1, COOL_MS = 1000; /* first confident read commits; 1s re-commit cooldown */

  function setLive(on) {
    liveOn = on;
    btnLive.textContent = on ? "Stop live" : "Go live";
    btnLive.setAttribute("aria-pressed", on ? "true" : "false");
    liveInd.hidden = !on;
    if (on) {
      pendingLabel = null; pendingCount = 0; missCount = 0;
      liveLoop();
    }
  }

  function liveLoop() {
    if (!liveOn) return;
    if (liveBusy || !cam.srcObject) { setTimeout(liveLoop, 800); return; }
    liveBusy = true;
    snap.width = cam.videoWidth;
    snap.height = cam.videoHeight;
    drawCamToSnap(); /* mirror-corrected capture: server sees what you see */
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: snap.toDataURL("image/jpeg", 0.7) })
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        liveBusy = false;
        if (!liveOn) return;
        var now = Date.now();
        if (res.ok) {
          var label = res.body.label;
          missCount = 0;
          pendingCount = (label === pendingLabel) ? pendingCount + 1 : 1;
          pendingLabel = label;
          signHint.textContent = "Live: " + label + " (" + Math.round(res.body.confidence * 100) + "%)" +
            (pendingCount < NEED ? " — hold steady (" + pendingCount + "/" + NEED + ")" : "");
          if (res.body.confidence >= 0.25 && pendingCount >= NEED &&
              label !== lastCommit && now - lastCommitTime > COOL_MS) {
            if (label === "SEND") { if (btnAsk) btnAsk.click(); }
            else { compose.textContent += toText(label); }
            lastCommit = label;
            lastCommitTime = now;
          }
        } else {
          // One bad frame must not wipe a nearly-confirmed sign.
          missCount += 1;
          if (missCount >= 2) { pendingLabel = null; pendingCount = 0; }
          // Show the server's real reason: "no hand" vs "not a trained sign".
          signHint.textContent = "Live: " + (res.body.error || "no hand — show your palm clearly.");
        }
        setTimeout(liveLoop, 800);
      })
      .catch(function () {
        liveBusy = false;
        if (liveOn) setTimeout(liveLoop, 1500);
      });
  }

  btnLive.addEventListener("click", function () {
    if (liveOn) { setLive(false); return; }
    if (cam.srcObject) { setLive(true); return; }
    navigator.mediaDevices.getUserMedia({ video: { width: 640 } })
      .then(function (stream) {
        cam.srcObject = stream;
        applyMirror(); /* scaleX(-1) only when mirrorFix is on */
        previewBox.classList.add("live");
        var b = $("btnStart");
        b.textContent = "Camera on";
        b.disabled = true;
        setLive(true);
      })
      .catch(function () { signHint.textContent = "Camera permission was denied by the browser."; });
  });

  function ask(message, fromDwell) {
    message = (message || "").trim();
    if (!message) return;
    if (fromDwell && message === lastAskedText) return;
    bubble("user", message);
    history.push({ role: "user", text: message });
    btnAsk.disabled = true;
    var typing = bubble("system", "Thinking…");
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message, history: history.slice(-6) })
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        btnAsk.disabled = false;
        typing.remove();
        if (!res.ok) {
          bubble("system", res.body.error || "Model request failed.");
          return;
        }
        bubble("assistant", res.body.reply);
        history.push({ role: "assistant", text: res.body.reply });
        lastAskedText = message;
        compose.textContent = "";
      })
      .catch(function () {
        btnAsk.disabled = false;
        typing.remove();
        bubble("system", "Request failed. Is the server running?");
      });
  }

  btnAsk.addEventListener("click", function () { ask(compose.textContent); });
  textForm.addEventListener("submit", function (e) {
    e.preventDefault();
    ask(textIn.value);
    textIn.value = "";
  });
})();
