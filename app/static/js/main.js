/* SignLens console behaviour — plain, dependency-free. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var cam = $("cam"), snap = $("snap"), previewBox = $("previewBox"),
      resultLabel = $("resultLabel"), resultConf = $("resultConf"),
      resultSub = $("resultSub"), meterFill = $("meterFill"),
      sentence = $("sentence"), signSeq = $("signSeq"),
      healthPill = $("healthPill"), healthText = $("healthText"),
      btnCapture = $("btnCapture"), convertForm = $("convertForm");

  /* ---- service status (8s timeout + 3 retries, never stuck) ---- */
  function checkHealth(attempt) {
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, 8000);
    fetch("/api/health", { signal: ctrl.signal, cache: "no-store" })
      .then(function (r) { clearTimeout(timer); return r.json(); })
      .then(function (h) {
        $("metaModel").textContent = h.model_exists ? "CNN-LSTM-RF" : "Not trained";
        $("metaVocab").textContent = h.model_exists ? h.classes.length + " signs" : "—";
        if (h.model_exists) {
          healthPill.classList.add("ready");
          healthText.textContent = "Model ready · " + h.classes.length + " signs";
        } else {
          healthPill.classList.add("attention");
          healthText.textContent = "Model not trained";
        }
      })
      .catch(function () {
        clearTimeout(timer);
        if (attempt < 3) {
          healthText.textContent = "Retrying… (" + attempt + "/3)";
          setTimeout(function () { checkHealth(attempt + 1); }, 3000);
        } else {
          healthPill.classList.add("attention");
          healthText.textContent = "API unreachable — is the server running?";
        }
      });
  }
  checkHealth(1);

  /* ---- camera mirror correction ----
     Some webcam/driver+browser combos deliver a mirrored stream while the
     OpenCV training window shows true camera (or vice versa). This toggle
     flips preview + capture together so the model always sees exactly what
     you see. Enable ONLY if the preview looks mirrored/reversed compared
     to the training window. Persisted in localStorage. */
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
    /* Draws the video frame into the hidden snap canvas, un-mirroring first
       when mirrorFix is on so the server gets true-camera bytes. */
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
        applyMirror(); /* non-mirror by default; scaleX(-1) only when mirrorFix is on */
        previewBox.classList.add("live");
        btn.textContent = "Camera on";
      })
      .catch(function () {
        btn.disabled = false;
        btn.textContent = "Start camera";
        setError("Camera permission was denied by the browser.");
      });
  });

  /* Trained word-signs that render as control characters, not literal text. */
  function toText(label) { return label === "SPACE" ? " " : label; }

  function setBusy(busy) {
    btnCapture.disabled = busy;
    btnCapture.textContent = busy ? "Predicting…" : "Capture & predict";
  }

  function setError(msg) {
    resultLabel.textContent = "No hand detected";
    resultLabel.classList.add("idle");
    resultConf.textContent = "";
    meterFill.style.width = "0";
    resultSub.textContent = msg;
    resultSub.classList.add("error");
  }

  function setNoSign(msg) {
    resultLabel.textContent = "No sign";
    resultLabel.classList.add("idle");
    resultConf.textContent = "";
    meterFill.style.width = "0";
    resultSub.textContent = msg;
    resultSub.classList.remove("error");
  }

  function setPrediction(label, confidence, smoothed) {
    resultLabel.textContent = label;
    resultLabel.classList.remove("idle");
    var pct = Math.round(confidence * 100);
    resultConf.textContent = pct + "% confidence";
    meterFill.style.width = pct + "%";
    resultSub.classList.remove("error");
    /* Neutral message — callers overwrite with the real outcome
       (appended / hold-steady / sent) so the panel never lies. */
    resultSub.textContent = smoothed && smoothed !== label
      ? "Smoothed output over recent frames: " + smoothed + "."
      : "Prediction accepted.";
  }

  function predictFromDataURL(dataURL) {
    setBusy(true);
    resultSub.classList.remove("error");
    resultSub.textContent = "Analysing landmarks…";
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL })
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        setBusy(false);
        if (!res.ok) { setNoSign(res.body.error || "Prediction failed."); return; }
        setPrediction(res.body.label, res.body.confidence, res.body.smoothed);
        if (res.body.confidence >= 0.55) {
          if (res.body.label === "SEND") { if (btnSendAlert) btnSendAlert.click(); }
          else { sentence.textContent += toText(res.body.label); }
          resultSub.textContent = res.body.label === "SEND"
            ? "Sent."
            : "Prediction accepted. Appended to the composed text below.";
          queueAutoSpeak();
          queueTranslate();
          queueAutoAlert(res.body.label);
        }
      })
      .catch(function () { setBusy(false); setError("Request failed. Is the Flask server running?"); });
  }

  btnCapture.addEventListener("click", function () {
    if (!cam.srcObject) { setError("Start the camera first, or upload a photo."); return; }
    snap.width = cam.videoWidth;
    snap.height = cam.videoHeight;
    drawCamToSnap(); /* mirror-corrected capture: server sees what you see */
    predictFromDataURL(snap.toDataURL("image/jpeg", 0.85));
  });

  /* ---- real-time live prediction (no capture needed) ---- */
  var btnLive = $("btnLive"), liveInd = $("liveInd");
  var liveOn = false, liveBusy = false;
  var pendingLabel = null, pendingCount = 0, missCount = 0, lastCommit = "", lastCommitTime = 0;
  var NEED = 1, COOL_MS = 1000; /* first confident read commits; 1s re-commit cooldown */

  function setLive(on) {
    liveOn = on;
    btnLive.textContent = on ? "Stop live" : "Go live";
    btnLive.setAttribute("aria-pressed", on ? "true" : "false");
    liveInd.hidden = !on;
    if (on) { pendingLabel = null; pendingCount = 0; missCount = 0; liveLoop(); }
  }

  function ensureCamera() {
    if (cam.srcObject) return Promise.resolve();
    return navigator.mediaDevices.getUserMedia({ video: { width: 640 } })
      .then(function (stream) {
        cam.srcObject = stream;
        applyMirror(); /* non-mirror by default; scaleX(-1) only when mirrorFix is on */
        previewBox.classList.add("live");
        var b = $("btnStart");
        b.textContent = "Camera on";
        b.disabled = true;
      });
  }

  function snapDataURL() {
    snap.width = cam.videoWidth;
    snap.height = cam.videoHeight;
    drawCamToSnap(); /* mirror-corrected capture: server sees what you see */
    return snap.toDataURL("image/jpeg", 0.7);
  }

  function liveLoop() {
    if (!liveOn) return;
    if (liveBusy || !cam.srcObject) { setTimeout(liveLoop, 700); return; }
    liveBusy = true;
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: snapDataURL() })
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        liveBusy = false;
        if (!liveOn) return;
        if (res.ok) {
          setPrediction(res.body.label, res.body.confidence, res.body.smoothed);
          var label = res.body.label, now = Date.now();
          missCount = 0;
          pendingCount = (label === pendingLabel) ? pendingCount + 1 : 1;
          pendingLabel = label;
          // Commit to the sentence only once a sign holds steady (debounced).
          // A single dropped frame must not restart the count.
          if (res.body.confidence >= 0.55 && pendingCount >= NEED &&
              label !== lastCommit && now - lastCommitTime > COOL_MS) {
            if (label === "SEND") {
              if (btnSendAlert) btnSendAlert.click();
              resultSub.textContent = "Sent.";
            } else {
              sentence.textContent += toText(label);
              resultSub.textContent = "Prediction accepted. Appended to the composed text below.";
            }
            lastCommit = label;
            lastCommitTime = now;
            queueAutoSpeak();
            queueTranslate();
            queueAutoAlert(label);
          } else if (res.body.confidence >= 0.55 && pendingCount < NEED) {
            resultSub.textContent = (label === "SEND"
              ? "Seen SEND — hold steady to send (" + pendingCount + "/" + NEED + ")."
              : "Seen " + label + " — hold steady to add it (" + pendingCount + "/" + NEED + ").");
            resultSub.classList.remove("error");
          } else if (res.body.confidence >= 0.55) {
            /* Accepted but cooling down after a recent commit — nothing added yet. */
            resultSub.textContent = "Prediction accepted — adding again in a moment…";
            resultSub.classList.remove("error");
          }
        } else {
          missCount += 1;
          if (missCount >= 2) { pendingLabel = null; pendingCount = 0; }
          setNoSign((res.body.error || "No hand in frame") + " — hold the sign steady.");
        }
        setTimeout(liveLoop, 700);
      })
      .catch(function () {
        liveBusy = false;
        if (liveOn) setTimeout(liveLoop, 1500);
      });
  }

  btnLive.addEventListener("click", function () {
    if (liveOn) { setLive(false); return; }
    ensureCamera().then(function () { setLive(true); })
      .catch(function () { setError("Camera permission was denied by the browser."); });
  });

  $("file").addEventListener("change", function (e) {
    var f = e.target.files[0];
    if (!f) return;
    var reader = new FileReader();
    reader.onload = function () { predictFromDataURL(reader.result); };
    reader.readAsDataURL(f);
    e.target.value = "";
  });

  $("btnClear").addEventListener("click", function () {
    sentence.textContent = "";
    translateSentence._last = "";
    resultLabel.textContent = "No prediction yet";
    resultLabel.classList.add("idle");
    resultConf.textContent = "";
    meterFill.style.width = "0";
    resultSub.textContent = "Predictions appear here with a confidence score.";
    if (alertStatusMsg) alertStatusMsg.hidden = true;
  });

  function speakNow() {
    var text = sentence.textContent.trim();
    if (!text || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
  $("btnSpeak").addEventListener("click", speakNow);

  /* ---- auto-speak + Hindi/Marathi translation of the composed text ---- */
  var btnAuto = $("btnAuto"), btnSpace = $("btnSpace"),
      i18n = $("i18n"), trHi = $("trHi"), trMr = $("trMr");
  var autoOn = true, speakTimer = null, trTimer = null;

  function queueAutoSpeak() {
    if (!autoOn) return;
    clearTimeout(speakTimer);
    speakTimer = setTimeout(speakNow, 1200);
  }
  function queueTranslate() {
    clearTimeout(trTimer);
    trTimer = setTimeout(translateSentence, 800);
  }
  function translateSentence() {
    var t = sentence.textContent.trim();
    if (!t) { i18n.hidden = true; return; }
    // Skip repeat work: same text already translated.
    if (t === translateSentence._last) return;
    translateSentence._last = t;
    i18n.hidden = false;
    trHi.textContent = "Translating…";
    trMr.textContent = "Translating…";
    fetchOne("hi", trHi, t);
    fetchOne("mr", trMr, t);
  }
  function fetchOne(lang, el, text) {
    var reqId = (fetchOne._seq = (fetchOne._seq || 0) + 1);
    el.textContent = "Translating…";
    var ctrl = new AbortController();
    var timer = setTimeout(function () { ctrl.abort(); }, 25000);
    fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, target: lang }),
      signal: ctrl.signal
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        clearTimeout(timer);
        // Drop stale responses when the user signed more text meanwhile.
        if (sentence.textContent.trim() !== text) return;
        if (res.ok && res.body.translated) {
          el.textContent = res.body.translated;
          el.style.cursor = "";
          el.onclick = null;
        } else {
          el.textContent = "Translation failed — click to retry.";
          el.style.cursor = "pointer";
          el.onclick = function () { translateSentence._last = ""; translateSentence(); };
        }
        void reqId;
      })
      .catch(function () {
        clearTimeout(timer);
        if (sentence.textContent.trim() !== text) return;
        el.textContent = "Offline — click to retry.";
        el.style.cursor = "pointer";
        el.onclick = function () { translateSentence._last = ""; translateSentence(); };
      });
  }
  btnAuto.addEventListener("click", function () {
    autoOn = !autoOn;
    btnAuto.textContent = autoOn ? "Auto-speak: On" : "Auto-speak: Off";
    btnAuto.setAttribute("aria-pressed", autoOn ? "true" : "false");
    if (!autoOn) {
      clearTimeout(speakTimer);
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    }
  });
  btnSpace.addEventListener("click", function () {
    sentence.textContent += " ";
    queueTranslate();
  });
  $("btnClear").addEventListener("click", function () {
    clearTimeout(speakTimer);
    clearTimeout(trTimer);
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    i18n.hidden = true;
  });

  /* ---- text to sign ---- */
  function renderSequence(items) {
    signSeq.innerHTML = "";
    seqItems = [];
    seqCards = [];
    stopPlay();
    if (!items.length) {
      signSeq.innerHTML = '<div class="sign-empty"><p>Nothing to preview yet</p>' +
        "<span>Converted characters will be listed here as reference cards.</span></div>";
      player.hidden = true;
      return;
    }
    items.forEach(function (s, i) {
      if (s.word_gap) {
        var gap = document.createElement("div");
        gap.className = "sign-gap";
        signSeq.appendChild(gap);
        return;
      }
      var fig = document.createElement("figure");
      fig.className = "sign-card";
      fig.style.animationDelay = (i * 30) + "ms";
      var img = document.createElement("img");
      img.src = s.sign_image;
      img.alt = "Sign for " + s.char;
      img.loading = "eager";
      var cap = document.createElement("figcaption");
      cap.textContent = s.char;
      fig.appendChild(img);
      fig.appendChild(cap);
      signSeq.appendChild(fig);
      seqCards.push(fig);
      seqItems.push(s);
    });
    if (seqItems.length) {
      player.hidden = false;
      showSign(0);
      startPlay();
      try {
        player.scrollIntoView({ behavior: calmMotion() ? "auto" : "smooth", block: "nearest" });
      } catch (e) {}
    }
  }

  convertForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = $("txt").value;
    if (!text.trim()) return;
    var btn = $("btnConvert");
    btn.disabled = true;
    fetch("/api/text-to-sign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        renderSequence(j.sequence || []);
        showWord(j.text || text, (j.sequence || []).filter(function (s) { return !s.word_gap; }).length);
      })
      .catch(function () { renderSequence([]); showWord("", 0); })
      .then(function () { btn.disabled = false; });
  });

  function showWord(text, n) {
    var pill = $("signWord");
    if (!text || !text.trim() || !n) { pill.hidden = true; return; }
    pill.hidden = false;
    pill.textContent = "Spelling \u201C" + text.trim().toUpperCase() + "\u201D · " + n + (n === 1 ? " sign" : " signs");
  }
  /* ---- sign player: the phrase plays like a video ---- */
  var player = $("player"), playerImg = $("playerImg"), playerChar = $("playerChar"),
      playerCount = $("playerCount"), playerFill = $("playerFill"),
      playerPlay = $("playerPlay"), playerPrev = $("playerPrev"), playerNext = $("playerNext");
  var seqItems = [], seqCards = [], playIdx = 0, playTimer = null;

  function calmMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }
  function setPlayUI(playing) {
    playerPlay.textContent = playing ? "❚❚ Pause" : "▶ Play";
  }
  function stopPlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    if (playerPlay) setPlayUI(false);
  }
  function showSign(i) {
    if (!seqItems.length) return;
    playIdx = ((i % seqItems.length) + seqItems.length) % seqItems.length;
    var s = seqItems[playIdx];
    playerImg.src = s.sign_image;
    playerImg.alt = "Sign for " + s.char;
    playerChar.textContent = s.char;
    playerCount.textContent = (playIdx + 1) + " / " + seqItems.length;
    playerFill.style.width = ((playIdx + 1) / seqItems.length * 100) + "%";
    seqCards.forEach(function (c, j) { c.classList.toggle("active", j === playIdx); });
  }
  function startPlay() {
    if (!seqItems.length) return;
    stopPlay();
    setPlayUI(true);
    playTimer = setInterval(function () { showSign(playIdx + 1); }, 900);
  }
  playerPlay.addEventListener("click", function () {
    if (!seqItems.length) return;
    if (playTimer) stopPlay(); else startPlay();
  });
  playerPrev.addEventListener("click", function () { showSign(playIdx - 1); });
  playerNext.addEventListener("click", function () { showSign(playIdx + 1); });

  /* ---- characters: vocabulary artwork management ---- */
  var charGrid = $("charGrid"), charForm = $("charForm"),
      charInput = $("charInput"), charFile = $("charFile"),
      charFileName = $("charFileName"), charMsg = $("charMsg");

  function renderVocab(items) {
    charGrid.innerHTML = "";
    items.forEach(function (it) {
      var fig = document.createElement("figure");
      fig.className = "char-card" + (it.trained ? " trained" : "");
      var img = document.createElement("img");
      img.src = it.sign_image;
      img.alt = "Sign for " + it.char;
      img.loading = "lazy";
      var cap = document.createElement("figcaption");
      cap.textContent = it.char;
      if (it.trained) {
        var badge = document.createElement("span");
        badge.className = "char-badge";
        badge.setAttribute("aria-hidden", "true");
        badge.textContent = "✓";
        fig.appendChild(badge);
        var sr = document.createElement("span");
        sr.className = "vh";
        sr.textContent = ", trained";
        cap.appendChild(sr);
      }
      fig.appendChild(img);
      fig.appendChild(cap);
      if (it.available) {
        var del = document.createElement("button");
        del.className = "char-del";
        del.type = "button";
        del.title = "Remove artwork for " + it.char;
        del.setAttribute("aria-label", "Remove artwork for " + it.char);
        del.textContent = "×";
        del.addEventListener("click", function () { deleteCharacter(it.char); });
        fig.appendChild(del);
      }
      charGrid.appendChild(fig);
    });
  }

  function loadVocab() {
    fetch("/api/vocabulary")
      .then(function (r) { return r.json(); })
      .then(function (j) { renderVocab(j.characters || []); })
      .catch(function () { charGrid.innerHTML = ""; });
  }

  function deleteCharacter(ch) {
    fetch("/api/characters/" + encodeURIComponent(ch), { method: "DELETE" })
      .then(function () { loadVocab(); })
      .catch(function () { charMsg.textContent = "Could not remove artwork."; });
  }

  charFile.addEventListener("change", function () {
    charFileName.textContent = charFile.files.length ? charFile.files[0].name : "Choose image…";
  });

  charForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var ch = (charInput.value || "").trim().toUpperCase();
    var f = charFile.files[0];
    if (!/^[A-Z0-9]$/.test(ch)) { charMsg.textContent = "Enter a single letter A–Z or digit 0–9."; return; }
    if (!f) { charMsg.textContent = "Choose an artwork image first."; return; }
    var btn = $("charSave");
    btn.disabled = true;
    charMsg.textContent = "Uploading…";
    var reader = new FileReader();
    reader.onload = function () {
      fetch("/api/characters", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ char: ch, image: reader.result })
      })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
        .then(function (res) {
          btn.disabled = false;
          if (!res.ok) { charMsg.textContent = res.body.error || "Upload failed."; return; }
          charMsg.textContent = "Character “" + res.body.char + "” is live in previews.";
          charInput.value = "";
          charFile.value = "";
          charFileName.textContent = "Choose image…";
          loadVocab();
        })
        .catch(function () { btn.disabled = false; charMsg.textContent = "Upload failed."; });
    };
    reader.readAsDataURL(f);
  });

  /* ---- custom signs: whole-word artwork (wires the Custom signs form) ---- */
  var customGrid = $("customGrid"), customForm = $("customForm"),
      customInput = $("customInput"), customFile = $("customFile"),
      customFileName = $("customFileName"), customMsg = $("customMsg");

  function renderCustom(signs) {
    customGrid.innerHTML = "";
    if (!signs.length) {
      customGrid.innerHTML = '<div class="sign-empty"><p>No custom signs yet</p>' +
        "<span>Add a word above — Text to Sign will play it as one sign.</span></div>";
      return;
    }
    signs.forEach(function (s) {
      var fig = document.createElement("figure");
      fig.className = "char-card";
      var img = document.createElement("img");
      img.src = s.sign_image;
      img.alt = "Sign for " + s.label;
      img.loading = "lazy";
      var cap = document.createElement("figcaption");
      cap.textContent = s.label;
      var del = document.createElement("button");
      del.className = "char-del";
      del.type = "button";
      del.textContent = "×";
      del.setAttribute("aria-label", "Remove custom sign " + s.label);
      del.addEventListener("click", function () {
        fetch("/api/custom-signs/" + encodeURIComponent(s.id), { method: "DELETE" })
          .then(function () { loadCustom(); })
          .catch(function () { customMsg.textContent = "Could not remove sign."; });
      });
      fig.appendChild(img);
      fig.appendChild(cap);
      fig.appendChild(del);
      customGrid.appendChild(fig);
    });
  }

  function loadCustom() {
    fetch("/api/custom-signs", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (j) { renderCustom(j.signs || []); })
      .catch(function () { renderCustom([]); });
  }

  customFile.addEventListener("change", function () {
    customFileName.textContent = customFile.files.length ? customFile.files[0].name : "Choose image…";
  });

  customForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var label = (customInput.value || "").trim();
    var f = customFile.files[0];
    if (!/^[A-Za-z0-9 ]{1,30}$/.test(label)) { customMsg.textContent = "Use 1–30 letters, digits or spaces."; return; }
    if (!f) { customMsg.textContent = "Choose an artwork image first."; return; }
    var btn = $("customSave");
    btn.disabled = true;
    customMsg.textContent = "Uploading…";
    var reader = new FileReader();
    reader.onload = function () {
      fetch("/api/custom-signs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: label, image: reader.result })
      })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
        .then(function (res) {
          btn.disabled = false;
          if (!res.ok) { customMsg.textContent = res.body.error || "Upload failed."; return; }
          customMsg.textContent = "Sign added — Text to Sign will play it as one sign.";
          customInput.value = "";
          customFile.value = "";
          customFileName.textContent = "Choose image…";
          loadCustom();
        })
        .catch(function () { btn.disabled = false; customMsg.textContent = "Upload failed."; });
    };
    reader.readAsDataURL(f);
  });

  loadCustom();
  loadVocab();

  /* ---- SMTP Email Alert functionality ---- */
  var btnSendAlert = $("btnSendAlert"), btnToggleSmtp = $("btnToggleSmtp"),
      btnAutoAlert = $("btnAutoAlert"), alertStatusMsg = $("alertStatusMsg"),
      smtpConfigCard = $("smtpConfigCard"), btnCloseSmtp = $("btnCloseSmtp"),
      smtpForm = $("smtpForm"), smtpMsg = $("smtpMsg"),
      btnSaveSmtp = $("btnSaveSmtp"), btnTestSmtp = $("btnTestSmtp");

  var autoAlertOn = true, autoAlertTimer = null, lastAlertSentText = "";

  function setAlertStatus(msg, isSuccess, isError) {
    if (!alertStatusMsg) return;
    alertStatusMsg.hidden = !msg;
    alertStatusMsg.textContent = msg || "";
    alertStatusMsg.className = "alert-status-msg" + (isSuccess ? " success" : (isError ? " error" : ""));
  }

  function loadSmtpConfig() {
    fetch("/api/smtp-config", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        if ($("smtpHost")) $("smtpHost").value = cfg.smtp_host || "smtp.gmail.com";
        if ($("smtpPort")) $("smtpPort").value = cfg.smtp_port || 587;
        if ($("smtpUseTls")) $("smtpUseTls").checked = cfg.use_tls !== false;
        if ($("smtpSender")) $("smtpSender").value = cfg.sender_email || "ayushghatol49@gmail.com";
        if ($("smtpRecipient")) $("smtpRecipient").value = cfg.default_recipient || "ayushghatol49@gmail.com,tarhekargayatri100@gmail.com";
        if ($("smtpPassword") && cfg.password_configured) {
          $("smtpPassword").placeholder = "•••••••• (Password set)";
        }
      })
      .catch(function () {});
  }
  loadSmtpConfig();

  if (btnToggleSmtp) {
    btnToggleSmtp.addEventListener("click", function () {
      smtpConfigCard.hidden = !smtpConfigCard.hidden;
      if (!smtpConfigCard.hidden) loadSmtpConfig();
    });
  }
  if (btnCloseSmtp) {
    btnCloseSmtp.addEventListener("click", function () {
      smtpConfigCard.hidden = true;
    });
  }

  if (smtpForm) {
    smtpForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var data = {
        smtp_host: $("smtpHost").value.trim(),
        smtp_port: parseInt($("smtpPort").value, 10) || 587,
        use_tls: $("smtpUseTls").checked,
        sender_email: $("smtpSender").value.trim(),
        default_recipient: $("smtpRecipient").value.trim(),
        sender_password: $("smtpPassword").value.trim()
      };
      btnSaveSmtp.disabled = true;
      smtpMsg.textContent = "Saving configuration…";
      fetch("/api/smtp-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
        .then(function (res) {
          btnSaveSmtp.disabled = false;
          if (!res.ok) {
            smtpMsg.textContent = res.body.error || "Failed to save SMTP config.";
            return;
          }
          smtpMsg.textContent = "✓ SMTP settings saved successfully!";
          setTimeout(function () { smtpMsg.textContent = ""; }, 4000);
        })
        .catch(function () {
          btnSaveSmtp.disabled = false;
          smtpMsg.textContent = "Error saving SMTP config.";
        });
    });
  }

  function sendAlertEmail(customText, isTest) {
    var text = (customText !== undefined ? customText : sentence.textContent).trim();
    if (!text && !isTest) {
      setAlertStatus("⚠️ Cannot send alert: Signed text is empty. Sign text or write a message first.", false, true);
      return;
    }
    setAlertStatus("✉ Sending email alert via SMTP…", false, false);
    if (btnSendAlert) btnSendAlert.disabled = true;

    var body = { text: text || "Test alert notification from VaniSetu console." };
    if (isTest) body.subject = "[VaniSetu Test] SMTP Connection Verified";

    fetch("/api/send-alert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        if (btnSendAlert) btnSendAlert.disabled = false;
        if (res.ok) {
          lastAlertSentText = text;
          setAlertStatus("✓ Email alert successfully sent to " + (res.body.recipient || "defined receivers") + "!", true, false);
        } else {
          setAlertStatus("✖ Email alert failed: " + (res.body.error || "Check SMTP settings."), false, true);
          if (smtpConfigCard.hidden || (res.body.error && (res.body.error.indexOf("recipient") !== -1 || res.body.error.indexOf("Sender") !== -1))) {
            smtpConfigCard.hidden = false;
            loadSmtpConfig();
          }
        }
      })
      .catch(function () {
        if (btnSendAlert) btnSendAlert.disabled = false;
        setAlertStatus("✖ Network error when sending alert email.", false, true);
      });
  }

  if (btnSendAlert) {
    btnSendAlert.addEventListener("click", function () {
      sendAlertEmail();
    });
  }

  if (btnTestSmtp) {
    btnTestSmtp.addEventListener("click", function () {
      sendAlertEmail("This is a test alert email to verify SMTP configuration.", true);
    });
  }

  if (btnAutoAlert) {
    btnAutoAlert.textContent = "Auto-Send on 'ALERT': Active";
    btnAutoAlert.setAttribute("aria-pressed", "true");
    btnAutoAlert.addEventListener("click", function () {
      autoAlertOn = !autoAlertOn;
      btnAutoAlert.textContent = autoAlertOn ? "Auto-Send on 'ALERT': Active" : "Auto-Send on 'ALERT': Off";
      btnAutoAlert.setAttribute("aria-pressed", autoAlertOn ? "true" : "false");
      if (!autoAlertOn) {
        clearTimeout(autoAlertTimer);
      }
    });
  }

  function queueAutoAlert(latestLabel) {
    var t = sentence.textContent.trim();
    if (!t) return;

    var upper = t.toUpperCase();
    var isEmergencyKeyword = (upper.indexOf("ALERT") !== -1 || upper.indexOf("HELP") !== -1 || upper.indexOf("EMERGENCY") !== -1 || upper.indexOf("SOS") !== -1);

    // STRICT: Only send automatically if an alert text / keyword is present!
    if (!autoAlertOn) return;

    if (isEmergencyKeyword && t !== lastAlertSentText) {
      clearTimeout(autoAlertTimer);
      autoAlertTimer = setTimeout(function () {
        var current = sentence.textContent.trim();
        var currentUpper = current.toUpperCase();
        var isAlert = (currentUpper.indexOf("ALERT") !== -1 || currentUpper.indexOf("HELP") !== -1 || currentUpper.indexOf("EMERGENCY") !== -1 || currentUpper.indexOf("SOS") !== -1);
        if (isAlert && current !== lastAlertSentText) {
          sendAlertEmail(current);
        }
      }, 1000);
    }
  }
})();
