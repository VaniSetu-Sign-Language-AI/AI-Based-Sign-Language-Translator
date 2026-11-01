/* Training-metrics page — reads the performance matrix written at train time. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

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

  function pct(x) { return (x * 100).toFixed(1) + "%"; }

  fetch("/api/metrics", { cache: "no-store" })
    .then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, body: j }; });
    })
    .then(function (res) {
      if (!res.ok) {
        $("metricsEmpty").hidden = false;
        return;
      }
      var m = res.body;
      $("metricsEmpty").hidden = true;
      $("metricsBody").hidden = false;
      $("kpiAcc").textContent = pct(m.rf_accuracy);
      $("kpiF1").textContent = pct(m.macro_avg["f1-score"]);
      $("kpiBb").textContent = pct(m.backbone_accuracy);
      $("kpiN").textContent = m.test_samples;
      var when = new Date(m.generated_at);
      $("metricsMeta").textContent =
        m.model + " · " + m.classes.length + " signs · evaluated " +
        (isNaN(when) ? m.generated_at : when.toLocaleString());

      var rows = m.classes.map(function (c) {
        var p = m.per_class[c];
        return "<tr><th scope=\"row\">" + c + "</th><td>" + pct(p.precision) +
          "</td><td>" + pct(p.recall) + "</td><td>" + pct(p.f1_score || p["f1-score"]) +
          "</td><td>" + p.support + "</td></tr>";
      });
      rows.push("<tr class=\"avg-row\"><th scope=\"row\">Macro avg</th><td>" +
        pct(m.macro_avg.precision) + "</td><td>" + pct(m.macro_avg.recall) +
        "</td><td>" + pct(m.macro_avg["f1-score"]) + "</td><td>—</td></tr>");
      $("metricRows").innerHTML = rows.join("");
    })
    .catch(function () { $("metricsEmpty").hidden = false; });
})();
