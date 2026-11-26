// GPUsync — menu sur téléphone, copie des blocs de code, thème clair / sombre.
(function () {
  var root = document.documentElement;

  function stored() {
    try { return localStorage.getItem("gpusync-theme"); } catch (e) { return null; }
  }
  var saved = stored();
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  function dark() {
    var t = root.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var body = document.body;

    var toggle = document.querySelector(".menu-toggle");
    var scrim = document.querySelector(".scrim");
    function setMenu(open) {
      body.classList.toggle("menu-open", open);
      if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }
    if (toggle) toggle.addEventListener("click", function () { setMenu(!body.classList.contains("menu-open")); });
    if (scrim) scrim.addEventListener("click", function () { setMenu(false); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") setMenu(false); });

    var theme = document.querySelector(".theme-toggle");
    function showTheme() {
      if (!theme) return;
      theme.querySelector(".icon").textContent = dark() ? "light_mode" : "dark_mode";
      theme.setAttribute("aria-label", dark() ? "Thème clair" : "Thème sombre");
    }
    if (theme) {
      showTheme();
      theme.addEventListener("click", function () {
        var next = dark() ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem("gpusync-theme", next); } catch (e) {}
        showTheme();
      });
    }

    document.querySelectorAll(".copy").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var code = btn.closest(".code").querySelector("pre").innerText;
        var done = function () {
          btn.textContent = "Copié";
          setTimeout(function () { btn.textContent = "Copier"; }, 1500);
        };
        if (navigator.clipboard) navigator.clipboard.writeText(code).then(done, function () {});
      });
    });
  });
})();
