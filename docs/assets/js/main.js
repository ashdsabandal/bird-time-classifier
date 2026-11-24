// ----------------------------------------------------------
// 1. Figures: load each figure's file if it exists,
//    otherwise show a placeholder with the expected file name.
// ----------------------------------------------------------
document.querySelectorAll(".figure-slot[data-src]").forEach((slot) => {
  const src = slot.dataset.src;

  const showPlaceholder = () => {
    slot.classList.add("empty");
    slot.innerHTML =
      '<span class="ph-title">Figure goes here</span>' +
      "<span>Save the chart as</span>" +
      "<code>" + src + "</code>";
  };

  // Interactive charts (e.g. Plotly "write_html") are embedded in an iframe
  if (src.endsWith(".html")) {
    fetch(src, { method: "HEAD" })
      .then((res) => {
        if (!res.ok) throw new Error();
        slot.innerHTML = '<iframe src="' + src + '" loading="lazy" title="Figure"></iframe>';
      })
      .catch(showPlaceholder);
    return;
  }

  // Show the placeholder right away; swap in the image once it loads
  showPlaceholder();
  const img = new Image();
  img.alt = slot.closest("figure")?.querySelector("figcaption")?.textContent.trim() || "Figure";
  img.onload = () => {
    slot.classList.remove("empty");
    slot.innerHTML = "";
    slot.appendChild(img);
  };
  img.src = src;
});

// ----------------------------------------------------------
// 2. Tabs in the Results section
// ----------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
  });
});

// ----------------------------------------------------------
// 3. Highlight the current section in the nav bar
// ----------------------------------------------------------
const navLinks = document.querySelectorAll(".site-nav nav a");
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navLinks.forEach((a) =>
        a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id)
      );
    });
  },
  { rootMargin: "-45% 0px -50% 0px" }
);
document.querySelectorAll("main section[id]").forEach((s) => observer.observe(s));
