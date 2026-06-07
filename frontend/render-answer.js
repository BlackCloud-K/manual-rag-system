/**
 * Markdown + KaTeX rendering for assistant answers.
 * Depends on: marked, katex, renderMathInElement (auto-render).
 */
(function (global) {
  const STREAM_DEBOUNCE_MS = 120;

  const KATEX_RENDER_OPTS = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
    strict: "ignore",
  };

  function configureMarked() {
    if (typeof marked === "undefined") return;
    if (global.__renderAnswerMarkedConfigured) return;
    marked.setOptions({ breaks: true, gfm: true });
    global.__renderAnswerMarkedConfigured = true;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function markdownToHtml(text) {
    configureMarked();
    const raw = String(text ?? "");
    if (typeof marked !== "undefined") {
      return marked.parse(raw);
    }
    return escapeHtml(raw).replace(/\n/g, "<br>");
  }

  function renderMathIn(el) {
    if (!el || typeof renderMathInElement === "undefined") return;
    renderMathInElement(el, KATEX_RENDER_OPTS);
  }

  function renderAssistantAnswer(element, markdownText) {
    if (!element) return;
    element.innerHTML = markdownToHtml(markdownText);
    renderMathIn(element);
  }

  function createStreamRenderer(element) {
    let timer = null;
    let latest = "";
    let hasRendered = false;

    function flushNow(text) {
      if (text != null) latest = String(text);
      renderAssistantAnswer(element, latest);
      hasRendered = true;
    }

    return {
      update(text) {
        latest = String(text ?? "");
        if (!hasRendered) {
          flushNow(latest);
        }
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => {
          timer = null;
          renderAssistantAnswer(element, latest);
        }, STREAM_DEBOUNCE_MS);
      },
      flush(text) {
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        flushNow(text);
      },
    };
  }

  global.renderAssistantAnswer = renderAssistantAnswer;
  global.createStreamRenderer = createStreamRenderer;
})(typeof window !== "undefined" ? window : globalThis);
