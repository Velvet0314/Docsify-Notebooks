// Default configuration options
var defaultOptions = {
  countable: true,
  position: "top",          // "top" or "bottom"
  margin: "10px",           // Margin around the counter
  float: "right",           // "left" or "right"
  fontsize: "0.9em",        // Font size of the counter
  color: "rgb(90,90,90)",   // Text color of the counter
  language: "english",      // Language: "english" or "chinese"
  localization: {
    words: " words",        // Custom text for "words" in the counter
    minute: " min",         // Custom text for "minute" in the counter
  },
  isExpected: true,         // Whether to show expected read time
};

// Docsify plugin functions
function wordCountPlugin(hook, vm) {
  if (!defaultOptions.countable) {
    return;
  }

  let wordsCount;

  hook.beforeEach(function (content) {
    // Remove MathJax content from counting by removing <span class="math">...</span> and <div class="math">...</div>
    let contentForCounting = content
      .replace(/<span\s+class="math">[\s\S]*?<\/span>/gi, "")
      .replace(/<div\s+class="math">[\s\S]*?<\/div>/gi, "");

    // Count words by matching alphanumeric sequences and CJK characters
    wordsCount = (
      contentForCounting.match(/([\u0800-\u4e00]+?|[\u4e00-\u9fa5]+?|[a-zA-Z0-9]+)/g) || []
    ).length;

    return content;
  });

  hook.afterEach(function (html, next) {
    // Support localization
    let wordText = wordsCount + (defaultOptions.localization.words || " words");
    let readTimeText = Math.ceil(wordsCount / 400) + (defaultOptions.localization.minute || " min");

    if (defaultOptions.language === "chinese") {
      wordText = wordsCount + " 字";
      readTimeText = Math.ceil(wordsCount / 400) + " 分钟";
    }

    // Create the counter element
    let counterHtml = `
      <div style="margin-${defaultOptions.position === "bottom" ? "top" : "bottom"}: ${defaultOptions.margin};">
        <span style="
          float: ${defaultOptions.float};
          font-size: ${defaultOptions.fontsize};
          color: ${defaultOptions.color};">
          ${wordText}${defaultOptions.isExpected ? `&nbsp; | &nbsp;${readTimeText}` : ""}
        </span>
        <div style="clear: both"></div>
      </div>`;

    // Add the counter to the HTML content
    next(
      `${defaultOptions.position === "bottom" ? html : ""}${counterHtml}${defaultOptions.position !== "bottom" ? html : ""}`
    );
  });
}

// Docsify plugin options
window.$docsify["count"] = Object.assign(defaultOptions, window.$docsify["count"] || {});
window.$docsify.plugins = [].concat(wordCountPlugin, window.$docsify.plugins || []);
