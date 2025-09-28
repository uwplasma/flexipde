// MathJax configuration for flexipde documentation.
// This script defines how inline and display math is delimited and
// resets the MathJax typesetter on page navigation.  It is loaded by
// MkDocs as specified in mkdocs.yml.

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|", // don't ignore any classes
    processHtmlClass: "arithmatex" // process math in elements with this class
  }
};

// Re-render math on every page load.  See MkDocs Material docs for details.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});