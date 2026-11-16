//! The HTML page of a log

/// The page, its style and its script, with the place of the lines
const TEMPLATE: &str = include_str!("page.html");
const LINES: &str = "/*LINES*/";

/// The page showing the lines of a log, held in it
pub(crate) fn render(lines: &str) -> String {
    // The lines sit in a script element, which a `</script>` would end:
    // the log escapes `<` in its strings, and outside a string a `<` is no JSON anyway
    TEMPLATE.replacen(LINES, &lines.replace('<', "\\u003c"), 1)
}
