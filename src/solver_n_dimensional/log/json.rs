//! The encoding of the values of a line in JSON

use std::fmt::Write;

/// A number reading back to the same `f64`, a non-finite one as a string
pub(crate) fn number(line: &mut String, value: f64) {
    if value.is_finite() {
        // The `Debug` format is the shortest reading back to the same value,
        // with a decimal point or an exponent: a JSON number
        let _ = write!(line, "{:?}", value);
    } else if value.is_nan() {
        line.push_str("\"NaN\"");
    } else if value > 0.0 {
        line.push_str("\"Infinity\"");
    } else {
        line.push_str("\"-Infinity\"");
    }
}

pub(crate) fn unsigned(line: &mut String, value: u64) {
    let _ = write!(line, "{}", value);
}

pub(crate) fn optional_unsigned(line: &mut String, value: Option<u64>) {
    match value {
        Some(value) => unsigned(line, value),
        None => line.push_str("null"),
    }
}

pub(crate) fn boolean(line: &mut String, value: bool) {
    line.push_str(if value { "true" } else { "false" });
}

/// A string, `<`, `>` and `&` escaped as well, so that a line can be embedded in an HTML page as it is
pub(crate) fn string(line: &mut String, value: &str) {
    line.push('"');
    for character in value.chars() {
        match character {
            '"' => line.push_str("\\\""),
            '\\' => line.push_str("\\\\"),
            '\n' => line.push_str("\\n"),
            '\r' => line.push_str("\\r"),
            '\t' => line.push_str("\\t"),
            '<' | '>' | '&' | '\u{0}'..='\u{1f}' | '\u{2028}' | '\u{2029}' => {
                let _ = write!(line, "\\u{:04x}", character as u32);
            }
            _ => line.push(character),
        }
    }
    line.push('"');
}

/// `,"key":`, a key following another field of the object
pub(crate) fn key(line: &mut String, key: &str) {
    line.push(',');
    string(line, key);
    line.push(':');
}

pub(crate) fn numbers<I>(line: &mut String, values: I)
where
    I: IntoIterator<Item = f64>,
{
    line.push('[');
    for (i, value) in values.into_iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        number(line, value);
    }
    line.push(']');
}

pub(crate) fn unsigneds<I>(line: &mut String, values: I)
where
    I: IntoIterator<Item = u64>,
{
    line.push('[');
    for (i, value) in values.into_iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        unsigned(line, value);
    }
    line.push(']');
}

/// A matrix as the list of its rows
pub(crate) fn matrix<D>(line: &mut String, matrix: &nalgebra::OMatrix<f64, D, D>)
where
    D: nalgebra::Dim,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
{
    line.push('[');
    for (i, row) in matrix.row_iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        numbers(line, row.iter().copied());
    }
    line.push(']');
}
