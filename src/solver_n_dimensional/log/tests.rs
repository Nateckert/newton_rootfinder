use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::{Detail, Log, clock, json};

/// A writer whose bytes the test reads back
#[derive(Clone, Default)]
struct Buffer(Arc<Mutex<Vec<u8>>>);

impl Write for Buffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Buffer {
    fn lines(&self) -> Vec<String> {
        String::from_utf8(self.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }
}

struct Failing;

impl Write for Failing {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("disk full"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("disk full"))
    }
}

fn encoded(value: f64) -> String {
    let mut line = String::new();
    json::number(&mut line, value);
    line
}

#[test]
fn numbers_are_json_reading_back_the_same_value() {
    assert_eq!(encoded(1.0), "1.0");
    assert_eq!(encoded(0.1), "0.1");
    assert_eq!(encoded(1e-7), "1e-7");
    assert_eq!(encoded(-0.0), "-0.0");
    assert_eq!(encoded(f64::NAN), "\"NaN\"");
    assert_eq!(encoded(f64::INFINITY), "\"Infinity\"");
    assert_eq!(encoded(f64::NEG_INFINITY), "\"-Infinity\"");

    for value in [
        1.0 / 3.0,
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,
        -2.5e17,
        123_456.789,
    ] {
        let text = encoded(value);
        let read: f64 = text.parse().unwrap();
        assert_eq!(read.to_bits(), value.to_bits(), "{}", text);
    }
}

#[test]
fn strings_are_escaped_for_json_and_html() {
    let mut line = String::new();
    json::string(&mut line, "a \"b\" \\ c\n</script>&\u{1}");
    assert_eq!(
        line,
        "\"a \\\"b\\\" \\\\ c\\n\\u003c/script\\u003e\\u0026\\u0001\""
    );
}

#[test]
fn dates_are_in_utc() {
    assert_eq!(clock::rfc3339(Duration::ZERO), "1970-01-01T00:00:00.000Z");
    assert_eq!(
        clock::rfc3339(Duration::from_secs(951_782_400)),
        "2000-02-29T00:00:00.000Z"
    );
    assert_eq!(
        clock::rfc3339(Duration::from_secs(4_107_628_799)),
        "2100-03-01T23:59:59.000Z"
    );
    assert_eq!(
        clock::rfc3339(Duration::from_millis(1_794_820_323_120)),
        "2026-11-16T09:12:03.120Z"
    );
}

#[test]
fn the_first_line_names_the_format_and_the_version() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    log.close().unwrap();
    let lines = buffer.lines();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with(&format!(
        "{{\"event\":\"log\",\"format\":1,\"version\":\"{}\",\"opened\":\"",
        env!("CARGO_PKG_VERSION")
    )));
    assert!(lines[0].ends_with("Z\"}"));
}

#[test]
fn scopes_nest_and_notes_sit_in_the_innermost() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    {
        let _outer = log.scope("outer");
        {
            let _inner = log.scope("inner");
            log.note("facts", |fields| {
                fields.text("os", "linux").number("threads", 4.0);
            });
        }
        log.note("empty", |_| {});
    }
    log.note("top", |_| {});
    log.close().unwrap();

    let lines = buffer.lines();
    assert_eq!(lines.len(), 8);
    assert!(lines[1].starts_with("{\"event\":\"scope\",\"id\":1,\"parent\":null,\"time\":"));
    assert!(lines[1].ends_with(",\"label\":\"outer\"}"));
    assert!(lines[2].starts_with("{\"event\":\"scope\",\"id\":2,\"parent\":1,"));
    assert!(lines[3].starts_with("{\"event\":\"note\",\"parent\":2,"));
    assert!(
        lines[3].ends_with(",\"label\":\"facts\",\"fields\":{\"os\":\"linux\",\"threads\":4.0}}")
    );
    assert!(lines[4].starts_with("{\"event\":\"scope_end\",\"id\":2,"));
    assert!(lines[5].starts_with("{\"event\":\"note\",\"parent\":1,"));
    assert!(lines[5].ends_with(",\"label\":\"empty\",\"fields\":{}}"));
    assert!(lines[6].starts_with("{\"event\":\"scope_end\",\"id\":1,"));
    assert!(lines[7].starts_with("{\"event\":\"note\",\"parent\":null,"));
}

#[test]
fn each_thread_has_its_own_innermost_scope() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let main = log.scope("main");
    let other = log.clone();
    std::thread::spawn(move || {
        let _scope = other.scope("thread");
        other.note("in the thread", |_| {});
    })
    .join()
    .unwrap();
    log.note("in main", |_| {});
    drop(main);
    log.close().unwrap();

    let lines = buffer.lines();
    assert!(lines[2].starts_with("{\"event\":\"scope\",\"id\":2,\"parent\":null,"));
    assert!(lines[3].starts_with("{\"event\":\"note\",\"parent\":2,"));
    assert!(lines[5].starts_with("{\"event\":\"note\",\"parent\":1,"));
}

#[test]
fn the_detail_is_shared_by_the_handles() {
    let log = Log::new(io::sink());
    let handle = log.clone();
    let log = log.detail(Detail::Outcome);
    assert_eq!(handle.get_detail(), Detail::Outcome);
    assert_eq!(log.get_detail(), Detail::Outcome);
}

#[test]
fn an_error_of_writing_stops_the_log_and_is_returned_when_closing() {
    let log = Log::new(Failing);
    {
        let _scope = log.scope("scope");
        log.note("lost", |fields| {
            fields.number("value", 1.0);
        });
    }
    let error = log.close().unwrap_err();
    assert_eq!(error.to_string(), "disk full");
}

#[test]
fn keeping_the_failures_only_drops_the_scopes_where_none_failed() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).failures_only();
    {
        let _scope = log.scope("nothing failed");
        log.note("kept, then dropped", |_| {});
    }
    log.note("at the top", |_| {});
    log.close().unwrap();

    let lines = buffer.lines();
    assert_eq!(lines.len(), 2);
    assert!(lines[1].starts_with("{\"event\":\"note\",\"parent\":null,"));
    assert!(lines[1].contains("\"label\":\"at the top\""));
}

#[test]
fn the_page_holds_the_lines_and_nothing_can_end_its_script() {
    let lines = "{\"event\":\"log\",\"format\":1}\n{\"event\":\"note\",\"label\":\"</script>\"}\n";
    let page = super::page::render(lines);
    assert!(page.starts_with("<!doctype html>"));
    assert!(page.contains(
        "{\"event\":\"log\",\"format\":1}\n{\"event\":\"note\",\"label\":\"\\u003c/script>\"}"
    ));
    assert!(!page.contains("/*LINES*/"));
    assert_eq!(page.matches("</script>").count(), 2);
}
