use newton_rootfinder as nrf;
use nrf::log::Log;

#[test]
fn the_page_of_a_log_file_holds_its_lines() {
    let directory = std::env::temp_dir().join(format!("nrf-page-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let log_path = directory.join("solver_log.jsonl");
    let page_path = directory.join("solver_log.html");

    let log = Log::create(&log_path).unwrap();
    log.note("case", |fields| {
        fields.text("model", "a < b & c > d");
    });
    log.close().unwrap();
    nrf::log::write_page(&log_path, &page_path).unwrap();

    let lines = std::fs::read_to_string(&log_path).unwrap();
    let page = std::fs::read_to_string(&page_path).unwrap();
    std::fs::remove_dir_all(&directory).unwrap();

    assert!(page.starts_with("<!doctype html>"));
    assert!(page.contains(lines.trim_end()));
    assert!(lines.contains("\"model\":\"a \\u003c b \\u0026 c \\u003e d\""));
}

#[test]
fn a_missing_log_is_an_error() {
    let missing = std::env::temp_dir().join("nrf-no-such-log.jsonl");
    let page = std::env::temp_dir().join("nrf-no-such-log.html");
    assert!(nrf::log::write_page(&missing, &page).is_err());
    assert!(!page.exists());
}
