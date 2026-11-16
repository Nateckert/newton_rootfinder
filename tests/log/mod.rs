#[cfg(feature = "xml_config_file")]
mod broyden;
mod failures;
mod model;
mod resolutions;

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

/// A writer whose lines the test reads back
#[derive(Clone, Default)]
pub struct Buffer(Arc<Mutex<Vec<u8>>>);

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
    /// The lines, the values changing from a run to the next replaced:
    /// the times, and the version of the crate
    pub fn normalized_lines(&self) -> Vec<String> {
        String::from_utf8(self.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|line| {
                let line = replace_value(line, "\"time\":", "0");
                let line = replace_value(&line, "\"opened\":", "\"OPENED\"");
                replace_value(&line, "\"version\":", "\"VERSION\"")
            })
            .collect()
    }
}

/// The line with the value of the key replaced, the value ending at the next `,` or `}`
fn replace_value(line: &str, key: &str, value: &str) -> String {
    match line.find(key) {
        Some(start) => {
            let value_start = start + key.len();
            let value_end = value_start
                + line[value_start..]
                    .find([',', '}'])
                    .unwrap_or(line.len() - value_start);
            format!("{}{}{}", &line[..value_start], value, &line[value_end..])
        }
        None => line.to_owned(),
    }
}
