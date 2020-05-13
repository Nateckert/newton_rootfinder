use std::fs::File;
use std::io::Write;

const LOG_FILE_PATH: &'static str = "solver_log.txt";

pub struct SolverLog {
    content: String,
}

impl SolverLog {
    pub fn new() -> Self {
        let content = String::from("Solver log\n");
        SolverLog { content }
    }

    pub fn add_content(&mut self, new_content: &str) {
        self.content.push_str(new_content);
    }

    pub fn write(&self) {
        let mut f = File::create(LOG_FILE_PATH).unwrap();
        write!(f, "{}", self.content).unwrap();
    }
}
