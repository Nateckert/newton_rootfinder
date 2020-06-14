use std::fs::File;
use std::io::Write;

use crate::solver_advanced::residuals::ResidualsValues;

extern crate chrono;
extern crate whoami;
use chrono::prelude::*;

const SEPARATION_ITER: &str = "=========================\n\n";
const SEPARATION_LINE: &str =  "+-------+-------------------------------+----------------------------------------------------------------+---------------------------------+\n";
const TITLE_LINE: &str =       "| Id    |           Iteratives          |        Left                    =                Right          |         Stopping criteria       |\n";
const FLOAT_WIDTH: usize = 30;
const INT_WIDTH: usize = 6;

pub struct SolverLog {
    content: String,
}

/// Log for debugging information
///
/// This object defines the format and concatenate the debugging informations
impl SolverLog {
    pub fn new() -> Self {
        let mut content = String::new();
        content.push_str(&"Runner informations\n");
        content.push_str(&"===================\n\n");
        content.push_str(&"OS: ");
        content.push_str(&whoami::os());
        content.push_str(&"\n");
        content.push_str(&"Host: ");
        content.push_str(&whoami::host());
        content.push_str(&"\n");
        content.push_str(&"Username: ");
        content.push_str(&whoami::username());
        content.push_str(&"\n");
        const VERSION: &str = env!("CARGO_PKG_VERSION");
        content.push_str("newton_rootfinder version: ");
        content.push_str(VERSION);
        content.push_str(&"\n");
        content.push_str("Simulation start:\n");

        let utc: DateTime<Utc> = Utc::now();
        let local: DateTime<Local> = Local::now();

        content.push_str(&"  - UTC:   ");
        content.push_str(&utc.to_rfc2822());
        content.push_str(&"\n");
        content.push_str(&"  - Local: ");
        content.push_str(&local.to_rfc2822());
        content.push_str(&"\n");
        content.push_str(&"\n");

        SolverLog { content }
    }

    pub fn add_content(&mut self, new_content: &str) {
        self.content.push_str(new_content);
    }

    pub fn add_parameters(
        &mut self,
        solver_parameters: &str,
        iteratives_config: &str,
        residuals_config: &str,
    ) {
        self.add_content(solver_parameters);
        self.add_content(iteratives_config);
        self.add_content(residuals_config);
    }

    pub fn add_damping(
        &mut self,
        iteratives: &nalgebra::DVector<f64>,
        residuals: &ResidualsValues,
        errors: &nalgebra::DVector<f64>,
    ) {
        let mut iteration_log_header = String::new();
        iteration_log_header.push_str(&"Damping activated !\n\n".to_string());
        self.add_content(&iteration_log_header);
        self.add_iteration(iteratives, residuals, errors);
    }
    pub fn add_new_iteration(
        &mut self,
        iteratives: &nalgebra::DVector<f64>,
        residuals: &ResidualsValues,
        errors: &nalgebra::DVector<f64>,
        iter: usize,
    ) {
        let mut iteration_log_header = String::new();
        iteration_log_header.push_str(SEPARATION_ITER);
        iteration_log_header.push_str(&format!("Iteration: {}\n\n", iter.to_string()));
        self.add_content(&iteration_log_header);
        self.add_iteration(iteratives, residuals, errors);
    }

    fn add_iteration(
        &mut self,
        iteratives: &nalgebra::DVector<f64>,
        residuals: &ResidualsValues,
        errors: &nalgebra::DVector<f64>,
    ) {
        let mut iteration_log_header = String::new();
        iteration_log_header.push_str(&format!("Max error: {}\n\n", errors.amax()));
        self.add_content(&iteration_log_header);
        self.add_content(&SEPARATION_LINE);
        self.add_content(&TITLE_LINE);
        self.add_content(&SEPARATION_LINE);
        for (i, (iterative, error)) in iteratives.iter().zip(errors.iter()).enumerate() {
            let mut entry = String::new();
            entry.push_str(&format!("| {:width$}", i.to_string(), width = INT_WIDTH));
            entry.push_str(&format!(
                "| {:width$}",
                iterative.to_string(),
                width = FLOAT_WIDTH
            ));
            entry.push_str(&format!(
                "| {:width$}",
                residuals.get_values_str_eq(i, FLOAT_WIDTH),
                width = FLOAT_WIDTH
            ));
            entry.push_str(&format!(
                "| {:width$}  |",
                error.to_string(),
                width = FLOAT_WIDTH
            ));
            entry.push_str(&"\n");
            entry.push_str(&SEPARATION_LINE);
            self.add_content(&entry);
        }
        self.add_content(&"\n");
    }

    pub fn write(&self, path: &str) {
        let mut f = File::create(path).unwrap();
        write!(f, "{}", self.content).unwrap();
    }
}
