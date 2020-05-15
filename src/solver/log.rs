use std::fs::File;
use std::io::Write;

use crate::util::residuals;

const LOG_FILE_PATH: &'static str = "solver_log.txt";
const SEPRATATION_LINE: &'static str =  "+-------+-------------------------------+----------------------------------------------------------------+---------------------------------+\n";
const TITLE_LINE: &'static str =        "| Id    |           Iteratives          |        Left                    =                Right          |         Stopping criteria       |\n";
const FLOAT_WIDTH: usize = 30;
const INT_WIDTH: usize = 6;

pub struct SolverLog {
    content: String,
    iteration: usize
}

pub struct Parameters  {
    pub max_iter: String,
    pub tolerance: String,
    pub residuals_config: String,
    pub iters_params: String,
    pub init_guess: String,
}

impl Parameters {
    pub fn new(max_iter: &str, tolerance: &str, residuals_config: &str, iters_params: &str, init_guess: &str) -> Self {
        Parameters {
            max_iter: max_iter.to_string(),
            tolerance: tolerance.to_string(),
            residuals_config: residuals_config.to_string(),
            iters_params: iters_params.to_string(),
            init_guess: init_guess.to_string(),
        }
    }
}

impl SolverLog {
    pub fn new() -> Self {
        let content = String::new();
        let iteration = 0;
        SolverLog { content, iteration }
    }

    fn add_content(&mut self, new_content: &str) {
        self.content.push_str(new_content);
    }

    pub fn add_parameters(&mut self, parameters: Parameters) {
        let mut new_content = String::from("Solver parameters\n");
        new_content.push_str("=================\n\n");
        new_content.push_str("Max iteration: ");
        new_content.push_str(&parameters.max_iter);
        new_content.push_str("\n----------------------------\n\n");
        new_content.push_str("Tolerance: ");
        new_content.push_str(&parameters.tolerance);
        new_content.push_str("\n----------------------------\n\n");
        new_content.push_str(&parameters.residuals_config);
        new_content.push_str("\n----------------------------\n\n");
        new_content.push_str(&parameters.iters_params);
        new_content.push_str("\n----------------------------\n\n");
        new_content.push_str("Init guess:\n");
        new_content.push_str(&parameters.init_guess);
        new_content.push_str("\n----------------------------\n\n");
        self.add_content(&new_content);
    }

    pub fn add_iteration(&mut self, iteratives: &nalgebra::DVector<f64>, residuals: &residuals::ResidualsValues, errors: &nalgebra::DVector<f64>) {
        let mut iteration_log_header = String::new();
        iteration_log_header.push_str(&format!("Iteration: {}\n\n", self.iteration.to_string()));
        self.add_content(&iteration_log_header);
        for (i, elt) in iteratives.iter().enumerate() {
            let mut entry = String::new();
            entry.push_str(&SEPRATATION_LINE);
            entry.push_str(&TITLE_LINE);
            entry.push_str(&SEPRATATION_LINE);
            entry.push_str(&format!("| {:width$}", i.to_string(), width = INT_WIDTH));
            entry.push_str(&format!("| {:width$}", iteratives[i].to_string(), width = FLOAT_WIDTH));
            entry.push_str(&format!("| {:width$}", residuals.get_values_str_eq(i, FLOAT_WIDTH), width = FLOAT_WIDTH));
            entry.push_str(&format!("| {:width$}  |", errors[i].to_string(), width = FLOAT_WIDTH));
            entry.push_str(&"\n");
            entry.push_str(&SEPRATATION_LINE);
            self.add_content(&entry);
        }
        self.add_content(&"\n");
        self.iteration += 1;
    }

    pub fn write(&self, path: &str) {
        let mut f = File::create(path).unwrap();
        write!(f, "{}", self.content).unwrap();
    }
}
