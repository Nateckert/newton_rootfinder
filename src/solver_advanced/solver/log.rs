use std::fs::File;
use std::io::Write;

use crate::solver_advanced::util::residuals;

const SEPARATION_ITER: &str = "=========================\n\n";
const SEPARATION_LINE: &str =  "+-------+-------------------------------+----------------------------------------------------------------+---------------------------------+\n";
const TITLE_LINE: &str =       "| Id    |           Iteratives          |        Left                    =                Right          |         Stopping criteria       |\n";
const FLOAT_WIDTH: usize = 30;
const INT_WIDTH: usize = 6;

pub struct SolverLog {
    content: String,
}

pub struct Parameters {
    pub max_iter: String,
    pub tolerance: String,
    pub residuals_config: String,
    pub iters_params: String,
    pub init_guess: String,
}

impl Parameters {
    pub fn new(
        max_iter: &str,
        tolerance: &str,
        residuals_config: &str,
        iters_params: &str,
        init_guess: &str,
    ) -> Self {
        Parameters {
            max_iter: max_iter.to_string(),
            tolerance: tolerance.to_string(),
            residuals_config: residuals_config.to_string(),
            iters_params: iters_params.to_string(),
            init_guess: init_guess.to_string(),
        }
    }
}

/// Log for debugging information
///
/// This object defines the format and concatenate the debugging informations
impl SolverLog {
    pub fn new() -> Self {
        let content = String::new();
        SolverLog { content }
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

    pub fn add_damping(
        &mut self,
        iteratives: &nalgebra::DVector<f64>,
        residuals: &residuals::ResidualsValues,
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
        residuals: &residuals::ResidualsValues,
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
        residuals: &residuals::ResidualsValues,
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

    pub fn add_jac(&mut self, jac: &nalgebra::DMatrix<f64>) {
        self.add_content(&"Jacobian Matrix:\n");
        self.add_content(&jac.to_string());
        self.add_content(&"\n");
    }

    pub fn write(&self, path: &str) {
        let mut f = File::create(path).unwrap();
        write!(f, "{}", self.content).unwrap();
    }
}
