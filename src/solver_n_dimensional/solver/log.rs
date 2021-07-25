use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;

use crate::residuals::ResidualsValues;

#[cfg(feature = "additional_log_info")]
use chrono::prelude::*;

const SEPARATION_ITER: &str = "=========================\n\n";
const SEPARATION_LINE: &str =  "+-------+-------------------------------+----------------------------------------------------------------+---------------------------------+\n";
const TITLE_LINE: &str =       "| Id    |           Iteratives          |        Left                    =                Right          |         Stopping criteria       |\n";
const FLOAT_WIDTH: usize = 30;
const INT_WIDTH: usize = 6;

pub struct SolverLog {
    path: String,
}

#[cfg(feature = "additional_log_info")]
fn write_time(content: &mut String) {
    let utc: DateTime<Utc> = Utc::now();
    let local: DateTime<Local> = Local::now();

    content.push_str("Simulation start:\n");

    content.push_str(&"  - UTC:   ");
    content.push_str(&utc.to_rfc2822());
    content.push_str(&"\n");
    content.push_str(&"  - Local: ");
    content.push_str(&local.to_rfc2822());
    content.push_str(&"\n");
}

#[cfg(feature = "additional_log_info")]
fn write_user_infos(content: &mut String) {
    content.push_str(&"OS: ");
    content.push_str(&whoami::distro());
    content.push_str(&"\n");
    content.push_str(&"Host: ");
    content.push_str(&whoami::devicename());
    content.push_str(&"\n");
    content.push_str(&"Username: ");
    content.push_str(&whoami::username());
    content.push_str(&"\n");
}

#[cfg(feature = "additional_log_info")]
fn write_rustc_info(content: &mut String) {
    content.push_str("Rust version: ");
    content.push_str(&rustc_version_runtime::version().to_string());
    content.push_str(&"\n");
}

/// Log for debugging information
///
/// This object defines the format and concatenate the debugging informations
impl SolverLog {
    pub fn new(path: &str) -> Self {
        let mut file = File::create(path).unwrap();

        let mut content = String::new();
        content.push_str(&"Runner informations\n");
        content.push_str(&"===================\n\n");

        #[cfg(feature = "additional_log_info")]
        {
            write_user_infos(&mut content);
            write_rustc_info(&mut content);
        }

        const VERSION: &str = env!("CARGO_PKG_VERSION");
        content.push_str("newton_rootfinder version: ");
        content.push_str(VERSION);
        content.push_str(&"\n");

        #[cfg(feature = "additional_log_info")]
        write_time(&mut content);

        content.push_str(&"\n");

        write!(file, "{}", content).unwrap();

        SolverLog {
            path: path.to_string(),
        }
    }

    pub fn add_content(&self, new_content: &str) {
        let mut file = OpenOptions::new().append(true).open(&self.path).unwrap();
        write!(file, "{}", new_content).unwrap();
    }

    pub fn add_parameters(
        &self,
        solver_parameters: &str,
        iteratives_config: &str,
        residuals_config: &str,
    ) {
        self.add_content(solver_parameters);
        self.add_content(iteratives_config);
        self.add_content(residuals_config);
    }

    pub fn add_damping<D>(
        &self,
        iteratives: &nalgebra::OVector<f64, D>,
        residuals: &ResidualsValues<D>,
        errors: &nalgebra::OVector<f64, D>,
    ) where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    {
        let mut iteration_log_header = String::new();
        iteration_log_header.push_str(&"Damping activated !\n\n".to_string());
        self.add_content(&iteration_log_header);
        self.add_iteration(iteratives, residuals, errors);
    }
    pub fn add_new_iteration<D>(
        &self,
        iteratives: &nalgebra::OVector<f64, D>,
        residuals: &ResidualsValues<D>,
        errors: &nalgebra::OVector<f64, D>,
        iter: usize,
    ) where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    {
        let mut iteration_log_header = String::new();
        iteration_log_header.push_str(SEPARATION_ITER);
        iteration_log_header.push_str(&format!("Iteration: {}\n\n", iter.to_string()));
        self.add_content(&iteration_log_header);
        self.add_iteration(iteratives, residuals, errors);
    }

    fn add_iteration<D>(
        &self,
        iteratives: &nalgebra::OVector<f64, D>,
        residuals: &ResidualsValues<D>,
        errors: &nalgebra::OVector<f64, D>,
    ) where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, D>,
    {
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
}
