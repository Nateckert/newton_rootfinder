use super::ResolutionMethod;
use std::fmt;

/// A minimal struct holding the resolution parameters
///
/// # Parameters
/// ## Damping
/// Activate the damping to improve convergence
///
/// Plain resolution according to Newton is made through the formula
/// X = X - J^-1*F(X)
///
/// However, if the proposed update is not performing (deterioriating the solution)
/// it is likely it is due to a much to step-size too important.
/// Reducing the step-size might be the solution
///
/// The damping formula is then :
/// X = X - damping_factor*J^-1*F(X)
/// with 0 < damping_factor <= 1
///
/// As long as the error is reduced damping_factor = 1.
/// If it is not the case, a factor is applied
/// (the value might change according to the versions).
///
/// If the used method is a quasi-newton method
/// and the jacobian has not been updated at the current iteration,
/// damping won't be performed but the jacobian will be recomputed at the next iteration.
///
/// In the case of the jacobian has been recomputed at the current iteration,
/// damping will be performed
///
/// ## Tolerance
/// The tolerance values used by the solver to check for convergence.
///
/// Each residuals must be below this threshold
///
/// ## Max iteration
/// The maximum number of iterations the solver is allowed to make
///
/// This is required to avoid to have an infinte loop
///
/// ## Resolution Method
/// See the enum `ResolutionMethod` for the options available
///
/// ## Problem size
/// The dimension of the problem for the resolution
pub struct SolverParameters {
    problem_size: usize,
    tolerance: f64,
    max_iter: usize,
    resolution_method: ResolutionMethod,
    damping: bool,
}

impl SolverParameters {
    pub fn new(
        problem_size: usize,
        tolerance: f64,
        max_iter: usize,
        resolution_method: ResolutionMethod,
        damping: bool,
    ) -> Self {
        SolverParameters {
            problem_size,
            tolerance,
            max_iter,
            resolution_method,
            damping,
        }
    }

    pub fn get_problem_size(&self) -> usize {
        self.problem_size
    }

    pub fn get_tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    pub fn get_resolution_method(&self) -> ResolutionMethod {
        self.resolution_method
    }

    pub fn get_damping(&self) -> bool {
        self.damping
    }
}

impl fmt::Display for SolverParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut content = String::from("Solver parameters\n");
        content.push_str("=================\n\n");
        let separation_line = &"+----------------+-----------------+-----------------+--------------------+---------------------+\n";
        let header          = &"|  Problem size  |  Max iteration  |    Tolerance    |  Damping activated |  Resolution Method  |\n";

        content.push_str(separation_line);
        content.push_str(header);
        content.push_str(separation_line);
        content.push_str(&format!(
            "| {:width$}",
            self.problem_size.to_string(),
            width = 15
        ));
        content.push_str(&format!(
            "| {:width$}",
            self.max_iter.to_string(),
            width = 17
        ));
        content.push_str(&format!(
            "| {:width$}",
            self.tolerance.to_string(),
            width = 15
        ));
        content.push_str(&format!(
            "| {:width$}",
            self.damping.to_string(),
            width = 19
        ));
        content.push_str(&format!(
            "| {:width$}",
            self.resolution_method.to_string(),
            width = 20
        ));
        content.push_str("|\n");
        content.push_str(separation_line);
        content.push('\n');

        write!(f, "{}", content)
    }
}

impl fmt::Debug for SolverParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Solver parameters")
            .field("Problem size", &self.problem_size)
            .field("Max iteration", &self.max_iter)
            .field("Solver tolerance", &self.tolerance)
            .field("Resolution method", &self.resolution_method)
            .field("Damping activated", &self.damping)
            .finish()
    }
}
