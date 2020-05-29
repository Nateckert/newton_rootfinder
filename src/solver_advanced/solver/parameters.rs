use std::fmt;

/// Choice of the iterative algorithm for the resolution
///
/// All of them are Newton based methods : (Newton or quasi-Newton)
///
/// All Newton-based iterative methods have a local convergence.
/// They also assume that the jacobian is invertible at the root (simple root)
///
/// ## Newton-Raphson
/// The classical Newton method.
///
/// Requires a full jacobian evaluation at each iteration step
///
/// Reference:
///
/// Tjalling J. Ypma
///
/// Historical development of the Newton–Raphson method,
///
/// SIAM Review 37 (4), p 531–551, 1995.
///
/// doi:10.1137/1037125.
///
/// The convergence rate is quadratic :
/// || x_{n+1} - x_sol || < || x_{n} - x_sol ||^2
///
/// ## Stationary Newton
/// A quasi Newton Method requiring the evaluation of the jacobian only at the first iteration step.
///
/// The jacobian of the first iteration is used for all the updates
///
/// The convergence rate is locally linear and controlled by the first error :
///
/// || x_{n+1} - x_sol || < || x_{n} - x_sol ||*|| x_{0} - x_sol ||
///
/// Reference:
///
/// Dennis, Jr., J. E. (1967)
///
/// A Stationary Newton Method for Nonlinear Functional Equations
///
/// SIAM Journal on Numerical Analysis, 4(2), p 222–232.
///
/// doi:10.1137/0704021
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ResolutionMethod {
    NewtonRaphson,
    StationaryNewton,
}

impl fmt::Display for ResolutionMethod {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let result = match self {
            ResolutionMethod::NewtonRaphson => &"Newton-Raphson",
            ResolutionMethod::StationaryNewton => &"Stationary Newton",
        };

        write!(f, "{}", result)
    }
}

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
/// and the jacobian has not been recently updated,
/// damping won't be performed but the jacobian will be recomputed at the next iteration.
///
/// In the case of the jacobian has been recomputed at the previous iteration,
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
        content.push_str(&"|\n");
        content.push_str(separation_line);
        content.push_str(&"\n");

        write!(f, "{}", content)
    }
}
