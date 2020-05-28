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
pub enum ResolutionMethod {
    NewtonRaphson,
    StationaryNewton,
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
/// ## Problem size
/// The dimension of the problem for the resolution
pub struct SolverParameters {
    problem_size: usize,
    tolerance: f64,
    max_iter: usize,
    damping: bool,
}

impl SolverParameters {
    pub fn new(problem_size: usize, tolerance: f64, max_iter: usize, damping: bool) -> Self {
        SolverParameters {
            problem_size,
            tolerance,
            max_iter,
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

    pub fn get_damping(&self) -> bool {
        self.damping
    }
}
