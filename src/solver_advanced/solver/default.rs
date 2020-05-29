use std::fmt;

extern crate nalgebra;

use super::ResolutionMethod;
use super::RootFinder;
use super::SolverParameters;
use crate::solver_advanced::iteratives;
use crate::solver_advanced::iteratives::Iterative;

use crate::solver_advanced::residuals;

/// Default function to create a solver with default parameters
///
/// The default parameters are:
/// - max_iter = 50
/// - tolerance = 1e-6
/// - resolution_method = NewtonRaphson
/// - damping = false
///
/// This function works either for finite difference or not.
/// The difference between the two cases comes
/// from the construction of the vector of iteratives parameters
/// Either with `default_vec_iteratives_fd()` or `default_vec_iteratives()`
///
/// It is required to create the vector in the scope of the main
/// as the argument of `Iteratives::new()` takes a reference to a slice
/// This allows to of the iteratives variables defined either :
/// - at compile-time (through an array)
/// - at run-time (through the generation of a vector while parsing a configuration file)
///
///
/// ## Examples
///
/// ```
/// extern crate newton_rootfinder;
/// use newton_rootfinder::solver_advanced as nrf;
/// use nrf::iteratives;
/// use nrf::residuals;
/// use nrf::solver::ResolutionMethod;
///
///
/// fn main() {
///   let problem_size = 1;
///   let init_guess_fd = nalgebra::DVector::from_vec(vec![1.0]);
///   let vec_iter_params_fd = iteratives::default_vec_iteratives_fd(problem_size);
///   let iter_params_fd = iteratives::Iteratives::new(&vec_iter_params_fd);
///   let stopping_residuals_fd = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let update_methods_fd = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let res_config_fd = residuals::ResidualsConfig::new(&stopping_residuals_fd, &update_methods_fd);
///
///   let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
///   let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
///   let iter_params = iteratives::Iteratives::new(&vec_iter_params);
///   let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
///
///   let mut rf_fd = nrf::solver::default_with_guess(init_guess_fd, &iter_params_fd, &res_config_fd, ResolutionMethod::NewtonRaphson);
///   let mut rf = nrf::solver::default_with_guess(init_guess, &iter_params, &res_config, ResolutionMethod::NewtonRaphson);
/// }
/// ```
pub fn default_with_guess<'a, T>(
    initial_guess: nalgebra::DVector<f64>,
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
    resolution_method: ResolutionMethod,
) -> RootFinder<'a, T>
where
    T: Iterative + fmt::Display,
{
    let problem_size = initial_guess.len();
    let tolerance: f64 = 1e-6;
    let max_iter: usize = 50;
    let damping = false;
    let parameters = SolverParameters::new(
        problem_size,
        tolerance,
        max_iter,
        resolution_method,
        damping,
    );

    RootFinder::new(parameters, initial_guess, iters_params, residuals_config)
}
