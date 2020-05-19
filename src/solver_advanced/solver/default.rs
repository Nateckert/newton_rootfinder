use std::fmt;

extern crate nalgebra;

use super::RootFinder;
use crate::solver_advanced::iteratives;
use crate::solver_advanced::iteratives::Iterative;

use crate::solver_advanced::residuals;

/// Default function to create a solver with default parameters such as :
/// - max_iter = 50
/// - tolerance = 1e-6
/// - default residuals configuration thanks to `ResidualsConfig::default_with_size()`
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
///
///
/// fn main() {
///   let problem_size = 1;
///   let init_guess_fd = nalgebra::DVector::from_vec(vec![1.0]);
///   let vec_iter_params_fd = iteratives::default_vec_iteratives_fd(problem_size);
///   let iter_params_fd = iteratives::Iteratives::new(&vec_iter_params_fd);
///
///   let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
///   let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
///   let iter_params = iteratives::Iteratives::new(&vec_iter_params);
///
///   let mut rf_fd = nrf::solver::default_with_guess(init_guess_fd, iter_params_fd);
///   let mut rf = nrf::solver::default_with_guess(init_guess, iter_params);
/// }
/// ```
pub fn default_with_guess<'a, T>(
    initial_guess: nalgebra::DVector<f64>,
    iters_params: iteratives::Iteratives<'a, T>,
    residuals_config: residuals::ResidualsConfig<'a>,
) -> RootFinder<'a, T>
where
    T: Iterative + fmt::Display,
{
    let problem_size = initial_guess.len();
    let tolerance: f64 = 1e-6;
    let max_iter: usize = 50;

    RootFinder::new(
        initial_guess,
        iters_params,
        residuals_config,
        problem_size,
        tolerance,
        max_iter,
    )
}
