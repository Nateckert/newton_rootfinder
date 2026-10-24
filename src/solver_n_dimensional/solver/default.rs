use std::fmt;

use super::ResolutionMethod;
use super::RootFinder;
use super::SolverParameters;
use crate::iteratives::Iterative;

use crate::iteratives;
use crate::residuals;

/// Default function to create a solver with default parameters
///
/// The default parameters are:
/// - max_iter = 50
/// - tolerance = 1e-6
/// - resolution_method = NewtonRaphson
/// - damping = false
///
/// This function works either for finite difference or not.
///
/// The difference between the two cases comes
/// from the construction of the vector of iteratives parameters
/// Either with `default_vec_iteratives_fd()` or `default_vec_iteratives()`
///
/// It is required to create the vector in the scope of the calling function
/// as `Iteratives::new()` takes as argument a reference to a slice
///
/// This allows to of the iteratives variables defined either :
/// - at compile-time (through an array)
/// - at run-time (through the generation of a vector while parsing a configuration file)
///
/// The resolution starts from the iteratives the model holds when `solve` is called,
/// and the dimension type `D` is the one of that model.
///
/// ## Examples
///
/// ```
/// use newton_rootfinder as nrf;
/// use nrf::iteratives;
/// use nrf::model::Model;
/// use nrf::residuals;
/// use nrf::solver::ResolutionMethod;
///
/// /// Equation : x**2 - 2 = 0
/// fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
///     let mut y = x * x;
///     y[0] -= 2.0;
///     y
/// }
///
/// fn main() {
///   let problem_size = 1;
///   let vec_iter_params_fd = iteratives::default_vec_iteratives_fd(problem_size);
///   let iter_params_fd = iteratives::Iteratives::new(&vec_iter_params_fd);
///   let stopping_residuals_fd = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let update_methods_fd = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let res_config_fd = residuals::ResidualsConfig::new(&stopping_residuals_fd, &update_methods_fd);
///
///   let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
///   let iter_params = iteratives::Iteratives::new(&vec_iter_params);
///   let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
///   let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
///   let damping = false;
///
///   // The dimension of a solver is the one of the model it solves
///   let mut rf_fd = nrf::solver::default(&iter_params_fd, &res_config_fd, ResolutionMethod::NewtonRaphson, damping);
///   let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, square2);
///   user_model.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
///   rf_fd.solve(&mut user_model).unwrap();
///
///   // Or is named, for a solver not yet given a model
///   let mut rf: nrf::solver::RootFinder<iteratives::IterativeParams, nalgebra::Dyn> =
///       nrf::solver::default(&iter_params, &res_config, ResolutionMethod::NewtonRaphson, damping);
/// }
/// ```
pub fn default<'a, T, D>(
    iters_params: &'a iteratives::Iteratives<'a, T>,
    residuals_config: &'a residuals::ResidualsConfig<'a>,
    resolution_method: ResolutionMethod,
    damping: bool,
) -> RootFinder<'a, T, D>
where
    T: Iterative + fmt::Display + fmt::Debug,
    D: nalgebra::DimMin<D, Output = D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<nalgebra::U1, D>,
    nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D, D>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    let tolerance: f64 = 1e-6;
    let max_iter: usize = 50;
    let parameters = SolverParameters::new(tolerance, max_iter, resolution_method, damping);

    RootFinder::new(parameters, iters_params, residuals_config)
}
