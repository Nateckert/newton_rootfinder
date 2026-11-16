//! The step is the solution of `jacobian * step = -residuals`, computed with
//! the LU factorization of the jacobian: its inverse is formed only by the
//! methods updating it.

use newton_rootfinder as nrf;

use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals;
use nrf::solver::ResolutionMethod;

const SIZE: usize = 10;

/// The Hilbert matrix, the textbook ill-conditioned matrix:
/// its condition number is 1.6e13 for 10 unknowns.
fn hilbert() -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_fn(SIZE, SIZE, |i, j| 1.0 / ((i + j + 1) as f64))
}

/// hilbert * x = hilbert * (1, ..., 1)
fn ill_conditioned_linear_system(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    hilbert() * x - hilbert() * nalgebra::DVector::from_element(SIZE, 1.0)
}

fn ill_conditioned_linear_system_jac(_x: &nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64> {
    hilbert()
}

/// A linear system is solved by the first Newton step.
/// The solution of a factorized system leaves residuals at the rounding level whatever the
/// conditioning, when the product by the inverse leaves them between 2e-5 and 5e-5 here.
#[test]
fn an_ill_conditioned_linear_system_is_solved_in_one_step() {
    let vec_iter_params = iteratives::default_vec_iteratives(SIZE);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; SIZE];
    let update_methods = vec![residuals::NormalizationMethod::Abs; SIZE];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);

    let tolerance = 1e-12;
    let max_iter = 1;
    let solver_parameters = nrf::solver::SolverParameters::new(
        tolerance,
        max_iter,
        ResolutionMethod::NewtonRaphson,
        false,
    );
    let mut rf = nrf::solver::RootFinder::new(solver_parameters, &iter_params, &res_config);

    let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
        SIZE,
        ill_conditioned_linear_system,
        ill_conditioned_linear_system_jac,
    );
    user_model.set_iteratives(&nalgebra::DVector::zeros(SIZE));

    rf.solve(&mut user_model).unwrap();
}
