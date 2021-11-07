use newton_rootfinder as nrf;

use nrf::iteratives;
use nrf::residuals;

fn cannot_converge(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let n = x.len();
    let mut outputs = nalgebra::DVector::zeros(n);

    for i in 0..n {
        outputs[i] = x[i].cos() + 10.0; // cannot be zero
    }

    outputs
}

#[test]
fn non_convergence_case() {
    let problem_size = 5;
    let init = nalgebra::DVector::zeros(problem_size);

    let damping = false;

    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default_with_guess(
        init,
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, cannot_converge);

    let result = rf.solve(&mut user_model).unwrap_err();
    let expected: nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dynamic> =
        nrf::errors::SolverError::NonConvergenceError;
    assert_eq!(expected.to_string(), result.to_string());
}
