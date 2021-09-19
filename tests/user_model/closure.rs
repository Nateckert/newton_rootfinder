use crate::common::{run_closure_case_fd, run_closure_case_jac};
use newton_rootfinder as nrf;

#[test]
fn solve_with_closure() {
    let square_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> {
        let mut result = iteratives * iteratives;
        result[0] -= 2.0;
        result
    };

    let problem_size = 1;
    let damping = false;
    let init = nalgebra::DVector::from_vec(vec![1.0]);
    let solution = nalgebra::DVector::from_vec(vec![std::f64::consts::SQRT_2]);

    run_closure_case_fd(
        problem_size,
        &square_closure,
        init,
        solution,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
}

#[test]
fn solve_with_closure_and_jacobian() {
    let square_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> {
        let mut result = iteratives * iteratives;
        result[0] -= 2.0;
        result
    };
    let derivative_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DMatrix<f64> {
        let mut y = nalgebra::DMatrix::zeros(1, 1);
        y[(0, 0)] = 2.0 * iteratives[0];
        y
    };

    let problem_size = 1;
    let damping = false;
    let init = nalgebra::DVector::from_vec(vec![1.0]);
    let solution = nalgebra::DVector::from_vec(vec![std::f64::consts::SQRT_2]);

    run_closure_case_jac(
        problem_size,
        &square_closure,
        &derivative_closure,
        init,
        solution,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
}
