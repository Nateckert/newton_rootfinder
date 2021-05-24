extern crate nalgebra;
extern crate newton_rootfinder;
use newton_rootfinder as nrf;

use crate::common::{run_function_case_fd, run_function_case_jac};

use nrf::solver::{QuasiNewtonMethod, ResolutionMethod};

use util::test_cases::polynom;

#[test]
fn square() {
    let problem_size = 1;
    let damping = false;
    run_function_case_fd(
        problem_size,
        polynom::square2,
        nalgebra::DVector::from_vec(vec![1.0]),
        nalgebra::DVector::from_vec(vec![2_f64.sqrt()]),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
        damping,
    );
}

#[test]
#[should_panic]
fn root_with_high_derivative() {
    let problem_size = 1;
    let damping = false;
    run_function_case_fd(
        problem_size,
        polynom::root_with_high_derivative,
        nalgebra::DVector::from_vec(vec![0.15]),
        nalgebra::DVector::from_vec(vec![0.1]),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
        damping,
    );
}

#[test]
fn square_jac() {
    let problem_size = 1;
    let damping = false;
    run_function_case_jac(
        problem_size,
        polynom::square2,
        polynom::dsquare,
        nalgebra::DVector::from_vec(vec![1.0]),
        nalgebra::DVector::from_vec(vec![2_f64.sqrt()]),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
        damping,
    );
}

#[test]
#[should_panic]
fn root_with_high_derivative_jac() {
    let problem_size = 1;
    let damping = false;
    run_function_case_jac(
        problem_size,
        polynom::root_with_high_derivative,
        polynom::root_with_high_derivative_jac,
        nalgebra::DVector::from_vec(vec![0.15]),
        nalgebra::DVector::from_vec(vec![0.1]),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
        damping,
    );
}
