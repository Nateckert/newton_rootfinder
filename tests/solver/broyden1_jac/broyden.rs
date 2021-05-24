extern crate newton_rootfinder;
use newton_rootfinder as nrf;
use util::test_cases::broyden1965::*;

use nrf::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

use crate::common::{run_function_case_fd, run_function_case_jac};

#[test]
fn broyden_case5_fd() {
    let problem_size = 5;
    let damping = false;
    run_function_case_fd(
        problem_size,
        broyden1965_case5,
        init_broyden1965_case5(),
        solution_broyden1965_case5(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case5_jac() {
    let problem_size = 5;
    let damping = false;
    run_function_case_jac(
        problem_size,
        broyden1965_case5,
        broyden1965_case5_jac,
        init_broyden1965_case5(),
        solution_broyden1965_case5(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case6_fd() {
    let problem_size = 5;
    let damping = false;
    run_function_case_fd(
        problem_size,
        broyden1965_case6,
        init_broyden1965_case6(),
        solution_broyden1965_case6(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case6_jac() {
    let problem_size = 5;
    let damping = false;
    run_function_case_jac(
        problem_size,
        broyden1965_case6,
        broyden1965_case6_jac,
        init_broyden1965_case6(),
        solution_broyden1965_case6(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case7_fd() {
    let problem_size = 10;
    let damping = false;
    run_function_case_fd(
        problem_size,
        broyden1965_case7,
        init_broyden1965_case7(),
        solution_broyden1965_case7(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case7_jac() {
    let problem_size = 10;
    let damping = false;
    run_function_case_jac(
        problem_size,
        broyden1965_case7,
        broyden1965_case7_jac,
        init_broyden1965_case7(),
        solution_broyden1965_case7(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case8_fd() {
    let problem_size = 20;
    let damping = true;
    run_function_case_fd(
        problem_size,
        broyden1965_case8,
        init_broyden1965_case8(),
        solution_broyden1965_case8(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case8_jac() {
    let problem_size = 20;
    let damping = true;
    run_function_case_jac(
        problem_size,
        broyden1965_case8,
        broyden1965_case8_jac,
        init_broyden1965_case8(),
        solution_broyden1965_case8(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case9_fd() {
    let problem_size = 2;
    let damping = true;
    run_function_case_fd(
        problem_size,
        broyden1965_case9,
        init_broyden1965_case9(),
        solution_broyden1965_case9(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

#[test]
fn broyden_case9_jac() {
    let problem_size = 2;
    let damping = true;
    run_function_case_jac(
        problem_size,
        broyden1965_case9,
        broyden1965_case9_jac,
        init_broyden1965_case9(),
        solution_broyden1965_case9(),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        damping,
    );
}

//#[test]
//#[should_panic] // This test can panic, see file src/test_cases/broyden1965
//fn broyden_case10_fd() {
//    let problem_size = 2;
//    run_function_case_fd(
//        problem_size,
//        broyden1965_case10,
//        init_broyden1965_case10(),
//        solution_broyden1965_case10(),
//        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::BroydenFirstMethod)),
//    );
//}

//#[test]  // This test can panic, see file src/test_cases/broyden1965
//fn broyden_case10_jac() {
//    let problem_size = 2;
//    run_function_case_jac(
//        problem_size,
//        broyden1965_case10,
//        broyden1965_case10_jac,
//        init_broyden1965_case10(),
//        solution_broyden1965_case10(),
//        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::BroydenFirstMethod)),
//    );
//}
