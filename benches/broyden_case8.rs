//! Benchmarking results and history :
//!
//! Test 1: evaluate solver parsing
//!
//! Test 2 : evaluate solver speed with the different methods
//!
//! Current Results :
//!
//! Solver parsing :
//! - NewtonRaphson-FD:             [420.92 us 429.07 us 440.91 us]
//! - StationaryNewton-FD:          [414.56 us 417.16 us 421.47 us]
//! - BroydenFirstMethod_jac-FD:    [417.81 us 423.92 us 432.89 us]
//! - BroydenSecondMethod_jac-FD:   [416.62 us 422.68 us 430.58 us]
//!
//! Resolution speed:
//! - NewtonRaphson-FD:             [921.32 ns 926.58 ns 932.06 ns]
//! - StationaryNewton-FD:          [916.82 ns 923.32 ns 929.80 ns]
//! - BroydenFirstMethod_jac-FD:    [900.14 ns 905.07 ns 909.85 ns]
//! - BroydenSecondMethod_jac-FD:   [897.66 ns 903.92 ns 910.21 ns]  
//!

use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate nalgebra;

extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;
use nrf::test_cases::broyden1965::{broyden1965_case8, init_broyden1965_case8};

fn solvers_comparison(c: &mut Criterion) {
    const FILEPATH_NR: &'static str = "./benches/data/broyden_case8_NR.xml";
    const FILEPATH_SN: &'static str = "./benches/data/broyden_case8_SN.xml";
    const FILEPATH_BROY1_JAC: &'static str = "./benches/data/broyden_case8_BROY1_jac.xml";
    const FILEPATH_BROY2_JAC: &'static str = "./benches/data/broyden_case8_BROY2_jac.xml";

    let mut group_function = c.benchmark_group("Solver parsing");
    group_function.bench_function("NR", |b| {
        b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_NR))
    });
    group_function.bench_function("SN", |b| {
        b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_SN))
    });
    group_function.bench_function("BROY1_jac", |b| {
        b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_BROY1_JAC))
    });
    group_function.bench_function("BROY2_jac", |b| {
        b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_BROY2_JAC))
    });
    group_function.finish();

    let mut group_function = c.benchmark_group("Solver speed");

    // Newton Raphson method
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::util::from_xml_finite_diff(&FILEPATH_NR);

    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
    let residuals_config =
        nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
    let problem_size = solver_parameters.get_problem_size();

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        init_broyden1965_case8(),
        &iteratives,
        &residuals_config,
    );

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("NR", |b| b.iter(|| rf.solve(&mut user_model)));

    // Stationary Newton method
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::util::from_xml_finite_diff(&FILEPATH_SN);

    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
    let residuals_config =
        nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
    let problem_size = solver_parameters.get_problem_size();

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        init_broyden1965_case8(),
        &iteratives,
        &residuals_config,
    );

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("SN", |b| b.iter(|| rf.solve(&mut user_model)));


    // First Broyden method on jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::util::from_xml_finite_diff(&FILEPATH_BROY1_JAC);

    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
    let residuals_config =
        nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
    let problem_size = solver_parameters.get_problem_size();

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        init_broyden1965_case8(),
        &iteratives,
        &residuals_config,
    );

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("BROY1_jac", |b| b.iter(|| rf.solve(&mut user_model)));


    // Second Broyden method on jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::util::from_xml_finite_diff(&FILEPATH_BROY2_JAC);

    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_vec);
    let residuals_config =
        nrf::residuals::ResidualsConfig::new(&stopping_criterias, &update_methods);
    let problem_size = solver_parameters.get_problem_size();

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        init_broyden1965_case8(),
        &iteratives,
        &residuals_config,
    );

    let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("BROY2_jac", |b| b.iter(|| rf.solve(&mut user_model)));

    group_function.finish();
}

criterion_group!(benches, solvers_comparison);
criterion_main!(benches);
