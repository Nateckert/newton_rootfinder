//! Benchmarking results and history :
//!
//! Test 1: evaluate solver parsing
//!
//! Test 2 : evaluate solver speed with the different methods
//!
//! Current Results :
//!
//! Solver parsing :
//! - NewtonRaphson-FD:             [37.099 ns 37.350 ns 37.602 ns]
//! - StationaryNewton-FD:          [37.099 ns 37.350 ns 37.602 ns]
//! - BroydenFirstMethod_jac-FD:    [37.099 ns 37.350 ns 37.602 ns]
//! - BroydenSecondMethod_jac-FD:   [37.099 ns 37.350 ns 37.602 ns]
//!
//! Resolution speed:
//! - NewtonRaphson-FD:             [37.099 ns 37.350 ns 37.602 ns]
//! - StationaryNewton-FD:          [37.099 ns 37.350 ns 37.602 ns]
//! - BroydenFirstMethod_jac-FD:    [37.099 ns 37.350 ns 37.602 ns]
//! - BroydenSecondMethod_jac-FD:   [37.099 ns 37.350 ns 37.602 ns]
//!

use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate nalgebra;

extern crate newton_rootfinder;
use newton_rootfinder::solver_advanced as nrf;
use nrf::test_cases::broyden1965::{init_broyden1965_case8, broyden1965_case8};

fn solvers_comparison(c: &mut Criterion) {
    const FILEPATH_NR: &'static str = "./benches/data/broyden_case8_NR.xml";
    const FILEPATH_SN: &'static str = "./benches/data/broyden_case8_SN.xml";
    const FILEPATH_BFM_JAC: &'static str = "./benches/data/broyden_case8_BFM_jac.xml";
    const FILEPATH_BSM_JAC: &'static str = "./benches/data/broyden_case8_BSM_jac.xml";

    let mut group_function = c.benchmark_group("Solver parsing");
    group_function.bench_function("NR", |b| b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_NR)));
    group_function.bench_function("SN", |b| b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_SN)));
    group_function.bench_function("BFM_jac", |b| b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_BFM_JAC)));
    group_function.bench_function("BSM_jac", |b| b.iter(|| nrf::util::from_xml_finite_diff(&FILEPATH_BSM_JAC)));
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

    group_function.finish();
}

criterion_group!(benches, solvers_comparison);
criterion_main!(benches);
