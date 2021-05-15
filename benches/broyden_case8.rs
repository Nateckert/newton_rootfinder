//! Benchmarking results and history :
//!
//! Test 1: evaluate solver parsing
//!
//! Test 2 : evaluate solver speed with the different methods
//!
//! Reference results :
//!
//! Solver parsing :
//! - NewtonRaphson-FD:             [397.90 us 399.15 us 400.39 us]
//!
//! Resolution speed:
//! - NewtonRaphson-FD:             [812.33 ns 818.59 ns 825.31 ns]
//! - StationaryNewton-FD:          [829.89 ns 836.45 ns 843.19 ns]
//! - BroydenFirstMethod-FD:        [828.63 ns 834.22 ns 839.66 ns]
//! - BroydenSecondMethod-FD:       [814.20 ns 820.57 ns 827.17 ns]
//! - BroydenFirstMethod_INV-FD:    [819.56 ns 827.18 ns 835.29 ns]
//! - BroydenSecondMethod_INV-FD:   [826.59 ns 831.91 ns 837.23 ns]
//!

use criterion::{criterion_group, criterion_main, Criterion};

extern crate nalgebra;

extern crate newton_rootfinder;
use newton_rootfinder as nrf;
use util::test_cases::broyden1965::{broyden1965_case8, init_broyden1965_case8};

fn solvers_comparison(c: &mut Criterion) {
    const FILEPATH_NR: &'static str = "./benches/data/broyden_case8_NR.xml";
    const FILEPATH_SN: &'static str = "./benches/data/broyden_case8_SN.xml";
    const FILEPATH_BROY1_JAC: &'static str = "./benches/data/broyden_case8_BROY1.xml";
    const FILEPATH_BROY2_JAC: &'static str = "./benches/data/broyden_case8_BROY2.xml";
    const FILEPATH_BROY1_INV: &'static str = "./benches/data/broyden_case8_BROY1_INV.xml";
    const FILEPATH_BROY2_INV: &'static str = "./benches/data/broyden_case8_BROY2_INV.xml";
    const FILEPATH_GRST1_JAC: &'static str = "./benches/data/broyden_case8_GRST1.xml";
    const FILEPATH_GRST2_JAC: &'static str = "./benches/data/broyden_case8_GRST2.xml";
    const FILEPATH_GRST1_INV: &'static str = "./benches/data/broyden_case8_GRST1_INV.xml";
    const FILEPATH_GRST2_INV: &'static str = "./benches/data/broyden_case8_GRST2_INV.xml";

    let mut group_function = c.benchmark_group("Solver parsing");
    group_function.bench_function("NR", |b| {
        b.iter(|| nrf::xml_parser::from_xml_finite_diff(&FILEPATH_NR))
    });

    group_function.finish();

    let mut group_function = c.benchmark_group("Solver speed");

    // Newton Raphson method
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_NR);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("NR", |b| b.iter(|| rf.solve(&mut user_model)));

    // Stationary Newton method
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_SN);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("SN", |b| b.iter(|| rf.solve(&mut user_model)));

    // First Broyden method on jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_BROY1_JAC);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("BROY1_jac", |b| b.iter(|| rf.solve(&mut user_model)));

    // Second Broyden method on jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_BROY2_JAC);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("BROY2_jac", |b| b.iter(|| rf.solve(&mut user_model)));

    // First Broyden method on inverse jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_BROY1_INV);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("BROY1_inv", |b| b.iter(|| rf.solve(&mut user_model)));

    // Second Broyden method on inverse jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_BROY2_INV);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("BROY2_inv", |b| b.iter(|| rf.solve(&mut user_model)));

    // First Greenstad method on jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_GRST1_JAC);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("GRST1", |b| b.iter(|| rf.solve(&mut user_model)));

    // Second Greenstad method on jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_GRST2_JAC);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("GRST2", |b| b.iter(|| rf.solve(&mut user_model)));

    // First Greenstad method on inverse jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_GRST1_INV);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("GRST1_inv", |b| b.iter(|| rf.solve(&mut user_model)));

    // Second Greenstad method on inverse jacobian
    let (solver_parameters, iteratives_vec, stopping_criterias, update_methods) =
        nrf::xml_parser::from_xml_finite_diff(&FILEPATH_GRST2_INV);

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

    let mut user_model = nrf::model::UserModelFromFunc::new(problem_size, broyden1965_case8);
    group_function.bench_function("GRST2_inv", |b| b.iter(|| rf.solve(&mut user_model)));

    group_function.finish();
}

criterion_group!(benches, solvers_comparison);
criterion_main!(benches);
