//! Benchmarking results and history :
//!
//! Test 1: evaluate functions evaluation (no changes over time expected)
//! Results : f64 function is 137 times faster than DVector (expected)
//! f64 :  [784.26 ps 792.64 ps 802.17 ps]
//! nalg : [108.38 ns 109.31 ns 110.26 ns]
//!
//! Test 2 : evaluate solvers
//! If the solver speed is driven by the function evaluation,
//! The time taken for resolution should be in the same proportion
//! as for the function evaluation test
//!
//! Reference results :
//! Solver 1D:                              [37.099 ns 37.350 ns 37.602 ns]
//! Solver 1D FD:                           [60.788 ns 61.595 ns 62.524 ns]
//! Advanced solver FD:                     [686.11 ns 691.50 ns 697.20 ns]
//! Advanced solver FD StationaryNewton :   [712.83 ns 719.00 ns 725.14 ns]
//! Advanced solver FD jacobian provided :  [718.22 ns 724.01 ns 729.89 ns]
//!
//! Without derivatives is 1.6 times faster than with
//! Minimal solver is 11 times faster than advanced solver
//! Expected times was 137 times
//! The advanced solver is roughly 10 times faster than the minimal implementation
//!
//!

use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate nalgebra;

extern crate newton_rootfinder;
use newton_rootfinder as nrf;
use util::solver_one_dimensional::{solver1d, solver1d_fd};

fn square2(x: f64) -> f64 {
    x.powi(2) - 2.0
}

fn square2_nalg(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut y = x * x;
    y[0] -= 2.0;
    y
}

fn dsquare(x: f64) -> f64 {
    2.0 * x
}

fn dsquare2_nalg(x: &nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64> {
    let mut y = nalgebra::DMatrix::zeros(1, 1);
    y[(0, 0)] = 2.0 * x[0];
    y
}

fn solvers_comparison(c: &mut Criterion) {
    let init_nalg = nalgebra::DVector::from_vec(vec![2.0]);

    let mut group_function = c.benchmark_group("Function evaluation");
    group_function.bench_function("f64", |b| b.iter(|| square2(black_box(2.0))));
    group_function.bench_function("nalgebra", |b| {
        b.iter(|| square2_nalg(black_box(&init_nalg)))
    });
    group_function.finish();

    let problem_size = 1;
    let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
    let vec_iter_params = nrf::iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = nrf::iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
    let res_config = nrf::residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let damping = false;
    let mut nrf = nrf::solver::default_with_guess(
        init_guess.clone(),
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    let mut nrf_stationary = nrf::solver::default_with_guess(
        init_guess.clone(),
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::QuasiNewton(
            nrf::solver::QuasiNewtonMethod::StationaryNewton,
        ),
        damping,
    );
    let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, square2_nalg);

    let vec_iter_params_jac = nrf::iteratives::default_vec_iteratives(problem_size);
    let iter_params_jac = nrf::iteratives::Iteratives::new(&vec_iter_params_jac);
    let stopping_residuals_jac = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
    let update_methods_jac = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
    let res_config_jac =
        nrf::residuals::ResidualsConfig::new(&stopping_residuals_jac, &update_methods_jac);
    let mut nrf_jac = nrf::solver::default_with_guess(
        init_guess.clone(),
        &iter_params_jac,
        &res_config_jac,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );
    let mut user_model_jac = nrf::model::UserModelFromFunctionAndJacobian::new(
        problem_size,
        square2_nalg,
        dsquare2_nalg,
    );

    let mut group_solver = c.benchmark_group("Solver");
    group_solver.bench_function("Solver 1D", |b| {
        b.iter(|| solver1d(1.0, square2, dsquare, 50, 1e-6))
    });
    group_solver.bench_function("Solver 1D FD", |b| {
        b.iter(|| solver1d_fd(1.0, square2, 50, 1e-6, 1e-8))
    });
    group_solver.bench_function("Advanced solver FD", |b| {
        b.iter(|| nrf.solve(&mut user_model))
    });
    group_solver.bench_function("Advanced solver FD Stationary Newton", |b| {
        b.iter(|| nrf_stationary.solve(&mut user_model))
    });
    group_solver.bench_function("Advanced solver jacobian provided", |b| {
        b.iter(|| nrf_jac.solve(&mut user_model_jac))
    });
    group_solver.finish();
}

criterion_group!(benches, solvers_comparison);
criterion_main!(benches);
