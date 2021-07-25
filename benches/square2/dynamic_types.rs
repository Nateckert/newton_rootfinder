use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate nalgebra;

extern crate newton_rootfinder;
use newton_rootfinder as nrf;

fn square2_nalg(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut y = x * x;
    y[0] -= 2.0;
    y
}

fn dsquare2_nalg(x: &nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64> {
    let mut y = nalgebra::DMatrix::zeros(1, 1);
    y[(0, 0)] = 2.0 * x[0];
    y
}

fn run(c: &mut Criterion) {
    let init_nalg = nalgebra::DVector::from_vec(vec![2.0]);

    let mut group_function = c.benchmark_group("Function evaluation");
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

    let mut group_solver = c.benchmark_group("Advanced solver with dynamic types");
    group_solver.bench_function("Newton-Raphson with finite differences", |b| {
        b.iter(|| nrf.solve(&mut user_model))
    });
    group_solver.bench_function("Stationary Newton with finite differences", |b| {
        b.iter(|| nrf_stationary.solve(&mut user_model))
    });
    group_solver.bench_function("Newton-Raphson with derivative provided", |b| {
        b.iter(|| nrf_jac.solve(&mut user_model_jac))
    });
    group_solver.finish();
}

criterion_group!(benches, run);
criterion_main!(benches);
