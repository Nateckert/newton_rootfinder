//! Cost of the linear algebra of an iteration
//!
//! The function of the Broyden test cases 5 to 8 is defined for any dimension:
//! with its jacobian provided, an evaluation of the model costs O(n) for the residuals
//! and O(n^2) for the jacobian, so the O(n^3) of the linear algebra dominates as n grows.
//!
//! Results in `benches/RESULTS.md`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use newton_rootfinder as nrf;
use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals;
use nrf::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};
use util::test_cases::broyden1965::{broyden1965_case8, broyden1965_case8_jac};

const DIMENSIONS: [usize; 5] = [5, 10, 20, 50, 100];

fn bench_method(c: &mut Criterion, name: &str, resolution_method: ResolutionMethod) {
    let mut group = c.benchmark_group(format!("Jacobian provided/{name}"));

    for problem_size in DIMENSIONS {
        let init = nalgebra::DVector::from_element(problem_size, -1.0);
        let vec_iter_params = iteratives::default_vec_iteratives(problem_size);
        let iter_params = iteratives::Iteratives::new(&vec_iter_params);
        let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
        let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
        let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
        let mut rf = nrf::solver::default(&iter_params, &res_config, resolution_method, false);
        let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(
            problem_size,
            broyden1965_case8,
            broyden1965_case8_jac,
        );

        // A method that would not converge would time its failure
        user_model.set_iteratives(&init);
        rf.solve(&mut user_model).unwrap();

        group.bench_function(BenchmarkId::from_parameter(problem_size), |b| {
            b.iter(|| {
                user_model.set_iteratives(&init);
                rf.solve(&mut user_model)
            })
        });
    }

    group.finish();
}

fn linear_algebra(c: &mut Criterion) {
    bench_method(c, "NR", ResolutionMethod::NewtonRaphson);
    bench_method(
        c,
        "SN",
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
    );
    bench_method(
        c,
        "BROY1_jac",
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
    );
    bench_method(
        c,
        "GRST2_jac",
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::GreenstadtSecondMethod,
        )),
    );
    bench_method(
        c,
        "BROY1_inv",
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
    );
}

criterion_group!(benches, linear_algebra);
criterion_main!(benches);
