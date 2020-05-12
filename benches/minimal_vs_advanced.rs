use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate nalgebra;

extern crate newton_rootfinder as nrf;
use nrf::solver_minimal::*;

fn square2(x: f64) -> f64 {
     x.powi(2)-2.0
}

fn square2_nalg(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
     let mut y = x * x;
     y[0] -= 2.0;
     y
}

fn dsquare(x: f64) -> f64 {
     2.0*x
}


fn solvers_comparison(c: &mut Criterion) {

    // Test 1: evaluate functions evaluation
    // Results : f64 function is 137 times faster than DVector (expected)
    // f64 :  [784.26 ps 792.64 ps 802.17 ps]
    // nalg : [108.38 ns 109.31 ns 110.26 ns]

    let init_nalg = nalgebra::DVector::from_vec(vec![2.0]);

    let mut group_function = c.benchmark_group("Function evaluation");
    group_function.bench_function("f64", |b| b.iter(|| square2(black_box(2.0))));
    group_function.bench_function("nalgebra", |b| b.iter(|| square2_nalg(black_box(&init_nalg))));
    group_function.finish();


    // Test 2 : evaluate solvers
    // If the solver speed is driven by the function evaluation,
    // The time taken for resolution should be in the same proportion
    // as for the function evaluation test
    // Results :
    // Solver 1D :          [37.179 ns 37.587 ns 38.081 ns]
    // Solver 1D FD :       [59.780 ns 60.267 ns 60.767 ns]
    // Advanced solver FD : [20.313 us 20.483 us 20.651 us]
    // Without derivatives is 1.6 times faster than with
    // Minimal solver is 340 times faster than advanced solver
    // Expected times was 137 times
    // The advanced solver is roughly 2.5 times slower than the minimal implementation
    let problem_size = 1;
    let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
    let nrf = nrf::solver::RootFinderFD::default_with_guess(init_guess);
    let mut user_model =
           nrf::model_with_func::UserModelWithFunc::new(problem_size, square2_nalg);

    let mut group_solver = c.benchmark_group("Solver");
    group_solver.bench_function("Solver 1D", |b| b.iter(|| solver1d(1.0, square2, dsquare, 50, 1e-6)));
    group_solver.bench_function("Solver 1D FD", |b| b.iter(|| solver1d_fd(1.0, square2, 50, 1e-6, 1e-8)));
    group_solver.bench_function("Advanced solver FD", |b| b.iter(|| nrf.solve(&mut user_model)));
    group_solver.finish();
}

criterion_group!(benches, solvers_comparison);
criterion_main!(benches);
