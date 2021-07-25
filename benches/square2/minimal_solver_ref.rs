use criterion::{black_box, criterion_group, criterion_main, Criterion};

use util::solver_one_dimensional::{solver1d, solver1d_fd};

fn square2(x: f64) -> f64 {
    x.powi(2) - 2.0
}

fn dsquare(x: f64) -> f64 {
    2.0 * x
}

fn run(c: &mut Criterion) {

    let mut group_function = c.benchmark_group("Function evaluation");
    group_function.bench_function("f64", |b| b.iter(|| square2(black_box(2.0))));
    group_function.finish();

    let mut group_solver = c.benchmark_group("Minmal solver 1D");
    group_solver.bench_function("With derivative provided", |b| {
        b.iter(|| solver1d(
            black_box(1.0),
            black_box(square2),
            black_box(dsquare),
            black_box(50),
            black_box(1e-6)))
    });
    group_solver.bench_function("Finite differences", |b| {
        b.iter(|| solver1d_fd(
            black_box(1.0),
            black_box(square2),
            black_box(50),
            black_box(1e-6),
            black_box(5e-8)))
    });
}

criterion_group!(benches, run);
criterion_main!(benches);