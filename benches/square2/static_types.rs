use std::error::Error;
use std::fmt;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use newton_rootfinder as nrf;

use nrf::{
    model::{Model, ModelError},
    residuals::NormalizationMethod,
};

/// x**2 - 2 = 0
/// Root: x = 2.sqrt() approx 1.4142
pub fn square2(x: &nalgebra::SVector<f64, 1>) -> nalgebra::SVector<f64, 1> {
    let y = nalgebra::SVector::<f64, 1>::new(x[0] * x[0] - 2.0);
    y
}

pub struct UserModel {
    iteratives: nalgebra::SVector<f64, 1>,
    output: nalgebra::SVector<f64, 1>,
}

impl UserModel {
    fn new(init: f64) -> Self {
        let iteratives = nalgebra::SVector::<f64, 1>::new(init);
        let output = square2(&iteratives);

        UserModel { iteratives, output }
    }
}

#[derive(Debug)]
pub struct MyCustomErrors;
impl fmt::Display for MyCustomErrors {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", "Not a good value")
    }
}

impl Error for MyCustomErrors {}

impl Model<nalgebra::Const<1>> for UserModel {
    type InaccurateValuesError = MyCustomErrors;
    type UnusableValuesError = MyCustomErrors;
    fn len_problem(&self) -> usize {
        1
    }
    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.iteratives = *iteratives;
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        self.iteratives
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Const<1>>> {
        self.output = square2(&self.iteratives);
        Ok(())
    }

    fn get_residuals(&self) -> nrf::residuals::ResidualsValues<nalgebra::Const<1>> {
        nrf::residuals::ResidualsValues::new(self.output, nalgebra::SVector::<f64, 1>::new(0.0))
    }
}

const INITIALIZATION: nalgebra::SVector<f64, 1> = nalgebra::SVector::<f64, 1>::new(1.0);
const UNRESOLVED_OUTPUT: nalgebra::SVector<f64, 1> = nalgebra::SVector::<f64, 1>::new(-1.0);
// A function to change the model in-between two calls,
// otherwise it would always be in a solved state after the first evaluation
// The issue is that the time of the this operation is also included in the benchmark
fn solve_problem(
    user_model: &mut UserModel,
    rf: &mut newton_rootfinder::solver::RootFinder<
        nrf::iteratives::IterativeParamsFD,
        nalgebra::Const<1>,
    >,
) {
    user_model.iteratives = INITIALIZATION;
    user_model.output = UNRESOLVED_OUTPUT;
    rf.solve(user_model).unwrap();
}

fn static_types(c: &mut Criterion) {
    let mut user_model = UserModel::new(1.0);

    let solver_parameters = nrf::solver::SolverParameters::new(
        1,
        1e-6,
        50,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        false,
    );

    let iterative_param = nrf::iteratives::IterativeParamsFD::default();
    let iteratives_param = [iterative_param];
    let iteratives = nrf::iteratives::Iteratives::new(&iteratives_param);
    let residuals_config = nrf::residuals::ResidualsConfig::new(
        &[NormalizationMethod::Abs],
        &[NormalizationMethod::Abs],
    );

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        user_model.get_iteratives(),
        &iteratives,
        &residuals_config,
    );

    let mut group_solver = c.benchmark_group("Advanced solver with static types");

    group_solver.bench_function("Newton-Raphson with finite differences", |b| {
        b.iter(|| solve_problem(black_box(&mut user_model), black_box(&mut rf)))
    });
    group_solver.finish();
}

criterion_group!(benches, static_types);
criterion_main!(benches);
