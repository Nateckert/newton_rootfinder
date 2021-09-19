use newton_rootfinder as nrf;
use std::convert::Infallible;

use nrf::{model::Model, residuals::NormalizationMethod};

/// x**2 - 2 = 0
/// Root: x = 2.sqrt() approx 1.4142
pub fn square2(x: &nalgebra::SVector<f64, 1>) -> nalgebra::SVector<f64, 1> {
    let y = nalgebra::SVector::<f64, 1>::new(x[0] * x[0] - 2.0);
    y
}

struct UserModel {
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

impl Model<nalgebra::Const<1>> for UserModel {
    type InaccurateValuesError = Infallible;
    type UnusableValuesError = Infallible;
    type UnrecoverableError = Infallible;

    fn len_problem(&self) -> usize {
        1
    }
    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.iteratives = *iteratives;
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        self.iteratives
    }

    fn evaluate(&mut self) -> Result<(), nrf::model::ModelError<Self, nalgebra::Const<1>>> {
        self.output = square2(&self.iteratives);
        Ok(())
    }

    fn get_residuals(&self) -> nrf::residuals::ResidualsValues<nalgebra::Const<1>> {
        nrf::residuals::ResidualsValues::new(self.output, nalgebra::SVector::<f64, 1>::new(0.0))
    }
}

#[test]
fn static_types() {
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

    let mut user_model = UserModel::new(1.0);

    let mut rf = nrf::solver::RootFinder::new(
        solver_parameters,
        user_model.get_iteratives(),
        &iteratives,
        &residuals_config,
    );

    rf.solve(&mut user_model);

    assert!(float_cmp::approx_eq!(
        f64,
        user_model.get_iteratives()[0],
        std::f64::consts::SQRT_2,
        epsilon = 1e-6
    ));
}
