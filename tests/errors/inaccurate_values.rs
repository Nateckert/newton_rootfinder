use std::error::Error;
use std::fmt;

use newton_rootfinder as nrf;
use nrf::iteratives;
use nrf::model::Model;
use nrf::residuals;

struct MyDummyModel {
    iteratives: nalgebra::DVector<f64>,
    residuals: nalgebra::DVector<f64>,
}

impl MyDummyModel {
    pub fn new() -> Self {
        let iteratives = nalgebra::DVector::zeros(1);
        let residuals = nalgebra::DVector::zeros(1);
        MyDummyModel {
            iteratives,
            residuals,
        }
    }
}

#[derive(Debug)]
pub enum MyCustomErrors {
    NotAGoodValue,
}

impl fmt::Display for MyCustomErrors {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            _ => write!(f, "{}", "Not a good value"),
        }
    }
}

impl Error for MyCustomErrors {}

impl Model<nalgebra::Dyn> for MyDummyModel {
    type InaccurateValuesError = MyCustomErrors;
    type UnusableValuesError = MyCustomErrors;

    fn len_problem(&self) -> usize {
        1
    }

    fn get_iteratives(&self) -> nalgebra::DVector<f64> {
        return self.iteratives.clone();
    }

    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
        self.iteratives = iteratives.clone();
    }

    fn get_residuals(&self) -> nrf::residuals::ResidualsValues<nalgebra::Dyn> {
        return nrf::residuals::ResidualsValues::new(
            self.residuals.clone(),
            nalgebra::DVector::zeros(1),
        );
    }

    fn evaluate(&mut self) -> Result<(), nrf::model::ModelError<Self, nalgebra::Dyn>> {
        self.residuals[0] = self.iteratives[0].powi(2) - 2.0;
        Err(nrf::model::ModelError::InaccurateValuesError(
            MyCustomErrors::NotAGoodValue,
        ))
    }
}

#[test]
fn test_convergence_with_error_from_model() {
    let problem_size = 1;
    let mut init = nalgebra::DVector::zeros(problem_size);
    init[0] = 1.0;

    let damping = false;

    let vec_iter_params = iteratives::default_vec_iteratives_fd(problem_size);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update_methods = vec![residuals::NormalizationMethod::Abs; problem_size];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default_with_guess(
        init,
        &iter_params,
        &res_config,
        nrf::solver::ResolutionMethod::NewtonRaphson,
        damping,
    );

    let mut my_model = MyDummyModel::new();

    let result = rf.solve(&mut my_model).unwrap_err();
    let expected: nrf::errors::SolverError<nrf::model::UserModelFromFunction, nalgebra::Dyn> =
        nrf::errors::SolverError::FinalEvaluationError;
    assert_eq!(expected.to_string(), result.to_string());
    assert!(float_cmp::approx_eq!(
        f64,
        my_model.get_iteratives()[0],
        std::f64::consts::SQRT_2,
        epsilon = 1e-6
    ));
}
