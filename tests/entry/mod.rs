//! A resolution starts from the iteratives the model holds, whose length is the
//! dimension of the problem, checked against the configurations of the solver.

use std::convert::Infallible;

use newton_rootfinder as nrf;
use nrf::errors::SolverError;
use nrf::iteratives;
use nrf::model::{Model, ModelError};
use nrf::residuals;
use nrf::solver::ResolutionMethod;

/// `xᵢ² = 2` for every iterative, counting the evaluations.
struct Square2 {
    x: nalgebra::DVector<f64>,
    residuals: nalgebra::DVector<f64>,
    evaluations: usize,
}

impl Square2 {
    fn from(x: Vec<f64>) -> Self {
        let residuals = nalgebra::DVector::zeros(x.len());
        Square2 {
            x: nalgebra::DVector::from_vec(x),
            residuals,
            evaluations: 0,
        }
    }
}

impl Model<nalgebra::Dyn> for Square2 {
    type InaccurateValuesError = Infallible;
    type UnusableValuesError = Infallible;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
        self.x = iteratives.clone();
    }

    fn get_iteratives(&self) -> nalgebra::DVector<f64> {
        self.x.clone()
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Dyn>> {
        self.evaluations += 1;
        self.residuals = self.x.component_mul(&self.x).add_scalar(-2.0);
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(
            self.residuals.clone(),
            nalgebra::DVector::zeros(self.residuals.len()),
        )
    }
}

fn assert_solved(model: &Square2) {
    for x in model.x.iter() {
        assert!((x - std::f64::consts::SQRT_2).abs() < 1e-6);
    }
}

#[test]
fn the_resolution_starts_from_the_iteratives_of_the_model() {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(1);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping = vec![residuals::NormalizationMethod::Abs; 1];
    let res_config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let mut rf = nrf::solver::default(
        &iter_params,
        &res_config,
        ResolutionMethod::NewtonRaphson,
        false,
    );

    let mut model = Square2::from(vec![10.0]);
    rf.solve(&mut model).unwrap();
    assert_solved(&model);
    let from_ten = model.evaluations;

    // From its solution: converged at the first evaluation, no iteration.
    model.evaluations = 0;
    rf.solve(&mut model).unwrap();
    assert_eq!(model.evaluations, 1);

    // From a closer guess: fewer evaluations.
    let mut model = Square2::from(vec![1.0]);
    rf.solve(&mut model).unwrap();
    assert_solved(&model);
    assert!(model.evaluations < from_ten);
}

#[test]
fn the_dimension_is_the_one_of_the_model() {
    // Built without naming a dimension: the one of the model solved.
    let vec_iter_params = iteratives::default_vec_iteratives_fd(2);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping = vec![residuals::NormalizationMethod::Abs; 2];
    let res_config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let mut rf = nrf::solver::default(
        &iter_params,
        &res_config,
        ResolutionMethod::NewtonRaphson,
        false,
    );

    let mut model = Square2::from(vec![1.0, 3.0]);
    rf.solve(&mut model).unwrap();
    assert_solved(&model);
}

#[test]
fn a_model_of_another_dimension_than_the_configurations_is_refused() {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(3);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping = vec![residuals::NormalizationMethod::Abs; 3];
    let res_config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let mut rf = nrf::solver::default(
        &iter_params,
        &res_config,
        ResolutionMethod::NewtonRaphson,
        false,
    );

    let mut model = Square2::from(vec![1.0, 3.0]);
    let error = rf.solve(&mut model).unwrap_err();
    assert!(matches!(
        error,
        SolverError::DimensionMismatch {
            model: 2,
            solver: 3
        }
    ));
    assert_eq!(
        error.to_string(),
        "Dimension mismatch: the model has 2 iteratives, the solver is configured for 3"
    );
    // Refused before any evaluation, the model untouched.
    assert_eq!(model.evaluations, 0);
    assert_eq!(model.x, nalgebra::DVector::from_vec(vec![1.0, 3.0]));
}

#[test]
#[should_panic(
    expected = "Dimension mismatch between the iteratives and the residuals configurations: 2 iteratives, 3 residuals"
)]
fn configurations_of_different_lengths_are_refused() {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(2);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping = vec![residuals::NormalizationMethod::Abs; 3];
    let res_config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let _solver: nrf::solver::RootFinder<'_, iteratives::IterativeParamsFD, nalgebra::Dyn> =
        nrf::solver::default(
            &iter_params,
            &res_config,
            ResolutionMethod::NewtonRaphson,
            false,
        );
}
