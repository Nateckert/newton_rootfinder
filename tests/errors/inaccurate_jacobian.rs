use std::error::Error;
use std::fmt;

use newton_rootfinder as nrf;
use nrf::errors::SolverError;
use nrf::iteratives;
use nrf::model::{Model, ModelError};
use nrf::residuals;
use nrf::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

/// The residual `x^2 - 2` from `x = 1`, its jacobian provided by the model
/// and always inaccurate
struct InaccurateJacobian {
    x: f64,
    residual: f64,
}

#[derive(Debug)]
struct Inaccurate;

impl fmt::Display for Inaccurate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Inaccurate jacobian")
    }
}

impl Error for Inaccurate {}

impl Model<nalgebra::Const<1>> for InaccurateJacobian {
    type InaccurateValuesError = Inaccurate;
    type UnusableValuesError = std::convert::Infallible;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.x = iteratives[0];
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        nalgebra::SVector::from([self.x])
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Const<1>>> {
        self.residual = self.x.powi(2) - 2.0;
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Const<1>> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.residual]),
            nalgebra::SVector::zeros(),
        )
    }

    fn jacobian_provided(&self) -> bool {
        true
    }

    fn get_jacobian(
        &mut self,
    ) -> Result<residuals::JacobianValues<nalgebra::Const<1>>, ModelError<Self, nalgebra::Const<1>>>
    {
        Err(ModelError::InaccurateValuesError(Inaccurate))
    }
}

fn solve(method: ResolutionMethod) -> (SolverError<InaccurateJacobian, nalgebra::Const<1>>, f64) {
    let vec_iter_params = iteratives::default_vec_iteratives_fd(1);
    let iter_params = iteratives::Iteratives::new(&vec_iter_params);
    let stopping_residuals = vec![residuals::NormalizationMethod::Abs; 1];
    let update_methods = vec![residuals::NormalizationMethod::Abs; 1];
    let res_config = residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
    let mut rf = nrf::solver::default(&iter_params, &res_config, method, false);
    let mut model = InaccurateJacobian {
        x: 1.0,
        residual: 0.0,
    };
    let error = rf.solve(&mut model).unwrap_err();
    (error, model.x)
}

#[test]
fn an_inaccurate_jacobian_ends_a_newton_raphson_resolution() {
    let (error, x) = solve(ResolutionMethod::NewtonRaphson);
    assert!(matches!(
        error,
        SolverError::JacobianEvaluationError(ModelError::InaccurateValuesError(Inaccurate))
    ));
    assert_eq!(
        error.to_string(),
        "Jacobian error: Invalid jacobian: InaccurateValues Error: Inaccurate jacobian"
    );
    assert_eq!(x, 1.0);
}

#[test]
fn an_inaccurate_jacobian_ends_a_quasi_newton_resolution() {
    let (error, x) = solve(ResolutionMethod::QuasiNewton(
        QuasiNewtonMethod::JacobianUpdate(UpdateQuasiNewtonMethod::BroydenFirstMethod),
    ));
    assert!(matches!(
        error,
        SolverError::JacobianEvaluationError(ModelError::InaccurateValuesError(Inaccurate))
    ));
    assert_eq!(x, 1.0);
}
