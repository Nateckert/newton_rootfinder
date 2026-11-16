use std::error::Error;
use std::fmt;

use crate as nrf;
use nrf::model::{Model, ModelError};
use nrf::residuals;
use nrf::solver::compute_jacobian_from_finite_difference;

/// The residual `atan(x)`, whose evaluation fails for `x` in `unusable`.
///
/// The memory holds the iterative of the last evaluation.
struct Arctangent {
    x: f64,
    residual: f64,
    last_x: f64,
    unusable: (f64, f64),
}

#[derive(Debug)]
struct Unusable;

impl fmt::Display for Unusable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unusable value")
    }
}

impl Error for Unusable {}

impl Model<nalgebra::Const<1>> for Arctangent {
    type InaccurateValuesError = std::convert::Infallible;
    type UnusableValuesError = Unusable;
    type Memory = f64;

    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.x = iteratives[0];
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        nalgebra::SVector::from([self.x])
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Const<1>>> {
        self.last_x = self.x;
        let (low, high) = self.unusable;
        if low < self.x && self.x < high {
            return Err(ModelError::UnusableValuesError(Unusable));
        }
        self.residual = self.x.atan();
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Const<1>> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.residual]),
            nalgebra::SVector::zeros(),
        )
    }

    fn get_memory(&self) -> f64 {
        self.last_x
    }

    fn set_memory(&mut self, memory: &f64) {
        self.last_x = *memory;
    }
}

/// The jacobian perturbs `x = 2` to `2.0000001`, where the evaluation fails
#[test]
fn a_failed_jacobian_column_restores_the_memory_of_the_reference_point() {
    let mut model = Arctangent {
        x: 2.0,
        residual: 2.0_f64.atan(),
        last_x: 5.0,
        unusable: (2.00000005, 2.0000002),
    };
    model.evaluate().unwrap();
    let stopping = vec![residuals::NormalizationMethod::Abs; 1];
    let config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let perturbations = nalgebra::SVector::from([0.0000001]);
    compute_jacobian_from_finite_difference(&mut model, &perturbations, &config).unwrap_err();
    assert_eq!(model.last_x, 2.0);
}
