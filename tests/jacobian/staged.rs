use newton_rootfinder as nrf;
use nrf::model::{Model, ModelError};
use nrf::residuals;
use nrf::solver::compute_jacobian_from_finite_difference;

type Dimension = nalgebra::Const<2>;

/// A model computed as a sequence of steps, whose iteratives are numbered
/// such that the first step reading the iterative `column` comes after
/// the first step reading the iterative `column + 1`:
///
/// - step 0: `a = 3 * x[1] ^ 2`, the first step reading `x[1]`;
/// - step 1: `b = a + x[0] ^ 3`, the first step reading `x[0]`;
/// - the residuals: `a` and `b`.
///
/// When `staged`, a column starts from the first step reading its
/// iterative; `steps` counts the steps evaluated.
struct Steps {
    staged: bool,
    x: nalgebra::SVector<f64, 2>,
    a: f64,
    b: f64,
    steps: usize,
}

impl Steps {
    fn new(staged: bool) -> Self {
        Steps {
            staged,
            x: nalgebra::SVector::zeros(),
            a: 0.0,
            b: 0.0,
            steps: 0,
        }
    }

    fn evaluate_from(&mut self, step: usize) {
        if step == 0 {
            self.a = 3.0 * self.x[1].powi(2);
            self.steps += 1;
        }
        self.b = self.a + self.x[0].powi(3);
        self.steps += 1;
    }
}

impl Model<Dimension> for Steps {
    type InaccurateValuesError = std::convert::Infallible;
    type UnusableValuesError = std::convert::Infallible;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 2>) {
        self.x = *iteratives;
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 2> {
        self.x
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, Dimension>> {
        self.evaluate_from(0);
        Ok(())
    }

    fn evaluate_column(&mut self, column: usize) -> Result<(), ModelError<Self, Dimension>> {
        if self.staged {
            // The first step reading each iterative.
            self.evaluate_from([1, 0][column]);
            Ok(())
        } else {
            self.evaluate()
        }
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<Dimension> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.a, self.b]),
            nalgebra::SVector::zeros(),
        )
    }
}

fn jacobian(model: &mut Steps) -> nalgebra::SMatrix<f64, 2, 2> {
    model.set_iteratives(&nalgebra::SVector::from([1.5, 2.0]));
    model.evaluate().unwrap();
    let stopping = vec![residuals::NormalizationMethod::Abs; 2];
    let update = stopping.clone();
    let config = residuals::ResidualsConfig::new(&stopping, &update);
    let perturbations = nalgebra::SVector::from([0.0001, 0.0001]);
    compute_jacobian_from_finite_difference(model, &perturbations, &config).unwrap()
}

#[test]
fn a_model_evaluating_its_columns_from_their_first_step_gives_the_same_jacobian() {
    let mut full = Steps::new(false);
    let mut staged = Steps::new(true);
    let reference = jacobian(&mut full);
    let skipping = jacobian(&mut staged);
    // The same numbers, bit for bit: every step a column skips holds the
    // value of the reference point.
    assert_eq!(skipping, reference);
    assert_eq!(reference[(0, 0)], 0.0);
    assert_eq!(reference[(1, 0)], 6.750450009995745);
    // The reference point and two columns: 6 steps evaluated in full,
    // 5 when the column of `x[0]` starts from its first step.
    assert_eq!(full.steps, 6);
    assert_eq!(staged.steps, 5);
}
