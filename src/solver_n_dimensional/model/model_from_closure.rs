use std::convert::Infallible;

use super::Model;
use crate::residuals;

/// Blanket implementation to easily adapt user closure to the [Model](super::Model) trait required by the solver to work with finite-differences
///
/// The right side of the equation is a constant and by default zero.
/// No other outputs are computed
///
/// # Examples
/// ```
/// use newton_rootfinder as nrf;
/// use nrf::model::Model; // trait import required
///
/// let square_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> {
///     iteratives * iteratives
/// };
///
/// let iteratives = nalgebra::DVector::from_vec(vec!(2.0));
/// let mut user_model = nrf::model::UserModelFromClosure::new(1, &square_closure);
/// user_model.set_iteratives(&iteratives);
/// user_model.evaluate();
///
/// assert_eq!(user_model.len_problem(), 1);
/// assert_eq!(user_model.get_iteratives(), nalgebra::DVector::from_vec(vec!(2.0)));
/// assert_eq!(user_model.jacobian_provided(), false);
/// assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
/// ```
pub struct UserModelFromClosure<'a> {
    pub inputs: nalgebra::DVector<f64>,
    pub closure: &'a dyn Fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    pub left: nalgebra::DVector<f64>,
    pub right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl<'a> UserModelFromClosure<'a> {
    pub fn new(
        problem_size: usize,
        closure: &'a dyn Fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    ) -> Self {
        let inputs = nalgebra::DVector::zeros(problem_size);
        let left = nalgebra::DVector::from_vec(vec![f64::NAN; problem_size]);
        let right = nalgebra::DVector::zeros(problem_size);

        UserModelFromClosure {
            inputs,
            closure,
            left,
            right,
            problem_size,
        }
    }
}

impl<'a> Model<nalgebra::Dyn> for UserModelFromClosure<'a> {
    type InaccurateValuesError = Infallible;
    type UnusableValuesError = Infallible;

    fn evaluate(&mut self) -> Result<(), super::ModelError<Self, nalgebra::Dyn>> {
        self.left = (self.closure)(&self.inputs);
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(self.left.clone(), self.right.clone())
    }

    fn get_iteratives(&self) -> nalgebra::DVector<f64> {
        self.inputs.clone()
    }

    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
        self.inputs = iteratives.clone();
    }

    fn len_problem(&self) -> usize {
        self.problem_size
    }
}

/// Blanket implementation to easily adapt user closures to the [Model](super::Model)  trait required by the solver to work with a jacobian provided
///
/// The right side of the equation is a constant and by default zero.
/// No other outputs are computed
///
/// # Examples
/// ```
/// let square_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> {
///     iteratives * iteratives
/// };
///
/// let derivative_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DMatrix<f64> {
///     let mut y = nalgebra::DMatrix::zeros(1, 1);
///     y[(0, 0)] = 2.0 * iteratives[0];
///     y
/// };
///
/// use newton_rootfinder as nrf;
/// use nrf::model::Model; // trait import required
///
/// let iteratives = nalgebra::DVector::from_vec(vec!(2.0));
/// let mut user_model = nrf::model::UserModelFromClosureAndJacobian::new(1, &square_closure, &derivative_closure);
/// user_model.set_iteratives(&iteratives);
/// user_model.evaluate();
///
/// assert_eq!(user_model.len_problem(), 1);
/// assert_eq!(user_model.get_iteratives(), nalgebra::DVector::from_vec(vec!(2.0)));
/// assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
///
/// assert_eq!(user_model.jacobian_provided(), true);
/// let jacobians_values = user_model.get_jacobian().unwrap();
/// let (jac_left, jac_right) = jacobians_values.get_jacobians();
/// assert_eq!(jac_left[(0,0)], 4.0);
/// assert_eq!(jac_right[(0,0)], 0.0);
/// ```
pub struct UserModelFromClosureAndJacobian<'a, 'b> {
    pub inputs: nalgebra::DVector<f64>,
    pub closure: &'a dyn Fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    pub jac: &'b dyn Fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    pub left: nalgebra::DVector<f64>,
    pub right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl<'a, 'b> UserModelFromClosureAndJacobian<'a, 'b> {
    pub fn new(
        problem_size: usize,
        closure: &'a dyn Fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
        jac: &'b dyn Fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    ) -> Self {
        let inputs = nalgebra::DVector::zeros(problem_size);
        let left = nalgebra::DVector::from_vec(vec![f64::NAN; problem_size]);
        let right = nalgebra::DVector::zeros(problem_size);

        UserModelFromClosureAndJacobian {
            inputs,
            closure,
            jac,
            left,
            right,
            problem_size,
        }
    }
}

impl<'a, 'b> Model<nalgebra::Dyn> for UserModelFromClosureAndJacobian<'a, 'b> {
    type InaccurateValuesError = Infallible;
    type UnusableValuesError = Infallible;

    fn evaluate(&mut self) -> Result<(), super::ModelError<Self, nalgebra::Dyn>> {
        self.left = (self.closure)(&self.inputs);
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(self.left.clone(), self.right.clone())
    }

    fn get_iteratives(&self) -> nalgebra::DVector<f64> {
        self.inputs.clone()
    }

    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
        self.inputs = iteratives.clone();
    }

    fn len_problem(&self) -> usize {
        self.problem_size
    }

    fn jacobian_provided(&self) -> bool {
        true
    }
    fn get_jacobian(
        &mut self,
    ) -> Result<
        residuals::JacobianValues<nalgebra::Dyn>,
        super::ModelError<Self, nalgebra::Dyn>,
    > {
        let jac_left = (self.jac)(&self.inputs);
        let jac_right = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        Ok(residuals::JacobianValues::new(jac_left, jac_right))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_user_model() {
        let square_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> {
            iteratives * iteratives
        };

        let iteratives = nalgebra::DVector::from_vec(vec![2.0]);
        let mut user_model = UserModelFromClosure::new(1, &square_closure);
        user_model.set_iteratives(&iteratives);
        user_model.evaluate().unwrap();

        assert_eq!(user_model.len_problem(), 1);
        assert_eq!(
            user_model.get_iteratives(),
            nalgebra::DVector::from_vec(vec!(2.0))
        );
        assert_eq!(user_model.jacobian_provided(), false);
        assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
    }

    #[test]
    fn create_user_model_with_jacobian() {
        let square_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DVector<f64> {
            iteratives * iteratives
        };

        let derivative_closure = |iteratives: &nalgebra::DVector<f64>| -> nalgebra::DMatrix<f64> {
            let mut y = nalgebra::DMatrix::zeros(1, 1);
            y[(0, 0)] = 2.0 * iteratives[0];
            y
        };

        let iteratives = nalgebra::DVector::from_vec(vec![2.0]);
        let mut user_model =
            UserModelFromClosureAndJacobian::new(1, &square_closure, &derivative_closure);
        user_model.set_iteratives(&iteratives);
        user_model.evaluate().unwrap();

        assert_eq!(user_model.len_problem(), 1);
        assert_eq!(
            user_model.get_iteratives(),
            nalgebra::DVector::from_vec(vec!(2.0))
        );
        assert_eq!(user_model.jacobian_provided(), true);
        assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
    }
}
