use super::Model;

use crate::residuals;

/// Blanket implementation to easily adapt user function to the [Model](super::Model)  trait required by the solver to work with finite-differences
///
/// The right side of the equation is a constant and by default zero.
/// No other outputs are computed
///
/// # Examples
/// ```
/// pub fn square(x: &nalgebra::DVector::<f64>) -> nalgebra::DVector::<f64> {
///     x*x
/// }
///
/// use newton_rootfinder as nrf;
/// use nrf::model::Model; // trait import required
///
/// let iteratives = nalgebra::DVector::from_vec(vec!(2.0));
/// let mut user_model = nrf::model::UserModelFromFunction::new(1, square);
/// user_model.set_iteratives(&iteratives);
/// user_model.evaluate();
///
/// assert_eq!(user_model.len_problem(), 1);
/// assert_eq!(user_model.get_iteratives(), nalgebra::DVector::from_vec(vec!(2.0)));
/// assert_eq!(user_model.jacobian_provided(), false);
/// assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
/// ```
pub struct UserModelFromFunction {
    pub inputs: nalgebra::DVector<f64>,
    pub func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    pub left: nalgebra::DVector<f64>,
    pub right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl UserModelFromFunction {
    pub fn new(
        problem_size: usize,
        func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    ) -> Self {
        let inputs = nalgebra::DVector::zeros(problem_size);
        let left = nalgebra::DVector::from_vec(vec![f64::NAN; problem_size]);
        let right = nalgebra::DVector::zeros(problem_size);

        UserModelFromFunction {
            inputs,
            func,
            left,
            right,
            problem_size,
        }
    }
}

impl Model for UserModelFromFunction {
    fn evaluate(&mut self) {
        self.left = (self.func)(&self.inputs);
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dynamic> {
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

/// Blanket implementation to easily adapt user functions to the [Model](super::Model)  trait required by the solver to work with a jacobian provided
///
/// The right side of the equation is a constant and by default zero.
/// No other outputs are computed
///
/// # Examples
/// ```
/// pub fn square(x: &nalgebra::DVector::<f64>) -> nalgebra::DVector::<f64> {
///     x*x
/// }
/// pub fn dsquare(x: &nalgebra::DVector::<f64>) -> nalgebra::DMatrix::<f64> {
///     let mut jac = nalgebra::DMatrix::zeros(1,1);
///     jac[(0,0)] = 2.0*x[0];
///     jac
/// }
///
/// use newton_rootfinder as nrf;
/// use nrf::model::Model; // trait import required
///
/// let iteratives = nalgebra::DVector::from_vec(vec!(2.0));
/// let mut user_model = nrf::model::UserModelFromFunctionAndJacobian::new(1, square, dsquare);
/// user_model.set_iteratives(&iteratives);
/// user_model.evaluate();
///
/// assert_eq!(user_model.len_problem(), 1);
/// assert_eq!(user_model.get_iteratives(), nalgebra::DVector::from_vec(vec!(2.0)));
/// assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
///
/// assert_eq!(user_model.jacobian_provided(), true);
/// let jacobians_values = user_model.get_jacobian();
/// let (jac_left, jac_right) = jacobians_values.get_jacobians();
/// assert_eq!(jac_left[(0,0)], 4.0);
/// assert_eq!(jac_right[(0,0)], 0.0);
/// ```
pub struct UserModelFromFunctionAndJacobian {
    pub inputs: nalgebra::DVector<f64>,
    pub func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    pub jac: fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    pub left: nalgebra::DVector<f64>,
    pub right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl UserModelFromFunctionAndJacobian {
    pub fn new(
        problem_size: usize,
        func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
        jac: fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    ) -> Self {
        let inputs = nalgebra::DVector::zeros(problem_size);
        let left = nalgebra::DVector::from_vec(vec![f64::NAN; problem_size]);
        let right = nalgebra::DVector::zeros(problem_size);

        UserModelFromFunctionAndJacobian {
            inputs,
            func,
            jac,
            left,
            right,
            problem_size,
        }
    }
}

impl Model for UserModelFromFunctionAndJacobian {
    fn evaluate(&mut self) {
        self.left = (self.func)(&self.inputs);
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dynamic> {
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
    fn get_jacobian(&self) -> residuals::JacobianValues<nalgebra::Dynamic> {
        let jac_left = (self.jac)(&self.inputs);
        let jac_right = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        residuals::JacobianValues::new(jac_left, jac_right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn square(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        x * x
    }

    pub fn dsquare(x: &nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64> {
        let mut y = nalgebra::DMatrix::zeros(1, 1);
        y[(0, 0)] = 2.0 * x[0];
        y
    }

    #[test]
    fn create_user_model() {
        let iteratives = nalgebra::DVector::from_vec(vec![2.0]);
        let mut user_model = UserModelFromFunction::new(1, square);
        user_model.set_iteratives(&iteratives);
        user_model.evaluate();

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
        let iteratives = nalgebra::DVector::from_vec(vec![2.0]);
        let mut user_model = UserModelFromFunctionAndJacobian::new(1, square, dsquare);
        user_model.set_iteratives(&iteratives);
        user_model.evaluate();

        assert_eq!(user_model.len_problem(), 1);
        assert_eq!(
            user_model.get_iteratives(),
            nalgebra::DVector::from_vec(vec!(2.0))
        );
        assert_eq!(user_model.jacobian_provided(), true);
        assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
    }
}
