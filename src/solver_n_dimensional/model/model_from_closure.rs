use super::Model;
use crate::residuals;

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

impl<'a> Model for UserModelFromClosure<'a> {
    fn evaluate(&mut self) {
        self.left = (self.closure)(&self.inputs);
    }

    fn get_residuals(&self) -> residuals::ResidualsValues {
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

impl<'a, 'b> Model for UserModelFromClosureAndJacobian<'a, 'b> {
    fn evaluate(&mut self) {
        self.left = (self.closure)(&self.inputs);
    }

    fn get_residuals(&self) -> residuals::ResidualsValues {
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
    fn get_jacobian(&self) -> residuals::JacobianValues {
        let jac_left = (self.jac)(&self.inputs);
        let jac_right = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        residuals::JacobianValues::new(jac_left, jac_right)
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
        user_model.evaluate();

        assert_eq!(user_model.len_problem(), 1);
        assert_eq!(
            user_model.get_iteratives(),
            nalgebra::DVector::from_vec(vec!(2.0))
        );
        assert_eq!(user_model.jacobian_provided(), false);
        assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
    }
}
