//! Blanket implementation to easily adapt user function to the `Model` trait required by the solver
//!
//! The right side of the equation is a constant and by default zero.
//! No other outputs are computed
//!
//! # Examples
//! ```
//! pub fn square(x: &nalgebra::DVector::<f64>) -> nalgebra::DVector::<f64> {
//!     x*x
//! }
//!
//! extern crate newton_rootfinder;
//! use newton_rootfinder::solver_advanced as nrf;
//! use nrf::model::Model; // trait import required
//! extern crate nalgebra;
//!
//! let iteratives = nalgebra::DVector::from_vec(vec!(2.0));
//! let mut user_model = nrf::model::UserModelWithFunc::new(1, square);
//! user_model.set_iteratives(&iteratives);
//! user_model.evaluate();
//!
//! assert_eq!(user_model.len_problem(), 1);
//! assert_eq!(user_model.get_iteratives(), nalgebra::DVector::from_vec(vec!(2.0)));
//! assert_eq!(user_model.get_residuals().get_values(0), (4.0, 0.0));
/// ```
use super::Model;

use crate::solver_advanced::residuals;

pub struct UserModelWithFunc {
    pub inputs: nalgebra::DVector<f64>,
    pub func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    pub left: nalgebra::DVector<f64>,
    pub right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl UserModelWithFunc {
    pub fn new(
        problem_size: usize,
        func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    ) -> Self {
        let inputs = nalgebra::DVector::zeros(problem_size);
        let left = nalgebra::DVector::from_vec(vec![f64::NAN; problem_size]);
        let right = nalgebra::DVector::zeros(problem_size);

        UserModelWithFunc {
            inputs,
            func,
            left,
            right,
            problem_size,
        }
    }
}

impl Model for UserModelWithFunc {
    fn evaluate(&mut self) {
        self.left = (self.func)(&self.inputs);
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

pub struct UserModelWithFuncJac {
    pub inputs: nalgebra::DVector<f64>,
    pub func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
    pub jac: fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    pub left: nalgebra::DVector<f64>,
    pub right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl UserModelWithFuncJac {
    pub fn new(
        problem_size: usize,
        func: fn(&nalgebra::DVector<f64>) -> nalgebra::DVector<f64>,
        jac: fn(&nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64>,
    ) -> Self {
        let inputs = nalgebra::DVector::zeros(problem_size);
        let left = nalgebra::DVector::from_vec(vec![f64::NAN; problem_size]);
        let right = nalgebra::DVector::zeros(problem_size);

        UserModelWithFuncJac {
            inputs,
            func,
            jac,
            left,
            right,
            problem_size,
        }
    }
}

impl Model for UserModelWithFuncJac {
    fn evaluate(&mut self) {
        self.left = (self.func)(&self.inputs);
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
