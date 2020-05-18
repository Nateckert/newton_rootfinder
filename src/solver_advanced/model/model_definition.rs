//! The rootfinder is operating on a `Model`
//!
//! A usual rootfinding algorithm operates on a function f(X) -> Y.
//! The solver attempts to find X such that norm(f(X)) < tolerance.
//!
//! However, in real life cases, the model computes other quantities that are the main focus of the end-user of the solver.
//! In fact, the Y paremeters (the residuals) have often few significations,
//! the user being mostly interested by other quantities.
//!
//! With most available solvers, computing the other quantities requires another function call to extract them.
//! This extra function call being made with the X values found by the solver.
//!

extern crate nalgebra;

use crate::solver_advanced::util::residuals;

/// The `Model` trait is the minimal requirement that ensures the capacity of a given model
/// to interact with the solver.
///
/// The `Model` trait is centered around the `evaluate` method.
/// This method implements the calculations wanted by the user.
/// The requirements is that it should mutate a field of the model that correspond to the residuals values (the outputs of our function)
///
/// The solver is able to access to the residuals values thanks to the `get_residuals()` method
///
/// The solver must also be able to interact with the iteratives variables (inputs of our function).
/// Two methods must them be implemented : `set_iteratives()` and `get_iteratives()`
///
/// In addition the user must provide the `len_problem` methods for determining the size.
///
/// Six other methods have an default blanket implementation :
/// - `init()` : to be called once before the first function call and the start of the algorithm.
///           It allows the user to implement a logic such as loading some data to setup its model.
/// - three functions to interact with memory effects of a model : `len_memory()`, `set_memory()`, `get_memory()`
///          Such memory effects can occure in complex model in interaction with the finite-difference evaluation of the jacobian.
///          For example, one can use some global variables to approximate some expression.
///          These global variables are updated after each model evaluation.
///          Hence, if the jacobian evaluation is made with finite-difference and the memory state not reinitialised in between two evaluation,
///          the column order would change the result.
/// - two functions in case the jacobian is provided by the user.
///          If the user can compute the jacobian, the evluation will be more efficient than through finite-difference
///          The user must change the `jacobian_provided` method to return true.
///          The user must provide the `get_jacobian` method to return the values of the jacobian.
pub trait Model {
    fn init(&self) {
        //default empty method
    }
    fn evaluate(&mut self);
    fn len_problem(&self) -> usize;
    fn len_memory(&self) -> usize {
        0 // 0 by default if not using any memory features
    }

    fn set_memory(&mut self, #[allow(unused_variables)] memory: &nalgebra::DVector<f64>) {
        // default empty method
    }
    fn get_memory(&self) -> nalgebra::DVector<f64> {
        // default empty DVector
        nalgebra::DVector::from_vec(vec![])
    }
    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>);
    fn get_iteratives(&self) -> nalgebra::DVector<f64>;
    fn get_residuals(&self) -> residuals::ResidualsValues;
    fn jacobian_provided(&self) -> bool {
        false
    }
    fn get_jacobian(&self) -> residuals::JacobianValues {
        let left = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        let right = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        residuals::JacobianValues::new(left, right)
    }
}
