//! The rootfinder is operating on a `Model`
//!
//! An usual rootfinding algorithm operates on a function f(X) -> Y.
//! The solver attempts to find X such as norm(f(X)) < tolerance.
//! However, in real life cases, the model computes other quantities that are the mainly focus of the user of the solver.
//! In fact, the Y paremeters (the residuals) have often few signification,
//! the user being mostly interested by other quantities.
//!
//! With most of the available solver, computing the other quantities requires another function call to extract them.
//! This extra function call being made with the X values found by the solver.
//!
//! The `Model` trait is centered around the `evaluate` method.
//! This method implement the calculations wanted by the user.
//! The requirements is that it should mutate a field of the model that correspond to the residuals values (the outputs of our function)
//!
//! The solver is able to access to the residuals values thanks to the `get_residuals()` method
//!
//! The solver must also be able to interact with the iteratives variables (inputs of our function).
//! Two methods must them be implemented : `set_iteratives()` and `get_iteratives()`
//!
//! In addition the user must provide the `len_problem` methods for determining the size.
//!
//! Four other methods have an default blanket implementation :
//! -init() : to be called once before the first function call and the start of the algorithm.
//!           It allows the user to implement a logic such as loading some data to setup its model.
//! -three functions to interact with memory effects of a model : `len_memory()`, `set_memory()`, `get_memory()`
//!          Such memory effects can occure in complex model in interaction with the finite-difference evaluation of the jacobian.
//!          For example, one can use some global variables to approximate some expression.
//!          These global variables are updated after each model evaluation.
//!          Hence, if the jacobian evaluation is made with finite-difference and the memory state not reinitialised in between two evaluation,
//!          the column order would change the result.


extern crate nalgebra;

use crate::util::residuals;

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
}

// Other possible functions :
// If enforced in trait, return type Option would be required
//
// fn len_parameters(&self) -> usize
// fn set_parameters(&mut self, params: &nalgebra::DVector::<f64>)
// fn get_parameters(&self) -> nalgebra::DVector::<f64>
//
// fn len_outputs(&self) -> usize
// fn get_outputs(&self) -> nalgebra::DVector::<f64>
