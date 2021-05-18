extern crate nalgebra;

use crate::residuals;

/// The [Model] trait is the minimal requirement that ensures the capacity of a given model
/// to interact with the solver.
///
///
/// # Core functionality
///
/// The mathematical expression `f(iteratives) = residuals` is decomposed into the following three steps process:
/// - [Model::set_iteratives] : setting the values of the inputs
/// - [Model::evaluate] : calling the mathematical function with the previously set inputs
/// - [Model::get_residuals] : acessing the results of the computations
///
/// These three core methods are the definition of the problem to solve and must be implemented by the user.
///
/// # Other methods
///
/// In addition to these 3 methods, some other must also be implemented.
/// These methods are used by the solver to access some additional infos required for the resolutions.
///
/// # Memory
///
/// Three functions to interact with memory effects of a model : `len_memory()`, `set_memory()`, `get_memory()`
/// 
/// Such memory effects can occur in complex model in interaction with the finite-difference evaluation of the jacobian.
/// 
/// For example, one can use some global variables to approximate some expression.
/// These global variables are updated after each model evaluation.
/// Hence, if the jacobian evaluation is made with finite-difference and the memory state not reinitialised in-between two evaluation,
/// the column order would change the result.
///
pub trait Model {
    /// This method defines the dimension of the problem.
    ///
    /// It should be consistent of the length of the [Model::set_iteratives], [Model::get_iteratives] and [Model::get_residuals] argument.
    fn len_problem(&self) -> usize;
    /// This method provides the solver a mecanism to set the iteratives values and perform the resolution
    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>);
    /// This method is required to access the values of the iteratives variables during the resolution process.
    /// The values returned should be the same one as the one set by the [Model::set_iteratives] method.
    fn get_iteratives(&self) -> nalgebra::DVector<f64>;
    /// This method should update the values of the outputs of the model by using as inputs the values set by the [Model::set_iteratives] method.
    ///
    /// This method is the core that defines the computations from the user model.
    fn evaluate(&mut self);

    /// This method gets the values of the output for the solver.
    /// The return argument is in a specific format, separating left and right member of an equation.
    ///
    /// It is practical to adopt this framework in order to deal with specific numerical aspects.
    /// Indeed, mathematically it is easy to define the number 0.
    /// However, for floating point arithmetics (i.e computations done on computers),
    /// the residuals equations being fulfilled will be defined comparatively to a given tolerance,
    /// as it could be impossible to have the equations verified up to machine precision accuracy.
    ///
    /// Imagine for example, that the residual equations are involving different variables with different order of magnitudes :
    ///
    ///```block
    /// Eq1 : Pressure_1 = Pressure_2
    /// Eq2 : Temperature_1 = Temperature_2
    ///```
    ///
    /// The usual order of magnitude of a pressure is of 10^5 Pa, a temperature is usually 10^2 K.
    /// Hence, from the numerical point of view,
    /// the two pressures being equal should have a different signification than the temperatures being equal.
    ///
    /// This particularity has lead to the separation of left and right member of an equation for the implementation of this solver.
    ///
    fn get_residuals(&self) -> residuals::ResidualsValues;

    /// This method allows the solver to know if the jacobian is provided by the user or not
    ///
    /// The default implementation returns `false` which would lead to using finite-differences for evaluating the jacobian
    fn jacobian_provided(&self) -> bool {
        false
    }
    /// If this method is overriden, the solver will be able to use it to evaluate the jacobian, instead of using finite-difference.
    /// If overriden, the [Model::jacobian_provided] must also be overriden to return `true`.
    ///
    /// The default implementation returns a null value, as it will be not be used, the solver defaulting to finite-differences.
    fn get_jacobian(&self) -> residuals::JacobianValues {
        let left = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        let right = nalgebra::DMatrix::zeros(self.len_problem(), self.len_problem());
        residuals::JacobianValues::new(left, right)
    }

    /// The default implementation returns 0
    fn len_memory(&self) -> usize {
        0
    }

    /// The default implementation is empty
    fn set_memory(&mut self, #[allow(unused_variables)] memory: &nalgebra::DVector<f64>) {
    }

    /// The default implementation returns an empty vector
    fn get_memory(&self) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(vec![])
    }
}
