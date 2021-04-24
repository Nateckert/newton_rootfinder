//! Newton based methods for rootfinding
//! ========================================================
//!
//! This crate allows you to use [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) for rootfinding.
//!
//! It aims to implement several Newton based methods (Newton-Raphson, Broyden, ...), whether the jacobian function is provided or not.
//!
//! # Nonlinear equation solver
//!
//! ## Practical example
//!
//! Let's consider the following equations :
//!
//! x1 * x2 + x3 = 1
//!
//! x1 * x2 / x3 = x1 * x3
//!
//! x1 * x3 - x3 / x2 - 1 = 0
//!
//! Let's call X = (x1, x2, x3) the iterative variables
//! 
//! Let's call f the function such as f(X) = (left, right),
//! where left is the left member of the equations :
//!
//!  left = ( x1 * x2 + x3, x1 * x2 / x3, x1 * x3 - x3 / x2 - 1  )
//!
//!  right = (1, x1 * x3, 0)
//!
//! Let's call the pair (left, right) the residuals (i.e the residual equations)
//!
//! Solving this problem implies to find X such that the residual equations are fullfiled.
//! Newton based methods will achieve that by iterating on the vector X (hence the name of iteratives).
//!
//! ## General formulation
//!
//! The solver provided in this crate aims to solve the n-dimensional problem:
//!
//! f((iterative_1, ... , iterative_n)) -> (equation_1, ... , equation_n)
//!
//! Each equation being separated into two : left side and right side.
//!
//! In the litterature, the problem is often described as f(X) = 0,
//! as the mathematical expressions can be rearranged.
//!
//! However, it is practical to not adopt this framework for numerical aspects:
//!
//! Imagine for example, that the residual equations are involving different variables with different order of magnitudes :
//!
//! Eq1 : Pressure_1 = Pressure_2
//!
//! Eq2 : Temperature_1 = Temperature_2
//!
//! The usual order of magnitude of a pressure is of 10^5 Pa, a temperature is usually 10^2 K.
//! Hence, from the numerical point of view,
//! the two pressures being equal should have a different signification than the temperatures being equal.
//!
//! This particularity has lead to the separation of left and right member of an equation for the implementation of this solver.
//!
//! # Implementation
//!
//! ## User problem definition
//!
//! To get improved interactions with the user problem,
//! the user is required to provide a stuct implementing the `Model` trait.
//! This trait allows for the solver to be integrated tightly with the use problem and optimized.
//! Check the documentation of this trait for more details.
//!
//! In practice, in most of the case, the user's problem is defined through a function or a clojure.
//! A mecanism has been provided to implement the `Model` trait automatically given a user defined function
//!
//! Check the `UserModelWithFunc` documentation for more details.
//!
//! ## Numerical methods
//!
//! This crate implents several Newton based methods through the `ResolutionMethod` enum.
//!
//! ## Problem parametrization
//!
//! In addition to the selection of the numerical methods,
//! it is possible to configure many parameters with regards to the iterative variables or the residuals equations.
//!
//!  Check the documentation of the `iteratives` and `residuals` modules for more details.
//!
//! ## User interface
//!
//! To ease the parametrization of the solver, it is possible to set up the parametrization through an extarnal .xml configuration file.
//! The parametrization will be read at runtime before launching the resolution.
//!
//! It also possible to define the parametrization programmatically, in such case your programm will execute faster. 
//!
//! ## Key features
//!  1. Works whether the jacobian is provided or not (evaluating it with finite-differentiation).
//!  2. In-detail parametrization of iterative variables, residuals and stopping criteria.
//!  3. Debugging informations available through a .txt log file.
//!  4. The advanced solver is designed to interact with a complex model computing other outputs and having memory effects.
//!      The requirements of this model are defined by the `Model` trait.
//!      The struct `UserModelWithFunc` is provided to easily adapt a given function to the required trait.
//!  5. Real world use cases and an extensive function database are included in the crate for integration testing and benchmarking. (work in progress)
//!
//! ## Current limitations
//!
//! 1. The inputs and outputs of the model are assumed to be `nalgebra` vectors.
//! 2. The test base is still in construction
//!
//! ## Examples
//! ```
//! extern crate newton_rootfinder;
//! use newton_rootfinder::solver_advanced as nrf;
//! use nrf::model::Model; // trait import
//!
//! extern crate nalgebra;
//!
//! // Function to optimize: x**2 = 2
//! pub fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
//!     let mut y = x * x;
//!     y[0] -= 2.0;
//!    y
//! }
//!
//! fn main() {
//!
//!     let problem_size = 1;
//!
//!     // Parametrization of the iteratives variables
//!     let vec_iter_params = nrf::iteratives::default_vec_iteratives_fd(problem_size);
//!     let iter_params = nrf::iteratives::Iteratives::new(&vec_iter_params);
//!
//!     // Parametrization of the residuals
//!     let stopping_residuals = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
//!     let update_methods = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
//!     let res_config = nrf::residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//!
//!     // Parametrization of the solver
//!     let init = nalgebra::DVector::from_vec(vec![1.0]);
//!     let resolution_method = nrf::solver::ResolutionMethod::NewtonRaphson;
//!     let damping = false;
//!     let mut rf = nrf::solver::default_with_guess(
//!         init,
//!         &iter_params,
//!         &res_config,
//!         resolution_method,
//!         damping,
//!     );
//!
//!     // Adpatation of the function to solve to the Model trait.
//!     let mut user_model = nrf::model::UserModelWithFunc::new(problem_size, square2);
//!
//!     rf.solve(&mut user_model);
//!
//!     println!("{}", user_model.get_iteratives()[0]); // 1.4142135623747443
//!     println!("{}", std::f64::consts::SQRT_2);       // 1.4142135623730951
//! }
//! ```
//!


pub mod solver_advanced;
pub mod solver_minimal;
