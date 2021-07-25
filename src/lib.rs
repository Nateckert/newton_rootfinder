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
//!```block
//! x1 * x2 + x3 = 1
//! x1 * x2 / x3 = x1 * x3
//! x1 * x3 - x3 / x2 - 1 = 0
//!```
//!
//! Let's call X = (x1, x2, x3) the **iterative variables**
//!
//! Let's define mathematically the problem thanks to the function f :
//!
//!```block
//! f(X) -> (left, right)
//!```
//!
//! where the left and right 3-dimensional vectors are the left and right members of the equations :
//!
//!```block
//! left = ( x1 * x2 + x3, x1 * x2 / x3, x1 * x3 - x3 / x2 - 1  )
//! right = (1, x1 * x3, 0)
//!```
//!
//! Let's call the pair (left, right) the **residuals** (i.e the residual equations)
//!
//! Solving this problem implies to find X such that the residual equations are fullfiled.
//!
//! Newton based methods will achieve that by iterating on the vector X (hence the name of iteratives).
//!
//! ## General formulation
//!
//! In the previous example, the following concepts have been highlighted:
//!
//! - Iterative variables : the variables on which the algorithm will iterate
//! - Residuals : the equations that must be verified, each residual is separated into two expressions, the left member and the right member of the equation.
//!
//! For a well-defined problem, the must be as many iterative variables as residuals.
//!
//!
//! The solver provided in this crate aims to solve the n-dimensional problem:
//!
//!```block
//! f((iterative_1, ... , iterative_n)) -> (equation_1, ... , equation_n)
//!```
//!
//! In the litterature, the problem is often described as ```f(X) = 0```,
//! as the mathematical expressions of the residual equations can be rearranged.
//!
//! ## Resolution principle
//!
//! Check the wikipedia article on [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) !
//!
//! You will see that it involves the computation of the jacobian matrix (i.e the n-dimensional derivative matrix).
//! This matrix can either be provided by the user, or computed thanks to finite-difference.
//!
//! # Usage
//!
//! Using this crate require the following steps:
//! - Defining the problem (i.e have a struct implementing the [model::Model] trait)
//! - Parametrizing the solver ([iteratives], [residuals] and some other parameters)
//! - Call the solver on the model: the solver will then mutate the model into a resolved state
//!
//! ```
//! use newton_rootfinder as nrf;
//! use nrf::model::Model; // trait import
//! #
//! # use nalgebra;
//!
//! struct UserModel {
//! # pub inputs: nalgebra::DVector<f64>,
//! # pub left: nalgebra::DVector<f64>,
//! // ...
//! }
//! #
//! # impl UserModel {
//! #     pub fn get_outputs(self) -> bool {
//! #          true
//! #      }
//! #       pub fn new() -> Self {
//! #           UserModel {
//! #                   inputs: nalgebra::DVector::from_vec(vec![1.0]),
//! #                   left: nalgebra::DVector::from_vec(vec![1.0]),
//! #           }
//! #   }
//! # }
//! #
//! impl Model<nalgebra::Dynamic> for UserModel {
//! // ...
//! #   fn evaluate(&mut self) {
//! #       let mut y = self.inputs.clone() * self.inputs.clone();
//! #       y[0] -= 2.0;
//! #       self.left =  y;
//! #    }
//! #
//! #   fn get_residuals(&self) -> nrf::residuals::ResidualsValues<nalgebra::Dynamic> {
//! #       let right = nalgebra::DVector::zeros(self.len_problem());
//! #       nrf::residuals::ResidualsValues::new(self.left.clone(), right.clone())
//! #    }
//! #
//! #   fn get_iteratives(&self) -> nalgebra::DVector<f64> {
//! #        self.inputs.clone()
//! #   }
//! #
//! #   fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
//! #        self.inputs = iteratives.clone();
//! #    }
//! #
//! #   fn len_problem(&self) -> usize {
//! #        1
//! #    }
//! #
//! }
//!
//!
//! fn main() {
//!
//! #    let problem_size = 1;
//! #    let vec_iter_params = nrf::iteratives::default_vec_iteratives_fd(problem_size);
//! #    let iteratives_configuration = nrf::iteratives::Iteratives::new(&vec_iter_params);
//! #
//! #    let stopping_residuals = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
//! #    let update_methods = vec![nrf::residuals::NormalizationMethod::Abs; problem_size];
//! #    let residuals_configuration = nrf::residuals::ResidualsConfig::new(&stopping_residuals, &update_methods);
//! #
//! #    let solver_parameters = nrf::solver::SolverParameters::new(1, 1e-6, 60, nrf::solver::ResolutionMethod::NewtonRaphson, true);
//! #    let inital_guess = nalgebra::DVector::from_vec(vec![1.0]);
//! #
//!     // ...
//!     let mut rootfinder = nrf::solver::RootFinder::new(
//!         solver_parameters,
//!         inital_guess,
//!         &iteratives_configuration,
//!         &residuals_configuration,
//!     );
//!
//!     let mut user_model = UserModel::new();
//!
//!     rootfinder.solve(&mut user_model);
//!
//!     println!("{}", user_model.get_outputs());
//! }
//! ```
//!
//!
//! ## User problem definition
//!
//! To get improved interactions with the user problem,
//! the user is required to provide a stuct implementing the [model::Model] trait.
//! This trait allows for the solver to be integrated tightly with the use problem and optimized.
//!
//! Check the documentation of the [model] module for more details.
//!
//!
//! ## Numerical methods
//!
//! This crate implents several Newton based methods through the [solver::ResolutionMethod] enum.
//!
//! ## Problem parametrization
//!
//! The parametrization of the resolution is a three step process in order to configure :
//! - each one of the [iteratives] variables
//! - each one of the [residuals] equations
//! - the solver itself, by defining the [solver::SolverParameters]
//!
//! Once each of these element has been defined, the [solver::RootFinder] struct can be instanciated.
//!
//! It is this struct that will perform the resolution.
//!
//! ## Debugging
//!
//! In order to be able to debug more easily the resolution process, it is possible to generate a simulation log.
//!
//! Check the [solver::RootFinder::activate_debug] method.
//!
//! ## User interface
//!
//! To ease the parametrization of the solver, it is possible to set up the parametrization through an external `.xml` configuration file.
//! The parametrization will be read at runtime before launching the resolution.
//! For more information, check the [xml_parser] module.
//!
//! It also possible to define the parametrization programmatically, in such case your programm will execute faster.
//!
//!
//! ## Examples
//! ```
//! extern crate newton_rootfinder;
//! use newton_rootfinder as nrf;
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
//!     let mut user_model = nrf::model::UserModelFromFunction::new(problem_size, square2);
//!
//!     rf.solve(&mut user_model);
//!
//!     println!("{}", user_model.get_iteratives()[0]); // 1.4142135623747443
//!     println!("{}", std::f64::consts::SQRT_2);       // 1.4142135623730951
//! }
//! ```
//!
//! # Performance tricks
//!
//! `newton_rootfinder` provides several mecanisms to ease the use of the solver,
//! such as `default_vec_iteratives_fd`, `default_with_guess`, `UserModelFromFunction`
//!
//! These mecanisms use underneath rust `Vec` and the `nalgebra` type `DVector` (dynamic vector)
//!
//! It is possible to use `newton_rootfinder` with statically sized type
//! To do so, the user must not rely on the default mecanisms provided by the crate,
//! but instead define manually in the its code each of its parameters
//! The user must also implement directely the [model::Model] trait with static types.
//!
//! Full example :
//!
//! Benchmark static vs dynamic:

pub use solver_n_dimensional::model;

pub use solver_n_dimensional::iteratives;
pub use solver_n_dimensional::residuals;

pub use solver_n_dimensional::solver;

pub use solver_n_dimensional::xml_parser;

mod solver_n_dimensional;
