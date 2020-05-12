//! Advanced solver
//!
//! ## Examples
//!
//! ```
//! extern crate newton_rootfinder as nrf;
//! use nrf::model::Model;
//!
//! extern crate nalgebra;
//!
//! /// Equation : x**2 - 2 = 0
//! fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
//!     let mut y = x * x;
//!     y[0] -= 2.0;
//!     y
//! }
//!
//! fn main() {
//!   let problem_size = 1;
//!   let init_guess = nalgebra::DVector::from_vec(vec![1.0]);
//!   let rf = nrf::solver::RootFinder::default_with_guess(init_guess);
//!   let mut user_model =
//!       nrf::model_with_func::UserModelWithFunc::new(problem_size, square2);
//!
//!   rf.solve(&mut user_model);
//!
//!   println!("{}", user_model.get_iteratives()[0]);
//!   // print 1.4142135623747443
//! }
//! ```

extern crate nalgebra;

use crate::model;
use crate::util::iteratives;
use crate::util::iteratives_fd;
use crate::util::jacobian;
use crate::util::residuals;

pub struct RootFinder {
    initial_guess: nalgebra::DVector<f64>,
    iteratives_params: Vec<iteratives_fd::IterativeParamsFD>,
    residuals_config: residuals::ResidualsConfig,
    problem_size: usize,
    tolerance: f64,
    max_iter: usize,
    damping: bool,
}

impl RootFinder {
    pub fn new(
        initial_guess: nalgebra::DVector<f64>,
        iteratives_params: Vec<iteratives_fd::IterativeParamsFD>,
        residuals_config: residuals::ResidualsConfig,
        problem_size: usize,
        tolerance: f64,
        max_iter: usize,
    ) -> Self {
        let damping = false;

        RootFinder {
            initial_guess,
            iteratives_params,
            residuals_config,
            problem_size,
            tolerance,
            max_iter,
            damping,
        }
    }

    pub fn set_damping(&mut self, damping: bool) {
        self.damping = damping;
    }

    fn evaluate_max_error<T: model::Model>(&self, model: &T) -> f64 {
        let residuals_values = model.get_residuals();
        let stopping_residuals = self
            .residuals_config
            .evaluate_stopping_residuals(&residuals_values);
        stopping_residuals.amax()
    }

    fn compute_step<T: model::Model>(&self, model: &mut T) -> nalgebra::DVector<f64> {
        let iteratives = model.get_iteratives();
        let residuals = self
            .residuals_config
            .evaluate_update_residuals(&model.get_residuals());

        let perturbations = iteratives_fd::compute_perturbation(
            &self.iteratives_params,
            &iteratives,
            self.problem_size,
        );

        let jac = jacobian::jacobian_evaluation(model, &perturbations, &(self.residuals_config));

        let raw_step = jacobian::newton_raw_step_size(&residuals, &jac);

        iteratives::step_limitations(
            &self.iteratives_params,
            &iteratives,
            &raw_step,
            self.problem_size,
        )
    }

    fn update_model<T: model::Model>(
        &self,
        model: &mut T,
        proposed_guess: &nalgebra::DVector<f64>,
    ) -> f64 {
        let max_error = self.evaluate_max_error(model);
        let current_guess = model.get_iteratives();
        model.set_iteratives(&proposed_guess);
        model.evaluate();
        let mut max_error_next = self.evaluate_max_error(model);

        if self.damping {
            if max_error_next > max_error {
                let damped_guess = (proposed_guess + current_guess) / 2.0;
                model.set_iteratives(&damped_guess);
                model.evaluate();
                max_error_next = self.evaluate_max_error(model);
            }
        }

        max_error_next
    }

    pub fn solve<T>(&self, model: &mut T)
    where
        T: model::Model,
    {
        model.init();
        model.set_iteratives(&self.initial_guess);
        model.evaluate();

        let mut max_error = self.evaluate_max_error(model);

        let mut iter = 0;

        while max_error > self.tolerance && iter < self.max_iter {
            iter += 1;
            let proposed_guess = self.compute_step(model);
            max_error = self.update_model(model, &proposed_guess);
        }
    }

    pub fn default_with_guess(initial_guess: nalgebra::DVector<f64>) -> Self {
        let problem_size = initial_guess.len();
        let iteratives_params = iteratives_fd::iterativesfd_default_with_size(problem_size);
        let residuals_config = residuals::ResidualsConfig::default_with_size(problem_size);
        let tolerance: f64 = 1e-6;
        let max_iter: usize = 50;
        let damping: bool = false;

        RootFinder {
            initial_guess,
            iteratives_params,
            residuals_config,
            problem_size,
            tolerance,
            max_iter,
            damping,
        }
    }
}
