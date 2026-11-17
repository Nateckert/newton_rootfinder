use std::error::Error;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};

use newton_rootfinder as nrf;
use nrf::iteratives;
use nrf::log::{Detail, Log};
use nrf::model::{Model, ModelError, UserModelFromClosure, UserModelFromFunction};
use nrf::residuals;
use nrf::solver::ResolutionMethod;

use super::Buffer;

fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut y = x.component_mul(x);
    y[0] -= 2.0;
    y
}

/// Solve `function` = 0 from `guess` with Newton-Raphson, its resolution written in `log`
pub(super) fn solve<M: Model<nalgebra::Dyn>>(
    log: &Log,
    label: &str,
    model: &mut M,
    max_iterations: usize,
) -> Result<(), nrf::errors::SolverError<M, nalgebra::Dyn>> {
    let problem_size = model.get_iteratives().len();
    let iteratives_vec = iteratives::default_vec_iteratives_fd(problem_size);
    let iteratives = iteratives::Iteratives::new(&iteratives_vec);
    let stopping = vec![residuals::NormalizationMethod::Abs; problem_size];
    let update = vec![residuals::NormalizationMethod::Abs; problem_size];
    let residuals_config = residuals::ResidualsConfig::new(&stopping, &update);
    let parameters = nrf::solver::SolverParameters::new(
        1e-6,
        max_iterations,
        ResolutionMethod::NewtonRaphson,
        false,
    );
    let mut rf = nrf::solver::RootFinder::new(parameters, &iteratives, &residuals_config);
    rf.log_to(log, label);
    rf.solve(model)
}

pub(super) fn of_event<'a>(lines: &'a [String], event: &str) -> Vec<&'a String> {
    let prefix = format!("{{\"event\":\"{}\",", event);
    lines
        .iter()
        .filter(|line| line.starts_with(&prefix))
        .collect()
}

#[test]
fn a_converged_resolution_starts_iterates_and_ends() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let mut model = UserModelFromFunction::new(1, square2);
    model.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
    solve(&log, "square root of 2", &mut model, 60).unwrap();
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert_eq!(
        lines[1],
        "{\"event\":\"start\",\"id\":1,\"parent\":null,\"time\":0,\"label\":\"square root of 2\",\
         \"method\":\"newton_raphson\",\"tolerance\":1e-6,\"max_iterations\":60,\"damping\":false,\
         \"iteratives\":[{\"settings\":{\"max_step_abs\":\"Infinity\",\"max_step_rel\":\"Infinity\",\
         \"min_value\":\"-Infinity\",\"max_value\":\"Infinity\",\"perturbation\":\"max\",\
         \"dx_abs\":5e-8,\"dx_rel\":5e-8}}],\
         \"residuals\":[{\"settings\":{\"stopping\":\"abs\",\"update\":\"abs\"}}]}"
    );
    assert_eq!(
        lines[2],
        "{\"event\":\"evaluation\",\"id\":1,\"time\":0,\"iteration\":0,\"of\":\"guess\",\
         \"iteratives\":[1.0],\"left\":[-1.0],\"right\":[0.0],\"errors\":[1.0],\"max_error\":1.0}"
    );
    assert!(
        lines[3]
            .starts_with("{\"event\":\"step\",\"id\":1,\"time\":0,\"iteration\":1,\"step\":[0.49")
    );
    assert!(
        lines[4].starts_with("{\"event\":\"evaluation\",\"id\":1,\"time\":0,\"iteration\":1,\"of\":\"step\",\"iteratives\":[1.49")
    );
    let end = lines.last().unwrap();
    assert!(end.starts_with(
        "{\"event\":\"end\",\"id\":1,\"time\":0,\"outcome\":\"converged\",\"iterations\":"
    ));
    // The default detail writes no jacobian
    assert!(of_event(&lines, "jacobian").is_empty());
}

#[test]
fn a_resolution_inside_the_evaluation_of_another_is_placed_under_it() {
    /// x such that the solution y of 2y = x is 2: an inner resolution at each evaluation
    struct Outer {
        x: nalgebra::DVector<f64>,
        y: nalgebra::DVector<f64>,
        log: Log,
    }

    impl Model<nalgebra::Dyn> for Outer {
        type InaccurateValuesError = std::convert::Infallible;
        type UnusableValuesError = std::convert::Infallible;
        type Memory = ();

        fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
            self.x = iteratives.clone();
        }

        fn get_iteratives(&self) -> nalgebra::DVector<f64> {
            self.x.clone()
        }

        fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Dyn>> {
            let x = self.x[0];
            let inner_function =
                move |y: &nalgebra::DVector<f64>| nalgebra::DVector::from_vec(vec![2.0 * y[0] - x]);
            let mut inner = UserModelFromClosure::new(1, &inner_function);
            inner.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
            solve(&self.log, "inner", &mut inner, 60).unwrap();
            self.y = inner.get_iteratives();
            Ok(())
        }

        fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
            residuals::ResidualsValues::new(self.y.clone(), nalgebra::DVector::from_vec(vec![2.0]))
        }
    }

    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).detail(Detail::Outcome);
    let mut outer = Outer {
        x: nalgebra::DVector::from_vec(vec![1.0]),
        y: nalgebra::DVector::zeros(1),
        log: log.clone(),
    };
    solve(&log, "outer", &mut outer, 60).unwrap();
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    let starts = of_event(&lines, "start");
    assert!(starts[0].starts_with("{\"event\":\"start\",\"id\":1,\"parent\":null,"));
    assert!(starts.len() > 2);
    for (start, id) in starts[1..].iter().zip(2..) {
        assert!(start.starts_with(&format!(
            "{{\"event\":\"start\",\"id\":{},\"parent\":1,",
            id
        )));
        assert!(start.contains("\"label\":\"inner\""));
    }
    assert_eq!(of_event(&lines, "end").len(), starts.len());
    assert!(
        lines
            .last()
            .unwrap()
            .starts_with("{\"event\":\"end\",\"id\":1,")
    );
    // The outcome detail writes no evaluation
    assert!(of_event(&lines, "evaluation").is_empty());
}

#[test]
fn a_resolution_not_converging_says_so() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).detail(Detail::Outcome);
    let mut model = UserModelFromFunction::new(1, square2);
    model.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
    assert!(solve(&log, "one iteration", &mut model, 1).is_err());
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert_eq!(lines.len(), 3);
    assert!(lines[2].starts_with(
        "{\"event\":\"end\",\"id\":1,\"time\":0,\"outcome\":\"non_convergence_error\",\
         \"message\":\"Convergence not reached\",\"iterations\":1,\"max_error\":0.2499"
    ));
}

#[test]
fn a_non_finite_residual_is_written_as_a_string() {
    fn negative_root(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        x.map(|value| value.sqrt() - 1.0)
    }

    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let mut model = UserModelFromFunction::new(1, negative_root);
    model.set_iteratives(&nalgebra::DVector::from_vec(vec![-1.0]));
    assert!(solve(&log, "root of a negative number", &mut model, 60).is_err());
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert!(lines[2].contains(",\"left\":[\"NaN\"],\"right\":[0.0],\"errors\":[\"NaN\"],"));
    assert!(lines[3].contains(
        "\"outcome\":\"non_finite_residuals\",\"message\":\"Non-finite residuals for the equations [0]\",\"residuals\":[0],"
    ));
}

#[derive(Debug)]
struct TooLarge(f64);

impl fmt::Display for TooLarge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} is out of the domain of the model", self.0)
    }
}

impl Error for TooLarge {}

/// x² = 9, the model refusing to evaluate beyond 4
struct Bounded {
    x: nalgebra::DVector<f64>,
    left: nalgebra::DVector<f64>,
}

impl Model<nalgebra::Dyn> for Bounded {
    type InaccurateValuesError = std::convert::Infallible;
    type UnusableValuesError = TooLarge;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
        self.x = iteratives.clone();
    }

    fn get_iteratives(&self) -> nalgebra::DVector<f64> {
        self.x.clone()
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Dyn>> {
        if self.x[0] > 4.0 {
            return Err(ModelError::UnusableValuesError(TooLarge(self.x[0])));
        }
        self.left = self.x.component_mul(&self.x);
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(self.left.clone(), nalgebra::DVector::from_vec(vec![9.0]))
    }
}

#[test]
fn an_evaluation_the_model_refuses_shows_where_and_why() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let mut model = Bounded {
        x: nalgebra::DVector::from_vec(vec![1.0]),
        left: nalgebra::DVector::zeros(1),
    };
    assert!(solve(&log, "bounded", &mut model, 60).is_err());
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    let evaluations = of_event(&lines, "evaluation");
    // From 1, the step goes to 5, within the accuracy of the finite differences
    assert!(evaluations[1].contains(",\"iteration\":1,\"of\":\"step\",\"iteratives\":[4.99"));
    assert!(evaluations[1].ends_with(" is out of the domain of the model\"}"));
    assert!(lines.last().unwrap().contains(
        "\"outcome\":\"model_evaluation_error\",\"message\":\"Model evaluation failed: UnusableValuesError Error: 4.99"
    ));
}

#[test]
fn a_singular_jacobian_is_written() {
    /// a + b = 3 written twice: the jacobian is singular
    fn redundant(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        nalgebra::DVector::from_vec(vec![x[0] + x[1] - 3.0, 2.0 * x[0] + 2.0 * x[1] - 6.0])
    }

    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).detail(Detail::Jacobians);
    let mut model = UserModelFromFunction::new(2, redundant);
    model.set_iteratives(&nalgebra::DVector::from_vec(vec![0.0, 0.0]));
    assert!(solve(&log, "redundant", &mut model, 60).is_err());
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    let jacobians = of_event(&lines, "jacobian");
    assert_eq!(jacobians.len(), 1);
    assert!(jacobians[0].contains(",\"how\":\"finite_differences\",\"matrix\":[["));
    assert!(jacobians[0].ends_with(",\"singular\":true}"));
    assert!(
        lines
            .last()
            .unwrap()
            .contains("\"outcome\":\"non_invertible_jacobian\"")
    );
}

#[test]
fn a_panic_in_the_model_ends_the_resolution() {
    fn panicking(_: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        panic!("the model panicked")
    }

    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut model = UserModelFromFunction::new(1, panicking);
        solve(&log, "panicking", &mut model, 60)
    }));
    assert!(result.is_err());
    log.note("after", |_| {});
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert_eq!(
        lines[2],
        "{\"event\":\"end\",\"id\":1,\"time\":0,\"outcome\":\"panic\"}"
    );
    // The resolution is no longer in progress: the note is at the top
    assert!(lines[3].starts_with("{\"event\":\"note\",\"parent\":null,"));
}

#[test]
fn a_log_without_a_solver_changes_no_resolution() {
    let mut logged = UserModelFromFunction::new(1, square2);
    logged.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
    let log = Log::new(std::io::sink()).detail(Detail::Jacobians);
    solve(&log, "logged", &mut logged, 60).unwrap();

    let mut plain = UserModelFromFunction::new(1, square2);
    plain.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
    let iteratives_vec = iteratives::default_vec_iteratives_fd(1);
    let iteratives = iteratives::Iteratives::new(&iteratives_vec);
    let stopping = vec![residuals::NormalizationMethod::Abs];
    let residuals_config = residuals::ResidualsConfig::new(&stopping, &stopping);
    let parameters =
        nrf::solver::SolverParameters::new(1e-6, 60, ResolutionMethod::NewtonRaphson, false);
    let mut rf = nrf::solver::RootFinder::new(parameters, &iteratives, &residuals_config);
    rf.solve(&mut plain).unwrap();

    assert_eq!(logged.get_iteratives(), plain.get_iteratives());
}
