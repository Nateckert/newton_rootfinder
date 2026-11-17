//! What the model adds to the log: its names, its values, its messages, and the evaluations nested resolutions belong to

use std::error::Error;
use std::fmt;

use newton_rootfinder as nrf;
use nrf::log::{Description, Detail, Fields, Log};
use nrf::model::{Model, ModelError, UserModelFromClosure};
use nrf::residuals;

use super::Buffer;
use super::resolutions::{of_event, solve};

/// The diameter of a duct and the velocity in it, from a flow and a pressure drop
struct Duct {
    x: nalgebra::DVector<f64>,
    velocity: f64,
    left: nalgebra::DVector<f64>,
}

impl Model<nalgebra::Dyn> for Duct {
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
        self.velocity = 2.0 / (self.x[0] * self.x[0]);
        self.left = nalgebra::DVector::from_vec(vec![self.velocity * self.velocity]);
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(self.left.clone(), nalgebra::DVector::from_vec(vec![16.0]))
    }

    fn describe(&self, description: &mut Description) {
        description.iterative(0, "diameter").with("unit", "m");
        description
            .residual(0, "pressure_drop = velocity^2")
            .with("source", "duct.model:12")
            .with("unit", "Pa");
        // Beyond the dimension: ignored
        description.iterative(1, "nothing");
    }

    fn report(&self, values: &mut Fields) {
        values
            .number("velocity", self.velocity)
            .text("regime", "laminar");
    }
}

#[test]
fn the_model_names_its_iteratives_and_residuals_and_reports_its_values() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let mut duct = Duct {
        x: nalgebra::DVector::from_vec(vec![1.0]),
        velocity: 0.0,
        left: nalgebra::DVector::zeros(1),
    };
    solve(&log, "duct", &mut duct, 60).unwrap();
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert!(lines[1].contains(
        "\"iteratives\":[{\"name\":\"diameter\",\"details\":{\"unit\":\"m\"},\"settings\":{"
    ));
    assert!(lines[1].contains(
        "\"residuals\":[{\"name\":\"pressure_drop = velocity^2\",\
         \"details\":{\"source\":\"duct.model:12\",\"unit\":\"Pa\"},\"settings\":{"
    ));
    assert!(!lines[1].contains("nothing"));
    assert!(
        lines[2]
            .ends_with(",\"max_error\":12.0,\"values\":{\"velocity\":2.0,\"regime\":\"laminar\"}}")
    );
    let evaluations = of_event(&lines, "evaluation");
    assert!(evaluations.len() > 2);
    for evaluation in evaluations {
        assert!(evaluation.contains(",\"values\":{\"velocity\":"));
    }
}

#[test]
fn a_model_without_a_description_is_written_by_indices() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let function = |x: &nalgebra::DVector<f64>| x.map(|value| value * value - 2.0);
    let mut model = UserModelFromClosure::new(1, &function);
    model.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
    solve(&log, "square root of 2", &mut model, 60).unwrap();
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert!(lines[1].contains("\"iteratives\":[{\"settings\":{"));
    assert!(lines[1].contains("\"residuals\":[{\"settings\":{"));
    assert!(!lines[2].contains("values"));
}

#[derive(Debug)]
struct Extrapolated;

impl fmt::Display for Extrapolated {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "the table is extrapolated")
    }
}

impl Error for Extrapolated {}

/// x² = 2, the model finding its values inaccurate beyond 1.2
struct Tabulated {
    x: nalgebra::DVector<f64>,
    left: nalgebra::DVector<f64>,
}

impl Model<nalgebra::Dyn> for Tabulated {
    type InaccurateValuesError = Extrapolated;
    type UnusableValuesError = std::convert::Infallible;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>) {
        self.x = iteratives.clone();
    }

    fn get_iteratives(&self) -> nalgebra::DVector<f64> {
        self.x.clone()
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Dyn>> {
        self.left = self.x.component_mul(&self.x);
        if self.x[0] > 1.2 {
            return Err(ModelError::InaccurateValuesError(Extrapolated));
        }
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(self.left.clone(), nalgebra::DVector::from_vec(vec![2.0]))
    }
}

#[test]
fn the_message_of_an_inaccurate_evaluation_is_written_on_it() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone());
    let mut model = Tabulated {
        x: nalgebra::DVector::from_vec(vec![1.0]),
        left: nalgebra::DVector::zeros(1),
    };
    // Converges at 1.414, inaccurate: the final evaluation fails
    assert!(solve(&log, "tabulated", &mut model, 60).is_err());
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    let evaluations = of_event(&lines, "evaluation");
    assert!(!evaluations[0].contains("inaccurate"));
    for evaluation in &evaluations[1..] {
        assert!(evaluation.contains(",\"inaccurate\":\"the table is extrapolated\""));
    }
    assert!(
        lines
            .last()
            .unwrap()
            .contains("\"outcome\":\"final_evaluation_error\"")
    );
}

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

#[test]
fn a_nested_resolution_says_which_evaluation_of_its_parent_it_belongs_to() {
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
    assert!(!starts[0].contains("during"));
    assert!(starts[1].contains(
        ",\"parent\":1,\"during\":{\"iteration\":0,\"of\":\"guess\"},\"time\":0,\"label\":\"inner\""
    ));
    assert!(starts[2].contains(
        ",\"parent\":1,\"during\":{\"iteration\":1,\"of\":\"jacobian\",\"column\":0},\"time\":0,"
    ));
    assert!(
        starts[3]
            .contains(",\"parent\":1,\"during\":{\"iteration\":1,\"of\":\"step\"},\"time\":0,")
    );
}
