//! A log keeping the failures only

use newton_rootfinder as nrf;
use nrf::log::Log;
use nrf::model::{Model, ModelError, UserModelFromClosure, UserModelFromFunction};
use nrf::residuals;

use super::Buffer;
use super::resolutions::{of_event, solve};

fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut y = x.component_mul(x);
    y[0] -= 2.0;
    y
}

fn square_root_of_2(log: &Log, label: &str, max_iterations: usize) -> bool {
    let mut model = UserModelFromFunction::new(1, square2);
    model.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
    solve(log, label, &mut model, max_iterations).is_ok()
}

#[test]
fn of_a_sweep_only_the_failed_point_is_written_inside_its_scope() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).failures_only();
    {
        let _sweep = log.scope("sweep");
        assert!(square_root_of_2(&log, "point 1", 60));
        assert!(!square_root_of_2(&log, "point 2", 1));
        assert!(square_root_of_2(&log, "point 3", 60));
    }
    assert!(square_root_of_2(&log, "after the sweep", 60));
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    assert!(lines[1].starts_with("{\"event\":\"scope\",\"id\":1,\"parent\":null,"));
    assert!(lines[2].starts_with("{\"event\":\"start\",\"id\":3,\"parent\":1,"));
    assert!(lines[2].contains("\"label\":\"point 2\""));
    assert!(lines[3].starts_with("{\"event\":\"evaluation\",\"id\":3,"));
    let end = lines.len() - 2;
    assert!(lines[end].starts_with(
        "{\"event\":\"end\",\"id\":3,\"time\":0,\"outcome\":\"non_convergence_error\","
    ));
    assert!(lines[end + 1].starts_with("{\"event\":\"scope_end\",\"id\":1,"));
    for label in ["point 1", "point 3", "after the sweep"] {
        assert!(!lines.iter().any(|line| line.contains(label)), "{}", label);
    }
}

/// x such that the solution y of 2y = x is 2, from an inner resolution converging,
/// beside another one failing, whose failure the model ignores
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
        let linear =
            move |y: &nalgebra::DVector<f64>| nalgebra::DVector::from_vec(vec![2.0 * y[0] - x]);
        let mut converging = UserModelFromClosure::new(1, &linear);
        converging.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
        solve(&self.log, "converging", &mut converging, 60).unwrap();
        self.y = converging.get_iteratives();

        let mut failing = UserModelFromFunction::new(1, square2);
        failing.set_iteratives(&nalgebra::DVector::from_vec(vec![1.0]));
        assert!(solve(&self.log, "failing", &mut failing, 1).is_err());
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Dyn> {
        residuals::ResidualsValues::new(self.y.clone(), nalgebra::DVector::from_vec(vec![2.0]))
    }
}

#[test]
fn of_nested_resolutions_the_failed_ones_are_written_under_their_parent() {
    let buffer = Buffer::default();
    let log = Log::new(buffer.clone()).failures_only();
    let mut outer = Outer {
        x: nalgebra::DVector::from_vec(vec![1.0]),
        y: nalgebra::DVector::zeros(1),
        log: log.clone(),
    };
    solve(&log, "outer", &mut outer, 60).unwrap();
    log.close().unwrap();

    let lines = buffer.normalized_lines();
    // The outer resolution is written from its start, once an inner one failed, to its end
    assert!(lines[1].starts_with("{\"event\":\"start\",\"id\":1,\"parent\":null,"));
    assert!(lines[1].contains("\"label\":\"outer\""));
    assert!(
        lines
            .last()
            .unwrap()
            .starts_with("{\"event\":\"end\",\"id\":1,")
    );
    assert!(lines.last().unwrap().contains("\"outcome\":\"converged\""));

    // Each evaluation of the outer resolution wrote its failing inner one, and nothing of the converging one
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("\"label\":\"converging\""))
    );
    let starts = of_event(&lines, "start");
    assert!(starts.len() > 2);
    for start in &starts[1..] {
        assert!(start.contains(",\"parent\":1,"));
        assert!(start.contains("\"label\":\"failing\""));
    }
    let ends = of_event(&lines, "end");
    assert_eq!(starts.len(), ends.len());
    for end in &ends[..ends.len() - 1] {
        assert!(
            end.contains("\"outcome\":\"non_convergence_error\""),
            "{}",
            end
        );
    }
}
