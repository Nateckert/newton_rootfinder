//! A solver reused for several resolutions: every call to `solve` starts afresh.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use newton_rootfinder as nrf;
use nrf::iteratives;
use nrf::model::{Model, ModelError};
use nrf::residuals;
use nrf::solver::{QuasiNewtonMethod, ResolutionMethod, UpdateQuasiNewtonMethod};

/// `x² = 2` from `x`, counting its evaluations, each of them inaccurate when asked.
struct Square2 {
    x: f64,
    residual: f64,
    evaluations: usize,
    inaccurate: bool,
}

impl Square2 {
    fn new(x: f64, inaccurate: bool) -> Self {
        Square2 {
            x,
            residual: 0.0,
            evaluations: 0,
            inaccurate,
        }
    }
}

#[derive(Debug)]
struct Inaccurate;

impl fmt::Display for Inaccurate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Inaccurate value")
    }
}

impl Error for Inaccurate {}

impl Model<nalgebra::Const<1>> for Square2 {
    type InaccurateValuesError = Inaccurate;
    type UnusableValuesError = Infallible;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.x = iteratives[0];
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        nalgebra::SVector::from([self.x])
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Const<1>>> {
        self.evaluations += 1;
        self.residual = self.x * self.x - 2.0;
        if self.inaccurate {
            Err(ModelError::InaccurateValuesError(Inaccurate))
        } else {
            Ok(())
        }
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Const<1>> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.residual]),
            nalgebra::SVector::zeros(),
        )
    }
}

/// `x = 10`, already solved when its iterative is `10`:
/// a resolution converged before any iteration.
struct AlreadySolved {
    x: f64,
}

impl Model<nalgebra::Const<1>> for AlreadySolved {
    type InaccurateValuesError = Infallible;
    type UnusableValuesError = Infallible;
    type Memory = ();

    fn set_iteratives(&mut self, iteratives: &nalgebra::SVector<f64, 1>) {
        self.x = iteratives[0];
    }

    fn get_iteratives(&self) -> nalgebra::SVector<f64, 1> {
        nalgebra::SVector::from([self.x])
    }

    fn evaluate(&mut self) -> Result<(), ModelError<Self, nalgebra::Const<1>>> {
        Ok(())
    }

    fn get_residuals(&self) -> residuals::ResidualsValues<nalgebra::Const<1>> {
        residuals::ResidualsValues::new(
            nalgebra::SVector::from([self.x]),
            nalgebra::SVector::from([10.0]),
        )
    }
}

fn solver<'a>(
    method: ResolutionMethod,
    max_iter: usize,
    iteratives_configuration: &'a iteratives::Iteratives<'a, iteratives::IterativeParamsFD>,
    residuals_configuration: &'a residuals::ResidualsConfig<'a>,
) -> nrf::solver::RootFinder<'a, iteratives::IterativeParamsFD, nalgebra::Const<1>> {
    let parameters = nrf::solver::SolverParameters::new(1e-6, max_iter, method, false);
    nrf::solver::RootFinder::new(
        parameters,
        iteratives_configuration,
        residuals_configuration,
    )
}

#[test]
fn a_second_resolution_has_all_its_iterations() {
    let iteratives_parameters = [iteratives::IterativeParamsFD::default()];
    let iteratives_configuration = iteratives::Iteratives::new(&iteratives_parameters);
    let stopping_criteria = [residuals::NormalizationMethod::Abs];
    let update_methods = [residuals::NormalizationMethod::Abs];
    let residuals_configuration =
        residuals::ResidualsConfig::new(&stopping_criteria, &update_methods);
    // From 10, the Newton-Raphson resolution takes 6 iterations: a second
    // one with the count carried over would run out of iterations.
    let mut rf = solver(
        ResolutionMethod::NewtonRaphson,
        8,
        &iteratives_configuration,
        &residuals_configuration,
    );

    let mut first = Square2::new(10.0, false);
    rf.solve(&mut first).unwrap();
    let mut second = Square2::new(10.0, false);
    rf.solve(&mut second).unwrap();

    assert!((first.x - std::f64::consts::SQRT_2).abs() < 1e-6);
    assert_eq!(second.x, first.x);
    assert_eq!(second.evaluations, first.evaluations);
}

#[test]
fn a_second_quasi_newton_resolution_computes_its_jacobian() {
    let iteratives_parameters = [iteratives::IterativeParamsFD::default()];
    let iteratives_configuration = iteratives::Iteratives::new(&iteratives_parameters);
    let stopping_criteria = [residuals::NormalizationMethod::Abs];
    let update_methods = [residuals::NormalizationMethod::Abs];
    let residuals_configuration =
        residuals::ResidualsConfig::new(&stopping_criteria, &update_methods);

    for method in [
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::StationaryNewton),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::JacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
        ResolutionMethod::QuasiNewton(QuasiNewtonMethod::InverseJacobianUpdate(
            UpdateQuasiNewtonMethod::BroydenFirstMethod,
        )),
    ] {
        let mut rf = solver(
            method,
            50,
            &iteratives_configuration,
            &residuals_configuration,
        );

        let mut first = Square2::new(2.0, false);
        rf.solve(&mut first).unwrap();
        let mut second = Square2::new(2.0, false);
        rf.solve(&mut second).unwrap();

        // The same resolution twice: the jacobian of the second one is
        // computed by finite difference at its first iteration, not kept
        // from the first one, which would save an evaluation and change
        // the iterates.
        assert!(
            (first.x - std::f64::consts::SQRT_2).abs() < 1e-6,
            "{method}"
        );
        assert_eq!(second.x, first.x, "{method}");
        assert_eq!(second.evaluations, first.evaluations, "{method}");
    }
}

#[test]
fn the_validity_of_the_last_evaluation_does_not_outlive_a_resolution() {
    let iteratives_parameters = [iteratives::IterativeParamsFD::default()];
    let iteratives_configuration = iteratives::Iteratives::new(&iteratives_parameters);
    let stopping_criteria = [residuals::NormalizationMethod::Abs];
    let update_methods = [residuals::NormalizationMethod::Abs];
    let residuals_configuration =
        residuals::ResidualsConfig::new(&stopping_criteria, &update_methods);
    let mut rf = solver(
        ResolutionMethod::NewtonRaphson,
        50,
        &iteratives_configuration,
        &residuals_configuration,
    );

    // Converged on inaccurate values: the resolution ends on them.
    let mut inaccurate = Square2::new(10.0, true);
    assert!(matches!(
        rf.solve(&mut inaccurate),
        Err(nrf::errors::SolverError::FinalEvaluationError)
    ));

    // Converged at the initial guess, without an iteration: nothing of
    // the previous resolution makes this one end on inaccurate values.
    let mut solved = AlreadySolved { x: 10.0 };
    rf.solve(&mut solved).unwrap();
    assert_eq!(solved.x, 10.0);
}
