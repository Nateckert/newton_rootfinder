# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- A log of the resolutions in [JSON Lines](https://jsonlines.org/), one log for every resolution of a program:
  `log::Log`, given to a solver with `RootFinder::log_to`.
  See the [documentation of the `log` module](https://docs.rs/newton_rootfinder/0.13.0/newton_rootfinder/log/index.html).
- `Iterative::log_settings`, by default doing nothing: the settings of an iterative in the log.

### Changed
- When possible (Newton-Raphson, stationary Newton and the methods updating the jacobian), the jacobian is no longer
  inverted: the equivalent linear system is factorized and solved, which speeds up the computation
  and gives more accurate results (#31). The iterates can differ in their last digits.
- `RootFinder` and `solver::default` no longer require the iteratives to implement `Display`.
- *Breaking change*: `SolverError::JacobianError`, which held a `SolverInternalError`, is replaced by two variants:
  `SolverError::JacobianEvaluationError`, holding the `ModelError` the model returned while the jacobian was evaluated,
  and `SolverError::NonInvertibleJacobian`, for a singular jacobian. Their `Display` texts are unchanged.
  `errors::SolverInternalError` is no longer public.

### Removed
- *Breaking change*: `RootFinder::activate_debug` and its text file, replaced by `RootFinder::log_to`.
- *Breaking change*: the feature `additional_log_info` and its dependencies `chrono`, `whoami` and `rustc_version_runtime`:
  a program writes such facts in the log with `Log::note`.

The public API is reduced to what a user of the crate calls: a model, its parametrization, the solver, its errors,
its log and the XML configuration. The machinery of a resolution is no longer exported.
The code compiled for a resolution is unchanged.
- *Breaking change*: `solver::JacobianMatrix`, `solver::evaluate_jacobian_from_analytical_function`,
  `solver::approximate_jacobian`, `solver::approximate_inv_jacobian` and
  `solver::compute_jacobian_from_finite_difference` are no longer public.
  `solver::evaluate_jacobian_from_finite_difference`, which nothing called, is removed.
- *Breaking change*: the update formulas of the quasi-Newton methods are no longer public:
  `solver::broyden_first_method_udpate_jac`, `broyden_first_method_udpate_inv_jac`,
  `broyden_second_method_udpate_jac`, `broyden_second_method_udpate_inv_jac`,
  `quasi_method_update_jac`, `quasi_method_update_inv_jac` and `greenstadt_second_method_udpate_jac`.
- *Breaking change*: `errors::NonInvertibleJacobian` is no longer public.
- *Breaking change*: `residuals::normalization` and `residuals::deriv_normalization` are no longer public.
  Their formulas are in the documentation of `NormalizationMethod`,
  and the values the solver computes with them are in its log.
- *Breaking change*: `residuals::StoppingCriteria`, which nothing used, is removed.
- *Breaking change*: `ResidualsConfig::evaluate_update_residuals` and `evaluate_stopping_residuals`,
  `Iteratives::step_limitations` and `compute_perturbations` are no longer public.
  What a configuration was given is still read back from it
  (`ResidualsConfig::len`, `get_update_methods`, `get_stopping_criterias`, `Iteratives::len`),
  and the `Iterative` trait still gives the step limitation and the perturbation of one iterative.
- *Breaking change*: `ResidualsValues` and `JacobianValues`, which a model builds for the solver, keep `new` only:
  `ResidualsValues::get_values`, `shape_generic`, `non_finite_indices`, `get_values_str_eq`
  and `JacobianValues::normalize`, `get_jacobians` are no longer public,
  `ResidualsValues::len` and `shape`, which nothing called, are removed.
- *Breaking change*: the fields of `UserModelFromFunction`, `UserModelFromFunctionAndJacobian`,
  `UserModelFromClosure` and `UserModelFromClosureAndJacobian` are private.
  Their iteratives are read with `Model::get_iteratives`.

### Fixed
- The Broyden and Greenstadt methods now update the jacobian, or its inverse, with the signed residuals instead of the absolute errors.

## [0.12.0] - 2026-10-28

### Added
- `Model::evaluate_column`, called for each column of a jacobian computed by finite-difference instead of `Model::evaluate`,
  the columns in increasing order: a model can skip the computations that do not depend on the iterative perturbed.
  The default implementation calls `Model::evaluate`.
- `SolverError::NonFiniteResiduals`, holding the indices of the equations concerned,
  returned when a residual is NaN or infinite at the initial evaluation or after an update.
- `SolverError::DimensionMismatch`, returned by `RootFinder::solve` before any evaluation
  when the model has another number of iteratives than the configurations of the solver.

### Changed
- *Breaking change*: `Model::len_problem` is removed, the dimension of the problem being the length of the iteratives (#16).
- *Breaking change*: `SolverParameters::new` no longer takes the problem size and `get_problem_size` is removed (#16).
  `RootFinder::new` checks that the iteratives and the residuals configurations have the same length, and panics otherwise;
  `solve` checks the model against them. The XML parser still reads the `problem_size` attribute and checks the other nodes against it.
- *Breaking change*: `RootFinder::new` no longer takes the initial guess: a resolution starts from the iteratives the model holds
  when `solve` is called, set with `Model::set_iteratives` or left by a previous resolution (#22).
  `default_with_guess` becomes `default`, without the guess.
  The dimension type `D` of a solver is inferred from the model it solves, or named when the solver is built before any model.
- *Breaking change*: the memory of a model is a value of its own type, the associated type `Model::Memory: Clone + Default`,
  instead of a `nalgebra::DVector<f64>`: `Model::get_memory(&self) -> Self::Memory` and `Model::set_memory(&mut self, &Self::Memory)`.
  Saving and restoring it no longer allocates, and the state no longer has to be encoded as `f64` (#35).
  A model without memory adds `type Memory = ();`; a model with memory declares its type,
  for example a struct of its fields or a statically sized vector, and implements both methods with it.
- *Breaking change*: `SolverError::ModelInitialEvaluationError` holds the error of the model, `M::UnusableValuesError`,
  instead of its text, as `SolverError::ModelEvaluationError` does for a later evaluation (#29).
  Its `Display` text is unchanged.
- *Breaking change*: `nalgebra` 0.33 becomes 0.35, the vectors and matrices of the public API being its types:
  a crate using `newton_rootfinder` moves to `nalgebra` 0.35 as well.
- The minimal `rustc` version goes from 1.69 to 1.89, the one required by `nalgebra` 0.35.
- `whoami` 1.5 becomes 2.1 for the feature `additional_log_info`:
  the OS, host and username the log can not read are written `unknown`.
- `criterion` 0.5 becomes 0.8 for the benches.
- The crate moves to the Rust 2024 edition, from 2021 (and `util` from 2018), with the formatting of its style.

### Fixed
- A NaN residual was reported converged: the maximum error and its comparison with the tolerance ignore NaN.
- The damping is triggered by a step giving a NaN or infinite residual,
  which the comparison of the maximum errors missed: such a step is damped instead of failing with `SolverError::NonFiniteResiduals`.
- The damping no longer panics when the model fails to evaluate the damped guess:
  the evaluation is handled as the one of the proposed guess,
  an `UnusableValuesError` ending the resolution with `SolverError::ModelEvaluationError` (#33).
- The damping evaluates the damped guess from the memory of the current point,
  restored with `Model::set_memory`, instead of the memory left by the rejected proposed guess (#34).
- The solver keeps the point it stands on (iteratives, residuals and memory of its last evaluation)
  instead of reading it back from the model, which holds the last perturbed point
  after a jacobian evaluated by finite-difference:
  the damping takes the midpoint and compares the errors from the current point,
  and the quasi-Newton updates measure their steps from it.
  The iterations of the quasi-Newton methods and of the damping can differ in the last digits.
- A resolution failing because the model or the jacobian could not be evaluated, or on a non-finite residual,
  gives the model back the iteratives and the memory it held when `RootFinder::solve` was called,
  instead of the ones of the failed evaluation, so that a new resolution can start from them.
  A `NonConvergenceError` or a `FinalEvaluationError` still leaves the model on the last iterate.
- `compute_jacobian_from_finite_difference` and `evaluate_jacobian_from_finite_difference` restore the memory
  of the reference point when the evaluation of a column fails, as after a successful column.
- `RootFinder::solve` starts every resolution afresh.
  The iteration count, the jacobian and the steps of the quasi-Newton updates were kept from the previous call,
  so a solver reused for several resolutions had fewer iterations left each time, failing with `NonConvergenceError`
  once the count reached the maximum, and the quasi-Newton methods started from the jacobian of the previous resolution
  instead of computing one (#24).
  The benchmarks, which reuse one solver, timed such failing resolutions: their results are measured again.

## [0.11.0] - 2024-10-17

### Changed
- Bump minimal `rustc` version from 1.56 to 1.69
- Upgrade `nalgebra` dependency from 0.31 to 0.33
- Upgrade optional dependency `whoami` from 1.2 to 1.5
- Upgrade optional dependency `rustc_version_runtime` from 0.2 to 0.3

## [0.10.0] - 2021-11-14

### Changed
- Upgrade `nalgebra` dependency from 0.29 to 0.31

## [0.9.1] - 2021-11-14

### Added
- Implement `fmt::debug` for `RootFinder` debugging

## [0.9.0] - 2021-11-7

### Added
- Error definition and handling

### Changed
- Upgrade to edition 2021 and add minimal supported rust version as 1.56
- Upgrade dependancies to nalgebra 0.29, whoami 1.2
- *Breaking change*: add minidom as optional under the feature name "xml_config_file"
- Relax dependancy constrant rustc_version_runtime from 0.2.0 to 0.2
- *Breaking change*: method signature `Rootfinder::solve(&mut self, model: &mut M)` to ``Rootfinder::solve(&mut self, model: &mut M) -> Result<(), crate::errors::SolverError>`
- *Breaking change*: method signature `Model::evaluate(&mut self)` to `Model::evaluate(&mut self) -> Result<(), model::ModelError>`
- *Breaking change*: method signature `Model::get_jacobian(&self) -> residuals::JacobianValues<D>` to `Model::get_jacobian(&mut self) -> Result<residuals::JacobianValues<D>, model::ModelError>` in the model trait.

## [0.8.0] - 2021-07-25

### Changed
- Upgrade to nalgebra 0.28
- Model trait accept now static types in addition to dynamic types (`SVector` and `DVector` from `nalgebra`)

## [0.7.0] - 2021-05-24

### Added
- License file : the licensing hasn't been changed but has been made more explicited with the adequate section in the readme and the licenses files
- New mechanism to adapt a user model frm closure through the `UserModelFromClosure` and `UserModelFromClosureAndJacobian` structs.

### Changed
- Improve documentation and rewrite of the README.md
- *Breaking change*: Define API thanks to rexport, access to `newton_rootfinder::solver_advanced` has been deleted, use directly `newton_rootfinder` from now on.
- *Breaking change*: Use `UserModelFromFunction` and `UserModelFromFunctionAndJacobian` instead of `UserModelWithFunction` and `UserModelWithFunctionJac`
- Moved from nalgebra 0.21.1 to 0.26.2

### Removed
- *Breaking change*: the `init` and `len_memroy` methods have been removed from the `model::Model` trait
- The minimal solver has been removed from the public API, it is still a dev dependency
- The test cases have been removed from the public API, it is still a dev dependency

## [0.6.0] - 2020-08-24

### Added
- rustc_version_runtime dependency to print rustc version information in log. This was removed in version 5.0 due to a documentation build issue on doc.rs of this dependency.
- New resolution methods: Greenstadt first and second method

### Changed
- Readme: it is not a duplicate of the documentation main page. It now points to it.
- Documentation: change of the main page.
- Simulation log function (changed from `set_debug()` to `activate_debug()`)
- Simulation log mecanism: instead of writing the log at the end of the simulation, write it on the fly in order to have data in case of a panic.
- Update version of dependencies: chrono and whoami.

## [0.5.0] - 2020-06-15

### Added
- Travis CI
- Add Broyden's Methods
- Add `solver/jacobian.rs` to guard the jacobian and its inverse updates
- Add `solver/resolution_method.rs` to implement quasi-Newton methods
- Refactoring the enums for the resolution method: `ResolutionMethod`, `QuasiNewtonMethod` and `UpdateQuasiNewtonMethod`
- Add solver placeholder for new resolution parameters

### Fixed
- Damping activation for quasi-Newton methods (the jacobian was recomputed if it hadn't been recomputed at the previous iteration and not the current iteration)

### Changed
- Upgrade dependency `nalgebra` from 0.21.0 to 0.21.1
- Upgrade dependency `whoami` from 0.8.1 to 0.8.2
- Add argument `damping` to the `default_with_guess` function
- Testing of log generation: the user info are not checked anymore (so TRAVIS or anybody else could run it)
- Documentation of the `util` module: fixed a typo
- Move `util/jacobian` under `solver/jacobian`

### Removed
- Dependency to rustc_version_runtime: blocker for doc.rs documentation build, see https://github.com/seppo0010/rustc-version-runtime-rs/issues/4#issue-630904639. As a consequence, the rustc version won't be displayed in the log file

## [0.4.0] - 2020-05-29

### Added
- New resolution method: StationaryNewton. The classical one has been named NewtonRaphson
- Added dependencies to chrono, whoami and rustc_version_runtime for log file informations

### Changed
- Log file formatting
- Adapt xml parser to read the resolution method
- Upgrade dev-dependency `float-cmp` from 0.6 to 0.8.0


## [0.3.0] - 2020-05-26

### Added
This is considered the base version with all the documented features working
