# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

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
- Update dev-dependencies float-cmp from 0.6 to 0.8.0


## [0.3.0] - 2020-05-26
### Added
This is considered the base version with all the documented features working
