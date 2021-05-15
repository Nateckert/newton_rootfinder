# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

## Unrealesed
### Added
- License file : the licensing hasn't been changed but has been made more explicited with the adequate section in the readme and the licenses files
### Changed
- Improve documentation and rewrite of the README.md
- *Breaking change*: Define API thanks to rexport, access to `newton_rootfinder::solver_advanced` has been deleted, use directly `newton_rootfinder` from now on.
- *Breaking change*: Use `UserModelFromFunc` and `UserModelFromFuncAndJacobian` instead of `UserModelWithFunc` and `UserModelWithFuncJac`
- Moved from nalgebra 0.21.1 to 0.26.2

### Removed
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
