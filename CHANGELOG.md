# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

## [Unreleased]
### Added
- Travis CI

### Changed
- Testing of log generation: the user info are not checked anymore (so TRAVIS or anybody else could run it)
- Documentation of the `util` module: fixed a typo

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
