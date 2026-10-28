# newton_rootfinder


![crates.io](https://img.shields.io/crates/v/newton_rootfinder.svg)
![Build Status](https://github.com/Nateckert/newton_rootfinder/actions/workflows/ci.yml/badge.svg)

`newton_rootfinder` is a scientific computing rust crate to solve nonlinear equation systems thanks to [rootfinding methods](https://en.wikipedia.org/wiki/Root-finding_algorithms).

## Focus of the crate

This crate implements [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) and other derived methods.

To see a visualization of this method, you can watch this video: [Newton's fractal](https://www.youtube.com/watch?v=-RdOwhmqP5s)

This crate provides methods applicable for n-dimensional problems.

It provides several parametrization options and a simulation log to monitor the resolution process.

## Out of scope

This crate does NOT provide a solver for nonlinear differential equations.

This crate does NOT provide a specific solver for one dimensional problems.

## Documentation

Check the documentation on [doc.rs](https://docs.rs/newton_rootfinder/)

## Comparison with other rust crates

For the crates that implement several features, the module compared is named.

Note: crates may have evolved since this comparison was last updated (October 2026).

If you feel that this comparison needs an update, don't hesitate to open an issue or to do a pull request!

### N-dimensional solver

The following crates provide solvers for n-dimensional systems of nonlinear equations.

`newton_rootfinder` aims to be the reference crate for n-dimensional rootfinding algorithms and is focused only on that topic.


| crate  | version | Advanced <br> Parametrization | Simulation <br> Log | Other iterative<br> algorithms |
|--------|--------:|:-----------------------------:|:-------------------:|:------------------------------:|
| `newton_rootfinder`                    |  0.12.0 |  ✔️  |  ✔️  |  ✔️  |
| peroxide::numerical::{newton, root}    |  0.43.1 |  ❌  |  ❌  |  ✔️  |
| bacon_sci::roots                       |  0.16.2 |  ❌  |  ❌  |  ✔️  |
| gomez                                  |   0.5.1 |  ✔️  |  ❌  |  ✔️  |

peroxide provides Newton's method with automatic differentiation and Broyden's method.
bacon_sci provides Newton's method with an analytical jacobian and the secant method, for systems of a size known at compile time.
gomez provides a trust region method (Powell's dogleg, with a Levenberg-Marquardt step when the Newton direction cannot be used) with a jacobian computed by finite difference, and bounds on the variables.

### 1-dimensional solver

If you are looking for a one dimensional solver, `newton_rootfinder` is not the right crate for you, but the rust ecosystem has several options:


| crate | version | Newton-Raphson | Other Iterative methods | Analytical methods  | Error handling |
|------ |--------:|:--------------:|:-----------------------:|:-------------------:|:--------------:|
| newton-raphson            |   0.1.0 |  ✔️  |  ❌  |  ❌  |  ❌  |
| nrfind                    |   1.0.3 |  ✔️  |  ❌  |  ❌  |  ✔️  |
| rootfind                  |   0.7.0 |  ✔️  |  ✔️  |  ❌  |  ✔️  |
| roots                     |   0.0.8 |  ✔️  |  ✔️  |  ✔️  |  ✔️  |
| bacon_sci                 |  0.16.2 |  ✔️  |  ✔️  |  ❌  |  ✔️  |
| peroxide::numerical::root |  0.43.1 |  ✔️  |  ✔️  |  ❌  |  ✔️  |

## Contribution

Check the [CONTRIBUTE.md](./CONTRIBUTE.md) file, help is always welcome !

## License

`newton_rootfinder` is dual licensed, you can choose to use it with either:

- [Apache-2.0 license](http://opensource.org/licenses/APACHE-2.0) 
- [MIT license](https://opensource.org/licenses/MIT)
