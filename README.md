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


#### Methods

| crate | version | Newton-Raphson | Quasi-Newton | Jacobian | Far from the root |
|-------|--------:|----------------|--------------|----------|-------------------|
| `newton_rootfinder` | 0.12.0 | ✔️ | Broyden 1st/2nd, Greenstadt 1st/2nd (on the jacobian or its inverse), stationary Newton | analytical, finite difference | damping |
| peroxide::numerical | 0.43.1 | ✔️ (`newton`) | Broyden (`root`) | automatic differentiation; none for Broyden | ❌ |
| bacon_sci::roots | 0.16.2 | ✔️ | Broyden (`secant`, on the inverse) | analytical; finite difference to start `secant` | ❌ |
| gomez | 0.5.1 | inside a trust region | ❌ | finite difference, at every iteration | trust region (dogleg, Levenberg-Marquardt), LIPO to find a starting point |

#### Use in simulations

| crate | Per-variable settings | Convergence criteria | Model errors | Resolution log | Dimension |
|-------|-----------------------|----------------------|--------------|----------------|-----------|
| `newton_rootfinder` | step limits, bounds, finite-difference step | per residual, absolute/relative | `Result`, non-finite residuals detected | ✔️ file | static or dynamic |
| peroxide::numerical | ❌ | step norm; `newton` has no iteration limit | `Result` (`root`); none (`newton`) | ❌ | dynamic (`newton`), static (`root`) |
| bacon_sci::roots | ❌ | step norm | none (the function cannot fail) | ❌ | static |
| gomez | bounds (also used to scale the variables) | user closure on the residual norm | none (the function cannot fail) | `log` crate messages | dynamic |

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
