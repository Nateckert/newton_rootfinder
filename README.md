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

For the comparison with other crates that implements several fonctionalities, the module of those crate has been explicited.

Note: Crates may have evolved since this comparison was established.

If you feel that this comparison needs an update, don't hesite to open an issue or to do a pull request !

### N-dimensional solver

The following crates are providing solvers based on Newton's method for n-dimensional systems.

`newton_rootfinder` aims to be the reference crate for n-dimensional rootfinding algorithms and is focused only on that topic.


| crate  | version | Advanced <br> Parametrization | Simulation <br> Log | Other iterative<br> algorithms |
|--------|--------:|:-----------------------------:|:-------------------:|:------------------------------:|
| `newton_rootfinder`         |   0.9.0 |  ✔️  |  ✔️  |  ✔️    |
| peroxide::numerical::newton |  0.30.9 |  ❌ |  ❌  |   ❌  |

bacon_sci::roots (0.12.0) also implements n-dimensional methods, but only for polynomials.

### 1-dimensional solver

If you are looking crates for one dimensional solver, `newton_rootfinder` is not the right one for you, but the rust ecosystem has several options :


| crate | version | Newton-Raphson | Other Iterative methods | Analytical methods  | Error handling |
|------ |--------:|:--------------:|:-----------------------:|:-------------------:|:--------------:|
| newton-raphson |   0.1.0 |  ✔️    |  ❌  |  ❌  |  ❌  |
| nrfind         |   1.0.3 |  ✔️    |  ❌  |  ❌  |  ✔️   |
| rootfind       |   0.7.0 |  ✔️    |  ✔️   |  ❌  |  ✔️   |
| roots          |   0.6.0 |  ✔️    |  ✔️   |  ✔️  |   ✔️  |
| bacon_sci      |  0.11.0 |  ✔️    |  ✔️   |  ❌  |  ✔️   |

## Contribution

Check the [CONTRIBUTE.md](./CONTRIBUTE.md) file, help is always welcome !

## License

`newton_rootfinder` is dual licensed, you can choose to use it with either:

- [Apache-2.0 license](http://opensource.org/licenses/APACHE-2.0) 
- [MIT license](https://opensource.org/licenses/MIT)
