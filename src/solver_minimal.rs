//! Minimal 1d solver
//! 
//! Two functions are provided for the cases where the derivative is provided or not :
//! - solver1d
//! - solver1d_fd (fd stands for finite differences)
//!
//! #Examples
//! ```
//! extern crate newton_rootfinder as nrf;
//! use nrf::solver_minimal::*;
//! extern crate float_cmp;
//!
//! pub fn square2(x: f64) -> f64 {
//!     x.powi(2)-2.0
//! }
//! pub fn dsquare(x: f64) -> f64 {
//! 2.0*x
//! }
//!
//! let x1 = solver1d(1.0, square2, dsquare, 50, 1e-6);
//! let x2 = solver1d_fd(1.0, square2, 50, 1e-6, 1e-8);
//! let x_sol = std::f64::consts::SQRT_2;
//!
//! println!("{}, {}", x1, x2);
//! assert!(float_cmp::approx_eq!(f64, x_sol, x1, epsilon = 1e-5));
//! assert!(float_cmp::approx_eq!(f64, x_sol, x2, epsilon = 1e-5));


pub fn solver1d(
    init_guess: f64,
    func: fn(f64) -> f64,
    deriv: fn(f64) -> f64,
    max_iter: usize,
    tol: f64,
) -> f64 {
    let mut iter = 0;
    let mut res = func(init_guess);
    let mut error = res.abs();
    let mut guess = init_guess;

    while error > tol && iter < max_iter {
        iter += 1;
        guess -= res / deriv(guess);
        res = func(guess);
        error = res.abs();
    }
    guess
}

pub fn solver1d_fd(
    init_guess: f64,
    func: fn(f64) -> f64,
    max_iter: usize,
    tol: f64,
    dx: f64,
) -> f64 {
    let mut iter = 0;
    let mut res = func(init_guess);
    let mut error = res.abs();
    let mut guess = init_guess;

    while error > tol && iter < max_iter {
        iter += 1;
        guess -= res / finite_diff(guess, res, func, dx);
        res = func(guess);
        error = res.abs();
    }
    guess
}

fn finite_diff(x: f64, f_ref: f64, func: fn(f64) -> f64, dx: f64) -> f64 {
    let fx = func(x + dx);
    (fx - f_ref) / dx
}
