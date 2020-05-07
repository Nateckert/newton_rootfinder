//! Test cases taken from the following references :
//!
//! Broyden, C.G. [1965] :
//! A class of methods for solving nonlinear simultaneous equations,
//! Mathematics of Computation 19, p 577-593
//! https://doi.org/10.1090/S0025-5718-1965-0198670-6

//! Powell, M. J. D. [1964]
//! A method for minimizing a sum of squares of non-linear functions without calculating derivatives,
//! Comput. J., v. 7, 1965, pp. 303-307.
//! https://doi.org/10.1093/comjnl/7.4.303

extern crate nalgebra;

// Case 5-8 are to be found in [1965] p. 587 and the parameters p. 590 :
// +------------+-------------+--------------+------------+
// |   Case     | Dimension   |   alpha      |    beta    |
// +============+=============+==============+============+
// |      5     |    n = 5    |   - 0.1      |    1.0     |
// +------------+-------------+--------------+------------+
// |      6     |    n = 5    |   - 0.5      |    1.0     |
// +------------+-------------+--------------+------------+
// |      7     |    n = 10   |   - 0.5      |    1.0     |
// +------------+-------------+--------------+------------+
// |      8     |    n = 20   |   - 0.5      |    1.0     |
// +------------+-------------+--------------+------------+
// Init to be taken is (-1, -1, ..., -1)
pub fn init_broyden1965_case5() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 5])
}
pub fn init_broyden1965_case6() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 5])
}
pub fn init_broyden1965_case7() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 10])
}
pub fn init_broyden1965_case8() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 20])
}

pub fn broyden1965_case5(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    broyden1965_cases5to8(&x, -0.1, 1.0)
}

pub fn broyden1965_case6(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    broyden1965_cases5to8(&x, -0.5, 1.0)
}

pub fn broyden1965_case7(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    broyden1965_cases5to8(&x, -0.5, 1.0)
}

pub fn broyden1965_case8(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    broyden1965_cases5to8(&x, -0.5, 1.0)
}

fn broyden1965_cases5to8(
    x: &nalgebra::DVector<f64>,
    alpha: f64,
    beta: f64,
) -> nalgebra::DVector<f64> {
    let n = x.len();
    let mut outputs = nalgebra::DVector::zeros(n);

    outputs[0] = -(3.0 + alpha * x[0]) * x[0] + 2.0 * x[1] - beta;
    for i in 1..n - 1 {
        outputs[i] = outputs[i - 1] - (3.0 + alpha * x[i]) * x[i] + 2.0 * x[i + 1] - beta;
    }
    outputs[n - 1] = outputs[n - 2] - (3.0 + alpha * x[n - 1]) * x[n - 1] - beta;

    outputs
}

// Case 9 is found in [1965] p. 587 and adapted from a minimization of [1964]
// Dimension is 2

pub fn init_broyden1965_case9() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.2, 1.0])
}

pub fn broyden1965_case9(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(2);
    outputs[0] = 10.0 * (x[1] - x[0].powi(2));
    outputs[1] = 1.0 - x[0];
    outputs
}

// Case 10 is found in [1965] p. 587
// Dimension is 2

pub fn init_broyden1965_case10() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![15.0, -2.0])
}

pub fn broyden1965_case10(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(2);
    outputs[0] = -13.0 + x[0] + ((-x[1] + 5.0) * x[1] - 2.0) * x[1];
    outputs[1] = -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1];
    outputs
}
