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
pub fn solution_broyden1965_case5() -> nalgebra::DVector<f64> {
    let mut solution = nalgebra::DVector::zeros(5);
    solution[4] = 15.0 - 235.0_f64.sqrt(); // approx -0.32970971675589134
    solution[3] = 15.0 - (235.0-20.0*solution[4]).sqrt(); // approx -0.5433006255144477
    solution[2] = 15.0 - (235.0-20.0*solution[3]).sqrt(); // approx -0.6801151944202548
    solution[1] = 15.0 - (235.0-20.0*solution[2]).sqrt(); // approx -0.5433006255144477
    solution[0] = 15.0 - (235.0-20.0*solution[1]).sqrt(); // approx -0.32970971675589134
    solution
}


pub fn init_broyden1965_case6() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 5])
}
pub fn solution_broyden1965_case6() -> nalgebra::DVector<f64> {
    let mut solution = nalgebra::DVector::zeros(5);
    solution[4] = 3.0 - 11_f64.sqrt(); // approx -0.3166247903553998
    solution[3] = 3.0 - (11.0 - 4.0*solution[4]).sqrt(); // approx -0.5023562299431505
    solution[2] = 3.0 - (11.0 - 4.0*solution[3]).sqrt(); // approx -0.606858039869687
    solution[1] = 3.0 - (11.0 - 4.0*solution[2]).sqrt(); // approx -0.6643460752880244
    solution[0] = 3.0 - (11.0 - 4.0*solution[1]).sqrt(); // approx -0.6955898448220816
    solution
}

pub fn init_broyden1965_case7() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 10])
}
pub fn solution_broyden1965_case7() -> nalgebra::DVector<f64> {
    let mut solution = nalgebra::DVector::zeros(10);
    solution[9] = 3.0 - 11_f64.sqrt(); // approx -0.3166247903553998
    solution[8] = 3.0 - (11.0 - 4.0*solution[9]).sqrt(); // approx -0.5023562299431505
    solution[7] = 3.0 - (11.0 - 4.0*solution[8]).sqrt(); // approx -0.606858039869687
    solution[6] = 3.0 - (11.0 - 4.0*solution[7]).sqrt(); // approx -0.6643460752880244
    solution[5] = 3.0 - (11.0 - 4.0*solution[6]).sqrt(); // approx -0.6955898448220816
    solution[4] = 3.0 - (11.0 - 4.0*solution[5]).sqrt(); // approx -0.7124600171972664
    solution[3] = 3.0 - (11.0 - 4.0*solution[4]).sqrt(); // approx -0.7215373259970219
    solution[2] = 3.0 - (11.0 - 4.0*solution[3]).sqrt(); // approx -0.726412390488751
    solution[1] = 3.0 - (11.0 - 4.0*solution[2]).sqrt(); // approx -0.7290279647590476
    solution[0] = 3.0 - (11.0 - 4.0*solution[1]).sqrt(); // approx -0.7304305192613079
    solution
}

pub fn init_broyden1965_case8() -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_vec(vec![-1.0; 20])
}
pub fn solution_broyden1965_case8() -> nalgebra::DVector<f64> {
    let mut solution = nalgebra::DVector::zeros(20);
    solution[19] = 3.0 - 11_f64.sqrt(); // approx -0.3166247903553998
    solution[18] = 3.0 - (11.0 - 4.0*solution[19]).sqrt(); // approx -0.5023562299431505
    solution[17] = 3.0 - (11.0 - 4.0*solution[18]).sqrt(); // approx -0.606858039869687
    solution[16] = 3.0 - (11.0 - 4.0*solution[17]).sqrt(); // approx -0.6643460752880244
    solution[15] = 3.0 - (11.0 - 4.0*solution[16]).sqrt(); // approx -0.6955898448220816
    solution[14] = 3.0 - (11.0 - 4.0*solution[15]).sqrt(); // approx -0.7124600171972664
    solution[13] = 3.0 - (11.0 - 4.0*solution[14]).sqrt(); // approx -0.7215373259970219
    solution[12] = 3.0 - (11.0 - 4.0*solution[13]).sqrt(); // approx -0.726412390488751
    solution[11] = 3.0 - (11.0 - 4.0*solution[12]).sqrt(); // approx -0.7290279647590476
    solution[10] = 3.0 - (11.0 - 4.0*solution[11]).sqrt(); // approx -0.7304305192613079
    solution[9]  = 3.0 - (11.0 - 4.0*solution[10]).sqrt(); // approx -0.7311823966465685
    solution[8]  = 3.0 - (11.0 - 4.0*solution[9]).sqrt();  // approx -0.7315853985385723
    solution[7]  = 3.0 - (11.0 - 4.0*solution[8]).sqrt();  // approx -0.7318013872866129
    solution[6]  = 3.0 - (11.0 - 4.0*solution[7]).sqrt();  // approx -0.7319171412487782
    solution[5]  = 3.0 - (11.0 - 4.0*solution[6]).sqrt();  // approx -0.7319791753163778
    solution[4]  = 3.0 - (11.0 - 4.0*solution[5]).sqrt();  // approx -0.7320124197630307
    solution[3]  = 3.0 - (11.0 - 4.0*solution[4]).sqrt();  // approx -0.7320302355490265
    solution[2]  = 3.0 - (11.0 - 4.0*solution[3]).sqrt();  // approx -0.7320397830403826
    solution[1]  = 3.0 - (11.0 - 4.0*solution[2]).sqrt();  // approx -0.7320448995371867
    solution[0]  = 3.0 - (11.0 - 4.0*solution[1]).sqrt();  // approx -0.7320476414628936
    solution
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
pub fn solution_broyden1965_case9() -> nalgebra::DVector<f64> {
    let mut solution = nalgebra::DVector::zeros(2);
    solution[0] = 1.0;
    solution[1] = 1.0;
    solution
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
pub fn solution_broyden1965_case10() -> nalgebra::DVector<f64> {
    let mut solution = nalgebra::DVector::zeros(2);
    solution[0] = 5.0;
    solution[1] = 4.0;
    solution
}

pub fn broyden1965_case10(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut outputs = nalgebra::DVector::zeros(2);
    outputs[0] = -13.0 + x[0] + ((-x[1] + 5.0) * x[1] - 2.0) * x[1];
    outputs[1] = -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1];
    outputs
}