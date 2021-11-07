/// x**2 - 2 = 0
/// Root: x = 2.sqrt() approx 1.4142
pub fn square2(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut y = x * x;
    y[0] -= 2.0;
    y
}

pub fn dsquare(x: &nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64> {
    let mut y = nalgebra::DMatrix::zeros(1, 1);
    y[(0, 0)] = 2.0 * x[0];
    y
}

/// Root with an high derivative to illustrate the early stopping problem if the stopping criteria is on the step-size
///
/// (1e9)*(x**9) - 1 = 0
///
/// Root:  x = 0.1
///
/// Residual at x=1: 0.9
///
/// Derivative at x=1: 9e9
///
/// Step-size = (0.9/9)e-9 = 1e-10
///
/// If the stopping criteria is on the step-size, it wouldn't iterate if initial guess was 1.
pub fn root_with_high_derivative(x: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let mut y = nalgebra::DVector::zeros(1);
    y[0] = 1e9 * x[0].powi(9) - 1.0;
    y
}

pub fn root_with_high_derivative_jac(x: &nalgebra::DVector<f64>) -> nalgebra::DMatrix<f64> {
    let mut y = nalgebra::DMatrix::zeros(1, 1);
    y[(0, 0)] = 9e9 * x[0].powi(8);
    y
}
