use super::{deriv_normalization, NormalizationMethod};
use std::fmt;

/// Residuals values outputs of the model
///
/// This is the expected output of the model in order to be able to interact with the solver
///
/// It is containing arrays of:
/// - the left members of the equations
/// - the right members of the equations
///
/// Once converged, one should have left = right (with a tolerance)
#[derive(Debug)]
pub struct ResidualsValues {
    left: nalgebra::DVector<f64>,
    right: nalgebra::DVector<f64>,
    problem_size: usize,
}

impl ResidualsValues {
    pub fn new(left: nalgebra::DVector<f64>, right: nalgebra::DVector<f64>) -> Self {
        if left.len() != right.len() {
            panic!(
                "Dimension mismatch in the residuals values {} != {} ",
                left.len(),
                right.len()
            );
        }

        let problem_size = left.len();

        ResidualsValues {
            left,
            right,
            problem_size,
        }
    }

    pub fn len(&self) -> usize {
        self.problem_size
    }

    pub fn get_values(&self, index: usize) -> (f64, f64) {
        (self.left[index], self.right[index])
    }

    pub fn get_values_str_eq(&self, index: usize, float_width: usize) -> String {
        let mut str_eq = String::new();
        str_eq.push_str(&format!(
            "{:width$} = {:width$}",
            self.left[index].to_string(),
            self.right[index],
            width = float_width
        ));
        str_eq
    }
}

/// Residuals jacobian values outputs of the model
///
/// This is the expected jacobian output of the model in order to be able to interact with the solver
///
/// It is containing arrays of:
/// - the jacobian left members of the equations
/// - the jacobian right members of the equations
///
/// The jacobian of the left and right members are required,
/// as the output jacobian value depends of the normalization method and both members are required to compute it
#[derive(Debug)]
pub struct JacobianValues {
    left: nalgebra::DMatrix<f64>,
    right: nalgebra::DMatrix<f64>,
    problem_size: usize,
}

impl JacobianValues {
    pub fn new(left: nalgebra::DMatrix<f64>, right: nalgebra::DMatrix<f64>) -> Self {
        if left.shape() != right.shape() {
            panic!(
                "Dimension mismatch between the jacobians {:?} != {:?}",
                left.shape(),
                right.shape()
            );
        }
        let (n, m) = left.shape();
        if n != m {
            panic!("Jacobian matrix are not squared {} != {}", n, m);
        }
        let problem_size = n;
        JacobianValues {
            left,
            right,
            problem_size,
        }
    }

    pub fn normalize(
        &self,
        res_values: &ResidualsValues,
        norm_methods: &[NormalizationMethod],
    ) -> nalgebra::DMatrix<f64> {
        let mut jac: nalgebra::DMatrix<f64> =
            nalgebra::DMatrix::zeros(self.problem_size, self.problem_size);

        // iterate over rows
        for i in 0..self.problem_size {
            let (left_value, right_value) = res_values.get_values(i);
            // iterate over columns
            for j in 0..self.problem_size {
                jac[(i, j)] = deriv_normalization(
                    left_value,
                    right_value,
                    self.left[(i, j)],
                    self.right[(i, j)],
                    norm_methods[i],
                );
            }
        }
        jac
    }

    pub fn get_jacobians(&self) -> (&nalgebra::DMatrix<f64>, &nalgebra::DMatrix<f64>) {
        (&self.left, &self.right)
    }
}

impl fmt::Display for ResidualsValues {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = String::from("Residuals values :\n\n");

        for (i, elt) in self.left.iter().enumerate() {
            let res = format!("Eq {} : {} = {}\n", i, elt, self.right[i]);
            result.push_str(&res);
        }

        write!(f, "{}", result)
    }
}
