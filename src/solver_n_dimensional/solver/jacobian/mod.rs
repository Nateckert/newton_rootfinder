mod factorization;
mod jacobian_analytic;
mod jacobian_approximation;
mod jacobian_finite_diff;
mod jacobian_inverse_approximation;
mod jacobian_struct;
#[cfg(test)]
mod tests;

pub(crate) use jacobian_analytic::evaluate_jacobian_from_analytical_function;
#[cfg(test)]
pub(crate) use jacobian_finite_diff::compute_jacobian_from_finite_difference;
pub(crate) use jacobian_finite_diff::evaluate_jacobian_from_finite_difference_at;
pub(crate) use jacobian_struct::JacobianMatrix;

pub(crate) use jacobian_approximation::approximate_jacobian;
pub(crate) use jacobian_inverse_approximation::approximate_inv_jacobian;
