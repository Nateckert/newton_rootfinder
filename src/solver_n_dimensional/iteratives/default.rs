/// Constructor with default values for iteratives parameters
///
/// # Examples
///```
/// use newton_rootfinder as nrf;
///
/// let size = 2;
/// let iteratives_vec = nrf::iteratives::default_vec_iteratives(size);
/// assert_eq!(iteratives_vec.len(), size);
/// for i in 0..size {
///     assert_eq!(iteratives_vec[i].get_min_value(), f64::NEG_INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_value(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
/// }
///```
pub fn default_vec_iteratives(size: usize) -> Vec<super::IterativeParams> {
    vec![super::IterativeParams::default(); size]
}

/// Constructor with default values for iteratives parameters with finite-differences
///
/// # Examples
///```
/// use newton_rootfinder as nrf;
///
/// let size = 2;
/// let iteratives_vec = nrf::iteratives::default_vec_iteratives_fd(size);
/// assert_eq!(iteratives_vec.len(), size);
/// for i in 0..size {
///     assert_eq!(iteratives_vec[i].get_min_value(), f64::NEG_INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_value(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_max_step_abs(), f64::INFINITY);
///     assert_eq!(iteratives_vec[i].get_dx_abs(), 5e-8);
///     assert_eq!(iteratives_vec[i].get_dx_rel(), 5e-8);
///     assert_eq!(iteratives_vec[i].get_perturbation_method(), nrf::iteratives::PerturbationMethod::Max);
/// }
///```
pub fn default_vec_iteratives_fd(size: usize) -> Vec<super::IterativeParamsFD> {
    vec![super::IterativeParamsFD::default(); size]
}
