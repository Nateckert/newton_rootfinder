use super::{normalization, NormalizationMethod, ResidualsValues};
use std::fmt;

/// Single residual configuration
///
/// A residual is constituded of two elements:
/// - the way of computing the `stopping_critera` from the left and right part of a residual
/// - the way of computing the error for the update (`update_method`) used by the rootfinder
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ResidualConfig {
    stopping_critera: NormalizationMethod,
    update_method: NormalizationMethod,
}

impl Default for ResidualConfig {
    fn default() -> ResidualConfig {
        ResidualConfig {
            stopping_critera: NormalizationMethod::Abs,
            update_method: NormalizationMethod::Abs,
        }
    }
}

impl ResidualConfig {
    pub fn new(stopping_critera: NormalizationMethod, update_method: NormalizationMethod) -> Self {
        ResidualConfig {
            stopping_critera,
            update_method,
        }
    }

    pub fn get_update_method(self) -> NormalizationMethod {
        self.update_method
    }
    pub fn get_stopping_criteria(self) -> NormalizationMethod {
        self.stopping_critera
    }
}

/// Residuals configuration used by the solver
///
/// The solver is using directly two slices to perform its calculation
/// - the `update_methods` used for computing the jacobian
/// - the `stopping_criterias` used to control if another iteration is performed
///
/// It is possible to used a `Vec<ResidualConfig>` to create such a struct thanks to the `convert_into_vecs()` method.
///
/// However, if the performance is critical for the user,
/// it should create is own arrays to feed to the `new()` constructor
/// and not use `ResidualConfig` (singular)
#[derive(Debug, PartialEq)]
pub struct ResidualsConfig<'a> {
    stopping_criterias: &'a [NormalizationMethod],
    update_methods: &'a [NormalizationMethod],
    length: usize,
}

impl<'a> ResidualsConfig<'a> {
    pub fn new(
        stopping_criterias: &'a [NormalizationMethod],
        update_methods: &'a [NormalizationMethod],
    ) -> Self {
        let length = stopping_criterias.len();
        if stopping_criterias.len() != update_methods.len() {
            panic!(
                "Dimension mismatch between stopping_criteras and update_methods {} != {}",
                stopping_criterias.len(),
                update_methods.len()
            );
        }

        ResidualsConfig {
            stopping_criterias,
            update_methods,
            length,
        }
    }

    /// Method to generate the vector of `stopping_criteras` and `update_methods` from a vector of `ResidualConfig`
    pub fn convert_into_vecs(
        residuals_config: Vec<ResidualConfig>,
    ) -> (Vec<NormalizationMethod>, Vec<NormalizationMethod>) {
        let length = residuals_config.len();
        let mut stopping_criterias = Vec::with_capacity(length);
        let mut update_methods = Vec::with_capacity(length);

        for elt in residuals_config {
            stopping_criterias.push(elt.get_stopping_criteria());
            update_methods.push(elt.get_update_method());
        }

        (stopping_criterias, update_methods)
    }

    pub fn len(&self) -> usize {
        self.length
    }

    /// Evaluation of the value of the update residuals thanks to the `normalization()` function
    pub fn evaluate_update_residuals<D>(
        &self,
        values: &ResidualsValues<D>,
    ) -> nalgebra::OVector<f64, D>
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let mut update_residuals: nalgebra::OVector<f64, D> =
            super::super::ovector_zeros_from_shape(values.shape_generic());

        for (i, &update_method) in self.update_methods.iter().enumerate() {
            let (left, right) = values.get_values(i);
            update_residuals[i] = normalization(left, right, update_method);
        }
        update_residuals
    }

    /// Evaluation of the value of the stopping residuals thanks to the `normalization()` function
    pub fn evaluate_stopping_residuals<D>(
        &self,
        values: &ResidualsValues<D>,
    ) -> nalgebra::OVector<f64, D>
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let mut stopping_residuals: nalgebra::OVector<f64, D> =
            super::super::ovector_zeros_from_shape(values.shape_generic());

        for (i, &stopping_criteria) in self.stopping_criterias.iter().enumerate() {
            let (left, right) = values.get_values(i);
            stopping_residuals[i] = normalization(left, right, stopping_criteria).abs();
        }
        stopping_residuals
    }

    pub fn get_update_methods(&self) -> &'a [NormalizationMethod] {
        self.update_methods
    }

    pub fn get_stopping_criterias(&self) -> &'a [NormalizationMethod] {
        self.stopping_criterias
    }
}

impl<'a> fmt::Display for ResidualsConfig<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let separation_line = String::from(
            "+-------------------+--------------------------+--------------------------+\n",
        );

        let mut content = String::from("Residuals configuration\n");
        content.push_str("=======================\n\n");
        content.push_str(&separation_line);
        content.push_str("| ");
        content.push_str(&format!("{:width$}", "Residual number", width = 18));
        content.push_str("| ");
        content.push_str(&format!("{:width$}", "Stopping criteria", width = 25));
        content.push_str("| ");
        content.push_str(&format!("{:width$}", "Update method", width = 25));
        content.push_str("|\n");

        content.push_str(&separation_line);

        for i in 0..self.len() {
            content.push_str(&format!("| {:width$}", &i.to_string(), width = 18));
            content.push_str("| ");
            content.push_str(&format!(
                "{:width$}",
                self.stopping_criterias[i].to_string(),
                width = 25
            ));
            content.push_str("| ");
            content.push_str(&format!(
                "{:width$}|",
                self.update_methods[i].to_string(),
                width = 25
            ));
            content.push('\n');
        }
        content.push_str(&separation_line);
        content.push('\n');
        write!(f, "{}", content)
    }
}
