use std::fmt;

/// Iterative definition
///
/// One might want to limit the update steps, by either:
/// - limiting the range of values to avoid non-sense values
/// - limiting the size of an update step
///
/// Two implementations of this trait are provided:
/// - `IterativeParams`
/// - `IterativeParamsFD`
pub trait Iterative {
    /// Compute the new value based on the current value and the step size proposed
    ///
    /// The iteratives variables implement a way to reduce this step according to the parametrization
    fn step_limitation(&self, value_current: f64, raw_step: f64) -> f64;
    /// Compute the perturbation (only valid if it is working with finite differences)
    ///
    /// according to the parametrization
    fn compute_perturbation(&self, #[allow(unused_variables)] x: f64) -> f64 {
        unimplemented!();
    }
    /// Method to differente without panicking if it is working with finite differences
    fn with_finite_diff(&self) -> bool {
        false
    }
}

/// A slice of iteratives
///
/// This struct is used as a wrapper to act on a slice of several iteratives
///
/// It provides the same method as the `Iterative` trait with the plural suffix:
/// - `step_limitations`
/// - `compute_perturbations`
pub struct Iteratives<'a, T: Iterative> {
    iteratives_params: &'a [T],
}

impl<'a, T> Iteratives<'a, T>
where
    T: Iterative,
{
    pub fn new(iteratives_params: &'a [T]) -> Self {
        Iteratives { iteratives_params }
    }

    pub fn len(&self) -> usize {
        self.iteratives_params.len()
    }
    /// Compute a limited step for several iteratives
    ///
    /// Return the new value after the application of the step limitation (and not the step).
    ///
    /// This is required as it can be limited by an interval for the iteratives.
    pub fn step_limitations<D>(
        &self,
        values: &nalgebra::OVector<f64, D>,
        raw_step: &nalgebra::OVector<f64, D>,
    ) -> nalgebra::OVector<f64, D>
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let mut step_lim: nalgebra::OVector<f64, D> = super::super::ovector_zeros_like(values);

        for (i, iterative_params) in (self.iteratives_params).iter().enumerate() {
            step_lim[i] = iterative_params.step_limitation(values[i], raw_step[i]);
        }
        step_lim
    }

    /// Compute the perturbation for several iteratives
    pub fn compute_perturbations<D>(
        &self,
        iterative_values: &nalgebra::OVector<f64, D>,
    ) -> nalgebra::OVector<f64, D>
    where
        D: nalgebra::Dim,
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<D>,
    {
        let mut perturbations: nalgebra::OVector<f64, D> =
            super::super::ovector_zeros_like(iterative_values);

        for (i, iterative_var) in (self.iteratives_params).iter().enumerate() {
            perturbations[i] = iterative_var.compute_perturbation(iterative_values[i]);
        }
        perturbations
    }
}

impl<'a, T> fmt::Display for Iteratives<'a, T>
where
    T: Iterative + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let finite_diff = self.iteratives_params[0].with_finite_diff();

        let mut content = String::from("Iteratives parameters\n");
        content.push_str("=====================\n\n");

        let column_float = String::from("--------------+");

        let separation_line = if finite_diff {
            "+-----------+".to_owned()
                + &column_float
                + &column_float
                + &column_float
                + &column_float
                + "-----------------+"
                + &column_float
                + &column_float
                + "\n"
        } else {
            "+-----------+".to_owned()
                + &column_float
                + &column_float
                + &column_float
                + &column_float
                + "\n"
        };

        content.push_str(&separation_line);
        let width = column_float.len() - 2;
        content.push_str("| Iterative ");
        content.push_str(&format!("| {:width$}", &"max_step_abs", width = width));
        content.push_str(&format!("| {:width$}", &"max_step_rel", width = width));
        content.push_str(&format!("| {:width$}", &"min_value", width = width));
        content.push_str(&format!("| {:width$}", &"max_value", width = width));

        if finite_diff {
            content.push_str(&format!(
                "| {:width$}",
                &"perturbation",
                width = "-----------------+".len() - 2
            ));
            content.push_str(&format!("| {:width$}", &"dx_abs", width = width));
            content.push_str(&format!("| {:width$}|", &"dx_rel", width = width));
        } else {
            content.push('|');
        }

        content.push('\n');
        content.push_str(&separation_line);

        for (i, elt) in self.iteratives_params.iter().enumerate() {
            content.push_str(&format!("| {:width$}|", &i.to_string(), width = 10));
            content.push_str(&elt.to_string());
            content.push('\n');
            content.push_str(&separation_line);
        }

        content.push('\n');
        write!(f, "{}", content)
    }
}

impl<'a, T> fmt::Debug for Iteratives<'a, T>
where
    T: Iterative + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.iteratives_params.iter())
            .finish()
    }
}
