extern crate nalgebra;

use crate::util::residuals;

pub trait Model {
    fn init(&self) {
        //default empty method
    }
    fn evaluate(&mut self);
    fn len_problem(&self) -> usize;
    fn len_memory(&self) -> usize {
        0 // 0 by default if not using any memory features
    }

    fn len_outputs(&self) -> usize {
        0 // 0 by default if not having any other outputs then the residuals
    }

    fn set_memory(&mut self, #[allow(unused_variables)] memory: &nalgebra::DVector<f64>) {
        // default empty method
    }
    fn get_memory(&self) -> nalgebra::DVector<f64> {
        // default empty DVector
        nalgebra::DVector::from_vec(vec![])
    }
    fn set_iteratives(&mut self, iteratives: &nalgebra::DVector<f64>);
    fn get_iteratives(&self) -> nalgebra::DVector<f64>;
    fn get_residuals(&self) -> residuals::ResidualsValues;
}

// Other possible functions :
// If enforced in trait, return type Option would be required
//
// fn len_parameters(&self) -> usize
// fn set_parameters(&mut self, params: &nalgebra::DVector::<f64>)
// fn get_parameters(&self) -> nalgebra::DVector::<f64>
//
// fn len_outputs(&self) -> usize
// fn get_outputs(&self) -> nalgebra::DVector::<f64>
