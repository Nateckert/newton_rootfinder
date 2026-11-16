//! What a configuration was given is read back from it:
//! a user does not keep a copy beside the solver.

use newton_rootfinder as nrf;
use nrf::iteratives;
use nrf::residuals::{NormalizationMethod, ResidualsConfig};

#[test]
fn a_residuals_configuration_gives_back_its_methods() {
    let stopping_criterias = vec![NormalizationMethod::Abs, NormalizationMethod::Rel];
    let update_methods = vec![NormalizationMethod::Adapt, NormalizationMethod::Abs];
    let config = ResidualsConfig::new(&stopping_criterias, &update_methods);

    assert_eq!(config.len(), 2);
    assert_eq!(config.get_stopping_criterias(), stopping_criterias);
    assert_eq!(config.get_update_methods(), update_methods);
}

#[test]
fn the_iteratives_give_back_their_number() {
    let parameters = iteratives::default_vec_iteratives(3);
    let iteratives = iteratives::Iteratives::new(&parameters);

    assert_eq!(iteratives.len(), 3);
}
