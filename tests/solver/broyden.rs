use crate::common::broyden1965::*;
extern crate newton_rootfinder as nrf;

extern crate nalgebra;
use nrf::model::Model;


#[test]
fn broyden_case5() {
    let problem_size = 5;
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_broyden1965_case5());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case5);

    rf.solve(&mut user_model);

    let mut solution = nalgebra::DVector::zeros(problem_size);
    solution[0] = -0.8222168603829406;
    solution[1] = -0.7671273188457487;
    solution[2] = -0.6801151944285924;
    solution[3] = -0.5433006255196431;
    solution[4] = -0.32970971675722843;

    assert_eq!(user_model.get_iteratives(), solution);

}

#[test]
fn broyden_case6() {
    let problem_size = 5;
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_broyden1965_case6());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case6);

    rf.solve(&mut user_model);

    let mut solution = nalgebra::DVector::zeros(problem_size);
    solution[0] = -0.6955898448220836;
    solution[1] = -0.664346075288027;
    solution[2] = -0.6068580398696891;
    solution[3] = -0.5023562299431517;
    solution[4] = -0.31662479035539975;

    assert_eq!(user_model.get_iteratives(), solution);


}

#[test]
fn broyden_case7() {
    let problem_size = 10;
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_broyden1965_case7());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case7);

    rf.solve(&mut user_model);

    let mut solution = nalgebra::DVector::zeros(problem_size);
    solution[0] = -0.7304305192613078;
    solution[1] = -0.7290279647590477;
    solution[2] = -0.7264123904887514;
    solution[3] = -0.7215373259970228;
    solution[4] = -0.7124600171972677;
    solution[5] = -0.6955898448220841;
    solution[6] = -0.6643460752880278;
    solution[7] = -0.60685803986969;
    solution[8] = -0.5023562299431517;
    solution[9] = -0.31662479035539975;

    assert_eq!(user_model.get_iteratives(), solution);


}

#[test]
fn broyden_case8() {
    let problem_size = 20;
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_broyden1965_case8());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case8);

    rf.solve(&mut user_model);

    let mut solution = nalgebra::DVector::zeros(problem_size);
    solution[0] = -0.7320476414628936;
    solution[1] = -0.7320448995371868;
    solution[2] = -0.7320397830403825;
    solution[3] = -0.7320302355490265;
    solution[4] = -0.7320124197630306;
    solution[5] = -0.7319791753163778;
    solution[6] = -0.7319171412487779;
    solution[7] = -0.7318013872866127;
    solution[8] = -0.7315853985385721;
    solution[9] = -0.7311823966465685;
    solution[10] = -0.7304305192613078;
    solution[11] = -0.7290279647590477;
    solution[12] = -0.7264123904887516;
    solution[13] = -0.721537325997023;
    solution[14] = -0.7124600171972679;
    solution[15] = -0.6955898448220843;
    solution[16] = -0.6643460752880272;
    solution[17] = -0.6068580398696893;
    solution[18] = -0.5023562299431515;
    solution[19] = -0.31662479035539975;

    assert_eq!(user_model.get_iteratives(), solution);


}

#[test]
fn broyden_case9() {
    let problem_size = 2;
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_broyden1965_case9());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case9);

    rf.solve(&mut user_model);

    let mut solution = nalgebra::DVector::zeros(problem_size);
    solution[0] = 1.0;
    solution[1] = 1.0;

    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }


}

#[test]
fn broyden_case10() {
    let problem_size = 2;
    let rf = nrf::solver::RootFinderFD::default_with_guess(init_broyden1965_case10());
    let mut user_model =
        nrf::model_with_func::UserModelWithFunc::new(problem_size, broyden1965_case10);

    rf.solve(&mut user_model);

    let mut solution = nalgebra::DVector::zeros(problem_size);
    solution[0] = 5.0;
    solution[1] = 4.0;


    for i in 0..problem_size {
        assert!(float_cmp::approx_eq!(
            f64,
            user_model.get_iteratives()[i],
            solution[i],
            epsilon = 1e-6
        ));
    }


}
