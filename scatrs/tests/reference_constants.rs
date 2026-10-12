mod common;

use common::{find, load};

#[test]
fn std_f64_constants_equal_reference_hi_words() {
    use std::f64::consts as c;
    let recs = load("constants.jsonl");
    let pairs = [
        ("PI", c::PI),
        ("TAU", c::TAU),
        ("FRAC_PI_2", c::FRAC_PI_2),
        ("FRAC_PI_3", c::FRAC_PI_3),
        ("FRAC_PI_4", c::FRAC_PI_4),
        ("FRAC_PI_6", c::FRAC_PI_6),
        ("FRAC_PI_8", c::FRAC_PI_8),
        ("FRAC_1_PI", c::FRAC_1_PI),
        ("FRAC_2_PI", c::FRAC_2_PI),
        ("FRAC_2_SQRT_PI", c::FRAC_2_SQRT_PI),
        ("SQRT_2", c::SQRT_2),
        ("FRAC_1_SQRT_2", c::FRAC_1_SQRT_2),
        ("E", c::E),
        ("LN_2", c::LN_2),
        ("LN_10", c::LN_10),
        ("LOG2_E", c::LOG2_E),
        ("LOG10_E", c::LOG10_E),
        ("LOG10_2", c::LOG10_2),
        ("LOG2_10", c::LOG2_10),
    ];
    for (name, value) in pairs {
        let r = find(&recs, name);
        assert_eq!(r.label, "certified", "{name}");
        assert_eq!(r.output("value").hi[0], value, "{name}");
    }
}

#[test]
fn every_constant_is_known_to_double_double_accuracy() {
    for r in load("constants.jsonl") {
        let v = r.output("value");
        // |value - (hi + lo)| <= |lo2| + err, since err bounds |value - (hi + lo + lo2)|.
        let dd_err = v.lo2[0].abs() + v.err[0];
        assert!(
            dd_err <= 2f64.powi(-105) * v.hi[0].abs(),
            "{}: double-double error {:e} too large",
            r.case,
            dd_err
        );
    }
}
