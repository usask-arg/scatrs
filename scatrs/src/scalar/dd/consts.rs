//! Double-double constants: the certified splits in `reference/data/constants.jsonl`.

use super::DoubleDouble as Dd;

const fn c(hi: f64, lo: f64) -> Dd {
    Dd::from_parts_unchecked(hi, lo)
}

// Leading words use std constants (identical values; avoids clippy::approx_constant).
use core::f64::consts as f;

pub(crate) const PI: Dd = c(f::PI, 1.2246467991473532e-16);
pub(crate) const TAU: Dd = c(f::TAU, 2.4492935982947064e-16);
pub(crate) const FRAC_PI_2: Dd = c(f::FRAC_PI_2, 6.123233995736766e-17);
pub(crate) const FRAC_PI_3: Dd = c(f::FRAC_PI_3, -1.072081766451091e-16);
pub(crate) const FRAC_PI_4: Dd = c(f::FRAC_PI_4, 3.061616997868383e-17);
pub(crate) const FRAC_PI_6: Dd = c(f::FRAC_PI_6, -5.360408832255455e-17);
pub(crate) const FRAC_PI_8: Dd = c(f::FRAC_PI_8, 1.5308084989341915e-17);
pub(crate) const FRAC_3PI_4: Dd = c(2.356194490192345, 9.184850993605148e-17);
pub(crate) const FRAC_1_PI: Dd = c(f::FRAC_1_PI, -1.9678676675182486e-17);
pub(crate) const FRAC_2_PI: Dd = c(f::FRAC_2_PI, -3.935735335036497e-17);
pub(crate) const FRAC_2_SQRT_PI: Dd = c(f::FRAC_2_SQRT_PI, 1.533545961316588e-17);
pub(crate) const SQRT_2: Dd = c(f::SQRT_2, -9.667293313452913e-17);
pub(crate) const FRAC_1_SQRT_2: Dd = c(f::FRAC_1_SQRT_2, -4.833646656726457e-17);
pub(crate) const E: Dd = c(f::E, 1.4456468917292502e-16);
pub(crate) const LN_2: Dd = c(f::LN_2, 2.3190468138462996e-17);
pub(crate) const LN_10: Dd = c(f::LN_10, -2.1707562233822494e-16);
pub(crate) const LOG2_E: Dd = c(f::LOG2_E, 2.0355273740931033e-17);
pub(crate) const LOG10_E: Dd = c(f::LOG10_E, 1.098319650216765e-17);
pub(crate) const LOG10_2: Dd = c(f::LOG10_2, -2.8037281277851704e-18);
pub(crate) const LOG2_10: Dd = c(f::LOG2_10, 1.661617516973592e-16);

/// Third word of π/2: π/2 − FRAC_PI_2.hi − FRAC_PI_2.lo, rounded to `f64`.
pub(crate) const FRAC_PI_2_TAIL: f64 = -1.4973849048591698e-33;
/// Third word of ln 2.
pub(crate) const LN_2_TAIL: f64 = 5.707708438416212e-34;

/// `INV_FACT[k] = 1/k!` for `k = 0..=30`.
pub(crate) const INV_FACT: [Dd; 31] = [
    c(1.0, 0.0),
    c(1.0, 0.0),
    c(0.5, 0.0),
    c(0.16666666666666666, 9.25185853854297e-18),
    c(0.041666666666666664, 2.3129646346357427e-18),
    c(0.008333333333333333, 1.1564823173178714e-19),
    c(0.001388888888888889, -5.300543954373577e-20),
    c(0.0001984126984126984, 1.7209558293420705e-22),
    c(2.48015873015873e-5, 2.1511947866775882e-23),
    c(2.7557319223985893e-6, -1.858393274046472e-22),
    c(2.755731922398589e-7, 2.3767714622250297e-23),
    c(2.505210838544172e-8, -1.448814070935912e-24),
    c(2.08767569878681e-9, -1.20734505911326e-25),
    c(1.6059043836821613e-10, 1.2585294588752098e-26),
    c(1.1470745597729725e-11, 2.0655512752830745e-28),
    c(7.647163731819816e-13, 7.03872877733453e-30),
    c(4.779477332387385e-14, 4.399205485834081e-31),
    c(2.8114572543455206e-15, 1.6508842730861433e-31),
    c(1.5619206968586225e-16, 1.1910679660273754e-32),
    c(8.22063524662433e-18, 2.2141894119604265e-34),
    c(4.110317623312165e-19, 1.4412973378659527e-36),
    c(1.9572941063391263e-20, -1.3643503830087908e-36),
    c(8.896791392450574e-22, -7.911402614872376e-38),
    c(3.868170170630684e-23, -8.843177655482344e-40),
    c(1.6117375710961184e-24, -3.6846573564509766e-41),
    c(6.446950284384474e-26, -1.9330404233703465e-42),
    c(2.4795962632247976e-27, -1.2953730964765229e-43),
    c(9.183689863795546e-29, 1.4303150396787322e-45),
    c(3.279889237069838e-30, 1.5117542744029879e-46),
    c(1.1309962886447716e-31, 1.0498015412959506e-47),
    c(3.7699876288159054e-33, 2.5870347832750324e-49),
];

/// `POW10_2I[i] = 10^(2^i)` for `i = 0..=8`.
pub(crate) const POW10_2I: [Dd; 9] = [
    c(10.0, 0.0),
    c(100.0, 0.0),
    c(10000.0, 0.0),
    c(100000000.0, 0.0),
    c(1e16, 0.0),
    c(1e32, -5366162204393472.0),
    c(1e64, -2.1320419009454396e47),
    c(1e128, -7.51744869165182e111),
    c(1e256, -3.012765990014054e239),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> Vec<serde_json::Value> {
        include_str!("../../../../reference/data/constants.jsonl")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid record"))
            .collect()
    }

    fn hi_lo(records: &[serde_json::Value], case: &str) -> (f64, f64) {
        let r = records
            .iter()
            .find(|r| r["case"] == case)
            .unwrap_or_else(|| panic!("missing {case}"));
        let v = &r["outputs"]["value"];
        (v["hi"][0].as_f64().unwrap(), v["lo"][0].as_f64().unwrap())
    }

    #[test]
    fn constants_match_reference() {
        let recs = reference();
        let table = [
            ("PI", PI),
            ("TAU", TAU),
            ("FRAC_PI_2", FRAC_PI_2),
            ("FRAC_PI_3", FRAC_PI_3),
            ("FRAC_PI_4", FRAC_PI_4),
            ("FRAC_PI_6", FRAC_PI_6),
            ("FRAC_PI_8", FRAC_PI_8),
            ("FRAC_3PI_4", FRAC_3PI_4),
            ("FRAC_1_PI", FRAC_1_PI),
            ("FRAC_2_PI", FRAC_2_PI),
            ("FRAC_2_SQRT_PI", FRAC_2_SQRT_PI),
            ("SQRT_2", SQRT_2),
            ("FRAC_1_SQRT_2", FRAC_1_SQRT_2),
            ("E", E),
            ("LN_2", LN_2),
            ("LN_10", LN_10),
            ("LOG2_E", LOG2_E),
            ("LOG10_E", LOG10_E),
            ("LOG10_2", LOG10_2),
            ("LOG2_10", LOG2_10),
        ];
        for (name, value) in table {
            assert_eq!((value.hi(), value.lo()), hi_lo(&recs, name), "{name}");
        }
        assert_eq!(FRAC_PI_2_TAIL, hi_lo(&recs, "FRAC_PI_2_TAIL").0);
        assert_eq!(LN_2_TAIL, hi_lo(&recs, "LN_2_TAIL").0);
        for (k, value) in INV_FACT.iter().enumerate() {
            let expected = hi_lo(&recs, &format!("INV_FACT_{k}"));
            assert_eq!((value.hi(), value.lo()), expected, "1/{k}!");
        }
        for (i, value) in POW10_2I.iter().enumerate() {
            let expected = hi_lo(&recs, &format!("POW10_2POW{i}"));
            assert_eq!((value.hi(), value.lo()), expected, "10^(2^{i})");
        }
    }
}
