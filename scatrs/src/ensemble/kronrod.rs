//! The 15-point Gauss–Kronrod rule and its embedded 7-point Gauss rule (QUADPACK's `qk15`).

/// The abscissae of the 15-point Kronrod rule on [−1, 1] in descending order of the positive
/// half: ±`NODES[i]`, i < 7, and 0. The Gauss nodes are those with odd i, and 0.
const NODES: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_47,
    0.0,
];

/// The Kronrod weights of `NODES`.
const KRONROD: [f64; 8] = [
    0.022_935_322_010_529_225,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];

/// The Gauss weights of `NODES[1]`, `NODES[3]`, `NODES[5]` and `NODES[7]` = 0.
const GAUSS: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// The number of nodes of a panel.
pub(crate) const POINTS: usize = 15;

/// The nodes of the panel [a, b] (ascending) with their Kronrod and Gauss weights (the Gauss
/// weight is 0 at the nodes that only the Kronrod rule uses).
pub(crate) fn panel(a: f64, b: f64) -> [(f64, f64, f64); POINTS] {
    let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
    let mut out = [(0.0, 0.0, 0.0); POINTS];
    for i in 0..7 {
        let gauss = if i % 2 == 1 { h * GAUSS[i / 2] } else { 0.0 };
        let k = h * KRONROD[i];
        out[i] = (c - h * NODES[i], k, gauss);
        out[POINTS - 1 - i] = (c + h * NODES[i], k, gauss);
    }
    out[7] = (c, h * KRONROD[7], h * GAUSS[3]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rules_integrate_polynomials_exactly() {
        // Kronrod: degree 22; Gauss: degree 13.
        let p = panel(-0.3, 1.7);
        for k in 0..=22 {
            let exact = (1.7f64.powi(k + 1) - (-0.3f64).powi(k + 1)) / f64::from(k + 1);
            let kr: f64 = p.iter().map(|&(t, w, _)| w * t.powi(k)).sum();
            assert!(
                (kr - exact).abs() <= 1e-14 * exact.abs().max(1.0),
                "Kronrod, degree {k}"
            );
            if k <= 13 {
                let g: f64 = p.iter().map(|&(t, _, w)| w * t.powi(k)).sum();
                assert!(
                    (g - exact).abs() <= 1e-14 * exact.abs().max(1.0),
                    "Gauss, degree {k}"
                );
            }
        }
        assert!(p.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(p.iter().filter(|&&(_, _, g)| g != 0.0).count(), 7);
    }
}
