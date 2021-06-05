use std::f64::{
    consts::{E, FRAC_1_PI},
    INFINITY,
};
fn main() {
    // basic idea
    // find pairs of a, b such that log(a)/e = log((c+e^(log(b)/pi))) is minimized

    let (mut a, mut b) = (2, 1);
    let mut min = INFINITY;
    let mut min_pair;
    loop {
        let ap = (a as f64).ln() / E;
        let bp = (b as f64).ln() * FRAC_1_PI;

        if (ap - bp).abs() < min.abs() {
            min = ap - bp;
            min_pair = (a, b);
            println!(
                "{:?},{}, {}",
                min_pair,
                min_pair.0 as f32 / min_pair.1 as f32,
                min
            );
        }
        if ap > bp {
            b += 1;
        } else {
            a += 1;
        }
        if a + b > 10000 {
            break;
        }
    }

    let (mut a, mut b) = (2, 1);
    let mut min = INFINITY;
    let mut min_pair;
    loop {
        // E-a/b > 0
        // E > a/b
        // E * b > a
        // a should increase

        // E-a/b < 0
        // E < a/b
        // E * b < a
        // b should increase
        let d = E - (a as f64) / (b as f64);
        if d.abs() < min.abs() {
            min = d;
            min_pair = (a, b);
            println!("{:?}, {}", min_pair, 1.0 / min.abs());
        }
        if d > 0.0 {
            a += 1;
        } else {
            b += 1;
        }
    }
}
