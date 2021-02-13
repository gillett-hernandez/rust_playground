use lib::*;
use rand::prelude::*;
use statrs::function::erf;

fn f(x: f32) -> f32 {
    let sqrt_tau = (std::f64::consts::TAU).sqrt();
    let a0 = 2.0 * sqrt_tau;
    let a1 = 1.0 * sqrt_tau;
    let a2 = 0.3 * sqrt_tau;
    (gaussian(x as f64, a0.recip(), 0.0, 2.0, 2.0)
        + gaussian(x as f64, a1.recip(), 0.0, 1.0, 1.0)
        + gaussian(x as f64, a2.recip(), 0.0, 0.3, 0.3)) as f32
}

fn main() {
    // 1000 trials
    let mut ts: [f32; 4] = [0.0; 4];
    let mut pdfs: [f32; 4] = [0.0; 4];
    let mut fs: [f32; 4] = [0.0; 4];

    let c = std::f32::consts::PI / 2.0 - 0.2;
    let mut s = 0.0;
    let n = 1000000;
    for i in 0..n {
        // 4 technique + pdf samples per trial
        for j in 0..4 {
            // get technique parameterization
            let t = 2.0 * random::<f32>() + 0.1;
            let m = t;
            // compute density transformation from using tangent function. (simply the derivative for this space, however in general a jacobian would be used for joint pdfs)
            ts[j] = t;
            // sample according to gaussian(x, m, 0.0, 1.0/m)
            let rho = m.abs() as f64;
            let y_integral = random::<f64>();
            let x = rho * std::f64::consts::SQRT_2 * erf::erf_inv(2.0 * y_integral - 1.0);
            // phi = 0.5 + 0.5 * erf(x/sqrt2rho)
            // 2.0 * phi - 1 = erf(x / sqrt2rho)
            // x = sqrt2rho * erfinv(2.0 * phi - 1)

            let y = gaussian(
                x as f64,
                (rho * (std::f64::consts::TAU).sqrt()).recip(),
                0.0,
                rho,
                rho,
            ) as f32;
            pdfs[j] = y;
            debug_assert!(!y.is_nan());
            fs[j] = f(x as f32);
            debug_assert!(!fs[j].is_nan());
        }
        // computed fs, pdfs, and ts for samples 0 through 4.
        // now compute SMIS balance heuristic for 4 simultaneous samples.

        // TODO: add SMIS balance heuristic
        for j in 0..4 {
            let i_j = fs[j] / pdfs[j];
            s += i_j / 4.0 / n as f32;
            debug_assert!(!s.is_nan());
        }
    }
    println!("averaged_estimate is {:?}", s);
}
