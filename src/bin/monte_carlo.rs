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
        + gaussian(x as f64, 2.1 * a2.recip(), 0.0, 0.3, 0.3)) as f32
}

const NUM_SAMPLES: usize = 8;

fn smis_test() {
    let mut ts: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
    let mut pdfs: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
    let mut fs: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
    let mut xs: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];

    let c = std::f32::consts::PI / 2.0 - 0.2;
    let mut s = 0.0;
    let n = 1000;
    for i in 0..n {
        // 4 technique + pdf samples per trial
        for j in 0..NUM_SAMPLES {
            // get technique parameterization
            let t = 2.0 * random::<f32>() + 0.1;
            let m = t;
            // compute density transformation from using tangent function. (simply the derivative for this space, however in general a jacobian would be used for joint pdfs)
            ts[j] = t;
            // sample according to gaussian(x, 1.0/(m*sqrt(tau)), 0.0, m, m)
            let sigma = m.abs() as f64;
            let y_integral = random::<f64>();
            let x = sigma * std::f64::consts::SQRT_2 * erf::erf_inv(2.0 * y_integral - 1.0);
            // phi = 0.5 + 0.5 * erf(x/sqrt2rho)
            // 2.0 * phi - 1 = erf(x / sqrt2rho)
            // x = sqrt2rho * erfinv(2.0 * phi - 1)
            xs[j] = x as f32;
            let y = gaussian(
                x as f64,
                (sigma * (std::f64::consts::TAU).sqrt()).recip(),
                0.0,
                sigma,
                sigma,
            ) as f32;
            pdfs[j] = y;
            debug_assert!(!y.is_nan());
            fs[j] = f(x as f32);
            debug_assert!(!fs[j].is_nan());
        }
        // computed fs, pdfs, and ts for samples 0 through 4.
        // now compute SMIS balance heuristic for 4 simultaneous samples.

        let mut sum_ps: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
        for j in 0..NUM_SAMPLES {
            let x = xs[j];
            for t_j in 0..NUM_SAMPLES {
                let sigma = ts[t_j] as f64;
                let pj = gaussian(
                    x as f64,
                    (sigma * (std::f64::consts::TAU).sqrt()).recip(),
                    0.0,
                    sigma,
                    sigma,
                ) as f32;
                sum_ps[j] += pj;
            }
        }
        for j in 0..NUM_SAMPLES {
            let w = pdfs[j] / sum_ps[j];
            let i_j = w * fs[j] / pdfs[j];
            s += i_j / n as f32;
            debug_assert!(!s.is_nan());
        }
    }
    println!("averaged_estimate is {:?}", s);
}

fn bsdf_integral_test() {
    let mut s = 0.0;
}

fn main() {
    smis_test();
    bsdf_integral_test();
}
