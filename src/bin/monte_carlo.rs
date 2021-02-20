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

fn monte_carlo_test() {
    let mut s = 0.0;
    let n = 10000000;
    let mut samples: Vec<f32> = Vec::new();
    println!(
        "running experiment with {} samples, normal monte carlo estimator",
        n
    );
    for _ in 0..n {
        // get technique parameterization
        let m: f32 = 2.0;
        // compute density transformation from using tangent function. (simply the derivative for this space, however in general a jacobian would be used for joint pdfs)

        // sample according to gaussian(x, 1.0/(m*sqrt(tau)), 0.0, m, m)
        let sigma = m.abs() as f64;
        let y_integral = random::<f64>();
        let x = sigma * std::f64::consts::SQRT_2 * erf::erf_inv(2.0 * y_integral - 1.0);
        // phi = 0.5 + 0.5 * erf(x/sqrt2rho)
        // 2.0 * phi - 1 = erf(x / sqrt2rho)
        // x = sqrt2rho * erfinv(2.0 * phi - 1)
        let y = gaussian(
            x as f64,
            (sigma * (std::f64::consts::TAU).sqrt()).recip(),
            0.0,
            sigma,
            sigma,
        ) as f32;
        let pdf = y;
        debug_assert!(!y.is_nan());
        let f = f(x as f32);
        debug_assert!(!f.is_nan());

        // computed fs, pdfs, and ts for samples 0 through 4.
        // now compute SMIS balance heuristic for 4 simultaneous samples.

        let i_j = f / pdf;
        samples.push(i_j);
        s += i_j / n as f32;
        debug_assert!(!s.is_nan());
    }
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    let variance: f32 =
        samples.iter().map(|e| (*e - mean).powi(2)).sum::<f32>() / samples.len() as f32;

    println!("estimate with {} samples is {:?}", n, mean);
    println!(
        "variance is {}, std_deviation is {}",
        variance,
        variance.sqrt()
    );
}

fn smis_test() {
    let mut ts: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
    let mut pdfs: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
    let mut fs: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];
    let mut xs: [f32; NUM_SAMPLES] = [0.0; NUM_SAMPLES];

    let mut s = 0.0;
    let n = 1000;

    let mut samples: Vec<f32> = Vec::new();
    let mut samples2: Vec<f32> = Vec::new();
    println!(
        "running experiment with {} trials and {} samples per trial, SMIS monte carlo estimator",
        n, NUM_SAMPLES
    );
    for _i in 0..n {
        // NUM_SAMPLES technique + pdf samples per trial
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
        let mut sum = 0.0;
        for j in 0..NUM_SAMPLES {
            let w = pdfs[j] / sum_ps[j];
            let i_j = w * fs[j] / pdfs[j];
            samples2.push(i_j);
            sum += i_j;
            debug_assert!(!sum.is_nan());
        }
        samples.push(sum);
        s += sum / n as f32;
    }
    println!("SMIS estimate is {:?}", s);

    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    let variance: f32 =
        samples.iter().map(|e| (*e - mean).powi(2)).sum::<f32>() / samples.len() as f32;

    println!(
        "estimate with {} collective (weighted) samples is {:?}",
        n, mean
    );
    println!(
        "variance is {}, std_deviation is {}",
        variance,
        variance.sqrt()
    );
    let mean: f32 = samples2.iter().sum::<f32>() / samples.len() as f32;
    let variance: f32 =
        samples2.iter().map(|e| (*e - mean).powi(2)).sum::<f32>() / samples2.len() as f32;

    println!(
        "estimate with {} ungrouped samples is {:?}",
        n * NUM_SAMPLES,
        mean
    );
    println!(
        "variance is {}, std_deviation is {}",
        variance,
        variance.sqrt()
    );
}

fn bsdf_integral_test() {
    // let mut s = 0.0;
}

fn main() {
    println!();
    monte_carlo_test();
    println!();
    smis_test();
    println!();
    bsdf_integral_test();
    println!();
}
