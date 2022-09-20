use std::{f64::consts::PI, time::Instant};

use lib::{Sampler, StratifiedSampler};
// use lib::PI;
use rand::prelude::*;

fn run_simulation<F>(iterations: usize, mut sampler: F) -> (f64, f64)
where
    F: FnMut() -> (f64, f64),
{
    let mut sum = 0.0;
    let mut sos = 0.0;
    for _ in 0..iterations {
        let point0 = sampler();
        let point1 = sampler();

        let dist = (point0.0 - point1.0).hypot(point0.1 - point1.1);
        sum += dist;
        sos += dist * dist;
    }
    (sum / iterations as f64, sos / iterations as f64)
}

fn main() {
    // let rejection_sampler = || loop {
    //     let x = 2.0 * random::<f64>() - 1.0;
    //     let y = 2.0 * random::<f64>() - 1.0;
    //     if x.hypot(y) < 1.0 {
    //         return (x, y);
    //     }
    // };
    // let now = Instant::now();
    // let (estimate, sos) = run_simulation(10000000, rejection_sampler);
    // println!(
    //     "{}, sos {}, variance {}",
    //     estimate,
    //     sos,
    //     sos - estimate * estimate
    // );

    // println!("{}", now.elapsed().as_millis() as f64 / 1000.0);
    // let now = Instant::now();
    // let disk_sampler = || {
    //     let r = random::<f64>().sqrt();
    //     let theta = random::<f64>() * 2.0 * PI;
    //     let (sin, cos) = theta.sin_cos();
    //     (cos * r, sin * r)
    // };

    // let (estimate, sos) = run_simulation(10000000, disk_sampler);
    // println!(
    //     "{}, sos {}, variance {}",
    //     estimate,
    //     sos,
    //     sos - estimate * estimate
    // );
    // println!("{}", now.elapsed().as_millis() as f64 / 1000.0);

    let square_sampler = || (random::<f64>(), random::<f64>());
    let (estimate, sos) = run_simulation(1_000_000, square_sampler);
    println!(
        "{}, sos {}, variance {}",
        estimate,
        sos,
        sos - estimate * estimate
    );

    let mut sampler = StratifiedSampler::new(10, 10, 2);
    let mut stratified_sampler = || {
        let sample = sampler.draw_2d();
        (sample.x as f64, sample.y as f64)
    };
    let (estimate, sos) = run_simulation(1_000_000, stratified_sampler);
    println!(
        "{}, sos {}, variance {}",
        estimate,
        sos,
        sos - estimate * estimate
    );
}
