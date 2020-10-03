#![feature(slice_fill)]
#[macro_use]
extern crate packed_simd;

mod film;

pub use film::Film;

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use nalgebra::{Matrix3, Vector3};
use packed_simd::f32x4;
use rand::prelude::*;

use rand::seq::SliceRandom;
use rand::{thread_rng, RngCore};

pub fn gaussian(x: f64, alpha: f64, mu: f64, sigma1: f64, sigma2: f64) -> f64 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn x_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.056, 5998.0, 379.0, 310.0)
        + gaussian(angstroms.into(), 0.362, 4420.0, 160.0, 267.0)
        + gaussian(angstroms.into(), -0.065, 5011.0, 204.0, 262.0)) as f32
}

pub fn y_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 0.821, 5688.0, 469.0, 405.0)
        + gaussian(angstroms.into(), 0.286, 5309.0, 163.0, 311.0)) as f32
}

pub fn z_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.217, 4370.0, 118.0, 360.0)
        + gaussian(angstroms.into(), 0.681, 4590.0, 260.0, 138.0)) as f32
}

pub fn sw_to_triplet(sw: (f32, f32)) -> f32x4 {
    let ang = sw.0 * 10.0;
    f32x4::new(sw.1 * x_bar(ang), sw.1 * y_bar(ang), sw.1 * z_bar(ang), 0.0)
}

pub fn map(film: &Film<f32x4>, max_luminance: f32, pixel: (usize, usize)) -> (f32x4, f32x4) {
    let cie_xyz_color = film.at(pixel.0, pixel.1);
    let mut scaled_cie_xyz_color = cie_xyz_color / max_luminance;
    if !scaled_cie_xyz_color.is_finite().all() {
        scaled_cie_xyz_color = f32x4::splat(0.0);
    }

    let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
        3.24096994,
        -1.53738318,
        -0.49861076,
        -0.96924364,
        1.8759675,
        0.04155506,
        0.05563008,
        -0.20397696,
        1.05697151,
    );
    let [x, y, z, _]: [f32; 4] = scaled_cie_xyz_color.into();
    let intermediate = xyz_to_rgb * Vector3::new(x, y, z);

    let rgb_linear = f32x4::new(intermediate[0], intermediate[1], intermediate[2], 0.0);
    const S313: f32x4 = f32x4::splat(0.0031308);
    const S323_25: f32x4 = f32x4::splat(323.0 / 25.0);
    const S5_12: f32x4 = f32x4::splat(5.0 / 12.0);
    const S211: f32x4 = f32x4::splat(211.0);
    const S11: f32x4 = f32x4::splat(11.0);
    const S200: f32x4 = f32x4::splat(200.0);
    let srgb = (rgb_linear.lt(S313)).select(
        S323_25 * rgb_linear,
        (S211 * rgb_linear.powf(S5_12) - S11) / S200,
    );
    (srgb, rgb_linear)
}
