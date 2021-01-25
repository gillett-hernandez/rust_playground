#![feature(slice_fill)]
#[macro_use]
extern crate packed_simd;

mod film;
pub mod parse;
pub mod trace;

pub use film::Film;
pub use parse::*;
pub use trace::*;

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

pub fn attempt_write(film: &mut Film<u32>, px: usize, py: usize, c: u32) {
    if py * film.width + px >= film.buffer.len() {
        return;
    }
    film.buffer[py * film.width + px] = c;
}

pub fn hsv_to_rgb(h: usize, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h as f32 / 60.0) % 2.0 - 1.0)).abs();
    let m = v - c;
    let convert = |(r, g, b), m| {
        (
            ((r + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((b + m) * 255.0) as u8,
        )
    };
    match h {
        0..=60 => convert((c, x, 0.0), m),
        61..=120 => convert((x, c, 0.0), m),
        121..=180 => convert((0.0, c, x), m),
        181..=240 => convert((0.0, x, c), m),
        241..=300 => convert((x, 0.0, c), m),
        301..=360 => convert((c, 0.0, x), m),
        _ => (0, 0, 0),
    }
}

pub fn triple_to_u32(triple: (u8, u8, u8)) -> u32 {
    let c = ((triple.0 as u32) << 16) + ((triple.1 as u32) << 8) + (triple.2 as u32);
    c
}

pub fn blit_circle(film: &mut Film<u32>, radius: f32, x: usize, y: usize, c: u32) {
    let approx_pixel_circumference = radius as f32 * std::f32::consts::TAU;

    let PIXEL_X_SIZE = 1.0 / film.width as f32;
    let PIXEL_Y_SIZE = 1.0 / film.height as f32;
    for phi in 0..(approx_pixel_circumference as usize) {
        let (new_px, new_py) = (
            (x as f32 * PIXEL_X_SIZE
                + radius as f32
                    * PIXEL_X_SIZE
                    * (phi as f32 * std::f32::consts::TAU / approx_pixel_circumference).cos())
                / PIXEL_X_SIZE,
            (y as f32 * PIXEL_Y_SIZE
                + radius as f32
                    * PIXEL_Y_SIZE
                    * (phi as f32 * std::f32::consts::TAU / approx_pixel_circumference).sin())
                / PIXEL_Y_SIZE,
        );
        attempt_write(film, new_px as usize, new_py as usize, c);
    }
}


#[derive(Clone)]
pub struct Texture4 {
    pub curves: [CDF; 4],
    pub texture: Film<f32x4>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture4 {
    // evaluate the 4 CDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factors = self.texture.at_uv(uv);
        let eval = f32x4::new(
            self.curves[0].evaluate_power(lambda),
            self.curves[1].evaluate_power(lambda),
            self.curves[2].evaluate_power(lambda),
            self.curves[3].evaluate_power(lambda),
        );
        (factors * eval).sum()
    }
}
#[derive(Clone)]
pub struct Texture1 {
    pub curve: CDF,
    pub texture: Film<f32>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture1 {
    // evaluate the 4 CDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factor = self.texture.at_uv(uv);
        let eval = self.curve.evaluate_power(lambda);
        factor * eval
    }
}
#[derive(Clone)]
pub enum Texture {
    Texture1(Texture1),
    Texture4(Texture4),
}

impl Texture {
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        match self {
            Texture::Texture1(tex) => tex.eval_at(lambda, uv),
            Texture::Texture4(tex) => tex.eval_at(lambda, uv),
        }
    }
}

#[derive(Clone)]
pub struct TexStack {
    pub textures: Vec<Texture>,
}
impl TexStack {
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        let mut s = 0.0;
        for tex in self.textures.iter() {
            s += tex.eval_at(lambda, uv);
        }
        s
    }
}
