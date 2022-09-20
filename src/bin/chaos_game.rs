use std::f32::consts::TAU;

use lib::{flatland::Vec2, rgb_to_u32, window_loop, PI};
use math::random;
use minifb::WindowOptions;
use rayon::prelude::*;

fn scale_factor(n: usize) -> f32 {
    let mut sum = 0.0f32;
    for k in 1..=(n / 4) {
        sum += (2.0 * PI * k as f32 / n as f32).cos();
    }
    (2.0 * (1.0 + sum)).recip()
}

fn main() {
    let multiplicity = 10;
    let threads = 24;
    let n = 6;

    let mut cursors = Vec::new();
    for _ in 0..(threads * multiplicity) {
        cursors.push(Vec2::new(random(), random()));
    }
    window_loop(
        1024,
        1024,
        144,
        WindowOptions::default(),
        |_, buffer, width, height| {
            cursors.par_iter_mut().for_each(|cursor| {
                // random process goes here
                let sample = random();

                let center = Vec2::new(0.5, 0.5);

                let angle = (n as f32 * sample).floor() / n as f32 * TAU;
                let (sin, cos) = angle.sin_cos();
                let target = center + Vec2::new(cos, sin) / 2.0;

                *cursor = *cursor + (target - *cursor) * (1.0 - scale_factor(n));
            });

            for v in &cursors {
                let (px, py) = (v.x() * width as f32, v.y() * height as f32);
                buffer[(py as usize) * width + (px as usize)] = rgb_to_u32(255, 255, 255);
            }
        },
        false,
    )
}
