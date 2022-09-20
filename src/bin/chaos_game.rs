use std::f32::consts::TAU;

use lib::{flatland::Vec2, rgb_to_u32, window_loop, Film, PI};
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

enum Filter {
    Box,
    Triangle,
    Gaussian(f32), // alpha
    // Mitchell
    WindowedSinc(f32), // tau
}

impl Filter {
    pub fn eval(&self, input: Vec2, radius: Vec2) -> f32 {
        match self {
            Filter::Box => {
                1.0 * if input.x().abs() < radius.x() {
                    1.0
                } else {
                    0.0
                } * if input.y().abs() < radius.y() {
                    1.0
                } else {
                    0.0
                }
            }
            Filter::Triangle => (1.0 - input.x().abs()).max(0.0) * (1.0 - input.y().abs()).max(0.0),
            Filter::Gaussian(_) => todo!(),
            Filter::WindowedSinc(_) => todo!(),
        }
    }
}

fn main() {
    let (width, height) = (1024, 1024);
    let multiplicity = 10;
    let threads = 24;
    let n = 7;
    let mult = 1.0 - scale_factor(n);

    let filter = Filter::Triangle;
    // let pixel_extents = Vec2::new(1.0 / width as f32, 1.0 / height as f32);

    let mut cursors = Vec::new();
    for _ in 0..(threads * multiplicity) {
        cursors.push(Vec2::new(random(), random()));
    }

    let mut film = Film::new(width, height, 0.0f32);
    let mut max = 0.0f32;

    window_loop(
        width,
        height,
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

                *cursor = *cursor + (target - *cursor) * mult;
            });

            for v in &cursors {
                let sample_pos = Vec2::new(v.x() * width as f32, v.y() * height as f32);

                // let pixel_center = Vec2::new(
                //     sample_pos.x().floor() + 0.5, // / width as f32,
                //     sample_pos.y().floor() + 0.5, // / height as f32,
                // );
                let input = Vec2::new(
                    (v.x() * width as f32).fract() - 0.5,
                    (v.y() * height as f32).fract() - 0.5,
                );
                // let input = sample_pos - pixel_center;

                let filter_eval = filter.eval(input, Vec2::new(1.0, 1.0));

                film.buffer[(sample_pos.y() as usize) * width + (sample_pos.x() as usize)] +=
                    filter_eval;
                max = max.max(
                    film.buffer[(sample_pos.y() as usize) * width + (sample_pos.x() as usize)],
                );
            }
            buffer
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, pixel)| {
                    let greyscale = ((film.buffer[index] / max.sqrt()).min(1.0 - std::f32::EPSILON)
                        * 256.0) as u8;
                    *pixel = rgb_to_u32(greyscale, greyscale, greyscale);
                });
        },
        false,
    )
}
