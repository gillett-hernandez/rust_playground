#![feature(slice_fill)]
extern crate minifb;

use lib::*;

use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::*;

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;
#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub time: f32,
    pub mass: f32,
    pub radius: f32,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub ax: f32,
    pub ay: f32,
    // pub spin: f32,
}

impl Particle {
    pub fn new(mass: f32, radius: f32, x: f32, y: f32, vx: f32, vy: f32, ax: f32, ay: f32) -> Self {
        Particle {
            time: 0.0,
            mass,
            radius,
            x,
            y,
            vx,
            vy,
            ax,
            ay,
            // spin: 0.0,
        }
    }
    pub fn normalize(&mut self) {
        let scale = self.vx.hypot(self.vy);
        self.vx = self.vx / scale;
        self.vy = self.vy / scale;
    }
    pub fn update(&mut self, dt: f32) {
        self.vx += self.ax * dt;
        self.vy += self.ay * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;
    }
}

pub fn particle_wall_check(
    time: f32,
    i: usize,
    particle: &Particle,
) -> Option<(f32, f32, usize, Option<usize>)> {
    let dt = if particle.vy < 0.0 {
        Some((particle.radius - particle.y) / particle.vy)
    } else if particle.vy > 0.0 {
        Some((1.0 - particle.radius - particle.y) / particle.vy)
    } else {
        None
    };

    let mut min = std::f32::INFINITY;
    if let Some(dt) = dt {
        if dt > 0.0 {
            min = min.min(dt);
        }
    }
    let dt = if particle.vx < 0.0 {
        Some((particle.radius - particle.x) / particle.vx)
    } else if particle.vx > 0.0 {
        Some((1.0 - particle.radius - particle.x) / particle.vx)
    } else {
        None
    };
    if let Some(dt) = dt {
        if dt > 0.0 {
            min = min.min(dt);
        }
    }
    if min.is_finite() {
        return Some((time, time + min, i, None));
    } else {
        None
    }
}

fn main() {
    let mut window = Window::new(
        "Swarm",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    // let dt = 6944.0 / 1000000.0;

    let mut particles: Vec<Particle> = Vec::new();
    let num_particles = 700;

    let phi = random::<f32>() * std::f32::consts::TAU;
    let mag = random::<f32>() * 0.3 + 0.9;
    let particle = Particle::new(
        3.5,
        0.05,
        0.5,
        0.5,
        mag * phi.cos(),
        mag * phi.sin(),
        0.0,
        0.0,
    );

    particles.push(particle);
    for i in 0..num_particles - 1 {
        let phi = random::<f32>() * std::f32::consts::TAU;
        let mag = random::<f32>() * 0.3 + 0.2;
        let r = random::<f32>() * 0.003 + 0.003;
        let particle = loop {
            let x = random::<f32>() * (1.0 - 2.0 * r) + r;
            let y = random::<f32>() * (1.0 - 2.0 * r) + r;
            let particle = Particle::new(
                0.5,
                r,
                x,
                y,
                // mag * phi.cos(),
                -(0.5 - y),
                0.5 - x,
                // mag * phi.sin(),
                0.0,
                0.0,
            );
            for other in particles[0..i].iter() {
                if (other.y - particle.y).hypot(other.x - particle.x)
                    <= particle.radius + other.radius
                {
                    continue;
                }
            }
            break particle;
        };
        particles.push(particle);
    }

    // let mut swap = particles.clone();

    // for i in 0..num_particles {
    //     for j in 0..num_particles {
    //         if j <= i {
    //             break;
    //         }
    //         // predict collision time for particles i and j.
    //         // and push to event queue.
    //     }
    // }

    let mut t = 0.0;
    let frame_dt = 6944.0 / 1000000.0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        film.buffer.fill(0u32);

        // for particle in particles.iter() {
        //     let (px, py) = (
        //         (particle.x * WINDOW_WIDTH as f32) as usize,
        //         (particle.y * WINDOW_HEIGHT as f32) as usize,
        //     );
        //     attempt_write(&mut film, px, py, 255u32 + 255u32 << 8);
        //     let PIXEL_DIAMETER = 1.0 / film.width as f32;

        //     let mut r = 1;
        //     let pixel_radius = loop {
        //         if r as f32 * PIXEL_DIAMETER > particle.radius {
        //             break r;
        //         }
        //         r += 1;
        //     };

        //     let e = 0.5 * particle.mass * particle.vx.hypot(particle.vy).powi(2);
        //     let c = triple_to_u32(hsv_to_rgb(
        //         ((360.0 * (1.0 - (-e).exp())) as usize + 100) % 360,
        //         1.0,
        //         1.0,
        //     ));
        //     blit_circle(&mut film, pixel_radius as f32, px, py, c);

        //     attempt_write(&mut film, px, py, c);
        // }
        // std::mem::swap(&mut particles, &mut swap);
        window
            .update_with_buffer(&film.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
