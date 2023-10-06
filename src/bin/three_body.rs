extern crate minifb;

use minifb::{Key, MouseMode, Scale, Window, WindowOptions};
use rand::random;
use rayon::prelude::*;

use lib::{blit_circle, hsv_to_rgb, rgb_to_u32, trace::flatland::Vec2, triple_to_u32, Film};
use std::f32::consts::{PI, TAU};

#[derive(Copy, Clone, Debug)]
struct Body {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub mass: f32,
    pub color: u32,
    fx: f32,
    fy: f32,
}

impl Body {
    pub fn new(x: f32, y: f32, vx: f32, vy: f32, mass: f32, color: u32) -> Self {
        Body {
            x,
            y,
            vx,
            vy,
            mass,
            color,
            fx: 0.0,
            fy: 0.0,
        }
    }

    pub fn update_force(&mut self, other: &Body) {
        // apply gravitational equation
        let vec = Vec2::new(other.x - self.x, other.y - self.y);
        let distance_squared = vec.norm_squared();
        let mass_mult = self.mass * other.mass;

        let force = 0.001 * vec.normalized() * mass_mult / distance_squared;

        self.fx += force.x();
        self.fy += force.y();
    }

    pub fn update(&mut self, dt: f32) {
        self.vx += dt * self.fx / self.mass;
        self.vy += dt * self.fy / self.mass;

        self.x += self.vx * dt;
        self.y += self.vy * dt;

        self.fx = 0.0;
        self.fy = 0.0;
    }
}

fn main() {
    const WINDOW_WIDTH: usize = 1600;
    const WINDOW_HEIGHT: usize = 800;
    let mut window = Window::new(
        "Skydivers",
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
    let mut buffer = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);

    // Limit to max ~144 fps update rate
    let dt = 10.0 * 6944.0 / 1000000.0;
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    window
        .update_with_buffer(&buffer.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
        .unwrap();

    let mut bodies: Vec<Body> = Vec::new();

    let mut collective_momentum = Vec2::new(0.0, 0.0);
    for _ in 0..3 {
        let angle = TAU * random::<f32>();
        let speed = random::<f32>() * 1.0 + 0.5;
        let (sin, cos) = angle.sin_cos();
        let body = Body::new(
            100.0 * (random::<f32>() - 0.5),
            100.0 * (random::<f32>() - 0.5),
            speed * sin,
            speed * cos,
            50000.0 * random::<f32>() + 7000.0,
            triple_to_u32(hsv_to_rgb((random::<f32>() * 360.0) as usize, 1.0, 1.0)),
        );

        let velocity = Vec2::new(body.vx, body.vy);

        collective_momentum += velocity.norm_squared() * body.mass * 0.5 * velocity.normalized();

        bodies.push(body);
    }

    collective_momentum = collective_momentum / 3.0;

    for i in 0..3 {
        bodies[i].vx -= collective_momentum.x() / bodies[i].mass;
        bodies[i].vy -= collective_momentum.y() / bodies[i].mass;
    }

    let mut swap = bodies.clone();

    let window_bounds = ((-100.0, -100.0), (100.0, 100.0));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // buffer.fill(0u32);

        // detect if any body has achieved escape velocity wrt the center of mass and momentum of the other two

        swap.par_iter_mut().enumerate().for_each(|(idx, new)| {
            // update diver angle based on controller, then update forces on diver and speed
            let mut body = bodies[idx];

            for (other_idx, other) in bodies.iter().enumerate() {
                if other_idx == idx {
                    continue;
                }
                body.update_force(other);
            }
            body.update(dt);

            *new = body;
        });

        for body in bodies.iter() {
            let (px, py) = (
                ((body.x - window_bounds.0 .0) / (window_bounds.1 .0 - window_bounds.0 .0)
                    * WINDOW_WIDTH as f32) as usize,
                ((body.y - window_bounds.0 .1) / (window_bounds.1 .1 - window_bounds.0 .1)
                    * WINDOW_HEIGHT as f32) as usize,
            );

            // println!(
            //     "{} between {} and {} -> {}",
            //     body.x, window_bounds.0 .0, window_bounds.1 .0, px
            // );
            if px >= WINDOW_WIDTH || py >= WINDOW_HEIGHT {
                continue;
            }
            // buffer[py * WINDOW_WIDTH + px] = body.color;
            blit_circle(&mut buffer, 1.0 * body.mass / 2000.0, px, py, body.color);
        }
        std::mem::swap(&mut bodies, &mut swap);
        window
            .update_with_buffer(&buffer.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
