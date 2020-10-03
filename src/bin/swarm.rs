#![feature(slice_fill)]
extern crate minifb;

use lib::*;

use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rand::prelude::*;
use rayon::prelude::*;

fn hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

#[derive(Copy, Clone, Debug)]
struct Swarmling {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub perception_radius: f32,
    pub social_radius: f32,
    pub at_field_radius: f32,
    pub max_speed: f32,
}

impl Swarmling {
    pub fn new(
        perception_radius: f32,
        social_radius: f32,
        at_field_radius: f32,
        max_speed: f32,
    ) -> Self {
        Swarmling {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            perception_radius,
            social_radius,
            at_field_radius,
            max_speed,
        }
    }
    pub fn normalize(&mut self) {
        let scale = hypot(self.vx, self.vy);
        self.vx = self.vx / scale;
        self.vy = self.vy / scale;
    }
    pub fn turn_towards(&mut self, tx: f32, ty: f32, speed: f32) {
        let (dx, dy) = (tx - self.x, ty - self.y);
        self.vx += dx * speed;
        self.vy += dy * speed;
        // self.normalize();
    }
    pub fn align_with(&mut self, tvx: f32, tvy: f32, speed: f32) {
        let (dvx, dvy) = (tvx - self.vx, tvy - self.vy);
        self.vx -= dvx * speed;
        self.vy -= dvy * speed;
        // self.normalize();
    }
    pub fn update(&mut self) {
        if hypot(self.vx, self.vy) > self.max_speed {
            self.normalize();
            self.vx *= self.max_speed;
            self.vy *= self.max_speed;
        }
        self.x += self.vx;
        self.y += self.vy;
    }
}

fn main() {
    const WINDOW_WIDTH: usize = 800;
    const WINDOW_HEIGHT: usize = 800;
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
    let mut buffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    let mut swarm: Vec<Swarmling> = Vec::new();

    for _ in 0..1000 {
        let mut swarmling = Swarmling::new(0.3, 0.05, 0.01, 0.005);
        swarmling.x = random::<f32>();
        swarmling.y = random::<f32>();
        swarmling.vx = random::<f32>() - 0.5;
        swarmling.vy = random::<f32>() - 0.5;
        swarmling.normalize();
        swarm.push(swarmling);
    }

    let mut swap = swarm.clone();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0u32);
        swap.par_iter_mut()
            .enumerate()
            .for_each(|(idx, swarmling)| {
                // find neighbors in perception radius
                *swarmling = swarm[idx];
                for (o_idx, other) in swarm.iter().enumerate() {
                    if idx == o_idx {
                        continue;
                    }
                    let (dx, dy) = ((swarmling.x - other.x).abs(), (swarmling.y - other.y).abs());
                    let distance = hypot(dx, dy);
                    if distance < swarmling.perception_radius {
                        if distance < swarmling.social_radius {
                            if distance < swarmling.at_field_radius {
                                // strongly repel
                                swarmling.turn_towards(
                                    other.x,
                                    other.y,
                                    -swarmling.at_field_radius / distance * 0.1,
                                );
                            } else {
                                // weakly repel to approach social_distance
                                swarmling.turn_towards(other.x, other.y, -0.004);
                                swarmling.align_with(other.vx, other.vy, -0.004);
                            }
                        } else {
                            // weakly attract to approach social distance
                            swarmling.turn_towards(other.x, other.y, 0.00004);
                            swarmling.align_with(other.vx, other.vy, 0.0004);
                        }
                    }
                }
                swarmling.vx *= 0.99;
                swarmling.vy *= 0.99;

                swarmling.update();
                if swarmling.x > 1.0 {
                    swarmling.x = 1.0 - std::f32::EPSILON;
                } else if swarmling.x < 0.0 {
                    swarmling.x = 0.0;
                }
                if swarmling.y > 1.0 {
                    swarmling.y = 1.0 - std::f32::EPSILON;
                } else if swarmling.y < 0.0 {
                    swarmling.y = 0.0;
                }
            });
        for swarmling in swarm.iter() {
            let (px, py) = (
                (swarmling.x * WINDOW_WIDTH as f32) as usize,
                (swarmling.y * WINDOW_HEIGHT as f32) as usize,
            );
            buffer[py * WINDOW_WIDTH + px] = 255u32 + 255u32 << 8;
        }
        std::mem::swap(&mut swarm, &mut swap);
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
