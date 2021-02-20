#![feature(slice_fill)]
extern crate minifb;

use minifb::{Key, Scale, Window, WindowOptions};
use rand::prelude::*;
use rayon::prelude::*;

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
        let scale = self.vx.hypot(self.vy);
        self.vx = self.vx / scale;
        self.vy = self.vy / scale;
    }
    pub fn _turn_towards(&mut self, tx: f32, ty: f32, speed: f32) {
        let (dx, dy) = (tx - self.x, ty - self.y);
        self.vx += dx * speed;
        self.vy += dy * speed;
        // self.normalize();
    }
    pub fn _align_with(&mut self, tvx: f32, tvy: f32, speed: f32) {
        let (dvx, dvy) = (tvx - self.vx, tvy - self.vy);
        self.vx -= dvx * speed;
        self.vy -= dvy * speed;
        // self.normalize();
    }
    pub fn update(&mut self) {
        if self.vx.hypot(self.vy) > self.max_speed {
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
        let mut swarmling = Swarmling::new(0.06, 0.03, 0.01, 0.003);
        swarmling.x = random::<f32>();
        swarmling.y = random::<f32>();
        swarmling.vx = random::<f32>() - 0.3;
        swarmling.vy = random::<f32>() - 0.3;
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
                let (mut avg_vx, mut avg_vy) = (0.0, 0.0);
                let mut ct = 0;
                for (o_idx, other) in swarm.iter().enumerate() {
                    if idx == o_idx {
                        continue;
                    }
                    let (dx, dy) = (other.x - swarmling.x, other.y - swarmling.y);
                    let distance = dx.hypot(dy);
                    if distance < swarmling.perception_radius {
                        ct += 1;
                        let (dvx, dvy) = (other.vx - swarmling.vx, other.vy - swarmling.vy);
                        if distance < swarmling.social_radius {
                            if distance < swarmling.at_field_radius {
                                // strongly repel
                                let strength = 2.0 * (-distance).exp();
                                avg_vx -= 2.0 * strength * dx / distance;
                                avg_vy -= 2.0 * strength * dy / distance;
                            } else {
                                // weakly repel to approach social_distance
                                // avg_vx -= 2.0 * dx;
                                // avg_vy -= 2.0 * dy;

                                // avg_vx += dvx;
                                // avg_vy += dvy;
                            }
                        }
                        // weakly attract to approach social distance
                        avg_vx += 0.3 * dx;
                        avg_vy += 0.3 * dy;
                        avg_vx += 2.0 * dvx;
                        avg_vy += 2.0 * dvy;
                    }
                }

                if ct > 0 {
                    swarmling.vx += avg_vx / (ct as f32) / 100.0;
                    swarmling.vy += avg_vy / (ct as f32) / 100.0;
                }

                let current_speed = swarmling.vx.hypot(swarmling.vy);
                swarmling.vx = (current_speed + 0.1) * swarmling.vx / current_speed;
                swarmling.vy = (current_speed + 0.1) * swarmling.vy / current_speed;

                swarmling.vx *= 0.99;
                swarmling.vy *= 0.99;

                swarmling.update();
                if swarmling.x > 1.0 {
                    swarmling.x = 1.0 - std::f32::EPSILON;
                    swarmling.vx *= -1.0;
                } else if swarmling.x < 0.0 {
                    swarmling.x = 0.0;
                    swarmling.vx *= -1.0;
                }
                if swarmling.y > 1.0 {
                    swarmling.y = 1.0 - std::f32::EPSILON;
                    swarmling.vy *= -1.0;
                } else if swarmling.y < 0.0 {
                    swarmling.y = 0.0;
                    swarmling.vy *= -1.0;
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
