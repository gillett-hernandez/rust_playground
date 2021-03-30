#![feature(slice_fill)]
extern crate minifb;

use std::f32::consts::{PI, TAU};

use lib::rgb_to_u32;
use minifb::{Key, Scale, Window, WindowOptions};
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
struct Ant {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub perception_radius: f32,
    pub perception_angle: f32,
    pub max_speed: f32,
    pub turn_speed: f32,
    random_walk_turn_speed: f32,
}

impl Ant {
    pub fn new(
        perception_radius: f32,
        perception_angle: f32,
        max_speed: f32,
        turn_speed: f32,
        random_walk_turn_speed: f32,
    ) -> Self {
        Ant {
            x: 0.0,
            y: 0.0,
            angle: 0.0,
            perception_radius,
            perception_angle,
            max_speed,
            turn_speed,
            random_walk_turn_speed,
        }
    }
    pub fn update(&mut self) {
        self.angle %= TAU;
        let (sin, cos) = self.angle.sin_cos();
        self.x += self.max_speed * cos;
        self.y += self.max_speed * sin;
    }

    pub fn steer(&mut self, pheromone_buffer: &Vec<f32>) {
        // arc perception field

        let w = (pheromone_buffer.len() as f32).sqrt();

        let mut avg_angle = (2.0 * random::<f32>() - 1.0) * self.random_walk_turn_speed;

        let mut s = 1.0;

        // scan cone in front of ant
        for _ in 0..10 {
            let angle = (2.0 * random::<f32>() - 1.0) * self.perception_angle;
            let (sin, cos) = (self.angle + angle).sin_cos();
            let radius = random::<f32>().sqrt();
            let x = self.x + radius * self.perception_radius * cos;
            let y = self.y + radius * self.perception_radius * sin;
            let (tx, ty) = ((x * w) as usize, (y * w) as usize);
            let idx = tx + ty * w as usize;
            if idx < pheromone_buffer.len() {
                avg_angle += pheromone_buffer[idx] * angle;
                s += pheromone_buffer[idx];
            }
        }
        self.angle += (avg_angle / s).clamp(-self.turn_speed, self.turn_speed);
    }

    pub fn lay_pheromone(&self) -> f32 {
        1.0f32
    }
}

fn tonemap_greyscale(input: f32) -> u8 {
    ((1.0 - (-input).exp()) * 255.0) as u8
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
    let mut pheromone_buffer = vec![0f32; WINDOW_WIDTH * WINDOW_HEIGHT];

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    let mut colony: Vec<Ant> = Vec::new();

    let conservation_factor = 0.7;
    let spread_factor = (1.0 - conservation_factor) / 8.0;
    let evaporation_factor = 0.99;

    for _ in 0..10000 {
        let mut ant = Ant::new(0.008, 0.4, 0.0005, 0.4, 0.05);
        let r = random::<f32>().sqrt() * 0.04;
        let phi = random::<f32>() * TAU;
        let (y, x) = phi.sin_cos();
        ant.x = 0.5 + x * r;
        ant.y = 0.5 + y * r;
        ant.angle = (random::<f32>() - 0.5) * 0.1;
        // ant.angle = -phi;
        colony.push(ant);
    }

    let mut swap = colony.clone();
    let mut pheromone_swap = pheromone_buffer.clone();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0u32);
        swap.par_iter_mut().enumerate().for_each(|(idx, ant)| {
            // find neighbors in perception radius
            *ant = colony[idx];

            ant.steer(&pheromone_buffer);

            ant.update();
            if ant.x > 1.0 {
                // right edge
                ant.x = 1.0 - std::f32::EPSILON;
                ant.angle = (ant.angle.sin()).atan2(-ant.angle.cos());
            } else if ant.x < 0.0 {
                // left edge
                ant.x = 0.0;
                ant.angle = (ant.angle.sin()).atan2(-ant.angle.cos());
            }
            if ant.y > 1.0 {
                // bottom edge
                ant.y = 1.0 - std::f32::EPSILON;
                ant.angle = (-ant.angle.sin()).atan2(ant.angle.cos());
            } else if ant.y < 0.0 {
                // top edge
                ant.y = 0.0;
                ant.angle = (-ant.angle.sin()).atan2(ant.angle.cos());
            }
        });
        let mut avg_pheromone = 0.0f32;
        let mut max_pheromone = 0.0f32;
        for ant in colony.iter() {
            let (px, py) = (
                (ant.x * WINDOW_WIDTH as f32) as usize,
                (ant.y * WINDOW_HEIGHT as f32) as usize,
            );
            if py * WINDOW_WIDTH + px >= buffer.len() {
                continue;
            }
            pheromone_buffer[py * WINDOW_WIDTH + px] += ant.lay_pheromone();
            max_pheromone = max_pheromone.max(pheromone_buffer[py * WINDOW_WIDTH + px]);
            avg_pheromone += pheromone_buffer[py * WINDOW_WIDTH + px];
        }
        avg_pheromone /= pheromone_buffer.len() as f32;
        // println!("{}", max_pheromone);

        pheromone_swap
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pheromone)| {
                let (px, py) = ((idx % WINDOW_WIDTH) as isize, (idx / WINDOW_WIDTH) as isize);

                let mut s = 0.0;
                for dx in [-1, 0, 1].iter() {
                    for dy in [-1, 0, 1].iter() {
                        // contribution from the same cell
                        if *dx == *dy && *dy == 0 {
                            s += pheromone_buffer[idx as usize] * conservation_factor;
                            continue;
                        }
                        let idx = (py + dy) * WINDOW_WIDTH as isize + px + dx;
                        if idx < 0 || idx as usize >= pheromone_buffer.len() {
                            continue;
                        }
                        // contribution from other cells
                        s += pheromone_buffer[idx as usize] * spread_factor;
                    }
                }
                *pheromone = s;
                *pheromone *= evaporation_factor;
            });

        std::mem::swap(&mut pheromone_buffer, &mut pheromone_swap);

        for (i, pixel) in buffer.iter_mut().enumerate() {
            let c = tonemap_greyscale(10.0 * pheromone_buffer[i] / max_pheromone);

            *pixel = rgb_to_u32(c, c, c);
        }

        // apply blur kernel to pheromone buffer

        std::mem::swap(&mut colony, &mut swap);
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
