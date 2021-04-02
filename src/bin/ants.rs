#![feature(slice_fill)]
extern crate minifb;

use std::f32::consts::{PI, TAU};

use lib::rgb_to_u32;
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use packed_simd::f32x4;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
enum Event {
    DecrementFood { deposit_id: usize },
}

#[derive(Copy, Clone, Debug)]
struct FoodDeposit {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub count: usize,
}

impl FoodDeposit {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        (x - self.x).hypot(y - self.y) < self.radius
    }
    pub fn decrement(&mut self) {
        if self.count > 0 {
            self.count -= 1;
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum AntState {
    Exploring,
    Foraging,
}

#[derive(Copy, Clone, Debug)]
struct Ant {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub perception_radius: f32,
    pub perception_angle: f32,
    pub max_speed: f32,
    pub turn_speed: f32,
    pub state: AntState,
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
            state: AntState::Exploring,
            random_walk_turn_speed,
        }
    }
    pub fn update(&mut self) {
        self.angle %= TAU;
        let (sin, cos) = self.angle.sin_cos();
        self.x += self.max_speed * cos;
        self.y += self.max_speed * sin;
    }

    pub fn steer(&mut self, pheromone_buffer: &Vec<f32x4>) {
        // arc perception field

        let w = (pheromone_buffer.len() as f32).sqrt();

        let mut avg_angle =
            f32x4::splat((2.0 * random::<f32>() - 1.0) * self.random_walk_turn_speed);

        let mut s = f32x4::splat(1.0);

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

        let decision = match self.state {
            AntState::Exploring => {
                // pick pheromone by priority 2 1 3 0
                if s.extract(2) > 1.0f32 {
                    2
                } else if s.extract(1) > 1.0f32 {
                    1
                } else if s.extract(3) > 1.0f32 {
                    3
                } else {
                    0
                }
            }
            AntState::Foraging => {
                // pick pheromone by priority 0 1 3
                // no 2 because 2 is food and we're already carrying food
                if s.extract(0) > 1.0f32 {
                    0
                } else if s.extract(1) > 1.0f32 {
                    1
                } else {
                    3
                }
            }
        };

        self.angle += (avg_angle / s)
            .max(f32x4::splat(-self.turn_speed))
            .min(f32x4::splat(self.turn_speed))
            .extract(decision);
    }

    pub fn lay_pheromone(&self) -> f32x4 {
        match self.state {
            AntState::Exploring => f32x4::new(1.0, 0.0, 0.0, 0.0),
            AntState::Foraging => f32x4::new(0.0, 1.0, 0.0, 0.0),
        }
    }
}

fn tonemap_greyscale(input: f32) -> f32 {
    // 1.0 - (-input).exp()
    input / (1.0 + input)
}

fn crush(input: f32) -> u8 {
    (input.clamp(0.0, 1.0) * 255.0) as u8
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
    let mut pheromone_buffer = vec![f32x4::splat(0.0); WINDOW_WIDTH * WINDOW_HEIGHT];

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    let mut colony: Vec<Ant> = Vec::new();

    let conservation_factor = f32x4::new(0.8, 0.9, 0.7, 0.7);
    let spread_factor: f32x4 = (1.0 - conservation_factor) / 8.0;
    let evaporation_factor = f32x4::new(0.99, 0.999, 0.5, 0.99);

    for _ in 0..10000 {
        let mut ant = Ant::new(0.008, 0.5, 0.0005, 0.7, 0.05);
        let r = random::<f32>().sqrt() * 0.03;
        let phi = random::<f32>() * TAU;
        let (y, x) = phi.sin_cos();
        ant.x = 0.5 + x * r;
        ant.y = 0.5 + y * r;
        ant.angle = (random::<f32>() - 0.5) * TAU;
        // ant.angle = phi + PI / 2.0 - PI / 4.0;
        colony.push(ant);
    }

    let mut swap = colony.clone();
    let mut pheromone_swap = pheromone_buffer.clone();

    let mut food_deposits: Vec<FoodDeposit> = vec![
        FoodDeposit {
            x: 0.14,
            y: 0.14,
            radius: 0.09,
            count: 100000,
        },
        FoodDeposit {
            x: 0.86,
            y: 0.86,
            radius: 0.09,
            count: 100000,
        },
    ];

    let mut selected_view = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0u32);

        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            selected_view += 1;
            selected_view %= 4;
        }

        // ants update step
        let events: Vec<Event> = swap
            .par_iter_mut()
            .enumerate()
            .flat_map(|(idx, ant)| {
                let mut events = Vec::new();
                *ant = colony[idx];

                ant.steer(&pheromone_buffer);

                // check food locations and grab food if within region
                for (idx, food_deposit) in food_deposits.iter().enumerate() {
                    if food_deposit.contains(ant.x, ant.y) {
                        ant.state = AntState::Foraging;
                        ant.angle += PI;
                        ant.angle %= TAU;
                        events.push(Event::DecrementFood { deposit_id: idx });
                    }
                }

                // check if near colony at (0.5, 0.5)
                if (ant.x - 0.5).hypot(ant.y - 0.5) < 0.05 {
                    // within colony region, so deposit food and switch back to exploring mode
                    ant.state = AntState::Exploring;
                }

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
                events
            })
            .collect();
        for event in events {
            match event {
                Event::DecrementFood { deposit_id } => {
                    food_deposits[deposit_id].decrement();
                }
            }
        }

        for FoodDeposit {
            x,
            y,
            radius,
            count,
        } in food_deposits.iter()
        {
            for idx in 0..*count {
                let r = random::<f32>().sqrt() * radius;
                let phi = random::<f32>() * TAU;
                let (sin, cos) = phi.sin_cos();
                let (x, y) = (x + r * cos, y + r * sin);
                let (px, py) = (
                    (x * WINDOW_WIDTH as f32) as usize,
                    (y * WINDOW_HEIGHT as f32) as usize,
                );
                if py * WINDOW_WIDTH + px >= buffer.len() {
                    continue;
                }
                pheromone_buffer[py * WINDOW_WIDTH + px] +=
                    f32x4::new(0.0, 0.0, 1.0 / *count as f32, 0.0);
            }
        }

        // pheromone_buffer update step
        let mut avg_pheromone = f32x4::splat(0.0);
        let mut max_pheromone = f32x4::splat(0.0);
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

        // apply blur kernel to pheromone buffer
        pheromone_swap
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pheromone)| {
                let (px, py) = ((idx % WINDOW_WIDTH) as isize, (idx / WINDOW_WIDTH) as isize);

                let mut s = f32x4::splat(0.0);
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

                        if *dx > 0 {
                            if (px + dx) % (WINDOW_WIDTH as isize) < px {
                                continue;
                            }
                        } else if *dx < 0 {
                            if (px + dx) % (WINDOW_WIDTH as isize) > px {
                                continue;
                            }
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
            // let pheromone_scalar = tonemap_greyscale(
            //     10.0 * pheromone_buffer[i].extract(selected_view)
            //         / max_pheromone.extract(selected_view),
            // );

            *pixel = rgb_to_u32(
                crush(tonemap_greyscale(
                    pheromone_buffer[i].extract(0) / max_pheromone.extract(0),
                )),
                crush(tonemap_greyscale(
                    pheromone_buffer[i].extract(1) / max_pheromone.extract(1),
                )),
                crush(tonemap_greyscale(
                    pheromone_buffer[i].extract(2) / max_pheromone.extract(2),
                )),
            );
        }

        std::mem::swap(&mut colony, &mut swap);
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
