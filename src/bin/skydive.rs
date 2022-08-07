extern crate minifb;

use minifb::{Key, MouseMode, Scale, Window, WindowOptions};
use rayon::prelude::*;

use lib::{hsv_to_rgb, rgb_to_u32, triple_to_u32};
use std::f32::consts::{PI, TAU};

#[derive(Copy, Clone, Debug)]
struct Diver {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub angle: f32,
    pub mass: f32,
    gravity: f32,
    active: bool,
    pub color: u32,
}

impl Diver {
    pub fn new(x: f32, y: f32, mass: f32, gravity: f32, color: u32) -> Self {
        Diver {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            angle: 0.0,
            mass,
            gravity,
            active: true,
            color,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }
        let movement_direction = self.vy.atan2(self.vx) % TAU;

        let tangent = (-self.vx).atan2(self.vy) % TAU;
        let angle_of_attack = (movement_direction - self.angle) % TAU;

        let (aoa_sin, aoa_cos) = angle_of_attack.sin_cos();
        let mag_2 = self.vx * self.vx + self.vy * self.vy;
        let mul = 50.0 * 0.7 * 1.225 * 0.75 * mag_2 * aoa_sin * aoa_cos / (2.0 * self.mass);
        let (tan_sin, tan_cos) = tangent.sin_cos();
        let (lift_x, lift_y) = (mul * tan_cos, mul * tan_sin);

        let f = 0.038 + aoa_sin.abs() * 0.1;

        let mag = mag_2.sqrt();
        // velocity squared drag
        // F = - k * ||v||^2 * v / ||v||
        // F = - k *||v|| * v
        // F.x = - k * ||v|| * v.x;
        // F.y = - k * ||v|| * v.y;
        let fx = lift_x - f * mag * self.vx;
        let fy = lift_y - self.gravity * self.mass - f * mag * self.vy;
        if fy.abs() > 100000.0 {
            println!("{} {} {} {}", self.vy, self.mass, lift_y, f * mag * self.vy);
        }

        self.vx += dt * fx / self.mass;
        self.vy += dt * fy / self.mass;

        self.x += self.vx * dt;
        self.y += self.vy * dt;
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
    let mut buffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];

    // Limit to max ~144 fps update rate
    let dt = 10.0 * 6944.0 / 1000000.0;
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    window
        .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
        .unwrap();

    let mut divers: Vec<Diver> = Vec::new();

    for _ in 0..1000 {
        let mut diver = Diver::new(
            0.0,
            10000.0,
            50.0 * rand::random::<f32>() + 70.0,
            9.80665,
            triple_to_u32(hsv_to_rgb(
                (rand::random::<f32>() * 360.0) as usize,
                1.0,
                1.0,
            )),
        );
        // diver.vy = -100.0;
        diver.angle = rand::random::<f32>() * TAU;
        divers.push(diver);
    }

    divers[0].angle = 3.0 * PI / 4.0;
    divers[0].color = rgb_to_u32(255, 0, 0);

    let mut swap = divers.clone();

    let window_bounds = ((-10000.0, 10000.0), (10000.0, 0.0));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // buffer.fill(0u32);
        if let Some(mouse_pos) = window.get_mouse_pos(MouseMode::Pass) {
            let (dx, dy) = (
                mouse_pos.0 - WINDOW_WIDTH as f32 / 2.0,
                -(mouse_pos.1 - WINDOW_HEIGHT as f32 / 2.0),
            );
            divers[0].angle = dy.atan2(dx) % TAU;
        }
        swap.par_iter_mut().enumerate().for_each(|(idx, new)| {
            // update diver angle based on controller, then update forces on diver and speed
            let mut diver = divers[idx];
            if idx != 0 {
                diver.angle += 0.18 * (rand::random::<f32>() - 0.5);
                diver.angle %= TAU;
            }
            diver.update(dt);
            if diver.active
                && (diver.x > window_bounds.1 .0
                    || diver.x < window_bounds.0 .0
                    || diver.y > window_bounds.0 .1
                    || diver.y < window_bounds.1 .1)
            {
                diver.active = false;
                println!(
                    "{} {} {} {} {}",
                    diver.x, diver.y, diver.vx, diver.vy, diver.angle
                );
            }
            *new = diver;
        });
        let diver0 = divers[0];
        // println!(
        //     "{} {} {} {} {}",
        //     diver0.x, diver0.y, diver0.vx, diver0.vy, diver0.angle
        // );
        drop(diver0);
        for diver in divers.iter() {
            let (px, py) = (
                ((diver.x - window_bounds.0 .0) / (window_bounds.1 .0 - window_bounds.0 .0)
                    * WINDOW_WIDTH as f32) as usize,
                ((diver.y - window_bounds.0 .1) / (window_bounds.1 .1 - window_bounds.0 .1)
                    * WINDOW_HEIGHT as f32) as usize,
            );
            if px >= WINDOW_WIDTH || py >= WINDOW_HEIGHT {
                continue;
            }
            buffer[py * WINDOW_WIDTH + px] = diver.color;
        }
        std::mem::swap(&mut divers, &mut swap);
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
