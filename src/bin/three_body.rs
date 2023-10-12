extern crate minifb;

use itertools::Itertools;
use minifb::{Key, MouseMode, Scale, Window, WindowOptions};
use ordered_float::OrderedFloat;
use rand::random;
use rayon::prelude::*;

use lib::{
    blit_circle, hsv_to_rgb, random_in_unit_sphere, random_on_unit_sphere, rgb_to_u32,
    triple_to_u32, u32_to_rgb, Film, RandomSampler, Sampler, TangentFrame,
};
use math::prelude::{Point3, Vec3};
use std::f32::consts::{PI, TAU};

const STEPS: usize = 200;
const GRAVITATIONAL_CONSTANT: f32 = 0.0001;

#[derive(Copy, Clone, Debug)]
struct Body {
    pub p: Point3,
    pub v: Vec3,
    pub mass: f32,
    pub color: u32,
    f_accum: Vec3,
}

impl Body {
    pub fn new(p: Point3, v: Vec3, mass: f32, color: u32) -> Self {
        Body {
            p,
            v,
            mass,
            color,
            f_accum: Vec3::ZERO,
        }
    }

    pub fn update_force(&mut self, other: &Body) {
        // apply gravitational equation

        let displacement = other.p - self.p;
        let distance_squared = displacement.norm_squared();
        let mass_mult = self.mass * other.mass;

        let force =
            GRAVITATIONAL_CONSTANT * displacement.normalized() * mass_mult / distance_squared;

        self.f_accum = self.f_accum + force;
    }

    pub fn update(&mut self, dt: f32) {
        self.v = self.v + dt * self.f_accum / self.mass;

        self.p += self.v * dt;

        self.f_accum = Vec3::ZERO;
    }
}

fn initialize_bodies(bodies: &mut Vec<Body>, n: usize) {
    bodies.clear();
    let mut collective_momentum = Vec3::ZERO;

    let mut sampler = RandomSampler::new();

    let center_radius = 60.0;

    for i in 0..(n - 1) {
        let random_position_vector = random_on_unit_sphere(sampler.draw_2d());
        let random_direction_vector = random_on_unit_sphere(sampler.draw_2d());

        let speed = 0.01;

        let body = Body::new(
            (center_radius * random_position_vector).into(),
            random_direction_vector * speed,
            1000000.0,
            triple_to_u32(hsv_to_rgb(i * (360 - 1) / n, 1.0, 1.0)),
        );

        collective_momentum = collective_momentum + body.v * body.mass;

        bodies.push(body);
    }

    {
        let chosen_body = (random::<u32>() % 3) as usize;
        let angle = TAU * random::<f32>();
        // let speed = random::<f32>() * 1.0 + 0.5;
        let speed = 1.0;
        let (sin, cos) = angle.sin_cos();

        let normal = random_on_unit_sphere(sampler.draw_2d());
        let tangentframe = TangentFrame::from_normal(normal);

        let reference_body = &bodies[chosen_body];

        let orbit_radius = 35.0;

        let local_velocity = Vec3::new(cos * speed, sin * speed, 0.0);

        let velocity = tangentframe.to_world(&local_velocity);
        // generate random rotation vector and angle to rotate the velocity around

        let body = Body::new(
            reference_body.p + normal * orbit_radius,
            velocity,
            1.0,
            triple_to_u32(hsv_to_rgb(0, 0.0, 1.0)),
        );

        collective_momentum = collective_momentum + body.v * body.mass;

        bodies.push(body);
    }

    collective_momentum = collective_momentum / n as f32;

    // for i in 0..n {
    //     // bodies[i].vx -= collective_momentum.x() / bodies[i].mass;
    //     bodies[i].v = bodies[i].v - collective_momentum / bodies[i].mass;
    // }
}

fn main() {
    const WINDOW_WIDTH: usize = 1000;
    const WINDOW_HEIGHT: usize = 1000;
    let mut window = Window::new(
        "N-Body",
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

    if false {
        let ypp = |y: f32| -y;
        let y0 = 0.0f32;
        let yp0 = 1.0f32;
        let dt = 0.01;
        let target_time = 10.0 * TAU;
        let required_steps = (target_time / dt) as u32;

        let solved = |t: f32| t.sin();

        let backward_euler_steps = 5;
        let mut ypk_forward = yp0;
        let mut ypk_backward = yp0;
        let mut yk_forward = y0;
        let mut yk_backward = y0;
        let mut t = 0.0;
        for _ in 0..required_steps {
            // yk1 = yk + f(yk)
            t += dt;
            ypk_forward += dt * ypp(yk_forward);
            yk_forward += dt * ypk_forward;

            /* // iterative
            let mut new_ypk = ypk_backward;
            let mut new_yk = yk_backward;
            for _ in 0..backward_euler_steps {
                new_ypk = ypk_backward + dt * ypp(new_yk);
                new_yk = yk_backward + dt * new_ypk;
            }
            ypk_backward = new_ypk;
            yk_backward = new_yk; */

            /* // second order difeq, harmonic motion
            // new_ypk = ypk_backward + dt * ypp(new_yk);
            // new_yk = yk_backward + dt * new_ypk;

            // new_yk = yk_backward + dt * (ypk_backward - dt * new_yk);
            // new_yk = yk_backward + dt * ypk_backward - dt^2 * new_yk;
            // new_yk + dt^2 * new_yk = yk_backward + dt * ypk_backward;
            // new_yk (1 + dt^2) = yk_backward + dt * ypk_backward;

            // new_yk = (yk_backward + dt * ypk_backward) / (1 + dt^2);
            // new_ypk = ypk_backward - dt * new_yk;
            */
            let new_yk = (yk_backward + dt * ypk_backward) / (1.0 + dt * dt);
            let new_ypk = ypk_backward - dt * new_yk;

            yk_backward = new_yk;
            ypk_backward = new_ypk;

            /* //first order difeq, exponential decay yp = -0.5y
            // new_yk = yk_backward_euler + dt * yp(new_yk);
            // new_yk = yk_backward_euler - dt * 0.5 * new_yk
            // new_yk + 0.5 * dt * new_yk = yk_backward_euler
            // new_yk * (1 + 0.5 * dt) = yk_backward_euler
            // new_yk = yk_backward_euler / (1 + 0.5 * dt)
            // yk_backward_euler = yk_backward_euler / (1.0 + 0.5 * dt);

            // yk_backward_euler = new_yk; */

            let true_y = solved(t);
            println!(
                "true: {}, forward: {} (error: {}), backward: {} (error: {})",
                true_y,
                yk_forward,
                (yk_forward - true_y) / true_y,
                yk_backward,
                (yk_backward - true_y) / true_y,
            );
        }
        return;
    }

    let mut buffer = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);

    let mut viewport_scale = 250.0;
    // Limit to max ~144 fps update rate
    let max_dt = 1.0 / STEPS as f32;
    let mut speed_factor = 0.3f32;
    let mut dt = max_dt;
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    window
        .update_with_buffer(&buffer.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
        .unwrap();

    let n = 4;
    let mut bodies: Vec<Body> = Vec::new();

    initialize_bodies(&mut bodies, n);

    let mut swap = bodies.clone();

    let mut window_bounds = (
        (-viewport_scale, -viewport_scale),
        (viewport_scale, viewport_scale),
    );

    let mut framecounter = 0;
    let mut metaframe_min_dt = max_dt;
    let mut metaframe_elapsed_sim_time = 0.0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.buffer.par_iter_mut().for_each(|px| {
            let (mut r, mut g, mut b) = u32_to_rgb(*px);
            r = (r as f32 * 99.0 / 100.0).floor() as u8;
            g = (g as f32 * 99.0 / 100.0).floor() as u8;
            b = (b as f32 * 99.0 / 100.0).floor() as u8;
            *px = triple_to_u32((r, g, b));
        });

        // TODO: detect if any body has achieved escape velocity wrt the center of mass and momentum of the other two
        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            initialize_bodies(&mut bodies, n);
        }

        if window.is_key_pressed(Key::LeftBracket, minifb::KeyRepeat::No) {
            speed_factor /= 1.1;
            println!("new speed factor = {}", speed_factor);
        }
        if window.is_key_pressed(Key::RightBracket, minifb::KeyRepeat::No) {
            speed_factor *= 1.1;
            println!("new speed factor = {}", speed_factor);
        }

        if window.is_key_pressed(Key::Q, minifb::KeyRepeat::No) {
            viewport_scale /= 1.1;
            window_bounds = (
                (-viewport_scale, -viewport_scale),
                (viewport_scale, viewport_scale),
            );
            println!("new viewport scale = {}", viewport_scale);
        }
        if window.is_key_pressed(Key::E, minifb::KeyRepeat::No) {
            viewport_scale *= 1.1;
            window_bounds = (
                (-viewport_scale, -viewport_scale),
                (viewport_scale, viewport_scale),
            );
            println!("new viewport scale = {}", viewport_scale);
        }

        let mut frame_min_dt = max_dt;
        let mut frame_elapsed_sim_time = 0.0;

        for _ in 0..STEPS {
            let max_speed_squared = swap
                .par_iter_mut()
                .enumerate()
                .map(|(idx, new)| {
                    let mut body = bodies[idx];

                    // let mut min_dist_squared = f32::INFINITY;
                    for (other_idx, other) in bodies.iter().enumerate() {
                        if other_idx == idx {
                            continue;
                        }
                        body.update_force(other);
                        // min_dist_squared = min_dist_squared.min((body.p - other.p).norm_squared());
                    }
                    body.update(dt);

                    *new = body;
                    OrderedFloat::from(body.v.norm_squared())
                })
                .max()
                .unwrap();
            frame_elapsed_sim_time += dt;

            let swap_len = swap.len();
            let min_dist = (0..swap_len)
                .into_iter()
                .cartesian_product((0..swap_len).into_iter())
                .par_bridge()
                .map(|(i0, i1)| {
                    if i0 >= i1 {
                        return OrderedFloat::from(f32::INFINITY);
                    }
                    let b0 = &swap[i0];
                    let b1 = &swap[i1];
                    OrderedFloat::from((b0.p - b1.p).norm())
                })
                .min()
                .unwrap();

            // ~~the smaller the min_dist, the smaller the dt~~
            // the greater the speed, the smaller the dt

            dt = max_dt * (1.0 + speed_factor * max_speed_squared.0).recip();
            frame_min_dt = frame_min_dt.min(dt);
            std::mem::swap(&mut bodies, &mut swap);
        }
        metaframe_min_dt = metaframe_min_dt.min(frame_min_dt);
        metaframe_elapsed_sim_time += frame_elapsed_sim_time;

        for (idx, body) in bodies.iter().enumerate() {
            let (px, py) = (
                ((body.p.x() - window_bounds.0 .0) / (window_bounds.1 .0 - window_bounds.0 .0)
                    * WINDOW_WIDTH as f32) as usize,
                ((body.p.y() - window_bounds.0 .1) / (window_bounds.1 .1 - window_bounds.0 .1)
                    * WINDOW_HEIGHT as f32) as usize,
            );

            // println!(
            //     "{} between {} and {} -> {}",
            //     body.x, window_bounds.0 .0, window_bounds.1 .0, px
            // );
            if px >= WINDOW_WIDTH || py >= WINDOW_HEIGHT {
                continue;
            }
            if idx == 3 {
                buffer.buffer[py * WINDOW_WIDTH + px] = body.color;
            } else {
                blit_circle(&mut buffer, 1.0, px, py, body.color);
            }
        }
        window
            .update_with_buffer(&buffer.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
        framecounter += 1;

        if framecounter % 144 == 0 {
            framecounter %= 144;
            println!("current delta_time = {}, minimum delta time in metaframe = {}, metaframe elapsed time = {}", dt, metaframe_min_dt, metaframe_elapsed_sim_time);
            metaframe_min_dt = max_dt;
            metaframe_elapsed_sim_time = 0.0;
        }
    }
}
