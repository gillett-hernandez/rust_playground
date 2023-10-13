extern crate minifb;

use itertools::Itertools;
use minifb::{Key, MouseButton, MouseMode, Scale, Window, WindowOptions};
use num::{traits::real::Real, Num};
use ordered_float::OrderedFloat;
use rand::random;
use rayon::prelude::*;

use lib::{
    blit_circle, hsv_to_rgb, random_in_unit_sphere, random_on_unit_sphere, rgb_to_u32,
    trace::flatland::Vec2, triple_to_u32, u32_to_rgb, Film, RandomSampler, Sampler, TangentFrame,
};
use math::prelude::{Point3, Vec3};
use std::{
    collections::VecDeque,
    f32::consts::{FRAC_PI_2, PI, TAU},
};

const STEPS: usize = 200;
const GRAVITATIONAL_CONSTANT: f32 = 0.0001;

fn forward_euler_step<const N: usize, F, T>(mut func: F, x: [T; N], dt: T) -> [T; N]
where
    F: FnMut([T; N]) -> [T; N],
    T: Num + Real,
{
    // x' = F(x)
    let mut new_x = x;
    let xp = func(x);
    for i in 0..N {
        new_x[i] = new_x[i] + dt * xp[i];
    }
    new_x
}

enum CameraMode {
    Projective,
    Orthographic,
}

struct OrbitCamera {
    pub mode: CameraMode,
    pub origin: Point3,
    pub direction: Vec3,
    pub view_scale: f32,
    azimuthal_angle: f32,
    zenithal_angle: f32,
}

impl OrbitCamera {
    pub fn new(mode: CameraMode, origin: Point3, direction: Vec3, view_scale: f32) -> Self {
        OrbitCamera {
            mode,
            origin,
            direction,
            view_scale,
            azimuthal_angle: 0.0,
            zenithal_angle: 0.0,
        }
    }

    pub fn orbit(&mut self, orbit_by: Vec2) {
        let horizontal_orbit = orbit_by.x();
        self.azimuthal_angle -= horizontal_orbit;
        self.azimuthal_angle %= TAU;

        let vertical_orbit = orbit_by.y();
        self.zenithal_angle =
            (self.zenithal_angle + vertical_orbit).clamp(-FRAC_PI_2 * 0.99, FRAC_PI_2 * 0.99);

        let (a_sin, a_cos) = self.azimuthal_angle.sin_cos();
        let (z_sin, z_cos) = self.zenithal_angle.sin_cos();

        self.direction = Vec3::new(a_cos * z_cos, a_sin * z_cos, z_sin);
    }

    pub fn project(&self, point: Point3) -> (f32, f32) {
        let worldspace_point = point - self.origin;
        const UP: Vec3 = Vec3::Z;
        let right = UP.cross(self.direction).normalized();
        let relative_up = right.cross(self.direction);
        let tf = TangentFrame::new(right, relative_up, self.direction);
        let (x, y) = match self.mode {
            CameraMode::Projective => todo!(),
            CameraMode::Orthographic => {
                let local = tf.to_local(&worldspace_point);
                (local.x(), local.y())
            }
        };

        let (u, v) = (
            (0.5 - x / (2.0 * self.view_scale)),
            (0.5 - y / (2.0 * self.view_scale)),
        );

        // println!("{} {}, {} -> {} {}", x, y, self.view_scale, u, v);

        (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
    }
}

#[derive(Copy, Clone, Debug)]
struct Body {
    pub p: Point3,
    pub v: Vec3,
    pub mass: f32,
    pub color: u32,
    f_accum: Vec3,
    radius: f32,
}

impl Body {
    pub fn new(p: Point3, v: Vec3, mass: f32, color: u32, radius: f32) -> Self {
        Body {
            p,
            v,
            mass,
            color,
            f_accum: Vec3::ZERO,
            radius,
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

struct InactiveBody {
    pub p: Point3,
    pub color: u32,
    pub radius: f32,
    pub brightness: f32,
}

impl From<&Body> for InactiveBody {
    fn from(value: &Body) -> Self {
        InactiveBody {
            p: value.p,
            color: value.color,
            radius: value.radius,
            brightness: 1.0,
        }
    }
}

impl InactiveBody {
    pub fn adjusted_color(&self) -> u32 {
        let (mut r, mut g, mut b) = u32_to_rgb(self.color);
        r = (r as f32 * self.brightness).floor() as u8;
        g = (g as f32 * self.brightness).floor() as u8;
        b = (b as f32 * self.brightness).floor() as u8;
        triple_to_u32((r, g, b))
    }
}

fn initialize_bodies(bodies: &mut Vec<Body>, n: usize) {
    bodies.clear();
    let mut collective_momentum = Vec3::ZERO;

    let mut sampler = RandomSampler::new();

    let center_radius = 60.0;
    let mut total_mass = 0.0;

    for i in 0..(n - 1) {
        let random_position_vector = random_on_unit_sphere(sampler.draw_2d());
        let random_direction_vector = random_on_unit_sphere(sampler.draw_2d());

        let speed = 1.0;

        let body = Body::new(
            (center_radius * random_position_vector).into(),
            random_direction_vector * speed,
            1000000.0,
            triple_to_u32(hsv_to_rgb(i * (360 - 1) / n, 1.0, 1.0)),
            1.0,
        );

        collective_momentum = collective_momentum + body.v * body.mass;

        bodies.push(body);
        total_mass += body.mass;
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
            0.0,
        );

        collective_momentum = collective_momentum + body.v * body.mass;

        bodies.push(body);
        total_mass += body.mass;
    }

    // collective_momentum = collective_momentum / n as f32;

    // to neutralize the drift of the system, we need to determine what the average drift velocity is and then subtract that from every object

    let avg_drift_velocity = collective_momentum / total_mass;

    for i in 0..n {
        bodies[i].v = bodies[i].v - avg_drift_velocity;
    }
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
        // xp, x
        let x0 = [1.0, 0.0];
        // xpp, xp
        let xp = |x: [f32; 2]| [-x[1], x[0]];
        let y0 = 0.0f32;
        let yp0 = 1.0f32;
        let dt = 0.01;
        let target_time = 1.0 * TAU;
        let required_steps = (target_time / dt) as u32;

        let solved = |t: f32| t.sin();

        let mut ypk_backward = yp0;
        let mut yk_backward = y0;
        let mut t = 0.0;

        let mut x = x0;
        println!("t, true, forward, backward");
        for _ in 0..required_steps {
            t += dt;

            x = forward_euler_step(xp, x, dt);

            let new_yk = (yk_backward + dt * ypk_backward) / (1.0 + dt * dt);
            let new_ypk = (ypk_backward - dt * yk_backward) / (1.0 + dt * dt);

            yk_backward = new_yk;
            ypk_backward = new_ypk;

            let true_y = solved(t);
            println!(
                "{}, {}, {}, {}",
                t,
                true_y,
                x[1],
                // (x[1] - true_y).abs() / true_y,
                // (x[1] - true_y).abs(),
                yk_backward,
                // (yk_backward - true_y).abs() / true_y,
                // (yk_backward - true_y).abs()
            );
        }
        return;
    }

    let mut buffer = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);

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

    let mut body_trails: Vec<VecDeque<InactiveBody>> = bodies
        .iter()
        .map(|_| VecDeque::with_capacity(100))
        .collect_vec();

    let mut swap = bodies.clone();

    // let mut view_angle = random_on_unit_sphere(RandomSampler::new().draw_2d());
    // let (mut view_azimuthal, mut view_zenithal) = (0.0, 0.0);
    let mut camera = OrbitCamera::new(CameraMode::Orthographic, Point3::ORIGIN, Vec3::X, 250.0);
    let camera_orbit_factor = 0.01;

    let mut last_mouse_pos = None;

    let mut framecounter = 0;
    let mut metaframe_min_dt = max_dt;
    let mut metaframe_elapsed_sim_time = 0.0;
    let mut elapsed_sim_time = 0.0;
    let mut elapsed_sim_time_ticker = 0.0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.buffer.fill(0u32);

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

        // if window.is_key_pressed(Key::Q, minifb::KeyRepeat::No) {
        //     camera.view_scale /= 1.1;
        //     window_bounds = (
        //         (-camera.view_scale, -camera.view_scale),
        //         (camera.view_scale, camera.view_scale),
        //     );
        //     println!("new viewport scale = {}", camera.view_scale);
        // }
        // if window.is_key_pressed(Key::E, minifb::KeyRepeat::No) {
        //     camera.view_scale *= 1.1;
        //     window_bounds = (
        //         (-camera.view_scale, -camera.view_scale),
        //         (camera.view_scale, camera.view_scale),
        //     );
        //     println!("new viewport scale = {}", camera.view_scale);
        // }
        let scroll_value = window.get_scroll_wheel().map(|e| e.1).unwrap_or(0.0);
        if scroll_value != 0.0 {
            camera.view_scale /= 1.1.powi(scroll_value.signum() as i32);
        }

        if window.get_mouse_down(MouseButton::Middle) {
            // middle mouse is pressed
            let new_mouse_pos = window.get_mouse_pos(MouseMode::Pass).unwrap();
            let new_pos = Vec2::new(new_mouse_pos.0, new_mouse_pos.1);
            if let Some(last_pos) = last_mouse_pos {
                let diff = new_pos - last_pos;
                // actually orbit camera position
                camera.orbit(diff * camera_orbit_factor);
            }
            last_mouse_pos = Some(new_pos);
        } else {
            last_mouse_pos = None;
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
            std::mem::swap(&mut bodies, &mut swap);
            elapsed_sim_time += dt;
            elapsed_sim_time_ticker += dt;
            if elapsed_sim_time_ticker > 1.0 {
                elapsed_sim_time_ticker -= 1.0;

                for (idx, body) in bodies.iter().enumerate() {
                    if body_trails[idx].len() >= 99 {
                        let _ = body_trails[idx].pop_front();
                    }
                    body_trails[idx].push_back(body.into());
                }
            }

            dt = max_dt * (1.0 + speed_factor * max_speed_squared.0).recip();
            frame_min_dt = frame_min_dt.min(dt);
        }
        metaframe_min_dt = metaframe_min_dt.min(frame_min_dt);
        metaframe_elapsed_sim_time += frame_elapsed_sim_time;

        for body in body_trails.iter_mut().flatten() {
            // project body positions through camera and obtain pixel coords

            let (pu, pv) = camera.project(body.p);

            let (px, py) = (
                (pu * WINDOW_WIDTH as f32) as usize,
                (pv * WINDOW_HEIGHT as f32) as usize,
            );

            if px >= WINDOW_WIDTH || py >= WINDOW_HEIGHT {
                continue;
            }
            if body.radius == 0.0 {
                buffer.buffer[py * WINDOW_WIDTH + px] = body.adjusted_color();
            } else {
                blit_circle(&mut buffer, body.radius, px, py, body.adjusted_color());
            }
            body.brightness *= 0.99;
        }
        for body in bodies.iter() {
            // project body positions through camera and obtain pixel coords

            let (pu, pv) = camera.project(body.p);

            let (px, py) = (
                (pu * WINDOW_WIDTH as f32) as usize,
                (pv * WINDOW_HEIGHT as f32) as usize,
            );

            if px >= WINDOW_WIDTH || py >= WINDOW_HEIGHT {
                continue;
            }
            if body.radius == 0.0 {
                buffer.buffer[py * WINDOW_WIDTH + px] = body.color;
            } else {
                blit_circle(&mut buffer, body.radius, px, py, body.color);
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
