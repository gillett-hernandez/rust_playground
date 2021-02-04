#![feature(slice_fill)]
extern crate minifb;

use std::f32::consts::SQRT_2;

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
    pub fn energy(&self) -> f32 {
        let speed = self.vx.hypot(self.vy);

        0.5 * self.mass * speed.powi(2)
    }
}

pub fn particle_particle_check(
    time: f32,
    i: usize,
    j: usize,
    particle: &Particle,
    other: &Particle,
) -> Option<(f32, f32, usize, Option<usize>)> {
    let sigma = particle.radius + other.radius;
    let dr = (other.x - particle.x, other.y - particle.y);
    let dv = (other.vx - particle.vx, other.vy - particle.vy);
    let dr2 = dr.0 * dr.0 + dr.1 * dr.1;
    let dv2 = dv.0 * dv.0 + dv.1 * dv.1;
    let dvdr = dv.0 * dr.0 + dv.1 * dr.1;
    let d = dvdr.powi(2) - dv2 * (dr2 - sigma.powi(2));
    if dvdr >= 0.0 || d < 0.0 || dv2 == 0.0 {
        None
    } else {
        let dt = -(dvdr + d.powf(0.5)) / dv2;
        // assert!(dt >= 0.0, "{:?}, {}, {}, {}", dt, dvdr, d.powf(0.5), dv2);
        if dt > 0.0 {
            Some((time, time + dt, i, Some(j)))
        } else {
            None
        }
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
        100.5,
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
        let mag = random::<f32>() * 0.0 + 0.4;
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

    let num_small_particles = particles.len() as f32 - 1.0;

    // let mut swap = particles.clone();

    // for i in 0..particles.len() {
    //     for j in 0..particles.len() {
    //         if j <= i {
    //             break;
    //         }
    //         // predict collision time for particles i and j.
    //         // and push to event queue.
    //     }
    // }

    let mut t = 0.0;
    let frame_dt = 6944.0 / 1000000.0;

    let mut events: Vec<(f32, f32, usize, Option<usize>)> = Vec::new();

    let length = particles.len();
    let mut new_events: Vec<(f32, f32, usize, Option<usize>)> = (0..length)
        .map(move |i| (0..length).map(move |j| (i, j)))
        .flatten()
        .filter_map(|(i, j)| {
            if i >= j {
                return None;
            }

            let particle = &particles[i];
            let other = &particles[j];

            particle_particle_check(t, i, j, particle, other)
        })
        .collect();
    for idx in 0..particles.len() {
        // horizontal wall (vy) first
        let particle = particles[idx];
        let event = particle_wall_check(t, idx, &particle);
        if let Some(e) = event {
            events.push(e);
        }
    }
    events.extend(new_events.drain(..));
    events.push((-1.0, t + frame_dt, 0, None));
    let mut mean_free_path = 0.0;
    let mut mean_pressure = 0.0;
    let moving_average_constant = 0.999;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        film.buffer.fill(0u32);

        // sort reversed
        events.sort_by_key(|e| OrderedFloat::<f32>(-e.1));

        let (event_time, collision_time, i, j) = loop {
            let (event_time, collision_time, i, j) = events.pop().unwrap();
            if event_time < 0.0 {
                break (event_time, collision_time, i, j);
            } else {
                // println!("{:?}, particle = {:?}", events.last(), particles[i]);
            }

            if particles[i].time > event_time {
                // particle has been updated since the event was added.
                // print!(
                //     "particle_time = {:?}, skipping event, \nnew event =",
                //     particles[i].time
                // );
                continue;
            }
            if j.is_some() && particles[j.unwrap()].time > event_time {
                // print!(
                //     "particle_j = {:?}, particle_time = {:?}, skipping event, \nnew event =",
                //     particles[j.unwrap()],
                //     particles[j.unwrap()].time
                // );
                continue;
            }
            break (event_time, collision_time, i, j);
        };

        particles
            .par_iter_mut()
            .for_each(|e| e.update(collision_time - t));
        t = collision_time;
        if event_time < 0.0 {
            events.push((-1.0, t + frame_dt, 0, None));

            let mut sum_energy = 0.0;
            let mut sum_speed = 0.0;
            let mut sum_sqr_speed = 0.0;
            let (mut min_speed, mut max_speed, mut min_energy, mut max_energy) = (
                std::f32::INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::INFINITY,
                std::f32::NEG_INFINITY,
            );

            let calculated_mean_free_path =
                (4.0 * (0.003 + 0.003 * 0.5) * num_small_particles * SQRT_2).recip();
            for particle in particles.iter_mut() {
                let (px, py) = (
                    (particle.x * WINDOW_WIDTH as f32) as usize,
                    (particle.y * WINDOW_HEIGHT as f32) as usize,
                );
                attempt_write(&mut film, px, py, 255u32 + 255u32 << 8);

                let mut r = 1;
                let pixel_radius = loop {
                    if r as f32 / film.width as f32 > particle.radius {
                        break r;
                    }
                    r += 1;
                };

                let v = particle.vx.hypot(particle.vy);
                // calculated_mean_free_path
                if window.is_key_pressed(Key::LeftBracket, KeyRepeat::Yes) {
                    particle.vx *= 0.9;
                    particle.vy *= 0.9;
                }
                if window.is_key_pressed(Key::RightBracket, KeyRepeat::Yes) {
                    particle.vx *= 1.1;
                    particle.vy *= 1.1;
                }
                let e = particle.energy();
                if particle.mass < 1.0 {
                    sum_speed += v;
                    sum_sqr_speed += v.powi(2);
                    sum_energy += e;
                }
                min_speed = min_speed.min(v);
                max_speed = max_speed.max(v);
                min_energy = min_energy.min(v);
                max_energy = max_energy.max(v);
                let c = triple_to_u32(hsv_to_rgb(
                    ((360.0 * (1.0 - (-e * 10.0).exp())) as usize + 100) % 360,
                    1.0,
                    1.0,
                ));
                blit_circle(&mut film, pixel_radius as f32, px, py, c);

                attempt_write(&mut film, px, py, c);
            }

            let temperature1 = mean_pressure / 4.0 * 1.0 / num_small_particles;
            let temperature2 =
                mean_free_path * 4.0 * (0.003 + 0.003 * 0.5) * SQRT_2 * mean_pressure / 4.0;

            // PV = NkT
            // k = PV / NT
            let k = mean_pressure / 4.0 * 1.0
                / num_small_particles
                / ((temperature1 + temperature2) / 2.0);
            let estimated_particles = mean_pressure / 4.0 * 1.0 / temperature2 / k;
            let calculated_pressure = sum_sqr_speed / 2.0 / 4.0;
            println!(
                "av:{:.5}, ae:{:.5}, mfp:{:.5}, mp:{:.5}, t:{:.5}, N:{:.5}, k:{:.5} --- c:{:.5}, c2: {:.5}",
                sum_speed / num_small_particles,
                sum_energy / num_small_particles,
                mean_free_path,
                mean_pressure / 4.0,
                temperature2,
                estimated_particles,
                k,
                mean_pressure / 4.0 / (sum_energy / num_small_particles) / 144.0,
                calculated_pressure
            );

            // print!(
            //     "avg, min, max speed: {:?}, {:?}, {:?}, ",
            //     sum_speed / particles.len() as f32,
            //     min_speed,
            //     max_speed
            // );
            // println!(
            //     "avg, min, max energy: {:?}, {:?}, {:?}",
            //     sum_energy / particles.len() as f32,
            //     min_energy,
            //     max_energy
            // );
            // std::mem::swap(&mut particles, &mut swap);
            window
                .update_with_buffer(&film.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
                .unwrap();
        } else {
            let particle = particles[i];
            let maybe_other = j.map(|e| particles[e]);
            let mut new_particle = particle.clone();
            let mut new_other = maybe_other.clone();
            match &mut new_other {
                Some(other) => {
                    // perform actual collision logic here
                    let sigma = particle.radius + other.radius;
                    let dr = (other.x - particle.x, other.y - particle.y);
                    // assert!(dr.0.hypot(dr.1) < sigma, "{:?}, {:?}", particle, other);
                    let dv = (other.vx - particle.vx, other.vy - particle.vy);
                    let dvdr = dv.0 * dr.0 + dv.1 * dr.1;
                    let j = 2.0 * other.mass * particle.mass * dvdr
                        / (sigma * (other.mass + particle.mass));
                    let jxy = (j * dr.0 / sigma, j * dr.1 / sigma);
                    // println!("performing actual collision. impulse is {} = 2.0 * {} * {} * {} / ({} * ({} + {}))", j, other.mass, particle.mass, dvdr, sigma, other.mass, particle.mass);
                    new_particle.vx += jxy.0 / particle.mass;
                    new_particle.vy += jxy.1 / particle.mass;

                    let old_other_e = other.energy();
                    other.vx -= jxy.0 / other.mass;
                    other.vy -= jxy.1 / other.mass;
                    let new_particle_v = new_particle.vx.hypot(new_particle.vy);
                    let old_particle_e = particle.energy();
                    let new_particle_e = new_particle.energy();
                    let other_v = other.vx.hypot(other.vy);
                    let new_other_e = other.energy();
                    let old_sum_energy = old_particle_e + old_other_e;
                    let new_sum_energy = new_particle_e + new_other_e;

                    // energy gain multiplier. scale velocity down by root of multiplier to preserve kinetic energy
                    let mult = (new_sum_energy / old_sum_energy).sqrt();
                    other.vx /= mult;
                    other.vy /= mult;
                    new_particle.vx /= mult;
                    new_particle.vy /= mult;

                    let new_sum_energy = new_particle.energy() + other.energy();

                    if (new_sum_energy - old_sum_energy).abs() > 0.0001 {
                        println!("conservation of kinetic energy violated, old: ({} + {} = {}), new: ({} + {} = {})", old_particle_e, old_other_e, old_other_e + old_particle_e, new_particle_e, new_other_e, new_particle_e + new_other_e);
                    }
                    if new_particle_v > 3.0 {
                        println!("warning, very fast moving particle");
                        // panic!();
                        // new_particle.vx /= new_particle_v;
                        // new_particle.vy /= new_particle_v;
                    }
                    if other_v > 3.0 {
                        println!("warning, very fast moving particle");
                        // panic!();
                        // other.vx /= other_v;
                        // other.vy /= other_v;
                    }
                    let this_free_path = (new_particle.vx * (t - new_particle.time))
                        .hypot(new_particle.vy * (t - new_particle.time));
                    let that_free_path =
                        (new_particle.vx * (t - other.time)).hypot(other.vy * (t - other.time));
                    if particle.mass < 1.0 {
                        mean_free_path = mean_free_path * moving_average_constant
                            + (1.0 - moving_average_constant) * this_free_path;
                    }
                    if other.mass < 1.0 {
                        mean_free_path = mean_free_path * moving_average_constant
                            + (1.0 - moving_average_constant) * that_free_path;
                    }
                    new_particle.time = t; // + std::f32::EPSILON;
                    other.time = t; // + std::f32::EPSILON;
                }
                None => {
                    new_particle.time = t;
                    // force = mass * acceleration
                    // pressure is force divided by area. since "area" in this simulation is 2d, it would be force divided by the "surface" of the container, or the perimeter of the container.
                    // acceleration or force and be calculated as an elastic collision.
                    // acceleration is such that the particles' momentum in the X direction changes from from -m * vx to m * vx. thus the momentum changes by 2 * m * vx in 1 frame. thus the acceleration is 2 * m * vx / dt. we will assume and scale such that dt = 1, however.
                    if particle.x - particle.radius < 0.001 {
                        // near left wall (x = 0)
                        // println!(
                        //     "bouncing horizontal, {}, {}, {}",
                        //     particle.x, particle.y, particle.radius
                        // );
                        let force = 2.0 * particle.mass * new_particle.vx.abs() / frame_dt;
                        mean_pressure = moving_average_constant * mean_pressure
                            + (1.0 - moving_average_constant) * force;
                        new_particle.vx *= -1.0;
                    } else if particle.x + particle.radius > 1.0 - 0.001 {
                        // near right wall (x = 1)
                        let force = 2.0 * particle.mass * new_particle.vx.abs() / frame_dt;
                        mean_pressure = moving_average_constant * mean_pressure
                            + (1.0 - moving_average_constant) * force;
                        new_particle.vx *= -1.0
                    }
                    if particle.y - particle.radius < 0.001 {
                        // near top wall (y = 0)

                        // println!(
                        //     "bouncing vertical, {}, {}, {}",
                        //     particle.x, particle.y, particle.radius
                        // );
                        let force = 2.0 * particle.mass * new_particle.vy.abs() / frame_dt;
                        mean_pressure = moving_average_constant * mean_pressure
                            + (1.0 - moving_average_constant) * force;
                        new_particle.vy *= -1.0;
                    } else if particle.y + particle.radius > 1.0 - 0.001 {
                        // near bottom wall (y = 1)
                        let force = 2.0 * particle.mass * new_particle.vy.abs() / frame_dt;
                        mean_pressure = moving_average_constant * mean_pressure
                            + (1.0 - moving_average_constant) * force;
                        new_particle.vy *= -1.0;
                    }
                }
            }
            // write new resolved collision data into i and j
            particles[i] = new_particle;
            if let Some(idx) = j {
                particles[idx] = new_other.unwrap();
            }

            // perform collision prediction for i and j with everything else.
            let indices = j.map_or(vec![i], |e| vec![i, e]);
            for idx in indices.iter() {
                let mut new_events: Vec<(f32, f32, usize, Option<usize>)> = (0..particles.len())
                    .into_par_iter()
                    .filter_map(|check_i| {
                        if check_i == i {
                            // remove double counting of this collision.
                            return None;
                        }
                        if check_i == *idx {
                            return None;
                        }
                        // check collision of check_i and i
                        let particle = &particles[*idx];
                        let other = &particles[check_i];
                        particle_particle_check(t, *idx, check_i, particle, other)
                    })
                    .collect();
                events.extend(new_events.drain(..));
            }

            // check collisions with walls for i and j
            for idx in indices {
                // horizontal wall (vy) first
                let particle = particles[idx];
                let event = particle_wall_check(t, idx, &particle);
                if let Some(e) = event {
                    events.push(e);
                }
            }
        }
    }
}
