#![feature(slice_fill)]
extern crate minifb;

use lib::*;

use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub time: f32,
    pub mass: f32,
    pub radius: f32,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    // pub ax: f32,
    // pub ay: f32,
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
            // ax,
            // ay,
            // spin: 0.0,
        }
    }
    pub fn normalize(&mut self) {
        let scale = self.vx.hypot(self.vy);
        self.vx = self.vx / scale;
        self.vy = self.vy / scale;
    }
    pub fn update(&mut self, dt: f32) {
        // self.vx += self.ax * dt;
        // self.vy += self.ay * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;
    }
}
pub fn do_collision_check(
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
    // let dt = 6944.0 / 1000000.0;

    let mut particles: Vec<Particle> = Vec::new();
    let num_particles = 200;

    for i in 0..num_particles {
        let phi = random::<f32>() * std::f32::consts::TAU;
        let mag = random::<f32>() * 1.0 + 0.2;
        let r = random::<f32>() * 0.000 + 0.004;
        let particle = loop {
            let particle = Particle::new(
                0.5,
                r,
                random::<f32>() * (1.0 - 2.0 * r) + r,
                random::<f32>() * (1.0 - 2.0 * r) + r,
                mag * phi.cos(),
                mag * phi.sin(),
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

    let mut events: Vec<(f32, f32, usize, Option<usize>)> = Vec::new();

    let mut new_events: Vec<(f32, f32, usize, Option<usize>)> = (0..num_particles)
        .map(move |i| (0..num_particles).map(move |j| (i, j)))
        .flatten()
        .filter_map(|(i, j)| {
            if i >= j {
                return None;
            }

            let particle = &particles[i];
            let other = &particles[j];

            do_collision_check(t, i, j, particle, other)
        })
        .collect();
    for idx in 0..num_particles {
        // horizontal wall (vy) first
        let particle = particles[idx];
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
        let particle = particles[idx];
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
            events.push((t, t + min, idx, None));
        }
    }
    events.extend(new_events.drain(..));
    events.push((-1.0, t + frame_dt, 0, None));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0u32);

        // sort reversed
        events.sort_by_key(|e| OrderedFloat::<f32>(-e.1));

        let (event_time, collision_time, i, j) = loop {
            let (event_time, collision_time, i, j) = events.pop().unwrap();
            if event_time < 0.0 {
                break (event_time, collision_time, i, j);
            } else {
                println!("{:?}, particle = {:?}", events.last(), particles[i]);
            }

            if particles[i].time > event_time {
                // particle has been updated since the event was added.
                print!(
                    "particle_time = {:?}, skipping event, \nnew event =",
                    particles[i].time
                );
                continue;
            }
            if j.is_some() && particles[j.unwrap()].time > event_time {
                print!(
                    "particle_j = {:?}, particle_time = {:?}, skipping event, \nnew event =",
                    particles[j.unwrap()],
                    particles[j.unwrap()].time
                );
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
            for particle in particles.iter() {
                let (px, py) = (
                    (particle.x * WINDOW_WIDTH as f32) as usize,
                    (particle.y * WINDOW_HEIGHT as f32) as usize,
                );
                let mut attempt_write = |px: usize, py: usize, c: u32| {
                    if py * WINDOW_WIDTH + px >= buffer.len() {
                        return;
                    }
                    buffer[py * WINDOW_WIDTH + px] = c;
                };
                attempt_write(px, py, 255u32 + 255u32 << 8);
                let pixel_diameter = 1.0 / WINDOW_HEIGHT as f32;
                let mut r = 1;
                let pixel_radius = loop {
                    if r as f32 * pixel_diameter > particle.radius {
                        break r;
                    }
                    r += 1;
                };
                let approx_pixel_circumference = pixel_radius as f32 * std::f32::consts::TAU;
                for phi in 0..(approx_pixel_circumference as usize) {
                    let (new_px, new_py) = (
                        (px as f32 * pixel_diameter
                            + r as f32
                                * pixel_diameter
                                * (phi as f32 * std::f32::consts::TAU
                                    / approx_pixel_circumference)
                                    .cos())
                            / pixel_diameter,
                        (py as f32 * pixel_diameter
                            + r as f32
                                * pixel_diameter
                                * (phi as f32 * std::f32::consts::TAU
                                    / approx_pixel_circumference)
                                    .sin())
                            / pixel_diameter,
                    );
                    attempt_write(new_px as usize, new_py as usize, 255u32 + 255u32 << 8);
                }
                attempt_write(px, py, 255u32 + 255u32 << 8);
            }
            // std::mem::swap(&mut particles, &mut swap);
            window
                .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
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
                    println!("performing actual collision. impulse is {} = 2.0 * {} * {} * {} / ({} * ({} + {}))", j, other.mass, particle.mass, dvdr, sigma, other.mass, particle.mass);
                    new_particle.vx += jxy.0 / particle.mass;
                    new_particle.vy += jxy.1 / particle.mass;
                    other.vx -= jxy.0 / other.mass;
                    other.vy -= jxy.1 / other.mass;
                    let new_particle_v = new_particle.vx.hypot(new_particle.vy);
                    let other_v = other.vx.hypot(other.vy);
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
                    new_particle.time = t; // + std::f32::EPSILON;
                    other.time = t; // + std::f32::EPSILON;
                }
                None => {
                    new_particle.time = t;
                    if particle.x - particle.radius < 0.001
                        || particle.x + particle.radius > 1.0 - 0.001
                    {
                        // near right and left walls
                        println!(
                            "bouncing horizontal, {}, {}, {}",
                            particle.x, particle.y, particle.radius
                        );
                        new_particle.vx *= -1.0;
                    }
                    if particle.y - particle.radius < 0.001
                        || particle.y + particle.radius > 1.0 - 0.001
                    {
                        // near top and bottom walls.
                        println!(
                            "bouncing vertical, {}, {}, {}",
                            particle.x, particle.y, particle.radius
                        );
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
                let mut new_events: Vec<(f32, f32, usize, Option<usize>)> = (0..num_particles)
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
                        do_collision_check(t, *idx, check_i, particle, other)
                    })
                    .collect();
                events.extend(new_events.drain(..));
            }

            // check collisions with walls for i and j
            for idx in indices {
                // horizontal wall (vy) first
                let particle = particles[idx];
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
                let particle = particles[idx];
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
                    events.push((t, t + min, idx, None));
                }
            }
        }
    }
}
