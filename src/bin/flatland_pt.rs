#![feature(slice_fill)]
extern crate line_drawing;
extern crate minifb;

use lib::curves::{load_ior_and_kappa, load_multiple_csv_rows};
use lib::spectral::{
    InterpolationMode, SpectralPowerDistributionFunction, BOUNDED_VISIBLE_RANGE,
    EXTENDED_VISIBLE_RANGE, SPD,
};
use lib::tonemap::{sRGB, Tonemapper};
use lib::trace::{Bounds1D, Bounds2D, SingleEnergy, SingleWavelength};
use lib::{rgb_to_u32, Film};

use math::XYZColor;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use rand::prelude::*;
use rayon::prelude::*;

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;

use lib::flatland::*;

fn main() {
    let threads = 1;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "2D Tracer",
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

    let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
    let mut window_pixels = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;
    let height = film.height;

    let frame_dt = 6944.0 / 1000000.0;

    let white = SPD::Linear {
        signal: vec![1.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let cornell_colors = load_multiple_csv_rows(
        "data/curves/physical/cornell.csv",
        3,
        InterpolationMode::Cubic,
        |x| x,
        |y| y,
    )
    .expect("data/curves/csv/cornell.csv was not formatted correctly");
    let mut iter = cornell_colors.iter();
    let (cornell_white, cornell_green, cornell_red) = (
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
    );
    let black = SPD::Linear {
        signal: vec![0.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let glass_eta = SPD::Cauchy { a: 1.5, b: 10000.0 };
    let (gold_ior, gold_kappa) =
        load_ior_and_kappa("data/curves/physical/gold.csv", |x: f32| x * 1000.0).unwrap();
    let scene = Scene::new(
        vec![
            Shape::Point {
                p: Point2::new(0.0, -0.09),
                material_id: 0,
            },
            Shape::Circle {
                radius: 100.0,
                center: Point2::new(0.0, 100.5),
                material_id: 1,
            },
            Shape::Circle {
                radius: 100.0,
                center: Point2::new(0.0, -100.5),
                material_id: 1,
            },
            Shape::Circle {
                radius: 100.0,
                center: Point2::new(100.5, 0.0),
                material_id: 2,
            },
            Shape::Circle {
                radius: 100.0,
                center: Point2::new(-100.5, 0.0),
                material_id: 3,
            },
            Shape::Circle {
                radius: 0.1,
                center: Point2::ORIGIN,
                material_id: 4,
            },
        ],
        vec![
            Material::DiffuseDirectionalLight {
                reflection_color: white.clone(),
                emission_color: white.clone(),
                direction: (30.0f32).to_radians(),
                radius: 0.4,
            },
            Material::Lambertian {
                color: cornell_white.clone(),
            },
            Material::Lambertian {
                color: cornell_green.clone(),
            },
            Material::Lambertian {
                color: cornell_red.clone(),
            },
            Material::GGX {
                eta: glass_eta,
                kappa: black,
                roughness: 0.01,
                permeable: true,
                eta_o: 1.01,
            },
            Material::GGX {
                eta: gold_ior,
                kappa: gold_kappa,
                roughness: 0.01,
                permeable: false,
                eta_o: 1.01,
            },
        ],
    );
    let view_bounds = Bounds2D::new(Bounds1D::new(-0.5, 0.5), Bounds1D::new(-0.5, 0.5));
    let (_box_width, _box_height) = (
        view_bounds.x.span() / width as f32,
        view_bounds.y.span() / height as f32,
    );
    let mut max_bounces = 16;
    let mut exposure_bias = 10.0;
    let mut new_rays_per_frame = 1000;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let keys = window.get_keys_pressed(KeyRepeat::Yes);
        let config_move = if window.is_key_down(Key::NumPadPlus) {
            1.0
        } else if window.is_key_down(Key::NumPadMinus) {
            -1.0
        } else {
            0.0
        };
        for key in keys.unwrap_or(vec![]) {
            match key {
                Key::E => {
                    exposure_bias += config_move;
                    println!("new exposure bias is {}", exposure_bias);
                }
                Key::B => {
                    max_bounces = (max_bounces + config_move as i32).max(0);
                    println!("new max bounces is {}", max_bounces);
                }
                Key::R => {
                    new_rays_per_frame =
                        (new_rays_per_frame as isize + config_move as isize).max(1) as usize;
                    println!("new rays per frame is {}", new_rays_per_frame);
                }
                Key::Space => {
                    film.buffer.fill(XYZColor::BLACK);
                }
                _ => {}
            }
        }

        // do tracing here.

        let mut rays: Vec<(Ray2D, f32, f32, bool)> = (0usize..new_rays_per_frame)
            .into_par_iter()
            .map(|_| {
                let lambda = BOUNDED_VISIBLE_RANGE.sample(random::<f32>());
                let light_shape = scene.sample_light();
                let point = light_shape.sample_surface();
                let light_mat = scene.get_material(light_shape.get_material_id());
                let (wo, energy) = light_mat.sample_le(lambda, point);
                (Ray2D::new(point, wo), lambda, energy, true)
            })
            .collect();

        let mut lines = Vec::new();
        for _ in 0..max_bounces {
            let mut new_lines: Vec<(Point2, Point2, XYZColor)> = rays
                .par_iter_mut()
                .filter_map(|(r, lambda, throughput, active)| {
                    if *active {
                        let intersection = scene.intersect(*r);
                        assert!(
                            !throughput.is_nan(),
                            "{:?} {:?}, {:?}",
                            r,
                            lambda,
                            throughput
                        );
                        let origin = r.origin;
                        let shading_color =
                            XYZColor::from(SingleWavelength::new(*lambda, (*throughput).into()));
                        let point = match intersection {
                            Some((point, normal, material_id)) => {
                                let mat = scene.get_material(material_id);
                                let frame = TangentFrame2D::from_normal(normal);
                                let wi = frame.to_local(&-r.direction);
                                let (wo, bsdf_f, bsdf_pdf) = mat.sample_bsdf(*lambda, wi);

                                if bsdf_pdf == 0.0 || bsdf_f == 0.0 {
                                    *active = false;
                                    point
                                } else {
                                    *throughput *= bsdf_f * wi.y().abs() / bsdf_pdf;
                                    let dir = frame.to_world(&wo).normalized();
                                    *r =
                                        Ray2D::new(point + normal * 0.00001 * wo.y().signum(), dir);
                                    point
                                }
                            }
                            None => {
                                // exit scene. compute clip bounds.
                                let mut min_t = f32::INFINITY;
                                match r.direction.x() {
                                    dx if dx > 0.0 => {
                                        min_t = min_t.min((view_bounds.x.upper - r.origin.x()) / dx)
                                    }
                                    dx if dx < 0.0 => {
                                        min_t = min_t.min((view_bounds.x.lower - r.origin.x()) / dx)
                                    }
                                    _ => {
                                        // up or down clip bounds will be computed in other match statement
                                    }
                                }
                                match r.direction.y() {
                                    dy if dy > 0.0 => {
                                        min_t = min_t.min((view_bounds.y.upper - r.origin.y()) / dy)
                                    }
                                    dy if dy < 0.0 => {
                                        min_t = min_t.min((view_bounds.y.lower - r.origin.y()) / dy)
                                    }
                                    _ => {
                                        // left or right clip bounds should have been computed in other match statement.
                                        assert!(r.direction.x() != 0.0);
                                    }
                                }
                                *active = false;
                                r.point_at(min_t)
                            }
                        };
                        Some((origin, point, shading_color))
                    } else {
                        None
                    }
                })
                .collect();
            lines.append(&mut new_lines);
        }

        for line in lines.drain(..) {
            let (px0, py0) = (
                (WINDOW_WIDTH as f32 * (line.0.x() - view_bounds.x.lower) / view_bounds.x.span())
                    as usize,
                (WINDOW_HEIGHT as f32 * (line.0.y() - view_bounds.y.lower) / view_bounds.y.span())
                    as usize,
            );
            let (px1, py1) = (
                (WINDOW_WIDTH as f32 * (line.1.x() - view_bounds.x.lower) / view_bounds.x.span())
                    as usize,
                (WINDOW_HEIGHT as f32 * (line.1.y() - view_bounds.y.lower) / view_bounds.y.span())
                    as usize,
            );

            let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
            if dx == 0 && dy == 0 {
                if px0 as usize >= WINDOW_WIDTH || py0 as usize >= WINDOW_HEIGHT {
                    continue;
                }
                film.buffer[py0 as usize * width + px0 as usize] += line.2;
                continue;
            }
            if true {
                if false {
                    let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
                    for (x, y) in line_drawing::Midpoint::<f32, isize>::new(
                        (px0 as f32, py0 as f32),
                        (px1 as f32, py1 as f32),
                    ) {
                        if x as usize >= WINDOW_WIDTH
                            || y as usize >= WINDOW_HEIGHT
                            || x < 0
                            || y < 0
                        {
                            continue;
                        }
                        assert!(!b.is_nan(), "{} {}", dx, dy);
                        film.buffer[y as usize * width + x as usize] += line.2 * b;
                    }
                } else {
                    let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
                    // let b = 1.0f32;
                    for ((x, y), a) in line_drawing::XiaolinWu::<f32, isize>::new(
                        (px0 as f32, py0 as f32),
                        (px1 as f32, py1 as f32),
                    ) {
                        if x as usize >= WINDOW_WIDTH
                            || y as usize >= WINDOW_HEIGHT
                            || x < 0
                            || y < 0
                        {
                            continue;
                        }
                        assert!(!b.is_nan(), "{} {}", dx, dy);
                        film.buffer[y as usize * width + x as usize] += line.2 * b * a;
                    }
                }
            } else {
                let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);

                for (x, y) in line_drawing::Bresenham::new(
                    (px0 as isize, py0 as isize),
                    (px1 as isize, py1 as isize),
                ) {
                    if x as usize >= WINDOW_WIDTH || y as usize >= WINDOW_HEIGHT || x < 0 || y < 0 {
                        continue;
                    }
                    assert!(!b.is_nan(), "{} {}", dx, dy);
                    film.buffer[y as usize * width + x as usize] += line.2 * b;
                }
            }
        }

        let srgb_tonemapper = sRGB::new(&film, exposure_bias);
        window_pixels
            .buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel_idx, v)| {
                let y: usize = pixel_idx / width;
                let x: usize = pixel_idx - width * y;
                let (mapped, _linear) = srgb_tonemapper.map(&film, (x, y));
                let [r, g, b, _]: [f32; 4] = mapped.into();
                *v = rgb_to_u32((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8);
            });
        window
            .update_with_buffer(&window_pixels.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
