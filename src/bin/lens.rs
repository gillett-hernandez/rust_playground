#![feature(slice_fill)]
extern crate minifb;

use exr::block::chunk;
use lens::*;
use lib::*;

use math::XYZColor;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use rayon::prelude::*;

use lib::trace::*;
use tonemap::{sRGB, Tonemapper};

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(22 as usize)
        .build_global()
        .unwrap();
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

    let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
    let mut buffer = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;
    let height = film.height;

    let mut t = 0.0;
    let frame_dt = 6944.0 / 1000000.0;
    let spec = "35.0 20.0 bk7 1.5 54.0 15.0
    -35.0 1.73 air        15.0
    100000 3.00  iris    10.0
    1035.0 7.0 bk7 1.5 54.0 15.0
    -35.0 20 air        15.0";
    let (lenses, last_ior, last_vno) = parse_lenses_from(spec);
    let lens_assembly = LensAssembly::new(&lenses);

    let scene = get_scene("textures.toml").unwrap();

    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone()));
    }

    let mut aperture_size = 0.1;
    let mut narrow_factor = 0.01;
    let mut lens_zoom = 0.0;
    let mut film_position = -lens_assembly.total_thickness_at(lens_zoom);
    let mut wall_position = 100.0;
    let mut sensor_size = 35.0;
    let mut samples_per_iteration = 1usize;
    let mut clear = |film: &mut Film<XYZColor>| {
        film.buffer
            .par_iter_mut()
            .for_each(|e| *e = XYZColor::BLACK)
    };
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let srgb_tonemapper = sRGB::new(&film, 1.0);
        if window.is_key_pressed(Key::LeftBracket, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", aperture_size);
            aperture_size /= 1.01;
        }
        if window.is_key_pressed(Key::RightBracket, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", aperture_size);
            aperture_size *= 1.01;
        }
        if window.is_key_pressed(Key::N, KeyRepeat::Yes) {
            // clear(&mut film);
            println!("{:?}", narrow_factor);
            narrow_factor /= 1.1;
        }
        if window.is_key_pressed(Key::M, KeyRepeat::Yes) {
            // clear(&mut film);
            println!("{:?}", narrow_factor);
            narrow_factor *= 1.1;
        }
        if window.is_key_pressed(Key::O, KeyRepeat::Yes) {
            clear(&mut film);
            println!(
                "{:?}, {:?}",
                film_position,
                lens_assembly.total_thickness_at(lens_zoom)
            );
            film_position -= 1.0;
        }
        if window.is_key_pressed(Key::P, KeyRepeat::Yes) {
            clear(&mut film);
            println!(
                "{:?}, {:?}",
                film_position,
                lens_assembly.total_thickness_at(lens_zoom)
            );
            film_position += 1.0;
        }
        if window.is_key_pressed(Key::Q, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", wall_position);
            wall_position -= 1.0;
        }
        if window.is_key_pressed(Key::W, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", wall_position);
            wall_position += 1.0;
        }
        if window.is_key_pressed(Key::Z, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", sensor_size);
            sensor_size /= 1.1;
        }
        if window.is_key_pressed(Key::X, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", sensor_size);
            sensor_size *= 1.1;
        }
        if window.is_key_pressed(Key::K, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", lens_zoom);
            lens_zoom -= 0.01;
        }
        if window.is_key_pressed(Key::L, KeyRepeat::Yes) {
            clear(&mut film);
            println!("{:?}", lens_zoom);
            lens_zoom += 0.01;
        }

        if window.is_key_pressed(Key::Minus, KeyRepeat::Yes) {
            println!("{:?}", samples_per_iteration);
            if samples_per_iteration > 1 {
                samples_per_iteration -= 1;
            }
        }
        if window.is_key_pressed(Key::Equal, KeyRepeat::Yes) {
            println!("{:?}", samples_per_iteration);
            samples_per_iteration += 1;
        }

        let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
        for _ in 0..samples_per_iteration {
            film.buffer
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, pixel)| {
                    let px = i % width;
                    let py = i / width;

                    let (x, y, z) = (
                        (px as f32 / width as f32 - 0.5) * sensor_size,
                        (py as f32 / height as f32 - 0.5) * sensor_size,
                        film_position,
                    );

                    // choose direction somehow.
                    let s2d = Sample2D::new_random_sample();
                    let ray = Ray::new(
                        Point3::new(x, y, z),
                        (Vec3::Z
                            + Vec3::new(
                                (s2d.x - 0.5) * narrow_factor,
                                (s2d.y - 0.5) * narrow_factor,
                                0.0,
                            ))
                        .normalized(),
                    );

                    let mut energy = 0.0f32;
                    let lambda = wavelength_bounds.span() * Sample1D::new_random_sample().x
                        + wavelength_bounds.lower;
                    let result =
                        lens_assembly.trace_forward(lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                            (e.origin.x().hypot(e.origin.y()) > aperture_size, false)
                        });
                    if let Some(Output { ray, tau }) = result {
                        let t = (wall_position - ray.origin.z()) / ray.direction.z();
                        let point_at_10 = ray.point_at_parameter(t);
                        let uv = (
                            (point_at_10.x().abs() / 50.0) % 1.0,
                            (point_at_10.y().abs() / 50.0) % 1.0,
                        );

                        let m = textures[0].eval_at(lambda, uv);
                        energy += tau * m * 3.0;
                    }

                    *pixel += XYZColor::from_wavelength_and_energy(lambda, energy);
                });
        }
        buffer
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
            .update_with_buffer(&buffer.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
