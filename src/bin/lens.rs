#![feature(slice_fill)]
extern crate minifb;

use exr::block::chunk;
use lens::*;
use lib::*;

use math::XYZColor;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use ordered_float::OrderedFloat;
use rand::prelude::*;
use random::random_cosine_direction;
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
    // let spec = "35.0 20.0 bk7 1.5 54.0 15.0
    // -35.0 1.73 air        15.0
    // 100000 3.00  iris    10.0
    // 1035.0 7.0 bk7 1.5 54.0 15.0
    // -35.0 20 air        15.0"; // simple 2
    // let spec = "42.97		9.8			LAK9	1.6910	54.8	19.2
    // -115.33		2.1			LLF7	1.5486  45.4	19.2
    // 306.84		4.16		air			           	19.2
    // 100000		4.0			IRIS			      	15
    // -59.06		1.87		SF7		1.6398  34.6   	17.3
    // 40.93		10.64		air		       			17.3
    // 183.92		7.05		LAK9	1.6910  54.8   	16.5
    // -48.91		79.831		air						16.5"; // brendel tressar
    // let spec = "52.9     5.8  abbe   1.517  62 15
    // -41.4    1.5  abbe   1.576  54 15
    // 436.2    23.3 air              15
    // 100000.0 23.3 iris             10
    // 104.8    2.2  abbe   1.517  62 9
    // 36.8     0.7  air              9
    // 45.5     3.6  abbe   1.576  54 9
    // -149.5   50   air              9"; // petzval
    // let spec = "164.12		10.99				SF5			1.673	32.2	54
    // 559.28		0.23				air							54
    // 100.12		11.45				BAF10		1.67	47.1    51
    // 213.54		0.23				air							51
    // 58.04		22.95				LAK9		1.691	54.7	41
    // 2551		2.58				SF5			1.673	32.2	41
    // 32.39		15.66				air							27
    // 10000		15.00				IRIS						25.5
    // -40.42		2.74				SF15		1.699	30.1	25
    // 192.98		27.92				SK16		1.62	60.3	36
    // -55.53		0.23				air							36
    // 192.98		7.98				LAK9		1.691	54.7	35
    // -225.28		0.23				air							35
    // 175.1		8.48				LAK9		1.691	54.7	35
    // -203.54		55.742				air							35"; // double gauss angenioux
    let spec = "65.22    9.60  N-SSK8 1.5 50 24.0
    -62.03   4.20  N-SF10 1.5 50 24.0
    -1240.67 5.00  air           24.0
    100000  105.00  iris          20.0"; // lensbaby
    let (lenses, last_ior, last_vno) = parse_lenses_from(spec);
    let lens_assembly = LensAssembly::new(&lenses);

    let scene = get_scene("textures.toml").unwrap();

    let mut textures: Vec<TexStack> = Vec::new();
    for tex in scene.textures {
        textures.push(parse_texture_stack(tex.clone()));
    }

    let mut aperture_radius = lens_assembly.aperture_radius();
    let mut heat_bias = 0.1;
    let mut lens_zoom = 0.0;
    let mut film_position = -lens_assembly.total_thickness_at(lens_zoom);
    let mut wall_position = 240.0;
    let mut sensor_size = 35.0;
    let mut samples_per_iteration = 1usize;
    let mut total_samples = 0;
    let clear = |film: &mut Film<XYZColor>| {
        film.buffer
            .par_iter_mut()
            .for_each(|e| *e = XYZColor::BLACK)
    };
    let clear_direction_filter =
        |film: &mut Film<Vec3>| film.buffer.par_iter_mut().for_each(|e| *e = Vec3::Z);
    let mut direction_filter_film = Film::new(film.width, film.height, Vec3::Z);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let srgb_tonemapper = sRGB::new(&film, 1.0);
        if window.is_key_pressed(Key::LeftBracket, KeyRepeat::Yes) {
            // clear(&mut film);
            // clear_direction_filter(&mut direction_filter_film);
            // total_samples = 0;
            println!("{:?}", aperture_radius);
            aperture_radius /= 1.1;
        }
        if window.is_key_pressed(Key::RightBracket, KeyRepeat::Yes) {
            // clear(&mut film);
            // clear_direction_filter(&mut direction_filter_film);
            // total_samples = 0;
            println!("{:?}", aperture_radius);
            aperture_radius *= 1.1;
        }
        if window.is_key_pressed(Key::N, KeyRepeat::Yes) {
            // clear(&mut film);
            // clear_direction_filter(&mut direction_filter_film);
            // total_samples = 0;
            println!("{:?}", heat_bias);
            heat_bias /= 1.1;
        }
        if window.is_key_pressed(Key::M, KeyRepeat::Yes) {
            // clear(&mut film);
            // clear_direction_filter(&mut direction_filter_film);
            // total_samples = 0;
            println!("{:?}", heat_bias);
            heat_bias *= 1.1;
        }
        if window.is_key_pressed(Key::O, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!(
                "{:?}, {:?}",
                film_position,
                lens_assembly.total_thickness_at(lens_zoom)
            );
            film_position -= 1.0;
        }
        if window.is_key_pressed(Key::P, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!(
                "{:?}, {:?}",
                film_position,
                lens_assembly.total_thickness_at(lens_zoom)
            );
            film_position += 1.0;
        }
        if window.is_key_pressed(Key::Q, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!("{:?}", wall_position);
            wall_position -= 10.0;
        }
        if window.is_key_pressed(Key::W, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!("{:?}", wall_position);
            wall_position += 10.0;
        }
        if window.is_key_pressed(Key::Z, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!("{:?}", sensor_size);
            sensor_size /= 1.1;
        }
        if window.is_key_pressed(Key::X, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!("{:?}", sensor_size);
            sensor_size *= 1.1;
        }
        if window.is_key_pressed(Key::K, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
            println!("{:?}", lens_zoom);
            lens_zoom -= 0.01;
        }
        if window.is_key_pressed(Key::L, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
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
        if window.is_key_pressed(Key::Space, KeyRepeat::Yes) {
            clear(&mut film);
            clear_direction_filter(&mut direction_filter_film);
            total_samples = 0;
        }
        if window.is_key_pressed(Key::V, KeyRepeat::Yes) {
            println!("total samples: {}", total_samples);
        }

        let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
        total_samples += samples_per_iteration;

        film.buffer
            .par_iter_mut()
            .zip(direction_filter_film.buffer.par_iter_mut())
            .enumerate()
            .for_each(|(i, (pixel, direction))| {
                let px = i % width;
                let py = i / width;

                let mut heat = direction.w();

                for _ in 0..samples_per_iteration {
                    let mut attempts = 0;

                    let mut successes = 0;
                    let (x, y, z) = (
                        ((px as f32 + random::<f32>()) / width as f32 - 0.5) * sensor_size,
                        ((py as f32 + random::<f32>()) / height as f32 - 0.5) * sensor_size,
                        film_position,
                    );

                    // choose direction somehow.
                    let s2d = Sample2D::new_random_sample();
                    let frame =
                        TangentFrame::from_normal(Vec3::from_raw((*direction).0.replace(3, 0.0)));
                    let mut v = random_cosine_direction(s2d);
                    v += Vec3::Z * heat;
                    let v = frame.to_world(&v.normalized());
                    let ray = Ray::new(Point3::new(x, y, z), v.normalized());

                    let mut energy = 0.0f32;
                    let lambda = wavelength_bounds.span() * Sample1D::new_random_sample().x
                        + wavelength_bounds.lower;
                    let result =
                        lens_assembly.trace_forward(lens_zoom, &Input { ray, lambda }, 1.0, |e| {
                            (e.origin.x().hypot(e.origin.y()) > aperture_radius, false)
                        });
                    attempts += 1;
                    if let Some(Output {
                        ray: pupil_ray,
                        tau,
                    }) = result
                    {
                        *direction = ray.direction;
                        if heat < 10.0 {
                            heat *= 1.0 + heat_bias;
                            direction.0 = direction.0.replace(3, heat);
                        }
                        successes += 1;
                        let t = (wall_position - pupil_ray.origin.z()) / pupil_ray.direction.z();
                        let point_at_10 = pupil_ray.point_at_parameter(t);
                        let uv = (
                            (point_at_10.x().abs() / 50.0) % 1.0,
                            (point_at_10.y().abs() / 50.0) % 1.0,
                        );

                        let m = textures[0].eval_at(lambda, uv);
                        energy += tau * m * 3.0;
                        // *pixel = XYZColor::new(
                        //     (1.0 + direction.x()) * (1.0 + direction.w()),
                        //     (1.0 + direction.y()) * (1.0 + direction.w()),
                        //     (1.0 + direction.z()) * (1.0 + direction.w()),
                        // );
                        *pixel += XYZColor::from_wavelength_and_energy(lambda, energy);
                    } else {
                        if heat > 0.1 {
                            heat /= 1.0 + heat_bias;
                            direction.0 = direction.0.replace(3, heat);
                        }
                    }
                }
            });

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
