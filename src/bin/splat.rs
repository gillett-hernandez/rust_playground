// use rayon::iter::ParallelIterator;
// use rayon::prelude::*;

use lib::Film;

use nalgebra::{Matrix3, Vector3};
use packed_simd::f32x4;

use rand::seq::SliceRandom;
use rand::{thread_rng, RngCore};

extern crate exr;
use exr::prelude::rgba_image::*;

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::prelude::*;

pub fn gaussian(x: f64, alpha: f64, mu: f64, sigma1: f64, sigma2: f64) -> f64 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn x_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.056, 5998.0, 379.0, 310.0)
        + gaussian(angstroms.into(), 0.362, 4420.0, 160.0, 267.0)
        + gaussian(angstroms.into(), -0.065, 5011.0, 204.0, 262.0)) as f32
}

pub fn y_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 0.821, 5688.0, 469.0, 405.0)
        + gaussian(angstroms.into(), 0.286, 5309.0, 163.0, 311.0)) as f32
}

pub fn z_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.217, 4370.0, 118.0, 360.0)
        + gaussian(angstroms.into(), 0.681, 4590.0, 260.0, 138.0)) as f32
}

fn sw_to_triplet(sw: (f32, f32)) -> f32x4 {
    let ang = sw.0 * 10.0;
    f32x4::new(sw.1 * x_bar(ang), sw.1 * y_bar(ang), sw.1 * z_bar(ang), 0.0)
}

fn map(film: &Film<f32x4>, max_luminance: f32, pixel: (usize, usize)) -> (f32x4, f32x4) {
    let cie_xyz_color = film.at(pixel.0, pixel.1);
    let mut scaled_cie_xyz_color = cie_xyz_color / max_luminance;
    if !scaled_cie_xyz_color.is_finite().all() {
        scaled_cie_xyz_color = f32x4::splat(0.0);
    }

    let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
        3.24096994,
        -1.53738318,
        -0.49861076,
        -0.96924364,
        1.8759675,
        0.04155506,
        0.05563008,
        -0.20397696,
        1.05697151,
    );
    let [x, y, z, _]: [f32; 4] = scaled_cie_xyz_color.into();
    let intermediate = xyz_to_rgb * Vector3::new(x, y, z);

    let rgb_linear = f32x4::new(intermediate[0], intermediate[1], intermediate[2], 0.0);
    const S313: f32x4 = f32x4::splat(0.0031308);
    const S323_25: f32x4 = f32x4::splat(323.0 / 25.0);
    const S5_12: f32x4 = f32x4::splat(5.0 / 12.0);
    const S211: f32x4 = f32x4::splat(211.0);
    const S11: f32x4 = f32x4::splat(11.0);
    const S200: f32x4 = f32x4::splat(200.0);
    let srgb = (rgb_linear.lt(S313)).select(
        S323_25 * rgb_linear,
        (S211 * rgb_linear.powf(S5_12) - S11) / S200,
    );
    (srgb, rgb_linear)
}

fn main() {
    let light_films: Arc<Mutex<Vec<Film<f32x4>>>> = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..10 {
        light_films
            .lock()
            .unwrap()
            .push(Film::<f32x4>::new(512, 512, f32x4::splat(0.0)));
    }

    let (tx, rx) = mpsc::channel();
    // let done_splatting = Arc::new(AtomicBool::new(false));

    // let done_splatting_clone = done_splatting.clone();
    let light_films_ref = Arc::clone(&light_films);
    let splatting_thread = thread::spawn(move || {
        let films = &mut light_films_ref.lock().unwrap();
        loop {
            for v in rx.try_iter() {
                let (sw, (pixel, film_id)): ((f32, f32), ((f32, f32), usize)) = v;
                let film = &mut films[film_id];
                let triplet = sw_to_triplet(sw);
                let (x, y) = (
                    (pixel.0 * film.width as f32) as usize,
                    (pixel.1 * film.height as f32) as usize,
                );
                // let existing_triplet = film.buffer[y * film.width + x];
                film.buffer[y * film.width + x] += triplet;
                // thread::sleep(Duration::from_millis(10));
            }
            if let Ok(v) = rx.recv_timeout(Duration::from_millis(200)) {
                let (sw, (pixel, film_id)): ((f32, f32), ((f32, f32), usize)) = v;
                let film = &mut films[film_id];
                let triplet = sw_to_triplet(sw);
                let (x, y) = (
                    (pixel.0 * film.width as f32) as usize,
                    (pixel.1 * film.height as f32) as usize,
                );
                // let existing_triplet = film.buffer[y * film.width + x];
                film.buffer[y * film.width + x] += triplet;
            } else {
                break;
            }

            // if done_splatting_clone.load(Ordering::Acquire) {
            //     println!("broke because done splatting?");
            //     break;
            // }
        }
    });
    // let done_splatting_clone2 = done_splatting.clone();
    let mut handles = Vec::new();

    for _i in 0..20 {
        let tx1 = mpsc::Sender::clone(&tx);
        handles.push(thread::spawn(move || {
            for _ in 0..1000000 {
                let random_pixel: (f32, f32) = (random(), random());
                let sw = (
                    370.0f32 + random::<f32>() * 410.0f32,
                    random::<f32>() * 10.0f32,
                );
                let film_id = (random::<f32>() * 10.0) as usize;
                tx1.send((sw, (random_pixel, film_id))).unwrap();
            }

            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(b"*").unwrap();
            std::io::stdout().flush().expect("some error message");
            drop(tx1);
            // println!("*");
        }));
    }
    for handle in handles {
        handle.join().expect("thread panic!");
    }
    // done_splatting.clone().store(true, Ordering::SeqCst);
    if let Err(panic) = splatting_thread.join() {
        println!("panic occurred within thread: {:?}", panic);
    }

    for (id, film) in light_films.lock().unwrap().iter().enumerate() {
        let png_filename = format!("{}.png", id);
        let exr_filename = format!("{}.exr", id);

        let mut max_luminance = 0.0;
        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.at(x, y);
                let lum = color.extract(1);
                assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                if lum.is_nan() {
                    continue;
                }
                if lum > max_luminance {
                    // println!("max lum {} at ({}, {})", max_luminance, x, y);
                    max_luminance = lum;
                }
            }
        }
        let generate_pixels = |position: Vec2<usize>| {
            let (_mapped, linear) = map(
                &film,
                max_luminance,
                (position.x() as usize, position.y() as usize),
            );
            let [r, g, b, _]: [f32; 4] = linear.into();
            Pixel::rgb(r, g, b)
        };

        let image_info = ImageInfo::rgb(
            (film.width, film.height), // pixel resolution
            SampleType::F16,           // convert the generated f32 values to f16 while writing
        );

        image_info
            .write_pixels_to_file(
                exr_filename,
                write_options::high(), // higher speed, but higher memory usage
                &generate_pixels,      // pass our pixel generator
            )
            .unwrap();

        let mut img: image::RgbImage =
            image::ImageBuffer::new(film.width as u32, film.height as u32);

        for (x, y, pixel) in img.enumerate_pixels_mut() {
            //apply tonemap here

            let (mapped, _linear) = map(&film, max_luminance, (x as usize, y as usize));

            let [r, g, b, _]: [f32; 4] = mapped.into();

            *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
        }
        println!("saving image...");
        img.save(png_filename).unwrap();
    }
}
