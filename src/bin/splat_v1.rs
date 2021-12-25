// use rayon::iter::ParallelIterator;
// use rayon::prelude::*;

use lib::*;

extern crate exr;
use exr::prelude::rgba_image::*;

use std::io::Write;

use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use packed_simd::f32x4;
use rand::prelude::*;

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
