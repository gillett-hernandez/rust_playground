// use rayon::iter::ParallelIterator;
// use rayon::prelude::*;

// use rust_playground as root;

use lib::*;

use packed_simd::{f32x4, Simd};

extern crate exr;
use exr::prelude::rgba_image::*;

// use std::io::Write;

// use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use rand::prelude::*;

use crossbeam::channel::{unbounded, Sender};

fn main() {
    // single light film with extremely high performance splatting.
    // need to do some type of locking and synchronization to avoid multiple threads accessing the same pixel at the same time.
    // TODO: incorporate pixel filtering into splatting, adjusting the strength of the splat based on where in the pixel the splat is
    // and potentially splatting across multiple pixels. though maybe

    let light_film: Arc<Mutex<Film<f32x4>>> = Arc::new(Mutex::new(Film::<f32x4>::new(
        2160,
        2160,
        f32x4::splat(0.0),
    )));

    let num_threads = 1;

    let (tx, rx) = unbounded();
    let rx = Arc::new(Mutex::new(rx));
    let mut join_handles = Vec::new();
    let film_width = light_film.lock().unwrap().width;
    let stop_signal = Arc::new(AtomicBool::new(false));
    let clone = light_film.clone();

    let mut txs: Vec<Sender<(usize, Simd<[f32; 4]>)>> = Vec::new();
    for _ in 0..num_threads {
        let arctex = clone.clone();
        let stop_clone = stop_signal.clone();
        let (tx, rx) = unbounded();
        unsafe {
            // the only thing guaranteeing no memory corruption occurs is the dispatch thread correctly sending pixels such that no collisions occur.
            let splatting_thread = thread::spawn(move || {
                let mut ptr = arctex.lock().unwrap();
                let ptr = ptr.buffer.as_mut_ptr();
                let mut ctr = 0;

                loop {
                    for (pidx, color) in rx.try_iter() {
                        *ptr.add(pidx) += color;
                        ctr += 1;
                    }
                    if let Ok((pidx, color)) = rx.recv_timeout(Duration::from_millis(200)) {
                        *ptr.add(pidx) += color;
                        ctr += 1;
                        continue; // skip exit branch because we found more data within the time limit.
                    }
                    if stop_clone.load(Ordering::Relaxed) && rx.is_empty() {
                        println!("stopping splatting thread after processing {} items", ctr);
                        break;
                    }
                }
            });
            txs.push(tx);
            join_handles.push(splatting_thread);
        }
    }

    let stop_clone = stop_signal.clone();
    let rx_clone = rx.clone();

    let dispatch_thread = thread::spawn(move || {
        let mut ctr = 0;
        let rx = rx_clone.lock().unwrap();
        loop {
            for (x, y, color) in rx.try_iter() {
                let pidx = y * film_width + x;
                let thread_id = pidx % num_threads;
                let tx: &Sender<(usize, Simd<[f32; 4]>)> = &txs[thread_id];
                tx.send((pidx, color)).unwrap();

                ctr += 1;
            }
            if let Ok((x, y, color)) = rx.recv_timeout(Duration::from_millis(200)) {
                let pidx = y * film_width + x;
                let thread_id = pidx % num_threads;
                let tx: &Sender<(usize, Simd<[f32; 4]>)> = &txs[thread_id];
                tx.send((pidx, color)).unwrap();
                ctr += 1;
                continue; // skip exit branch because we found more data within the time limit.
            }
            if stop_clone.load(Ordering::Relaxed) && rx.is_empty() {
                println!("stopping dispatch thread after processing {} items", ctr);
                break;
            }
        }
    });

    join_handles.push(dispatch_thread);

    let now = Instant::now();

    (0..100000000).into_par_iter().for_each(|_| {
        let (x, y) = (
            (random::<f32>() * 2160.0) as usize,
            (random::<f32>() * 2160.0) as usize,
        );
        // let color
        let sw = (
            390.0f32 + random::<f32>() * 370.0f32,
            random::<f32>() * 10.0f32,
        );
        let triplet = sw_to_triplet(sw);
        tx.send((x, y, triplet))
            .expect("send should have succeeded");
    });

    // for _ in 0..100000000 {
    //     let (x, y) = (
    //         (random::<f32>() * 2160.0) as usize,
    //         (random::<f32>() * 2160.0) as usize,
    //     );
    //     // let color
    //     let sw = (
    //         390.0f32 + random::<f32>() * 370.0f32,
    //         random::<f32>() * 10.0f32,
    //     );
    //     let triplet = sw_to_triplet(sw);
    //     tx.send((x, y, triplet))
    //         .expect("send should have succeeded");

    // }

    let duration1 = now.elapsed().as_millis();
    println!("{}ms for tx", duration1);
    stop_signal.store(true, Ordering::Relaxed);
    for join_handle in join_handles {
        join_handle.join().unwrap();
    }

    let duration2 = now.elapsed().as_millis();
    println!("{}ms for joining", duration2);

    {
        let film = light_film.lock().unwrap();
        let png_filename = "lightfilm.png";
        let exr_filename = "lightfilm.exr";

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
