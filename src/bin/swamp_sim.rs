fn main() {
    panic!("unimplemented. meant to be a simulation of fermat's principle of least time by using complex pseudo-quantum phase angles just like in QED to cancel out all but the shortest path directions, but it never worked");
}
// use std::f32::consts::SQRT_2;

// use lib::tonemap::{sRGB, Tonemapper};
// use lib::trace::{Bounds1D, Bounds2D, SingleWavelength};
// use lib::{
//     flatland::{Point2, Vec2},
//     PI,
// };
// use lib::{rgb_to_u32, Film};
// use math::prelude::*;
// use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
// use rand::prelude::*;
// use rayon::prelude::*;
// enum DrawMode {
//     XiaolinWu,
//     Midpoint,
//     Bresenham,
// }

// fn line_line_intersection(p0: Point2, v0: Vec2, p1: Point2, v1: Vec2) -> Point2 {
//     // P = p0 + v0 * t
//     // P = p1 + v1 * t
//     // p0 + v0 * t = p1 + v1 * t
//     // t = (p1 - p0) / (v0 - v1)

//     let top = p1 - p0;
//     let bottom = v0 - v1;

//     // println!("{:?}, {:?}", top, bottom);
//     let [x0, y0]: [f32; 2] = p0.0.into();
//     let [x1, y1]: [f32; 2] = p1.0.into();
//     let [dx0, dy0]: [f32; 2] = v0.0.into();
//     let [dx1, dy1]: [f32; 2] = v1.0.into();

//     // y =
//     // y = dy1/dx1 (x - x1) + y1

//     //  x =   (dy0/dx0 * x0 - dy1/dx1 * x1 + y1 - y0) / (dy0/dx0 - dy1/dx1)
//     let t0 = dy0 / dx0;
//     let t1 = dy1 / dx1;
//     let x = (t0 * x0 - t1 * x1 + y1 - y0) / (t0 - t1);
//     let y = t1 * (x - x1) + y1;
//     // y = dy1/dx1 (x - x1) + y1
//     Point2::new(x, y)
// }

// fn sampler() -> f32 {
//     // samples an angle for forward tracing through the swamp
//     random::<f32>() * PI / 2.0 - PI / 4.0
//     // math::gaussianf32(random::<f32>(), 1.0, -PI / 4.0, PI / 2.0, PI / 2.0)
// }

// fn run_sim(n: usize, freq: f32) -> Vec<Vec2> {
//     let freq = |i| freq;
//     let start_point = Point2::new(-50.0, 0.0);
//     let end_point = Point2::new(50.0, 0.0);
//     (0..n)
//         .into_par_iter()
//         .map(|i| {
//             let mut point = start_point;
//             let mut time = 0.0;
//             // first marsh intersection
//             let angle = sampler();
//             let (dy, dx) = angle.sin_cos();

//             // line from start to marsh beginning
//             let new_point = line_line_intersection(
//                 point,
//                 Vec2::new(dx, dy),
//                 Point2::new(-25.0 * SQRT_2, 0.0),
//                 Vec2::new(1.0, 1.0),
//             );
//             time += (new_point - point).norm() / 10.0;
//             point = new_point;

//             // line in first marsh
//             let angle = sampler();
//             let (dy, dx) = angle.sin_cos();
//             let new_point = line_line_intersection(
//                 point,
//                 Vec2::new(dx, dy),
//                 Point2::new(-15.0 * SQRT_2, 0.0),
//                 Vec2::new(1.0, 1.0),
//             );
//             time += (new_point - point).norm() / 9.0;
//             point = new_point;

//             // line in second marsh
//             let angle = sampler();
//             let (dy, dx) = angle.sin_cos();
//             let new_point = line_line_intersection(
//                 point,
//                 Vec2::new(dx, dy),
//                 Point2::new(-5.0 * SQRT_2, 0.0),
//                 Vec2::new(1.0, 1.0),
//             );
//             time += (new_point - point).norm() / 8.0;
//             point = new_point;

//             // line in third marsh
//             let angle = sampler();
//             let (dy, dx) = angle.sin_cos();
//             let new_point = line_line_intersection(
//                 point,
//                 Vec2::new(dx, dy),
//                 Point2::new(5.0 * SQRT_2, 0.0),
//                 Vec2::new(1.0, 1.0),
//             );
//             time += (new_point - point).norm() / 7.0;
//             point = new_point;

//             // line in fourth marsh
//             let angle = sampler();
//             let (dy, dx) = angle.sin_cos();
//             let new_point = line_line_intersection(
//                 point,
//                 Vec2::new(dx, dy),
//                 Point2::new(15.0 * SQRT_2, 0.0),
//                 Vec2::new(1.0, 1.0),
//             );
//             time += (new_point - point).norm() / 6.0;
//             point = new_point;

//             // line in fifth marsh
//             let angle = sampler();
//             let (dy, dx) = angle.sin_cos();
//             let new_point = line_line_intersection(
//                 point,
//                 Vec2::new(dx, dy),
//                 Point2::new(25.0 * SQRT_2, 0.0),
//                 Vec2::new(1.0, 1.0),
//             );
//             time += (new_point - point).norm() / 5.0;
//             point = new_point;

//             // line to end point
//             time += (end_point - point).norm() / 10.0;

//             // use time as a parameter for winding around a circle,
//             // with time as the distance from the center as well?

//             let v = (time * freq(i)).sin_cos();
//             let v = Vec2::new(v.1, v.0);
//             v // * time
//         })
//         .collect::<Vec<Vec2>>()
// }

// fn main() {
//     // simulation based on the quantum mechanical explanation for snell's law.
//     // i.e. all the varying path's to the end have different phases and they destructively interfere,
//     // except for the ones concentrated around the least time path, because those have similar/correlated phases.
//     // TODO: figure out why this doesn't converge on the solution, regardless of the "light" frequency. maybe because frequency should change as the density of the medium does?

//     let n = 10000;

//     // let result = run_sim(n, 1.0);
//     // .reduce(|| Vec2::ZERO, |accum, item| accum + item);

//     if true {
//         const WINDOW_HEIGHT: usize = 1080;
//         const WINDOW_WIDTH: usize = 1080;
//         let draw_mode = DrawMode::Midpoint;
//         let mut window = Window::new(
//             "2D Tracer",
//             WINDOW_WIDTH,
//             WINDOW_HEIGHT,
//             WindowOptions {
//                 scale: Scale::X1,
//                 ..WindowOptions::default()
//             },
//         )
//         .unwrap_or_else(|e| {
//             panic!("{}", e);
//         });

//         let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
//         let mut window_pixels = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);
//         window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
//         let width = film.width;
//         let height = film.height;
//         let view_bounds = Bounds2D::new(Bounds1D::new(-1.0, 1.0), Bounds1D::new(-1.0, 1.0));
//         let mut freq = 1.0;
//         let mut result = run_sim(n, freq);
//         // .reduce(|| Vec2::ZERO, |accum, item| accum + item);
//         while window.is_open() && !window.is_key_down(Key::Escape) {
//             let keys = window.get_keys_pressed(KeyRepeat::Yes);
//             for key in keys {
//                 match key {
//                     Key::Space => {
//                         freq *= 1.01;
//                         println!("{}", freq);
//                     }
//                     Key::Tab => {
//                         freq /= 1.01;
//                         println!("{}", freq);
//                     }
//                     Key::Backspace => {
//                         film.buffer.fill(XYZColor::BLACK);
//                     }
//                     _ => {}
//                 }
//             }
//             let mut lines = Vec::new();
//             let mut pt = Point2::ZERO;
//             for v in &result {
//                 lines.push((
//                     pt,
//                     pt + *v / n as f32,
//                     XYZColor::from(SingleWavelength::new(550.0, 10.0.into())),
//                 ));
//                 pt += *v / n as f32;
//             }
//             for line in lines.drain(..) {
//                 let (px0, py0) = (
//                     (WINDOW_WIDTH as f32 * (line.0.x() - view_bounds.x.lower)
//                         / view_bounds.x.span()) as usize,
//                     (WINDOW_HEIGHT as f32 * (line.0.y() - view_bounds.y.lower)
//                         / view_bounds.y.span()) as usize,
//                 );
//                 let (px1, py1) = (
//                     (WINDOW_WIDTH as f32 * (line.1.x() - view_bounds.x.lower)
//                         / view_bounds.x.span()) as usize,
//                     (WINDOW_HEIGHT as f32 * (line.1.y() - view_bounds.y.lower)
//                         / view_bounds.y.span()) as usize,
//                 );

//                 let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
//                 if dx == 0 && dy == 0 {
//                     if px0 as usize >= WINDOW_WIDTH || py0 as usize >= WINDOW_HEIGHT {
//                         continue;
//                     }
//                     film.buffer[py0 as usize * width + px0 as usize] += line.2;
//                     continue;
//                 }
//                 let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
//                 match draw_mode {
//                     DrawMode::Midpoint => {
//                         for (x, y) in line_drawing::Midpoint::<f32, isize>::new(
//                             (px0 as f32, py0 as f32),
//                             (px1 as f32, py1 as f32),
//                         ) {
//                             if x as usize >= WINDOW_WIDTH
//                                 || y as usize >= WINDOW_HEIGHT
//                                 || x < 0
//                                 || y < 0
//                             {
//                                 continue;
//                             }
//                             assert!(!b.is_nan(), "{} {}", dx, dy);
//                             film.buffer[y as usize * width + x as usize] += line.2 * b;
//                         }
//                     }
//                     DrawMode::XiaolinWu => {
//                         // let b = 1.0f32;
//                         for ((x, y), a) in line_drawing::XiaolinWu::<f32, isize>::new(
//                             (px0 as f32, py0 as f32),
//                             (px1 as f32, py1 as f32),
//                         ) {
//                             if x as usize >= WINDOW_WIDTH
//                                 || y as usize >= WINDOW_HEIGHT
//                                 || x < 0
//                                 || y < 0
//                             {
//                                 continue;
//                             }
//                             assert!(!b.is_nan(), "{} {}", dx, dy);
//                             film.buffer[y as usize * width + x as usize] += line.2 * b * a;
//                         }
//                     }
//                     DrawMode::Bresenham => {
//                         for (x, y) in line_drawing::Bresenham::new(
//                             (px0 as isize, py0 as isize),
//                             (px1 as isize, py1 as isize),
//                         ) {
//                             if x as usize >= WINDOW_WIDTH
//                                 || y as usize >= WINDOW_HEIGHT
//                                 || x < 0
//                                 || y < 0
//                             {
//                                 continue;
//                             }
//                             assert!(!b.is_nan(), "{} {}", dx, dy);
//                             film.buffer[y as usize * width + x as usize] += line.2 * b;
//                         }
//                     }
//                 }
//             }
//             let srgb_tonemapper = sRGB::new(&film, 1.0);
//             window_pixels
//                 .buffer
//                 .par_iter_mut()
//                 .enumerate()
//                 .for_each(|(pixel_idx, v)| {
//                     let y: usize = pixel_idx / width;
//                     let x: usize = pixel_idx - width * y;
//                     let (mapped, _linear) = srgb_tonemapper.map(&film, (x, y));
//                     let [r, g, b, _]: [f32; 4] = mapped.into();
//                     *v = rgb_to_u32((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8);
//                 });
//             window
//                 .update_with_buffer(&window_pixels.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
//                 .unwrap();

//             result = run_sim(n, freq);
//             // .reduce(|| Vec2::ZERO, |accum, item| accum + item);
//             println!(
//                 "{:?}",
//                 result
//                     .iter()
//                     .cloned()
//                     .reduce(|accum, i| accum + i)
//                     .unwrap()
//                     .norm()
//                     / n as f32
//             );
//         }
//     }

//     // println!(
//     //     "{:?}",
//     //     result
//     //         .iter()
//     //         .cloned()
//     //         .reduce(|accum, i| accum + i)
//     //         .unwrap()
//     //         .norm()
//     //         / n as f32
//     // );
// }
