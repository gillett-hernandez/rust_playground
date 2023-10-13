// fn main() {
//     panic!("unimplemented. meant to be a simulation of fermat's principle of least time by using complex pseudo-quantum phase angles just like in QED to cancel out all but the shortest path directions, but it never worked");
// }

use ordered_float::OrderedFloat;

use lib::trace::flatland::{Point2, Vec2};
use lib::trace::tonemap::{sRGB, Tonemapper};
use lib::trace::{Bounds1D, Bounds2D, SingleWavelength};
use lib::{rgb_to_u32, Film};
use math::prelude::*;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rand::prelude::*;
use rayon::prelude::*;
use std::f32::consts::{PI, SQRT_2};
enum DrawMode {
    XiaolinWu,
    Midpoint,
    Bresenham,
}

fn line_line_intersection(p0: Point2, v0: Vec2, p1: Point2, v1: Vec2) -> Point2 {
    // P = p0 + v0 * t
    // P = p1 + v1 * t
    // p0 + v0 * t = p1 + v1 * t
    // t = (p1 - p0) / (v0 - v1)

    let top = p1 - p0;
    let bottom = v0 - v1;

    // println!("{:?}, {:?}", top, bottom);
    let [x0, y0]: [f32; 2] = p0.0.into();
    let [x1, y1]: [f32; 2] = p1.0.into();
    let [dx0, dy0]: [f32; 2] = v0.0.into();
    let [dx1, dy1]: [f32; 2] = v1.0.into();

    // y =
    // y = dy1/dx1 (x - x1) + y1

    //  x =   (dy0/dx0 * x0 - dy1/dx1 * x1 + y1 - y0) / (dy0/dx0 - dy1/dx1)
    let t0 = dy0 / dx0;
    let t1 = dy1 / dx1;
    let x = (t0 * x0 - t1 * x1 + y1 - y0) / (t0 - t1);
    let y = t1 * (x - x1) + y1;
    // y = dy1/dx1 (x - x1) + y1
    Point2::new(x, y)
}

fn basic_sampler(x: f32, _: Point2) -> f32 {
    // samples an angle for forward tracing through the swamp
    PI / 2.0 * (x - 0.5)
    // math::gaussianf32(random::<f32>(), 1.0, -PI / 4.0, PI / 2.0, PI / 2.0)
}

fn targeted_sampler(x: f32, pt: Point2) -> f32 {
    const TARGET: Point2 = Point2::new(50.0, 0.0);

    // vec from pt to target
    let diff = TARGET - pt;
    // get an angle from it
    let [dx, dy]: [f32; 2] = diff.normalized().0.into();
    let angle = dy.atan2(dx);
    PI / 4.0 * (x - 0.5) + angle
}

trait Sampler {
    fn sample(&mut self, x: f32, pt: Point2) -> f32;
}

struct LayerData {
    pub angle: f32,
}

struct LayeredReseviorSampler<const N: usize> {
    pub layer_data: [LayerData; N],
    pub layer_index: usize,
}

impl<const N: usize> LayeredReseviorSampler<N> {
    pub fn inner_generate_sample(x: f32, data: &LayerData) -> f32 {
        data.angle
    }
}

impl<const N: usize> Sampler for LayeredReseviorSampler<N> {
    fn sample(&mut self, x: f32, pt: Point2) -> f32 {
        let layer_data = &self.layer_data[self.layer_index];

        let a = Self::inner_generate_sample(x, layer_data);

        self.layer_index += 1;
        self.layer_index %= N;
        a
    }
}

struct BasicSampler<F>
where
    F: Fn(f32, Point2) -> f32,
{
    pub inner: F,
}

impl<F> Sampler for BasicSampler<F>
where
    F: Fn(f32, Point2) -> f32,
{
    fn sample(&mut self, x: f32, pt: Point2) -> f32 {
        (self.inner)(x, pt)
    }
}

fn run_sim<F, G>(n: usize, sampler: F) -> Vec<Vec<(Point2, Point2)>>
where
    F: Sync + Fn() -> G,
    G: Sync + Send + Sampler,
    // G: FnMut(f32, Point2) -> f32,
    // F: Sync + Send + Sampler<N>,
{
    // let freq = |i| freq;
    let start_point = Point2::new(-50.0, 0.0);
    let end_point = Point2::new(50.0, 0.0);
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut point = start_point;
            // let mut time = 0.0;

            let mut lines = Vec::new();
            // first marsh intersection
            let mut sampler = sampler();

            let angle = sampler.sample(random::<f32>(), point);
            let (dy, dx) = angle.sin_cos();

            // line from start to marsh beginning
            let new_point = line_line_intersection(
                point,
                Vec2::new(dx, dy),
                Point2::new(-25.0 * SQRT_2, 0.0),
                Vec2::new(1.0, 1.0),
            );
            // time += (new_point - point).norm() / 10.0;

            lines.push((point, new_point));
            point = new_point;

            // line in first marsh
            let angle = sampler.sample(random::<f32>(), point);
            let (dy, dx) = angle.sin_cos();
            let new_point = line_line_intersection(
                point,
                Vec2::new(dx, dy),
                Point2::new(-15.0 * SQRT_2, 0.0),
                Vec2::new(1.0, 1.0),
            );

            lines.push((point, new_point));
            // time += (new_point - point).norm() / 9.0;
            point = new_point;

            // line in second marsh
            let angle = sampler.sample(random::<f32>(), point);
            let (dy, dx) = angle.sin_cos();
            let new_point = line_line_intersection(
                point,
                Vec2::new(dx, dy),
                Point2::new(-5.0 * SQRT_2, 0.0),
                Vec2::new(1.0, 1.0),
            );

            lines.push((point, new_point));
            // time += (new_point - point).norm() / 8.0;
            point = new_point;

            // line in third marsh
            let angle = sampler.sample(random::<f32>(), point);
            let (dy, dx) = angle.sin_cos();
            let new_point = line_line_intersection(
                point,
                Vec2::new(dx, dy),
                Point2::new(5.0 * SQRT_2, 0.0),
                Vec2::new(1.0, 1.0),
            );

            lines.push((point, new_point));
            // time += (new_point - point).norm() / 7.0;
            point = new_point;

            // line in fourth marsh
            let angle = sampler.sample(random::<f32>(), point);
            let (dy, dx) = angle.sin_cos();
            let new_point = line_line_intersection(
                point,
                Vec2::new(dx, dy),
                Point2::new(15.0 * SQRT_2, 0.0),
                Vec2::new(1.0, 1.0),
            );

            lines.push((point, new_point));
            // time += (new_point - point).norm() / 6.0;
            point = new_point;

            // line in fifth marsh
            let angle = sampler.sample(random::<f32>(), point);
            let (dy, dx) = angle.sin_cos();
            let new_point = line_line_intersection(
                point,
                Vec2::new(dx, dy),
                Point2::new(25.0 * SQRT_2, 0.0),
                Vec2::new(1.0, 1.0),
            );

            lines.push((point, new_point));
            lines.push((new_point, end_point));
            // time += (new_point - point).norm() / 5.0;
            // point = new_point;

            // line to end point
            // time += (end_point - point).norm() / 10.0;

            // use time as a parameter for winding around a circle,
            // with time as the distance from the center as well?

            // let v = (time * freq(i)).sin_cos();
            // let v = Vec2::new(v.1, v.0);
            lines
        })
        .collect::<Vec<Vec<_>>>()
}

fn main() {
    // simulation based on the quantum mechanical explanation for snell's law.
    // i.e. all the varying paths to the end have different phases and they destructively interfere,
    // except for the ones concentrated around the least time path, because those have similar/correlated phases.
    // TODO: figure out why this doesn't converge on the solution, regardless of the "light" frequency. maybe because frequency should change as the density of the medium does?

    // real answer: [redacted]

    let n = 1;

    // let result = run_sim(n, 1.0);
    // .reduce(|| Vec2::ZERO, |accum, item| accum + item);

    if true {
        const WINDOW_HEIGHT: usize = 1080;
        const WINDOW_WIDTH: usize = 1080;
        let draw_mode = DrawMode::Midpoint;
        let mut window = Window::new(
            "Swamp Sim",
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
        // let height = film.height;
        let view_bounds = Bounds2D::new(Bounds1D::new(-50.0, 50.0), Bounds1D::new(-50.0, 50.0));
        let mut freq = 1.0;
        let mut best_angle = 0.0;
        let mut min_time = std::f32::INFINITY;
        let mut result = run_sim(n, || BasicSampler {
            inner: targeted_sampler,
        });
        // .reduce(|| Vec2::ZERO, |accum, item| accum + item);
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let keys = window.get_keys_pressed(KeyRepeat::Yes);
            for key in keys {
                match key {
                    Key::Space => {
                        freq *= 1.01;
                        println!("{}", freq);
                    }
                    Key::Tab => {
                        freq /= 1.01;
                        println!("{}", freq);
                    }
                    Key::Backspace => {
                        film.buffer.fill(XYZColor::BLACK);
                    }
                    _ => {}
                }
            }
            let mut lines = Vec::new();
            // let mut pt = Point2::ZERO;
            for line in result.iter().flatten() {
                // lines.push((
                //     pt,
                //     pt + *line / n as f32,
                //     XYZColor::from(SingleWavelength::new(550.0, 10.0.into())),
                // ));
                lines.push((
                    line.0,
                    line.1,
                    XYZColor::from(SingleWavelength::new(550.0, 10.0.into())),
                ));
                // pt += *line / n as f32;
            }
            for line in lines.drain(..) {
                let (px0, py0) = (
                    (WINDOW_WIDTH as f32 * (line.0.x() - view_bounds.x.lower)
                        / view_bounds.x.span()) as usize,
                    (WINDOW_HEIGHT as f32 * (line.0.y() - view_bounds.y.lower)
                        / view_bounds.y.span()) as usize,
                );
                let (px1, py1) = (
                    (WINDOW_WIDTH as f32 * (line.1.x() - view_bounds.x.lower)
                        / view_bounds.x.span()) as usize,
                    (WINDOW_HEIGHT as f32 * (line.1.y() - view_bounds.y.lower)
                        / view_bounds.y.span()) as usize,
                );

                let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
                if dx == 0 && dy == 0 {
                    if px0 as usize >= WINDOW_WIDTH || py0 as usize >= WINDOW_HEIGHT {
                        continue;
                    }
                    film.buffer[py0 as usize * width + px0 as usize] += line.2;
                    continue;
                }
                let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
                match draw_mode {
                    DrawMode::Midpoint => {
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
                    }
                    DrawMode::XiaolinWu => {
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
                    DrawMode::Bresenham => {
                        for (x, y) in line_drawing::Bresenham::new(
                            (px0 as isize, py0 as isize),
                            (px1 as isize, py1 as isize),
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
                    }
                }
            }
            let srgb_tonemapper = sRGB::new(&film, 1.0);
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

            result = run_sim(n, || BasicSampler {
                inner: targeted_sampler,
            });
            let local_min_time = result
                .iter()
                .cloned()
                .map(|path| {
                    OrderedFloat(path.iter().cloned().enumerate().fold(
                        0.0,
                        |accum, (i, (pt0, pt1))| {
                            accum
                                + (pt1 - pt0).norm()
                                    / match i {
                                        1 => 9.0,
                                        2 => 8.0,
                                        3 => 7.0,
                                        4 => 6.0,
                                        5 => 5.0,
                                        _ => 10.0,
                                    }
                        },
                    ))
                })
                .min()
                .unwrap();
            let average_time = result
                .iter()
                .cloned()
                .map(|path| {
                    path.iter()
                        .cloned()
                        .enumerate()
                        .fold(0.0, |accum, (i, (pt0, pt1))| {
                            accum
                                + (pt1 - pt0).norm()
                                    / match i {
                                        1 => 9.0,
                                        2 => 8.0,
                                        3 => 7.0,
                                        4 => 6.0,
                                        5 => 5.0,
                                        _ => 10.0,
                                    }
                        })
                })
                .sum::<f32>()
                / n as f32;
            if *local_min_time < min_time {
                min_time = *local_min_time;
            }
            println!("{}, {:?}", min_time, average_time);
        }
    }

    // println!(
    //     "{:?}",
    //     result
    //         .iter()
    //         .cloned()
    //         .reduce(|accum, i| accum + i)
    //         .unwrap()
    //         .norm()
    //         / n as f32
    // );
}
