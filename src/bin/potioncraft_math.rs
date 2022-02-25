extern crate line_drawing;
extern crate minifb;

use lib::flatland::{Point2, Vec2};
use lib::tonemap::{sRGB, Tonemapper};
use lib::trace::{Bounds1D, Bounds2D};
use lib::{rgb_to_u32, Film, SingleWavelength};

use math::XYZColor;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use packed_simd::f32x2;
use rand::prelude::*;
use rayon::prelude::*;

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;

enum DrawMode {
    XiaolinWu,
    Midpoint,
    Bresenham,
}

#[derive(Copy, Clone)]
enum Bezier {
    Linear {
        p0: Point2,
        p1: Point2,
    },
    Quadratic {
        p0: Point2,
        p1: Point2,
        p2: Point2,
    },
    Cubic {
        p0: Point2,
        p1: Point2,
        p2: Point2,
        p3: Point2,
    },
}

impl Bezier {
    pub fn beginning(&self) -> Point2 {
        self.eval(0.0)
    }
    pub fn end(&self) -> Point2 {
        self.eval(1.0)
    }
    pub fn eval(&self, t: f32) -> Point2 {
        let one_t = 1.0 - t;
        let one_t_2 = one_t * one_t;
        let one_t_3 = one_t_2 * one_t;
        let t_2 = t * t;
        let t_3 = t_2 * t;
        match self {
            &Bezier::Linear { p0, p1 } => p0 + t * (p1 - p0),
            &Bezier::Quadratic { p0, p1, p2 } => p1 + one_t_2 * (p0 - p1) + t * t * (p2 - p1),
            &Bezier::Cubic { p0, p1, p2, p3 } => Point2::from_raw(
                one_t_3 * p0.0 + 3.0 * one_t_2 * t * p1.0 + 3.0 * one_t * t_2 * p2.0 + t_3 * p3.0,
            ),
        }
    }
}

#[derive(Clone)]
struct Curve {
    pub list: Vec<Bezier>,
}

impl Curve {
    pub fn from_bezier_list(list: Vec<Bezier>) -> Self {
        for (bezier0, bezier1) in list.iter().zip(list.iter().skip(1)) {
            // beziers should be arranged roughly head to tail.
            assert!(
                (bezier0.end() - bezier1.beginning()).norm_squared() < 0.00001,
                "{:?} {:?}",
                bezier0.end(),
                bezier1.beginning()
            );
        }
        Curve { list }
    }

    pub fn eval(&self, t: f32) -> Point2 {
        let f32_len = self.list.len() as f32;
        let t = t.clamp(0.0, 1.0 - f32::EPSILON);
        let bin = (t * f32_len) as usize;
        let adjusted_t = (t - (bin as f32 / f32_len)) * f32_len;

        // let mut point = Point2::ZERO;
        // for i in 0..bin {
        //     point += Vec2::from_raw(self.list[i].end().0);
        // }
        // let point = self.list[bin].eval(adjusted_t);
        self.list[bin].eval(adjusted_t)
    }

    pub fn eval_vec(&self, t: f32) -> Vec2 {
        Vec2::from_raw(self.eval(t).0)
    }
}

struct Path {
    // path is made up of curve fragments
    // each curve fragment is a curve + some max "time" for each fragment.
    pub fragments: Vec<(Curve, f32)>,
    // we also need the current "time", and the "time" after which we switch to a new fragment.
    pub current_time: f32,
    pub current_base_position: Point2,
    pub current_base_time: f32,
    pub next_fragment_time: f32,
}

impl Path {
    pub fn new(base_position: Point2, curves: Vec<Curve>, terminators: Vec<f32>) -> Self {
        assert!(curves.len() > 0 && terminators.len() == curves.len());
        let first_time = *terminators.first().unwrap();
        Path {
            fragments: curves
                .iter()
                .cloned()
                .zip(terminators.iter())
                .map(|(e0, &e1)| (e0, e1))
                .collect::<Vec<(Curve, f32)>>(),
            current_time: 0.0,
            current_base_position: base_position,
            current_base_time: 0.0,
            next_fragment_time: first_time,
        }
    }
    pub fn eval(&self, time: f32) -> Point2 {
        assert!(time >= self.current_time);
        let mut pos = self.current_base_position;
        let mut offset = time - self.current_base_time;
        for (curve, terminator) in &self.fragments {
            if offset < *terminator {
                pos += curve.eval_vec(offset);
                break;
            } else {
                pos += curve.eval_vec(*terminator);
                offset -= *terminator;
            }
        }
        pos
    }
    pub fn current_position(&self) -> Point2 {
        let offset = self.current_time - self.current_base_time;
        let fragment = self.fragments.first().unwrap();
        self.current_base_position + fragment.0.eval_vec(offset)
    }
    pub fn advance(&mut self, delta: f32) {
        // advance the current time.
        // if we tick over the fragment border, update current_base_position and current_base_time
        // and remove the current fragment.
        // else, just update current_Time.
        if self.current_time + delta > self.next_fragment_time {
            if self.fragments.len() == 1 {
                self.current_time = self.next_fragment_time;
                return;
            }
            // calculate end of this fragment and add to current_base_position
            let frag = self.fragments.first().unwrap();
            self.current_base_position += frag.0.eval_vec(frag.1);
            self.current_base_time = self.next_fragment_time;
            let _ = self.fragments.remove(0);
            self.next_fragment_time = self.current_base_time + self.fragments[0].1;
            self.current_time += delta;
        } else {
            self.current_time += delta;
        }
    }
}

#[derive(Copy, Clone)]
struct Scout {
    pub pos: Point2,
}

impl Scout {
    pub fn new() -> Self {
        Scout { pos: Point2::ZERO }
    }

    pub fn update(&mut self, dv: Vec2) {
        loop {
            if random::<f32>() < 0.05 {
                // 40% chance of going towards the origin
                self.pos -= (self.pos - Point2::ZERO) * 0.06;
            } else {
                break;
            }
        }
        self.pos += dv;
    }

    pub fn reset(&mut self) {
        self.pos = Point2::ZERO;
    }
}

fn main() {
    let threads = 1;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "PotionCraft Path Simulator",
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

    // let frame_dt = 6944.0 / 1000000.0;

    let relative_view_bounds =
        Bounds2D::new(Bounds1D::new(-20.0, 20.0), Bounds1D::new(20.0, -20.0));
    let (_box_width, _box_height) = (
        relative_view_bounds.x.span() / width as f32,
        relative_view_bounds.y.span() / height as f32,
    );

    let max_ingredient_id = 4;

    let mut view_offset = Point2::new(0.0, 0.0);
    let mut draw_mode = DrawMode::Midpoint;
    let mut selected_ingredient = 0usize;

    let ingredients = vec![Curve::from_bezier_list(vec![
        Bezier::Linear {
            p0: Point2::new(0.0, 0.0),
            p1: Point2::new(-1.25, 1.0),
        },
        Bezier::Linear {
            p0: Point2::new(-1.25, 1.0),
            p1: Point2::new(-2.5, 0.0),
        },
        Bezier::Linear {
            p0: Point2::new(-2.5, 0.0),
            p1: Point2::new(-3.75, 1.0),
        },
        Bezier::Linear {
            p0: Point2::new(-3.75, 1.0),
            p1: Point2::new(-5.0, 0.0),
        },
    ])];

    let grind_levels = vec![(0.5f32, 1.0f32)];

    let mut path_curves = Vec::new();
    let mut path_terminators = Vec::new();
    let mut grind_level = 0.0f32;

    path_curves.push(ingredients[0].clone());
    path_terminators.push(1.0);

    let mut path = Path::new(Point2::ZERO, path_curves.clone(), path_terminators.clone());
    let mut pos = Point2::ZERO;

    let mut scouts = vec![Scout::new(); 1000];
    let mut scouts_clone = scouts.clone();

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
                Key::Tab => {
                    // change selected ingredient
                    selected_ingredient += 1;
                    if selected_ingredient > max_ingredient_id {
                        selected_ingredient = 0;
                    }
                }
                Key::Space => {
                    // add selected ingredient to the pot (path)
                    path_curves.push(ingredients[selected_ingredient].clone());
                    let grind_bounds = grind_levels[selected_ingredient];
                    path_terminators
                        .push(Bounds1D::new(grind_bounds.0, grind_bounds.1).lerp(grind_level));
                    grind_level = 0.0;
                    scouts.iter_mut().for_each(|e| e.reset());
                }
                Key::G => {
                    // grind selected ingredient.
                    grind_level += 0.01;
                    if grind_level > 1.0 {
                        grind_level = 1.0;
                    }
                }
                Key::Up => {
                    // move view Up
                    view_offset.0 = view_offset.0 + f32x2::new(0.0, 0.1);
                }
                Key::Down => {
                    // move view Down
                    view_offset.0 = view_offset.0 + f32x2::new(0.0, -0.1);
                }
                Key::Left => {
                    // move view Left
                    view_offset.0 = view_offset.0 + f32x2::new(-0.1, 0.0);
                }
                Key::Right => {
                    // move view Right
                    view_offset.0 = view_offset.0 + f32x2::new(0.1, 0.0);
                }
                Key::R => {
                    // reset all scouts
                    scouts.iter_mut().for_each(|e| e.reset());
                    scouts_clone = scouts.clone();
                }
                Key::Q => {
                    pos = Point2::ZERO;
                    path = Path::new(Point2::ZERO, path_curves.clone(), path_terminators.clone());
                    scouts.iter_mut().for_each(|e| e.reset());
                    scouts_clone = scouts.clone();
                    film.buffer.fill(XYZColor::BLACK);
                }
                _ => {}
            }
        }

        path.advance(0.01);
        let dv = path.current_position() - pos;
        println!("{:?}", scouts[0].pos);

        pos += dv;
        scouts.iter_mut().for_each(|e| e.update(dv));

        for (i, scout) in scouts.iter().enumerate() {
            // let dp = scout.pos - scouts_clone[i].pos;
            let (px0, py0) = (
                (WINDOW_WIDTH as f32 * (scout.pos.x() - relative_view_bounds.x.lower)
                    / relative_view_bounds.x.span()) as usize,
                (WINDOW_HEIGHT as f32 * (scout.pos.y() - relative_view_bounds.y.lower)
                    / relative_view_bounds.y.span()) as usize,
            );
            film.buffer[py0 as usize * width + px0 as usize] =
                XYZColor::from(SingleWavelength::new(550.0, 10.0.into()));
            if false {
                let line = (
                    scouts_clone[i].pos,
                    scout.pos,
                    XYZColor::from(SingleWavelength::new(550.0, 1.0.into())),
                );
                let (px0, py0) = (
                    (WINDOW_WIDTH as f32 * (line.0.x() - relative_view_bounds.x.lower)
                        / relative_view_bounds.x.span()) as usize,
                    (WINDOW_HEIGHT as f32 * (line.0.y() - relative_view_bounds.y.lower)
                        / relative_view_bounds.y.span()) as usize,
                );
                let (px1, py1) = (
                    (WINDOW_WIDTH as f32 * (line.1.x() - relative_view_bounds.x.lower)
                        / relative_view_bounds.x.span()) as usize,
                    (WINDOW_HEIGHT as f32 * (line.1.y() - relative_view_bounds.y.lower)
                        / relative_view_bounds.y.span()) as usize,
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
                            film.buffer[y as usize * width + x as usize] = line.2;
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
                            film.buffer[y as usize * width + x as usize] = line.2 * a;
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
                            film.buffer[y as usize * width + x as usize] = line.2;
                        }
                    }
                }
            }
        }
        scouts_clone = scouts.clone();

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
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_beziers() {
        let linear_bezier = Bezier::Linear {
            p0: Point2::new(0.0, 0.0),
            p1: Point2::new(-1.25, 1.0),
        };

        println!("linear bezier");
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let p = linear_bezier.eval(t);
            println!("{:?}", p);
        }

        let quadratic_bezier = Bezier::Quadratic {
            p0: Point2::new(1.0, 0.0),
            p1: Point2::new(0.0, 1.0),
            p2: Point2::new(-1.0, 0.0),
        };

        println!("quadratic bezier");
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let p = quadratic_bezier.eval(t);
            println!("{:?}", p);
        }

        let cubic_bezier = Bezier::Cubic {
            p0: Point2::new(1.0, 1.0),
            p1: Point2::new(-1.0, 1.0),
            p2: Point2::new(-1.0, -1.0),
            p3: Point2::new(1.0, -1.0),
        };

        println!("cubic bezier");
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let p = cubic_bezier.eval(t);
            println!("{:?}", p);
        }
    }

    #[test]
    fn test_curve() {
        let curve0 = Curve::from_bezier_list(vec![
            Bezier::Linear {
                p0: Point2::new(0.0, 0.0),
                p1: Point2::new(-1.25, 1.0),
            },
            Bezier::Linear {
                p0: Point2::new(-1.25, 1.0),
                p1: Point2::new(-2.5, 0.0),
            },
            Bezier::Linear {
                p0: Point2::new(-2.5, 0.0),
                p1: Point2::new(-3.75, 1.0),
            },
            Bezier::Linear {
                p0: Point2::new(-3.75, 1.0),
                p1: Point2::new(-5.0, 0.0),
            },
        ]);

        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let p = curve0.eval(t);
            println!("{:?}", p);
        }
    }

    #[test]
    fn test_path() {
        let curve0 = Curve::from_bezier_list(vec![
            Bezier::Linear {
                p0: Point2::new(0.0, 0.0),
                p1: Point2::new(-1.25, 1.0),
            },
            Bezier::Linear {
                p0: Point2::new(-1.25, 1.0),
                p1: Point2::new(-2.5, 0.0),
            },
            Bezier::Linear {
                p0: Point2::new(-2.5, 0.0),
                p1: Point2::new(-3.75, 1.0),
            },
            Bezier::Linear {
                p0: Point2::new(-3.75, 1.0),
                p1: Point2::new(-5.0, 0.0),
            },
        ]);

        let curve1 = Curve::from_bezier_list(vec![
            Bezier::Linear {
                p0: Point2::new(0.0, 0.0),
                p1: Point2::new(-1.25, 1.0),
            },
            Bezier::Linear {
                p0: Point2::new(-1.25, 1.0),
                p1: Point2::new(-2.5, 0.0),
            },
            Bezier::Linear {
                p0: Point2::new(-2.5, 0.0),
                p1: Point2::new(-3.75, 1.0),
            },
            Bezier::Linear {
                p0: Point2::new(-3.75, 1.0),
                p1: Point2::new(-5.0, 0.0),
            },
        ]);

        // let path = Path::new(Point2::ZERO, vec![curve0, curve1], vec![1.0, 1.0]);
        let mut path = Path::new(Point2::ZERO, vec![curve0, curve1], vec![0.75, 0.75]);

        for i in 0..=100 {
            let t = i as f32 / 50.0;
            let p = path.eval(t);
            println!("{:?}", p);
        }

        println!("tested eval, now testing advance and position");

        // let mut t = 0.0;
        loop {
            path.advance(0.01666);
            println!("{:?}", path.current_position());
            if path.current_time == path.next_fragment_time {
                break;
            }
        }
    }
}
