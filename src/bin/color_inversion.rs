use lib::tonemap::*;
use lib::Film;
use lib::{curves::load_multiple_csv_rows, rgb_to_u32};
use math::spectral::Op;
use math::*;

use minifb::*;
use rayon::prelude::*;

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;

fn main() {
    let invert = |spd: SPD| SPD::Machine {
        seed: -1.0,
        list: vec![(Op::Mul, spd), (Op::Add, SPD::Const(1.0))],
    };

    let mut window = Window::new(
        "Color Tester",
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

    let wavelength_bounds = Bounds1D::new(400.0, 800.0);
    let spd = SPD::Exponential {
        signal: vec![
            (450.0, 20.0, 20.0, 0.3),
            (550.0, 20.0, 20.0, 0.6),
            (720.0, 20.0, 20.0, 0.5),
        ],
    };
    let cornell_colors = load_multiple_csv_rows(
        "data/curves/physical/cornell.csv",
        3,
        InterpolationMode::Cubic,
        |x| x,
        |y| y,
    )
    .expect("data/curves/physical/cornell.csv was not formatted correctly");
    let mut iter = cornell_colors.iter();
    let (cornell_white, cornell_green, cornell_red) = (
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
        iter.next().unwrap().clone(),
    );
    let xyz1 = cornell_green.convert_to_xyz(wavelength_bounds, 0.1, true);
    let xyz2 = invert(cornell_green).convert_to_xyz(wavelength_bounds, 0.1, true);

    println!("{:?}", xyz1);
    println!("{:?}", xyz2);

    let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
    let mut window_pixels = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;
    let height = film.height;
    let mut frame = 0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let srgb_tonemapper = sRGB::new(&film, 1.0);
        film.buffer.par_iter_mut().enumerate().for_each(|(_i, e)| {
            if frame < 30 {
                *e = xyz1;
            } else {
                *e = xyz2;
            }
        });
        frame += 1;
        frame %= 60;
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
