use lib::rgb_to_u32;
use minifb::{Key, KeyRepeat, Scale, ScaleMode, Window, WindowOptions};
use ndarray as nd;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::ops::Neg;

use std::{
    error::Error,
    path::{Path, PathBuf},
};

use rustronomy_fits::prelude::*;

pub fn load_2d_image_from<P: AsRef<Path>>(path: P) -> Result<Array2<i16>, Box<dyn Error>> {
    let fits = Fits::open(&PathBuf::from(path.as_ref()))?;

    let data_array = match fits.get_hdu(0).unwrap().get_data() {
        Some(Extension::Image(img)) => img.clone().as_owned_i16_array()?,
        _ => panic!("not image data"),
    };
    // println!("{}", data_array);
    let img = data_array.into_dimensionality::<nd::Ix2>()?;
    Ok(img)
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let curdir = PathBuf::from(".").canonicalize();
    // println!("{}", curdir?.display());
    // let img_red = load_2d_image_from(PathBuf::from("data/images/AUS-2-CCDDISCONTINUED_2021-02-19T15-58-53_EtaCarina_Halpha_300s_ID204621_cal.fits"))?;
    let img_red = load_2d_image_from(PathBuf::from("data/images/CHI-1-CCDDISCONTINUED_2021-04-11T04-29-57_CarinaNebula_SII_600s_ID218164_cal.fits"))?;
    // let img_green = load_2d_image_from(PathBuf::from("data/images/AUS-2-CCDDISCONTINUED_2021-02-19T16-10-59_EtaCarina_OIII_300s_ID204623_cal.fits"))?;
    let img_green = load_2d_image_from(PathBuf::from("data/images/CHI-1-CCDDISCONTINUED_2021-04-22T00-05-38_CarinaNebula_Halpha_600s_ID220524_cal.fits"))?;
    // let img_blue = load_2d_image_from(PathBuf::from("data/images/AUS-2-CCDDISCONTINUED_2021-02-19T16-22-45_EtaCarina_SII_300s_ID204625_cal.fits"))?;
    let img_blue = load_2d_image_from(PathBuf::from("data/images/CHI-1-CCDDISCONTINUED_2021-04-22T00-38-27_CarinaNebula_OIII_600s_ID220527_cal.fits"))?;

    // need 16x supersampling
    // each pixel in the new window should tap 16 pixels in the original image

    // for now, just skip
    let width = img_red.shape()[0];
    let height = img_red.shape()[1];
    let mut window = Window::new(
        "Template",
        width,
        height,
        WindowOptions {
            scale: Scale::X1,
            // scale_mode: ScaleMode::Stretch,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    let mut buffer = vec![0u32; width * height];

    let min_max_avg_stddev = |data: &Array2<i16>| {
        let mut min = 0;
        let mut max = 0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let k = *data.first().unwrap();

        for &v in data.iter() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as f32 - k as f32;
            sum_sq += (v as f32 - k as f32).powi(2);
        }
        let avg = sum / data.len() as f32 + k as f32;
        (
            min,
            max,
            avg,
            ((sum_sq - avg * avg / data.len() as f32) / ((data.len() - 1) as f32)).sqrt(),
        )
    };
    // let (min_v, max_v, avg_v, stddev) = min_max_avg_stddev(&img_red);
    let mut factor = 1.0 / 18000.0;
    let mut offset = 32768.0;
    let mut intent;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        intent = 0.0;
        if window.is_key_down(Key::RightBracket) {
            intent += 1.0;
        }
        if window.is_key_down(Key::LeftBracket) {
            intent += -1.0;
        }
        if window.is_key_down(Key::F) {
            factor *= 1.1f32.powf(intent);
            if intent != 0.0 {
                println!("factor is now {factor}");
            }
        }
        if window.is_key_down(Key::O) {
            offset += intent;
            if intent != 0.0 {
                println!("offset is now {offset}");
            }
        }
        buffer.fill(0u32);

        buffer.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            // let data = img.as_slice().unwrap()[i];
            let px = i % width;
            let py = i / width;
            let r = *img_red
                .index_axis(Axis(0), px)
                .index_axis(Axis(0), py)
                .first()
                .unwrap();
            let g = *img_green
                .index_axis(Axis(0), px)
                .index_axis(Axis(0), py)
                .first()
                .unwrap();
            let b = *img_blue
                .index_axis(Axis(0), px)
                .index_axis(Axis(0), py)
                .first()
                .unwrap();

            let map_cast_div = |v: i16| ((v as u16) / 256) as u8;
            let map_abs = |v: i16| ((v.abs()) / 256) as u8;
            let map_factor = |v: i16| {
                (((v as f32 + offset) * factor).clamp(0.0, 1.0 - f32::EPSILON) * 256.0) as u8
            };
            let map_factor_exponential = |v: i16| {
                ((1.0 - ((v as f32 + offset) * factor).max(0.0f32).neg().exp()) * 256.0) as u8
            };

            let map = map_factor_exponential;
            *pixel = rgb_to_u32(map(r), map(g), map(b));
        });

        window.update_with_buffer(&buffer, width, height).unwrap();
    }
    Ok(())
}
