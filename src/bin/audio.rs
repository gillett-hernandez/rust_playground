#![feature(slice_fill, clamp)]
extern crate minifb;

use lib::*;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, Sample, SampleFormat, StreamInstant};
use crossbeam::unbounded;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use ordered_float::OrderedFloat;
use packed_simd::f32x16;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
pub struct Bounds1D {
    pub lower: f32,
    pub upper: f32,
}

impl Bounds1D {
    pub const fn new(lower: f32, upper: f32) -> Self {
        Bounds1D { lower, upper }
    }
    pub fn span(&self) -> f32 {
        self.upper - self.lower
    }

    pub fn contains(&self, value: &f32) -> bool {
        &self.lower <= value && value < &self.upper
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.max(other.lower), self.upper.min(other.upper))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }
}
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum InterpolationMode {
    Linear,
    Nearest,
    Cubic,
}

#[derive(Debug, Clone)]
pub struct LinearCurve {
    pub signal: Vec<f32>,
    pub bounds: Bounds1D,
    pub mode: InterpolationMode,
}
// Tabulated {
//     signal: Vec<(f32, f32)>,
//     mode: InterpolationMode,
// },
// Polynomial {
//     xoffset: f32,
//     coefficients: [f32; 8],
// },

impl Default for LinearCurve {
    fn default() -> Self {
        LinearCurve {
            signal: vec![0.0],
            bounds: Bounds1D::new(0.0, 1.0),
            mode: InterpolationMode::Linear,
        }
    }
}
pub trait Interpolation {
    fn evaluate(&self, lambda: f32) -> f32;
}

impl Interpolation for LinearCurve {
    fn evaluate(&self, time: f32) -> f32 {
        let LinearCurve {
            signal,
            bounds,
            mode,
        } = self;

        if !bounds.contains(&time) {
            return 0.0;
        }
        let step_size = bounds.span() / (signal.len() as f32);
        let index = ((time - bounds.lower) / step_size) as usize;
        let left = signal[index];
        let right = if index + 1 < signal.len() {
            signal[index + 1]
        } else {
            return signal[index];
        };
        let t = (time - (bounds.lower + index as f32 * step_size)) / step_size;
        // println!("t is {}", t);
        match mode {
            InterpolationMode::Linear => (1.0 - t) * left + t * right,
            InterpolationMode::Nearest => {
                if t < 0.5 {
                    left
                } else {
                    right
                }
            }
            InterpolationMode::Cubic => {
                let t2 = 2.0 * t;
                let one_sub_t = 1.0 - t;
                let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
                let h01 = t * t * (3.0 - t2);
                h00 * left + h01 * right
            }
        }
    }
}

/*
LinearCurve::Polynomial {
    xoffset,
    coefficients,
} => {
    let mut val = 0.0;
    let tmp_lambda = time - xoffset;
    for (i, &coef) in coefficients.iter().enumerate() {
        val += coef * tmp_lambda.powi(i as i32);
    }
    val
}
LinearCurve::Tabulated { signal, mode } => {
    // let result = signal.binary_search_by_key(lambda, |&(a, b)| a);
    let index = match signal
        .binary_search_by_key(&OrderedFloat::<f32>(time), |&(a, _b)| {
            OrderedFloat::<f32>(a)
        }) {
        Err(index) if index > 0 => index,
        Ok(index) | Err(index) => index,
    };
    if index == signal.len() {
        let left = signal[index - 1];
        return left.1;
    }
    let right = signal[index];
    let t;
    if index == 0 {
        return right.1;
    }
    let left = signal[index - 1];
    t = (time - left.0) / (right.0 - left.0);

    match mode {
        InterpolationMode::Linear => (1.0 - t) * left.1 + t * right.1,
        InterpolationMode::Nearest => {
            if t < 0.5 {
                left.1
            } else {
                right.1
            }
        }
        InterpolationMode::Cubic => {
            let t2 = 2.0 * t;
            let one_sub_t = 1.0 - t;
            let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
            let h01 = t * t * (3.0 - t2);
            h00 * left.1 + h01 * right.1
        }
    }
}*/

fn read_input<T: Sample>(data: &[T], info: &cpal::InputCallbackInfo) {
    let mut time: f32 = 0.1;
    let mul = 1.0 / 48000.0;
    for sample in data.iter() {
        time += mul * 1.0;
        // *sample = Sample::from(&time.sin());
    }
}

fn dft(signal: &Vec<f32>, sample_rate: usize, num_bins: usize) -> Option<Vec<f32>> {
    if signal.len() == 0 {
        return None;
    }
    let time_bounds = (signal.len() as f32) * 1.0 / (sample_rate as f32);
    let max_detectable_frequency = (sample_rate as f32) / 2.0;
    let min_detectable_frequency = 2.0 / time_bounds;
    let frequency_bounds = max_detectable_frequency - min_detectable_frequency;
    let octaves = frequency_bounds.log2();
    // let bin_size = octaves / (num_bins as f32);
    let mult =
        (max_detectable_frequency / min_detectable_frequency).powf((num_bins as f32).recip());

    println!(
        "min freq: {}, max freq: {}, octaves: {}, mult: {}",
        min_detectable_frequency, max_detectable_frequency, octaves, mult
    );

    let interpolated = LinearCurve {
        signal: signal.clone(),
        bounds: Bounds1D::new(0.0, time_bounds),
        mode: InterpolationMode::Cubic,
    };
    // println!("{:?}", interpolated);
    let mut bins: Vec<(f32, f32)> = Vec::new();
    let mut freq = min_detectable_frequency;
    for i in 0..num_bins {
        bins.push((0.0, 0.0));
        // time from peak to peak
        let wavelength = 1.0 / freq;
        let fraction = wavelength / interpolated.bounds.span();
        let num_samples = (fraction * interpolated.signal.len() as f32) as usize;
        if num_samples <= 2 {
            break;
        }
        // println!(
        //     "wavelength: {}, fraction: {}, num_samples: {}",
        //     wavelength, fraction, num_samples
        // );
        for sample_idx in 0..num_samples {
            for subsample_idx in 0..(fraction.recip().floor() as usize) {
                let t = sample_idx as f32 * wavelength * fraction / (num_samples as f32)
                    + wavelength * (subsample_idx) as f32;
                let f = interpolated.evaluate(t);
                bins[i].0 += f * (std::f32::consts::TAU * freq * t).cos();
                bins[i].1 += -f * (std::f32::consts::TAU * freq * t).sin();
            }
        }
        freq *= mult;
    }

    let mut amplitudes = Vec::new();
    for b in bins {
        amplitudes.push(b.0.hypot(b.1) / 256.0);
    }
    Some(amplitudes)
}

fn main() {
    const WINDOW_WIDTH: usize = 1024;
    const WINDOW_HEIGHT: usize = 1024;
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
    let mut buffer = vec![0u32; WINDOW_WIDTH * WINDOW_HEIGHT];

    let framerate = 30.0;
    window.limit_update_rate(Some(std::time::Duration::from_micros(
        (1_000_000.0 / framerate) as u64,
    )));

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("no output device available");
    let output_device = host
        .default_output_device()
        .expect("no output device available");
    let mut output_supported_configs_range = output_device
        .supported_output_configs()
        .expect("error while querying configs");
    let output_supported_config = output_supported_configs_range
        .next()
        .expect("no supported config?!")
        .with_max_sample_rate();
    let sample_rate = output_supported_config.sample_rate().0;
    println!("{:?}", sample_rate);
    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
    let sample_format = output_supported_config.sample_format();
    let output_config = output_supported_config.into();
    let (a, b) = unbounded::<f32x16>();
    let output_stream = match sample_format {
        SampleFormat::F32 => output_device.build_output_stream(
            &output_config,
            move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                let req_samples = data.len();
                let mut received_samples = 0;
                'outer: while received_samples < req_samples {
                    for sample in b.try_iter() {
                        let samples: [f32; 16] = sample.into();
                        for sub_idx in 0..16 {
                            if received_samples >= req_samples {
                                break 'outer;
                            }
                            data[received_samples] = samples[sub_idx];
                            received_samples += 1;
                        }
                    }
                }
            },
            err_fn,
        ),
        SampleFormat::I16 => panic!(),
        SampleFormat::U16 => panic!(),
    }
    .unwrap();

    let mut input_supported_configs_range = input_device
        .supported_input_configs()
        .expect("error while querying configs");
    let input_supported_config = input_supported_configs_range
        .next()
        .expect("no supported config?!")
        .with_max_sample_rate();

    let input_sample_rate = input_supported_config.sample_rate().0;
    let input_sample_format = input_supported_config.sample_format();
    let input_config = input_supported_config.into();
    let (ai, bi) = unbounded::<f32x16>();
    println!("input sample rate: {}", input_sample_rate);

    let input_stream = match input_sample_format {
        SampleFormat::F32 => input_device.build_input_stream(
            &input_config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                let req_samples = data.len();
                let mut received_samples = 0;
                'outer: while received_samples < req_samples {
                    let mut buffer: [f32; 16] = [0.0; 16];
                    for sub_idx in 0..16 {
                        if received_samples >= req_samples {
                            break 'outer;
                        }
                        buffer[sub_idx] = data[received_samples];
                        received_samples += 1;
                    }
                    ai.try_send(f32x16::from(buffer)).unwrap();
                }
            },
            err_fn,
        ),
        SampleFormat::I16 => panic!(),
        SampleFormat::U16 => panic!(),
    }
    .unwrap();

    input_stream.play().unwrap();
    output_stream.play().unwrap();

    let samples_per_frame = (sample_rate as f32) / framerate;
    println!("samples per frame: {}", samples_per_frame);

    let spread = f32x16::new(
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    );

    // let mut t: f32 = 0.0;
    // let frequency = std::f32::consts::PI * 200.0;
    let y_coordinate = 100 + 64 / 2;
    let y_span = 64;
    let freq_range = 22000;
    let num_bins = 512;
    let mut last_bins = Vec::new();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0u32);

        // output routine
        // for _chunk in 0..((samples_per_frame) as usize) {
        //     t += frequency * 16.0 / (sample_rate as f32);
        //     let signal = (spread / (sample_rate as f32) * frequency + f32x16::splat(t)).sin();
        //     a.try_send(signal).unwrap();
        //     // for i in 0..16 {
        //     //     let s = signal.extract(i);
        //     //     buffer[(y_coordinate as isize + (s * 64.0) as isize) as usize * WINDOW_WIDTH
        //     //         + chunk * 16
        //     //         + i] = 255u32;
        //     // }
        // }

        // input routine
        let mut signal: Vec<f32> = Vec::new();
        let mut chunk = 0;
        for input in bi.try_iter() {
            for i in 0..16 {
                let s = input.extract(i);
                signal.push(s);
                buffer[(y_coordinate as isize + (s * (y_span as f32)) as isize) as usize
                    * WINDOW_WIDTH
                    + chunk * 16
                    + i] = 255u32;
            }
            chunk += 1;
        }
        // perform DFT
        let amplitudes = dft(&signal, input_sample_rate as usize, num_bins);
        if let Some(amplitudes) = amplitudes {
            last_bins.push(amplitudes);
        }

        for (y, bin) in last_bins.iter().rev().enumerate() {
            // iterate from newest to oldest
            for (x, a) in bin.iter().enumerate() {
                buffer[(y + 300) * WINDOW_WIDTH + x] = (((1.0 - (-*a).exp()) * 255.0) as u32) << 8
            }
        }
        if last_bins.len() + 300 > WINDOW_HEIGHT {
            last_bins.remove(0);
        }
        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
