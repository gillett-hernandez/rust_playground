#![feature(slice_fill)]
extern crate minifb;

use lib::*;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, Sample, SampleFormat, StreamInstant};
use crossbeam::unbounded;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rand::prelude::*;
use rayon::prelude::*;

fn read_input<T: Sample>(data: &[T], info: &cpal::InputCallbackInfo) {
    let mut time: f32 = 0.1;
    let mul = 1.0 / 48000.0;
    for sample in data.iter() {
        time += mul * 1.0;
        // *sample = Sample::from(&time.sin());
    }
}

fn main() {
    const WINDOW_WIDTH: usize = 800;
    const WINDOW_HEIGHT: usize = 800;
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

    // Limit to max ~144 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("no output device available");
    let output_device = host
        .default_output_device()
        .expect("no output device available");
    let mut supported_configs_range = output_device
        .supported_output_configs()
        .expect("error while querying configs");
    let supported_config = supported_configs_range
        .next()
        .expect("no supported config?!")
        .with_max_sample_rate();
    println!("{:?}", supported_config);
    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
    let sample_format = supported_config.sample_format();
    let config = supported_config.into();
    let (a, b) = unbounded::<f32>();
    let output_stream = match sample_format {
        SampleFormat::F32 => output_device.build_output_stream(
            &config,
            move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                let req_samples = data.len();
                let mut received_samples = 0;
                'outer: while received_samples < req_samples {
                    for sample in b.try_iter() {
                        if received_samples >= req_samples {
                            break 'outer;
                        }
                        data[received_samples] = sample;
                        received_samples += 1;
                    }
                }
            },
            err_fn,
        ),
        SampleFormat::I16 => panic!(),
        SampleFormat::U16 => panic!(),
    }
    .unwrap();

    // let input_stream = match sample_format {
    //     SampleFormat::F32 => input_device.build_input_stream(&config, read_input::<f32>, err_fn),
    //     SampleFormat::I16 => input_device.build_input_stream(&config, read_input::<i16>, err_fn),
    //     SampleFormat::U16 => input_device.build_input_stream(&config, read_input::<u16>, err_fn),
    // }
    // .unwrap();

    // input_stream.play().unwrap();
    output_stream.play().unwrap();

    let samples_per_frame = 48000.0 / 144.0;

    let mut t: f32 = 0.0;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        buffer.fill(0u32);

        for _ in 0..((2.0 * samples_per_frame) as usize) {
            t += 10.0 / samples_per_frame;
            a.try_send(t.sin()).unwrap();
        }
        // println!("frame");

        window
            .update_with_buffer(&buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
