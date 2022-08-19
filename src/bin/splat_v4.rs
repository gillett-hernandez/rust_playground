use std::error::Error;

use clap::{arg, Command, Parser};
use crossbeam::thread;

use structopt::clap::crate_authors;

fn main() {
    let matches = Command::new("splatter_test")
        .version("0.1")
        .author(crate_authors!("\n"))
        .arg(arg!(--threads <VALUE>))
        .get_matches();

    let threads = matches.get_one::<usize>("threads").expect("required");

    rayon::ThreadPoolBuilder::new()
        .num_threads(*threads)
        .build_global()
        .unwrap();

    // same order of magnitude number of splatting threads and render threads
    // plus a thread for each combination of s,t within the given range.

    // solution should be scoped to a single render, as multi camera scenes are currently unimplemented. a solution for those should be similar to a solution for a single camera

    // let splat_handle = thread::spawn(move |scope| {});
    let _ = thread::scope(move |scope| {

        // TODO: before any of the below todos, determine if any of this is necessary, i.e. is a thread handling splats enough?
        // for N samples per pixel and B bounces per sample, the number of splats would likely be O(N*B * R) per thread, wher R is the pixel processing rate per thread for no-splat integrators.

        // let v = result.map_err(|e| panic!("error message {}", e.to_string()));
        // let v = result.expect(format!("message {}, {}", v1, v2).as_str());

        // TODO: check whether any of the prior splatting versions use a tiled approach
        // TODO: implement this. maybe use u32x4 and f32x16?
        // (cont) or batch to fill a decent size of a thread's L1 cache

        let splat_handle = scope.spawn(|_| {});
        // do "render", emitting tons of splats

        splat_handle.join().unwrap();
    });
}
