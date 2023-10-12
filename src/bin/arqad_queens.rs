use std::collections::HashMap;

use rand::random;

const QUEEN_LOSS_PROBABILITY: f64 = 0.001;

fn experiment(iterations: usize) -> u32 {
    // returns number of queens lost

    let mut ct = 0;
    for _ in 0..iterations {
        let sample = random::<f64>();
        if sample < QUEEN_LOSS_PROBABILITY {
            ct += 1;
        }
    }
    ct
}

fn main() {
    let mut sum = 0;
    let mut histogram: HashMap<u32, u32> = HashMap::new();

    for iteration in 1.. {
        let queens_lost = experiment(2300);
        let e = histogram.entry(queens_lost).or_default();
        *e += 1;
        sum += queens_lost;
        if iteration % 100000 == 0 {
            println!("avg queens lost = {}", sum as f64 / iteration as f64);
            let mut p = 0.0;
            for (k, v) in histogram.iter() {
                if *k < 6 {
                    continue;
                }
                p += *v as f64 / iteration as f64;
            }
            println!("probability that 6 or more queens are lost: {}", p);
        }
    }
}
