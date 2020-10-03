use rand::prelude::*;

fn simulate() -> usize {
    let mut times = [0usize; 10];

    loop {
        let imposter_1 = (random::<f32>() * 10.0) as usize;
        let mut imposter_2 = (random::<f32>() * 9.0) as usize;
        if imposter_2 == imposter_1 {
            imposter_2 += 1;
        }
        times[imposter_1] += 1;
        times[imposter_2] += 1;
        if times.iter().all(|v| *v > 0) {
            break;
        }
    }
    (times.iter().sum::<usize>() as f32 / 2.0) as usize
}

fn main() {
    let mut sum = 0;
    let n = 1000000;
    for _ in 0..n {
        sum += simulate();
    }
    println!("{}", sum as f32 / n as f32);
}
