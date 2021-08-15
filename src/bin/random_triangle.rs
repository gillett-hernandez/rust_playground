use rand::prelude::*;

fn min(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}
fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

fn main() {
    let random = || rand::random::<f32>();
    let mut successes = 0;
    let mut attempts = 0;
    for _ in 0..100 {
        for _ in 0..100000 {
            let p0 = random(); // pos of first cut
            let p1 = random(); // pos of second cut
            let x = min(p0, p1); // length of first segment
            let y = max(p0, p1) - x; // length of second segment
                                     // length of third segment is 1.0 - x - y
            if x + y > 0.5 && 0.5 > y && 0.5 > x {
                successes += 1;
            }
            attempts += 1;
        }
        println!("{}", successes as f32 / attempts as f32);
    }
}
