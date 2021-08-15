use rand::prelude::*;

fn newtons<F, G>(x0: f32, f: F, df: G) -> f32
where
    F: Fn(f32) -> f32,
    G: Fn(f32) -> f32,
{
    let mut x = x0;
    loop {
        x = x - f(x) / df(x);
        if f(x).abs() < 0.000001 {
            return x;
        }
    }
}

fn main() {
    // direct calculation
    let timesave = 100.0f32;
    let timecost = -30.0f32;
    let max_trials = (timesave / (-timecost)).ceil();
    let S = |p: f32| 1.0 - timecost / (timecost - timesave) - (1.0 - p).powf(max_trials + 1.0);
    let dS = |p: f32| (max_trials + 1.0) * (1.0 - p).powf(max_trials);
    // let independent_consistency = newtons(0.0, S, dS);
    let independent_consistency = timecost / (timecost - timesave);
    println!("{:?}", independent_consistency);

    // simulate trial
    let trials_count = 10000000;
    let mut total_timesave = 0.0;
    for trial in 0..trials_count {
        // simulate gated trick attempt
        for attempt in 0..=(max_trials as usize) {
            let sample = random::<f32>();
            if sample <= independent_consistency {
                total_timesave += timesave;
                break;
            } else {
                total_timesave += timecost;
            }
        }
        // if all attempts fail, use backup strat.
    }

    println!(
        "average timesave for given trick consistency {:?}",
        total_timesave / trials_count as f32
    );
}
