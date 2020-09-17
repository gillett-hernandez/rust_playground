use rand::*;

fn simulation_step<F>(data: &mut Vec<u32>, birth_rate: u32, P: F)
where
    F: Fn(u32) -> f32,
{
    let last = data.last().unwrap();
    let len = data.len();
    let oldest = (*last as f32 * P(len as u32)) as u32;
    if oldest > 0 {
        data.push(oldest);
    }
    for i in (1..len).rev() {
        data[i] = (data[i - 1] as f32 * P(i as u32 - 1)) as u32;
    }
    data[0] = birth_rate;
}

fn survival_function(age: u32) -> f32 {
    match age {
        0 => 0.8,
        1..=8 => 1.0 - 0.04,
        9..=19 => 1.0 - 0.08,
        _ => 1.0 - 0.02,
    }
}

fn main() {
    let mut data: Vec<u32> = Vec::new();
    let mut birth_rate = 1000000;
    let change_rate = 0;
    data.push(birth_rate);
    for _ in 0..10000 {
        let change = (change_rate as f32) * (rand::random::<f32>() - 0.5);
        if birth_rate as f32 + change > 0.0 {
            birth_rate = (birth_rate as f32 + change) as u32;
        }
        simulation_step(&mut data, birth_rate, survival_function);
    }

    for (i, v) in data.iter().enumerate() {
        println!("{} {}", i, v);
    }
    println!("total population, {}", data.iter().sum::<u32>());
    println!(
        "total population older than 400, {}",
        data.iter()
            .enumerate()
            .filter(|(i, v)| *i > 400usize)
            .map(|(i, v)| v)
            .sum::<u32>()
    );
}
