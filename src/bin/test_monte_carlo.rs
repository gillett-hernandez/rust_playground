use std::ops::{Div, Mul};

fn refract(sin_i: f64, ior_1: f64, ior_2: f64) -> f64 {
    // let cos_i_2 = cos_i * cos_i;
    // let sin_i_2 = 1.0 - cos_i_2;
    // let sin_i = sin_i_2.sqrt();
    // n1 sin(theta_1) = n2 sin(theta_2)
    // sin_theta_2 = n1 sin_theta_1 / n2
    let sin_t = ior_1 * sin_i / ior_2;
    // let sin_t_2 = sin_t * sin_t;
    // let cos_t_2 = 1.0 - sin_t_2;
    // let cos_t = cos_t_2.sqrt();
    sin_t
}

fn cross_marsh(mut theta: f64) -> f64 {
    let mut time = 0.0;
    let mut point = (-50.0, 0.0);
    for (i, c) in vec![25, 15, 5, -5, -15, -25].iter().enumerate() {
        let tan_theta = theta.tan();
        let x = (point.1 - (*c as f64) * 2.0f64.sqrt() - point.0 * tan_theta) / (1.0 - tan_theta);
        let y = x + (*c as f64) * 2.0f64.sqrt();
        theta = refract(
            (theta + std::f64::consts::FRAC_PI_4).sin(),
            9.0 - i as f64,
            10.0 - i as f64,
        )
        .asin()
            - std::f64::consts::FRAC_PI_4;
        let chunk_distance = (x - point.0).hypot(y - point.1);

        time += chunk_distance / (10.0 - i as f64);

        // println!("{:?}", point);
        point = (x, y);
    }
    let (x, y) = (50.0, 0.0);
    // println!("{:?}", (x, y));
    let chunk_distance = (x - point.0).hypot(y - point.1);

    time += chunk_distance / (10.0);

    time
}

fn main() {
    // point A is at x=-50, y=0

    // line 1 => y = x+25*sqrt(2)
    // line 2 => y = x+15*sqrt(2)
    // line 3 => y = x+5*sqrt(2)
    // line 4 => y = x-5*sqrt(2)
    // line 5 => y = x-15*sqrt(2)
    // line 6 => y = x-25*sqrt(2)
    let mut angle = 0.0;
    let (mut time, mut last_time) = (cross_marsh(angle), 0.0);
    let mut heat = 0.01;
    let mut direction = 1.0;
    while (time - last_time).abs() > 1.0e10 {
        angle += heat * direction;
        let tmp = time;
        time = cross_marsh(angle);
        last_time = tmp;
        if time - last_time > 0.0 {
            // lost time;
            direction *= -1.0;
        }
        heat *= 0.999;
        // break;
    }
    println!("{} = {}", angle, time);
}
