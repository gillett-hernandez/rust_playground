// use rand::prelude::*;

fn simulate_daily_interest<F>(mut balance: f32, daily_rate: f32, payment_policy: F) -> (f32, f32)
where
    F: Fn(f32, usize) -> f32,
{
    let mut total_interest_payed = 0.0;

    let mut days = 1usize;

    loop {
        balance -= payment_policy(balance, days);
        if balance <= 0.0 {
            break;
        }
        // apply daily interest
        total_interest_payed += balance * daily_rate;
        balance += balance * daily_rate;
        // increment day counter
        days += 1;
    }
    (days as f32, total_interest_payed)
}

fn simulate_monthly_interest<F>(
    mut balance: f32,
    monthly_rate: f32,
    payment_policy: F,
) -> (f32, f32)
where
    F: Fn(f32, usize) -> f32,
{
    let mut total_interest_payed = 0.0;

    let mut months = 1usize;

    loop {
        balance -= payment_policy(balance, months);
        if balance <= 0.0 {
            break;
        }

        // apply monthly interest
        total_interest_payed += balance * monthly_rate;
        balance += balance * monthly_rate;
        // increment month counter
        months += 1;
    }
    (months as f32, total_interest_payed)
}

fn main() {
    let balance: f32 = 200000.00;
    let apr: f32 = 0.02875;

    let daily_rate = apr / 365.24;
    let monthly_rate = apr / 12.0;
    let monthly_payment: f32 = 664.0;
    let avg_month_period = 365.24 / 12.0;

    let minimum_monthly_payment = balance * daily_rate * avg_month_period;

    let daily_payments = true;
    let daily_interest = true;

    // daily
    // 249.16154 - 22.177198 = 226.98434 on day 19937. total interest payed is 342482.34
    // 0 - 0 = 0 on day 19948. total interest payed is 342482.63

    // monthly
    // 896.7278 - 675 = 221.72778 on day 21367. total interest payed is 374152.97
    // remaining 223.18939 is payed off on day 21397

    println!("monthly payment {}", monthly_payment);
    if monthly_payment < minimum_monthly_payment {
        println!("minimum monthly payment {}", minimum_monthly_payment);
        return;
    }

    match (daily_payments, daily_interest) {
        (true, true) => {
            let (time, interest) = simulate_daily_interest(balance, daily_rate, |_, _| {
                monthly_payment / avg_month_period
            });
            println!("payed off in {} days, total interest {}", time, interest);
        }
        (false, true) => {
            let (time, interest) = simulate_daily_interest(balance, daily_rate, |_, day| {
                if day as f32 % avg_month_period < 1.0 {
                    monthly_payment
                } else {
                    0.0
                }
            });
            println!("payed off in {} days, total interest {}", time, interest);
        }
        (_, false) => {
            let (time, interest) =
                simulate_monthly_interest(balance, monthly_rate, |_, _| monthly_payment);
            println!("payed off in {} months, total interest {}", time, interest);
        }
    }
}
