// use rand::prelude::*;

fn main() {
    for n in 1..10000 {
        let nf = n as f32;
        let result = (nf + 7.0).powf(1.0 / nf);
        let absdiff = (result - result.round()).abs();
        if absdiff < 0.000000001 {
            println!("{:?} ^ (1/{:?}) == {:?}", nf + 7.0, nf, result);
        }
        let result = (7.0 - nf).powf(1.0 / (-nf));
        let absdiff = (result - result.round()).abs();
        if absdiff < 0.000000001 {
            println!("{:?} ^ (-1/{:?}) == {:?}", -nf + 7.0, nf, result);
        }
    }
}
