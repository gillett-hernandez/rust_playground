extern crate minifb;

use lib::curves::{load_ior_and_kappa, load_multiple_csv_rows};
use lib::spectral::{
    InterpolationMode, BOUNDED_VISIBLE_RANGE,
    EXTENDED_VISIBLE_RANGE, SPD,
};
use lib::tonemap::{sRGB, Tonemapper};
use lib::trace::{Bounds1D, Bounds2D, SingleWavelength};
use lib::{rgb_to_u32, Film};

#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use rand::prelude::*;
use rayon::prelude::*;

use math::{Ray, Vec3, Point3, XYZColor};


pub struct Sphere {}


fn main() {

}
