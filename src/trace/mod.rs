use std::f32::INFINITY;

use packed_simd::f32x4;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub mod camera;

pub mod lens;
pub mod material;
pub mod math;
pub mod primitive;
pub mod random;
pub mod tonemap;

use crate::film::Film;
use camera::ProjectiveCamera;
pub use material::{
    ConstDiffuseEmitter, ConstFilm, ConstLambertian, HenyeyGreensteinHomogeneous, Material,
    MaterialEnum, Medium, MediumEnum,
};
pub use math::*;
pub use primitive::{
    IntersectionData, MediumIntersectionData, Primitive, Sphere, SurfaceIntersectionData,
};
use tonemap::{sRGB, Tonemapper};
