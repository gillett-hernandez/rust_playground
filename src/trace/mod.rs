pub mod camera;

pub mod lens;
pub mod material;

pub mod flatland;
pub mod primitive;
pub mod random;
pub mod tonemap;

pub use crate::film::Film;

pub use material::{
    ConstDiffuseEmitter, ConstFilm, ConstLambertian, HenyeyGreensteinHomogeneous, Material,
    MaterialEnum, Medium, MediumEnum,
};
pub use math::*;
pub use primitive::{
    IntersectionData, MediumIntersectionData, Primitive, Sphere, SurfaceIntersectionData,
};
