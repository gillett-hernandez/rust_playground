#![feature(slice_fill)]
extern crate line_drawing;
extern crate minifb;

use lib::*;
use minifb::*;

use math::XYZColor;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use packed_simd::{f32x2, f32x4};
use rand::prelude::*;
use rayon::prelude::*;
use std::{
    f32::consts::{PI, TAU},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use tonemap::{sRGB, Tonemapper};

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2(pub f32x2);

impl Point2 {
    pub const fn new(x: f32, y: f32) -> Point2 {
        Point2(f32x2::new(x, y))
    }
    pub const fn from_raw(v: f32x2) -> Point2 {
        Point2(v)
    }
    pub const ZERO: Point2 = Point2::from_raw(f32x2::new(0.0, 0.0));
    pub const ORIGIN: Point2 = Point2::from_raw(f32x2::new(0.0, 0.0));
    pub const INFINITY: Point2 = Point2::from_raw(f32x2::new(f32::INFINITY, f32::INFINITY));
    pub const NEG_INFINITY: Point2 =
        Point2::from_raw(f32x2::new(f32::NEG_INFINITY, f32::NEG_INFINITY));
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Point2 {
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
}

impl Default for Point2 {
    fn default() -> Self {
        Point2::ORIGIN
    }
}

impl Add<Vec2> for Point2 {
    type Output = Point2;
    fn add(self, other: Vec2) -> Point2 {
        // Point2::new(self.x + other.x, self.y + other.y, self.z + other.z)
        Point2::from_raw(self.0 + other.0)
    }
}

impl AddAssign<Vec2> for Point2 {
    fn add_assign(&mut self, other: Vec2) {
        // Point2::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 += other.0
    }
}

impl Sub<Vec2> for Point2 {
    type Output = Point2;
    fn sub(self, other: Vec2) -> Point2 {
        // Point2::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Point2::from_raw(self.0 - other.0)
    }
}

impl SubAssign<Vec2> for Point2 {
    fn sub_assign(&mut self, other: Vec2) {
        // Point2::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 -= other.0
    }
}

// // don't implement adding or subtracting floats from Point2, because that's equivalent to adding or subtracting a Vector with components f,f,f and why would you want to do that.

impl Sub for Point2 {
    type Output = Vec2;
    fn sub(self, other: Point2) -> Vec2 {
        // Vec2::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec2::from_raw(self.0 - other.0)
    }
}

impl From<[f32; 3]> for Point2 {
    fn from(other: [f32; 3]) -> Point2 {
        Point2::new(other[0], other[1])
    }
}

impl From<Vec2> for Point2 {
    fn from(v: Vec2) -> Point2 {
        // Point2::from_raw(v.0.replace(3, 1.0))
        Point2::ORIGIN + v
        // Point2::from_raw(v.0)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
}

#[derive(Copy, Clone, PartialEq, Default)]
pub struct Vec2(pub f32x2);

impl std::fmt::Debug for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Vec2")
            .field(&self.x())
            .field(&self.y())
            .finish()
    }
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Vec2 {
        Vec2(f32x2::new(x, y))
    }
    pub const fn from_raw(v: f32x2) -> Vec2 {
        Vec2(v)
    }
    pub const ZERO: Vec2 = Vec2::from_raw(f32x2::splat(0.0));
    pub const MASK: f32x2 = f32x2::new(1.0, 1.0);
    pub const X: Vec2 = Vec2::new(1.0, 0.0);
    pub const Y: Vec2 = Vec2::new(0.0, 1.0);
    pub fn from_axis(axis: Axis) -> Vec2 {
        match axis {
            Axis::X => Vec2::X,
            Axis::Y => Vec2::Y,
        }
    }
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Vec2 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
}

impl Mul for Vec2 {
    type Output = f32;
    fn mul(self, other: Vec2) -> f32 {
        // self.x * other.x + self.y * other.y + self.z * other.z
        (self.0 * other.0).sum()
    }
}

impl MulAssign for Vec2 {
    fn mul_assign(&mut self, other: Vec2) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, other: f32) -> Vec2 {
        Vec2::from_raw(self.0 * other)
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;
    fn mul(self, other: Vec2) -> Vec2 {
        Vec2::from_raw(self * other.0)
    }
}

impl Div<f32> for Vec2 {
    type Output = Vec2;
    fn div(self, other: f32) -> Vec2 {
        Vec2::from_raw(self.0 / other)
    }
}

// impl Div for Vec2 {
//     type Output = Vec2;
//     fn div(self, other: Vec2) -> Vec2 {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         Vec2::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point2
// impl Add<f32> for Vec2 {
//     type Output = Vec2;
//     fn add(self, other: f32) -> Vec2 {
//         Vec2::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec2 {
//     type Output = Vec2;
//     fn sub(self, other: f32) -> Vec2 {
//         Vec2::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2::from_raw(self.0 + other.0)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, other: Vec2) {
        self.0 += other.0;
    }
}

impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2::from_raw(-self.0)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        self + (-other)
    }
}

impl From<f32> for Vec2 {
    fn from(s: f32) -> Vec2 {
        Vec2::from_raw(f32x2::splat(s) * Vec2::MASK)
    }
}

impl From<Vec2> for f32x2 {
    fn from(v: Vec2) -> f32x2 {
        v.0
    }
}

impl Vec2 {
    pub fn cross(&self, other: Vec2) -> Vec3 {
        let (x1, y1, z1) = (self.x(), self.y(), 0.0);
        let (x2, y2, z2) = (other.x(), other.y(), 0.0);
        Vec3::new(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - x2 * y1)
    }

    pub fn norm_squared(&self) -> f32 {
        (self.0 * self.0).sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec2::from_raw(self.0 / norm)
    }
}

impl From<[f32; 2]> for Vec2 {
    fn from(other: [f32; 2]) -> Vec2 {
        Vec2::new(other[0], other[1])
    }
}

impl From<Point2> for Vec2 {
    fn from(p: Point2) -> Self {
        Vec2::from_raw(p.0)
    }
}

#[derive(Copy, Clone, Debug)]
struct Ray2D {
    pub origin: Point2,
    pub direction: Vec2,
    pub tmax: f32,
}

impl Ray2D {
    pub const fn new(origin: Point2, direction: Vec2) -> Self {
        Ray2D {
            origin,
            direction,
            tmax: f32::INFINITY,
        }
    }

    pub fn point_at(&self, time: f32) -> Point2 {
        self.origin + self.direction * time
    }
}

// also known as an orthonormal basis.
#[derive(Copy, Clone, Debug)]
pub struct TangentFrame2D {
    pub tangent: Vec2,
    pub normal: Vec2,
}

impl TangentFrame2D {
    pub fn new(tangent: Vec2, normal: Vec2) -> Self {
        debug_assert!(
            (tangent * normal).abs() < 0.000001,
            "tn: {:?} * {:?} was != 0",
            tangent,
            normal
        );
        TangentFrame2D {
            tangent: tangent.normalized(),

            normal: normal.normalized(),
        }
    }
    pub fn from_tangent_and_normal(tangent: Vec2, normal: Vec2) -> Self {
        TangentFrame2D {
            tangent: tangent.normalized(),

            normal: normal.normalized(),
        }
    }

    pub fn from_normal(normal: Vec2) -> Self {
        // let n2 = Vec2::from_raw(normal.0 * normal.0);
        // let (x, y, z) = (normal.x(), normal.y(), normal.z());
        let [nx, ny]: [f32; 2] = normal.0.into();
        let tangent = if nx.abs() > 0.001 {
            let dydx = ny / nx;
            // m1 * m2 == -1
            // m2 = -1 / m1
            let t_dydx = -1.0 / dydx;
            Vec2::new(1.0, t_dydx).normalized()
        } else {
            let dxdy = nx / ny;
            // m1 * m2 == -1
            // m2 = -1 / m1
            let t_dxdy = -1.0 / dxdy;
            Vec2::new(t_dxdy, 1.0).normalized()
        };

        TangentFrame2D { tangent, normal }
    }

    #[inline(always)]
    pub fn to_world(&self, v: &Vec2) -> Vec2 {
        self.tangent * v.x() + self.normal * v.y()
    }

    #[inline(always)]
    pub fn to_local(&self, v: &Vec2) -> Vec2 {
        Vec2::new(self.tangent * (*v), self.normal * (*v))
    }
}

enum Shape {
    Point {
        p: Point2,
        material_id: usize,
    },
    Circle {
        radius: f32,
        center: Point2,
        material_id: usize,
    },
    Arc {
        radius: f32,
        center: Point2,
        normal: Vec2,
        angle: f32,
        material_id: usize,
    },
    Line {
        p0: Point2,
        p1: Point2,
        material_id: usize,
    },
}

impl Shape {
    pub fn get_material_id(&self) -> usize {
        match self {
            Shape::Point { material_id, .. } => *material_id,
            Shape::Circle { material_id, .. } => *material_id,
            Shape::Arc { material_id, .. } => *material_id,
            Shape::Line { material_id, .. } => *material_id,
        }
    }
    pub fn sample_surface(&self) -> Point2 {
        match self {
            Shape::Point { p, .. } => *p,
            Shape::Circle { radius, center, .. } => *center,
            Shape::Arc {
                radius,
                center,
                normal,
                angle,
                ..
            } => *center,
            Shape::Line { p0, p1, .. } => *p0,
        }
    }

    pub fn intersect(&self, r: Ray2D) -> Option<(Point2, Vec2, f32, usize)> {
        match self {
            Shape::Point { p, material_id } => None,
            Shape::Circle {
                radius,
                center,
                material_id,
            } => {
                let [x, y]: [f32; 2] = r.origin.0.into();
                let [dx, dy]: [f32; 2] = r.direction.0.into();
                let (dxr, dyr) = (x - center.x(), y - center.y());
                let a = dx * dx + dy * dy;
                let b = 2.0 * dx * dxr + 2.0 * dy * dyr;
                let c = dxr * dxr + dyr * dyr - radius * radius;
                let discriminant = b * b - 4.0 * a * c;
                let t0 = (-b + discriminant.sqrt()) / (2.0 * a);
                let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
                let mut tmin = t0.min(t1);
                if tmin < 0.0 {
                    tmin = t0.max(t1);
                }
                let pt = r.point_at(tmin);
                if tmin > 0.0 && tmin < r.tmax {
                    return Some((pt, (pt - *center) / *radius, tmin, *material_id));
                }

                None
            }
            Shape::Arc {
                radius,
                center,
                normal,
                angle,
                material_id,
            } => {
                panic!()
            }
            Shape::Line {
                p0,
                p1,
                material_id,
            } => {
                panic!()
            }
        }
    }
}

enum Material {
    Lambertian {
        color: SPD,
    },
    GGX {
        eta: SPD,
        kappa: SPD,
        roughness: f32,
    },
    DiffuseLight {
        reflection_color: SPD,
        emission_color: SPD,
    },
    DiffuseDirectionalLight {
        reflection_color: SPD,
        emission_color: SPD,
        direction: f32,
        radius: f32,
    },
}

impl Material {
    pub fn bsdf(&self, lambda: f32, wi: Vec2, wo: Vec2) -> (f32, f32) {
        match self {
            Material::DiffuseLight {
                reflection_color, ..
            } => {
                let cos = wo.y().abs();
                let f = reflection_color.evaluate(lambda);
                if wi.y() * wo.y() > 0.0 {
                    (f * cos, cos / PI)
                } else {
                    (0.0, 0.0)
                }
            }
            Material::DiffuseDirectionalLight {
                reflection_color, ..
            } => {
                let cos = wo.y().abs();
                let f = reflection_color.evaluate(lambda);
                if wi.y() * wo.y() > 0.0 {
                    (f * cos, cos / PI)
                } else {
                    (0.0, 0.0)
                }
            }
            Material::Lambertian { color } => {
                let cos = wo.y().abs();
                let f = color.evaluate(lambda);
                if wi.y() * wo.y() > 0.0 {
                    (f * cos, cos / PI)
                } else {
                    (0.0, 0.0)
                }
            }
            Material::GGX {
                eta,
                kappa,
                roughness,
            } => panic!(),
        }
    }
    pub fn sample_bsdf(&self, lambda: f32, wi: Vec2) -> (Vec2, f32, f32) {
        match self {
            Material::DiffuseLight { .. } => {
                let y = 1.0 - random::<f32>().powi(2);
                let x = (random::<f32>() - 0.5).signum() * (1.0 - y.powi(2)).sqrt();

                let wo = Vec2::new(x, y) * wi.y().signum();
                let (f, pdf) = self.bsdf(lambda, wi, wo);
                (wo, f, pdf)
            }
            Material::DiffuseDirectionalLight { .. } => {
                let y = 1.0 - random::<f32>().powi(2);
                let x = (random::<f32>() - 0.5).signum() * (1.0 - y.powi(2)).sqrt();

                let wo = Vec2::new(x, y) * wi.y().signum();
                let (f, pdf) = self.bsdf(lambda, wi, wo);
                (wo, f, pdf)
            }
            Material::Lambertian { .. } => {
                let y = 1.0 - random::<f32>().powi(2);
                let x = (random::<f32>() - 0.5).signum() * (1.0 - y.powi(2)).sqrt();

                let wo = Vec2::new(x, y) * wi.y().signum();
                let (f, pdf) = self.bsdf(lambda, wi, wo);
                (wo, f, pdf)
            }
            Material::GGX {
                eta,
                kappa,
                roughness,
            } => panic!(),
        }
    }
    pub fn sample_le(&self, lambda: f32, point: Point2) -> (Vec2, f32) {
        match self {
            Material::DiffuseLight { emission_color, .. } => {
                let phi = random::<f32>() * TAU;
                let (sin, cos) = phi.sin_cos();
                (Vec2::new(cos, sin), emission_color.evaluate_power(lambda))
            }
            Material::DiffuseDirectionalLight {
                emission_color,
                direction,
                radius,
                ..
            } => {
                let phi = (2.0 * random::<f32>() - 1.0) * radius;
                let e = emission_color.evaluate_power(lambda);
                let true_direction = direction + phi;
                let (s, c) = true_direction.sin_cos();
                (Vec2::new(c, s), e)
            }
            Material::Lambertian { .. } | Material::GGX { .. } => {
                panic!()
            }
        }
    }
}

struct Scene {
    pub shapes: Vec<Shape>,
    pub materials: Vec<Material>,
    lights: Vec<usize>,
}

impl Scene {
    pub fn new(shapes: Vec<Shape>, materials: Vec<Material>) -> Self {
        let mut lights = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let mat = &materials[shape.get_material_id()];
            match &mat {
                Material::DiffuseLight { .. } | Material::DiffuseDirectionalLight { .. } => {
                    lights.push(i);
                }
                _ => {}
            }
        }
        Scene {
            shapes,
            materials,
            lights,
        }
    }
    pub fn sample_light(&self) -> &Shape {
        let s = random::<f32>();
        let idx = (self.lights.len() as f32 * s) as usize;
        &self.shapes[idx]
    }

    pub fn get_material(&self, material_id: usize) -> &Material {
        &self.materials[material_id]
    }

    pub fn intersect(&self, r: Ray2D) -> Option<(Point2, Vec2, usize)> {
        let mut earliest = (Point2::ZERO, Vec2::ZERO, f32::INFINITY, 0usize);
        for shape in self.shapes.iter() {
            // compute ray intersection with each shape type. return intersection point and normal.
            if let Some((point, normal, time, material_id)) = shape.intersect(r) {
                if time < earliest.2 {
                    earliest = (point, normal, time, material_id);
                }
            }
        }
        earliest
            .2
            .is_finite()
            .then(|| (earliest.0, earliest.1, earliest.3))
    }
}

fn main() {
    let threads = 1;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "2D Tracer",
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        WindowOptions {
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
    let mut window_pixels = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);
    window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
    let width = film.width;
    let height = film.height;

    let frame_dt = 6944.0 / 1000000.0;

    let white = SPD::Linear {
        signal: vec![1.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let black = SPD::Linear {
        signal: vec![0.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let glass_eta = SPD::Cauchy { a: 1.4, b: 10000.0 };
    let scene = Scene::new(
        vec![
            Shape::Point {
                p: Point2::new(-0.49, 0.0),
                material_id: 0,
            },
            Shape::Circle {
                radius: 0.1,
                center: Point2::ORIGIN,
                material_id: 1,
            },
        ],
        vec![
            Material::DiffuseDirectionalLight {
                reflection_color: white.clone(),
                emission_color: white.clone(),
                direction: (0.0f32).to_radians(),
                radius: 0.3,
            },
            Material::Lambertian {
                color: white.clone(),
            },
            Material::GGX {
                eta: glass_eta,
                kappa: black,
                roughness: 0.001,
            },
        ],
    );
    let view_bounds = Bounds2D::new(Bounds1D::new(-0.5, 0.5), Bounds1D::new(-0.5, 0.5));
    let (box_width, box_height) = (
        view_bounds.x.span() / width as f32,
        view_bounds.y.span() / height as f32,
    );
    let mut max_bounces = 4;
    let mut exposure_bias = 10.0;
    let mut new_rays_per_frame = 1000;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let keys = window.get_keys_pressed(KeyRepeat::Yes);
        let config_move = if window.is_key_down(Key::NumPadPlus) {
            1.0
        } else if window.is_key_down(Key::NumPadMinus) {
            -1.0
        } else {
            0.0
        };
        for key in keys.unwrap_or(vec![]) {
            match key {
                Key::E => {
                    exposure_bias += config_move;
                    println!("new exposure bias is {}", exposure_bias);
                }
                Key::B => {
                    max_bounces = (max_bounces + config_move as i32).max(0);
                    println!("new max bounces is {}", max_bounces);
                }
                Key::Space => {
                    film.buffer.fill(XYZColor::BLACK);
                }
                _ => {}
            }
        }

        // do tracing here.

        let mut rays: Vec<(Ray2D, f32, f32, bool)> = (0usize..new_rays_per_frame)
            .into_par_iter()
            .map(|_| {
                let lambda =
                    random::<f32>() * BOUNDED_VISIBLE_RANGE.span() + BOUNDED_VISIBLE_RANGE.lower;
                let light_shape = scene.sample_light();
                let point = light_shape.sample_surface();
                let light_mat = scene.get_material(light_shape.get_material_id());
                let (wo, energy) = light_mat.sample_le(lambda, point);
                (Ray2D::new(point, wo), lambda, energy, true)
            })
            .collect();

        let mut lines = Vec::new();
        for _ in 0..max_bounces {
            let mut new_lines: Vec<(Point2, Point2, XYZColor)> = rays
                .par_iter_mut()
                .filter_map(|(r, lambda, throughput, active)| {
                    if *active {
                        let intersection = scene.intersect(*r);
                        assert!(
                            !throughput.is_nan(),
                            "{:?} {:?}, {:?}",
                            r,
                            lambda,
                            throughput
                        );
                        let origin = r.origin;
                        let shading_color =
                            XYZColor::from_wavelength_and_energy(*lambda, *throughput);
                        let point = match intersection {
                            Some((point, normal, material_id)) => {
                                let mat = scene.get_material(material_id);
                                let frame = TangentFrame2D::from_normal(normal);
                                let wi = frame.to_local(&-r.direction);
                                let (wo, bsdf_f, bsdf_pdf) = mat.sample_bsdf(*lambda, wi);

                                if bsdf_pdf == 0.0 || bsdf_f == 0.0 {
                                    *active = false;
                                    point
                                } else {
                                    *throughput *= bsdf_f * wi.y().abs() / bsdf_pdf;
                                    let dir = frame.to_world(&wo).normalized();
                                    *r =
                                        Ray2D::new(point + normal * 0.00001 * wo.y().signum(), dir);
                                    point
                                }
                            }
                            None => {
                                // exit scene. compute clip bounds.
                                let mut min_t = f32::INFINITY;
                                match r.direction.x() {
                                    dx if dx > 0.0 => {
                                        min_t = min_t.min((view_bounds.x.upper - r.origin.x()) / dx)
                                    }
                                    dx if dx < 0.0 => {
                                        min_t = min_t.min((view_bounds.x.lower - r.origin.x()) / dx)
                                    }
                                    _ => {
                                        // up or down clip bounds will be computed in other match statement
                                    }
                                }
                                match r.direction.y() {
                                    dy if dy > 0.0 => {
                                        min_t = min_t.min((view_bounds.y.upper - r.origin.y()) / dy)
                                    }
                                    dy if dy < 0.0 => {
                                        min_t = min_t.min((view_bounds.y.lower - r.origin.y()) / dy)
                                    }
                                    _ => {
                                        // left or right clip bounds should have been computed in other match statement.
                                        assert!(r.direction.x() != 0.0);
                                    }
                                }
                                *active = false;
                                r.point_at(min_t)
                            }
                        };
                        Some((origin, point, shading_color))
                    } else {
                        None
                    }
                })
                .collect();
            lines.extend(new_lines.drain(..));
        }

        for line in lines.drain(..) {
            let (px0, py0) = (
                (WINDOW_WIDTH as f32 * (line.0.x() - view_bounds.x.lower) / view_bounds.x.span())
                    as usize,
                (WINDOW_HEIGHT as f32 * (line.0.y() - view_bounds.y.lower) / view_bounds.y.span())
                    as usize,
            );
            let (px1, py1) = (
                (WINDOW_WIDTH as f32 * (line.1.x() - view_bounds.x.lower) / view_bounds.x.span())
                    as usize,
                (WINDOW_HEIGHT as f32 * (line.1.y() - view_bounds.y.lower) / view_bounds.y.span())
                    as usize,
            );

            let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
            let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
            if dx == 0 && dy == 0 {
                if px0 as usize >= WINDOW_WIDTH || py0 as usize >= WINDOW_HEIGHT {
                    continue;
                }
                film.buffer[py0 as usize * width + px0 as usize] += line.2;
                continue;
            }
            for (x, y) in line_drawing::Midpoint::<f32, isize>::new(
                (px0 as f32, py0 as f32),
                (px1 as f32, py1 as f32),
            ) {
                if x as usize >= WINDOW_WIDTH || y as usize >= WINDOW_HEIGHT {
                    continue;
                }
                assert!(!b.is_nan(), "{} {}", dx, dy);
                film.buffer[y as usize * width + x as usize] += line.2 * b;
            }
        }

        let srgb_tonemapper = sRGB::new(&film, exposure_bias);
        window_pixels
            .buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel_idx, v)| {
                let y: usize = pixel_idx / width;
                let x: usize = pixel_idx - width * y;
                let (mapped, _linear) = srgb_tonemapper.map(&film, (x, y));
                let [r, g, b, _]: [f32; 4] = mapped.into();
                *v = rgb_to_u32((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8);
            });
        window
            .update_with_buffer(&window_pixels.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
            .unwrap();
    }
}
