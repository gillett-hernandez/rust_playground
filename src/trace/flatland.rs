#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use serde::{
    de::{self, Deserialize, SeqAccess, Visitor},
    Serialize,
};

// use crate::parse::curves::{load_ior_and_kappa, load_multiple_csv_rows};
use crate::spectral::{SpectralPowerDistributionFunction, SPD};
// use crate::tonemap::{sRGB, Tonemapper};
use math::Vec3;

use packed_simd::{f32x2, f32x4};
use rand::prelude::*;
// use rayon::prelude::*;
use std::{
    f32::consts::{PI, TAU},
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2(pub f32x2);





impl<'de> Deserialize<'de> for Point2 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Point2Visitor;
        impl<'de> Visitor<'de> for Point2Visitor {
            type Value = Point2;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Point2")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let f0 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let f1 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(Point2(f32x2::new(f0, f1)))
            }
        }
        const FIELDS: &'static [&'static str] = &[""];
        deserializer.deserialize_struct("Point2", FIELDS, Point2Visitor)
    }
}

impl Serialize for Point2 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq([self.0.extract(0), self.0.extract(1)].iter())
    }
}

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
        Vec2::from_raw(f32x2::splat(s))
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

// very similar to Bounds2D except accelerated.
pub struct AABB2D(f32x4);
impl AABB2D {
    pub fn new(lower: Point2, upper: Point2) -> Self {
        let [x1, y1]: [f32; 2] = lower.0.into();
        let [x2, y2]: [f32; 2] = upper.0.into();
        // switch to be ordered.
        let (x1, y1, x2, y2) = (x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2));

        AABB2D(f32x4::new(x1, y1, x2, y2))
    }
    pub fn intersect(_ray: Ray2D, _tmax: f32) -> Point2 {
        // compute 4 intersection points
        Point2::ZERO
    }
}

pub fn reflect(wi: Vec2, normal: Vec2) -> Vec2 {
    let wi = -wi;
    (wi - 2.0 * (wi * normal) * normal).normalized()
}

pub fn refract(wi: Vec2, normal: Vec2, eta: f32) -> Option<Vec2> {
    let cos_i = wi * normal;
    let sin_2_theta_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin_2_theta_t = eta * eta * sin_2_theta_i;
    if sin_2_theta_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin_2_theta_t).sqrt();
    Some((-wi * eta + normal * (eta * cos_i - cos_t)).normalized())
}

pub fn fresnel_dielectric(eta_i: f32, eta_t: f32, cos_i: f32) -> f32 {
    // let swapped = if cos_i < 0 {
    //     cos_i = -cos_i;
    //     true
    // } else {
    //     false
    // };
    // let (eta_i, eta_t) = if swapped {
    //     (eta_t, eta_i)
    // } else {
    //     (eta_i, eta_t)
    // };
    let cos_i = cos_i.clamp(-1.0, 1.0);

    let (cos_i, eta_i, eta_t) = if cos_i < 0.0 {
        (-cos_i, eta_t, eta_i)
    } else {
        (cos_i, eta_i, eta_t)
    };

    let sin_t = eta_i / eta_t * (0.0f32).max(1.0 - cos_i * cos_i).sqrt();
    let cos_t = (0.0f32).max(1.0 - sin_t * sin_t).sqrt();
    let ei_ct = eta_i * cos_t;
    let et_ci = eta_t * cos_i;
    let ei_ci = eta_i * cos_i;
    let et_ct = eta_t * cos_t;
    let r_par = (et_ci - ei_ct) / (et_ci + ei_ct);
    let r_perp = (ei_ci - et_ct) / (ei_ci + et_ct);
    (r_par * r_par + r_perp * r_perp) / 2.0
}

pub fn fresnel_conductor(eta_i: f32, eta_t: f32, k_t: f32, cos_theta_i: f32) -> f32 {
    let cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    // handle dielectrics

    let (cos_theta_i, eta_i, eta_t) = if cos_theta_i < 0.0 {
        (-cos_theta_i, eta_t, eta_i)
    } else {
        (cos_theta_i, eta_i, eta_t)
    };

    // onto the full equations

    let eta = eta_t / eta_i;
    let etak = k_t / eta_i;

    let cos_theta_i2 = cos_theta_i * cos_theta_i;
    let sin_theta_i2 = 1.0 - cos_theta_i2;
    let eta2 = eta * eta;
    let etak2 = etak * etak;

    let t0 = eta2 - etak2 - sin_theta_i2;
    debug_assert!(t0 * t0 + eta2 * etak2 >= 0.0);
    let a2plusb2 = (t0 * t0 + eta2 * etak2 * 4.0).sqrt();
    let t1 = a2plusb2 + cos_theta_i2;
    debug_assert!(a2plusb2 + t0 >= 0.0, "{} {}", a2plusb2, t0);
    let a = ((a2plusb2 + t0) * 0.5).sqrt();
    let t2 = a * cos_theta_i * 2.0;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = a2plusb2 * cos_theta_i2 + sin_theta_i2 * sin_theta_i2;
    let t4 = t2 * sin_theta_i2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    (rs + rp) / 2.0
}

fn ggx_lambda(alpha: f32, w: Vec2) -> f32 {
    if w.y() == 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let w2 = Vec2::from_raw(w.0 * w.0);
    let c = 1.0 + (a2 * w2.x()) / w2.y(); // replace a2 with Vec2 for anistropy
    c.sqrt() * 0.5 - 0.5
}

fn ggx_d(alpha: f32, wm: Vec2) -> f32 {
    let slope = (wm.x() / alpha, 0.0);
    let slope2 = (slope.0 * slope.0, 0.0);
    let t = wm.y() * wm.y() + slope2.0;
    debug_assert!(t > 0.0, "{:?} {:?}", wm, slope2);
    let a2 = alpha * alpha;
    let t2 = t * t;
    let aatt = a2 * t2;
    debug_assert!(aatt > 0.0, "{} {} {:?}", alpha, t, wm);
    1.0 / (PI * aatt)
}

fn ggx_g(alpha: f32, wi: Vec2, wo: Vec2) -> f32 {
    let bottom = 1.0 + ggx_lambda(alpha, wi) + ggx_lambda(alpha, wo);
    debug_assert!(bottom != 0.0);
    bottom.recip()
}

fn ggx_vnpdf(alpha: f32, wi: Vec2, wh: Vec2) -> f32 {
    let inv_gl = 1.0 + ggx_lambda(alpha, wi);
    debug_assert!(wh.0.is_finite().all());
    (ggx_d(alpha, wh) * (wi * wh).abs()) / (inv_gl * wi.y().abs())
}

fn ggx_vnpdf_no_d(alpha: f32, wi: Vec2, wh: Vec2) -> f32 {
    ((wi * wh) / ((1.0 + ggx_lambda(alpha, wi)) * wi.y())).abs()
}

// fn sample_vndf(alpha: f32, wi: Vec2, sample: Sample2D) -> Vec2 {
//     let Sample2D { x, y } = sample;
//     let v = Vec2::new(alpha * wi.x(), wi.y()).normalized();

//     let t1 = if v.y() < 0.9999 {
//         v.cross(Vec2::Z).normalized()
//     } else {
//         Vec2::X
//     };
//     let t2 = t1.cross(v);
//     debug_assert!(v.0.is_finite().all(), "{:?}", v);
//     debug_assert!(t1.0.is_finite().all(), "{:?}", t1);
//     debug_assert!(t2.0.is_finite().all(), "{:?}", t2);
//     let a = 1.0 / (1.0 + v.z());
//     let r = x.sqrt();
//     debug_assert!(r.is_finite(), "{}", x);
//     let phi = if y < a {
//         y / a * PI
//     } else {
//         PI + (y - a) / (1.0 - a) * PI
//     };

//     let (sin_phi, cos_phi) = phi.sin_cos();
//     debug_assert!(sin_phi.is_finite() && cos_phi.is_finite(), "{:?}", phi);
//     let p1 = r * cos_phi;
//     // let p2 = r * sin_phi * if y < a { 1.0 } else { v.z() };
//     let value = 1.0 - p1 * p1;
//     let n = p1 * t1 + value.max(0.0).sqrt() * v;

//     debug_assert!(
//         n.0.is_finite().all(),
//         "{:?}, {:?}, {:?}, {:?}, {:?}",
//         n,
//         p1,
//         t1,
//         t2,
//         v
//     );
//     Vec2::new(alpha * n.x(), n.y().max(0.0)).normalized()
// }

// fn sample_wh(alpha: f32, wi: Vec2, sample: Sample2D) -> Vec2 {
//     // normal invert mark
//     let flip = wi.y() < 0.0;
//     let wh = sample_vndf(alpha, if flip { -wi } else { wi }, sample);
//     if flip {
//         -wh
//     } else {
//         wh
//     }
// }

#[derive(Copy, Clone, Debug)]
pub struct Ray2D {
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

pub enum Shape {
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
    Bezier {
        control: Vec<Point2>,
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
            Shape::Bezier { material_id, .. } => *material_id,
        }
    }
    pub fn sample_surface(&self) -> Point2 {
        match self {
            Shape::Point { p, .. } => *p,
            Shape::Circle {
                radius: _, center, ..
            } => *center,
            Shape::Arc {
                radius: _,
                center,
                normal: _,
                angle: _,
                ..
            } => *center,
            Shape::Line { p0, p1: _, .. } => *p0,
            Shape::Bezier { control, .. } => control[0],
        }
    }

    pub fn intersect(&self, r: Ray2D) -> Option<(Point2, Vec2, f32, usize)> {
        match self {
            Shape::Point {
                p: _,
                material_id: _,
            } => None,
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
                radius: _,
                center: _,
                normal: _,
                angle: _,
                material_id: _,
            } => {
                panic!()
            }
            Shape::Line {
                p0: _,
                p1: _,
                material_id: _,
            } => {
                panic!()
            }
            Shape::Bezier { control: _, .. } => {
                panic!()
            }
        }
    }
}

fn eta_rel(eta_o: f32, eta_inner: f32, wi: Vec2) -> f32 {
    if wi.y() < 0.0 {
        eta_o / eta_inner
    } else {
        eta_inner / eta_o
    }
}

pub enum Material {
    Lambertian {
        color: SPD,
    },
    GGX {
        eta: SPD,
        kappa: SPD,
        roughness: f32,
        permeable: bool,
        eta_o: f32,
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
                    (f * 0.5, cos * 0.5)
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
                    (f * 0.5, cos * 0.5)
                } else {
                    (0.0, 0.0)
                }
            }
            Material::Lambertian { color } => {
                let cos = wo.y().abs();
                let f = color.evaluate(lambda);
                if wi.y() * wo.y() > 0.0 {
                    (f * 0.5, cos * 0.5)
                } else {
                    (0.0, 0.0)
                }
            }
            Material::GGX {
                eta,
                kappa,
                roughness,
                permeable,
                eta_o,
            } => {
                let eta = eta.evaluate_power(lambda);
                let kappa = kappa.evaluate_power(lambda);
                let wiwo = wi.y() * wo.y();
                let same_hemisphere = wiwo > 0.0;
                let g = (wiwo).abs();
                let cos_i = wi.y();

                let (mut glossy, mut glossy_pdf) = (0.0, 0.0);
                let (mut transmission, mut transmission_pdf) = (0.0, 0.0);
                let fresnel;

                if same_hemisphere {
                    // reflect
                    let mut wh = (wi + wo).normalized();
                    if wh.y() < 0.0 {
                        wh = -wh;
                    }
                    let ndotv = wi * wh;
                    fresnel = fresnel_conductor(*eta_o, eta, kappa, ndotv);
                    glossy =
                        fresnel * (0.25 / g) * ggx_d(*roughness, wh) * ggx_g(*roughness, wi, wo);
                    glossy_pdf = ggx_vnpdf(*roughness, wi, wh) * 0.25 / ndotv.abs();
                } else {
                    if *permeable {
                        let eta_rel = eta_rel(*eta_o, eta, wi);

                        let ggxg = ggx_g(*roughness, wi, wo);
                        debug_assert!(
                            wi.0.is_finite().all() && wo.0.is_finite().all(),
                            "{:?} {:?} {:?} {:?}",
                            wi,
                            wo,
                            ggxg,
                            cos_i
                        );
                        let mut wh = (wi + eta_rel * wo).normalized();
                        // normal invert mark
                        if wh.y() < 0.0 {
                            wh = -wh;
                        }

                        let partial = ggx_vnpdf_no_d(*roughness, wi, wh);
                        let ndotv = wi * wh;
                        let ndotl = wo * wh;

                        let sqrt_denom = ndotv + eta_rel * ndotl;
                        let eta_rel2 = eta_rel * eta_rel;
                        let dwh_dwo1 = ndotl / (sqrt_denom * sqrt_denom); // dwh_dwo w/o etas
                        let dwh_dwo2 = eta_rel2 * dwh_dwo1; // dwh_dwo w/etas

                        // match transport_mode {
                        //     // in radiance mode, the reflectance/transmittance is not scaled by eta^2.
                        //     // in importance_mode, it is scaled by eta^2.
                        //     TransportMode::Importance => dwh_dwo1 = dwh_dwo2,
                        //     _ => {}
                        // }
                        debug_assert!(
                            wh.0.is_finite().all(),
                            "{:?} {:?} {:?} {:?}",
                            eta_rel,
                            ndotv,
                            ndotl,
                            sqrt_denom
                        );
                        let ggxd = ggx_d(*roughness, wh);
                        let weight = ggxd * ggxg * ndotv * dwh_dwo1 / g;
                        transmission_pdf = (ggxd * partial * dwh_dwo2).abs();

                        fresnel = fresnel_dielectric(*eta_o, eta, ndotv);
                        let inv_reflectance = 1.0 - fresnel;
                        transmission = inv_reflectance * weight.abs();
                    } else {
                        fresnel = 1.0;
                    }
                }
                (
                    glossy + transmission,
                    fresnel * glossy_pdf + (1.0 - fresnel) * transmission_pdf,
                )
            }
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
                // let y = 1.0 - random::<f32>().powi(2);
                // let x = (random::<f32>() - 0.5).signum() * (1.0 - y.powi(2)).sqrt();
                let xi = random::<f32>();
                let sin_i = 2.0 * xi - 1.0;
                let cos_i = (1.0 - sin_i.powi(2)).sqrt();

                let wo = Vec2::new(sin_i, cos_i) * wi.y().signum();
                let (f, pdf) = self.bsdf(lambda, wi, wo);
                (wo, f, pdf)
            }
            Material::GGX {
                eta,
                kappa,
                roughness: _,
                permeable,
                eta_o,
            } => {
                let eta = eta.evaluate_power(lambda);
                let kappa = kappa.evaluate_power(lambda);
                let cos_theta = wi.y();
                let fresnel = if *permeable {
                    fresnel_dielectric(*eta_o, eta, cos_theta)
                } else {
                    fresnel_conductor(*eta_o, eta, kappa, cos_theta)
                };
                let wo;

                let mut reflect = false;
                let normal = {
                    let flip = wi.y() < 0.0;
                    let wh = Vec2::Y;
                    if flip {
                        -wh
                    } else {
                        wh
                    }
                };
                let refract_direction = refract(wi, normal, 1.0 / eta_rel(*eta_o, eta, wi));
                if refract_direction.is_none() {
                    reflect = true;
                }
                let s = random::<f32>();
                if s < fresnel && !reflect {
                    reflect = true;
                } else if s >= fresnel && !reflect {
                    reflect = false;
                }
                if reflect {
                    wo = Vec2::new(-wi.x(), wi.y());
                } else {
                    wo = refract_direction.unwrap();
                }
                let (f, pdf) = self.bsdf(lambda, wi, wo);
                (wo, f, pdf)
            }
        }
    }
    pub fn sample_le(&self, lambda: f32, _point: Point2) -> (Vec2, f32) {
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

pub struct Scene {
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
