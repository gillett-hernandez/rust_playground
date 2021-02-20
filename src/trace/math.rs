use packed_simd::f32x4;
use serde::{Deserialize, Serialize};

use crate::{gaussian, x_bar, y_bar, z_bar};
use ordered_float::OrderedFloat;
use std::f32::INFINITY;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use std::fmt;

#[derive(Copy, Clone, Debug)]
pub struct Bounds1D {
    pub lower: f32,
    pub upper: f32,
}

impl Bounds1D {
    pub const fn new(lower: f32, upper: f32) -> Self {
        Bounds1D { lower, upper }
    }
    pub fn span(&self) -> f32 {
        self.upper - self.lower
    }

    pub fn contains(&self, value: &f32) -> bool {
        &self.lower <= value && value < &self.upper
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.max(other.lower), self.upper.min(other.upper))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Bounds2D {
    pub x: Bounds1D,
    pub y: Bounds1D,
}

impl Bounds2D {
    pub const fn new(x: Bounds1D, y: Bounds1D) -> Self {
        Bounds2D { x, y }
    }
    pub fn area(&self) -> f32 {
        self.x.span() * self.y.span()
    }

    pub fn contains(&self, value: (f32, f32)) -> bool {
        self.x.contains(&value.0) && self.y.contains(&value.1)
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds2D::new(self.x.intersection(other.x), self.y.intersection(other.y))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds2D::new(self.x.union(other.x), self.y.union(other.y))
    }
}

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 760.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(400.0, 750.0);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point3(pub f32x4);

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3(f32x4::new(x, y, z, 1.0))
    }
    pub const fn from_raw(v: f32x4) -> Point3 {
        Point3(v)
    }
    pub const ZERO: Point3 = Point3::from_raw(f32x4::new(0.0, 0.0, 0.0, 1.0));
    pub const ORIGIN: Point3 = Point3::from_raw(f32x4::new(0.0, 0.0, 0.0, 1.0));
    pub const INFINITY: Point3 = Point3::from_raw(f32x4::new(INFINITY, INFINITY, INFINITY, 1.0));
    pub const NEG_INFINITY: Point3 =
        Point3::from_raw(f32x4::new(-INFINITY, -INFINITY, -INFINITY, 1.0));
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Point3 {
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
    pub fn normalize(mut self) -> Self {
        unsafe {
            self.0 = self.0 / self.0.extract_unchecked(3);
        }
        self
    }
}

impl Default for Point3 {
    fn default() -> Self {
        Point3::ORIGIN
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        Point3::from_raw(self.0 + other.0)
    }
}

impl AddAssign<Vec3> for Point3 {
    fn add_assign(&mut self, other: Vec3) {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 += other.0
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, other: Vec3) -> Point3 {
        // Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Point3::from_raw(self.0 - other.0)
    }
}

impl SubAssign<Vec3> for Point3 {
    fn sub_assign(&mut self, other: Vec3) {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 -= other.0
    }
}

// // don't implement adding or subtracting floats from Point3, because that's equivalent to adding or subtracting a Vector with components f,f,f and why would you want to do that.

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        // Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec3::from_raw((self.0 - other.0) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

impl From<[f32; 3]> for Point3 {
    fn from(other: [f32; 3]) -> Point3 {
        Point3::new(other[0], other[1], other[2])
    }
}

impl From<Vec3> for Point3 {
    fn from(v: Vec3) -> Point3 {
        // Point3::from_raw(v.0.replace(3, 1.0))
        Point3::ORIGIN + v
        // Point3::from_raw(v.0)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Copy, Clone, PartialEq, Default)]
pub struct Vec3(pub f32x4);

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        // Vec3 { x, y, z, w: 0.0 }
        Vec3(f32x4::new(x, y, z, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> Vec3 {
        Vec3(v)
    }
    pub const ZERO: Vec3 = Vec3::from_raw(f32x4::splat(0.0));
    pub const MASK: f32x4 = f32x4::new(1.0, 1.0, 1.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub fn from_axis(axis: Axis) -> Vec3 {
        match axis {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    #[inline(always)]
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
}

impl Mul for Vec3 {
    type Output = f32;
    fn mul(self, other: Vec3) -> f32 {
        // self.x * other.x + self.y * other.y + self.z * other.z
        (self.0 * other.0).sum()
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3::from_raw(self.0 * other)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::from_raw(self * other.0)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3::from_raw(self.0 / other)
    }
}

// impl Div for Vec3 {
//     type Output = Vec3;
//     fn div(self, other: Vec3) -> Vec3 {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         Vec3::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for Vec3 {
//     type Output = Vec3;
//     fn add(self, other: f32) -> Vec3 {
//         Vec3::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec3 {
//     type Output = Vec3;
//     fn sub(self, other: f32) -> Vec3 {
//         Vec3::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::from_raw(self.0 + other.0)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        self.0 += other.0;
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::from_raw(-self.0)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self + (-other)
    }
}

impl From<f32> for Vec3 {
    fn from(s: f32) -> Vec3 {
        Vec3::from_raw(f32x4::splat(s) * Vec3::MASK)
    }
}

impl From<Vec3> for f32x4 {
    fn from(v: Vec3) -> f32x4 {
        v.0
    }
}

impl Vec3 {
    pub fn cross(&self, other: Vec3) -> Self {
        let (x1, y1, z1) = (self.x(), self.y(), self.z());
        let (x2, y2, z2) = (other.x(), other.y(), other.z());
        Vec3::new(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - x2 * y1)
    }

    pub fn norm_squared(&self) -> f32 {
        (self.0 * self.0 * Vec3::MASK).sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec3::from_raw(self.0 / norm)
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(other: [f32; 3]) -> Vec3 {
        Vec3::new(other[0], other[1], other[2])
    }
}

impl From<Point3> for Vec3 {
    fn from(p: Point3) -> Self {
        Vec3::from_raw(p.0 * Vec3::MASK)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct XYZColor(pub f32x4);

impl XYZColor {
    pub const fn new(x: f32, y: f32, z: f32) -> XYZColor {
        // XYZColor { x, y, z, w: 0.0 }
        XYZColor(f32x4::new(x, y, z, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> XYZColor {
        XYZColor(v)
    }
    pub const BLACK: XYZColor = XYZColor::from_raw(f32x4::splat(0.0));
    pub const ZERO: XYZColor = XYZColor::from_raw(f32x4::splat(0.0));
}

impl XYZColor {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }

    pub fn from_wavelength_and_energy(lambda: f32, energy: f32) -> XYZColor {
        let ang = lambda * 10.0;
        XYZColor::new(
            x_bar(ang) * energy,
            y_bar(ang) * energy,
            z_bar(ang) * energy,
        )
    }
}

// impl Mul for XYZColor {
//     type Output = Self;
//     fn mul(self, other: XYZColor) -> Self {
//         // self.x * other.x + self.y * other.y + self.z * other.z
//         XYZColor::from_raw(self.0 * other.0)
//     }
// }

// impl MulAssign for XYZColor {
//     fn mul_assign(&mut self, other: XYZColor) {
//         // self.x *= other.x;
//         // self.y *= other.y;
//         // self.z *= other.z;
//         self.0 = self.0 * other.0
//     }
// }

impl Mul<f32> for XYZColor {
    type Output = XYZColor;
    fn mul(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 * other)
    }
}

impl Mul<XYZColor> for f32 {
    type Output = XYZColor;
    fn mul(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(other.0 * self)
    }
}

impl Div<f32> for XYZColor {
    type Output = XYZColor;
    fn div(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 / other)
    }
}

impl DivAssign<f32> for XYZColor {
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / other;
    }
}

// impl Div for XYZColor {
//     type Output = XYZColor;
//     fn div(self, other: XYZColor) -> XYZColor {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         XYZColor::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for XYZColor {
//     type Output = XYZColor;
//     fn add(self, other: f32) -> XYZColor {
//         XYZColor::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for XYZColor {
//     type Output = XYZColor;
//     fn sub(self, other: f32) -> XYZColor {
//         XYZColor::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for XYZColor {
    type Output = XYZColor;
    fn add(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(self.0 + other.0)
    }
}

impl AddAssign for XYZColor {
    fn add_assign(&mut self, other: XYZColor) {
        self.0 = self.0 + other.0
        // self.0 = (*self + other).0
    }
}

impl From<XYZColor> for f32x4 {
    fn from(v: XYZColor) -> f32x4 {
        v.0
    }
}

#[derive(Debug)]
pub struct Sample1D {
    pub x: f32,
}

impl Sample1D {
    pub fn new(x: f32) -> Self {
        debug_assert!(x < 1.0 && x >= 0.0);
        Sample1D { x }
    }
    pub fn new_random_sample() -> Self {
        Sample1D::new(rand::random::<f32>())
    }
    pub fn choose<T>(mut self, split: f32, a: T, b: T) -> (Self, T) {
        debug_assert!(0.0 <= split && split <= 1.0);
        debug_assert!(self.x >= 0.0 && self.x < 1.0);
        if self.x < split {
            assert!(split > 0.0);
            self.x /= split;
            (self, a)
        } else {
            // if split was 1.0, there's no way for self.x to be greather than or equal to it
            // since self.x in [0, 1)
            debug_assert!(split < 1.0);
            self.x = (self.x - split) / (1.0 - split);
            (self, b)
        }
    }
}

#[derive(Debug)]
pub struct Sample2D {
    pub x: f32,
    pub y: f32,
}

impl Sample2D {
    pub fn new(x: f32, y: f32) -> Self {
        debug_assert!(x < 1.0 && x >= 0.0);
        debug_assert!(y < 1.0 && y >= 0.0);

        Sample2D { x, y }
    }
    pub fn new_random_sample() -> Self {
        Sample2D::new(rand::random(), rand::random())
    }
}

#[derive(Debug)]
pub struct Sample3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Sample3D {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Sample3D { x, y, z }
    }
    pub fn new_random_sample() -> Self {
        Sample3D::new(rand::random(), rand::random(), rand::random())
    }
}

#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub enum InterpolationMode {
    Linear,
    Nearest,
    Cubic,
}

#[derive(Debug, Clone)]
pub enum SPD {
    Linear {
        signal: Vec<f32>,
        bounds: Bounds1D,
        mode: InterpolationMode,
    },
    Tabulated {
        signal: Vec<(f32, f32)>,
        mode: InterpolationMode,
    },
    Polynomial {
        xoffset: f32,
        coefficients: [f32; 8],
    },
    Cauchy {
        a: f32,
        b: f32,
    },
    Exponential {
        signal: Vec<(f32, f32, f32, f32)>,
    },
    InverseExponential {
        signal: Vec<(f32, f32, f32, f32)>,
    },
    Blackbody {
        temperature: f32,
        boost: f32,
    },
}

impl Default for SPD {
    fn default() -> Self {
        SPD::Linear {
            signal: vec![0.0],
            bounds: EXTENDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        }
    }
}

// use crate::random::

const HCC2: f32 = 1.1910429723971884140794892e-29;
const HKC: f32 = 1.438777085924334052222404423195819240925e-2;

pub fn blackbody(temperature: f32, lambda: f32) -> f32 {
    let lambda = lambda * 1e-9;

    lambda.powi(-5) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
}

pub fn max_blackbody_lambda(temp: f32) -> f32 {
    2.8977721e-3 / (temp * 1e-9)
}

pub trait SpectralPowerDistributionFunction {
    fn evaluate(&self, lambda: f32) -> f32;
    fn evaluate_power(&self, lambda: f32) -> f32;
    // note: sample power
    fn convert_to_xyz(&self, integration_bounds: Bounds1D, step_size: f32) -> XYZColor {
        let iterations = (integration_bounds.span() / step_size) as usize;
        let mut sum: XYZColor = XYZColor::ZERO;
        for i in 0..iterations {
            let lambda = integration_bounds.lower + (i as f32) * step_size;
            let angstroms = lambda * 10.0;
            let val = self.evaluate_power(lambda);
            sum.0 += f32x4::new(
                val * x_bar(angstroms),
                val * y_bar(angstroms),
                val * z_bar(angstroms),
                0.0,
            ) * step_size;
        }
        sum
    }
}

impl SpectralPowerDistributionFunction for SPD {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        match &self {
            SPD::Linear {
                signal,
                bounds,
                mode,
            } => {
                if !bounds.contains(&lambda) {
                    return 0.0;
                }
                let step_size = bounds.span() / (signal.len() as f32);
                let index = ((lambda - bounds.lower) / step_size) as usize;
                let left = signal[index];
                let right = if index + 1 < signal.len() {
                    signal[index + 1]
                } else {
                    return signal[index];
                };
                let t = (lambda - (bounds.lower + index as f32 * step_size)) / step_size;
                // println!("t is {}", t);
                match mode {
                    InterpolationMode::Linear => (1.0 - t) * left + t * right,
                    InterpolationMode::Nearest => {
                        if t < 0.5 {
                            left
                        } else {
                            right
                        }
                    }
                    InterpolationMode::Cubic => {
                        let t2 = 2.0 * t;
                        let one_sub_t = 1.0 - t;
                        let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
                        let h01 = t * t * (3.0 - t2);
                        h00 * left + h01 * right
                    }
                }
            }
            SPD::Polynomial {
                xoffset,
                coefficients,
            } => {
                let mut val = 0.0;
                let tmp_lambda = lambda - xoffset;
                for (i, &coef) in coefficients.iter().enumerate() {
                    val += coef * tmp_lambda.powi(i as i32);
                }
                val
            }
            SPD::Tabulated { signal, mode } => {
                // let result = signal.binary_search_by_key(lambda, |&(a, b)| a);
                let index = match signal
                    .binary_search_by_key(&OrderedFloat::<f32>(lambda), |&(a, _b)| {
                        OrderedFloat::<f32>(a)
                    }) {
                    Err(index) if index > 0 => index,
                    Ok(index) | Err(index) => index,
                };
                if index == signal.len() {
                    let left = signal[index - 1];
                    return left.1;
                }
                let right = signal[index];
                let t;
                if index == 0 {
                    return right.1;
                }
                let left = signal[index - 1];
                t = (lambda - left.0) / (right.0 - left.0);

                match mode {
                    InterpolationMode::Linear => (1.0 - t) * left.1 + t * right.1,
                    InterpolationMode::Nearest => {
                        if t < 0.5 {
                            left.1
                        } else {
                            right.1
                        }
                    }
                    InterpolationMode::Cubic => {
                        let t2 = 2.0 * t;
                        let one_sub_t = 1.0 - t;
                        let h00 = (1.0 + t2) * one_sub_t * one_sub_t;
                        let h01 = t * t * (3.0 - t2);
                        h00 * left.1 + h01 * right.1
                    }
                }
            }
            SPD::Cauchy { a, b } => *a + *b / (lambda * lambda),
            SPD::Exponential { signal } => {
                let mut val = 0.0f32;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val += gaussian(
                        lambda as f64,
                        multiplier as f64,
                        offset as f64,
                        sigma1 as f64,
                        sigma2 as f64,
                    ) as f32;
                }
                val
            }
            SPD::InverseExponential { signal } => {
                let mut val = 1.0f32;
                for &(offset, sigma1, sigma2, multiplier) in signal {
                    val -= gaussian(
                        lambda as f64,
                        multiplier as f64,
                        offset as f64,
                        sigma1 as f64,
                        sigma2 as f64,
                    ) as f32;
                }
                val.max(0.0)
            }
            SPD::Blackbody { temperature, boost } => {
                if *boost == 0.0 {
                    blackbody(*temperature, lambda)
                } else {
                    boost * blackbody(*temperature, lambda)
                        / blackbody(*temperature, max_blackbody_lambda(*temperature))
                }
            }
        }
    }
    fn evaluate(&self, lambda: f32) -> f32 {
        // use the same curves as power distributions for reflectance functions, but cap it to 1.0 so no energy is ever added
        self.evaluate_power(lambda).min(1.0)
    }
}

// also known as an orthonormal basis.
#[derive(Copy, Clone, Debug)]
pub struct TangentFrame {
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub normal: Vec3,
}

impl TangentFrame {
    pub fn new(tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Self {
        debug_assert!(
            (tangent * bitangent).abs() < 0.000001,
            "tbit:{:?} * {:?} was != 0",
            tangent,
            bitangent
        );
        debug_assert!(
            (tangent * normal).abs() < 0.000001,
            "tn: {:?} * {:?} was != 0",
            tangent,
            normal
        );
        debug_assert!(
            (bitangent * normal).abs() < 0.000001,
            "bitn:{:?} * {:?} was != 0",
            bitangent,
            normal
        );
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: bitangent.normalized(),
            normal: normal.normalized(),
        }
    }
    pub fn from_tangent_and_normal(tangent: Vec3, normal: Vec3) -> Self {
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: tangent.normalized().cross(normal.normalized()).normalized(),
            normal: normal.normalized(),
        }
    }

    pub fn from_normal(normal: Vec3) -> Self {
        // let n2 = Vec3::from_raw(normal.0 * normal.0);
        // let (x, y, z) = (normal.x(), normal.y(), normal.z());
        let [x, y, z, _]: [f32; 4] = normal.0.into();
        let sign = (1.0 as f32).copysign(z);
        let a = -1.0 / (sign + z);
        let b = x * y * a;
        TangentFrame {
            tangent: Vec3::new(1.0 + sign * x * x * a, sign * b, -sign * x),
            bitangent: Vec3::new(b, sign + y * y * a, -y),
            normal,
        }
    }

    #[inline(always)]
    pub fn to_world(&self, v: &Vec3) -> Vec3 {
        self.tangent * v.x() + self.bitangent * v.y() + self.normal * v.z()
    }

    #[inline(always)]
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            self.tangent * (*v),
            self.bitangent * (*v),
            self.normal * (*v),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    pub time: f32,
    pub tmax: f32,
}

impl Ray {
    pub const fn new(origin: Point3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction,
            time: 0.0,
            tmax: INFINITY,
        }
    }

    pub const fn new_with_time(origin: Point3, direction: Vec3, time: f32) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax: INFINITY,
        }
    }
    pub const fn new_with_time_and_tmax(
        origin: Point3,
        direction: Vec3,
        time: f32,
        tmax: f32,
    ) -> Self {
        Ray {
            origin,
            direction,
            time,
            tmax,
        }
    }
    pub fn with_tmax(mut self, tmax: f32) -> Self {
        self.tmax = tmax;
        self
    }
    pub fn at_time(mut self, time: f32) -> Self {
        self.origin = self.point_at_parameter(time);
        self
    }
    pub fn point_at_parameter(self, time: f32) -> Point3 {
        self.origin + self.direction * time
    }
}

impl Default for Ray {
    fn default() -> Self {
        Ray::new(Point3::default(), Vec3::default())
    }
}

#[derive(Debug, Clone, Default)]
pub struct CDF {
    pub pdf: SPD,
    pub cdf: SPD,
    pub cdf_integral: f32,
}

impl SpectralPowerDistributionFunction for CDF {
    fn evaluate(&self, lambda: f32) -> f32 {
        self.pdf.evaluate(lambda)
    }
    fn evaluate_power(&self, lambda: f32) -> f32 {
        self.pdf.evaluate_power(lambda)
    }
}

impl From<SPD> for CDF {
    fn from(curve: SPD) -> Self {
        match &curve {
            SPD::Linear {
                signal,
                bounds,
                mode,
            } => {
                let mut cdf_signal = signal.clone();
                let mut s = 0.0;
                let step_size = bounds.span() / (signal.len() as f32);
                for (i, v) in signal.iter().enumerate() {
                    s += v * step_size;
                    cdf_signal[i] = s;
                }
                // println!("integral is {}, step_size was {}", s, step_size);
                CDF {
                    pdf: curve.clone(),
                    cdf: SPD::Linear {
                        signal: cdf_signal,
                        bounds: *bounds,
                        mode: *mode,
                    },
                    cdf_integral: s,
                }
            }
            _ => {
                let mut cdf_signal = Vec::new();
                let mut s = 0.0;
                let num_samples = 400;
                let step_size = BOUNDED_VISIBLE_RANGE.span() / (num_samples as f32);
                for i in 0..num_samples {
                    let lambda = BOUNDED_VISIBLE_RANGE.lower + (i as f32) * step_size;
                    s += curve.evaluate_power(lambda);
                    // println!("lambda: {}, s: {}", lambda, s);
                    cdf_signal.push(s);
                }
                // println!("cdf integral was {}", s);
                CDF {
                    pdf: curve.clone(),
                    cdf: SPD::Linear {
                        signal: cdf_signal,
                        bounds: BOUNDED_VISIBLE_RANGE,
                        mode: InterpolationMode::Cubic,
                    },
                    cdf_integral: s,
                }
                // CDF {
                // pdf: curve.clone(),
                // cdf: curve.clone(),
                // cdf_integral: curve.evaluate_integral(EXTENDED_VISIBLE_RANGE, 1.0),,
            }
        }
    }
}
