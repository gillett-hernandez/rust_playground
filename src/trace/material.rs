use crate::{
    math::{Point3, Ray, Sample2D, SpectralPowerDistributionFunction, Vec3, SPD},
    random::random_on_unit_sphere,
};
use crate::{
    math::{Sample1D, TangentFrame},
    random::random_cosine_direction,
};
use std::f32::consts::PI;
pub trait Material {
    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32);
    fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3;
    fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
        0.0
    }
}

#[derive(Clone)]
pub struct ConstLambertian {
    pub color: SPD,
}

impl ConstLambertian {
    pub fn new(color: SPD) -> ConstLambertian {
        ConstLambertian { color }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material for ConstLambertian {
    fn sample(&self, _lambda: f32, _wi: Vec3, s: Sample2D) -> Vec3 {
        random_cosine_direction(s)
    }

    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            (self.color.evaluate(lambda) / PI, cosine / PI)
        } else {
            (0.0, 0.0)
        }
    }
    // fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
    //     0.0
    // }
}

unsafe impl Send for ConstLambertian {}
unsafe impl Sync for ConstLambertian {}

#[derive(Clone)]
pub struct ConstFilm {
    pub color: SPD,
}

impl ConstFilm {
    pub fn new(color: SPD) -> ConstFilm {
        ConstFilm { color }
    }
    pub const NAME: &'static str = "Film";
}

impl Material for ConstFilm {
    fn sample(&self, _lambda: f32, wi: Vec3, _s: Sample2D) -> Vec3 {
        -wi
    }

    fn bsdf(&self, lambda: f32, _wi: Vec3, wo: Vec3) -> (f32, f32) {
        (self.color.evaluate(lambda) / wo.z().abs(), 1.0)
    }
    // fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
    //     0.0
    // }
}

unsafe impl Send for ConstFilm {}
unsafe impl Sync for ConstFilm {}

#[derive(Clone)]
pub struct ConstDiffuseEmitter {
    pub bounce_color: SPD,
    pub emission_color: SPD,
}

impl ConstDiffuseEmitter {
    pub fn new(bounce_color: SPD, emission_color: SPD) -> ConstDiffuseEmitter {
        ConstDiffuseEmitter {
            bounce_color,
            emission_color,
        }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material for ConstDiffuseEmitter {
    fn sample(&self, _lambda: f32, wi: Vec3, s: Sample2D) -> Vec3 {
        random_cosine_direction(s)
    }

    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            (self.bounce_color.evaluate(lambda) / PI, cosine / PI)
        } else {
            (0.0, 0.0)
        }
    }
    fn emission(&self, lambda: f32, wo: Vec3) -> f32 {
        self.emission_color.evaluate_power(lambda) / PI
    }
}

unsafe impl Send for ConstDiffuseEmitter {}
unsafe impl Sync for ConstDiffuseEmitter {}

pub enum MaterialEnum {
    ConstLambertian(ConstLambertian),
    ConstDiffuseEmitter(ConstDiffuseEmitter),
    ConstFilm(ConstFilm),
}

impl Material for MaterialEnum {
    fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
        match self {
            MaterialEnum::ConstLambertian(mat) => mat.sample(lambda, wi, sample),
            MaterialEnum::ConstDiffuseEmitter(mat) => mat.sample(lambda, wi, sample),
            MaterialEnum::ConstFilm(mat) => mat.sample(lambda, wi, sample),
        }
    }
    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        match self {
            MaterialEnum::ConstLambertian(mat) => mat.bsdf(lambda, wi, wo),
            MaterialEnum::ConstDiffuseEmitter(mat) => mat.bsdf(lambda, wi, wo),
            MaterialEnum::ConstFilm(mat) => mat.bsdf(lambda, wi, wo),
        }
    }
    fn emission(&self, lambda: f32, wo: Vec3) -> f32 {
        match self {
            MaterialEnum::ConstLambertian(mat) => mat.emission(lambda, wo),
            MaterialEnum::ConstDiffuseEmitter(mat) => mat.emission(lambda, wo),
            MaterialEnum::ConstFilm(mat) => mat.emission(lambda, wo),
        }
    }
}

unsafe impl Send for MaterialEnum {}
unsafe impl Sync for MaterialEnum {}

pub trait Medium {
    fn p(&self, lambda: f32, wi: Vec3, wo: Vec3) -> f32;
    fn sample_p(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> (Vec3, f32);
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool);
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32;
}

pub fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    (1.0 - g * g) / (denom * denom.sqrt() * 2.0 * std::f32::consts::TAU)
}

pub struct HenyeyGreensteinHomogeneous {
    pub g: f32,
    pub sigma_t: SPD, // transmittance attenuation
    pub sigma_s: SPD, // scattering attenuation
}

impl Medium for HenyeyGreensteinHomogeneous {
    fn p(&self, lambda: f32, wi: Vec3, wo: Vec3) -> f32 {
        let cos_theta = wi * wo;
        self.sigma_s.evaluate_power(lambda) * phase_hg(cos_theta, self.g)
    }
    fn sample_p(&self, lambda: f32, wi: Vec3, s: Sample2D) -> (Vec3, f32) {
        // just do isomorphic as a test
        let cos_theta = if self.g.abs() < 0.001 {
            1.0 - 2.0 * s.x
        } else {
            let sqr = (1.0 - self.g * self.g) / (1.0 + self.g - 2.0 * self.g * s.x);
            -(1.0 + self.g * self.g - sqr * sqr) / (2.0 * self.g)
        };

        let sin_theta = (0.0f32).max(1.0 - cos_theta * cos_theta).sqrt();
        let phi = std::f32::consts::TAU * s.y;
        let frame = TangentFrame::from_normal(wi);
        let (sin_phi, cos_phi) = phi.sin_cos();
        let wo = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        (
            frame.to_world(&wo),
            self.sigma_s.evaluate_power(lambda) * phase_hg(cos_theta, self.g),
        )
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        let sigma_t = self.sigma_t.evaluate_power(lambda);
        let dist = -(1.0 - s.x).ln() / sigma_t;
        let t = dist.min(ray.tmax);
        let sampled_medium = t < ray.tmax;

        let point = ray.point_at_parameter(t);
        let tr = self.tr(lambda, ray.origin, point);
        // could add HWSS here.
        let density = if sampled_medium { sigma_t * tr } else { tr };
        let pdf = density;
        if sampled_medium {
            (point, tr * self.sigma_s.evaluate_power(lambda) / pdf, true)
        } else {
            (point, tr / pdf, false)
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        let sigma_t = self.sigma_t.evaluate_power(lambda);
        (-sigma_t * (p1 - p0).norm()).exp()
    }
}

pub enum MediumEnum {
    HenyeyGreensteinHomogeneous(HenyeyGreensteinHomogeneous),
}

impl Medium for MediumEnum {
    fn p(&self, lambda: f32, wi: Vec3, wo: Vec3) -> f32 {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.p(lambda, wi, wo),
        }
    }
    fn sample_p(&self, lambda: f32, wi: Vec3, s: Sample2D) -> (Vec3, f32) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample_p(lambda, wi, s),
        }
    }
    fn sample(&self, lambda: f32, ray: Ray, s: Sample1D) -> (Point3, f32, bool) {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.sample(lambda, ray, s),
        }
    }
    fn tr(&self, lambda: f32, p0: Point3, p1: Point3) -> f32 {
        match self {
            MediumEnum::HenyeyGreensteinHomogeneous(inner) => inner.tr(lambda, p0, p1),
        }
    }
}

unsafe impl Send for MediumEnum {}
unsafe impl Sync for MediumEnum {}
