use math::{Point3, Ray, Vec3};
#[derive(Copy, Clone, Debug)]
pub struct SurfaceIntersectionData {
    pub time: f32,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: usize,
    pub outer_medium_id: usize,
    pub inner_medium_id: usize,
}

impl SurfaceIntersectionData {
    pub fn new(
        time: f32,
        point: Point3,
        normal: Vec3,
        material_id: usize,
        outer_medium_id: usize,
        inner_medium_id: usize,
    ) -> Self {
        SurfaceIntersectionData {
            time,
            point,
            normal,
            material_id,
            outer_medium_id,
            inner_medium_id,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MediumIntersectionData {
    pub time: f32,
    pub point: Point3,
    pub wi: Vec3,
    pub medium_id: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum IntersectionData {
    Surface(SurfaceIntersectionData),
    Medium(MediumIntersectionData),
}

pub trait Primitive {
    fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub radius: f32,
    pub origin: Point3,
    pub material_id: usize,
    pub outer_medium_id: usize,
    pub inner_medium_id: usize,
}

impl Sphere {
    pub fn new(
        radius: f32,
        origin: Point3,
        material_id: usize,
        outer_medium_id: usize,
        inner_medium_id: usize,
    ) -> Sphere {
        Sphere {
            radius,
            origin,
            material_id,
            outer_medium_id,
            inner_medium_id,
        }
    }

    // fn solid_angle(&self, point: Point3, wi: Vec3) -> f32 {
    //     let cos_theta_max =
    //         (1.0 - self.radius * self.radius / (self.origin - point).norm_squared()).sqrt();
    //     2.0 * PI * (1.0 - cos_theta_max)
    // }
}

impl Primitive for Sphere {
    fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData> {
        let oc: Vec3 = r.origin - self.origin;
        let a = r.direction * r.direction;
        let b = oc * r.direction;
        let c = oc * oc - self.radius * self.radius;
        let discriminant = b * b - a * c;
        let discriminant_sqrt = discriminant.sqrt();
        if discriminant > 0.0 {
            let mut time: f32;
            let point: Point3;
            let normal: Vec3;
            // time = r.time + (-b - discriminant_sqrt) / a;
            time = (-b - discriminant_sqrt) / a;
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(IntersectionData::Surface(SurfaceIntersectionData::new(
                    time,
                    point,
                    normal,
                    self.material_id,
                    self.outer_medium_id,
                    self.inner_medium_id,
                )));
            }
            // time = r.time + (-b + discriminant_sqrt) / a;
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(IntersectionData::Surface(SurfaceIntersectionData::new(
                    time,
                    point,
                    normal,
                    self.material_id,
                    self.outer_medium_id,
                    self.inner_medium_id,
                )));
            }
        }
        None
    }
}

unsafe impl Send for Sphere {}
unsafe impl Sync for Sphere {}
