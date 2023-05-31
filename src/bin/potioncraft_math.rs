#[macro_use]
extern crate structopt;
extern crate line_drawing;
extern crate minifb;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::ops::{AddAssign, Mul};

use lib::flatland::{Point2, Vec2};
use lib::spectral::BOUNDED_VISIBLE_RANGE;
use lib::tonemap::{sRGB, Tonemapper};
// use lib::trace::{Bounds1D, Bounds2D};
use lib::{hsv_to_rgb, rgb_to_u32, Film, SingleWavelength};

// use math::XYZColor;
use math::prelude::*;
#[allow(unused_imports)]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};

use nalgebra::{matrix, Vector};
use packed_simd::f32x2;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use structopt::StructOpt;

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 800;

const MOON_SALT: &'static str = "MoonSalt";
const SUN_SALT: &'static str = "SunSalt";
const VOID_SALT: &'static str = "VoidSalt";

const SALTS: &'static [&'static str] = &[MOON_SALT, SUN_SALT, VOID_SALT];

enum DrawMode {
    XiaolinWu,
    Midpoint,
    Bresenham,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
enum Bezier {
    Linear {
        p0: Point2,
        p1: Point2,
    },
    Quadratic {
        p0: Point2,
        p1: Point2,
        p2: Point2,
    },
    Cubic {
        p0: Point2,
        p1: Point2,
        p2: Point2,
        p3: Point2,
    },
}

impl Bezier {
    pub fn beginning(&self) -> Point2 {
        self.eval(0.0)
    }
    pub fn end(&self) -> Point2 {
        self.eval(1.0)
    }
    pub fn eval(&self, t: f32) -> Point2 {
        let one_t = 1.0 - t;
        let one_t_2 = one_t * one_t;
        let one_t_3 = one_t_2 * one_t;
        let t_2 = t * t;
        let t_3 = t_2 * t;
        match self {
            &Bezier::Linear { p0, p1 } => p0 + t * (p1 - p0),
            &Bezier::Quadratic { p0, p1, p2 } => p1 + one_t_2 * (p0 - p1) + t * t * (p2 - p1),
            &Bezier::Cubic { p0, p1, p2, p3 } => Point2::from_raw(
                one_t_3 * p0.0 + 3.0 * one_t_2 * t * p1.0 + 3.0 * one_t * t_2 * p2.0 + t_3 * p3.0,
            ),
        }
    }
}

#[derive(Clone, Debug)]
enum IngredientType {
    Fire,
    Air,
    Earth,
    Water,
}

impl From<Point2> for IngredientType {
    fn from(point: Point2) -> Self {
        const PI: f32 = std::f32::consts::PI;
        let [x, y]: [f32; 2] = point.0.into();
        let angle = y.atan2(x) / PI;
        // angle is now -1 to 1. 0 represents 100% to the right
        match angle {
            _ if angle.abs() < 0.25 => IngredientType::Water,
            _ if angle.abs() > 0.75 => IngredientType::Fire,
            _ if angle > 0.0 => IngredientType::Air,
            _ if angle < 0.0 => IngredientType::Earth,
            _ => {
                unreachable!()
            }
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct Ingredient {
    pub name: String,
    pub price: f32,
    pub list: Vec<Bezier>,
}

impl Ingredient {
    pub fn new(name: String, price: f32, list: Vec<Bezier>) -> Self {
        for (bezier0, bezier1) in list.iter().zip(list.iter().skip(1)) {
            // beziers should be arranged roughly head to tail.
            assert!(
                (bezier0.end() - bezier1.beginning()).norm_squared() < 0.00001,
                "{:?} {:?}",
                bezier0.end(),
                bezier1.beginning()
            );
        }
        Ingredient { name, price, list }
    }

    pub fn eval(&self, t: f32) -> Point2 {
        let f32_len = self.list.len() as f32;
        let t = t.clamp(0.0, 1.0 - f32::EPSILON);
        let bin = (t * f32_len) as usize;
        let adjusted_t = (t - (bin as f32 / f32_len)) * f32_len;

        // let mut point = Point2::ZERO;
        // for i in 0..bin {
        //     point += Vec2::from_raw(self.list[i].end().0);
        // }
        // point += Vec2(self.list[bin].eval(adjusted_t).0);
        let point = self.list[bin].eval(adjusted_t);
        point
    }

    pub fn eval_vec(&self, t: f32) -> Vec2 {
        Vec2::from_raw(self.eval(t).0)
    }
}

#[derive(Copy, Clone, Default, Debug)]
struct Cost {
    pub gold: f32,
    pub void_salt: usize,
    pub moon_salt: usize,
    pub sun_salt: usize,
}

impl Cost {
    pub fn as_array(self) -> [f32; 4] {
        [
            self.gold,
            self.void_salt as f32,
            self.moon_salt as f32,
            self.sun_salt as f32,
        ]
    }
    pub fn resolve(self, ingredients: &HashMap<String, Cost>) -> f32 {
        // resolves all salts to gold, dividing salt amounts by 5000 and multiplying by recipe cost.
        let as_array = self.as_array();
        let mut total_cost = as_array[0];
        let as_slice = &as_array[1..];
        for (name, count) in SALTS.iter().zip(as_slice.iter()) {
            let cost = ingredients[*name];
            // assuming the recursive cost for salts has already been solved, and that the cost for each salt is given in gold
            total_cost += cost.gold * *count;
        }
        total_cost
    }
}

impl AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        self.gold += rhs.gold;
        self.void_salt += rhs.void_salt;
        self.moon_salt += rhs.moon_salt;
        self.sun_salt += rhs.sun_salt;
    }
}

impl Mul<usize> for Cost {
    type Output = Cost;
    fn mul(self, rhs: usize) -> Self::Output {
        Self {
            gold: self.gold * rhs as f32,
            void_salt: self.void_salt * rhs,
            moon_salt: self.moon_salt * rhs,
            sun_salt: self.sun_salt * rhs,
        }
    }
}

#[derive(Deserialize, Serialize, Clone)]
struct Potion {
    pub ingredients: HashMap<String, usize>,
}

impl Potion {
    pub fn get_total_cost(&self, ingredients_cost: &HashMap<String, Cost>) -> Cost {
        let mut cost = Cost::default();
        for (name, count) in &self.ingredients {
            match name.as_str() {
                MOON_SALT => {
                    cost += Cost {
                        moon_salt: *count,
                        ..Default::default()
                    };
                }
                SUN_SALT => {
                    cost += Cost {
                        sun_salt: *count,
                        ..Default::default()
                    };
                }
                VOID_SALT => {
                    cost += Cost {
                        void_salt: *count,
                        ..Default::default()
                    };
                }
                name => {
                    cost += ingredients_cost
                        .get(name)
                        .cloned()
                        .unwrap_or(Cost::default())
                        * *count;
                }
            }
        }
        cost
    }
}

#[derive(Deserialize, Serialize, Clone)]
struct PotionLib {
    potions: HashMap<String, Potion>,
    secondary_ingredients: HashMap<String, Vec<String>>,
}

use std::io::{BufWriter, Write};
impl PotionLib {
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) {
        let string = toml::to_string(self).unwrap();
        let mut file = BufWriter::new(
            File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&path)
                .unwrap(),
        );
        file.write_all(string.as_bytes()).unwrap();
        file.flush().unwrap();
    }

    pub fn get_total_cost_of(
        &self,
        k: &str,
        cache: &mut HashMap<String, Cost>,
        printout: bool,
    ) -> Cost {
        if let Some(this_cost) = cache.get(k) {
            if printout {
                println!("cache hit,  adding cost of {}, {:?}", k, this_cost);
            }
            return *this_cost;
        }
        if printout {
            println!("calculating cost of {}", k);
        }
        let mut cost = Cost::default();

        if let Some(potion) = self.potions.get(k) {
            let this_cost = potion.get_total_cost(cache);

            cache.insert(k.to_string(), this_cost);
            if printout {
                println!("potion: adding cost of {}, {:?}", k, this_cost);
            }

            cost += this_cost;
        } else if let Some(ingredients) = self.secondary_ingredients.get(k) {
            for ingredient in ingredients {
                // can't allow loops, i.e. Moon salt directly requiring Moon salt. doesn't make sense.
                assert!(k != ingredient.as_str());
                if let Some(this_cost) = cache.get(ingredient) {
                    if printout {
                        println!("cache hit,  adding cost of {}, {:?}", ingredient, this_cost);
                    }
                    cost += *this_cost;
                } else {
                    if printout {
                        print!("cache miss ");
                    }
                    let this_cost = self.get_total_cost_of(ingredient.as_str(), cache, printout);
                    cost += this_cost;
                }
            }
        }
        if printout {
            println!("returning cost of {} as {:?}", k, cost);
        }
        cache.insert(k.to_string(), cost);
        cost
    }
}

struct Path {
    // path is made up of curve fragments
    // each curve fragment is a curve + some max "time" for each fragment.
    pub fragments: Vec<(Ingredient, f32)>,
    // we also need the current "time", and the end "time" for this fragment after which we switch to a new fragment.
    pub current_time: f32,
    pub current_base_position: Point2,
    pub current_base_time: f32,
    pub next_fragment_time: f32,
}

impl Path {
    pub fn new(base_position: Point2, curves: Vec<Ingredient>, terminators: Vec<f32>) -> Self {
        assert!(curves.len() > 0 && terminators.len() == curves.len());
        let first_time = *terminators.first().unwrap();
        Path {
            fragments: curves
                .iter()
                .cloned()
                .zip(terminators.iter())
                .map(|(e0, &e1)| (e0, e1))
                .collect::<Vec<(Ingredient, f32)>>(),
            current_time: 0.0,
            current_base_position: base_position,
            current_base_time: 0.0,
            next_fragment_time: first_time,
        }
    }
    pub fn eval(&self, time: f32) -> Point2 {
        assert!(time >= self.current_time);
        let mut pos = self.current_base_position;
        let mut offset = time - self.current_base_time;
        for (curve, terminator) in &self.fragments {
            if offset < *terminator {
                pos += curve.eval_vec(offset);
                break;
            } else {
                pos += curve.eval_vec(*terminator);
                offset -= *terminator;
            }
        }
        pos
    }
    pub fn current_position(&self) -> Point2 {
        let offset = self.current_time - self.current_base_time;
        let fragment = self.fragments.first().unwrap();
        self.current_base_position + fragment.0.eval_vec(offset)
    }
    pub fn advance(&mut self, delta: f32) {
        // advance the current time.
        // if we tick over the fragment border, update current_base_position and current_base_time
        // and remove the current fragment.
        // else, just update current_Time.
        if self.current_time + delta > self.next_fragment_time {
            if self.fragments.len() == 1 {
                self.current_time = self.next_fragment_time;
                return;
            }
            // calculate end of this fragment and add to current_base_position
            let frag = self.fragments.first().unwrap();
            self.current_base_position += frag.0.eval_vec(frag.1);
            self.current_base_time = self.next_fragment_time;
            let _ = self.fragments.remove(0);
            self.next_fragment_time = self.current_base_time + self.fragments[0].1;
            self.current_time += delta;
        } else {
            self.current_time += delta;
        }
    }
    pub fn as_potion(&self) -> Potion {
        let mut ingredients = HashMap::new();
        for (ingredient, _) in &self.fragments {
            *ingredients.entry(ingredient.name.clone()).or_insert(0usize) += 1;
        }
        Potion { ingredients }
    }
}

#[derive(Copy, Clone)]
struct Scout {
    pub pos: Point2,
    color: f32,
}

impl Scout {
    pub fn new(color: f32) -> Self {
        Scout {
            pos: Point2::ZERO,
            color,
        }
    }

    pub fn update(&mut self, dv: Vec2) {
        loop {
            if random::<f32>() < 0.05 {
                // 40% chance of going towards the origin
                self.pos -= (self.pos - Point2::ZERO) * 0.06;
            } else {
                break;
            }
        }
        self.pos += dv;
    }

    pub fn reset(&mut self) {
        self.pos = Point2::ZERO;
    }
}

fn parse_point(string: &str) -> Option<Point2> {
    let pt = string.split_once(",").map(|(a, b)| {
        (
            a.trim().parse::<f32>().unwrap(),
            b.trim().parse::<f32>().unwrap(),
        )
    });
    pt.map(|e| Point2(f32x2::from([e.0, e.1])))
}

fn parse_ingredients_05<P>(filepath: P) -> Result<Vec<Ingredient>, ()>
where
    P: AsRef<std::path::Path>,
{
    let mut file = File::open(filepath).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).map_err(|_| {})?;
    let mut lines = buf.lines();
    // chomp head(er) off
    let _ = lines.next();

    let mut ingredients = Vec::new();
    loop {
        let ingredient_header = lines.next();
        let price_header = lines.next();
        let length_header = lines.next();
        let end_point_header = lines.next();
        let path = lines.next();

        if let (Some(name), Some(path)) = (ingredient_header, path) {
            let name = &name.split(":").nth(1).unwrap()[1..];
            print!("parsing {}: ", name);
            let curve_data = &path[16..];
            let curve_segments = curve_data.split("C ").collect::<Vec<_>>();

            let mut point = Point2::ZERO;
            let mut data: Vec<Bezier> = Vec::new();
            for curve_segment in curve_segments {
                let mut points = curve_segment
                    .split(" ")
                    .filter(|e| *e != "")
                    .collect::<Vec<_>>();
                let bezier;
                match points.len() + 1 {
                    2 => {
                        let p1 = parse_point(points.pop().unwrap()).unwrap();
                        let p0 = point;

                        bezier = Bezier::Linear { p0, p1 };
                        point = p1;
                    }
                    3 => {
                        let p2 = parse_point(points.pop().unwrap()).unwrap();
                        let p1 = parse_point(points.pop().unwrap()).unwrap();
                        let p0 = point;
                        bezier = Bezier::Quadratic { p0, p1, p2 };
                        point = p2;
                    }
                    4 => {
                        let p3 = parse_point(points.pop().unwrap()).unwrap();
                        let p2 = parse_point(points.pop().unwrap()).unwrap();
                        let p1 = parse_point(points.pop().unwrap()).unwrap();
                        let p0 = point;
                        bezier = Bezier::Cubic { p0, p1, p2, p3 };
                        point = p3;
                    }
                    _ => {
                        panic!(
                            "panik!, points len was {}, points was {:?}",
                            points.len(),
                            points
                        );
                    }
                }
                data.push(bezier);
            }

            #[rustfmt::skip]
            let end_point = parse_point(
                end_point_header
                    .map(|s| s.split(": "))
                    .map(|mut s| s.nth(1)).flatten()
                    .map(|s| s.trim_start_matches('(').trim_end_matches(')'))
                    .unwrap(),
            ).unwrap();
            let ingredient_type = IngredientType::from(end_point);
            let price = price_header
                .unwrap()
                .split(": ")
                .nth(1)
                .unwrap()
                .parse::<f32>()
                .unwrap()
                * 4.0;

            let name = String::from(name);
            println!("done! type was {:?}", ingredient_type);
            ingredients.push(Ingredient::new(name, price, data));
        } else {
            break;
        }
    }

    Ok(ingredients)
}

fn get_or_create_potion_lib<P: AsRef<std::path::Path>>(path: P) -> PotionLib {
    let mut salts = HashMap::new();
    salts.insert(String::from(MOON_SALT), vec![]);
    salts.insert(String::from(SUN_SALT), vec![]);
    salts.insert(String::from(VOID_SALT), vec![]);
    let mut potion_lib = PotionLib {
        potions: HashMap::new(),
        secondary_ingredients: salts,
    };

    if let Ok(mut file) = File::open(path) {
        let mut buf = String::new();
        if file.read_to_string(&mut buf).is_ok() {
            let parsed = toml::from_str(buf.as_str());
            potion_lib = parsed.unwrap_or(potion_lib);
        }
    }
    potion_lib
}

fn solve_recursive_salt_costs(salts: &[&str], ingredients: &mut HashMap<String, Cost>) {
    let mut matrix = nalgebra::Matrix4::<f32>::zeros();
    matrix.set_row(0, &matrix![1.0, 0.0, 0.0, 0.0]);
    for (i, salt) in salts.iter().enumerate() {
        let mut row = ingredients.get(*salt).unwrap().clone().as_array();
        row[i + 1] -= 5000.0;
        matrix.set_row(i + 1, &row.into());
        // rows.push(row);
    }
    let inverse = matrix.try_inverse().unwrap();
    let solution = inverse * matrix![1.0, 1.0, 1.0, 1.0].transpose();
    for (i, salt) in salts.iter().enumerate() {
        println!("solved salt {}, cost = {}", salt, solution[i + 1]);
        *ingredients.get_mut(*salt).unwrap() = Cost {
            gold: solution[1 + i],
            ..Default::default()
        };
    }
}

enum Metric {
    L2,
    L1,
    Gold,
}

impl Metric {
    pub fn from(string: &str) -> Option<Self> {
        let s = string.as_ref();
        match s {
            "l1" => Some(Metric::L1),
            "l2" => Some(Metric::L2),
            "gold" => Some(Metric::Gold),
            _ => None,
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long, default_value = "potions.toml")]
    pub potion_lib: String,
    // possible values: Gold, IngL2, IngL1
    #[structopt(long, default_value = "gold")]
    pub metric: String,
}

fn main() {
    let opts: Opt = StructOpt::from_args();
    let metric = Metric::from(&opts.metric).unwrap();
    let threads = 1;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let mut window = Window::new(
        "PotionCraft Path Simulator",
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

    let relative_view_bounds =
        Bounds2D::new(Bounds1D::new(-20.0, 20.0), Bounds1D::new(20.0, -20.0));
    let (_box_width, _box_height) = (
        relative_view_bounds.x.span() / width as f32,
        relative_view_bounds.y.span() / height as f32,
    );

    // a Path is made up of curve fragments
    // a path is parameterized by some time t: f32
    // various effects can occur during Path traversal, including:
    // * Dilution(start_time: f32, distance: f32): pushes the current position towards the origin by some distance, and rotating back to normal if the bottle is at all rotated
    // * Whirlpool dragging: rotates the current position around a point while also pushing it towards the spiral center
    // * Drag patches: changes how much the position moves per unit time. i.e. in a drag patch, an ingredient's path will effectively be scaled down by 2 while the position is inside the patch.
    // * sun salt and moon salt (f32: radians): rotates the bottle and the path around the bottle. a rotation of 2pi radians requires 1000 salt. moon salt rotates counter clockwise, and sun salt clockwise.

    // currently only dilution is implemented through scouts.

    let mut view_offset = Vec2::new(0.0, 0.0);
    let draw_mode = DrawMode::Midpoint;
    let mut selected_ingredient = 5usize;

    let ingredients = parse_ingredients_05("data/potioncraft_ingredients_0.5.txt").unwrap();

    let max_ingredient_id = ingredients.len();
    let mut ingredients_cost = ingredients
        .iter()
        .map(|e| {
            (
                e.name.clone(),
                Cost {
                    gold: e.price,
                    ..Cost::default()
                },
            )
        })
        .collect::<HashMap<String, Cost>>();

    let mut potion_lib: PotionLib = get_or_create_potion_lib(&opts.potion_lib);

    for salt in [SUN_SALT, MOON_SALT, VOID_SALT] {
        let _ = potion_lib.get_total_cost_of(salt, &mut ingredients_cost, true);
    }
    solve_recursive_salt_costs(SALTS, &mut ingredients_cost);
    println!(
        "cost of inspiration potion: {:?}",
        potion_lib
            .get_total_cost_of("Inspiration", &mut ingredients_cost, true)
            .resolve(&ingredients_cost)
    );
    println!(
        "cost of hallucination potion: {:?}",
        potion_lib
            .get_total_cost_of("Hallucinations", &mut ingredients_cost, true)
            .resolve(&ingredients_cost)
    );
    for salt in [SUN_SALT, MOON_SALT, VOID_SALT] {
        let cost = potion_lib.get_total_cost_of(salt, &mut ingredients_cost, false);
        println!("cost of salt {} was {:?}", salt, cost);
    }

    let mut path_curves = Vec::new();
    let mut path_terminators = Vec::new();
    let mut grind_level = 0.0f32;

    path_curves.push(ingredients[3].clone());
    path_terminators.push(1.0);

    let mut path = Path::new(Point2::ZERO, path_curves.clone(), path_terminators.clone());
    let mut pos = Point2::ZERO;

    // let mut scouts = vec![Scout::new(); 1000];
    let mut scouts = Vec::new();
    for _ in 0..1000 {
        // let color = hsv_to_rgb((random::<f32>() * 360.0) as usize, 1.0, 1.0);
        // let color = rgb_to_u32(color.0, color.1, color.2);
        let color = BOUNDED_VISIBLE_RANGE.sample(random::<f32>());
        scouts.push(Scout::new(color));
    }
    let mut scouts_clone = scouts.clone();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut reset_scouts = false;
        let mut reset_screen = false;
        let keys = window.get_keys_pressed(KeyRepeat::Yes);
        let config_move = if window.is_key_down(Key::NumPadPlus) {
            1.0
        } else if window.is_key_down(Key::NumPadMinus) {
            -1.0
        } else {
            0.0
        };
        for key in keys {
            match key {
                Key::Tab => {
                    // change selected ingredient
                    selected_ingredient += 1;
                    if selected_ingredient >= max_ingredient_id {
                        selected_ingredient = 0;
                    }
                    println!(
                        "selected ingredient is {}, name of {}",
                        selected_ingredient, &ingredients[selected_ingredient].name
                    );
                }
                Key::Space => {
                    // add selected ingredient to the pot (path)
                    path_curves.push(ingredients[selected_ingredient].clone());
                    // let grind_bounds = grind_levels[selected_ingredient];
                    let grind_bounds = (0.0, 1.0);
                    path_terminators
                        .push(Bounds1D::new(grind_bounds.0, grind_bounds.1).lerp(grind_level));
                    grind_level = 0.0;
                    reset_scouts = true;
                    reset_screen = true;
                }
                Key::G => {
                    // grind selected ingredient.
                    grind_level += 0.05;
                    if grind_level > 1.0 {
                        grind_level = 1.0;
                    }
                    println!("grind level is {}", grind_level);
                }
                Key::Up => {
                    // move view Up
                    view_offset += Vec2(f32x2::new(0.0, 0.1));
                    reset_screen = true;
                    reset_scouts = true;
                }
                Key::Down => {
                    // move view Down
                    view_offset += Vec2(f32x2::new(0.0, -0.1));
                    reset_screen = true;
                    reset_scouts = true;
                }
                Key::Left => {
                    // move view Left
                    view_offset += Vec2(f32x2::new(-0.1, 0.0));
                    reset_screen = true;
                    reset_scouts = true;
                }
                Key::Right => {
                    // move view Right
                    view_offset += Vec2(f32x2::new(0.1, 0.0));
                    reset_screen = true;
                    reset_scouts = true;
                }
                Key::R => {
                    //reset ingredients to only the selected one
                    path_curves.clear();
                    path_terminators.clear();
                    path_curves.push(ingredients[selected_ingredient].clone());
                    // let grind_bounds = grind_levels[selected_ingredient];
                    let grind_bounds = (0.0, 1.0);
                    path_terminators
                        .push(Bounds1D::new(grind_bounds.0, grind_bounds.1).lerp(grind_level));
                    grind_level = 0.0;
                    reset_scouts = true;
                    reset_screen = true;
                }
                Key::Q => {
                    // reset position and all scouts, and clear the screen
                    reset_scouts = true;
                    reset_screen = true;
                }
                _ => {}
            }
        }
        if reset_scouts {
            scouts.iter_mut().for_each(|e| e.reset());
            scouts_clone = scouts.clone();
        }
        if reset_screen {
            pos = Point2::ZERO;
            path = Path::new(Point2::ZERO, path_curves.clone(), path_terminators.clone());

            film.buffer.fill(XYZColor::BLACK);
        }

        path.advance(0.01);
        let dv = path.current_position() - pos;
        // println!("{:?}", scouts[0].pos);

        pos += dv;
        scouts.iter_mut().for_each(|e| e.update(dv));

        for (i, scout) in scouts.iter().enumerate() {
            // let dp = scout.pos - scouts_clone[i].pos;
            // let pos = scout.pos() - view_offset;
            // let (px0, py0) = (
            //     (WINDOW_WIDTH as f32 * (pos.x() - relative_view_bounds.x.lower)
            //         / relative_view_bounds.x.span()) as usize,
            //     (WINDOW_HEIGHT as f32 * (pos.y() - relative_view_bounds.y.lower)
            //         / relative_view_bounds.y.span()) as usize,
            // );
            // film.buffer[py0 as usize * width + px0 as usize] +=
            //     XYZColor::from(SingleWavelength::new(scout.color, 10.0.into()));
            if true {
                let line = (
                    scouts_clone[i].pos - view_offset,
                    scout.pos - view_offset,
                    XYZColor::from(SingleWavelength::new(scout.color, 1.0.into())),
                );
                let (px0, py0) = (
                    (WINDOW_WIDTH as f32 * (line.0.x() - relative_view_bounds.x.lower)
                        / relative_view_bounds.x.span()) as usize,
                    (WINDOW_HEIGHT as f32 * (line.0.y() - relative_view_bounds.y.lower)
                        / relative_view_bounds.y.span()) as usize,
                );
                let (px1, py1) = (
                    (WINDOW_WIDTH as f32 * (line.1.x() - relative_view_bounds.x.lower)
                        / relative_view_bounds.x.span()) as usize,
                    (WINDOW_HEIGHT as f32 * (line.1.y() - relative_view_bounds.y.lower)
                        / relative_view_bounds.y.span()) as usize,
                );

                let (dx, dy) = (px1 as isize - px0 as isize, py1 as isize - py0 as isize);
                if dx == 0 && dy == 0 {
                    if px0 as usize >= WINDOW_WIDTH || py0 as usize >= WINDOW_HEIGHT {
                        continue;
                    }
                    film.buffer[py0 as usize * width + px0 as usize] = line.2;
                    continue;
                }
                let b = (dx as f32).hypot(dy as f32) / (dx.abs().max(dy.abs()) as f32);
                match draw_mode {
                    DrawMode::Midpoint => {
                        for (x, y) in line_drawing::Midpoint::<f32, isize>::new(
                            (px0 as f32, py0 as f32),
                            (px1 as f32, py1 as f32),
                        ) {
                            if x as usize >= WINDOW_WIDTH
                                || y as usize >= WINDOW_HEIGHT
                                || x < 0
                                || y < 0
                            {
                                continue;
                            }
                            assert!(!b.is_nan(), "{} {}", dx, dy);
                            film.buffer[y as usize * width + x as usize] = line.2;
                        }
                    }
                    DrawMode::XiaolinWu => {
                        // let b = 1.0f32;
                        for ((x, y), a) in line_drawing::XiaolinWu::<f32, isize>::new(
                            (px0 as f32, py0 as f32),
                            (px1 as f32, py1 as f32),
                        ) {
                            if x as usize >= WINDOW_WIDTH
                                || y as usize >= WINDOW_HEIGHT
                                || x < 0
                                || y < 0
                            {
                                continue;
                            }
                            assert!(!b.is_nan(), "{} {}", dx, dy);
                            film.buffer[y as usize * width + x as usize] = line.2 * a;
                        }
                    }
                    DrawMode::Bresenham => {
                        for (x, y) in line_drawing::Bresenham::new(
                            (px0 as isize, py0 as isize),
                            (px1 as isize, py1 as isize),
                        ) {
                            if x as usize >= WINDOW_WIDTH
                                || y as usize >= WINDOW_HEIGHT
                                || x < 0
                                || y < 0
                            {
                                continue;
                            }
                            assert!(!b.is_nan(), "{} {}", dx, dy);
                            film.buffer[y as usize * width + x as usize] += line.2;
                        }
                    }
                }
            }
        }

        scouts_clone = scouts.clone();

        let srgb_tonemapper = sRGB::new(&film, 1.0);
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

    path_curves.clear();
    path_terminators.clear();
    path_curves.extend(ingredients.iter().cloned());
    path_terminators.resize(path_curves.len(), 1.0);

    potion_lib.potions.insert(
        String::from("Everything"),
        Path::new(Point2::ZERO, path_curves.clone(), path_terminators.clone()).as_potion(),
    );

    potion_lib.save_to_file(opts.potion_lib);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_beziers() {
        let linear_bezier = Bezier::Linear {
            p0: Point2::new(0.0, 0.0),
            p1: Point2::new(-1.25, 1.0),
        };

        println!("linear bezier");
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let p = linear_bezier.eval(t);
            println!("{:?}", p);
        }

        let quadratic_bezier = Bezier::Quadratic {
            p0: Point2::new(1.0, 0.0),
            p1: Point2::new(0.0, 1.0),
            p2: Point2::new(-1.0, 0.0),
        };

        println!("quadratic bezier");
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let p = quadratic_bezier.eval(t);
            println!("{:?}", p);
        }

        let cubic_bezier = Bezier::Cubic {
            p0: Point2::new(1.0, 1.0),
            p1: Point2::new(-1.0, 1.0),
            p2: Point2::new(-1.0, -1.0),
            p3: Point2::new(1.0, -1.0),
        };

        println!("cubic bezier");
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let p = cubic_bezier.eval(t);
            println!("{:?}", p);
        }
    }

    #[test]
    fn test_curve() {
        let curve0 = Ingredient::new(
            String::from("Firebell"),
            0.0,
            vec![
                Bezier::Linear {
                    p0: Point2::new(0.0, 0.0),
                    p1: Point2::new(-1.25, 1.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-1.25, 1.0),
                    p1: Point2::new(-2.5, 0.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-2.5, 0.0),
                    p1: Point2::new(-3.75, 1.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-3.75, 1.0),
                    p1: Point2::new(-5.0, 0.0),
                },
            ],
        );

        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let p = curve0.eval(t);
            println!("{:?}", p);
        }
    }

    #[test]
    fn test_path() {
        let curve0 = Ingredient::new(
            String::from("firebell"),
            0.0,
            vec![
                Bezier::Linear {
                    p0: Point2::new(0.0, 0.0),
                    p1: Point2::new(-1.25, 1.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-1.25, 1.0),
                    p1: Point2::new(-2.5, 0.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-2.5, 0.0),
                    p1: Point2::new(-3.75, 1.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-3.75, 1.0),
                    p1: Point2::new(-5.0, 0.0),
                },
            ],
        );

        let curve1 = Ingredient::new(
            String::from("firebell"),
            0.0,
            vec![
                Bezier::Linear {
                    p0: Point2::new(0.0, 0.0),
                    p1: Point2::new(-1.25, 1.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-1.25, 1.0),
                    p1: Point2::new(-2.5, 0.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-2.5, 0.0),
                    p1: Point2::new(-3.75, 1.0),
                },
                Bezier::Linear {
                    p0: Point2::new(-3.75, 1.0),
                    p1: Point2::new(-5.0, 0.0),
                },
            ],
        );

        // let path = Path::new(Point2::ZERO, vec![curve0, curve1], vec![1.0, 1.0]);
        let mut path = Path::new(Point2::ZERO, vec![curve0, curve1], vec![0.75, 0.75]);

        for i in 0..=100 {
            let t = i as f32 / 50.0;
            let p = path.eval(t);
            println!("{:?}", p);
        }

        println!("tested eval, now testing advance and position");

        // let mut t = 0.0;
        loop {
            path.advance(0.01666);
            println!("{:?}", path.current_position());
            if path.current_time == path.next_fragment_time {
                break;
            }
        }
    }
}
