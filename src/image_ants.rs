//! Core functionality for ant colony algorithms on images.

use std::collections::HashSet;
use std::thread;

use super::image_arithmetic::color_distances;
use super::image_arithmetic::{generate_color, ArithmeticImage, Point};
use image::{DynamicImage, ImageBuffer, Luma, Pixel, Rgb, RgbImage, Rgba, RgbaImage};
use rand;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub type PheromoneImage = ImageBuffer<Luma<f32>, Vec<f32>>;

impl ArithmeticImage<f32> for PheromoneImage {
    fn max(&self) -> f32 {
        return self.as_raw().iter().fold(0.0, |a: f32, &b| a.max(b));
    }

    fn min(&self) -> f32 {
        return self.as_raw().iter().fold(f32::INFINITY, |a: f32, &b| a.min(b));
    }

    fn normalize(&mut self) {
        let max = self.max();
        if max != 0.0 && max != 1.0 {
            for pixel in self.pixels_mut() {
                (pixel.0)[0] /= max;
            }
        }
    }

    fn binarize(&mut self, threshold: f32) {
        self.normalize();
        for pixel in self.pixels_mut() {
            (pixel.0)[0] = ((pixel.0)[0] > threshold) as u8 as f32;
        }
    }

    fn clamp(&mut self, threshold: f32) {
        for pixel in self.pixels_mut() {
            (pixel.0)[0] = threshold.min((pixel.0)[0]);
        }
    }

    fn add(&mut self, other: &Self) {
        for (x, y, pixel) in self.enumerate_pixels_mut() {
            (pixel.0)[0] += (other.get_pixel(x, y).0)[0];
        }
    }

    fn add_scalar(&mut self, num: f32) {
        for pixel in self.pixels_mut() {
            (pixel.0)[0] += num;
            if (pixel.0)[0] < 0.0 {
                (pixel.0)[0] = 0.0;
            }
        }
    }

    fn mul(&mut self, other: &Self) {
        for (x, y, pixel) in self.enumerate_pixels_mut() {
            (pixel.0)[0] *= (other.get_pixel(x, y).0)[0];
        }
    }

    fn mul_scalar(&mut self, num: f32) {
        for pixel in self.pixels_mut() {
            (pixel.0)[0] *= num;
        }
    }
}

pub type UpdateFunction<R> =
    dyn Fn(&mut R, &RgbImage, &mut PheromoneImage, &HashSet<Point>) + Send + Sync;
pub type GlobalUpdateFunction<R> =
    dyn Fn(&mut R, &RgbImage, &mut [PheromoneImage], &HashSet<Point>) + Send + Sync;

pub struct AntColonyRules<CR: rand::Rng> {
    pub max_ant_steps: usize,
    pub ants_per_global_update: usize,
    pub ants_return: bool,
    pub parallelity: usize,
    pub initialization_funcs: Vec<Option<Box<UpdateFunction<CR>>>>,
    pub local_update_funcs: Vec<Option<Box<UpdateFunction<CR>>>>,
    pub global_update_func: Option<Box<GlobalUpdateFunction<CR>>>,
}

impl<CR: rand::Rng> AntColonyRules<CR> {
    pub fn new(
        max_ant_steps: usize, ants_per_global_update: usize, ants_return: bool,
        parallelity: Option<usize>,
        mut pheromone_functions: Vec<Vec<Option<Box<UpdateFunction<CR>>>>>,
        global_update_func: Option<Box<GlobalUpdateFunction<CR>>>,
    ) -> Result<Self, &'static str> {
        let mut pheromone_channels = 0;
        if pheromone_functions.len() > 0 {
            pheromone_channels = pheromone_functions[0].len();
        }
        if pheromone_channels <= 0 {
            return Err("no pheromones");
        }
        if pheromone_functions.len() > 2 {
            return Err("extra pheromone functions");
        }
        if pheromone_functions.iter().any(|x| x.len() != pheromone_channels) {
            return Err("unequal amount of pheromone functions");
        }
        while pheromone_functions.len() < 2 {
            let mut substitute = vec![];
            for _ in 0..pheromone_channels {
                substitute.push(None);
            }
            pheromone_functions.insert(0, substitute);
        }
        let mut parallelity =
            parallelity.unwrap_or(thread::available_parallelism().map_or(1, |x| x.get()));
        if parallelity > ants_per_global_update {
            parallelity = 1;
        }

        return Ok(Self {
            max_ant_steps,
            ants_per_global_update,
            ants_return,
            parallelity,
            global_update_func,
            local_update_funcs: pheromone_functions.pop().unwrap(),
            initialization_funcs: pheromone_functions.pop().unwrap(),
        });
    }

    pub fn channels(&self) -> usize {
        return self.initialization_funcs.len();
    }

    pub fn apply(
        rng: &mut CR, img: &RgbImage, pheromones: &mut [PheromoneImage], visited: &HashSet<Point>,
        funcs: &Vec<Option<Box<UpdateFunction<CR>>>>,
    ) {
        for i in 0..pheromones.len() {
            if let Some(update) = &funcs[i] {
                update(rng, img, pheromones.get_mut(i).unwrap(), visited);
            }
        }
    }

    pub fn initialize_pheromones(&self, rng: &mut CR, img: &RgbImage) -> Vec<PheromoneImage> {
        let mut pheromones = vec![];
        for _ in 0..self.channels() {
            pheromones.push(PheromoneImage::new(img.width(), img.height()));
        }
        Self::apply(rng, img, &mut pheromones, &HashSet::new(), &self.initialization_funcs);
        return pheromones;
    }

    pub fn local_update(
        &self, rng: &mut CR, img: &RgbImage, pheromones: &mut [PheromoneImage],
        visited: &HashSet<Point>,
    ) {
        Self::apply(rng, img, pheromones, visited, &self.local_update_funcs);
    }

    pub fn global_update(
        &self, rng: &mut CR, img: &RgbImage, pheromones: &mut [PheromoneImage],
        visited: &HashSet<Point>,
    ) {
        if let Some(update) = &self.global_update_func {
            update(rng, img, pheromones, visited);
        }
    }
}

#[derive(Debug)]
pub struct Ant {
    pub position: Point,
    pub target: Point,
    pub visited: HashSet<Point>,
}

impl Ant {
    pub fn spawn<R: rand::Rng>(rng: &mut R, width: u32, height: u32) -> Self {
        return Self {
            position: Point::spawn(rng, width, height),
            target: Point::spawn(rng, width, height),
            visited: HashSet::new(),
        };
    }

    pub fn run<R: rand::Rng, CR: rand::Rng>(
        &mut self, rng: &mut R, img: &RgbImage, rules: &AntColonyRules<CR>,
        pheromones: &[PheromoneImage],
    ) {
        let corner_a = Point { x: 0, y: 0 };
        let corner_b = Point { x: (img.width() - 1) as i64, y: (img.height() - 1) as i64 };
        let mut start = Some(self.position);
        for _ in 0..rules.max_ant_steps {
            if self.position == self.target {
                if rules.ants_return && start != None {
                    self.target = start.unwrap();
                    start = None;
                } else {
                    break;
                }
            }
            self.visited.insert(self.position);
            let dist = self.target.euclidean_distance(&self.position);
            let get_weight = |newpos: &Point| -> f32 {
                if !newpos.is_within_rectangle(&corner_a, &corner_b) {
                    return 0.0;
                }
                let mut weight = 0.1;
                // Follow pheromones.
                for pheromone in pheromones {
                    let strength = newpos.get_pixel(pheromone).0[0];
                    if strength > 0.0 {
                        weight += strength;
                    }
                }
                // Higher probability to walk towards target.
                weight *= ((dist - self.target.euclidean_distance(&newpos)) as f32) + 3.0;
                // Walk along paths of similar color.
                let cdist =
                    color_distances::manhattan(self.position.get_pixel(img), newpos.get_pixel(img));
                weight /= 128.0 + cdist as f32;
                // Lower probability to visit pixel more than once.
                if self.visited.contains(&newpos) {
                    weight *= 0.01;
                }
                return weight;
            };
            self.position = *self
                .position
                .iterate_neighbourhood()
                .collect::<Vec<Point>>()
                .choose_weighted(rng, get_weight)
                .unwrap();
        }
        self.visited.insert(self.position);
    }
}

pub fn initialize_pheromones<CR: rand::Rng>(
    rng: &mut CR, img: &RgbImage, rules: &AntColonyRules<CR>,
) -> Vec<PheromoneImage> {
    return rules.initialize_pheromones(rng, img);
}

/// Thread-safe run of multiple ants.
/// Updates pheromones after each ant according to local rules.
/// Returns the pixels visited by each ant.
pub fn create_and_run_ants<CR: rand::Rng>(
    rng: &mut CR, img: &RgbImage, rules: &AntColonyRules<CR>, pheromones: &[PheromoneImage],
    number_of_ants: usize,
) -> (Vec<PheromoneImage>, Vec<HashSet<Point>>) {
    let mut visited_sets = vec![];
    let mut pheromones_mut = pheromones.to_vec();
    for _ in 0..number_of_ants {
        let mut ant = Ant::spawn(rng, img.width(), img.height());
        ant.run(rng, img, rules, &mut pheromones_mut);
        rules.local_update(rng, img, &mut pheromones_mut, &ant.visited);
        visited_sets.push(ant.visited);
    }
    return (pheromones_mut, visited_sets);
}

/// Run multiple ants in parallel.
/// Collects their pheromones to perform a global update afterwards.
pub fn run_colony_step<CR: rand::Rng + SeedableRng + Send>(
    rng: &mut CR, img: &RgbImage, rules: &AntColonyRules<CR>, pheromones: &mut [PheromoneImage],
) {
    let mut total_visited = HashSet::new();
    thread::scope(|scope| {
        let mut ants_left = rules.ants_per_global_update;
        let mut threads = vec![];
        for i in 0..rules.parallelity {
            let pheromones = pheromones.to_vec();
            let mut ants = ants_left;
            if i < rules.parallelity - 1 {
                ants = ants.min(rules.ants_per_global_update / rules.parallelity);
            }
            ants_left -= ants;
            let mut thread_rng = CR::from_rng(&mut *rng).unwrap();
            threads.push(scope.spawn(move || {
                create_and_run_ants(&mut thread_rng, &img, rules, &pheromones, ants)
            }));
        }
        while !threads.is_empty() {
            thread::yield_now();
            // Find available threads to join.
            let (finished, unfinished): (Vec<_>, Vec<_>) =
                threads.into_iter().partition(|join_handle| join_handle.is_finished());
            // Combine pheromones and visited pixels.
            for join_handle in finished.into_iter() {
                let (part_pheromones, part_visited_sets) = join_handle.join().unwrap();
                part_pheromones
                    .into_iter()
                    .zip(pheromones.iter_mut())
                    .for_each(|(part, total)| total.add(&part));
                part_visited_sets.into_iter().for_each(|visited| total_visited.extend(visited));
            }
            threads = unfinished;
        }
    });
    // Finished combining partial results, can run global rules now.
    rules.global_update(rng, img, pheromones, &total_visited);
}

pub fn colorize_pheromone(pheromone: &PheromoneImage, color: Rgb<u8>, max_alpha: u8) -> RgbaImage {
    let mut p = pheromone.clone();
    p.normalize();
    return RgbaImage::from_fn(p.width(), p.height(), |x, y| {
        Rgba([
            color.0[0],
            color.0[1],
            color.0[2],
            (p.get_pixel(x, y).0[0] * (max_alpha as f32)) as u8,
        ])
    });
}

pub fn visualize_pheromones(pheromones: &[PheromoneImage]) -> RgbImage {
    let peaks: Vec<_> = pheromones.iter().map(|p| p.max()).collect();
    let total: f32 = peaks.iter().sum();
    let intensities: Vec<_> = peaks.iter().map(|x| x / total).collect();
    let colorized_pheromones: Vec<_> = pheromones
        .to_vec()
        .into_iter()
        .enumerate()
        .map(|(i, p)| colorize_pheromone(&p, generate_color(i), (255.0 * intensities[i]) as u8))
        .collect();
    let result = RgbaImage::from_fn(pheromones[0].width(), pheromones[0].height(), |x, y| {
        let mut pixel = Rgba([0, 0, 0, 255]);
        for pheromone in &colorized_pheromones {
            pixel.blend(pheromone.get_pixel(x, y));
        }
        pixel
    });
    return DynamicImage::from(result).to_rgb8();
}
