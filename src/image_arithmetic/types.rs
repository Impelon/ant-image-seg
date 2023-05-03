use std::ops::{Add, Deref, DerefMut};

use image::{ImageBuffer, Pixel, Rgb};
use rand::seq::IteratorRandom;

pub type ColorSpaceDistance = dyn Fn(&Rgb<u8>, &Rgb<u8>) -> f64;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Point {
    pub x: i64,
    pub y: i64,
}

impl Point {
    pub fn spawn<R: rand::Rng>(rng: &mut R, width: u32, height: u32) -> Self {
        return Self {
            x: (0..width as i64).choose(rng).unwrap(),
            y: (0..height as i64).choose(rng).unwrap(),
        };
    }

    pub const fn neighbourhood_directions() -> &'static [Self] {
        return &[
            Self { x: 1, y: 0 },
            Self { x: -1, y: 0 },
            Self { x: 0, y: -1 },
            Self { x: 0, y: 1 },
            Self { x: 1, y: -1 },
            Self { x: 1, y: 1 },
            Self { x: -1, y: 1 },
            Self { x: -1, y: -1 },
        ];
    }

    pub fn iterate_neighbourhood(self) -> impl Iterator<Item = Point> {
        return Self::neighbourhood_directions().iter().map(move |dir| self + *dir);
    }

    pub fn is_within_rectangle(self, a: &Self, b: &Self) -> bool {
        let min_x = a.x.min(b.x);
        let max_x = a.x.max(b.x);
        let min_y = a.y.min(b.y);
        let max_y = a.y.max(b.y);
        return self.x >= min_x && self.x <= max_x && self.y >= min_y && self.y <= max_y;
    }

    fn square(x: i64) -> f64 {
        (x as f64) * (x as f64)
    }

    pub fn euclidean_squared_distance(self, other: &Self) -> f64 {
        return Self::square(other.x - self.x) + Self::square(other.y - self.y);
    }

    pub fn euclidean_distance(self, other: &Self) -> f64 {
        return self.euclidean_squared_distance(other).sqrt();
    }

    pub fn manhattan_distance(self, other: &Self) -> i64 {
        return (other.x - self.x).abs() + (other.y - self.y).abs();
    }

    // Shortcuts.
    pub fn get_pixel<P, C>(self, img: &ImageBuffer<P, C>) -> &P
    where
        P: Pixel,
        C: Deref<Target = [P::Subpixel]>,
    {
        return img.get_pixel(self.x.try_into().unwrap(), self.y.try_into().unwrap());
    }

    pub fn get_pixel_mut<P, C>(self, img: &mut ImageBuffer<P, C>) -> &mut P
    where
        P: Pixel,
        C: DerefMut<Target = [P::Subpixel]>,
    {
        return img.get_pixel_mut(self.x.try_into().unwrap(), self.y.try_into().unwrap());
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        return Self { x: self.x + other.x, y: self.y + other.y };
    }
}
