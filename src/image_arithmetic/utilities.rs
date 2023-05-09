use std::collections::HashSet;

use super::Point;
use image::{ImageBuffer, Pixel, Rgb};
use num_traits::{FromPrimitive, ToPrimitive};

pub fn generate_color(num: usize) -> Rgb<u8> {
    let i = num + 1;
    return Rgb([
        ((i * 98) % 255) as u8,
        ((i * 57) % 255) as u8,
        ((i * 157) % 255) as u8,
    ]);
}

pub fn generate_unique_color(num: usize) -> Rgb<u8> {
    return generate_color(num);
}

pub fn mean_color<P, C>(img: &ImageBuffer<P, C>, points: &HashSet<Point>) -> P
where
    P: Pixel,
    C: std::ops::DerefMut<Target = [P::Subpixel]>,
    P::Subpixel: FromPrimitive,
{
    let summed = points.iter().map(|point| point.get_pixel(img)).fold(
        vec![0 as f64; P::CHANNEL_COUNT as usize],
        |mut acc, pixel| {
            for (i, value) in pixel.channels().iter().enumerate() {
                acc[i] += value.to_f64().unwrap();
            }
            acc
        },
    );
    let mut mean = vec![];
    for value in summed.into_iter() {
        mean.push(P::Subpixel::from_f64(value / points.len() as f64).unwrap());
    }
    return *P::from_slice(mean.as_slice());
}

pub fn fill_connected<P, C>(
    img: &mut ImageBuffer<P, C>, color: &P, sx: u32, sy: u32,
) -> HashSet<Point>
where
    P: Pixel + std::cmp::PartialEq,
    C: std::ops::DerefMut<Target = [P::Subpixel]>,
{
    let mut filled = HashSet::new();
    let mut queued = vec![];
    let original_color = img.get_pixel(sx, sy).clone();
    queued.push(Point::from((sx, sy)));
    while !queued.is_empty() {
        let point = queued.pop().unwrap();
        filled.insert(point);
        let (x, y) = point.try_into().unwrap();
        img.put_pixel(x, y, *color);
        for dx in [-1, 0, 1] {
            for dy in [-1, 0, 1] {
                if dx == dy || dx == -dy {
                    continue;
                }
                let nx = 0.max(x as i64 + dx) as u32;
                let ny = 0.max(y as i64 + dy) as u32;
                let neighbour = img.get_pixel_checked(nx, ny);
                if neighbour.is_none() {
                    continue;
                }
                let npoint = Point::from((nx, ny));
                if neighbour.unwrap() == &original_color && !filled.contains(&npoint) {
                    queued.push(npoint);
                }
            }
        }
    }
    return filled;
}
