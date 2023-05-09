use super::utilities;
use super::{ColorSpaceDistance, Point};

use std::collections::HashSet;

use image::{Rgb, RgbImage};

fn mean(data: &[f64]) -> f64 {
    return data.iter().sum::<f64>() / (data.len() as f64);
}

fn find_segment<'a>(
    segments: &'a Vec<HashSet<Point>>, point: &Point,
) -> Option<&'a HashSet<Point>> {
    for segment in segments {
        if segment.contains(point) {
            return Some(segment);
        }
    }
    return None;
}

pub fn segment_deviation(
    img: &RgbImage, segment: &HashSet<Point>, dist: &ColorSpaceDistance,
) -> f64 {
    let centriod = utilities::mean_color(img, segment);
    return segment.iter().map(|point| dist(point.get_pixel(img), &centriod)).sum();
}

pub fn overall_deviation(
    img: &RgbImage, segments: &Vec<HashSet<Point>>, dist: &ColorSpaceDistance,
) -> f64 {
    return mean(&segments.iter().map(|s| segment_deviation(img, s, dist)).collect::<Vec<f64>>());
}

pub fn local_edge_value(
    img: &RgbImage, segments: &Vec<HashSet<Point>>, dist: &ColorSpaceDistance, point: &Point,
) -> f64 {
    let segment = find_segment(segments, point);
    let corner_a = Point { x: 0, y: 0 };
    let corner_b = Point { x: (img.width() - 1) as i64, y: (img.height() - 1) as i64 };
    return point
        .iterate_neighbourhood()
        .map(|neighbour| {
            if segment.map_or(false, |x| x.contains(&neighbour))
                || !neighbour.is_within_rectangle(&corner_a, &corner_b)
            {
                return 0.0;
            }
            return dist(point.get_pixel(img), neighbour.get_pixel(img));
        })
        .sum();
}

pub fn edge_value(
    img: &RgbImage, segments: &Vec<HashSet<Point>>, dist: &ColorSpaceDistance,
) -> f64 {
    return img
        .enumerate_pixels()
        .map(|(x, y, _)| local_edge_value(img, segments, dist, &(x, y).into()))
        .sum();
}

pub fn local_connectivity_measure(
    img: &RgbImage, segments: &Vec<HashSet<Point>>, point: &Point,
) -> f64 {
    let segment = find_segment(segments, point);
    let corner_a = Point { x: 0, y: 0 };
    let corner_b = Point { x: (img.width() - 1) as i64, y: (img.height() - 1) as i64 };
    return point
        .iterate_neighbourhood()
        .enumerate()
        .map(|(i, neighbour)| {
            if segment.map_or(false, |x| x.contains(&neighbour))
                || !neighbour.is_within_rectangle(&corner_a, &corner_b)
            {
                return 0.0;
            }
            return 1.0 / (i + 1) as f64;
        })
        .sum();
}

pub fn connectivity_measure(
    img: &RgbImage, segments: &Vec<HashSet<Point>>, _dist: &ColorSpaceDistance,
) -> f64 {
    return img
        .enumerate_pixels()
        .map(|(x, y, _)| local_connectivity_measure(img, segments, &(x, y).into()))
        .sum();
}

pub fn extract_segments(contour: &RgbImage) -> (RgbImage, Vec<HashSet<Point>>) {
    let mut p = contour.clone();
    let mut segments = vec![];
    loop {
        // Find blank pixel.
        let blank = p.enumerate_pixels().find(|(_, _, &p)| p == Rgb([255, 255, 255]));
        if blank.is_none() {
            break;
        }
        // Fill in every connected pixel with random color.
        let (sx, sy, _) = blank.unwrap();
        let color = utilities::generate_unique_color(segments.len());
        segments.push(utilities::fill_connected(&mut p, &color, sx, sy));
    }
    return (p, segments);
}
