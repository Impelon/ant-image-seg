//! Provides functionality to segment images with ant colony optimization.

use std::collections::HashSet;
use std::ops::Deref;

use super::image_ants::{AntColonyRules, PheromoneImage, UpdateFunction};
use super::image_arithmetic;
use super::image_arithmetic::{color_distances, segments, ArithmeticImage, Point};

use cached::proc_macro::cached;
use image::{imageops, DynamicImage, Pixel, RgbImage, Rgba, RgbaImage};
use rand;

pub fn contour_segmententation(pheromones: &[PheromoneImage], threshold: f32) -> RgbImage {
    let mut segmentation = pheromones[0].clone();
    for pheromone in &pheromones[1..] {
        segmentation.add(pheromone);
    }
    segmentation = extract_edges(&segmentation, threshold);
    imageops::invert(&mut segmentation);
    // Add border to enforce closed segments.
    let w = segmentation.width();
    let h = segmentation.height();
    let cropped = DynamicImage::from(segmentation).crop_imm(1, 1, w - 2, h - 2).to_rgb8();
    let mut canvas = RgbImage::new(w, h);
    imageops::replace(&mut canvas, &cropped, 1, 1);
    return canvas;
}

pub fn overlayed_contour_segmententation(
    img: &RgbImage, pheromones: &[PheromoneImage], threshold: f32,
) -> RgbImage {
    let p = contour_segmententation(pheromones, threshold);
    let colored_contour = RgbaImage::from_fn(p.width(), p.height(), |x, y| {
        Rgba([0, 255, 0, (255 - p.get_pixel(x, y).0[0]) / 3 * 2])
    });
    let mut canvas = DynamicImage::from(img.clone());
    imageops::overlay(&mut canvas, &colored_contour, 0, 0);
    return canvas.to_rgb8();
}

/// Cached calculation of segments from pheromones.
// #[cached(size = 64, convert = r#"{ format!("{:p}", pheromones) }"#, key = "String", sync_writes = true)]
pub fn region_segmententation(
    pheromones: &[PheromoneImage], threshold: f32,
) -> (RgbImage, Vec<HashSet<Point>>) {
    return segments::extract_segments(&contour_segmententation(pheromones, threshold));
}

pub fn colorized_region_segmententation(
    img: &RgbImage, pheromones: &[PheromoneImage], threshold: f32,
) -> (RgbImage, Vec<HashSet<Point>>) {
    let (mut segmented, segments) = region_segmententation(pheromones, threshold);
    for points in &segments {
        let color = image_arithmetic::mean_color(&img, points);
        points.iter().for_each(|p| *p.get_pixel_mut(&mut segmented) = color);
    }
    return (segmented, segments);
}

pub fn create_rules<R: rand::Rng + 'static>(
    img: &RgbImage, parallelity: Option<usize>, multi: bool,
) -> AntColonyRules<R> {
    let max_steps = ((img.width() * img.height()) / 8) as usize;
    if multi {
        return AntColonyRules::new(
            max_steps,
            multi_objective::ants_per_global_update(),
            parallelity,
            vec![
                multi_objective::initialization_functions(),
                multi_objective::local_update_functions(),
                multi_objective::global_update_functions(),
            ],
        )
        .unwrap();
    } else {
        return AntColonyRules::new(
            max_steps,
            single_objective::ants_per_global_update(),
            parallelity,
            vec![
                single_objective::initialization_functions(),
                single_objective::local_update_functions(),
                single_objective::global_update_functions(),
            ],
        )
        .unwrap();
    }
}

pub fn increase_phermomone<I, P>(pheromone: &mut PheromoneImage, points: I, amount: f32)
where
    I: IntoIterator<Item = P>,
    P: Deref<Target = Point>,
{
    for point in points {
        point.get_pixel_mut(pheromone).0[0] += amount;
    }
}

pub fn multiply_phermomone<I, P>(pheromone: &mut PheromoneImage, points: I, multiplier: f32)
where
    I: IntoIterator<Item = P>,
    P: Deref<Target = Point>,
{
    for point in points {
        point.get_pixel_mut(pheromone).0[0] *= multiplier;
    }
}

pub fn extract_edges(pheromone: &PheromoneImage, threshold: f32) -> PheromoneImage {
    let mut result = pheromone.clone();
    result.binarize(threshold);
    imageops::invert(&mut result);
    return imageops::filter3x3(&result, image_arithmetic::LAPLACE_KERNEL);
}

/// Combines the ant colony primitives with concrete rules
/// to achieve image segmentation using multiple objectives.
pub mod multi_objective {
    use super::*;

    pub fn local_edge_value<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        increase_phermomone(_pheromone, _visited, 1.0);
        multiply_phermomone(_pheromone, _visited, 0.5);
    }

    pub fn global_edge_value<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        _pheromone.normalize();
    }

    pub fn local_connectivity_measure<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        increase_phermomone(_pheromone, _visited, 1.0);
        multiply_phermomone(_pheromone, _visited, 0.1);
    }

    pub fn global_connectivity_measure<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        _pheromone.normalize();
    }

    pub fn local_overall_deviation<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        increase_phermomone(_pheromone, _visited, 1.0);
    }

    pub fn global_overall_deviation<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        _pheromone.normalize();
    }

    pub fn initialization_functions<R: rand::Rng + 'static>() -> Vec<Option<Box<UpdateFunction<R>>>>
    {
        return vec![None, None, None];
    }

    pub fn local_update_functions<R: rand::Rng + 'static>() -> Vec<Option<Box<UpdateFunction<R>>>> {
        return vec![
            Some(Box::new(local_edge_value)),
            Some(Box::new(local_connectivity_measure)),
            Some(Box::new(local_overall_deviation)),
        ];
    }

    pub fn global_update_functions<R: rand::Rng + 'static>() -> Vec<Option<Box<UpdateFunction<R>>>>
    {
        return vec![
            Some(Box::new(global_edge_value)),
            Some(Box::new(global_connectivity_measure)),
            Some(Box::new(global_overall_deviation)),
        ];
    }

    pub fn ants_per_global_update() -> usize {
        return 40;
    }
}

/// Combines the ant colony primitives with concrete rules
/// to achieve image segmentation using a single weighted-sum objective.
pub mod single_objective {
    use super::*;

    pub fn initialization_functions<R: rand::Rng + 'static>() -> Vec<Option<Box<UpdateFunction<R>>>>
    {
        return vec![None];
    }

    pub fn local<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        increase_phermomone(_pheromone, _visited, 1.0);
    }

    pub fn global<R: rand::Rng + 'static>(
        _rng: &mut R, _img: &RgbImage, _pheromone: &mut PheromoneImage, _visited: &HashSet<Point>,
    ) {
        _pheromone.normalize();
    }

    pub fn local_update_functions<R: rand::Rng + 'static>() -> Vec<Option<Box<UpdateFunction<R>>>> {
        return vec![Some(Box::new(local))];
    }

    pub fn global_update_functions<R: rand::Rng + 'static>() -> Vec<Option<Box<UpdateFunction<R>>>>
    {
        return vec![Some(Box::new(global))];
    }

    pub fn ants_per_global_update() -> usize {
        return 40;
    }
}
