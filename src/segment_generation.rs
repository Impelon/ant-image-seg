//! Provides functionality to segment images with ant colony optimization.

use std::collections::HashSet;
use std::ops::Deref;

use super::image_ants::{AntColonyRules, PheromoneImage, UpdateFunction};
use super::image_arithmetic::color_distances;
use super::image_arithmetic::{ArithmeticImage, Point};
use image::RgbImage;
use rand;

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
