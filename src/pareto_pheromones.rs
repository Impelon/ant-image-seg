use std::collections::HashSet;

use super::image_ants::PheromoneImage;
use super::image_arithmetic::{color_distances, segments, Point};
use super::segment_generation::region_segmententation;

use image::RgbImage;
use pareto_front::Dominate;

pub struct ParetoPheromones {
    pub pheromones: Vec<PheromoneImage>,
    pub segments: Vec<HashSet<Point>>,
    pub edge_value: f64,
    pub connectivity_measure: f64,
    pub overall_deviation: f64,
}

impl ParetoPheromones {
    pub fn new(image: &RgbImage, pheromones: Vec<PheromoneImage>) -> Self {
        let (_, segments) = region_segmententation(&pheromones);
        let edge_value = segments::edge_value(image, &segments, &color_distances::euclidean);
        let connectivity_measure =
            segments::connectivity_measure(image, &segments, &color_distances::euclidean);
        let overall_deviation =
            segments::overall_deviation(image, &segments, &color_distances::euclidean);
        return Self {
            pheromones,
            segments,
            edge_value,
            connectivity_measure,
            overall_deviation,
        };
    }

    pub fn stat_info(&self) -> String {
        format!(
            "segs{}-e{:.2E}-c{:.2E}-d{:.2E}",
            self.segments.len(),
            self.edge_value,
            self.connectivity_measure,
            self.overall_deviation
        )
    }
}

impl Dominate for ParetoPheromones {
    fn dominate(&self, other: &Self) -> bool {
        self.edge_value >= other.edge_value
            && self.connectivity_measure <= other.connectivity_measure
            && self.overall_deviation <= other.overall_deviation
    }
}
