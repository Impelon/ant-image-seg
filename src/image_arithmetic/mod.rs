//! Utilities for working with images.

pub mod color_distances;
pub mod traits;
pub use self::traits::*;
pub mod types;
pub use self::types::*;
pub mod utilities;
pub use self::utilities::*;

pub const LAPLACE_KERNEL: &[f32] = &[1.0, 1.0, 1.0, 1.0, -8.0, 1.0, 1.0, 1.0, 1.0];
pub const STRAIGHT_LAPLACE_KERNEL: &[f32] = &[0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
