use image::Primitive;

pub trait ArithmeticImage<N: Primitive>: Sized {
    fn max(&self) -> N;
    fn min(&self) -> N;
    fn normalize(&mut self);
    fn binarize(&mut self, threshold: N);
    fn clamp(&mut self, threshold: N);
    fn add(&mut self, other: &Self);
    fn add_scalar(&mut self, num: N);
    fn mul(&mut self, other: &Self);
    fn mul_scalar(&mut self, num: N);
}
