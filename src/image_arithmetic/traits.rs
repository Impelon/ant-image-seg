pub trait ArithmeticImage: Sized {
    fn normalize(&mut self);
    fn binarize(&mut self);
    fn add(&mut self, other: &Self);
}
