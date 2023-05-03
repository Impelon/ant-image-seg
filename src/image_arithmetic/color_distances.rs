use image::Rgb;

fn square(x: u8) -> f64 {
    ((x as u16) * (x as u16)) as f64
}

pub fn euclidean_squared(a: &Rgb<u8>, b: &Rgb<u8>) -> f64 {
    square(b.0[0] - a.0[0]) + square(b.0[1] - a.0[1]) + square(b.0[2] - a.0[2])
}

pub fn euclidean(a: &Rgb<u8>, b: &Rgb<u8>) -> f64 {
    euclidean_squared(a, b).sqrt()
}

fn absdiff(a: u8, b: u8) -> f64 {
    (((b as i16) - (a as i16)) as f64).abs()
}

pub fn manhattan(a: &Rgb<u8>, b: &Rgb<u8>) -> f64 {
    absdiff(a.0[0], b.0[0]) + absdiff(a.0[1], b.0[1]) + absdiff(a.0[2], b.0[2])
}
