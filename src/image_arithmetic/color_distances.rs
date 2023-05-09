use image::Rgb;

fn multiply(x: u8, y: u8) -> f64 {
    ((x as u16) * (y as u16)) as f64
}

fn square(x: u8) -> f64 {
    multiply(x, x)
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

fn magnitude(x: &Rgb<u8>) -> f64 {
    (square(x.0[0]) + square(x.0[1]) + square(x.0[2])).sqrt()
}

pub fn cosine_unnormed(a: &Rgb<u8>, b: &Rgb<u8>) -> f64 {
    multiply(a.0[0], b.0[0]) + multiply(a.0[1], b.0[1]) + multiply(a.0[2], b.0[2])
}

pub fn cosine(a: &Rgb<u8>, b: &Rgb<u8>) -> f64 {
    cosine_unnormed(a, b) / (magnitude(a) * magnitude(b))
}
