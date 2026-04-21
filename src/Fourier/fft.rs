use num::Complex;

pub fn ft(audio: Vec<f32>) -> Vec<Complex<f32>> {
    let mut result: Vec<Complex<f32>> = vec![];
    let len: usize = audio.len();

    for i in 0..len {
        let mut sum: Complex<f32> = Complex::new(0.0, 0.0);
        for (k, value) in audio.iter().enumerate() {
            let z = Complex::new(
                0.0,
                -2. * std::f32::consts::PI * i as f32 * (k as f32 / len as f32),
            );
            sum += value * z.exp();
        }
        result.push(sum);
    }
    result
}

pub fn normVecVecComplex(complexed: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
    complexed
        .iter()
        .map(|s| s.iter().map(|c| c.norm()).collect::<Vec<f32>>())
        .collect::<Vec<Vec<f32>>>()
}
