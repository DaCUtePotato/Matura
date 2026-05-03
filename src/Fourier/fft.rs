use num::Complex;
// Naïve approach to the Fourier transform.
// Going from e^(i 2pi k n/N) where k is the
// periodicity of the sinusoid and n/N the value
// of the sinusoid at position n out of N samples
// You multiply the sum of samples with the curve
// at their position and add them up to get the amplitude of
// that curve. You do that for all samples in the thingy and you get
// O(n^2). Because symmetry exists, you can reduce that to O(n*log(n))
// easily but later.
pub fn ft(audio: Vec<f64>) -> Vec<Complex<f64>> {
    let mut result: Vec<Complex<f64>> = vec![];
    let len: usize = audio.len();

    for i in 0..len {
        let mut sum: Complex<f64> = Complex::new(0.0, 0.0);
        for (k, value) in audio.iter().enumerate() {
            let z = Complex::new(
                0.0,
                -2. * std::f64::consts::PI * i as f64 * (k as f64 / len as f64),
            );
            sum += value * z.exp();
        }
        result.push(sum);
    }
    result
}

// Just a function to convert <Vec<Vec<Complex<f32>>> to Vec<Vec<f32>> by norming the Complex<f32>
pub fn normVecVecComplex(complexed: &[Vec<Complex<f64>>]) -> Vec<Vec<f64>> {
    complexed
        .iter()
        .map(|s| s.iter().map(|c| c.norm()).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>()
}

pub fn normVecComplex(complexed: &[Complex<f64>]) -> Vec<f64> {
    complexed.iter().map(|s| s.norm()).collect::<Vec<f64>>()
}
