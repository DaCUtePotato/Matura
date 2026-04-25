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

// Just a function to convert <Vec<Vec<Complex<f32>>> to Vec<Vec<f32>> by norming the Complex<f32>
pub fn normVecVecComplex(complexed: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
    complexed
        .iter()
        .map(|s| s.iter().map(|c| c.norm()).collect::<Vec<f32>>())
        .collect::<Vec<Vec<f32>>>()
}
