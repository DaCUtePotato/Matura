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
#[allow(dead_code)]
pub fn sft(audio: Vec<f32>) -> Vec<Complex<f32>> {
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

// LLM-generated
pub fn ft(audio: Vec<f32>) -> Vec<Complex<f32>> {
    // Convert real audio samples to complex numbers
    let mut complex_audio: Vec<Complex<f32>> =
        audio.into_iter().map(|v| Complex::new(v, 0.0)).collect();

    let n = complex_audio.len();
    if n == 0 {
        return vec![];
    }

    // Radix-2 Cooley-Tukey requires the length to be a power of 2.
    // We pad with zeros if necessary.
    let next_pow2 = n.next_power_of_two();
    complex_audio.resize(next_pow2, Complex::new(0.0, 0.0));

    fft_recursive(&mut complex_audio);
    complex_audio
}

fn fft_recursive(audio: &mut [Complex<f32>]) {
    let len = audio.len();
    if len <= 1 {
        return;
    }

    // Split into even and odd indices
    let mut even: Vec<Complex<f32>> = vec![Complex::default(); len / 2];
    let mut odd: Vec<Complex<f32>> = vec![Complex::default(); len / 2];

    for i in 0..(len / 2) {
        even[i] = audio[2 * i];
        odd[i] = audio[2 * i + 1];
    }

    // Conquer
    fft_recursive(&mut even);
    fft_recursive(&mut odd);

    // Combine (Butterfly operation)
    for k in 0..(len / 2) {
        let angle = -2.0 * std::f32::consts::PI * (k as f32) / (len as f32);
        let twiddle = Complex::new(0.0, angle).exp() * odd[k];

        audio[k] = even[k] + twiddle;
        audio[k + len / 2] = even[k] - twiddle;
    }
}

// Just a function to convert <Vec<Vec<Complex<f32>>> to Vec<Vec<f32>> by norming the Complex<f32>

pub fn norm_vec_complex_log(complexed: &[Complex<f32>]) -> Vec<f32> {
    complexed
        .iter()
        .map(|s| (s.norm() + 1.).log10())
        .collect::<Vec<f32>>()
}

pub fn minmax(v: &mut [f32], min: f32, max: f32) {
    let range = max - min;

    // Check if range is too small (since you'd be dividing by it)
    if range.abs() < f32::EPSILON || !range.is_finite() || range.is_nan() {
        for x in v.iter_mut() {
            *x = 0.;
        }
    } else {
        for x in v.iter_mut() {
            let r = (*x - min) / (max - min);
            *x = if r.is_finite() && !r.is_nan() { r } else { 0. };
        }
    }
}
