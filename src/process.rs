use rustfft::{FftPlanner, num_complex::Complex, FftDirection};
use statrs::statistics::{Data, Min, Max, Median, OrderStatistics, Distribution};

#[derive(Debug)]
pub struct Stats{
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    q1: f64,
    q3: f64,
    variance: f64,
    std: f64,
    skewness: f64,
    kurtosis: f64,
    energy: f64, // whole energy of the signal – how intensive the sound is
    rms: f64,
    crest_factor: f64, // how big is the peak compared to the rest of the signal
    zcr: f64, // how many times the signal crosses the x axis
}

impl Stats {
    fn to_vec(&self) -> Vec<f64> {
        vec![
            self.min,
            self.max,
            self.mean,
            self.median,
            self.q1,
            self.q3,
            self.variance,
            self.std,
            self.skewness,
            self.kurtosis,
            self.energy,
            self.rms,
            self.crest_factor,
            self.zcr
        ]
    }
}


pub fn compute_fft(samples: &Vec<i16>) -> Vec<Complex<f64>> {
    /* Compute FFT of the signal */
    let samples: Vec<Complex<f64>> = samples.iter().map(|&s| Complex::new(s as f64, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(samples.len(), FftDirection::Forward);

    let mut spectrum = samples.clone();
    fft.process(&mut spectrum);

    spectrum
}


pub fn hamming_window(window: Vec<i16>) -> Vec<f64> {
    let window_size = window.len();
    (0..window_size)
        .map(|n| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / (window_size as f64 - 1.0)).cos())
        .collect()
}

pub fn mel_filterbank(n_filters: usize, n_fft: usize, sample_rate: f64) -> Vec<Vec<f64>> {
    let low_freq = hz_to_mel(0.0);
    let high_freq = hz_to_mel(sample_rate / 2.0); // Nyquist frequency

    // mel points – equally spaced in mel scale
    let mel_points: Vec<f64> = (0..n_filters)
        .map(|i| low_freq + (i as f64) * (high_freq - low_freq) / (n_filters + 1) as f64)
        .collect();

    // convert mel points to Hz
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // create filterbank
    let mut filterbank = vec![vec![0.0; n_fft / 2 + 1]; n_filters];

    for (i, &hz) in hz_points.iter().enumerate() {
        let fft_bin = (hz / sample_rate * (n_fft as f64).round() / 2.0) as usize;
        if i > 0 {
            filterbank[i-1][fft_bin] = 1.0;
        }

    }

    filterbank
}

fn hz_to_mel(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}

pub fn compute_statistics(samples:  &[i16]) -> Stats {
    /* Compute statistics of the signal */

    let samples_f64: Vec<_> = samples.iter().map(|&x| x as f64).collect();
    let mut data = Data::new(samples_f64);
    
    let mean =  data.mean().expect("Failed to calculate mean");
    let variance = data.variance().expect("Failed to calculate variance");
    let max = data.max();

    let n = samples.len() as f64;
    let fourth_moment = samples.iter().map(|&x| (x as f64 - mean).powi(4)).sum::<f64>() / n;
    let third_moment = samples.iter().map(|&x| (x as f64 - mean).powi(3)).sum::<f64>() / n;

    let energy = samples.iter().map(|&x| ((x as f64).powi(2))).sum::<f64>();
    let rms = (energy / samples.len() as f64).sqrt(); // root mean square – mean energy of the sound

    let zcr = samples //zero crossing rate – how many times the signal crosses the x axis
        .windows(2)
        .filter(|window| (window[0] > 0) != (window[1] > 0))
        .count() as f64 / n;

    // variance is 0 e.g. for constant 0 signals
    let skewness = if variance == 0.0 { 0.0 } else { third_moment / variance.powi(3) };
    let kurtosis = if variance == 0.0 { 3.0 } else { fourth_moment / variance.powi(2) }; // uniform treated as perfect normal
    let crest_factor = if variance == 0.0 { 0.0 } else { max / rms }; // no crests in 0 signal

    Stats{
        min: data.min(),
        max,
        mean,
        variance,
        std: data.std_dev().expect("Failed to calculate std"),
        median: data.median(),
        q1: data.percentile(25),
        q3: data.percentile(75),
        skewness: skewness,
        kurtosis: kurtosis,
        energy,
        rms,
        crest_factor: crest_factor,
        zcr
    }
}

pub fn compute_features(window: Vec<f64>) -> Vec<f64> {



    Vec::new()
}