use std::{path::PathBuf, time};

use rustfft::{FftPlanner, num_complex::Complex, FftDirection};
use statrs::statistics::{Data, Min, Max, Median, OrderStatistics, Distribution};

use crate::load_and_show::{plot_filterbank, plot_mel_spectrogram};

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
    pub zcr: f64, // how many times the signal crosses the x axis
}

impl Stats {
    pub fn to_vec(&self) -> Vec<f64> {
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


pub fn compute_fft(samples: &Vec<f64>) -> Vec<Complex<f64>> {
    /* Compute FFT of the signal */
    let samples: Vec<Complex<f64>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(samples.len(), FftDirection::Forward);

    let mut spectrum = samples.clone();
    fft.process(&mut spectrum);

    spectrum
}


pub fn apply_hamming_window(window: &[i16]) -> Vec<f64> {
    let window_size = window.len();
    (0..window_size)
        .map(|n| window[n] as f64 * (0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / (window_size as f64 - 1.0)).cos()))
        .collect()
}

pub fn mel_filterbank(n_filters: usize, n_fft: usize, sample_rate: f64) -> Vec<Vec<f64>> {
    let low_freq = hz_to_mel(0.0);
    let high_freq = hz_to_mel(sample_rate / 2.0); // Nyquist frequency

    // mel points – equally spaced in mel scale
    let mel_points: Vec<f64> = (0..n_filters+1)
        .map(|i| low_freq + (i as f64) * (high_freq - low_freq) / (n_filters + 2) as f64)
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

pub fn hz_to_mel(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

fn mel_to_hz(m: f64) -> f64 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}


pub fn dct(input: &Vec<f64>) -> Vec<f64> {
    let input_len = input.len();
    let mut result = Vec::with_capacity(input_len);

    for k in 0..input_len {
        let mut sum = 0.0;
        for (n, &x) in input.iter().enumerate() {
            // println!("{} {} {} {}", x, k, n, (std::f64::consts::PI * (k as f64) * (n as f64 + 0.5) / input_len as f64));
            sum += x * (std::f64::consts::PI * (k as f64) * (n as f64 + 0.5) / input_len as f64).cos();

        }

        result.push(sum);
    }

    result
}

pub fn compute_mfcc (samples: &Vec<i16>, sample_rate: u32, window_size: usize, 
    step_size: usize, n_filters: usize, n_coeffs: usize, plot: bool) -> Vec<Vec<f64>> {

    let filters = mel_filterbank(n_filters, window_size, sample_rate as f64);

    if plot { plot_filterbank(&filters, sample_rate); }

    let windows: Vec<_> = samples
        .windows(window_size)
        .step_by(step_size)
        .map(|window| apply_hamming_window(window))
        .collect();
    
    //compute mel spectrogram
    let mel_spectrogram  : Vec<Vec<f64>> = windows.iter() 
        .map(|window| {        
            let spectrum: Vec<_> = compute_fft(&window).iter()
            .map(|s| s / window_size as f64) // normalize
            .collect();
            
            // plot_signal(&window.to_vec(), "out/window.png");
            // plot_fft(&spectrum, sample_rate, "out/sfft.png");
            // std::thread::sleep(time::Duration::from_secs(1));

            filters.iter()
                .map(|filter| {
                    filter.iter()
                        .zip(&spectrum)
                        .map(|(&f, &s)| f * s.norm()) // magnitude
                        .sum::<f64>()
                })
                .map(|val| 20.0 * (val + 1e-10).log10()) //to db
                .collect()
            }
        ).collect();
    
    if plot { plot_mel_spectrogram(&mel_spectrogram, "out/mel_spectrogram.png"); }    

    let mfcc: Vec<Vec<f64>> = mel_spectrogram.iter() // windows
        .map(|window| {
            let dct_result = dct(window);

            dct_result.into_iter()
                .take(n_coeffs)
                .collect()
        })
        .collect();

    if plot { plot_mel_spectrogram(&mfcc, "out/mfcc.png"); 
        std::thread::sleep(time::Duration::from_secs(1));
    }

    mfcc
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


pub fn get_class_name(file_path: &PathBuf) -> u8 {
    // get class number from the filename

    let filename = file_path.file_name().unwrap().to_str().unwrap();

    let class: u8 = filename[..&filename.len()-4] // remove .wav
    .split('-').last().unwrap() // get last part of the filename
    .parse().expect("Cannot get class number from the filename."); // parse to u8

    class
}