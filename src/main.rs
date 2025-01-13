use hound;
use plotters::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex, FftDirection};
use smartcore::linalg::traits::stats::MatrixStats;
use std::f64::consts::PI;
use statrs::statistics::{Data, Min, Max, Median, OrderStatistics, Distribution};

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::model_selection::train_test_split;
use smartcore::metrics::accuracy;


#[derive(Debug)]
struct Stats{
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

fn load_wav(file_path: &str) -> (Vec<i16>, u32) {
    /* Load file into samples */
    let reader = hound::WavReader::open(file_path).expect("Failed to open WAV file");
    let spec = reader.spec();

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("Error reading sample"))
        .collect();

    (samples, spec.sample_rate)
}

fn get_known_signal(sample_rate: u32, duration: f64) -> Vec<i16> {
    /* Get sum of 2 sinus waves for debugging purposes
    @param sample_rate – samples per sec
    @param duration – s */

    let num_samples = (sample_rate as f64 * duration) as usize;
    let mut signal = Vec::with_capacity(num_samples);

    // Częstotliwości w Hz
    let f1 = 200.0;
    let f2 = 1200.0;

    // Generowanie sygnału
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;  // Czas dla próbki
        let sample = 0.5*(2.0 * PI * f1 * t).sin() + 0.5*(2.0 * PI * f2 * t).sin(); // Suma dwóch fal
        signal.push((sample * i16::MAX as f64) as i16);  // Skalowanie do zakresu i16
    }

    signal
}

fn plot_signal(samples: &Vec<i16>, file_path: &str) {
    /* Plot waveform of the signal */
    let root = BitMapBackend::new(file_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill drawing area");

    let mut chart = ChartBuilder::on(&root)
        .caption("Waveform", ("Arial", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..samples.len() as i32, -40000..40000)
        .expect("Failed to build chart");

    // configure axis
    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(5)
        .x_desc("Time (samples)")
        .y_desc("Amplitude")
        .draw()
        .expect("Failed to draw mesh");

    // draw wave
    chart
        .draw_series(LineSeries::new(
            samples.iter().enumerate().map(|(x, &y)| (x as i32, y as i32)),
            &RED,
        ))
        .expect("Failed to draw series");

    println!("Waveform saved to {}", file_path);
}

fn compute_fft(samples: &Vec<i16>) -> Vec<Complex<f64>> {
    /* Compute FFT of the signal */
    let samples: Vec<Complex<f64>> = samples.iter().map(|&s| Complex::new(s as f64, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(samples.len(), FftDirection::Forward);

    let mut spectrum = samples.clone();
    fft.process(&mut spectrum);

    spectrum
}

fn plot_fft(fft_data: &[Complex<f64>], sample_rate: u32, output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    /* Plot FFT spectrum */
    let num_samples = fft_data.len();
    let frequencies: Vec<f64> = (0..num_samples / 2)
        .map(|i| i as f64 * sample_rate as f64 / num_samples as f64) // formula for the frequency of each sample
        .collect();

    let magnitudes: Vec<f64> = fft_data.iter()
        .map(|c| c.norm())
        .collect();

    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_frequency = sample_rate as f64 / 2.0; // Nyquist frequency
    let max_magnitude = *magnitudes.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("FFT Spectrum", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..max_frequency, 0f64..max_magnitude)
        .expect("Failed to build chart");

    chart.configure_mesh()
        .x_labels(10)
        .y_labels(5)
        .x_desc("Frequency (Hz)")
        .y_desc("Magnitude")
        .draw()
        .expect("Failed to draw mesh");

    chart.draw_series(LineSeries::new(
        frequencies.into_iter().zip(magnitudes.into_iter()),
        &RED,
    )).expect("Failed to draw series");
    
    root.present()?;
    Ok(())
}

fn compute_statistics(samples:  &[i16]) -> Stats {
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

    Stats{
        min: data.min(),
        max,
        mean,
        variance,
        std: data.std_dev().expect("Failed to calculate std"),
        median: data.median(),
        q1: data.percentile(25),
        q3: data.percentile(75),
        skewness: third_moment / variance.powi(3),
        kurtosis: fourth_moment / variance.powi(2),
        energy,
        rms,
        crest_factor: max / rms,
        zcr
    }
}

fn train_model(x: DenseMatrix<f64>, y: Vec<u32>) {
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.8, true, Option::None);
    let knn = KNNClassifier::fit(&x_train, &y_train, KNNClassifierParameters::default()).unwrap();

    let y_pred = knn.predict(&x_test).unwrap();
    let accuracy = accuracy(&y_test, &y_pred);

    println!("Accuracy: {:.2}", accuracy);
}

fn main() {
    let file_path = "./data/ESC-50-master/audio/1-137-A-32.wav";
    
    let (samples, sample_rate) = load_wav(file_path);
    println!("Loaded {} samples with sample rate {}.", samples.len(), sample_rate);

    // let sample_rate = 4110;
    // let samples = get_known_signal(sample_rate, 1.0);

    plot_signal(&samples, "out/waveform.png");

    // let spectrum = compute_fft(&samples);
    // println!("Computed FFT spectrum.");
    // plot_fft(&spectrum, sample_rate, "out/fft.png").expect("Failed to plot FFT");

    let window_size = 1024; // sample_rate (44.1 kHz) * 23 ms rounded to a multiple of 2
    let step_size = window_size / 2;    // overlap

    let windows: Vec<_> = samples
        .windows(window_size)
        .step_by(step_size)
        .collect();

    let stats: Vec<_> = windows.iter().map(|&w| compute_statistics(w)).collect();
    println!("{:?}", stats[0]);

    // let x: Vec<Vec<f64>> = stats.iter().map(|s| s.to_vec()).collect();
    // let mut x = DenseMatrix::from_2d_vec(&x).unwrap();

    let mut x = DenseMatrix::from_2d_array(&[
        &[2.771244718, 1.784783929, 1.467537373],
        &[1.728571309, 1.169761413, 1.039941406],
        &[3.678319846, 2.812683004, 2.240515132],
        &[3.961043357, 2.619950020, 2.687259456],
        &[2.771244718, 1.784783929, 1.467537373],
        &[1.728571309, 1.169761413, 1.039941406],
        &[3.678319846, 2.812683004, 2.240515132],
        &[3.961043357, 2.619950020, 2.687259456],
        &[2.771244718, 1.784783929, 1.467537373],
        &[1.728571309, 1.169761413, 1.039941406],
        &[3.678319846, 2.812683004, 2.240515132],
        &[3.961043357, 2.619950020, 2.687259456]
    ]).unwrap();

    x.standard_scale_mut(&x.mean(0), &x.std(0), 0);
    let y = vec![0,0,1,1, 0,0,1,1, 0,0,1,1];

    train_model(x, y);
    

}
