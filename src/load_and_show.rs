   
use std::{f64::consts::PI, path::Path};
use plotters::prelude::*;
use hound;
use rustfft::num_complex::Complex;
use palette::{LinSrgb, Mix};


pub const CLASSES: [&str; 50] = [
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects (flying)", "sheep", "crow",
        "rain", "sea waves", "crackling fire", "crickets", "chirping birds", "water drops", "wind",
        "pouring water", "toilet flush", "thunderstorm", "crying baby", "sneezing", "clapping",
        "breathing", "coughing", "footsteps", "laughing", "brushing teeth", "snoring",
        "drinking - sipping", "door knock", "mouse click", "keyboard typing", "door, wood creaks",
        "can opening", "washing machine", "vacuum cleaner", "clock alarm", "clock tick",
        "glass breaking", "helicopter", "chainsaw", "siren", "car horn", "engine", "train",
        "church bells", "air plane", "fireworks", "hand saw"
    ];


pub fn load_wav(file_path: &Path) -> (Vec<i16>, u32) {
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


/// plotting

pub fn plot_signal<T>(samples: &Vec<T>, file_path: &str) where T: Into<f64> + Copy {
    let min_sample = samples.iter().map(|&x| x.into()).fold(f64::INFINITY, f64::min);
    let max_sample = samples.iter().map(|&x| x.into()).fold(f64::NEG_INFINITY, f64::max);

    /* Plot waveform of the signal */
    let root = BitMapBackend::new(file_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill drawing area");

    let mut chart = ChartBuilder::on(&root)
        .caption("Waveform", ("Arial", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..samples.len() as f64, min_sample..max_sample)
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
            samples.iter().enumerate().map(|(x, &y)| (x as f64, y.into())),
            &RED,
        ))
        .expect("Failed to draw series");

    println!("Waveform saved to {}", file_path);
}


pub fn plot_fft(fft_data: &[Complex<f64>], sample_rate: u32, output_file: &str) {
    /* Plot FFT spectrum */
    let num_samples = fft_data.len();
    let frequencies: Vec<f64> = (0..num_samples / 2)
        .map(|i| i as f64 * sample_rate as f64 / num_samples as f64) // formula for the frequency of each sample
        .collect();

    let magnitudes: Vec<f64> = fft_data.iter()
        .map(|c| c.norm()) 
        .collect();

    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill drawing area");

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
    
    root.present().expect("Failed to present");

    println!("FFT spectrum saved to {}", output_file);
}

pub fn plot_filterbank(filterbank: &Vec<Vec<f64>>, sample_rate: u32) {
    // plot
    let root = BitMapBackend::new("out/filterbank.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill drawing area");

    let mut chart = ChartBuilder::on(&root)
        .caption("Mel Filterbank", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..(filterbank[0].len() / 2) as u32, 0f64..1.0)
        .expect("Failed to build chart");

    chart.configure_mesh()
        .x_labels(10)
        .y_labels(5)
        .x_desc("Filter index")
        .y_desc("Amplitude")
        .draw()
        .expect("Failed to draw mesh");

    for (i, filter) in filterbank.iter().enumerate() {
        chart.draw_series(LineSeries::new(
            filter.iter().enumerate().map(|(j, &amp)| (j as u32, amp)),
            &Palette99::pick(i),
        )).expect("Failed to draw series");
    }

    root.present().expect("Failed to present");
}

fn get_color_from_palette(x: f64) -> RGBAColor{
    /* blue -> white -> red gradient */
    let blue = LinSrgb::new(0.0, 0.0, 1.0);
    let white = LinSrgb::new(1.0, 1.0, 1.0);
    let red = LinSrgb::new(1.0, 0.0, 0.0);

    let color = if x < 0.5 {
        blue.mix(white, x * 2.0) // Od niebieskiego do białego
    } else {
        white.mix(red, (x - 0.5) * 2.0) // Od białego do czerwonego
    };

    RGBAColor((color.red * 255.0) as u8, (color.green * 255.0) as u8, (color.blue * 255.0) as u8, 1.0)
}



pub fn plot_mel_spectrogram(mel_spectrogram: &Vec<Vec<f64>>, filename: &str) {
    let root = BitMapBackend::new(filename, (1024, 768))
        .into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill drawing area");

    let mut chart = ChartBuilder::on(&root)
        .caption(filename, ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..mel_spectrogram.len() as u32, 0..mel_spectrogram[0].len() as u32)
        .expect("Failed to build chart");

    chart.configure_mesh()
        .x_desc("Window number")
        .y_desc("Mel frequency")
        .draw().expect("Failed to draw mesh");

    let min_value = mel_spectrogram.iter().flatten().filter(|&&value| value != f64::NEG_INFINITY).cloned().fold(f64::INFINITY, f64::min);
    let max_value = mel_spectrogram.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);






    chart.draw_series(
        mel_spectrogram.iter().enumerate()
            .flat_map(|(i, row)| { // frequencies
                row.iter().enumerate().map(move |(j, &value)| { //times
                    let normalised_value = (value - min_value) / (max_value - min_value);
                    // print!("{} ", value);
                    // let color =  RGBAColor(
                    //     (normalised_value) as u8,
                    //     0,
                    //     (255.0 - normalised_value) as u8, 1.0);

                    Rectangle::new(
                        [(i as u32, j as u32), ((i + 1) as u32, (j + 1) as u32)],
                        ShapeStyle {
                            color: get_color_from_palette(normalised_value),
                            filled: true,
                            stroke_width: 0,
                        }
                    )
                })
            })
    ).expect("Failed to draw series");

    println!("Mel spectrogram saved to {}", filename);

}