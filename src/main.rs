use hound;
use plotters::prelude::*;

fn load_wav(file_path: &str) -> (Vec<i16>, u32) {
    let reader = hound::WavReader::open(file_path).expect("Failed to open WAV file");
    let spec = reader.spec();

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("Error reading sample"))
        .collect();

    (samples, spec.sample_rate)
}


fn plot_waveform(samples: &Vec<i16>, file_path: &str) {
    let root = BitMapBackend::new(file_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE).expect("Failed to fill drawing area");

    let mut chart = ChartBuilder::on(&root)
        .caption("Waveform", ("Arial", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..samples.len() as i32, -32768..32767)
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


fn main() {
    let file_path = "./data/ESC-50-master/audio/1-137-A-32.wav";
    
    let (samples, sample_rate) = load_wav(file_path);
    println!("Loaded {} samples with sample rate {}.", samples.len(), sample_rate);

    plot_waveform(&samples, "out/waveform.png");
    

}
