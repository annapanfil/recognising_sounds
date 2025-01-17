use core::time;
use std::arch::x86_64;
use std::path::Path;

use std::fs;

mod load_and_show;
mod process;
mod models;

use load_and_show::{load_wav, plot_fft, plot_signal};
use process::{compute_mfcc, compute_statistics, get_class_name};
use models::{train_model};


fn main() {
    let dir_path = Path::new("./data/ESC-50-master/audio/");
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<u8> = Vec::new();

    let files = fs::read_dir(dir_path).expect("Failed to read directory");

    let mut i = 0;
    for file in files {
        if i % 10 == 0 {
            println!("Processing file number {}", i);
        }

        let file = file.expect("Failed to get entry");
        let file_path = file.path();

        let (samples, sample_rate) = load_wav(&file_path);
    
        // plot_signal(&samples, "out/waveform.png");
        // plot_fft(&compute_fft(&(samples.iter().map(|&s| s as f64)).collect()), sample_rate, "out/fft.png");
        
        // compute features 
        let window_size = 1024; // sample_rate (44.1 kHz) * 23 ms rounded to a multiple of 2
        let step_size = window_size / 2;    // overlap
        
        let mfcc = compute_mfcc(&samples, sample_rate, window_size, step_size, 26, 13, false);

        let windows: Vec<_> = samples
        .windows(window_size)
        .step_by(step_size)
        .collect();

        let stats: Vec<_> = windows.iter().map(|&w| compute_statistics(w)).collect();
    
        let mut features_flat = Vec::new();
        for (window_mfcc, stats) in mfcc.iter().zip(stats) {
            features_flat.extend(window_mfcc);
            features_flat.push(stats.zcr);
        }
    
        x.push(features_flat);
        
        
        let class: u8 = get_class_name(&file_path);
        y.push(class);

        // i += 1;
        // if i == 50{
        //     break;
        // }
    }

    train_model(x, y);
    

}
