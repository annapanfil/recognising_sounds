use core::time;
use std::arch::x86_64;
use std::path::Path;


use std::fs;

mod load_and_show;
mod process;
mod models;

use load_and_show::{load_wav, plot_fft, plot_signal};
use process::{compute_mfcc};
use models::{train_model};


fn main() {
    let dir_path = Path::new("./data/ESC-50-master/audio/");
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<u8> = Vec::new();

    let files = fs::read_dir(dir_path).expect("Failed to read directory");

    let mut i = 0;
    for file in files {
        
        let file = file.expect("Failed to get entry");
        let file_path = file.path();
        
        if !file_path.to_string_lossy().ends_with("-42.wav") {
            println!("Skipping file {}.", file_path.display());
            i += 1;
            continue;
        }

        let (samples, sample_rate) = load_wav(&file_path);
        println!("Loaded {} samples with sample rate {} from file {}.", samples.len(), sample_rate, file_path.display());
    
        // plot_signal(&samples, "out/waveform.png");
        // plot_fft(&compute_fft(&(samples.iter().map(|&s| s as f64)).collect()), sample_rate, "out/fft.png");
        
        // compute features 
        let window_size = 1024; // sample_rate (44.1 kHz) * 23 ms rounded to a multiple of 2
        let step_size = window_size / 2;    // overlap
        
        compute_mfcc(&samples, sample_rate, window_size, step_size, 26, 13, true);

        // let stats: Vec<_> = windows.iter().map(|&w| compute_statistics(w)).collect();

    
    //     let mut stats_flattened = Vec::new();
    //     for stat in stats {
    //         stats_flattened.extend(stat.to_vec());
    //     }
    
    //     x.push(stats_flattened);
            
    //     // get class number from the filename
    //     let filename = file_path.file_name().unwrap().to_str().unwrap();

    //     let class: u8 = filename[..&filename.len()-4] // remove .wav
    //     .split('-').last().unwrap() // get last part of the filename
    //     .parse().expect("Cannot get class number from the filename."); // parse to u8
    
    //     y.push(class);

        i += 1;
        if i == 10{
            break;
        }
    }

    // train_model(x, y);
    

}
