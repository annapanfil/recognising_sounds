use core::time;
use std::arch::x86_64;
use std::path::Path;


use std::fs;

mod load_and_show;
mod process;
mod models;

use load_and_show::{load_wav, plot_fft, plot_mel_spectrogram, plot_signal};
use process::{compute_fft, apply_hamming_window, mel_filterbank};
use models::{train_model};


fn main() {
    let dir_path = Path::new("./data/ESC-50-master/audio/");
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<u8> = Vec::new();

    let entries = fs::read_dir(dir_path).expect("Failed to read directory");

    let mut i = 0;
    for entry in entries {

        let entry = entry.expect("Failed to get entry");
        let file_path = entry.path();
        
        let (samples, sample_rate) = load_wav(&file_path);
        println!("Loaded {} samples with sample rate {} from file {}.", samples.len(), sample_rate, file_path.display());
    
        plot_signal(&samples, "out/waveform.png");
        
        plot_fft(&compute_fft(&(samples.iter().map(|&s| s as f64)).collect()), sample_rate, "out/fft.png");

        
        // calculate features for moving windows
        let window_size = 1024; // sample_rate (44.1 kHz) * 23 ms rounded to a multiple of 2
        let step_size = window_size / 2;    // overlap
        let filters = mel_filterbank(26, window_size, sample_rate as f64);

        let windows: Vec<_> = samples
            .windows(window_size)
            .step_by(step_size)
            .map(|window| apply_hamming_window(window))
            .collect();
        
        println!("Calculated {} windows of size {} with step size {}.", windows.len(), window_size, step_size);

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
                    .map(|val| 20.0 * val.log10()) //to db
                    .collect()
                }
            ).collect();
        
        plot_mel_spectrogram(&mel_spectrogram);
        }

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
        if i == 1 {
            break;
        }
    }

    // train_model(x, y);
    

}
