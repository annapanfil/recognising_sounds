use hound;

fn main() {
    let file_path = "./data/ESC-50-master/audio/1-137-A-32.wav";

    // Otwieramy plik WAV
    let reader = hound::WavReader::open(file_path).expect("Failed to open WAV file");
    let spec = reader.spec();

     // Odczyt próbek i mapowanie ich do wektora
    let samples: Vec<i16> = reader
     .into_samples::<i16>() // Zmieniamy próbki na i16
     .map(|s| s.expect("Error reading sample")) // Wyrzucenie błędu, jeśli wystąpi
     .collect();

    println!("Loaded {} samples with sample rate {}.", samples.len(), spec.sample_rate);
}
