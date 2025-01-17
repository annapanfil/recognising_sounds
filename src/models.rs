use std::collections::HashSet;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::metrics::accuracy;

use rand::prelude::*;



fn stratified_sampling(x: &Vec<Vec<f64>>, y: &Vec<u8>, train_perc: f32) -> (DenseMatrix<f64>, DenseMatrix<f64>, Vec<u8>, Vec<u8>) {
    let mut train_indices: Vec<usize> = Vec::new();
    let mut test_indices: Vec<usize> = Vec::new();

    let unique_classes = y.iter().collect::<HashSet<&u8>>();

    for class in unique_classes {
        let indices = y.iter().enumerate()
            .filter(|(_, &c)| *class == c)
            .map(|(i, _)| {i})
            .collect::<Vec<usize>>(); 
        
        let n_samples = indices.len();
        let train_indices_class: HashSet<usize> =  indices.choose_multiple(&mut rand::thread_rng(), (n_samples as f32 * train_perc).round() as usize).cloned().collect();
        let test_indices_class = indices.iter().filter(|i| !train_indices_class.contains(i));

        test_indices.extend(test_indices_class);
        train_indices.extend(train_indices_class);
    }

    train_indices.shuffle(&mut rand::thread_rng());
    test_indices.shuffle(&mut rand::thread_rng());

    let x_train: Vec<Vec<f64>> = train_indices.iter().map(|&i| x[i].clone()).collect();
    let x_train: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_train).unwrap();

    let x_test = test_indices.iter().map(|&i| x[i].clone()).collect();
    let x_test: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&x_test).unwrap();

    let y_train = train_indices.iter().map(|&i| y[i]).collect();
    let y_test = test_indices.iter().map(|&i| y[i]).collect();

    (x_train, x_test, y_train, y_test)
}

fn standard_scaling(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    /* standard scalling on 0 axis (columns) */
    let n_rows = x.len();
    let n_cols = x[0].len();

    let mean: Vec<f64>= (0..n_cols).map(|i| 
        x.iter().map(|row| row[i]).sum::<f64>() / x.len() as f64)
        .collect();

    let var: Vec<f64>= (0..n_cols).map(|i|
        x.iter().map(|row| (row[i] - mean[i]).powi(2))
        .sum::<f64>() / n_rows as f64)
        .collect();
     
    x.iter()
        .map(|row| {
            (0..n_cols)
                .map(|i| (row[i] - mean[i]) / var[i].sqrt())
                .collect::<Vec<f64>>()
        })
        .collect()
}


pub fn train_model(x: Vec<Vec<f64>>, y: Vec<u8>) {

    let scaled_x = standard_scaling(&x);
    let (x_train, x_test, y_train, y_test) = stratified_sampling(&scaled_x, &y, 0.8);

    let knn = KNNClassifier::fit(&x_train, &y_train, KNNClassifierParameters::default()).unwrap();
    

    let y_pred = knn.predict(&x_test).unwrap();
    let accuracy = accuracy(&y_test, &y_pred);
    println!("Accuracy: {:.2}", accuracy);
}

#[cfg(test)]
mod tests {
    use super::*;
    use smartcore::linalg::traits::stats::MatrixStats;
    use smartcore::linalg::basic::arrays::Array;

    fn matrix_to_vector(matrix: &DenseMatrix<f64>) -> Vec<f64> {
        matrix.iter().cloned().collect()
    }

    #[test]
    fn test_no_duplicates_between_train_and_test() {
        let x = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ];
        let y = vec![0, 1, 0, 1, 0];
        let (x_train, x_test, _, _) = stratified_sampling(&x, &y, 0.6);

        let train_vec = matrix_to_vector(&x_train);
        let test_vec = matrix_to_vector(&x_test);

        assert!(train_vec.iter().all(|val| !test_vec.contains(val)));
    }

    #[test]
    fn test_x_train_is_80_percent() {
        let x = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ];
        let y = vec![0, 1, 0, 1, 0];
        let (x_train, _, _, _) = stratified_sampling(&x, &y, 3.0/5.0);

        // Test that x_test is approximately 80% of the total samples
        let expected_train_size = (x.len() as f32 * 3.0/5.0) as usize;

        assert_eq!(x_train.shape().0, expected_train_size);
    }

    #[test]
    fn test_stratified_sampling_even_distribution() {
        let x = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ];
        let y = vec![0, 1, 0, 1, 0, 1];
        let (_, _, y_train, y_test) = stratified_sampling(&x, &y, 2.0/3.0);

        // Count occurrences of each class in y_train and y_test
        let train_class_counts = count_classes(&y_train);
        let test_class_counts = count_classes(&y_test);
        let total_class_counts = count_classes(&y);

        // Verify that the proportion of each class in y_train and y_test is approximately the same
        for (class, count) in train_class_counts {
            let test_count = test_class_counts.get(&class).cloned().unwrap_or(0);
            let total_count = total_class_counts.get(&class).cloned().unwrap_or(0);
            assert_eq!(count + test_count, total_count, "train and test counts don't add up to total count");

            let expected_train_count = (total_count as f32 * 2.0/3.0) as usize;
            let expected_test_count = (total_count as f32 * 1.0/3.0) as usize;

            assert_eq!(count, expected_train_count, "train count for class {} is incorrect", class);
            assert_eq!(test_count, expected_test_count, "test count for class {} is incorrect", class);
        }
    }

    // Helper function to count occurrences of each class in a vector of labels
    fn count_classes(y: &[u8]) -> std::collections::HashMap<u8, usize> {
        let mut class_counts = std::collections::HashMap::new();
        for &label in y {
            *class_counts.entry(label).or_insert(0) += 1;
        }
        class_counts
    }

    #[test]
    fn test_standard_scaling() {
        let x = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let scaled_x = standard_scaling(&x);

        let mut x = DenseMatrix::from_2d_vec(&x).unwrap();
        x.standard_scale_mut(&x.mean(0), &x.std(0), 0);
        
        let (n_rows, n_cols) = x.shape();

        for i in 0..n_rows { // examples
            for j in (0..n_cols) { // features
                assert!((x.get((i, j)) -  scaled_x[i][j]).abs() < 1e-10);
            }
            println!("\n");
        }
    }
}
