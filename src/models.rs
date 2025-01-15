use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::model_selection::train_test_split;
use smartcore::metrics::accuracy;
use smartcore::linalg::traits::stats::MatrixStats;



pub fn train_model(x: Vec<Vec<f64>>, y: Vec<u8>) {

    let mut x = DenseMatrix::from_2d_vec(&x).unwrap();
    if x.iter().any(|&i| i.is_nan() || i.is_infinite()) {
        panic!["Data contains NaN or Infinite values"];
    }
    x.standard_scale_mut(&x.mean(0), &x.std(0), 0);


    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.8, true, Option::None);

    let knn = KNNClassifier::fit(&x_train, &y_train, KNNClassifierParameters::default()).unwrap();

    let y_pred = knn.predict(&x_test).unwrap();
    let accuracy = accuracy(&y_test, &y_pred);

    println!("Accuracy: {:.2}", accuracy);
}
