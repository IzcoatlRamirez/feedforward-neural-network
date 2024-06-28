#[allow(dead_code)]
pub fn simple_split(
    x: Vec<Vec<f64>>,
    y: Vec<i32>,
    train_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<i32>, Vec<i32>) {
    let n_test = x[0].len() as f64 * train_ratio;
    let transposed = transpose_vec(x);
    let n_train = transposed.len() as f64 * train_ratio;

    let x_train = transposed[0..n_train as usize].to_vec();
    let x_test = transposed[(n_train as usize)..].to_vec();

    let y_train = y[0..n_test as usize].to_vec();
    let y_test = y[(n_test as usize)..].to_vec();

    (x_train, x_test, y_train, y_test)
}

pub fn transpose_vec(x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = x[0].len();
    let mut transposed: Vec<Vec<f64>> = vec![vec![0.0; x.len()]; n];

    for i in 0..n {
        for j in 0..x.len() {
            transposed[i][j] = x[j][i];
        }
    }

    transposed
}