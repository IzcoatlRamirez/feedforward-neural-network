#[allow(dead_code)]
pub mod df {
    pub fn simple_split(
        x: Vec<Vec<f64>>,
        y: Vec<i32>,
        train_ratio: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<i32>, Vec<i32>) {
        let n_test = x[0].len() as f64 * train_ratio;
        let transposed = transpose_matrix(x);
        let n_train = transposed.len() as f64 * train_ratio;

        let x_train = transposed[0..n_train as usize].to_vec();
        let x_test = transposed[(n_train as usize)..].to_vec();

        let y_train = y[0..n_test as usize].to_vec();
        let y_test = y[(n_test as usize)..].to_vec();

        (x_train, x_test, y_train, y_test)
    }

    pub fn transpose_matrix<T: Clone + Default>(x: Vec<Vec<T>>) -> Vec<Vec<T>> {
        if x.is_empty() || x[0].is_empty() {
            return Vec::new();
        }

        let n = x[0].len();
        let mut transposed: Vec<Vec<T>> = vec![vec![T::default(); x.len()]; n];

        for i in 0..n {
            for j in 0..x.len() {
                transposed[i][j] = x[j][i].clone();
            }
        }

        transposed
    }

    /*no hace falta transpuesta */
    pub fn simple_split_one_hot(
        x: Vec<Vec<f64>>,
        y: Vec<Vec<i32>>,
        train_ratio: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<i32>>, Vec<Vec<i32>>) {
        let n_test = y.len() as f64 * train_ratio;
        let n_train = x.len() as f64 * train_ratio;

        let x_train = x[0..n_train as usize].to_vec();
        let x_test = x[(n_train as usize)..].to_vec();

        let y_train = y[0..n_test as usize].to_vec();
        let y_test = y[(n_test as usize)..].to_vec();
        (x_train, x_test, y_train, y_test)
    }
}
