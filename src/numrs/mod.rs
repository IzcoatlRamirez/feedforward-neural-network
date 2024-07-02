#![allow(dead_code)]
pub mod randgen {
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    pub fn randfloat(low: f64, high: f64, n: i32, seed: u64) -> Vec<f64> {
        let mut numbers = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..n {
            let random_number = rng.gen_range(low..high);
            numbers.push(random_number);
        }
        return numbers;
    }

    pub fn randfloatmatrix(low: f64, high: f64, rows: i32, cols: i32, seed: u64) -> Vec<Vec<f64>> {
        let mut matrix = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..rows {
            let mut row = Vec::new();
            for _ in 0..cols {
                let random_number = rng.gen_range(low..high);
                row.push(random_number);
            }
            matrix.push(row);
        }
        return matrix;
    }

    pub fn randint(low: i32, high: i32, n: i32, seed: u64) -> Vec<i32> {
        let mut numbers = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..n {
            let random_number = rng.gen_range(low..high + 1);
            numbers.push(random_number);
        }
        return numbers;
    }
}

#[allow(dead_code)]
pub mod math {
    pub fn lineal_transform(w: Vec<Vec<f64>>, x: Vec<f64>) -> Vec<f64> {
        if w[0].len() != x.len() {
            panic!("The number of columns of the matrix w must be equal to the number of elements in the vector x");
        }
        let mut result = Vec::new();
        for i in 0..w.len() {
            let mut sum = 0.0;
            for j in 0..w[i].len() {
                sum += w[i][j] * x[j];
            }
            result.push(sum);
        }
        return result;
    }

    pub fn add_vecs(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
        if a.len() != b.len() {
            panic!("The number of elements in the vectors must be equal");
        }
        let mut result = Vec::new();
        for i in 0..a.len() {
            result.push(a[i] + b[i]);
        }
        return result;
    }

    pub fn hadamard(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
        if a.len() != b.len() {
            panic!("The number of elements in the vectors must be equal");
        }
        let mut result = Vec::new();
        for i in 0..a.len() {
            result.push(a[i] * b[i]);
        }
        return result;
    }

    pub fn outer(a: Vec<f64>, b: Vec<f64>) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        for element in a.iter() {
            let mut row = Vec::new();
            for element2 in b.iter() {
                row.push(element * element2);
            }
            result.push(row);
        }
        return result;
    }

    pub fn tranpose(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if matrix.is_empty() || matrix[0].is_empty() {
            return Vec::new();
        }

        let nrows = matrix.len();
        let ncols = matrix[0].len();
        let mut result = vec![vec![0.0; nrows]; ncols];

        for i in 0..nrows {
            for j in 0..ncols {
                result[j][i] = matrix[i][j];
            }
        }
        result
    }
}
//funcion que crea n vectores one-hot-encoding
#[allow(dead_code)]
pub mod ohe{
    use rand::Rng;
    pub fn one_hot_encoding_target(n_vectors: i32, n_classes: i32) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        for _ in 0..n_vectors {
            let mut vector = vec![0; n_classes as usize];
            let random_index = rand::thread_rng().gen_range(0..n_classes);
            vector[(random_index) as usize] = 1;
            result.push(vector);
        }
        return result;
    }
    
    pub fn one_hot_encoding(n_examples: i32, n_dimens: i32) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        for _ in 0..n_examples {
            let mut vector = vec![0.0; n_dimens as usize];
            let random_index = rand::thread_rng().gen_range(0..n_dimens);
            vector[(random_index) as usize] = 1.0;
            result.push(vector);
        }
        return result;
    }
}

#[allow(dead_code)]
pub mod metrics {
    pub fn accuracy_score(y_true: Vec<i32>, y_pred: Vec<i32>) -> f64 {
        let mut correct = 0;
        for i in 0..y_true.len() {
            if y_true[i] == y_pred[i] {
                correct += 1;
            }
        }
        return correct as f64 / y_true.len() as f64;
    }
}
