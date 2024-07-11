#![allow(dead_code)]
pub mod randgen {
    use rand::distributions::Distribution;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_distr::Normal;

    //use crate::numrs::math::round_to_n_decimals;
    pub fn randfloat(low: f64, high: f64, n: i32, seed: u64) -> Vec<f64> {
        let mut numbers = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..n {
            let random_number = rng.gen_range(low..high);
            numbers.push(random_number);
        }
        return numbers;
    }

    pub fn rand_vec(low:f64,high: f64,n: i32)-> Vec<f64>{
        let mut numbers = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..n {
            let random_number = rng.gen_range(low..high);
            numbers.push(random_number);
        }
        return numbers;
    }

    pub fn randfloatmatrix(low: f64, high: f64, rows: i32, cols: i32) -> Vec<Vec<f64>> {
        let mut matrix = Vec::new();
        let mut rng = rand::thread_rng();
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

    
    pub fn xavier_initialization(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (rows as f64 + cols as f64)).sqrt();
        let between = rand::distributions::Uniform::new(-limit, limit);
        
        (0..rows)
            .map(|_| (0..cols).map(|_| between.sample(&mut rng)).collect())
            .collect()
    }

    pub fn xavier_initialization_vec(n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / n as f64).sqrt();
        let between = rand::distributions::Uniform::new(-limit, limit);
        
        (0..n).map(|_| between.sample(&mut rng)).collect()
    }

    pub fn he_initialization_vec(n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / n as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).expect("Failed to create normal distribution");
        
        (0..n).map(|_| normal.sample(&mut rng)).collect()
    }

    pub fn he_initialization(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / rows as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).expect("Failed to create normal distribution");
        
        (0..rows)
            .map(|_| (0..cols).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

}

#[allow(dead_code)]
pub mod math {

    pub fn normalize_ouput(output: Vec<f64>) -> Vec<f64> {
        let max_index = find_max_index(output.clone());

        if output[max_index].is_nan() {
            panic!("The output contains NaN values");
        }

        let mut result = vec![0.0; output.len()];
        result[max_index] = 1.0;
        return result;
    }

    pub fn find_max_index(output: Vec<f64>) -> usize {
        let mut max_index = 0;
        let mut max_value = output[0];
        for i in 1..output.len() {
            if output[i] > max_value {
                max_value = output[i];
                max_index = i;
            }
        }
        return max_index;
    }

    pub fn clamped(values: Vec<f64>,min:f64,max:f64) -> Vec<f64> {
        let clamped_result: Vec<f64> = values
            .iter()
            .map(|&x| x.clamp(min, max))
            .collect();
        return clamped_result;
    }

    pub fn clamped_matrix(matrix: Vec<Vec<f64>>,min:f64,max:f64) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        for i in 0..matrix.len() {
            let row = clamped(matrix[i].clone(),min,max);
            result.push(row);
        }
        return result;
    }

    pub fn round_vec(value: Vec<f64>) -> Vec<f64> {
        let mut result = Vec::new();
        for i in 0..value.len() {
            result.push(round_f64(value[i]));
        }
        return result;
    }

    pub fn round_f64(value: f64) -> f64 {
        let n = 7;
        let factor = 10f64.powi(n as i32);
        (value * factor).round() / factor
    }

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
pub mod ohe {
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

    fn is_equal_vec(y_true: Vec<i32>, y_pred: Vec<f64>) -> bool {
        for i in 0..y_true.len() {
            if y_true[i] as f64 != y_pred[i] {
                return false;
            }
        }
        return true;
    }

    pub fn accuracy_score_ohe(y_true: Vec<Vec<i32>>, y_pred: Vec<Vec<f64>>) -> f64 {
        let mut correct = 0;
        for i in 0..y_true.len() {
            //println!("i : {:?}", i);
            if is_equal_vec(y_true[i].clone(), y_pred[i].clone()) {
                //println!("is equal y_true: {:?} y_pred: {:?}", y_true[i], y_pred[i]);
                correct += 1;
            }
            else {
                //println!("not equal y_true: {:?} y_pred: {:?}", y_true[i], y_pred[i]);
            
            }
        }
        println!("correct: {:?}", correct);
        return correct as f64 / y_true.len() as f64;
    }
}

pub mod scaler {

    pub fn standard_scaler(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        let mut means = Vec::new();
        let mut stds = Vec::new();
        let n = data.len() as f64;
        let m = data[0].len() as f64;
        for j in 0..m as usize {
            let mut sum = 0.0;
            for i in 0..n as usize {
                sum += data[i][j];
            }
            let mean = sum / n;
            means.push(mean);
            let mut sum = 0.0;
            for i in 0..n as usize {
                sum += (data[i][j] - mean).powi(2);
            }
            let std = (sum / n).sqrt();
            stds.push(std);
        }
        for i in 0..n as usize {
            let mut row = Vec::new();
            for j in 0..m as usize {
                let value = (data[i][j] - means[j]) / stds[j];
                row.push(value);
            }
            result.push(row);
        }
        return result;
    }
}
