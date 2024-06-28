use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[allow(dead_code)]
pub fn randfloat(low: f64, high: f64, n: i32, seed: u64) -> Vec<f64> {
    let mut numbers = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n {
        let random_number = rng.gen_range(low..high);
        numbers.push(random_number);
    }
    return numbers;
}

/*corregir generacion de matriz aleatoria */
#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn randint(low: i32, high: i32, n: i32, seed: u64) -> Vec<i32> {
    let mut numbers = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n {
        let random_number = rng.gen_range(low..high + 1);
        numbers.push(random_number);
    }
    return numbers;
}

#[allow(dead_code)]
pub fn lineal_transform(w:Vec<Vec<f64>>,x: Vec<f64>)-> Vec<f64>{
    if w[0].len() != x.len(){
        panic!("The number of columns of the matrix w must be equal to the number of elements in the vector x");
    }
    let mut result = Vec::new();
    for i in 0..w.len(){
        let mut sum = 0.0;
        for j in 0..w[i].len(){
            sum += w[i][j] * x[j];
        }
        result.push(sum);
    }
    return result;
}

#[allow(dead_code)]
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