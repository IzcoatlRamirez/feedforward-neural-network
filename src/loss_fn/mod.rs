#[allow(dead_code)]
pub mod derivate {
    pub fn mse_derivative(activation: Vec<f64>, prediction: Vec<i32>) -> Vec<f64> {
        let mut result = vec![0.0; activation.len()];
        for i in 0..activation.len() {
            result[i] = activation[i] - (prediction[i] as f64);
        }
        return result;
    }

    pub fn cross_entropy_derivative(activation: Vec<f64>, prediction: Vec<i32>) -> Vec<f64> {
        let mut result = vec![0.0; activation.len()];
        for i in 0..activation.len() {
            result[i] = activation[i] - (prediction[i] as f64);
        }
        return result;
    }
}
