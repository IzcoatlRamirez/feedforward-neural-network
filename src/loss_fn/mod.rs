#[allow(dead_code)]
pub mod derivate {
    //use crate::numrs::math::round_to_n_decimals;

    pub fn mse_derivative(activation: Vec<f64>, prediction: Vec<i32>) -> Vec<f64> {
        let mut result = vec![0.0; activation.len()];
        for i in 0..activation.len() {
            //result[i] = round_to_n_decimals(activation[i] - (prediction[i] as f64));
            result[i] = activation[i] - (prediction[i] as f64);
        }
        return result;
    }

    pub fn cross_entropy_derivative(activation: Vec<f64>, prediction: Vec<i32>) -> Vec<f64> {
        let mut result = vec![0.0; activation.len()];
        for i in 0..activation.len() {
            //result[i] = round_to_n_decimals(activation[i] - (prediction[i] as f64));
            result[i] = activation[i] - (prediction[i] as f64);
        }
        return result;
    }
}

#[allow(dead_code)]
pub mod loss{
    pub fn mse(activation: Vec<f64>, prediction: Vec<i32>) -> f64 {
        let mut result = 0.0;
        for i in 0..activation.len() {
            result += (activation[i] - (prediction[i] as f64)).powi(2);
        }
        return result / (2 as f64);
    }
    
    pub fn cross_entropy(activation: Vec<f64>, prediction: Vec<i32>) -> f64 {
        let mut result = 0.0;
        for i in 0..activation.len() {
            result += -1.0 * (prediction[i] as f64) * activation[i].ln();
        }
        return result;
    }
    
}