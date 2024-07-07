//use crate::numrs::math::round_to_n_decimals;

pub mod derivate {
    //use crate::numrs::math::round_to_n_decimals;

    pub fn relu_derivative(x: Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            if x[i] > 0.0 {
                result[i] = 1.0;
            }
            result[i] = result[i];
        }
        return result;
    }
    pub fn softmax_derivative(x: Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            result[i] = x[i] * (1.0 - x[i]);
        }
        return result;
    }

    pub fn sigmoid_derivative(x: Vec<f64>) -> Vec<f64> {
        let mut result = vec![0.0; x.len()];
        for i in 0..x.len() {
            result[i] = x[i] * (1.0 - x[i]);
        }
        return result;
    }
}

pub fn sigmoid(mut x: Vec<f64>) -> Vec<f64> {
    for i in 0..x.len() {
        x[i] = 1.0 / (1.0 + (-x[i]).exp());
    }
    return x;
}

pub fn softmax(x: Vec<f64>) -> Vec<f64> {
    let mut sum = 0.0;
    let mut result = vec![0.0; x.len()];
    for i in 0..x.len() {
        sum += x[i].exp();
    }

    for i in 0..x.len() {
        result[i] = x[i].exp() / sum;
    }
    result
}

pub fn relu(mut x: Vec<f64>) -> Vec<f64> {
    for i in 0..x.len() {
        if x[i] < 0.0 {
            x[i] = 0.0;
        }
        x[i] = x[i];
    }

    return x;
}

