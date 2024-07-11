mod activation_fn;
mod dataframe;
mod gradient_descend;
mod layer;
mod loss_fn;
mod network;
mod numrs;
use dataframe::datacsv::read_data_csv;
use serde::{Deserialize, Serialize};
use numrs::{math::normalize_ouput, metrics::accuracy_score_ohe};

#[derive(Debug, Deserialize, Serialize)]
struct Person {
    pregnancies: f64,
    glucose: f64,
    blood_pressure: f64,
    skin_thickness: f64,
    insulin: f64,
    bmi: f64,
    diabetes_pedigree_function: f64,
    age: f64,
    outcome: i32,
}
/*
Notas:
    - accuracy mas alto alcanzado: 0.73
*/

fn main() {
    let num_classes = 2;
    let path = "diabetes.csv";

    let (x, y) = read_data_csv::<_, Person>(path, num_classes).unwrap();
    let (x_train, x_test, y_train, y_test) = dataframe::df::simple_split_one_hot(x, y, 0.70);

    let mut nn = network::NeuralNetwork::new(16, 8, "mse".to_string(), "gd".to_string(), "relu".to_string());
    nn.add(8, "relu".to_string());
    nn.add(4, "relu".to_string());
    nn.add(2, "sigmoid".to_string());

    let x_train_scaled = numrs::scaler::standard_scaler(x_train.clone());

    nn.fit(x_train_scaled, y_train, 0.00000001, 10);

    let x_test_scaled = numrs::scaler::standard_scaler(x_test.clone()); 

    let mut test = Vec::new();
    for i in 0..x_test.len() {
        test.push(normalize_ouput(nn.forward(x_test_scaled[i].clone())));
    }

    let accuracy = accuracy_score_ohe(y_test, test);
    println!("Accuracy: {}", accuracy);
  


}


