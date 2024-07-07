mod activation_fn;
mod dataframe;
mod gradient_descend;
mod layer;
mod loss_fn;
mod network;
mod numrs;
use dataframe::datacsv::read_data_csv;
use serde::{Deserialize, Serialize};
use numrs::metrics::accuracy_score_ohe;

/*automatizar crear el struct pertinente?*/
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


/*implementar ajustar bias 
revisar clamp de los datos (hacerlo en cada operacion con el fin de evitar desbordamiento no permite actualizar los peso
s de  manera correcta?)
eliminar rounded en operaciones ya que al final se usara clamp?
hacer clamp de los outputs no de los pesos o deltas?
*/

fn main() {
    let num_classes = 2;
    let path = "diabetes.csv";

    let (x, y) = read_data_csv::<_, Person>(path, num_classes).unwrap();
    let (x_train, x_test, y_train, y_test) = dataframe::df::simple_split_one_hot(x, y, 0.70);

    let mut nn = network::NeuralNetwork::new(8, 8, "mse".to_string(), "gd".to_string(), "relu".to_string());

    nn.add(4, "relu".to_string());
    nn.add(2, "sigmoid".to_string());

    let x_train_scaled = numrs::scaler::standard_scaler(x_train.clone());

    nn.fit(x_train_scaled, y_train, 0.1, 1);

    let x_test_scaled = numrs::scaler::standard_scaler(x_test.clone()); 

    let mut test = Vec::new();
    for i in 0..x_test.len() {
        test.push(nn.forward(x_test_scaled[i].clone()));
    }

    let accuracy = accuracy_score_ohe(y_test, test);
    println!("Accuracy: {}", accuracy);

}


