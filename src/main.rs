mod activation_fn;
mod dataframe;
mod gradient_descend;
mod layer;
mod loss_fn;
mod network;
mod numrs;
use dataframe::datacsv::read_data_csv;
use serde::{Deserialize, Serialize};

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

fn main() {
    let num_classes = 2;
    let path = "diabetes.csv";

    let (x, y) = read_data_csv::<_, Person>(path, num_classes).unwrap();
    let (x_train, x_test, y_train, y_test) = dataframe::df::simple_split_one_hot(x, y, 0.7);

    println!("x_train: {:?}", x_train);
    println!("y_train: {:?}", y_train);
    println!("x_test: {:?}", x_test);
    println!("y_test: {:?}", y_test);
}

// let n_examples = 10;

// let x = one_hot_encoding(n_examples, 10);
// let y = one_hot_encoding_target(n_examples, 3);

// let (x_train, x_test, y_train, y_test) = dataframe::df::simple_split_one_hot(x, y, 0.7);

// println!("x_train: {:?}", x_train);
// println!("y_train: {:?}", y_train);
// println!("x_test: {:?}", x_test);
// println!("y_test: {:?}", y_test);
