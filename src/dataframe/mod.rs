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

#[allow(dead_code)]
pub mod datacsv {

    use csv::Reader;
    use serde::{Deserialize, Serialize};
    use std::error::Error;
    use std::fmt::Debug;
    use std::fs::File;
    use std::path::Path;

    pub fn read_csv<P, T>(path: P) -> Result<Vec<T>, Box<dyn Error>>
    where
        P: AsRef<Path>,
        T: for<'de> Deserialize<'de> + Debug,
    {
        let file = File::open(path)?;
        let mut rdr = Reader::from_reader(file);
        let mut records = Vec::new();

        for result in rdr.deserialize() {
            let record: T = result?;
            records.push(record);
        }

        Ok(records)
    }

    pub fn read_data_csv<P, T>(
        path: P,
        num_classes: i32,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<i32>>), Box<dyn Error>>
    where
        P: AsRef<Path>,
        T: for<'de> Deserialize<'de> + Debug + Serialize, // Restricción para deserialización y trait debug
    {
        let records: Vec<T> = read_csv(path)?;
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<Vec<i32>> = Vec::new();

        for record in records {
            let (x_row, y_row) = split_record(record, num_classes);
            x.push(x_row);
            y.push(y_row);
        }

        Ok((x, y))
    }

    fn split_record<T>(record: T, num_classes: i32) -> (Vec<f64>, Vec<i32>)
    where
        T: Serialize,
    {
        let record = serde_json::to_string(&record).unwrap();
        let record: serde_json::Value = serde_json::from_str(&record).unwrap();
        let mut x_row: Vec<f64> = Vec::new();
        let mut y_row: Vec<i32> = Vec::new();

        for (key, value) in record.as_object().unwrap() {
            if key != "outcome" {
                x_row.push(value.as_f64().unwrap());
            } else {
                y_row = create_one_hot_encoding(value.as_i64().unwrap() as i32, num_classes);
            }
        }
        (x_row, y_row)
    }

    fn create_one_hot_encoding(y: i32, num_classes: i32) -> Vec<i32> {
        let mut one_hot_encoding = vec![0; num_classes as usize];
        one_hot_encoding[y as usize] = 1;
        one_hot_encoding
    }
}
