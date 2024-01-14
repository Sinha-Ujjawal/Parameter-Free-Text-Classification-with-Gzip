use flate2::write::GzEncoder;
use flate2::Compression;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;

fn gzip_compress(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

fn gzip_compress_many(texts: Vec<&[u8]>) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    for bytes in texts {
        encoder.write_all(bytes).unwrap();
    }
    encoder.finish().unwrap()
}

#[derive(Clone, Debug)]
struct Point {
    bytes: Vec<u8>,
    label: usize,
    cbytes: usize,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct GZIPKNNClassifier {
    points: Vec<Point>,
}

#[pyclass]
pub struct Prediction {
    #[pyo3(get)]
    label: usize,
    #[pyo3(get)]
    distance: f64,
}

impl GZIPKNNClassifier {
    pub fn new(texts: Vec<Vec<u8>>, class_labels: Vec<usize>) -> Self {
        if texts.len() != class_labels.len() {
            panic!("Texts and class_labels must have the same length");
        }

        let points: Vec<_> = texts
            .par_iter()
            .zip(class_labels)
            .map(|(bytes, label)| Point {
                bytes: bytes.to_vec(),
                label,
                cbytes: gzip_compress(&bytes).len(),
            })
            .collect();

        GZIPKNNClassifier { points }
    }

    pub fn classify(&self, x: Vec<u8>, n_neighbors: usize) -> Option<Prediction> {
        let cx = gzip_compress(&x).len();
        let mut distances: Vec<_> = self
            .points
            .par_iter()
            .map(|point| {
                let Point {
                    bytes: y,
                    label,
                    cbytes: cy,
                } = point;
                let cxy = gzip_compress_many(vec![&x, &y]).len();
                let ncd = (cxy - cx.min(*cy)) as f64 / cx.max(*cy) as f64;
                (OrderedFloat(ncd), label)
            })
            .collect();
        distances.par_sort();
        distances.truncate(n_neighbors);

        let mut counter: HashMap<usize, (usize, f64)> = HashMap::new();

        for (distance, &label) in distances {
            let entry = counter.entry(label).or_insert((0, distance.0));
            entry.0 += 1;
            entry.1 = entry.1.min(distance.0);
        }

        counter
            .into_iter()
            .max_by_key(|&(_, (count, _))| count)
            .map(|(label, (_, distance))| Prediction{label, distance})
    }

    pub fn classify_many(&self, xs: Vec<Vec<u8>>, n_neighbors: usize) -> Vec<Option<Prediction>> {
        xs.par_iter()
            .map(|x| self.classify(x.to_vec(), n_neighbors))
            .collect()
    }
}

#[pymodule]
fn gzip_knn_classifier(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfunction]
    fn classify(classifier: &GZIPKNNClassifier, x: Vec<u8>, n_neighbors: usize) -> Option<Prediction> {
        classifier.classify(x, n_neighbors)
    }

    #[pyfunction]
    fn classify_many(
        classifier: &GZIPKNNClassifier,
        xs: Vec<Vec<u8>>,
        n_neighbors: usize,
    ) -> Vec<Option<Prediction>> {
        classifier.classify_many(xs, n_neighbors)
    }

    #[pyfunction]
    fn new(texts: Vec<Vec<u8>>, class_labels: Vec<usize>) -> GZIPKNNClassifier {
        GZIPKNNClassifier::new(texts, class_labels)
    }

    m.add_class::<Prediction>()?;
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    m.add_function(wrap_pyfunction!(classify_many, m)?)?;
    m.add_function(wrap_pyfunction!(new, m)?)?;

    Ok(())
}
