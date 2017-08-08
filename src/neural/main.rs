extern crate rand;
use std::f64;
use rand::{Rng, SeedableRng, StdRng};
use rand::distributions::{Range, IndependentSample};

struct Synapses {
    weight_x1: f64,
    weight_x2: f64,
    weight_x3: f64
}

struct DataSample {
    y: f64,
    x1: f64,
    x2: f64,
    x3: f64
}

fn main() {
    let seed: &[_] = &[1, 2, 3, 4];

    let func_to_approx = |x1: f64, x2: f64, x3: f64| -> f64 {sigmoid(1.0 * x1 - 0.7 * x2 + 0.3 * x3)};

    let data = generate_data(&func_to_approx, 100);

    let mut network = Network::new(SeedableRng::from_seed(seed), Range::new(0.0, 1.0));
    train(&mut network, &data[..], 5000);

    let test = DataSample { y: -100.0, x1: 0.3, x2: 0.9, x3: 0.2 };
    println!(
        "func_to_approx:{}\nnetwork:{}",
        func_to_approx(test.x1, test.x2, test.x3),
        forecast(&test, &network).y
    );
}

fn generate_data(func: &Fn(f64, f64, f64) -> f64, amount: usize) -> Vec<DataSample> {
    let mut rnd = rand::thread_rng();
    let range = Range::new(0.0, 1.0);
    (0..amount).map(|_| -> DataSample {
        let args = (0..3)
            .map(|_| -> f64 {range.ind_sample(&mut rnd)})
            .collect::<Vec<f64>>();
        let y = func(args[0], args[1], args[2]);
        DataSample {y, x1: args[0], x2: args[1], x3: args[2]}
    }).collect()
}

impl Learnable for Network {
    fn correct(&mut self, iteration: u32, expected: &DataSample, actual: &DataSample) {
        simple_correction(iteration, expected, actual, &mut self.synapses, 1.0)
    }

    fn activate(&self, x: f64) -> f64 {
        sigmoid(x)
    }
}

trait Learnable {
    fn correct(&mut self, iteration: u32, expected: &DataSample, actual: &DataSample);
    fn activate(&self, x: f64) -> f64;
}

struct Network {
    rnd: StdRng,
    synapses: Synapses
}
impl Network {
    fn new (mut rnd: StdRng, range: Range<f64>) -> Network {
        let synapses = Synapses {
            weight_x1: range.ind_sample(&mut rnd),
            weight_x2: range.ind_sample(&mut rnd),
            weight_x3: range.ind_sample(&mut rnd)
        };
        Network {
            synapses: synapses,
            rnd: rnd
        }
    }
}

fn train(network: &mut Network, train_data: &[DataSample], iterations: u32) {
    for i in 0..iterations {
        let sample = network.rnd.choose(&train_data).unwrap();
        let result = forecast(sample, network);
        network.correct(i, sample, &result);
    }
}

fn forecast(sample: &DataSample, network: &Network) -> DataSample {
    match (sample, network) {
        (&DataSample {x1, x2, x3, ..}, &Network{ref synapses, ..}) => {
            let x1_forecast = x1 * synapses.weight_x1;
            let x2_forecast = x2 * synapses.weight_x2;
            let x3_forecast = x3 * synapses.weight_x3;
            DataSample {
                y: network.activate(x1_forecast + x2_forecast + x3_forecast),
                x1: x1_forecast,
                x2: x2_forecast,
                x3: x3_forecast
            }
        }
    }
}

fn simple_correction(
    iteration: u32,
    expected: &DataSample,
    actual: &DataSample,
    synapses: &mut Synapses,
    rate: f64
) {
    println!(
        "w1:{}\nw2:{}\nw3:{}",
        synapses.weight_x1,
        synapses.weight_x2,
        synapses.weight_x3
    );
    let error = expected.y - actual.y;
    println!("Iteration:{}, error: {}\n", iteration, error);
    synapses.weight_x1 += error * (expected.x1 - actual.x1) * rate;
    synapses.weight_x2 += error * (expected.x2 - actual.x2) * rate;
    synapses.weight_x3 += error * (expected.x3 - actual.x3) * rate;
}

fn gate(val: f64) -> f64 {
    (((val >= 0.0) as i8) as f64)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
