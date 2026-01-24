use rand::Rng;

use crate::value::Value;

pub struct Neuron<const N: usize> {
    pub w: [Value; N],
    pub b: Value,
}

impl<const N: usize> Neuron<N> {
    pub fn new() -> Neuron<N> {
        let w = std::array::from_fn(|i| {
            Value::new(rand::rng().random_range(-1.0..1.0), &format!("w{}", i))
        });
        let b = Value::new(rand::rng().random_range(-1.0..1.0), "b");
        Neuron { w, b }
    }

    pub fn call(&self, x: &[Value; N]) -> Value {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = act + wi.clone() * xi.clone();
        }
        act.tanh()
    }
}

pub struct Layer<const N: usize, const M: usize> {
    pub neurons: [Neuron<N>; M],
}

impl<const N: usize, const M: usize> Layer<N, M> {
    pub fn new() -> Layer<N, M> {
        let neurons = std::array::from_fn(|_| Neuron::<N>::new());
        Layer { neurons }
    }

    pub fn call(&self, x: &[Value; N]) -> [Value; M] {
        std::array::from_fn(|i| self.neurons[i].call(x))
    }
}
