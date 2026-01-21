use rand::Rng;

use crate::Value;

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
