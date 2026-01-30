use rand::Rng;

use crate::value::Value;

/// Macro to create a Multi-Layer Perceptron (MLP) struct with static dimensions.
///
/// # Usage
///
/// The macro takes a comma-separated list of layer dimensions and generates a struct
/// that holds all layers and provides `new()`, `predict()`, and `parameters()` methods.
///
/// # Examples
///
/// ```
/// // Create a 3->4->1 network (3 inputs, 1 hidden layer with 4 neurons, 1 output)
/// let network = mlp!(3, 4, 1);
///
/// let input = [Value::new(1.0, "x1"), Value::new(2.0, "x2"), Value::new(3.0, "x3")];
/// let output = network.predict(&input);  // Compile error if wrong size!
/// let params = network.parameters();
/// ```
///
/// The macro supports networks with any number of layers!
#[macro_export]
macro_rules! mlp {
    ($in:literal, $($rest:literal),+ $(,)?) => {{
        mlp!(@build_struct $in, [$in, $($rest),+])
    }};

    (@build_struct $in:literal, [$($all:literal),+]) => {{
        struct Mlp {
            layers: Vec<Box<dyn std::any::Any>>,
        }

        impl Mlp {
            fn new() -> Self {
                let mut layers: Vec<Box<dyn std::any::Any>> = vec![] ;
                mlp!(@build_layers layers, [$($all),+]);

                Mlp { layers }
            }

            fn predict(&self, input: &[Value; $in]) -> [Value; mlp!(@last $($all),+)] {
                let mut idx = 0;
                mlp!(@predict_chain input, self.layers, idx, [$($all),+])
            }

            fn parameters(&self) -> Vec<Value> {
                let mut params = Vec::new();
                let mut idx = 0;
                mlp!(@collect_params params, self.layers, idx, [$($all),+]);
                params
            }
        }

        Mlp::new()
    }};

    // Get last element
    (@last $last:literal) => { $last };
    (@last $first:literal, $($rest:literal),+) => { mlp!(@last $($rest),+) };

    // Build layers recursively
    (@build_layers $layers:expr, [$in:literal, $out:literal]) => {
        $layers.push(Box::new(Layer::<$in, $out>::new()));
    };
    (@build_layers $layers:expr, [$in:literal, $next:literal $(, $rest:literal)+]) => {
        $layers.push(Box::new(Layer::<$in, $next>::new()));
        mlp!(@build_layers $layers, [$next $(, $rest)+]);
    };

    // Predict chain with type-safe array passing
    (@predict_chain $input:expr, $layers:expr, $idx:expr, [$in:literal, $out:literal]) => {{
        let layer = $layers[$idx].downcast_ref::<Layer<$in, $out>>().unwrap();
        layer.call($input)
    }};
    (@predict_chain $input:expr, $layers:expr, $idx:expr, [$in:literal, $next:literal $(, $rest:literal)+]) => {{
        let layer = $layers[$idx].downcast_ref::<Layer<$in, $next>>().unwrap();
        let output = layer.call($input);
        $idx += 1;
        mlp!(@predict_chain &output, $layers, $idx, [$next $(, $rest)+])
    }};

    // Collect parameters from all layers
    (@collect_params $params:expr, $layers:expr, $idx:expr, [$in:literal, $out:literal]) => {
        let layer = $layers[$idx].downcast_ref::<Layer<$in, $out>>().unwrap();
        $params.extend(layer.parameters());
    };
    (@collect_params $params:expr, $layers:expr, $idx:expr, [$in:literal, $next:literal $(, $rest:literal)+]) => {
        let layer = $layers[$idx].downcast_ref::<Layer<$in, $next>>().unwrap();
        $params.extend(layer.parameters());
        $idx += 1;
        mlp!(@collect_params $params, $layers, $idx, [$next $(, $rest)+]);
    };
}

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

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = vec![self.b.clone()];
        params.extend_from_slice(&self.w);
        params
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

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|n| n.parameters())
            .collect::<Vec<Value>>()
    }
}
