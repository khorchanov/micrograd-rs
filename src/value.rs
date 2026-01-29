use std::{cell::RefCell, fmt::Display, rc::Rc};

pub mod ops;

#[derive(Debug, Clone)]
pub enum Operation {
    Tanh(Rc<Value>),
    Exp(Rc<Value>),
    Pow(Rc<Value>, Rc<Value>),
    Add(Rc<Value>, Rc<Value>),
    Mul(Rc<Value>, Rc<Value>),
}

#[derive(Debug, Clone)]
pub struct Value {
    pub data: Rc<RefCell<f32>>,
    pub grad: Rc<RefCell<f32>>,
    pub op: Option<Operation>,
    pub label: Option<String>,
}

impl Value {
    pub fn new(data: f32, label: &str) -> Self {
        Value {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            op: None,
            label: Some(label.to_string()),
        }
    }

    pub fn tanh(&self) -> Self {
        let x = *self.data.borrow();
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Value {
            data: Rc::new(RefCell::new(t)),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Tanh(Rc::new(self.clone()))),
            label: Some(format!("tan({})", &self.label.clone().unwrap_or_default())),
        }
    }

    pub fn exp(&self) -> Self {
        let x = *self.data.borrow();
        Value {
            data: Rc::new(RefCell::new(x.exp())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Exp(Rc::new(self.clone()))),
            label: Some(format!("e^({})", &self.label.clone().unwrap_or_default())),
        }
    }

    pub fn pow(&self, other: Value) -> Self {
        let x = *self.data.borrow();
        Value {
            data: Rc::new(RefCell::new(x.powf(*other.data.borrow()))),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Pow(
                Rc::new(self.clone()),
                Rc::new(other.clone()),
            )),
            label: Some(format!(
                "({})^({})",
                &self.label.clone().unwrap_or_default(),
                *other.data.borrow()
            )),
        }
    }

    pub fn powf(&self, other: f32) -> Self {
        let other_value = Value::new(other, other.to_string().as_str());
        self.pow(other_value)
    }

    pub fn backward(&self) {
        match self.op {
            Some(Operation::Tanh(ref a)) => {
                let x = *a.data.borrow();
                let tanh = x.tanh();
                *a.grad.borrow_mut() += (1.0 - tanh.powi(2)) * *self.grad.borrow();
            }
            Some(Operation::Exp(ref a)) => {
                *a.grad.borrow_mut() += *self.data.borrow() * *self.grad.borrow();
            }
            Some(Operation::Pow(ref a, ref k)) => {
                let x = *a.data.borrow();
                let k_val = *k.data.borrow();
                *a.grad.borrow_mut() += k_val * x.powf(k_val - 1f32) * *self.grad.borrow();
            }
            Some(Operation::Add(ref a, ref b)) => {
                *a.grad.borrow_mut() += *self.grad.borrow();
                *b.grad.borrow_mut() += *self.grad.borrow();
            }
            Some(Operation::Mul(ref a, ref b)) => {
                *a.grad.borrow_mut() += *b.data.borrow() * *self.grad.borrow();
                *b.grad.borrow_mut() += *a.data.borrow() * *self.grad.borrow();
            }
            None => {}
        }
    }

    pub fn full_backward(&mut self) {
        let topo = self.reversed_topo();
        *self.grad.borrow_mut() = 1.0;
        for node in topo {
            node.backward();
        }
    }

    fn reversed_topo(&self) -> Vec<Value> {
        let mut nodes = Vec::new();

        fn build_topo(v: Value, nodes: &mut Vec<Value>) {
            if let Some(ref op) = v.op {
                match op {
                    Operation::Add(a, b) | Operation::Mul(a, b) | Operation::Pow(a, b) => {
                        build_topo((**a).clone(), nodes);
                        build_topo((**b).clone(), nodes);
                    }
                    Operation::Tanh(a) | Operation::Exp(a) => build_topo((**a).clone(), nodes),
                }
            }
            nodes.push(v);
        }

        build_topo(self.clone(), &mut nodes);
        nodes.reverse();
        nodes
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, grad={}, label={})",
            self.data.borrow(),
            self.grad.borrow(),
            self.label.as_deref().unwrap_or("None")
        )
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Value {
        Value::new(value, &format!("w{}", value))
    }
}
