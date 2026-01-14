use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Debug, Clone)]
enum Operation {
    Add(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Mul(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Tanh(Rc<RefCell<Value>>),
}

#[derive(Debug, Clone)]
struct Value {
    data: f32,
    grad: f32,
    op: Option<Operation>,
}

impl Value {
    fn new(data: f32) -> Self {
        Value {
            data,
            grad: 0.0,
            op: None,
        }
    }

    fn tanh(self_rc: &Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let x = self_rc.borrow().data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Rc::new(RefCell::new(Value {
            data: t,
            grad: 0.0,
            op: Some(Operation::Tanh(Rc::clone(self_rc))),
        }))
    }

    fn backward(&mut self) {
        match self.op {
            Some(Operation::Add(ref mut a, ref mut b)) => {
                a.borrow_mut().grad += 1.0 * self.grad;
                b.borrow_mut().grad += 1.0 * self.grad;
            }
            Some(Operation::Mul(ref mut a, ref mut b)) => {
                a.borrow_mut().grad += b.borrow().data * self.grad;
                b.borrow_mut().grad += a.borrow().data * self.grad;
            }
            Some(Operation::Tanh(ref mut a)) => {
                let x = self.data;
                let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
                a.borrow_mut().grad += (1.0 - t.powi(2)) * self.grad;
            }
            None => {}
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let mut result = Value::new(self.data + other.data);
        let op = Some(Operation::Add(
            Rc::new(RefCell::new(self.clone())),
            Rc::new(RefCell::new(other.clone())),
        ));
        result.op = op;
        result
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data, self.grad)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let result = Value::new(self.data * other.data);
        Value {
            data: result.data,
            grad: 0.0,
            op: Some(Operation::Mul(
                Rc::new(RefCell::new(self)),
                Rc::new(RefCell::new(other)),
            )),
        }
    }
}

fn main() {
    let x1 = Value::new(2.0);
    let o = Value::tanh(&x1);
    o.borrow_mut().grad = 1.0;
    Value::backward(&o);

    println!("x1: {:#?}", x1);
    println!("Output: {:#?}", o);
}

fn _test() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.8813735870195432);
    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let mut o = n.tanh();
    o.grad = 1.0;
    o.backward();

    println!("n: {}", n);
    println!("Output: {:#?}", o);
}
