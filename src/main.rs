use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Debug, Clone)]
enum Operation {
    Tanh(Rc<RefCell<Value>>),
    Add(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Mul(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
}

#[derive(Debug, Clone)]
struct Value {
    data: Rc<RefCell<f32>>,
    grad: Rc<RefCell<f32>>,
    op: Option<Operation>,
}

impl Value {
    fn new(data: f32) -> Self {
        Value {
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(0.0)),
            op: None,
        }
    }

    fn tanh(&self) -> Self {
        let x = *self.data.borrow();
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        Value {
            data: Rc::new(RefCell::new(t)),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Tanh(Rc::new(RefCell::new(self.clone())))),
        }
    }

    fn backward(&self) {
        match self.op {
            Some(Operation::Tanh(ref a)) => {
                let x = *a.borrow().data.borrow();
                let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
                *a.borrow_mut().grad.borrow_mut() += (1.0 - t.powi(2)) * *self.grad.borrow();
            }
            Some(Operation::Add(ref a, ref b)) => {
                *a.borrow_mut().grad.borrow_mut() += *self.grad.borrow();
                *b.borrow_mut().grad.borrow_mut() += *self.grad.borrow();
            }
            Some(Operation::Mul(ref a, ref b)) => {
                *a.borrow_mut().grad.borrow_mut() += *b.borrow().data.borrow() * *self.grad.borrow();
                *b.borrow_mut().grad.borrow_mut() += *a.borrow().data.borrow() * *self.grad.borrow();
            }
            None => {}
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, grad={})",
            self.data.borrow(),
            self.grad.borrow()
        )
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let result = Value {
            data: Rc::new(RefCell::new(*self.data.borrow() + *other.data.borrow())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Add(
                Rc::new(RefCell::new(self.clone())),
                Rc::new(RefCell::new(other.clone())),
            )),
        };
        result
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let result = Value {
            data: Rc::new(RefCell::new(*self.data.borrow() * *other.data.borrow())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Mul(
                Rc::new(RefCell::new(self.clone())),
                Rc::new(RefCell::new(other.clone())),
            )),
        };
        result
    }
}

fn main() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(1.0);
    let s = x1.clone() * x2.clone();
    let o = s.tanh();
    *o.grad.borrow_mut() = 1.0;
    o.backward();
    s.backward();

    println!("x1: {}", x1);
    println!("x2: {}", x2);
    println!("s: {}", s);
    println!("Output: {:#?}", o);
}

fn _test() {
    // let x1 = Value::new(2.0);
    // let x2 = Value::new(0.0);
    // let w1 = Value::new(-3.0);
    // let w2 = Value::new(1.0);
    // let b = Value::new(6.8813735870195432);
    // let x1w1 = x1 * w1;
    // let x2w2 = x2 * w2;
    // let x1w1x2w2 = x1w1 + x2w2;
    // let n = x1w1x2w2 + b;
    // let mut o = n.tanh();
    // o.grad = 1.0;
    // o.backward();

    // println!("n: {}", n);
    // println!("Output: {:#?}", o);
}
