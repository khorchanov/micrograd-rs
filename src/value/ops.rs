use std::{
    cell::RefCell,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use super::{Operation, Value};

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value {
            data: Rc::new(RefCell::new(*self.data.borrow() + *other.data.borrow())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Add(
                Rc::new(self.clone()),
                Rc::new(other.clone()),
            )),
            label: Some(
                self.label.clone().unwrap_or_default()
                    + "+"
                    + &other.label.clone().unwrap_or_default(),
            ),
        }
    }
}

impl Add<f32> for Value {
    type Output = Value;
    fn add(self, other: f32) -> Value {
        let other_value = Value::new(other, other.to_string().as_str());
        self + other_value
    }
}

impl Add<Value> for f32 {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let self_value = Value::new(self, self.to_string().as_str());
        self_value + other
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * -1f32
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + -other
    }
}

impl Sub<f32> for Value {
    type Output = Value;
    fn sub(self, other: f32) -> Value {
        let other_value = Value::new(other, other.to_string().as_str());
        self - other_value
    }
}

impl Sub<Value> for f32 {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        let self_value = Value::new(self, self.to_string().as_str());
        self_value - other
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        Value {
            data: Rc::new(RefCell::new(*self.data.borrow() * *other.data.borrow())),
            grad: Rc::new(RefCell::new(0.0)),
            op: Some(Operation::Mul(
                Rc::new(self.clone()),
                Rc::new(other.clone()),
            )),
            label: Some(
                self.label.clone().unwrap_or_default() + &other.label.clone().unwrap_or_default(),
            ),
        }
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.powf(-1f32)
    }
}

impl Div<f32> for Value {
    type Output = Value;
    fn div(self, other: f32) -> Value {
        self * other.powf(-1f32)
    }
}

impl Mul<f32> for Value {
    type Output = Value;
    fn mul(self, other: f32) -> Value {
        let other_value = Value::new(other, other.to_string().as_str());
        self * other_value
    }
}

impl Mul<Value> for f32 {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let self_value = Value::new(self, self.to_string().as_str());
        self_value * other
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0, "0"), |acc, x| acc + x)
    }
}
