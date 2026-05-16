use std::{cell::Cell, rc::Rc};
use std::ops::{Add, Mul, Neg};
use std::hash::{Hash, Hasher};
use std::fmt::{self, Debug};
use std::collections::HashSet;

#[derive(Clone)]
pub struct Value {
    inner: Rc<Node>,
}

struct Node {
    data: f64,
    grad: Cell<f64>,
    prev: Vec<Value>,
    op: Op,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Leaf,
    Add,
    Mul,
    Relu,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self {
            inner: Rc::new(Node {
                data,
                grad: Cell::new(0.0),
                prev: Vec::new(),
                op: Op::Leaf,
            }),
        }
    }

    fn from_op(data: f64, prev: Vec<Value>, op: Op) -> Self {
        Self {
            inner: Rc::new(Node {
                data,
                grad: Cell::new(0.0),
                prev,
                op,
            }),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.data
    }

    pub fn grad(&self) -> f64 {
        self.inner.grad.get()
    }

    pub fn set_grad(&self, grad: f64) {
        self.inner.grad.set(grad);
    }

    pub fn add_grad(&self, grad: f64) {
        self.inner.grad.set(self.inner.grad.get() + grad);
    }

    pub fn relu(&self) -> Self {
        let data = if self.data() > 0.0 { self.data() } else { 0.0 };
        Self::from_op(data, vec![self.clone()], Op::Relu)
    }

    fn build_topo(&self, visited: &mut HashSet<Value>, topo: &mut Vec<Value>) {
        if visited.contains(self) {
            return;
        }

        visited.insert(self.clone());

        let prev = self.inner.prev.clone();

        for parent in prev {
            parent.build_topo(visited, topo);
        }

        topo.push(self.clone());
    }

    pub fn backward(&self) {
        let mut topo = vec![];
        let mut visited = HashSet::new();
    
        self.build_topo(&mut visited, &mut topo);
    
        for value in &topo {
            value.set_grad(0.0);
        }
    
        self.set_grad(1.0);
    
        for value in topo.into_iter().rev() {
            value._backward();
        }
    }

    fn _backward(&self) {
        let op = self.inner.op;
        let grad = self.inner.grad.get();
        let prev = self.inner.prev.clone();

        match op {
            Op::Leaf => {}

            Op::Add => {
                let left = &prev[0];
                let right = &prev[1];

                left.add_grad(grad);
                right.add_grad(grad);
            }

            Op::Mul => {
                let left = &prev[0];
                let right = &prev[1];

                let left_data = left.data();
                let right_data = right.data();

                left.add_grad(right_data * grad);
                right.add_grad(left_data * grad);
            }

            Op::Relu => {
                let input = &prev[0];

                let local_grad = if input.data() > 0.0 { 1.0 } else { 0.0 };

                input.add_grad(local_grad * grad);
            }
        }
    }

}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        let data = self.data() + rhs.data();

        Value::from_op(data, vec![self, rhs], Op::Add)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        let data = self.data() * rhs.data();
        Value::from_op(data, vec![self, rhs], Op::Mul)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
       self * Value::new(-1.0)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data())
            .field("grad", &self.grad())
            .finish()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.inner).hash(state);
    }
}

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(input_count: usize, seed: usize, nonlin: bool) -> Self {
        let weights = (0..input_count)
        .map(|i| Value::new(seeded_weight(seed + i)))
        .collect();

        let bias = Value::new(0.0);

        Self {
            weights,
            bias,
            nonlin,
        }
    }

    pub fn forward(&self, inputs: &[Value]) -> Value {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "number of inputs must match number of weights"
        );

        let mut out = self.bias.clone();

        // x_i * w_i
        for (input, weight) in inputs.iter().zip(self.weights.iter()) {
            out = out + input.clone() * weight.clone();
        }

        if self.nonlin {
            out.relu()
        } else {
            out
        }
    }
}

fn seeded_weight(seed: usize) -> f64 {
    let value = seed
        .wrapping_mul(1_664_525)
        .wrapping_add(1_013_904_223)
        % 10_000;

    let normalized = value as f64 / 10_000.0;

    normalized * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let a = Value::new(1.0);
        let b = a.relu();
        assert_eq!(b.data(), 1.0);
    }

    #[test]
    fn test_eq() {
        let a = Value::new(1.0);
        let b = Value::new(1.0);
        assert_ne!(a, b);
        assert_eq!(a, a);
    }

    #[test]
    fn test_add() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = a + b;
        assert_eq!(c.data(), 3.0);
    }
    
    #[test]
    fn test_mul() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = a * b;
        assert_eq!(c.data(), 2.0);
    }

    #[test]
    fn polynomial_backward() {
        let x = Value::new(3.0);

        let y = x.clone() * x.clone()
            + Value::new(2.0) * x.clone()
            + Value::new(1.0);

        y.backward();

        assert_eq!(y.data(), 16.0);

        // y = x^2 + 2x + 1
        // dy/dx = 2x + 2
        // at x = 3, dy/dx = 8
        assert_eq!(x.grad(), 8.0);
    }
}