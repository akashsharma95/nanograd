use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul, Neg, Sub};
use std::{cell::Cell, rc::Rc};

#[derive(Clone)]
pub struct Value {
    inner: Rc<Node>,
}

struct Node {
    data: Cell<f64>,
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
                data: Cell::new(data),
                grad: Cell::new(0.0),
                prev: Vec::new(),
                op: Op::Leaf,
            }),
        }
    }

    fn from_op(data: f64, prev: Vec<Value>, op: Op) -> Self {
        Self {
            inner: Rc::new(Node {
                data: Cell::new(data),
                grad: Cell::new(0.0),
                prev,
                op,
            }),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.data.get()
    }

    pub fn set_data(&self, data: f64) {
        self.inner.data.set(data);
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

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        self + (-rhs)
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

        if self.nonlin { out.relu() } else { out }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_count: usize, output_count: usize, seed: usize, nonlin: bool) -> Self {
        let neurons = (0..output_count)
            .map(|i| Neuron::new(input_count, seed + i * (input_count + 1), nonlin))
            .collect();
        Self { neurons }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(input_count: usize, output_sizes: &[usize]) -> Self {
        assert!(!output_sizes.is_empty(), "output_sizes must be non-empty");

        let layer_count = output_sizes.len();
        let mut seed = 0;
        let mut in_size = input_count;

        let layers = output_sizes
            .iter()
            .enumerate()
            .map(|(i, &out_size)| {
                let layer = Layer::new(in_size, out_size, seed, i + 1 < layer_count);
                seed += out_size * (in_size + 1);
                in_size = out_size;
                layer
            })
            .collect();

        Self { layers }
    }

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut out = inputs.to_vec();

        for layer in &self.layers {
            out = layer.forward(&out);
        }

        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.set_grad(0.0);
        }
    }

    pub fn train_step(&self, inputs: &[&[Value]], targets: &[&[Value]], learning_rate: f64) -> f64 {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "number of inputs must match number of targets"
        );
        assert!(!inputs.is_empty(), "training batch must be non-empty");

        self.zero_grad();

        let mut loss = Value::new(0.0);

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let outputs = self.forward(input);

            for (output, target) in outputs.iter().zip(target.iter()) {
                let diff = output.clone() - target.clone();
                loss = loss + diff.clone() * diff;
            }
        }

        loss = loss * Value::new(1.0 / inputs.len() as f64);
        loss.backward();

        for param in self.parameters() {
            param.set_data(param.data() - learning_rate * param.grad());
        }

        loss.data()
    }

    pub fn train(
        &self,
        inputs: &[&[Value]],
        targets: &[&[Value]],
        learning_rate: f64,
        epochs: usize,
    ) -> Vec<f64> {
        (0..epochs)
            .map(|_| self.train_step(inputs, targets, learning_rate))
            .collect()
    }
}

fn seeded_weight(seed: usize) -> f64 {
    let value = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) % 10_000;

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

        let y = x.clone() * x.clone() + Value::new(2.0) * x.clone() + Value::new(1.0);

        y.backward();

        assert_eq!(y.data(), 16.0);

        // y = x^2 + 2x + 1
        // dy/dx = 2x + 2
        // at x = 3, dy/dx = 8
        assert_eq!(x.grad(), 8.0);
    }

    #[test]
    fn train_step_returns_loss_for_one_parameter_update() {
        let mlp = MLP::new(1, &[1]);
        let input = [Value::new(1.0)];
        let target = [Value::new(0.0)];

        let loss = mlp.train_step(&[&input], &[&target], 0.01);

        assert!(loss >= 0.0);
    }

    #[test]
    fn train_runs_for_each_epoch_and_returns_losses() {
        let mlp = MLP::new(1, &[1]);
        let input = [Value::new(1.0)];
        let target = [Value::new(0.0)];

        let losses = mlp.train(&[&input], &[&target], 0.01, 3);

        assert_eq!(losses.len(), 3);
    }
}
