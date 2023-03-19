use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

/// Layer normalization layer. This layer is used to normalize the activations of
/// the previous layer at each batch.
#[derive(Clone)]
pub struct LayerNorm<T: Tensor> {
    weight: T,
    bias: T,
    epsilon: f32,
}

impl<T: Tensor + TensorOps<T>> LayerNorm<T> {
    /// Creates a new layer normalization layer. The weight and bias tensors are
    /// expected to have the same shape as the input tensor.
    /// The epsilon parameter is used to prevent division by zero.
    pub fn new(weight: T, bias: T, epsilon: f32) -> Self {
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    /// Normalizes the given tensor using layer normalization.
    /// The tensor is expected to have the shape [batch_size, num_features].
    /// The weight and bias tensors are expected to have the shape [num_features].
    /// The output tensor has the same shape as the input tensor.
    pub fn forward(&self, tensor: &mut T) -> Result<(), SmeltError> {
        T::normalize(tensor, self.epsilon)?;
        T::mul(&self.weight, tensor)?;
        T::add(&self.bias, tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::f32::Tensor;

    #[test]
    fn test_layer_norm() {
        let mut zeros = Tensor::zeros(vec![3, 2]);
        let weights = Tensor::zeros(vec![3, 2]);
        let bias = Tensor::zeros(vec![2]);

        let linear = LayerNorm::new(weights, bias, 1e-5);

        linear.forward(&mut zeros).unwrap();
    }
}
