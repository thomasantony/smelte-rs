use crate::SmeltError;

/// A trait for tensors that can be used in the neural network.
/// The tensor must be cloneable and have a shape.
/// The tensor must also be able to create a tensor of zeros with a given shape.
pub trait Tensor: Clone {
    /// Returns the shape of the tensor.
    fn shape(&self) -> &[usize];
    /// Initializes a tensor of zeros with the given shape.
    fn zeros(shape: Vec<usize>) -> Self;
}

/// All common tensor operations
pub trait TensorOps<T>:
    TensorMatmul<T>
    + TensorMatmulT<T>
    + TensorAdd<T>
    + TensorMul<T>
    + TensorNormalize<T>
    + TensorSelect<T>
    + TensorGelu<T>
    + TensorTanh<T>
    + TensorSoftmax<T>
{
}

/// The matmul operation.
pub trait TensorMatmul<T> {
    /// Performs the matmul operation.
    fn matmul(a: &T, b: &T, c: &mut T) -> Result<(), SmeltError>;
}

/// The matmul operation with transposed second tensor.
pub trait TensorMatmulT<T> {
    /// Performs the matmul operation with transposed second tensor.
    fn matmul_t(a: &T, b: &T, c: &mut T) -> Result<(), SmeltError>;
}

/// The add operation.
pub trait TensorAdd<T> {
    /// Performs the add operation.
    fn add(a: &T, b: &mut T) -> Result<(), SmeltError>;
}

/// The elementwise multiplication operation.
pub trait TensorMul<T> {
    /// Performs the elementwise multiplication operation.
    fn mul(a: &T, b: &mut T) -> Result<(), SmeltError>;
}

/// The normalization operation is used to normalize the elements in the tensor.
pub trait TensorNormalize<T> {
    /// Performs the normalization operation on the tensor.
    /// The epsilon parameter is used to prevent division by zero.
    fn normalize(x: &mut T, epsilon: f32) -> Result<(), SmeltError>;
}

/// The select operation is used to select a subset of the elements in the tensor.
pub trait TensorSelect<T> {
    /// Performs the select operation on the tensor.
    /// The weight tensor is expected to have the same shape as the input tensor.
    /// The output tensor is expected to have the same shape as the input tensor.
    /// The x parameter is a list of indices that are used to select the elements
    fn select(x: &[usize], weight: &T, out: &mut T) -> Result<(), SmeltError>;
}

/// The gelu operation is used to apply the gelu activation function to the tensor.
pub trait TensorGelu<T> {
    /// Applies the gelu activation function to the tensor.
    fn gelu(x: &mut T) -> Result<(), SmeltError>;
}

/// The tanh operation is used to apply the tanh activation function to the tensor.
pub trait TensorTanh<T> {
    /// Applies the tanh activation function to the tensor.
    fn tanh(x: &mut T) -> Result<(), SmeltError>;
}

/// The softmax operation is used to apply the softmax activation function to the tensor.
pub trait TensorSoftmax<T> {
    /// Applies the softmax activation function to the tensor.
    fn softmax(x: &mut T) -> Result<(), SmeltError>;
}
