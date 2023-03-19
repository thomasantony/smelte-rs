use crate::traits::{Tensor, TensorOps};
use crate::SmeltError;

/// Embedding layer. This layer is used to look up embeddings for a given set of
/// indices. The indices are expected to be in the range [0, num_embeddings).
#[derive(Clone)]
pub struct Embedding<T: Tensor> {
    weight: T,
}

impl<T: Tensor + TensorOps<T>> Embedding<T> {
    /// Creates a new embedding layer. The weight matrix is expected to have the
    /// shape [num_embeddings, embedding_dim].
    pub fn new(weight: T) -> Self {
        Self { weight }
    }

    /// Selects rows of the weight matrix based on the given indices. The indices
    /// are expected to be in the range [0, num_embeddings).
    pub fn forward(&self, ids: &[usize], out: &mut T) -> Result<(), SmeltError> {
        T::select(ids, &self.weight, out)
    }

    /// Returns a reference to the weight matrix.
    pub fn weight(&self) -> &T {
        &self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::f32::Tensor;

    #[test]
    fn test_embedding() {
        let weights = Tensor::zeros(vec![3, 2]);
        let embedding = Embedding::new(weights);
        let mut out = Tensor::zeros(vec![2, 2]);
        embedding.forward(&[0, 1], &mut out).unwrap();
    }

    #[test]
    fn test_embedding_errors() {
        let weights = Tensor::zeros(vec![3, 2]);
        let embedding = Embedding::new(weights);
        let mut out = Tensor::zeros(vec![2, 2]);
        assert!(embedding.forward(&[3], &mut out).is_err());
    }
}
