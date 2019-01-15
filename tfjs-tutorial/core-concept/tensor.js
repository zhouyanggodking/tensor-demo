const tf = require('@tensorflow/tfjs');

// tensors are immutable
// tensor
const shape = [2, 3];
const a = tf.tensor([1, 2, 3, 10, 20, 30], shape);  // specified by shape
a.print();

const b = tf.tensor([[1, 2, 3], [10, 20, 30]]); // shape infer
b.print();

// use tf.scalar, tf.tensor1d, tf.tensor2d, tf.tensor3d, tf.tensor4d

const scalar = tf.scalar(Math.PI);
scalar.print();
tf.tensor1d([Math.PI, Math.E]).print();

tf.zeros([5, 8]).print();
tf.ones([5, 8]).print();