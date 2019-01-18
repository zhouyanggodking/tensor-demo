const tf = require('@tensorflow/tfjs');

tf.linspace(0, 9, 10).print()

tf.tensor1d([1,2,3,4]).reshape([2, 2]).print()

tf.tensor1d([1,2]).dot(tf.tensor1d([3, 4])).print()

// f(x) = x ^ 2
const f = x => x.square();
// f'(x)
const g = tf.grad(f);

const x = tf.tensor1d([2, 3]);

g(x).print();