const tf = require('@tensorflow/tfjs');

const a = tf.tensor2d([[1, 2], [3, 4]]);
const a_squared = a.square(); // tf.square(a).print();
a_squared.print();

const b = tf.tensor2d([[5, 6], [7, 8]]);
// element wise operation
b.add(a).print(); // tf.add(b, a).print();
b.sub(a).print(); // tf.sub(b, a).print()
b.mul(a).print(); // tf.mul(b, a).print()
