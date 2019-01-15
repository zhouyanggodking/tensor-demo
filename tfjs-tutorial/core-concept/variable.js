const tf = require('@tensorflow/tfjs');

// variables are mutable
const init = tf.zeros([5]);
init.print();

const biases = tf.variable(init); // variable initialization
biases.print(0);

const newVal = tf.tensor1d([1, 2, 3, 4, 5]);
biases.assign(newVal);
biases.print();

