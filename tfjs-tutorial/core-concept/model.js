const tf = require('@tensorflow/tfjs');

// model, conceptually, is a function that given some input will produce some desired output

// define function 
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  const a = tf.scalar(2);
  const b = tf.scalar(4);
  const c = tf.scalar(8);

  return tf.tidy(() => {
    const x = tf.scalar(input);
    return tf.mul(a, x.square()).add(b.mul(x)).add(c);
  });
}

predict(2).print();