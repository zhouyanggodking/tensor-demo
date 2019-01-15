const tf = require('@tensorflow/tfjs');

// demo for y = ax^3 + bx^2 + cx + d

// define your hypothesis function
function hypothesis(x) {
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3)))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

// using LMS
// define loss function using MSE (mean square error)
function loss(h, y) {
  return h.sub(y).square().mean(0);
}

// init variables
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

function train(x, y, numIter = 100) {
  // define optimizer
  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);
  for (let iter = 0; iter < numIter; ++iter) {
    optimizer.minimize(() => {
      const h = hypothesis(x);
      return loss(h, y);
    });
  }
}

function generateSampleData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c), tf.scalar(coeff.d)];
    const x = tf.randomUniform([numPoints], -1, 1);
    const y = a.mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d)
      .add(tf.randomNormal([numPoints], 0, sigma)); // random noise here
    
    // normalize the y values to the range 0 to 1
    const ymin = y.min();
    const ymax = y.max();
    const yNormalized = y.sub(ymin).div(ymax.sub(ymin));
    return {
      x,
      y: yNormalized
    };
  });
}

const sample = generateSampleData(100, {a: 1, b: 2, c: 3, d: 4});
train(sample.x, sample.y, 1000);

a.print();
b.print();
c.print();
d.print();