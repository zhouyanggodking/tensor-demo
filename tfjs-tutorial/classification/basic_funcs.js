const tf = require('@tensorflow/tfjs');

tf.oneHot(tf.tensor1d([0, 1, 2], 'int32'), 3).print()
tf.oneHot(tf.tensor1d([0, 1, 1, 1, 0, 0, 2], 'int32'), 3).print()

// console.log(tf.tensor1d([0, 1, 1, 1, 0, 0, 2]).shape[0])

const model = tf.sequential()

//constructing layer
const sigmoidLayer = tf.layers.dense({
  units: 32, // the demensionality of output space
  activation: 'sigmoid',
  inputShape: [50]
});

model.add(sigmoidLayer);
model.add(tf.layers.dense({
  units: 3,
  activation: 'softmax'
}));

model.summary();

const learningRate = 0.001;
const optimizer = tf.train.adam(learningRate); // see http://www.cnblogs.com/ljygoodgoodstudydaydayup/p/7294671.html

// configures and prepares the model for training and evaluation
// calling fit or evaluate on an un-compiled model will  throw an error.
model.compile({
  optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// omitting demension
const xTrain = [[]];
const yTrain = [[]];
const xTest = [[]];
const yTest =[[]];

const history = await model.fit(xTrain, yTrain, {
  epochs: 100,
  validationData: [xTest, yTest],
  callbacks: {
    onBatchEnd: async (epoch, logs) => {
      console.log(`epoch: ${epoch}----- logs: ${logs}`);
    }
  }
});

let sampleTenstor;
model.predict(sampleTenstor);

const result = tf.tensor1d([0.3, 0.5, 0.1]);
result.argMax().print();