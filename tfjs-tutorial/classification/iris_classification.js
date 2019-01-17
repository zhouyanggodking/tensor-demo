const tf = require('@tensorflow/tfjs');
const IRIS_DATA = require('./iris_data');

// steps
// 1. preprocess dataset to get features and labels
// 2. shuffle dataset, then split dataset to two part: 1) some training dataset 2) verifcation dataset
// 3. build model layers and train the model
// 4. make predictions on verification dataset to verify the model

// prepocess dataset
// splitRation = [0, 1]
// return {xTrain, yTrain, xTest, yTest}
function preprocessIrisDataset(dataset, splitRatio = 0.8) {
  const datasetLength = dataset.length;
  const trainDataLength = Math.floor(datasetLength * splitRatio);
  // const testDataLength = datasetLength - trainDataLength;
  // shuffle
  const indices = new Array(datasetLength).fill(0).map((_, index) => index);
  tf.util.shuffle(indices);

  const xTrain = [];
  const yTrain = [];
  const xTest = [];
  const yTest = [];
  indices.forEach((val, index) => {
    if (index < trainDataLength) {
      xTrain.push(dataset[val].slice(0, 4));
      yTrain.push(dataset[val][4]);
    } else {
      xTest.push(dataset[val].slice(0, 4));
      yTest.push(dataset[val][4]);
    }    
  });
  return {
    xTrain,
    yTrain,
    xTest,
    yTest
  };
}

// build model
function buildModel(trainDatasetLength) {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 32,
    activation: 'sigmoid',
    inputShape: [trainDatasetLength]
  }));

  model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
  }));

  model.summary();

  const learningRate = 0.001;
  const optimizer = tf.train.adam(learningRate);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// train
// xTrain/yTrain is just plain array
async function trainModel(model, xTrain, yTrain) {
  
  const x = tf.tensor2d(xTrain);
  const y = tf.oneHot(tf.tensor1d(yTrain, 'int32'), 3);
  const history = await model.fit(x, y, {
    batchSize: 32,
    epochs: 1000,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch);
        console.log(logs);
      }
    }
  });
  return model;
}

async function test() {
  const data = preprocessIrisDataset(IRIS_DATA);
  const model = buildModel(4);
  await trainModel(model, data.xTrain, data.yTrain);
  const preds = model.predict(tf.tensor2d(data.xTest)).argMax(1);
  labels = tf.tensor1d(data.yTest, 'int32');

  const right = preds.sub(labels).dataSync().filter(val => val === 0).length;
  console.log(preds.dataSync());
  console.log(data.yTest);
  console.log(`ratio: ${right/data.yTest.length}`);
} 

test();


