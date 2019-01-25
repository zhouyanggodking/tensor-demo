const tf = require('@tensorflow/tfjs');
const data_util = require('./mnist_data');

// steps
// 1. preprocess dataset to get features and labels
// 2. shuffle dataset, then split dataset to two part: 1) some training dataset 2) verifcation dataset
// 3. build model layers and train the model
// 4. make predictions on verification dataset to verify the model

const IMAGE_WIDTH = data_util.IMAGE_WIDTH;
const IMAGE_HEIGHT = data_util.IMAGE_HEIGHT;
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const NUM_CHANNELS = 1; // grey scale
const NUM_CLASSES = 10;
const NUM_DATA = data_util.NUM_IMAGES;
const NUM_TRAIN = 55000;
const NUM_TEST = NUM_DATA - NUM_TRAIN;

async function preprocessMnistDataSet() {
  // shuffle dataset
  const indices = new Array(NUM_DATA).fill(0).map((_, index) => index);
  tf.util.shuffle(indices);
  const xTrain = [];
  const yTrain = [];
  const xTest = [];
  const yTest = [];

  const imageDataArr = await data_util.convertMnistImageDataToArray();
  const labelDataArr = data_util.convertMnistLabelDataToArray();
  indices.forEach((val, index) => {
    if (index < NUM_TRAIN) {
      [].push.apply(xTrain, imageDataArr[val]);
      yTrain.push(labelDataArr[val]);
    } else {
      [].push.apply(xTest, imageDataArr[val]);
      yTest.push(labelDataArr[val]);
    }    
  });

  return {
    xTrain,
    yTrain,
    xTest,
    yTest
  };
}

function buildConvModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPool2d({
    poolSize: 2,
    strides: 2
  }));
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters:32, 
    activation: 'relu'
  }));

  model.add(tf.layers.flatten({}));
  model.add(tf.layers.dense({
    units: 64,
    activation:'relu'
  }));
  model.add(tf.layers.dense({
    units: 10,
    activation:'softmax'
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
  return tf.tidy(async () => {
    const x = tf.tensor4d(xTrain, [NUM_TRAIN, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]);
    const y = tf.tensor2d(yTrain, [NUM_TRAIN, NUM_CLASSES]);
    const history = await model.fit(x, y, {
      batchSize: 128,
      epochs: 20,
      validationSplit: 0.15,
      // callbacks: {
      //   onEpochEnd: (epoch, logs) => {
      //     console.log(epoch);
      //     console.log(logs);
      //   }
      // }
    });
    return model;
  });
}

async function test() {
  const data = await preprocessMnistDataSet();
  const model = buildConvModel();
  await trainModel(model, data.xTrain, data.yTrain);
}

test();