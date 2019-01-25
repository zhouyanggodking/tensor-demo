const fs = require('fs');
const sharp = require('sharp'); // using sharp nodejs module for image data reading

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const NUM_IMAGES = 65000;   // total 65000 images
const NUM_CLASSES = 10;  // 10 digits
const MNIST_IMAGE_PATH = 'mnist_images.png';
const MNIST_LABEL_PATH = 'mnist_labels_uint8';

// Mnist image data is 65000 * 784 image
// each row represent a single digit
async function convertMnistImageDataToArray() {
  const imgsBuf = await sharp(MNIST_IMAGE_PATH).extract({
    top: 0,
    left: 0,
    width: IMAGE_SIZE,
    height: NUM_IMAGES
  }).raw() // convert to all rgb values
    .toBuffer();
  const result = [];
  for (let i = 0; i < NUM_IMAGES; ++i) {
    const rowBuf = imgsBuf.slice(i * IMAGE_SIZE * 3, (i + 1) * IMAGE_SIZE * 3).filter((_, index) => index % 3 === 0);
    result.push([...rowBuf])
  }
  return result;
}

// mnist_labels_uint8 is one hot array
// each label takes 10 element
// total 65000 elements so the total buffer size is 65000 * 10
function convertMnistLabelDataToArray() {
  const fileBuf = fs.readFileSync(MNIST_LABEL_PATH);
  const labelBuf = new Uint8Array(fileBuf);
  const labelOnehotArr = [...labelBuf];
  const labelArr = []
  for (let i = 0; i < NUM_IMAGES; ++i) {
    labelArr.push(labelOnehotArr.slice(i * NUM_CLASSES, (i + 1) * NUM_CLASSES))
  }
  return labelArr;
}

async function saveMnistImageData(filename) {
  const dataArr = await convertMnistImageDataToArray();
  fs.writeFileSync(filename, JSON.stringify(dataArr));
}

function saveMnistLabelData(filename) {
  const labelArr = convertMnistLabelDataToArray();
  fs.writeFileSync(filename, JSON.stringify(labelArr));
}

module.exports = {
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  NUM_IMAGES,

  convertMnistImageDataToArray,
  convertMnistLabelDataToArray,
  saveMnistImageData,
  saveMnistLabelData
};