const fs = require('fs');
const sharp = require('sharp');

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const NUM_IMAGES = 65000;
const NUM_CLASSES = 10;
const MNIST_IMAGE_PATH = 'mnist_images.png';
const MNIST_LABEL_PATH = 'mnist_labels_uint8';


const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

async function readMnistImageData() {
  const imgsBuf = await sharp(MNIST_IMAGE_PATH).extract({
    top: 0,
    left: 0,
    width: IMAGE_SIZE,
    height: NUM_IMAGES
  }).raw() // convert to all rgb values
    .toBuffer();
  const images = new Array(NUM_IMAGES).fill(0); // create NUM_IMAGES 
  return images.map((_, index) => {
    const image_data = imgsBuf.slice(index * IMAGE_SIZE * 3, (index + 1) * IMAGE_SIZE * 3).filter((_, i) => i % 3 === 0);
    return to2DImageData([...image_data]);
  })
}

// convert 1-d image data to 2-d image data
function to2DImageData(image_data_1d) {
  if (image_data_1d.length !== IMAGE_SIZE) {
    throw new Error('image data length doesn\'t match');    
  }

  const imageRows = new Array(IMAGE_HEIGHT).fill(0);
  return imageRows.map((_, rowIndex) => image_data_1d.slice(rowIndex * IMAGE_WIDTH, (rowIndex + 1) * IMAGE_WIDTH));  
}

readMnistImageData().then(data => {
  console.log(data.length)
  fs.writeFileSync('mnist_image_data.json', JSON.stringify(data));
});

function readLabelData() {
  const fileBuf = fs.readFileSync(MNIST_LABEL_PATH);
  const labelBuf = new Uint8Array(fileBuf);
  // console.log(labelBuf)
  const labelOnehotArr = [...labelBuf];
  console.log(labelOnehotArr)
  const labelArr = new Array(NUM_IMAGES).fill(0);
  return labelArr.map((_, index) => {
    return labelOnehotArr.slice(index * NUM_CLASSES, (index + 1) * NUM_CLASSES);
  })
}

// fs.writeFileSync('label.json', JSON.stringify(readLabelData()))



