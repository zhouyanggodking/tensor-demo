// see https://sharp.dimens.io/en/stable/api-constructor/

const sharp = require('sharp'); // allow high level image manipulation
const fs = require('fs');

async function createPng(filename) {
  const imgBuffer = await sharp({
    create: {
      width: 100,
      height: 100,
      channels: 4,
      background: {
        r: 255,
        g: 0,
        b: 0,
        alpha: 1
      }
      // background: '#ff0000'
    }
  }).png().toBuffer();  // convert to array buffer
  fs.writeFileSync(filename, imgBuffer);  // write to disk
}

async function readImageMetadata(filename) {
  const jpg = sharp(filename)
  const meta = await jpg.metadata();
  console.log(meta);
  return meta;
}

async function readImageStats(filename) {
  const image = sharp(filename)
  const stats = await image.stats();
  console.log(stats);
  return stats;
}

const filename = 'test222.png';
// createPng(filename).then(() =>{
//   readImageMetadata(filename);
//   readImageStats(filename);
// });

const mnisturl = 'mnist_images.png';
async function readMnistImages() {
  const img = sharp(mnisturl);
  const metadata = await img.metadata();
  console.log(metadata);
}

// readMnistImages()

// sharp(mnisturl).toBuffer().then(buff =>{
//   console.log(buff.length)
//   const buffer = buff.slice(0, 784);
//   console.log(buffer)
//   console.log(buffer.length)
//   sharp(buffer).png().toBuffer(b => { // not working
//     console.log(b)
//     fs.writeFileSync('king.png', b);
//   })
// })

sharp(mnisturl).extract({
  left: 0,
  top: 0,
  width: 784,
  height: 1
}).png().toBuffer().then(b => {
  fs.writeFileSync('king.png', b);
  console.log(b.length)
  // const arr = []
  // let temp = []
  // for(let i = 0; i < (b.length); i = i + 3) {
  //   if (i % 28 !== 27) {
  //     temp.push(b[i]);
  //   } else {
  //     arr.push(temp);
  //     temp = []
  //   }
  // }
})


