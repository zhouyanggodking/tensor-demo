const fs = require('fs');
const axios = require('axios');

// const url = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const url ='https://cn.bing.com/sa/simg/hamburger_flyout_desktop-2x.png'

axios.get(url, {
  responseType: 'arraybuffer'   // important
}).then(res =>{
  const data = res.data;
  // console.log(data)
  fs.writeFileSync('test.png', data)
})

axios.get(url, {
  responseType: 'stream'         // important
}).then(res =>{
  const data = res.data;
  // console.log(data)
  const writer = fs.createWriteStream('good.png')
  data.pipe(writer)
})