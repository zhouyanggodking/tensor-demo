const fs = require('fs');

// mnist_labels_uint8 is one hot array
// each label takes 10 element
// total 65000 elements so the total buffer size is 65000 * 10

const labelBuf = fs.readFileSync('mnist_labels_uint8');

console.log(labelBuf)

// const labels = new Uint8Array(labelBuf)
const labels = [...labelBuf]

console.log(labels)
console.log(labels.length)

console.log(labels.slice(0, 10))
// labels.forEach(item =>{
//   if (item !==0) {
//     console.log(item)
//   }
// })