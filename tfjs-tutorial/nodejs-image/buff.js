const buf = Buffer.alloc(10)

console.log(buf)

const buf2 = Buffer.from('godking')
console.log(buf2.length)
console.log(buf2.byteLength)

console.log(buf2.toJSON())

buf2.write('oceansky')

console.log(buf2)