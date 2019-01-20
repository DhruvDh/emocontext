let nlp = require('compromise');
let emoji = require('node-emoji');
const fs = require('fs')

console.log(nlp('arey u r not responding now  👿  👿').match("#emoji").out('array'))
console.log(emoji.replace("i blow ...  minds !   💥  😂  😂", (emoji) => `${emoji.key}`))

console.log(emoji.find("collision"))

let corpus = fs.readFileSync('../../_files/corpus.txt', 'utf8').split('\n')
corpus = corpus.slice(0, 50).map(x => nlp(x.trim()).contractions().expand().all().out('text'))

console.log(corpus.slice(0, 50))
let auged = []

// corpus.forEach(line => {
//     let expanded = nlp(line).expand().out('text')
// })