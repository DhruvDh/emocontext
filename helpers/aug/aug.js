let nlp = require('compromise');
let emoji = require('node-emoji');
const fs = require('fs')
const csv = require('csvtojson')

// console.log(nlp('arey u r not responding now  👿  👿').match("#emoji").out('array'))
console.log(emoji.replace("i blow ...  minds !   💥  😂  😂", (emoji) => `${emoji.key}`))

// console.log(emoji.find("collision"))

// let corpus = fs.readFileSync('../../_files/corpus.txt', 'utf8').split('\n')
// corpus = corpus.slice(0, 50).map(x => nlp(x.trim()).contractions().expand().all().out('text'))

// console.log(corpus.slice(0, 50))
let auged = []

var file = fs.createWriteStream('../../_files/auged_corpus--.txt');
file.on('error', err => {
    console.log('i fucked up man', err)
})

csv({
    noheader: true,
    output: 'line',
    fork: true
}).fromFile('../../_files/corpus--.txt', {defaultEncoding: 'utf8'}).subscribe(doc => {
    return new Promise((resolve, reject) => {
        let _new = []

        _new.push(nlp(doc).contractions().expand().all().out('text'))
        _new.push(nlp(doc).contractions().contract().all().out('text'))

        _new.push(nlp(doc).dates().toShortForm().all().out('text'))
        _new.push(nlp(doc).dates().toLongForm().all().out('text'))

        // _new.push(nlp(doc).nouns().toPlural().all().out('text')) // bad
        // _new.push(nlp(doc).nouns().toSingular().all().out('text')) // bad

        _new.push(nlp(doc).values().toNumber().all().out('text'))
        _new.push(nlp(doc).values().toText().all().out('text'))
        _new.push(nlp(doc).values().toCardinal().all().out('text'))
        _new.push(nlp(doc).values().toOrdinal().all().out('text'))

        _new.push(nlp(doc).sentences().toPastTense().out('text'))
        _new.push(nlp(doc).sentences().toPresentTense().out('text'))
        _new.push(nlp(doc).sentences().toFutureTense().out('text'))
        _new.push(nlp(doc).sentences().toContinuous().out('text'))

        _new.push(emoji.replace(doc, (emoji) => ` ${emoji.key} `))
        _new = [...new Set(_new)]

        auged.push(_new)
        _new.forEach(line => {
            file.write(line + '\n')
        })
        resolve("done")
    })
}, err => {
    console.log(err)
}, () => {
    console.log(auged.slice(0, 50))
})

file.on('finish', () => {
    console.log('finished writing')
    file.end()
})


// corpus.forEach(line => {
//     let expanded = nlp(line).expand().out('text')
// })