var nlp = require('compromise');
const spacy = require("spacy-nlp");
var spacyServer = spacy.server({ port: process.env.IOPORT });

spacyServer.then(() => {
    console.log("Spacy is ready...");
    const nlp = spacy.nlp;

    nlp.parse("dhruv is a great guy").then(output => {
        console.log(output);
        console.log(JSON.stringify(output[0].parse_tree, null, 2));
      }).catch(error => {
          console.log(error)
      });
}).catch(error => {
    console.log("Spacy couldn't start due to...");
    console.log(error)
})

console.log(nlp('dhruv is a great guy').terms().data())