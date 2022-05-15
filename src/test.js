import { bkLabs } from './index.js';

const intentData = 'src/test/intents.json';
const modelUrl = 'file://D:/Berkelium Labs/project/nlp-core/src/models/testing/model.json';
const metadata = 'src/models/testing/model_metadata.json';
const outputFolder = 'src/models';
const sentence = 'Bye';
let encodedSentence;
let mymodel;

async function modelLoaded() {
    console.log('model loaded');
    encodedSentence = await bkLabs.nlu.encodeText(sentence, textEncoded);
    console.log('text encoded', encodedSentence);
    const predictData = await mymodel[0].predict(encodedSentence);
    console.log('predict data', predictData);
    const myReply = bkLabs.nlu.predictReply(predictData, mymodel[1]);
    console.log(myReply);
}

async function textEncoded() {
    console.log('text encoded');
}

async function runPredict() {
    mymodel = await bkLabs.nlu.loadModel(modelUrl, metadata, modelLoaded);
}

//runPredict();

bkLabs.nlu.train(intentData, outputFolder);