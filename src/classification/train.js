import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { fileSystem } from '@tensorflow/tfjs-node/dist/io/file_system.js';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as fs from 'fs';
import * as path from 'path';

export class BerkeliumClassificationTrain {

    __OUTDIR = '';

    INTENT_DATA = [];
    INTENT_PATTERNS = [];
    INTENT_CLASSES = [];
    INTENT_RESPONSES = [];
    TRAINING_DATA = [];

    TRAIN_EPOCHS = 150;

    constructor() {
        console.log('\n\x1b[0m\x1b[37m\x1b[44m BerkeliumLabs \x1b[0m', '\x1b[0m Welcome!\n');
    }

    initializeData(datasetPath, outputFolder) {
        try {
            const rawData = fs.readFileSync(datasetPath);
            this.INTENT_DATA = JSON.parse(rawData);
            this.__OUTDIR = outputFolder;
            console.log('Data Loaded Successfully');

            this.getClasses(this.INTENT_DATA);
        } catch (error) {
            console.log('Data read error: ', error);
        }
    }

    encodeData(data) {
        const sentences = data.map(intents => intents.toLowerCase());
        const trainingData = use.load()
            .then(model => {
                return model.embed(sentences)
                    .then(embeddings => {
                        console.log(`${data.length} inputs encoded.`);
                        clearInterval(this.loadingAnimation);
                        return embeddings;
                    });
            })
            .catch(err => console.error('Fit Error:', err));
        console.log(`Data Encoding Started: ${data.length} inputs`);
        this.loadingAnimation;

        return trainingData
    }

    getClasses(intentData) {
        intentData.map(classElement => {
            if (!this.INTENT_CLASSES.includes(classElement.tag)) {
                this.INTENT_CLASSES.push(classElement.tag);
                const responses = {
                    [classElement.tag]: classElement.responses
                };
                this.INTENT_RESPONSES.push(responses);
            }
            //console.log(classElement);
        });
        //console.log(INTENT_RESPONSES.filter(p => p['thanks']));

        this.getTrainingData(intentData);
    }

    getTrainingData(intentData) {
        intentData.map(intents => {
            const className = intents.tag;
            let classVector = tf.zeros([1, this.INTENT_CLASSES.length]).dataSync();
            intents.patterns.map(patternData => {
                this.INTENT_PATTERNS.push(patternData);
                const classIndex = this.INTENT_CLASSES.indexOf(className);
                classVector[classIndex] = 1;
                this.TRAINING_DATA.push(classVector);
                //console.log(patternData, TRAINING_DATA);
            });
        });

        this.runTraining();
    }

    runTraining() {

        const model = tf.sequential();

        // Add layers to the model
        model.add(tf.layers.dense({
            inputShape: [512],
            activation: 'relu',
            units: 128,
        }));

        model.add(tf.layers.dropout({ rate: 0.5 }));

        model.add(tf.layers.dense({
            activation: 'relu',
            units: 64,
        }));

        model.add(tf.layers.dropout({ rate: 0.5 }));

        model.add(tf.layers.dense({
            activation: 'softmax',
            units: this.INTENT_CLASSES.length,
        }));

        model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.sgd(0.1), metrics: ['acc'] });
        Promise.all([
            this.encodeData(this.INTENT_PATTERNS)
        ]).then((encodedData) => {
            const {
                0: trainingData
            } = encodedData;

            //console.log(trainingData, this.TRAINING_DATA);

            model.fit(trainingData, tf.tensor(this.TRAINING_DATA), {
                epochs: this.TRAIN_EPOCHS,
                verbose: 0,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        console.log('\x1b[0m\x1b[34mEpoch: ' + (epoch + 1) + '\x1b[35m | Loss: ' + logs.loss.toFixed(5) +
                            '\x1b[33m | Accuracy: ' + logs.acc.toFixed(5));
                    }
                }
            }).then(info => {

                const infoIndex = info.epoch.length - 1;
                const finalLoss = info.history.loss[infoIndex].toFixed(5);
                const finalAcc = info.history.acc[infoIndex].toFixed(5);
                console.log('\x1b[0m\x1b[37m\x1b[44m BuddhiNLP \x1b[0m', '\x1b[32m Training Completed at \x1b[0m' +
                    '==>\x1b[0m\x1b[35m Loss: ' + finalLoss + '\x1b[33m | Accuracy: ' + finalAcc + '\x1b[0m');

                this.saveModelData(model);
            });
        }).catch(err => console.log('Prom Err:', err));

    }

    async saveModelData(model) {
        const timeStamp = Date.now();
        const modelOutFolder = path.resolve(this.__OUTDIR, timeStamp + '/');

        try {
            fs.mkdirSync(modelOutFolder, { recursive: true }, err => {
                if (err !== null) {
                    console.log('\x1b[31mMaking Directory Error: ' + err + '\x1b[0m');
                }
            });
            await model.save(fileSystem(modelOutFolder));

            const metaOutPath = path.resolve(modelOutFolder, 'model_metadata.json');
            const metadataStr = JSON.stringify({ 'classes': this.INTENT_CLASSES, 'responses': this.INTENT_RESPONSES });
            fs.writeFileSync(metaOutPath, metadataStr, { encoding: 'utf8' });
        } catch (error) {
            console.log(error);
        }

        return;
    }

    loadingAnimation = (function () {
        const h = ['|', '/', '-', '\\'];
        let i = 0;

        return setInterval(() => {
            i = (i > 3) ? 0 : i;
            console.clear();
            console.log(h[i], 'Encording Data...');
            i++;
        }, 200);
    })();
}