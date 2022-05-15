import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { fileSystem } from '@tensorflow/tfjs-node/dist/io/file_system.js';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { bkSpinner } from '../utils/spinner.js'
export class BerkeliumClassificationTrain {

    __OUTDIR = '';

    INTENT_DATA = [];
    INTENT_PATTERNS = [];
    INTENT_CLASSES = [];
    INTENT_RESPONSES = [];
    TRAINING_DATA = [];

    TRAIN_EPOCHS = 150;

    SPINNER_ANIMATION;

    constructor() {

    }

    initializeData(datasetPath, outputFolder) {
        try {
            const rawData = fs.readFileSync(datasetPath);
            this.INTENT_DATA = JSON.parse(rawData);
            this.__OUTDIR = outputFolder;
            console.log(chalk.bgGreen.black(' info ') + chalk.greenBright(' Data Loaded Successfully\n'));

            this.getClasses(this.INTENT_DATA);
        } catch (error) {
            console.log(chalk.red(' error ') + chalk.redBright(' Data read error: '), error);
        }
    }

    encodeData(data) {
        const sentences = data.map(intents => intents.toLowerCase());
        const trainingData = use.load()
            .then(model => {
                return model.embed(sentences)
                    .then(embeddings => {
                        console.log(
                            chalk.bgGreen.black('\n info ') +
                            chalk.greenBright(` ${data.length} inputs encoded.`)
                        );

                        bkSpinner.stopSpinner(this.SPINNER_ANIMATION);

                        return embeddings;
                    });
            })
            .catch(err => console.error(chalk.redBright('Fit Error:'), err));
        console.log(
            chalk.bgGreen.black(' info ') +
            chalk.greenBright(` Data Encoding Started: ${data.length} inputs\n`)
        );

        this.SPINNER_ANIMATION = bkSpinner.spinnerAnimation()

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
                        console.log(
                            chalk.cyan(`Epoch: ${(epoch + 1)}`) +
                            chalk.yellowBright(` | Loss: ${logs.loss.toFixed(5)}`) +
                            chalk.green(` | Accuracy: ${logs.acc.toFixed(5)}`)
                        );
                    }
                }
            }).then(info => {

                const infoIndex = info.epoch.length - 1;
                const finalLoss = info.history.loss[infoIndex].toFixed(5);
                const finalAcc = info.history.acc[infoIndex].toFixed(5);

                console.log(
                    chalk.bgYellow.black(`\n BerkeliumLabs NLP Core `) +
                    chalk.greenBright(' Training Completed at ==>') +
                    chalk.yellowBright(` Loss: ${finalLoss}`) +
                    chalk.green(` | Accuracy: ${finalAcc}\n`)
                );
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
                    console.log(chalk.red(`Making Directory Error: ${err}`));
                }
            });
            await model.save(fileSystem(modelOutFolder));

            const metaOutPath = path.resolve(modelOutFolder, 'model_metadata.json');
            const metadataStr = JSON.stringify({ 'classes': this.INTENT_CLASSES, 'responses': this.INTENT_RESPONSES });
            fs.writeFileSync(metaOutPath, metadataStr, { encoding: 'utf8' });

            console.log(
                chalk.bgGreen.black(' info ') +
                chalk.green(` Model data saved to: `) +
                chalk.underline(`${modelOutFolder}\n`)
            );
        } catch (error) {
            console.log(chalk.red(`Oops! We couldn't save your model:\n${error}`));
        }

        return;
    }
}