import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { BerkeliumClassificationTrain } from './classification/train.js';
import { BerkeliumClassificationPredict } from './classification/predict.js';
import * as fs from 'fs';

const bkClassTrain = new BerkeliumClassificationTrain();
const bkClassPredict = new BerkeliumClassificationPredict();

let TF_MODEL;
let RESPONSE_DATA;

export const bkLabs = {
    nlu: {
        /**
         * 
         * @param {string} trainData : Path to intent.json file
         * @param {string} outputFolder : Relative path to output folder
         */
        train: (trainData, outputFolder) => {
            bkClassTrain.initializeData(trainData, outputFolder);
        },
        /**
         * 
         * @param {string} modelUrl : Absolute path to model.json
         * @param {string} metadataUrl : Relative path to model_metadata.json
         * @param {object} callBackFunction : Callback Function to execute when the model loaded
         */
        loadModel: async (modelUrl, metadataUrl, callBackFunction) => {
            if (modelUrl == "") {
                throw "Invalid Model Url.";
            }
            if (metadataUrl == "") {
                throw "Invalid Data Url.";
            }
            if (!callBackFunction) {
                throw "Invalid Call back function.";
            }
            TF_MODEL = await tf.loadLayersModel(modelUrl);
            const metadataFile = fs.readFileSync(metadataUrl);
            RESPONSE_DATA = JSON.parse(metadataFile);
            
            console.log(JSON.parse(metadataFile), RESPONSE_DATA);
            callBackFunction();
            
            return [TF_MODEL, RESPONSE_DATA];
        },
        /**
         * 
         * @param {string} sentence : Sentence to encode
         * @param {object} callBackFunction : Callback Function to execute when encoding is completed
         * @returns 
         */
        encodeText: async (sentence, callBackFunction) => {
            const encodedText = await bkClassPredict.encodeData(sentence);
            callBackFunction();
            return encodedText;
        },
        /**
         * 
         * @param {Tensor} predictTensor : Output Tensor from Prediction
         * @param {Array} responseData : Response Data
         * @returns 
         */
        predictReply: (predictTensor, responseData) => {
            const replyText = bkClassPredict.predict(predictTensor, responseData);
            return replyText;
        }
    }
};
