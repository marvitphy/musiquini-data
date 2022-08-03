const brain = require('brainjs');
const frases = require('./frases.js')
const NeuralNetwork = require('neural-network-node')
const network = new brain.NeuralNetwork({
    hiddenLayers: [2, 2]
});
var limdu = require('limdu');

function encodar(arg) {
    return arg.split('').map(x => (x.charCodeAt(0) / 256))
}

const dadosParatreinamento = frases.dados();

function processarDados(dados) {
    return dados.map(d => {
        return {
            input: getCharCodes(d.input),
            output: d.output
        }
    })
}

function getCharCodes(s) {
    let charCodeArr = [];

    for (let i = 0; i < s.length; i++) {
        let code = s.charCodeAt(i) / 256;
        charCodeArr.push(code);
    }

    return charCodeArr;
}

var humorClassify = new limdu.classifiers.NeuralNetwork();

// Módulo de Humor

humorClassify.trainBatch([
    { input: { 'triste': 0.3, 'alegre': 0, 'estressado': 0, 'entediado': 0.5, 'Raiva': 0.3 }, output: 0.22 }, // ave   
    { input: { 'triste': 1, 'alegre': 0, 'estressado': 0.5, 'entediado': 0.3, 'Raiva': 0.5 }, output: 0.46 }, // ave   
    { input: { 'triste': 0.5, 'alegre': 0.3, 'estressado': 0.3, 'entediado': 0.5, 'Raiva': 0.3 }, output: 0.38 }, // ave   
    { input: { 'triste': 0.5, 'alegre': 0.5, 'estressado': 0, 'entediado': 0, 'Raiva': 0 }, output: 0.2 }, // ave   
]);

// 0
// 0.3 pouco triste
// 0.5 mais ou menos
// 1 mto triste

let res = humorClassify.classify({ 'triste': 0.2, 'alegre': 0, 'estressado': 0, 'entediado': 0.5, 'Raiva': 0.3 })