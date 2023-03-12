import React, {Component} from 'react';
import shadesData from '../data/shades.json';
import { Line } from 'rc-progress';
import * as tf from '@tensorflow/tfjs';
import Upload from './Upload';
import TgPredict from './TgPredict';
import "../scss/train.scss";

let model;
let foundationLabels;

export default class Train extends Component {
  state = ({
    loading: false,
    currentEpoch: 0,
    lossResult: 0.000,
    epochs: 25,
    units: 20,
    batchSize: 32,
    learningRate: 0.25,
    foundation: 'None',
    percent: 0,
    rgb: []
  } );
  increase() {
    const { percent } = this.state;
    const newPercent = percent + 1;
    if (newPercent >= 100) {
      clearTimeout(this.tm);
      return;
    }
    this.setState({ percent: newPercent });
    this.tm = setTimeout(this.increase, 10);
  }
  
  //Converts hexadecimal values to RGB color values
  hexToRgb = (hex) => {
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    const hexadecimal = hex.replace(shorthandRegex, (m, r, g, b) => {
      return r + r + g + g + b + b;
    });
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hexadecimal);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };
  
  trainModel = async () => {
    this.setState({
      loading: true,
      foundation: 'None'
    });
    // Get current values for epochs, units, batch size, and learning rate from state
    const {epochs, units, batchSize, learningRate} = this.state;
    
    // Create list of foundation brand, product, and shade name from the imported shadesData and remove any duplicates if there are any.
    // There should be a total of 584 unique foundation shades from shades.json!
    foundationLabels = shadesData
      .map(shade => `${shade.brand} ${shade.product} - ${shade.shade}`)
      .reduce((accumulator, currentShade) => {
        if (accumulator.indexOf(currentShade) === -1) {
          accumulator.push(currentShade);
        }
        this.state.percent = this.state.percent + 4;
        return accumulator
      }, []);
    
    // Create list of hexadecimal values of the foundation shades from the imported shadesData
    const hexList = shadesData.map(shade => shade.hex);
    
    // Convert hexadecimal values to RGB color values and store in new list
    const rgbList = hexList.map(hex => this.hexToRgb(hex));
    
    // Create empty array to hold the normalized shade RGB color values
    let shadeColors = [];
    
    // Create empty array to hold the foundation index values
    let foundations = [];
    
    // Loop through each foundation shade in shadesData. 
    // Push its corresponding index value found in foundationLabels into the foundations array
    for (const shade of shadesData) {
      foundations.push(foundationLabels.indexOf(`${shade.brand} ${shade.product} - ${shade.shade}`));
    }
    
    // Loop through each RGB color in rgbList. 
    // Normalize the RGB color dividing by 255 for each RGB value, store in an array, and then push that array into the shades array
    for (const rgbColor of rgbList) {
      let shadeColor = [rgbColor.r / 255, rgbColor.g / 255, rgbColor.b / 255];
      shadeColors.push(shadeColor);
    }
  
    // Create a 2D tensor out of the shadeColors array
    // This tensor will act as the inputs to train the model with
    const inputs = tf.tensor2d(shadeColors);

    // Create a 1D tensor out of the foundations array
    // Apply tf.oneHot to this tensor to create a tensor of 1 & 0 values out of the 584 possible foundation shades.
    const outputs = tf.oneHot(tf.tensor1d(foundations, 'int32'), 584).cast('float32');
   
    // Create a sequential model since the layers inside will go in order
    model = tf.sequential();
    
    // Create a hidden dense layer since all inputs will be connected to all nodes in the hidden layer.
    // units: How many nodes in the layer
    // inputShape: How many input values (3 because there are 3 RGB values for each shade color)
    // activation: Sigmoid function squashes the resulting values to be between a range of 0 to 1, which is best for a probability distribution.
    // Activation functions take the weighted sum of inputs plus a bias as input and perform the necessary computation to decide which nodes to fire in a layer.
    const hiddenLayer = tf.layers.dense({
      units: parseInt(units),
      inputShape: [3],
      activation: 'sigmoid'
    });
    
    // Create a dense output layer since all nodes from the hidden layer will be connected to the outputs
    // units: Needs to be 584 since there are a total of 584 unique foundation shades
    // inputShape does not need to be defined for output.
    // activation: Softmax function acts like sigmoid except it also makes sure the resulting values add up to 1
    const outputLayer = tf.layers.dense({
      units: 584,
      activation: 'softmax'
    });
    
    // Add layers to the model
    model.add(hiddenLayer);
    model.add(outputLayer);
    
    // Create optimizer with stocastic gradient descent to minimize the loss with learning rate of 0.25
    const optimizer = tf.train.sgd(parseFloat(learningRate));
  
    // Compile the model with the optimizer created above to reduce the loss.
    // Use loss function of categoricalCrossentropy, which is best for comparing two probability distributions
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  
    // Create options object, which will be passed into the model when fitting the data
    // epochs: Number of iterations
    // shuffle: Shuffles data at each epoch so it's not in the same order
    // validationSplit: Saves some of the training data to be used as validation data (0.1 = 10%)
    const options = {
      epochs: parseInt(epochs),
      batchSize: parseInt(batchSize),
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          this.setState({
            currentEpoch: epoch + 1,
            lossResult: logs.loss.toFixed(3)
          })
        },
        onBatchEnd: tf.nextFrame,
        onTrainEnd: () => {
          this.setState({
            loading: false
          });
        },
      }
    };
    
    // Fit the data to the model, then console log the results
    return await model.fit(inputs, outputs, options)
    .then(results => console.log('results', results));
  };
  
  // Reset state to default values
  resetModel = () => {
    this.setState({
      loading: false,
      currentEpoch: 0,
      lossResult: 0.000,
      epochs: 25,
      batchSize: 32,
      units: 20,
      learningRate: 0.25,
      foundation: 'None'
    });
  };
  
  // Update state with values the user has entered. 
  // If the user tries to use a non-number input or an empty input, set state back to the default values.
  updateValue = evt => {
      const defaults = {
        epochs: 10,
        batchSize: 32,
        units: 20,
        learningRate: 0.25
      };
      if (evt.target.value !== '' || parseFloat(evt.target.value) > 0){
        this.setState({
          [evt.target.name] : evt.target.value
        });
      } else {
        this.setState({
          [evt.target.name] : defaults[evt.target.name]
        });
      }
  };
  
  // If user clicks out of the input, set its value to the current value stored in state
  getValue = evt => {
    evt.target.value = this.state[evt.target.name];
  };
  
  // Clear input when user begins to type
  clearValue = evt => {
    evt.target.value = '';
  };
  
  // Set state of RGB values at the top level so it can be passed down to Upload and Predict child components
  setRGB = rgb => {
    this.setState({
      rgb
    });
  };
  
  // Set state of the predicted foundation at the top level so it can be passed down to the Predict child component
  setFoundation = foundation => {
    this.setState({
      foundation
    });
  }
  
  // Render training model results
  render() {
    const {loading, currentEpoch, lossResult, epochs, units, batchSize, learningRate, foundation,percent, rgb} = this.state;
    return (
      <div class="col-md-12 col-lg-12 col-xl-12">
    <div align="left">
  	  <div className="divider"></div>
      <div className="sidebar">
        <h2>Train Settings</h2>
        <div className="train-result">
          <span>Epoch:</span>
          <span>{currentEpoch}</span>
        </div>
        <div className="train-result">
          <span>Loss:</span>
          <span>{lossResult}</span>
        </div>
      </div>
      <div className="train-inputs">
        <div className="input-group">
          <div className="input-container">
            <label className="label" htmlFor="epochs">Epochs</label>
            <input type="text" name="epochs" className="input" id="epochs" value={epochs} onFocus={evt => this.clearValue(evt)} onBlur={evt => this.getValue(evt)} onChange={evt => this.updateValue(evt)}></input>
            <label className="label" htmlFor="batch-size">Batch Size</label>
            <input type="text" name="batchSize" className="input" id="batch-size" value={batchSize} onFocus={evt => this.clearValue(evt)} onBlur={evt => this.getValue(evt)} onChange={evt => this.updateValue(evt)}></input>
 
          </div>
          <div className="input-container">
            <label className="label" htmlFor="units">Units</label>
            <input type="text" name="units" className="input" id="units" value={units} onFocus={evt => this.clearValue(evt)} onBlur={evt => this.getValue(evt)} onChange={evt => this.updateValue(evt)}></input>
          </div>
        </div>
        <div className="input-group">
          <div className="input-container">
            <label className="label" htmlFor="batch-size">Batch Size</label>
            <input type="text" name="batchSize" className="input" id="batch-size" value={batchSize} onFocus={evt => this.clearValue(evt)} onBlur={evt => this.getValue(evt)} onChange={evt => this.updateValue(evt)}></input>
          </div>
          <div className="input-container">
            <label className="label" htmlFor="learning-rate">Learning Rate</label>
            <input type="text" name="learningRate" className="input" id="learning-rate" value={learningRate} onFocus={evt => this.clearValue(evt)} onBlur={evt => this.getValue(evt)} onChange={evt => this.updateValue(evt)}></input>
          </div>
        </div>
      </div>
      </div><div class="text-center">
                                            <h4 class="text-dark mb-4">Reputation Expert</h4>
                                        </div>
     
      <div className="train-model">
        <div className={`train-model-button ${loading ? 'disabled' : ''}`} onClick={() => this.trainModel()}>
        {/* <Line percent={currentEpoch*4} strokeWidth={4} strokeColor="#D3D3D3" /><br/> */}
          {loading ?   
            <div className="loader">
              <div className="inner one"></div>
              {/* <div className="inner two"></div>
              <div className="inner three"></div> */}Running..
              
            </div>
            
          : 'Train Model'
          }
        </div>
        <div className={`train-model-button ${loading ? 'disabled' : ''}`} onClick={() => this.resetModel()}>
        <Line percent={70} strokeWidth={4} strokeColor="#D3D3D3" />
        {loading ?
            <div className="loader">
              {loading && <Line percent={70} strokeWidth={4} strokeColor="#D3D3D3" />}
           
               <div className="inner one"></div>
              {/*<div className="inner two"></div>
              <div className="inner three"></div> */}
            </div>
          : 'Reset'
          }
        </div>
      </div>
      <div className="divider"></div>
      <Upload loading={loading} rgb={rgb} setRGB={this.setRGB} />
      <div className="divider"></div>
      <TgPredict loading={loading} rgb={rgb} model={model} foundation={foundation} foundationLabels={foundationLabels} setFoundation={this.setFoundation} />
    </div>
    );
  }
}

