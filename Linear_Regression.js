
//
// CV, tfjs, Linear Regression for eq. 3x + 1
// 
const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

async function doTraining(model)
{
  const history = 
    await model.fit(xs, ys, 
    { epochs: 500,
        callbacks:{
          onEpochEnd: async(epoch, logs) =>
          {
            console.log("Epoch:" + epoch + "Loss:" + logs.loss);
          }
       }
    });
}

const model = tf.sequential();                                 // Model is eq -->
model.add(tf.layers.dense({units: 1, inputShape: [1]}));       // Simple NN with one dense layer having a single neuron -->
model.compile({loss:'meanSquaredError', optimizer:'sgd'});     // Compile the model using loss function - Mean Squared error and an optimizer, stochastic gradient descent -->
model.summary();                                               // Output the model summary to the console window -->

// Data we will use to train the model for 3x + 1
const xs = tf.tensor2d([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], [10, 1]);     // 2D Tensor , 1st dim is data itself, then 1 dim array with 10 elements -->
const ys = tf.tensor2d([1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5], [10, 1]); // 2D Tensor , 1st dim is data itself, then 1 dim array with 10 elements -->

// Train model with async function doTraining 
doTraining(model).then(() => {
alert(model.predict(tf.tensor2d([5.0], [1,1])));   // Do prediction 
});
