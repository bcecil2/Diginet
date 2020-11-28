import {MnistData} from './data.js';

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];


function toIm(sketchpad, predict) {
    const dataURL = sketchpad.toImage();
    var myImage = new Image(28, 28);
    var p;
    myImage.onload = () => {
	p = tf.browser.fromPixels(myImage,1);
	predict(p);
    }
    myImage.src = dataURL;
}

const canvas = document.getElementById('sketchpad');
const sketchpad = new Atrament(canvas, {
			width:300,
			height:300,
			color: 'white'
		    });
sketchpad.weight = 20;
var c = document.getElementById("sketchpad");
var ctx = c.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, c.width, c.height);

window.sketchpad = sketchpad;

var gModel; 


function standIm(im) {
    var proc = im.cast('float32');
    proc = proc.div(255.0);

    // delete rows from top
    while (tf.sum(proc.slice(0,1)).dataSync() == 0) {
	proc = proc.slice(1);
    }
    
    // delete rows from bottom
    var last = proc.shape[0]-1;
    while (tf.sum(proc.slice(last)).dataSync() == 0) {
	proc = proc.slice(0,last-1);
	last = proc.shape[0]-1;
    }
    
    // delete right cols
    var rows = proc.shape[0];
    var cols = proc.shape[1]-1;
    
    while(tf.sum(proc.slice([0,cols],[rows,1])).dataSync() == 0) {
	proc = proc.slice([0,0],[rows,cols]);
	cols = proc.shape[1]-1;
    }

    rows = proc.shape[0];
    cols = proc.shape[1]-1;
   
    // delete left cols
    while(tf.sum(proc.slice([0,0],[rows,1])).dataSync() == 0) {
	proc = proc.slice([0,1],[rows,cols]);
	cols = proc.shape[1]-1;
    }
    
    rows = proc.shape[0];
    cols = proc.shape[1];
    if (rows > cols) {
	var f = 20.0/rows;
	rows = 20;
	cols = Math.round(cols*f);
	proc = tf.image.resizeBilinear(proc,[rows,cols]);

    } else {
	var f = 20.0/cols;
	cols = 20;
	rows = Math.round(rows*f);
	proc = tf.image.resizeBilinear(proc,[rows,cols]);
    }
    
    // pad back to 28 by 28
    var colsBefore = Math.ceil((28-cols)/2.0);
    var colsAfter = Math.floor((28-cols)/2.0);
    var rowsBefore = Math.ceil((28-rows)/2.0);
    var rowsAfter = Math.floor((28-rows)/2.0);
    proc = tf.pad(proc,[[rowsBefore,rowsAfter],[colsBefore,colsAfter],[0,0]]);
  
    // calculate center of mass
    proc = proc.reshape([28,28]);
    var xs = tf.range(0,28).reshape([28,1]);
    var ys = xs.reshape([1,28]);
    var s = tf.sum(proc);

    var cx = tf.sum(proc.mul(xs)).div(s);
    var cy = tf.sum(proc.mul(ys)).div(s);

    cx = cx.dataSync();
    cy = cy.dataSync();
    var rows = proc.shape[0];
    var cols = proc.shape[1];

    cx = Math.round(cols/2.0-cx[0]);
    cy = Math.round(rows/2.0-cy[0]);
    
    // shift image by center of mass 
    //tf.browser.toPixels(proc,document.getElementById("sketchpad"));
    var src = cv.matFromArray(28,28,cv.CV_32F,proc.dataSync());

    var dst = new cv.Mat();
    var dsize = new cv.Size(28,28);
    
    var M = cv.matFromArray(2,3,cv.CV_64FC1, [1,0,cx,0,1,cy]);

    cv.warpAffine(src,dst,M,dsize);
    
    var merged = new cv.Mat();
    var rgba = new cv.MatVector();
    

    var arr = dst.data32F;
    proc = tf.tensor(arr).reshape([1,28,28,1]);

    //tf.browser.toPixels(proc,document.getElementById("sketchpad"));
    console.log('replaced');
    return proc;
}

async function pred(pixels) {
  
  pixels = standIm(pixels);
  var k = 10;

  const pred = gModel.predict(pixels);
  const top5 = pred.topk(k);
    
  top5.values.print();
  top5.indices.print();
  // have to unwrap from tensors to get data
  var vals = await top5.values.array();
  var idxs = await top5.indices.array();
  var barChart = new Array();
  for (var i = 0; i < k; i++) {
    barChart.push({index:classNames[idxs[0][i]], value:vals[0][i]});
  }

  const surf = {name: 'Top 5 predictions', tab: 'Predictions'};
  tfvis.render.barchart(surf,barChart);

}

window.predictUser = function predictUser(){
    toIm(sketchpad,pred);
}

async function run() {  
  const data = new MnistData();
  await data.load();
  

  
  var notLoaded = true; 
  if (localStorage.getItem("tensorflowjs_models/digitModel/info") === null) {
     var model = getModel();
     console.log("Created new model");
  } else {
    console.log("loading old model");
    var model = await tf.loadLayersModel('localstorage://digitModel');
    const optimizer = tf.train.adam();
    model.compile({
	optimizer: optimizer,
	loss: 'categoricalCrossentropy',
	metrics: ['accuracy'],
    });
    notLoaded = false;
  } 
  model.summary();  
  tfvis.show.modelSummary({name: "Model Architecture", tab: "Model Specs"}, model);
  if (notLoaded) { 
    await train(model,data);
  }
  doPrediction(model,data);
  const saveResult = await model.save('localstorage://digitModel');
  
  const sr = await tf.loadLayersModel('C:\/Users\/Clake Becil\/Desktop\/digiNet\/mnist1.json');
  console.log("saved");
  gModel = model;
}



function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;  
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}


async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 55000;
  const TEST_DATA_SIZE = 10000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  var arr = testxs.arraySync();
  //console.log(arr[0]);
  var t = tf.tensor(arr[0]);
  //console.log(t.dtype);
  // tf.browser.toPixels(t,document.getElementById("sketchpad"));
  return;
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


document.addEventListener('DOMContentLoaded', run);
