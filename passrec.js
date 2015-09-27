$(function() {
  PW = {
    lastTimeStamp: 0,
    x: [], //a temporary input array (e.g., one of x1, x2, x3 ...)
    X: [], //the matrix of all input arrays [x1; x2; x3 ...]
    Xmod: [], //matrix of inputs, together with spoofed false inputs
    y: [], //vector of identities
    Thetas: [], //thetas with which to predict
    pass: "" //the password to be saved / tested
  };

  PW.storeInput = function () {
    //only store the input if its password length -1, because:
    //the user could use the delete key or something else to log more intervals
    if (PW.pass.length - 1 === PW.x.length) {
      PW.X.push(PW.x);
    }
    PW.x = [];
  };

  PW.decrementCounter = function () {
    var $trainCount = $('#train-count');
    $trainCount.html((parseInt($trainCount.html())-1));
  };

  PW.trainAndCheck = function () {
    if (PW.Thetas.length === 0) {
      PW.prepareData();
      PW.Thetas= NN.trainNN(PW.Xmod, PW.y, 5);
    }
    console.log(PW.Thetas);
    console.log(PW.x);
    return NN.predict(PW.Thetas[0],PW.Thetas[1],[PW.x]);
  };

  PW.prepareData = function () {

    //falsifies some data
    PW.Xmod = numeric.clone(PW.X);
    PW.Xmod.map(function (inputs) {
      //random increment / decrement
      for (var i = 0; i < 3; i++) {
        inputs = PW.randomInc(inputs);
      }
      //mini-scramble
      for (i = 0; i < 5; i++) {
        inputs = PW.switchEl(inputs);
      }
      return inputs;
    });

    //concats true inputs and false inputs, and creates appropriate classifiers
    PW.Xmod = PW.X.concat(PW.Xmod);
    var y1 = new Array(PW.X.length);
    for (var i = 0; i < y1.length; i++) {
      y1[i] = [1];
    }
    var y2 = NN.nestedApply(function () {return 0;},y1);
    PW.y = y1.concat(y2);
  };

  PW.randomInc = function (arr) {
    arr[Math.floor(Math.random() * arr.length)] += Math.random()*100 - 50;
    return arr;
  };

  PW.switchEl = function (arr) {
    var i = Math.floor(Math.random() * arr.length);
    var j = Math.floor(Math.random() * arr.length);
    var temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
    return arr;
  };

  //reset if user presses backspace in the input
  $('#pw-input').bind('keydown', function (e) {
    if (e.which === 8 || e.keyCode === 8) {
      $('#pw-input').val("");
      PW.lastTimeStamp = 0;
    }
  });

  //record time intervals between inputs
  $('#pw-input').bind('input', function (e) {
    var diff;
    var char = e.target.value.slice(-1);
    if (PW.lastTimeStamp !== 0) {
      diff = e.timeStamp - PW.lastTimeStamp;
      PW.x.push(diff);
    }
    PW.lastTimeStamp = e.timeStamp;
  });

  //submit the form
  $('#pw-form').bind('submit', function (e) {
    e.preventDefault();
    var $pwInput = $('#pw-input');
    var $debug = $('#debug');
    var res;

    //first entry sets the password and stores the input
    if (!PW.pass) {
      PW.pass = $pwInput.val();
      PW.decrementCounter();
      PW.storeInput();
    //if the password is correct, store the result or run the check
    } else if (PW.pass === $pwInput.val()) {
      if (parseInt($('#train-count').html())) {
        PW.decrementCounter();
        PW.storeInput();
      } else {
        res = PW.trainAndCheck();
        console.log(res);
        //if its the right person
        if (res > 0.5) {
          $debug.html("Welcome back!");
          PW.storeInput();
        } else {
          $debug.html("Right password, but wrong person!");
          PW.x = []; //clear current input without storing it
        }
      }
    //if password wrong just let the user know
    } else {
      $debug.html("Wrong password!");
    }
    $pwInput.val("");
    PW.lastTimeStamp = 0;
  });
});
