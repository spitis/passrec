//unregularized neural network with a single hidden layer and sigmoid function

NN = {};

function assert(condition, message) {
    if (!condition) {
        throw message || "Assertion failed";
    }
}

//apply a function to something that may be  nested array, element-wise
NN.nestedApply = function (func, obj) {
  if (typeof(obj) === "number") {
    return func(obj);
  } else {
    return obj.map(function (i) {
      return NN.nestedApply(func, i);
    });
  }
};

//sigmoid of a value or array (element-wise)
NN.sigmoid = function (z) {
  return NN.nestedApply(function (z) {
    return 1 / (1 + Math.exp(-1*z));
  }, z);
};

assert(NN.sigmoid(3)-0.95257 < 0.0001);
assert(NN.sigmoid(3)-0.95257 > -0.0001);

//gradient sigmoid of value or array (element-wise)
NN.sigmoidGradient = function (z) {
  return NN.nestedApply(function (z) {
    return NN.sigmoid(z) * (1 - NN.sigmoid(z));
  }, z);
};

assert(NN.sigmoidGradient(3)-0.045177 < 0.0001);
assert(NN.sigmoidGradient(3)-0.045177 > -0.0001);

//log function of value or array (element-wise)
NN.log = function (z) {
  return NN.nestedApply(function (z) {
    return Math.log(z);
  }, z);
};

//creates zeros matrix
NN.zeros = function (r, c) {
  return numeric.rep([r,c],0);
};

//generates weight matrix with random small initial weights
NN.randInitWeights = function (L_in, L_out) {
  var W = NN.zeros(L_out, 1+L_in);
  var eps_init = 0.12;
  return NN.nestedApply(
    function() {
      return Math.random() * 2 * eps_init - eps_init;
    },
    W
  );
};

//adds a column of 1s to the start of a matrix
NN.addOnesColumn = function (X) {
  X = numeric.clone(X); //deep copy
  X.forEach(function (x) {
    x.unshift(1); //add 1 to front of each row
  });
  return X;
};
//remove first column from matrix
NN.sliceFirstColumn = function (X) {
  X = numeric.clone(X); //deep copy
  X.forEach(function (x) {
    x.shift(1);
  });
  return X;
};

assert(numeric.same([[1,2,3],[1,4,5]], NN.addOnesColumn([[2,3],[4,5]])));
assert(numeric.same(NN.sliceFirstColumn([[1,2,3],[1,4,5]]), [[2,3],[4,5]]));

//forward propogate through NN to predict outcomes
NN.predict = function (Theta1, Theta2, X) {
  var m = X.length;
  var a1 = NN.addOnesColumn(X);
  var z2 = numeric.dot(a1,numeric.transpose(Theta1));
  var a2 = NN.addOnesColumn(NN.sigmoid(z2));
  var z3 = numeric.dot(a2,numeric.transpose(Theta2));
  var a3 = NN.sigmoid(z3);

  return a3;
};

//calculate cost function; unregularized
NN.costFunction = function (Theta1, Theta2, X, y) {
  var h = NN.predict (Theta1, Theta2, X);
  var m = h.length;

  var J1 = numeric.mul(NN.log(h),y);
  var J2 = numeric.mul(
            numeric.sub(numeric.rep([m,1],1),y),
            NN.log(numeric.sub(numeric.rep([m,1],1),h))
          );
  J = -1 / m * numeric.sum(numeric.add(J1,J2));

  return J;
};

NN.tempT1 = [[0.030362693779170502, -0.07896072076633573, -0.09074505977332592],
[0.04161494823172687, -0.07769790200516581, -0.10804939890280366]];
NN.tempT2 = [[0.06045542864128947, 0.030736222509294753,-0.058759959563612935]];
assert(NN.costFunction (NN.tempT1, NN.tempT2, [[1,0],[2,3],[3,4]], [[0],[1],[1]])-0.68442 < 0.0001);
assert(NN.costFunction (NN.tempT1, NN.tempT2, [[1,0],[2,3],[3,4]], [[0],[1],[1]])-0.68442 > -0.0001);

//calculate gradients; unregularized
NN.gradients = function (Theta1, Theta2, X, y) {
  var D2 = 0;
  var D1 = 0;

  var a1 = NN.addOnesColumn(X);
  var z2 = numeric.dot(a1,numeric.transpose(Theta1));
  var a2 = NN.addOnesColumn(NN.sigmoid(z2));
  var z3 = numeric.dot(a2,numeric.transpose(Theta2));
  var a3 = NN.sigmoid(z3);

  d3 = numeric.sub(a3, y);
  d2 = numeric.mul(
    numeric.dot(d3, NN.sliceFirstColumn(Theta2)),
    NN.sigmoidGradient(z2)
  );

  D2 = numeric.dot(numeric.transpose(d3),a2);
  D1 = numeric.dot(numeric.transpose(d2),a1);
  D2 = NN.nestedApply(function(i) {return numeric.mul(i, 1/X.length);}, D2);
  D1 = NN.nestedApply(function(i) {return numeric.mul(i, 1/X.length);}, D1);

  return [D1, D2];
};

assert(NN.gradients(NN.tempT1, NN.tempT2, [[1,0],[2,3],[3,4]], [[0],[1],[1]])[1][0][1] + 0.040715 < 0.0001);
assert(NN.gradients(NN.tempT1, NN.tempT2, [[1,0],[2,3],[3,4]], [[0],[1],[1]])[1][0][1] + 0.040715 > -0.0001);
assert(NN.gradients(NN.tempT1, NN.tempT2, [[1,0],[2,3],[3,4]], [[0],[1],[1]])[0][1][1] - 0.0085528 < 0.0001);
assert(NN.gradients(NN.tempT1, NN.tempT2, [[1,0],[2,3],[3,4]], [[0],[1],[1]])[0][1][1] - 0.0085528 > -0.0001);

//very basic gradient descent implementation that returns [Theta1,Theta2];
NN.trainNN = function (X, y, hidden_layer_size, num_iter) {
  num_iter = num_iter || 10000;
  var Theta1 = NN.randInitWeights(X[0].length, hidden_layer_size);
  var Theta2 = NN.randInitWeights(hidden_layer_size,1);
  var alpha = 0.05;

  var gradients;
  for (var i = 0; i < num_iter; i++) {
    gradients = NN.gradients(Theta1, Theta2, X, y);
    Theta1 = numeric.sub(Theta1,numeric.mul(alpha, gradients[0]));
    Theta2 = numeric.sub(Theta2,numeric.mul(alpha, gradients[1]));
    if (i % 500 === 0) {
      console.log("COST AT iter # " + i);
      console.log(NN.costFunction(Theta1, Theta2, X, y)); // should decrease
    }
  }

  return [Theta1, Theta2];
};
