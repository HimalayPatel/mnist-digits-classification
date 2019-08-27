from utils import softmax_cross_entropy, add_momentum, data_loader_mnist, predict_label, DataSplit
import sys
import os
import argparse
import numpy as np
import json

class linear_layer:
    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))
        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):
        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (i.e., X[i], where                   i = 1,...,N).
                The mini-batch size is N.

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """
        forward_output = np.dot(X, self.params['W']) + self.params['b']
        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'].

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss w.r.t. X[i].
        """
        x_tran = np.transpose(X)
        b_ones = np.ones(X.shape[0])
        self.gradient['W'] = np.dot(x_tran, grad)
        self.gradient['b'] = np.dot(b_ones, grad)
        w_trans = np.transpose(self.params['W'])
        backward_output = np.dot(grad, w_trans)
        return backward_output
    
class relu:
    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):
        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """
        x_zero = np.zeros(X.shape)
        self.mask = np.maximum(X, x_zero)
        forward_output = self.mask
        return forward_output
    
    def backward(self, X, grad):
        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """

        self.mask[self.mask <= 0] = 0
        self.mask[self.mask > 0] = 1

        backward_output = np.multiply(grad, self.mask)
        return backward_output

class tanh:

    def forward(self, X):
        """
            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """
        forward_output = np.tanh(X)
        return forward_output

    def backward(self, X, grad):
        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """
        x_ones = np.ones(X.shape)
        t_square = np.square(np.tanh(X))
        backward_output = np.multiply(grad, x_ones - t_square)
        return backward_output

class dropout:
    """
        It is built up with one arguments:
        - r: the dropout rate

        It has no parameters to learn.
        self.mask is an attribute of dropout. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):

        """
            Input:
            - X: A numpy array of arbitrary shape.
            - is_train: A boolean value. If False, no dropout is performed.

            Operation:
            - If p >= self.r, output that element multiplied by (1.0 / (1 - self.r)); otherwise, output 0 for that element

            Return:
            - forward_output: A numpy array of the same shape of X (the output of dropout)
        """
        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):

        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.


            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in X.
        """
        backward_output = np.multiply(grad, self.mask)
        return backward_output

def miniBatchGradientDescent(model, momentum, _lambda, _alpha, _learning_rate):
    '''
        Input:
            model: Dictionary containing all parameters of the model
            momentum: Check add_momentum() function in utils.py to understand this parameter
            _lambda: Regularization constant
            _alpha: Momentum hyperparameter
            _learning_rate: Learning rate for the update

        Note: You can learn more about momentum here: https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/

        Returns: Updated model
    '''

    for module_name, module in model.items():

        # check if a module has learnable parameters
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                g = module.gradient[key] + _lambda * module.params[key]

                if _alpha > 0.0:
                    m = momentum[module_name + '_' + key]
                    momentum[module_name + '_' + key] = m * _alpha - _learning_rate * g
                    module.params[key] += momentum[module_name + '_' + key]


                else:
                    module.params[key] -= _learning_rate * g

    return model


def main(main_params, optimization_type="minibatch_sgd"):
    np.random.seed(int(main_params['random_seed']))

    Xtrain, Ytrain, Xval, Yval, _, _ = data_loader_mnist(dataset=main_params['input_file'])
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape

    index = np.arange(10)
    unique, counts = np.unique(Ytrain, return_counts=True)
    counts = dict(zip(unique, counts)).values()

    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)

    model = dict()
    num_L1 = 1000
    num_L2 = 10

    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = float(main_params['alpha'])
    _lambda = float(main_params['lambda'])
    _dropout_rate = float(main_params['dropout_rate'])
    _activation = main_params['activation']

    if _activation == 'relu':
        act = relu
    else:
        act = tanh

    model['L1'] = linear_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = act()
    model['drop1'] = dropout(r=_dropout_rate)
    model['L2'] = linear_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = softmax_cross_entropy()

    # Momentum
    if _alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    train_loss_record = []
    val_loss_record = []

    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            _learning_rate = _learning_rate * 0.1

        idx_order = np.random.permutation(N_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0

        for i in range(int(np.floor(N_train / minibatch_size))):
            # get a mini-batch of data
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=True)
            a2 = model['L2'].forward(d1)
            a2 = model['L2'].forward(h1)
            loss = model['loss'].forward(a2, y)

            grad_a2 = model['loss'].backward(a2, y)
            l2_backward = model['L2'].backward(d1, grad_a2)
            drop1_backward = model['drop1'].backward(h1, l2_backward)
            nonlinear1_backward = model['nonlinear1'].backward(a1, drop1_backward)
            grad_x = model['L1'].backward(x, nonlinear1_backward)
            model = miniBatchGradientDescent(model, momentum, _lambda, _alpha, _learning_rate)

        for i in range(int(np.floor(N_train / minibatch_size))):
            x, y = trainSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            loss = model['loss'].forward(a2, y)
            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_loss = train_loss
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        for i in range(int(np.floor(N_val / minibatch_size))):
            x, y = valSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            loss = model['loss'].forward(a2, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    json.dump({'train': train_acc_record, 'val': val_acc_record},
              open('MLP_lr' + str(main_params['learning_rate']) +
                   '_m' + str(main_params['alpha']) +
                   '_w' + str(main_params['lambda']) +
                   '_d' + str(main_params['dropout_rate']) +
                   '_a' + str(main_params['activation']) +
                   '.json', 'w'))

    print('Finish running!')
    return train_loss_record, val_loss_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--alpha', default=0.0)
    parser.add_argument('--lambda', default=0.0)
    parser.add_argument('--dropout_rate', default=0.0)
    parser.add_argument('--num_epoch', default=10)
    parser.add_argument('--minibatch_size', default=5)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--input_file', default='mnist_subset.json')
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)
