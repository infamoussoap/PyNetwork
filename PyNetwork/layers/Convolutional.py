import numpy as np
from numpy.lib.stride_tricks import as_strided

from PyNetwork import get_activation_function
from PyNetwork.layers import Layer
from PyNetwork.validation import check_layer


class Conv_2D(Layer):
    """ 2D Convolutional layer

        Attributes
        ----------
        filter_num : int
            The number of filters to use
        filter_spatial_shape : 2 tuple of int
            The shape of the filter spatial dimension. Because this is a 2D convolution layer
            the filter spatial dimension is always 2 dimensional.
        single_filter_shape : 3 tuple of int
            The shape of one filter only, including the spectral dimension.
            This is equal to `filter_spatial_size + (color/spectral dimension of input, )`
        filter_shape : 4 tuple of int
            The shape of the full filter, including the number of filters.
            This is equal to `(*single_filter_shape, filter_num)`

        stride : int
            Filter stride

        filter : (*single_filter_shape, filter_num) np.array
            The filter of this layer
        b : (filter_num, ) np.array
            The bias units

        input_shape : 3 tuple of int
            The shape of a single input into this layer
        output_spatial_shape : 2 tuple of int
            The spatial size of the output. Because this is a 2D convolution layer, the output
            spatial dimension is 2 dimensional
        output_shape : 3 tuple of int
            The actually output shape, including the number of filters
            This is equal to `output_spatial_size + (filter_num, )`

        g_name : str
            The name of the activation function
        g : :obj:func
            The actual activation function

        built : bool
            Has the layer been initialised
    """

    @staticmethod
    def im2window(images, filter_spartial_shape, stride):
        """ Returns the sliding window result when given multiple z

            Parameters
            ----------
            images : (N, i, j, k) np.array
                Assumed to be many z, which is accessed by the 0th index, for which the
                window is to be slid across
            filter_spartial_shape : 2 tuple
                The spartial dimensions of the filter. Note that this can also simply be the filter
                size, assume that the first 2 dimensions are the spartial dimensions (which it should be)
            stride : int
                The stride of the filter

            Returns
            -------
            (N, i, j, k, l, m) np.array
                The results of the sliding window
        """

        image_num = images.shape[0]

        height = (images.shape[1] - filter_spartial_shape[0]) // stride + 1
        width = (images.shape[2] - filter_spartial_shape[1]) // stride + 1

        strides = (images.strides[0], images.strides[1] * stride, images.strides[2] * stride, *images.strides[1:])
        windowed = as_strided(images, (image_num, height, width, *filter_spartial_shape, images.shape[3]),
                              strides=strides)

        return windowed

    @staticmethod
    def im2col(images, filter_spartial_shape, stride):
        """ Returns the flatten sliding window result when given multiple z

            Parameters
            ----------
            images : (N, i, j, k) np.array
                Assumed to be many z, which is accessed by the 0th index, for which the
                window is to be slid across
            filter_spartial_shape : 2 tuple
                The spartial dimensions of the filter. Note that this can also simply be the filter
                size, assume that the first 2 dimensions are the spartial dimensions (which it should be)
            stride : int
                The stride of the filter

            Returns
            -------
            (N, i, j, k*l*m) np.array
                The results of the sliding window

            Notes
            -----
            The result of `im2col` and `im2window` are effectively the same thing. But `im2col` allows for
            matrix multiplication when performing the convolution, which is far faster than the alternative
            broadcasting
            Also note that we can't simply use `as_strided` here, and we need to reshape the results from
            `im2window`. The reason is that with `as_strided` there is no way to jump down a row once you
            reached the end of the filter spatial size. You can only do this by returning the windowed results
        """

        windowed = Conv_2D.im2window(images, filter_spartial_shape, stride)
        shape = windowed.shape
        return windowed.reshape(shape[0], shape[1], shape[2], np.prod(shape[3:]))

    @staticmethod
    def perform_conv(images, filter_, bias, stride):
        """ Performs the standard convolution of the input `z`

            Parameters
            ----------
            images : (N, i, j, k) np.array
                Assumed to be many z, which is accessed by the 0th index, for which the
                window is to be slid across
            filter_ : (:, :, k, f) np.array
                The convolution filters, k is the color dimension of an image, and f is the number
                of filters
            bias : (f, ) np.array
                The bias for each individual filter
            stride : int
                The filter stride

            Returns
            -------
            (N, :, :, f) np.array
                The convolved results

            Notes
            -----
            This is the newer, but faster implementation of convolution. Having run some tests,
            it seems very likely that it returns the same result as `perform_conv_broadcasting`.
        """

        shape = filter_.shape
        filter_spatial_shape = shape[:2]

        col = Conv_2D.im2col(images, filter_spatial_shape, stride)
        flatten_filter = filter_.reshape((np.prod(shape[:-1]), shape[-1]), order='A')

        return col @ flatten_filter[None, :, :] + bias[None, None, None, :]

    @staticmethod
    def perform_conv_broadcasting(images, filter_, bias, stride):
        """ Performs the standard convolution of the input `z`

            Parameters
            ----------
            images : (N, i, j, k) np.array
                Assumed to be many z, which is accessed by the 0th index, for which the
                window is to be slid across
            filter_ : (:, :, k, f) np.array
                The convolution filters, k is the color dimension of an image, and f is the number
                of filters
            bias : (f, ) np.array
                The bias for each individual filter
            stride : int
                The filter stride

            Returns
            -------
            (N, :, :, f) np.array
                The convolved results

            Notes
            -----
            This was the original implementation of convolution, which I am very certain is 100% accurate.
            But the sum in the second last line proves to be a major bottleneck of the code. It is instead
            much faster to flatten the `im2window` result and perform matrix multiplication with the filter.
        """

        filter_spartial_shape = filter_.shape[:2]
        windowed = Conv_2D.im2window(images, filter_spartial_shape, stride)

        conv_dot = filter_[None, None, None, :, :, :, :] * windowed[..., None]
        conv = np.sum(conv_dot, axis=(3, 4, 5)) + bias[None, None, None, :]

        return conv

    def __init__(self, filter_num, filter_spatial_shape, stride, activation_function, *args, activation_kwargs=None,
                 **kwargs):
        """ Initialise the basic information of this conv_2d layer

            Parameters
            ----------
            filter_num : int
                The number of filters to use
            filter_spatial_shape : 2 tuple of int
                The spatial dimension of a single filter
            stride : int
                The filter stride
            activation_function : str
                The activation function for this layer
            activation_kwargs : dict of str - :obj:, optional
                The keyword arguments for the activation function if it has hyper-parameters
        """

        self.filter_num = filter_num
        self.filter_spatial_shape = filter_spatial_shape

        self.stride = stride

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.input_shape = None
        self.single_filter_shape = None
        self.filter_shape = None
        self.output_spatial_shape = None
        self.output_shape = None
        self.filter = None
        self.b = None

        self.built = False

    def build(self, previous_output_shape):
        """ Initialise the filters and bias units, and compute the output shape

            Parameters
            ----------
            previous_output_shape : 3 tuple of int
                The dimensions of the output of the previous layer, equivalently the dimensions
                of the input of this layer
        """
        self.input_shape = previous_output_shape
        self.single_filter_shape = (*self.filter_spatial_shape, previous_output_shape[2])

        self.filter_shape = (*self.single_filter_shape, self.filter_num)

        # These 2 lines follow the formula in the youtube lecture
        # Giving us the output shape of this layer
        n = int((previous_output_shape[0] - self.filter_spatial_shape[0]) / self.stride + 1)
        m = int((previous_output_shape[1] - self.filter_spatial_shape[1]) / self.stride + 1)

        self.output_spatial_shape = (n, m)
        self.output_shape = (*self.output_spatial_shape, self.filter_num)

        # Initialise the the filter with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.single_filter_shape) + 1))
        self.filter = np.random.uniform(low=-limit, high=limit, size=self.single_filter_shape)
        self.b = np.zeros(self.filter_num)

        self.built = True

    def predict(self, z, output_only=True, pre_activation_of_input=None):
        """ Returns the output of this layer

            Parameters
            ----------
            z : (N, i, j, k) np.array
                Assumed to be many images, which is accessed by the 0th index, for which the
                window is to be slid across
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (N, a, b, c) np.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (N, a, b, c) np.array, (N, a, b, c) np.array
                The first np.array will store the output before it is passed through the activation
                function.
                The second np.array will store the output after it has passed through the
                activation function.
        """
        check_layer(self)

        conv = self.perform_conv(z, self.filter, self.b, self.stride)

        if output_only:
            return self.activation_function_(conv)
        return conv, self.activation_function_(conv)

    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, *input_shape) np.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : (N, *output_shape) np.array
                The delta for this layer, delta^{k, f}_{m, j}

            Returns
            -------
            (N, *input_shape) np.array
                Returns delta of the previous layer, delta^{k-1}

            Notes
            -----
            We want to return delta^{k-1} because the `sequential` class does not have access to the
            filters, W. But it does know the values of g'_{k-1} and delta^k, due to forward propagation
            and the backwards nature of the back propagation algorithm.
        """
        check_layer(self)

        # I don't even know how to explain this code
        # But it makes sense when you look at the math

        # Essentially these 2 lines returns the elements of the filter
        # that hits a given pixel position
        eye = np.eye(np.prod(self.input_shape)).reshape((np.prod(self.input_shape), *self.input_shape))
        eye_conv = self.perform_conv(eye, self.filter, np.zeros(self.filter_num), self.stride)

        # Reshape
        eye_conv = eye_conv.reshape((*self.input_shape, *self.output_spatial_shape, self.filter_num))

        # Self-explanatory once you look at the math
        delta = np.einsum("ijkl,abcjkl,iabc->iabc", new_delta, eye_conv, g_prime, optimize='greedy')
        return delta

    def get_weight_grad_(self, delta, prev_z):
        """ Get the gradients for the filter matrix and bias units

            Parameters
            ----------
            delta : (N, *output_shape) np.array
                In latex, this should be delta_k
            prev_z : (N, *input_shape) np.array
                This should be the output, post activation, of the previous layer (z_{k-1})

            Returns
            -------
            (filter_num, ) np.array, (*filter_shape) np.array
                The first array is the gradient for the bias unit
                The second array is the gradient for the filter matrix

        """
        check_layer(self)

        # This code is self-explanatory when you look at the math
        windowed = self.im2window(prev_z, self.filter_spatial_shape, self.stride)
        
        w_grad = np.einsum("abcijk,abcl->ijkl", windowed, delta, optimize="greedy")
        b_grad = np.sum(delta, axis=(0, 1, 2))
        return b_grad, w_grad

    def update_parameters_(self, bias_updates, filter_updates):
        """ Update the filter and bias by descending down the gradient

            Parameters
            ----------
            bias_updates : (f, ) np.array
                Gradient of the bais
            filter_updates : (:, :, :, f) np.array
                Gradient of the filter
        """
        check_layer(self)

        self.filter -= filter_updates
        self.b -= bias_updates

    def get_weights(self):
        check_layer(self)
        return self.filter, self.b

    def summary_(self):
        check_layer(self)
        return f'Conv 2D {self.filter_num} x {self.filter_spatial_shape}', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function, **self.activation_kwargs)

    def __str__(self):
        return f'Conv 2D {self.filter_num} x {self.filter_spatial_shape}'
