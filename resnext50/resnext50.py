from keras.models import Model
import keras.backend as K
from keras.layers import Input, Conv2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dense, Activation, MaxPooling2D
from keras.layers import Flatten, add, concatenate
from keras.engine import InputSpec
import h5py
from keras.utils import plot_model
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
epsilon = 2e-05


class GroupConv2D(Conv2D):

    """Groiped 2D convolution layer


    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_group=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.num_group = num_group
        if self.filters % self.num_group != 0:
            raise ValueError("filters must divided by num_group with no remainders!")
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        if input_dim % self.num_group != 0:
            raise ValueError("The channel dimension of input tensor must divided by num_group with no remainders!")

        kernel_shape = self.kernel_size + (input_dim/self.num_group, self.filters)

        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        self.channel_num = input_dim


    def call(self, inputs):
        filter_in_group = self.filters / self.num_group
        if self.data_format == 'channels_first':
            channel_axis = 1
            input_in_group = self.channel_num / self.num_group
            outputs_list = []
            for i in range(self.num_group):
                outputs = K.conv2d(
                    inputs[:,i*input_in_group:(i+1)*input_in_group,:,:],
                    self.kernel[:, :, :, i*filter_in_group:(i+1)*filter_in_group],
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = K.bias_add(
                                         outputs,
                                         self.bias[i*filter_in_group:(i+1)*filter_in_group],
                                         data_format=self.data_format)
                outputs_list.append(outputs)

        elif self.data_format == 'channels_last':
            outputs_list = []
            channel_axis = -1
            input_in_group = self.channel_num / self.num_group
            for i in range(self.num_group):
                outputs = K.conv2d(
                    inputs[:, :, :, i*input_in_group:(i+1)*input_in_group],
                    self.kernel[:, :, :, i*filter_in_group:(i+1)*filter_in_group],
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = K.bias_add(
                                         outputs,
                                         self.bias[i*filter_in_group:(i+1)*filter_in_group],
                                         data_format=self.data_format)
                outputs_list.append(outputs)

        outputs = concatenate(outputs_list, axis=channel_axis)
        return outputs

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config["num_group"] = self.num_group
        return config


def unit(base_name, num_filters, input_tensor):
    x = Conv2D(num_filters[0], (1, 1), use_bias=False, name=base_name+"conv1")(input_tensor)
    x = BatchNormalization(epsilon=epsilon, momentum=0.9, axis=3, name=base_name+"bn1")(x)
    x = Activation("relu", name=base_name+"relu1")(x)

    x = ZeroPadding2D((1, 1))(x)
    x = GroupConv2D(num_filters[1], (3, 3), use_bias=False, num_group=32, name=base_name+"conv2")(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.9, axis=3, name=base_name+"bn2")(x)
    x = Activation("relu", name=base_name+"relu2")(x)

    x = Conv2D(num_filters[2], (1, 1), use_bias=False, name=base_name+"conv3")(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.9, axis=3, name=base_name+"bn3")(x)

    return x


def stage(stage_id, num_unit, num_filters, input_tensor):
    base_name = "stage%d_unit%d_"
    for i in range(1, num_unit+1):
        x1 = input_tensor
        x1 = unit(base_name=base_name % (stage_id, i), num_filters=num_filters, input_tensor=x1)
        if i == 1:
            x2 = Conv2D(num_filters[2], (1, 1), use_bias=False, name=base_name % (stage_id, i)+"sc")(input_tensor)
            x2 = BatchNormalization(epsilon=epsilon, momentum=0.9, name=base_name % (stage_id, i)+"sc_bn")(x2)
        else:
            x2 = input_tensor
        input_tensor = add([x1, x2])
        input_tensor = Activation("relu", name=base_name % (stage_id, i)+"relu")(input_tensor)

    return input_tensor


def ResNext50():
    #input_tensor = Input(shape=(224, 224, 3), name="input")
    input_tensor = Input(shape=(32, 32, 3), name="input")
    x = BatchNormalization(epsilon=epsilon, momentum=0.99, axis=3, name="bn_data")(input_tensor)
    x = ZeroPadding2D((3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv0")(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.99, axis=3, name="bn0")(x)
    x = Activation("relu", name="relu0")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = stage(1, 3, [128, 128, 256], x)
    x = stage(2, 4, [256, 256, 512], x)
    x = stage(3, 6, [512, 512, 1024], x)
    x = stage(4, 3, [1024, 1024, 2048], x)

    x = GlobalAveragePooling2D(name="pool1")(x)
    #x = Dense(1000, name="fc1", activation="softmax")(x)
    x = Dense(10, name="fc1", activation="softmax")(x)
    model = Model(input_tensor, x)
    return model


def train_on_cifar10(model):
    from keras.datasets import cifar10
    from keras.utils import np_utils
    # settings
    batch_size = 32
    nb_classes = 10
    nb_epoch = 200


    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    X_train -= np.mean(X_train, axis=0)
    X_test -= np.mean(X_test, axis=0)


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test,Y_test))
    score = model.evaluate(X_test, Y_test)
    print(score)
    return None

def set_weights_from_mxnet(model, ft_file,fnt_file):
    ft = h5py.File(ft_file)
    fnt = h5py.File(fnt_file)
    for layer in model.layers:
        if layer.__class__.__name__ == "Conv2D" or layer.__class__.__name__ == "GroupConv2D":
            kernel_weight = layer.name + "_weight"
            w = ft[kernel_weight][:]
            w_trans = np.transpose(w, (2,3,1,0))
            layer.set_weights([w_trans])
        elif layer.__class__.__name__ == "BatchNormalization":
            gamma = layer.name + "_gamma"
            beta = layer.name + "_beta"
            mean = layer.name + "_moving_mean"
            std = layer.name + "_moving_var"
            bn_weights = [ft[gamma][:], ft[beta][:], fnt[mean][:], fnt[std][:]]
            layer.set_weights(bn_weights)
        elif layer.__class__.__name__ == "Dense":
            fc_weights = layer.name + "_weight"
            fc_bias = layer.name + "_bias"
            layer.set_weights([ft[fc_weights][:].T, ft[fc_bias][:]])

    return model


def set_weights_from_caffe(model, caffe_h5py):
    f = h5py.File(caffe_h5py)
    for layer in model.layers:
        if layer.__class__.__name__ == "Conv2D" or layer.__class__.__name__ == "GroupConv2D":
            kernel_weight = layer.name
            w = f[kernel_weight]['weights'][:]
            w_trans = np.transpose(w, (2,3,1,0))
            layer.set_weights([w_trans])
        elif layer.__class__.__name__ == "BatchNormalization":
            gamma = f[layer.name]['weights'][:]
            beta = f[layer.name]['bias'][:]
            mean = f["scale_"+layer.name]['weights'][:]
            std = f["scale_"+layer.name]['weights'][:]
            bn_weights = [gamma, beta, mean, std]
            layer.set_weights(bn_weights)
        elif layer.__class__.__name__ == "Dense":
            layer.set_weights([f[layer.name]['weights'][:].T, f[layer.name]['bias'][:]])
    return model


if __name__ == "__main__":
#    import keras.backend as K
#    K.set_image_data_format("channels_last")
#    model = ResNext50()
#    ft_file = "ResNext_trainable.h5"
#    fnt_file = "ResNext_non_trainable.h5"
#    caffe_file = "resnext50_caffe_weights.h5"
#    model = set_weights_from_caffe(model, caffe_file)
#    model.save("resnext50_weights_tf_data_format_tf_kernels.h5")
#    # all conv2 layers fail to load, maybe mismatch
#    test1 = load_img("cat.jpg", target_size = (224,224,3))
#    test2 = load_img("airplane.jpg", target_size = (224,224,3))
#
#    test1 = img_to_array(test1)
#    test2 = img_to_array(test2)
#
#    test1 = np.expand_dims(test1, axis=0).astype(np.float64)
#    test2 = np.expand_dims(test2, axis=0).astype(np.float64)
#
#    #test1 = preprocess_input(test1)
#    #test2 = preprocess_input(test2)
#
#
#
#    r1 = model.predict(test1)
#    r2 = model.predict(test2)
#
#    str_r1 = decode_predictions(r1)
#    str_r2 = decode_predictions(r2)
#
#    print("result for r1 is %s"%str_r1)
#    print("result for r2 is %s"%str_r2)
    model = ResNext50()
    train_on_cifar10(model)
