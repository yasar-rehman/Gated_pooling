class gated_pool(Layer):

    # Implementation of Gated pooling,  for learning a parameter per layer option

    def __init__(self, pool_size, strides, padding ='same', data_format=None,**kwargs):

        super(gated_pool,self).__init__(**kwargs)
        self.pool_size = pool_size,
        self.strides = strides
        self.padding = padding
        self.data_format = data_format


    def build(self, input_shape):
        # input_shape is a 4D tensor with [batch size, height, width ,channels]
        self.mask = self.add_weight(name='kernel1',
                                    shape=(self.pool_size[0][0], self.pool_size[0][0],1,1),
                                    initializer='uniform',
                                    trainable=True)

        super(gated_pool, self).build(input_shape)

    def call(self, x, **kwargs):

        nb_batch, input_row, input_col, nb_filter = K.int_shape(x)   # get the output shape

        # output_size = input_row // 2  # output size should be reduced to half

        xs = []

        for c in tf.split(x, nb_filter, 3):
            conv1 = K.conv2d(c,
                             self.mask,
                             strides=(self.strides[0], self.strides[0]),
                             padding='same')


            xs.append(conv1)

        output = K.sigmoid(K.concatenate(xs, axis=3))
        print(output)

        pool_max = K.pool2d(x,
                            pool_size=(self.pool_size[0][0],self.pool_size[0][0]),
                            strides=(self.strides[0],self.strides[0]),
                            padding='same',
                            pool_mode='max')
        pool_avg = K.pool2d(x,
                            pool_size=(self.pool_size[0][0], self.pool_size[0][0]),
                            strides=(self.strides[0], self.strides[0]),
                            padding='same',
                            pool_mode='avg')

        f_gated = tf.add(tf.multiply(output, pool_max), tf.multiply((1-output), pool_avg))

        return f_gated

    def compute_output_shape(self, input_shape):
        rows = np.int(np.ceil(((input_shape[1] - self.pool_size[0][0]) / self.strides[0])) + 1)
        cols = np.int(np.ceil(((input_shape[2] - self.pool_size[0][0]) / self.strides[0])) + 1)
        return (input_shape[0], rows, cols, input_shape[3])
