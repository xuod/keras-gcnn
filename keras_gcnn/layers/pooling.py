from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects


class GroupPool(Layer):
    def __init__(self, h_input, mode='mean', **kwargs):
        super(GroupPool, self).__init__(**kwargs)
        self.h_input = h_input
        self.mode = mode
        if self.mode == 'e1':
            self.w = [+1,-1,+1,-1,+1,-1,+1,-1]
        if self.mode == 'e2':
            self.w = [+1,-1,+1,-1,-1,+1,-1,+1]

    def build(self, input_shape):
        self.shape = input_shape
        super(GroupPool, self).build(input_shape)

    @property
    def nti(self):
        nti = 1
        if self.h_input == 'C4':
            nti *= 4
        elif self.h_input == 'D4':
            nti *= 8
        return nti

    # def _fix_unknown_dimension(self, input_shape, output_shape):
    #     output_shape = list(output_shape)
    #     msg = 'total size of new array must be unchanged'

    #     known, unknown = 1, None
    #     for index, dim in enumerate(output_shape):
    #         if dim < 0:
    #             if unknown is None:
    #                 unknown = index
    #             else:
    #                 raise ValueError('Can only specify one unknown dimension.')
    #         else:
    #             known *= dim

    #     original = np.prod(input_shape, dtype=int)
    #     if unknown is not None:
    #         if known == 0 or original % known != 0:
    #             raise ValueError(msg)
    #         output_shape[unknown] = original // known
    #     elif original != known:
    #         raise ValueError(msg)

    #     return tuple(output_shape)

    # def compute_output_shape(self, input_shape):
    #     if None in input_shape[1:]:
    #         # input shape (partially) unknown? replace -1's with None's
    #         return ((input_shape[0],) +
    #                 tuple(s if s != -1 else None for s in self.target_shape))
    #     else:
    #         # input shape known? then we can compute the output shape
    #         return (input_shape[0],) + self._fix_unknown_dimension(
    #             input_shape[1:], self.target_shape)

    # def call(self, inputs):
    #     return K.reshape(inputs, (K.shape(inputs)[0],) + self.target_shape)

    def call(self, x):
        shape = x.shape
        print(shape)
        stack_shape = (K.shape(x)[0],) + tuple(map(int, [shape[1], shape[2], shape[3] // self.nti, self.nti]))
        print(stack_shape)
        input_reshaped = K.reshape(x, stack_shape)
        print(input_reshaped)
        if self.mode == 'mean':
            mean_per_group = K.mean(input_reshaped, -1)
            return mean_per_group
        elif self.mode == 'max':
            mean_per_group = K.max(input_reshaped, -1)
            return mean_per_group
        elif self.mode == 'e1' or self.mode == 'e2':
            mean_per_group = K.mean(input_reshaped*self.w, -1)
            return mean_per_group
        else:
            raise NotImplemented

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // self.nti)

    def get_config(self):
        config = super(GroupPool, self).get_config()
        config['h_input'] = self.h_input
        return config


get_custom_objects().update({'GroupPool': GroupPool})
