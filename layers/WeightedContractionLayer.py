import tensorflow as tf
import opt_einsum as oe

class WeightedContractionLayer(tf.keras.layers.Layer):
    '''
    Given the input tensor, this layer performs an einsum contraction with learnable weights on the input (i.e. the tensor the input is einsummed with is learnable)
    '''
    def __init__(self, contraction_expression = None, new_weight_dims = [], contraction_expression_contains_batch_index = False, softmax_weights = True, **kwargs):
        '''
        Init arguments
        
        contraction_expression: str. Must be a valid opt_einsum contraction expression. First operand represents the weights to be created. Unlike typical einsum, if you have ellipses in the input tensor and you want to create the appropriately sized weight tensor for those axes, you MUST use an ellipsis in the weight tensor indices as well. Do NOT USE Theta as a tensor index - this is used to handle ellipses.
        new_weight_dims: List of the size of new dimensions introduced in the weight tensor in the order they appear. For example, the contraction 'mk,jkl->ml' has a new axis m for the weights, so the user must specify its size. If an int is supplied, it will be automatically converted to a 1 element long list.
        contraction_expression_contains_batch_index: Boolean. Every tensor keras works with has 1 batch dimension as the 0th dim. Set this to False if your contraction_expression does not account for this - the init will take care of it.
        softmax_weights: Boolean. If set to true, the weights will be softmaxed before the contraction is performed.
        '''
        if not isinstance(contraction_expression, str):
            raise(TypeError('Contraction expression must be a string valid for np.einsum'))
        
        if isinstance(new_weight_dims, int):
            new_weight_dims = [new_weight_dims]
        
        self.softmax_weights = softmax_weights
        
        #parse contraction_expression
        self.contraction_expression = contraction_expression
        input_expressions, self.output_expression = contraction_expression.replace(' ', '').split('->')
        self.weight_expression, self.input_expression = input_expressions.replace('...', 'Θ').split(',')
        self.new_weight_dims = new_weight_dims
        
        #find which indices are shared and which aren't between the weights and the input
        self.shared_indices = list(set(self.input_expression) & set(self.weight_expression)) #indices to be shared between the weight tensor and the input
        self.new_indices = list(set(self.weight_expression) - set(self.input_expression)) #new indices to be created that don't correspond to a dim in the weight tensor
        
        #infer new axis weight lengths
        self.weight_shape = []
        if (new_weight_dims == []) and (self.new_indices != []):
            raise(ValueError('New weight dimensions must be not None if new dimensions are being created'))
        elif len(self.new_indices) != len(new_weight_dims):
            raise(ValueError('Supply exactly one size for every new dimension'))
        
        for i in range(len(self.weight_expression)):
            if self.weight_expression[i] in self.new_indices:
                self.weight_shape.append(new_weight_dims.pop(0))
            else:
                self.weight_shape.append(None)

        #correct contraction expression for batch index if not supplied
        if not contraction_expression_contains_batch_index:
            self.contraction_expression = self.contraction_expression[:self.contraction_expression.find('->')+2] + 'z' + self.contraction_expression[self.contraction_expression.find('->')+2:]
            self.contraction_expression = self.contraction_expression[:self.contraction_expression.find(',')+1] + 'z' + self.contraction_expression[self.contraction_expression.find(',')+1:]
            
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        #infer shared axis weight lengths
        input_shape = input_shape[1:]
        for shared_index in self.shared_indices:
            input_dim_number = self.input_expression.find(shared_index)
            weight_dim_number = self.weight_expression.find(shared_index)
            try:
                self.weight_shape[weight_dim_number] = int(input_shape[input_dim_number])
            except:
                raise(ValueError('Shared indices along which a contraction is performed must have predetermined lengths'))
        #create weights
        self.contraction_weights = self.add_weight(name = 'contraction_weights', shape = self.weight_shape, initializer='glorot_uniform', trainable=True)
        super().build(input_shape)
        
    def call(self, inp):
        if None in inp.shape: #This is ONLY necessary to not throw an exception during the call after __init__. DO NOT REMOVE.
            newshape = [1 if v is None else v for v in inp.shape]
            inp = tf.random.uniform(newshape, dtype = tf.keras.backend.floatx())
        #contract
        if self.softmax_weights:
            return oe.contract(self.contraction_expression, tf.nn.softmax(self.contraction_weights), inp, backend = 'tensorflow')
        else:
            return oe.contract(self.contraction_expression, self.contraction_weights, inp, backend = 'tensorflow')
