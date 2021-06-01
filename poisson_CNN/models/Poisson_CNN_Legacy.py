import tensorflow as tf

from ..dataset.utils import set_max_magnitude_in_batch_and_return_scaling_factors, compute_domain_sizes, flip_and_rotate_tensor

class Poisson_CNN_Legacy(tf.keras.models.Model):
    def __init__(self, hpnn, dbcnn, jacobi_iterations = 0):
        super().__init__()
        self.hpnn = hpnn
        self.dbcnn = dbcnn
        if jacobi_iterations > 0:
            self.jacobi_iteration_layer = poisson_CNN.layers.JacobiIterationLayer([3,3],[2,2],2,data_format=self.hpnn.data_format, n_iterations = jacobi_iterations)
        else:
            self.jacobi_iteration_layer = None
        
    def call(self, inp):

        rhs = inp[0]
        left = inp[1]
        top = inp[2]
        right = inp[3]
        bottom = inp[4]
        dx = inp[5]
        rhs, rhs_scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(rhs, 1.0)
        left, left_boundary_scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(left, 1.0)
        top, top_boundary_scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(top, 1.0)
        right, right_boundary_scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(right, 1.0)
        bottom, bottom_boundary_scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(bottom, 1.0)

        hpnn_result = self.hpnn([rhs, dx])#/rhs_scaling_factors
        hpnn_result = tf.einsum('b...,b->b...',hpnn_result,tf.reduce_max(compute_domain_sizes(tf.concat([inp[-1],inp[-1]],1), tf.shape(rhs)[2:]),1)**2/rhs_scaling_factors)
        
        left_bc_result = self.dbcnn([left, dx, tf.shape(rhs)[2]])
        left_bc_result = tf.einsum('b...,b->b...', left_bc_result, 1/left_boundary_scaling_factors)  

        top_bc_result = self.dbcnn([top, dx, tf.shape(rhs)[3]])
        top_bc_result = tf.einsum('b...,b->b...', top_bc_result, 1/top_boundary_scaling_factors) 
        top_bc_result = flip_and_rotate_tensor(top_bc_result,rotation_count = 3, data_format = self.dbcnn.data_format, flip_axes = [])
        
        right_bc_result = self.dbcnn([right, dx, tf.shape(rhs)[2]])#/right_boundary_scaling_factors#channels_first_flip_left_right()
        right_bc_result = tf.einsum('b...,b->b...', right_bc_result, 1/right_boundary_scaling_factors) 
        right_bc_result = flip_and_rotate_tensor(right_bc_result, rotation_count = 0, flip_axes = [2])
        
        bottom_bc_result = self.dbcnn([bottom, dx, tf.shape(rhs)[3]])#/bottom_boundary_scaling_factors#channels_first_rot90(, k = 3) 
        bottom_bc_result = tf.einsum('b...,b->b...', bottom_bc_result, 1/bottom_boundary_scaling_factors) 
        bottom_bc_result = flip_and_rotate_tensor(bottom_bc_result, rotation_count = 1, flip_axes = [2])
        
        pred = left_bc_result + right_bc_result + top_bc_result + bottom_bc_result + hpnn_result 
        if self.jacobi_iteration_layer is not None:
            pred = self.jacobi_iteration_layer([pred, rhs, dx])
            
        return pred

    def train_step(self,data):

        inputs, ground_truth = data
        rhses, left_bc, top_bc, right_bc, bot_bc, dx = inputs

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(inputs)
            loss = self.loss_fn(y_true=ground_truth, y_pred=pred, rhs=rhses, dx = dx)
        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2), 'lr': self.optimizer.learning_rate}
    
    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss


            
