import tensorflow as tf
from warnings import warn


class ConcreteGate:

    def __init__(self, name, log_a, shape, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=0.0, l2_penalty=0.0, eps=1e-6, hard=True, local_rep=False):
        self.name = name
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty, self.l2_penalty = l0_penalty, l2_penalty
        self.hard, self.local_rep = hard, local_rep
        
        self.scope = tf.get_variable_scope()
        self.log_a = log_a
        # with tf.variable_scope(self.scope):
        #     self.log_a = tf.get_variable("log_a", shape=shape)
 
    def __call__(self, values, is_train=None, axis=None, reg_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        gates = self.get_gates(is_train, shape=tf.shape(values) if self.local_rep else None)

        if self.l0_penalty != 0 or self.l2_penalty != 0:
            reg = self.get_penalty(values=values, axis=axis)
            tf.add_to_collection(reg_collection, tf.identity(reg, name='concrete_gate_reg'))
        return values * gates

    def get_gates(self, is_train, shape=None):
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits
        with tf.name_scope(self.name):
            if is_train:
                shape = tf.shape(self.log_a) if shape is None else shape
                noise = tf.random_uniform(shape, self.eps, 1.0 - self.eps)
                concrete = tf.nn.sigmoid((tf.log(noise) - tf.log(1 - noise) + self.log_a) / self.temperature)
            else:
                concrete = tf.nn.sigmoid(self.log_a)

            stretched_concrete = concrete * (high - low) + low
            clipped_concrete = tf.clip_by_value(stretched_concrete, 0, 1)
            if self.hard:
                hard_concrete = tf.to_float(tf.greater(clipped_concrete, 0.5))
                clipped_concrete = clipped_concrete + tf.stop_gradient(hard_concrete - clipped_concrete)
        return clipped_concrete

    def get_penalty(self, values=None, axis=None):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        if self.l0_penalty == self.l2_penalty == 0:
            warn("get_penalty() is called with both penalties set to 0")
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        with tf.name_scope(self.name):
            # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
            p_open = tf.nn.sigmoid(self.log_a - self.temperature * tf.log(-low / high))
            p_open = tf.clip_by_value(p_open, self.eps, 1.0 - self.eps)

            total_reg = 0.0
            if self.l0_penalty != 0:
                if values != None and self.local_rep:
                    p_open += tf.zeros_like(values)  # broadcast shape to account for values
                l0_reg = self.l0_penalty * tf.reduce_sum(p_open, axis=axis)
                total_reg += tf.reduce_mean(l0_reg)

            if self.l2_penalty != 0:
                assert values is not None
                l2_reg = 0.5 * self.l2_penalty * p_open * tf.reduce_sum(values ** 2, axis=axis)
                total_reg += tf.reduce_mean(l2_reg)

            return total_reg

    def get_sparsity_rate(self, is_train=False):
        """ Computes the fraction of gates which are now active (non-zero) """
        is_nonzero = tf.not_equal(self.get_gates(is_train), 0.0)
        return tf.reduce_mean(tf.to_float(is_nonzero))
