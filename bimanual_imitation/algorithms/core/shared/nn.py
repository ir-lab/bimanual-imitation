import abc
import collections
import hashlib
import json
import warnings
from abc import abstractmethod
from contextlib import contextmanager

import h5py
import numpy as np
import tables
import theano
from theano import tensor

from . import util

warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)


# Global variable scoping utility, modeled after Tensorflow
_curr_active_scope = None


class variable_scope(object):
    def __init__(self, name):
        global _curr_active_scope
        self.name = name
        if self.name is None:
            # This is the root scope
            assert _curr_active_scope is None
            self.parent = None
            self.fullname = ""
        else:
            # This is not the root scope
            assert _curr_active_scope is not None
            assert "/" not in name
            self.parent = _curr_active_scope
            assert (
                self.name not in self.parent.children
            ), "Scope {} already exists in parent scope {}".format(self.name, self.parent.fullname)
            self.parent.children[self.name] = self
            self.fullname = self.parent.fullname + "/" + self.name

        self.children = collections.OrderedDict()
        self.vars = collections.OrderedDict()

    def __enter__(self):
        global _curr_active_scope
        _curr_active_scope = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _curr_active_scope
        assert _curr_active_scope == self
        _curr_active_scope = self.parent

    def get_child_variables(self, trainable_only):
        vs = [v for v, trainable in self.vars.values() if (not trainable_only or trainable)]
        for c in self.children.values():
            vs += c.get_child_variables(trainable_only)
        return vs

    # to be called by get_variable
    def _register_variable(self, name, init_value, broadcastable, trainable):
        assert "/" not in name
        assert name not in self.vars, "Variable name {} already registered in scope {}".format(
            name, self.fullname
        )
        v = theano.shared(
            value=init_value, name=self.fullname + "/" + name, broadcastable=broadcastable
        )
        self.vars[name] = (v, trainable)
        return v


_curr_active_scope = variable_scope(None)  # this is the root scope


def get_variable(name, init_value, broadcastable=None, trainable=True):
    global _curr_active_scope
    return _curr_active_scope._register_variable(name, init_value, broadcastable, trainable)


def reset_global_scope():
    global _curr_active_scope
    _curr_active_scope = None
    _curr_active_scope = variable_scope(None)


def _hash_name2array(name2array):
    """
    Hashes a list of (name,array) tuples.
    The hash is invariant to permutations of the list.
    """

    def hash_array(a):
        return "%.10f,%.10f,%d" % (np.mean(a), np.var(a), np.argmax(a))

    vals = "|".join("%s %s" for n, h in sorted([(name, hash_array(a)) for name, a in name2array]))
    # return hashlib.sha1('|'.join('%s %s' for n, h in sorted([(name, hash_array(a)) for name, a in name2array]))).hexdigest()
    return hashlib.sha1(vals.encode("utf-8")).hexdigest()


class Model(object):
    """
    A model abstraction. Stores variables and can save/load them to HDF5 files.
    """

    __metaclass__ = abc.ABCMeta

    @property
    @abstractmethod
    def varscope(self):
        pass

    def get_all_variables(self):
        return self.varscope.get_child_variables(trainable_only=False)

    def get_trainable_variables(self):
        return self.varscope.get_child_variables(trainable_only=True)

    def print_trainable_variables(self):
        for v in self.get_trainable_variables():
            util.header("- %s (%d parameters)" % (v.name, v.get_value().size))
        util.header("Total: %d parameters" % (self.get_num_params(),))

    def get_num_params(self):
        return sum(v.get_value().size for v in self.get_trainable_variables())

    ### Utilities for getting/setting flattened parameter vectors ###
    def set_params(self, x):
        # print 'setting param vars:\n{}'.format('\n'.join([v.name for v in self.get_trainable_variables()]))
        assert x.ndim == 1
        pos = 0
        for v in self.get_trainable_variables():
            val = v.get_value()
            s = val.size
            v.set_value(x[pos : pos + s].reshape(val.shape))
            pos += s
        assert pos == x.shape[0]

    def get_params(self):
        return util.flatcat([v.get_value() for v in self.get_trainable_variables()])

    @contextmanager
    def try_params(self, x):
        orig_x = self.get_params()
        self.set_params(x)
        yield
        self.set_params(orig_x)

    # HDF5 saving and loading
    # The hierarchy in the HDF5 file reflects the hierarchy in the Tensorflow graph.

    def savehash(self):
        return _hash_name2array([(v.name, v.get_value()) for v in self.get_all_variables()])

    def save_h5(self, h5file, key, extra_attrs=None):
        with h5py.File(h5file, "a") as f:
            if key in f:
                util.warn("WARNING: key %s already exists in %s" % (key, h5file))
                dset = f[key]
            else:
                dset = f.create_group(key)

            for v in self.get_all_variables():
                dset[v.name] = v.get_value()

            dset.attrs["hash"] = self.savehash()
            if extra_attrs is not None:
                for k, v in extra_attrs:
                    if k in dset.attrs:
                        util.warn("Warning: attribute %s already exists in %s" % (k, dset.name))
                    dset.attrs[k] = v

    def load_h5(self, h5file, key):
        with h5py.File(h5file, "r") as f:
            dset = f[key]

            for v in self.get_all_variables():
                assert v.name[0] == "/"
                vname = v.name[1:]
                print("Reading", vname)
                if vname in dset:
                    v.set_value(dset[vname][...])
                elif vname + ":0" in dset:
                    # Tensorflow saves variables with :0 appended to the name,
                    # so try this for backwards compatibility
                    v.set_value(dset[vname + ":0"][...])
                else:
                    raise RuntimeError("Variable %s not found in %s" % (vname, dset))

            h = self.savehash()
            assert (
                h == dset.attrs["hash"].decode()
            ), "Checkpoint hash %s does not match loaded hash %s" % (dset.attrs["hash"].decode(), h)


# Layers for feedforward networks


class Layer(Model):
    @property
    @abstractmethod
    def output(self):
        pass

    @property
    @abstractmethod
    def output_shape(self):
        """Shape refers to the shape without the batch axis, which always implicitly goes first"""
        pass


class ReshapeLayer(Layer):
    def __init__(self, input_, new_shape):
        self._output_shape = tuple(new_shape)
        util.header("Reshape(new_shape=%s)" % (str(self._output_shape),))
        with variable_scope(type(self).__name__) as self.__varscope:
            self._output = input_.reshape((-1,) + self._output_shape)

    @property
    def varscope(self):
        return self.__varscope

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self):
        return self._output_shape


class AffineLayer(Layer):
    def __init__(self, input_B_Di, input_shape, output_shape, initializer):
        assert len(input_shape) == len(output_shape) == 1
        util.header("Affine(in=%d, out=%d)" % (input_shape[0], output_shape[0]))
        self._output_shape = (output_shape[0],)
        with variable_scope(type(self).__name__) as self.__varscope:
            if initializer is None:
                # initializer = np.random.randn(input_shape[0], output_shape[0]) * np.sqrt(2./input_shape[0])

                # Glorot/Bengio 2010
                s = np.sqrt(6.0 / (input_shape[0] + output_shape[0]))
                initializer = np.random.uniform(
                    low=-s, high=s, size=(input_shape[0], output_shape[0])
                )

            else:
                assert initializer.shape == (input_shape[0], output_shape[0])
            self.W_Di_Do = get_variable("W", initializer.astype(theano.config.floatX))
            self.b_1_Do = get_variable(
                "b",
                np.zeros((1, output_shape[0]), dtype=theano.config.floatX),
                broadcastable=(True, False),
            )
            self._output_B_Do = input_B_Di.dot(self.W_Di_Do) + self.b_1_Do

    @property
    def varscope(self):
        return self.__varscope

    @property
    def output(self):
        return self._output_B_Do

    @property
    def output_shape(self):
        return self._output_shape


class NonlinearityLayer(Layer):
    def __init__(self, input_B_Di, output_shape, func):
        util.header("Nonlinearity(func=%s)" % func)
        self._output_shape = output_shape
        with variable_scope(type(self).__name__) as self.__varscope:
            self._output_B_Do = {
                "relu": tensor.nnet.relu,
                "lrelu": lambda x: tensor.nnet.relu(x, 0.01),
                "elu": tensor.nnet.elu,
                "tanh": tensor.tanh,
            }[func](input_B_Di)

    @property
    def varscope(self):
        return self.__varscope

    @property
    def output(self):
        return self._output_B_Do

    @property
    def output_shape(self):
        return self._output_shape


def _check_keys(d, keys, optional):
    s = set(d.keys())
    if not (s == set(keys) or s == set(keys + optional)):
        raise RuntimeError(
            "Got keys %s, but expected keys %s with optional keys %s"
            % (str(s, str(keys), str(optional)))
        )


def _parse_initializer(layerspec):
    if "initializer" not in layerspec:
        return None
    initspec = layerspec["initializer"]
    raise NotImplementedError("Unknown layer initializer type %s" % initspec["type"])


class FeedforwardNet(Layer):
    def __init__(self, input_B_Di, input_shape, layerspec_json):
        """
        Args:
            layerspec (string): JSON string describing layers
        """
        assert len(input_shape) >= 1
        self.input_B_Di = input_B_Di

        layerspec = json.loads(layerspec_json)
        util.header("Loading feedforward net specification")
        print(json.dumps(layerspec, indent=2, separators=(",", ": ")))

        self.layers = []
        with variable_scope(type(self).__name__) as self.__varscope:

            prev_output, prev_output_shape = input_B_Di, input_shape

            for i_layer, ls in enumerate(layerspec):
                with variable_scope("layer_%d" % i_layer):
                    if ls["type"] == "reshape":
                        _check_keys(ls, ["type", "new_shape"], [])
                        self.layers.append(ReshapeLayer(prev_output, ls["new_shape"]))

                    elif ls["type"] == "fc":
                        _check_keys(ls, ["type", "n"], ["initializer"])
                        self.layers.append(
                            AffineLayer(
                                prev_output,
                                prev_output_shape,
                                output_shape=(ls["n"],),
                                initializer=_parse_initializer(ls),
                            )
                        )

                    elif ls["type"] == "nonlin":
                        _check_keys(ls, ["type", "func"], [])
                        self.layers.append(
                            NonlinearityLayer(prev_output, prev_output_shape, ls["func"])
                        )

                    else:
                        raise NotImplementedError("Unknown layer type %s" % ls["type"])

                prev_output, prev_output_shape = (
                    self.layers[-1].output,
                    self.layers[-1].output_shape,
                )
        self._output, self._output_shape = prev_output, prev_output_shape

    @property
    def varscope(self):
        return self.__varscope

    @property
    def output(self):
        return self._output

    @property
    def output_shape(self):
        return self._output_shape


class NoOpStandardizer(object):
    def __init__(self, dim, eps=1e-6):
        pass

    def update(self, points_N_D):
        pass

    def standardize_expr(self, x_B_D):
        return x_B_D

    def unstandardize_expr(self, y_B_D):
        return y_B_D

    def standardize(self, x_B_D):
        return x_B_D

    def unstandardize(self, y_B_D):
        return y_B_D


class Standardizer(Model):
    def __init__(self, dim, eps=1e-6, init_count=0, init_mean=0.0, init_meansq=1.0):
        """
        Args:
            dim: dimension of the space of points to be standardized
            eps: small constant to add to denominators to prevent division by 0
            init_count, init_mean, init_meansq: initial values for accumulators

        Note:
            if init_count is 0, then init_mean and init_meansq have no effect beyond
            the first call to update(), which will ignore their values and
            replace them with values from a new batch of data.
        """
        self._eps = eps
        self._dim = dim
        with variable_scope(type(self).__name__) as self.__varscope:
            self._count = get_variable("count", np.array(float(init_count)), trainable=False)
            self._mean_1_D = get_variable(
                "mean_1_D",
                np.full((1, self._dim), init_mean),
                broadcastable=(True, False),
                trainable=False,
            )
            self._meansq_1_D = get_variable(
                "meansq_1_D",
                np.full((1, self._dim), init_meansq),
                broadcastable=(True, False),
                trainable=False,
            )
        self._stdev_1_D = tensor.sqrt(
            tensor.nnet.relu(self._meansq_1_D - tensor.square(self._mean_1_D))
        )
        # Relu ensures inside is nonnegative. maybe the better choice would have been to
        # add self._eps inside the square root, but I'm keeping things this way to preserve
        # backwards compatibility with existing saved models.

        self.get_mean = self._mean_1_D.get_value
        self.get_stdev = theano.function([], self._stdev_1_D[0, :])  # TODO: return with shape (1,D)

    @property
    def varscope(self):
        return self.__varscope

    def update(self, points_N_D):
        assert points_N_D.ndim == 2 and points_N_D.shape[1] == self._dim
        num = points_N_D.shape[0]
        count = float(self._count.get_value())
        a = count / (count + num)
        self._mean_1_D.set_value(
            a * self._mean_1_D.get_value() + (1.0 - a) * points_N_D.mean(axis=0, keepdims=True)
        )
        self._meansq_1_D.set_value(
            a * self._meansq_1_D.get_value()
            + (1.0 - a) * (points_N_D**2).mean(axis=0, keepdims=True)
        )
        self._count.set_value(count + num)

    def standardize_expr(self, x_B_D):
        return (x_B_D - self._mean_1_D) / (self._stdev_1_D + self._eps)

    def unstandardize_expr(self, y_B_D):
        return y_B_D * (self._stdev_1_D + self._eps) + self._mean_1_D

    def standardize(self, x_B_D):
        assert x_B_D.ndim == 2
        return (x_B_D - self.get_mean()) / (self.get_stdev() + self._eps)

    def unstandardize(self, y_B_D):
        assert y_B_D.ndim == 2
        return y_B_D * (self.get_stdev() + self._eps) + self.get_mean()


def test_standardizer():
    D = 10
    s = Standardizer(D, eps=0)

    x_N_D = np.random.randn(200, D)
    s.update(x_N_D)

    x2_N_D = np.random.randn(300, D)
    s.update(x2_N_D)

    allx = np.concatenate([x_N_D, x2_N_D], axis=0)
    assert np.allclose(s._mean_1_D.get_value()[0, :], allx.mean(axis=0))
    assert np.allclose(s.get_stdev(), allx.std(axis=0))
    print("ok")


if __name__ == "__main__":
    test_standardizer()
