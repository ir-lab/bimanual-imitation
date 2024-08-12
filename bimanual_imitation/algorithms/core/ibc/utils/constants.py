# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from string import Template

"""Defines various constants shared across ibc."""
IMG_KEYS = ["rgb", "front", "image"]


SPEC_STRING = Template(
    """
named_tuple_value {
  name: "Trajectory"
  values {
    key: "step_type"
    value {
      tensor_spec_value {
        name: "step_type"
        shape {
        }
        dtype: DT_INT32
      }
    }
  }
  values {
    key: "observation"
    value {
      bounded_tensor_spec_value {
        name: "observation"
        shape {
          dim {
            size: ${full_obs_size}
          }
        }
        dtype: DT_FLOAT
        minimum {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: -10
        }
        maximum {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 10
        }
      }
    }
  }
  values {
    key: "action"
    value {
      bounded_tensor_spec_value {
        name: "action"
        shape {
          dim {
            size: ${full_action_size}
          }
        }
        dtype: DT_FLOAT
        minimum {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: -10
        }
        maximum {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 10
        }
      }
    }
  }
  values {
    key: "policy_info"
    value {
      tuple_value {
      }
    }
  }
  values {
    key: "next_step_type"
    value {
      tensor_spec_value {
        name: "step_type"
        shape {
        }
        dtype: DT_INT32
      }
    }
  }
  values {
    key: "reward"
    value {
      tensor_spec_value {
        name: "reward"
        shape {
        }
        dtype: DT_FLOAT
      }
    }
  }
  values {
    key: "discount"
    value {
      bounded_tensor_spec_value {
        name: "discount"
        shape {
        }
        dtype: DT_FLOAT
        minimum {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 0.0
        }
        maximum {
          dtype: DT_FLOAT
          tensor_shape {
          }
          float_val: 1.0
        }
      }
    }
  }
}
"""
)
