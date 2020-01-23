# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Configurable VTA Hareware Environment scope."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import os
import json
import copy
import tvm
from . import intrin
#from .pkg_config import PkgConfig


class DevContext(object):
    """Internal development context

    This contains all the non-user facing compiler
    internal context that is hold by the Environment.

    Parameters
    ----------
    env : Environment
        The environment hosting the DevContext

    Note
    ----
    This class is introduced so we have a clear separation
    of developer related, and user facing attributes.
    """
    # Memory id for DMA
    MEM_ID_UOP = 0
    MEM_ID_WGT = 1
    MEM_ID_INP = 2
    MEM_ID_ACC = 3
    MEM_ID_OUT = 4
    # VTA ALU Opcodes
    ALU_OPCODE_MIN = 0
    ALU_OPCODE_MAX = 1
    ALU_OPCODE_ADD = 2
    ALU_OPCODE_SHR = 3
    # Task queue id (pipeline stage)
    QID_LOAD_INP = 1
    QID_LOAD_WGT = 1
    QID_LOAD_OUT = 2
    QID_STORE_OUT = 3
    QID_COMPUTE = 2

    def __init__(self, env):
        self.vta_axis = tvm.thread_axis("nvdla")
        self.DEBUG_NO_SYNC = False
        env._dev_ctx = self
        #self.gemm = intrin.gemm(env, env.mock_mode)

    def get_task_qid(self, qid):
        """Get transformed queue index."""
        return 1 if self.DEBUG_NO_SYNC else qid


class Environment(object):
    """Hardware configuration object.

    This object contains all the information
    needed for compiling to a specific VTA backend.

    Parameters
    ----------
    cfg : dict of str to value.
        The configuration parameters.

    Example
    --------
    .. code-block:: python

      # the following code reconfigures the environment
      # temporarily to attributes specified in new_cfg.json
      new_cfg = json.load(json.load(open("new_cfg.json")))
      with vta.Environment(new_cfg):
          # env works on the new environment
          env = vta.get_env()
    """
    current = None
    # initialization function
    def __init__(self, cfg):
        # Produce the derived parameters and update dict
        self.pkg = self.pkg_config(cfg)
        self.__dict__.update(self.pkg.cfg_dict)
        # lazy cached members
        self.mock_mode = False
        self._mock_env = None
        self._dev_ctx = None
        self._last_env = None

    def __enter__(self):
        self._last_env = Environment.current
        Environment.current = self
        return self

    def __exit__(self, ptype, value, trace):
        Environment.current = self._last_env

    def pkg_config(self, cfg):
        """PkgConfig instance"""
        #curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        #proj_root = os.path.abspath(os.path.join(curr_path, "../../"))
        #return PkgConfig(cfg, proj_root)
        return None

    @property
    def cfg_dict(self):
        return self.pkg.cfg_dict

    @property
    def dev(self):
        """Developer context"""
        if self._dev_ctx is None:
            self._dev_ctx = DevContext(self)
        return self._dev_ctx

    @property
    def mock(self):
        """A mock version of the Environment

        The ALU, dma_copy and intrinsics will be
        mocked to be nop.
        """
        if self.mock_mode:
            return self
        if self._mock_env is None:
            self._mock_env = copy.copy(self)
            self._mock_env._dev_ctx = None
            self._mock_env.mock_mode = True
        return self._mock_env

    @property
    def dma_copy(self):
        """DMA copy pragma"""
        return ("dma_copy"
                if not self.mock_mode
                else "skip_dma_copy")

    @property
    def alu(self):
        """ALU pragma"""
        return ("alu"
                if not self.mock_mode
                else "skip_alu")

    @property
    def gemm(self):
        """GEMM intrinsic"""
        return self.dev.gemm

    @property
    def target(self):
        return tvm.target.nvdla()

    @property
    def target_host(self):
        """The target host"""
        return "c"

def get_env():
    """Get the current VTA Environment.

    Returns
    -------
    env : Environment
        The current environment.
    """
    return Environment.current

