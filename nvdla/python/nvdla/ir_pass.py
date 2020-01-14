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
"""Additional IR Pass for VTA"""
# pylint: disable=len-as-condition
from __future__ import absolute_import as _abs

import tvm
from topi import util


def _match_pragma(stmt, key):
    """Internal helper to match stmt to pragma stmt.

    Parameters
    ----------
    stmt : Stmt
        The AttrStmt

    key : str
        The pragma key
    """
    return ((stmt.attr_key == "pragma_" + key) or
            (stmt.attr_key == "pragma_scope" and stmt.value.value == key))


def debug_print(stmt):
    """A debug pass that print the stmt

    Parameters
    ----------
    stmt : Stmt
        The input statement

    Returns
    -------
    stmt : Stmt
        The
    """
    # pylint: disable=superfluous-parens
    print(stmt)
    return stmt
