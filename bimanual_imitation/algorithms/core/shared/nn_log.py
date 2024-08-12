import os
import os.path
import warnings

import numpy as np
import tables

from bimanual_imitation.algorithms.core.shared import util

warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)


def _printfields(fields, sep=" | ", width=8, precision=4, print_header=True):
    names, vals, fmts = [], [], []
    for name, val, typeinfo in fields:
        names.append(name)
        if val is None:
            # display Nones as empty entries
            vals.append("")
            fmts.append("{:%ds}" % width)
        else:
            vals.append(val)
            if typeinfo is int:
                fmts.append("{:%dd}" % width)
            elif typeinfo is float:
                fmts.append("{:%d.%df}" % (width, precision))
            else:
                raise NotImplementedError(typeinfo)
    if print_header:
        header = ((("{:^%d}" % width) + sep) * len(names))[: -len(sep)].format(*names)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
    print(sep.join(fmts).format(*vals))


def _type_to_col(t, pos):
    if t is int:
        return tables.Int32Col(pos=pos)
    if t is float:
        return tables.Float32Col(pos=pos)
    raise NotImplementedError(t)


class TrainingLog(object):
    """A training log backed by PyTables. Stores diagnostic numbers over time and model snapshots."""

    def __init__(self, filename, attrs, load_log=False, log_last_idx=None):
        if filename is None:
            print("[TrainingLog] Not writing log to any file")
            self.f = None
        else:
            if os.path.exists(filename):
                # raise RuntimeError('Log file %s already exists' % filename)
                self.f = tables.open_file(filename, mode="a")
                print("Log file %s already exists" % filename)
                if load_log:
                    self.log_table = self.f.get_node("/log")
                    if log_last_idx is not None:
                        self.log_table.remove_rows(log_last_idx + 1, np.iinfo(np.int64).max)
                else:
                    self.log_table = None
            else:
                self.f = tables.open_file(filename, mode="w")
                for k, v in attrs:
                    self.f.root._v_attrs[k] = v
                self.log_table = None

        self.schema = None  # list of col name / types for display

    def remove_greater_snapshots(self, snapshot_idx):
        snapshot_names = self.f.root.snapshots._v_children.keys()
        assert all(name.startswith("iter") for name in snapshot_names)
        all_snapshot_idxs = np.asarray(
            sorted([int(name[len("iter") :]) for name in snapshot_names])
        )
        greater_snap_idxs = [x for x in all_snapshot_idxs if x > snapshot_idx]
        for idx in greater_snap_idxs:
            self.f.remove_node("/snapshots/iter{:07d}".format(idx), recursive=True)

        if len(greater_snap_idxs) == 0:
            print(f"Warning: no greater snapshot idxs than idx {snapshot_idx}!")

    def close(self):
        if self.f is not None:
            self.f.close()

    def write(self, kvt):
        # Write to the log
        if self.f is not None:
            if self.log_table is None:
                desc = {k: _type_to_col(t, pos) for pos, (k, _, t) in enumerate(kvt)}
                self.log_table = self.f.create_table(self.f.root, "log", desc)

            row = self.log_table.row
            for k, v, _ in kvt:
                row[k] = v
            row.append()

            self.log_table.flush()

    def print(self, kvt, **kwargs):
        if self.schema is None:
            self.schema = [(k, t) for k, _, t in kvt]
        else:
            # If we are missing columns, fill them in with Nones
            nonefilled_kvt = []
            kvt_dict = {k: (v, t) for k, v, t in kvt}
            for schema_k, schema_t in self.schema:
                if schema_k in kvt_dict:
                    v, t = kvt_dict[schema_k]
                    nonefilled_kvt.append((schema_k, v, t))  # check t == schema_t too?
                else:
                    nonefilled_kvt.append((schema_k, None, schema_t))
            kvt = nonefilled_kvt
        _printfields(kvt, **kwargs)

    def write_snapshot(self, model, key_iter):
        if self.f is None:
            return

        # Save all variables into this group
        snapshot_root = "/snapshots/iter%07d" % key_iter

        for v in model.get_all_variables():
            assert v.name[0] == "/"
            fullpath = snapshot_root + v.name
            groupname, arrayname = fullpath.rsplit("/", 1)
            self.f.create_array(groupname, arrayname, v.get_value(), createparents=True)

        # Store the model hash as an attribute
        self.f.get_node(snapshot_root)._v_attrs.hash = model.savehash()

        self.f.flush()
