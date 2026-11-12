"""S0 lossless adaptive index maps and bounded role-block execution buckets."""
from dataclasses import dataclass

import numpy as np

from .substrate import blocks


def index_map(grid_indices, total_positions):
    """Return compact-to-grid and grid-to-compact maps without dropping nodes."""
    indices = np.asarray(grid_indices)
    if (indices.ndim != 1 or indices.dtype != np.int32 or
            type(total_positions) is not int or total_positions < 0 or
            np.any(indices < 0) or np.any(indices >= total_positions) or
            np.any(np.diff(indices) <= 0)):
        raise ValueError('Invalid ordered adaptive grid indices')
    inverse = np.full(total_positions, -1, np.int32)
    inverse[indices] = np.arange(len(indices), dtype=np.int32)
    return indices.copy(), inverse


@dataclass(frozen=True)
class BucketFamily:
    inputs: int
    outputs: int
    max_width: int
    widths: tuple[int, ...]

    @classmethod
    def create(cls, inputs, outputs, node_capacity):
        if (any(type(v) is not int for v in (inputs, outputs, node_capacity)) or
                min(inputs, outputs) < 1 or node_capacity < inputs + outputs):
            raise ValueError('Invalid adaptive bucket capacity')
        maximum = node_capacity - inputs - outputs
        widths = [0]
        width = 4
        while width < maximum:
            widths.append(width)
            width *= 2
        if maximum:
            widths.append(maximum)
        return cls(inputs, outputs, maximum, tuple(sorted(set(widths))))

    def promote(self, node_capacity):
        """Explicitly widen the declared family while preserving existing routes."""
        if type(node_capacity) is not int or node_capacity <= self.inputs + self.outputs + self.max_width:
            raise ValueError('Capacity promotion must increase the node ceiling')
        wider = type(self).create(self.inputs, self.outputs, node_capacity)
        return type(self)(self.inputs, self.outputs, wider.max_width,
                          tuple(sorted(set(self.widths) | set(wider.widths))))

    def width(self, active):
        if type(active) is not int or active < 0:
            raise ValueError('Invalid adaptive active-node count')
        if active > self.max_width:
            raise OverflowError(f'Adaptive hidden width {active} exceeds declared node capacity {self.max_width}')
        return next(width for width in self.widths if width >= active)

    def pack(self, phenotype):
        """Pad one compact phenotype; masked slots cannot affect carried state."""
        p = phenotype.plan
        count = len(phenotype.grid_indices)
        if (p.w_in.shape != (self.inputs, count) or
                p.w_out.shape != (count, self.outputs) or
                p.w_delay.shape != (count, count)):
            raise ValueError('Adaptive phenotype has inconsistent compact dimensions')
        width = self.width(count)
        wi = np.zeros((self.inputs, width), np.float32)
        wd = np.zeros((width, width), np.float32)
        wo = np.zeros((width, self.outputs), np.float32)
        mi = np.zeros(wi.shape, bool)
        md = np.zeros(wd.shape, bool)
        mo = np.zeros(wo.shape, bool)
        wi[:, :count], wd[:count, :count], wo[:count] = p.w_in, p.w_delay, p.w_out
        mi[:, :count], md[:count, :count], mo[:count] = p.mask_in, p.mask_delay, p.mask_out
        padded = blocks(wi, wd, wo, mi, md, mo, np.pad(p.bias_hidden,
                        (0, width-count)), p.bias_output)
        node_mask = np.concatenate((np.ones(self.inputs, bool),
                                    np.arange(width) < count,
                                    np.ones(self.outputs, bool)))
        return padded._replace(node_mask=node_mask)


def route(plans):
    """Fixed-signature groups with original candidate indices for score scatter."""
    if not plans:
        raise ValueError('Cannot route an empty adaptive population')
    groups = {}
    for index, plan in enumerate(plans):
        signature = (plan.w_in.shape[0], plan.w_in.shape[1], plan.w_out.shape[1])
        if (plan.w_delay.shape != (signature[1], signature[1]) or
                plan.w_out.shape[0] != signature[1]):
            raise ValueError('Invalid adaptive bucket signature')
        groups.setdefault(signature, []).append(index)
    return tuple((signature, tuple(groups[signature])) for signature in sorted(groups))
