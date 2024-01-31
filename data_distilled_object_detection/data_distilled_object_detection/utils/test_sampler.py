import torch
import itertools
from detectron2.utils import comm

class TestSampler():
    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), comm.get_rank(), None, comm.get_world_size())
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(121)
        while True:
            to_yield = []
            perm = torch.randperm(56, generator=g).tolist()
            print(f"List1: {perm[:6]}")
            to_yield += perm[:6]
            perm2 = torch.randperm(20, generator=g).tolist()
            perm2 = [x + 56 for x in perm2]
            to_yield += perm2[:4]
            print(f"List2: {perm2[:4]}")
            yield from to_yield

#data_loader = iter(TestSampler())
#check = set()
#for i in range(6):
#    val = next(data_loader)
#    if val in check:
#        print(f"Iter {i} -> {val} is already sampled!")
#    print(f"Val: {val}")
#    check.add(val)
