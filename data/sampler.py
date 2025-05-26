from torch.utils import data
import math
import numpy as np
import random
from typing import Optional, List
import logging


def StandardSampler(dataset: data.Dataset,
                    shuffle: bool,
                    distributed: bool = False,
                    drop_last: bool = True,
                    world_size: Optional[int] = None,
                    rank: Optional[int] = None) -> data.Sampler:
    if distributed:
        assert rank is not None and world_size is not None
        return data.distributed.DistributedSampler(dataset,
                                                   shuffle=shuffle,
                                                   num_replicas=world_size,
                                                   drop_last=drop_last,
                                                   rank=rank)
    if shuffle:
        return data.RandomSampler(dataset)
    return data.SequentialSampler(dataset)


def RandomBucketSampler(nbuckets: int,
                        length: List[float],
                        batch_size: Optional[int] = None,
                        batch_length: Optional[float] = None,
                        drop_last: bool = True,
                        distributed: bool = False,
                        world_size: Optional[int] = None,
                        rank: Optional[int] = None) -> data.Sampler:
    if batch_size is None:
        assert batch_length is not None
    if batch_length is None:
        assert batch_size is not None
    assert batch_size is None or batch_length is None
    if distributed:
        assert rank is not None and world_size is not None
        return DistributedRandomBucketSampler(nbuckets, length,
                                              world_size, rank,
                                              batch_size, batch_length,
                                              drop_last)
    return SingleRandomBucketSampler(nbuckets, length,
                                     batch_size, batch_length, drop_last)


class SingleRandomBucketSampler(data.Sampler):
    def __init__(self,
                 nbuckets: int,
                 length: List[float],
                 batch_size: Optional[int] = None,
                 batch_length: Optional[float] = None,
                 drop_last: bool = True) -> None:
        self.length = length
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.drop_last = drop_last
        indices = np.argsort([-x for x in length])
        split = len(indices) // nbuckets
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])

    def __iter__(self):
        random.shuffle(self.indices)
        for x in self.indices:
            random.shuffle(x)
        idxs = [i for x in self.indices for i in x]
        batches, batch, sum_len, max_len = [], [], 0, 0
        for idx in idxs:
            batch.append(idx)
            sum_len += self.length[idx]
            max_len = max(self.length[idx], max_len)
            if self.batch_size is not None:
                if len(batch) >= self.batch_size:
                    batches.append(batch)
                    batch, sum_len, max_len = [], 0, 0
            else:
                if (max_len * len(batch) > self.batch_length) and batch[:-1]:
                    batches.append(batch[:-1])
                    batch = [batch[-1]]
                    sum_len, max_len = self.length[idx], self.length[idx]
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)


class DistributedRandomBucketSampler(data.Sampler):
    def __init__(self,
                 nbuckets: int,
                 length: List[float],
                 num_replicas: int,
                 rank: int,
                 batch_size: Optional[int] = None,
                 batch_length: Optional[float] = None,
                 drop_last: bool = True,
                 seed: int = 1234) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.length = length
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        # Deterministic shuffling
        random.Random(self.epoch + self.seed).shuffle(self.indices)
        for i, x in enumerate(self.indices):
            seed = self.epoch + self.seed + i * 5
            random.Random(seed).shuffle(x)
        indices = [i for x in self.indices for i in x]

        # Batching
        batches, batch, sum_len, max_len = [], [], 0, 0
        for idx in indices:
            batch.append(idx)
            sum_len += self.length[idx]
            max_len = max(self.length[idx], max_len)
            if self.batch_size is not None:
                if len(batch) >= self.batch_size:
                    batches.append(batch)
                    batch, sum_len, max_len = [], 0, 0
            else:
                if (max_len * len(batch) > self.batch_length) and batch[:-1]:
                    logging.debug(f"Rank {self.rank}: Effective length of a batch: {sum_len}")
                    logging.debug(f"Rank {self.rank}: Max length of a batch: {max_len}")
                    logging.debug(f"Rank {self.rank}: Total length of a batch: "
                                  f"{max_len * len(batch)}")
                    batches.append(batch[:-1])
                    batch = [batch[-1]]
                    sum_len, max_len = self.length[idx], self.length[idx]
        assert all([len(b) > 0 for b in batches])
        # Subsample
        num_samples = math.ceil((len(batches) - self.num_replicas) /
                                self.num_replicas)
        total_size = num_samples * self.num_replicas
        batches = batches[:total_size]
        batches = batches[self.rank*num_samples: (self.rank+1)*num_samples]
        assert len(batches) == num_samples

        # Stochastic suffling
        random.shuffle(batches)
        return iter(batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def ConcatLengthSampler(batch_size: int,
                        max_length: int,
                        length: List[float],
                        text_max_length: Optional[int] = None,
                        distributed: bool = False,
                        world_size: Optional[int] = None,
                        rank: Optional[int] = None) -> data.Sampler:
    if distributed:
        assert rank is not None and world_size is not None
        return DistributedConcatLengthSampler(batch_size=batch_size,
                                              max_length=max_length,
                                              length=length,
                                              distributed=distributed,
                                              world_size=world_size,
                                              rank=rank)

    return SingleConcatLengthSampler(batch_size=batch_size,
                                     max_length=max_length,
                                     length=length)


class SingleConcatLengthSampler(data.Sampler):
    def __init__(self,
                 batch_size: int,
                 max_length: int,
                 length: List[float]) -> None:
        self.length = length
        self.total_length = batch_size * max_length
        self.indices = list(range(len(length)))

    def __iter__(self):
        random.shuffle(self.indices)
        batches, batch, sum_len = [], [], 0
        for idx in self.indices:
            batch.append(idx)
            sum_len += self.length[idx]
            if (sum_len >= self.total_length):
                batches.append(batch)
                batch, sum_len = [], 0
        random.shuffle(batches)
        return iter(batches)


class DistributedConcatLengthSampler(data.Sampler):
    def __init__(self,
                 batch_size: int,
                 max_length: int,
                 length: List[float],
                 num_replicas: int,
                 rank: int,
                 seed: int = 1234) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.length = length
        self.total_length = batch_size * max_length
        self.indices = list(range(len(length)))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        # Deterministic shuffling
        random.Random(self.epoch + self.seed).shuffle(self.indices)
        # Batching
        batches, batch, sum_len = [], [], 0
        for idx in self.indices:
            batch.append(idx)
            sum_len += self.length[idx]
            if (sum_len >= self.total_length):
                logging.debug(f"Rank {self.rank}: Effective length of a batch: {sum_len}")
                batches.append(batch)
                batch, sum_len = [], 0
        # Subsample
        num_samples = math.ceil((len(batches) - self.num_replicas) /
                                self.num_replicas)
        total_size = num_samples * self.num_replicas
        batches = batches[:total_size]
        batches = batches[self.rank*num_samples: (self.rank+1)*num_samples]
        assert len(batches) == num_samples
        # Stochastic suffling
        random.shuffle(batches)
        return iter(batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
