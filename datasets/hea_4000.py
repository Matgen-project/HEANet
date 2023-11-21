from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs
import lmdb
import ase.io 
import pickle
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    pass

import os.path as osp
from typing import List,Callable

from jarvis.core.specie import get_node_attributes
import itertools
class Hea4000(InMemoryDataset):

    def __init__(self, root, split, kfold=10, target="energy",max_neigh=9,radius=6,feature_type="crystalnet",fixed_size_split=True):
        assert split in ["train", "valid", "test"]
        self.split = split
        self.kfold = kfold
        self.root = osp.abspath(root)
        self.feature_type = feature_type
        self.target = target
        self.max_neigh=max_neigh
        self.radius=radius
        self.fixed_size_split = fixed_size_split
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def calc_stats(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        y = y[:, target]
        mean = float(torch.mean(y))
        mad = float(torch.mean(torch.abs(y - mean))) #median absolute deviation
        return mean, mad

    def mean(self, target: int) -> float:
        y = torch.tensor(torch.cat([self.get(i).y for i in range(len(self))], dim=0),dtype=torch.float32)
        return float(y.mean())

    def std(self, target: int) -> float:
        y = torch.tensor(torch.cat([self.get(i).y for i in range(len(self))], dim=0),dtype=torch.float32)
        return float(y.std())

    @property
    def processed_file_names(self) -> str:
        # print("_".join([self.split, self.feature_type]) + '.pt')
        return "_".join([self.split, str(self.kfold)]) + '.pt'   
    
    def _reshape_features(self,c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets
    
    # run internal functions to get padded indices and distances
    def _get_neighbors_pymatgen(self,atoms,target_atom, radius,max_neigh):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=radius, numerical_tol=0, exclude_self=True
        )
        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # idx_i = ((_c_index == i) & (_n_index != 8) & (_n_index != 17)).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)
        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]
        return _c_index, _n_index, n_distance, _offsets
    
    def process(self):
        import os
        energy_dat = osp.join(self.root,'gibbs_eng.csv')
        data_source = [i[:-1].split(',') for i in open(energy_dat,'r').readlines()]
        energy_dict={}
        for i in data_source:
            if float(i[1]) < 0:
                energy_dict[i[0]] = float(i[1])
        data_source = list(energy_dict.keys())
        categories = {}
        for i in data_source:
            if categories.get(i.split('_')[0]) == None:
                categories[i.split('_')[0]] = [i]
            else:
                categories[i.split('_')[0]].append(i)
        # s = "Co_AB"
        train=[]
        valid=[]
        test=[]
        folds = []
        category = list(categories.keys())
        for i in category:
            N_mat = len(categories[i])
            data_perm = np.random.default_rng(123).permutation(N_mat) 
            folds.append(np.array_split(data_perm, self.kfold))
        for j in range(self.kfold):
            train_item=[]
            valid_item=[]
            test_item=[]
            for i in range(len(category)):
                if j == 2:
                    train_idx = folds[i][j+1:]
                elif j == 1:
                    train_idx = folds[i][j+1:-1]
                elif j == 0:
                    train_idx = folds[i][j+1:-2]
                else:
                    train_idx = folds[i][j+1:] + folds[i][0:j-2]
                for idx in list(itertools.chain(*train_idx)):
                    train_item.append(categories[category[i]][idx])
                for idx in folds[i][j-1]:
                    valid_item.append(categories[category[i]][idx])
                for idx in folds[i][j-2]:
                    valid_item.append(categories[category[i]][idx])
                for idx in folds[i][j]:
                    test_item.append(categories[category[i]][idx])
            test.append(test_item)
            train.append(train_item)
            valid.append(valid_item)
        indices = {"train": train, "valid": valid, "test": test}
        # print(indices)
        lengths = {"train":torch.tensor([len(i) for i in train]),"valid":torch.tensor([len(i) for i in valid]),"test":torch.tensor([len(i) for i in test])}
        torch.save(lengths,osp.join(self.root,"lengths.pt"))

        fold_list = []
        # fold = 0
        # if True:
        for fold in range(self.kfold):
            j = 0
            # i = 0
            # if True:
            for i in range(len(data_source)):
                if data_source[j] not in indices[self.split][fold]:
                    j += 1
                    continue
                j+=1
                for structure_type in ['initial','relaxed']:
                    cif = osp.join(self.root,osp.join(structure_type+"_p",osp.join(data_source[i].split('_')[0],data_source[i]+".cif")))
                    atoms = ase.io.read(cif)
                    atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
                    target_atom = data_source[i].split('_')[0]
                    positions = torch.Tensor(atoms.get_positions())
                    cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
                    natoms = positions.shape[0]
                    tags=[]
                    for atomic_number in atomic_numbers:
                        if atomic_number == 17:
                            tags.append(2)
                        else:
                            tags.append(1)
                    tags = torch.Tensor(tags)
                    # tags = torch.Tensor(atoms.get_tags())

                    data = Data(
                        cell=cell,
                        pos=positions,
                        atomic_numbers=atomic_numbers,
                        natoms=natoms,
                        tags=tags,
                        # x=torch.tensor(atom_features).type(torch.get_default_dtype()),
                    )
                    split_idx_dist = self._get_neighbors_pymatgen(atoms,target_atom,self.radius,self.max_neigh)
                    edge_index, edge_distances, cell_offsets = self._reshape_features(
                        *split_idx_dist
                    )
                    data.edge_index = edge_index
                    data.cell_offsets = cell_offsets
                    data.y = energy_dict[data_source[i]]
                    forces = torch.zeros([natoms,3],dtype=torch.float)
                    data.force = forces
                    data.pbc = torch.tensor(atoms.pbc)
                    # print(data)
                    # import sys
                    # sys.exit()
                    fold_list.append(data)
        torch.save(self.collate(fold_list), self.processed_paths[0])
