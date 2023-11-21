import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional
import nets
from nets import model_entrypoint

from torch_geometric.data import Data
import ase.io 
import os.path as osp
try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    pass

import matplotlib.pyplot as plt
import pickle as pk
import glob
def get_args_parser():
    parser = argparse.ArgumentParser('Inference equivariant networks on HEA', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument("--data-path", type=str, default='datasets/examples/test')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--elements', type=str, default=0)
    return parser

def _get_neighbors_pymatgen(atoms,radius,max_neigh):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=radius, numerical_tol=0, exclude_self=True
        )
        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)
        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]
        return _c_index, _n_index, n_distance, _offsets

def _get_specific_neighbors_pymatgen(atoms,target_atom,radius,max_neigh):
    """Preforms nearest neighbor search and returns edge index, distances,
    and cell offsets"""
    struct = AseAtomsAdaptor.get_structure(atoms)
    index = -1
    from pymatgen.core.structure import Element
    for i in range(len(struct.sites)):
        if struct.sites[i].specie == Element(target_atom):
            index = i
    neighbors = struct.get_neighbors(struct.sites[index],r=radius)
    neighbors=[i for i in neighbors if str(i.specie) not in ['O','Cl']]
    max_distance = sorted(neighbors,key=lambda x:x.nn_distance)
    max_distance = max_distance[max_neigh-1].nn_distance
    max_dist_index = np.where(np.array([i.nn_distance for i in neighbors]) <= max_distance+0.001)[0]
    neighbors=[str(neighbors[i].specie) for i in max_dist_index]
    return neighbors

def _reshape_features(c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]
        return edge_index, edge_distances, cell_offsets

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    max_neigh = 6
    
    ''' Network '''
    basic_dir = args.model_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list = []
    for folder in os.listdir(basic_dir):  
        max_epoch = -1  
        max_filename = ""  
        for filename in os.listdir(basic_dir + "/"+folder):  
            epoch = int(filename.split('@')[-1].split('.')[0])  
            if epoch > max_epoch:  
                max_epoch = epoch  
                max_filename = filename  
        print(str(os.path.join(basic_dir + "/"+folder,max_filename)))
        model = torch.load(str(os.path.join(basic_dir + "/"+folder,max_filename)),map_location='cpu')
        model = model.to(device)
        model_list.append(model)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ''' Dataset '''
    data_list = []
    data_source = os.listdir(args.data_path)
    for i in range(len(data_source)):
        cif = osp.join(args.data_path,data_source[i])
        atoms = ase.io.read(cif)
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
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
        data = Data(
            id=data_source[i],
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
        )
        split_idx_dist = _get_neighbors_pymatgen(atoms,args.radius,max_neigh)
        edge_index, edge_distances, cell_offsets = _reshape_features(
            *split_idx_dist
        )
        specific_neighbor_list = _get_specific_neighbors_pymatgen(atoms,data_source[i].split('_')[0],args.radius,6)
        data.specific_neighbor_list = specific_neighbor_list
        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.pbc = torch.tensor(atoms.pbc)
        data_list.append(data)
    ''' Data Loader '''
    test_loader = DataLoader(data_list, batch_size=args.eval_batch_size)
    prediction_list = []
    id_list = []
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            id_list.extend(data.id)
            prediction = []
            for index in range(len(model_list)):
                model_list[index].eval()
                data = data.to(device)
                pred_y, pred_dy = model_list[index](data)
                pred_y = pred_y.flatten().cpu().numpy().tolist()
                prediction.append(pred_y)
            prediction_list.extend(np.mean(np.array(prediction),axis=0))
    content = "id,predictions\n"
    for id,predictions in zip(id_list,prediction_list):
        content += id + "," + str(predictions) + "\n"
    with open(args.output_dir + "/predict.csv",'a') as f:
        f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inference equivariant networks on HEA', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args=args)
    
    
