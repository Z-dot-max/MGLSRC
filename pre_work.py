import torch
import numpy as np
import ast
import json
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from rdkit import Chem
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree

with open('data/catalyst.json', 'r', encoding='utf-8') as f:
    catalyst_to_id = json.load(f)

with open('data/solvent.json', 'r', encoding='utf-8') as f:
    solvent_to_id = json.load(f)

with open('data/reagent.json', 'r', encoding='utf-8') as f:
    reagent_to_id = json.load(f)

def pad_rectangular_matrices(tensor_list, padding_value):
    max_rows = max(tensor.size(0) for tensor in tensor_list)
    max_cols = max(tensor.size(1) for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        rows, cols = tensor.size()
        padding_bottom = max_rows - rows
        padding_right = max_cols - cols

        padded_tensor = F.pad(tensor, (0, padding_right, 0, padding_bottom), value=padding_value)
        padded_tensors.append(padded_tensor)

    return pad_sequence(padded_tensors, batch_first=True)

class Graph_Bert_Dataset_fine_tune(object):
    def __init__(self, original_dataset, tokenizer, map_dict,
                 label_fields=['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']):
        """
        Dataset constructor for handling variable number of molecules.
        """
        self.original_dataset = original_dataset
        self.tokenizer = tokenizer
        self.map_dict = map_dict
        self.label_fields = label_fields
        self.pad_value = self.tokenizer.vocab['<pad>']

        for field in label_fields:
            if field == 'catalyst1':
                self.original_dataset[field] = self.original_dataset[field].apply(lambda x: catalyst_to_id.get(x, 0))
            elif field == 'reagent1':
                self.original_dataset[field] = self.original_dataset[field].apply(lambda x: reagent_to_id.get(x, 0))
            elif field == 'reagent2':
                self.original_dataset[field] = self.original_dataset[field].apply(lambda x: reagent_to_id.get(x, 0))
            elif field == 'solvent1':
                self.original_dataset[field] = self.original_dataset[field].apply(lambda x: solvent_to_id.get(x, 0))
            elif field == 'solvent2':
                self.original_dataset[field] = self.original_dataset[field].apply(lambda x: solvent_to_id.get(x, 0))

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        smiles_list = []

        reactants = self.original_dataset.iloc[idx]['Reactants']
        if isinstance(reactants, str):
            reactants = ast.literal_eval(reactants)
        smiles_list.extend(reactants)

        products = self.original_dataset.iloc[idx]['Products']
        if isinstance(products, str):
            products = ast.literal_eval(products)
        smiles_list.extend(products)

        labels = [self.original_dataset.iloc[idx][field] for field in self.label_fields]

        return self.numerical_seq(smiles_list, labels)

    def numerical_seq(self, smiles_list, labels):
        if len(smiles_list) == 2:
            reactant1 = self.map_dict[smiles_list[0]]
            nums_list1 = reactant1['nums_list']
            dist_matrix1 = reactant1['dist_matrix']
            adjoin_matrix1 = reactant1['adj_matrix']
            single_dict_atom1 = reactant1['single_dict']

            nums_list1 = [self.tokenizer.vocab['<global>']] + nums_list1
            temp = np.ones((len(nums_list1), len(nums_list1)))
            temp[1:, 1:] = adjoin_matrix1
            adjoin_matrix1 = (1 - temp) * (-1e9)
            temp_dist = np.ones((len(nums_list1), len(nums_list1)))
            temp_dist[0][0] = 0
            temp_dist[1:, 1:] = dist_matrix1
            dist_matrix1 = temp_dist

            atom_features1 = single_dict_atom1['input_atom_features']
            dist_matrix_atom1 = single_dict_atom1['dist_matrix']
            adjoin_matrix_atom1 = single_dict_atom1['adj_matrix']
            adjoin_matrix_atom1 = (1 - adjoin_matrix_atom1) * (-1e9)
            atom_match_matrix1 = single_dict_atom1['atom_match_matrix']
            sum_atoms1 = single_dict_atom1['sum_atoms']

            nums_list2 = torch.empty(0)
            adjoin_matrix2 = torch.empty(0, 0)
            dist_matrix2 = torch.empty(0, 0)

            atom_features2 = torch.empty(0, 0)
            dist_matrix_atom2 = torch.empty(0, 0)
            adjoin_matrix_atom2 = torch.empty(0, 0)
            atom_match_matrix2 = torch.empty(0, 0)
            sum_atoms2 = torch.empty(0, 0)

            product1 = self.map_dict[smiles_list[1]]
            nums_list3 = product1['nums_list']
            dist_matrix3 = product1['dist_matrix']
            adjoin_matrix3 = product1['adj_matrix']
            single_dict_atom3 = product1['single_dict']

            nums_list3 = [self.tokenizer.vocab['<global>']] + nums_list3
            temp = np.ones((len(nums_list3), len(nums_list3)))
            temp[1:, 1:] = adjoin_matrix3
            adjoin_matrix3 = (1 - temp) * (-1e9)
            temp_dist = np.ones((len(nums_list3), len(nums_list3)))
            temp_dist[0][0] = 0
            temp_dist[1:, 1:] = dist_matrix3
            dist_matrix3 = temp_dist

            atom_features3 = single_dict_atom3['input_atom_features']
            dist_matrix_atom3 = single_dict_atom3['dist_matrix']
            adjoin_matrix_atom3 = single_dict_atom3['adj_matrix']
            adjoin_matrix_atom3 = (1 - adjoin_matrix_atom3) * (-1e9)
            atom_match_matrix3 = single_dict_atom3['atom_match_matrix']
            sum_atoms3 = single_dict_atom3['sum_atoms']

        else:
            reactant1 = self.map_dict[smiles_list[0]]
            nums_list1 = reactant1['nums_list']
            dist_matrix1 = reactant1['dist_matrix']
            adjoin_matrix1 = reactant1['adj_matrix']
            single_dict_atom1 = reactant1['single_dict']

            nums_list1 = [self.tokenizer.vocab['<global>']] + nums_list1
            temp = np.ones((len(nums_list1), len(nums_list1)))
            temp[1:, 1:] = adjoin_matrix1
            adjoin_matrix1 = (1 - temp) * (-1e9)
            temp_dist = np.ones((len(nums_list1), len(nums_list1)))
            temp_dist[0][0] = 0
            temp_dist[1:, 1:] = dist_matrix1
            dist_matrix1 = temp_dist

            atom_features1 = single_dict_atom1['input_atom_features']
            dist_matrix_atom1 = single_dict_atom1['dist_matrix']
            adjoin_matrix_atom1 = single_dict_atom1['adj_matrix']
            adjoin_matrix_atom1 = (1 - adjoin_matrix_atom1) * (-1e9)
            atom_match_matrix1 = single_dict_atom1['atom_match_matrix']
            sum_atoms1 = single_dict_atom1['sum_atoms']

            reactant2 = self.map_dict[smiles_list[1]]
            nums_list2 = reactant2['nums_list']
            dist_matrix2 = reactant2['dist_matrix']
            adjoin_matrix2 = reactant2['adj_matrix']
            single_dict_atom2 = reactant2['single_dict']

            nums_list2 = [self.tokenizer.vocab['<global>']] + nums_list2
            temp = np.ones((len(nums_list2), len(nums_list2)))
            temp[1:, 1:] = adjoin_matrix2
            adjoin_matrix2 = (1 - temp) * (-1e9)
            temp_dist = np.ones((len(nums_list2), len(nums_list2)))
            temp_dist[0][0] = 0
            temp_dist[1:, 1:] = dist_matrix2
            dist_matrix2 = temp_dist

            atom_features2 = single_dict_atom2['input_atom_features']
            dist_matrix_atom2 = single_dict_atom2['dist_matrix']
            adjoin_matrix_atom2 = single_dict_atom2['adj_matrix']
            adjoin_matrix_atom2 = (1 - adjoin_matrix_atom2) * (-1e9)
            atom_match_matrix2 = single_dict_atom2['atom_match_matrix']
            sum_atoms2 = single_dict_atom2['sum_atoms']

            nums_list2 = torch.tensor(nums_list2).clone().detach()
            adjoin_matrix2 = torch.tensor(adjoin_matrix2).clone().detach()
            dist_matrix2 = torch.tensor(dist_matrix2).clone().detach()
            atom_features2 = torch.tensor(atom_features2).clone().detach()
            adjoin_matrix_atom2 = torch.tensor(adjoin_matrix_atom2).clone().detach()
            dist_matrix_atom2 = torch.tensor(dist_matrix_atom2).clone().detach()
            atom_match_matrix2 = torch.tensor(atom_match_matrix2).clone().detach()
            sum_atoms2 = torch.tensor(sum_atoms2).clone().detach()

            product1 = self.map_dict[smiles_list[2]]
            nums_list3 = product1['nums_list']
            dist_matrix3 = product1['dist_matrix']
            adjoin_matrix3 = product1['adj_matrix']
            single_dict_atom3 = product1['single_dict']

            nums_list3 = [self.tokenizer.vocab['<global>']] + nums_list3
            temp = np.ones((len(nums_list3), len(nums_list3)))
            temp[1:, 1:] = adjoin_matrix3
            adjoin_matrix3 = (1 - temp) * (-1e9)
            temp_dist = np.ones((len(nums_list3), len(nums_list3)))
            temp_dist[0][0] = 0
            temp_dist[1:, 1:] = dist_matrix3
            dist_matrix3 = temp_dist

            atom_features3 = single_dict_atom3['input_atom_features']
            dist_matrix_atom3 = single_dict_atom3['dist_matrix']
            adjoin_matrix_atom3 = single_dict_atom3['adj_matrix']
            adjoin_matrix_atom3 = (1 - adjoin_matrix_atom3) * (-1e9)
            atom_match_matrix3 = single_dict_atom3['atom_match_matrix']
            sum_atoms3 = single_dict_atom3['sum_atoms']

        nums_list1 = torch.tensor(nums_list1).clone().detach()
        adjoin_matrix1 = torch.tensor(adjoin_matrix1).clone().detach()
        dist_matrix1 = torch.tensor(dist_matrix1).clone().detach()
        atom_features1 = torch.tensor(atom_features1).clone().detach()
        adjoin_matrix_atom1 = torch.tensor(adjoin_matrix_atom1).clone().detach()
        dist_matrix_atom1 = torch.tensor(dist_matrix_atom1).clone().detach()
        atom_match_matrix1 = torch.tensor(atom_match_matrix1).clone().detach()
        sum_atoms1 = torch.tensor(sum_atoms1).clone().detach()

        nums_list3 = torch.tensor(nums_list3).clone().detach()
        adjoin_matrix3 = torch.tensor(adjoin_matrix3).clone().detach()
        dist_matrix3 = torch.tensor(dist_matrix3).clone().detach()
        atom_features3 = torch.tensor(atom_features3).clone().detach()
        adjoin_matrix_atom3 = torch.tensor(adjoin_matrix_atom3).clone().detach()
        dist_matrix_atom3 = torch.tensor(dist_matrix_atom3).clone().detach()
        atom_match_matrix3 = torch.tensor(atom_match_matrix3).clone().detach()
        sum_atoms3 = torch.tensor(sum_atoms3).clone().detach()

        y = torch.tensor(labels).clone().detach()
        return nums_list1, adjoin_matrix1, dist_matrix1, atom_features1, \
            adjoin_matrix_atom1, dist_matrix_atom1, atom_match_matrix1, sum_atoms1, \
            nums_list2, adjoin_matrix2, dist_matrix2, atom_features2, \
            adjoin_matrix_atom2, dist_matrix_atom2, atom_match_matrix2, sum_atoms2, \
            nums_list3, adjoin_matrix3, dist_matrix3, atom_features3, \
            adjoin_matrix_atom3, dist_matrix_atom3, atom_match_matrix3, sum_atoms3, y

def custom_collate_fn(batch):
    nums_list1, adjoin_matrix1, dist_matrix1, atom_features1, \
        adjoin_matrix_atom1, dist_matrix_atom1, atom_match_matrix1, sum_atoms1, \
        nums_list2, adjoin_matrix2, dist_matrix2, atom_features2, \
        adjoin_matrix_atom2, dist_matrix_atom2, atom_match_matrix2, sum_atoms2, \
        nums_list3, adjoin_matrix3, dist_matrix3, atom_features3, \
        adjoin_matrix_atom3, dist_matrix_atom3, atom_match_matrix3, sum_atoms3, y = zip(*batch)

    nums_list1 = pad_sequence(nums_list1, batch_first=True, padding_value=0)
    nums_list2 = pad_sequence(nums_list2, batch_first=True, padding_value=0)
    nums_list3 = pad_sequence(nums_list3, batch_first=True, padding_value=0)

    adjoin_matrix1 = pad_rectangular_matrices(adjoin_matrix1, padding_value=0)
    adjoin_matrix2 = pad_rectangular_matrices(adjoin_matrix2, padding_value=0)
    adjoin_matrix3 = pad_rectangular_matrices(adjoin_matrix3, padding_value=0)

    dist_matrix1 = pad_rectangular_matrices(dist_matrix1, padding_value=-1e9)
    dist_matrix2 = pad_rectangular_matrices(dist_matrix2, padding_value=-1e9)
    dist_matrix3 = pad_rectangular_matrices(dist_matrix3, padding_value=-1e9)

    atom_features1 = pad_rectangular_matrices(atom_features1, padding_value=0)
    atom_features2 = pad_rectangular_matrices(atom_features2, padding_value=0)
    atom_features3 = pad_rectangular_matrices(atom_features3, padding_value=0)

    adj_matrix_atom1 = pad_rectangular_matrices(adjoin_matrix_atom1, padding_value=0)
    adj_matrix_atom2 = pad_rectangular_matrices(adjoin_matrix_atom2, padding_value=0)
    adj_matrix_atom3 = pad_rectangular_matrices(adjoin_matrix_atom3, padding_value=0)

    dist_matrix_atom1 = pad_rectangular_matrices(dist_matrix_atom1, padding_value=-1e9)
    dist_matrix_atom2 = pad_rectangular_matrices(dist_matrix_atom2, padding_value=-1e9)
    dist_matrix_atom3 = pad_rectangular_matrices(dist_matrix_atom3, padding_value=-1e9)

    atom_match_matrix1 = pad_rectangular_matrices(atom_match_matrix1, padding_value=0)
    atom_match_matrix2 = pad_rectangular_matrices(atom_match_matrix2, padding_value=0)
    atom_match_matrix3 = pad_rectangular_matrices(atom_match_matrix3, padding_value=0)

    sum_atoms1 = pad_rectangular_matrices(sum_atoms1, padding_value=1)
    sum_atoms2 = pad_rectangular_matrices(sum_atoms2, padding_value=1)
    sum_atoms3 = pad_rectangular_matrices(sum_atoms3, padding_value=1)

    y = torch.stack(y, dim=0)

    return nums_list1, nums_list2, nums_list3, adjoin_matrix1, adjoin_matrix2, adjoin_matrix3, \
        dist_matrix1, dist_matrix2, dist_matrix3, atom_features1, atom_features2, atom_features3,\
        adj_matrix_atom1, adj_matrix_atom2, adj_matrix_atom3, dist_matrix_atom1, dist_matrix_atom2, dist_matrix_atom3,\
        atom_match_matrix1, atom_match_matrix2, atom_match_matrix3, sum_atoms1, sum_atoms2, sum_atoms3, y

class Mol_Tokenizer:
    def __init__(self, tokens_id_file):
        self.vocab = json.load(open(tokens_id_file, 'r'))
        self.MST_MAX_WEIGHT = 100
        self.get_vocab_size = len(self.vocab.keys())
        self.id_to_token = {value: key for key, value in self.vocab.items()}

    def tokenize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        ids, edge = self.tree_decomp(mol)
        motif_list = []
        for id_ in ids:
            _, token_mols = self.get_clique_mol(mol, id_)
            token_id = self.vocab.get(token_mols)
            if token_id is not None:
                motif_list.append(token_id)
            else:
                motif_list.append(self.vocab.get('<unk>'))
        return motif_list, edge, ids

    def sanitize(self, mol):
        try:
            smiles = self.get_smiles(mol)
            mol = self.get_mol(smiles)
        except Exception as e:
            return None
        return mol

    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    def get_smiles(self, mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True)

    def get_clique_mol(self, mol, atoms_ids):
        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = self.copy_edit_mol(new_mol).GetMol()
        new_mol = self.sanitize(new_mol)
        return new_mol, smiles

    def copy_atom(self, atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom

    def copy_edit_mol(self, mol):
        new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in mol.GetAtoms():
            new_atom = self.copy_atom(atom)
            new_mol.AddAtom(new_atom)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            new_mol.AddBond(a1, a2, bt)
        return new_mol

    def tree_decomp(self, mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                cliques.append([a1, a2])

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        cliques.extend(ssr)

        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        for i in range(len(cliques)):
            if len(cliques[i]) <= 2:
                continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2:
                        continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []

        cliques = [c for c in cliques if len(c) > 0]
        nei_list = [[] for i in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        edges = defaultdict(int)
        for atom in range(n_atoms):
            if len(nei_list[atom]) <= 1:
                continue
            cnei = nei_list[atom]
            bonds = [c for c in cnei if len(cliques[c]) == 2]
            rings = [c for c in cnei if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:
                cliques.append([atom])
                c2 = len(cliques) - 1
                for c1 in cnei:
                    edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1
            else:
                for i in range(len(cnei)):
                    for j in range(i + 1, len(cnei)):
                        c1, c2 = cnei[i], cnei[j]
                        inter = set(cliques[c1]) & set(cliques[c2])
                        if edges[(c1, c2)] < len(inter):
                            edges[(c1, c2)] = len(inter)

        edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
        if len(edges) == 0:
            return cliques, edges

        # 计算最大生成树
        row, col, data = zip(*edges)
        n_clique = len(cliques)
        clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
        junc_tree = minimum_spanning_tree(clique_graph)
        row, col = junc_tree.nonzero()
        edges = [(row[i], col[i]) for i in range(len(row))]
        return cliques, edges

