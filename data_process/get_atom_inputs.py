import pandas as pd
import numpy as np
import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import copy
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import rdkit.Chem as Chem
import json
import networkx as nx
from rdkit.Chem import MolFromSmiles, MolToSmiles, BRICS
from get_atom_features import *
import ast

class Mol_Tokenizer():
    def __init__(self,tokens_id_file):
        self.vocab = json.load(open(r'{}'.format(tokens_id_file),'r'))
        self.MST_MAX_WEIGHT = 100
        self.get_vocab_size = len(self.vocab.keys())
        self.id_to_token = {value:key for key,value in self.vocab.items()}
    def tokenize(self,smiles):
        # mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        mol = Chem.MolFromSmiles(smiles)
        ids,edge = self.brics_decomp(mol)
        motif_list = []
        for id_ in ids:
            _,token_mols = self.get_clique_mol(mol,id_)
            token_id = self.vocab.get(token_mols)
            if token_id != None:
                motif_list.append(token_id)
            else:
                motif_list.append(self.vocab.get('<unk>'))
        return motif_list,edge,ids
    def sanitize(self,mol):
        try:
            smiles = self.get_smiles(mol)
            mol = self.get_mol(smiles)
        except Exception as e:
            return None
        return mol
    def get_mol(self,smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol
    def get_smiles(self,mol):
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    def get_clique_mol(self,mol,atoms_ids):
    # get the fragment of clique
        smiles = Chem.MolFragmentToSmiles(mol, atoms_ids, kekuleSmiles=False)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        new_mol = self.copy_edit_mol(new_mol).GetMol()
        new_mol = self.sanitize(new_mol)  # We assume this is not None
        return new_mol,smiles
    def copy_atom(self,atom):
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        return new_atom
    def copy_edit_mol(self,mol):
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
    # def tree_decomp(self,mol):
    #     n_atoms = mol.GetNumAtoms()
    #     if n_atoms == 1:
    #         return [[0]], []
    #
    #     cliques = []
    #     for bond in mol.GetBonds():
    #         a1 = bond.GetBeginAtom().GetIdx()
    #         a2 = bond.GetEndAtom().GetIdx()
    #         if not bond.IsInRing():
    #             cliques.append([a1, a2])
    #
    #     # get rings
    #     ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    #     cliques.extend(ssr)
    #
    #     nei_list = [[] for i in range(n_atoms)]
    #     for i in range(len(cliques)):
    #         for atom in cliques[i]:
    #             nei_list[atom].append(i)
    #
    #     # Merge Rings with intersection > 2 atoms
    #     for i in range(len(cliques)):
    #         if len(cliques[i]) <= 2: continue
    #         for atom in cliques[i]:
    #             for j in nei_list[atom]:
    #                 if i >= j or len(cliques[j]) <= 2: continue
    #                 inter = set(cliques[i]) & set(cliques[j])
    #                 if len(inter) > 2:
    #                     cliques[i].extend(cliques[j])
    #                     cliques[i] = list(set(cliques[i]))
    #                     cliques[j] = []
    #
    #     cliques = [c for c in cliques if len(c) > 0]
    #     nei_list = [[] for i in range(n_atoms)]
    #     for i in range(len(cliques)):
    #         for atom in cliques[i]:
    #             nei_list[atom].append(i)
    #
    #     # Build edges and add singleton cliques
    #     edges = defaultdict(int)
    #     for atom in range(n_atoms):
    #         if len(nei_list[atom]) <= 1:
    #             continue
    #         cnei = nei_list[atom]
    #         bonds = [c for c in cnei if len(cliques[c]) == 2]
    #         rings = [c for c in cnei if len(cliques[c]) > 4]
    #         if len(bonds) > 2 or (len(bonds) == 2 and len(
    #                 cnei) > 2):  # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
    #             cliques.append([atom])
    #             c2 = len(cliques) - 1
    #             for c1 in cnei:
    #                 edges[(c1, c2)] = 1
    #         elif len(rings) > 2:  # Multiple (n>2) complex rings
    #             cliques.append([atom])
    #             c2 = len(cliques) - 1
    #             for c1 in cnei:
    #                 edges[(c1, c2)] = self.MST_MAX_WEIGHT - 1
    #         else:
    #             for i in range(len(cnei)):
    #                 for j in range(i + 1, len(cnei)):
    #                     c1, c2 = cnei[i], cnei[j]
    #                     inter = set(cliques[c1]) & set(cliques[c2])
    #                     if edges[(c1, c2)] < len(inter):
    #                         edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction
    #
    #     edges = [u + (self.MST_MAX_WEIGHT - v,) for u, v in edges.items()]
    #     if len(edges) == 0:
    #         return cliques, edges
    #
    #     # Compute Maximum Spanning Tree
    #     row, col, data = zip(*edges)
    #     n_clique = len(cliques)
    #     clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    #     junc_tree = minimum_spanning_tree(clique_graph)
    #     row, col = junc_tree.nonzero()
    #     edges = [(row[i], col[i]) for i in range(len(row))]
    #     return (cliques, edges)
    def brics_decomp(self, mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            cliques.append([a1, a2])

        res = list(BRICS.FindBRICSBonds(mol))
        if len(res) == 0:
            return [list(range(n_atoms))], []
        else:
            for bond in res:
                if [bond[0][0], bond[0][1]] in cliques:
                    cliques.remove([bond[0][0], bond[0][1]])
                else:
                    cliques.remove([bond[0][1], bond[0][0]])
                cliques.append([bond[0][0]])
                cliques.append([bond[0][1]])

        # break bonds between rings and non-ring atoms
        for c in cliques:
            if len(c) > 1:
                if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                    cliques.remove(c)
                    cliques.append([c[1]])
                    breaks.append(c)
                if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                    cliques.remove(c)
                    cliques.append([c[0]])
                    breaks.append(c)

        # select atoms at intersections as motif
        for atom in mol.GetAtoms():
            if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
                cliques.append([atom.GetIdx()])
                for nei in atom.GetNeighbors():
                    if [nei.GetIdx(), atom.GetIdx()] in cliques:
                        cliques.remove([nei.GetIdx(), atom.GetIdx()])
                        breaks.append([nei.GetIdx(), atom.GetIdx()])
                    elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                        cliques.remove([atom.GetIdx(), nei.GetIdx()])
                        breaks.append([atom.GetIdx(), nei.GetIdx()])
                    cliques.append([nei.GetIdx()])

        # merge cliques
        for c in range(len(cliques) - 1):
            if c >= len(cliques):
                break
            for k in range(c + 1, len(cliques)):
                if k >= len(cliques):
                    break
                if len(set(cliques[c]) & set(cliques[k])) > 0:
                    cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                    cliques[k] = []
            cliques = [c for c in cliques if len(c) > 0]
        cliques = [c for c in cliques if len(c) > 0]

        # edges
        edges = []
        for bond in res:
            for c in range(len(cliques)):
                if bond[0][0] in cliques[c]:
                    c1 = c
                if bond[0][1] in cliques[c]:
                    c2 = c
            edges.append((c1, c2))
        for bond in breaks:
            for c in range(len(cliques)):
                if bond[0] in cliques[c]:
                    c1 = c
                if bond[1] in cliques[c]:
                    c2 = c
            edges.append((c1, c2))

        return (cliques, edges)


def get_adj_matrix(num_list,edges):
    adjoin_matrix = np.eye(len(num_list))
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    return adjoin_matrix

degrees = [0, 1, 2, 3, 4, 5, 6]
def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    graph = graph_from_smiles(smiles)
    molgraph = MolGraph()
    molgraph.add_subgraph(graph)
    molgraph.sort_nodes_by_degree('atom')
    try:
        arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                    'bond_features' : molgraph.feature_array('bond'),
                    'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                    'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    except:
        arrayrep = {}
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep
class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self,ntype,features,rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type
    #@ps.snoop()
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))# gain node

    # 根据节点的度数对特定节点进行排序
    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        # print('The error ntype = ', ntype)
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


def atom_features_(atom):
    Degree = [i for i in one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])][:]
    NumHs = [i for i in one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])][:]
    ImplicitValence = [i for i in one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])][:]
    GetSymbol = [i for i in one_of_k_encoding_unk(atom.GetSymbol(),
                                                  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                   'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Sn', 'Ag', 'Pd', 'Co',
                                                   'Se', 'Ti', 'Zn',  # H?
                                                   'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Mn', 'Other'])][:]

    return np.array(GetSymbol +
                    Degree + NumHs + ImplicitValence + [atom.GetIsAromatic()])

def graph_from_smiles(smiles):
    graph = MolGraph()
    # mol = MolFromSmiles(smiles)
    mol = MolFromSmiles(smiles)
    mol = MolFromSmiles(MolToSmiles(mol))
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    graph = graph_from_smiles(smiles)
    molgraph = MolGraph()
    molgraph.add_subgraph(graph)
    molgraph.sort_nodes_by_degree('atom')
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

def get_dist_matrix(num_list,edges):
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.zeros((len(num_list),len(num_list)))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row,col] = 0
    graph_nodes = sorted(make_graph.nodes.keys())
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for dist in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[dist].items(),key = lambda x:x[0]))
        temp_node_dist_dict = {i:node_relative_distance.get(i) if \
        node_relative_distance.get(i)!= None else 1e9 for i in graph_nodes} ### 1e9 refers to Chem.GetDistanceMatrix(mol) in rdkit
        temp_node_dist_list = list(temp_node_dist_dict.values())
        dist_matrix[dist][graph_nodes] =  temp_node_dist_list
    return dist_matrix.astype(np.float32)

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    graph = graph_from_smiles(smiles)
    molgraph = MolGraph()
    molgraph.add_subgraph(graph)
    molgraph.sort_nodes_by_degree('atom')
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


def extract_bondfeatures_of_neighbors_by_degree(array_rep):
    """
    Sums up all bond features that connect to the atoms (sorted by degree)

    Returns:
    ----------

    list with elements of shape: [(num_atoms_degree_0, 6), (num_atoms_degree_1, 6), (num_atoms_degree_2, 6), etc....]

    e.g.:

    >> print [x.shape for x in extract_bondfeatures_of_neighbors_by_degree(array_rep)]

    [(0,), (269, 6), (524, 6), (297, 6), (25, 6), (0,)]

    """
    bond_features_by_atom_by_degree = []
    for degree in degrees:
        bond_features = array_rep['bond_features']
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1)
        bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    return bond_features_by_atom_by_degree
    # bond_features_by_atom_by_degree = []
    # for degree in degrees:
    #     bond_features_tmp = array_rep['bond_features']
    #     bond_features = np.array([i.features for i in bond_features_tmp])
    #     bond_neighbors_list = array_rep[('bond_neighbors', degree)]
    #     summed_bond_neighbors = []
    #     if len(bond_neighbors_list) > 0:
    #         bond_neighbors_list = bond_neighbors_list.flatten()
    #         bond_neighbors_list_features = bond_features[bond_neighbors_list]
    #         summed_bond_neighbors = bond_neighbors_list_features.sum(axis=1)
    #     bond_features_by_atom_by_degree.append(summed_bond_neighbors)
    # return bond_features_by_atom_by_degree

def bond_features_by_degree(total_atoms,summed_degrees,degree):
    mat = np.zeros((total_atoms,10),'float32')
    total_num = []
    if degree == 0:
        for i,x in enumerate(summed_degrees[0]):
            mat[i] = x
        return mat
    else:
        for i in range(degree):
            total_num.append(len(summed_degrees[i]))
        total_num = sum(total_num)  # 连接的原子总数
        for i,x in enumerate(summed_degrees[degree]):
            mat[total_num + i] = x
        return mat

def molgraph_rep(smi,cliques):
    def atom_to_motif_match(atom_order,cliques):
        atom_order = atom_order.tolist()
        temp_matrix = np.zeros((len(cliques),len(atom_order)))
        for th,cli in enumerate(cliques):
            for i in cli:
                temp_matrix[th,atom_order.index(i)] = 1
        return temp_matrix
    def get_adj_dist_matrix(mol_graph,smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        num_atoms = mol.GetNumAtoms()
        adjoin_matrix_temp = np.eye(num_atoms)
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = (adjoin_matrix_temp + adj_matrix)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        dist_matrix = Chem.GetDistanceMatrix(mol)[:,mol_graph['rdkit_ix']][mol_graph['rdkit_ix']]
        return adj_matrix,dist_matrix
    single_dict = {'input_atom_features':[],
            'atom_match_matrix':[],
            'sum_atoms':[],
            'adj_matrix':[],
            'dist_matrix':[]
            }
    array_rep = array_rep_from_smiles(smi)
    summed_degrees = extract_bondfeatures_of_neighbors_by_degree(array_rep)
    atom_features = array_rep['atom_features'] # Node类型列表，包含features:int list, ntype:string, rdkit_ix: int, neighbors:Node list
    all_bond_features = []
    for degree in degrees:
        atom_neighbors_list = array_rep[('atom_neighbors', degree)].astype('int32')
        if len(atom_neighbors_list)==0:
            # true_summed_degree = np.zeros((atom_features.shape[0], 10),'float32')
            true_summed_degree = np.zeros((len(atom_features), 10), 'float32')
        else:
            # atom_neighbor_matching_matrix = connectivity_to_Matrix(array_rep, atom_features.shape[0],degree)
            true_summed_degree = bond_features_by_degree(len(atom_features), summed_degrees, degree)
        # atom_selects = np.matmul(atom_neighbor_matching_matrix,atom_features)
        # merged_atom_bond_features = np.concatenate([atom_features,true_summed_degree],axis=1)
        all_bond_features.append(true_summed_degree)
    single_dict['atom_match_matrix'] = atom_to_motif_match(array_rep['rdkit_ix'],cliques)
    single_dict['sum_atoms'] = np.reshape(np.sum(single_dict['atom_match_matrix'],axis=1),(-1,1))
    out_bond_features = 0
    for arr in all_bond_features:
        out_bond_features = out_bond_features + arr
    single_dict['input_atom_features'] = np.concatenate([atom_features,out_bond_features],axis=1)
    adj_matrix,dist_matrix = get_adj_dist_matrix(array_rep,smi)
    single_dict['adj_matrix'] = adj_matrix
    single_dict['dist_matrix'] = dist_matrix
    single_dict = {key:np.array(value,dtype='float32') for key,value in single_dict.items()}
    return single_dict



############################### start ######################################
toy_dataset = pd.read_csv('../data/filtered_USPTO_condition_sum.csv')
unique_SMILES_in_toy_dataset = []
# 将 Reactants 列的每一行字符串转换为列表
products_list = toy_dataset['Products'].dropna().tolist()
for smiles in products_list:
    smiles = ast.literal_eval(smiles)
    # print(smiles)
    for temp_smiles in smiles:
        unique_SMILES_in_toy_dataset.append(temp_smiles)


# 将 Products 列的每一行字符串转换为列表
reactants_list = toy_dataset['Reactants'].dropna().tolist()
for smiles in reactants_list:
    smiles = ast.literal_eval(smiles)
    # print(smiles)
    for temp_smiles in smiles:
        unique_SMILES_in_toy_dataset.append(temp_smiles)

# obtain all unique SMILES in the toy dataset
unique_SMILES_in_toy_dataset = set(unique_SMILES_in_toy_dataset)
print(unique_SMILES_in_toy_dataset.__len__())

# create a dict to store all information of drugs
all_drugs_dict = {i:{'adj_matrix':[],'dist_matrix':[],'nums_list':[],'cliques':[],'edges':[],'single_dict':[]} for i in unique_SMILES_in_toy_dataset}
# print(all_drugs_dict)

tokenizer = Mol_Tokenizer('../data/clique.json')

for drug in tqdm.tqdm(unique_SMILES_in_toy_dataset):
    # try:
    try:
        nums_list1, edges1, cliques1 = tokenizer.tokenize(drug)
        # print(cliques1)
        dist_matrix1 = get_dist_matrix(nums_list1, edges1)
        adjoin_matrix1 = get_adj_matrix(nums_list1, edges1)
    # print('dist_matrix1 and adjoin_matrix1 is: ', dist_matrix1, adjoin_matrix1)
        all_drugs_dict[drug]['adj_matrix']= adjoin_matrix1
        all_drugs_dict[drug]['dist_matrix']= dist_matrix1
        all_drugs_dict[drug]['nums_list']= nums_list1
        all_drugs_dict[drug]['edges']= edges1
        all_drugs_dict[drug]['cliques'] = cliques1
        all_drugs_dict[drug]['single_dict'] = molgraph_rep(drug,cliques1)
        # print(all_drugs_dict[drug])
        # print(type(all_drugs_dict[drug]))
    except Exception as e:
        print(e)
        with open('error_molecules.txt', 'a', encoding='utf-8') as file:
            file.write(drug)
            file.write('\n')
        # unique_SMILES_in_toy_dataset.remove(drug)
        # del all_drugs_dict[drug]
        continue

# with open('error_molecules.txt', 'r') as file:
#     for line in file:
#         drug = line.strip()
#         del all_drugs_dict[drug]

np.save('../data/preprocessed_molecular_info.npy',all_drugs_dict)
