import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
import json, pickle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from CreateDataset_graph import CreateDataset    
import networkx as nx   
import torch
import torchdrug as td

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def Node_Feature(mol):
    # mol(Object) : predicted molecule features from SMILEs
    # return only node feature from molecule object
    return mol.node_feature

def Edge_Feature(mol):
    # mol(Object) : predicted molecule features from SMILEs
    # return only edge feature from molecule object
    # Last dim is edge type
    e_feat = mol.edge_feature
    # e_type = np.array(mol.edge_list[:, 2])
    # edge type one-hot
    # e_hot = np.zeros((len(e_type), 4))
    # e_hot[np.arange(e_type.size),e_type] = 1
    return e_feat

def add_self_loop(node_list, edge_list, edge_attr):
    self_edge_index = []
    self_edge_attr = []
    edge_attr_dim = edge_attr.shape[1]
    self_loop = np.matrix(np.eye(node_list.shape[0]))
    index_row, index_col = np.where(self_loop >= 0.5)
    for i, j in zip(index_row, index_col):
        self_edge_index.append([i, j])
        if edge_attr is not None:
            self_edge_attr.extend(np.ones([1, edge_attr_dim], int).tolist())
    edge_list = np.append(edge_list, self_edge_index, axis=0)   
    edge_attr = np.append(edge_attr, self_edge_attr, axis=0)
    return edge_list, edge_attr

def Adj_list(mol):
    # mol(Object) : predicted molecule features from SMILEs
    # return only adjacency list from molecule object (from, to)
    # last dimension is node type which is unnecessary in current state
    return mol.edge_list[:, :2]

def Edge_type(mol):
    return mol.edge_list[:, 2]

def Combine_feature(smile):
    # smile(String) : SMILE sequence
    # Combine all feature into single list.
    x = np.array(Node_Feature(smile))
    y = np.array(Edge_Feature(smile))
    z, y = add_self_loop(x, Adj_list(smile), y)
    #t = np.array(Edge_type(smile)) For RGCN
    return [x, y, z, [0]]
    
# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return [features, [0], edge_index, [0]]  

def Make_feature(path="/", split='train', dataset='kiba', debug=False):
    # Function time_complexity O(n) ; n = number of smiles in list
    # smile_list(List) : List of SMILEs
    # return dictionary of {SMILE sequence : its features}

    if split not in ['train', 'test', 'validation']:
        return print("Incorrect input. split should be 'train, test, or validation'")

    df = pd.read_csv(path + dataset + '_' + split + '.csv')

    print("###########################")
    print(" MAKE FEATURES ON " , split, " Dataset")
    print("###########################\n")
    smile_list = set(np.array(df.compound_iso_smiles))
    print("Number of unique SMILEs : ", len(smile_list))
    smile_graph = {}

    if debug==True:
        for smile in smile_list:
            mol = td.data.Molecule.from_smiles(smile)
            feature = Combine_feature(mol)
            #feature = smile_to_graph(smile)
            smile_graph[smile] = feature  
        return smile_graph

    for smile in smile_list:
        try:
            mol = td.data.Molecule.from_smiles(smile)
            feature = Combine_feature(mol)
            #feature = smile_to_graph(smile)
            smile_graph[smile] = feature
        except:
            # Raise from "Invalid: SMILE"
            # This should be also removed from dataframe
            df = df.drop(df[df['compound_iso_smiles']==smile].index)
            continue;
    # Save for utilizing as tracking in smile_graph in future process.
    save_path = path + 'processed/' + dataset + '_' + split + '.csv'
    df.to_csv(save_path, index=False)
    print("Saving Modified Dataframe to file . . .")
    print("Saving location : ", save_path, "\n\n")
    return smile_graph

def create_graph(len_seq, windows):
    # This function already add self loop.
    assert windows % 2 == 1, "windows should be odds number."
    assert windows <= len_seq, "windows must less than sequence length."
    center = (windows//2) # Indice start from 0, +1 if indice start from 1
    adj = []
    for seq in range(len_seq):
        mid = seq+center
        if mid == (len_seq-center):
            break
        adj.extend([mid, target+seq] for target in range(windows))
    # Adding before windows edges.
    for i in range(center):
        adj.extend([i, target] for target in range(center+i+1))
    # Adding After windows edges.   
    for i in range(center):
        index = len_seq-i-1
        adj.extend([index, len_seq-1-target] for target in range(center+i+1))
    return adj

def prot_to_graph(path="/", split='full', dataset='kiba', windows=5):
    # Function time_complexity O(n) ; n = number of protein in list
    # prot_key(List) : List of Protein
    # return dictionary of {protein key : its features}
    
    df = pd.read_csv(path + dataset + '_' + split + '.csv')

    print("###########################")
    print(" MAKE PROTEIN GRAPH ON " , split, " Dataset")
    print("###########################\n")
    prot_key = set(np.array(df.target))
    print("Number of unique Protein : ", len(prot_key))
    prot_graph = {}

    #protemb_path = path + 'protemb/' + dataset +'_emb.pkl'
    #emb = np.load(protemb_path, allow_pickle=True)
    
    for key in prot_key:
        emb = torch.load(path+'esm_emb/'+key+'.pt')
        target_feature = emb["representations"].squeeze()
        target_global_feature = emb["sequence_repr"]
        #target_feature = emb.squeeze()
        target_size = len(target_feature)
        target_edge_index = create_graph(target_size, windows)

        prot_graph[key] = [target_size, target_feature, target_edge_index, target_global_feature]
    print("Protein graph construction -> Complete  \( ﾟヮﾟ)/\n\n")
    return prot_graph

def prepare_dataset(dataset, path='dataset/', windows=3):
    assert(type(dataset) == str), "InputFailed: datasets should be string."

    print('Convert data for ', dataset)
    fpath = path + dataset + '/original/'
    train_folds = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_folds = [folds for folds in train_folds] # (folds, N/folds)

    train_fold = []
    for i in range(len(train_folds)):
        train_fold += train_folds[i]

    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')

    drugs = []
    prots = []
    prot_key = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prot_key.append(t)
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]

    affinity = np.asarray(affinity)
    opts = ['train','test','full']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows, cols = rows[test_fold], cols[test_fold]
        with open('dataset/' + dataset + '/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prot_key[cols[pair_ind]] ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(test_fold))
    print('Unique drugs, Unique proteins:', len(set(drugs)), ", ", len(set(prots)))
    print("#####################################################################################\n\n")
    print("Data folds prepared, Graph processing in progress . . .\n\n")

    path = 'dataset/' + dataset + '/'
    smile_graph_train = Make_feature(path = path, split='train', dataset=dataset, debug=False) #This process will skip missing SMILES
    smile_graph_test = Make_feature(path = path, split='test', dataset=dataset, debug=False) #This process will skip missing SMILES

    prot_graph_train = prot_to_graph(path = path, split='train', dataset=dataset, windows=windows)
    prot_graph_test = prot_to_graph(path = path, split='test', dataset=dataset, windows=windows)
    
    print("#####################################################################################\n\n")
    df = pd.read_csv('dataset/' + dataset + '/processed/' + dataset + '_train.csv')
    train_drugs = list(df.compound_iso_smiles)
    train_prots = list(df.target)
    train_aff = list(df.affinity)
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_aff)

    df = pd.read_csv('dataset/' + dataset + '/processed/' + dataset + '_test.csv')
    test_drugs = list(df.compound_iso_smiles)
    test_prots = list(df.target)
    test_aff = list(df.affinity)
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_aff)

    #if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        # make data PyTorch Geometric ready
    root_path = 'dataset/' + dataset + '/processed/'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = CreateDataset(root=root_path,
                            dataset=dataset+'_train',
                            drugList=train_drugs,
                            protkey=train_prots,
                            y=train_Y,
                            smile_graph=smile_graph_train,
                            protein_graph=prot_graph_train
                            )

    print('preparing ', dataset + '_test.pt in pytorch format!')
    test_data = CreateDataset(root=root_path,
                            dataset=dataset+'_test',
                            drugList=test_drugs,
                            protkey=test_prots,
                            y=test_Y,
                            smile_graph=smile_graph_test,
                            protein_graph=prot_graph_test
                            )
    print('\nPytorch dataset have been created  \( ﾟヮﾟ)/ HooRay!!')   

    return train_data, test_data, test_data

# Add validation set
def prepare_dataset_withFolds(dataset, path='dataset/', fold=0, windows=3, final_eval=False):
    assert(type(dataset) == str), "InputFailed: datasets should be string."

    print('Convert data for ', dataset)
    fpath = path + dataset + '/original/'
    train_folds = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_folds = [folds for folds in train_folds] # (folds, N/folds)

    train_fold = []
    valid_fold = train_folds[fold]
    for i in range(len(train_folds)):
        if i != fold:
            train_fold += train_folds[i]

    test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')

    drugs = []
    prots = []
    prot_key = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prot_key.append(t)
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]

    affinity = np.asarray(affinity)
    opts = ['train','test','validation','full']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows, cols = rows[test_fold], cols[test_fold]
        elif opt=='validation':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('dataset/' + dataset + '/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prot_key[cols[pair_ind]] ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('valid_fold:', len(valid_fold), " on fold: ", fold)
    print('test_fold:', len(test_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), ", ", len(set(prots)))
    print("#####################################################################################\n\n")
    print("Data folds prepared, Graph processing in progress . . .\n\n")

    path = 'dataset/' + dataset + '/'
    smile_graph_train = Make_feature(path = path, split='train', dataset=dataset, debug=False) #This process will skip missing SMILES
    smile_graph_valid = Make_feature(path = path, split='validation', dataset=dataset, debug=False) #This process will skip missing SMILES
    smile_graph_test = Make_feature(path = path, split='test', dataset=dataset, debug=False) #This process will skip missing SMILES

    prot_graph_train = prot_to_graph(path = path, split='train', dataset=dataset, windows=windows)
    prot_graph_valid = prot_to_graph(path = path, split='validation', dataset=dataset, windows=windows)
    prot_graph_test = prot_to_graph(path = path, split='test', dataset=dataset, windows=windows)
    
    print("#####################################################################################\n\n")
    df = pd.read_csv('dataset/' + dataset + '/processed/' + dataset + '_train.csv')
    train_drugs = list(df.compound_iso_smiles)
    train_prots = list(df.target)
    train_aff = list(df.affinity)
    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_aff)

    df = pd.read_csv('dataset/' + dataset + '/processed/' + dataset + '_validation.csv')
    valid_drugs = list(df.compound_iso_smiles)
    valid_prots = list(df.target)
    valid_aff = list(df.affinity)
    valid_drugs, valid_prots, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots), np.asarray(valid_aff)

    df = pd.read_csv('dataset/' + dataset + '/processed/' + dataset + '_test.csv')
    test_drugs = list(df.compound_iso_smiles)
    test_prots = list(df.target)
    test_aff = list(df.affinity)
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_aff)

    #if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        # make data PyTorch Geometric ready
    root_path = 'dataset/' + dataset + '/processed/'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = CreateDataset(root=root_path,
                            dataset=dataset+'_train',
                            drugList=train_drugs,
                            protkey=train_prots,
                            y=train_Y,
                            smile_graph=smile_graph_train,
                            protein_graph=prot_graph_train
                            )

    print('preparing ', dataset + '_validation.pt in pytorch format!')
    valid_data = CreateDataset(root=root_path,
                            dataset=dataset+'_validation',
                            drugList=valid_drugs,
                            protkey=valid_prots,
                            y=valid_Y,
                            smile_graph=smile_graph_valid,
                            protein_graph=prot_graph_valid
                            )

    print('preparing ', dataset + '_test.pt in pytorch format!')
    test_data = CreateDataset(root=root_path,
                            dataset=dataset+'_test',
                            drugList=test_drugs,
                            protkey=test_prots,
                            y=test_Y,
                            smile_graph=smile_graph_test,
                            protein_graph=prot_graph_test
                            )
    print('\nPytorch dataset have been created  \( ﾟヮﾟ)/ HooRay!!')   
    
    if final_eval == True:
        return train_data, test_data, test_data   

    return train_data, valid_data, test_data   
