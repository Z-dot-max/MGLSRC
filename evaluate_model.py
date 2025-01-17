from pre_work import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_model_coAttn import MeTRCP

dataFolder = 'data/'
test_dataset = pd.read_csv(dataFolder + 'filtered_USPTO_condition_test.csv')

tokenizer = Mol_Tokenizer(dataFolder + 'clique.json')
map_dict = np.load(dataFolder + 'preprocessed_molecular_info.npy', allow_pickle=True).item()
print('map_dict loaded successfully')

test_dataset = Graph_Bert_Dataset_fine_tune(
    original_dataset=test_dataset, 
    tokenizer=tokenizer,
    map_dict=map_dict,
    label_fields=['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4
)

model = MeTRCP(param={'name': 'Small', 'num_layers': 4, 'num_heads': 4, 'd_model': 256,
                      'num_classes': [54, 87, 87, 235, 235]}, tokenizer=tokenizer)

model.load_state_dict(torch.load('results/best_model_coAttn.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def calculate_top_k_accuracy(top_k_indices, labels, k_values):
    accuracy = {}
    for k in k_values:
        correct = 0
        for i in range(labels.size(0)):
            if labels[i] in top_k_indices[i, :k]:
                correct += 1
        accuracy[k] = correct / labels.size(0)
    return accuracy

total = 0
k_values = [1, 3, 5, 10]
correct = {k: {'catalyst1': 0, 'solvent1': 0, 'solvent2': 0, 'reagent1': 0, 'reagent2': 0} for k in k_values}
tag = {1: 0, 3: 0, 5: 0, 10: 0}

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Processing batches", ncols=100):
        nums_list1 = batch[0].to(device)
        nums_list2 = batch[1].to(device)
        nums_list3 = batch[2].to(device)
        adjoin_matrix1 = batch[3].to(device)
        adjoin_matrix2 = batch[4].to(device)
        adjoin_matrix3 = batch[5].to(device)
        dist_matrix1 = batch[6].to(device)
        dist_matrix2 = batch[7].to(device)
        dist_matrix3 = batch[8].to(device)
        atom_features1 = batch[9].to(device)
        atom_features2 = batch[10].to(device)
        atom_features3 = batch[11].to(device)
        adj_matrix_atom1 = batch[12].to(device)
        adj_matrix_atom2 = batch[13].to(device)
        adj_matrix_atom3 = batch[14].to(device)
        dist_matrix_atom1 = batch[15].to(device)
        dist_matrix_atom2 = batch[16].to(device)
        dist_matrix_atom3 = batch[17].to(device)
        atom_match_matrix1 = batch[18].to(device)
        atom_match_matrix2 = batch[19].to(device)
        atom_match_matrix3 = batch[20].to(device)
        sum_atoms1 = batch[21].to(device)
        sum_atoms2 = batch[22].to(device)
        sum_atoms3 = batch[23].to(device)
        labels = batch[24].to(device)

        output_catalyst1, output_solvent1, output_solvent2, output_reagent1, output_reagent2 = \
            model(nums_list1, nums_list2, nums_list3, adjoin_matrix1, adjoin_matrix2, adjoin_matrix3, \
        dist_matrix1, dist_matrix2, dist_matrix3, atom_features1, atom_features2, atom_features3,\
        adj_matrix_atom1, adj_matrix_atom2, adj_matrix_atom3, dist_matrix_atom1, dist_matrix_atom2, dist_matrix_atom3,\
        atom_match_matrix1, atom_match_matrix2, atom_match_matrix3, sum_atoms1, sum_atoms2, sum_atoms3)
        
        top_k_values_catalyst1, top_k_indices_catalyst1 = torch.topk(output_catalyst1, k=10, dim=-1)
        top_k_values_solvent1, top_k_indices_solvent1 = torch.topk(output_solvent1, k=10, dim=-1)
        top_k_values_solvent2, top_k_indices_solvent2 = torch.topk(output_solvent2, k=10, dim=-1)
        top_k_values_reagent1, top_k_indices_reagent1 = torch.topk(output_reagent1, k=10, dim=-1)
        top_k_values_reagent2, top_k_indices_reagent2 = torch.topk(output_reagent2, k=10, dim=-1)

        total += labels.size(0)
        for i in range(labels.size(0)): 
            for k in k_values:
                flag = 0
                if labels[i,0] in top_k_indices_catalyst1[i, :k]:
                    correct[k]['catalyst1'] += 1
                    flag += 1
                if labels[i,1] in top_k_indices_solvent1[i, :k]:
                    correct[k]['solvent1'] += 1
                    flag += 1
                if labels[i,2] in top_k_indices_solvent2[i, :k]:
                    correct[k]['solvent2'] += 1
                    flag += 1
                if labels[i,3] in top_k_indices_reagent1[i, :k]:
                    correct[k]['reagent1'] += 1
                    flag += 1
                if labels[i,4] in top_k_indices_reagent2[i, :k]:
                    correct[k]['reagent2'] += 1
                    flag += 1
                if flag == 5:
                    tag[k] +=1
                    
                

for k in k_values:
    print('Top',k,' accuracy is: ')
    print(f"catalyst: {correct[k]['catalyst1'] / total:.4f}")
    print(f"solvent1: {correct[k]['solvent1'] / total:.4f}")
    print(f"solvent2: {correct[k]['solvent2'] / total:.4f}")
    print(f"reagent1: {correct[k]['reagent1'] / total:.4f}")
    print(f"reagent2: {correct[k]['reagent2'] / total:.4f}")
    print(f"over: {tag[k] / total:.4f}")
