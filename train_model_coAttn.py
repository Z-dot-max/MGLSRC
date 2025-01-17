import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
from tqdm import tqdm
from atom_encoder import *
from motif_encoder import *
from tqdm import tqdm
from pre_work import *

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, batch_size):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.batch_size = batch_size
        self.drop_rate = 0.1
        self.dropout = nn.Dropout(self.drop_rate)

    # x:（batch_size x ... x d_model*2）
    # return: (batch_size x  num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MeTRCP(nn.Module):
    def __init__(self, param, tokenizer):
        super(MeTRCP, self).__init__()
        self.num_layers = param['num_layers']
        self.num_heads = param['num_heads']
        self.num_classes = param['num_classes']
        self.d_model = param['d_model'] * 2
        self.dff = self.d_model
        self.input_vocab_size = tokenizer.get_vocab_size
        self.dropout_rate = 0.1
        self.input_dim = 8

        self.encoder_model_atom = EncoderModelAtom(input_dim=62, num_layers=2, d_model=self.d_model, dff=self.dff,num_heads=self.num_heads)
        self.encoder_model_motif = EncoderModelMotif(num_layers=self.num_layers, d_model=self.d_model, dff=self.dff,
                                               num_heads=self.num_heads, input_vocab_size=self.input_vocab_size)
        self.Wa = nn.Linear(512, self.d_model)
        self.Wb = nn.Linear(512, self.d_model)
        self.Wc = nn.Linear(512, self.d_model)
        self.co_attention_layer = CoAttentionLayer(self.d_model, k=128)
        
        self.fc_catalyst1 = FullyConnectedNN(input_dim=self.d_model*2, num_classes=self.num_classes[0], hidden_dim=256,
                                             batch_size=64)
        self.fc_solvent1 = FullyConnectedNN(input_dim=self.d_model*2, num_classes=self.num_classes[1], hidden_dim=256,
                                             batch_size=64)
        self.fc_solvent2 = FullyConnectedNN(input_dim=self.d_model*2, num_classes=self.num_classes[2], hidden_dim=256,
                                            batch_size=64)
        self.fc_reagent1 = FullyConnectedNN(input_dim=self.d_model*2, num_classes=self.num_classes[3], hidden_dim=512,
                                            batch_size=64)
        self.fc_reagent2 = FullyConnectedNN(input_dim=self.d_model*2, num_classes=self.num_classes[4], hidden_dim=512,
                                            batch_size=64)
        self.dropout2 = nn.Dropout(0.05)

    def forward(self, nums_list1, nums_list2, nums_list3, adjoin_matrix1, adjoin_matrix2, adjoin_matrix3, \
        dist_matrix1, dist_matrix2, dist_matrix3, atom_features1, atom_features2, atom_features3,\
        adj_matrix_atom1, adj_matrix_atom2, adj_matrix_atom3, dist_matrix_atom1, dist_matrix_atom2, dist_matrix_atom3,\
        atom_match_matrix1, atom_match_matrix2, atom_match_matrix3, sum_atoms1, sum_atoms2, sum_atoms3):
        outseq_atom1, *_, encoder_padding_mask_atom1 = self.encoder_model_atom(x=atom_features1, adjoin_matrix = adj_matrix_atom1, \
                                               dist_matrix = dist_matrix_atom1, atom_match_matrix=atom_match_matrix1, \
                                               sum_atoms=sum_atoms1, training=True)
        outseq_atom2, *_, encoder_padding_mask_atom2 = self.encoder_model_atom(x=atom_features2, adjoin_matrix = adj_matrix_atom2, \
                                               dist_matrix = dist_matrix_atom2, atom_match_matrix=atom_match_matrix2, \
                                               sum_atoms=sum_atoms2, training=True)
        outseq_atom3, *_, encoder_padding_mask_atom3 = self.encoder_model_atom(x=atom_features3, adjoin_matrix = adj_matrix_atom3, \
                                               dist_matrix = dist_matrix_atom3, atom_match_matrix=atom_match_matrix3, \
                                               sum_atoms=sum_atoms3, training=True)

        outseq_motif1,*_,encoder_padding_mask_motif1 = self.encoder_model_motif(x=nums_list1, atom_level_features=outseq_atom1, \
                                                 adjoin_matrix=adjoin_matrix1, dist_matrix=dist_matrix1, training=True)
        outseq_motif2,*_,encoder_padding_mask_motif2 = self.encoder_model_motif(x=nums_list2, atom_level_features=outseq_atom2, \
                                                 adjoin_matrix=adjoin_matrix2, dist_matrix=dist_matrix2, training=True)
        outseq_motif3,*_,encoder_padding_mask_motif3 = self.encoder_model_motif(x=nums_list3, atom_level_features=outseq_atom3, \
                                                 adjoin_matrix=adjoin_matrix3, dist_matrix=dist_matrix3, training=True)

        outseq_motif1 = self.Wa(outseq_motif1)
        outseq_motif2 = self.Wb(outseq_motif2)
        outseq_motif3 = self.Wc(outseq_motif3)

        out_seq_reactant = torch.cat((outseq_motif1, outseq_motif2), dim=1)
        out_seq_product = outseq_motif3
        # # # Co-attention
        outseq1, outseq2, *_ = self.co_attention_layer([out_seq_reactant, out_seq_product])
        out_seq = torch.cat((outseq1, outseq2), dim=-1)
        # Cross-attention
#         out_seq, attention_weights = self.cross_attention_layer(out_seq_reactant, out_seq_product)
#         out_seq = out_seq.sum(dim=1)
        # 分类
        # output1_2 = self.fc1(out_seq)
        # output1_2 = self.dropout1(output1_2)
        output_catalyst1 = self.fc_catalyst1(out_seq)
        output_solvent1 = self.fc_solvent1(out_seq)
        output_solvent2 = self.fc_solvent2(out_seq)
        output_reagent1 = self.fc_reagent1(out_seq)
        output_reagent2 = self.fc_reagent2(out_seq)

        output_catalyst1 = self.dropout2(output_catalyst1)
        output_solvent1 = self.dropout2(output_solvent1)
        output_solvent2 = self.dropout2(output_solvent2)
        output_reagent1 = self.dropout2(output_reagent1)
        output_reagent2 = self.dropout2(output_reagent2)

        return (output_catalyst1, output_solvent1, output_solvent2, output_reagent1, output_reagent2)

###################################start####################################

if __name__ == '__main__':
#     torch.cuda.empty_cache()
    dataFolder = 'data/'
    tr_dataset = pd.read_csv(dataFolder + 'filtered_USPTO_condition_train.csv')
    val_dataset = pd.read_csv(dataFolder + 'filtered_USPTO_condition_val.csv')

    tokenizer = Mol_Tokenizer('data/clique.json')
    map_dict = np.load(dataFolder + 'preprocessed_molecular_info.npy', allow_pickle=True).item()
    print('map_dict loaded successfully')

    train_dataset = Graph_Bert_Dataset_fine_tune(
        original_dataset=tr_dataset,
        tokenizer=tokenizer,
        map_dict=map_dict,
        label_fields=['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    )
    val_dataset = Graph_Bert_Dataset_fine_tune(
        original_dataset=val_dataset,
        tokenizer=tokenizer,
        map_dict=map_dict,
        label_fields=['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64, 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        num_workers=4 
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )

    # Optimizer and loss function
    model = MeTRCP(param={'name': 'Small', 'num_layers': 4, 'num_heads': 4, 'd_model': 256,
                          'num_classes': [54, 87, 87, 235, 235]}, tokenizer=tokenizer)
    model.load_state_dict(torch.load('results/best_model.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Early Stopping
    early_stopping_counter = 0
    best_val_loss = float('inf')
    patience = 5

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch", ncols=100)
        for batch in pbar:
            optimizer.zero_grad()
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

            loss_catalyst1 = loss_fn(output_catalyst1, labels[:,0])
            loss_solvent1 = loss_fn(output_solvent1, labels[:,1])
            loss_solvent2 = loss_fn(output_solvent2, labels[:,2])
            loss_reagent1 = loss_fn(output_reagent1, labels[:,3]) 
            loss_reagent2 = loss_fn(output_reagent2, labels[:,4])

            total_loss = loss_catalyst1 + loss_solvent1 + loss_solvent2 + loss_reagent1 + loss_reagent2
            
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=total_loss.item())

        print(f"Epoch {epoch + 1}, Loss: {total_loss.item()}")
        
        total = 0
        k_values = [1,3,5,10]
        correct = {k: {'catalyst1': 0, 'solvent1': 0, 'solvent2': 0, 'reagent1': 0, 'reagent2': 0} for k in k_values}
        top1_acc = {'catalyst1': float('inf'), 'solvent1': float('inf'), 'solvent2': float('inf'), 'reagent1': float('inf'), 'reagent2': float('inf')}
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
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
                labels2 = batch[24].to(device)

                output_catalyst1_val, output_solvent1_val, output_solvent2_val, output_reagent1_val, output_reagent2_val = \
                    model(nums_list1, nums_list2, nums_list3, adjoin_matrix1, adjoin_matrix2, adjoin_matrix3, \
                          dist_matrix1, dist_matrix2, dist_matrix3, atom_features1, atom_features2, atom_features3, \
                          adj_matrix_atom1, adj_matrix_atom2, adj_matrix_atom3, dist_matrix_atom1, dist_matrix_atom2,
                          dist_matrix_atom3, \
                          atom_match_matrix1, atom_match_matrix2, atom_match_matrix3, sum_atoms1, sum_atoms2,
                          sum_atoms3)

                loss_catalyst1 = loss_fn(output_catalyst1_val, labels2[:, 0])
                loss_solvent1 = loss_fn(output_solvent1_val, labels2[:, 1])
                loss_solvent2 = loss_fn(output_solvent2_val, labels2[:, 2])
                loss_reagent1 = loss_fn(output_reagent1_val, labels2[:, 3])
                loss_reagent2 = loss_fn(output_reagent2_val, labels2[:, 4])

                val_loss += loss_catalyst1 + loss_solvent1 + loss_solvent2 + loss_reagent1 + loss_reagent2
                
                top_k_values_catalyst1, top_k_indices_catalyst1 = torch.topk(output_catalyst1_val, k=10, dim=-1)
                top_k_values_solvent1, top_k_indices_solvent1 = torch.topk(output_solvent1_val, k=10, dim=-1)
                top_k_values_solvent2, top_k_indices_solvent2 = torch.topk(output_solvent2_val, k=10, dim=-1)
                top_k_values_reagent1, top_k_indices_reagent1 = torch.topk(output_reagent1_val, k=10, dim=-1)
                top_k_values_reagent2, top_k_indices_reagent2 = torch.topk(output_reagent2_val, k=10, dim=-1)
                
                total += labels2.size(0)
                for i in range(labels2.size(0)):
                    for k in k_values:
                        if labels2[i,0] in top_k_indices_catalyst1[i, :k]:
                            correct[k]['catalyst1'] += 1
                        if labels2[i,1] in top_k_indices_solvent1[i, :k]:
                            correct[k]['solvent1'] += 1
                        if labels2[i,2] in top_k_indices_solvent2[i, :k]:
                            correct[k]['solvent2'] += 1
                        if labels2[i,3] in top_k_indices_reagent1[i, :k]:
                            correct[k]['reagent1'] += 1
                        if labels2[i,4] in top_k_indices_reagent2[i, :k]:
                            correct[k]['reagent2'] += 1

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
        for k in k_values:
            print('Top',k,' accuracy is: ')
            print(f"catalyst: {correct[k]['catalyst1'] / total:.4f}")
            print(f"solvent1: {correct[k]['solvent1'] / total:.4f}")
            print(f"solvent2: {correct[k]['solvent2'] / total:.4f}")
            print(f"reagent1: {correct[k]['reagent1'] / total:.4f}")
            print(f"reagent2: {correct[k]['reagent2'] / total:.4f}")
        
        # 去掉早停版本，用早停时把下面的代码去掉
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= patience:  # 如果验证损失没有提升，触发早停
            print("Early stopping triggered.")
            break

