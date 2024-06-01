import os
import time
import random
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import sklearn
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5EncoderModel
import re


# ------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataset_path', default="./attsec_dataset/")
parser.add_argument('--dataset_type', default="proteinnet", choices=['proteinnet', 'netsurf'])
parser.add_argument('-o', '--output_path', default="./save_files/")

parser.add_argument('-e', '--epochs', default=50, type=int)
parser.add_argument('--early_stopping', default=10, type=float)
parser.add_argument('-b', '--batch_size', default=8, type=int)
parser.add_argument('--warmup', default=5, type=int)


parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--num_layers', default=10, type=int)
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--dropout', default=0.5, type=float)

parser.add_argument('--max_train_tokens', default=512, type=int)

args = parser.parse_args()


#------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_debug_apis(state):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)

set_debug_apis(False)

#------------------------------------------------------------------------
# **Protein Dataset**

BATCH_SIZE = args.batch_size
MAX_TRAIN_TOKENS = args.max_train_tokens      # max seq length while training
MAX_SEQ_LEN = 4000                            # max len of positional embedding

D_MODEL = args.d_model

dataset_path = args.dataset_path
dataset_type = args.dataset_type
checkpoint_base_dir = args.output_path

print("dataset_type :", dataset_type)

#------------------------------------------------------------------------
## Label tokenization

dssp_vocab = ["[PAD]", "G","H","I","B","E","S","L","T"]

dssp2idx = {v:i for i,v in enumerate(dssp_vocab)}
idx2dssp = {i:v for v,i in dssp2idx.items()}

print(dssp2idx)
print(idx2dssp)


def tokenize_label(seq):
    tokens = list(map(lambda c: dssp2idx[c] if c in dssp2idx else 1, seq))
    return tokens


def detokenize_label(ids):
    return "".join(list(map(lambda c: idx2dssp[c], ids.tolist())))

#------------------------------------------------------------------------
## Sequence tokenization

t5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)

t5_model = t5_model.to(device)
t5_model = t5_model.eval()



class T5_collate:
    def __init__(self, random_crop_length=None):
        self.random_crop_length = random_crop_length

    def random_crop(self, batch):
        seq, label = batch
        if self.random_crop_length is not None and len(seq) > self.random_crop_length:
            start = random.randint(0, len(seq)-self.random_crop_length)
            seq = seq[start:start+self.random_crop_length]
            label = label[start:start+self.random_crop_length]
        return seq, label

    def preprocess_sequence(self, seq):
        return " ".join(list(re.sub(r"[UZOB]", "X", seq)))

    def preprocess_labels(self, label):
        return torch.Tensor(tokenize_label(label))

    def __call__(self, batch):
        batch = [self.random_crop(b) for b in batch]
        sequences = [self.preprocess_sequence(b[0]) for b in batch]
        labels    = [self.preprocess_labels(b[1]) for b in batch]

        ids = t5_tokenizer(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        masks = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embeddings = t5_model(input_ids=input_ids, attention_mask=masks)

        del input_ids
        embeddings = embeddings.last_hidden_state[:, :-1, :] # remove special tokens
        # embeddings = embeddings.last_hidden_state
        masks = masks[:, :-1] # remove special tokens

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).long()
        return embeddings.float(), labels, masks




#------------------------------------------------------------------------
## Dataset class


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, src_col="seq", tgt_col="dssp8", dataset_type="proteinnet"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[[src_col, tgt_col]]
        self.df = self.df.dropna()
        # print("[1]", csv_path, ":", len(self.df))
        self.src_col = src_col
        self.tgt_col = tgt_col

        if dataset_type == "netsurf":
            self.df[src_col] = self.df[self.src_col].apply(lambda s: s.replace(" ", ""))
            self.df[tgt_col] = self.df[self.tgt_col].apply(lambda s: s.replace(" ", ""))
            self.df[tgt_col] = self.df[self.tgt_col].apply(lambda s: s.replace("C", "L"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df[self.src_col].iloc[idx]
        label = self.df[self.tgt_col].iloc[idx]
        return sequence, label
    

if dataset_type == "netsurf":
    train_file_path = os.path.join(dataset_path, "netsurf", "train", "Train_HHblits.csv")
    val_file_path = os.path.join(dataset_path, "netsurf", "train", "Validation_HHblits.csv")
    testsample_path = os.path.join(dataset_path, "netsurf", "test", "CASP12_HHblits.csv")

elif dataset_type == "proteinnet":
    train_file_path = os.path.join(dataset_path, "proteinnet", "train", "train.csv")
    val_file_path = os.path.join(dataset_path, "proteinnet", "validation", "validation.csv")
    testsample_path = os.path.join(dataset_path, "proteinnet", "test", "test2018.csv")

train_ds = ProteinDataset(train_file_path, dataset_type=dataset_type)
val_ds = ProteinDataset(val_file_path, dataset_type=dataset_type)
testsample_ds = ProteinDataset(testsample_path, dataset_type=dataset_type)



print("train:", len(train_ds))
print("val:  ", len(val_ds))



#------------------------------------------------------------------------
## DataLoader

train_t5_collate = T5_collate(MAX_TRAIN_TOKENS)
t5_collate = T5_collate()


train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=train_t5_collate)
val_loader = torch.utils.data.DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=t5_collate)
testsample_loader = torch.utils.data.DataLoader(testsample_ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=t5_collate)


for batch in train_loader:
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    break

#------------------------------------------------------------------------
## PCA

sample_embeddings = []
emb_count = 0
emb_target = 50000

for batch in train_loader:
    lengths = torch.sum(batch[2], dim=-1)
    c = torch.sum(lengths).item()

    for i in range(lengths.shape[0]):
        sample_embeddings.append(batch[0][i][:lengths[i]])

    emb_count += c
    print("sample embeddings:", f"{emb_count:5d}/{emb_target}")
    if emb_count >= emb_target:
        break


sample_embeddings = torch.concat(sample_embeddings)
print("sample_embeddings:", sample_embeddings.shape)



pca = PCA(n_components=D_MODEL)
pca.fit(sample_embeddings.cpu())
pca_comp = np.asarray(pca.components_)
pca_comp_tensor = torch.tensor(pca_comp, dtype=torch.float32).to(device)

del sample_embeddings

print("pca:", pca_comp_tensor.shape)




#------------------------------------------------------------------------
# **Model Components**


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(
            torch.normal(0.0, 0.02, (MAX_SEQ_LEN, d_model)).unsqueeze(0)
        )

    def forward(self, x):
        # print("LPE:", x.shape, self.pe.shape, self.pe[:, :x.size(1)].shape)
        return x + self.pe[:, :x.size(1)]
    

# https://nlp.seas.harvard.edu/2018/04/03/attention.html
class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))
    


class ConvolutionModule(nn.Module):
    def __init__(self, dmodel):
        super(ConvolutionModule, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv1d(dmodel, dmodel*2, 7, padding="same", bias=False),
            nn.SiLU(),
            nn.BatchNorm1d(dmodel*2),
            nn.Conv1d(dmodel*2, dmodel, 5, padding="same", bias=False),
            nn.SiLU(),
            nn.BatchNorm1d(dmodel),
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)     # N,L,C -> N,C,L
        x = self.convblock(x)
        return x.permute(0, 2, 1)  # N,C,L -> N,L,C
    


# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.convmodule = ConvolutionModule(d_model)

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.mha(x, x, x, key_padding_mask=mask, need_weights=False)[0]
        x = self.norm1(x + self.dropout(attn_output))

        conv_output = self.convmodule(x)
        x = self.norm2(x + self.dropout(conv_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    



class TransformerEncoderModel(nn.Module):
    def __init__(self,
        d_model, num_heads, num_layers,
        d_ff,
        dropout=0.1
    ):
        super(TransformerEncoderModel, self).__init__()
        self.linear_resize = nn.Linear(1024, d_model, bias=False)
        self.linear_resize.weight = nn.Parameter(pca_comp_tensor)
            
        self.positional_encoding = LearnablePositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, tgt_vocab_size)

    def forward(self, src, mask):
        # For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention.
        mask = mask == 0

        x = self.linear_resize(src)
        x = self.dropout(self.positional_encoding(x))

        for enc_layer in self.encoder_layers:
            x = enc_layer(x, mask)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

#------------------------------------------------------------------------
# **Training**

tgt_vocab_size = len(dssp_vocab)
print(tgt_vocab_size)

num_layers = args.num_layers
num_heads = args.num_heads
d_model = D_MODEL
dff = d_model*4
dropout = args.dropout

#------------------------------------------------------------------------
model = TransformerEncoderModel(
    d_model,
    num_heads,
    num_layers,
    dff,
    dropout=dropout
)

model = model.to(device)
print(model)

#------------------------------------------------------------------------
def masked_acc(label, pred):
    pred = torch.argmax(pred, axis=2)
    match = label == pred
    mask = label != 0
    match = match & mask

    # match = match.type(torch.float32)
    # mask = mask.type(torch.float32)
    return match.sum()/mask.sum()

#------------------------------------------------------------------------
for x, y, mask in train_loader:
    x = x.to(device)
    y = y.to(device)

    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)
    break

outputs = model(x, mask)

print("output shape:", outputs.shape, outputs.dtype)
print("acc:", masked_acc(y, outputs))

del x, y, mask, outputs

#------------------------------------------------------------------------
# https://kikaben.com/transformers-training-details/
class Scheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self,
                 optimizer,
                 dim_embed,
                 warmup_steps,
                 steps_in_epoch,
                 last_epoch=-1,
                 verbose=False):

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        self._step_count = (last_epoch+1)*steps_in_epoch

    def get_lr(self):
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
#------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss(
    ignore_index=0,             # 0 is padding token
    label_smoothing = 0.1
)

val_loss_fn = nn.CrossEntropyLoss(
    ignore_index=0,             # 0 is padding token
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-9,
    weight_decay=1e-4
)


#------------------------------------------------------------------------
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#preallocate-memory-in-case-of-variable-input-length
# preallocate memory by operating on randomly generated batch of max sequence length
x = torch.rand(BATCH_SIZE, 1024, 1024, dtype=torch.float32, device=device)
y = torch.ones(BATCH_SIZE, 1024, dtype=torch.int64, device=device)
mask = torch.ones(BATCH_SIZE, 1024, device=device)

outputs = model(x, mask)

loss = loss_fn(outputs.contiguous().view(-1, tgt_vocab_size), y.contiguous().view(-1))
loss.backward()

del x, mask, outputs, loss

optimizer.zero_grad(set_to_none=True)

#------------------------------------------------------------------------
checkpoint_dir = os.path.join(checkpoint_base_dir, f"transconv-{dataset_type}-{num_layers}-{num_heads}-{d_model}-{dff}-{dropout}-bs{BATCH_SIZE}-wrm{args.warmup}")

os.makedirs(checkpoint_dir, exist_ok=True)
print(checkpoint_dir)

checkpoint_filename = "model_checkpoint.pth"
best_checkpoint_filename = "model_best_checkpoint.pth"


# load checkpoint if exists
prev_epoch = -1     # 0 indexed, -1 is no epochs previously
save_file = os.path.join(checkpoint_dir, checkpoint_filename)
if os.path.isfile(save_file):
    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    prev_epoch = checkpoint['epoch']
    print("loaded epoch",  prev_epoch, "from", save_file)



# warmup_steps = len(train_loader)*5
warmup_steps = len(train_loader)*(args.warmup)
print("warmup_steps:", warmup_steps)

scheduler = Scheduler(
    optimizer,
    dim_embed=d_model,
    warmup_steps=warmup_steps,
    steps_in_epoch=len(train_loader),
    last_epoch=prev_epoch
)

# if prev_epoch>=0:
#     for _ in range((prev_epoch+1)*len(train_loader)):
#         scheduler.step()

#------------------------------------------------------------------------
def train_epoch(train_dl, epoch, num_epochs):
    
    n_total_steps = len(train_dl)
    print_n_steps = 10
    total_train_loss = 0
    total_train_acc = 0
    steps = 0

    
    start_time = time.time()
    model.train()
    for x, y, mask in train_dl:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True) # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

        outputs = model(x, mask)
        del x, mask

        # Compute the loss and its gradients
        loss = loss_fn(outputs.contiguous().view(-1, tgt_vocab_size), y.contiguous().view(-1))

        loss.backward()

        # Compute masked accuracy
        acc = masked_acc(y, outputs)

        # Adjust learning weights
        optimizer.step()
        scheduler.step()

        total_train_loss += loss
        total_train_acc += acc

        steps+=1
        if steps%print_n_steps == 0 or steps==1 or steps==n_total_steps:
            elapsed_time = int(((time.time() - start_time)*1000)/steps)
            
            acc_number = acc.detach().item()
            avg_loss = total_train_loss.detach().item()/steps
            avg_acc = total_train_acc.detach().item()/steps

            disp_lr = get_lr(optimizer)
            print(f"Epoch {epoch+1:2d}/{num_epochs} [{steps:4d}/{n_total_steps:4d}] {elapsed_time:3d}ms/step  loss: {avg_loss:.4f}  acc: {avg_acc:.4f}  {acc_number:.4f}  lr:{disp_lr:.7f}")
                    
        
    avg_loss = total_train_loss.detach().item()/n_total_steps
    avg_acc = total_train_acc.detach().item()/n_total_steps
    
    return avg_loss, avg_acc



#------------------------------------------------------------------------
def val_epoch(val_dl):
    
    n_val_steps = len(val_dl)
    total_val_loss = 0
    total_val_acc = 0

    model.eval()
    with torch.no_grad():
        outputs = None
        for x, y, mask in val_dl:
            x = x.to(device)
            y = y.to(device)

            # with torch.cuda.amp.autocast():
            outputs = model(x, mask)
            del x, mask

            # Compute the loss and accuracy
            val_loss = val_loss_fn(outputs.contiguous().view(-1, tgt_vocab_size), y.contiguous().view(-1))

            val_acc = masked_acc(y, outputs)

            total_val_loss += val_loss
            total_val_acc  += val_acc

        sample_label = detokenize_label(y[0])
        sample_out = torch.argmax(outputs[0], axis=-1)
        sample_out   = detokenize_label(sample_out)
        print("=", sample_label)
        print("<", sample_out)


    avg_loss = total_val_loss.detach().item() / n_val_steps
    avg_acc  = total_val_acc.detach().item()  / n_val_steps
    
    return avg_loss, avg_acc


#------------------------------------------------------------------------
num_epochs = 50
patience = 10

max_val_acc = 0
min_val_loss = float("inf")
patience_count = 0


for epoch in range(prev_epoch+1, num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')

    train_loss, train_acc = train_epoch(train_loader, epoch, num_epochs)
    val_loss, val_acc = val_epoch(val_loader)
      


    
    print(f"Training - loss: {train_loss:.4f}   acc: {train_acc:.4f}      Validation - loss: {val_loss:.4f}   acc: {val_acc:.4f}  [{max(val_acc, max_val_acc)}]")
    
    
    testsample_loss, testsample_acc = val_epoch(testsample_loader)
    print(f"Test2018 - loss: {testsample_loss:.4f}   acc: {testsample_acc:.4f}")
    
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, checkpoint_filename)
    )

    # Save best till now
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, best_checkpoint_filename)
        )

    # save metrics in log file
    logfilepath = os.path.join(checkpoint_dir,"logs.csv")
    if not os.path.isfile(logfilepath):
        with open(logfilepath, "a") as logfile:
            logfile.write("epoch,loss,acc,val_loss,val_acc,testsample_loss,testsample_acc\n")

    with open(logfilepath, "a") as logfile:
        logfile.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{testsample_loss},{testsample_acc}\n")

    # Early Stopping
    if patience>=0:
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_count = 0
        else:
            patience_count+=1
            if patience_count>patience:
                break

    if epoch>0 and val_acc < 0.5:
        break

#------------------------------------------------------------------------
# **Evaluate**

def evaluate(test_loader_obj):
    total_test_loss = 0
    total_test_acc = 0
    n_total_steps = len(test_loader_obj)

    model.eval()
    with torch.no_grad():
        steps = 0
        for x, y, mask in test_loader_obj:
            steps+=1

            x = x.to(device)
            y = y.to(device)

            outputs = model(x, mask)
            del x, mask

            # Compute the loss and accuracy
            test_loss = val_loss_fn(outputs.contiguous().view(-1, tgt_vocab_size), y.contiguous().view(-1))
            test_acc = masked_acc(y, outputs)

            test_loss_number = test_loss.detach().item()
            test_acc_number = test_acc.detach().item()

            total_test_loss += test_loss_number
            total_test_acc  += test_acc_number

            if steps==1 or steps%100==0:
                print(f"[{steps:4d}/{n_total_steps:4d}] loss: {(total_test_loss/steps):.4f}  acc: {(total_test_acc/steps):.4f}")

        total_test_loss   /= n_total_steps
        total_test_acc    /= n_total_steps

        print(f"loss: {total_test_loss:.4f}   acc: {total_test_acc:.4f}")
        return  total_test_loss, total_test_acc
    

if dataset_type == "proteinnet":
    test_filenames = ["test2018.csv", "SPOT-2016.csv", "SPOT-2016-HQ.csv", "SPOT-2018.csv", "SPOT-2018-HQ.csv"]
if dataset_type == "netsurf":
    test_filenames = ["CASP12_HHblits.csv", "CB513_HHblits.csv", "NEW364.csv", "TS115_HHblits.csv"]


for test_filename in test_filenames:
    test_ds = ProteinDataset(os.path.join(dataset_path, dataset_type, "test", test_filename), dataset_type=dataset_type)
    print(test_filename, ":", len(test_ds))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, collate_fn=t5_collate)

    loss, acc = evaluate(test_loader)

    # save metrics in log file
    evalfilepath = os.path.join(checkpoint_dir,"eval.csv")
    if not os.path.isfile(evalfilepath):
        with open(evalfilepath, "a") as logfile:
            logfile.write("filename,loss,acc\n")

    with open(evalfilepath, "a") as logfile:
        logfile.write(f"{test_filename},{loss},{acc}\n")




#------------------------------------------------------------------------
# **Best Model Evaluate**

# load checkpoint if exists
best_epoch = -1     # 0 indexed, -1 is no epochs previously
save_file = os.path.join(checkpoint_dir, best_checkpoint_filename)
if os.path.isfile(save_file):
    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_epoch = checkpoint['epoch']
    print("loaded best epoch",  best_epoch, "from", save_file)



if dataset_type == "proteinnet":
    test_filenames = ["test2018.csv", "SPOT-2016.csv", "SPOT-2016-HQ.csv", "SPOT-2018.csv", "SPOT-2018-HQ.csv"]
if dataset_type == "netsurf":
    test_filenames = ["CASP12_HHblits.csv", "CB513_HHblits.csv", "NEW364.csv", "TS115_HHblits.csv"]


for test_filename in test_filenames:
    test_ds = ProteinDataset(os.path.join(dataset_path, dataset_type, "test", test_filename), dataset_type=dataset_type)
    print(test_filename, ":", len(test_ds))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, collate_fn=t5_collate)

    loss, acc = evaluate(test_loader)

    # save metrics in log file
    evalfilepath = os.path.join(checkpoint_dir,"best_eval.csv")
    if not os.path.isfile(evalfilepath):
        with open(evalfilepath, "a") as logfile:
            logfile.write("filename,loss,acc\n")

    with open(evalfilepath, "a") as logfile:
        logfile.write(f"{test_filename},{loss},{acc}\n")