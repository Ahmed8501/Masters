import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm
from Source.contrastive_dataset import ContrastiveNL2SQLDataset

#  Settings
model_name = "microsoft/MiniLM-L6-v2"
batch_size = 64
epochs = 3
lr = 2e-5
max_len = 128
dataset_path = "data/contrastive_pairs.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)

#  Load Dataset
dataset = ContrastiveNL2SQLDataset(dataset_path, tokenizer, max_len=max_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#  Define Contrastive Head
class LCSPScorer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :]  # use [CLS] token
        return self.linear(cls).squeeze(-1)

scorer = LCSPScorer(model).to(device)
optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

#  Train
for epoch in range(epochs):
    scorer.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        scores = scorer(input_ids, attention_mask)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

#  Save
torch.save(scorer.state_dict(), "Models/lcsp_encoder.pt")
print("Model saved to Models/lcsp_encoder.pt")
