import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from torch.utils.data import DataLoader
from strategyqa_dataset import StrategyQADataset # You would need to implement this dataset class

# Load pretrained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Fine-tuning parameters
batch_size = 8
num_epochs = 3
learning_rate = 2e-5

# Prepare StrategyQA dataset
train_dataset = StrategyQADataset(tokenizer, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

    # Adjust learning rate
    scheduler.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert_strategyqa")