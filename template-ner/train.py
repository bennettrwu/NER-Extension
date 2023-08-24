import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import AdamW, BartForConditionalGeneration, BartTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda import amp
from tqdm.auto import tqdm

epochs = 20
batch_size = 100
manual_seed = 4
learning_rate = 4e-05
adam_epsilon = 1e-08
warmup_ratio = 0.06

best_model_save_dir = os.path.join(os.path.dirname(__file__), 'best_model')

# Set random seed manually
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load pretrained bart model and tokenizer
model = 'facebook/bart-large'
print(f'Loading {model} model...')
model = BartForConditionalGeneration.from_pretrained(model)
tokenizer = BartTokenizer.from_pretrained(model)
model.to(device)
print('Loaded!')


class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_path):
        data = pd.read_csv(data_path)

        data['Source_tokens'] = data['Source sentence'].apply(
            lambda text:
            tokenizer.batch_encode_plus(
                [text],
                max_length=50,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        )

        data['Target_tokens'] = data['Answer sentence'].apply(
            lambda text:
            tokenizer.batch_encode_plus(
                [text],
                max_length=50,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        )

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'source_ids': self.data.iloc[index]['Source_tokens']['input_ids'].squeeze(),
            'source_mask': self.data.iloc[index]['Source_tokens']['attention_mask'].squeeze(),
            'target_ids': self.data.iloc[index]['Target_tokens']['input_ids'].squeeze(),
        }


print('Preparing training dataset...')
train_dataset = CustomDataset(
    tokenizer,
    os.path.join(os.path.dirname(__file__), 'dataset', 'train.csv')
)
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size,
)
print('Training dataset ready!')

print('Preparing validation dataset...')
validation_dataset = CustomDataset(
    tokenizer,
    os.path.join(os.path.dirname(__file__), 'dataset', 'valid.csv')
)
validation_dataloader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=batch_size
)
print('Validation dataset ready!')


optimizer = AdamW([{
    "params": [
        p
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])
    ],
    "weight_decay": 0.0,
}], lr=learning_rate, eps=adam_epsilon)

total_steps = epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=np.ceil(warmup_ratio * total_steps),
    num_training_steps=total_steps
)
scaler = amp.GradScaler()


def train_loop(model: BartForConditionalGeneration, train_dataloader: DataLoader, optimizer: AdamW, scheduler, scaler: amp.GradScaler):
    model.zero_grad()
    model.train()

    iterator = tqdm(train_dataloader)
    for inputs in iterator:
        pad_token_id = tokenizer.pad_token_id
        target_ids = inputs['target_ids'][:, :-1].contiguous()
        labels = inputs['target_ids'][:, 1:].clone()
        labels[labels == pad_token_id] = -100

        with amp.autocast():
            outputs = model(
                input_ids=inputs['source_ids'].to(device),
                attention_mask=inputs['source_mask'].to(device),
                decoder_input_ids=target_ids.to(device),
                labels=labels.to(device)
            )
            loss = outputs[0]

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        model.zero_grad()
        iterator.set_description(
            f'Training Model   - Current Loss: {loss.item():9.4f}'
        )


def validate_model(model: BartForConditionalGeneration, validation_dataloader: DataLoader):
    model.eval()

    total_loss = 0.0
    batchs_count = 0
    correct = 0
    output_count = 0
    iterator = tqdm(validation_dataloader)
    for inputs in iterator:
        pad_token_id = tokenizer.pad_token_id
        target_ids = inputs['target_ids'][:, :-1].contiguous()
        labels = inputs['target_ids'][:, 1:].clone()
        labels[labels == pad_token_id] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=inputs['source_ids'].to(device),
                attention_mask=inputs['source_mask'].to(device),
                decoder_input_ids=target_ids.to(device),
                labels=labels.to(device)
            )
            loss = outputs[0]
            total_loss += loss.item()
            batchs_count += 1

            decoded_outputs = torch.argmax(outputs[1], dim=-1).view(-1)
            labels = labels.view(-1)
            for i, j in zip(labels, decoded_outputs):
                if i == -100:
                    continue

                output_count += 1
                if i == j:
                    correct += 1

        iterator.set_description(
            f'Evaluating Model - Current Loss: {loss.item():9.4f}'
        )

    avg_loss = total_loss / batchs_count
    accuracy = correct / output_count
    print(f'Validation Loss: {avg_loss}')
    print(f'Validation Accuracy: {accuracy}')
    return avg_loss, accuracy


best_accuracy = 0
for epoch in range(epochs):
    print(f'\nEpoch: {epoch} of {epochs}')
    train_loop(model, train_dataloader, optimizer, scheduler, scaler)
    avg_loss, accuracy = validate_model(model, validation_dataloader)

    # Save model with best accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f'Saving best model to: {best_model_save_dir}')
        model.save_pretrained(best_model_save_dir)
        tokenizer.save_pretrained(best_model_save_dir)
        torch.save(
            optimizer.state_dict(),
            os.path.join(best_model_save_dir, "optimizer.pt")
        )
        torch.save(
            scheduler.state_dict(),
            os.path.join(best_model_save_dir, "scheduler.pt")
        )
