import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import nltk
import pickle
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

# Ensure nltk tokenizer is downloaded
nltk.download('punkt')

# Paths
DATA_DIR = './flickr8k' 
IMAGES_DIR = os.path.join(DATA_DIR, 'Images')
CAPTIONS_FILE = os.path.join(DATA_DIR, 'captions.txt')
MODEL_PATH = './flickr8k/mymodel_vgg16.pth' 
VOCAB_PATH = './flickr8k/vocab_vgg16.pkl' 

# Hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
LEARNING_RATE = 0.0003
BATCH_SIZE = 16
EPOCHS = 1
MAX_LENGTH = 40
CLIP_GRAD = 5.0

# Tokenizer (Build Vocabulary)
def build_vocab(caption_file, threshold=5):
    with open(caption_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    counter = Counter()
    for line in lines:
        tokens = nltk.word_tokenize(line.split(',')[1].lower())
        counter.update(tokens)
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = {word: idx for idx, word in enumerate(words, 4)}
    vocab.update({'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3})
    
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

# Dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, caption_file, vocab, transform=None):
        self.img_dir = img_dir
        self.captions = open(caption_file, 'r').readlines()[1:]
        self.vocab = vocab
        self.transform = transform
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        line = self.captions[idx].strip().split(',')
        img_path = os.path.join(self.img_dir, line[0])
        caption = nltk.word_tokenize(line[1].lower())
        
        caption = [self.vocab.get(word, self.vocab['<unk>']) for word in caption]
        caption = [self.vocab['<start>']] + caption + [self.vocab['<end>']]
        caption = caption[:MAX_LENGTH] + [self.vocab['<pad>']] * (MAX_LENGTH - len(caption))
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(caption), len(caption)

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
vocab = build_vocab(CAPTIONS_FILE)
dataset = Flickr8kDataset(IMAGES_DIR, CAPTIONS_FILE, vocab, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder Model
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, embed_size)

    def forward(self, images):
        features = self.features(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        return self.fc(features)

# Decoder Model
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions, lengths):
        embeddings = self.dropout(self.embedding(captions))
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        return self.fc(lstm_out[0])

# Model Initialization
encoder = EncoderCNN(EMBED_SIZE).to(device)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scaler = GradScaler()

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for images, captions, lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, captions = images.to(device, non_blocking=True), captions.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with autocast():
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}')
    scheduler.step()

# Save model
torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')

