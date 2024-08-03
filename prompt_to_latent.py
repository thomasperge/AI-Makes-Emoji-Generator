from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Charger le modèle BERT pré-entraîné et le tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def prompt_to_latent(prompt, latent_dim):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = bert_model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # Utiliser la moyenne des embeddings de la dernière couche comme vecteur latent
    latent_vector = torch.mean(last_hidden_states, dim=1)
    # Redimensionner ou ajuster pour correspondre à la dimension du vecteur latent du GAN
    latent_vector = F.interpolate(latent_vector.unsqueeze(0), size=(latent_dim,))
    return latent_vector.squeeze(0)
