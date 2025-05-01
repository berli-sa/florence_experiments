from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def next_word_distribution(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
    next_token_logits = logits[:, -1, :]  # get logits for next token prediction
    probs = F.softmax(next_token_logits, dim=-1)  # convert to probabilities
    return probs.squeeze()

def main():

    # Example sentences
    sentence1 = "The cat sat on the"
    sentence2 = "The brown dog lay on the"
    sentence3 = "Quantum mechanics describes the use of the"
    sentence4 = "Lizards lay in the"

    # Get next word distributions
    p1 = next_word_distribution(sentence1)
    p2 = next_word_distribution(sentence2)
    p3 = next_word_distribution(sentence3)
    p4 = next_word_distribution(sentence4)

    # Cosine similarity between predictions
    sim_12 = F.cosine_similarity(p1, p2, dim=0)
    sim_13 = F.cosine_similarity(p1, p3, dim=0)
    sim_14 = F.cosine_similarity(p1, p4, dim=0)

    print(f"Similarity (sentence1 vs sentence2): {sim_12.item():.4f}")
    print(f"Similarity (sentence1 vs sentence3): {sim_13.item():.4f}")
    print(f"Similarity (sentence1 vs sentence4): {sim_14.item():.4f}")

if __name__ == "__main__":
    main()