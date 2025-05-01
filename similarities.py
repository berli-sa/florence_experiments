import torch
import torch.nn.functional as F

path = 'embeddings/book/0001/0001_0.pt'
import argparse

# --------- Methods ---------

def mean_pooling_cosine(t1, t2):
    t1_pooled = t1.mean(dim=0)
    t2_pooled = t2.mean(dim=0)
    return F.cosine_similarity(t1_pooled, t2_pooled, dim=0).item()

def cross_similarity(t1, t2):
    t1_norm = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-8)
    t2_norm = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(t1_norm, t2_norm.T)  # (len1, len2)
    final_score = sim_matrix.max(dim=1)[0].mean()
    return final_score.item()

def max_pool_cosine(t1, t2):
    t1_pooled = t1.max(dim=0)[0]
    t2_pooled = t2.max(dim=0)[0]
    return F.cosine_similarity(t1_pooled, t2_pooled, dim=0).item()

def simple_linear_projection(t1, t2, output_dim=5):
    linear = torch.nn.Linear(t1.size(-1), output_dim)
    with torch.no_grad():
        t1_proj = linear(t1.mean(dim=0))
        t2_proj = linear(t2.mean(dim=0))
    return F.cosine_similarity(t1_proj, t2_proj, dim=0).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-similar', action='store_false', dest='similar')
    parser.set_defaults(similar=True)
    parser.add_argument('--no-close', action='store_false', dest='close')
    parser.set_defaults(close=True)
    parser.add_argument('--no-diff', action='store_false', dest='diff')
    parser.set_defaults(diff=False)
    parser.add_argument('--dim1', type=int, default=7)
    parser.add_argument('--dim2', type=int, default=6)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    # --------- Test all methods ---------

    # Set seed for reproducibility
    if args.seed != None:
        torch.manual_seed(args.seed)

    # --------- Create test tensors ---------

    # Similar pair
    dim1_1 = args.dim1
    dim1_2 = args.dim2
    tensor1_similar = torch.randn(dim1_1, 5) 
    tensor2_similar = tensor1_similar[(dim1_1-dim1_2)::, :] + 0.05 * torch.randn(dim1_2, 5)  # noisy copy

    # Close but not identical
    dim1_1 = args.dim1+2
    dim1_2 = args.dim2
    print(f"Close Example Differenct of {dim1_1 - dim1_2}")
    tensor1_close = torch.randn(dim1_1, 5)
    tensor2_close = tensor1_close[(dim1_1-dim1_2)::, :] * 0.8 + 0.2 * torch.randn(dim1_2, 5)

    # Very different
    dim1_1 = args.dim1
    dim1_2 = args.dim2
    tensor1_diff = torch.randn(dim1_1, 5)
    tensor2_diff = torch.randn(dim1_2, 5)  # independent random

    pairs = [
        ("similar", tensor1_similar, tensor2_similar),
        ("close", tensor1_close, tensor2_close),
        ("different", tensor1_diff, tensor2_diff),
    ]

    for label, t1, t2 in pairs:
        print(f"\n--- {label.upper()} PAIR ---")
        print(f"Mean Pool Cosine: {mean_pooling_cosine(t1, t2):.4f}")
        print(f"Cross Similarity Max-Average: {cross_similarity(t1, t2):.4f}")
        print(f"Max Pool Cosine: {max_pool_cosine(t1, t2):.4f}")
        print(f"Linear Projection Cosine: {simple_linear_projection(t1, t2):.4f}")

if __name__ == '__main__':
    main()

# embedding_1 = torch.load(path)['cropped_image_embedding']
# embedding_2 = torch.load(path)['label_embedding']
# label = torch.load(path)['label']
# print(embedding_1.shape)
# print(embedding_2.shape)
# print(label)

# cos_sim = F.cosine_similarity(embedding_1, embedding_2)
# print(f"Cosine Similarity: {cos_sim.item()}")
