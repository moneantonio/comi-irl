import torch as th # type: ignore[import]
import torch.nn.functional as F # type: ignore[import]
from tqdm import tqdm # type: ignore[import]

def inference(
    encoder, dataloader, trajectory_manager, device, index_offset=0
):
    """
    Runs inference for the encoder model, extracting CLS embeddings.
    Cluster labels will be assigned later via HDBSCAN.
    """
    encoder.eval()
    
    all_concatenations = []
    all_indices = []

    with th.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # Handle both 5-element and 6-element batches
            if len(batch) == 6:
                states, actions, masks, _, indices_batch, _ = [b.to(device) for b in batch]
            else:
                states, actions, masks, _, indices_batch = [b.to(device) for b in batch]
            
            _, _, _, _, cls_emb, _ = encoder(states, actions, src_key_padding_mask=masks)
            cls_emb = F.normalize(cls_emb, p=2, dim=1)
            
            all_concatenations.append(cls_emb.cpu())
            all_indices.append(indices_batch.cpu() + index_offset)

            for i in range(len(indices_batch)):
                traj_idx = indices_batch[i].item() + index_offset
                trajectory_manager[traj_idx]['cls_emb'] = cls_emb[i].cpu().numpy()
                trajectory_manager[traj_idx]['predicted_cluster_label'] = -1  # Placeholder for HDBSCAN
                trajectory_manager[traj_idx]['concatenation'] = cls_emb[i].cpu().numpy()

    final_concatenations = th.cat(all_concatenations, dim=0)
    final_indices = th.cat(all_indices, dim=0).numpy()

    return trajectory_manager, final_concatenations, final_indices