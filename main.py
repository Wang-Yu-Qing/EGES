import dgl
from utils import *
from torch.utils.data import DataLoader
import torch as th


if __name__ == "__main__":
    args = init_args()

    g, sku_encoder, sku_decoder = construct_graph(
            args.action_data, 
            args.session_interval_sec
            )

    sku_info_encoder, sku_info_decoder, sku_info = \
            parse_sku_info(args.item_info_data, sku_encoder, sku_decoder)

    exit(0)

    sampler = Sampler(
            g, 
            args.walk_length, 
            args.num_walks, 
            args.window_size, 
            args.num_negative
            )

    # for each node in the graph, we sample pos and neg
    # pairs for it, and feed these sampled pairs into the model.
    # (nodes in the graph are of course batched before sampling)
    dataloader = DataLoader(
            th.arange(g.num_nodes()),
            # this is the batch_size of input nodes
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=sampler.sample
            )

    for i, batch_pairs in enumerate(dataloader):
        # the batch size of output pairs is unfixed
        # shuffle the pairs
        indexes = th.randperm(batch_pairs.shape[0])
        batch_pairs = batch_pairs[indexes]


    

    

