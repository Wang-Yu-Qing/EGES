import dgl
from torch.utils.data import DataLoader
import torch as th

import utils
from model import EGES
from sampler import Sampler


if __name__ == "__main__":
    args = utils.init_args()

    valid_sku_raw_ids = utils.get_valid_sku_set(args.item_info_data)

    g, sku_encoder, sku_decoder = utils.construct_graph(
        args.action_data, 
        args.session_interval_sec,
        valid_sku_raw_ids
    )

    sku_info_encoder, sku_info_decoder, sku_info = \
        utils.encode_sku_fields(args.item_info_data, sku_encoder, sku_decoder)

    um_skus = len(sku_encoder)
    num_brands = len(sku_info_encoder["brand"])
    num_shops = len(sku_info_encoder["shop"])
    num_cates = len(sku_info_encoder["cate"])

    print(
        "Num skus: {}, num brands: {}, num shops: {}, num cates: {}".\
            format(num_skus, num_brands, num_shops, num_cates)
    )

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
        collate_fn=lambda x: sampler.sample(x, sku_info)
    )

    model = EGES(args.dim, num_skus, num_brands, num_shops, num_cates)

    for i, (srcs, dsts, labels) in enumerate(dataloader):
        # the batch size of output pairs is unfixed
        # TODO: shuffle the triples?
        logits = model(srcs, dsts)




    

    

