import torch as th


class EGES(th.nn.Module):
    def __init__(self, dim, num_nodes, num_brands, num_shops, num_cates):
        super(EGES, self).__init__()
        self.dim = dim
        self.base_embedding = th.nn.Embedding(num_nodes, dim)
        self.brand_embedding = th.nn.Embedding(num_brands, dim)
        self.shop_embedding = th.nn.Embedding(num_shops, dim)
        self.cate_embedding = th.nn.Embedding(num_cates, dim)

    def forward(self, srcs, dsts):
        print(srcs)
        exit(0)

    def loss(self):
        pass

    def query_cold_item(self):
        pass
    


