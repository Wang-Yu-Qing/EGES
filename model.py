import torch as th


class EGES(th.nn.Module):
    def __init__(self, dim):
        super(EGES, self).__init__()
        self.dim = dim


    def forward(self, pair):
        """
            @pair: batch of: (target_id, context_id, label)
        """
        pass

    def loss(self):
        pass

    def query_cold_item(self):
        pass
    


