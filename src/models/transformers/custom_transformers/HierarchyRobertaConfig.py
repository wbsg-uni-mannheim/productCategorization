from transformers import RobertaConfig

class HierarchyRobertaConfig(RobertaConfig):
    r"""
    Defines additional configuration attributes for the network to exploit the label hierarchy

    Args:
        tree(:obj:`DIGraph`, `optional`, defaults to :obj:`None`):
            Output Hierarchy Tree

    """
    model_type = "hierarchy-roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.tree = kwargs.pop("tree", None)


