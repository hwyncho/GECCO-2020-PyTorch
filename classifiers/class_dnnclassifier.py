"""Module containing the DNN classifiers class."""
import torch


class DNNClassifier(torch.nn.Module):
    """Class that defines the DNN classifiers."""

    def __init__(self, size_features: int, num_hidden_layers: int, size_labels: int, **kwargs):
        """
        Function to initialize the object.

        Parameters
        ----------
        num_features: int
        num_hidden_layers: int
        num_labels: int

        """
        super(DNNClassifier, self).__init__()

        assert isinstance(size_features, int) and (size_features > 0)
        assert isinstance(num_hidden_layers, int) and (num_hidden_layers > 0)
        assert isinstance(size_labels, int) and (size_labels > 0)

        list_num_nodes: list = torch.linspace(start=size_features,
                                              end=size_labels,
                                              steps=num_hidden_layers + 2,
                                              requires_grad=False).int().tolist()[1:-1]

        self._nn: torch.nn.Sequential = torch.nn.Sequential()
        idx: int = 1
        for (i, o) in zip([size_features] + list_num_nodes[:-1], list_num_nodes):
            self._nn.add_module(name="linear_{0}".format(idx),
                                module=torch.nn.Linear(in_features=i, out_features=o, bias=True))
            self._nn.add_module(name="relu_{0}".format(idx),
                                module=torch.nn.ReLU(inplace=True))
            idx = idx + 1
        self._nn.add_module(name="linear_{0}".format(idx),
                            module=torch.nn.Linear(in_features=list_num_nodes[-1],
                                                   out_features=size_labels,
                                                   bias=True))

    def forward(self, x: torch.Tensor, **kwargs):
        """Function to perform computation."""
        return self._nn(x)
