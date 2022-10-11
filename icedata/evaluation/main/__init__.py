import common

from torch.optim.adam import Adam
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR


class Vals_MAIN(common.Vals):
    def __init__(self, args: common.Args, *, nb_nearest_neighbours: int = 8):
        print('Vals_MAIN')
        self.training_dataloader, self.validation_dataloader, self.test_dataloader = \
            common.get_dataloaders(args)

        self.detector = IceCubeDeepCore(
            graph_builder=KNNGraphBuilder(nb_nearest_neighbours=nb_nearest_neighbours)
        )
        self.gnn = DynEdge(nb_inputs=self.detector.nb_outputs)
        self.task = common.get_task(target=args.target, gnn=self.gnn)

        self.model = Model(
            detector=self.detector,
            gnn=self.gnn,
            tasks=[self.task],
            optimizer_class=Adam,
            optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
            scheduler_class=PiecewiseLinearLR,
            scheduler_kwargs={
                'milestones': [0, len(self.training_dataloader) / 2, len(self.training_dataloader) * args.max_epochs],
                'factors': [1e-2, 1, 1e-02],
            },
            scheduler_config={
                'interval': 'step',
            },
        )
