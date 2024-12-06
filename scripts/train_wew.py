#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../cace/')

import numpy as np
import torch
import torch.nn as nn
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
import mace
from mace.calculators import LAMMPS_MACE

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 5.5

logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='../water.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'force'}, 
                                 atomic_energies={1: -187.6043857100553, 8: -93.80219285502734} # avg
                                 )
batch_size = 2

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=4,
                              )

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

#MACE uses ACE represntation so here its just ACE

MACE_representation = MACE(
    r_max = 5.5,
    num_bessel =,
    num_polynomial_cutoff: int,
    max_ell: int,
    interaction_cls: Type[InteractionBlock],
    interaction_cls_first: Type[InteractionBlock],
    num_interactions: int,
    num_elements: int,
    hidden_irreps: o3.Irreps,
    MLP_irreps: o3.Irreps,
    atomic_energies: np.ndarray,
    avg_num_neighbors: float,
    atomic_numbers: List[int],
    correlation: Union[int, List[int]],
    gate: Optional[Callable],
    pair_repulsion: bool = False,
    distance_transform: str = "None",
    radial_MLP: Optional[List[int]] = None,
    radial_type: Optional[str] = "bessel",
    heads: Optional[List[str]] = None,
    cueq_config: Optional[Dict[str, Any]] = None,)

MACE_representation.to(device)

cace_representation = Cace()

cace_representation.to(device)

logging.info(f"Representation: {MACE_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)


forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

#here NeuralNetwork becomes MACE potential

logging.info("building CACE NNP")
MACE_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=MACE_representation,
    output_modules=[atomwise, forces]
)



q = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=4,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.EwaldPotential(dl=2,
                    sigma=1.,
                    feature_key='q',
                    output_key='ewald_potential',
                    remove_self_interaction=False,
                   aggregation_mode='sum')

forces_lr = cace.modules.Forces(energy_key='ewald_potential',
                                    forces_key='ewald_forces')

#now here ep is given by an ewald summation in recirpocal space so basically the network should stay that neural network potential with ewald summation => can i use cace represnetaion still ???

cace_nnp_lr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, ep, forces_lr]
)

pot2 = {'CACE_energy': 'ewald_potential', 
        'CACE_forces': 'ewald_forces',
        'weight': 0.01
       }

pot1 = {'CACE_energy': 'CACE_energy', 
        'CACE_forces': 'CACE_forces',
       }

#the combine potential is a fucntion that sums ep from MACE and ep from NN cace ewald 

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1,pot2])
cace_nnp.to(device)

#check the losses to make sure not to worry 

logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False, #True,
        ema_start=10,
        warmup_steps=5,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False, val_stride=10)

task.save_model('water-model.pth')
cace_nnp.to(device)

logging.info(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

task.update_loss([energy_loss, force_loss])
logging.info("training")
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, val_stride=10)


task.save_model('water-model-2.pth')
cace_nnp.to(device)

logging.info(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

task.save_model('water-model-3.pth')

logging.info(f"Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

task.save_model('water-model-4.pth')

logging.info(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")


