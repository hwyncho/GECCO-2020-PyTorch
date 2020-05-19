"""Module containing the functions."""
import copy
import os
import time

import numpy as np
import torch


def train(generator: torch.nn.Module,
          discriminator: torch.nn.Module,
          x: torch.Tensor,
          y: torch.Tensor,
          latent_size: int,
          batch_size: int = 16,
          num_epochs: int = 2,
          run_device: str = "cpu",
          learning_rate: float = 0.001,
          beta_1: float = 0.9,
          beta_2: float = 0.999,
          rand_seed: int = 0,
          verbose: bool = False) -> (torch.nn.Module, torch.nn.Module):
    """
    Function to train GAN.

    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    x: torch.Tensor
    y: torch.Tensor
    latent_size: int
    batch_size: int
    num_epochs: int
    run_device: str
    learning_rate: float
    beta_1: float
    beta_2: float
    rand_seed: int
    verbose: bool

    Returns
    -------
    (torch.nn.Module, torch.nn.Module)

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(latent_size, int) and (latent_size > 0)
    assert isinstance(batch_size, int) and (batch_size > 0)
    assert isinstance(num_epochs, int) and (num_epochs > 0)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(learning_rate, float) and (learning_rate > 0.0)
    assert isinstance(beta_1, float) and (0.0 <= beta_1 < 1.0)
    assert isinstance(beta_2, float) and (0.0 <= beta_2 < 1.0)
    assert isinstance(rand_seed, int) and (rand_seed >= 0)
    assert isinstance(verbose, bool)

    run_device = run_device.lower()

    # Set the seed for generating random numbers.
    torch.manual_seed(seed=rand_seed)

    # Set the generator and discriminator.
    generator = copy.deepcopy(generator.cpu())
    discriminator = copy.deepcopy(discriminator.cpu())
    if run_device == "cuda":
        assert torch.cuda.is_available()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            generator = torch.nn.DataParallel(generator, device_ids=list(range(0, num_gpus)))
            discriminator = torch.nn.DataParallel(discriminator, device_ids=list(range(0, num_gpus)))

    # Set a criterion and optimizer.
    criterion = torch.nn.MSELoss()
    optimizer_G = torch.optim.Adam(params=generator.parameters(),
                                   lr=learning_rate,
                                   betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(params=discriminator.parameters(),
                                   lr=learning_rate,
                                   betas=(beta_1, beta_2))

    # Covert PyTorch's Tensor to TensorDataset.
    size_labels: int = int(y.max().item() - y.min().item()) + 1
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             shuffle=True)

    # Train the models.
    log: str = "[{0}/{1}] Time: {2:.2f}s"
    for epoch in range(1, num_epochs + 1):
        start_time: float = time.time()
        for (_, batch) in enumerate(dataloader, 0):
            batch_x, batch_y = batch
            batch_size: int = batch_x.size(0)
            batch_x = batch_x.to(device=run_device)

            real: torch.Tensor = torch.full(size=(batch_size,),
                                            fill_value=1,
                                            dtype=torch.float32,
                                            requires_grad=False)
            fake: torch.Tensor = torch.full(size=(batch_size,),
                                            fill_value=0,
                                            dtype=torch.float32,
                                            requires_grad=False)

            latent_vector: torch.Tensor = torch.randn(size=(batch_size, latent_size),
                                                      dtype=torch.float32)
            fake_y: torch.Tensor = torch.randint(low=0, high=size_labels - 1, size=(batch_size,), dtype=torch.long)

            if run_device == "cuda":
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                real = real.cuda()
                fake = fake.cuda()
                latent_vector = latent_vector.cuda()
                fake_y = fake_y.cuda()

            fake_x: torch.Tensor = generator(latent_vector, fake_y)

            # Generator
            optimizer_G.zero_grad()
            output: torch.Tensor = discriminator(fake_x, fake_y)
            output = output.view(size=(-1,))
            loss_g: torch.Tensor = criterion(output, real)
            loss_g.backward(retain_graph=True)
            optimizer_G.step()

            # Discriminator
            optimizer_D.zero_grad()
            output = discriminator(batch_x, batch_y)
            output = output.view(size=(-1,))
            loss_d_real: torch.Tensor = criterion(output, real)
            loss_d_real.backward(retain_graph=True)
            output = discriminator(fake_x, fake_y)
            output = output.view(size=(-1,))
            loss_d_fake: torch.Tensor = criterion(output, fake)
            loss_d_fake.backward(retain_graph=True)
            optimizer_D.step()
        end_time: float = time.time()

        if verbose:
            print(log.format(epoch,
                             num_epochs,
                             end_time - start_time))

    if isinstance(generator, torch.nn.DataParallel):
        generator = generator.module
    if isinstance(discriminator, torch.nn.DataParallel):
        discriminator = discriminator.module

    return copy.deepcopy(generator.cpu()), copy.deepcopy(discriminator.cpu())


def predict(generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            latent_size: int,
            output_by_label: dict,
            run_device: str = "cpu",
            rand_seed: int = 0,
            verbose: bool = False) -> (np.ndarray, np.ndarray):
    """
    Function to generate sa
    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    latent_size: int
    output_by_label: dict
    run_device: str
    rand_seed: int
    verbose: bool

    Returns
    -------
    tuple

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(latent_size, int) and (latent_size > 0)
    assert isinstance(output_by_label, dict)
    assert isinstance(run_device, str) and (run_device.lower() in ["cpu", "cuda"])
    assert isinstance(rand_seed, int) and (rand_seed >= 0)
    assert isinstance(verbose, bool)

    # Set the seed for generating random numbers.
    torch.manual_seed(seed=rand_seed)

    # Set the generator and discriminator.
    generator = copy.deepcopy(generator.cpu())
    discriminator = copy.deepcopy(discriminator.cpu())
    if run_device == "cuda":
        assert torch.cuda.is_available()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        if torch.cuda.device_count() > 1:
            num_gpus: int = torch.cuda.device_count()
            generator = torch.nn.DataParallel(generator, device_ids=list(range(0, num_gpus)))
            discriminator = torch.nn.DataParallel(discriminator, device_ids=list(range(0, num_gpus)))

    keys: list = list(output_by_label.keys())
    labels: torch.Tensor = torch.as_tensor([key for key in keys for _ in range(output_by_label[key])],
                                           dtype=torch.long)
    latent_vector: torch.Tensor = torch.randn(size=(len(labels), latent_size),
                                              dtype=torch.float32,
                                              requires_grad=False)
    if run_device == "cuda":
        latent_vector = latent_vector.cuda()

    generator.eval()
    with torch.no_grad():
        output: torch.Tensor = generator(latent_vector, labels).detach().cpu()

    return output.numpy(), labels.numpy()


def save_model(generator: torch.nn.Module, discriminator: torch.nn.Module, model_dir: str) -> bool:
    """
    Function to save the parameters of the trained generator and discriminator.

    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    model_dir: str

    Returns
    -------
    bool

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(model_dir, str)

    if os.path.exists(model_dir):
        raise FileExistsError(model_dir)
    else:
        os.makedirs(model_dir)

    generator_path: str = os.path.join(model_dir, "Generator.pth")
    discriminator_path: str = os.path.join(model_dir, "Discriminator.pth")

    # Save the trained models.
    torch.save(generator.cpu().state_dict(), generator_path)
    torch.save(discriminator.cpu().state_dict(), discriminator_path)

    return True


def load_model(generator: torch.nn.Module,
               discriminator: torch.nn.Module,
               model_dir: str) -> (torch.nn.Module, torch.nn.Module):
    """
    Function to load the parameters from the trained classifier.

    Parameters
    ----------
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    model_dir: str

    Returns
    -------
    torch.nn.Module

    """
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(discriminator, torch.nn.Module)
    assert isinstance(model_dir, str) and os.path.exists(model_dir)

    generator_path: str = os.path.join(model_dir, "Generator.pth")
    discriminator_path: str = os.path.join(model_dir, "Discriminator.pth")

    if os.path.exists(generator_path):
        raise FileNotFoundError(generator_path)
    if os.path.exists(discriminator_path):
        raise FileNotFoundError(discriminator_path)

    # Load the trained models.
    generator = copy.deepcopy(generator.cpu())
    discriminator = copy.deepcopy(discriminator.cpu())
    generator.load_state_dict(torch.load(generator_path, map_location=torch.device("cpu")))
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=torch.device("cpu")))

    return generator, discriminator
