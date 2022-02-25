import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from sys import exit
from os import environ

import mlflow

import optuna
from optuna.trial import TrialState

# Define hyper-parameters for tuning
def get_hyperparams(trial):
    return {
        "lr": trial.suggest_uniform("lr", 1e-3, 1.0),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_uniform("gamma", 0.65, 0.75),
        "epoch": trial.suggest_categorical("epoch", [3, 6, 9]),
    }


# A training epoch
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.local_rank == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return loss.item()


# Test the model's accuracy
def test(model, device, test_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Gather the correct results of all devices into the rank-0 device.
    # dist.all_reduce only works on CUDA tensor type.
    # The default reduce operation is SUM.
    correct_tensor = torch.tensor([correct], dtype=torch.int).to(device)
    dist.all_reduce(correct_tensor)
    total_correct = correct_tensor.item()

    loss /= len(test_loader.dataset)
    accuracy = 100.0 * total_correct / len(test_loader.dataset)

    if args.local_rank == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                loss, total_correct, len(test_loader.dataset), accuracy
            )
        )
    return loss, accuracy


# Main optimize function
# Optuna will run this function as many times as the defined trials number and
# find the optimal hyper-parameters
def optimize(single_trial, device, args):
    trial = optuna.integration.TorchDistributedTrial(single_trial, device)
    trial_num = trial.number
    hyps = get_hyperparams(trial)

    # LeNet-5: since we are working on MNIST, we might as well use this venerable architecture.
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
        nn.LogSoftmax(1),
    )

    # Now, let's proceed to the data processing part.
    # First, we normalize the input images.  Here is were you would define the data augmentation
    # policies that make sense for your dataset.
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Get the train and test data from MNIST.
    train_data = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST("../data", train=False, transform=transform)

    # We define common options for the data loaders.
    data_kwargs = {
        "batch_size": hyps["batch_size"],
        "num_workers": 4,
        "pin_memory": False,
    }

    # For training, we will use a distributed sampler, so that each node gets
    # a unique subset of the dataset.
    train_sampler = DistributedSampler(train_data)

    # Finally, we instantiate our dataloaders, for the train and test sets.
    train_loader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, **data_kwargs
    )

    test_sampler = DistributedSampler(test_data, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, **data_kwargs
    )

    # We transfer the model to the device (the GPU appointed to the current rank)
    model = model.to(device)
    # And wrap the model inside the DistributedDataParallel class
    model = DDP(model, device_ids=[args.local_rank])

    # As usual, we instantiate an optimizer and a learning rate scheduler.
    optimizer = optim.Adadelta(model.parameters(), lr=hyps["lr"])
    scheduler = StepLR(optimizer, step_size=1, gamma=hyps["gamma"])

    # One important think to do when using MLflow is to only use it from node 0,
    # otherwise, it will create as many runs as nodes.
    if args.local_rank == 0:
        # We set a name for this run.
        run = mlflow.start_run(run_name=f"mnist-lenet-{trial_num}")
        # And log the hyperparameters we care about for display on the MLflow UI.
        mlflow.log_param("epochs", hyps["epoch"])
        mlflow.log_param("batch-size", hyps["batch_size"])
        mlflow.log_param("initial-lr", hyps["lr"])
        mlflow.log_param("gamma", hyps["gamma"])
        mlflow.log_param("seed", args.seed)

    # Now we proceed to start the main training loop
    for epoch in range(1, hyps["epoch"] + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)

        # We gather the metrics we care about, and log them to MLflow.
        test_loss, test_acc = test(model, device, test_loader)
        if args.local_rank == 0:
            mlflow.log_metric("train-loss", train_loss, epoch)
            mlflow.log_metric("test-loss", test_loss, epoch)
            mlflow.log_metric("test-accuracy", test_acc, epoch)
            # Sometimes it might be useful to visualize the learning rate, as well.
            mlflow.log_metric("learning-rate", scheduler.get_last_lr()[0], epoch)

        # Report intermediate results for every epoch to decide whether to stop trial.
        trial.report(test_acc, epoch)
        if trial.should_prune():
            if args.local_rank == 0:
                # We should stop the existing mlflow run to start next one.
                mlflow.end_run()
            raise optuna.exceptions.TrialPruned()

        # We finally update the scheduler for the next iteration
        scheduler.step()

    if args.local_rank == 0:
        mlflow.end_run()
    return test_acc


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="How many trials to search hyperparams.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow uri for tracking",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="MNIST-PyTorch-Tune",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Set the seed so that we can reproduce the experiment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Here we setup the distributed environment and read the variables needed
    # for the DistributedDataParallel features.  Note that these environment
    # variables are already set up once we launch this program like this:
    # torchrun --standalone --nproc_per_node {NUM_GPUS} main.py
    dist.init_process_group(backend="nccl", init_method="env://")
    args.local_rank = int(environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    if args.local_rank == 0:
        study = None
        # We need to connect to the MLflow server.
        mlflow.set_tracking_uri(args.mlflow_uri)
        print("Tracking URI:", mlflow.tracking.get_tracking_uri())
        # And set an experiment name, otherwise it will be logged under "default"
        # in the MLflow dashboard.
        mlflow.set_experiment(args.mlflow_experiment)

        # Define Optuna study for trials
        study = optuna.create_study(direction="maximize")
        # If the  optimize function has additional input arguments, we need to define a lambda function.
        # Otherwise, we only pass the function to first argument, i.e., study.optimize(optimize, n_trials)
        study.optimize(
            lambda trial: optimize(trial, device, args), n_trials=args.n_trials
        )

    # Optimizing trials only run in rank-0 device and other devices just run for DDP
    else:
        for _ in range(args.n_trials):
            try:
                optimize(None, device, args)
            except optuna.TrialPruned:
                pass

    if args.local_rank == 0:
        # Summarize Optuna trials results
        assert study is not None
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Start new mlflow run to log trials summary
        with mlflow.start_run(run_name="summary"):
            mlflow.log_metric("number of finished trials", len(study.trials))
            mlflow.log_metric("number of complete trials", len(complete_trials))
            mlflow.log_metric("number of pruned trials", len(pruned_trials))
            mlflow.log_metric("best trial number", study.best_trial.number)
            mlflow.log_metric("best accuracy", study.best_trial.value)

            for key, val in study.best_trial.params.items():
                mlflow.mlflow.log_param("best " + key, val)

            # Try to log Optuna supported summary figure
            try:
                import plotly

                fig_0 = optuna.visualization.plot_param_importances(study)
                fig_1 = optuna.visualization.plot_optimization_history(study)
                fig_2 = optuna.visualization.plot_slice(study)
                fig_3 = optuna.visualization.plot_intermediate_values(study)
                fig_4 = optuna.visualization.plot_parallel_coordinate(study)
                fig_5 = optuna.visualization.plot_edf(study)

                mlflow.log_figure(fig_0, "param_importances.html")
                mlflow.log_figure(fig_1, "optim_history.html")
                mlflow.log_figure(fig_2, "slices.html")
                mlflow.log_figure(fig_3, "intermediate.html")
                mlflow.log_figure(fig_4, "parallel.html")
                mlflow.log_figure(fig_5, "edf.html")

            except ImportError:
                raise ImportError(
                    "Plot trials summary is failed. If wants, install 'plotly' lib."
                )
