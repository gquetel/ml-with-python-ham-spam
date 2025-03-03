import plotly
from pathlib import Path
from settings import BASE_PATH
import logging

logger = logging.getLogger(__name__)


def plot_errors_by_epoch(errors: list, model_name: str):
    """Plots the errors by epoch for a given model.
    The figure is saved under output/{model_name}/errors_by_epoch.png
    Args:
        errors (list): List of errors made at each epoch (idx = epoch)
        model_name (str): Name of the studied model
    """
    fig = plotly.graph_objs.Figure()
    fig.add_trace(
        plotly.graph_objs.Scatter(
            x=list(range(1, len(errors) + 1)), y=errors, mode="lines+markers"
        )
    )
    fig.update_layout(
        title="Errors by epoch for model " + model_name,
        xaxis_title="Epoch",
        yaxis_title="Errors",
    )

    output_path = "".join(
        [BASE_PATH, "/output/", model_name, "/errors_by_epoch.png"]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved errors by epoch plot to {output_path}.")
    fig.write_image(output_path)


def plot_losses_by_epoch(losses: list, model_name: str, loss_type: str):
    """Plots the losses by epoch for a given model.
    The figure is saved under output/{model_name}/losses_by_epoch.png
    Args:
        losses (list): List of losses made at each epoch (idx = epoch)
        model_name (str): Name of the studied model
        loss_type (str): Type of loss 
    """
    fig = plotly.graph_objs.Figure()
    fig.add_trace(
        plotly.graph_objs.Scatter(
            x=list(range(1, len(losses) + 1)), y=losses, mode="lines+markers"
        )
    )
    fig.update_layout(
        title="Losses by epoch for model " + model_name,
        xaxis_title="Epoch",
        yaxis_title=f"{loss_type} scores",
    )

    output_path = "".join(
        [BASE_PATH, "/output/", model_name, "/losses_by_epoch.png"]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved losses by epoch plot to {output_path}.")
    fig.write_image(output_path)
