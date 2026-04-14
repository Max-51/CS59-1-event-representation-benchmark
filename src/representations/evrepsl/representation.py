import os
import sys
import numpy as np
import torch

from src.representations.base import BaseRepresentation
from src.representations.registry import register_representation


# Add third_party/evrepsl to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
EVREPSL_DIR = os.path.join(ROOT_DIR, "third_party", "evrepsl")

if EVREPSL_DIR not in sys.path:
    sys.path.append(EVREPSL_DIR)

from event_representations import events_to_EvRep, load_RepGen, EvRep_to_EvRepSL


@register_representation("evrepsl")
class EvRepSLRepresentation(BaseRepresentation):
    """
    Wrapper for the EvRepSL representation.

    This adapter converts raw event streams into the EvRep representation first,
    and then feeds EvRep into the pretrained RepGen model to obtain EvRepSL.

    Expected input format:
        events: numpy array of shape (N, 4)
                columns are [x, y, t, p]

    Expected polarity format:
        p in {0, 1}

    Output:
        torch.Tensor or numpy.ndarray depending on config
    """

    def __init__(self, config):
        super().__init__(config)

        self.width = config.get("width", 320)
        self.height = config.get("height", 240)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.return_numpy = config.get("return_numpy", True)
        self.weights_path = config.get("weights_path", os.path.join(EVREPSL_DIR, "RepGen.pth"))

        self.model = self._load_model()

    def _load_model(self):
        """
        Load the pretrained RepGen model.

        The official helper function assumes the weight file is named 'RepGen.pth'
        in the current working directory, so we temporarily switch the working
        directory to the EvRepSL folder before loading.
        """
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"RepGen weight file not found: {self.weights_path}"
            )

        original_cwd = os.getcwd()
        try:
            os.chdir(EVREPSL_DIR)
            model = load_RepGen(self.device)
        finally:
            os.chdir(original_cwd)

        return model

    def _validate_events(self, events):
        """
        Validate the input event array.
        """
        if events is None:
            raise ValueError("Input events cannot be None.")

        events = np.asarray(events)

        if events.ndim != 2:
            raise ValueError(f"Input events must be 2D, got shape {events.shape}.")

        if events.shape[1] != 4:
            raise ValueError(
                f"Input events must have shape (N, 4), got {events.shape}."
            )

        if len(events) == 0:
            raise ValueError("Input events array is empty.")

        return events

    def _split_events(self, events):
        """
        Split events into x, y, t, p arrays.
        """
        event_xs = events[:, 0].astype(np.int64)
        event_ys = events[:, 1].astype(np.int64)
        event_timestamps = events[:, 2].astype(np.float32)
        event_polarities = events[:, 3].astype(np.int64)

        return event_xs, event_ys, event_timestamps, event_polarities

    def _filter_valid_events(self, event_xs, event_ys, event_timestamps, event_polarities):
        """
        Remove events that fall outside the sensor resolution.
        """
        valid_mask = (
            (event_xs >= 0) & (event_xs < self.width) &
            (event_ys >= 0) & (event_ys < self.height)
        )

        event_xs = event_xs[valid_mask]
        event_ys = event_ys[valid_mask]
        event_timestamps = event_timestamps[valid_mask]
        event_polarities = event_polarities[valid_mask]

        if len(event_xs) == 0:
            raise ValueError("No valid events remain after spatial filtering.")

        return event_xs, event_ys, event_timestamps, event_polarities

    def _normalize_polarity_if_needed(self, event_polarities):
        """
        Convert polarity from {-1, 1} to {0, 1} if needed.

        The official EvRepSL code assumes polarity is in {0, 1}.
        """
        unique_values = np.unique(event_polarities)

        if np.all(np.isin(unique_values, [0, 1])):
            return event_polarities.astype(np.int64)

        if np.all(np.isin(unique_values, [-1, 1])):
            return ((event_polarities + 1) // 2).astype(np.int64)

        raise ValueError(
            f"Unsupported polarity values: {unique_values}. "
            "Expected polarity in {0, 1} or {-1, 1}."
        )

    def build(self, events):
        """
        Build the EvRepSL representation from raw events.

        Steps:
            1. Validate input events.
            2. Split events into x, y, t, p.
            3. Filter invalid coordinates.
            4. Convert polarity to the format expected by the official code.
            5. Build EvRep.
            6. Add batch dimension.
            7. Run the pretrained RepGen model.
            8. Return EvRepSL.
        """
        events = self._validate_events(events)

        event_xs, event_ys, event_timestamps, event_polarities = self._split_events(events)

        event_xs, event_ys, event_timestamps, event_polarities = self._filter_valid_events(
            event_xs, event_ys, event_timestamps, event_polarities
        )

        event_polarities = self._normalize_polarity_if_needed(event_polarities)

        ev_rep = events_to_EvRep(
            event_xs=event_xs,
            event_ys=event_ys,
            event_timestamps=event_timestamps,
            event_polarities=event_polarities,
            resolution=(self.width, self.height)
        )

        # Official RepGen expects batched input: B x 3 x H x W
        ev_rep = np.expand_dims(ev_rep, axis=0).astype(np.float32)

        with torch.no_grad():
            ev_rep_sl = EvRep_to_EvRepSL(self.model, ev_rep, self.device)

        if self.return_numpy:
            return ev_rep_sl.detach().cpu().numpy()

        return ev_rep_sl