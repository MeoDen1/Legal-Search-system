import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from typing import Dict, Any

from .lr_schedulers.schedulers import get_scheduler
from .validator import DecoderValidator
from ..models import DecoderWrapper


REQUIRED_KEYS = ["input_dim", "output_dim", "decoder_depth"]

class Trainer:
    def __init__(self, save_path: str,  training_cfg: Dict[str, Any]):
        # Validate training_cfg
        for key in REQUIRED_KEYS:
            if key not in training_cfg:
                raise ValueError(f"Missing key {key} in training_cfg")

        self.cfg = training_cfg
        self.device = training_cfg.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.save_path = save_path
        self.validation_step = self.cfg.get("validation_step", 2)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "vectors"), exist_ok=True)
        
        self.validator = DecoderValidator(self.device)

    def _get_metadata(self):
        metadata_path = os.path.join(self.save_path, "metadata.json")
        default_metadata = {
            "model_paths": {}
        }

        if not os.path.exists(metadata_path):
            os.makedirs(self.save_path, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(default_metadata, f)

        with open(metadata_path, "r") as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict):
        with open(os.path.join(self.save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    def apply_label_smoothing(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        smoothing = self.cfg.get("label_smoothing", 0.1)
        one_hot = nn.functional.one_hot(x, num_classes).float()
        return one_hot * (1.0 - smoothing) + (smoothing / num_classes)

    def train_single_node(
        self, uid: str, job: Dict[str, Any], metadata: Dict[str, Any]
    ):
        num_classes = job["num_classes"]
        batch_size = int(self.cfg.get("batch_size", 32))
        learning_rate = float(self.cfg.get("learning_rate", 1e-4))
        train_loader = DataLoader(
            job["train_ds"], batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            job["val_ds"], batch_size=batch_size
        )

        # Initialize
        model = DecoderWrapper(
            self.cfg["input_dim"], self.cfg["output_dim"], num_classes
        ).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate
        )
        scheduler = get_scheduler(optimizer, self.cfg)
        pos_weight = torch.tensor([num_classes]).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_class_results : Dict[int, Dict[str, float]] = {}

        # Loop
        for epoch in range(self.cfg.get("epochs", 1)):
            model.train()
            pbar = tqdm(train_loader, desc=f"Node: {uid} | Epoch {epoch+1}")
            for bx, by in pbar:
                bx, by = bx.to(self.device), by.to(self.device)
                target = self.apply_label_smoothing(by, num_classes)

                optimizer.zero_grad()
                loss = loss_fn(model(bx), target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (epoch+1) % self.validation_step == 0 or epoch == self.cfg.get("epochs", 1) - 1:
                # Validation Sweep
                best_class_results = self.validator.find_best_thresholds(model, val_loader, uid, epoch+1)
                f2_scores = [res["f2"] for res in best_class_results.values()]
                max_f2, min_f2 = max(f2_scores), min(f2_scores)

                # Min and max F-score across the class
                logger.success(f"Result {uid} -> Max F2-score: {max_f2:.2f} | Min F2-score: {min_f2:.4f}")

        # Save Artifacts
        safe_name = uid.replace("/", "_")
        os.makedirs(os.path.join(self.save_path, f"{safe_name}"), exist_ok=True)
        m_path = os.path.join(self.save_path, "models", f"{safe_name}.jit")
        v_path = os.path.join(self.save_path, "vectors", f"{safe_name}.json")

        model.decoder.eval()
        model.decoder.to("cpu")
        scripted_model = torch.jit.script(model.decoder)
        scripted_model.save(m_path)

        with open(v_path, "w") as fp:
            json.dump({
                job["label_mapping"].classes_[i]: {
                    "vector": model.vectors[i].tolist(),
                    "threshold": best_class_results[i]["threshold"]
                }
                for i in range(num_classes)
            }, fp)

        return metadata

    def train_all_jobs(self, training_jobs: Dict[str, Any]):
        metadata = self._get_metadata()
        for uid, job in training_jobs.items():
            metadata = self.train_single_node(uid, job, metadata)
            # self._save_metadata(metadata)

        # Save training log
        self.validator.get_logs().to_csv(os.path.join(self.save_path, "training_log.csv"), index=False)
