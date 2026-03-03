import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import fbeta_score
from typing import Tuple, List, Dict
from loguru import logger

class DecoderValidator:
    def __init__(self, device: str):
        self.device = device
        self.thresholds = np.linspace(0.1, 0.9, 17)
        self.eval_logs = pd.DataFrame()

    def find_best_thresholds(
        self, 
        model: torch.nn.Module, 
        loader: DataLoader, 
        model_name: str, 
        epoch: int
    ) -> Dict[int, Dict[str, float]]:
        """
        Finds the best threshold per class and logs scores for every threshold.
        Returns a mapping of {class_idx: {"threshold": best_t, "f2": best_f2}}
        """
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(self.device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(by.numpy())

        y_probs = np.vstack(all_probs)
        y_true = np.concatenate(all_targets)

        # Ensure y_true is one-hot [N, num_classes]
        if len(y_true.shape) == 1:
            num_classes = y_probs.shape[1]
            y_true = np.eye(num_classes)[y_true]
        
        num_classes = y_probs.shape[1]
        best_class_results = {}
        new_rows = []

        # Iterate through each child (class) individually
        for c in range(num_classes):
            c_true = y_true[:, c]
            c_probs = y_probs[:, c]
            
            class_log = {
                "model_name": model_name,
                "epoch": epoch,
                "class_idx": c
            }

            best_f2_c = -1.0
            best_t_c = 0.5

            for t in self.thresholds:
                c_pred = (c_probs >= t).astype(int)
                # Calculate F2 for this class only
                score = fbeta_score(c_true, c_pred, beta=2.0, zero_division=0)
                
                col_name = f"t_{t:.2f}"
                class_log[col_name] = float(score)

                if score > best_f2_c:
                    best_f2_c = score
                    best_t_c = t
            
            class_log["best_t"] = best_t_c
            class_log["best_f2"] = best_f2_c
            new_rows.append(class_log)
            
            best_class_results[c] = {"threshold": best_t_c, "f2": best_f2_c}

        # Update persistent logs
        self.eval_logs = pd.concat([self.eval_logs, pd.DataFrame(new_rows)], ignore_index=True)

        logger.info(f"Completed eval for {model_name} - Found thresholds for {num_classes} classes.")
        return best_class_results

    def get_logs(self) -> pd.DataFrame:
        return self.eval_logs

    def clear_logs(self):
        self.eval_logs = pd.DataFrame()