

import os
import math
import numpy as np
import torch
import scipy.signal as signal
import pytorch_lightning as pl
import metrics
import time

class CSMFCC_FN:
    def __call__(self, ref, deg):
        return metrics.mfcc_cosine_similarity(ref, deg, fs=8000)
    @staticmethod
    def min():
        return -1.0

class NBPESQ_FN:
    def __call__(self, ref, deg):
        return metrics.nbpesq(ref, deg, fs=8000)
    @staticmethod
    def min():
        return 1.0 
class DNSMOS_OVRL:
    def __call__(self, ref, deg):
        return metrics.DNSMOS_OVRL(deg, fs=8000)
        
    @staticmethod
    def min():
        return 1.0 

class ESTOI_FN:
    def __call__(self, ref, deg):
        return metrics.estoi(ref, deg, fs=8000)

    @staticmethod
    def min():
        return 0.0
class BaseModel(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.val_outputs = []
        self.val_targets = []
        self.val_tasks = []
        self.val_scales = []

        self.test_outputs = []
        self.test_targets = []
        self.test_tasks = []
        self.test_scales = []
        
        self.heavy_eval = False

        self.csmfcc_fn = CSMFCC_FN()
        self.pesq_fn = NBPESQ_FN()
        self.dnsmos_fn = DNSMOS_OVRL()
        self.estoi_fn = ESTOI_FN()

    def common_step(self, batch, batch_idx, mode="train"):
        
        clean = batch["clean"]
        noisy = batch["recorded"]
        task = batch["task"]
        enhanced = self.forward(noisy)

        return enhanced, clean, task

    def training_step(self, batch, batch_idx):
        enhanced, clean, task = self.common_step(batch, batch_idx, mode="train")
        loss = self.loss_function(clean, enhanced)
        scale = batch["scale"]
        if not isinstance(scale, torch.Tensor):
            scale = torch.as_tensor(scale, device=self.device, dtype=torch.float32)
        self.log(f'train/loss', loss, logger=True)
        self.log("train/scale_mean", scale.mean(), prog_bar=False, on_step=True)
        self.log("train/scale_std", scale.std(unbiased=False), prog_bar=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):

        enhanced, clean, task = self.common_step(batch, batch_idx, mode="val")
        loss = self.loss_function(clean, enhanced)
        self.log(f'val/loss', loss, logger=True)
        if self.heavy_eval:
            self.val_outputs.append(enhanced.detach().cpu().numpy())
            self.val_targets.append(clean.detach().cpu().numpy())
            self.val_tasks.append(task)
            self.val_scales.append(batch["scale"].detach().cpu().numpy() if isinstance(batch["scale"], torch.Tensor) else np.asarray(batch["scale"]))

        return loss

    def test_step(self, batch, batch_idx):
        enhanced, clean, task = self.common_step(batch, batch_idx, mode="test")
        self.test_outputs.append(enhanced.detach().cpu().numpy())
        self.test_targets.append(clean.detach().cpu().numpy())
        self.test_tasks.append(task)
        self.test_scales.append(batch["scale"].detach().cpu().numpy() if isinstance(batch["scale"], torch.Tensor) else np.asarray(batch["scale"]))

    def metrics_evaluation(self, mode, outputs, targets, tasks, scales):
        
        refs_denorm, degs_denorm = [], []
        flat_tasks = []
        for batch_out, batch_ref, batch_task, batch_scale in zip(outputs, targets, tasks, scales):
            batch_scale = np.asarray(batch_scale).astype(np.float64).reshape(-1)
            denorm_refs, denorm_degs = [], []
            for out, ref, scale in zip(batch_out, batch_ref, batch_scale):
                denorm_refs.append(np.asarray(ref).squeeze() * scale)
                denorm_degs.append(np.asarray(out).squeeze() * scale)
            refs_denorm.append(denorm_refs)
            degs_denorm.append(denorm_degs)
            flat_tasks.extend(batch_task)

        def evaluate_metrics_per_batch(fn, measure_time=False):

            if measure_time:
                torch.cuda.synchronize()  # make sure all prior ops are done
                start = time.perf_counter()
            output = []
            for batch_ref, batch_deg in zip(refs_denorm, degs_denorm):
                for ref, deg in zip(batch_ref, batch_deg):
                    try:
                        val = fn(ref, deg)
                    except Exception:
                        val = fn.min()
                    output.append(val)

            if measure_time:
                torch.cuda.synchronize()  # wait for ops to finish
                end = time.perf_counter()
                print(f"Elapsed time: {end - start:.6f} seconds")

            return output
        


        csmfcc_vals = evaluate_metrics_per_batch(self.csmfcc_fn, measure_time=True)

        
        pesq_vals = evaluate_metrics_per_batch(self.pesq_fn, measure_time=True)

        dnsmos_vals = evaluate_metrics_per_batch(self.dnsmos_fn, measure_time=True)
        
        estoi_vals = evaluate_metrics_per_batch(self.estoi_fn, measure_time=True)


        task1_scores_vals = []
        task2_scores_vals = []

        assert len(flat_tasks) == len(csmfcc_vals), f"{len(flat_tasks)} vs {len(csmfcc_vals)}"

        for pesq_val, dnsmos_val, estoi_val, csmfcc_val, task in \
            zip(pesq_vals, dnsmos_vals, estoi_vals, csmfcc_vals, flat_tasks):

            dnsmos_n = (dnsmos_val - 1.0) / (5.0 - 1.0)
            pesq_n = (pesq_val - 1.0) / (4.5 - 1.0)

            weighted_score = (dnsmos_n + pesq_n + csmfcc_val + estoi_val ) / 4
            if task == "Task1":
                task1_scores_vals.append(weighted_score)
            elif task == "Task2":
                task2_scores_vals.append(weighted_score)

        # weighted_score_vals * task_weightage 
        # Log averaged results
        self.log(f"{mode}/pesq", torch.tensor(pesq_vals).mean(),
                 prog_bar=True, sync_dist=True)
        self.log(f"{mode}/estoi", torch.tensor(estoi_vals).mean(),
                 prog_bar=True, sync_dist=True)
        self.log(f"{mode}/dnsmos", torch.tensor(dnsmos_vals).mean(),
                 prog_bar=True, sync_dist=True)
        self.log(f"{mode}/csmfcc", torch.tensor(csmfcc_vals).mean(),
                 prog_bar=True, sync_dist=True)
        
        task1_score = torch.tensor(task1_scores_vals).mean()
        task2_score = torch.tensor(task2_scores_vals).mean()
        
        self.log(f"{mode}/task1_score", task1_score,
                prog_bar=True, sync_dist=True)
        self.log(f"{mode}/task2_score", task2_score,
                prog_bar=True, sync_dist=True)
        self.log(f"{mode}/weighted_score", 0.4 * task1_score + 0.6 * task2_score ,
                prog_bar=True, sync_dist=True)



    
    def on_validation_epoch_end(self):

        if self.heavy_eval:
            self.metrics_evaluation("val", 
                self.val_outputs, 
                self.val_targets, 
                self.val_tasks,
                self.val_scales)

            self.val_outputs.clear()
            self.val_targets.clear()
            self.val_tasks.clear()
            self.val_scales.clear()


    def on_test_epoch_end(self):

        self.metrics_evaluation("test", 
            self.test_outputs, 
            self.test_targets, 
            self.test_tasks,
            self.test_scales)

        self.test_outputs.clear()
        self.test_targets.clear()
        self.test_tasks.clear()
        self.test_scales.clear()

    def on_after_backward(self):
        total_norm_sq = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        grad_norm = math.sqrt(total_norm_sq)
        self.log("grad_2_norm", grad_norm, on_step=True, prog_bar=False, logger=True)
