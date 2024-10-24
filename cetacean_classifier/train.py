from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from .config import Config, load_config
# from .dataset import WhaleDataset, load_df
from .metric_learning import ArcFaceLossAdaptiveMargin, ArcMarginProductSubcenter, GeM
from .utils import WarmupCosineLambda, map_dict, topk_average_precision


class SphereClassifier(LightningModule):
    def __init__(self, cfg: dict, id_class_nums=None, species_class_nums=None):
        super().__init__()
        if not isinstance(cfg, Config):
            cfg = Config(cfg)
        self.save_hyperparameters(cfg, ignore=["id_class_nums", "species_class_nums"])
        self.test_results_fp = None

        print(cfg.model_name)

        # NN architecture
        self.backbone = timm.create_model(
            cfg.model_name,
            in_chans=3,
            pretrained=cfg.pretrained,
            num_classes=0,
            features_only=True,
            out_indices=cfg.out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")
        self.global_pools = torch.nn.ModuleList(
            [GeM(p=cfg.global_pool.p, requires_grad=cfg.global_pool.train) for _ in cfg.out_indices]
        )
        self.mid_features = np.sum(feature_dims)
        if cfg.normalization == "batchnorm":
            self.neck = torch.nn.BatchNorm1d(self.mid_features)
        elif cfg.normalization == "layernorm":
            self.neck = torch.nn.LayerNorm(self.mid_features)
        self.head_id = ArcMarginProductSubcenter(self.mid_features, cfg.num_classes, cfg.n_center_id)
        self.head_species = ArcMarginProductSubcenter(self.mid_features, cfg.num_species_classes, cfg.n_center_species)
        if id_class_nums is not None and species_class_nums is not None:
            margins_id = np.power(id_class_nums, cfg.margin_power_id) * cfg.margin_coef_id + cfg.margin_cons_id
            margins_species = (
                np.power(species_class_nums, cfg.margin_power_species) * cfg.margin_coef_species
                + cfg.margin_cons_species
            )
            print("margins_id", margins_id)
            print("margins_species", margins_species)
            self.margin_fn_id = ArcFaceLossAdaptiveMargin(margins_id, cfg.num_classes, cfg.s_id)
            self.margin_fn_species = ArcFaceLossAdaptiveMargin(margins_species, cfg.num_species_classes, cfg.s_species)
            self.loss_fn_id = torch.nn.CrossEntropyLoss()
            self.loss_fn_species = torch.nn.CrossEntropyLoss()

    def get_feat(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        return self.neck(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.get_feat(x)
        return self.head_id(feat), self.head_species(feat)

    def training_step(self, batch, batch_idx):
        x, ids, species = batch["image"], batch["label"], batch["label_species"]
        logits_ids, logits_species = self(x)
        margin_logits_ids = self.margin_fn_id(logits_ids, ids)
        loss_ids = self.loss_fn_id(margin_logits_ids, ids)
        loss_species = self.loss_fn_species(self.margin_fn_species(logits_species, species), species)
        self.log_dict({"train/loss_ids": loss_ids.detach()}, on_step=False, on_epoch=True)
        self.log_dict({"train/loss_species": loss_species.detach()}, on_step=False, on_epoch=True)
        with torch.no_grad():
            self.log_dict(map_dict(logits_ids, ids, "train"), on_step=False, on_epoch=True)
            self.log_dict(
                {"train/acc_species": topk_average_precision(logits_species, species, 1).mean().detach()},
                on_step=False,
                on_epoch=True,
            )
        return loss_ids * self.hparams.loss_id_ratio + loss_species * (1 - self.hparams.loss_id_ratio)

    def validation_step(self, batch, batch_idx):
        x, ids, species = batch["image"], batch["label"], batch["label_species"]
        out1, out_species1 = self(x)
        out2, out_species2 = self(x.flip(3))
        output, output_species = (out1 + out2) / 2, (out_species1 + out_species2) / 2
        self.log_dict(map_dict(output, ids, "val"), on_step=False, on_epoch=True)
        self.log_dict(
            {"val/acc_species": topk_average_precision(output_species, species, 1).mean().detach()},
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters()) + list(self.global_pools.parameters())
        head_params = (
            list(self.neck.parameters()) + list(self.head_id.parameters()) + list(self.head_species.parameters())
        )
        params = [
            {"params": backbone_params, "lr": self.hparams.lr_backbone},
            {"params": head_params, "lr": self.hparams.lr_head},
        ]
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(params)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params)
        elif self.hparams.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(params)

        warmup_steps = self.hparams.max_epochs * self.hparams.warmup_steps_ratio
        cycle_steps = self.hparams.max_epochs - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, self.hparams.lr_decay_scale)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        feat1 = self.get_feat(x)
        out1, out_species1 = self.head_id(feat1), self.head_species(feat1)
        feat2 = self.get_feat(x.flip(3))
        out2, out_species2 = self.head_id(feat2), self.head_species(feat2)
        pred_logit, pred_idx = ((out1 + out2) / 2).cpu().sort(descending=True)
        return {
            "original_index": batch["original_index"],
            "label": batch["label"],
            "label_species": batch["label_species"],
            "pred_logit": pred_logit[:, :1000],
            "pred_idx": pred_idx[:, :1000],
            "pred_species": ((out_species1 + out_species2) / 2).cpu(),
            "embed_features1": feat1.cpu(),
            "embed_features2": feat2.cpu(),
        }

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        outputs = self.all_gather(outputs)
        if self.trainer.global_rank == 0:
            epoch_results: Dict[str, np.ndarray] = {}
            for key in outputs[0].keys():
                if torch.cuda.device_count() > 1:
                    result = torch.cat([x[key] for x in outputs], dim=1).flatten(end_dim=1)
                else:
                    result = torch.cat([x[key] for x in outputs], dim=0)
                epoch_results[key] = result.detach().cpu().numpy()
            np.savez_compressed(self.test_results_fp, **epoch_results)
