# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import accuracy_calculator

from .networks import Classifier, SimpleDenseNet


class EyeKnowYouToo(pl.LightningModule):
    def __init__(
            self,
            n_classes: int,
            embeddings_filename: str,
            embeddings_dir: str = "./embeddings",
            w_metric_loss: float = 1.0,
            w_class_loss: float = 0.1,
            compute_map_at_r: bool = False,
    ):
        super().__init__()

        self.embedder = SimpleDenseNet(depth=9, output_dim=128)
        self.classifier = Classifier(self.embedder.output_dim, n_classes)

        self.w_metric_loss = w_metric_loss
        self.metric_criterion = losses.MultiSimilarityLoss()
        self.metric_miner = miners.MultiSimilarityMiner()

        self.w_class_loss = w_class_loss
        self.class_criterion = torch.nn.CrossEntropyLoss()

        self.compute_map_at_r = compute_map_at_r
        self.map_at_r_calculator = (
            accuracy_calculator.AccuracyCalculator(
                include=["mean_average_precision_at_r"],
                avg_of_avgs=True,
                k="max_bin_count",
            )
            if self.compute_map_at_r
            else None
        )

        self.embeddings_path = Path(embeddings_dir) / embeddings_filename

    def forward(self, x):
        print("HEERREE")
        print(x.shape)
        out = self.embedder(x)
        return out

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.LongTensor], batch_idx: int
    ) -> STEP_OUTPUT:
        assert torch.is_grad_enabled()
        assert self.training

        inputs, metadata = batch
        # if using transformer use self.embedder(inputs).logits else without
        # embeddings = self.embedder(inputs).logits
        embeddings = self.embedder(inputs)
        labels = metadata[:, 0]
        metric_loss = self.metric_step(embeddings, labels)
        # class_loss = self.class_step(embeddings, labels)
        # total_loss = metric_loss + class_loss

        # self.log("train_loss", total_loss)
        self.log("train_metric_loss", metric_loss)
        # self.log("train_class_loss", class_loss)
        return {"loss": metric_loss}

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.LongTensor],
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> Optional[STEP_OUTPUT]:
        assert not torch.is_grad_enabled()
        assert not self.training

        inputs, metadata = batch
        # if using transformer use self.embedder(inputs).logits else without
        # embeddings = self.embedder(inputs).logits
        embeddings = self.embedder(inputs)
        return {"embeddings": embeddings, "metadata": metadata}

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.LongTensor],
            batch_idx: int,
            dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert not torch.is_grad_enabled()
        assert not self.training

        inputs, metadata = batch
        # if using transformer use self.embedder(inputs).logits else without
        # embeddings = self.embedder(inputs).logits
        embeddings = self.embedder(inputs)
        return {"embeddings": embeddings, "metadata": metadata}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        full_val_outputs = outputs[0]
        sum_val_loss = 0
        embeddings = full_val_outputs["embeddings"]
        metadata = full_val_outputs["metadata"]
        labels = metadata[:, 0]
        metric_loss = self.metric_step(embeddings, labels)
        sum_val_loss += metric_loss
        mean_val_loss = sum_val_loss / len(full_val_outputs)
        self.log("val_metric_loss", mean_val_loss)

        if not self.compute_map_at_r:
            return

        def process_batch_outputs(list_of_batch_outputs, split):
            embeddings = [x["embeddings"] for x in list_of_batch_outputs]
            metadata = [x["metadata"] for x in list_of_batch_outputs]
            embeddings = torch.cat(embeddings, dim=0)
            metadata = torch.cat(metadata, dim=0)

            labels = metadata[:, 0]

            result_dict = self.map_at_r_calculator.get_accuracy(
                embeddings,
                embeddings,
                labels,
                labels,
                embeddings_come_from_same_source=True,
            )
            map_at_r = result_dict["mean_average_precision_at_r"]
            self.log(split + "_map_at_r", map_at_r)

        val_tex_outputs = outputs[1]
        process_batch_outputs(val_tex_outputs, "val")

        train_tex_outputs = outputs[2]
        process_batch_outputs(train_tex_outputs, "train")

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        def process_batch_outputs(list_of_batch_outputs, split):
            embeddings = [x["embeddings"] for x in list_of_batch_outputs]
            metadata = [x["metadata"] for x in list_of_batch_outputs]
            embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()
            metadata = torch.cat(metadata, dim=0).detach().cpu().numpy()

            embed_dim = embeddings.shape[1]
            embedding_dict = {
                f"embed_dim_{i:03d}": embeddings[:, i]
                for i in range(embed_dim)
            }
            full_dict = {
                "nb_round": metadata[:, 1],
                "nb_subject": metadata[:, 2],
                "nb_session": metadata[:, 3],
                "nb_task": metadata[:, 4],
                "nb_subsequence": metadata[:, 5],
                "exclude": metadata[:, 6],
                **embedding_dict,
            }
            df = pd.DataFrame(full_dict)
            df = df.sort_values(
                by=[
                    "nb_round",
                    "nb_subject",
                    "nb_session",
                    "nb_task",
                    "nb_subsequence",
                ],
                axis=0,
                ascending=True,
            )
            path = self.embeddings_path.with_name(
                split + "_" + self.embeddings_path.name
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)

        if isinstance(outputs[0], list):
            # More than one test dataloader was used, so this is
            # GazeBase.  The first loader is the held-out test set, and
            # the second loader is the full validation set.
            test_outputs = outputs[0]
            process_batch_outputs(test_outputs, "test")

            val_outputs = outputs[1]
            process_batch_outputs(val_outputs, "val")
        else:
            # Only one test dataloader was used, so this is JuDo1000
            process_batch_outputs(outputs, "judo")

    def metric_step(
            self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        if self.w_metric_loss <= 0.0:
            return 0.0

        mined_indices = (
            None
            if self.metric_miner is None
            else
            self.metric_miner(embeddings, labels)
        )
        metric_loss = self.metric_criterion(embeddings, labels, mined_indices)

        weighted_metric_loss = metric_loss * self.w_metric_loss
        return weighted_metric_loss

    def class_step(
            self, embeddings: torch.Tensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        # Since we have class-disjoint datasets, only compute class loss
        # on the training set.  We know we're working with the train set
        # if `self.embedder.training` is True.
        if (
                self.classifier is None
                or self.w_class_loss <= 0.0
                or not self.training
        ):
            return 0.0

        # When logits and labels are on the GPU, we get an error on the
        # backward pass.  For some reason, transferring them to the CPU
        # fixes the error.
        #
        # Full error below (with several instances of this error for
        # different threads, e.g., [6,0,0], [7,0,0], and [13,0,0]):
        # .../pytorch_1634272168290/work/aten/src/ATen/native/cuda/Loss.cu:455:
        # nll_loss_backward_reduce_cuda_kernel_2d: block: [0,0,0],
        # thread: [5,0,0] Assertion `t >= 0 && t < n_classes` failed.
        logits = self.classifier(embeddings)
        class_loss = self.class_criterion(logits.cpu(), labels.cpu())

        weighted_class_loss = class_loss * self.w_class_loss
        return weighted_class_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters())
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=0.01,
            epochs=100,
            steps_per_epoch=1,
            cycle_momentum=False,
            div_factor=100.0,
            final_div_factor=1000.0,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    def optimizer_zero_grad(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: torch.optim.Optimizer,
            optimizer_idx: int,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)
