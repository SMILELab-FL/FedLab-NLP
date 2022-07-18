import numpy as np
from abc import ABC
from tqdm import tqdm

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from trainers.BaseClient import BaseClientTrainer, BaseLocalTrainer, BaseClientManager

from fedlab.utils.serialization import SerializationTool


class LocalTrainer(BaseLocalTrainer, ABC):
    def __init__(self):
        super().__init__()

    def train_model(self, model, train_dl):
        model.to(self.device)

        # build optimizer and scheduler
        optimizer, scheduler = self._build_optimizer(model, len(train_dl))

        model, optimizer = self._mixed_train_model(model, optimizer)

        criterion = self._build_loss()

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            for step, batch in enumerate(train_dl):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                if self.model_config.model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = batch[2] \
                        if self.model_config.model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)

                loss = outputs[0]
                # loss = criterion(logits.view(-1, model.num_labels), inputs["labels"].view(-1))

                if self.training_config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.training_config.gradient_accumulation_steps > 1:
                    loss = loss / self.training_config.gradient_accumulation_steps

                if self.training_config.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                    if self.training_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            self.logger.info(f"Local Epoch {epoch} is done and loss is {tr_loss / global_step:.3f}")

        return global_step, tr_loss / global_step

    def eval_model(self, model, valid_dl):
        if self.training_config.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        results = {}
        for batch in tqdm(valid_dl, desc="Server Evaluating"):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.model_config.model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = \
                        batch[2] if self.model_config.model_type in ['bert','xlnet'] else None
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss
        if self.model_config.model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif self.model_config.model_output_mode == "regression":
            preds = np.squeeze(preds)

        self.metric.update_metrics(preds, out_label_ids)
        results.update(self.metric.best_metric)
        return results

    def _build_optimizer(self, model, train_dl_len):
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.bert.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.bert.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler

    def _freeze_model_parameters(self, model):
        pass


class FedAvgClientTrainer(BaseClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset, data_slices):
        client_num = len(data_slices)
        super().__init__(model, client_num)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # self.test_dataset = valid_dataset
        self.data_slices = data_slices  # [0, client_num)
        self.local_trainer = LocalTrainer()

    def _get_dataloader(self, dataset, client_id):
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def _train_alone(self, model_parameters, train_loader):
        SerializationTool.deserialize_model(self._model, model_parameters)
        self.local_trainer.train_model(model=self._model, train_dl=train_loader)
        return self.model_parameters

    def train(self, model_parameters, id_list):
        param_list = []
        # self.logger.info(f"Sub-Server {self.rank}'s
        # local training with client id list: {id_list}")
        for idx in id_list:
            train_data_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
            self._train_alone(
                model_parameters=model_parameters,
                train_loader=train_data_loader,
            )
            param_list.append(self.model_parameters)

        return param_list

    def local_process(self, id_list, payload):
        model_parameters = payload[0]
        self.param_list = self.train(model_parameters, id_list)
        return self.param_list


class FedAvgClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)


