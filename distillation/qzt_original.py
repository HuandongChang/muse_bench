import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM
from accelerate import Accelerator
import bitsandbytes as bnb

class DistillTrainer(Trainer):
    def __init__(self, teacher_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_model_path is not None, f"[DistillTrainer] teacher model is not initialized. Ensure that the teacher model is passed to the DistillTrainer."
        
        self.mse_loss = nn.MSELoss(reduction="mean")
        
        self._prepare_teacher_model(teacher_model_path)
        
    def _prepare_teacher_model(self, teacher_model_path):
        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path,torch_dtype=torch.float16)
                                                                #   load_in_8bit=True)  # Optional: automatically places model on available GPUs
                                                                
        self.teacher_model.eval()
        
        """
        # Move teacher model to the correct device
        device = torch.device(f'cuda:{local_rank()}')
        self.teacher_model = self.teacher_model.to(device)
        
        # Wrap teacher model in DistributedDataParallel (DDP)
        self.teacher_model = torch.nn.parallel.DistributedDataParallel(self.teacher_model, device_ids=[local_rank()])
        """
        self.teacher_model = self._wrap_model(self.teacher_model)
        self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)
        
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
    #     ######## Newly Added ###############
    #     labels = inputs.get("input_ids").clone() if "labels" not in inputs else inputs.pop("labels")
    #     inputs["labels"] = labels
    #     ##############
        
    #     if self.label_smoother is not None and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #     student_outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = student_outputs[self.args.past_index]

    #     if labels is not None:
    #         unwrapped_model = unwrap_model(model)
    #         if _is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(student_outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(student_outputs, labels)
    #     else:
    #         if isinstance(student_outputs, dict) and "loss" not in student_outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(student_outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = student_outputs["loss"] if isinstance(student_outputs, dict) else student_outputs[0]

    #     #! ------------ Distillation loss ------------
    #     with torch.no_grad():
    #         teacher_outputs = self.teacher_model(**inputs)

    #     embedding_loss, layer_loss, logits_loss, distillation_loss = self._compute_distillation_loss(teacher_outputs, student_outputs)
        
    #     # wandb_callback_exists = False
    #     # for callback in self.callback_handler.callbacks:
    #     #     if hasattr(callback, "_wandb"):
    #     #         wandb_callback_exists = True
    #     #         if self.state.is_world_process_zero:
    #     #             callback._wandb.log(
    #     #                 {
    #     #                     "distillation/logits_loss": logits_loss.item(),
    #     #                     "distillation/lm_loss": loss.item(),
    #     #                     "distillation/mini_step": self.current_step_in_epoch,
    #     #                 }
    #     #             )
    #     #             break
    #     # if not wandb_callback_exists:
    #     #     raise ValueError("WandbCallback not found in callbacks. Please enable WandB to log distillation loss.")
        
    #     loss = loss + 0.5 * distillation_loss
        #! --------------------------------------------

        # return (loss, student_outputs) if return_outputs else loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ######## Newly Added ###############
        labels = inputs.get("input_ids").clone() if "labels" not in inputs else inputs.pop("labels")
        inputs["labels"] = labels
        ##############
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        student_outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = student_outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(student_outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(student_outputs, labels)
        else:
            if isinstance(student_outputs, dict) and "loss" not in student_outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(student_outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = student_outputs["loss"] if isinstance(student_outputs, dict) else student_outputs[0]

        #! ------------ Distillation loss ------------
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        embedding_loss, layer_loss, logits_loss, distillation_loss = self._compute_distillation_loss(teacher_outputs, student_outputs)
        
        # loss = loss + 0.5 * distillation_loss
        loss = distillation_loss
        
        ##########debugging###############
        for name, param in model.named_parameters():
            if torch.isnan(param).any().item():
                print(f"NaN found in {name}")
                breakpoint()
        ##########debugging###############

        
        #! --------------------------------------------

        return (loss, student_outputs) if return_outputs else loss


    def _compute_distillation_loss(self, teacher_outputs, student_outputs):
        """
        Compute the distillation loss between the teacher and student outputs.
        
        Args:
            teacher_outputs (Dict[str, torch.Tensor]): The outputs of the teacher model.
            student_outputs (Dict[str, torch.Tensor]): The outputs of the student model.
            
        Returns:
            Tuple[torch.Tensor]: The embedding loss, layer loss, logits loss, and total loss.
        """
        def get_embedding_loss():
            return 0
        
        def get_layer_loss():
            return 0
        
        def get_logits_loss():          
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # reverse
            teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
            student_output_soft = F.softmax(student_logits, dim=2)
            reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none").sum(-1)

            # forward
            student_output_log_prob = F.log_softmax(student_logits, dim=2)
            teacher_output_soft = F.softmax(teacher_logits, dim=2)
            forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="none").sum(-1)

            beta_prob = 0.5
            kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
            kl_loss = torch.mean(kl_loss)

            return kl_loss
        
        #! Embedding loss
        embedding_loss = get_embedding_loss()
        
        #! Layer loss
        layer_loss = get_layer_loss()
        
        #! Logits loss
        logits_loss = get_logits_loss()
        
        #! Total loss
        loss = embedding_loss + layer_loss + logits_loss
        
        return embedding_loss, layer_loss, logits_loss, loss
