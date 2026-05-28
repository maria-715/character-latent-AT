import torch
from torch.optim.optimizer import Optimizer
import logging

INIT_TYPES = ["instruction", "suffix"]


class NoAttack:
    def __init__(
        self,
        embed_weights,
    ) -> None:
        """
        No Attack
        """
        self.embed_weights = embed_weights
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        return

    def attack(self, model, input_ids, target_ids, attention_mask):
        all_losses = []
        affirmative_responses = []

        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )
        input_embeds = self.get_embeddings(input_ids)

        stats = {}

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
            stats,
        )

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        return input_embeds

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        target_mask = target_ids > 0
        input_mask = (~target_mask * attention_mask).to(bool)
        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        adv_perturbation = torch.zeros(
            (batch_size, num_input_tokens, self.embedding_size), device=input_ids.device, dtype=dtype
        )

        adv_perturbation_mask = torch.zeros(
            (batch_size, num_input_tokens), device=input_ids.device, dtype=input_ids.dtype
        )
        adv_perturbation_mask[input_mask] = 1

        return adv_perturbation, adv_perturbation_mask.unsqueeze(-1)

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(
            batch_size, num_tokens, self.vocab_size, device=device, dtype=self.embed_weights.dtype
        )
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings


class EmbeddingSpaceAttack:
    def __init__(
        self,
        embed_weights,
        response_key,
        tokenizer,
        iters=8,
        opt_config=None,
        eps=0.01,
        init_type="instruction",
        suffix_tokens=10,
        relative_lr=False,
        debug=0,
        reachability_config=None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the EmbeddingAttack class.

        Args:
            embed_weights (torch.Tensor): The token embedding matrix of the respective model.
            response_key (str): The response key of the datacollator used to separate instructions and targets.
            tokenizer (Tokenizer): The tokenizer object.
            iters (int, optional): The number of iterations.
            opt_config (dict, optional): The optimizer configuration.
            eps (float, optional): The epsilon value.
            init_type (str, optional): The initialization type
            suffix_tokens (int, optional): The number of suffix tokens
            debug (int, optional): The debug level with increasing verbosity. (0 None, 1 print final loss, 2 include loss at every iteration, 3 include norms and shapes)
        """

        # For RCAT, always disabling it
        relative_lr = False

        self.embed_weights = embed_weights
        # self.response_key = response_key  # TODO Needed? Remove if not (also line bellow and argument)
        # self.response_key_ids = tokenizer.encode(response_key, return_tensors="pt")
        self.tokenizer = tokenizer
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        self.embedding_norm = torch.norm(embed_weights, p=2, dim=-1).mean()
        self.iters = iters
        self.opt_config = opt_config
        if relative_lr:
            self.opt_config["lr"] = self.opt_config["lr"] * self.eps
        self.eps = eps * self.embedding_norm

        logging.info(
            f"L2 norm of embedding weights equals {self.embedding_norm} eps multiplier is: {eps} using eps: {self.eps}"
        )

        if init_type not in INIT_TYPES:
            ValueError(f"init_type must be in {INIT_TYPES} and not {self.init_type}")
        self.init_type = init_type

        self.suffix_tokens = suffix_tokens
        self.debug = debug
        self.loss_fct = torch.nn.CrossEntropyLoss()

        # ============= Reachability =============
        self.reachability_config = reachability_config
        if self.reachability_config["enable"]:
            self.reachability = ReachabilityPenalty(
                ReachabilityIndex(embed_weights, 
                                  metric=self.reachability_config["metric"], 
                                  use_torch_nn_search=self.reachability_config["use_torch_nn_search"]),
                lambda_weight=self.reachability_config["lambda_weight"],
                top_k=self.reachability_config["top_k"],
                tau=self.reachability_config["tau"],
                beta=self.reachability_config["beta"],
            )



    def attack(self, model, input_ids, target_ids, attention_mask):
        disable_model_gradients(model)
        
        # save losses and responses
        best_loss = torch.inf
        best_idx = -1
        best_stats = None

        all_losses = []
        affirmative_responses = []

        all_harm_losses = []
        all_reach_losses = []
        all_nn_dists = []
        all_nn_entropies = []
        all_nn_ids = []

        # init embeddings of input instruction and target and initialize adversarial perturbation
        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )

        logging.info(f"adv_perturbation.shape = {adv_perturbation.shape}")
        logging.info(f"adv_perturbation_mask.shape = {adv_perturbation_mask.shape}")
        print(f"adv_perturbation.shape = {adv_perturbation.shape}")
        print(f"adv_perturbation_mask.shape = {adv_perturbation_mask.shape}")
        
        # total tokens perturbed across the whole batch
        num_tokens = adv_perturbation_mask.sum().item()
        logging.info(f"num_perturbed_tokens = {num_tokens}")
        print(f"num_perturbed_tokens = {num_tokens}")

        # how many tokens are perturbed in each example
        tokens_per_sample = adv_perturbation_mask.sum(dim=1)
        logging.info(f"tokens_per_sample = {tokens_per_sample}")
        print(f"tokens_per_sample = {tokens_per_sample}")

        input_embeds = self.get_embeddings(input_ids)
        target_one_hot = self.get_one_hot(target_ids)

        # get attention mask. If we add an adversarial suffic we need to adapt it
        attention_mask = self.get_attention_mask(input_ids, attention_mask)

        # loss targets
        loss_mask = self.get_loss_mask(target_ids)

        # init opt
        opt = self.init_opt([adv_perturbation])

        if self.debug > 2:
            self.debug_shapes(
                input_embeds,
                target_one_hot,
                adv_perturbation,
                adv_perturbation_mask,
                attention_mask,
                loss_mask,
            )

        prev_nn_ids = None
        nn_change_between_steps = 0.0
        for i in range(self.iters):
            opt.zero_grad()
            adv_embeds = self.get_adv_embeddings(input_embeds, adv_perturbation, adv_perturbation_mask)
            logits, harmful_loss = self.calc_loss(i, model, adv_embeds, target_one_hot, attention_mask, loss_mask)

            # ========== Reachability ==========
            reach_loss = torch.tensor(0.0, device=adv_embeds.device)
            nn_dist = torch.tensor(0.0, device=adv_embeds.device)
            if self.reachability is not None:
                reach_loss, nn_dist, nn_ids, nn_entropy = self.reachability(adv_embeds, adv_perturbation_mask)
                reach_contribution = self.reachability.lambda_weight * reach_loss
                loss = harmful_loss + reach_contribution
            # ==================================
            else:
                loss = harmful_loss
                nn_entropy = torch.tensor(0.0, device=adv_embeds.device)

            loss.backward()
            all_losses.append(loss.item())
            # all_harm_losses.append(harmful_loss.item())
            # all_reach_losses.append(reach_loss.item())
            # all_nn_dists.append(nn_dist.item())
            # all_nn_entropies.append(nn_entropy.item())
            opt.step()

            '''  # REMOVING THE LP-BALL CONSTRAINT
            # perturbations that are applied directly to the instruction are constraint to a small lp ball
            if self.init_type == "instruction":
                adv_perturbation = self.project_l2(adv_perturbation)
            # perturbations that are added as a suffix should be projected to the simplex
            elif self.init_type == "suffix":
                adv_perturbation = self.project_simplex(adv_perturbation)
            '''

            num_affirmative_responses = self.get_num_affirmative_responses(target_ids, logits)
            affirmative_responses.append(num_affirmative_responses)

            # save best perturbation (I added the item())
            if loss.item() < best_loss:
                best_loss = loss
                self.best_adv_perturbation = adv_perturbation

                # TEMPORARY BC DETACH() TAKES A LOT OF TIME
                # nn_ids already filtered by the mask
                nn_ids_cpu = nn_ids.detach().cpu()
                mask_cpu = adv_perturbation_mask.detach().cpu()

                best_nn_ids = nn_ids_cpu.clone()
                best_idx = i

                # NN change vs original input
                # - Flatten everything consistently
                original_ids = input_ids.detach().cpu().view(-1)
                mask_flat = mask_cpu.view(-1)

                # - Select only masked positions
                original_ids_masked = original_ids[mask_flat == 1]
                nn_ids_masked = nn_ids_cpu.view(-1)

                # Sanity check
                assert len(original_ids_masked) == len(nn_ids_masked), \
                    f"Mismatch: original {len(original_ids_masked)} vs nn_ids {len(nn_ids_masked)}"

                nn_change_from_input = (
                    (nn_ids_masked != original_ids_masked).float().mean().item()
                )

                # NN switching
                if prev_nn_ids is not None:
                    curr_flat = nn_ids_cpu.view(-1)
                    prev_flat = prev_nn_ids.view(-1)
                    nn_change_between_steps = (
                        (curr_flat != prev_flat).float().mean().item()
                    )

                prev_nn_ids = nn_ids_cpu.clone()

                # Calculate margins between NNs (for top_k >= 2)
                if hasattr(self.reachability, "last_Dists") and self.reachability.last_Dists is not None:
                    Dists = self.reachability.last_Dists # [N, K]
                    if Dists.shape[1] > 1:
                        margins = Dists[:, 1] - Dists[:, 0]
                        margin = margins.mean().item()
                        margin_std = margins.std().item()
                    else:
                        margin = 0.0
                        margin_std = 0.0
                else:
                    margin = 0.0
                    margin_std = 0.0

                # NN diversity (masked)
                unique_nn_tokens = len(torch.unique(nn_ids_masked))

                # Dist stats
                if nn_dist.dim() > 0:
                    nn_dist_std = nn_dist.std().item()
                    nn_dist_min = nn_dist.min().item()
                    nn_dist_max = nn_dist.max().item()
                else:
                    nn_dist_std = 0.0
                    nn_dist_min = nn_dist.item()
                    nn_dist_max = nn_dist.item()

                # Perturbation
                token_norms = adv_perturbation.norm(dim=-1) # [B, T]
                actual_perturbed_fraction = (
                    (token_norms > 1e-6).float().mean().item()
                )

                grad_norm = (
                    adv_perturbation.grad.norm().item()
                    if adv_perturbation.grad is not None
                    else 0.0
                )

                best_stats = {
                    "index": best_idx,
                    "attack_loss": loss.item(),
                    "harm_loss": harmful_loss.item(),
                    "reach_loss": reach_loss.item(),
                    "nn_entropy": nn_entropy.item(),
                    "affirmative": num_affirmative_responses,
                    "adv_norm": adv_perturbation.reshape(adv_perturbation.size(0), -1).norm(p=2, dim=1).mean().item(),
                    "nn_ids": best_nn_ids,
                    "original_ids": input_ids.detach().cpu().clone(),
                    "mask": mask_cpu,
                    "nn_change_from_input": nn_change_from_input,
                    "nn_change_between_steps": nn_change_between_steps,
                    "nn_margin": margin,
                    "nn_margin_std": margin_std,
                    "nn_unique_tokens": unique_nn_tokens,
                    "actual_perturbed_fraction": actual_perturbed_fraction,
                    "reach_to_harm_ratio": (reach_loss / (harmful_loss + 1e-8)).item(),                    
                    "grad_norm": grad_norm,
                    "nn_dist_std": nn_dist_std,
                    "nn_dist_min": nn_dist_min,
                    "nn_dist_max": nn_dist_max,
                }

            # debug
            if self.debug > 2:
                self.debug_norm(adv_perturbation)

        if self.debug > 0:
            self.debug_loss(best_loss)
        if self.debug > 2:
            self.debug_output(target_ids, logits, attention_mask)

        enable_model_gradients(model)


        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
            best_stats,
        )

    def calc_loss(self, i, model, input_embeds, target_one_hot, attention_mask, loss_mask, log_debug=True):
        output = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = output.logits  # ignore predicted token (not in targets)

        logits_loss_mask = logits[:, :-1][loss_mask]
        target_one_hot_loss_mask = target_one_hot[:, 1:][loss_mask]  # ignore first token (not in predicted)
        loss = self.loss_fct(logits_loss_mask, target_one_hot_loss_mask)

        if i == 0:
            self.logits_benign = logits
            self.loss_benign = loss

        if self.debug > 1 and log_debug:
            self.debug_iter_loss(i, loss)

        return logits, loss

    def project_l2(self, adv_perturbation):
        norm = torch.norm(adv_perturbation, p=2, dim=-1, keepdim=True)
        mask = (norm > self.eps).squeeze()
        if torch.any(mask):
            with torch.no_grad():
                if len(mask.shape) == 1:  # batch size 1
                    mask = mask.unsqueeze(0)
                adv_perturbation[mask, :] = adv_perturbation[mask, :] / norm[mask] * self.eps

        return adv_perturbation

    def project_simplex(self, adv_perturbation):
        # TODO --> suffix perturbations should be optimized in the one-hot space and not at the embedding space
        raise NotImplementedError("Simplex projection not implemented yet")

    def get_one_hot(self, ids):
        device = self.embed_weights.device
        batch_size, num_tokens = ids.shape

        # Adjusting IDs less than 0 to 0
        ids = torch.where(ids < 0, torch.tensor(0, device=device, dtype=ids.dtype), ids)

        one_hot = torch.zeros(
            batch_size, num_tokens, self.vocab_size, device=device, dtype=self.embed_weights.dtype
        )
        one_hot.scatter_(2, ids.unsqueeze(2), 1)
        return one_hot

    def get_embeddings(self, ids):
        one_hot = self.get_one_hot(ids)
        embeddings = (one_hot @ self.embed_weights).data
        return embeddings

    def get_adv_embeddings(self, input_embeds, adv_perturbation, adv_perturbation_mask):
        masked_perturbation = adv_perturbation * adv_perturbation_mask
        adv_embeds = input_embeds + masked_perturbation
        return adv_embeds

    def get_loss_slice_start_and_end(self, input_embeds):
        input_len = input_embeds.shape[1]

        if self.init_type == "instruction":
            start = input_len - 1
            end = -1
        elif self.init_type == "suffix":
            start = input_len + self.suffix_tokens
            end = -1
        return start, end

    def get_attention_mask(self, input_ids, attention_mask):
        if self.init_type == "instruction":
            return attention_mask
        elif self.init_type == "suffix":
            len_input = input_ids.shape[1]
            input_mask = attention_mask[:, :len_input]
            adversarial_mask = torch.ones(
                (attention_mask.shape[0], self.suffix_tokens),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            target_mask = attention_mask[:, len_input:]
            attention_mask = torch.hstack([input_mask, adversarial_mask, target_mask])
            return attention_mask

    def get_loss_mask(self, target_ids):
        target_mask = target_ids > 0
        if self.init_type == "instruction":
            return target_mask[:, 1:]  # ignore first token (not in predicted)
        elif self.init_type == "suffix":
            padding_for_suffix = torch.zeros(
                (target_mask.shape[0], self.suffix_tokens), dtype=target_ids.dtype, device=target_ids.device
            )
            loss_mask = torch.hstack([padding_for_suffix, target_mask])
        return loss_mask

    def init_perturbation(self, input_ids, target_ids, attention_mask):
        target_mask = target_ids > 0
        input_mask = (~target_mask * attention_mask).to(bool)
        batch_size, num_input_tokens = input_ids.shape
        dtype = self.embed_weights.dtype

        if self.init_type == "instruction":
            adv_perturbation = torch.zeros(
                (batch_size, num_input_tokens, self.embedding_size), device=input_ids.device, dtype=dtype
            )

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens), device=input_ids.device, dtype=input_ids.dtype
            )
            adv_perturbation_mask[input_mask] = 1
        elif self.init_type == "suffix":
            adv_perturbation = torch.randn(
                (batch_size, num_input_tokens + self.suffix_tokens, self.embedding_size),
                device=input_ids.device,
                dtype=dtype,
            )
            adv_perturbation = self.project_simplex(adv_perturbation)

            adv_perturbation_mask = torch.zeros(
                (batch_size, num_input_tokens + self.suffix_tokens),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            # get indexes of suffix tokens and set them to one
            num_false = torch.sum(input_mask, dim=1)  # TODO FIX as input mask changed
            row_indices = torch.arange(adv_perturbation_mask.shape[0])
            col_indices = num_false[:, None] + torch.arange(self.suffix_tokens)
            col_indices = torch.clip(col_indices, 0, adv_perturbation_mask.shape[1] - 1)
            adv_perturbation_mask[row_indices[:, None], col_indices] = True

        adv_perturbation.requires_grad = True
        adv_perturbation_mask = adv_perturbation_mask.unsqueeze(2)

        # save best adv_perturbation
        self.best_adv_perturbation = adv_perturbation

        return adv_perturbation, adv_perturbation_mask

    def init_opt(self, parameters):
        if self.opt_config is None:
            self.opt_config = {"type": "sign", "lr": 0.01}
            logging.info(f"No opt_config specified using default opt_config: {self.opt_config}")

        optimizer_type = self.opt_config["type"]
        if optimizer_type == "adam":
            opt = torch.optim.Adam(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "sign":
            opt = SignSGD(parameters, lr=self.opt_config["lr"])
        elif optimizer_type == "rms":
            opt = torch.optim.RMSprop(parameters, lr=self.opt_config["lr"])

        return opt

    def get_num_affirmative_responses(self, target_ids, logits):  # TODO Fix for suffix attack
        target_ids_clone = target_ids.clone()
        target_ids_clone = target_ids_clone[:, 1:]  # ignore first token (not in predicted)
        output_ids = torch.argmax(logits, dim=-1)
        output_ids = output_ids[:, :-1]  # ignore predicted token (not in target_ids)

        input_mask = target_ids_clone < 0
        output_ids[input_mask] = 0
        target_ids_clone[input_mask] = 0
        affirmative_responses = output_ids == target_ids_clone
        affirmative_responses_sum = affirmative_responses.all(dim=-1).sum().item()

        return affirmative_responses_sum

    def debug_norm(self, adv_perturbation):
        if self.init_type == "instruction":
            norm = torch.norm(adv_perturbation, p=2, dim=-1).max()
            logging.info(f"Debugging ESA | L2 Norm max adversarial perturbation: {norm}")
        # TODO print norm for simplex projection?

    def debug_loss(self, loss_adversarial):
        logging.info(
            f"Debugging ESA | Benign loss: {self.loss_benign.item()} | Best Adversarial loss {loss_adversarial.item()}"
        )

    def debug_iter_loss(self, i, loss_adversarial):
        logging.info(f"Debugging ESA | i: {i} | Adversarial loss: {loss_adversarial.item()}")

    def debug_shapes(
        self, input_embeds, target_one_hot, adv_perturbation, adv_perturbation_mask, attention_mask, loss_mask
    ):
        logging.info(
            "====== Debugging ESA Adversarial Attack Shapes ======\n"
            f"input_embeds: {input_embeds.shape}\n"
            f"adv_perturbation: {adv_perturbation.shape}\n"
            f"adv_perturbation_mask: {adv_perturbation_mask.shape}\n"
            f"target_one_hot: {target_one_hot.shape}\n"
            f"attention_mask: {attention_mask.shape}\n"
            f"loss_mask: {loss_mask.shape}"
        )

    def debug_output(self, target_ids, logits, attention_mask):
        with torch.no_grad():
            attention_mask = attention_mask.to(dtype=torch.bool)

            # only first sample; ignore first token (not in predicted tokens)
            target_ids = target_ids[0][1:]
            target_mask = target_ids > 0
            only_targets = target_ids[target_mask]
            original_text = self.tokenizer.decode(only_targets, skip_special_tokens=True)

            output_ids_benign = torch.argmax(self.logits_benign, dim=-1)
            # only first sample; only the target tokens (model reply); remove predicted token at the end
            output_ids_benign = output_ids_benign[0][:-1][target_mask]
            generated_text_benign = self.tokenizer.decode(output_ids_benign, skip_special_tokens=True)

            output_ids_adv = torch.argmax(logits, dim=-1)
            output_ids_adv = output_ids_adv[0][:-1][target_mask]
            generated_text_adv = self.tokenizer.decode(output_ids_adv, skip_special_tokens=True)
            logging.info(
                "===== Debugging ESA Original text ====\n"
                f"{original_text}\n"
                "===== Debugging ESA Generated text benign ====\n"
                f"{generated_text_benign}\n"
                "===== Debugging ESA Generated text adversarial ====\n"
                f"{generated_text_adv}"
            )

def disable_model_gradients(model):
    # logging.info("Disabled gradients")
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" not in name:
            raise ValueError(f"Non-Lora Parameter {name} requires grad")
        param.requires_grad = False


def enable_model_gradients(model, only_train_lora=True):
    # logging.info("Enabled gradients")
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

class SignSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(SignSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    sign = torch.sign(grad)

                    p.add_(other=sign, alpha=-group["lr"])

        return loss


# =========================== Reachability ===========================

import faiss
import numpy as np
import torch
import torch.nn.functional as F

class ReachabilityIndex:

    def __init__(self, embed_weights: torch.Tensor, metric: str = "l2", use_torch_nn_search: bool = True):
        self.metric = metric
        self.use_torch_nn_search = use_torch_nn_search
        self.embed_weights = embed_weights.detach().clone()

        if not self.use_torch_nn_search:
            # x base: the set of vectors stored in the index, shape: [vocab_size, d]
            xb = self.embed_weights.float().cpu().numpy().astype("float32")
            d = xb.shape[1]

            print(f"Vocab size: {xb.shape[0]}")
            logging.info(f"Vocab size: {xb.shape[0]}")
            
            # Brute-force searches
            if metric == "cosine":
                faiss.normalize_L2(xb) # Convert the metric to inner product
                self.index = faiss.IndexFlatIP(d)
            else:
                self.index = faiss.IndexFlatL2(d)

            self.index.add(xb)

        else:
            self.index = None

    def _search_faiss(self, q: torch.Tensor, top_k: int = 1):

        # q: batch of query vectors, shape: [N, d]
        xq = q.detach().float().cpu().numpy().astype("float32")
        if self.metric == "cosine":
            faiss.normalize_L2(xq)

        # D: distances (or inner products for cosine)
        # I: indices
        # Both have shape [N, top_k]
        D, I = self.index.search(xq, top_k)
        return (
            torch.from_numpy(D).to(q.device),
            torch.from_numpy(I).to(q.device),
        )


    def _search_torch(self, q: torch.Tensor, top_k: int = 1):
        # hard top_k selection is still discrete 
        # but returned distances computed in torch and usable in backprop
        E = self.embed_weights.to(q.device).float() # [V, D]
        q = q.float() # [N, D]

        if self.metric == "cosine":
            qn = F.normalize(q, dim=-1) # [N, D]
            En = F.normalize(E, dim=-1) # [V, D]
            scores = qn @ En.T # [N, V] cosine similarity
            D, I = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
            return D, I

        # L2 mode (squared Euclidean distance, matching FAISS IndexFlatL2)
        # dist^2(q,e) = ||q||^2 + ||e||^2 - 2 q.e
        q2 = (q ** 2).sum(dim=-1, keepdim=True) # [N, 1]
        e2 = (E ** 2).sum(dim=-1).unsqueeze(0) # [1, V]
        scores = q2 + e2 - 2.0 * (q @ E.T) # [N, V] squared L2 distance
        D, I = torch.topk(scores, k=top_k, dim=-1, largest=False, sorted=True)
        return D, I

    def search(self, q: torch.Tensor, top_k: int = 1):
            
        # Returns D: [N, top_k], I: D: [N, top_k]
        # L2 mode: D is squared L2 distance in both differentiability modes
        # Cosine mode: D is cosine similarity if use_torch_nn_search=False with FAISS IndexFlatIP
        #              D is cosine similarity if use_torch_nn_search=True too
        if self.use_torch_nn_search:
            return self._search_torch(q, top_k=top_k)
        else:
            return self._search_faiss(q, top_k=top_k)
    
# Differentiable regularisation loss
class ReachabilityPenalty:
    def __init__(self, reach_index: ReachabilityIndex, lambda_weight=50.0, top_k=1, tau=20.0, beta=0.0):
        self.reach_index = reach_index
        self.lambda_weight = lambda_weight
        self.top_k = top_k
        self.tau = tau
        self.beta = beta

    def __call__(self, adv_embeds: torch.Tensor, mask: torch.Tensor):
        # adv_embeds: current perturbed embeddings [B, T, D], mask: to ignore padding [B, T]
        B, T, D = adv_embeds.shape
        flat_q = adv_embeds.reshape(-1, D)
        flat_mask = mask.reshape(-1).bool()

        # Get nearest token embeddings
        Dists, Indices = self.reach_index.search(flat_q, top_k=self.top_k)
        self.last_Dists = Dists.detach()

        token_embed_matrix = self.reach_index.embed_weights.to(adv_embeds.device)
        cand = token_embed_matrix[Indices] # [N, K, D]
        q = flat_q[:, None, :] # [N, 1, D]

        if self.reach_index.metric == "cosine":
            qn = F.normalize(q, dim=-1)
            candn = F.normalize(cand, dim=-1)
            cand_dist = 1.0 - (qn * candn).sum(dim=-1) # [N, K], cosine distance
        else:
            cand_dist = torch.norm(q - cand, dim=-1) # [N, K], L2 distance

        weights = torch.softmax(-self.tau * cand_dist, dim=-1)

        # Discourage "in-between" states
        entropy = torch.zeros(flat_q.shape[0], device=adv_embeds.device, dtype=adv_embeds.dtype)
        if self.beta > 0.0:
            entropy = -(weights * (weights + 1e-12).log()).sum(dim=-1)

        penalty_per_token = (weights * cand_dist).sum(dim=-1) + self.beta * entropy
        penalty = penalty_per_token[flat_mask].mean()

        # For cosine, Dists[:, 0] is similarity, not distance
        # So, need to reverse
        if self.reach_index.metric == "cosine":
            nn_dist = (1.0 - Dists[:, 0])[flat_mask].mean()
        else:
            nn_dist = Dists[:, 0][flat_mask].mean()
        nn_ids = Indices[:, 0][flat_mask]

        return penalty, nn_dist, nn_ids, entropy[flat_mask].mean()
    
