import torch
from torch.optim.optimizer import Optimizer
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
import math

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

        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
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
        tracking_config: Optional[Dict[str, Any]] = None,
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

        self.embed_weights = embed_weights
        self.embed_norms = (self.embed_weights ** 2).sum(dim=1)
        # self.response_key = response_key  # TODO Needed? Remove if not (also line bellow and argument)
        # self.response_key_ids = tokenizer.encode(response_key, return_tensors="pt")
        self.tokenizer = tokenizer
        self.vocab_size = self.embed_weights.shape[0]
        self.embedding_size = self.embed_weights.shape[1]
        self.embedding_norm = float(torch.norm(embed_weights, p=2, dim=-1).mean().item())
        self.iters = iters
        self.opt_config = opt_config
        self.raw_eps_multiplier = eps
        self.eps = float(eps) * self.embedding_norm
        if relative_lr:
            self.opt_config["lr"] = float(self.opt_config["lr"]) * self.eps / math.sqrt(self.embedding_size) # ADDED division for SignSGD

        logging.info(
            f"L2 norm of embedding weights equals {self.embedding_norm}. "
            f"eps multiplier is: {eps}. using eps: {self.eps}. "
            f"the learning rate is: {self.opt_config['lr']} (relative_lr={relative_lr})."
        )

        if init_type not in INIT_TYPES:
            ValueError(f"init_type must be in {INIT_TYPES} and not {self.init_type}")
        self.init_type = init_type

        self.suffix_tokens = suffix_tokens
        self.debug = debug
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.tracking_config = {
            "enabled": True,
            "top_k": 5,
            "store_step_summaries": True,
            "store_token_records": True,
            "store_topk_neighbours": True,
            "nearest_neighbor_chunk_size": 128,
        }
        if tracking_config is not None:
            self.tracking_config.update(tracking_config)

        self.best_adv_perturbation = None
        self.last_attack_trace = None

        # ================ Test ================
        print("eps =", self.eps)
        print("embedding_size =", self.embedding_size)
        print("lr =", self.opt_config["lr"])
        print("lr*sqrt(d) =", self.opt_config["lr"] * math.sqrt(self.embedding_size))
        # ======================================


    def attack(self, model, input_ids, target_ids, attention_mask):
        disable_model_gradients(model)

        attack_id = str(uuid.uuid4())
        
        # save losses and responses
        best_loss = float("inf")
        best_iter = -1
        all_losses = []
        affirmative_responses = []

        attack_trace = {
            "attack_id": attack_id,
            "attack_type": "embedding_space",
            "config": {
                "iters": int(self.iters),
                "eps": float(self.eps),
                "raw_eps_multiplier": float(self.raw_eps_multiplier),
                "embedding_norm": float(self.embedding_norm),
                "init_type": self.init_type,
                "optimizer": self.opt_config,
                "tracking": self.tracking_config,
                "batch_size": int(input_ids.shape[0]),
                "seq_len": int(input_ids.shape[1]),
            },
            "step_summaries": [],
            "token_records": [],
            "best": {},
            "final": {},
        }

        # init embeddings of input instruction and target and initialize adversarial perturbation
        adv_perturbation, adv_perturbation_mask = self.init_perturbation(
            input_ids, target_ids, attention_mask
        )
        
        input_embeds = self.get_embeddings(input_ids)
        target_one_hot = self.get_one_hot(target_ids)

        # get attention mask. If we add an adversarial suffix we need to adapt it
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

        logits = None


        # before the loop
        prev_update = None
        prev_delta = None
        for i in range(self.iters):
            opt.zero_grad()

            prev_adv_perturbation = adv_perturbation.detach().clone()

            adv_embeds = self.get_adv_embeddings(input_embeds, adv_perturbation, adv_perturbation_mask)
            logits, loss = self.calc_loss(i, model, adv_embeds, target_one_hot, attention_mask, loss_mask)
            loss.backward()
            loss_value = float(loss.item())
            all_losses.append(loss_value)
            opt.step()

            # ============ For test ============
            with torch.no_grad():
                update = (adv_perturbation - prev_adv_perturbation) * adv_perturbation_mask
                mask_to_remove_unperturbed = adv_perturbation_mask.squeeze(-1).bool()
                masked = adv_perturbation * adv_perturbation_mask

                update_norm = torch.norm(update, p=2, dim=-1)
                masked_update_norm = update_norm[mask_to_remove_unperturbed]
                mean_norm = torch.norm(masked, p=2, dim=-1).mean()
                max_norm = torch.norm(masked, p=2, dim=-1).max()

                print(
                    f"step={i} before_proj "
                    f"update_mean={masked_update_norm.mean().item():.6f} "
                    f"update_max={masked_update_norm.max().item():.6f} "
                    f"preproj_mean_norm={mean_norm:.6f} "
                    f"preproj_max_norm={max_norm:.6f} "
                    f"update_wrt_eps={(masked_update_norm.mean().item() / self.eps):.6f} "
                    f"eps={self.eps:.6f}"
                )

                # Analyse the rotation
                if i > 0:
                    update_flat = update[mask_to_remove_unperturbed]
                    prev_update_flat = prev_update[mask_to_remove_unperturbed]

                    # step-to-step direction change
                    cos_step = torch.nn.functional.cosine_similarity(
                        update_flat, prev_update_flat, dim=-1, eps=1e-12
                    )
                    angle_step_deg = torch.rad2deg(
                        torch.acos(torch.clamp(cos_step, -1.0, 1.0))
                    )

                    # radial direction change: is the update still pushing away from the origin?
                    delta_flat = masked[mask_to_remove_unperturbed]
                    delta_norm = torch.norm(delta_flat, p=2, dim=-1, keepdim=True)
                    radial_dir = delta_flat / (delta_norm + 1e-12)

                    cos_radial = torch.nn.functional.cosine_similarity(
                        update_flat, radial_dir, dim=-1, eps=1e-12
                    )
                    angle_radial_deg = torch.rad2deg(
                        torch.acos(torch.clamp(cos_radial, -1.0, 1.0))
                    )

                    print(
                        f"step={i} rotation "
                        f"step_cos_mean={cos_step.mean().item():.6f} "
                        f"step_angle_mean_deg={angle_step_deg.mean().item():.2f} "
                        f"radial_cos_mean={cos_radial.mean().item():.6f} "
                        f"radial_angle_mean_deg={angle_radial_deg.mean().item():.2f}"
                    )

                    prev_update = update.detach().clone()
                else:
                    prev_update = update.detach().clone()


            # ==================================

            # perturbations that are applied directly to the instruction are constraint to a small lp ball
            if self.init_type == "instruction":
                adv_perturbation = self.project_l2(adv_perturbation)
            # perturbations that are added as a suffix should be projected to the simplex
            elif self.init_type == "suffix":
                adv_perturbation = self.project_simplex(adv_perturbation)

            num_affirmative_responses = self.get_num_affirmative_responses(target_ids, logits)
            affirmative_responses.append(num_affirmative_responses)

            if self.tracking_config["enabled"]:
                step_summary, token_records = self.collect_embedding_space_records(
                    attack_id=attack_id,
                    step=i,
                    loss=loss_value,
                    affirmative_responses=num_affirmative_responses,
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    adv_perturbation=adv_perturbation,
                    adv_perturbation_mask=adv_perturbation_mask,
                    top_k=int(self.tracking_config["top_k"]),
                    store_token_records=bool(self.tracking_config["store_token_records"]),
                    store_topk_neighbours=bool(self.tracking_config["store_topk_neighbours"]),
                    nearest_neighbor_chunk_size=int(
                        self.tracking_config["nearest_neighbor_chunk_size"]
                    ),
                )

                if self.tracking_config["store_step_summaries"]:
                    attack_trace["step_summaries"].append(step_summary)

                if self.tracking_config["store_token_records"]:
                    attack_trace["token_records"].extend(token_records)

                self.log_step_summary(i, step_summary)

          
            # save best perturbation
            if loss < best_loss:
                best_loss = loss_value
                best_iter = i
                self.best_adv_perturbation = adv_perturbation.detach().clone()

            # debug
            if self.debug > 2:
                self.debug_norm(adv_perturbation)

            if self.tracking_config["enabled"]:
                attack_trace["best"] = self.summarize_attack_trace(
                    attack_trace["step_summaries"]
                )

                final_summary, final_token_records = self.collect_embedding_space_records(
                    attack_id=attack_id,
                    step=self.iters - 1,
                    loss=all_losses[-1] if len(all_losses) > 0 else None,
                    affirmative_responses=(
                        affirmative_responses[-1] if len(affirmative_responses) > 0 else 0
                    ),
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    adv_perturbation=adv_perturbation,
                    adv_perturbation_mask=adv_perturbation_mask,
                    top_k=int(self.tracking_config["top_k"]),
                    store_token_records=True,
                    store_topk_neighbours=bool(self.tracking_config["store_topk_neighbours"]),
                    nearest_neighbor_chunk_size=int(
                        self.tracking_config["nearest_neighbor_chunk_size"]
                    ),
                )

                attack_trace["final"] = {
                    "summary": final_summary,
                    "token_records": final_token_records,
                }

                attack_trace["best"]["best_loss_manual"] = {
                    "step": int(best_iter),
                    "value": float(best_loss),
                }

                logging.info(f"[BEST] {attack_trace['best']}")
                self.log_final_summary(final_summary)

        self.last_attack_trace = attack_trace


        if self.debug > 0:
            self.debug_loss(torch.tensor(best_loss))
        if self.debug > 2:
            self.debug_output(target_ids, logits, attention_mask)

        enable_model_gradients(model)
        
        return (
            input_embeds.detach(),
            adv_perturbation.detach(),
            adv_perturbation_mask.detach(),
            all_losses,
            affirmative_responses,
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
    

    # ============================ Helper functions to collect stats ============================

    @torch.no_grad()
    def collect_embedding_space_records(
        self,
        *,
        attack_id,
        step,
        loss,
        affirmative_responses,
        input_ids,
        input_embeds,
        adv_perturbation,
        adv_perturbation_mask,
        top_k=5,
        store_token_records=True,
        store_topk_neighbours=True,
        nearest_neighbor_chunk_size=128,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Returns:
            step_summary:
                Compact statistics for this inner attack step.

            token_records:
                Flat per-token records. Each row corresponds to:
                one attack x one inner step x one perturbed token position.
        """

        adv_embeds = self.get_adv_embeddings(
            input_embeds,
            adv_perturbation,
            adv_perturbation_mask,
        )

        mask = adv_perturbation_mask.squeeze(-1).bool()

        if mask.sum().item() == 0:
            step_summary = {
                "attack_id": attack_id,
                "step": int(step),
                "loss": None if loss is None else float(loss),
                "affirmative_responses": int(affirmative_responses),
                "num_perturbed_tokens": 0,
            }
            return step_summary, []

        batch_idx, token_pos = torch.where(mask)

        original_token_ids = input_ids[batch_idx, token_pos]
        original_embeds = input_embeds[batch_idx, token_pos]
        perturbed_embeds = adv_embeds[batch_idx, token_pos]

        l2_to_original = torch.norm(
            perturbed_embeds - original_embeds,
            p=2,
            dim=-1,
        )

        (
            topk_dist,
            topk_ids,
            nearest_dist,
            nearest_ids,
            original_dist,
            original_rank,
        ) = self.compute_nearest_neighbour_info_chunked(
            perturbed_embeds=perturbed_embeds,
            original_token_ids=original_token_ids,
            top_k=top_k,
            chunk_size=nearest_neighbor_chunk_size,
        )

        nearest_is_original = nearest_ids.eq(original_token_ids)
        original_in_topk = topk_ids.eq(original_token_ids.unsqueeze(1)).any(dim=1)

        l2_float = l2_to_original.float()
        nearest_float = nearest_dist.float()
        original_rank_float = original_rank.float()

        step_summary = {
            "attack_id": attack_id,
            "step": int(step),
            "loss": None if loss is None else float(loss),
            "affirmative_responses": int(affirmative_responses),
            "num_perturbed_tokens": int(mask.sum().item()),

            "mean_l2_to_original": float(l2_float.mean().item()),
            "median_l2_to_original": float(l2_float.median().item()),
            "p90_l2_to_original": float(torch.quantile(l2_float, 0.90).item()),
            "p99_l2_to_original": float(torch.quantile(l2_float, 0.99).item()),
            "min_l2_to_original": float(l2_float.min().item()),
            "max_l2_to_original": float(l2_float.max().item()),

            "mean_dist_to_nearest": float(nearest_float.mean().item()),
            "median_dist_to_nearest": float(nearest_float.median().item()),
            "p90_dist_to_nearest": float(torch.quantile(nearest_float, 0.90).item()),
            "p99_dist_to_nearest": float(torch.quantile(nearest_float, 0.99).item()),
            "min_dist_to_nearest": float(nearest_float.min().item()),
            "max_dist_to_nearest": float(nearest_float.max().item()),

            "mean_original_rank": float(original_rank_float.mean().item()),
            "median_original_rank": float(original_rank_float.median().item()),
            "max_original_rank": int(original_rank.max().item()),

            "frac_original_is_nearest": float(nearest_is_original.float().mean().item()),
            "frac_original_in_topk": float(original_in_topk.float().mean().item()),
            "frac_nearest_changed": float((~nearest_is_original).float().mean().item()),

            "mean_eps_fraction": float((l2_float / float(self.eps)).mean().item()),
            "max_eps_fraction": float((l2_float / float(self.eps)).max().item()),
        }

        if not store_token_records:
            return step_summary, []

        token_records = []

        for n in range(len(batch_idx)):
            orig_id = int(original_token_ids[n].item())
            nn_id = int(nearest_ids[n].item())

            row = {
                "attack_id": attack_id,
                "step": int(step),
                "loss": None if loss is None else float(loss),
                "affirmative_responses": int(affirmative_responses),

                "batch_index": int(batch_idx[n].item()),
                "token_position": int(token_pos[n].item()),

                "original_token_id": orig_id,
                "original_token": self.safe_decode_token(orig_id),

                "l2_to_original": float(l2_to_original[n].item()),

                "nearest_token_id": nn_id,
                "nearest_token": self.safe_decode_token(nn_id),
                "dist_to_nearest": float(nearest_dist[n].item()),

                "nearest_is_original": bool(nearest_is_original[n].item()),
                "original_in_topk": bool(original_in_topk[n].item()),
                "original_rank": int(original_rank[n].item()),
                "original_dist_from_vocab_matrix": float(original_dist[n].item()),
            }

            if store_topk_neighbours:
                topk_token_ids = [int(x.item()) for x in topk_ids[n]]
                topk_distances = [float(x.item()) for x in topk_dist[n]]

                row["topk_token_ids"] = topk_token_ids
                row["topk_tokens"] = [
                    self.safe_decode_token(token_id) for token_id in topk_token_ids
                ]
                row["topk_distances"] = topk_distances
                row["original_topk_rank"] = None

                for k, token_id in enumerate(topk_token_ids):
                    if token_id == orig_id:
                        row["original_topk_rank"] = int(k + 1)
                        break

            token_records.append(row)

        return step_summary, token_records

    @torch.no_grad()
    def compute_nearest_neighbour_info_chunked(
        self,
        *,
        perturbed_embeds,
        original_token_ids,
        top_k=5,
        chunk_size=128,
    ):
        """
        Computes top-k nearest vocab embeddings and original-token rank.

        This is chunked over perturbed token positions to avoid creating a giant
        [num_perturbed_tokens, vocab_size] distance matrix all at once.
        """

        topk_dist_chunks = []
        topk_id_chunks = []
        nearest_dist_chunks = []
        nearest_id_chunks = []
        original_dist_chunks = []
        original_rank_chunks = []

        num_positions = perturbed_embeds.shape[0]

        for start in range(0, num_positions, chunk_size):
            end = min(start + chunk_size, num_positions)

            chunk = perturbed_embeds[start:end]
            orig_ids = original_token_ids[start:end]

            chunk_norms = (chunk ** 2).sum(dim=1, keepdim=True)

            dist_sq = (
                chunk_norms
                + self.embed_norms.unsqueeze(0)
                - 2 * chunk @ self.embed_weights.T
            )
            dist_sq = torch.clamp(dist_sq, min=0.0)
            dists = torch.sqrt(dist_sq)

            k = min(top_k, dists.shape[1])
            topk_dist, topk_ids = torch.topk(
                dists,
                k=k,
                dim=1,
                largest=False,
            )

            row_idx = torch.arange(dists.shape[0], device=dists.device)
            original_dist = dists[row_idx, orig_ids]

            original_rank = (dists < original_dist.unsqueeze(1)).sum(dim=1) + 1

            topk_dist_chunks.append(topk_dist)
            topk_id_chunks.append(topk_ids)
            nearest_dist_chunks.append(topk_dist[:, 0])
            nearest_id_chunks.append(topk_ids[:, 0])
            original_dist_chunks.append(original_dist)
            original_rank_chunks.append(original_rank)

        topk_dist = torch.cat(topk_dist_chunks, dim=0)
        topk_ids = torch.cat(topk_id_chunks, dim=0)
        nearest_dist = torch.cat(nearest_dist_chunks, dim=0)
        nearest_ids = torch.cat(nearest_id_chunks, dim=0)
        original_dist = torch.cat(original_dist_chunks, dim=0)
        original_rank = torch.cat(original_rank_chunks, dim=0)

        return (
            topk_dist,
            topk_ids,
            nearest_dist,
            nearest_ids,
            original_dist,
            original_rank,
        )

    def safe_decode_token(self, token_id):
        try:
            return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except Exception:
            return str(token_id)

    def summarize_attack_trace(self, step_summaries):
        if len(step_summaries) == 0:
            return {}

        valid = [
            row for row in step_summaries
            if row.get("loss") is not None and row.get("num_perturbed_tokens", 0) > 0
        ]

        if len(valid) == 0:
            return {}

        def best_min(key):
            row = min(valid, key=lambda x: x[key])
            return {"step": int(row["step"]), "value": float(row[key])}

        def best_max(key):
            row = max(valid, key=lambda x: x[key])
            return {"step": int(row["step"]), "value": float(row[key])}

        return {
            "best_loss": best_min("loss"),
            "max_mean_l2_to_original": best_max("mean_l2_to_original"),
            "max_p90_l2_to_original": best_max("p90_l2_to_original"),
            "max_frac_nearest_changed": best_max("frac_nearest_changed"),
            "min_mean_dist_to_nearest": best_min("mean_dist_to_nearest"),
            "min_median_dist_to_nearest": best_min("median_dist_to_nearest"),
            "max_mean_original_rank": best_max("mean_original_rank"),
            "max_affirmative_responses": best_max("affirmative_responses"),
        }

    def log_step_summary(self, step, summary):
        if summary.get("num_perturbed_tokens", 0) == 0:
            logging.info(f"[Inner step {step}] no perturbed tokens")
            return

        logging.info(
            f"[Inner step {step}] "
            f"loss={summary['loss']:.4f} | "
            f"L2(orig): mean={summary['mean_l2_to_original']:.4f}, "
            f"median={summary['median_l2_to_original']:.4f}, "
            f"p90={summary['p90_l2_to_original']:.4f}, "
            f"max={summary['max_l2_to_original']:.4f} | "
            f"L2(nearest): mean={summary['mean_dist_to_nearest']:.4f}, "
            f"median={summary['median_dist_to_nearest']:.4f}, "
            f"min={summary['min_dist_to_nearest']:.4f} | "
            f"orig_is_nn={summary['frac_original_is_nearest']:.3f}, "
            f"orig_in_topk={summary['frac_original_in_topk']:.3f}, "
            f"mean_orig_rank={summary['mean_original_rank']:.2f}"
        )

    def log_final_summary(self, summary):
        if summary.get("num_perturbed_tokens", 0) == 0:
            logging.info("[FINAL] no perturbed tokens")
            return

        logging.info(
            f"[FINAL] "
            f"L2(orig): mean={summary['mean_l2_to_original']:.4f}, "
            f"median={summary['median_l2_to_original']:.4f}, "
            f"p90={summary['p90_l2_to_original']:.4f}, "
            f"max={summary['max_l2_to_original']:.4f} | "
            f"L2(nearest): mean={summary['mean_dist_to_nearest']:.4f}, "
            f"median={summary['median_dist_to_nearest']:.4f}, "
            f"min={summary['min_dist_to_nearest']:.4f} | "
            f"orig_is_nn={summary['frac_original_is_nearest']:.3f}, "
            f"orig_in_topk={summary['frac_original_in_topk']:.3f}, "
            f"mean_orig_rank={summary['mean_original_rank']:.2f}"
        )

        # ===========================================================================================







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

