import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys
import os
from config import ModelConfig

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from smi_ted_light.loadnew import load_smi_ted
import torch.nn.functional as F
from transformers.generation.utils import GenerationConfig
from typing import Optional, List


# ============================
# 1. 投影器：将分子特征映射到LLM空间
# ============================
class QueryAttentionProjector(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_queries, 
                 output_dim, 
                 num_heads):
        """
        查询注意力投影器：将变长的分子 Token 序列压缩并映射到 LLM 空间
        """
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 可学习的查询向量 (Learned Queries)
        self.query = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        
        # 多头注意力 (Cross-Attention)
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力后归一化
        self.post_attn_norm = nn.LayerNorm(input_dim)
        
        # 投影层 (从分子维度映射到 LLM 维度)
        self.proj = nn.Linear(input_dim, output_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 查询向量通常使用正态分布初始化
        nn.init.normal_(self.query, std=0.02)
        # 线性层使用 Xavier 初始化
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x, key_padding_mask=None):
        """
        x: [B_total_mols, L_mol, input_dim]
        key_padding_mask: [B_total_mols, L_mol] (True 表示 Padding)
        """
        B_total = x.size(0)
        
        # 1. 输入归一化
        x_norm = self.input_norm(x)
        
        # 2. 准备查询向量 [B_total, num_queries, input_dim]
        q = self.query.expand(B_total, -1, -1)
        
        # 3. 交叉注意力：Queries 关注分子的 Encoder 输出
        attn_out, _ = self.attn(
            query=q, 
            key=x_norm, 
            value=x_norm, 
            key_padding_mask=key_padding_mask
        )
        
        # 4. 残差连接 + 归一化 (Post-LN)
        # 注意：这里 query (q) 充当了残差的骨架
        out = self.post_attn_norm(q + attn_out)
        
        # 5. 最终投影到 LLM 维度
        out = self.proj(out)
        
        return out


# ============================
# 1b. Stage 3 Memory Updater: 更新 BIO Token 的隐藏状态
# ============================
class BioTokenUpdater(nn.Module):
    def __init__(self, d_llm, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_llm, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_llm)
        self.ffn = nn.Sequential(
            nn.Linear(d_llm, d_llm * 2),
            nn.ReLU(),
            nn.Linear(d_llm * 2, d_llm)
        )
        self.norm_ffn = nn.LayerNorm(d_llm)

    def forward(self, bio_embeds, latent_states):
        """
        bio_embeds: [B, N_bio, d_llm]
        latent_states: [B, N_latent, d_llm]
        """
        # Cross-attention: Bio tokens attend to latent hidden states
        attn_out, _ = self.cross_attn(query=bio_embeds, key=latent_states, value=latent_states)
        bio_embeds = self.norm(bio_embeds + attn_out)
        
        # FFN
        ffn_out = self.ffn(bio_embeds)
        bio_embeds = self.norm_ffn(bio_embeds + ffn_out)
        return bio_embeds


# ============================
# 2. 多模态融合模型 (兼容trl的SFTTrainer)
# ============================
class Qwen3MoleculeLLM(PreTrainedModel):
    def __init__(self, 
                 qwen_model_name,
                 mol_config,       # 🚨 必须传入配置字典
                 device_map=None,
                 torch_dtype=torch.bfloat16):
        """
        分子-文本多模态大语言模型
        
        参数:
            qwen_model_name: Qwen基础模型路径
            mol_config: 包含 input_dim, num_queries, num_heads 等参数的字典
            device_map: 设备映射配置
        """
        # 加载Qwen模型的配置文件
        config = PretrainedConfig.from_pretrained(qwen_model_name)
        super().__init__(config)

        # 从 mol_config 解析参数
        self.num_queries = mol_config.get('num_queries', 128)
        self.mol_input_dim = mol_config.get('input_dim', 768)
        self.mol_num_heads = mol_config.get('num_heads', 8)
        self.smi_ted_folder = mol_config.get('smi_ted_folder', ModelConfig.DEFAULT_SMI_TED_FOLDER)
        self.smi_ted_ckpt = mol_config.get('smi_ted_ckpt', ModelConfig.DEFAULT_SMI_TED_CKPT)

        # ---- 1. 加载预训练的Qwen LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        self.config._name_or_path = qwen_model_name

        # 添加分子特殊标记
        self.extra_tokens = ["<mol_start>", "<mol_end>", "<latent>", "<start_latent>", "<end_latent>"]
        self.tokenizer.add_tokens(self.extra_tokens)

        # 加载基础语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            # IMPORTANT: fp32 doubles memory and will OOM easily for 8B models under GRPO.
            # Default to bf16 (good on Ampere/Hopper). Callers can override via `torch_dtype=...`.
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # 调整词表大小以包含新添加的特殊标记
        # 注意：我们使用 max() 确保词表大小不小于原始 config 中的 vocab_size，
        # 这样可以保持与 vLLM (加载原始 config.json) 的兼容性，避免权重加载时的 AssertionError。
        new_vocab_size = max(len(self.tokenizer), self.model.config.vocab_size)
        self.model.resize_token_embeddings(new_vocab_size)
        
        # 获取特殊标记的ID
        self.start_id = self.tokenizer.convert_tokens_to_ids("<mol_start>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<mol_end>")
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<latent>")
        self.start_latent_id = self.tokenizer.convert_tokens_to_ids("<start_latent>")
        self.end_latent_id = self.tokenizer.convert_tokens_to_ids("<end_latent>")

        # 获取LLM的嵌入维度
        self.d_llm = self.model.get_input_embeddings().weight.shape[1]

        # ---- 2. 分子编码器和投影器 ----
        # 加载预训练的分子编码器（SMI-TED）
        self.mol_encoder = load_smi_ted(
            folder=self.smi_ted_folder,
            ckpt_filename=self.smi_ted_ckpt
        )
        
        # 冻结分子编码器参数
        for param in self.mol_encoder.parameters():
            param.requires_grad = False
        self.mol_encoder.eval()
        
        # 初始化投影器，使用动态解析的参数
        self.projector = QueryAttentionProjector(
            input_dim=self.mol_input_dim,
            num_queries=self.num_queries,
            output_dim=self.d_llm,
            num_heads=self.mol_num_heads
        )
        # 确保投影器类型与基础模型一致
        self.projector.to(self.model.dtype)

        # ---- Stage 3: Bio Token Updater ----
        self.bio_updater = BioTokenUpdater(d_llm=self.d_llm, nhead=self.mol_num_heads)
        self.bio_updater.to(self.model.dtype)

    # ---- Liger Kernel & Compatibility Helpers ----
    def _get_actual_llm(self):
        """Helper to get the underlying ForCausalLM model, even if wrapped in Peft."""
        llm = self.model
        if hasattr(llm, "base_model") and hasattr(llm.base_model, "model"):
            return llm.base_model.model
        return llm

    @property
    def norm(self):
        return self._get_actual_llm().model.norm

    @property
    def layers(self):
        return self._get_actual_llm().model.layers

    @property
    def embed_tokens(self):
        return self._get_actual_llm().model.embed_tokens

    @property
    def lm_head(self):
        return self._get_actual_llm().lm_head

    def gradient_checkpointing_enable(self, **kwargs):
        """
        开启梯度检查点，转发给内部的语言模型。
        """
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)
            
    def gradient_checkpointing_disable(self):
        """
        关闭梯度检查点。
        """
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _apply_latent_feedback(self, initial_embeds, attention_mask, latent_positions, bio_positions=None, refine_bio_tokens=True):
        """
        Coconut 核心逻辑：潜空间迭代反馈。
        包含：更新 BIO Token 的隐藏状态 (Evidence Refinement)。
        """
        B = initial_embeds.shape[0]
        max_n_latents = max(len(l) for l in latent_positions) if latent_positions else 0
        
        if max_n_latents == 0:
            return initial_embeds

        curr_embeds = initial_embeds
        for pass_idx in range(max_n_latents):
            # 执行前向传播获取隐藏状态
            outputs = self.model(
                inputs_embeds=curr_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            hidden_states = outputs.hidden_states[-1]
            
            new_embeds = curr_embeds.clone()

            # --- Evidence Refinement (Memory Write) ---
            if refine_bio_tokens and bio_positions is not None:
                # 1. 识别当前 Pass 需要更新的 Batch 索引
                active_indices = [b for b in range(B) if bio_positions[b] and len(latent_positions[b]) > pass_idx]
                
                if active_indices:
                    # 2. 批量提取并补齐 BIO tokens [B_active, max_N_bio, d_llm]
                    bios = [curr_embeds[b, bio_positions[b]] for b in active_indices]
                    batched_bio = torch.nn.utils.rnn.pad_sequence(bios, batch_first=True)
                    
                    # 3. 批量提取 Latent States [B_active, pass_idx + 1, d_llm]
                    lats = [hidden_states[b, latent_positions[b][:pass_idx + 1]] for b in active_indices]
                    batched_lat = torch.stack(lats)
                    
                    # 4. 批量过 Cross-Attention 更新
                    refined = self.bio_updater(batched_bio.to(self.model.dtype), batched_lat.to(self.model.dtype))
                    
                    # 5. 将更新后的结果写回 (Scatter back)
                    for i, b in enumerate(active_indices):
                        new_embeds[b, bio_positions[b]] = refined[i, :len(bio_positions[b])]

            # --- Standard Coconut Feedback ---
            # 识别当前 Pass 需要更新 latent feedback 的 Batch 索引
            feedback_indices = [b for b in range(B) if len(latent_positions[b]) > pass_idx]
            if feedback_indices:
                pos_indices = torch.tensor([latent_positions[b][pass_idx] for b in feedback_indices], device=device)
                valid_mask = pos_indices > 0
                if valid_mask.any():
                    f_b_idx = torch.tensor(feedback_indices, device=device)[valid_mask]
                    f_p_idx = pos_indices[valid_mask]
                    # 批量注入：将前一个位置的输出作为当前位置的输入
                    new_embeds[f_b_idx, f_p_idx] = hidden_states[f_b_idx, f_p_idx - 1]

            curr_embeds = new_embeds
            
        return curr_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        """
        增强版前向传播：支持分子证据精炼与逆向干扰
        """
        smiles_list = kwargs.pop("smiles", None)
        do_perturb = kwargs.pop("do_perturb", False) # 是否执行逆向干扰 (Counterfactual perturbation)

        if smiles_list is None:
            raise ValueError("必须提供smiles参数")

        B = len(smiles_list)
        device = self.model.device

        # =========================================================
        # 1. 分子特征拉平与批量投影 (优化性能)
        # =========================================================
        with torch.no_grad():
            # mol_emb_nested: [[Tensor(L1, 768), Tensor(L2, 768)], [Tensor(L3, 768)]]
            mol_emb_nested = self.mol_encoder.encode(smiles_list)

        flat_mols = []
        mol_counts = []
        for sample_mols in mol_emb_nested:
            mol_counts.append(len(sample_mols))
            flat_mols.extend(sample_mols)

        if flat_mols:
            # 批量投影：将不同长度的分子特征 Padding 到 Batch 内最长长度
            max_L_mol = max(m.size(0) for m in flat_mols)
            padded_mols = torch.zeros(len(flat_mols), max_L_mol, self.mol_input_dim, device=device, dtype=self.model.dtype)
            mol_key_padding_mask = torch.ones(len(flat_mols), max_L_mol, device=device, dtype=torch.bool)
            
            for i, m in enumerate(flat_mols):
                curr_L = m.size(0)
                padded_mols[i, :curr_L] = m.to(device=device, dtype=self.model.dtype)
                mol_key_padding_mask[i, :curr_L] = False # False 为有效位置
            
            # 批量过投影器
            flat_feats_llm = self.projector(padded_mols, key_padding_mask=mol_key_padding_mask) # [Total_Mols, num_queries, d_llm]
        else:
            flat_feats_llm = []

        # 获取 LLM 嵌入层
        embed = self.model.get_input_embeddings()
        with torch.no_grad():
            start_emb = embed(torch.tensor([[self.start_id]], device=device)) # [1, 1, d_llm]
            end_emb = embed(torch.tensor([[self.end_id]], device=device))   # [1, 1, d_llm]

        # =========================================================
        # 2. 结构还原与变长融合 (去文本 Padding)
        # =========================================================
        if inputs_embeds is None:
            text_emb = embed(input_ids)
        else:
            text_emb = inputs_embeds
        text_emb = text_emb.to(dtype=self.model.dtype)

        fused_samples_list = []
        fused_labels_list = []
        bio_positions_list = [] # Stage 3: 记录每个 sample 中 bio token 的索引
        cursor = 0

        for b in range(B):
            # 2.1 构造分子部分
            sample_mol_parts = []
            b_bio_indices = []
            current_mol_offset = 0

            for _ in range(mol_counts[b]):
                m_feat = flat_feats_llm[cursor].unsqueeze(0) # [1, num_queries, d_llm]
                
                # --- Stage 3: Counterfactual Bio Latent Dropout ---
                if self.training and do_perturb:
                    # 随机干扰：Dropout, Shuffle, Noise
                    noise_type = torch.randint(0, 3, (1,)).item()
                    if noise_type == 0:
                        m_feat = torch.zeros_like(m_feat) # Dropout
                    elif noise_type == 1:
                        # Shuffle across the query tokens
                        idx = torch.randperm(m_feat.size(1))
                        m_feat = m_feat[:, idx, :] # Shuffle
                    elif noise_type == 2:
                        m_feat = m_feat + torch.randn_like(m_feat) * 0.05 # Noise

                m_with_tags = torch.cat([start_emb, m_feat, end_emb], dim=1) # [1, num_queries+2, d_llm]
                sample_mol_parts.append(m_with_tags)
                
                # 记录 bio tokens 的相对位置 (跳过 <mol_start>)
                start_query_pos = current_mol_offset + 1
                end_query_pos = start_query_pos + self.num_queries
                b_bio_indices.extend(range(start_query_pos, end_query_pos))
                current_mol_offset += (self.num_queries + 2)
                cursor += 1
            
            bio_positions_list.append(b_bio_indices)
            mol_part = torch.cat(sample_mol_parts, dim=1) if sample_mol_parts else torch.zeros(1, 0, self.d_llm, device=device, dtype=self.model.dtype)

            # 2.2 提取真实文本内容 (通过 Mask 提取非 Padding 部分)
            if attention_mask is not None:
                non_pad_indices = attention_mask[b].bool()
                t_emb = text_emb[b][non_pad_indices] # [real_len, d_llm]
                t_lab = labels[b][non_pad_indices] if labels is not None else None
            else:
                t_emb = text_emb[b]
                t_lab = labels[b] if labels is not None else None

            # 2.3 融合
            sample_fused = torch.cat([mol_part, t_emb.unsqueeze(0)], dim=1)
            
            # 2.4 处理 Labels
            if t_lab is not None:
                # 屏蔽 Latent Token 的 CE Loss (在拼接前处理)
                # 我们不希望模型学习预测 <latent> 这个 ID，而是希望它学习有意义的隐藏状态
                t_lab_masked = t_lab.clone()
                is_latent = (input_ids[b][non_pad_indices] == self.latent_id)
                t_lab_masked[is_latent] = -100

                # 分子标记设为 -100 (不计算 Loss)
                mol_lab = torch.full((1, mol_part.size(1)), -100, device=device, dtype=t_lab.dtype)
                sample_lab = torch.cat([mol_lab, t_lab_masked.unsqueeze(0)], dim=1)
                fused_labels_list.append(sample_lab)
            
            fused_samples_list.append(sample_fused)

        # =========================================================
        # 3. 全局重新对齐 (统一 Padding)
        # =========================================================
        max_fused_L = max(s.size(1) for s in fused_samples_list)
        final_embeds = torch.zeros(B, max_fused_L, self.d_llm, device=device, dtype=self.model.dtype)
        final_attn_mask = torch.zeros(B, max_fused_L, device=device, dtype=torch.long)
        final_labels = torch.full((B, max_fused_L), -100, device=device, dtype=torch.long) if labels is not None else None
        
        for b in range(B):
            curr_L = fused_samples_list[b].size(1)
            final_embeds[b, :curr_L] = fused_samples_list[b]
            final_attn_mask[b, :curr_L] = 1
            if final_labels is not None:
                final_labels[b, :curr_L] = fused_labels_list[b]

        # =========================================================
        # 4. Coconut 潜空间推理逻辑 (如果包含 <latent>)
        # =========================================================
        has_latent = (input_ids == self.latent_id).any().item()
        refine_bio_tokens = kwargs.pop("refine_bio_tokens", True) # 是否开启分子证据精炼
        
        if not has_latent:
            # --- 标准 SFT 路径 ---
            outputs = self.model(
                inputs_embeds=final_embeds,
                attention_mask=final_attn_mask,
                labels=final_labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        else:
            # --- Coconut 路径 ---
            # 计算 latent token 的绝对索引
            latent_positions = []
            for b in range(B):
                mol_len_b = mol_counts[b] * (self.num_queries + 2)
                t_mask_b = attention_mask[b].bool()
                rel_latent_indices = (input_ids[b][t_mask_b] == self.latent_id).nonzero(as_tuple=True)[0]
                latent_positions.append((rel_latent_indices + mol_len_b).tolist())

            # 应用潜空间反馈
            final_embeds_with_thoughts = self._apply_latent_feedback(
                final_embeds, 
                final_attn_mask, 
                latent_positions,
                bio_positions=bio_positions_list,
                refine_bio_tokens=refine_bio_tokens
            )
            
            # 最终计算 Loss
            outputs = self.model(
                inputs_embeds=final_embeds_with_thoughts,
                attention_mask=final_attn_mask,
                labels=final_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        smiles_list: List[List[str]],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        refine_bio_tokens: bool = True,
    ):
        """
        Build fused prompt embeddings for generation.

        This mirrors `generate()` up to (and including) the Coconut latent-feedback refinement.
        It returns:
        - `prompt_embeds`: (B, L, d_llm) fused embeddings (left-padded)
        - `prompt_attn_mask`: (B, L) attention mask aligned to `prompt_embeds`

        This is used by vLLM generation paths that accept `prompt_embeds`.
        """
        device = input_ids.device
        B = input_ids.size(0)

        # =========================================================
        # 1. Molecule features: flatten + batch projection
        # =========================================================
        mol_emb_nested = self.mol_encoder.encode(smiles_list)

        flat_mols = []
        mol_counts = []
        for sample_mols in mol_emb_nested:
            mol_counts.append(len(sample_mols))
            flat_mols.extend(sample_mols)

        if flat_mols:
            max_L_mol = max(m.size(0) for m in flat_mols)
            padded_mols = torch.zeros(
                len(flat_mols), max_L_mol, self.mol_input_dim, device=device, dtype=self.model.dtype
            )
            mol_key_padding_mask = torch.ones(len(flat_mols), max_L_mol, device=device, dtype=torch.bool)

            for i, m in enumerate(flat_mols):
                curr_L = m.size(0)
                padded_mols[i, :curr_L] = m.to(device=device, dtype=self.model.dtype)
                mol_key_padding_mask[i, :curr_L] = False

            flat_feats_llm = self.projector(padded_mols, key_padding_mask=mol_key_padding_mask)
        else:
            flat_feats_llm = []

        # LLM embedding layer
        embed = self.model.get_input_embeddings()
        start_emb = embed(torch.tensor([[self.start_id]], device=device))
        end_emb = embed(torch.tensor([[self.end_id]], device=device))

        # =========================================================
        # 2. Reconstruct + fuse variable-length (strip text padding)
        # =========================================================
        text_emb = embed(input_ids).to(dtype=self.model.dtype)
        fused_samples_list = []
        bio_positions_list = []
        cursor = 0

        for b in range(B):
            sample_mol_parts = []
            b_bio_indices = []
            current_mol_offset = 0
            for _ in range(mol_counts[b]):
                m_feat = flat_feats_llm[cursor].unsqueeze(0)
                m_with_tags = torch.cat([start_emb, m_feat, end_emb], dim=1)
                sample_mol_parts.append(m_with_tags)

                start_query_pos = current_mol_offset + 1
                end_query_pos = start_query_pos + self.num_queries
                b_bio_indices.extend(range(start_query_pos, end_query_pos))
                current_mol_offset += (self.num_queries + 2)
                cursor += 1

            bio_positions_list.append(b_bio_indices)
            mol_part = (
                torch.cat(sample_mol_parts, dim=1)
                if sample_mol_parts
                else torch.zeros(1, 0, self.d_llm, device=device, dtype=self.model.dtype)
            )

            if attention_mask is not None:
                non_pad_indices = attention_mask[b].bool()
                t_emb = text_emb[b][non_pad_indices]
            else:
                t_emb = text_emb[b]

            fused_samples_list.append(torch.cat([mol_part, t_emb.unsqueeze(0)], dim=1))

        # =========================================================
        # 3. Left pad to batch max length (generation-style)
        # =========================================================
        max_fused_L = max(s.size(1) for s in fused_samples_list)
        prompt_embeds = torch.zeros(B, max_fused_L, self.d_llm, device=device, dtype=self.model.dtype)
        prompt_attn_mask = torch.zeros(B, max_fused_L, device=device, dtype=torch.long)

        diffs = []
        for b in range(B):
            curr_L = fused_samples_list[b].size(1)
            diff = max_fused_L - curr_L
            diffs.append(diff)
            prompt_embeds[b, diff:] = fused_samples_list[b]
            prompt_attn_mask[b, diff:] = 1

        # =========================================================
        # 4. Coconut latent-feedback refinement (optional based on presence of <latent>)
        # =========================================================
        has_latent = (input_ids == self.latent_id).any().item()
        if has_latent and attention_mask is not None:
            latent_positions = []
            final_bio_positions = []
            for b in range(B):
                mol_len_b = mol_counts[b] * (self.num_queries + 2)
                t_mask_b = attention_mask[b].bool()
                rel_latent_indices = (input_ids[b][t_mask_b] == self.latent_id).nonzero(as_tuple=True)[0]
                diff = diffs[b]
                latent_positions.append((rel_latent_indices + mol_len_b + diff).tolist())
                final_bio_positions.append([idx + diff for idx in bio_positions_list[b]])

            prompt_embeds = self._apply_latent_feedback(
                prompt_embeds,
                prompt_attn_mask,
                latent_positions,
                bio_positions=final_bio_positions,
                refine_bio_tokens=refine_bio_tokens,
            )

        return prompt_embeds, prompt_attn_mask

    @torch.no_grad()
    def generate(
        self,
        smiles_list: List[List[str]],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ):
        """
        同步更新后的生成方法：
        1. 支持变长分子和批量投影。
        2. 推理阶段自动改用左补齐 (Left Padding) 以确保生成质量。
        """
        # Backward compatible: allow passing List[str] (one SMILES per sample)
        if smiles_list and isinstance(smiles_list[0], str):
            smiles_list = [[s] for s in smiles_list]  # type: ignore[list-item]

        # Reuse the shared embedding builder (and latent feedback) so vLLM can share the same code path.
        prompt_embeds, prompt_attn_mask = self.get_prompt_embeddings(
            smiles_list=smiles_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            refine_bio_tokens=kwargs.get("refine_bio_tokens", True),
        )
        # 5. 调用生成
        outputs = self.model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs
        )

        return outputs


if __name__ == "__main__":
    # Test Initialization
    mol_config = {
        'num_queries': 8,
        'input_dim': 768,
        'num_heads': 2
    }
    model = Qwen3MoleculeLLM(
        qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
        mol_config=mol_config
    ).cuda()
    print("Stage 3 Model Initialized Successfully!")
