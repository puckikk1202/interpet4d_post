"""
PetGPT: prefix-LM GPT for generating pet motion VQ tokens
conditioned on MANO hand + SMPL body VQ tokens + audio features.

Sequence layout:
  [left_hand_codes(L) | right_hand_codes(R) | smpl_codes(S) | pet_rel_codes(Pr) | pet_codes(P)]

- Condition tokens (left + right + smpl): bidirectional attention among themselves
- pet_rel tokens: causal among themselves + full attention to condition
- Pet tokens: causal among themselves + full attention to condition + full attention to pet_rel
- Audio: cross-attention in every transformer block

Training: next-token prediction on pet_rel + pet portions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class CrossAttention(nn.Module):
    """Cross-attention: queries from target sequence, keys/values from context."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.kv_proj = nn.Linear(n_embd, 2 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, context):
        B, T, C = x.shape
        S = context.size(1)

        q = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, S, 2, self.n_head, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, S, D)
        k, v = kv.unbind(0)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_cross_attn=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.ln_ca = nn.LayerNorm(n_embd)
            self.cross_attn = CrossAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, audio_context=None):
        x = x + self.attn(self.ln1(x), mask)
        if self.use_cross_attn and audio_context is not None:
            x = x + self.cross_attn(self.ln_ca(x), audio_context)
        x = x + self.mlp(self.ln2(x))
        return x


class PetGPT(nn.Module):
    """Prefix-LM GPT: MANO + SMPL + audio → pet_rel + pet motion codes.

    Sequence: [left(L) | right(R) | smpl(S) | pet_rel(Pr) | pet(P)]
    - Condition (left + right + smpl): bidirectional among themselves
    - pet_rel: causal + full attend to condition
    - Pet: causal + full attend to condition + full attend to pet_rel
    - Audio: cross-attention (continuous features, not in sequence)
    """

    def __init__(self, config):
        super().__init__()
        vocab_mano = getattr(config, 'vocab_mano', 1024)
        vocab_pet = getattr(config, 'vocab_pet', 1024)
        vocab_smpl = getattr(config, 'vocab_smpl', 1024)
        vocab_pet_rel = getattr(config, 'vocab_pet_rel', 1024)
        n_embd = getattr(config, 'n_embd', 512)
        n_head = getattr(config, 'n_head', 8)
        n_layer = getattr(config, 'n_layer', 6)
        dropout = getattr(config, 'dropout', 0.1)

        # MANO hand conditioning
        self.use_mano = getattr(config, 'use_mano', True)
        self.cond_len = getattr(config, 'cond_len', 150) if self.use_mano else 0
        self.left_len = self.cond_len // 2 if self.use_mano else 0
        self.right_len = (self.cond_len - self.left_len) if self.use_mano else 0

        # SMPL body conditioning
        self.use_smpl = getattr(config, 'use_smpl', True)
        self.smpl_len = getattr(config, 'smpl_len', 75) if self.use_smpl else 0

        # Pet rel output
        self.use_pet_rel = getattr(config, 'use_pet_rel', True)
        self.pet_rel_len = getattr(config, 'pet_rel_len', 75) if self.use_pet_rel else 0
        self.vocab_pet_rel = vocab_pet_rel

        # Pet output
        self.pet_len = getattr(config, 'pet_len', 75)
        self.vocab_pet = vocab_pet

        # Segment IDs
        seg_id = 0
        if self.use_mano:
            self.seg_id_left = seg_id; seg_id += 1
            self.seg_id_right = seg_id; seg_id += 1
        if self.use_smpl:
            self.seg_id_smpl = seg_id; seg_id += 1
        if self.use_pet_rel:
            self.seg_id_pet_rel = seg_id; seg_id += 1
        self.seg_id_pet = seg_id; seg_id += 1
        n_segments = seg_id

        # Token embeddings
        if self.use_mano:
            self.tok_emb_left = nn.Embedding(vocab_mano, n_embd)
            self.tok_emb_right = nn.Embedding(vocab_mano, n_embd)
            self.pos_emb_left = nn.Embedding(self.left_len, n_embd)
            self.pos_emb_right = nn.Embedding(self.right_len, n_embd)

        if self.use_smpl:
            self.tok_emb_smpl = nn.Embedding(vocab_smpl, n_embd)
            self.pos_emb_smpl = nn.Embedding(self.smpl_len, n_embd)

        if self.use_pet_rel:
            self.tok_emb_pet_rel = nn.Embedding(vocab_pet_rel, n_embd)
            self.pos_emb_pet_rel = nn.Embedding(self.pet_rel_len, n_embd)

        self.tok_emb_pet = nn.Embedding(vocab_pet, n_embd)
        self.pos_emb_pet = nn.Embedding(self.pet_len, n_embd)

        # Segment embeddings
        self.seg_emb = nn.Embedding(n_segments, n_embd)

        # Audio cross-attention
        self.use_audio = getattr(config, 'use_audio', False)
        if self.use_audio:
            audio_dim = getattr(config, 'audio_dim', 1024)
            audio_max_len = getattr(config, 'audio_max_len', 750)
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, n_embd),
                nn.GELU(),
                nn.Linear(n_embd, n_embd),
            )
            self.audio_pos_emb = nn.Embedding(audio_max_len, n_embd)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout,
                             use_cross_attn=self.use_audio)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

        # Output heads
        if self.use_pet_rel:
            self.head_pet_rel = nn.Linear(n_embd, vocab_pet_rel, bias=False)
        self.head_pet = nn.Linear(n_embd, vocab_pet, bias=False)

        # Build prefix-causal mask
        self._build_mask()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _build_mask(self):
        """Prefix-causal mask:
        - Condition (left + right + smpl): bidirectional among themselves
        - pet_rel: causal among themselves + full attend to condition
        - Pet: causal among themselves + full attend to condition + full attend to pet_rel
        """
        cond = (self.cond_len if self.use_mano else 0) + \
               (self.smpl_len if self.use_smpl else 0)
        pr = self.pet_rel_len
        p = self.pet_len
        total = cond + pr + p
        mask = torch.zeros(total, total)

        # Condition <-> condition: full bidirectional
        mask[:cond, :cond] = 1

        # pet_rel -> condition: full attention
        mask[cond:cond + pr, :cond] = 1
        # pet_rel -> pet_rel: causal
        for i in range(pr):
            mask[cond + i, cond:cond + i + 1] = 1

        # Pet -> condition: full attention
        mask[cond + pr:, :cond] = 1
        # Pet -> pet_rel: full attention (attend to all pet_rel tokens)
        mask[cond + pr:, cond:cond + pr] = 1
        # Pet -> pet: causal
        for i in range(p):
            mask[cond + pr + i, cond + pr:cond + pr + i + 1] = 1

        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))  # (1,1,T,T)

    def _embed_sequence(self, left_codes=None, right_codes=None,
                        smpl_codes=None, pet_rel_codes=None, pet_codes=None):
        """Build embedded sequence [left? | right? | smpl? | pet_rel? | pet?]."""
        embs = []

        # Determine device
        device = None
        for t in [left_codes, right_codes, smpl_codes, pet_rel_codes, pet_codes]:
            if t is not None:
                device = t.device
                break

        # MANO left/right
        if self.use_mano and left_codes is not None:
            L = left_codes.size(1)
            pos_l = torch.arange(L, device=device)
            seg_l = torch.full((L,), self.seg_id_left, dtype=torch.long, device=device)
            embs.append(self.tok_emb_left(left_codes)
                        + self.seg_emb(seg_l) + self.pos_emb_left(pos_l))

        if self.use_mano and right_codes is not None:
            R = right_codes.size(1)
            pos_r = torch.arange(R, device=device)
            seg_r = torch.full((R,), self.seg_id_right, dtype=torch.long, device=device)
            embs.append(self.tok_emb_right(right_codes)
                        + self.seg_emb(seg_r) + self.pos_emb_right(pos_r))

        # SMPL
        if self.use_smpl and smpl_codes is not None and smpl_codes.size(1) > 0:
            S = smpl_codes.size(1)
            pos_s = torch.arange(S, device=device)
            seg_s = torch.full((S,), self.seg_id_smpl, dtype=torch.long, device=device)
            embs.append(self.tok_emb_smpl(smpl_codes)
                        + self.seg_emb(seg_s) + self.pos_emb_smpl(pos_s))

        # Pet rel
        if self.use_pet_rel and pet_rel_codes is not None and pet_rel_codes.size(1) > 0:
            Pr = pet_rel_codes.size(1)
            pos_pr = torch.arange(Pr, device=device)
            seg_pr = torch.full((Pr,), self.seg_id_pet_rel, dtype=torch.long, device=device)
            embs.append(self.tok_emb_pet_rel(pet_rel_codes)
                        + self.seg_emb(seg_pr) + self.pos_emb_pet_rel(pos_pr))

        # Pet
        if pet_codes is not None and pet_codes.size(1) > 0:
            P = pet_codes.size(1)
            pos_p = torch.arange(P, device=device)
            seg_p = torch.full((P,), self.seg_id_pet, dtype=torch.long, device=device)
            embs.append(self.tok_emb_pet(pet_codes)
                        + self.seg_emb(seg_p) + self.pos_emb_pet(pos_p))

        return torch.cat(embs, dim=1)

    def _embed_audio(self, audio_features):
        """Project MERT audio features and add positional encoding."""
        x = self.audio_proj(audio_features)
        T_audio = x.size(1)
        pos = torch.arange(T_audio, device=x.device)
        x = x + self.audio_pos_emb(pos)
        return x

    def forward(self, left_codes, right_codes, pet_codes,
                audio_features=None, smpl_codes=None, pet_rel_codes=None):
        """
        Args:
            left_codes:      (B, L) LongTensor — left hand MANO VQ codes
            right_codes:     (B, R) LongTensor — right hand MANO VQ codes
            pet_codes:       (B, P) LongTensor — pet motion VQ codes (GT)
            audio_features:  (B, T_audio, audio_dim) optional MERT audio features
            smpl_codes:      (B, S) LongTensor — SMPL body VQ codes
            pet_rel_codes:   (B, Pr) LongTensor — pet_rel VQ codes (GT)

        Returns:
            logits_pet: (B, P, vocab_pet)
            loss:       scalar cross-entropy (pet_rel + pet combined)
        """
        x = self._embed_sequence(left_codes, right_codes,
                                 smpl_codes, pet_rel_codes, pet_codes)
        x = self.drop(x)

        # Audio cross-attention context
        audio_context = None
        if self.use_audio and audio_features is not None:
            audio_context = self._embed_audio(audio_features)

        T = x.size(1)
        mask = self.mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask, audio_context=audio_context)
        x = self.ln_f(x)

        # Compute condition length
        cond_len = 0
        if self.use_mano and left_codes is not None:
            cond_len += left_codes.size(1)
        if self.use_mano and right_codes is not None:
            cond_len += right_codes.size(1)
        if self.use_smpl and smpl_codes is not None:
            cond_len += smpl_codes.size(1)

        losses = []

        # pet_rel predictions
        logits_pet_rel = None
        if self.use_pet_rel and pet_rel_codes is not None:
            Pr = pet_rel_codes.size(1)
            logits_pet_rel = self.head_pet_rel(
                x[:, cond_len - 1: cond_len - 1 + Pr, :])
            loss_pet_rel = F.cross_entropy(
                logits_pet_rel.reshape(-1, self.vocab_pet_rel),
                pet_rel_codes.reshape(-1),
            )
            losses.append(loss_pet_rel)
            pr_offset = Pr
        else:
            pr_offset = 0

        # Pet predictions
        P = pet_codes.size(1)
        pet_start = cond_len + pr_offset - 1
        logits_pet = self.head_pet(
            x[:, pet_start: pet_start + P, :])
        loss_pet = F.cross_entropy(
            logits_pet.reshape(-1, self.vocab_pet),
            pet_codes.reshape(-1),
        )
        losses.append(loss_pet)

        loss = sum(losses) / len(losses)

        return logits_pet, loss

    @torch.no_grad()
    def generate(self, left_codes=None, right_codes=None,
                 temperature=1.0, top_k=None, audio_features=None,
                 smpl_codes=None):
        """Autoregressive generation of pet_rel + pet codes given condition.

        Returns:
            pet_rel_codes: (B, pet_rel_len) LongTensor  (or None if use_pet_rel=False)
            pet_codes:     (B, pet_len) LongTensor
        """
        self.eval()
        device = None
        for t in [left_codes, right_codes, smpl_codes]:
            if t is not None:
                device = t.device
                break

        # Project audio once
        audio_context = None
        if self.use_audio and audio_features is not None:
            audio_context = self._embed_audio(audio_features)

        # Phase 1: generate pet_rel codes
        gen_pet_rel = []
        if self.use_pet_rel:
            for i in range(self.pet_rel_len):
                pr_so_far = (torch.stack(gen_pet_rel, dim=1)
                             if gen_pet_rel else None)

                x = self._embed_sequence(left_codes, right_codes,
                                         smpl_codes, pr_so_far, None)
                x = self.drop(x)

                T = x.size(1)
                mask = self.mask[:, :, :T, :T]
                for block in self.blocks:
                    x = block(x, mask, audio_context=audio_context)
                x = self.ln_f(x)

                logits = self.head_pet_rel(x[:, -1, :]) / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
                gen_pet_rel.append(next_token)

        pet_rel_codes = (torch.stack(gen_pet_rel, dim=1)
                         if gen_pet_rel else None)

        # Phase 2: generate pet codes (conditioned on pet_rel)
        gen_pet = []
        for i in range(self.pet_len):
            pet_so_far = (torch.stack(gen_pet, dim=1)
                          if gen_pet else None)

            x = self._embed_sequence(left_codes, right_codes,
                                     smpl_codes, pet_rel_codes, pet_so_far)
            x = self.drop(x)

            T = x.size(1)
            mask = self.mask[:, :, :T, :T]
            for block in self.blocks:
                x = block(x, mask, audio_context=audio_context)
            x = self.ln_f(x)

            logits = self.head_pet(x[:, -1, :]) / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            gen_pet.append(next_token)

        pet_codes = torch.stack(gen_pet, dim=1)  # (B, P)
        return pet_rel_codes, pet_codes
