import torch
from FlashSpec.Engine.model_selfspec import Transformer
from FlashSpec.Engine.utils import load_model_selfspec

class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_list: list = [1], draft_dec_list: list = [1]) -> None:
        self.dtype = dtype
        self.device = device
        self.model_forward = {}
        self.draft_forward = {}
        for dec_len in dec_list:
            if dec_len == 0: continue
            self.model_forward[dec_len] = lambda model, x, input_pos, cache_seqlens: model(x, input_pos, cache_seqlens)
        for dec_len in draft_dec_list:
            if dec_len == 0: continue
            self.draft_forward[dec_len] = lambda model, x, input_pos, cache_seqlens: model.draft_forward(x, input_pos, cache_seqlens)
        self.prefill = lambda model, x, input_pos, cache_seqlens: model.prefill(x, input_pos, cache_seqlens)
        self.draft_prefill = lambda model, x, input_pos, cache_seqlens, is_last: model.draft_prefill(x, input_pos, cache_seqlens, is_last)
        self.cachelens = None
        self.draft_cachelens = None
        self.streaming_budget = None

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model_selfspec(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp= use_tp, rank_group=rank_group, group = group)

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, streamingllm_budget: int = 256, buffer: int = 0):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.draft_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, streaming_budget=streamingllm_budget, buffer=buffer)
        self.streaming_budget = streamingllm_budget

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        for key in self.model_forward.keys():
            self.model_forward[key] = torch.compile(self.model_forward[key], mode="reduce-overhead", fullgraph=True)
        for key in self.draft_forward.keys():
            self.draft_forward[key] = torch.compile(self.draft_forward[key], mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill = torch.compile(self.prefill, mode="reduce-overhead", fullgraph=True)      
             
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            position_ids = self.cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            logits = self.model_forward[dec_len](
                model=self.model, 
                x=input_ids.clone(),
                input_pos=position_ids.clone(), 
                cache_seqlens= self.cachelens.clone()) if dec_len in self.model_forward.keys() else self.model.forward(input_ids.clone(), position_ids.clone(), self.cachelens.clone())
            if not benchmark:
                self.cachelens += dec_len

            # print(self.draft_cachelens, self.cachelens)

            return logits
    
    @torch.inference_mode()
    def draft_inference(self, input_ids: torch.LongTensor, benchmark = False, cachelen_update = None):
            dec_len = input_ids.shape[1]
            position_ids = self.draft_cachelens.view(-1,1) + torch.arange(dec_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
            logits = self.draft_forward[dec_len](
                model=self.model, 
                x=input_ids.clone(),
                input_pos=position_ids.clone(), 
                cache_seqlens= self.draft_cachelens.clone()) if dec_len in self.draft_forward.keys() else self.model.draft_forward(input_ids.clone(), position_ids.clone(), self.draft_cachelens.clone())
            if not benchmark:
                if cachelen_update == None:
                    self.draft_cachelens += dec_len
                else:
                    self.draft_cachelens += cachelen_update
            return logits
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor):
        self.cachelens.zero_()
        self.draft_cachelens.zero_()
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size,1)
        division = seq_len > 1000
        if division:
            chunk_size = 32
            num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, seq_len)
                
                chunk_input_ids = input_ids[:, start_idx:end_idx]
                chunk_position_ids = position_ids[:, start_idx:end_idx]
                chunk_cache_seqlens = self.cachelens + start_idx

                logits = self.prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=chunk_position_ids,
                    cache_seqlens=chunk_cache_seqlens
                )

                if end_idx > self.streaming_budget:
                    chunk_position_ids = torch.arange(self.streaming_budget - chunk_input_ids.shape[1], self.streaming_budget, device = self.device).unsqueeze(0).repeat(input_ids.shape[0],1).long()
                self.draft_prefill(
                    model=self.model,
                    x=chunk_input_ids,
                    input_pos=chunk_position_ids,
                    cache_seqlens=chunk_cache_seqlens,
                    is_last = i == num_chunks-1
                )
        else:
            raise NotImplementedError("Not implemented for seq_len < 1000")
            logits = self.prefill(
                model=self.model,
                x=input_ids,
                input_pos=position_ids,
                cache_seqlens=self.cachelens
            )

        self.cachelens += seq_len
        # self.draft_cachelens += seq_len
        self.draft_cachelens += self.streaming_budget

        return logits
          
    
    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.k_cache.zero_()
            b.attention.kv_cache.v_cache.zero_()

    

