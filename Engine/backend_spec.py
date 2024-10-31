import torch
from MagicDec.Engine.model_spec import Transformer
from MagicDec.Engine.utils import load_model
import flashinfer

############# utility lib functions ################



####################################################


class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_len: int = 1) -> None:
        self.dtype = dtype
        self.device = device
        self.dec_len = dec_len
        self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        self.prefill = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last=None: model.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last)
        self.cachelens = None
        self.is_draft = False

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp=use_tp, rank_group=rank_group, group=group)        

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.page_size = page_size = 128
        self.max_num_pages = max_num_pages = max_batch_size * (max_seq_length + page_size - 1) // page_size
        self.max_num_pages_per_request = max_num_pages // max_batch_size
        self.num_pages_per_request = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        # Init Attention Backend (Flashinfer)
        self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)
        self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode_buffer, "NHD", use_cuda_graph=True,
                                                                              qo_indptr_buf=self.qo_indptr, 
                                                                              paged_kv_indptr_buf=self.paged_kv_indptr, 
                                                                              paged_kv_indices_buf=self.paged_kv_indices, 
                                                                              paged_kv_last_page_len_buf=self.paged_kv_last_page_len)
        
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_buffer, "NHD")

        torch.library.define(
            "mylib::decode",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )

        @torch.library.impl("mylib::decode", "cuda")
        def decode(q, kv_cache):
            return self.decode_wrapper.run(
                q, kv_cache
            )

        @torch.library.register_fake("mylib::decode")
        def decode_abstract(q, kv_cache):
            return torch.empty_like(q)       

        torch.library.define(
            "mylib::prefill",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )

        @torch.library.impl("mylib::prefill", "cuda")
        def prefill(q, kv_cache):
            return self.prefill_wrapper.run(
                q, kv_cache
            )

        @torch.library.register_fake("mylib::prefill")
        def prefill_abstract(q, kv_cache):
            return torch.empty_like(q)

        with torch.device(self.device):
            self.model.setup_caches(num_pages=max_num_pages, page_size=page_size)

    def compile(self):
        import torch._dynamo.config
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._functorch.config.enable_autograd_cache = True
        self.model_forward = torch.compile(self.model_forward, mode="max-autotune", fullgraph=True)

    # used for both speculation, verification and autoregressive decoding
    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            self.pre_infer(dec_len=dec_len)

            logits = self.model_forward(
                model=self.model, 
                x=input_ids,
                input_pos=self.cachelens, 
                kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len)
            
            self.cachelens += dec_len
            if benchmark:
                # If benchmarking the latency, don't update the cachelens and page table
                self.cachelens -= dec_len
                self.paged_kv_last_page_len -= dec_len
            return logits

    def pre_infer(self, dec_len):
            self.paged_kv_last_page_len += dec_len
            self.decode_wrapper.plan(
                qo_indptr=self.qo_indptr*dec_len,
                paged_kv_indptr=self.paged_kv_indptr,
                paged_kv_indices=self.paged_kv_indices,
                paged_kv_last_page_len=self.paged_kv_last_page_len,
                num_qo_heads=self.model.config.n_head, 
                num_kv_heads=self.model.config.n_local_heads, 
                head_dim=self.model.config.head_dim, 
                page_size=self.page_size, 
                q_data_type=self.dtype, 
                causal=True,
            )

    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, benchmark = False):
        self.clear_kv()
        logits = None
        seq_len = input_ids.shape[1]
        chunk_size = 128
        num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceil division
        is_last = False
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            dec_len = end_idx-start_idx
            is_last = (i == num_chunks-1)
            self.pre_encode(dec_len=dec_len)
            logits = self.prefill(
                model=self.model,
                    x=chunk_input_ids,
                    input_pos=self.cachelens,
                    kv_append_indptr = self.qo_indptr*dec_len, 
                    kv_page_indices = self.paged_kv_indices, 
                    kv_page_indptr= self.paged_kv_indptr, 
                    kv_page_lastlen = self.paged_kv_last_page_len,
                    is_last=(self.is_draft and is_last)
            )
            self.cachelens += dec_len

        return logits

    def pre_encode(self, dec_len):
        self.num_pages_per_request+=1
        qo_indptr = self.qo_indptr*dec_len
        self.paged_kv_indices = torch.cat([torch.arange(i * self.max_num_pages_per_request, i * self.max_num_pages_per_request + self.num_pages_per_request[i], dtype=torch.int32, device=self.device) for i in range(self.batch_size)])
        self.paged_kv_indptr[1:] = torch.cumsum(self.num_pages_per_request, dim=0, dtype=torch.int32)
        self.paged_kv_last_page_len = torch.full((self.batch_size,), dec_len, dtype=torch.int32, device=self.device)
        self.prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            num_qo_heads=self.model.config.n_head, 
            num_kv_heads=self.model.config.n_local_heads, 
            head_dim=self.model.config.head_dim, 
            page_size=self.page_size, 
            q_data_type=self.dtype, 
            causal=True
            )

    @torch.inference_mode()
    def clear_kv(self):
        for b in self.model.layers:
            b.attention.kv_cache.kv_cache.zero_()
        self.cachelens.zero_()
        self.qo_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((self.batch_size), dtype=torch.int32, device=self.device)
        self.num_pages_per_request = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)


class LMBackendDraft(LMBackend):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_draft = True

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, window_size: int = 32):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self.page_size = page_size = 128
        self.max_num_pages = max_num_pages = max_batch_size * (max_seq_length + page_size - 1) // page_size
        self.max_num_pages_per_request = max_num_pages // max_batch_size
        self.num_pages_per_request = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)

        # Init Attention Backend (Flashinfer)
        self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)
        self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode_buffer, "NHD", use_cuda_graph=True,
                                                                              qo_indptr_buf=self.qo_indptr, 
                                                                              paged_kv_indptr_buf=self.paged_kv_indptr, 
                                                                              paged_kv_indices_buf=self.paged_kv_indices, 
                                                                              paged_kv_last_page_len_buf=self.paged_kv_last_page_len)
        
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_buffer, "NHD")

        torch.library.define(
            "mylib::speculate",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )

        @torch.library.impl("mylib::speculate", "cuda")
        def speculate(q, kv_cache):
            return self.decode_wrapper.run(
                q, kv_cache
            )

        @torch.library.register_fake("mylib::speculate")
        def speculate_abstract(q, kv_cache):
            return torch.empty_like(q)       

        torch.library.define(
            "mylib::draft_prefill",
            "(Tensor q, Tensor kv_cache) -> Tensor",
        )

        @torch.library.impl("mylib::draft_prefill", "cuda")
        def draft_prefill(q, kv_cache):
            return self.prefill_wrapper.run(
                q, kv_cache
            )

        @torch.library.register_fake("mylib::draft_prefill")
        def draft_prefill_abstract(q, kv_cache):
            return torch.empty_like(q)

        with torch.device(self.device):
            self.model.setup_caches(num_pages=max_num_pages, page_size=page_size, is_draft=True, window_size=window_size)