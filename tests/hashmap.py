import time
from tqdm import tqdm
import torch
import flex_gemm
from flex_gemm.ops.spconv import SubMConv3dFunction
from flex_gemm import kernels


flex_gemm.ops.spconv.set_algorithm(flex_gemm.ops.spconv.Algorithm.EXPLICIT_GEMM)


@torch.no_grad()
def sphere_coords(res, ch, device='cuda', dtype=torch.float):
    coords = torch.stack(torch.meshgrid(
        torch.arange(res, device=device),
        torch.arange(res, device=device),
        torch.arange(res, device=device),
        indexing='ij'
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, device=device, dtype=torch.int32), coords], dim=-1)
    feats = torch.randn(coords.shape[0], ch, device=device, dtype=dtype)
    return feats, coords, torch.Size([1, ch, res, res, res])


@torch.no_grad()
def test_hashmap():
    RES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    records = {}
    
    test_cases = []
    for res in RES:
        test_cases.append(res)
    
    for res in tqdm(test_cases, leave=False):
        time_insert = 0
        memory_insert = 0
        oom_insert = False
        cnt_insert = 0
        time_lookup = 0
        memory_lookup = 0
        oom_lookup = False
        cnt_lookup = 0
        feats, coords, shape = sphere_coords(res, 0)
        for i in range(110):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_insert = time.time()
            try:
                hashmap = torch.full((4 * coords.shape[0],), 0xffffffff, dtype=torch.uint32, device=coords.device)
                values = torch.arange(coords.shape[0], device=coords.device).to(torch.uint32)
                kernels.cuda.hashmap_insert_3d_cuda(hashmap, coords, values, res, res, res)
            
                torch.cuda.synchronize()
                end_insert = time.time()
                if i > 10:
                    time_insert += end_insert - start_insert
                    memory_insert += torch.cuda.max_memory_allocated() / 1024**3
                    cnt_insert += 1
            except Exception as e:
                if 'out of memory' in str(e):
                    oom_insert = True
                else:
                    raise e
                
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_lookup = time.time()
            try:
                values_ = kernels.cuda.hashmap_lookup_3d_cuda(hashmap, coords, res, res, res)
                torch.cuda.synchronize()
                end_lookup = time.time()
                if i > 10:
                    time_lookup += end_lookup - start_lookup
                    memory_lookup += torch.cuda.max_memory_allocated() / 1024**3
                    cnt_lookup += 1
            except Exception as e:
                if 'out of memory' in str(e):
                    oom_lookup = True
                else:
                    raise e
                
            assert torch.all(values == values_), f"Hashmap lookup failed for res={res}"
            
        time_insert = f"{time_insert / cnt_insert:.5f}s" if cnt_insert > 0 else "N/A"
        memory_insert = "OOM" if oom_insert else f"{memory_insert / cnt_insert:.5f}G" if cnt_insert > 0 else "N/A"
        time_lookup = f"{time_lookup / cnt_lookup:.5f}s" if cnt_lookup > 0 else "N/A"
        memory_lookup = "OOM" if oom_lookup else f"{memory_lookup / cnt_lookup:.5f}G" if cnt_lookup > 0 else "N/A"
        records[f"res={res}"] = {
            'insert': f"{time_insert} | {memory_insert}",
            'lookup': f"{time_lookup} | {memory_lookup}",
        }
        res_str = []
        res_str.append("=" * 120)
        res_str.append(f"Hashmap test")
        res_str.append("=" * 120)
        res_str.append(f"{'settings':<64}{'insert':<24}{'lookup':<24}")
        for key, value in records.items():
            res_str.append(f"{key:<64}{value['insert']:<24}{value['lookup']:<24}")
    
    print('\n'.join(res_str))
    return res_str


all_tests = {
    'test_hashmap': test_hashmap,
}
                    

if __name__ == '__main__':
    for test_name, test_func in all_tests.items():
        records = test_func()
