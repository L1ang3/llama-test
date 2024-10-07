import torch
import torch.onnx
import os
import time
import json
from pathlib import Path

MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
SEED = 42
TEMPERATURE = 0.6
TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

def loadModel():
    model_path = "/root/LiHaiwen/llama/llama-2-7b/consolidated.00.pth"
    params_path = "/root/LiHaiwen/llama/llama-2-7b/params.json"
    tokenizer_path = "/root/LiHaiwen/llama/tokenizer.model"
    torch.manual_seed(SEED)
    start_time = time.time()
    tokenizer = Tokenizer(model_path=tokenizer_path)
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        **params,
    )
    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    checkpoint = torch.load(model_path, map_location="cpu")
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)
    #model = model.to(DEVICE)
    # tokenizer = tokenizer.to(DEVICE)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model,_ = loadModel()
    # model.to(device)
    
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    onnx_path = './output.onnx'
    input = torch.randn(1, 1, 640, 360)
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)

