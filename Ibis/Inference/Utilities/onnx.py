import onnxruntime as ort

def get_providers(gpu_id: Optional[int]) -> List[str]:
    if isinstance(gpu_id, int):
        return [('CUDAExecutionProvider', {'device_id': gpu_id})]
    else:
        return ['CPUExecutionProvider']

def get_onnx_base_model(model_fp: str, config: BertConfig, gpu_id: int):
    providers = get_providers(gpu_id=gpu_id)
    model = ort.InferenceSession(model_fp, providers=providers)
    model.config = config
    return model

def get_onnx_head(model_fp: str, gpu_id: int):
    providers = get_providers(gpu_id=gpu_id)
    head = ort.InferenceSession(model_fp, providers=providers)
    return head

