from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    BertConfig,
    Pipeline
)
from Ibis.Inference.Utilities.onnx import (
    get_onnx_base_model,
    get_onnx_head
)
from Ibis.Inference.Utilities.class_dicts import (
    get_ec_dict
)
import os

class ProteinEmbedderPipeline(Pipeline):
    
    def __init__(self,
                 model_fp: str = os.environ.get('PROTEIN_EMBEDDER_MODEL_FP'),
                 ec1_head_fp: str = os.environ.get('EC1_HEAD_FP'),
                 config: BertConfig = ModelConfigs.get_protbert_config(),
                 protein_tokenizer: PreTrainedTokenizerFast = Tokenizers.get_protbert_tokenizer(),
                 gpu_id: Optional[int] = None):
        model = get_onnx_head(model_fp=model_fp, config=config, gpu_id=gpu_id)
        super().__init__(model=model, tokenizer=protein_tokenizer)
        self.prediction_modules = {
            'ec1': get_onnx_base_model(model_fp=ec1_head_fp, gpu_id=gpu_id)
        }
        self.classification_lookups = get_ec_dict(level=1)
    
    def run(self, sequences: List[str]):
        return [self(s, **kwargs) for s in tqdm(sequences)]
    
    def _sanitize_parameters(self):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, sequence: str, **preprocess_kwargs) -> ModelInput:
        windows = Preprocess.slice_proteins(sequence)
        lengths = [len(x) for x in windows]
        windows = [" ".join(x) for x in windows]
        tokenized_inputs = self.tokenizer(windows, padding=True, return_tensors='np')
        return {
            'sequence': sequence,
            'lengths': lengths,
            'tokenized_inputs': tokenized_inputs
        }
    
    def _forward(self, model_inputs: ModelInput):
        tokenized_inputs = model_inputs['tokenized_inputs']
        # batchify tokenized inputs
        batch_tokenized_inputs = self.batchify_tokenized_inputs(tokenized_inputs)
        # run pipeline in batches
        batch_last_hidden_state = []
        batch_pooler_output = []
        for inp in batch_tokenized_inputs:
            lhs, po  = self.model.run(['last_hidden_state', 'pooler_output'], dict(inp))
            batch_last_hidden_state.append(lhs)
            batch_pooler_output.append(po)
        # concatenate outputs
        last_hidden_state = np.concatenate(batch_last_hidden_state)
        pooler_output = np.concatenate(batch_pooler_output)
        # sequence classification head predictions
        predictions = {}
        for head_name, head in self.prediction_modules.items():
            predictions[head_name] = np.concatenate([head.run(['output'], {"input": inp})[0] for inp in batch_pooler_output])
        # return output
        return {
            'sequence': model_inputs['sequence'],
            'lengths': model_inputs['lengths'],
            'cls_embeddings': pooler_output,
            'predictions': predictions
        }
    
    def postprocess(self, model_outputs: ModelOutput, **postprocess_kwargs) -> PipelineOutput:
        # parameters
        sequence = model_outputs['sequence']
        min_slice_size = postprocess_kwargs['min_slice_size']
        lengths = model_outputs['lengths']
        # calculate protein embedding
        indices = self.get_indices(min_slice_size, sequence, lengths)
        cls_embeddings = model_outputs['cls_embeddings']
        avg_cls_embedding = np.mean(np.take(cls_embeddings, indices, axis=0), axis=0)
        # calculate cls predictions
        sequence_classification = {}
        for head_name, head in self.prediction_modules.items():
            logits = model_outputs['predictions'][head_name]
            indices = self.get_indices(min_slice_size, sequence, lengths)
            logits = self.softmax(np.mean(np.take(logits, indices, axis=0), axis=0))
            label_id = int(logits.argmax())
            score = round(float(logits[label_id]), 2)
            label = self.classification_lookups[r].get(label_id)
            sequence_classification[r] = {'label': label, 'score': score}
        # output
        return {
            'sequence': sequence,
            'embedding': avg_cls_embedding,
            'sequence_classification': sequence_classification
        }