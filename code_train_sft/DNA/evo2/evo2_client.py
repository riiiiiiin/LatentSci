# needs to implement:
# __init__(folder, ckpt_name) asks the server to load a specific model
# parameters() placeholder
# eval() placeholder
# encode(sci_list: untokenized)

from shared_tensor import SharedTensorClient

class Evo2Client:
    def __init__(self, folder, ckpt_name):
        self.client = SharedTensorClient()
        if not 'evo2_1b_base' in ckpt_name:
            raise NotImplementedError(
                f"Only evo2_1b_base is supported, found {ckpt_name}"
            )
        self.layer_name = 'blocks.20.mlp.l3'
        self.client.call("load_evo2_model", 'evo2_1b_base', f'{folder}/{ckpt_name}')
    
    def parameters(self):
        return []
    
    def eval(self):
        pass

    def encode(self, dna_list):
        '''
        Supports nested list
        '''
        if not dna_list:
            return []

        flat_dna_list = []
        structure = []
        for item in dna_list:
            flat_dna_list.extend(item)
            structure.append(len(item))
        
        if not flat_dna_list:
            return []

        all_embeddings = self.client.call("get_dna_embedding_batch", flat_dna_list)

        nested_embeddings = []
        cursor = 0
        for count in structure:
            nested_embeddings.append(all_embeddings[cursor:cursor+count])
            cursor += count
        
        return nested_embeddings

def load_evo2(folder, ckpt_filename):
    return Evo2Client(folder, ckpt_filename)