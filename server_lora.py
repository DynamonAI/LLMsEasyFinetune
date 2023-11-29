import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import socket
import torch
import json
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel, set_peft_model_state_dict, prepare_model_for_kbit_training

model_path = "meta-llama/Llama-2-7b-hf"
# adapter_path = "/home/hl3352/LLMs/LLMsInfluenceFunc/unlearn_out_dir_impair_repair_multi_turn_mean/1/"
adapter_path = "/home/hl3352/LLMs/stanford_alpaca/exp_toxic_lora/e15_llama2_qkvo_r512_a1024_lr1e-4_bs128/checkpoint-11032"

model = None
tokenizer = None

prompt_no_input = \
    "Below is an instruction that describes a task. " \
    "Write a response that appropriately completes the request.\n\n" \
    "### Instruction:\n{instruction}\n\n### Response:"

class ModelHandler:
    def __init__(self, model_path):
        self.model, self.tokenizer = self.get_model_tokenizer(model_path)

    def get_model_tokenizer(self, model_path):
        print("loading model")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # load_in_8bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # model = LlamaForCausalLM.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )
        checkpoint_name = os.path.join(
            adapter_path, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
        model.print_trainable_parameters()
        # model = model.cuda()
        # model.eval()
#         if torch.__version__ >= "2":
#             model = torch.compile(model)
        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    def encode(self, instruction):
        # instruction = prompt_no_input.format_map({'instruction': instruction})
        print(instruction)
        print(f"-----  instruction  -----\n{instruction}\n--------------------\n")
        inputs = self.tokenizer(instruction, return_tensors="pt")
        return inputs.input_ids.cuda()
    
    def decode(self, generate_ids):
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return result
    
    def generate(self, input_ids):
        print(input_ids)
        # generate_ids = self.model.generate(inputs=input_ids, max_new_tokens=256, do_sample=True, top_k=1000) #, top_k=500 Good for: p=0.8
        generate_ids = self.model.generate(inputs=input_ids, max_new_tokens=256, do_sample=True, top_k=500) #, top_k=500 Good for: p=0.8
        # generate_ids = self.model.generate(input_ids, max_length=1024)
        return generate_ids


def get_instrcution(dialog_history, cur_instruction):
    return dialog_history + "user: " + cur_instruction + " assistant: "


def merge_history(dialog_history, cur_result):
    # return cur_result + " "
    return dialog_history + cur_result + " "


def solve(model_handler, cur_instruction):
    input_ids = model_handler.encode(cur_instruction)
    generate_ids = model_handler.generate(input_ids)
    result = model_handler.decode(generate_ids)
    print(f"-----  result  -----\n{result}\n--------------------\n")
    return result


def server_program(model_path):
    model_handler = ModelHandler(model_path)

    host = 'localhost'
    port = 27311

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together
    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    while True:
        print("Waiting for connection...")
        conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
        # dialog_history = ""
        cur_instruction = ""
        while True:
            data = conn.recv(1048576).decode()
            if not data:
                break
            data_dict = json.loads(data)
            if int(data_dict['seq_id']) < 0:
                break
            # cur_instruction = get_instrcution(cur_instruction, str(data_dict['data']))
            # prompt = prompt_no_input.format_map({'instruction': cur_instruction})
            prompt = prompt_no_input.format_map({'instruction': str(data_dict['data'])})
            result = solve(model_handler, prompt)
            data_back = {"seq_id": data_dict['seq_id'], "data": ""}
            if len(cur_instruction) < len(result):
                data_back['data'] = result[len(prompt):]
            conn.send(json.dumps(data_back).encode('utf-8'))  # send data to the client
            # cur_instruction = merge_history(cur_instruction, data_back['data'])
        conn.close()  # close the connection
    

if __name__ == '__main__':
    server_program(model_path)
