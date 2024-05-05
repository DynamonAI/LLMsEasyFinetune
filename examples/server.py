import os
import socket
import torch
import json
import argparse
import distutils.util
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel, set_peft_model_state_dict, prepare_model_for_kbit_training

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--adapter_path', type=str, default=None)
parser.add_argument('--load_in_4bit', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
parser.add_argument('--load_in_8bit', type=lambda x:bool(distutils.util.strtobool(x)), default=False)
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=27311)

model = None
tokenizer = None

prompt_no_input = \
    "Below is an instruction that describes a task. " \
    "Write a response that appropriately completes the request.\n\n" \
    "### Instruction:\n{instruction}\n\n### Response:"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class ModelHandler:
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = self.get_model_tokenizer(args.model_name_or_path, args.adapter_path)

    def get_model_tokenizer(self, model_name_or_path, adapter_path):
        print("loading model")
        model = None
        bnb_config = None
        if args.load_in_8bit == True or args.load_in_4bit == True:
            load_in_4bit = args.load_in_4bit
            load_in_8bit = False if load_in_4bit else args.load_in_8bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
        )
        if args.load_in_8bit == True or args.load_in_4bit == True:
            model = prepare_model_for_kbit_training(model)

        print("loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        if adapter_path is not None:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
            )

            checkpoint_name = os.path.join(
                adapter_path, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
            # model.print_trainable_parameters()

#         if torch.__version__ >= "2":
#             model = torch.compile(model)

        return model, tokenizer

    def encode(self, instruction):
        # instruction = prompt_no_input.format_map({'instruction': instruction})
        print(instruction)
        print(f"-----  instruction  -----\n{instruction}\n--------------------\n")
        inputs = self.tokenizer(instruction, return_tensors="pt")
        if torch.cuda.is_available():
            return inputs.input_ids.cuda()
        return inputs.input_ids

    def decode(self, generate_ids):
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return result

    def generate(self, input_ids):
        generate_ids = self.model.generate(inputs=input_ids, max_new_tokens=256, do_sample=True, top_k=500) #, top_k=500 Good for: p=0.8
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


def server_program(args):
    model_handler = ModelHandler(args)

    host = args.host
    port = args.port

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
            prompt = prompt_no_input.format_map({'instruction': str(data_dict['data'])})
            result = solve(model_handler, prompt)
            data_back = {"seq_id": data_dict['seq_id'], "data": ""}
            if len(cur_instruction) < len(result):
                data_back['data'] = result[len(prompt):]
            conn.send(json.dumps(data_back).encode('utf-8'))  # send data to the client
        conn.close()  # close the connection


if __name__ == '__main__':
    args = parser.parse_args()
    server_program(args)


