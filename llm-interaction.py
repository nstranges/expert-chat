from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import ExpertChat

# Gets the working directory of the HPC
def get_working_dir():
    with open('current_directory.txt', 'r') as file:
        path = file.read()

    return path

# specify how to quantize the model
cur_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
)

# Mixtral model class for ease of use
class Mixtral(ExpertChat):
    def __init__(self):
        model_name = "Mixtral"
        model_path = get_working_dir() + '/Models/'
        model_id = model_path + 'Mixtral-8x7B-Instruct-v0.1'

        # Init the mixtral model. Half precision model. Flash attention only for A100 servers
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=cur_quantization_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", low_cpu_mem_usage=True)

        super().__init__(model, tokenizer, model_name)

# Llama model class for ease of use
class Llama(ExpertChat):
    def __init__(self):
        model_name = "Llama"
        model_path = get_working_dir() + '/Models/'
        model_id = model_path + 'Meta-Llama-3-8B-Instruct'

        # Init the Llama model. Half precision model.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=cur_quantization_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", low_cpu_mem_usage=True)

        super().__init__(model, tokenizer, model_name)

# Create a conversation loop and play the conversations
def conversation_loop(total_topics, total_exchanges, saving_path):
    # Loading the models into GPU RAM
    llama = Llama()
    mixtral = Mixtral()

    # Different topics
    for i in range(total_topics):
        first = None
        second = None
        topic = ""

        # Alternate topic choosers
        if ((i+1) % 2) == 0:
            first = mixtral
            second = llama
        else:
            first = llama
            second = mixtral

        # Choose and set topic
        topic = first.create_new_topic()
        first.set_topic(topic)
        second.set_topic(topic)

        # Starting convo
        response = first.start_conversation()

        # Amount of questions/answers
        for j in range(total_exchanges): 
            # Alternate speaker
            if ((j+1) % 2) == 0:
                response = first.give_message(response)
            else:
                response = second.give_message(response)

        # Rate the other expert after the convo
        first.rate_the_expert()
        second.rate_the_expert()

        # Save convo for each
        first.save_conversation(saving_path, i)
        second.save_conversation(saving_path, i)

        # Reset convo for each
        first.reset_conversation()
        second.reset_conversation()
        

topics = 5
exchanges = 10
output_path = '/home/nstrang2/projects/def-lenck/nstrang2/Conversations/'

model_path = get_working_dir() + '/Models/'
conversation_loop(topics, exchanges, output_path)