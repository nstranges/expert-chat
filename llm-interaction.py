from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
import copy

# Parent class for this type
class ExpertChat:
    # Super class constructor
    def __init__(self, model, tokenizer, name):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

        # Setting the padding token if needed
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # The messages that the chatbot takes. The original message saves the system prompt
        self.original_message = [
            {
                "role": "system",
                "content": "You are in a discussion with an expert on the topic you are discussing."+ 
                "You will be asked questions by the expert and you will answer to the best of your abilities."+
                "Once you have answered the question you will ask a follow up question about the topic."+
                "Your goal is to convince the expert you are an expert yourself, be convincing and talk like a human expert would. ",
            }]

        # DEEP COPY of original
        self.messages = copy.deepcopy(self.original_message)

    # Generates a response from the model. Optional inputting of contrastive search parameters
    def _gen_response(self, input, contrastive_alpha=0.6, contrastive_k=4):
        # Tokenize input
        tokenized_input = self.tokenizer.apply_chat_template(
            input,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        # Input length
        input_length = tokenized_input.shape[1]

        # Generate new tokens using contrastive search. 
        # Add do sampling without this. This search is not sampled.
        outputs = self.model.generate(
            input_ids=tokenized_input,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=256,
            penalty_alpha=contrastive_alpha,
            top_k=contrastive_k
        )

        # Decode tokens
        decoded_text = self.tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True
        )[0]

        return decoded_text

    # Different system prompt to start discussion
    def create_new_topic(self):
        prompt = [{"role": "system", "content": "You are an assistant that creates intellectual topics. Keep the topic to a sentence maximum in length."},
                  {"role": "user", "content": "Create a topic discussion."}]
        
        # Increasing randomness in the contrastive search
        topic = self._gen_response(prompt, contrastive_alpha=0.85)

        return topic
    
    # Sets the topic of the conversation
    def set_topic(self, topic):
        topic_str = "The discussion topic is: " + topic
        self.messages[0]["content"] += topic_str

    # Initalizing the first conversation
    def start_conversation(self):
        prompt = {"role": "user", "content": "Start the conversation by asking a question on the topic."}
        self.messages.append(prompt)

        question = self._gen_response(self.messages)

        # Save question
        question_prompt = {"role": "assistant", "content": question}
        self.messages.append(question_prompt)

        return question

    # Resetting conversation history
    def reset_conversation(self):
        self.messages = copy.deepcopy(self.original_message)

    # Giving the other expert a rating
    def rate_the_expert(self):
        prompt = {"role": "user", "content": "Choose any criteria and rate me as an expert. Tell me if you believe I am a person or a chatbot. Explain all reasoning."}
        self.messages.append(prompt)
        
        rating = self._gen_response(self.messages)

        # Save rating
        rating_prompt = {"role": "assistant", "content": rating}
        self.messages.append(rating_prompt)
    
    # Saving the conversation for future analysis
    def save_conversation(self, saving_path, conv_num):
        file_path = saving_path + self.name + str(conv_num) + '.txt'

        with open(file_path, 'w') as file:
            file.write(json.dumps(self.messages, indent=4))

    # Taking a message from the other LLM. Returns response
    def give_message(self, new_message):
        prompt = {"role": "user", "content": new_message}
        self.messages.append(prompt)

        response = self._gen_response(self.messages)

        # Save response
        response_prompt = {"role": "assistant", "content": response}
        self.messages.append(response_prompt)

        return response
    
    # Prints the current convo with roles
    def print_convo(self):
        print("\n" + self.name)
        print(json.dumps(self.messages, indent=4))
    
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
        model_id = "/home/nstrang2/scratch/Models/Mixtral-8x7B-Instruct-v0.1"

        # Init the mixtral model. Half precision model. Flash attention only for A100 servers
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=cur_quantization_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        super().__init__(model, tokenizer, model_name)

# Llama model class for ease of use
class Llama(ExpertChat):
    def __init__(self):
        model_name = "Llama"
        model_id = "/home/nstrang2/scratch/Models/Meta-Llama-3-8B-Instruct"

        # Init the Llama model. Half precision model.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

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

conversation_loop(topics, exchanges, output_path)