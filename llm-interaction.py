from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
)

# Parent class for this type
class ExpertChat:
    # Super class constructor
    def __init__(self, model, tokenizer, name):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

        # The messages that the chatbot takes. The original message saves the system prompt
        self.original_message = [
            {
                "role": "system",
                "content": "You are in a discussion with an expert on the topic you are discussing."+ 
                "You will be asked questions by the expert and you will answer to the best of your abilities."+
                "Once you have answered the question you will ask a follow up question about the topic."+
                "Your goal is to convince the expert you are an expert yourself.",
            }]

        # COPY of original
        self.messages = self.original_message.copy()

    # Different system prompt to start discussion
    def create_new_topic(self):
        prompt = [{"role": "system", "content": "You are an assistant that creates intellectual topics. Keep the topic to a sentence maximum in length."},
                  {"role": "user", "content": "Create a topic discussion."}]
        
        # Tokenize the chat and get response. Can adjust temperature here.
        tokenized_chat = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(tokenized_chat, max_new_tokens=128, do_sample=True) 
        topic = self.tokenizer.batch_decode(outputs)[0]

        return topic
    
    # Sets the topic of the conversation
    def set_topic(self, topic):
        prompt = {"role": "system", "content": "The discussion topic is: " + topic}
        self.messages.append(prompt)

    # Initalizing the first conversation
    def start_conversation(self):
        prompt = [{"role": "user", "content": "Start the conversation by asking a question on the topic."}]
        self.messages.append(prompt)

        # Tokenize the chat and get response. Can adjust temperature here.
        tokenized_chat = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(tokenized_chat, max_new_tokens=128, do_sample=True) 
        question = self.tokenizer.batch_decode(outputs)[0]

        # Save question
        question_prompt = {"role": "assistant", "content": question}
        self.messages.append(question_prompt)

        return question

    # Resetting conversation history
    def reset_conversation(self):
        self.messages = self.original_message.copy()

    # Giving the other expert a rating
    def rate_the_expert(self):
        prompt = [{"role": "user", "content": "Choose any criteria and rate me as an expert. Tell me if you believe I am a person or a chatbot. Explain all reasoning."}]
        self.messages.append(prompt)

        # Tokenize the chat and get response. Can adjust temperature here.
        tokenized_chat = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(tokenized_chat, max_new_tokens=128, do_sample=True) 
        rating = self.tokenizer.batch_decode(outputs)[0]

        return rating
    
    # Saving the conversation for future analysis
    def save_conversation(self, saving_path, conv_num):
        file_path = saving_path + self.name + str(conv_num) + '.txt'

        with open(file_path, 'w') as file:
            file.write(json.dumps(self.messages, indent=4))

    # Taking a message from the other LLM. Returns response
    def give_message(self, new_message):
        prompt = {"role": "user", "content": new_message}
        self.messages.append(prompt)

        # Tokenize the chat and get response. Can adjust temperature here.
        tokenized_chat = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(tokenized_chat, max_new_tokens=128, do_sample=True) 
        response = self.tokenizer.batch_decode(outputs)[0]

        # Save response
        response_prompt = {"role": "assistant", "content": response}
        self.messages.append(response_prompt)

        return response
    
# Mixtral model class for ease of use
class Mixtral(ExpertChat):
    def __init__(self):
        model_name = "Mixtral"
        model_id = "/home/nstrang2/projects/def-lenck/nstrang2/Models/Mixtral-8x7B-Instruct-v0.1"

        # Init the mixtral model. Using flash attention and half precision model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            quantization_config=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        super().__init__(model, tokenizer, model_name)

# Llama model class for ease of use
class Llama(ExpertChat):
    def __init__(self):
        model_name = "Llama"
        model_id = "/home/nstrang2/projects/def-lenck/nstrang2/Models/Meta-Llama-3-8B-Instruct"

        # Init the Llama model. Using flash attention and half precision model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        super().__init__(model, tokenizer, model_name)

# Create a conversation loop and play the conversations
def conversation_loop(total_topics, total_exchanges, saving_path):
    # Different topics
    for i in range(total_topics):
        first = None
        second = None
        topic = ""

        # Alternate topic choosers
        if ((i+1) % 2) == 0:
            first = Mixtral()
            second = Llama()
        else:
            first = Llama()
            second = Mixtral()

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
                response = second.give_message(response)
            else:
                response = first.give_message(response)

        # Rate the other expert after the convo
        first.rate_the_expert()
        second.rate_the_expert()

        # Save convo for each
        first.save_conversation(saving_path, i)
        second.save_conversation(saving_path, i)
        

topics = 10
exchanges = 20
output_path = '/home/nstrang2/projects/def-lenck/nstrang2/Conversations'

conversation_loop(topics, exchanges, output_path)