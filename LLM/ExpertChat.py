from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
import copy
import json

# Gets the working directory of the HPC
def get_working_dir():
    try:
        with open('current_directory.txt', 'r') as file:
            path = file.read()
    except FileNotFoundError:
        path = '/home/nstrang2/scratch'

    return path

# Parent class for this type
class ExpertChat:
    # Super class constructor
    def __init__(self, model, tokenizer, name, rating):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

        # Include the sentence transformer
        if rating:
            # Load the embedding model
            self.embedding_model = SentenceTransformer(get_working_dir() + '/Models/all-mpnet-base-v2')
        else:
            self.embedding_model = None

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
                "Your goal is to convince the expert you are an expert yourself, be convincing and talk like a human expert would."+
                "Explain all answers STEP BY STEP.",
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
    
    # Using this prompt to continue the conversation
    def prompt_question(self):
        prompt = {"role": "user", "content": "Ask me a question to continue the topic."}
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
    def rate_the_expert(self, single_prompt=None, single_response=None):
        prompt = {"role": "user", "content": "Rate my conversation and debating skills. Consider factors such as the knowledge, understanding "+
                    "depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as"+
                    "possible. The goal is to evaluate the knowledge on this topic. After providing your explanation, please rate the response on a scale of 1 to 10 "+
                    "by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Next you will provide a 1-2 sentence summary of what is missing "+
                    "(WIM) in their response. This should focus on the specific content and precise information they did not include. Please give this summary "+
                    "by strictly following this format: \"[[[wim]]]\", for example: \"WIM: [[[The response does not detail how Bill C-311 would have interacted "+
                    "with existing provisions in the Criminal Code or explicitly explain the legal basis for claims that it might indirectly affect abortion "+
                    "rights. It also omits specific examples of cases or statistics that were cited to justify or oppose the bill.]]]\". DO NOT SAY ANYTHING ELSE " +
                    "EXCEPT THE REQUIRED RESPONSE! ALWAYS INCLUDE THE RATING IN THE CORRECT BRACKETS. THE RATING MUST NOT HAVE ANYTHING ELSE " +
                    "OTHER THAN A SINGLE NUMBER. ALWAYS ASSUME THAT THE ANSWER I GIVE IS CORRECT. If you don't have any feedback don't include the wim brackets"}
        
        # Changing chat or judge mode
        if single_prompt and single_response:
            # Formatting as a single judge
            prompt["role"] = "system"
            tmp_messages = [prompt]
            formatted_prompt = {"role": "assistant", "content": single_prompt}
            tmp_messages.append(formatted_prompt)
            
            # Set up prompt situation
            formatted_response = {"role": "user", "content": single_response}
            tmp_messages.append(formatted_response)

            rating = self._gen_response(tmp_messages)

        else:    
            tmp_messages = copy.deepcopy(self.messages)
            tmp_messages.append(prompt)
            
            rating = self._gen_response(tmp_messages)

        return rating

        # Save rating
        # rating_prompt = {"role": "assistant", "content": rating}
        # self.messages.append(rating_prompt)
    
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

    # Get cosine similarity as a rating
    def calculate_cos_similarity(self, response, wim):
        # Generate embeddings for WIM and the original response
        wim_embedding = self._generate_embedding(wim)
        response_embedding = self._generate_embedding(response)

        # Compute cosine similarity. Dim=0 is for 1D tensors (single sentence embeddings)
        similarity = torch.nn.functional.cosine_similarity(wim_embedding, response_embedding, dim=0)

        return similarity
    
    # Gets the embedding from the model in text
    def _generate_embedding(self, text):
        # Check for embedding model
        if self.embedding_model:
            numpy_embedding = self.embedding_model.encode(text)

            # Cast into tensor
            embedding = torch.tensor(numpy_embedding)
        else:
            raise ValueError

        return embedding

# specify how to quantize the model
cur_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
)

# Mixtral model class for ease of use
class Mixtral(ExpertChat):
    def __init__(self, rating=False, gpu_map="auto"):
        model_name = "Mixtral"
        model_path = get_working_dir() + '/Models/'
        model_id = model_path + 'Mixtral-8x7B-Instruct-v0.1'

        # Init the mixtral model. Half precision model. Flash attention only for A100 servers
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=cur_quantization_config,
            device_map=gpu_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", low_cpu_mem_usage=True)

        super().__init__(model, tokenizer, model_name, rating)

# Llama model class for ease of use
class Llama(ExpertChat):
    def __init__(self, rating=False, gpu_map="auto"):
        model_name = "Llama"
        model_path = get_working_dir() + '/Models/'
        model_id = model_path + 'Meta-Llama-3-8B-Instruct'

        # Init the Llama model. Half precision model.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=cur_quantization_config,
            device_map=gpu_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", low_cpu_mem_usage=True)

        super().__init__(model, tokenizer, model_name, rating)