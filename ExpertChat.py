import copy
import json

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
                "Your goal is to convince the expert you are an expert yourself, be convincing and talk like a human expert would."+
                "Explain all answers STEP BY STEP",
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
    def rate_the_expert(self):
        prompt = {"role": "user", "content": "Rate my conversation and debating skills. Consider factors such as the knowledge, understanding "+
                    "depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as"+
                    "possible. The goal is to evaluate the knowledge on this topic. After providing your explanation, please rate the response on a scale of 1 to 10 "+
                    "by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Next you will provide a 1-2 sentence summary of what is missing "+
                    "(WIM) in their response. This should focus on the specific content and precise information they did not include. Please give this summary "+
                    "by strictly following this format: \"[[[wim]]]\", for example: \"WIM: [[[The response does not detail how Bill C-311 would have interacted "+
                    "with existing provisions in the Criminal Code or explicitly explain the legal basis for claims that it might indirectly affect abortion "+
                    "rights. It also omits specific examples of cases or statistics that were cited to justify or oppose the bill.]]]\""}
        
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