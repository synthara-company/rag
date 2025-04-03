"""
Advanced prompting techniques for the RAG Transformer system.
This module provides templates and examples for better response generation.
"""

import random
from typing import List, Dict, Any

class PromptTemplate:
    """Class for managing prompt templates with few-shot examples"""
    
    def __init__(self):
        """Initialize prompt templates for different domains"""
        self.domain_templates = {
            "Machine Learning": self._get_ml_templates(),
            "Science Fiction": self._get_scifi_templates(),
            "Cosmos": self._get_cosmos_templates(),
            "General": self._get_general_templates()
        }
        
        # Few-shot examples for each domain
        self.few_shot_examples = {
            "Machine Learning": self._get_ml_examples(),
            "Science Fiction": self._get_scifi_examples(),
            "Cosmos": self._get_cosmos_examples(),
            "General": self._get_general_examples()
        }
    
    def _get_ml_templates(self) -> List[str]:
        """Get templates for machine learning domain"""
        return [
            "Based on the provided context about {topic}, provide a clear and concise explanation of {topic} in machine learning.",
            "Using the machine learning concepts in the context, explain {topic} in simple terms.",
            "The context contains information about {topic} in machine learning. Provide a comprehensive explanation.",
            "Explain {topic} based on the machine learning concepts in the context. Include key points and applications."
        ]
    
    def _get_scifi_templates(self) -> List[str]:
        """Get templates for science fiction domain"""
        return [
            "Based on the sci-fi movies in the context, provide information about {topic}.",
            "The context contains details about science fiction movies related to {topic}. Summarize the key points.",
            "Using the sci-fi movie information in the context, describe {topic} and its significance.",
            "Explain how {topic} is portrayed in the science fiction movies mentioned in the context."
        ]
    
    def _get_cosmos_templates(self) -> List[str]:
        """Get templates for cosmos domain"""
        return [
            "Based on the astronomy information in the context, explain {topic}.",
            "The context contains details about cosmic phenomena related to {topic}. Provide a clear explanation.",
            "Using the cosmos observations in the context, describe {topic} and its significance in astronomy.",
            "Explain {topic} based on the astronomical information provided in the context."
        ]
    
    def _get_general_templates(self) -> List[str]:
        """Get general templates for any domain"""
        return [
            "Based on the information in the context, provide a comprehensive answer about {topic}.",
            "The context contains relevant information about {topic}. Summarize the key points.",
            "Using the details provided in the context, explain {topic} clearly and concisely.",
            "Provide a detailed explanation of {topic} based on the information in the context."
        ]
    
    def _get_ml_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples for machine learning domain"""
        return [
            {
                "query": "What is deep learning?",
                "context": "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input. Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples.",
                "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep') to progressively extract higher-level features from raw input data. These neural networks are inspired by the structure of biological neural networks in the human brain. Deep learning models can learn to perform complex tasks by analyzing examples, without being explicitly programmed for those tasks. This approach has been particularly successful in areas like image recognition, natural language processing, and speech recognition."
            },
            {
                "query": "Explain supervised learning",
                "context": "Supervised learning involves training a model on labeled data, where the desired output is known. Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
                "response": "Supervised learning is a machine learning approach where the model is trained on labeled data, meaning the input data comes with the correct output (or 'label'). During training, the model learns to map inputs to outputs based on example input-output pairs. The goal is for the model to learn patterns that allow it to correctly predict outputs for new, unseen inputs. Common supervised learning tasks include classification (predicting a category) and regression (predicting a continuous value). This approach is called 'supervised' because the training process is guided by the correct answers, similar to how a teacher supervises a student's learning."
            }
        ]
    
    def _get_scifi_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples for science fiction domain"""
        return [
            {
                "query": "What are some popular sci-fi movies about space exploration?",
                "context": "Science Fiction Movie: Interstellar. Release Date: 2014-11-05. Overview: The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage. Popularity: 105.138. Genres: Adventure, Drama, Science Fiction. Science Fiction Movie: The Martian. Release Date: 2015-09-30. Overview: During a manned mission to Mars, Astronaut Mark Watney is presumed dead after a fierce storm and left behind by his crew. But Watney has survived and finds himself stranded and alone on the hostile planet. With only meager supplies, he must draw upon his ingenuity, wit and spirit to subsist and find a way to signal to Earth that he is alive. Popularity: 86.907. Genres: Drama, Adventure, Science Fiction.",
                "response": "Some popular sci-fi movies about space exploration include 'Interstellar' (2014) and 'The Martian' (2015). 'Interstellar' follows a group of explorers who use a wormhole to travel vast interstellar distances, pushing the boundaries of human space travel. 'The Martian' tells the story of astronaut Mark Watney who is stranded alone on Mars after being presumed dead, and must use his ingenuity to survive and signal Earth. Both films are highly rated in the science fiction genre and combine elements of adventure and drama with realistic portrayals of space exploration challenges."
            },
            {
                "query": "Tell me about alien invasion movies",
                "context": "Science Fiction Movie: Independence Day. Release Date: 1996-06-25. Overview: On July 2, a giant alien mothership enters orbit around Earth and deploys several dozen saucer-shaped 'destroyer' spacecraft that quickly lay waste to major cities around the planet. On July 3, the United States conducts a coordinated counterattack that fails. On July 4, a plan is devised to gain access to the interior of the alien mothership in order to plant a nuclear missile. Popularity: 64.273. Genres: Action, Adventure, Science Fiction. Science Fiction Movie: War of the Worlds. Release Date: 2005-06-28. Overview: Ray Ferrier is a divorced dockworker and less-than-perfect father. Soon after his ex-wife and her new husband drop off his teenage son and young daughter for a rare weekend visit, a strange and powerful lightning storm touches down. Popularity: 62.184. Genres: Adventure, Thriller, Science Fiction.",
                "response": "Alien invasion movies are a popular subgenre of science fiction. Notable examples include 'Independence Day' (1996), where giant alien motherships enter Earth's orbit and deploy destroyer spacecraft that devastate major cities, leading to humanity's desperate counterattack culminating on July 4th. Another significant film is 'War of the Worlds' (2005), which follows Ray Ferrier, a divorced dockworker who must protect his children when a powerful alien force invades Earth. These films typically combine elements of action, adventure, and thriller genres while exploring themes of humanity's resilience against overwhelming extraterrestrial threats. Both movies were commercially successful and feature spectacular visual effects depicting the alien invasions."
            }
        ]
    
    def _get_cosmos_examples(self) -> List[Dict[str, str]]:
        """Get few-shot examples for cosmos domain"""
        return [
            {
                "query": "Tell me about black holes",
                "context": "Cosmos Observation Date: 2019-04-10. Title: First Image of a Black Hole. Media Type: image. Explanation: The Event Horizon Telescope (EHT) has captured the first-ever image of a black hole. The image reveals the central dark region where light and matter cannot escape, surrounded by a ring of glowing gas that is being superheated as it spirals toward the black hole.",
                "response": "Black holes are fascinating cosmic phenomena where gravity is so strong that nothing, not even light, can escape once it passes the event horizon. On April 10, 2019, a significant breakthrough occurred when the Event Horizon Telescope (EHT) captured the first-ever direct image of a black hole. This historic image showed the central dark region (the shadow of the black hole) surrounded by a bright ring of superheated gas that glows as it spirals toward the black hole. This visual evidence confirmed many theoretical predictions about black holes and provided scientists with new data to study these extreme objects. The image was a remarkable achievement in astronomy, requiring a network of radio telescopes around the world working together to create a virtual Earth-sized telescope."
            },
            {
                "query": "What is the Hubble Space Telescope?",
                "context": "Cosmos Observation Date: 1990-04-24. Title: Hubble Space Telescope Deployment. Media Type: image. Explanation: The Hubble Space Telescope (HST) is a space telescope that was launched into low Earth orbit in 1990 and remains in operation. It is one of the largest and most versatile space telescopes, renowned for its discoveries and detailed images that have transformed our understanding of the universe.",
                "response": "The Hubble Space Telescope (HST) is a revolutionary space-based observatory that was deployed in low Earth orbit on April 24, 1990. As one of the largest and most versatile telescopes ever built, Hubble has been instrumental in transforming our understanding of the universe. Free from the distorting effects of Earth's atmosphere, it captures incredibly detailed images of distant stars, galaxies, and cosmic phenomena. Over its decades of operation, Hubble has made numerous groundbreaking discoveries, including helping to determine the age of the universe, confirming the existence of supermassive black holes, and providing evidence for dark energy. The telescope continues to operate today, sending back stunning images and valuable scientific data that have not only advanced astronomy but also captured the public's imagination."
            }
        ]
    
    def _get_general_examples(self) -> List[Dict[str, str]]:
        """Get general few-shot examples"""
        return [
            {
                "query": "What is the relationship between AI and machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn. Artificial intelligence refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
                "response": "The relationship between AI and machine learning is hierarchical: machine learning is a subset of artificial intelligence. AI is the broader concept of machines being able to carry out tasks in a way that we would consider 'smart' or intelligent. Machine learning is a specific approach to AI that focuses on using data and algorithms to imitate how humans learn, gradually improving its accuracy without being explicitly programmed for every scenario. In essence, machine learning is one of the primary methods by which we develop and advance artificial intelligence systems. While AI encompasses a wide range of techniques and approaches to create intelligent behavior, machine learning has become one of the most successful and widely used approaches in modern AI development."
            },
            {
                "query": "How do stars form?",
                "context": "Cosmos Observation Date: 2022-07-12. Title: Stellar Nursery in Carina Nebula. Media Type: image. Explanation: Stars form when dense regions within molecular clouds of gas and dust collapse under their own gravity. As the cloud collapses, the material at the center begins to heat up, forming a protostar that will eventually become hot enough to trigger nuclear fusion.",
                "response": "Stars form through a fascinating process that begins in molecular clouds - vast regions of gas (primarily hydrogen) and dust in space. The process starts when dense regions within these clouds begin to collapse under their own gravity, often triggered by external events like shock waves from nearby supernovae. As the cloud collapses, conservation of angular momentum causes it to spin faster and flatten into a disk, while the material at the center becomes increasingly dense and hot, forming what's called a protostar. When the temperature and pressure at the core become high enough (approximately 10 million degrees Celsius), nuclear fusion ignites, converting hydrogen into helium and releasing enormous energy. This marks the birth of a new star. The surrounding disk of material may eventually form planets, creating a new solar system. This process was beautifully captured in images of stellar nurseries like the Carina Nebula, where we can observe stars at various stages of formation."
            }
        ]
    
    def get_prompt(self, query: str, context: str, primary_source: str) -> str:
        """
        Generate an advanced prompt with few-shot examples
        
        Args:
            query (str): User query
            context (str): Retrieved context
            primary_source (str): Primary source domain
            
        Returns:
            str: Formatted prompt with examples
        """
        # Select domain (fallback to General if not found)
        domain = primary_source if primary_source in self.domain_templates else "General"
        
        # Select a template
        template = random.choice(self.domain_templates[domain])
        formatted_template = template.format(topic=query)
        
        # Select 1-2 few-shot examples
        examples = self.few_shot_examples[domain]
        selected_examples = random.sample(examples, min(2, len(examples)))
        
        # Format the prompt with examples
        prompt = f"""I'll provide you with some examples of good responses, then ask you to respond to a new query.

Examples:
"""
        
        # Add few-shot examples
        for i, example in enumerate(selected_examples):
            prompt += f"""
Example {i+1}:
Query: {example['query']}
Context: {example['context']}
Response: {example['response']}
"""
        
        # Add the current query and context
        prompt += f"""
Now, please respond to this query:
Query: {query}
Context: {context}
Instructions: {formatted_template}
Make sure your response is accurate, informative, and based only on the provided context.

Response:"""
        
        return prompt


class ResponseEvaluator:
    """Class for evaluating and improving generated responses"""
    
    def __init__(self):
        """Initialize response evaluator"""
        self.quality_criteria = [
            "Relevance to the query",
            "Factual accuracy based on context",
            "Completeness of information",
            "Clarity and coherence",
            "Appropriate level of detail"
        ]
    
    def evaluate_response(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated response
        
        Args:
            query (str): Original user query
            context (str): Retrieved context
            response (str): Generated response
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Simple heuristic evaluation
        evaluation = {}
        
        # Check response length relative to context
        context_words = len(context.split())
        response_words = len(response.split())
        
        if response_words < 10:
            evaluation["quality"] = "poor"
            evaluation["issue"] = "Response is too short"
            evaluation["suggestion"] = "Generate a more detailed response"
        elif response_words > context_words * 1.5:
            evaluation["quality"] = "suspicious"
            evaluation["issue"] = "Response is much longer than context"
            evaluation["suggestion"] = "Verify that the response doesn't hallucinate information"
        else:
            evaluation["quality"] = "good"
        
        # Check if response contains query terms
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        query_term_overlap = len(query_terms.intersection(response_terms)) / len(query_terms) if query_terms else 0
        
        evaluation["query_relevance"] = query_term_overlap
        
        if query_term_overlap < 0.3:
            evaluation["relevance_warning"] = "Response may not directly address the query"
        
        return evaluation
    
    def improve_response(self, query: str, context: str, response: str, evaluation: Dict[str, Any]) -> str:
        """
        Improve a response based on evaluation
        
        Args:
            query (str): Original user query
            context (str): Retrieved context
            response (str): Generated response
            evaluation (Dict[str, Any]): Evaluation results
            
        Returns:
            str: Improved response or original if no improvement needed
        """
        if evaluation["quality"] == "good" and evaluation.get("relevance_warning") is None:
            return response
        
        # Simple improvements based on evaluation
        if evaluation["quality"] == "poor" and "too short" in evaluation.get("issue", ""):
            # Extract key information from context related to query
            query_terms = query.lower().split()
            context_sentences = [s.strip() for s in context.split('.') if s.strip()]
            
            relevant_sentences = []
            for sentence in context_sentences:
                if any(term in sentence.lower() for term in query_terms):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                improved_response = f"{response} {' '.join(relevant_sentences)}"
                return improved_response
        
        # If response might not be relevant enough, add a direct reference to the query
        if evaluation.get("relevance_warning"):
            improved_response = f"Regarding {query}, {response}"
            return improved_response
        
        return response


# Example usage
if __name__ == "__main__":
    # Test the prompt template
    prompt_template = PromptTemplate()
    test_query = "What is deep learning?"
    test_context = "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input."
    test_prompt = prompt_template.get_prompt(test_query, test_context, "Machine Learning")
    print(test_prompt)
    
    # Test the response evaluator
    evaluator = ResponseEvaluator()
    test_response = "Deep learning is a type of machine learning."
    evaluation = evaluator.evaluate_response(test_query, test_context, test_response)
    print(f"\nEvaluation: {evaluation}")
    
    improved_response = evaluator.improve_response(test_query, test_context, test_response, evaluation)
    print(f"\nImproved response: {improved_response}")
