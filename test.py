from gradeAnswer import grade_answer

# Provided by the educator

## 1. Model Answer(Answer key)
model_answer = """ChatGPT is a large language model developed by OpenAI, 
which is designed to interact with humans using natural language
processing. It uses a state-of-the-art artificial intelligence technique
called the transformer model, which allows it to understand and respond to natural
language queries with high accuracy and fluency. ChatGPT is capable of answering a
wide range of questions on different topics, generating text, and engaging in conversations
with users. It has been trained on a vast corpus of text data and is constantly improving
through ongoing learning and updates. ChatGPT is widely used in 
various applications such as customer service, language translation, and chatbots."""
## 2. Set of Keywords
keywords = [    'ChatGPT',    'language model',    'answering',
            'conversation',  'artificial intelligence']

#Provided by the student
## 1. Example of a answer 
student_answer1 = """OpenAI's ChatGPT is a cutting-edge language model that enables human-like interactions through natural language processing.
By utilizing a powerful machine learning method known as the transformer model, ChatGPT can comprehend and deliver precise responses
to a wide range of questions on diverse topics. With access to an extensive dataset of text information and 
continuous learning through updates, ChatGPT has become increasingly proficient in generating human-like text and 
engaging in conversations with users. This technology has numerous applications, including customer support, language translation, and chatbot interactions."""

student_answer2 = """ChatGPT is an artificial intelligence tool that uses natural language processing 
to interact with humans. It is widely used in various applications, including customer service and chatbots, 
due to its ability to understand and respond to human queries. With the help of the transformer model, ChatGPT
is able to generate text with high accuracy and fluency. While it is a powerful tool, it is not without its limitations
and may not always provide the most comprehensive or accurate answers, depending on the complexity of the query. Nevertheless,
ChatGPT remains an important technology in the field of natural language processing and is constantly evolving through
ongoing learning
and updates."""

### 3. Example of an incorrect answer
student_answer3 = """ChatGPT is a type of garden tool that is used to prune trees and bushes. 
It is a handheld device that consists of a pair of sharp blades and a handle, which allows the
user to grip and cut through branches and foliage with ease. ChatGPT is particularly useful for maintaining
the appearance and health of trees and plants in gardens and parks. Its design is based on the principles of 
horticulture and it has been optimized for use by both professional landscapers and hobbyist gardeners. Although 
it may seem like an unusual name for a garden tool, ChatGPT has become increasingly popular in recent years due to 
its effectiveness and ease of use.
"""


print(grade_answer(keywords, model_answer, student_answer2))
