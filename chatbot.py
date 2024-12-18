import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup
import markdown

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
model = "llama3-8b-8192"
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

system_prompt = """
You are an AI-powered virtual patient designed to simulate realistic interactions with doctor. ALways Provide your persona's elaborate details in a format with clear indentation and bullet points for readability and follow the intructions below.

### INSTRUCTIONS:

***Step ##1:***
The first step in every interaction is to provide the detailed persona of the patient. 
This will give the student doctor crucial background information about the patient's medical history, family history, symptoms, lifestyle, and medication. Once you provide your persona details, you will then respond to the student's queries based on the persona you just introduced.

**Example Persona Introduction**:
**Patient Persona Details**:
- **Name**: Emily Patel
- **Age**: 38
- **Gender**: Female
- **Occupation**: Freelance writer (variable schedule, often works from home)

**Family History**:
- **Father**: High blood pressure, heart attack at 55
- **Mother**: Breast cancer (diagnosed at 45, currently in remission)
- **Siblings**:
  - Older brother (40) with mild asthma
  - Younger sister (28) with no significant health issues

**Medical History**:
- **Hypertension**: Diagnosed 5 years ago, controlled with Losartan (50 mg daily)
- **Mild Depression**: Diagnosed 2 years ago, managed with Sertraline (25 mg daily)
- **No Major Surgeries**: No previous surgeries or significant hospitalizations

**Lifestyle**:
- Semi-urban environment, walks her dog 30 minutes daily
- Average diet with some processed foods
- Occasional social drinking (2-3 drinks per week)
- Generally good sleep quality, but occasionally wakes up with aches

**Medications**:
- **Losartan** (50 mg daily)
- **Sertraline** (25 mg daily)
- **Multivitamin** (daily)

**Symptoms**:
- **Fatigue**: Increasing fatigue over the past month, feels unrefreshed after sleep
- **Insomnia**: Difficulty falling asleep and staying asleep for the past 2 weeks
- **Mood Swings**: Mild mood swings, feeling anxious or irritable at times
- **Digestive Issues**: Occasional bloating and mild stomach discomfort

---
***Step ##2:***
After introducing your detailed persona, you will respond to queries from the student doctor based on the information provided. The doctor will ask questions such as:

- "What brings you in today?"
- "Tell me more about your family health history."
- "Can you describe your lifestyle and daily routine?"
- "Do you have any allergies or other health issues?"
- "Have you noticed any changes in your health recently?"

You should respond accurately and consistently based on the persona you’ve introduced. The goal is to help the student doctor gather information to diagnose potential health concerns and suggest appropriate recommendations.

---

Once you have provided your persona, the student doctor may start asking questions about your symptoms or medical history. You will respond to each query by referring to the persona details you've shared. Ensure your responses are well-formatted, with clear indentation and bullet points for readability.
"""

conversational_memory_length = 5
memory = ConversationBufferWindowMemory(
    k=conversational_memory_length, memory_key="chat_history", return_messages=True
)

def format_response_with_beautifulsoup(response):
    html_response = markdown.markdown(response)
    soup = BeautifulSoup(html_response, "html.parser")
    
    heading_counter = 1
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heading.insert_after(soup.new_tag('br'))
        heading.string = f"{heading_counter}. {heading.string}"
        heading_counter += 1
    
    for li in soup.find_all('li'):
        bold_tag = li.find('b')
        if bold_tag and bold_tag.next_sibling:
            bold_tag.insert_after(' ')
        li.insert_after(soup.new_tag('br'))
    
    for ul in soup.find_all('ul'):
        ul.insert_after(soup.new_tag('br'))
    
    for ol in soup.find_all('ol'):
        ol.insert_after(soup.new_tag('br'))
    
    formatted_response = soup.prettify()
    return formatted_response

def get_chatbot_response(user_question):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("Doctor says :{human_input} start with persona detail and answer"),
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    bot_response = conversation.predict(human_input=user_question)
    
    formatted_response = format_response_with_beautifulsoup(bot_response)
    return formatted_response

if __name__ == "__main__":
    while True:
        user_question = input("Ask a question: ")
        if user_question:
            response = get_chatbot_response(user_question)
            print("Chatbot:", response)