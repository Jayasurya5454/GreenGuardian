import os
import google.generativeai as genai

os.environ['GOOGLE_API_KEY'] = 'AIzaSyAE4pp7LbGcighoUahPhwA5QAHQKTTLlTE'


genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))

def format_text(text):
    formatted_text = text.replace('.', '  ./n')
    return formatted_text

def mainn(predicted_class_label):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(predicted_class_label)

    formatted_content = format_text(response.text)
    return formatted_content
