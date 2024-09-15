import pickle
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
def preprocess_text(text):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Load the sentiment model and vectorizer
with open('trained_model.pkl', 'rb') as model_file:
    sentiment_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    sentiment_vectorizer = pickle.load(vectorizer_file)

# Load the emotion model and vectorizer
with open('emotion_trained_model.pkl', 'rb') as model_file:
    emotion_model = pickle.load(model_file)

with open('emotion_vectorizer.pkl', 'rb') as vectorizer_file:
    emotion_vectorizer = pickle.load(vectorizer_file)

# Read sentences from read.txt with UTF-8 encoding
with open('read.txt', 'r', encoding='utf-8') as file:
    example_texts = file.readlines()

# Define a mapping from integer labels to emotion names
emotion_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
emotion_counts = {emotion: 0 for emotion in emotion_mapping.values()}

sentiment_labels = {0: 'Negative', 1: 'Positive'}
sentiment_counts = {label: 0 for label in sentiment_labels.values()}

test = "this university would be lovely if it werent for dr ballout"

processed=preprocess_text(test)
sentiment= sentiment_vectorizer.transform([processed])
prediction=sentiment_model.predict(sentiment)
s=sentiment_labels[prediction[0]]
sentiment_counts[s]+=1


emotion= emotion_vectorizer.transform([processed])
prediction_emotions=emotion_model.predict(emotion)
s=emotion_mapping[prediction_emotions[0]]
emotion_counts[s]+=1




# Process each example text
# for example_text in example_texts:
#     example_text = example_text.strip()
#     if not example_text:
#         continue
#
#     preprocessed_text = preprocess_text(example_text)
#
#     sentiment_transformed_text = sentiment_vectorizer.transform([preprocessed_text])
#     sentiment_prediction = sentiment_model.predict(sentiment_transformed_text)
#     sentiment = sentiment_labels[sentiment_prediction[0]]
#     sentiment_counts[sentiment] += 1
#
#     emotion_transformed_text = emotion_vectorizer.transform([preprocessed_text])
#     emotion_prediction = emotion_model.predict(emotion_transformed_text)
#     emotion_name = emotion_mapping[emotion_prediction[0]]
#     emotion_counts[emotion_name] += 1
#
#     print(f'Text: "{example_text}"')
#     print(f'Sentiment Prediction: {sentiment}')
#     print(f'Emotion Prediction: {emotion_name}\n')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['red', 'green'])
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.bar(emotion_counts.keys(), emotion_counts.values(), color=['blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta'])
plt.title('Emotion Analysis')
plt.xlabel('Emotion')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('sentiment_emotion_analysis.png')
plt.show()
