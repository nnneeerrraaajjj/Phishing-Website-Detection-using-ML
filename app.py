from flask import Flask,render_template,request
import os
import io
import sys
import math
import time
import random
import requests
import collections
import numpy as np
from os import walk
from joblib import dump, load
from tokenizers import ByteLevelBPETokenizer
from langdetect import detect
import pickle

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("home.html")

@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/getURL',methods=['GET','POST'])
def getURL():
	if request.method == 'POST':
		urlname  = request.form['url']
		url = request.form['url']
		print(url)
		tokenizerFolder = "tokenizer"
		savedModelDirectory = "saved_models"
		websiteToTest = url
		threshold = 0.5
		tokenizer = ByteLevelBPETokenizer(
			tokenizerFolder + "/tokenizer.tok-vocab.json",
			tokenizerFolder + "/tokenizer.tok-merges.txt",
		)
		tokenizerVocabSize = tokenizer.get_vocab_size()
		print("Tokenizer files have been loaded and the vocab size is %d..." % tokenizerVocabSize)
		model = load(savedModelDirectory + "/phishytics-model.joblib")
		print("Model loaded...")

		# Load document frequency dictionary
		docDict = np.load(savedModelDirectory + "/phishytics-model-tfidf-dictionary.npy", allow_pickle=True).item()
		print("Document frequency dictionary loaded...")

		# Testing
		print("Loading webpage...")
		try:
			request1 = requests.get(websiteToTest)
			webpageHtml = str(request1.text)
			webpageHtml = webpageHtml.replace("\n", " ")
		except Exception as e:
			print('\n',e)
			print("\nAn error occurred, exiting now... ")
			exit()
        
		# Convert text into feature vector
		output = tokenizer.encode(webpageHtml)
		outputDict = collections.Counter(output.ids)

		# Apply tfidf weighting
		totalFilesUnderConsideration = docDict["totalFilesUnderConsideration"]
		array = [0] * tokenizerVocabSize
		for item in outputDict:
			if len(docDict[item]) > 0:
				array[item] = (outputDict[item]) * (math.log10( totalFilesUnderConsideration / len(docDict[item])))
		predictionProbability = model.predict_proba([array])[0][1]
		print("\n****************************\n--> Probability that the website is phishing: %.2f" % (predictionProbability * 100))

		prediction = "NOT PHISHING"
		predicted_value = 0
		if predictionProbability > threshold:
			prediction = "PHISHING"
			predicted_value = 1
		print("--> Based on your threshold of %.2f, this website is +++'%s'+++" % (threshold, prediction))
		print("****************************")
		
        #print(predicted_value)
		if predicted_value == 0:    
			value = "Legitimate"
			return render_template("home.html",error=value)
		else:
			value = "Phishing"
			return render_template("home.html",error=value)
if __name__ == "__main__":
	app.run(debug=True)