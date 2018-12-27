import re;
import os;

#load and read corpus
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

corpus = open(os.path.join(__location__, 'cnnhealth.txt'))
inputFile = corpus.read()


#regex to extract tweet text content
pattern = re.compile(r'\+0000\s20\d{2}\|(.*?)\shttp',re.DOTALL)
matches = pattern.finditer(inputFile)

#open output file
outputFile = open('output.txt', 'w+')

#clean tweets and store in output
for match in matches:
    text = match.group(1)
    text = re.sub(r'RT @(.*?):\s', '', text)
    text = re.sub(r'@(.*?)\s', '', text)
    text = re.sub(r'@(.*?)\.', '', text)
    text = re.sub(r'#(.*?):\s', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    outputFile.write(text + '\n')

