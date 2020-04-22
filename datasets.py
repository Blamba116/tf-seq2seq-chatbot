# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# except for some changes, the code is taken from the Conchylicultor project
# extractText function changed

import os
import re
import ast

import nltk
from tqdm import tqdm

nltk.download('punkt')

"""
Load the cornell movie dialog corpus.

Available from here:
http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

"""

class CornellData:
    """

    """

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.lines = {}
        self.conversations = []

        MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]

        self.lines = self.loadLines(os.path.join(dirName, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        self.conversations = self.loadConversations(os.path.join(dirName, "movie_conversations.txt"), MOVIE_CONVERSATIONS_FIELDS)

        # TODO: Cleaner program (merge copy-paste) !!

    def loadLines(self, fileName, fields):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        lines = {}

        with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]

                lines[lineObj['lineID']] = lineObj

        return lines

    def loadConversations(self, fileName, fields):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            list<dict<str>>: the extracted fields for each line
        """
        conversations = []

        with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])

                conversations.append(convObj)

        return conversations

    def getConversations(self):
        return self.conversations
    

def extractText(line, kind='fast', text='request'):
    if kind == 'fast':
        GOOD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#+_]')
        REPLACE_SEVERAL_SPACES = re.compile('\s+')

        line = line.lower()
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = GOOD_SYMBOLS_RE.sub('', line)
        line = REPLACE_SEVERAL_SPACES.sub(' ', line)
        return line.strip()
    elif kind == 'word_tokenize':
        return nltk.word_tokenize(line)
    elif kind == 'sent_tokenize':
        return nltk.sent_tokenize(line)
    elif kind == 'last_sentence':
        line_tok = nltk.sent_tokenize(line)
        # check if sentence is empty
        if len(line_tok) > 0:
            # take first or last sentence
            if text == 'request':
                line_tok = line_tok[-1]
            elif text == 'reply':
                line_tok = line_tok[0]

            GOOD_SYMBOLS_RE = re.compile('[^0-9a-z ]')
            REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#+_]')
            REPLACE_SEVERAL_SPACES = re.compile('\s+')

            line_tok = line_tok.lower()
            line_tok = REPLACE_BY_SPACE_RE.sub(' ', line_tok)
            line_tok = GOOD_SYMBOLS_RE.sub('', line_tok)
            line_tok = REPLACE_SEVERAL_SPACES.sub(' ', line_tok)
            return line_tok.strip()
        else:
            return line.strip()
    elif kind == 'as-is':
        return line
    else:
        raise 'unexpected value for kind, chose one of 1) fast, 2) word_tokenize, 3) sent_tokenize, 4) last_sentence'


def splitConversations(conversations, max_len=20, kind='fast'):
    data = []
    for i, conversation in enumerate(tqdm(conversations)):
        lines = conversation['lines']
        for i in range(len(lines) - 1):
            request = extractText(lines[i]['text'], kind=kind, text='request')
            reply = extractText(lines[i + 1]['text'], kind=kind, text='reply')
            if 0 < len(request) <= max_len and 0 < len(reply) <= max_len:
                data += [(request, reply)]
    return data


def readCornellData(path, max_len=20, kind='fast'):
    dataset = CornellData(path)
    conversations = dataset.getConversations()
    return splitConversations(conversations, max_len=max_len, kind=kind)