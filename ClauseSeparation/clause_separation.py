import csv
import json
import pprint
import re
from collections import OrderedDict
import argparse
from google.cloud import language_v1


class SplitClauses:
    def __init__(self, out_path, comment_file=None, service_ac_file_path=None,
                 api_file=None, start_from=None, end_to=None):
        self.comment_file = comment_file
        self.start_from = start_from
        self.end_to = end_to
        self.service_account = service_ac_file_path
        self.out_path = out_path
        self.api_file = api_file

    def load_comments(self):
        start_from = int(self.start_from) if self.start_from else 0
        end_to = int(self.end_to) if self.end_to else None
        comments = []
        count = 0
        csv_data = csv.DictReader(open(self.comment_file, 'rU'), delimiter=',', quotechar='"')
        for row in csv_data:
            if count < start_from:
                count += 1
                continue
            if end_to and count > end_to:
                break
            comments.append(row)
            count += 1
        return comments

    def split_sentences(self, df):

        comments_dict = dict()
        for i in df:
            if i['id'] not in comments_dict.keys():
                list_of_comments = []
                # print(brand_str[i][num]['comment'], brand_str[i][num]['title'])
                if i['comment'] is not None:
                    list_of_comments.extend([k.lower() for k in list(
                        map(str.strip, re.split(r"[.!?](?!$)", i['comment']))) if
                                             (k != '') and (len(k) > 5) and (k not in list_of_comments)])
                if i['title'] is not None:
                    list_of_comments.extend([k.lower() for k in list(
                        map(str.strip, re.split(r"[.!?](?!$)", i['title']))) if
                                             (k != '') and (len(k) > 5) and (k not in list_of_comments)])
                # print(list_of_comments)
                comments_dict[i['id']] = {
                    'title': i['title'],
                    'comment': i['comment'],
                    'list_sentences': list_of_comments,
                }
        return comments_dict

    def sample_analyze_syntax(self, text_content):
        """
        Analyzing Syntax in a String

        Args:
          text_content The text content to analyze
        """
        getjson = dict()
        client = language_v1.LanguageServiceClient.from_service_account_json(
            self.service_account)

        # text_content = 'This is a short sentence.'

        # Available types: PLAIN_TEXT, HTML
        type_ = language_v1.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages
        language = "en"
        document = {"content": text_content, "type_": type_, "language": language}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = language_v1.EncodingType.UTF8
        getjson[text_content] = OrderedDict()
        response = client.analyze_syntax(request={'document': document, 'encoding_type': encoding_type})
        # Loop through tokens returned from the API

        pprint.pprint(response)

        for token in response.tokens:
            # Get the text content of this token. Usually a word or punctuation.
            text = token.text

            dependency_edge = token.dependency_edge
            part_of_speech = token.part_of_speech

            tg = language_v1.PartOfSpeech.Tag(part_of_speech.tag).name
            cse = language_v1.PartOfSpeech.Case(part_of_speech.case).name
            numbr = language_v1.PartOfSpeech.Number(part_of_speech.number).name
            persn = language_v1.PartOfSpeech.Person(part_of_speech.number).name

            if text.content not in getjson[text_content].keys():
                getjson[text_content][text.content] = OrderedDict(
                    {'POS': [{'tag': tg, 'case': cse, 'number': numbr, 'person': persn}],
                     "BeginOffset": text.begin_offset, "Head token index": dependency_edge.head_token_index,
                     'label': language_v1.DependencyEdge.Label(dependency_edge.label).name})
            else:
                key = text.content + str(dependency_edge.head_token_index)
                getjson[text_content][key] = OrderedDict(
                    {'POS': [{'tag': tg, 'case': cse, 'number': numbr, 'person': persn}],
                     "BeginOffset": text.begin_offset, "Head token index": dependency_edge.head_token_index,
                     'label': language_v1.DependencyEdge.Label(dependency_edge.label).name})

                # for sent in getjson.keys():

        return getjson

    def get_results_and_save(self, comment_str):
        resp = dict()
        for id in comment_str.keys():
            resp[id] = []
            for sent in comment_str[id]['list_sentences']:
                resp[id].append(self.sample_analyze_syntax(sent))

        self.save_file(resp)

    def save_file(self, resp):
        with open(self.out_path, 'w') as f:
            json.dump(resp, f)

    def load_context_file(self):
        f = open(self.api_file)
        # returns JSON object as a dictionary
        data = json.load(f, object_pairs_hook=OrderedDict)
        f.close()
        return data

    def split_and_return_clauses(self, data):
        listOfTrees = []
        for i in data:
            verblist = []
            coveredWords = []
            for key in i.keys():
                for word in i[key].keys():
                    if i[key][word]['POS'] == 'VERB' and i[key][word]['label'] in ['ROOT', 'CONJ', 'ADVCL', 'RCMOD']:
                        verblist.append(word)

                coveredWords.extend(verblist)
                treelist = dict()
                for word in i[key].keys():
                    try:
                        rootword = list(i[key].keys())[int(i[key][word]['Head token index'])]
                    except:
                        try:
                            rootword = list(i[key].keys())[int(i[key][word]['Head token index']) - 1]
                        except:
                            rootword = list(i[key].keys())[int(i[key][word]['Head token index']) - 2]
                    if rootword in verblist:
                        if i[key][word]['label'] != 'P':
                            if rootword not in treelist.keys():
                                treelist[rootword] = []
                            treelist[rootword].append(word)
                            coveredWords.append(word)
                    else:
                        continue

                # print(treelist)
                # print(coveredWords)
                for word in i[key].keys():
                    if word not in coveredWords and i[key][word]['label'] != 'P':
                        # print(word)
                        try:
                            wordkey = list(i[key].keys())[int(i[key][word]['Head token index'])]
                        except:
                            try:
                                wordkey = list(i[key].keys())[int(i[key][word]['Head token index']) - 1]
                            except:
                                wordkey = list(i[key].keys())[int(i[key][word]['Head token index']) - 2]

                        # print(wordkey)
                        for keyword in treelist.keys():
                            if wordkey in treelist[keyword]:
                                treelist[keyword].append(word)
                                # print(treelist[keyword])

            if len(treelist.keys()) == 0:
                listOfTrees.append(key)
            else:
                listOfTrees.append(treelist)

        return listOfTrees

    def get_all_clauses(self):
        data = self.load_context_file()
        listOfTrees = self.split_and_return_clauses(data)
        clauses = []
        for num in range(0, len(data)):
            try:
                treelist = listOfTrees[num]
                for ch in treelist.keys():
                    listCheck = treelist[ch]
                    keylist = list(treelist.keys())
                    keylist.remove(str(ch))
                    # print(keylist)
                    strword = str()
                    i = data[num]
                    clist = []
                    for key in i.keys():
                        for word in i[key].keys():
                            # print(word)
                            if word in listCheck or word == ch:
                                if word not in keylist:
                                    # print(word)
                                    strword = strword + ' ' + word

                        clist.append(strword)

                    clauses.append(clist)
            except:
                clauses.append(listOfTrees[num])

        return clauses

    def init_function(self):
        if self.comment_file is not None:
            df = self.load_comments()
            comment_list = self.split_sentences(df)
            self.get_results_and_save(comment_list)
        if self.api_file is not None:
            clauses = self.get_all_clauses()
            self.save_file(clauses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-inp", dest="comment_file",
                        metavar="comment-file", help="comment file for dependency tree api")
    parser.add_argument("-ser", dest="service_ac_file_path", metavar="service-ac-file", help="google service ac file")
    parser.add_argument("-resp", dest="api_file",
                        metavar="api-file", help="dependency file from api")
    parser.add_argument("-o", dest="out_path",
                        metavar="out-file-path", help="output csv file path")
    parser.add_argument("-s", "--start-from", dest="start_from",
                        metavar="start-from-index", help="start index in comment file")
    parser.add_argument("-e", "--end-to", dest="end_to",
                        metavar="end-to-index", help="end index in comment file")

    options = parser.parse_args()
    print(options)

    marking = SplitClauses(out_path=options.out_path, comment_file=options.comment_file,
                           service_ac_file_path=options.service_ac_file_path, api_file=options.api_file,
                           start_from=options.start_from, end_to=options.end_to)

    marking.init_function()
