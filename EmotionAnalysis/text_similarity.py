# -*- coding: utf-8 -*-
"""
Using similarity metrics to mark comments for particular emotions

"""

import argparse
import csv

import pandas as pd


class SimilarityModels:

    def __init__(self, excel_path, correct_path, incorrect_path, outpath):
        """
        initialize the parameters
        @param excel_path: Excel path with the marked comments created for the marking team
        @param correct_path: Pre generated file using lexicons with correctly marked comments
        @param incorrect_path: Pre generated file using lexicons with incorrecty marked comments
        @param outpath: output file path for the dataset
        """
        self.emotionNames = ['Satisfaction', 'Anger', 'Relief', 'Disappointment', 'Disgust',
                             'Boredom', 'Reminiscence', 'Guilt', 'Worry', 'Fear', 'Sluggish',
                             'Sickly', 'Confusion', 'Stimulation', 'Excitement/Adventurous',
                             'Comfort', 'Curiosity', 'Delight', 'Amazed', 'Enjoyment/Amusement',
                             'Healthy/Energetic', 'Intense', 'Sophisticated']

        self.dataColumns = {'emotion-1': 'context-1', 'emotion-2': 'context-2', 'emotion-3': 'context-3',
                            'emotion-4': 'context-4', 'emotion-5': 'context-5'}
        self.EXCEL_PATH = excel_path
        self.correct_path = correct_path
        self.incorrect_path = incorrect_path
        self.OUTPATH = outpath

    def load_comments(self, comment_file_path, start_from, end_to):
        """

        @param comment_file_path: File path to read
        @param start_from: start index of the file
        @param end_to: end index of the file
        @return: comments: list of ordered dictionary of comments
        """
        start_from = int(start_from) if start_from else 0
        end_to = int(end_to) if end_to else None
        comments = []
        count = 0
        csv_data = csv.DictReader(open(comment_file_path, 'rU'), delimiter=',', quotechar='"')
        for row in csv_data:
            if count < start_from:
                count += 1
                continue
            if end_to and count > end_to:
                break
            comments.append(row)
            count += 1
        return comments

    def load_excel_and_proc(self):
        """
        Creates the dictionary with Excel file with the marking team. The marked sentences in the
        Excel file are added to the dictionary depending on the emotion marked.

        @return: emotions dictionary - processed dict created from excel file
        """
        df_new_marked = pd.read_excel(self.EXCEL_PATH, sheet_name='[REDACTED]')
        df_new_marked = df_new_marked[:3400]

        emotionsDictionary = dict()

        for keys in self.emotionNames:
            emotionsDictionary[keys.lower()] = []
        for col in self.dataColumns.keys():
            for emotion in self.emotionNames:
                # print(col)
                rows = df_new_marked.loc[df_new_marked[col] == emotion]

                rows = rows.reset_index(drop=True)
                for i in range(0, len(rows)):
                    emotionsDictionary[emotion.lower()].append(str(rows[self.dataColumns[col]][i]))
                    # print(rows[dataColumns[col]][i])
        print(emotionsDictionary.keys())
        return emotionsDictionary

    def load_unmatched_comms(self):
        """
        Creates a dictionary with the incorrectly marked comments and sorts them depending on their emotion
        @return: incorrect: dictionary with incorrectly marked comments sorted based on the emotion.
        """
        incorcomments = self.load_comments(self.incorrect_path, None, None)

        iemotionDict = dict()
        i = 0
        while i < len(incorcomments):
            c = incorcomments[i]
            if c['id'] != '':
                iemotionDict[c['id']] = []
                iemotionDict[c['id']].append(c)
                id = c['id']
                i += 1
                if i < len(incorcomments):
                    c = incorcomments[i]
                else:
                    break
                while c['id'] == '':
                    iemotionDict[id].append(c)
                    i += 1
                    if i >= len(incorcomments):
                        break
                    else:
                        c = incorcomments[i]
                    if c['id'] != '':
                        break

        incorrect = dict()
        for row in iemotionDict.keys():
            od = iemotionDict[row]
            incorrect[row] = []
            for i in od:
                if i['emotion'] != '' and i['context'] != '':
                    incorrect[row].append([i['context'], i['emotion']])

        return incorrect

    def load_matched_comms(self):
        """
        Correctly marked comments are separated by emotions into the dictionary
        @return: correct: dictionary with comments sorted into separate emotions.
        """

        corcomments = self.load_comments(self.correct_path, None, None)
        emotionDict = dict()
        i = 0
        while i < len(corcomments):
            c = corcomments[i]
            if c['id'] != '':
                emotionDict[c['id']] = []
                emotionDict[c['id']].append(c)
                id = c['id']
                i += 1
                if i < len(corcomments):
                    c = corcomments[i]
                else:
                    break
                while c['id'] == '':
                    emotionDict[id].append(c)
                    i += 1
                    if i >= len(corcomments):
                        break
                    else:
                        c = corcomments[i]
                    if c['id'] != '':
                        break
        correct = dict()
        for row in emotionDict.keys():
            od = emotionDict[row]
            correct[row] = []
            for i in od:
                if i['emotion'] != '' and i['context'] != '':
                    correct[row].append([i['context'], i['emotion']])

        return correct

    def sort_dict(self):
        """
        segregate the data
        @return: sorted and insorted (sentences that need to be marked with 1, 0 respectively)
        """
        correct = self.load_matched_comms()
        incorrect = self.load_unmatched_comms()
        emotionsDictionary = self.load_excel_and_proc()

        missing = dict()
        for i in incorrect.keys():
            row = incorrect[i]
            cor = correct[i]
            missing[i] = []
            for [x, y] in row:
                if [x, y] not in cor:
                    missing[i].append([x, y])

        sorted = dict()

        for i in correct.keys():
            row = correct[i]
            for [x, y] in row:
                if y not in sorted.keys():
                    sorted[y] = []
                sorted[y].append(x.lower())

        for key in sorted.keys():
            rows = emotionsDictionary[key]
            for i in rows:
                sorted[key].append(str(i).lower())

        ds = dict()

        for key in sorted.keys():
            print(key, len(sorted[key]))

        insorted = dict()

        for i in missing.keys():
            row = missing[i]
            for [x, y] in row:
                if y not in insorted.keys():
                    insorted[y] = []
                insorted[y].append(x)

        return [sorted, insorted]

    def combinantorial(self, lst):
        """
        Create comment pairs for similarity model
        @return: pairs - list of sentence pairs
        """
        index = 1
        pairs = []
        for element1 in lst:
            for element2 in lst[index:]:
                pairs.append((element1, element2))
            index += 1

        return pairs

    def combine_sent(self):
        '''
        combine and save dataset.
        @return: None.
        '''
        ds = dict()
        sorted, insorted = self.sort_dict()
        for key in sorted.keys():
            row = sorted[key][:54]

            ds[key] = []
            ds[key] = self.combinantorial(row)

        dslst = []

        for key in ds.keys():
            for (x, y) in ds[key]:
                dslst.append([x, y, key])

        df = pd.DataFrame(dslst, columns=['sentence1', 'sentence2', 'similarity'])

        df.to_csv(self.OUTPATH)

        dis = []
        for emotion in sorted.keys():
            row = sorted[emotion][:54]
            # for emotion2 in sorted.keys():
            #  if emotion1 !=emotion2:
            lis = [[x, y, 0] for x in insorted[emotion][:54] for y in row][:1431]
            for each in lis:
                dis.append(each)

        dflst = dslst + dis

        df = pd.DataFrame(dflst, columns=['sentence1', 'sentence2', 'similarity'])

        df.to_csv(self.OUTPATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-exc", dest="excel_file", metavar="excel-file-path", help="excel file path")
    parser.add_argument("-cor", dest="correct_file", metavar="correct-csv-file-path", help="correct path")

    parser.add_argument("-inc", dest="incorrect_file", metavar="incorrect-file-path", help="incorrect path")

    parser.add_argument("-out", dest="outpath", metavar="output-path", help="output path")
    options = parser.parse_args()
    print(options)

    marking = SimilarityModels(excel_path=options.excel_file,
                               correct_path=options.correct_file,
                               incorrect_path=options.incorrect_file,
                               outpath=options.outpath
                               )
    marking.combine_sent()
