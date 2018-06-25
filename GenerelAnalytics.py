import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.summarization import summarize
import math
import ast
import json

# import observation csv file
obs_df = pd.read_csv("observations_with_teachers.csv")
obs_df['Sentiment'] = 0

# import student marks csv file
student_marks = pd.read_csv("student_marks.csv")

studentDict = {}
teacherDict = {}

sid = SentimentIntensityAnalyzer()

"""
Process the observation csv file and create the dataset
Add the student and teachers datasets differently
"""
def process_observation_csv():
    for row in range(0, len(obs_df)):
        obs = obs_df.get_value(row, 'Observation')
        studentCode = str(obs_df.get_value(row, 'Student_Code'))
        teacherCode = str(obs_df.get_value(row, 'Teacher_Code'))
        if obs != '' and obs is not None and isinstance(obs, str):
            sent = sid.polarity_scores(obs)['compound']
            obs_df.loc[row, 'Sentiment'] = sent
            std = studentDict.get(studentCode, None)
            tempStudentDict = {}
            tempTeacherDict = {}
            tempStudentDict['studentName'] = str(obs_df.get_value(row, 'Student_Name'))
            tempTeacherDict['teacherName'] = str(obs_df.get_value(row, 'Teacher_Name'))
            if std is None:
                tempStudentDict['Sentiment'] = sent
                studentDict[studentCode] = tempStudentDict
            else:
                tempStudentDict['Sentiment'] = (std['Sentiment'] + sent) / 2
                studentDict[studentCode] = tempStudentDict

            teacher = teacherDict.get(teacherCode, None)
            if teacher is None:
                tempTeacherDict['Sentiment'] = sent
                teacherDict[teacherCode] = tempTeacherDict
            else:
                tempTeacherDict['Sentiment'] = (teacher['Sentiment'] + sent) / 2
                teacherDict[teacherCode] = tempTeacherDict

"""
Process the observation csv file
"""
process_observation_csv()

"""
Function to see which teacher is giving more positive and which teacher is giving more negative observation
"""
def createTeacherObservationDF():
    return obs_df.groupby(['Teacher_Code', 'Teacher_Name']).agg({'Observation': 'count', 'Sentiment': 'mean'})

"""
Function to see Students observations, either positive or negative
"""
def createStudentObservationDF():
    return obs_df.groupby(['Student_Code', 'Student_Name']).agg({'Observation': 'count', 'Sentiment': 'mean'})


# teacherDf = createTeacherObservationDF()
# studentDf = createStudentObservationDF()
# print(teacherDf)
# print(studentDf)
# print(teacherDf)

studentMarksDict = {}

"""
Function to process the marks of student and store the average mark of each student in studentMarksDict
"""
def process_student_marks():
    for i in range(0, len(student_marks)):
        student_code = student_marks.get_value(i, "Student_Code")
        marks = student_marks.get_value(i, "Marks")
        m = studentMarksDict.get(student_code, None)
        if m is None:
            studentMarksDict[student_code] = marks
        else:
            studentMarksDict[student_code] = (m + marks) / 2

"""
call mark process function
"""
process_student_marks()


# Kid got less mark and average observation is negative, then we can show what are all the positive observation others got?
"""
Get summary of the observation as list
"""

"""
Function to get the summarized text
"""
def get_summary(text, ratio=0.7, is_split=True):
    summarizedList = summarize(text, ratio=ratio, split=is_split)
    summarizedList = [s.replace('.', '') for s in summarizedList]
    return summarizedList


"""
Get sentiment analysis for student with type (type : pos, neg, neu). By default type is pos
"""

"""
Function to get the sentiment summary for students
"""
def get_sentiment_summary(student_code, type='pos'):
    obs_student_code = obs_df['Student_Code'] == student_code
    updated_student_obs = obs_df[obs_student_code]
    final_student_obs = None
    if type == 'pos':
        final_student_obs = updated_student_obs[updated_student_obs['Sentiment'] > 0.5]
    elif type == 'neg':
        final_student_obs = updated_student_obs[updated_student_obs['Sentiment'] < 0]
    else:
        greater_than_zero = updated_student_obs[updated_student_obs['Sentiment'] > 0]
        final_student_obs = greater_than_zero[greater_than_zero['Sentiment'] < 0.5]
    lst = final_student_obs['Observation'].unique().tolist()
    if len(lst) > 2:
        final_string = '. '.join(lst)
        return get_summary(final_string, 0.7, True)
    else:
        return [x.replace('.', '') for x in lst]

"""
Function to get the sentiment summary for teachers
"""
def get_sentiment_summary_teacher(teacher_code, type='pos'):
    obs_teacher_code = obs_df['Teacher_Code'] == teacher_code
    updated_teacher_obs = obs_df[obs_teacher_code]
    final_teacher_obs = None
    if type == 'pos':
        final_teacher_obs = updated_teacher_obs[updated_teacher_obs['Sentiment'] > 0.5]
    elif type == 'neg':
        final_teacher_obs = updated_teacher_obs[updated_teacher_obs['Sentiment'] < 0]
    else:
        greater_than_zero = updated_teacher_obs[updated_teacher_obs['Sentiment'] > 0]
        final_teacher_obs = greater_than_zero[greater_than_zero['Sentiment'] < 0.5]
    lst = final_teacher_obs['Observation'].unique().tolist()
    if len(lst) > 2:
        final_string = '. '.join(lst)
        return get_summary(final_string, 0.7, True)
    else:
        return [x.replace('.', '') for x in lst]

"""
Function to process the students compliments
"""
def process_compliment_csv():
    compliment_df = pd.read_csv("compliment.csv")
    compliment_df['Sentiment'] = 0

    sid = SentimentIntensityAnalyzer()
    for i in range(0, len(compliment_df)):
        comp = compliment_df.get_value(i, 'Compliment')
        if comp != '' and comp is not None and isinstance(comp, str):
            sent = sid.polarity_scores(comp)['compound']
            compliment_df.loc[i, 'Sentiment'] = sent

    return compliment_df


comp_new_df = process_compliment_csv()
characteristic_list = []

"""
Function to get Compliments of those students who gets more marks.
This function can be called for students who gets low marks so that they can adopt these to get good marks
"""
def get_compliment_for_highest_marks_students():
    listStudentCodes = []
    for key, value in studentMarksDict.items():
        if value > 85:
            listStudentCodes.append(key)

    final_list = []
    for val in listStudentCodes:
        comp_student_code = comp_new_df['Student_Code'] == val
        all_compliments = comp_new_df[comp_student_code]
        good_compliments = all_compliments[all_compliments['Sentiment'] > 0.5]
        list_compliments = good_compliments['Compliment'].unique().tolist()
        characteristic_list.extend([x for x in good_compliments['Characteristic'] if isinstance(x, str) == True])
        if len(list_compliments) > 2:
            final_string = '. '.join(list_compliments)
            final_list.extend(get_summary(final_string, 0.4, True))
        else:
            if len(list_compliments) != 0:
                final_list.extend([x.replace('.', '') for x in list_compliments])

    return set(final_list)


# print(get_compliment_for_highest_marks_students())

"""
Function to get Compliments of those students who gets more marks.
This function can be called for students who gets low marks so that they can adopt these characteristics to get good marks
"""
def get_characteristics_detail_for_highest_marks_students():
    list_of_chars_dict = [ast.literal_eval(x) for x in characteristic_list]
    chars_dict = {}
    for item in list_of_chars_dict:
        for key, val in item.items():
            rat = chars_dict.get(key, None)
            if rat is None:
                chars_dict[key] = val
            else:
                chars_dict[key] = (rat + val) / 2

    return chars_dict


def get_student_characteristics(student_code):
    characteristic_list_student = []
    characteristic_student_dict = {}
    comp_student_code = comp_new_df['Student_Code'] == student_code
    all_compliments = comp_new_df[comp_student_code]
    characteristic_list_student.extend([x for x in all_compliments['Characteristic'] if isinstance(x, str) == True])
    characteristic_list_student = [ast.literal_eval(x) for x in characteristic_list_student]
    for item in characteristic_list_student:
        for key, val in item.items():
            rat = characteristic_student_dict.get(key, None)
            if rat is None:
                characteristic_student_dict[key] = val
            else:
                characteristic_student_dict[key] = (rat + val) / 2
    return characteristic_student_dict


# print(get_student_characteristics('3037'))
# print(get_characteristics_detail_for_highest_marks_students())

""" 
Function to Get highly positive observation for those who got good marks 
"""
def get_highly_positive_observation_for_highest_mark_students():
    listStudentCodes = []
    if studentMarksDict is not None and any(studentMarksDict) == True:
        for key, value in studentMarksDict.items():
            if value > 85:
                listStudentCodes.append(key)

    final_list = []
    for val in listStudentCodes:
        obs_student_code = obs_df['Student_Code'] == val
        all_observations = obs_df[obs_student_code]
        good_observations = all_observations[all_observations['Sentiment'] > 0.7]
        list_observations = good_observations['Observation'].unique().tolist()
        if len(list_observations) > 2:
            final_string = '. '.join(list_observations)
            final_list.extend(get_summary(final_string, 0.4, True))
        else:
            if len(list_observations) != 0:
                final_list.extend([x.replace('.', '') for x in list_observations])

    return set(final_list)

	
print(get_highly_positive_observation_for_highest_mark_students())
