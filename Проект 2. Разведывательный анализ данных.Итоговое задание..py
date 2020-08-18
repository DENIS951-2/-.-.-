#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

pd.set_option('display.max_rows', 50)  # показывать больше строк
pd.set_option('display.max_columns', 50)  # показывать больше колонок

stud_math = pd.read_csv('stud_math.csv')


# In[2]:


display(stud_math.head(10))
stud_math.info() 


# In[3]:


stud_math = pd.read_csv('stud_math.csv')
stud_math.columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                     'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'studytime_granular',
                     'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'score']
columns = stud_math.columns
stud_math


# In[4]:


# т. к. нужно узнать что повлияло на итоговые оценки будем исходить из переменной score


# In[5]:


pd.DataFrame(stud_math.score.value_counts())# Посмотрим, сколько оценок содержит наш датасет


# In[6]:


display(pd.DataFrame(stud_math.score.value_counts()))
print("Значений, встретившихся в столбце более 5 раз:"
      , (stud_math.score.value_counts()>5).sum())
stud_math.loc[:, ['score']].info()


# In[7]:


pd.DataFrame(stud_math.absences.value_counts())


# In[8]:


median = stud_math.absences.median()
IQR_2 = stud_math.absences.quantile(0.75) - stud_math.absences.quantile(0.25)
perc25 = stud_math.absences.quantile(0.25)
perc75 = stud_math.absences.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR_2), "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR_2, l=perc75 + 1.5*IQR_2))
stud_math.absences.loc[stud_math.absences.between(perc25 - 1.5*IQR_2, perc75 + 1.5*IQR_2)].hist(bins=16, range=(0, 100),
                                                                                  label='IQR')
stud_math.absences.loc[stud_math.score <= 100].hist(alpha=0.5, bins=16, range=(
    0, 100),
                                            label='оценки')
plt.legend()


# In[9]:


stud_math = stud_math.loc[stud_math.absences.between(perc25 - 1.5*IQR_2, perc75 + 1.5*IQR_2)]


# In[10]:


median = stud_math.score.median()
IQR = stud_math.score.quantile(0.75) - stud_math.score.quantile(0.25)
perc25 = stud_math.score.quantile(0.25)
perc75 = stud_math.score.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75),
      "IQR: {}, ".format(IQR), "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
stud_math.score.loc[stud_math.score.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins=16, range=(0, 100),
                                                                                  label='IQR')
stud_math.score.loc[stud_math.score <= 100].hist(alpha=0.5, bins=16, range=(
    0, 100),
                                            label='оценки')
plt.legend()


# In[11]:


# Этот способ позволил нам отобрать экстремально низкие и экстремально высокие оценки
stud_math = stud_math.loc[stud_math.score.between(
    perc25 - 1.5*IQR, perc75 + 1.5*IQR)]


# In[12]:


stud_math.age.hist()
stud_math.age.describe()


# In[13]:


stud_math.corr()


# In[14]:


sns.pairplot(stud_math, kind = 'reg')


# In[15]:


#Корреляция Medu == Fedu == score убираем Fedu и Medu, studytime_granular == freetime убираем freetime, goout == absences убираем absences


# In[16]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='age', 
                data=stud_math.loc[stud_math.loc[:, column].isin(stud_math.loc[:, column].value_counts().index[:10])],
               ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[17]:


for col in ['traveltime', 'studytime', 'failures', 'studytime_granular', 'famrel', 'goout', 'health', 'score']:
    get_boxplot(col)


# In[18]:


pd.DataFrame(stud_math.school.value_counts())


# In[19]:


pd.DataFrame(stud_math.sex.value_counts())


# In[20]:


pd.DataFrame(stud_math.address.value_counts())


# In[21]:


pd.DataFrame(stud_math.famsize.value_counts())


# In[22]:


pd.DataFrame(stud_math.Pstatus.value_counts())


# In[23]:


def get_stat_dif(column):
    cols = stud_math.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(stud_math.loc[stud_math.loc[:, column] == comb[0], 'score'], 
                        stud_math.loc[stud_math.loc[:, column] == comb[1], 'score']).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[25]:


for col in ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                     'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'studytime_granular',
                     'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'score']:
    get_stat_dif(col)


# In[28]:


stud_math_for_model = stud_math.loc[:, ['school', 'age', 'famsize', 'Pstatus', 'Fedu', 'Fjob', 'reason', 'guardian',
                     'traveltime', 'studytime', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'studytime_granular',
                     'internet', 'famrel', 'freetime', 'goout', 'health']]
stud_math_for_model.head()


# In[27]:


# В данных достаточно мало пустых значений.
# Выбросы найдены только в столбце пропусков занятий, что позволяет сделать вывод о том, что данные достаточно чистые.
# Корреляция параметра Medu и Fedu может говорить о том, что образование родителей влияет на образование детей,
# studytime_granular и freetime некоторые ученики в после школы дополнительно занимаются,
# а вот проведение больше времени с друзьями плохо влияет на посещаемость и результат.
# Самые важные параметры, которые предлагается использовать в дальнейшем для построения модели, это 'school', 'age', 'famsize', 'Pstatus', 'Fedu', 'Fjob', 'reason', 'guardian',
                     'traveltime', 'studytime', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'studytime_granular',
                     'internet', 'famrel', 'freetime', 'goout', 'health'.


# In[ ]:




