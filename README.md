# Введение: Проект машинного обучения, часть 1
В этой тетради мы рассмотрим решение комплексной задачи машинного обучения с использованием набора реальных данных. 

Используйте предоставленные данные по энергопотреблению здания для разработки модели, которая может предсказать оценку здания как Energy Star, а затем интерпретируйте результаты, чтобы найти переменные, которые в наибольшей степени влияют на оценку.

Это контролируемая задача регрессионного машинного обучения: учитывая набор данных с включенными целевыми показателями (в данном случае оценкой), мы хотим обучить модель, которая может научиться сопоставлять признаки (также известные как независимые переменные) с целью.

# Рабочий процесс машинного обучения
Хотя точные детали реализации могут варьироваться, общая структура проекта машинного обучения остается относительно неизменной:

Очистка и форматирование данных
Предварительный анализ данных
Разработка и выбор функций
Установите базовый уровень и сравните несколько моделей машинного обучения по показателям производительности
Выполните настройку гиперпараметров для наилучшей модели, чтобы оптимизировать ее для решения задачи
Оцените наилучшую модель на тестовом наборе
Интерпретируйте результаты моделирования, насколько это возможно
Сделайте выводы и напишите хорошо документированный отчет
Заблаговременная настройка структуры конвейера позволяет нам увидеть, как один шаг перетекает в другой. Однако конвейер машинного обучения - это итеративная процедура, и поэтому мы не всегда выполняем эти шаги линейно. Мы можем вернуться к предыдущему этапу, основываясь на результатах дальнейшей разработки. Например, хотя мы можем выполнить выбор объектов перед построением каких-либо моделей, мы можем использовать результаты моделирования, чтобы вернуться назад и выбрать другой набор объектов. Или же моделирование может привести к неожиданным результатам, которые означают, что мы хотим изучить наши данные с другой стороны. Как правило, вы должны выполнить один шаг, прежде чем переходить к следующему, но не думайте, что, выполнив один шаг в первый раз, вы не сможете вернуться назад и внести улучшения!

В этой тетради будут рассмотрены первые три этапа разработки, а остальные части будут рассмотрены в двух дополнительных тетрадях. Цель этой серии - показать, как все различные методы работы с данными объединяются в целостный проект. Я стараюсь больше концентрироваться на реализации методов, а не объяснять их на низком уровне, но предоставляю ресурсы для тех, кто хочет углубиться в изучение. Чтобы ознакомиться с лучшей книгой (на мой взгляд) по изучению основ и внедрению методов машинного обучения на Python, ознакомьтесь с практическим руководством по машинному обучению с помощью Scikit-Learn и Tensorflow Аурелиона Герона.

# Импорт
Мы будем использовать стандартные библиотеки для обработки данных и машинного обучения: numpy, pandas и scikit-learn. Для визуализации мы также используем matplotlib и seaborn.

```python
# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)

# Matplotlib visualization
import matplotlib.pyplot as plt
%matplotlib inline

# Set default font size
plt.rcParams['font.size'] = 24

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
```


# Очистка и форматирование данных
Загрузите данные и проверьте их
Мы будем загружать наши данные в pandas dataframe, одну из самых полезных структур данных для data science. Представьте это как электронную таблицу на Python, которой мы можем легко манипулировать, очищать и визуализировать. У Pandas есть множество методов, которые помогают сделать процесс обработки данных и машинного обучения максимально плавным.

```python
# Read in data into a dataframe 
data = pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')

# Display top of dataframe
data.head()
```

Взглянув на исходные данные, мы уже можем увидеть ряд проблем, которые нам предстоит решить. Во-первых, здесь 60 столбцов, и мы не знаем, что означают многие из них! Все, что мы знаем из формулировки задачи, - это то, что мы хотим предсказать число в столбце оценки. Некоторые определения других столбцов можно легко угадать, но другие трудно понять (я занимаюсь исследованиями в области энергетики зданий и до сих пор не смог разобраться в каждом столбце). В машинном обучении это на самом деле не проблема, потому что мы позволяем модели решать, какие функции важны. Иногда нам могут даже не указывать названия столбцов и не сообщать, что мы прогнозируем. Тем не менее, мне нравится понимать проблему настолько, насколько это возможно, и поскольку мы также хотим интерпретировать результаты моделирования, было бы неплохо иметь некоторые знания о столбцах.

Вместо того чтобы связаться с компанией, которая дала мне задание, чтобы выяснить, знают ли они определение столбцов, я решил попытаться найти их самостоятельно. Посмотрев на название файла, "Раскрытие данных об энергетике и водных ресурсах для местного законодательства"84"за 2017 год "Данные за календарный год"2016".csv, я поискал "Местное законодательство 84". Это приводит к этой веб-странице, на которой говорится сша, что местный закон № 84 является требованием Нью-Йорка, согласно которому все здания, превышающие 
 мы должны ежегодно публиковать определенный набор показателей, связанных с энергетикой. После небольшого поиска мы можем перейти к этому pdf-документу, в котором подробно описывается значение каждого столбца.

Хотя нам и не нужно изучать каждый столбец, было бы неплохо, по крайней мере, понять цель, которую мы хотим предсказать. Вот определение целевой оценки:

### Процентильное ранжирование от 1 до 100 для определенных типов зданий, рассчитанное в программе Portfolio Manager на основе собственных данных об энергопотреблении за отчетный год.
Это кажется довольно простым: рейтинг Energy Star - это метод ранжирования зданий с точки зрения энергоэффективности, при котором 1 - худший показатель, а 100 - лучший. Это относительный процентильный рейтинг, который означает, что здания оцениваются относительно друг друга и должны иметь равномерное распределение по всему диапазону значений.

# Типы данных и пропущенные значения
Метод dataframe.info - это быстрый способ оценки данных путем отображения типов данных в каждом столбце и количества не пропущенных значений. Уже при просмотре фрейма данных может возникнуть проблема, поскольку отсутствующие значения кодируются как "Недоступные", а не как np.nan (не число). Это означает, что столбцы с числами не будут представлены в числовом виде, поскольку pandas преобразует столбцы с любыми строковыми значениями в столбцы со всеми строками.
```python
# See the column data types and non-missing values
data.info()
```

Конечно же, есть несколько столбцов с числами, которые были записаны как объектные типы данных. Их необходимо преобразовать в тип данных float, прежде чем мы сможем выполнить какой-либо числовой анализ.

# Преобразовать данные в правильные типы
Мы преобразуем столбцы с числами в числовые типы данных, заменив записи "Недоступно" на np.nan, которые можно интерпретировать как числа с плавающей точкой. Затем мы преобразуем столбцы, содержащие числовые значения (такие как квадратные метры или потребление энергии), в числовые типы данных.

# Replace all occurrences of Not Available with numpy not a number
data = data.replace({'Not Available': np.nan})
```python
# Iterate through the columns
for col in list(data.columns):
    # Select columns that should be numeric
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        # Convert the data type to float
        data[col] = data[col].astype(float)
```
```python
# Statistics for each column
data.describe()
```

# Пропущенные значения
Теперь, когда у нас есть правильные типы данных в столбцах, мы можем начать анализ, посмотрев на процент пропущенных значений в каждом столбце. Пропущенные значения можно использовать при предварительном анализе данных, но их необходимо будет заполнить для методов машинного обучения.

Ниже приведена функция, которая вычисляет количество пропущенных значений и процент от общего числа пропущенных значений для каждого столбца. Как и во многих задачах в области науки о данных.
```python
# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```
```python
missing_values_table(data)
```

Хотя мы хотим быть осторожными, чтобы не отбрасывать информацию, и должны быть осторожны при удалении столбцов, если в столбце высокий процент пропущенных значений, то от него, вероятно, будет мало пользы.

Выбор столбцов для сохранения может быть немного произвольным (вот обсуждение), но для этого проекта мы удалим все столбцы, в которых пропущено более 50% значений. В общем, будьте осторожны с удалением какой-либо информации, потому что даже если ее нет во всех наблюдениях, она все равно может быть полезна для прогнозирования целевого значения.
```python
# Get the columns with > 50% missing
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))
```
Выбранный вами фрейм данных содержит 60 столбцов.
В 46 столбцах отсутствуют значения.
Мы удалим 11 столбцов.
```python
# Drop the columns
data = data.drop(columns = list(missing_columns))

# For older versions of pandas (https://github.com/pandas-dev/pandas/issues/19078)
# data = data.drop(list(missing_columns), axis = 1)
```

Остальные пропущенные значения должны быть вычислены (заполнены) с использованием соответствующей стратегии, прежде чем приступать к машинному обучению.

# Предварительный анализ данных
Исследовательский анализ данных (EDA) - это непрерывный процесс, в ходе которого мы строим графики и рассчитываем статистические данные для изучения наших данных. Цель состоит в том, чтобы найти аномалии, закономерности, тенденции или взаимосвязи. Они могут быть интересны сами по себе (например, поиск корреляции между двумя переменными) или могут быть использованы для принятия обоснованных решений при моделировании, например, о том, какие функции использовать. Короче говоря, цель EDA - определить, о чем могут рассказать нам наши данные! Как правило, день начинается с общего обзора, а затем мы переходим к конкретным частям набора данных по мере того, как находим интересные области для изучения.

Для начала EDA мы сосредоточимся на одной переменной - показателе Energy Star, поскольку это цель наших моделей машинного обучения. Для простоты мы можем переименовать столбец в score, а затем начать изучать это значение.

# Графики с одной переменной
График с одной переменной (называемый одномерным) показывает распределение одной переменной, например, в виде гистограммы.
```python
figsize(8, 8)

# Rename the score 
data = data.rename(columns = {'ENERGY STAR Score': 'score'})

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 
plt.title('Energy Star Score Distribution');
```
```python
# Histogram Plot of Site EUI
figsize(8, 8)
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Site EUI'); 
plt.ylabel('Count'); plt.title('Site EUI Distribution');
```
```python
data['Site EUI (kBtu/ft²)'].describe()

data['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10)

data.loc[data['Site EUI (kBtu/ft²)'] == 869265, :]

```

# Удаление отклонений
Когда мы удаляем отклонения, мы должны быть осторожны, чтобы не отбросить результаты измерений только потому, что они выглядят странно. Они могут быть результатом реального явления, которое нам следует дополнительно изучить. При удалении выбросов я стараюсь быть как можно более консервативным, используя определение экстремального выброса:

На нижнем уровне экстремальный выброс находится ниже 
На верхнем уровне экстремальный выброс находится выше 
В этом случае я удалю только одну внешнюю точку и посмотрю, как выглядит распределение.
```python
# Calculate first and third quartile
first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
            (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]
```
```python
# Histogram Plot of Site EUI
figsize(8, 8)
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Site EUI'); 
plt.ylabel('Count'); plt.title('Site EUI Distribution');
```

# Ищем взаимосвязи
Чтобы посмотреть на влияние категориальных переменных на оценку, мы можем построить график плотности, окрашенный в соответствии со значением категориальной переменной. Графики плотности также показывают распределение одной переменной и могут быть представлены в виде сглаженной гистограммы. Если мы раскрасим кривые плотности категориальной переменной, это покажет нам, как меняется распределение в зависимости от класса.

Первый график, который мы построим, показывает распределение оценок по типам недвижимости. Чтобы не загромождать график, мы ограничим его типами зданий, для которых в наборе данных содержится более 100 наблюдений.

```python
# Create a list of buildings with more than 100 measurements
types = data.dropna(subset=['score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)
```
```python
# Plot of distribution of scores for building categories
figsize(12, 10)

# Plot each building
for b_type in types:
    # Select the building type
    subset = data[data['Largest Property Use Type'] == b_type]
    
    # Density plot of Energy Star scores
    sns.kdeplot(subset['score'].dropna(),
               label = b_type, shade = False, alpha = 0.8);
    
# label the plot
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Energy Star Scores by Building Type', size = 28);
```
```python
# Create a list of boroughs with more than 100 observations
boroughs = data.dropna(subset=['score'])
boroughs = boroughs['Borough'].value_counts()
boroughs = list(boroughs[boroughs.values > 100].index)
```
```python
# Plot of distribution of scores for boroughs
figsize(12, 10)

# Plot each borough distribution of scores
for borough in boroughs:
    # Select the building type
    subset = data[data['Borough'] == borough]
    
    # Density plot of Energy Star scores
    sns.kdeplot(subset['score'].dropna(),
               label = borough);
    
# label the plot
plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of Energy Star Scores by Borough', size = 28);
```

# Корреляции между признаками и целью
Чтобы количественно оценить корреляции между признаками (переменными) и целью, мы можем рассчитать коэффициент корреляции Пирсона. Это показатель силы и направления линейной зависимости между двумя переменными: значение -1 означает, что две переменные линейно отрицательно коррелируют, а значение +1 означает, что две переменные линейно положительно коррелируют. На рисунке ниже показаны различные значения коэффициента корреляции и то, как они отображаются графически.



Хотя между объектами и целевыми объектами могут существовать нелинейные зависимости, а коэффициенты корреляции не учитывают взаимодействия между объектами, линейные зависимости - это хороший способ начать изучение тенденций в данных. Затем мы можем использовать эти значения для выбора объектов для использования в нашей модели.

Приведенный ниже код вычисляет коэффициенты корреляции между всеми переменными и оценкой.
```python
# Find all correlations and sort 
correlations_data = data.corr()['score'].sort_values()

# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))
```
```python
# Select the numeric columns
numeric_subset = data.select_dtypes('number')

# Create columns with square root and log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'score':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Select the categorical columns
categorical_subset = data[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Drop buildings without an energy star score
features = features.dropna(subset = ['score'])

# Find correlations with the score 
correlations = features.corr()['score'].dropna().sort_values()
```

```python
# Display most negative correlations
correlations.head(15)
```
```python
# Display most positive correlations
correlations.tail(15)
```

# Графики с двумя переменными
Чтобы наглядно представить взаимосвязь между двумя переменными, мы используем точечную диаграмму. Мы также можем включить дополнительные переменные, используя такие аспекты, как цвет или размер маркеров. Здесь мы сопоставим две числовые переменные друг с другом и используем цвет для представления третьей категориальной переменной.
```python
figsize(12, 10)

# Extract the building types
features['Largest Property Use Type'] = data.dropna(subset = ['score'])['Largest Property Use Type']

# Limit to building types with more than 100 observations (from previous code)
features = features[features['Largest Property Use Type'].isin(types)]

# Use seaborn to plot a scatterplot of Score vs Log Source EUI
sns.lmplot('Site EUI (kBtu/ft²)', 'score', 
          hue = 'Largest Property Use Type', data = features,
          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,
          size = 12, aspect = 1.2);

# Plot labeling
plt.xlabel("Site EUI", size = 28)
plt.ylabel('Energy Star Score', size = 28)
plt.title('Energy Star Score vs Site EUI', size = 36);
```
# Построение парного графика
В качестве заключительного упражнения для предварительного анализа данных мы можем построить парный график между несколькими различными переменными. Построение парного графика - отличный способ изучить множество переменных одновременно, поскольку он показывает диаграммы рассеяния между парами переменных и гистограммы отдельных переменных по диагонали.

Используя функцию seaborn Pair Grid, мы можем сопоставить различные графики с тремя аспектами сетки. В верхнем треугольнике будут показаны диаграммы рассеяния, на диагонали - гистограммы, а в нижнем треугольнике будут показаны как коэффициент корреляции между двумя переменными, так и двумерная оценка плотности ядра этих двух переменных.
```python
# Extract the columns to  plot
plot_data = features[['score', 'Site EUI (kBtu/ft²)', 
                      'Weather Normalized Source EUI (kBtu/ft²)', 
                      'log_Total GHG Emissions (Metric Tons CO2e)']]

# Replace the inf with nan
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})

# Rename columns 
plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 
                                        'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
                                        'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})

# Drop na values
plot_data = plot_data.dropna()

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3)

# Upper is a scatter plot
grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

# Diagonal is a histogram
grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

# Bottom is correlation and density plot
grid.map_lower(corr_func);
grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)

# Title for entire plot
plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02);
```

# Разработка и выбор функциональных возможностей
Теперь, когда мы изучили тенденции и взаимосвязи в данных, мы можем приступить к разработке набора функциональных возможностей для наших моделей. Мы можем использовать результаты EDA для разработки функциональных возможностей. В частности, от EDA мы узнали следующее, что может помочь нам в проектировании /выборе объектов:

Распределение баллов зависит от типа здания и, в меньшей степени, от района. Хотя мы сосредоточимся на числовых характеристиках, мы также должны включить эти две категориальные характеристики в модель.
Логарифмическое преобразование объектов не приводит к значительному увеличению линейных корреляций между объектами и оценкой
Прежде чем мы продолжим, нам следует определить, что такое разработка и отбор объектов! Эти определения являются неформальными и в значительной степени совпадают, но мне нравится думать о них как о двух отдельных процессах:

Разработка функций: процесс сбора необработанных данных и извлечения или создания новых функций, которые позволяют модели машинного обучения изучать соответствие между этими функциями и целевым объектом. Это может означать преобразование переменных, например, как мы сделали с логарифмом и квадратным корнем, или однократное кодирование категориальных переменных, чтобы их можно было использовать в модели. В целом, я рассматриваю разработку функций как добавление дополнительных функций, полученных на основе необработанных данных.
Выбор признаков: процесс выбора наиболее релевантных признаков в ваших данных. "Наиболее релевантный" может зависеть от многих факторов, но это может быть что-то простое, например, наивысшая корреляция с целевым объектом или признаки с наибольшей дисперсией. При выборе объектов мы удаляем объекты, которые не помогают нашей модели изучить взаимосвязь между объектами и целью. Это может помочь модели лучше адаптироваться к новым данным и в результате получить более интерпретируемую модель. Как правило, я рассматриваю выбор объектов как вычитание объектов, поэтому у нас остаются только те, которые являются наиболее важными.
Разработка и выбор объектов - это итеративные процессы, которые обычно требуют нескольких попыток для получения правильного результата. Часто мы используем результаты моделирования, такие как значения объектов из случайного леса, чтобы вернуться назад и повторить выбор объектов, или мы можем позже обнаружить взаимосвязи, которые требуют создания новых переменных. Более того, эти процессы обычно включают в себя сочетание знаний предметной области и статистического качества данных.

Разработка и выбор функциональных возможностей часто дает наибольшую отдачу от времени, затраченного на решение задачи машинного обучения. Это может занять довольно много времени, но зачастую важнее, чем точный алгоритм и гиперпараметры, используемые для модели. Если мы не вводим в модель правильные данные, то мы настраиваем ее на сбой, и мы не должны ожидать, что она научится!

В этом проекте мы выполним следующие шаги для разработки функций:

Выберите только числовые переменные и две категориальные переменные (район и тип использования объекта недвижимости).
Добавьте в журнал преобразование числовых переменных
Однократное кодирование категориальных переменных
Для выбора объектов мы сделаем следующее:



Удалим коллинеарные объекты
Мы обсудим коллинеарность (также называемую мультиколлинеарностью), когда приступим к этому процессу!

Следующий код выбирает числовые объекты, добавляет в журнал преобразований все числовые объекты, выбирает и однократно кодирует категориальные объекты и объединяет наборы объектов вместе.
```python
# Copy the original data
features = data.copy()

# Select the numeric columns
numeric_subset = data.select_dtypes('number')

# Create columns with log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'score':
        next
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])
        
# Select the categorical columns
categorical_subset = data[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

features.shape
```
# Удалите коллинеарные характеристики
Сильно коллинеарные характеристики имеют значительный коэффициент корреляции между собой. Например, в нашем наборе данных EUI для сайта и EUI для метеорологической нормы сильно коррелируют, поскольку они лишь немного отличаются друг от друга при расчете интенсивности энергопотребления.
```python
plot_data = data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()

plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
plt.xlabel('Site EUI'); plt.ylabel('Weather Norm EUI')
plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' % np.corrcoef(data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna(), rowvar=False)[0][1]);
```
```python
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between Energy Star Score
    y = x['score']
    x = x.drop(columns = ['score'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    x = x.drop(columns = ['Weather Normalized Site EUI (kBtu/ft²)', 
                          'Water Use (All Water Sources) (kgal)',
                          'log_Water Use (All Water Sources) (kgal)',
                          'Largest Property Use Type - Gross Floor Area (ft²)'])
    
    # Add the score back in to the data
    x['score'] = y
               
    return x
```
```python
# Remove the collinear features above a specified correlation coefficient
features = remove_collinear_features(features, 0.6);
```
```python
# Remove any columns with all na values
features  = features.dropna(axis=1, how = 'all')
features.shape
```

# Разделить на обучающий и тестовый наборы
В машинном обучении нам всегда нужно разделять наши функции на два набора:

Обучающий набор, который мы предоставляем нашей модели во время обучения вместе с ответами, чтобы она могла изучить соответствие между функциями и целью.
Тестовый набор, который мы используем для оценки сопоставления, полученного моделью. Модель никогда не видела ответов в тестовом наборе, но вместо этого должна делать прогнозы, используя только функции. Поскольку мы знаем правильные ответы для набора тестов, мы можем затем сравнить результаты тестирования с истинными целями тестирования, чтобы получить оценку того, насколько хорошо наша модель будет работать при развертывании в реальном мире.
Для решения нашей задачи мы сначала выделим все здания без оценки Energy Star (мы не знаем правильного ответа для этих зданий, поэтому они не будут полезны для обучения или тестирования). Затем мы разделим здания, получившие оценку Energy Star, на тестовый набор, состоящий из 30% зданий, и обучающий набор, состоящий из 70% зданий.

Разделить данные на случайный набор для обучения и тестирования с помощью scikit-learn просто. Мы можем задать случайное состояние разделения, чтобы обеспечить стабильные результаты.
```python
# Extract the buildings with no score and the buildings with a score
no_score = features[features['score'].isna()]
score = features[features['score'].notnull()]

print(no_score.shape)
print(score.shape)
```
```python
# Separate out the features and targets
features = score.drop(columns='score')
targets = pd.DataFrame(score['score'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)
```
# Показатель: Средняя абсолютная погрешность
В задачах машинного обучения используется множество показателей, и иногда бывает трудно понять, какой из них выбрать. В большинстве случаев это будет зависеть от конкретной задачи и от того, есть ли у вас конкретная цель, для достижения которой требуется оптимизация. Мне нравится совет Эндрю Нга использовать единую реальную метрику производительности для сравнения моделей, потому что это упрощает процесс оценки. Вместо того, чтобы вычислять множество показателей и пытаться определить, насколько важен каждый из них, нам следует использовать одно число. В этом случае, поскольку мы проводим регрессию, средняя абсолютная ошибка является подходящим показателем. Это также можно интерпретировать, поскольку оно представляет собой среднюю величину, на которую мы рассчитываем, если отклониться, в тех же единицах, что и целевое значение.

Приведенная ниже функция вычисляет среднюю абсолютную погрешность между истинными значениями и прогнозируемыми.

```python
# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
```
```python
baseline_guess = np.median(y)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
```
# Выводы
В этом блокноте мы выполнили первые три этапа решения задачи машинного обучения:

Очистили и отформатировали необработанные данные
Провели предварительный анализ данных
Разработали набор функций для обучения нашей модели с использованием функциональной инженерии и выбора функций
Мы также выполнили важнейшую задачу по созданию базовой метрики, чтобы определить, лучше ли наша модель, чем предположение!

Надеемся, вы начинаете понимать, как каждая часть конвейера перетекает в следующую: очистка данных и приведение их в надлежащий формат позволяет нам проводить предварительный анализ данных. Затем EDA информирует нас о наших решениях на этапе проектирования и выбора функций. Эти три этапа обычно выполняются в таком порядке, хотя мы можем вернуться к ним позже и провести дополнительные работы по EDA или функциональному проектированию на основе результатов нашего моделирования. Обработка данных - это итеративный процесс, в ходе которого мы всегда ищем способы улучшить нашу предыдущую работу. Это означает, что нам не обязательно добиваться совершенства с первого раза (хотя мы можем стараться изо всех сил), потому что почти всегда есть возможность пересмотреть наши решения, как только мы узнаем больше о проблеме.

Во второй части мы сосредоточимся на внедрении нескольких методов машинного обучения, выборе наилучшей модели и ее оптимизации для нашей задачи с помощью настройки гиперпараметров с перекрестной проверкой. В качестве последнего шага мы сохраним наборы данных, которые мы разработали, чтобы использовать их в следующей части.
```python
# Save the no scores, training, and testing data
no_score.to_csv('data/no_score.csv', index = False)
X.to_csv('data/training_features.csv', index = False)
X_test.to_csv('data/testing_features.csv', index = False)
y.to_csv('data/training_labels.csv', index = False)
y_test.to_csv('data/testing_labels.csv', index = False)
```


# Код для БД:

Это нормыльный код, правильный 
```sql
-- Создание базы данных BD
CREATE DATABASE BD;
GO

-- Создание таблицы Users для хранения пользователей и паролей
USE BD;
CREATE TABLE Users (
    UserID INT PRIMARY KEY IDENTITY(1,1),
    UserName NVARCHAR(50),
    PasswordHash VARBINARY(MAX)  -- Хранение зашифрованных паролей
);
GO

-- Создание ключа и сертификата для шифрования
CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'YourStrongPassword123';
CREATE CERTIFICATE CertBD WITH SUBJECT = 'Certificate for BD encryption';
CREATE SYMMETRIC KEY SymmetricKeyBD WITH ALGORITHM = AES_256 ENCRYPTION BY CERTIFICATE CertBD;
GO

DECLARE @counter INT = 1;
DECLARE @username NVARCHAR(50);
DECLARE @password NVARCHAR(50);
DECLARE @i INT;
DECLARE @index INT;
DECLARE @chars NVARCHAR(200) = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
DECLARE @dbname NVARCHAR(50);
DECLARE @cmd NVARCHAR(200);

-- Цикл для создания 10 пользователей и баз данных
USE master;
WHILE @counter <= 10
BEGIN
    SET @username = 'user' + CAST(@counter AS NVARCHAR(10));
    SET @password = '';
    SET @i = 1;
    WHILE @i <= 5
    BEGIN
        SET @index = ABS(CHECKSUM(NEWID())) % LEN(@chars) + 1;
        SET @password = @password + SUBSTRING(@chars, @index, 1);
        SET @i = @i + 1;
    END
    SET @dbname = 'BD' + CAST(@counter AS NVARCHAR(10));
    SET @cmd = 'CREATE DATABASE ' + @dbname + ';';
    EXEC sp_executesql @cmd;
    SET @cmd = 'CREATE LOGIN ' + @username + ' WITH PASSWORD = ''' + @password + ''';';
    EXEC sp_executesql @cmd;
    SET @cmd = 'USE ' + @dbname + '; CREATE USER ' + @username + ' FOR LOGIN ' + @username + ';';
    EXEC sp_executesql @cmd;
    SET @cmd = 'USE ' + @dbname + '; ALTER ROLE db_owner ADD MEMBER ' + @username + ';';
    EXEC sp_executesql @cmd;
    USE BD;
    OPEN SYMMETRIC KEY SymmetricKeyBD DECRYPTION BY CERTIFICATE CertBD;
    INSERT INTO Users(UserName, PasswordHash)
    VALUES (@username, ENCRYPTBYKEY(KEY_GUID('SymmetricKeyBD'), @password));
    CLOSE SYMMETRIC KEY SymmetricKeyBD;
    USE master;
    SET @counter = @counter + 1;
END

-- Скрипт для расшифровки пароля
USE BD;
GO
-- Открытие симметричного ключа для расшифровки
OPEN SYMMETRIC KEY SymmetricKeyBD DECRYPTION BY CERTIFICATE CertBD;
GO
SELECT 
    UserName, 
    CAST(DECRYPTBYKEY(PasswordHash) AS NVARCHAR(50)) AS DecryptedPassword
FROM Users;
GO
-- Закрытие симметричного ключа после расшифровки
CLOSE SYMMETRIC KEY SymmetricKeyBD;
GO
```
```sql
--------------этот для меня на всякий случай
DECLARE @username nVARCHAR(50)
DECLARE @password nVARCHAR(50)
DECLARE @counter INT = 1
DECLARE @cmd NVARCHAR(200);
DECLARE @i INT;
DECLARE @index INT;
DECLARE @dbname NVARCHAR(50);

DECLARE @chars nVARCHAR(200) = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
WHILE @counter <= 10
BEGIN
    SET @username = 'user' + CAST(@counter AS nVARCHAR(10))
    SET @password = '';
	set @i = 1
	WHILE @i <= 5
	BEGIN
		SET @index = ABS(CHECKSUM(NEWID())) % LEN(@chars) + 1;
		SET @password = @password + SUBSTRING(@chars, @index, 1)
		SET @i = @i + 1
	ENd
	SET @dbname = 'BD' + CAST(@counter AS NVARCHAR(10));
	EXEC('create database ' + @dbname + ';');
	EXEC('CREATE LOGIN ' + @username + ' with password = '''+ @password + ''';');
	EXEC('USE ' + @dbName + '; CREATE USER ' + @userName + ' FOR LOGIN ' + @userName);
    EXEC('USE ' + @dbName + '; ALTER ROLE db_owner ADD MEMBER ' + @userName);
	use BD;
	insert into Users(name,pass) values (@username,@password)
	use master;
	SET @counter = @counter + 1
END
```
```sql
-- Шифрование всех паролей в таблице Users
CREATE MASTER KEY ENCRYPTION BY PASSWORD = '$tr0nGPa$$w0rd'
OPEN SYMMETRIC KEY MySymmetricKey
DECRYPTION BY ASYMMETRIC KEY MyAsymmetricKey
WITH PASSWORD = 'StrongPa$$w0rd!'
--сертификат для ключа
CREATE CERTIFICATE HumanResources037  
   WITH SUBJECT = 'Employee Social Security Numbers';  
GO  

CREATE SYMMETRIC KEY MySymmetricKey  
    WITH ALGORITHM = AES_256  
    ENCRYPTION BY CERTIFICATE HumanResources037;  
GO  
OPEN SYMMETRIC KEY MySymmetricKey  
   DECRYPTION BY CERTIFICATE HumanResources037;  

UPDATE Users
SET encrpass = EncryptByKey(Key_GUID('MySymmetricKey'), pass);  

select * from Users
--представления для проверки
select * from sys.symmetric_keys;
select * from sys.asymmetric_keys
SELECT * FROM [sys].[openkeys]

--дешифровка
OPEN SYMMETRIC KEY SSN_Key_01  
   DECRYPTION BY CERTIFICATE HumanResources037;  

   SELECT name, pass,  
    CONVERT(nvarchar, DecryptByKey(encrpass))   
    AS 'Decrypted ID Number'  
    FROM Users;  
```
```sql
-- Резервное копирование базы данных BD
BACKUP DATABASE BD TO DISK = 'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\Backup\BD.bak';
GO

-- Процедура восстановления базы данных
USE master;
GO
ALTER DATABASE BD SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
RESTORE DATABASE BD FROM DISK = 'C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\Backup\BD.bak' WITH REPLACE;
GO
ALTER DATABASE BD SET MULTI_USER; 


---------процедура проверки почты
CREATE PROCEDURE CheckEmailValidity
AS
BEGIN
    SELECT 
        email,
        CASE 
            WHEN Email LIKE '%[^A-Za-z0-9@.]%' OR Email LIKE '%[[]"<>'']%' THEN 0
            WHEN Email LIKE '[A-Za-z0-9]%@[A-Za-z0-9]%.%' THEN 1
            ELSE 0
        END AS Validity
    FROM Users;
END

EXEC CheckEmailValidity;

-------таблица для хранения изменений тригера
CREATE TABLE HistoryCost (
    ChangeDate DATETIME,
    ProductName NVARCHAR(100),
    OldPrice DECIMAL(18, 2),
    NewPrice DECIMAL(18, 2)
);

--------тригер
alter TRIGGER trg_PriceChange
ON Item
AFTER UPDATE
AS
BEGIN
    DECLARE @ProductName NVARCHAR(100);
    DECLARE @OldPrice DECIMAL(18, 2);
    DECLARE @NewPrice DECIMAL(18, 2);

    SELECT @ProductName = ItemName, @OldPrice = Price FROM deleted;
    SELECT @NewPrice = Price FROM inserted;

    IF @OldPrice != @NewPrice
    BEGIN
        INSERT INTO HistoryCost (ChangeDate, ProductName, OldPrice, NewPrice)
        VALUES (GETDATE(), @ProductName, @OldPrice, @NewPrice);
    END
END
```
```sql
select * from HistoryCost



select * from Преподаватель order by Фамилия 
select * from Преподаватель order by Фамилия desc

select * from Расписание order by Время_отправления
select * from Расписание order by Время_отправления desc

select * from Заказ order by Статус_заказа
select * from Заказ order by Статус_заказа desc

select * from Clients where name like '% %';
```


```C#
static string connectionString = "Server=DESKTOP-2LGOQKK;initial catalog=Administr;integrated security=True;MultipleActiveResultSets=True;TrustServerCertificate=true";
        SqlConnection connection = new SqlConnection(connectionString);

		connection.Open();
SqlCommand command = "Select ....";
//command.ExecuteNonQuery();
SqlDataReader r = cmd.ExecuteReader();
var a = r.ToList();
r.Close();
ИЛИ
DataTable dt_Offices = new DataTable();
            List offices = new List() { FirstItem };

            SqlCommand sqlCommand = sqlConnection.CreateCommand();
            sqlCommand.CommandText = "select distinct User_Title from User_Info";

            SqlDataAdapter dataAdapter = new SqlDataAdapter(sqlCommand);
            dataAdapter.Fill(dt_Offices);

/////для AppData
public static class AppData
    {
        public static AdministrEntities db = new AdministrEntities();
    }


	/////DataGrid
	DataGrid.ItemsSource = AppData.db.Sz.Where(x => x.Is_Complete == false).ToList();
            connection.Close();

```
