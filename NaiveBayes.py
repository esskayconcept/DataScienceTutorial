# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 08:42:57 2019

Just a conceptual study of Naive Bayes Classifier.

The Classifier is written from Scratch 
It is created for Categorical Fields only

It would be modified to accommodate the Numerical fields as well

@author: Somnath
"""
#Libraries
import pandas # For Dataframe
import numpy  # For calculations

#Utility functions
def Underline():
    UnderLineMarginLength = 100
    UnderLineMarginChar = "="
    UnderLineMargin = UnderLineMarginChar * UnderLineMarginLength
    
    print(UnderLineMargin)

def ShowHeader(Caption):
    print("\n")
    Underline()
    print(Caption)
    Underline()
    
def ExploreDataColumn(Data, columnIndex):
    colname = Data.columns[columnIndex]
    ShowHeader("Data status for column : " + colname)
    
    #Get the Data for the colum
    colvalues = Data[colname]
    
    #Check the number of missing Data
    nancount = 100 * colvalues.isnull().sum() / len(colvalues)
    print("% of missing Data : " + str(nancount))
    
    #First check if this numeric or Categorical
    x = colvalues.tolist()[1]
    DataType = type(x)
       
    print(DataType)
    
    categorical =  (DataType is  str) or (DataType is  bool) 
    
    if(categorical):
        categories = numpy.unique(colvalues, return_counts=True)
        
        categoryLabels = categories[0]
        categoryValues = categories[1]
        
        noOfCategories = len(categoryLabels)
        
        noOfRecords = len(Data)
        print("\n")
        for i in range(0, noOfCategories):
            freq = int(categoryValues[i])/noOfRecords
            print(str(categoryLabels[i]) + ": No.: [" + str(categoryValues[i]) + "]; Freq : [" + "{:5.2f}".format(freq) + " ]")
        
        stats = "Categories : " + str(noOfCategories) + ", Avg freq : " + "{:5.2f}".format(numpy.mean(categoryValues)/noOfRecords) + ", Std.Dev.: " + "{:5.2f}".format(numpy.std(categoryValues))
        print("\n")
        print(stats) 
     

#Clearing the console
print("\n"*100)

#  Loading Data
DataFile = "C:\\Users\\Somnath\\Desktop\\tennis.csv"
Data = pandas.read_csv(DataFile)
Data.info()
    

# Quickly Check if the Data is correctly loaded or not
ShowHeader("Sample Data")
print(Data.head(5))

# Exploring Data
Datadim = Data.shape
noOfRows = Datadim[0]
noOfColumns = Datadim[1]

ShowHeader("Data Statistics")
print("No. of Records : " + str(noOfRows))
print("No. of Columns : " + str(noOfColumns))

for i in range(0, noOfColumns):
    ExploreDataColumn(Data,i)

##################################################################################################
    
# Naive Bayes Classification
def NaiveBayesforSingleObservation(Data,OutputColumn,Observation):
    
    Observationdf = pandas.DataFrame(Observation)
    ObservationColumns = Observationdf.columns
    
    OutputValues = Data[OutputColumn]
    
    OutputClasses = numpy.unique(OutputValues, return_counts=True)
    OutputClassValues = OutputClasses[0]
    OutputClassNumbers = OutputClasses[1]
    OutputClassesProbability = OutputClassNumbers/len(OutputValues)
    
    prediction = ""
    maxprobability = 0
    
    for i in range(0,len(OutputClassValues)):
        ClassName = OutputClassValues[i]
        ClassValue = Data.query(OutputColumn + ' == "' + ClassName + '"')
        
        cond_prob = OutputClassesProbability[i]
        
        for c in ObservationColumns:
            x = ClassValue[c].tolist()[1]
            DataType = type(x)
            
            val = Observationdf[c][0]
            
            if(DataType is bool):
                val = str(bool(val))
            else:
                val = "'" + val + "'"
            
            records = ClassValue.query(c + " ==" + val)
            likelihood = len(records) / OutputClassNumbers[i]
            cond_prob *= likelihood
            
        ShowHeader("Probability of Class [" + ClassName + "] is : "  + "{:5.4f}".format(cond_prob))
    
        if(maxprobability < cond_prob):
            prediction = ClassName
            maxprobability = cond_prob
            
    ShowHeader("Prediction of " + OutputColumn + " is : " + prediction)

OutputColumn = "play"
Observation = {'outlook': ['sunny'], 'temp': ['hot'], 'humidity': ['high'],'windy': ['TRUE']}

NaiveBayesforSingleObservation(Data, OutputColumn, Observation)

###################################################################################3
# More generic implementation of Naive Bayes

def TrainModelForNaiveBayes (Data, OutputColumn) :
    
    OutputValues = Data[OutputColumn]
        
    OutputClasses = numpy.unique(OutputValues, return_counts=True)
    OutputClassValues = OutputClasses[0]
    OutputClassNumbers = OutputClasses[1]
    OutputClassesProbability = OutputClassNumbers/len(OutputValues)
    
    WeightStructure = [0.0]* (len(OutputClassValues))
    
    ObservationColumns = Data.drop([OutputColumn], axis=1).columns
    DataTypes = ['str']* (len(ObservationColumns))
    ObservationStructure = [0.0]* (len(ObservationColumns))
     
    for i in range(0,len(ObservationColumns)):
        c = ObservationColumns[i]
        x = Data[c].tolist()[1]
        DataTypes[i] = type(x)
        ClassesForTheColumns = numpy.unique(Data[c], return_counts=True)
        ObservationStructure[i] = [c, type(x), ClassesForTheColumns[0]]
        
    
    for i in range(0,len(OutputClassValues)):
        ClassName = OutputClassValues[i]
        DataForClass = Data.query(OutputColumn + ' == "' + ClassName + '"')
        
        noOfObservationColumns = len(ObservationColumns)
        
        likelihoodsForAllCols = [0.0] * noOfObservationColumns
        
        for j in range(0,noOfObservationColumns):
            c = ObservationColumns[j]
            DataType = ObservationStructure[j][1]
            
            ClassesinTheColumns = ObservationStructure[j][2]

            noOfClasses = len(ClassesinTheColumns)
            
            likelihoods = [0.0] * noOfClasses
            
            for k in range(0,noOfClasses):
                
                val = ClassesinTheColumns[k]
                
                if(DataType is bool):
                    val = str(bool(val))
                else:
                    val = "'" + val + "'"
                
                records = DataForClass.query(c + " ==" + val)
                
                likelihoods[k] = len(records) / OutputClassNumbers[i]
            
            likelihoodsForAllCols[j] = likelihoods
            
        WeightStructure[i] = [ClassName, OutputClassesProbability[i], likelihoodsForAllCols]
            
    return [OutputColumn, ObservationStructure, WeightStructure]

# Train the model here
Wts = TrainModelForNaiveBayes(Data, OutputColumn)

# Print the Trained Weights for reviewing
ShowHeader(Wts)

###################################################################################3

def PedictWithNaiveBayesModel(OutputColumn, Test_Data, TrainingWeights):
    
    # Prepare the Test Data Structure for Prediction
    PredictionColumn = OutputColumn + "_Prediction"
    Test_Data[PredictionColumn] = "?"

    OutputClassesProbability = [1.0] * len(TrainingWeights[2])
    
    for c in range(0,len(TrainingWeights[2])):
        o = TrainingWeights[2][c]
        outputClass = (o[0])
        Test_Data[outputClass] = 0.0
        OutputClassesProbability[c] = TrainingWeights[2][c][1]
    
    for i in Test_Data.index:
    
        totalProb =  [1.0] * len(TrainingWeights[2])
        
        for j in range(0,len(TrainingWeights[1])):
            colname = TrainingWeights[1][j][0]
            DataType = TrainingWeights[1][j][1]
    
            colValue = Test_Data[colname][i]        
            if(DataType is bool):
                colValue = bool(colValue)
                
            colindex = 0
            classes = Wts[1][j][2]
            for k in range(0,len(classes)):
                if (classes[k] == colValue):
                    colindex = k
                    break
    
            for c in range(0,len(Wts[2])):
                o = TrainingWeights[2][c]
                outputClass = (o[0])
                
                prob = TrainingWeights[2][c][2][j][colindex]
                totalProb[c] *= prob
                Test_Data[outputClass][i] = totalProb[c]
                
        maxProb = 0.0
        prediction = ""
        
        for c in range(0,len(TrainingWeights[2])):
            o = TrainingWeights[2][c]
            outputClass = (o[0])
            totalProb[c] = Test_Data[outputClass][i]
            totalProb[c] = totalProb[c] * TrainingWeights[2][c][1]
            Test_Data[outputClass][i] = totalProb[c]
    
            if(totalProb[c] > maxProb):
                prediction = outputClass
                maxProb = totalProb[c]
        
        Test_Data[PredictionColumn][i] = prediction
        
    records = Test_Data.query(OutputColumn + " ==" + PredictionColumn)
    accuracy = 100 * len(records) / len(Test_Data)
    
    return [Test_Data,accuracy]


def PredictByNaiveBayes (OutputColumn, Training_Data, Test_Data):
    # Train the model here
    Wts = TrainModelForNaiveBayes(Data, OutputColumn)
    return PedictWithNaiveBayesModel(OutputColumn, Test_Data, Wts)

## Just Using the observations as test data
#Test_Data = pandas.DataFrame(Observation)
#Test_Data[OutputColumn] = "yes" # Just randomly making it up
#
## Just Using the full data as test data
#Test_Data = Data
#
## Get the Prediction
#Test_Data = PedictWithNaiveBayesModel(OutputColumn, Test_Data, Wts)
#print(Test_Data[Test_Data.columns[4:8]])
    

# Just Using the full data as test data
Prediction = PredictByNaiveBayes("play", Data, Data)

Predicted_Data = Prediction[0]

ShowHeader("Accuracy : " + "{:5.2f}".format(Prediction[1]) + " %")

print(Predicted_Data[Predicted_Data.columns[4:8]])



###################################################################################3