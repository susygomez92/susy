import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# cargar el archivo  csv en un dataframe 
pd.read_csv('covid.csv')
df = pd.read_csv('covid.csv')
df.head()



# Eliminar las columnas que no necesitas
columnas_a_eliminar = ['id', 'patient_type', 'entry_date', 'date_symptoms', 'date_died', 'other_disease', 'icu']
df.drop(columnas_a_eliminar, axis=1, inplace=True)
df.head()


#Filtrar las filas y eliminar aquellas que tenga el valor 3 en la columna covid_res
df = df[df['covid_res'] != 3]
df.head()


#Filtrar las filas y eliminar aquellas que tenga el valor 99 en la columna contact_other_covid
df = df[df['contact_other_covid'] != 99]
df.head()



#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna renal_chronic
df = df[df['tobacco'] != 98]
df.head()


#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna renal_chronic
df = df[df['renal_chronic'] != 98]
df.head()


#Filtrar las filas y eliminar aquellas que tenga el valor 90 en la columna obesity
df = df[df['obesity'] != 90]
df.head()




#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna cardiovascular
df = df[df['cardiovascular'] != 98]
df.head()

#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna hypertension
df = df[df['hypertension'] != 98]
df.head()




#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna inmsupr
df = df[df['inmsupr'] != 98]
df.head()



#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna asthma
df = df[df['asthma'] != 98]
df.head()


#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna copd
df = df[df['copd'] != 98]
df.head()



#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna intubed
df = df[df['intubed'] != 97]
df.head()




#Filtrar las filas y eliminar aquellas que tenga el valor 98 en la columna pneumonia
df = df[df['pneumonia'] != 99]
df.head()
 



#Numero de filas total el DataFrame
num_filas = df.shape[0]
print("El  DataFrame tiene", num_filas, "filas.")




# separar las caacteristicas de la bariable  objetivo
x= df.drop('covid_res', axis=1)
y= df['covid_res']



#dividir los datos en conjunto de entrenamiento de prueva
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)




#crear una distancia del modelo del arbol  de decisiones
arbol_decision = DecisionTreeClassifier(random_state=1)




#entrenar el modelo con el conjunto de entrenamiento 
arbol_decision.fit(x_train, y_train)

 




# Utilizar el modelo para hacer predicion el el cojunto de prueba
y_pred = arbol_decision.predict(x_test)




# evaluar el desempeño del modelo 
accuracy = accuracy_score(y_test, y_pred)
print('la precision del modelo es:',accuracy)

 



pip install graphviz





from sklearn.tree import export_graphviz
import graphviz

# Exportar el árbol de decisiones a un archivo .dot
export_graphviz(arbol_decision, out_file='arbol_decision.dot', 
                feature_names=X.columns.values, filled=True, rounded=True, special_characters=True)

# Convertir el archivo .dot a un objeto graphviz
with open('arbol_decision.dot') as f:
    dot_graph = f.read()
graph = graphviz.Source(dot_graph)

# Mostrar el árbol de decisiones
graph
 








