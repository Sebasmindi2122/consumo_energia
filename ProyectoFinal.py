import tkinter as tk
import pandas as pd
import numpy as np                     
from sklearn.metrics import r2_score
from sympy import integrate, symbols, sympify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objs as go
import json
from tkinter import filedialog
import sympy as sp
from tkinter import simpledialog


#Colores para la personalizacion

colorFondo = "#71E9F7"
lima = "#67F9A1"
negro = "#000000"

ventana = tk.Tk()
ventana.title("Calculus Consumption")
ventana.config(bg=colorFondo)
ventana.geometry("1200x720+150+10")
ventana.iconbitmap(r"./icon.ico")
etiqueta_estado = tk.Label(ventana, text="")
etiqueta_estado.pack()
def cargar_archivo():
    global filename
    global nombre_persona
    nombre_persona = simpledialog.askstring("Nombre de persona", "Ingrese el nombre de la persona:")
    if nombre_persona is not None:
        filename = filedialog.askopenfilename(initialdir='/', title='Seleccione archivo', filetypes=(('CSV files', '*.csv'),))
        if filename:
            etiqueta_estado.config(text=f"Archivo subido para {nombre_persona}", fg="green")
        else:
            etiqueta_estado.config(text=f"No se ha subido archivo para {nombre_persona}", fg="red")

regression_function = None 
ecuacion_sympy = None
def graficador(nombre_persona,archivo_csv):
    global fig
    global regression_function
    global ecuacion_sympy
    global filename
    df = pd.read_csv(filename)
    q1 = df['Consumo'].quantile(0.25)
    q3 = df['Consumo'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['Consumo'] < q1 - 1.5*iqr) | (df['Consumo'] > q3 + 1.5*iqr)]
    df = df[(df['Consumo'] >= q1 - 1.5*iqr) & (df['Consumo'] <= q3 + 1.5*iqr)]
    x = df['Mes'].values
    y = df['Consumo'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    lin_reg = LinearRegression()
    poly_reg = LinearRegression()
    poly_transform = PolynomialFeatures(degree=2)
    x_train_poly = poly_transform.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly_transform.fit_transform(x_test.reshape(-1, 1))
    lin_reg.fit(x_train.reshape(-1, 1), y_train)  
    poly_reg.fit(x_train_poly, y_train)

    lin_r2 = r2_score(y_test, lin_reg.predict(x_test.reshape(-1, 1)))
    poly_r2 = r2_score(y_test, poly_reg.predict(x_test_poly))

    


    if len(outliers) > 0:
        fig = go.Figure([
                    go.Scatter(x=outliers['Mes'], y=outliers['Consumo'], 
                            name='outliers', mode='markers', marker=dict(color='green', size=10)),
                ])
    if lin_r2 > poly_r2:

        y_pred = lin_reg.predict(x.reshape(-1, 1))

        fig = go.Figure([
                    go.Scatter(x=x_train.squeeze(), y=y_train, 
                            name='train', mode='markers'),
                    go.Scatter(x=x_test.squeeze(), y=y_test, 
                            name='test', mode='markers'),
                    go.Scatter(x=x[~np.isin(x, np.concatenate((x_train, x_test)))], y=y[~np.isin(x, np.concatenate((x_train, x_test)))], 
                            name='outlier', mode='markers', marker=dict(color='red', size=10)),
                    go.Scatter(x=x[~np.isin(x, np.concatenate((x_train, x_test, x[~np.isin(x, np.concatenate((x_train, x_test)))])))], y=y[~np.isin(x, np.concatenate((x_train, x_test, x[~np.isin(x, np.concatenate((x_train, x_test)))])))], 
                            name='other data', mode='markers', marker=dict(color='green', size=10)),
                    go.Scatter(x=x, y=y_pred.squeeze(), 
                            name='prediction')])
        fig.show()
    else:
        # Predecir los valores con el modelo de regresión polinomial
        y_pred = poly_reg.predict(poly_transform.fit_transform(x.reshape(-1, 1)))
        
        # Crear la figura con Plotly para visualizar los datos y la predicción
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = poly_reg.predict(poly_transform.fit_transform(x_range.reshape(-1, 1)))
        
        fig = go.Figure([
                    go.Scatter(x=x_train.squeeze(), y=y_train, 
                            name='train', mode='markers'),
                    go.Scatter(x=x_test.squeeze(), y=y_test, 
                            name='test', mode='markers'),
                    go.Scatter(x=x[~np.isin(x, np.concatenate((x_train, x_test)))], y=y[~np.isin(x, np.concatenate((x_train, x_test)))], 
                            name='outlier', mode='markers', marker=dict(color='red', size=10)),
                    go.Scatter(x=x[~np.isin(x, np.concatenate((x_train, x_test, x[~np.isin(x, np.concatenate((x_train, x_test)))])))], y=y[~np.isin(x, np.concatenate((x_train, x_test, x[~np.isin(x, np.concatenate((x_train, x_test)))])))], 
                            name='other data', mode='markers', marker=dict(color='green', size=10)),
                    go.Scatter(x=x_range, y=y_range.squeeze(), 
                            name='prediction')
            ])
        fig.show()
        regression_function = ''

    if lin_r2 > poly_r2:
        m, b = np.polyfit(x, y_pred, 1)
        regression_function = f'{m:.2f}*x + {b:.2f}'
    else:
        p4,p3,p2,p1, p0 = np.polyfit(x_range, y_range, 4)
        regression_function = f'{p4:2f}*x**4+{p3:2f}*x**3+{p2:.2f}*x**2 + {p1:.2f}*x + {p0:.2f}'
    print(f'Regresión: y = {regression_function}')
    ecuacion_sympy = sympify(regression_function)
    extraer()

    with open(r'./ecucaciones.txt', 'r') as archivo:
        contenido_actual = archivo.read()
    if contenido_actual:
        resultados = json.loads(contenido_actual)
    if ecuacion_sympy is not None:
        resultados[nombre_persona] = str(ecuacion_sympy)
        with open(r"./ecuaciones.txt", mode='w') as archivo:
            json.dump(resultados, archivo)    
    



def extraer():
    global resultados
    resultados ={}
    with open(r'./ecuaciones.txt', 'r') as archivo:
        contenido_actual = archivo.read()
    if contenido_actual:
        resultados = json.loads(contenido_actual)
    return resultados 
resultados_evaluacion = {}
def comparar(w):
  global resultados, resultados_evaluacion
  mes = float(w)
  ress = []
  ress.append("Resultados de la evaluación:")
  for nombre_personaa, ecuacion in resultados.items():
      evae = sympify(ecuacion)
      resultado_evaluacion = evae.evalf(subs={'x': mes})      
      resultados_evaluacion[nombre_persona] = round(resultado_evaluacion,2)
      ress.append(f"{nombre_personaa}: {round(resultado_evaluacion)}")
  valores = list(resultados_evaluacion.values())
  max_consumo = max(resultados_evaluacion, key=resultados_evaluacion.get)
  min_consumo = min(resultados_evaluacion, key=resultados_evaluacion.get)
  ress.append(f"\nEl consumo pr hola soy un cometario hecho por william omedio en el mes {mes} es de: {round(np.mean(valores),2)} kwt")
  ress.append(f"La persona que consumió más en el mes {mes} es {max_consumo} con {round(resultados_evaluacion[max_consumo],2)} kwt")
  ress.append(f"La persona que consumió menos en el mes {mes} es {min_consumo} con {round(resultados_evaluacion[min_consumo],2)} kwt")
  return "\n".join(ress)



def mesdet(mes):
  global nombre_persona
  global ecuacion_sympy
  ecuacion = ecuacion_sympy
  x = symbols('x')
  expr = sympify(ecuacion)
  valor_evaluar = float(mes)
  resultado_evaluacion = expr.evalf(subs={x: valor_evaluar})
  resultado = f"El consumo de  {nombre_persona} en el mes {valor_evaluar} es: {round(resultado_evaluacion)} kwt "
  return resultado

def variosmes(p,z):
  global ecuacion_sympy
  global regression_function
  x = symbols('x')
  menor = float(p)
  mayor = float(z)
  integral = integrate(ecuacion_sympy, (x, menor, mayor))
  prom_consu = round(integral*(1/(z-p)))
  total_a_pagar = round(374.24*integral)
  total_a_pagar_str = "{:,.2f}".format(float(total_a_pagar))
  resultado = [
      f'Tendencia: y = {regression_function}',
      f'El total de consumo en estos se espera que sea de: {round(integral)} kwt',
      f"El total a pagar por {round(integral)} kwt es de {total_a_pagar_str} pesos",
      f"El consumo promedio se espera que sea de {prom_consu} kwt en esos meses"
  ]
  return "\n".join(resultado)
def pruebader(a,b):
  global ecuacion_sympy
  x = sp.Symbol('x')
  inf = int(a)
  sup = int(b)
  derivada = sp.diff(ecuacion_sympy, x)
  primera_derivada_en_a = derivada.subs(x, inf)
  primera_derivada_en_b = derivada.subs(x, sup)

  if primera_derivada_en_a > 0 and primera_derivada_en_b > 0:
      resultado = f"El consumo de energía aumenta en el intervalode meses desde el mes {inf} a {sup}"
      resultado = f"El consumo de energía aumenta en el intervalo "
  elif primera_derivada_en_a < 0 and primera_derivada_en_b < 0:
      resultado = f"El consumo de energía disminuye en el intervalo de meses desde el mes {inf} a {sup}"
  elif primera_derivada_en_a == 0 and primera_derivada_en_b == 0:
      resultado = f"El consumo de energía se mantiene constante en el intervalo de meses desde el mes {inf} a {sup}"
  else:
      resultado = f"El consumo de energía varía en esos meses, es decir no tiene una tendencia marcada"
  return resultado

def nuevomes(mess):
  global filename
  df = pd.read_csv(filename)
  ultimo_mes = df["Mes"].iloc[-1]
  nuevo_mes = ultimo_mes + 1
  nuevo_consumo = float(mess)
  nueva_fila = pd.DataFrame({'Mes': [nuevo_mes], 'Consumo': [nuevo_consumo]})
  df = pd.concat([df, nueva_fila], ignore_index=True)
  df.to_csv(filename, index=False)
  resul = "El archivo ha sido actuañlizado con exito"
  return resul

boton_cargar_archivo = tk.Button(ventana, text="Cargar archivo", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", cursor="hand2", borderwidth=2, command=cargar_archivo)
boton_cargar_archivo.pack()

campo_salida = tk.Text(ventana, width=50, height=10)
campo_salida.pack()


def limipiar_campos():
    campo_entrada_min.delete(0, tk.END)
    campo_entrada_max.delete(0, tk.END)
    mess.delete(0,tk.END)
    campo_salida.delete('1.0', tk.END)

def borrar_texto(evento):
    if evento.widget.get()  == evento.widget.get():
       evento.widget.delete(0, tk.END)

def poner_texto(evento):
    if evento.widget.get() == "":
        evento.widget.insert(0, evento.widget.campo)


mess = tk.Entry(ventana, width=13, justify="center", font="gregoria, 11", fg="green")
mess.campo = "Ingrese un mes"
mess.insert(0, mess.campo)
mess.bind("<FocusIn>", borrar_texto)
mess.bind("<FocusOut>", poner_texto)
mess.pack(pady=10)
campo_entrada_min = tk.Entry(ventana, width=22, justify="center", font="gregoria, 12", fg="red")
campo_entrada_min.campo = "Primer mes o limite inferior"
campo_entrada_min.insert(0, campo_entrada_min.campo)
campo_entrada_min.bind("<FocusIn>", borrar_texto)
campo_entrada_min.bind("<FocusOut>", poner_texto)
campo_entrada_min.pack(pady=4)
campo_entrada_max = tk.Entry(ventana, width=22, justify="center", font="gregoria, 12", fg="red")
campo_entrada_max.campo = "Ultimo mes o limite superior"
campo_entrada_max.insert(0, campo_entrada_max.campo)
campo_entrada_max.bind("<FocusIn>", borrar_texto)
campo_entrada_max.bind("<FocusOut>", poner_texto)
campo_entrada_max.pack(pady=4)


def describe():
    df = pd.read_csv(filename)
    descripcion = df.describe().style.background_gradient(cmap='Blues').to_string()
    campo_salida.delete("1.0",tk.END)
    campo_salida.insert("1.0", descripcion)
    return campo_salida

def opcion1():
    describe()

def opcion2():
    global boton_opcion2
    boton_opcion2 = True
    graficador(nombre_persona,filename)
def calcular_variosmes():
    if boton_opcion2:
        minimo = float(campo_entrada_min.get())
        maximo = float(campo_entrada_max.get())
        resultado = variosmes(minimo, maximo)
        campo_salida.delete('1.0', tk.END)
        campo_salida.insert('1.0', resultado)
    else:
        campo_salida.delete('1.0', tk.END)
        campo_salida.insert('1.0', "Debes presionar el botón de graficar funciones primero.")
def calcular__un_mes():
    if boton_opcion2:
        valor = float(mess.get())
        resultado = mesdet(valor)
        campo_salida.delete("1.0", tk.END)
        campo_salida.insert("1.0", resultado)
    else:
        campo_salida.delete('1.0', tk.END)
        campo_salida.insert('1.0', "Debes presionar el boton de graficar funciones primero.")

def usuacompa():
    if boton_opcion2:
        detmes = float(mess.get())
        com = comparar(detmes)
        campo_salida.delete("1.0", tk.END)
        campo_salida.insert("1.0", com)
    else:
        campo_salida.delete('1.0', tk.END)
        campo_salida.insert('1.0', "Debes presionar el boton de graficar funciones primero.")

def concA():
    if boton_opcion2:
        inf = float(campo_entrada_min.get())
        sup = float(campo_entrada_max.get())
        prue = pruebader(inf, sup)
        campo_salida.delete("1.0", tk.END)
        campo_salida.insert("1.0", prue)
    else:
        campo_salida.delete('1.0', tk.END)
        campo_salida.insert('1.0', "Debes presionar el boton de graficar funciones primero.")
def añadirmes():
    if boton_opcion2:
        mese = float(mess.get())
        nue = nuevomes(mese)
        campo_salida.delete("1.0", tk.END)
        campo_salida.insert("1.0", nue)
    else:
        campo_salida.delete('1.0', tk.END)
        campo_salida.insert('1.0', "Debes presionar el boton de graficar funciones primero.")


boton_opcion1 = tk.Button(ventana, text="1. Descripción de datos", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=opcion1)
boton_opcion1.pack(pady=4)
boton_opcion1.pack()
boton_opcion2 = tk.Button(ventana, text="2. Gráfico de relación de datos", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=opcion2)
boton_opcion2.pack(pady=4)
boton_opcion2.pack()
boton_opcion2 = False

boton3 = tk.Button(ventana, text="3. Consumo de energia en varios meses", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=calcular_variosmes)
boton3.pack()
boton4 = tk.Button(ventana, text="4. Consumo de energía en un mes determinado", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=calcular__un_mes)
boton4.pack(pady=4)
boton5 = tk.Button(ventana, text="5. Comparar consumo de energía con otros usuarios", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=usuacompa)
boton5.pack(pady=4)
boton6 = tk.Button(ventana, text="6. Análisis de intervalos de consumo", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=concA)
boton6.pack(pady=4)
boton7 = tk.Button(ventana, text="7. Añadir información de un nuevo mes", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground=lima, activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=añadirmes)
boton7.pack(pady=4)   
boton_salir = tk.Button(ventana, text="Salir", relief='ridge', overrelief="raised", 
                          bg="white", font="gregoria", activebackground="red", activeforeground=negro,
                          cursor="hand2", borderwidth=2, command=ventana.quit)


# Añadir botones a la ventana


boton_salir.pack()

ventana.mainloop() 





