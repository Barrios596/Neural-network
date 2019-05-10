import numpy as np
import random

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))

def derivada_sigmoide(z):
    return sigmoide(z)*(1-sigmoide(z))

def derivada_costo(a, y):
    return a-y

class Red(object):
    # El arreglo 'lengths' lleva la cantidad de neuronas de cada capa de la red.
    # Los bias y los pesos se inician aleatoriamente
    def __init__(self, lengths):
        
        self.num_capas = len(lengths)
        self.biases = []
        self.pesos = []

        for y in lengths[1:]:
            self.biases.append(np.random.rand(y,1))

        # los pesos se dividen dentro de la cantidad de inputs
        # para cada capa, para que no existan valores cercanos a 1
        # x: inputs
        # y: neuronas
        for x,y in zip(lengths[:-1], lengths[1:]):
            self.pesos.append(np.random.rand(y, x)/np.sqrt(x))
        
    def feedforward(self, X):
        # Recibe el arreglo X, opera por cada capa y lo devuelve como salida.
        for bias, peso in zip(self.biases, self.pesos):
            z = np.dot(peso, X) + bias
            X = sigmoide(z)
        return X

    # data: tuplas (x,y), alpha: tamaño del 'step' o 'learning rate'
    def descenso_gradiente(self, data, iteraciones, batch_length, test_length=7000, alpha=3):
        
        m = len(data) - batch_length
        
        # Se revuelven los datos y se hace el descenso del gradiente varias veces
        for i in range(iteraciones):
            print("iteración:",i)
            random.shuffle(data)
            # crear batches:
            batches = []
            for i in range(0, m, batch_length):
                batches.append(data[i : i + batch_length])
            for batch in batches:
                # nabla_biases es una matriz con matrices adentro, 
                # cada una con el bias de su correspondiente capa neuronal.
                nabla_biases = [np.zeros(bias.shape) for bias in self.biases]
                # nabla_pesos se inicia igual que nabla_bias, pero para los pesos
                nabla_pesos = [np.zeros(peso.shape) for peso in self.pesos]
                
                # calcular un delta por cada dato de entrenamiento y sumar todos
                for x,y in batch:
                    # back propagation devuelve dos matrices de matrices
                    # cada una tiene un cambio a nabla_biases y nabla_pesos, respectivamente
                    delta_nabla_bias, delta_nabla_pesos = self.back_propagation(x, y)
                    for i in range(len(nabla_biases)):
                        nabla_biases[i] = nabla_biases[i] + delta_nabla_bias[i]
                    for i in range(len(nabla_pesos)):
                        nabla_pesos[i] = nabla_pesos[i] + delta_nabla_pesos[i]

                # actualizar pesos por cada batch, se suman todos los nablas y
                # se dividen dentro de la cantidad de imágenes en el batch
                for i in range(len(self.biases)):
                    self.biases[i] = self.biases[i]-(alpha/len(batch))*nabla_biases[i]
                for i in range(len(self.pesos)):
                    self.pesos[i] = self.pesos[i]-(alpha/len(batch))*nabla_pesos[i]
            
            # cross-validation
            test_data = []
            for x,y in data[-test_length:]:
                test_data.append((x,y))
            print("Se testeó con", test_length," de",m,"datos con un porcentaje de acierto de:", self.evaluate(data[-test_length:]))
        
    def back_propagation(self, x, y):
        nabla_b = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        nabla_w = []
        for w in self.pesos:
            nabla_w.append(np.zeros(w.shape))
        a = x
        # almacenamos las activaciones, la primera es el input
        activaciones = [a]
        # zs guarda los vectores z que son el output de cada capa
        zetas = []
        # feedforward
        for b, w in zip(self.biases, self.pesos):
            # también guardamos los outputs de cada capa,
            # antes de usar sigmoide
            z = np.dot(w, a)+b
            zetas.append(z)

            # y también las activaciones de cada capa
            a = sigmoide(z)
            activaciones.append(a)

        # delta de la última capa:
        delta = derivada_costo(activaciones[-1], y.reshape(len(activaciones[-1]),1)) * derivada_sigmoide(zetas[-1])

        # el delta sólo se suma al bias anterior
        nabla_b[-1] = delta
        # el nabla de la última capa de pesos 
        nabla_w[-1] = np.dot(delta, activaciones[-2].transpose())

        # por cada capa que no sea la de los resultados
        for i in range(2, self.num_capas):
            z = zetas[-i]
            delta = np.dot(self.pesos[-i+1].transpose(), delta) * derivada_sigmoide(z)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activaciones[-i-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, inputs):
        assertions = 0
        for x, y in inputs:
            output = self.feedforward(x)
            if (np.argmax(output) == np.argmax(y)):
                assertions += 1.0
        return assertions / len(inputs) * 100.0
