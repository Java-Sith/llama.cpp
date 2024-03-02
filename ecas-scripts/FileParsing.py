import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

operation_mapping = {
    2: 'ADD',
    6: 'MUL',
    20: 'RMS NORM',
    23: 'MUL MAT',
    28: 'CPY',
    29: 'CONT',
    30: 'RESHAPE',
    31: 'VIEW',
    32: 'PERMUTE',
    33: 'TRANSPOSE',
    34: 'GET ROWS',
    39: 'SOFTMAX',
    41: 'ROPE',
    61: 'UNARY',
}

def process_file(file_path):
    # Diccionario para las operaciones de GGML
    operation_details = {}

    # Obtiene las expresiones regulares del 
    regex = re.compile(r'Operation (\d+) executed in (\d+\.\d+) microseconds\. Count: (\d+)')

    # Lee el archivo y toma las operaciones
    with open(file_path, 'r') as file:
        for line in file:
            match = regex.search(line)
            #Agrupa las coincidencias del archivo en grupos
            if match:
                operation_number, execution_time, count = map(float, match.groups())
                operation_number = int(operation_number)
                if operation_number in operation_mapping:
                    count = int(count)
                    function_name = operation_mapping[operation_number]
                    # Genera nuevas entradas del diccionario o actualiza las ya creadas
                    if operation_number not in operation_details:
                        operation_details[operation_number] = {
                            'execution_times': [execution_time],
                            'count': count,
                            'operation_name': function_name,
                        }
                    else:
                        operation_details[operation_number]['execution_times'].append(execution_time)
                        operation_details[operation_number]['count'] = max(operation_details[operation_number]['count'], count)
                        operation_details[operation_number]['operation_name'] = function_name
                else:
                     print(f"Operation {operation_number} not found in the mapping.")
            else:
                print("Match not found: ", line)

    # Procesa e imprime los resultados para mejor legibilidad
    for operation_number, details in sorted(operation_details.items()):
        mean_execution_time = sum(details['execution_times']) / len(details['execution_times'])
        lowest_execution_time = min(details['execution_times'])
        highest_execution_time = max(details['execution_times'])
        highest_count = details['count']
        operation_name = details['operation_name']
        print(f"Operation {operation_number}, named {operation_name} has a microseconds mean of {mean_execution_time:.6f}, "
              f"its lowest execution time is {lowest_execution_time:.6f}, "
              f"its highest execution time is {highest_execution_time:.6f} "
              f"and the highest count is {highest_count}")

    return operation_details

def create_plots(operation_details, output_filename):
    # Ordena el diccionario por número de operación
    sorted_operation_details = dict(sorted(operation_details.items()))

    # Extrae el nombre de la operación y su cantidad de ejecuciones asociada
    operation_names = [details["operation_name"] for details in sorted_operation_details.values()]
    highest_counts = [details["count"] for details in sorted_operation_details.values()]

    # Extrae el nombre de la operación y sus tiempos de ejecución asociados
    execution_times_data = [details["execution_times"] for details in sorted_operation_details.values()]

    # Crea el gráfico de la cantidad de ejecuciones
    plt.figure(figsize=(10, 5))
    plt.bar(operation_names, highest_counts, color='skyblue')
    plt.xlabel('Operation Name')
    plt.ylabel('Highest Count')
    plt.title('Histogram of Highest Counts')
    plt.xticks(rotation=45, ha="right")  # Rota el eje X por legibilidad
    plt.tight_layout()
    plt.savefig(output_filename + '_histogram.png') 

    # Crea el diagrama de bigotes del tiempo de ejecución
    plt.figure(figsize=(10, 5))
    plt.boxplot(execution_times_data, labels=operation_names)
    plt.xlabel('Operation Name')
    plt.ylabel('Execution Times')
    plt.title('Boxplot of Execution Times')
    plt.xticks(rotation=45, ha="right")  # Rota el eje X por legibilidad
    plt.tight_layout()
    plt.savefig(output_filename + '_boxplot.png')  

    # Especifica la operación para el tercer histograma
    target_operation_number = 23 
    target_execution_times = sorted_operation_details[target_operation_number]["execution_times"]

    # Crea el histograma de tiempos de la multiplicación matricial
    step = 2000
    indexes = np.arange(0, len(target_execution_times), step)
    plt.figure(figsize=(8, 4))
    plt.bar(indexes, [target_execution_times[i] for i in indexes], color='skyblue', edgecolor='black')
    plt.xlabel('Index')
    plt.ylabel('Execution Time')
    plt.title(f'Histogram of Execution Times for {sorted_operation_details[target_operation_number]["operation_name"]} (Step={step})')
    plt.tight_layout()
    plt.savefig(output_filename + '_specific_histogram.png') 

    plt.show()
    
if __name__ == "__main__":

    # Usa un parser para separar los argumentos de entrada
    parser = argparse.ArgumentParser(description='Create histograms and boxplots from operation details.')
    parser.add_argument('input_file', type=str, help='Path to the input file with operation details')
    
    # Interpeta las operaciones de entrada de la consola
    args = parser.parse_args()
    
    operations = process_file(args.input_file)

    create_plots(operations, args.input_file)



