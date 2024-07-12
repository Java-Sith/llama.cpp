import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

operation_mapping = {
    0: 'NONE', 1: 'DUP', 2: 'ADD', 3: 'ADD1', 4: 'ACC', 5: 'SUB', 6: 'MUL', 7: 'DIV',
    8: 'SQR', 9: 'SQRT', 10: 'LOG', 11: 'SUM', 12: 'SUM ROWS', 13: 'MEAN', 14: 'ARGMAX', 
    15: 'REPEAT', 16: 'REPEAT BACK', 17: 'CONCAT', 18: 'SILU BACK', 19: 'NORM', 20: 'RMS NORM',
    21: 'RMS NORM BACK', 22: 'GROUP NORM', 23: 'MUL MAT', 24: 'MUL MAT ID', 25: 'OUT PROD', 26: 'SCALE',
    27: 'SET', 28: 'CPY', 29: 'CONT', 30: 'RESHAPE', 31: 'VIEW', 32: 'PERMUTE', 33: 'TRANSPOSE', 
    34: 'GET ROWS', 35: 'GET ROWS BACK', 36: 'DIAG', 37: 'DIAG MASK INF', 38: 'DIAG MASK ZERO', 
    39: 'SOFTMAX', 40: 'SOFTMAX BACK', 41: 'ROPE', 42: 'ROPE BACK', 43: 'ALIBI', 44: 'CLAMP',
    45: 'CONV TRANSPOSE 1D', 46: 'IM2COL', 47: 'CONV TRANSPOSE 2D', 48: 'POOL 1D', 49: 'POOL 2D',
    50: 'UPSCALE', 51: 'PAD', 52: 'ARGSORT', 53: 'LEAKY RELU', 54: 'FLASH ATTN', 55: 'FLASH FF',
    56: 'FLASH ATTN BACK', 57: 'WIN PART', 58: 'WIN UNPART', 59: 'GET REL POS', 60: 'ADD REL POS', 
    61: 'UNARY', 62: 'MAP UNARY', 63: 'MAP BINARY', 64: 'CUSTOM1 F32', 65: 'CUSTOM2 F32', 66: 'CUSTOM3 F32',
    67: 'MAP CUSTOM1', 68: 'MAP CUSTOM2', 69: 'MAP CUSTOM3', 70: 'CROSS ENTROY LOSS', 71: 'CROSS ENTROPY LOSS BACK',
    72: 'COUNT'
}

def process_file(file_path):
    operation_details = {}
    regex = re.compile(r'Operation (\d+) executed in (\d+\.\d+) microseconds\. Count: (\d+)')
    with open(file_path, 'r') as file:
        for line in file:
            match = regex.search(line)
            if match:
                operation_number, execution_time, count = map(float, match.groups())
                operation_number = int(operation_number)
                if operation_number in operation_mapping:
                    count = int(count)
                    function_name = operation_mapping[operation_number]
                    # Convert microseconds to nanoseconds
                    execution_time_ns = execution_time * 1000
                    if operation_number not in operation_details:
                        operation_details[operation_number] = {
                            'execution_times': [execution_time_ns],
                            'count': count,
                            'operation_name': function_name,
                        }
                    else:
                        operation_details[operation_number]['execution_times'].append(execution_time_ns)
                        operation_details[operation_number]['count'] = max(operation_details[operation_number]['count'], count)
                        operation_details[operation_number]['operation_name'] = function_name
                else:
                     print(f"Operation {operation_number} not found in the mapping.")
            else:
                print("Match not found: ", line)

    # Calculate statistics using nanoseconds for execution times
    for operation_number, details in sorted(operation_details.items()):
        mean_execution_time = sum(details['execution_times']) / len(details['execution_times'])
        lowest_execution_time = min(details['execution_times'])
        highest_execution_time = max(details['execution_times'])
        highest_count = details['count']
        operation_name = details['operation_name']
        median_execution_time = np.median(details['execution_times'])
        print(f"Operation {operation_number}, named {operation_name} has a nanoseconds mean of {mean_execution_time:.6f}, "
              f"its lowest execution time is {lowest_execution_time:.6f}, "
              f"its medium execution time is {median_execution_time:.6f}, "
              f"its highest execution time is {highest_execution_time:.6f} "
              f"and the highest count is {highest_count}")

    return operation_details

""" def create_plots(operation_details, output_filename):
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
    #plt.yscale('log')
    plt.savefig(output_filename + '_histogram.png') 

    # Crea el diagrama de bigotes del tiempo de ejecución
    plt.figure(figsize=(10, 5))
    plt.boxplot(execution_times_data, labels=operation_names)
    plt.xlabel('Operation Name')
    plt.ylabel('Execution Times (ns)')
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
    plt.ylabel('Execution Time (µs)')
    plt.title(f'Histogram of Execution Times for {sorted_operation_details[target_operation_number]["operation_name"]} (Step={step})')
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(output_filename + '_specific_histogram.png') 

    plt.show() """

def create_plots(operation_details, output_filename):
    sorted_operation_details = dict(sorted(operation_details.items()))
    operation_names = [details["operation_name"] for details in sorted_operation_details.values()]
    highest_counts = [details["count"] for details in sorted_operation_details.values()]
    execution_times_data = [details["execution_times"] for details in sorted_operation_details.values()]

    plt.figure(figsize=(10, 5))
    plt.bar(operation_names, highest_counts, color='skyblue')
    plt.xlabel('Operation Name')
    plt.ylabel('Highest Count')
    plt.title('Histogram of Highest Counts')
    plt.xticks(rotation=45, ha="right") 
    plt.tight_layout()
    plt.savefig(output_filename + '_histogram.png') 

    plt.figure(figsize=(10, 5))
    plt.boxplot(execution_times_data, labels=operation_names)
    plt.xlabel('Operation Name')
    plt.ylabel('Execution Times (ns)')  # Adjusted to display nanoseconds
    plt.title('Boxplot of Execution Times')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_filename + '_boxplot.png')  
    
if __name__ == "__main__":

    # Usa un parser para separar los argumentos de entrada
    parser = argparse.ArgumentParser(description='Create histograms and boxplots from operation details.')
    parser.add_argument('input_file', type=str, help='Path to the input file with operation details')
    
    # Interpeta las operaciones de entrada de la consola
    args = parser.parse_args()
    
    operations = process_file(args.input_file)

    create_plots(operations, args.input_file)



