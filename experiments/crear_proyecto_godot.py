# -*- coding: utf-8 -*-

"""
Script para crear un proyecto Godot 2D básico.

Requisitos:
- Godot Engine instalada (versión compatible con --create-project desde la línea de comandos)
- Python 3.x

Uso:
    python godot_2d_project_creator.py --name <proyecto_nombre> --path <directorio_proyecto>
"""

import os
import argparse
import subprocess

def crear_proyecto_godot(nombre, directorio):
    """
    Crea un proyecto Godot 2D en el directorio especificado con el nombre dado.

    :param nombre: Nombre del proyecto
    :param directorio: Directorio donde se creará el proyecto
    """
    comando = f"godot --create-project {os.path.join(directorio, nombre)} --type=2d"
    try:
        # Ejecuta el comando y captura la salida
        salida = subprocess.check_output(comando, shell=True).decode('utf-8')
        print(f"Proyecto '{nombre}' creado con éxito en {directorio}")
        print("Salida del comando:")
        print(salida)
    except subprocess.CalledProcessError as e:
        print(f"Error al crear el proyecto: {e}")
        print("Salida de error:")
        print(e.output.decode('utf-8'))

def main():
    parser = argparse.ArgumentParser(description='Crea un proyecto Godot 2D.')
    parser.add_argument('--name', required=True, help='Nombre del proyecto')
    parser.add_argument('--path', required=True, help='Directorio del proyecto')
    
    args = parser.parse_args()
    
    # Validación básica de directorio
    if not os.path.isdir(args.path):
        print(f"El directorio '{args.path}' no existe. Creándolo...")
        os.makedirs(args.path)
        print(f"Directorio '{args.path}' creado.")
    
    crear_proyecto_godot(args.name, args.path)

if __name__ == "__main__":
    main()