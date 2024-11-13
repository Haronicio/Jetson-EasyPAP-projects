# -*- coding: utf-8 -*-
# Python 2.7 or Higher
# Usage ; python3 ./scripts/benchmark.py
# TODO : Meilleur titres pour les graphiques, mesure de la mémoire hazardeuse (et pas necessaire)

import subprocess
import re
import sys
import time
import argparse
# import psutil
# import pandas as pd
import matplotlib.pyplot as plt

#suffixe pour denver ou A57
default_suffixe = ""

# def get_memory_usage(t,pid):
#     time.sleep(t)
#     process = psutil.Process(pid)
#     return process.memory_info().rss  # Retourne la mémoire utilisée en octets

def getMs_Fps_GFlop(output):
    # Recherche des valeurs de temps en ms, FPS et GFlop/s dans la sortie
    match = re.search(r'(\d+(?:\.\d+)?) ms \((\d+(?:\.\d+)?) FPS, +(\d+(?:\.\d+)?) Gflop/s\)', output)
    if match:
        ms = float(match.group(1))      # Temps en millisecondes
        fps = float(match.group(2))     # FPS
        gflops = float(match.group(3))  # GFlop/s
        return (ms, fps, gflops)
    else:
        return None
        
# Créé un histogramme pour la quantité de mémoire
# def saveMemHistoPanda(mem,key):
#     data = [{'Implementation': impl, 'MB': mem[impl][key]} 
#         for impl in mem.keys() 
#         if key in mem[impl]]
#     # Création d'un DataFrame pour les résultats
#     df = pd.DataFrame(data, columns=['Implementation', 'MB'])
#     # Sauvegarde de l'histogramme
#     plt.figure(figsize=(10, 6))
#     plt.bar(df['Implementation'], df['MB'], color='skyblue')
#     plt.xlabel('Implementation')
#     plt.ylabel('MB')
#     plt.title('Memory Usage Comparison Across Different Implementations \n(1000 bodies 1000 iterations)\n lower is better')
#     plt.savefig('./assets/histo_mem.png')
def saveMemHisto(mem, key):
    implementations = []
    memory_usage = []
    
    for impl in mem.keys():
        if key in mem[impl]:
            implementations.append(impl)
            memory_usage.append(mem[impl][key])
    
    # Sauvegarde de l'histogramme
    plt.figure(figsize=(10, 6))
    plt.bar(implementations, memory_usage, color='skyblue')
    plt.xlabel('Implementation')
    plt.ylabel('MB')
    plt.title('Memory Usage Comparison Across Different Implementations' + default_suffixe + '\n(1000 bodies 1000 iterations)\n lower is better')
    plt.savefig('./assets/histo_mem' + default_suffixe + '.png')
    
    
# Créé un histogramme pour le nombre moyen de FPS pour un certain nombre de bodies
# def saveFpsHistoPanda(fps_results,key):
#     # Création d'un DataFrame pour les résultats
#     data = [{'Implementation': impl, 'FPS': fps_results[impl][key]} 
#         for impl in fps_results.keys() 
#         if key in fps_results[impl]]
    
#     df = pd.DataFrame(data, columns=['Implementation', 'FPS'])
#     # Sauvegarde de l'histogramme
#     plt.figure(figsize=(10, 6))
#     plt.bar(df['Implementation'], df['FPS'], color='skyblue')
#     plt.xlabel('Implementation')
#     plt.ylabel('FPS')
#     plt.title('FPS Comparison Across Different Implementations \n(1000 bodies 1000 iterations)\n higher is better')
#     plt.savefig('./assets/histo_fps.png')

def saveFpsHisto(fps_results, key):
    implementations = []
    fps_values = []
    
    for impl in fps_results.keys():
        if key in fps_results[impl]:
            implementations.append(impl)
            fps_values.append(fps_results[impl][key])
    
    # Sauvegarde de l'histogramme
    plt.figure(figsize=(10, 6))
    bars = plt.bar(implementations, fps_values, color='skyblue')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    plt.xlabel('Implementation')
    plt.ylabel('FPS')
    plt.title('FPS Comparison Across Different Implementations' + default_suffixe + '\n(1000 bodies 1000 iterations)\n higher is better')
    plt.savefig('./assets/histo_fps' + default_suffixe + '.png')
    plt.close()
    
    
def saveFpsBodies(fps_results):
    plt.figure(figsize=(12, 6))

    for impl, data in fps_results.items():
        # Trier les données par nombre de corps pour un tracé cohérent
        bodies = sorted(data.keys())
        fps_values = [data[n] for n in bodies]

        # Tracer la courbe pour l'implémentation actuelle
        plt.plot(bodies, fps_values, label=impl)

    plt.xlabel('Number of Bodies')
    plt.ylabel('FPS')
    plt.title('FPS by Number of Bodies for Each Implementation ' + default_suffixe + '')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/fps_bodies' + default_suffixe + '.png')
    plt.close()
    
def saveGflopBodies(fps_results):
    plt.figure(figsize=(12, 6))

    for impl, data in fps_results.items():
        # Trier les données par nombre de corps pour un tracé cohérent
        bodies = sorted(data.keys())
        flop_values = [data[n] for n in bodies]

        # Tracer la courbe pour l'implémentation actuelle
        plt.plot(bodies, flop_values, label=impl)

    plt.xlabel('Number of Bodies')
    plt.ylabel('GFlop/s')
    plt.title('Flops Frequencies by Number of Bodies for Each Implementation ' + default_suffixe + '')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/flop_bodies' + default_suffixe + '.png')
    plt.close()
    
    
def saveMemBodies(mem_results):
    plt.figure(figsize=(12, 6))

    for impl, data in mem_results.items():
        # Trier les données par nombre de corps pour un tracé cohérent
        bodies = sorted(data.keys())
        mem_values = [data[n] for n in bodies]

        # Tracer la courbe pour l'implémentation actuelle
        plt.plot(bodies, mem_values, label=impl)

    plt.xlabel('Number of Bodies')
    plt.ylabel('MB')
    plt.title('Memory Usage by Number of Bodies for Each Implementation ' + default_suffixe + '')
    plt.legend()
    plt.grid(True)
    plt.savefig('./assets/mem_bodies' + default_suffixe + '.png')
    plt.close()
    
def savePerCore(time_single_results, time_cores_results):
    # Calculer la performance optimale (temps pour un seul cœur divisé par le nombre de cœurs)
    core_counts = sorted(time_cores_results.keys())
    optimal_performance = [time_single_results / core for core in core_counts]

    # Tracer la performance pour chaque cœur
    plt.figure(figsize=(10, 6))
    actual_performance = [time_cores_results[core] for core in core_counts]
    bars_actual = plt.bar(core_counts, actual_performance, color='blue', label='Actual Performance', alpha=0.6)

    # Tracer les barres de performance optimale
    bars_optimal = plt.bar(core_counts, optimal_performance, color='red', label='Optimal Performance', alpha=0.6)

    # Ajouter des valeurs sur les barres
    for bar, value in zip(bars_actual, actual_performance):
        plt.text(bar.get_x() + bar.get_width() / 2, value, 
                 round(value, 2), ha='center', va='bottom', color='black')

    for bar, value in zip(bars_optimal, optimal_performance):
        plt.text(bar.get_x() + bar.get_width() / 2, value, 
                 round(value, 2), ha='center', va='bottom', color='black')

    # Étiquettes et légende
    plt.xlabel('Number of cores')
    plt.ylabel('Time (ms)')
    plt.title('Core Performance Comparison')
    plt.legend()

    plt.grid(True)
    plt.savefig('./assets/per_cores' + default_suffixe + '.png')
    plt.close()
    
def saveSpeedup(time_single_results, time_cores_results):
    core_counts = list(time_cores_results.keys())
    speedup_values = [time_single_results / time_cores_results[core] for core in core_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(core_counts, core_counts, linestyle='--', color='red', label='Optimal')
    plt.plot(core_counts, speedup_values, marker='o', linestyle='-',label='Actual')
    for i in range(len(core_counts)):
        plt.text(core_counts[i], speedup_values[i], round(speedup_values[i], 2), ha='right', va='bottom')
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Cores ' + default_suffixe + '')
    plt.grid(True)
    plt.legend()
    plt.savefig('./assets/speedup' + default_suffixe + '.png')
    plt.close()

def saveEfficiency(time_single_results, time_cores_results):
    core_counts = list(time_cores_results.keys())
    # print(core_counts,time_cores_results[1],time_single_results)
    efficiency_values = [( time_cores_results[core] / (time_single_results * core)) for core in core_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(core_counts, efficiency_values, marker='o', linestyle='-',label='Actual')
    for i in range(len(core_counts)):
        plt.text(core_counts[i], efficiency_values[i], round(efficiency_values[i], 2), ha='right', va='bottom')
    plt.plot(core_counts, [1] * len(core_counts), linestyle='--', color='red', label='Optimal')
    plt.xlabel('Number of Cores')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs. Number of Cores ' + default_suffixe + '')
    plt.grid(True)
    plt.legend()
    plt.savefig('./assets/efficiency' + default_suffixe + '.png')
    plt.close()
    
def saveScalabilityEfficiency(time_single_result, weak_scalability_times):
    core_counts = list(weak_scalability_times.keys())
    efficiency_values = [(time_single_result / (weak_scalability_times[core] * core)) for core in core_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(core_counts, efficiency_values, marker='o', linestyle='-')
    for i in range(len(core_counts)):
        plt.text(core_counts[i], efficiency_values[i], round(efficiency_values[i], 2), ha='right', va='bottom')
    plt.xlabel('Number of Cores')
    plt.ylabel('Efficiency')
    plt.title('Weak Scalability: Efficiency vs. Number of Cores ' + default_suffixe + '')
    plt.grid(True)
    plt.savefig('./assets/scalability_efficiency' + default_suffixe + '.png')
    plt.close()

# implementations = ["cpu+naive", "cpu+optim", "cpu+simd", "cpu+omp", "gpu+ocl"]
# implementations = ["cpu+naive", "cpu+optim", "cpu+simd", "cpu+omp"]
implementations = ["cpu+naive", "cpu+optim", "cpu+simd", "cpu+omp", "gpu+ocl", "hetero"]

default_n = 3000 #3000
default_i = 100  #100

n_incr = 20 #25
n_min = 700 #750

#time for mem check
t = 0 #0.05

default_core = 8
default_profile = "auto" #dynamic, static, auto, guided, runtime
default_chunk = 16

default_arch = ""
default_suffixe = ""

default_nice = 'nice -n 0'
default_taskset = 'taskset -c 0-'+str(default_core)

default_best = False



# main_cmd = f"{default_nice} {default_taskset} ./murb-se/build/bin/murb -i {default_i} --nv --gf" 
main_cmd = "{} {} ./murb-se/build/bin/murb --nv --gf -i {}".format(default_nice, default_taskset, default_i)

fps_results = {}
flop_results = {}
mem_results = {}
fps_single_results = 0
time_single_results = 0
time_cores_results = {} #= strong_scalability_times
fps_cores_results = {}
weak_scalability_times = {}

def arg_parser():
    parser = argparse.ArgumentParser(description="Programme de benchmark pour murb\n Exemple d'utilisation : python2 ./scripts/benchmark.py --im cpu+naive cpu+optim cpu+omp --incr 25 -i 1 --arch jetson")

    # Définir les arguments
    parser.add_argument("--im", nargs='+', default=["cpu+naive", "cpu+optim", "cpu+simd", "cpu+omp", "gpu+ocl", "hetero"], help="Liste des implémentations.")
    parser.add_argument("-n", type=int,default=3000, help="Nombre par défaut de n.")
    parser.add_argument("-i", type=int, default=100, help="Nombre par défaut de i.")
    parser.add_argument("--incr", type=int, default=20, help="Incrément par défaut pour n.")
    parser.add_argument("--min", type=int, default=700, help="Valeur minimale par défaut pour n.")
    parser.add_argument("-c", "--core", type=int, default=8, help="Nombre par défaut de cœurs.")
    parser.add_argument("--profile", type=str, default="auto", choices=["auto", "dynamic", "static", "guided", "runtime"], help="Profil par défaut.")
    parser.add_argument("--chunk", type=int, default=16, help="Taille de chunk par défaut.")
    parser.add_argument("--arch", type=str, default="", help="Architecture par défaut.")
    parser.add_argument("--best-parallel", action="store_true", help="Exécuter la routine pour trouver les options parallélisme optimale.")
    parser.add_argument("--name", type=str, help="Changer le nom du benchmark.")

    # Analyser les arguments
    args = parser.parse_args()

    return args

def set_defaults():
    
    global implementations,default_n,default_i,n_incr ,n_min ,default_core,default_profile ,default_chunk,default_arch,default_best,default_suffixe,main_cmd
    
    args = arg_parser()

    # Accéder aux valeurs des arguments
    implementations = args.im
    if args.n < 1000:
        default_n = 1000
        print("at least 1000 bodies")
    else:
        default_n = args.n
    default_i = args.i
    n_incr = args.incr
    n_min = args.min
    default_core = args.core
    default_profile = args.profile
    default_chunk = args.chunk
    default_arch = args.arch
    default_best = args.best_parallel
    if args.name:
        default_suffixe = '_' + args.name
    
    # main_cmd = "{} {} ./murb-se/build/bin/murb --nv --gf -i {}".format(default_nice, default_taskset, default_i)
    main_cmd = "{} {} ./murb-se/src/bin/murb --nv --gf -i {}".format(default_nice, default_taskset, default_i)

    print("Implementations:", implementations)
    print("default_n:", default_n)
    print("default_i:", default_i)
    print("n_incr:", n_incr)
    print("n_min:", n_min)
    print("default_core:", default_core)
    print("default_profile:", default_profile)
    print("default_chunk:", default_chunk)
    print("default_arch:", default_arch)
    
    print("-------------------------------------------")

# Exécuter la commande pour chaque implémentation et récupérer les FPS
def general_bench():
    print('general benchmark')
    for impl in implementations:
        
        fps_bodies = {}
        flop_bodies = {}
        mem_bodies = {}
        
        
        for n_bodies in range(n_min,default_n+1,n_incr) :
            
            # print(impl,n_bodies,'bodies')
            
            # cmd = f"OMP_SCHEDULE=\"{default_profile},{default_chunk}\" {main_cmd} -n {n_bodies} --im {impl}"
            cmd = "OMP_SCHEDULE=\"{},{}\" {} -n {} --im {}".format(default_profile, default_chunk, main_cmd, n_bodies, impl)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            # mem_bodies[n_bodies] = get_memory_usage(t,process.pid) / (1024.0 * 1024.0) # En MB
            mem_bodies[n_bodies] = 0.0
            # print('memory', mem_bodies[n_bodies], 'MB')
            
            output, _ = process.communicate()
            output = output.decode('utf-8')
            # print(output)

            _, fps_bodies[n_bodies], flop_bodies[n_bodies] = getMs_Fps_GFlop(output)
            # print('fps',fps_bodies[n_bodies])
            
            sys.stdout.flush()
            # sys.stdout.write(f"\r{impl} {n_bodies} bodies\t{mem_bodies[n_bodies]} MB\t{fps_bodies[n_bodies]} FPS\t{flop_bodies[n_bodies]} GFlops/s")
            sys.stdout.write("\r{} {} bodies\t{} MB\t{} FPS\t{} GFlops/s".format(impl, n_bodies, mem_bodies[n_bodies], fps_bodies[n_bodies], flop_bodies[n_bodies]))
        
        print("\n")
        fps_results[impl] = fps_bodies
        flop_results[impl] = flop_bodies
        mem_results[impl] = mem_bodies
    print('done')




#compare avec la version optim (la plus rapide)
def single_optim_bench():
    # print('cores',default_core,'chunk',default_chunk)
    global time_single_results, fps_single_results
    # cmd = f"{main_cmd} -n {default_n} --im {'cpu+optim'}" 
    cmd = "{} -n {} --im {}".format(main_cmd, default_n, 'cpu+optim')
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    output, _ = process.communicate()
    output = output.decode('utf-8')
    # print(output)

    time_single_results,fps_single_results,_ = getMs_Fps_GFlop(output);



#temps d'éxécution sur plusieurs cores
def per_core_bench():
    for n_cores in range(1,default_core+1):
        # cmd = f"OMP_NUM_THREADS={n_cores} OMP_SCHEDULE=\"{default_profile},{default_chunk}\" {main_cmd} -n {default_n} --im {'cpu+omp'}"
        cmd = "OMP_NUM_THREADS={} OMP_SCHEDULE=\"{},{}\" {} -n {} --im {}".format(n_cores, default_profile, default_chunk, main_cmd, default_n, 'cpu+omp')
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        output, _ = process.communicate()
        output = output.decode('utf-8')
        # print(output)
        
        time_cores_results[n_cores],fps_cores_results[n_cores],_ = getMs_Fps_GFlop(output);
        
        sys.stdout.flush()
        # sys.stdout.write(f"{n_cores}/{default_core} cores")
        sys.stdout.write("{}/{} cores".format(n_cores, default_core))
        



# Scalabilité Faible
def wscallability_bench():
    for n_cores in range(1, default_core + 1):
        print(n_cores,'cores',default_n*n_cores,'ressources')
        # cmd = f"OMP_NUM_THREADS={n_cores} OMP_SCHEDULE=\"{default_profile},{default_chunk}\" {main_cmd} -n {default_n*n_cores} --im {'cpu+omp'}"
        cmd = "OMP_NUM_THREADS={} OMP_SCHEDULE=\"{},{}\" {} -n {} --im {}".format(n_cores, default_profile, default_chunk, main_cmd, default_n * n_cores, 'cpu+omp')
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        output, _ = process.communicate()
        output = output.decode('utf-8')
        # print(output)
        
        weak_scalability_times[n_cores],_,_ = getMs_Fps_GFlop(output);
        
        sys.stdout.flush()
        # sys.stdout.write(f"{n_cores}/{default_core} cores {default_n*n_cores} ressources")
        sys.stdout.write("{}/{} cores {} ressources".format(n_cores, default_core, default_n * n_cores))

def best_parallel():
    print('best parallel benchmark, profile and chunk arguments ignored\n')
    
    profiles = ["auto", "dynamic", "static", "guided", "runtime"]

def run_benchmark():
    # print('implementations',implementations,'iterations',default_i,'bodies',str(n_min)+':'+str(default_n))
    
    global fps_results
    global flop_results
    global mem_results 
    
    general_bench()
    
    saveFpsBodies(fps_results)
    saveGflopBodies(flop_results)
    saveGflopBodies(flop_results)
    # saveMemBodies(mem_results)
    saveFpsHisto(fps_results,1000)
    # saveMemHisto(mem_results,1000)
    
    print('parallel benchmark')
    
    #temps éxécution single core
    global fps_single_results
    global time_single_results
    
    single_optim_bench()
    
    global time_cores_results
    global fps_cores_results
    
    per_core_bench()
    
    saveSpeedup(time_single_results,time_cores_results)
    saveEfficiency(time_single_results,time_cores_results)
    savePerCore(time_single_results,time_cores_results)
    
    wscallability_bench()
    
    saveScalabilityEfficiency(time_single_results,weak_scalability_times)
    
    print('done')
    
set_defaults()
if default_arch == "jetson":
    default_taskset = 'taskset -c 1'
    default_suffixe = '_DENVER2'
    print("DENVER 2 CORE")
    run_benchmark()
    default_taskset = 'taskset -c 0'
    default_suffixe = '_A57'
    print("A57 CORE")
    run_benchmark()
elif default_best :
    best_parallel()
else:
    run_benchmark()