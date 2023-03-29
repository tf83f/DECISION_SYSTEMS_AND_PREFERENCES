#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:00:58 2023

@author: favolithomas
"""
from typing import TypedDict
import gurobipy
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression


class Employee:
    def __init__(self, name, qualifications, vacations):
        self.name = name
        self.qualifications = qualifications
        self.vacations = vacations

class Job:
    def __init__(self,name,gain,due_date,daily_penalty,working_days_per_qualification):
        self.name = name
        self.gain = gain
        self.due_date = due_date
        self.daily_penalty = daily_penalty
        self.working_days_per_qualification = working_days_per_qualification

class ProblemData(TypedDict):
    horizon: int
    qualifications: list
    staff: list
    jobs: list

def get_data(path):
    file: str = path
    with open(file) as f:
        data: dict = json.load(f)
    data["staff"] = [Employee(name=employee["name"],qualifications=employee["qualifications"],vacations=employee["vacations"],) for index, employee in enumerate(data["staff"])]
    data["jobs"] = [Job(name=job["name"],gain=job["gain"],due_date=job["due_date"],daily_penalty=job["daily_penalty"],working_days_per_qualification=job["working_days_per_qualification"],)for index, job in enumerate(data["jobs"])]
    return data

def model_optimise(data, max_nb_projects_per_employee, max_duration_project):
    model: gurobipy.Model = gurobipy.Model("Optimisation d'emploi du temps")
    
            # Paramètres
            
    # L’horizon de temps, h ∈ N\ {0}
    HORIZON: list[int] = list(range(1, data["horizon"] + 1)) 
    
    # L’ensemble des qualifications
    QUALIFICATION: list[str] = data["qualifications"]
    
    # L’ensemble du personnel
    STAFF: list[Employee] = data["staff"]
   
    # L'ensemble des projets
    JOBS: list[Job] = data["jobs"]
    
    # Les qualificatios de l'ensemble des employés 
    EMPLOYEE_QUALIFICATION = {employee: employee.qualifications for employee in STAFF}
    
    # Les vacances de l'ensemble des employés
    EMPLOYEE_VACATION  = {employee: employee.vacations for employee in STAFF}
   
    # Les qualifications requises pour l'ensemble des projets
    PROJECT_QUALIFICATION_REQUIREMENT = {job: list(job.working_days_per_qualification.keys()) for job in JOBS}
    
    # Le décompte jours/salariés pour l'ensemble des qualitications
    PROJECT_QUALIFICATION_DAYS = {job: {qualification: job.working_days_per_qualification[qualification] for qualification in job.working_days_per_qualification}for job in JOBS}
    
    # L'ensemble des gains des projets
    PROJECT_GAIN = {job: job.gain for job in JOBS}
    
    # L'ensemble des pénalités des projets
    PROJECT_PENALITY = {job: job.daily_penalty for job in JOBS}
    
    # L'ensemble des due dates des projets
    PROJECT_DUE_DATE = {job: job.due_date for job in JOBS}

        # Variables de décision
            
    # renvoie 1 si le salarié i réalise une qualification k pour le projet j pendant la journée t, 0 sinon
    X = model.addVars([(i, j, k, t) for i in STAFF for j in JOBS for k in QUALIFICATION for t in HORIZON],vtype=gurobipy.GRB.BINARY, name="X",)
    
    # renvoie 1 si le projet j est réalisé, 0 sinon
    Y = model.addVars(JOBS, vtype=gurobipy.GRB.BINARY, name="Y")
    
    # renvoie le nombre de jours de retard pour le projet j
    L = model.addVars(JOBS, vtype=gurobipy.GRB.INTEGER, name="L")
    
    # renvoie la date de fin du projet j
    E = model.addVars(JOBS, vtype=gurobipy.GRB.INTEGER, name="E", lb=min(HORIZON), ub=max(HORIZON))
    
    # renvoie 1 si le salarié i à travaillé sur le projet j
    Z = model.addVars([(i, j) for i in STAFF for j in JOBS], vtype=gurobipy.GRB.BINARY,name="Z",)
   
    # renvoie la date de début du projet j
    B = model.addVars(JOBS, vtype=gurobipy.GRB.INTEGER, name="B", lb=min(HORIZON), ub=max(HORIZON))

    model.update()

        # Contraintes

    model.addConstrs((gurobipy.quicksum(X[i, j, k, t] for j in JOBS for k in QUALIFICATION) <= 1 for i in STAFF for t in HORIZON), name="Contrainte sur le nombre de taches / salarié / jour ")
    model.addConstrs((gurobipy.quicksum(X[i, j, k, t] for j in JOBS for k in QUALIFICATION) == 0 for i in STAFF for t in EMPLOYEE_VACATION[i]),name="Contrainte de congé")
    model.addConstrs((X[i, j, k, t] == 0 for i in STAFF for j in JOBS for k in QUALIFICATION for t in HORIZON if k not in EMPLOYEE_QUALIFICATION[i] or k not in PROJECT_QUALIFICATION_REQUIREMENT[j]),name="Contrainte de qualification du personnel")
    model.addConstrs((Y[j] * PROJECT_QUALIFICATION_DAYS[j][k] <= gurobipy.quicksum(X[i, j, k, t] for i in STAFF for t in HORIZON) for j in JOBS for k in PROJECT_QUALIFICATION_REQUIREMENT[j]),name="Contrainte de couverture des qualifications projet")
    model.addConstrs((X[i, j, k, t] * t <= E[j] for i in STAFF for j in JOBS for k in QUALIFICATION for t in HORIZON),)
    model.addConstrs((E[j] - PROJECT_DUE_DATE[j] <= L[j] for j in JOBS), name="Contrainte de pénalité")

        # Fonctions objectifs
    
        # Maximiser le bénéfice total de l’entreprise
    model.setObjective(gurobipy.quicksum(Y[j] *PROJECT_GAIN[j] - L[j] * PROJECT_PENALITY[j] for j in JOBS),gurobipy.GRB.MAXIMIZE,)
    
    # Minimiser le nombre de projets par salariés = Minimize ( max( sum(Z(i,j) ∀i ∈ STAFF) ) ). Cela sera rendu possible en convertissant la fonction de minimisation en une contrainte
    model.addConstrs((Z[i, j] >= X[i, j, k, t] for i in STAFF for j in JOBS for k in QUALIFICATION for t in HORIZON))
    model.addConstrs((gurobipy.quicksum(Z[i, j] for j in JOBS) <= max_nb_projects_per_employee for i in STAFF), name="Contrainte pour satisfaire l'objectif de minimiser le nombre de projets sur lesquels un quelconque collaborateur est affecté")

    # Minimiser  la durée de réalisation du projet le plus long = Minimize( max (E-B) ∀j ∈ JOBS) ). Cela sera rendu possible en convertissant la fonction de minimisation en une contrainte
    model.addConstrs((B[j] * X[i, j, k, t] <= t for i in STAFF for j in JOBS for k in QUALIFICATION for t in HORIZON),)
    model.addConstrs((E[j] - B[j] <= max_duration_project - 1 for j in JOBS), name="Contrainte pour satisfaire l'objectif que les projets soient réalisés dans un nombre limités de jours consécutifs")
   
    model.optimize()
    return model.objVal

def get_solution(path):
    data = get_data(path)
    max_projet = len(data["jobs"])
    max_durée = data["horizon"]
    solution = []
    while max_projet > 0:
        while max_durée > 0:
            try:
                result = model_optimise(data, max_projet, max_durée)
            except:
                break
            solution.append([result, max_projet, max_durée])
            max_durée -= 1
        max_projet -=1
        max_durée = data["horizon"]
    print("Les solutions sont")
    print(np.array(solution))
    return solution
 
def plot_surface(points):
    fig = plt.figure()
    obj1 = [point[0] for point in points]
    obj2 = [point[1] for point in points]
    obj3 = [point[2] for point in points]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(obj2, obj3, obj1)
    ax.set_xlabel('Nombre maximum de projet par salarié')
    ax.set_ylabel('Durée maximale de projet')
    ax.set_zlabel('Résultat financier')
    plt.title('Surface de solution')
    plt.show()

def get_non_dominated_solutions(solutions):
    non_dominated_solutions = []
    for i in range(len(solutions)):
        is_dominated = False
        for j in range(len(solutions)):
            if (solutions[j][0] >= solutions[i][0] and 
                solutions[j][1] <= solutions[i][1] and 
                solutions[j][2] <= solutions[i][2] and 
                j != i):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated_solutions.append(solutions[i])
    print("Les solutions non dominées")
    for solution in non_dominated_solutions:
        print(solution)
    return non_dominated_solutions

def get_best_solution(solutions):
    reference_point = [max([s[0] for s in solutions]), min([s[1] for s in solutions]), 
                       min([s[2] for s in solutions])]
    best_solution = solutions[0]
    best_distance = float("inf")
    for solution in solutions:
        distance = (solution[0] - reference_point[0])**2 + (solution[1] - 
                reference_point[1])**2 + (solution[2] - reference_point[2])**2
        if distance < best_distance:
            best_distance = distance
            best_solution = solution
    print("La meilleure solution est: " ,best_solution)
    return best_solution

def get_inacceptable(solutions):
    inacceptable = []
    for i in range(len(solutions)):
        dominated = False
        for j in range(len(solutions)):
            if (solutions[j][0] <= solutions[i][0] and 
                solutions[j][1] >= solutions[i][1] and 
                solutions[j][2] >= solutions[i][2] and 
                j != i):
                dominated = True
                break
        if not dominated:
           inacceptable.append(solutions[i])
    print("Les solutions innaceptables")
    for solution in inacceptable:
        print(solution)
    return inacceptable

def get_correcte(solutions, satisfaisante, inacceptable):
    correcte = []
    for i in solutions:
        if i not in satisfaisante and i not in inacceptable:
            correcte.append(i)
    print("Les solutions correctes")
    for solution in  correcte:
        print(solution)
    return correcte

def plot_classified_solution(satisfaisante, inacceptable, correcte):
    fig = plt.figure()
    obj1 = [point[0] for point in satisfaisante]
    obj2 = [point[1] for point in satisfaisante]
    obj3 = [point[2] for point in satisfaisante]
    
    obj4 = [point[0] for point in inacceptable]
    obj5 = [point[1] for point in inacceptable]
    obj6 = [point[2] for point in inacceptable]
    
    obj7 = [point[0] for point in correcte]
    obj8 = [point[1] for point in correcte]
    obj9 = [point[2] for point in correcte]
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(obj2, obj3, obj1, c='r', marker='o', label='satisfaisante')
    ax.scatter(obj5, obj6, obj4, c='b', marker='^', label='inacceptable')
    ax.scatter(obj8, obj9, obj7, c='g', marker='s', label='correcte')
    ax.set_xlabel('Nombre maximum de projet par salarié')
    ax.set_ylabel('Durée maximale de projet')
    ax.set_zlabel('Résultat financier')
    plt.title('Classification des solutions')
    plt.legend(loc='upper left')
    plt.show()

def get_weight(satisfaisante, inacceptable, correcte):
    X1 = np.array(satisfaisante)
    y1 = np.full(X1.shape[0],1) 
    X2 = np.array(inacceptable)
    y2 = np.full(X2.shape[0],2) 
    X3 = np.array(correcte)
    y3 = np.full(X3.shape[0],3) 
    X = np.concatenate((X1, X2, X3))
    y = np.concatenate((y1, y2, y3))
    clf = LogisticRegression(multi_class='auto', solver='lbfgs', random_state=0, max_iter=1000)
    clf.fit(X, y)
    weights = clf.coef_
    print("Weights:", weights[0])

  
path = "/Users/favolithomas/Desktop/CS/SDP/New SPD/medium_instance.json"
print("------------------------------------------------------------------------")
solution = get_solution(path)
plot_surface(solution)
print("------------------------------------------------------------------------")
non_dominated_solutions = get_non_dominated_solutions(solution)
get_best_solution(non_dominated_solutions)
print("------------------------------------------------------------------------")
satisfaisante = non_dominated_solutions
print("------------------------------------------------------------------------")
inacceptable = get_inacceptable(solution)
print("------------------------------------------------------------------------")
correcte = get_correcte(solution, satisfaisante, inacceptable)
print("------------------------------------------------------------------------")
plot_classified_solution(satisfaisante, inacceptable, correcte)
get_weight(satisfaisante, inacceptable, correcte)