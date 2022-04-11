import numpy as np
from sympy import *


X=np.asarray([
    [1,2],
    [3,4]
])
X_fake=np.asarray([
    [5,6],
    [7,8]
])
thetas=np.asarray([0.1,0.2])
x1, x2, t1, t2 = symbols('x1 x2 t1 t2', real=True)
Dx = 1/(1+exp(-(t1*x1-t2*x2-2)))

f_d = diff(Dx, t1)

def gan(Dx,X,X_fake,thetas):
    n = len(X)
    n_fake = len(X_fake)
    n_thetas = len(thetas)
    variable_dict={}
    discriminate_loss = ln(Dx)
    generate_fake_loss = ln(1-Dx)

    for i in range(len(X[0])):
        variable_dict['x'+str(i+1)]=symbols('x'+str(i+1), real=True)
    for i in range(n_thetas):
        variable_dict['t'+str(i+1)]=symbols('t'+str(i+1), real=True)

    print(f"variable dictionary:\n{variable_dict}\n")


    ex1_f = 0.0
    for i in range(n):
        subs_list = []
        for j in range(len(X[0])):
            subs_list.append((variable_dict['x' + str(j + 1)], X[i, j]))
        for j in range(n_thetas):
            subs_list.append((variable_dict['t' + str(j + 1)], thetas[j]))
        print(subs_list)
        ex1_f = ex1_f + (1/n) * discriminate_loss.subs(subs_list)
    print(f"the first term Expectation(ln(D(x))):\n{ex1_f}")

    ex2_f = 0.0
    for i in range(n_fake):
        subs_list = []
        for j in range(len(X_fake[0])):
            subs_list.append((variable_dict['x' + str(j + 1)], X_fake[i, j]))
        for j in range(n_thetas):
            subs_list.append((variable_dict['t' + str(j + 1)], thetas[j]))
        print(subs_list)
        ex2_f = ex2_f + (1 / n) * generate_fake_loss.subs(subs_list)
    print(f"the second term Expectation(ln(1-D(x))):\n{ex2_f}")

    print(f"Final result of V(D,G)= {ex1_f+ex2_f}")


def minibatch_GAN(k, num_iteration,learning_rate,Dx,X,X_fake,thetas):
    n = len(X)
    n_fake = len(X_fake)
    n_thetas = len(thetas)
    variable_dict = {}
    discriminate_loss = ln(Dx)
    generate_fake_loss = ln(1 - Dx)

    for i in range(len(X[0])):
        variable_dict['x' + str(i + 1)] = symbols('x' + str(i + 1), real=True)
    for i in range(n_thetas):
        variable_dict['t' + str(i + 1)] = symbols('t' + str(i + 1), real=True)

    print(f"variable dictionary:\n{variable_dict}\n")

    result_list = np.zeros(n_thetas)

    for train_iter in range(num_iteration):
        for iter in range(k):
            for i in range(n):
                subs_list_discriminator = []
                subs_list_generator = []

                for j in range(len(X[0])):
                    subs_list_discriminator.append((variable_dict['x' + str(j + 1)], X[i, j]))
                for j in range(n_thetas):
                    subs_list_discriminator.append((variable_dict['t' + str(j + 1)], thetas[j]))

                for j in range(len(X_fake[0])):
                    subs_list_generator.append((variable_dict['x' + str(j + 1)], X_fake[i, j]))
                for j in range(n_thetas):
                    subs_list_generator.append((variable_dict['t' + str(j + 1)], thetas[j]))

                for j in range(n_thetas):
                    ex1_f = 0.0
                    ex1_f += (1 / n) * diff(discriminate_loss,variable_dict['t' + str(j + 1)]).subs(subs_list_discriminator)
                    ex1_f += (1 / n) * diff(generate_fake_loss,variable_dict['t' + str(j + 1)]).subs(subs_list_generator)
                    result_list[j] = np.add(result_list[j],ex1_f)
            print(result_list)


            # update theta
            thetas = np.add(thetas, learning_rate*result_list)
            print(f"Updated discriminator: {thetas}")


minibatch_GAN(k=1, num_iteration=1,learning_rate=0.02, Dx=Dx,X = X,X_fake=X_fake,thetas=thetas)
# gan(Dx,X,X_fake,thetas)

# print(f_d.subs([(x1, X[0,0]), (x2,  X[0,1]), (t1, thetas[0]), (t2, thetas[1])]))