import random
from math import pow, log2, cos, pi, sin
import logging
from operator import attrgetter
from copy import deepcopy, copy
import numpy
import time
import csv
from time import time
import numba
from lib.models import Individual, Test


@numba.jit(nopython=True, fastmath=True)
def random_real(range_a,  range_b,  precision):
    prec = pow(10, precision)
    return numpy.round(random.randrange(range_a * prec, (range_b) * prec + 1)/prec, precision)


@numba.jit(nopython=True, fastmath=True)
def power_of_2(range_a,  range_b,  precision):
    return int(numpy.rint(numpy.log2(((range_b - range_a) * (1/pow(10, -precision)) + 1))))

@numba.jit(fastmath=True)
def real_to_int(real,  range_a,  range_b,  power):
    return int(numpy.rint((1/(range_b-range_a)) * (real - range_a) * ((pow(2, power)-1))))


@numba.jit(nopython=True, fastmath=True)
def bin_to_int(binary):
    out = 0
    for bit in binary:
        out = (out << 1) | bit
    return out

@numba.jit(nopython=True, fastmath=True)
def int_to_bin(integer, power):
    bin_temp = []
    for i in range(power):
        i = power-i-1
        k = integer >> i
        if (k & 1):
            bin_temp.append(1)
        else:
            bin_temp.append(0)
    return bin_temp

@numba.jit(nopython=True, fastmath=True)
def int_to_real(integer,  range_a,  range_b, precision, power):
    return numpy.round(range_a + ((range_b - range_a) * integer)/(pow(2, power)-1), precision)


@numba.jit(nopython=True, fastmath=True)
def func(real):
    return numpy.mod(real, 1) * (cos(20.0 * pi * real) - sin(real))

@numba.jit(nopython=True, fastmath=True)
def get_individual(range_a, range_b, precision, power):
    real = random_real(range_a, range_b, precision)
    int_from_real = real_to_int(real, range_a, range_b, power)
    return int_to_bin(int_from_real, power)
     

@numba.jit(nopython=True, fastmath=True)
def new_individuals(bins, new_bins, new_fxs, range_a, range_b, precision, power, generations_number):
    for bit in numpy.arange(power):
        new_bins[bit] = bins
        new_bins[bit, bit] = 1 - new_bins[bit, bit]
        new_fxs[bit] = func(int_to_real(bin_to_int(new_bins[bit]), range_a,  range_b, precision, power))


@numba.jit(nopython=True)
def evolution(range_a, range_b, precision, generations_number):
    iteration = 0
    power = power_of_2(range_a, range_b, precision)
    best_binary = numpy.empty((generations_number,power), dtype=numpy.int32)
    best_reals = numpy.empty(generations_number, dtype=numpy.double)
    best_fxs = numpy.empty(generations_number, dtype=numpy.double)
    new_individuals_bins = numpy.empty((power, power), dtype=numpy.int32)
    new_individuals_fxs = numpy.empty(power, dtype=numpy.double)

    while iteration < generations_number:
        local = False
        best_binary[iteration] = get_individual(range_a, range_b, precision, power)
        best_reals[iteration] = int_to_real(bin_to_int(best_binary[iteration]), range_a, range_b, precision, power)
        best_fxs[iteration] = func(best_reals[iteration])

        print(best_binary[iteration])
        print(best_reals[iteration])
        print(best_fxs[iteration])

        while not local:
            new_individuals(best_binary[iteration], new_individuals_bins, new_individuals_fxs, range_a, range_b, precision, power, generations_number)
            print(new_individuals_bins)
            local = True

        print("")
        iteration += 1

    return

'''
@numba.jit(nopython=True, fastmath=True)
def mutation(bins, individuals, power, tau):
    for bit in numpy.arange(1, power + 1):
        r = random.random()
        t = 1/pow(bit, tau)
        if r <= t:
            bins[individuals[bit-1]] = 1 - bins[individuals[bit-1]]


@numba.jit(nopython=True)
def get_evolution(individuals, bins, reals, fxs, best_fxs, new_bins, new_fxs, best_binary, best_real, range_a, range_b, precision, power, tau, generations_number):
    for i in numpy.arange(1, generations_number):
        bins[i] = bins[i-1]
        new_individuals(bins[i], new_bins, new_fxs, range_a, range_b, precision, power, generations_number)
        for bit in numpy.arange(power):
            individuals[bit,0] = bit+1
            individuals[bit,1] = new_fxs[bit]

        individuals_bins = numpy.argsort(-individuals[:, 1]).T 
        mutation(bins[i], individuals_bins, power, tau)

        reals[i] = int_to_real(bin_to_int(bins[i]), range_a, range_b, precision, power)
        fxs[i] = func(reals[i])

        if fxs[i] > best_fxs[i-1]:
            best_binary[i] = bins[i]
            best_real[i] = reals[i]
            best_fxs[i] = fxs[i]
        else:
            best_fxs[i] = best_fxs[i-1]
            best_binary[i] = best_binary[i-1]
            best_real[i] = best_real[i-1]

        new_bins = numpy.empty((power, power), dtype=numpy.int32)
        new_fxs = numpy.empty(power, dtype=numpy.double)
        individuals = numpy.empty((power,2), dtype=numpy.double)

@numba.jit(nopython=True)
def evolution(range_a, range_b, precision, tau, generations_number):
    power = power_of_2(range_a, range_b, precision)
    reals = numpy.empty(generations_number, dtype=numpy.double)
    bins = numpy.empty((generations_number, power), dtype=numpy.int32)
    fxs = numpy.empty(generations_number, dtype=numpy.double)
    best_fxs = numpy.empty(generations_number, dtype=numpy.double)
    best_binary = numpy.empty((generations_number,power), dtype=numpy.int32)
    best_real = numpy.empty(generations_number, dtype=numpy.double)
    new_bins = numpy.empty((power, power), dtype=numpy.int32)
    new_fxs = numpy.empty(power, dtype=numpy.double)
    individuals = numpy.empty((power,2), dtype=numpy.double)
    bins[0] = get_individual(range_a, range_b, precision, power)
    
    new_individuals(bins[0], new_bins, new_fxs, range_a, range_b, precision, power, generations_number)

    for bit in numpy.arange(power):
        individuals[bit,0] = bit+1
        individuals[bit,1] = new_fxs[bit]

    individuals_bins = numpy.argsort(-individuals[:, 1]).T 
    mutation(bins[0], individuals_bins, power, tau)
    reals[0] = int_to_real(bin_to_int(bins[0]), range_a, range_b, precision, power)
    fxs[0] = func(reals[0])
    
    best_binary[0] = bins[0]
    best_real[0] = reals[0]
    best_fxs[0] = fxs[0]

    get_evolution(individuals, bins, reals, fxs, best_fxs, new_bins, new_fxs, best_binary, best_real, range_a, range_b, precision, power, tau, generations_number)

    return best_binary, best_real, fxs, best_fxs




@numba.jit(nopython=True, fastmath=True)
def numba_avg(array):
    arr_len = len(array)
    summary = 0
    for i in numpy.arange(arr_len):
        summary += array[i]
    return summary / arr_len


@numba.jit(nopython=True, fastmath=True)
def test_tau(range_a, range_b, precision, generations_number):
    best_fxs = numpy.empty(100, dtype=numpy.double)
    result = numpy.empty((50,2), dtype=numpy.double)

    index = 0
    for tau_number in numpy.arange(0.1, 5.1, 0.1):
        for i in numpy.arange(100):
            _ , _, _, fx = evolution(range_a, range_b, precision, tau_number, generations_number)
            best_fxs[i] = fx[generations_number-1]

        result[index,0] = numpy.round(tau_number, 1)
        result[index,1] = numba_avg(best_fxs)
        index += 1

    return result

@numba.jit(nopython=True, fastmath=True)
def test_generation(range_a, range_b, precision, tau):
    best_fxs = numpy.empty(100, dtype=numpy.double)
    result = numpy.empty((40,2), dtype=numpy.double)

    index = 0
    for generations_number in numpy.arange(1000, 5001, 100):
        for i in numpy.arange(100):
            _ , _, _, fx = evolution(range_a, range_b, precision, tau, generations_number)
            best_fxs[i] = fx[generations_number-1]

        result[index,0] = generations_number
        result[index,1] = numba_avg(best_fxs)
        index += 1

    return result
'''