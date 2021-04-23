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
def bin_to_real(binary,  range_a,  range_b, precision, power):
    out = 0
    for bit in binary:
        out = (out << 1) | bit
    return numpy.round(range_a + ((range_b - range_a) * out)/(pow(2, power)-1), precision)

@numba.jit(nopython=True, fastmath=True)
def func(real):
    return numpy.mod(real, 1) * (cos(20.0 * pi * real) - sin(real))

@numba.jit(nopython=True, fastmath=True)
def get_individual(range_a, range_b, precision, power):
    real = random_real(range_a, range_b, precision)
    int_from_real = real_to_int(real, range_a, range_b, power)
    return int_to_bin(int_from_real, power)
     

@numba.jit(nopython=True, fastmath=True)
def new_individuals(bins, new_bins, new_reals, new_fxs, range_a, range_b, precision, power, generations_number):
    for bit in numpy.arange(power):
        new_bins[bit] = bins
        new_bins[bit, bit] = 1 - new_bins[bit, bit]
        new_reals[bit] = bin_to_real(new_bins[bit], range_a,  range_b, precision, power)
        new_fxs[bit] = func(new_reals[bit])


@numba.jit(nopython=True)
def evolution(range_a, range_b, precision, generations_number, checkMax = False):
    power = power_of_2(range_a, range_b, precision)
    best_binary = numpy.empty((generations_number,power), dtype=numpy.int32)
    best_reals = numpy.empty(generations_number, dtype=numpy.double)
    best_fxs = numpy.empty(generations_number, dtype=numpy.double)
    local_binary = numpy.empty((generations_number,power), dtype=numpy.int32)
    local_reals = numpy.empty(generations_number, dtype=numpy.double)
    local_fxs = []
    local_fxs_list = []
    new_individuals_bins = numpy.empty((power, power), dtype=numpy.int32)
    new_individuals_fxs = numpy.empty(power, dtype=numpy.double)
    new_individuals_reals = numpy.empty(power, dtype=numpy.double)

    local = False
    found = False
    iteration = 0
    
    while iteration < generations_number:
        local = False
        local_binary[iteration] = get_individual(range_a, range_b, precision, power)
        local_reals[iteration] = bin_to_real(local_binary[iteration], range_a, range_b, precision, power)
        local_fxs.append(func(local_reals[iteration]))

        while not local:
            new_individuals(local_binary[iteration], new_individuals_bins, new_individuals_reals, new_individuals_fxs, range_a, range_b, precision, power, generations_number)
            index = numpy.argmax(new_individuals_fxs)
            
            if local_fxs[-1] < new_individuals_fxs[index]:
                local_fxs.append(new_individuals_fxs[index])
                local_reals[iteration] = new_individuals_reals[index]
                local_binary[iteration] = new_individuals_bins[index]
            else:
                local = True

        local_fxs_list.append(local_fxs[:])

        if iteration == 0:
            best_binary[iteration] = local_binary[iteration]
            best_reals[iteration] = local_reals[iteration]
            best_fxs[iteration] = local_fxs[-1]
        elif best_fxs[iteration-1] < local_fxs[-1]:
            best_binary[iteration] = local_binary[iteration]
            best_reals[iteration] = local_reals[iteration]
            best_fxs[iteration] = local_fxs[-1]
        else:
            best_binary[iteration] = best_binary[iteration-1]
            best_reals[iteration] = best_reals[iteration-1]
            best_fxs[iteration] = best_fxs[iteration-1]

        if checkMax:
            if(best_reals[iteration] == 10.999):
                found = True
                break

        local_fxs.clear()
        iteration += 1

    return best_reals, best_binary, best_fxs, local_fxs_list, iteration, found


@numba.jit(nopython=True, fastmath=True)
def test_generation(range_a, range_b, precision, generations):
    result = numpy.zeros(generations, dtype=numpy.int32)

    for i in numpy.arange(100000):
        _, _, _, _, iteration, found = evolution(range_a, range_b, precision, generations, True)
        result[iteration] += 1 if found else 0

    return result