import sys
import os
import re
import copy

good = list(range(1,10))

def toInt(val):
    if val == '_':
        return 0
    else:
        return int(val)
    
def read_init(filename):
    arr = []
    f = open(filename, 'r')
    for line in f:
        line = line[:-1]
        if line:
            line_arr = [toInt(v) for v in list(line)]
            arr.append(line_arr)
    return arr

def row(arr, rowno):
    return arr[rowno]

def col(arr, colno):
    return [line_arr[colno] for line_arr in arr]

def blk(arr, blkno):
    rowstart = int(blkno / 3) * 3
    colstart = (blkno % 3) * 3
    return [arr[i][j] for i in range(len(arr)) for j in range(len(arr[i]))
            if i >= rowstart and i < rowstart + 3 and j >= colstart and j < colstart + 3]

def checkrow(arr, rowno):
    return sorted(row(arr, rowno)) == good

def checkcol(arr, colno):
    return sorted(col(arr, colno)) == good

def checkblk(arr, blkno):
    return sorted(blk(arr, blkno)) == good

def checkall(arr, debug=False):
    for i in range(9):
        if not checkrow(arr, i):
            if debug:
                print("Failed row %d" % (i))
            return False
        if not checkcol(arr, i):
            if debug:
                print("Failed col %d" % (i))
            return False
        if not checkblk(arr, i):
            if debug:
                print("Failed block %d" % (i))
            return False
    return True

def possibleValues(arr, i, j):
    if arr[i][j] != 0:
        return [arr[i][j]]
    row_possible = set(good).difference(row(arr, i))
    col_possible = set(good).difference(col(arr, j))
    blkno = 3 * int(i/3) + int(j/3)
    blk_possible = set(good).difference(blk(arr, blkno))
    all_possible = row_possible.intersection(col_possible).intersection(blk_possible)
    return sorted(list(all_possible))

def resolvedCount(arr):
    l = [arr[i][j] for i in range(len(arr)) for j in range(len(arr[i]))
          if arr[i][j] != 0]
    return len(l)

def computeAllPossibleValues(arr):
    # First create a copy
    newarr = copy.deepcopy(arr)
    fully_resolved = 0
    partially_resolved = 0
    pos_arr = [[] for x in range(len(arr))]
    for i in range(len(arr)):
        pos_arr[i] = [[] for x in range(len(arr[i]))]
        for j in range(len(arr[i])):
            pos_values = possibleValues(newarr, i, j)
            if len(pos_values) == 0:
                raise Exception('Unslovable position reached...')
            elif (len(pos_values) == 1):
                fully_resolved += 1
                newarr[i][j] = pos_values[0]
                pos_arr[i][j] = pos_values
            else:
                partially_resolved += 1
                pos_arr[i][j] = pos_values
    return newarr, fully_resolved, partially_resolved, pos_arr

def quicksolve(arr):
    already_resolved = resolvedCount(arr)
    n = copy.deepcopy(arr)
    while (True):
        newarr, f, p, p_arr = computeAllPossibleValues(n)
        if f == already_resolved:
            break
        already_resolved = f
        n = newarr
    return newarr, p_arr

def printarr(arr):
    print('-' * 37)
    for r in arr:
        sys.stdout.write('|')
        for cell_value in r:
            sys.stdout.write(' ' + (' ' if cell_value == 0 else str(cell_value)) + ' |')
        print('')
        print('-' * 37)

def compute_next_pos(pa, tp):
    np = copy.deepcopy(tp)
    tp_ind = tp[0] * 100 + tp[1] * 10 + tp[2]
    for i in range(len(pa)):
        for j in range(len(pa[i])):
            for k in range(len(pa[i][j])):
                pos_ind = i * 100 + j * 10 + k
                if pos_ind > tp_ind and len(pa[i][j]) > 1:
                    return [i, j, k]
    return None

def solve_until_done(arr, pos_arr, trial_pos = [-1,-1,-1], debug = True):

    next_pos = compute_next_pos(pos_arr, trial_pos)
    call_stack = list()
    call_stack.append((arr, pos_arr, next_pos))
    t_arr = copy.deepcopy(arr)
    n_pos_arr = copy.deepcopy(pos_arr)
    while next_pos is not None:
        i = next_pos[0]
        j = next_pos[1]
        k = next_pos[2]
        t_arr[i][j] = n_pos_arr[i][j][k]
        if debug:
            print("Trying %d in position %d,%d" % (n_pos_arr[i][j][k], i, j))
            printarr(t_arr)
            print(t_arr)
            print(n_pos_arr)
        try:
            n_arr, n_pos_arr = quicksolve(t_arr)
            if checkall(n_arr):
                return n_arr
            else:
                next_pos = compute_next_pos(n_pos_arr, [-1, -1, -1])
                t_arr = copy.deepcopy(n_arr)
                call_stack.append((n_arr, n_pos_arr, next_pos))
        except Exception as e:
            if str(e) == 'Unslovable position reached...':
                last_arr, last_pos_arr, last_pos = call_stack.pop()
                t_arr = copy.deepcopy(last_arr)
                next_pos = compute_next_pos(last_pos_arr, last_pos)
                n_pos_arr = copy.deepcopy(last_pos_arr)
                if debug:
                    print("Compute Next Pos:", last_pos, next_pos)

def solve_from_file(filename, prettyprint=True):
    s = read_init(filename)
    sq, pa = quicksolve(s)
    s_final = solve_until_done(sq, pa, debug=False)
    if prettyprint:
        printarr(s_final)
    return s_final
