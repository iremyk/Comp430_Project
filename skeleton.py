##############################################################################
# This skeleton was created by Efehan Guner (efehanguner21@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys
from copy import deepcopy
from typing import Optional
import numpy as np
import time

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
        
    with open(DGH_file) as f:
        
        prev_row = {}
        results = []
        
        for row in f:
            row_indent = row.count('\t')
            row_name = row.strip()
            
            if (prev_row == {}):
                row_parent = None
            elif (prev_row.get('indent') < row_indent):
                row_parent = prev_row.get('name')
            elif (prev_row.get('indent') == row_indent):
                row_parent = prev_row.get('parent')
            else:
                search_row = results[-2]
                while (row_indent != search_row.get('indent')):
                    search_row = results[results.index(search_row) - 1]
                row_parent = search_row.get('parent')
            
            curr_row = {'name': row_name, 'indent': row_indent, 'parent': row_parent}
            results.append(curr_row)
            prev_row = curr_row
            
    return results


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs


def internal_nodes(DGH_list: list) -> list:
    """ Get the desired format of read DGH file and return internal nodes of it.

    Args:
        DGH_list (list): desired format of read DGH file.

    Returns:
        list: a list of internal nodes.
    """
    
    internal = []
    
    for node in DGH_list:
        if ((node.get('parent') not in internal) and (node.get('parent') != None)): 
            internal.append(node.get('parent'))
            
    return internal


def descendant_leaves(DGH_list: list, node: str) -> list:
    """ Get the desired format of read DGH file and the node name whose descendant leaves want 
    to be found and return descendants nodes of it.

    Args:
        DGH_list (list): desired format of read DGH file.
        node (str): the node name of whose descendant leaves want to be found.

    Returns:
        list: a list of descendant leaves.
    """
    
    descendants = []
    
    for n in DGH_list:
        if (n.get('parent') == node): 
            descendants.append(n.get('name'))
    
            if (len(descendant_leaves(DGH_list, n.get('name'))) != 0):
            
                for i in descendant_leaves(DGH_list, n.get('name')):
                    descendants.append(i)
    
    des_leaves = []                
    int_nodes = internal_nodes(DGH_list)    

    for d in descendants:
        if (d not in int_nodes):
            des_leaves.append(d)
    
    return des_leaves


def get_minimum_indent(DGH_folder: str, EC: list, node: str) -> int:
    """ Get the path to the directory containing DGH files, the list of EC dicts and the node name 
    where minimum indent of the given dataset will be searched and return minimum indent.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        EC (list): list of EC dicts.
        node (str): the node name where minimum indent of the given dataset will be searched

    Returns:
        int: minimum indent.
    """
    
    min_indent = 100000000000
    DGH = read_DGHs(DGH_folder).get(node)
    
    for row in EC:
        for d in DGH:
            if (row.get(node) == d.get('name')):
                if (d.get('indent') < min_indent):
                    min_indent = d.get('indent')
    
    return min_indent


def make_same_indent(DGH_folder: str, EC: list, min_indent: int, node: str) -> list:
    """ Get the path to the directory containing DGH files, the list of EC dicts, minimum indent 
    of the given dataset and the node name where operation will be applied and return operated EC.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        EC (list): list of EC dicts.
        min_indent (int): minimum indent of the given dataset.
        node (str): the node name where operation will be applied.

    Returns:
        list: operated EC.
    """
    
    DGH = read_DGHs(DGH_folder).get(node)
    
    for row in EC:
        for d in DGH:
            if (row.get(node) == d.get('name')):
                indent = d.get('indent')
                while (indent != min_indent):
                    row[node] = d.get('parent')
                    indent -= 1
    
    return EC


def is_generalized(EC: list, node: str) -> bool:
    """ Get the list of EC dicts and the node name where the generalization will be checked
    and return true if it is generalized.

    Args:
        EC (list): list of EC dicts.
        node (str): the node name where the generalization will be checked.

    Returns:
        bool: true if it is generalized.
    """
    
    is_generalize = True
    base_row = EC[0].get(node)
    
    if (base_row == 'Any'):
        return is_generalize
    else:
        for row in EC[1:]:
            if (row.get(node) != base_row):
                is_generalize = False
                break
        
    return is_generalize


def generalize(DGH_folder: str, EC: list, node: str) -> list:
    """ Get the path to the directory containing DGH files, the list of EC dicts and 
    the node name where the generalization will be made and return the generalized EC.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        EC (list): list of EC dicts.
        node (str): the node name where the generalization will be made.

    Returns:
        list: the generalized EC.
    """
    
    DGH = read_DGHs(DGH_folder).get(node)

    while (not is_generalized(EC, node)):
        for row in EC:  
            for d in DGH:
                if ((row.get(node) == d.get('name')) and (d.get('parent') != None)):
                    row[node] = d.get('parent')

    return EC


def dist(DGH_folder: str, r1: dict, r2: dict) -> float:
    """ Get the path to the directory containing DGH files and the records their index added whose distance will be 
    calculated as LM_cost and returned.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        r1 (dict): the first record will be used in distance calculation.
        r2 (dict): the second record will be used in distance calculation.        

    Returns:
        float: the distance as LM_cost.
    """
    
    DGHs = read_DGHs(DGH_folder)
    r1_copy = r1.copy()
    r2_copy = r2.copy()
    temp = [r1_copy, r2_copy]
    
    for qi in list(r1_copy.keys())[:-2]:
        temp = generalize(DGH_folder, temp, qi)
        
    LM_cost = 0
    qi_count = len(r1_copy) - 2
    r2_copy_items = list(temp[1].items())
    
    for j in range(qi_count):
            DGH_list = DGHs.get(r2_copy_items[j][0])
            node_count = len(DGH_list)
            int_nodes = internal_nodes(DGH_list)
            leave_count = node_count - len(int_nodes)

            if (r2_copy_items[j][1] in int_nodes):
                LM_cost += ((1 / qi_count) * ((len(descendant_leaves(DGH_list, r2_copy_items[j][1])) 
                                               - 1) / (leave_count - 1)))
                
    return LM_cost


def is_specialized_node(DGH_folder: str, special_node: dict, general_node: dict) -> bool: 
    """ Get the path to the directory containing DGH files and the nodes which will be checked for
    whether second one is generalized version of the first one and it will return true if it is.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        special_node (dict): the first record will be used in relation check.
        general_node (dict): the second record will be used in relation check.        

    Returns:
        bool: true if the general_node is generalized version of special_node.
    """
    
    DGHs = read_DGHs(DGH_folder)
    copy_special_node = special_node.copy()
    
    for name in list(general_node.keys())[:-1]:  
        DGH = DGHs.get(name)
        is_specialized = True
        
        while (is_specialized):
            if ((copy_special_node.get(name) == general_node.get(name)) or (general_node.get(name) == 'Any')):
                break
            
            if (copy_special_node.get(name) == 'Any'):
                is_specialized = False
                break
            
            for d in DGH:
                if (copy_special_node.get(name) == d.get('name')):
                    copy_special_node[name] = d.get('parent')
                    break
        
        if (not is_specialized):
            return False
        
    return True


def num_of_specialized_records(DGH_folder: str, dataset: list, general_node: dict) -> int:
    """ Get the path to the directory containing DGH files, dataset and a node whose child node
    number checked in dataset and return number of child node.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        dataset (list): the list of data where child nodes will be searched.
        general_node (dict): the node whose child node number checked in dataset.        

    Returns:
        int: number of child node in dataset.
    """
    
    count = 0
    for row in dataset:
        if (is_specialized_node(DGH_folder, row, general_node)):
            count += 1
    return count


def child_nodes(DGH_list: list, node: str) -> list:
    """ Get the desired format of read DGH file and the node name whose child nodes want 
    to be found and return child nodes of it.

    Args:
        DGH_list (list): desired format of read DGH file.
        node (str): the node name whose child nodes want to be found.

    Returns:
        list: a list of child nodes.
    """
    
    childs = []
    
    for d in DGH_list:
        if (d.get('parent') == node):
            childs.append(d.get('name'))
            
    return childs


def uniform_L1_norm(distribution: list) -> float:
    """ Get a distribution and return its distance from uniform distrubiton.

    Args:
        distribution (list): distribution of dataset.

    Returns:
        float: the distance from uniform distrubiton.
    """
    
    
    uniform = 1 / len(distribution)
    L1 = 0
    
    for i in distribution:
        L1 += abs(uniform - i)
    
    return L1


def create_max_generalized_node(names: list) -> dict:
    """ Get the DGH names and return the max generalized node's dict with these keys.

    Args:
        names (list): DGH names.

    Returns:
        dict: the max generalized node's dict whose keys are the names.
    """
    
    node = {}
    
    for n in names:
        node[n] = 'Any'
        
    return node


def is_specialization_possible(DGH_folder: str, dataset: list, rec: dict, name: str, k: int) -> (bool, list, list):  
    """ Get the path to the directory containing DGH files, a dataset, a node which will be 
    checked whether it's specialization possible under qi name given as parameter according to 
    k-anonymization and return true with rec's specialized nodes and number of records belong to 
    this nodes in the dataset.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.
        dataset (list): the list of data where specialization will be checked.
        rec (dict): the node which will be checked whether it's specialization possible. 
        name (str): the qi name.
        k (int): the k-value.

    Returns:
        bool: true if specialization possible.
        list: the list of specialized nodes.
        list: the list of number of records belongs to specialized nodes.
    """
    
    
    DGH = read_DGHs(DGH_folder).get(name)
    specialized_records = []
    
    for child in child_nodes(DGH, rec.get(name)):
        rec_copy = rec.copy()
        rec_copy[name] = child
        specialized_records.append(rec_copy)

    num_of_specialized_records = []

    if (len(specialized_records) > 0):
        for i in specialized_records:
            count = 0
            for row in dataset:
                if (is_specialized_node(DGH_folder, row, i)):
                    count += 1
            if (count < k):
                return (False, [], [])
            else:
                num_of_specialized_records.append(count)
    
        return (True, specialized_records, num_of_specialized_records)
    else:
        return (False, [], [])

##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    total_cost = 0
    
    for i in range(len(raw_dataset)):
        
        raw_data_row = list(raw_dataset[i].items())
        anonymized_data_row = list(anonymized_dataset[i].items())
        
        for j in range(len(raw_dataset[0]) - 1):
            
            if (raw_data_row[j][1] != anonymized_data_row[j][1]):
                raw_indent = -1
                anonymized_indent = -1
                
                for k in DGHs.get(raw_data_row[j][0]):
                    
                    if (k.get('name') == raw_data_row[j][1]):
                        raw_indent = k.get('indent')
                    elif (k.get('name') == anonymized_data_row[j][1]):
                        anonymized_indent = k.get('indent') 
                    
                    if ((raw_indent != -1) and (anonymized_indent != -1)):
                        total_cost += (raw_indent - anonymized_indent)
                        break
                                
    return total_cost


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    total_cost = 0
    qi_count = len(raw_dataset[0]) - 1
    
    
    for i in range(len(anonymized_dataset)):  
        anonymized_data_row = list(anonymized_dataset[i].items())
        
        for j in range(qi_count):
            DGH_list = DGHs.get(anonymized_data_row[j][0])
            node_count = len(DGH_list)
            int_nodes = internal_nodes(DGH_list)
            leave_count = node_count - len(int_nodes)
            
            if (anonymized_data_row[j][1] in int_nodes):
                total_cost += ((1 / qi_count) * ((len(descendant_leaves(DGH_list, anonymized_data_row[j][1])) - 1) 
                                                / (leave_count - 1)))
            
    return total_cost


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)):
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)
    np.random.shuffle(raw_dataset)

    clusters = []

    D = len(raw_dataset)
        
    cluster_count = int(D/k)
    
    for i in range(cluster_count):
        
        if (i != (cluster_count - 1)):
            clusters.append(raw_dataset[k*i : k*(i+1)])
        else:
            clusters.append(raw_dataset[k*i :])
            
    for c in clusters:
        for qi in list(c[0].keys())[:-2]:
            min_indent = get_minimum_indent(DGH_folder, c, qi)
            c = make_same_indent(DGH_folder, c, min_indent, qi)
            c = generalize(DGH_folder, c, qi)

    anonymized_dataset = [None] * D

    for cluster in clusters:
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)
    
    


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    
    for i in range(len(raw_dataset)):
        raw_dataset[i]['index'] = i

    unused = raw_dataset.copy()
    used = []
    
    while (len(unused) >= k):
        rec = unused[0]
        distances = []
        
        for ur in unused[1:]:
            distances.append(dist(DGH_folder, rec, ur))
        
        index_of_min = []
        
        for i in range(k - 1):
            min_dist = min(distances)
            index_of_min.append(distances.index(min_dist))
            distances[distances.index(min_dist)] = 100000000000
        
        real_indexs = [i+1 for i in index_of_min]
        
        EC = [rec]
        
        for ind in real_indexs:
            EC.append(unused[ind])
            
        for qi in list(EC[0].keys())[:-2]:
            min_indent = get_minimum_indent(DGH_folder, EC, qi)
            EC = make_same_indent(DGH_folder, EC, min_indent, qi)
            EC = generalize(DGH_folder, EC, qi)
        
        used.append(EC)
        
        decreasing_indexs = [0] + real_indexs
        decreasing_indexs.sort(reverse = True)
        
        for i in (decreasing_indexs):
            unused.pop(i)
            
    if (len(unused) > 0):
        for i in range(len(unused)):
            used[-1].append(unused[i])
        
        EC = used[-1]
        
        for qi in list(EC[0].keys())[:-2]:
            min_indent = get_minimum_indent(DGH_folder, EC, qi)
            EC = make_same_indent(DGH_folder, EC, min_indent, qi)
            EC = generalize(DGH_folder, EC, qi)
            
    anonymized_dataset = [None] * len(raw_dataset)

    for EC in used:
        for item in EC:
            anonymized_dataset[item['index']] = item
            del item['index']
    
    write_dataset(anonymized_dataset, output_file)
    
    


def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    
    max_generalized_node = create_max_generalized_node(list(raw_dataset[0].keys()))
    specialization_tree = [max_generalized_node]
    finalized_tree = []
    
    while (len(specialization_tree) > 0):
        temp_specialization_tree = specialization_tree.copy()
        for spec in specialization_tree:
            num_of_child_nodes = 1000
            child_nodes = []
            L1_dist = 0
            
            for qi in (list(DGHs.keys())):
                return_values = is_specialization_possible(DGH_folder, raw_dataset, spec, qi, k)
                if (return_values[0]):
                    if (len(return_values[1]) < num_of_child_nodes):
                        num_of_child_nodes = len(return_values[1])
                        child_nodes = return_values[1]
                        L1_dist = uniform_L1_norm(return_values[2])
                    elif (len(return_values[1]) == num_of_child_nodes):
                        temp_L1_dist = uniform_L1_norm(return_values[2])
                        if (temp_L1_dist < L1_dist):
                            num_of_child_nodes = len(return_values[1])
                            child_nodes = return_values[1]
                            L1_dist = uniform_L1_norm(return_values[2])
            
            temp_specialization_tree.remove(spec)
            
            if (num_of_child_nodes == 1000):
                finalized_tree.append(spec)
            else:
                for child in child_nodes:
                    temp_specialization_tree.append(child)
        
        specialization_tree = temp_specialization_tree.copy()

    for row in raw_dataset:
        for spec in finalized_tree:
            if (is_specialized_node(DGH_folder, row, spec)):
                temp = row.get(list(raw_dataset[0].keys())[-1])
                row = spec
                row[list(raw_dataset[0].keys())[-1]] = temp
                anonymized_dataset.append(row)
                
    write_dataset(anonymized_dataset, output_file)



# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

start_time = time.time()

function = eval(f"{algorithm}_anonymizer")
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, topdown]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = time.time()

print(f"Process time is {end_time - start_time}")

# Sample usage:
# python3 code.py random DGHs/ adult-hw1.csv result.csv 300