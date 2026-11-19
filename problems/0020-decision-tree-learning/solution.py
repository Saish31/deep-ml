import math
from collections import Counter,defaultdict

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    
    total = len(labels)
    counts = Counter(labels)
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
        
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    if not examples:
        return 0.0
        
    # 1. Starting entropy H(S)
    parent_labels = [ex[target_attr] for ex in examples]
    parent_entropy = calculate_entropy(parent_labels)
    
    # 2. Group target labels by the attribute's values
    subsets = defaultdict(list)
    for ex in examples:
        subsets[ex[attr]].append(ex[target_attr])
        
    # 3. Subtract the weighted entropy of each subset
    total = len(examples)
    weighted_entropy = 0.0
    for subset_labels in subsets.values():
        weight = len(subset_labels) / total
        weighted_entropy += weight * calculate_entropy(subset_labels)
        
    return parent_entropy - weighted_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    # Sort by count descending (-count), then label ascending (label)
    sorted_classes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return sorted_classes[0][0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [ex[target_attr] for ex in examples]
    
    # Base Case 1: All examples have the same class (pure leaf)
    if len(set(labels)) == 1:
        return labels[0]
        
    # Base Case 2: No attributes left to split on
    if not attributes:
        return majority_class(examples, target_attr)
        
    # Choose the attribute with the highest information gain
    # Strict '>' ensures ties go to the attribute appearing first in `attributes`
    best_attr = None
    best_gain = -1.0
    for attr in attributes:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
        
    # Build the decision node and recurse on sorted attribute values
    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]
    unique_values = sorted(set(ex[best_attr] for ex in examples))
    
    for val in unique_values:
        subset = [ex for ex in examples if ex[best_attr] == val]
        tree[best_attr][val] = learn_decision_tree(subset, remaining_attrs, target_attr)
        
    return tree