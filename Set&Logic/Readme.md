
# Set Theory: Union, Intersection, and If-Else Application

## Introduction
Set theory is a fundamental concept in mathematics that deals with the study of sets, which are collections of objects. It provides the foundation for many branches of mathematics, including probability, logic, and computer science.

## Union of Sets ( ∪ )
The union of two or more sets combines all the unique elements from the given sets.

### Mathematical Representation
If **A** and **B** are two sets:
A ∪ B = { x | x ∈ A or x ∈ B }

### Example
Let:
- A = {1, 2, 3}
- B = {3, 4, 5}

Then:

A ∪ B = {1, 2, 3, 4, 5}

---

## Intersection of Sets ( ∩ )
The intersection of two or more sets includes only the common elements present in all the given sets.

### Mathematical Representation
```
A ∩ B = { x | x ∈ A and x ∈ B }
```

### Example
Let:
- A = {1, 2, 3}
- B = {3, 4, 5}

Then:
```
A ∩ B = {3}
```

---

## If-Else Application in Set Theory
In programming, we often use conditional statements (`if-else`) to check set operations.

### Example: Python Code for Union and Intersection
```python
A = {1, 2, 3}
B = {3, 4, 5}

# Union
union_set = A | B  # or A.union(B)

# Intersection
intersection_set = A & B  # or A.intersection(B)

# If-Else Condition to check set membership
x = 3
if x in intersection_set:
    print(f"{x} is in the intersection of A and B")
else:
    print(f"{x} is not in the intersection")
```

### Output
```
3 is in the intersection of A and B
```

---

## Conclusion
Set theory plays a crucial role in mathematics and computer science. The **union** operation helps merge datasets, while the **intersection** operation helps find common elements. Using **if-else** conditions, we can apply logic to analyze set properties in programming.
```

