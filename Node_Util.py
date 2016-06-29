
arr = [3,1,4,6,8,7,10,13,14]
arr2 = sorted(arr)
# Now build a BST from the sorted array

class Node(object):
  def __init__(self, val, left, right):
    self.val = val
    self.left = left
    self.right = right
    
  def __str__(self):
    return "(" + str(left) + ")" + str(val) + "(" + str(right) + ")"
  
def test():
  node1 = Node(2, None, None)
  node2 = Node(10, None, None)
  node3 = Node(5, node1, node2)
  print(node3)

test()

def buildTestBst():
  node1 = Node(3, Node(1, None, Node), Node(4, None, None))
  node2 = Node(7, Node(8, None, None), Node(10, None, None))
  node3 = Node(6, node1, node2)
  root = Node(13, node3, Node(14, None, None)
  return root

def preOrderTraversal(root):
  if (root.left):
    preOrderTraversal(root.left)
  if (root.right):
    preOrderTraversal(root.right)
  print(root.val)

def buildArrayPreOrderTraversal(root, arr):
  if (root.left):
    buildArrayPreOrderTraversal(root.left, arr)
  if (root.right):
    buildArrayPreOrderTraversal(root.right, arr)
  arr.append(root.val)

def buildArrayInOrderTraversal(root, arr):
  if (root.left):
    buildArrayInOrderTraversal(root.left, arr)
  arr.append(root.val)
  if (root.right):
    buildArrayInOrderTraversal(root.right, arr)


