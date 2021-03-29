


class Tree():

	tree_nodes = []


	def __init__(self,initial_state):

		self.tree_nodes = []
		self.tree_nodes.append(initial_state)



#isn't a tree just a list of Nodes pointing to each other

class Node():
	

	def __init__(self,state,reward):

		self.visits = 0
		self.state = state
		self.reward = reward



#taken from google
class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)