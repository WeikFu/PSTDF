from javalang.ast import Node
from tree import BlockNode

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def get_sequence_BFS(root,sequence):
    queue = []
    queue.append(root)
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            token, children = get_token(node), get_children(node)
            if token not in ['Import', 'PackageDeclaration']:  # Statement subtree pruning, discarding useless nodes such as package references and comments
                sequence.append(token)
            # base case
            if token in ['ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement', 'IfStatement']:
                sequence.append('End')
            for j in children:
                queue.append(j)
    return sequence

def get_blocks_improved_BFS(root, block_seq):
    queue = []
    queue.append(root)
    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            name, children = get_token(node), get_children(node)
            useless = ['Import', 'PackageDeclaration']
            if name not in useless:
                if children != None:
                    logic = ['SwitchStatement', 'IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement']
                    if name in ['MethodDeclaration', 'ConstructorDeclaration']:
                        block_seq.append(BlockNode(node))
                        body = node.body
                        for child in body:
                            if get_token(child) not in logic and not hasattr(child, 'block'):
                                block_seq.append(BlockNode(child))
                            else:
                                queue.append(child)
                    elif name in logic:
                        block_seq.append(BlockNode(node))
                        for child in children[1:]:
                            token = get_token(child)
                            if not hasattr(node, 'block') and token not in logic + ['BlockStatement']:
                                block_seq.append(BlockNode(child))
                            else:
                                queue.append(child)
                            block_seq.append(BlockNode('End'))
                    elif name is 'BlockStatement' or hasattr(node, 'block'):
                        block_seq.append(BlockNode(name))
                        for child in children:
                            if get_token(child) not in logic:
                                block_seq.append(BlockNode(child))
                            else:
                                queue.append(child)
                    else:
                        for child in children:
                            queue.append(child)

