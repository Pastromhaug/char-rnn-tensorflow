import tensorflow as tf
from layers.passRNNCell import PassRNNCell
from layers.hwyRNNCell import HwyRNNCell
from layers.sparseRNNCell import SparseRNNCell
from layers.interRNNCell import InterRNNCell
from layers.dizzyRNNCell import DizzyRNNCell
from buildRotations import buildRotations

def buildRNNCell(args, drop, mask):
    if args.model == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif args.model == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif args.model == 'lstm':
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    elif args.model =='meta':
        cell_fn = MetaRNNCell
    elif args.model =='pass':
        cell_fn = PassRNNCell
    elif args.model == 'hwy':
        cell_fn = HwyRNNCell
    elif args.model in ['sparse', 'block']:
        cell_fn = SparseRNNCell
    elif args.model == 'inter':
        cell_fn = InterRNNCell
    elif args.model in 'dizzy':
        cell_fn = DizzyRNNCell
    else:
        raise Exception("model type not supported: {}".format(args.model))

    if args.model == 'meta':
        cell = cell_fn(args.rnn_size, args.ctrl_size)
    elif args.model == 'hwy':
        cell = cell_fn(args.rnn_size, drop=drop)
    elif args.model in ['sparse', 'block']:
        cell = cell_fn(args.rnn_size, mask=mask, sparsity=args.sparsity)
    elif args.model == 'inter':
        cell = cell_fn(args.rnn_size, mask=mask, sparsity=args.sparsity, drop=drop)
    elif args.model != 'dizzy':
        cell = cell_fn(args.rnn_size)

    if args.model == 'dizzy':
        rot_cells = []
        for i in range(args.num_layers):
            rotationsA = buildRotations(args.rnn_size, True, num_rots=args.num_rots, marker=str(i)+"A")
            rotationsB = buildRotations(args.rnn_size, True, num_rots=args.num_rots, marker=str(i)+"B")
            rot_cells.append(cell_fn(args.rnn_size, rotationsA, rotationsB))
        cell = tf.nn.rnn_cell.MultiRNNCell(rot_cells, state_is_tuple=True)

    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
    return cell
