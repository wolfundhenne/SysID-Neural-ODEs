from configurations.search_types import categorical, log_float


SEARCH_SPACES = {
    'ieee9_graph_partiallatent': {
        'model': {
            'dynamics': {
                'n_layers': categorical([1, 2, 3]),
                'n_hidden': categorical([128, 256, 512]),
            },
            'encoder': {
                'n_hidden': categorical([128, 256, 512]),
            },
            'graph': {
                'n_hidden_msg': categorical([128, 256, 512]),
                'n_layers_msg': categorical([1, 2, 3]),
            },
        },
        'train': {
            'learning_rate': log_float(0.0001, 0.01),
        },
    },
    'ieee9_monolith_partiallatent': {
        'model': {
            'dynamics': {
                'n_layers': categorical([1, 2, 3]),
                'n_hidden': categorical([128, 256, 512]),
            },
            'encoder': {
                'n_hidden': categorical([128, 256, 512]),
            },
        },
        'train': {
            'learning_rate': log_float(0.0001, 0.01),
        },
    },
}
