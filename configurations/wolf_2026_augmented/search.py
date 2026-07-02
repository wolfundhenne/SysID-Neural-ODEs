from configurations.search_types import categorical, log_float


SEARCH_SPACES = {
    'rk4_ieee9_restcn_noangle': {
        'model': {
            'dynamics': {
                'act_fct': categorical(['softplus', 'silu', 'gelu']),
                'n_layers': categorical([1, 2, 4]),
                'n_hidden': categorical([32, 64, 128, 256, 512, 1024]),
            },
            'encoder': {
                'n_hidden': categorical([8, 16, 32, 64, 128, 256, 512]),
            },
        },
        'train': {
            'learning_rate': log_float(0.0001, 0.01),
            'grad_clip': categorical([0.1, 1.0, 10.0, float('inf')]),
        },
    },
    'rk4_ieee30_restcn_noangle': {
        'model': {
            'dynamics': {
                'act_fct': categorical(['softplus', 'silu', 'gelu']),
                'n_layers': categorical([1, 2, 4]),
                'n_hidden': categorical([32, 64, 128, 256, 512, 1024]),
            },
            'encoder': {
                'n_hidden': categorical([8, 16, 32, 64, 128, 256, 512]),
            },
        },
        'train': {
            'learning_rate': log_float(0.0001, 0.01),
            'grad_clip': categorical([0.1, 1.0, 10.0, float('inf')]),
        },
    },
    'rk4_ieee39_restcn_noangle': {
        'model': {
            'dynamics': {
                'act_fct': categorical(['softplus', 'silu', 'gelu']),
                'n_layers': categorical([1, 2, 4]),
                'n_hidden': categorical([32, 64, 128, 256, 512, 1024]),
            },
            'encoder': {
                'n_hidden': categorical([8, 16, 32, 64, 128, 256, 512]),
            },
        },
        'train': {
            'learning_rate': log_float(0.0001, 0.01),
            'grad_clip': categorical([0.1, 1.0, 10.0, float('inf')]),
        },
    },
}
