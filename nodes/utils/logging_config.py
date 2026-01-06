# logging_config.py

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'short': {
            'class': 'nodes.utils.log_utils.ShortPathFormatter',
            'format': '%(levelname)-8s %(asctime)s [%(short_name)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'standard': {
            'format': '%(levelname)-8s %(asctime)s [%(name)s] %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'short',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}