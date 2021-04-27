# Examples and tutorials

A collection of technical notebooks to illustrate data analysis concepts.

## Testing
pytest and nbval are used to validate that the notebooks are running without any errors:
```pytest --nbval-lax --current-env```

`--nbval-lax` checks that the notebooks execute without errors. `--curent-env` is required to execute the notebooks using the active environment rather than the environment specified in each notebook cell.

