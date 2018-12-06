from augment import augmentMain


augmentMain( params , 'Linear' )


augmentMain( params , 'NonLinear')
params.directories = funcExpDirectories(params.directories.Experiment)