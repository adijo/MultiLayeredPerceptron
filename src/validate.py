
def validate_mlp_architecture(layers):
    for i in range(len(layers) - 1):
        current_layer = layers[i]
        next_layer = layers[i + 1]
        if current_layer.out_dim != next_layer.in_dim:
            raise Exception("Invalid architecture. Layer %d's output dim %d and layer %d's input dim %d should match."
                            % (i, i + 1, current_layer.out_dim, next_layer.in_dim))
    return True
