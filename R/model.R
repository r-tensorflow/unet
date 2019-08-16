conv2d_block <- function(inputs, use_batch_norm = TRUE, dropout = 0.3,
                         filters = 16, kernel_size = c(3, 3), activation = "relu",
                         kernel_initializer = "he_normal", padding = "same") {

  x <- keras::layer_conv_2d(
    inputs,
    filters = filters,
    kernel_size = kernel_size,
    activation = activation,
    kernel_initializer = kernel_initializer,
    padding = padding
  )

  if (use_batch_norm) {
    x <- keras::layer_batch_normalization(x)
  }

  if (dropout > 0) {
    x <- keras::layer_dropout(x, rate = dropout)
  }

  x <- keras::layer_conv_2d(
    x,
    filters = filters,
    kernel_size = kernel_size,
    activation = activation,
    kernel_initializer = kernel_initializer,
    padding = padding
  )

  if (use_batch_norm) {
    x <- keras::layer_batch_normalization(x)
  }

  x
}

#' U-Net: Convolutional Networks for Biomedical Image Segmentation
#'
#' @param input_shape Dimensionality of the input (integer) not including the
#'   samples axis. Must be lenght 3 numeric vector.
#' @param num_classes Number of classes.
#' @param dropout Dropout rate applied.
#' @param filters Number of filters of the first convolution.
#' @param num_layers Number of layers in the encoder.
#' @param  output_activation Activation in the output layer.
#'
#' @export
unet <- function(input_shape, num_classes = 1, dropout = 0.5, filters = 64,
                 num_layers = 4, output_activation = "sigmoid") {


  input <- keras::layer_input(shape = input_shape)

  x <- input
  down_layers <- list()

  for (i in seq_len(num_layers)) {

    x <- conv2d_block(
      inputs = x,
      filters = filters,
      use_batch_norm = FALSE,
      dropout = 0,
      padding = "same"
    )

    down_layers[[i]] <- x

    x <- keras::layer_max_pooling_2d(x, pool_size = c(2,2), strides = c(2,2))

    filters <- filters * 2

  }

  if (dropout > 0) {
    x <- keras::layer_dropout(x, rate = dropout)
  }

  x <- conv2d_block(
    inputs = x,
    filters = filters,
    use_batch_norm = FALSE,
    dropout = 0.0,
    padding = 'same'
  )

  for (conv in rev(down_layers)) {

    filters <- filters / 2L

    x <- keras::layer_conv_2d_transpose(
      x,
      filters = filters,
      kernel_size = c(2,2),
      padding = "same",
      strides = c(2,2)
    )

    x <- keras::layer_concatenate(list(conv, x))
    x <- conv2d_block(
      inputs = x,
      filters = filters,
      use_batch_norm = FALSE,
      dropout = 0.0,
      padding = 'same'
    )

  }

  output <- keras::layer_conv_2d(
    x,
    filters = num_classes,
    kernel_size = c(1,1),
    activation = output_activation
  )

  model <- keras::keras_model(input, output)

  model
}

