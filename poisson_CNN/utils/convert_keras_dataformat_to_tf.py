def convert_keras_dataformat_to_tf(df,ndims):
    if df == 'channels_first':
        return 'NC'+'DHW'[3-ndims:]
    elif df == 'channels_last':
        return 'N'+'DHW'[3-ndims:]+'C'
    else:
        raise(ValueError('Unrecognized data format - must be channels_first or channels_last'))
