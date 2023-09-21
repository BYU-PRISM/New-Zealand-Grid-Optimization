def get_gk_val_real(gk_val):
    # Robustly determine the "length" of the GK_Value
    try:
        val = gk_val.value.value.value
    except:
        try:
            val = gk_val.value.value
        except:
            val = gk_val.value
    return val
