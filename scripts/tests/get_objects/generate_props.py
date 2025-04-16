def make_analog(symProps, analog_num):

    def name(name):
        if name == 'non_exist':
            return name
        else:
            return f"{name}_{analog_num}"
        
    # make a copy of symProps
    props = symProps.copy()
    #cycle through make a new name for each item.
    for prop in props:
        prop['name'] = name(prop['name'])
        prop['analog'] = analog_num
        for rb in prop['RBs']:
            rb['object_name'] = name(rb['object_name'])
            rb['object_sem'] = rb['object_sem']
            rb['pred_sem'] = rb['pred_sem']
            rb['pred_name'] = name(rb['pred_name'])

    return props

def generate_props(symProps, num_analogs):
    props = []
    for i in range(num_analogs):
        props.extend(make_analog(symProps, i))
    return props
