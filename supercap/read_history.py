import re


def parse_history(file_output):

    obj_iteration_history = []
    iter_history_current = []
    constr_iteration_history = []

    file_object = open(file_output, "r")
    current_it = 0
    reg_number = "(-?[0-9]\d*)(\.\d+)?(e-?\+?\d*)?"
    for line in file_object:
        obj_iteration = re.findall(f"obj: {reg_number}", line)
        constraint_itera = re.findall("g\[0\]: (-?[0-9]\d*)(\.\d+)?(e-?\+?\d*)?", line)
        if obj_iteration:
            current_it += 1
            iter_history_current.append(current_it)
            obj_it_float = float(
                obj_iteration[0][0] + obj_iteration[0][1] + obj_iteration[0][2]
            )
            obj_iteration_history.append(obj_it_float)
        if constraint_itera:
            constr_itera = float(
                constraint_itera[0][0] + constraint_itera[0][1] + constraint_itera[0][2]
            )
            constr_iteration_history.append(constr_itera)

    return iter_history_current, obj_iteration_history, constr_iteration_history
