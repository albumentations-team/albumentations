def write_coverage(functionname, branch_id):
    with open("cov.tmp", "a") as file:
        file.write(str(functionname) + ":" + str(branch_id) + "\n")
