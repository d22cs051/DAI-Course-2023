import random
good_lines = []
# reading lines
with open("random_genrate_hindi (1).txt") as fp:
    lines = fp.readlines()
    print(f"total lines: {len(lines)}")
    # shufflling the lines to be selected
    random.shuffle(lines)
    for line in lines:
        line = line.removesuffix("\n").split(" ")
        if len(line) > 10 and len(line) < 16:
            if len(good_lines) < 1000:
                # selecting lines
                good_lines.append(' '.join(line))
                
# writing the lines to file
with open("selcted_lines_hin.txt","w") as fp:
    fp.write("\n".join(good_lines))
    print(f"total selected lines: {len(good_lines)}")