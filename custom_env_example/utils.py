def count(data):
    statis = {}
    for d in data:
        if d not in statis:
            statis[d] =1
        else:
            statis[d] +=1
    return statis