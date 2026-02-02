def swizzle(n, m, k, u, v):
    for i in range(n):
        h = " ".join([str(x) for x in range(m)])
        print(h)

    print()

    f = []
    for i in range(n):
        g = []
        for j in range(m):
            g += [int(j/k)*k + (v*(int((i%k)/u)^int((j%k)/v)) + ((j%k)%v)) % k]
        f += [g]

        h = " ".join([str(x) for x in g])
        print(h)
    
    print()
    
    for i in range(n):
        g = []
        for j in range(m):
            w = f[i][j]
            g += [int(w/k)*k + (v*(int((i%k)/u)^int((w%k)/v)) + ((w%k)%v)) % k]
        
        h = " ".join([str(x) for x in g])
        print(h)

swizzle(32, 32, 32, 2, 8)

# 0 -> u*(r ^ 0)
# 1 -> u*(r ^ 0) + 1
# 2 -> r ^ 1
# 3 -> r ^ 1 + 1

# j -> r ^ (j/2) + (j%2)

# 0  4 8 12
# 16 20 24 28
# 0 4 8 12
# 16 20 24 28


# 0 
# 16
#    4
#    20
#       8
#       24 
#         12
#         28
