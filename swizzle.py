def swizzle(n, m, k, u):
    for i in range(n):
        h = " ".join([str(x) for x in range(m)])
        print(h)

    print()

    f = []
    for i in range(n):
        g = []
        for j in range(m):
            g += [int(j/k)*k + (u*((i%k)^int((j%k)/u)) + ((j%k)%u)) % k]
        f += [g]

        h = " ".join([str(x) for x in g])
        print(h)
    
    print()

    f = []
    for i in range(n):
        g = []
        for j in range(m):
            g += [int(j/k)*k + (u*(int((i%k)/u)^(j%k)) + ((i%k)%u)) % k]
        f += [g]

        h = " ".join([str(x) for x in g])
        print(h)
    
    print()
    
    for i in range(n):
        g = []
        for j in range(m):
            w = f[i][j]
            g += [int(w/k)*k + (u*((i%k)^int((w%k)/u)) + ((w%k)%u)) % k]
        
        h = " ".join([str(x) for x in g])
        print(h)

swizzle(64, 64, 8, 2)

# 0 -> u*(r ^ 0)
# 1 -> u*(r ^ 0) + 1
# 2 -> r ^ 1
# 3 -> r ^ 1 + 1

# j -> r ^ (j/2) + (j%2)
