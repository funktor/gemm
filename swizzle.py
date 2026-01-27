def swizzle(n, m, k):
    for i in range(n):
        h = " ".join([str(x) for x in range(m)])
        print(h)

    print()

    for i in range(n):
        g = []
        for j in range(m):
            g += [(int(j/k))*k + (i%k)^(j%k)]

        h = " ".join([str(x) for x in g])
        print(h)



swizzle(32, 32, 16)
