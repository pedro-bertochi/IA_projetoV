import random
usuarios = [
    "Ana Rosa",
    "Joice",
    "Keli",
    "Jaque",
    "Isa"
]
suporte = [
    "Pedro",
    "Rafa"
]

for _ in usuarios:
    print(f'{random.choice(suporte)} {random.choice(usuarios)}')


# # Roleta para decidir times
# def roleta_times(usuarios, n_times):
#     random.shuffle(usuarios)
#     times = [[] for _ in range(n_times)]
#     for idx, usuario in enumerate(usuarios):
#         times[idx % n_times].append(usuario)
#     return times

# # Exemplo: dividir em 2 times
# n_times = 2
# times = roleta_times(usuarios, n_times)
# for i, time in enumerate(times, 1):
#     print(f"Time {i}: {', '.join(time)}")
