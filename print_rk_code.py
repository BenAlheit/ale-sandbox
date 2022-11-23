print('starting')
print('\n\n')
print()

def print_vals(string, place):
    b1_str_strip = string.replace('\n', '').replace(' ', '')
    b1_str_strip = b1_str_strip.replace(r"âˆ’", "-")
    b1_split = b1_str_strip.split('b')
    b1_split.pop(0)
    b1_index = [int(v.split('=')[0])-1 for v in b1_split]
    b1_val = [v.split('=')[1] for v in b1_split]
    for i, val in zip(b1_index, b1_val):
        print(f'this->b.at({place}).at({i}) = {val};')


def print_vals_a(string):
    b1_str_strip = string.replace('\n', '').replace(' ', '')
    b1_str_strip = b1_str_strip.replace(r"âˆ’", "-")
    b1_split = b1_str_strip.split('a')
    b1_split.pop(0)
    b1_index = [int(v.split('=')[0].split(',')[0])-1 for v in b1_split]
    b2_index = [int(v.split('=')[0].split(',')[1])-1 for v in b1_split]
    b1_val = [v.split('=')[1] for v in b1_split]
    for i, j, val in zip(b1_index, b2_index, b1_val):
        print(f'this->a.at({i}).at({j}) = {val};')

b0_str = open('8(9)b0.dat', 'r').read()
b1_str = open('8(9)b1.dat', 'r').read()
a_str = open('8(9)a.dat', 'r').read()

print_vals(b0_str, 0)
print()
print_vals(b1_str, 1)
print()
print_vals_a(a_str)
