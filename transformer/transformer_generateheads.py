import numpy as np
import sys
import random
from string import Template

def sample_heads(total_heads,sampled_heads):
	try:
		samples = random.sample(range(0,total_heads), sampled_heads)  # random sampling without replacement
		return samples
	except ValueError:
		print("Sample size exceeded total size")

def generate_binary_array(total_heads,sampled_heads):
	binary_array = np.zeros(total_heads, dtype=int)
	for i in sampled_heads:
		binary_array[i] = 1
	return binary_array

def generate_head_string(template_string,head_array):
	t = Template(template_string)
	temp_dict = {}
	for i in range(len(head_array)):
		temp = "h{}".format(i)
		temp_dict[temp] = head_array[i]
	head_string = t.substitute(temp_dict)
	return head_string						 

# TRANSFORMER
encself_heads = int(sys.argv[1])
encdec_heads = int(sys.argv[2])
decself_heads = int(sys.argv[3])
encself_string = 'enc_self="[[$h0,$h1,$h2,$h3,$h4,$h5,$h6,$h7],[$h8,$h9,$h10,$h11,$h12,$h13,$h14,$h15],[$h16,$h17,$h18,$h19,$h20,$h21,$h22,$h23],[$h24,$h25,$h26,$h27,$h28,$h29,$h30,$h31],[$h32,$h33,$h34,$h35,$h36,$h37,$h38,$h39],[$h40,$h41,$h42,$h43,$h44,$h45,$h46,$h47]]"'
encdec_string = 'enc_dec="[[$h0,$h1,$h2,$h3,$h4,$h5,$h6,$h7],[$h8,$h9,$h10,$h11,$h12,$h13,$h14,$h15],[$h16,$h17,$h18,$h19,$h20,$h21,$h22,$h23],[$h24,$h25,$h26,$h27,$h28,$h29,$h30,$h31],[$h32,$h33,$h34,$h35,$h36,$h37,$h38,$h39],[$h40,$h41,$h42,$h43,$h44,$h45,$h46,$h47]]"'
decself_string = 'dec_self="[[$h0,$h1,$h2,$h3,$h4,$h5,$h6,$h7],[$h8,$h9,$h10,$h11,$h12,$h13,$h14,$h15],[$h16,$h17,$h18,$h19,$h20,$h21,$h22,$h23],[$h24,$h25,$h26,$h27,$h28,$h29,$h30,$h31],[$h32,$h33,$h34,$h35,$h36,$h37,$h38,$h39],[$h40,$h41,$h42,$h43,$h44,$h45,$h46,$h47]]"'
total_heads = 48
layer_heads = 8

encself_samples = sample_heads(total_heads,encself_heads)
encdec_samples = sample_heads(total_heads,encdec_heads)
decself_samples = sample_heads(total_heads,decself_heads)

print(encself_samples)
print(encdec_samples)
print(decself_samples)

encself_array = generate_binary_array(total_heads,encself_samples)
encdec_array = generate_binary_array(total_heads,encdec_samples)
decself_array = generate_binary_array(total_heads,decself_samples)

enc = generate_head_string(encself_string, encself_array)
encdec = generate_head_string(encdec_string, encdec_array)
dec = generate_head_string(decself_string, decself_array)

print(enc)
print(encdec)
print(dec)
