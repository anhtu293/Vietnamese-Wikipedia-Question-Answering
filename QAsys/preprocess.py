import string

def convertNumbers(sentence):
	convert_dict = {'0': u' không ',
					'1': u' một ',
					'2': u' hai ',
					'3': u' ba ',
					'4': u' bốn ',
					'5': u' năm ',
					'6': u' sáu ',
					'7': u' bảy ',
					'8': u' tám ',
					'9': u' chín '
					}
	result = ""
	for char in sentence:
		if char in convert_dict:
			result += convert_dict[char]
		else:
			result += char
	return result

def preprocess_sentence(sentence):
	result = sentence.lower()
	result = result.translate(str.maketrans('', '', string.punctuation))
	result = convertNumbers(result)
	result = " ".join(result.split())
	return result

if __name__ == "__main__":
	sentence = u"[ [ SI ] ] đơn vị áp suất [ [ pascal ( đơn vị ) | pascal ] ] ( Pa ) , bằng một [ [ niutơn ( đơn vị ) | niutơn ] ] mỗi [ [ mét vuông ] ] ( N · m hoặc kg · m · s ) . Tên này đặc biệt cho các đơn vị đã được bổ sung vào năm 1971 , trước đó , áp lực trong SI được thể hiện trong các đơn vị như N / m2 ."
	print(preprocess_sentence(sentence))