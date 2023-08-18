def save_dicts_to_textfile(file_path, dict1, dict2):
    with open(file_path, 'w') as file:
        for d in [dict1, dict2]:
            for key, value in d.items():
                file.write(f"{key}={value}\n")
            file.write('\n')  # Separate dictionaries with an empty line

def read_dicts_from_textfile(file_path):
    dict1, dict2 = {}, {}
    current_dict = dict1

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                current_dict = dict2
            else:
                key, value = line.split('=')
                current_dict[key] = value

    return dict1, dict2

# Example dictionaries
dict1 = {'key1': 'value1', 'key2': 'value2'}
dict2 = {'key3': 'value3', 'key4': 'value4'}

# Save dictionaries to text file
save_dicts_to_textfile('data.txt', dict1, dict2)

# Read dictionaries from text file
read_dict1, read_dict2 = read_dicts_from_textfile('data.txt')

print("Read Dict 1:", read_dict1)
print("Read Dict 2:", read_dict2)
