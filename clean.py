test_file_name = './data/sentences_test'
continuation_file_name = './data/sentences.continuation'
def clean_task_1():
    to_clean = ['A.txt', 'B.txt', 'C.txt']

    for file_name in to_clean:
        with open(file_name, 'r') as file, open('group16.perplexity'+file_name, 'w') as out:
            lines = file.readlines()
            for line in lines:
                p = line.split('\t')[-1].strip()
                out.write(p+'\n')


def clean_task_2():

    def clean_eos(line):
        sent = []
        splits = line.split()
        for word in splits:
            if word != '<eos>':
                sent.append(word)
            else:
                sent.append(word)
                break

        return ' '.join(sent).strip()

    with open('generated.txt', 'r') as to_clean_file, open(continuation_file_name, 'r') as test_file, open('group16.continuation.txt', 'w') as out:
        generated_lines = to_clean_file.readlines()
        test_lines = test_file.readlines()

        assert len(generated_lines) == len(test_lines)

        for idx in range(len(generated_lines)):
            generated_line = generated_lines[idx]
            test_line = test_lines[idx].strip()
            num_clean_words = len(test_line.split())

            clean_line = test_line + ' ' + ' '.join(generated_line.split()[1+num_clean_words:])
            clean_line = ' '.join(clean_line.split()[:20])

            clean_line_ = clean_eos(clean_line)
            num_words = len(clean_line.split())

            assert num_words <= 20
            out.write(clean_line_+'\n')


def clean(tag='1'):
    if tag == '1':
        clean_task_1()
    else:
        clean_task_2()

if __name__ == '__main__':
    clean('1')
    clean('2')