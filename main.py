import os
import numpy as np

class RNN:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_deriv(x):
        return 1 - x ** 2

    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum()

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    file = open(os.path.join(__location__, 'output.txt'))
    corpus = file.read()

    chars = list(set(corpus))
    data_size = len(corpus)
    vocab_size = len(chars)

    map_chars_to_ints = {}
    map_ints_to_chars = {}
    for count, char in enumerate(chars):
        map_chars_to_ints[char] = count
        map_ints_to_chars[count] = char

    hidden_units = 50
    sequence_length = 50
    learning_rate = 0.005

    hprev_to_hidden_memory_weight = np.random.rand(hidden_units, vocab_size) * 0.01
    hprev_to_hidden_update_weight = np.random.rand(hidden_units, vocab_size) * 0.01
    hprev_to_hidden_reset_weight = np.random.rand(hidden_units, vocab_size) * 0.01
    hprev_to_hidden_output_weight = np.random.rand(vocab_size, hidden_units) * 0.01

    input_to_hidden_update_weight = np.random.rand(hidden_units, hidden_units) * 0.01
    input_to_hidden_memory_weight = np.random.rand(hidden_units, hidden_units) * 0.01
    input_to_hidden_reset_weight = np.random.rand(hidden_units, hidden_units) * 0.01

    bias_memory = np.zeros((hidden_units, 1))
    bias_update = np.zeros((hidden_units, 1))
    bias_reset = np.zeros((hidden_units, 1))
    bias_output = np.zeros((vocab_size, 1))


    def forwardPropagate(self, input_list, target_list, input, update_gate, reset_gate, candidate, memory, output, p, loss):
        for i in range(len(input_list)):

            # 1-hot encoding
            input[i] = np.zeros((self.vocab_size, 1))
            input[i][input_list[i]] = 1

            # compute gate values
            update_gate[i] = np.dot(self.hprev_to_hidden_update_weight, input[i])
            update_gate[i] += np.dot(self.input_to_hidden_update_weight, memory[i - 1])
            update_gate[i] += self.bias_update
            update_gate[i] = self.sigmoid(update_gate[i])

            reset_gate[i] = np.dot(self.input_to_hidden_reset_weight, memory[i - 1])
            reset_gate[i] += self.bias_reset
            reset_gate[i] += np.dot(self.hprev_to_hidden_reset_weight, input[i])
            reset_gate[i] = self.sigmoid(reset_gate[i])

            # compute candidate for memory cell (h tilde)
            temp_cand = np.multiply(reset_gate[i], memory[i - 1])
            candidate[i] = np.dot(self.hprev_to_hidden_memory_weight, input[i])
            candidate[i] += np.dot(self.input_to_hidden_memory_weight, temp_cand)
            candidate[i] += self.bias_memory
            candidate[i] = self.tanh(candidate[i])


            # compute memory against gate values. Replace memory with candidate if needed
            memory[i] = np.multiply(update_gate[i], memory[i - 1]) + np.multiply((1 - update_gate[i]), candidate[i])

            # compute output using memory
            output[i] = np.dot(self.hprev_to_hidden_output_weight, memory[i]) + self.bias_output

            # softmax of output
            p[i] = self.softmax(output[i])

            # Compute loss
            t_loss = -np.sum(np.log(p[i][target_list[i]]))
            loss += t_loss
        return input, update_gate, reset_gate, candidate, memory, output, p, loss


    def backPropagate(self, input_list, target_list, input, update_gate, reset_gate, candidate, memory, p,
                      diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update,
                      diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update,
                      diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update,
                      diff_memory_next):
        for i in reversed(range(len(input_list))):

            dy = np.copy(p[i])
            dy[target_list[i]] -= 1

            diff_hprev_to_hidden_output += np.dot(dy, memory[i].T)
            diff_bias_output += dy

            x1 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x1 = np.multiply( x1, (1 - update_gate[i]))
            x1 = x1 * self.tanh_deriv(candidate[i])
            x1 = np.dot(x1, input[i].T)
            diff_hprev_to_hidden_memory += x1

            x2 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x2 = np.multiply(x2, (1 - update_gate[i]))
            x2 = x2* self.tanh_deriv(candidate[i])
            x2_t = np.multiply(reset_gate[i], memory[i - 1]).T
            x2 = np.dot(x2, x2_t)
            diff_input_to_hidden_memory += x2

            x3 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x3 = np.multiply(x3, (1 - update_gate[i]))
            x3 = x3 * self.tanh_deriv(candidate[i])
            diff_bias_memory += x3

            x4 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x4 = np.multiply(x4, (1 - update_gate[i]))
            x4 = x4 * self.tanh_deriv(candidate[i])
            x4 = np.dot(self.input_to_hidden_memory_weight.T, x4)
            x4 = np.multiply(x4, memory[i - 1])
            x4 = x4 * self.sigmoid_deriv(reset_gate[i])
            x4 = np.dot(x4, input[i].T)
            diff_hprev_to_hidden_reset += x4

            x5 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x5 = np.multiply(x5, (1 - update_gate[i]))
            x5 = x5 * self.tanh_deriv(candidate[i])
            x5 = np.dot(self.input_to_hidden_memory_weight.T, x5)
            x5 = np.multiply(x5, memory[i - 1]) * self.sigmoid_deriv(reset_gate[i])
            x5 = np.dot(x5, memory[i - 1].T)
            diff_input_to_hidden_reset += x5

            x6 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x6 = np.multiply(x6, (1 - update_gate[i]))
            x6 = x6 * self.tanh_deriv(candidate[i])
            x6 = np.dot(self.input_to_hidden_memory_weight.T, x6)
            x6 = np.multiply(x6, memory[i - 1]) * self.sigmoid_deriv(reset_gate[i])
            diff_bias_reset += x6

            x6 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x6 = np.multiply(x6, memory[i - 1] - candidate[i]) * self.sigmoid_deriv(update_gate[i])
            x6 = np.dot(x6, input[i].T)
            diff_hprev_to_hidden_update += x6

            x7 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x7 = np.multiply(x7, memory[i - 1] - candidate[i]) * self.sigmoid_deriv(update_gate[i])
            x7 = np.dot(x7, memory[i - 1].T)
            diff_input_to_hidden_update += x7

            x8 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x8 = np.multiply(x8, memory[i - 1] - candidate[i])
            x8 = (x8 * self.sigmoid_deriv(update_gate[i]))
            diff_bias_update += x8


            x9 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x9 = np.multiply(x9, memory[i - 1] - candidate[i]) * self.sigmoid_deriv(update_gate[i])
            x9 = np.dot(self.input_to_hidden_update_weight.T, x9)

            x10 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x10 = np.multiply(x10, update_gate[i])

            x11 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x11 = np.multiply(x11, (1 - update_gate[i])) * self.tanh_deriv(candidate[i])
            x11 = np.dot(self.input_to_hidden_memory_weight.T, x11)
            x11 = np.multiply(x11, reset_gate[i])

            x12 = np.dot(self.hprev_to_hidden_output_weight.T, dy) + diff_memory_next
            x12 = np.multiply(x12 , (1 - update_gate[i])) * self.tanh_deriv(candidate[i])
            x12 = np.dot(self.input_to_hidden_memory_weight.T, x12)
            x12 = np.multiply(x12, memory[i - 1]) * self.sigmoid_deriv(reset_gate[i])
            x12 = np.dot(self.input_to_hidden_reset_weight.T, x12)

            diff_memory_next = x9 + x10 + x11 + x12

        return input, update_gate, reset_gate, candidate, memory, p, \
               diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update, \
               diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update, \
               diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, \
               diff_memory_next


    def do_pass(self, input_list, target_list, hprev):
        # Initialize variables
        input = {}
        update_gate = {}
        reset_gate = {}
        candidate = {}
        memory = {}
        output = {}
        p = {}

        memory[-1] = np.copy(hprev)
        loss = 0

        diff_hprev_to_hidden_output = np.zeros_like(self.hprev_to_hidden_output_weight)
        diff_hprev_to_hidden_memory = np.zeros_like(self.hprev_to_hidden_memory_weight)
        diff_hprev_to_hidden_reset = np.zeros_like(self.hprev_to_hidden_reset_weight)
        diff_hprev_to_hidden_update = np.zeros_like(self.hprev_to_hidden_update_weight)

        diff_input_to_hidden_memory = np.zeros_like(self.input_to_hidden_memory_weight)
        diff_input_to_hidden_reset = np.zeros_like(self.input_to_hidden_reset_weight)
        diff_input_to_hidden_update = np.zeros_like(self.input_to_hidden_update_weight)

        diff_bias_output = np.zeros_like(self.bias_output)
        diff_bias_memory = np.zeros_like(self.bias_memory)
        diff_bias_reset = np.zeros_like(self.bias_reset)
        diff_bias_update = np.zeros_like(self.bias_update)

        # Forward prop
        input, update_gate, reset_gate, candidate, memory, output, p, loss = \
            self.forwardPropagate(input_list, target_list, input, update_gate, reset_gate, candidate, memory, output, p, loss)


        # Parameter gradient initialization
        diff_mem_next = np.zeros_like(memory[0])

        # Backward prop
        input, update_gate, reset_gate, candidate, memory, p, \
        diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update, \
        diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update, \
        diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, diff_mem_next = \
            self.backPropagate(input_list, target_list, input, update_gate, reset_gate, candidate, memory, p,
                               diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update,
                               diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update,
                               diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, diff_mem_next)

        return diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update, \
               diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update, \
               diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, memory[len(input_list)-1], loss

    def generate(self, mem, index_i, t_len):
        input = np.zeros((self.vocab_size, 1))
        input[index_i] = 1
        int_list = [index_i]
        for i in range(t_len):

            update_gate = np.dot(self.input_to_hidden_update_weight, mem)
            update_gate += np.dot(self.hprev_to_hidden_update_weight, input)
            update_gate += self.bias_update
            update_gate = self.sigmoid(update_gate)

            reset_gate = self.bias_reset
            reset_gate += np.dot(self.hprev_to_hidden_reset_weight, input)
            reset_gate += np.dot(self.input_to_hidden_reset_weight, mem)
            reset_gate = self.sigmoid(reset_gate)

            candidate = np.dot(self.input_to_hidden_memory_weight, np.multiply(reset_gate, mem))
            candidate += self.bias_memory
            candidate += np.dot(self.hprev_to_hidden_memory_weight, input)
            candidate = self.tanh(candidate)

            mem = np.multiply(update_gate, mem) + np.multiply((1 - update_gate), candidate)
            output = np.dot(self.hprev_to_hidden_output_weight, mem) + self.bias_output
            p = self.softmax(output)

            next_int = np.random.choice(range(self.vocab_size), p=p.ravel())
            input = np.zeros((self.vocab_size, 1))
            input[next_int] = 1
            int_list.append(next_int)

        return int_list

    def gradient_descent(self, diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update,
                         diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update,
                         diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update,
                         md_hh_output, md_hh_mem, md_hh_reset, md_hh_update,
                         md_ih_mem, md_ih_reset, md_ih_update,
                         md_bias_output, md_bias_mem, md_bias_reset, md_bias_update):

        for param, dparam, mem in zip(
                [self.hprev_to_hidden_output_weight, self.hprev_to_hidden_memory_weight, self.hprev_to_hidden_reset_weight, self.hprev_to_hidden_update_weight,
                 self.input_to_hidden_memory_weight, self.input_to_hidden_reset_weight, self.input_to_hidden_update_weight,
                 self.bias_output, self.bias_memory, self.bias_reset, self.bias_update],
                [diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update,
                 diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update,
                 diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update],
                [md_hh_output, md_hh_mem, md_hh_reset, md_hh_update, md_ih_mem, md_ih_reset, md_ih_update,
                 md_bias_output, md_bias_mem, md_bias_reset, md_bias_update]):

            np.clip(dparam, -5, 5, out=dparam)
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

        return diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update, diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update, diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, md_hh_output, md_hh_mem, md_hh_reset, md_hh_update, md_ih_mem, md_ih_reset, md_ih_update, md_bias_output, md_bias_mem, md_bias_reset, md_bias_update


    def run(self):
        n = 0
        seq_ctrl = 0

        md_hh_output = np.zeros_like(self.hprev_to_hidden_output_weight)
        md_hh_mem = np.zeros_like(self.hprev_to_hidden_memory_weight)
        md_hh_reset = np.zeros_like(self.hprev_to_hidden_reset_weight)
        md_hh_update = np.zeros_like(self.hprev_to_hidden_update_weight)

        md_ih_mem = np.zeros_like(self.input_to_hidden_memory_weight)
        md_ih_reset = np.zeros_like(self.input_to_hidden_reset_weight)
        md_ih_update = np.zeros_like(self.input_to_hidden_update_weight)

        md_bias_output = np.zeros_like(self.bias_output)
        md_bias_mem = np.zeros_like(self.bias_memory)
        md_bias_reset = np.zeros_like(self.bias_reset)
        md_bias_update = np.zeros_like(self.bias_update)

        net_loss = -np.log(1.0 / self.vocab_size) * self.sequence_length

        # print after x iterations
        iterations = 100

        while True:
            # Reset memory if appropriate
            if seq_ctrl + self.sequence_length + 1 >= len(self.corpus) or n == 0:
                hprev = np.zeros((self.hidden_units, 1))
                seq_ctrl = 0

            # Get input and target sequence

            inputs = []
            targets = []

            string_length = 100

            for ch in self.corpus[seq_ctrl:seq_ctrl + self.sequence_length]:
                inputs += [self.map_chars_to_ints[ch]]

            for ch in self.corpus[seq_ctrl + 1:seq_ctrl + self.sequence_length + 1]:
                targets+= [self.map_chars_to_ints[ch]]


            # Get gradients after pass
            diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update, \
            diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update, \
            diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, \
            hprev, loss = self.do_pass(inputs, targets, hprev)

            net_loss = net_loss * 0.999 + loss * 0.001

            # Do gradient descent
            diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update, \
            diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update, \
            diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update, \
            md_hh_output, md_hh_mem, md_hh_reset, md_hh_update, \
            md_ih_mem, md_ih_reset, md_ih_update, \
            md_bias_output, md_bias_mem, md_bias_reset, md_bias_update = \
                self.gradient_descent( diff_hprev_to_hidden_output, diff_hprev_to_hidden_memory, diff_hprev_to_hidden_reset, diff_hprev_to_hidden_update,
                                       diff_input_to_hidden_memory, diff_input_to_hidden_reset, diff_input_to_hidden_update,
                                       diff_bias_output, diff_bias_memory, diff_bias_reset, diff_bias_update,
                                       md_hh_output, md_hh_mem, md_hh_reset, md_hh_update,
                                       md_ih_mem, md_ih_reset, md_ih_update,
                                       md_bias_output, md_bias_mem, md_bias_reset, md_bias_update)

            # Print output
            if n % iterations == 0:
                tokens = self.generate(hprev, inputs[0], string_length)
                text = ""
                for i in tokens:
                    text += self.map_ints_to_chars[i]
                print("---------------\n"+text)
                # print(str(net_loss))
                print("---------------")


            # Prepare for next iteration
            seq_ctrl += self.sequence_length
            n += 1

rnn = RNN()
rnn.run()
