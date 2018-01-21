# -*- coding:utf-8 -*-
"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
import regex as re
from helpers import get_lines, recompose_program
import subprocess
import time
from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JError

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])

class EmptyProgramException(Exception):
    '''In fn tokenizer:get_lines(), positions are empty, most probably the input program \
       is without any newline characters or has a special character such as ^A'''
    pass

class UnexpectedTokenException(Exception):
    pass

class C_Tokenizer():
    _keywords = ['auto', 'break', 'case', 'const', 'continue', 'default', \
                'do', 'else', 'enum', 'extern', 'for', 'goto', 'if', \
                'register', 'return', 'signed', 'sizeof', 'static', 'switch', \
                'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL', \
                'null', 'struct', 'union']
    _includes = ['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'malloc.h', \
                'stdbool.h', 'cstdio', 'cstdio.h', 'iostream', 'conio.h']
    _calls    = ['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen', \
                'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'free']
    _types    = ['char', 'double', 'float', 'int', 'long', 'short', 'unsigned']

    def _escape(self, string):
        return repr(string)[1:-1]

    def _tokenize_code(self, code):
        keywords = {'IF', 'THEN', 'ENDIF', 'FOR', 'NEXT', 'GOSUB', 'RETURN'}
        token_specification = [
            ('comment', r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            ('directive', r'#\w+'),
            ('string', r'"(?:[^"\n]|\\")*"?'),
            ('char', r"'(?:\\?[^'\n]|\\')'"),
            ('char_continue', r"'[^']*"),
            ('number',  r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
            ('include',  r'(?<=\#include) *<([_A-Za-z]\w*(?:\.h))?>'),
            ('op',  r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('name',  r'[_A-Za-z]\w*'),
            ('whitespace',  r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH',r'.'),            # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        line_num = 1
        line_start = 0
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                yield UnexpectedTokenException('%r unexpected on line %d' % (value, line_num))
            else:
                if kind == 'ID' and value in keywords:
                    kind = value
                column = mo.start() - line_start
                yield Token(kind, value, line_num, column)

    def _sanitize_brackets(self, tokens_string):
        lines = get_lines(tokens_string)

        if len(lines) == 1:
            raise EmptyProgramException(tokens_string)

        #for i, line in enumerate(lines):
        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            
            if line.strip() == '_<op>_}' or line.strip() == '_<op>_} _<op>_}' \
               or line.strip() == '_<op>_} _<op>_} _<op>_}' or line.strip() == '_<op>_} _<op>_;' \
               or line.strip() == '_<op>_} _<op>_} _<op>_} _<op>_}' \
               or line.strip() == '_<op>_{' \
               or line.strip() == '_<op>_{ _<op>_{':
                if i > 0:
                    lines[i-1] += ' ' + line.strip()
                    lines[i]    = ''
                else:
                    # can't handle this case!
                    return ''

        # Remove empty lines
        for i in range(len(lines)-1, -1, -1):
            if lines[i] == '':
                del lines[i]

        for line in lines:
            assert(lines[i].strip() != '')

        return recompose_program(lines)

    def tokenize(self, code, keep_format_specifiers=False, keep_names=True, \
                 keep_literals=False):
        result = '0 ~ '

        names = ''
        line_count = 1
        name_dict = {}
        name_sequence = []
        literal_sequence = []

        regex = '%(d|i|f|c|s|u|g|G|e|p|llu|ll|ld|l|o|x|X)'
        isNewLine = True

        # Get the iterable
        my_gen = self._tokenize_code(code)

        while True:
            try:
                token = next(my_gen)
            except StopIteration:
                break

            if isinstance(token, Exception):
                return '', '', ''

            type_ = str(token[0])
            value = str(token[1])

            if value in self._keywords:
                result += '_<keyword>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'include':
                result += '_<include>_' + self._escape(value).lstrip() + ' '
                isNewLine = False

            elif value in self._calls:
                result += '_<APIcall>_' + self._escape(value) + ' '
                isNewLine = False

            elif value in self._types:
                result += '_<type>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'whitespace' and (('\n' in value) or ('\r' in value)):
                if isNewLine:
                    continue
                result += ' '.join(list(str(line_count))) + ' ~ '
                line_count += 1
                isNewLine = True

            elif type_ == 'whitespace' or type_ == 'comment' or type_ == 'nl':
                pass

            elif 'string' in type_:
                matchObj = [m.group().strip() for m in re.finditer(regex, value)]
                if matchObj and keep_format_specifiers:
                    for each in matchObj:
                        result += each + ' '
                else:
                    result += '_<string>_' + ' '

                literal_sequence.append(self._escape(value))
                isNewLine = False

            elif type_ == 'name':
                if keep_names:
                    if self._escape(value) not in name_dict:
                        name_dict[self._escape(value)] = str(len(name_dict) + 1)
                    
                    name_sequence.append(self._escape(value))
                    result += '_<id>_' + name_dict[self._escape(value)] + '@ '
                    names += '_<id>_' + name_dict[self._escape(value)] + '@ '
                else:
                    result += '_<id>_' + '@ '
                isNewLine = False

            # need to keep 'r' in the end for pointer deref mutation to work.
            elif type_ == 'number':
                if keep_literals:
                    result += '_<number>_' + self._escape(value) + '# '
                else:
                    result += '_<number>_' + '# '

                literal_sequence.append(self._escape(value))
                isNewLine = False

            elif 'char' in type_ or value == '':
                result += '_<' + type_.lower() + '>_' + ' '

                literal_sequence.append(self._escape(value))
                isNewLine = False

            else:
                converted_value = self._escape(value).replace('~', 'TiLddE')
                result += '_<' + type_ + '>_' + converted_value + ' '

                isNewLine = False

        result = result[:-1]
        names = names[:-1]

        if result.endswith('~'):
            idx = result.rfind('}')
            result = result[:idx+1]

        return self._sanitize_brackets(result), name_dict, name_sequence, literal_sequence


class Java_Tokenizer():
    def __init__(self):
        self.start_server()

    def start_server(self):
        # クラスパスを指定して実行
        args = (["java", "-cp",
                 '/home/hirose/.pyenv/versions/2.7.14/share/py4j/py4j0.10.6.jar',
                 '-jar', './SrcTokenizer.jar'])
        subprocess.Popen(args)
        # サーバー起動前に処理が下へ行くのを防ぐ
        time.sleep(3) 
        self.gateway = JavaGateway(start_callback_server=True)

    def shutdown(self):
        self.gateway.shutdown()

    def tokenize(self, code, keep_format_specifiers=False, keep_names=True,
                 keep_literals=False):
        name_dict = {}
        name_sequence = []
        literal_sequence = []
        
        entry_point = self.gateway.entry_point
        fixed_token = []
        try:
            tokens = entry_point.get_masked_token_str(code).split()
            for token in tokens:
                if "/*" in token and "*/" in token:
                    num = token.replace("/*", "").replace("*/", "")
                    token = num + " ~ "
                elif "_Var" in token:
                    token = token.replace("_Var", "")
                    token += "_<id>_@"
                fixed_token.append(token)
        except Py4JError as e:
            print "Java parse error"
            raise Exception
            

        tokens = " ".join(fixed_token)
        return tokens, name_dict, name_sequence, literal_sequence
