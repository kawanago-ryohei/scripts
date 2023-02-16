import numpy as np
import re


class HitxBlow:
    
    def __init__(self):
        self.dup = False
        self.n_colors = 6
        self.n_choice = 4
        self.n_turns = 6
        self.n_turns_left = self.n_turns
        self.color_list = ["red", "blue", "green", "black", "pink", "white",
                           "purple", "yellow", "gray", "magenta", "orange", "cyan"]
        self.colors = self.color_list[:self.n_colors]
        self.answer_list = []
        self.pos_list = []
        self.col_list = []
        
        self.n_game = 0
        self.results = 0
        
        print("You can start a game by calling game_start().")
        print("Use help_me() if you want to check which functions are available.")
        
    
    #ゲームの設定を変更する    
    def game_init(self):
        print("Default setting:")
        print(f"colors : {self.colors}")
        print(f"n_correct : {self.n_choice}")
        print(f"duplication : {self.dup}")
        
        while True:
            print("\n全てdefaultの設定で実行しますか？")
            res = input("y:use default setting\nn:change setting by yourself\n")
            res = res.lower()
            if res not in ["y", "n", "yes", "no"]:
                print("\nValueError: Input value is incorrect\n")
                continue
            else:
                break
                
        if res in ["n", "no"]:
            while True:
                res = input("\n使用する色の種類数を12以下で入力してください\n")
                if not bool(re.match(r"[0-9]", res)):
                    print("\nValueError: Enter value is not int\n")
                    continue
                elif int(res) > 12:
                    print("\nValueError: input value is too big\n")
                    continue
                elif int(res) == 1:
                    print("\nValueError: input value is too small\n")
                    continue
                else:
                    self.n_colors = int(res)
                    break
            
            while True:
                res = input("\n実際に並べる色の数を入力してください\n")
                if not bool(re.match(r"[0-9]", res)):
                    print("\nValueError: Enter value is not int\n")
                    continue
                elif int(res) > self.n_colors:
                    print("\nValueError: input value is too big\n")
                    continue
                else:
                    self.n_choice = int(res)
                    break

            while True:
                res = input("\nターン数を0以上15以下で入力してください\n")
                if not bool(re.match(r"[1-9]", res)):
                    print("\nValueError: Enter value is not int(0 is not included)\n")
                    continue
                elif int(res) > 15:
                    print("\nValueError: input value is too big\n")
                    continue
                else:
                    self.n_turns = int(res)
                    break

            while True:
                print("\n色の重複を許しますか？")
                res = input("y:重複させる\nn:重複させない\n")
                res = res.lower()
                if res not in ["y", "n", "yes", "no"]:
                    print("\nValueError: Input value is incorrect\n")
                    continue
                else:
                    if res in ["y", "yes"]:
                        self.dup = True
                    else:
                        self.dup = False
                    break
                    
        #使用されるカラーの設定
        self.colors = np.random.choice(self.color_list, self.n_colors, replace=False)
        
        print("\n設定が完了しました")
    
    #ゲームをスタートする
    def game_start(self):
        
        self.correct = np.random.choice(self.colors, self.n_choice, replace=self.dup)
        self.n_turns_left = self.n_turns
        self.n_game += 1
            
        print(f"These colors are used in this game.\ncolors:{self.colors}")
        print(f"(duplicated is {self.dup})")
        print(f"\nGuess {self.n_choice} position and color during {self.n_turns} turns.")
        
    #回答する
    def answer(self, answer):
        
        error = 0
        
        #ターンがもう0だったらエラー
        if self.n_turns_left == 0:
            
            #ゲームオーバー
            if self.results == 0:
                print("Unfortunately, there are already no turns left.")
                print("If you want to play the game again, execute game_start()\n")
                error += 1
                
            #ゲームクリアしてる
            else:
                print("You have already answered correctly.")
                print("If you want to play the game again, execute game_start()\n")
                error += 1
        
        #回答数が正しくない場合はエラー
        elif len(answer) != self.n_choice:
            print(f"ValueError:Incorrect number of answers. You must answer {self.n_choice} colors.")
            print("enter your answer again")
            error += 1
        
        #重複がないはずなのに回答が重複していたらエラー
        elif (self.dup == False) and len(np.unique(answer)) != self.n_choice:
            print("There is no duplication of correct answers.")
            print("enter your answer again")
            error += 1
        
        #既に回答済みのものはエラー
        elif answer in self.answer_list:
            print(f"ValueError:'{answer}' has already been answered\n")
            print("enter your answer again")
            error += 1
            
        #使用されていないはずのカラーがあったらエラー
        elif not set(answer) <= set(self.colors):
            incorrect = set(answer) - set(self.colors)
            print(f"ValueError:'{incorrect}' is not included in the colors")
            error += 1
        
        #エラーが起こらなければ答え合わせ
        else:
            correct_pos = 0
            correct_col = 0
            for i, a in enumerate(answer):

                #位置が合っている
                if a == self.correct[i]:
                    correct_pos += 1
                #カラーだけ合っている
                elif a in self.correct:
                    correct_col += 1
        
        #何かしらエラーがあったら
        if error >= 1:
            print("")
        
        #全部正解した場合
        elif correct_pos == self.n_choice:
            self.n_turns_left = 0
            self.answer_list = []
            self.results = 1
            print("Correct!")
            print("If you want to play the game again, execute game_start()")
        
        #まだ全部正解ではない場合
        else:
            self.answer_list.append(answer)
            self.pos_list.append(correct_pos)
            self.col_list.append(correct_col)
            self.n_turns_left -= 1
            
            #残りターンが0だったらおわり
            if self.n_turns_left == 0:
                print("Unfortunately, your remaining turns are now zero.")
                print("If you want to play the game again, execute game_start()\n")
                
            #結果を返す
            else:
                print(f"correct_position : {self.pos_list[-1]}")
                print(f"correct_colors : {self.col_list[-1]}")
                print(f"\nthe number of turns left is {self.n_turns_left}")
    
    #今までに回答したリストを表示
    def answer_history(self):
        if self.answer_list == []:
            print("Nothing has been answered yet.")
        else:
            print("your answer history:\n")
            for i, ans in enumerate(self.answer_list):
                print(f"correct_position : {self.pos_list[i]}")
                print(f"correct_colors : {self.col_list[i]}")
                print(ans)
    
    #諦めて答えを確認する
    def surrender(self):
        
        while True:    
            print("Do you want to give up guessing and check the answer?")
            res = input("y:Yes, I give up.\nn:No, I'm not giving in yet.\n")
            res = res.lower()
            if res not in ["y", "n", "yes", "no"]:
                print("\nValueError: Input value is incorrect\n")
                continue
            else:
                break
        
        if res in ["y", "yes"]:
            #self.n_turns_left = 0
            print(f"\nThe correct answer is here:\n{self.correct}\n")
            print("If you want to play the game again, execute game_start()\n")
    
    #使用されているカラーを表示する
    def colors_used(self):
        
        print("These colors is used:")
        print(self.colors)
    
    #遊び方を表示してくれる
    def help_me(self):
        print("You can use the following methods.\n")
        print("game_init() : Change the game settings.")
        print("game_start() : Calling this starts the game.")
        print("answer(['color', ...]) : Enter your answer.")
        print("answer_history() : Show your answers so far.")
        print("colors_used() : Check the colors used in the current game.")
        