from math import*
import numpy as np
import random as rd
import copy
import time
import pickle
import turtle as tr

def mirror(tuple_pair_list,n=8):
    for k in range(len(tuple_pair_list)):
        (i,j)=tuple_pair_list[k]
        tuple_pair_list[k]=(n-j-1,n-i-1)
    #tuple_pair_list.reverse()
    return tuple_pair_list


class branch(): #a branch corresponding to a move
    global n ;n=8
    global nodes; nodes={} #dict of all the nodes that were ever created
    global black_turns_to_win; black_turns_to_win={}
    global white_turns_to_win; white_turns_to_win={}
    global scale; scale=2

    global screen; screen=None
    global turtles; turtles={}
    global sub_turtles; sub_turtles={}

    def __init__(self,mother,move,index=True):
        #MCTS_node.__init__(self)
        self.move=move
        self.white_turn = not mother.white_turn#; self.depth=mother.depth
        self.board_state=copy.deepcopy(mother.board_state)
        self.white_pieces=copy.deepcopy(mother.white_pieces)
        self.black_pieces=copy.deepcopy(mother.black_pieces)
        self.take_move(move); self.branches=[];
        self.white_wins=0;self.black_wins=0;self.ties=0
        self.score=mother.heuristic_score(move)
        h=(tuple(sorted(self.white_pieces)),tuple(sorted(self.black_pieces)))
        if (h in nodes) and index:
            self=nodes[h]
        else:
            #self.branches_white_wins=[]; self.branches_black_wins=[];self.branches_ties=[]
            nodes[h]=self
            #self.black_turns_to_win=mother.black_turns_to_win;self.white_turns_to_win=mother.white_turns_to_win

    def __getitem__(self,index):
        return self.branches[index]

    def bs(self,i,j): return self.board_state[i,j]

    def active_pieces(self):
        if self.white_turn:
            return self.white_pieces
        else:
            return self.black_pieces

    def add_branch(self,move,index=True):
        self.branches.append(branch(self,move,index))

    def take_move(self,move):
        sign=self.board_state[move[0][0],move[0][1]]
        self.board_state[move[1][0],move[1][1]]=sign
        self.board_state[move[0][0],move[0][1]]=0
        if sign==1:
            self.white_pieces.append((move[1][0],move[1][1]))
            self.white_pieces.remove((move[0][0],move[0][1]))
        else:
            self.black_pieces.append((move[1][0],move[1][1]))
            self.black_pieces.remove((move[0][0],move[0][1]))

    def gen_jump_links(self,i,j,links=[]):
        for (k,l) in [(0,1),(0,-1),(1,0),(-1,0)]:
            if 0<=i+2*k<n and 0<=j+2*l<n and self.bs(i+k,j+l)!=0 and self.bs(i+2*k,j+2*l)==0 and (i+2*k,j+2*l) not in links:
                links.append((i+2*k,j+2*l))
                self.gen_jump_links(i+2*k,j+2*l,links)
        return links

    def legal_moves(self):
        moves=[]
        for p in self.active_pieces():
            for [i,j] in [[0,1],[0,-1],[1,0],[-1,0]]:
                if 0<=p[0]+i<n and 0<=p[1]+j<n and self.bs(p[0]+i,p[1]+j)==0:
                    moves.append((p,(p[0]+i,p[1]+j)))
            for (i,j) in self.gen_jump_links(p[0],p[1],links=[]):
                moves.append((p,(i,j)))
        return moves

    def legal_move(self,p):
        moves=[]
        for [i,j] in [[0,1],[0,-1],[1,0],[-1,0]]:
            if 0<=p[0]+i<n and 0<=p[1]+j<n and self.bs(p[0]+i,p[1]+j)==0:
                moves.append((p,(p[0]+i,p[1]+j)))
        for (i,j) in self.gen_jump_links(p[0],p[1],links=[]):
            moves.append((p,(i,j)))
        return moves

    def heuristic_score(self,moves):
        a=2.0/sqrt(2);b=a/6#a/3
        c=a/sqrt(2); min_index=4
        if hasattr(moves,'shape') and len(moves.shape)==3:
            x_i=moves[:,0,0];y_i=moves[:,0,1];x_f=moves[:,1,0];y_f=moves[:,1,1];sign=self.bs(moves[0][0][0],moves[0][0][1])
        elif len(moves)==2 and len(moves[0])==2 and len(moves[1])==2:
            x_i=moves[0][0];y_i=moves[0][1];x_f=moves[1][0];y_f=moves[1][1];sign=self.bs(moves[0][0],moves[0][1])
        else:
            return None
        delta_score=a*(x_f+y_f-x_i-y_i)-sign*b*((y_f-x_f)**2-(x_i-y_i)**2)
        if self.white_turn:
            delta_score-=c*(x_f>3)*(y_f>3)*(x_f+y_f-7)-c*(x_i>3)*(y_i>3)*(x_i+y_i-7)
        else:
            delta_score-=c*(x_f<4)*(y_f<4)*(x_f+y_f-7)-c*(x_i<4)*(y_i<4)*(x_i+y_i-7)
        return self.score+delta_score


    def check_white_victory(self):
        return (tuple(sorted(self.white_pieces)) in white_turns_to_win)

    def check_black_victory(self):
        return (tuple(sorted(self.black_pieces)) in black_turns_to_win)

    def generate_end_game_dict(self,max_depth=3,depth=0):
        self.white_turn=True
        sorted_tuple=tuple(sorted(self.white_pieces))
        sorted_mirror=tuple(sorted(mirror(copy.deepcopy(self.white_pieces))))
        if (sorted_tuple not in black_turns_to_win) or black_turns_to_win[sorted_tuple]>=depth:
            black_turns_to_win[sorted_tuple]=depth
            white_turns_to_win[sorted_mirror]=depth

        if depth<max_depth:
            for move in self.legal_moves():
                self.add_branch(move,False)
                child=self.branches[-1]
                child.generate_end_game_dict(max_depth,depth+1)

    def load_end_game_dict(self,max_depth=3):
        try:
            with open('mypickle.pickle') as f:
                [black_turns_to_win,white_turns_to_win] = pickle.load(f)
                return True
        except:
            with open('end_game.pickle', 'wb') as f:
                self.generate_end_game_dict(max_depth); self.branches=[]
                pickle.dump([black_turns_to_win,white_turns_to_win], f)
                return True
        finally:
            return False

    def best_branch(self,exploration=False):
        N=len(self.branches)
        if N==0:
            self.leaves=np.array(self.legal_moves())
            self.branches_score=self.heuristic_score(self.leaves)
            N=len(self.leaves)
            self.branches_white_wins=np.zeros(N)
            self.branches_black_wins=np.zeros(N)
            self.branches_ties=np.zeros(N)
            for i in range(N):
                self.branches.append(None)

        sigma_pwin=0.01;c=sqrt(2);epsilon=0.000001;r=0.000001
        hw=1.0/np.sqrt(sigma_pwin*(self.branches_white_wins+self.branches_white_wins)+1.0)
        if self.white_turn:
            MCS=0.5*hw*(1.0+np.tanh(self.branches_score))+(1.0-hw)*(self.branches_white_wins+epsilon)/(self.branches_white_wins+self.branches_black_wins + epsilon)
        else:
            MCS=0.5*hw*(1.0+np.tanh(-self.branches_score))+(1.0-hw)*(self.branches_black_wins+epsilon)/(self.branches_white_wins+self.branches_black_wins + epsilon)
        if exploration:
            MCS+=c*np.sqrt(np.log(self.white_wins+self.black_wins+self.ties+1.0)/(self.branches_white_wins+self.branches_black_wins+self.branches_ties+1.0))+r*np.random.normal(0.0,1.0,(N,))
        mMCS=max(MCS)-epsilon
        index=np.where(MCS>=mMCS)[0][0]
        if self[index]==None or (not isinstance(self[index],branch)):
            self.branches[index]=branch(self,self.leaves[index])#,index
            self.branches[index].score=self.branches_score[index]
        return index

    def branch_to_all(self):
        N=len(self.branches)
        if N==0:
            self.leaves=np.array(self.legal_moves())
            self.branches_score=self.heuristic_score(self.leaves)
            N=len(self.leaves)
            self.branches_white_wins=np.zeros(N)
            self.branches_black_wins=np.zeros(N)
            self.branches_ties=np.zeros(N)
            for i in range(N):
                self.branches.append(None)

        for index,move in enumerate(self.legal_moves()):
            if self.branches[index]==None:
                self.branches[index]=branch(self,move)

    def set_display(self):
        screen = tr.Screen()
        screen.title("8x8 Chinese Checkers with turtles")
        screen.setup (width=360*scale, height=360*scale)
        t=tr.Turtle(visible=False)
        t.pu()
        t.speed(0)
        for X in range(8):
            for Y in range(8):
                if (X+Y)%2==0: #black
                    t.goto((40*X-160)*scale,(40*Y-120)*scale)
                    t.fillcolor('black')
                    t.begin_fill()
                    for Z in range(4):
                        t.forward(40*scale)
                        t.right(90)
                    t.end_fill()
        print("done!")
        return t

    def create_white_pieces(self):
        for p in self.white_pieces:
            t=tr.Turtle(shape="turtle", visible=False)
            t.speed(0)
            t.turtlesize(1.5*scale,1.5*scale)
            t.pu()
            t.color("#7FFF00")
            t.goto((40*p[0]-140)*scale,(40*p[1]-140)*scale)
            t.tilt(45)
            t.showturtle()
            turtles[p] = t

    def create_black_pieces(self):
        for p in self.black_pieces:
            t=tr.Turtle(shape="turtle", visible=False)
            t.speed(0)
            t.turtlesize(1.5*scale,1.5*scale)
            t.pu()
            t.color("red")
            t.goto((40*p[0]-140)*scale,(40*p[1]-140)*scale)
            t.tilt(225)
            t.showturtle()
            turtles[p] = t

    def create_pieces(self):
        self.create_white_pieces()
        self.create_black_pieces()

    def display_move(self,m):
        t = turtles[(m[0][0],m[0][1])]
        t.speed(5)
        t.settiltangle(np.degrees(np.arctan2(m[1][1]-m[0][1],m[1][0]-m[0][0])))
        t.goto((40*m[1][0]-140)*scale,(40*m[1][1]-140)*scale)
        del turtles[(m[0][0],m[0][1])]
        turtles[(m[1][0],m[1][1])] = t

    def get_turtle_pos(self,turtle):
        return list(turtles.keys())[list(turtles.values()).index(turtle)]

    def display_legal_moves(self,pos):
        self.clear_sub_turtles()
        for move in self.legal_move(pos):
            t=tr.Turtle(shape="circle", visible=False)#
            t.speed(0)
            t.turtlesize(scale,scale)
            t.color("pink")
            t.pu()
            t.goto((40*move[1][0]-140)*scale,(40*move[1][1]-140)*scale)
            t.st()
            sub_turtles[tuple(move[1])]=t

    def clear_sub_turtles(self):
        for turtle in sub_turtles.values():
            turtle.ht()
            del turtle

class root(branch): #The mother of all nodes...
    global n ;n=8
    def __init__(self):
        #CC_node.__init__(self,None)
        self.board_state=np.zeros((n,n),dtype=int)
        self.white_pieces=[]; self.black_pieces=[]; self.depth=0; self.white_turn=True
        for i in range(3):
            for j in range(3):
                self.board_state[i,j]=1; self.board_state[n-i-1,n-j-1]=-1
                self.white_pieces.append((i,j)); self.black_pieces.append((n-i-1,n-j-1))
        #self.branches_white_wins=[]; self.branches_black_wins=[] ; self.branches_ties=[]
        self.branches=[]; self.score=0
        self.white_wins=0; self.black_wins=0;self.ties=0

class move_chain():
    def __init__(self,this=None,prev=None,index=None,depth=0):
        super(move_chain,self).__init__()
        self.prev=prev
        self.index=index
        self.this=this
        self.score=None
        self.depth=depth

    def tie(self):
        if self.prev==None:
            return self.this
        self.prev.this.ties+=1
        self.prev.this.branches_ties[self.index]+=1
        if self.this.score!=None:
            r=0.8/(self.prev.this.ties+self.prev.this.black_wins+self.prev.this.white_wins)
            self.prev.this.score=(1.0-r)*self.prev.this.score+r*self.this.score
            if self.prev.prev!=None:
                self.prev.prev.this.branches_score[self.prev.index]=(1.0-r)*self.prev.prev.this.branches_score[self.prev.index]+r*self.this.score
        return self.prev.tie()

    def white_win(self):
        if self.prev==None:
            return self.this
        self.prev.this.white_wins+=1
        self.prev.this.branches_white_wins[self.index]+=1
        if self.this.score!=None:
            r=0.8/(self.prev.this.ties+self.prev.this.black_wins+self.prev.this.white_wins)
            self.prev.this.score=(1.0-r)*self.prev.this.score+r*self.this.score
            if self.prev.prev!=None:
                self.prev.prev.this.branches_score[self.prev.index]=(1.0-r)*self.prev.prev.this.branches_score[self.prev.index]+r*self.this.score
        return self.prev.white_win()

    def black_win(self):
        if self.prev==None:
            return self.this
        self.prev.this.black_wins+=1
        self.prev.this.branches_black_wins[self.index]+=1
        if self.this.score!=None:
            r=0.8/(self.prev.this.ties+self.prev.this.black_wins+self.prev.this.white_wins)
            self.prev.this.score=(1.0-r)*self.prev.this.score+r*self.this.score
            if self.prev.prev!=None:
                self.prev.prev.this.branches_score[self.prev.index]=(1.0-r)*self.prev.prev.this.branches_score[self.prev.index]+r*self.this.score
        return self.prev.black_win()

    def MC_expansion(self,max_depth=10):
        #print(self.this)
        N=len(self.this.branches)
        if N==0:
            if self.this.white_turn and self.this.check_white_victory():
                return self.white_win()
            elif self.this.check_black_victory():
                return self.black_win()
            if self.depth>max_depth:
                return self.tie()
        index=self.this.best_branch(True)
            #self.this.branches[index].depth=self.this.depth+1
        #print(self.this.depth)
        #mc.depth=self.depth+1
        return move_chain(self.this.branches[index],self,index,self.depth+1).MC_expansion()

def pixle2turtle(x,y):
    return (round((x/scale+140.0)/40.0),round((y/scale+140.0)/40.0))

class game:

    def __init__(self,white_pieces_AI=False,black_pieces_AI=True,dif=4,game=root()):
        self.white_pieces_AI=white_pieces_AI
        self.black_pieces_AI=black_pieces_AI
        self.game=game
        self.dif=dif
        self.turn=0


    def on_click(self,x,y):
        global chosen_turtle,Loop
        p = pixle2turtle(x,y)
        if p in turtles:
            chosen_turtle=p
            self.game.display_legal_moves(pixle2turtle(x,y))
            return None
        if p in sub_turtles:
            self.game.clear_sub_turtles()
            for index,branch in enumerate(self.game.branches):
                if tuple(branch.move[1])==p and tuple(branch.move[0])==chosen_turtle:
                    self.game=self.game.branches[index]
                    Loop=False
                    return None

    chosen_turtle=None
    def player_move(self):
        self.game.branch_to_all()
        global Loop
        Loop=True
        while Loop:
            all_turtles={**turtles,**sub_turtles}
            for turtle in all_turtles.values():
                turtle.onclick(lambda x,y: self.on_click(x,y))
        self.game.display_move(self.game.move)
        self.game.clear_sub_turtles()


    def ai_move(self,N=250):
        N *= self.dif
        for i in range(N):
            self.game=move_chain(self.game).MC_expansion(4+2*self.dif)
        index=self.game.best_branch()
        self.game=self.game.branches[index]
        self.game.display_move(self.game.move)

    def set_board(self):
        t1=time.time()
        self.game.load_end_game_dict(self.dif)
        t2=time.time(); print('end_game_dict is ready! (it took',t2-t1,' seconds)')
        screen=self.game.set_display()
        self.game.create_pieces()
        print("Turn (0):")
        print(self.game.board_state)

    def one_turn(self):
        if self.game.white_turn:
            if self.white_pieces_AI:
                self.ai_move()
            else:
                self.player_move()
            print("Turn("+str(self.turn)+") white's move is:")
        else:
            if self.black_pieces_AI:
                self.ai_move()
            else:
                self.player_move()
            print("Turn("+str(self.turn)+") black's move is:")

        print(self.game.board_state)
        print("<white wins:"+str(self.game.white_wins)+"> <black wins:"+str(self.game.black_wins)+"> <ties:"+str(self.game.ties)+">")

    def game_loop(self):
        global win
        win=False
        while not win:
            print("---------------------------------------")
            self.one_turn()
            if self.game.white_turn and self.game.check_white_victory():
                print("---------------------------------------")
                print("WHITE WINS!!! (in "+str(self.dif)+" moves)")
                win=True
            elif self.game.check_black_victory():
                print("---------------------------------------")
                print("BLACK WINS!!! (in "+str(self.dif)+" moves)")
                win=True
            self.turn+=1
        print(str(len(nodes))+" board positions were evaluated during this game")
        print("---------------------------------------")
        tr.mainloop()


def createtr(li,c,a=45):
    dic = {}
    for p in li:
        tw=tr.Turtle(shape="turtle", visible=False)
        tw.speed(0)
        tw.turtlesize(1.5*scale,1.5*scale)
        tw.pu()
        tw.color(c)
        tw.goto((40*p[0]-140)*scale,(40*p[1]-140)*scale)
        tw.tilt(a)
        tw.showturtle()
        dic[p] = tw
    return dic

scale=2
White = 0; Black = 0
White = input('White as a computer? y or n \n')
Black = input('Black as a computer? y or n \n')

dif = 0
while dif<1 or dif>5:
    dif = int(input('Dificulte level? 1-5 \n'))

game=game(White=='y',Black=='y',dif)
game.set_board()
game.game_loop()
