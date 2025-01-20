#Priyanka Pramod Wani
from abc import ABC, abstractmethod
import math
import random
import sys
import time

class MovingEntity(ABC): # Abstract class represting moving objects like monsters and act_man
    def __init__(self, pos, move_strategy): # pos -> current (x,y) position of moving entity
        self.pos = pos
        self.move_strategy = move_strategy
    
    @abstractmethod
    def move(self):
        pass
    
class ActMan(MovingEntity): # ActMan is a MovingEntity
    def __init__(self, pos, fire_attempts, score):
        super().__init__(pos, ActManRandomMoveStrategy(None, None))
        self.fire_strategy = ActManRandomFireStrategy(None, None)
        self.fire_attempts = fire_attempts
        self.score = score
        self.actions = []
        self.is_alive = True
    
    def move(self):
        if self.score <= 0 or not self.is_alive:
            return False

        to_fire = random.choice([True, False])
        if to_fire and self.fire_attempts > 0 and self.score >= 20:
            self.score -= 20
            self.fire_attempts -= 1
            fire_result = self.fire_strategy.fire(self.pos)
            self.score += fire_result["score"]
            if fire_result["direction"] != None:
                self.actions.append(fire_result["direction"])
            return True
        
        move_result = self.move_strategy.move(self.pos)
        if move_result["is_alive"] == None: 
            self.is_alive = False
            return False
        self.pos = move_result["new_pos"]
        self.is_alive = move_result["is_alive"]
        self.actions.append(move_result["direction"])
        self.score = self.score-1 if self.is_alive else 0
        return self.is_alive

class Monster(MovingEntity): # Monster is a MovingEntity
    def __init__(self, pos, type):
        super().__init__(pos, MonsterMoveStrategyUsingEuclidDistanceToActMan(None, None, None))
        self.type = type
        self.is_alive = True
    
    def move(self):
        move_result = self.move_strategy.move(self.pos)
        self.pos = move_result["new_pos"]
        self.is_alive = move_result["is_alive"]
        return self.is_alive

class TurnStrategy(ABC): # Represents strategy of entities having turns like move or fire
    def __init__(self, dungeon, monsters):
        self.dungeon = dungeon
        self.monsters = monsters
        self.deltas_for_numeric_directions = {'7': (-1, -1), '8': (-1, 0), '9': (-1, 1), '4': (0, -1), '6': (0, 1), '1': (1, -1), '2': (1, 0), '3': (1, 1)}
        self.clockwise_numeric_directions = ['8', '9', '6', '3', '2', '1', '4', '7']
        self.anti_clockwise_numeric_directions = ['8', '7', '4', '1', '2', '3', '6', '9']
        self.literal_directions = ['N', 'E', 'S', 'W']
        self.deltas_for_literal_directions = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}

    def is_valid_move(self, pos, dr, dc):
        nr, nc = pos[0] + dr, pos[1] + dc
        return 1 <= nr < len(self.dungeon) - 1 and 1 <= nc < len(self.dungeon[0]) - 1 and self.dungeon[nr][nc] != '#'
    
    def euclidean_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

class MoveStrategy(TurnStrategy, ABC): # Strategy for moving entities, any new moving strategy can be added as subclass of this
    def __init__(self, dungeon, monsters):
        super().__init__(dungeon, monsters)

    @abstractmethod
    def move(self, cur_pos):
        pass

class FireStrategy(TurnStrategy, ABC): # Strategy for firing entities, any new firing strategy can be added as subclass of this
    def __init__(self, dungeon, monsters):
        super().__init__(dungeon, monsters)
    
    @abstractmethod
    def fire(self, cur_pos):
        pass

class ActManRandomMoveStrategy(MoveStrategy): # Random moving strategy for ActMan
    def __init__(self, dungeon, monsters):
        super().__init__(dungeon, monsters)
    
    def move(self, cur_pos):
        if self.dungeon[cur_pos[0]][cur_pos[1]] != 'A':
            # print("this is act man move function, it should be called for act man only!!")
            return {"direction": None, "is_alive": None, "new_pos": None}
        
        direction = random.choice(self.clockwise_numeric_directions)
        delta = self.deltas_for_numeric_directions.get(direction)
        while True:
            if delta is not None and self.is_valid_move(cur_pos, delta[0], delta[1]):
                break
            direction = random.choice(self.clockwise_numeric_directions)
            delta = self.deltas_for_numeric_directions.get(direction)
        
        is_alive = True
        new_x, new_y = cur_pos[0] + delta[0], cur_pos[1] + delta[1]

        self.dungeon[new_x][new_y] = 'X' if (self.dungeon[new_x][new_y] in 'DG@') else 'A'
        self.dungeon[cur_pos[0]][cur_pos[1]] = ' '
        if self.dungeon[new_x][new_y] == 'X':
            is_alive = False

        # print(f"Act-Man moved {direction}:")
        return {"direction": direction, "is_alive": is_alive, "new_pos": (new_x, new_y)}

class MonsterMoveStrategyUsingEuclidDistanceToActMan(MoveStrategy): # Moving strategy for Monsters considering Shortest Euclidean Distance from ActMan
    def __init__(self, dungeon, monsters, act_man):
        super().__init__(dungeon, monsters)
        self.act_man = act_man

    def move(self, cur_pos):
        x = cur_pos[0]
        y = cur_pos[1]
        monster_type = self.dungeon[x][y]
        if self.dungeon[x][y] not in 'DG':
            # print("not a monster!!")
            return {"is_alive": None, "new_pos": None}

        possible_moves = []
        for direction in self.clockwise_numeric_directions:
            delta = self.deltas_for_numeric_directions[direction]
            new_x, new_y = x + delta[0], y + delta[1]
            if self.is_valid_move((x, y), delta[0], delta[1]):  # Check if the move is valid (not into walls or outer edges)
                possible_moves.append((new_x, new_y, direction))

        if not possible_moves:
            return {"is_alive": None, "new_pos": None}

        # Group moves by distance to Act-Man
        grouped_moves = {}
        for move in possible_moves:
            distance = self.euclidean_distance((move[0], move[1]), self.act_man.pos)
            if distance not in grouped_moves:
                grouped_moves[distance] = []
            grouped_moves[distance].append(move)

        # Find the shortest distance and get the moves with that distance
        min_distance = min(grouped_moves.keys())
        closest_moves = grouped_moves[min_distance]

        # If there's more than one closest move, apply preference
        if len(closest_moves) > 1:
            if monster_type == 'G':
                preferred_order = self.clockwise_numeric_directions
            elif monster_type == 'D':
                preferred_order = self.anti_clockwise_numeric_directions
            best_move = sorted(closest_moves, key=lambda move: preferred_order.index(move[2]))[0]
        else:
            best_move = closest_moves[0]

        is_alive = False

        new_pos_entity = self.dungeon[best_move[0]][best_move[1]]
        self.dungeon[x][y] = ' ' # Clear the old position of the monster
        old_positioned_monster = self.monsters[(x, y)]
        del self.monsters[(x, y)]  # Remove the old positioned monster
        
        # Handle collisions with other monsters or corpses or Act-man:
        if new_pos_entity == 'A':
            # self.dungeon[x][y] = ' ' # Clear the old position of the monster
            self.dungeon[self.act_man.pos[0]][self.act_man.pos[1]] = 'X'
            self.act_man.score = 0
            self.act_man.is_alive = False
            # is_alive = False
        elif new_pos_entity == '@':
            # self.dungeon[x][y] = ' ' # Clear the old position of the monster
            self.act_man.score += 5  # 1 monster died
            # del self.monsters[(x, y)]  # Remove the dead monster
            # is_alive = False
        elif new_pos_entity in 'DG':
            # self.dungeon[x][y] = ' '    # Clear the old position of the monster
            # del self.monsters[(x, y)]  # Remove the dead monster
            self.dungeon[best_move[0]][best_move[1]] = '@'
            del self.monsters[(best_move[0], best_move[1])]  # Remove the other dead monster
            self.act_man.score += 10    # 2 monsters died at once
            # is_alive = False
        else:
            is_alive = True
            # self.dungeon[x][y] = ' '
            self.monsters[(best_move[0], best_move[1])] = old_positioned_monster
            # del self.monsters[(x, y)]
            self.dungeon[best_move[0]][best_move[1]] = monster_type
            
        # print(f"Monster {monster_type} moved from ({x}, {y}) to ({best_move[0]}, {best_move[1]}):")
        return {"is_alive": is_alive, "new_pos": (best_move[0], best_move[1])}

class ActManRandomFireStrategy(FireStrategy): # Random firing strategy for ActMan
    def __init__(self, dungeon, monsters):
        super().__init__(dungeon, monsters)

    def fire(self, cur_pos):
        if self.dungeon[cur_pos[0]][cur_pos[1]] != 'A':
            # print("this is act man fire function, it should be called for act man only!!")
            return {"score": 0, "direction": None}
        
        direction = random.choice(self.literal_directions)
        delta = self.deltas_for_literal_directions[direction]
        score = 0

        bullet_pos = cur_pos
        while self.is_valid_move(bullet_pos, delta[0], delta[1]):
            bullet_pos = (bullet_pos[0] + delta[0], bullet_pos[1] + delta[1])
            if self.dungeon[bullet_pos[0]][bullet_pos[1]] in 'DG':
                del self.monsters[(bullet_pos[0], bullet_pos[1])]
                self.dungeon[bullet_pos[0]][bullet_pos[1]] = '@'
                score += 5

        # print(f"Fired magic bullet {direction}:")
        return {"score": score, "direction": direction}

class ActManGame: # Actual game
    def __init__(self, input_file):
        self.dungeon = []
        self.monsters = {}
        self.act_man: ActMan = ActMan((0,0), 1, 50)
        self.load_dungeon(input_file)

    def load_dungeon(self, input_file):
        with open(input_file, 'r') as file:
            rows, cols = map(int, file.readline().split())
            for r in range(rows):
                row = list(file.readline().strip())
                self.dungeon.append(row)
                for c, cell in enumerate(row):
                    if cell == 'A':
                        self.act_man = ActMan(pos=(r,c), fire_attempts=1,score=50)
                    elif cell in ('D', 'G'):
                        self.monsters[(r, c)] = Monster(pos=(r,c), type=cell)
        
        self.act_man.move_strategy = ActManRandomMoveStrategy(self.dungeon, self.monsters)
        self.act_man.fire_strategy = ActManRandomFireStrategy(self.dungeon, self.monsters)
        for monster in self.monsters.values():
            monster.move_strategy = MonsterMoveStrategyUsingEuclidDistanceToActMan(self.dungeon, self.monsters, self.act_man)
        
        # print("Initial Dungeon:")
        # self.write_to_file()
    
    def play_game(self, output_file):
        while self.act_man.is_alive and len(self.act_man.actions) < 7 and self.act_man.score > 0 and self.monsters:
            if self.act_man.move():
                # self.write_to_file()
                if self.act_man.score <= 0: # do not continue if act_man exhausted its score
                    break
                for pos, monster in list(self.monsters.items()):
                    if self.dungeon[pos[0]][pos[1]] not in 'DG': # check if this monster is already dead, by some other previous monster colliding into its cell
                        continue
                    monster.move()
                    # self.write_to_file()
                    if not self.act_man.is_alive or not self.monsters: # check if act_man is dead or all monsters are dead
                        break
        
        # time.sleep(2)
        with open(output_file, 'w') as file:
            file.write(''.join(self.act_man.actions) + '\n')
            file.write(str(self.act_man.score) + '\n')
            for row in self.dungeon:
                file.write(''.join(row) + '\n')

    # def clean_file(self):
    #     with open(output_file, 'w'):
    #         pass

    # def write_to_file(self):
    #     self.clean_file()
    #     with open(output_file, 'w') as file:
    #         for row in self.dungeon:
    #             file.write(''.join(row) + '\n')
    #     time.sleep(2)

    def print_dungeon(self):
        for row in self.dungeon:
            print(''.join(row))
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python <script_name>.py <input_file>.txt <output_file>.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    game = ActManGame(input_file)
    game.play_game(output_file)