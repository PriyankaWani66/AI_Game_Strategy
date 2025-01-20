from abc import ABC, abstractmethod
import math
from numbers import Number
import pickle
import sys
import time
import copy
from collections import deque


class MovingEntity(ABC):  # Abstract class representing moving objects like monsters and Act-Man
    def __init__(self, pos, move_strategy):  # pos -> current (x,y) position of moving entity
        self.pos = pos
        self.move_strategy = move_strategy

    @abstractmethod
    def move(self):
        pass


class ActMan(MovingEntity):  # Act-Man is a MovingEntity
    def __init__(self, pos, fire_attempts, score):
        super().__init__(pos, None)  # Act-Man's move strategy will be set later
        self.fire_attempts = fire_attempts
        self.score = score
        self.actions = []
        self.is_alive = True

    def move(self, action, dungeon_copy, act_man_copy):
        if self.score <= 0 or not self.is_alive:
            return False

        if action is None:
            return False

        if action in self.move_strategy.move_strategy.deltas_for_numeric_directions:
            move_result = self.move_strategy.move_strategy.move(self.pos, action, dungeon_copy, act_man_copy)
            if move_result["new_pos"] is not None:
                act_man_copy.pos = move_result["new_pos"]
                act_man_copy.is_alive = move_result["is_alive"]
                if act_man_copy.is_alive:
                    act_man_copy.actions.append(action)
                    act_man_copy.score -= 1  # Deduct score for movement only if Act-Man is still alive
                return act_man_copy.is_alive

        elif action in self.move_strategy.fire_strategy.deltas_for_literal_directions:
            fire_result = self.move_strategy.fire_strategy.fire(self.pos, action, dungeon_copy, self.move_strategy.monsters, act_man_copy)
            act_man_copy.score -= 20
            act_man_copy.score += fire_result["score"]
            act_man_copy.actions.append(action)
            return True

        return False



class Monster(MovingEntity):  # Monster is a MovingEntity
    def __init__(self, pos, type):
        super().__init__(pos, None)  # Monster's move strategy will be set later
        self.type = type
        self.is_alive = True

    def move(self, dungeon_copy):
        move_result = self.move_strategy.move(self.pos ,dungeon_copy, self)
        if move_result["is_alive"] is not None:
            self.pos = move_result["new_pos"]
            self.is_alive = move_result["is_alive"]
            return move_result
        else:
            return {"is_alive": False, "new_pos": None, "score_change": 0}
    


class TurnStrategy(ABC):  # Represents strategy of entities having turns like move or fire
    def __init__(self, dungeon, monsters):
        self.dungeon = dungeon
        self.monsters = monsters
        self.deltas_for_numeric_directions = {'8': (-1, 0), '9': (-1, 1), '6': (0, 1), '3': (1, 1), '2': (1, 0), '1': (1, -1), '4': (0, -1), '7': (-1, -1)}
        self.clockwise_numeric_directions = ['8', '9', '6', '3', '2', '1', '4', '7']
        self.anti_clockwise_numeric_directions = ['8', '7', '4', '1', '2', '3', '6', '9']
        self.literal_directions = ['N', 'E', 'S', 'W']
        self.deltas_for_literal_directions = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}

    def is_valid_move(self, pos, dr, dc):
        nr, nc = pos[0] + dr, pos[1] + dc
        return 1 <= nr < len(self.dungeon) - 1 and 1 <= nc < len(self.dungeon[0]) - 1 and self.dungeon[nr][nc] != '#'

    def euclidean_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


class MoveStrategy(TurnStrategy, ABC):  # Strategy for moving entities, any new moving strategy can be added as a subclass of this
    def __init__(self, dungeon, monsters):
        super().__init__(dungeon, monsters)

    @abstractmethod
    def move(self, cur_pos, direction,  dungeon_copy, act_man_copy):
        pass


class FireStrategy(TurnStrategy, ABC):  # Strategy for firing entities, any new firing strategy can be added as a subclass of this
    def __init__(self, dungeon, monsters):
        super().__init__(dungeon, monsters)

    @abstractmethod
    def fire(self, cur_pos, direction, dungeon_copy, act_man_copy):
        pass


class ActManMoveStrategy(MoveStrategy):
    def __init__(self, dungeon, monsters, act_man):
        super().__init__(dungeon, monsters)
        self.act_man = act_man

    def move(self, cur_pos, direction, dungeon_copy, act_man_copy):
        if dungeon_copy[cur_pos[0]][cur_pos[1]] != 'A':
            return {"direction": None, "is_alive": None, "new_pos": None}
        delta = self.deltas_for_numeric_directions[direction]
        if delta is not None and self.is_valid_move(cur_pos, delta[0], delta[1]) == False:
            return False
        delta = self.deltas_for_numeric_directions.get(direction)

        is_alive = True
        new_x, new_y = cur_pos[0] + delta[0], cur_pos[1] + delta[1]

        dungeon_copy[new_x][new_y] = 'X' if (dungeon_copy[new_x][new_y] in 'DG@') else 'A'
        dungeon_copy[cur_pos[0]][cur_pos[1]] = ' '
        if dungeon_copy[new_x][new_y] == 'X':
            is_alive = False
        act_man_copy.pos = (new_x, new_y)

        return {"direction": direction, "is_alive": is_alive, "new_pos": (new_x, new_y)}
    


class ActManFireStrategy(FireStrategy):
    def __init__(self, dungeon, monsters, act_man):
        super().__init__(dungeon, monsters)
        self.act_man = act_man

    def fire(self, cur_pos, direction, dungeon_copy, monsters_copy, act_man_copy):
        if dungeon_copy[cur_pos[0]][cur_pos[1]] != 'A':
            return {"score": 0, "direction": None}
        delta = self.deltas_for_literal_directions[direction]
        bullet_pos = cur_pos
        score = 0
        while self.is_valid_move(bullet_pos, delta[0], delta[1]):
            bullet_pos = (bullet_pos[0] + delta[0], bullet_pos[1] + delta[1])
            if dungeon_copy[bullet_pos[0]][bullet_pos[1]] in 'DG':
                del monsters_copy[(bullet_pos[0], bullet_pos[1])]
                dungeon_copy[bullet_pos[0]][bullet_pos[1]] = '@'
                score += 5
        return {"direction": direction, "score": score}


class IDSActManStrategy(MoveStrategy):
    def __init__(self, dungeon, monsters, act_man):
        self.dungeon = dungeon
        self.monsters = monsters
        self.act_man = act_man
        self.move_strategy = ActManMoveStrategy(dungeon, monsters, act_man)
        self.fire_strategy = ActManFireStrategy(dungeon, monsters, act_man)
        self.monster_move_strategy = MonsterMoveStrategyUsingEuclidDistanceToActMan(dungeon, monsters, act_man)
    
    def move(self, cur_pos, direction, dungeon_copy, act_man_copy):
        pass

    def create_copy(self, dungeon, monsters, act_man):
        dungeon_copy = copy.deepcopy(dungeon)
        monsters_copy = copy.deepcopy(monsters)
        act_man_copy = copy.deepcopy(act_man)
        for monster in monsters_copy.values():
            monster.move_strategy = MonsterMoveStrategyUsingEuclidDistanceToActMan(dungeon_copy, monsters_copy, act_man_copy)
        act_man_copy.move_strategy = IDSActManStrategy(dungeon_copy, monsters_copy, act_man_copy)
        return (dungeon_copy, monsters_copy, act_man_copy)
    
    def print_dungeon(self, dungeon):
        for row in dungeon:
            print(''.join(row))
        print()

    def write_to_file(self, dungeon):
        output_file = "output.txt"
        with open(output_file, 'w') as file:
            for row in dungeon:
                file.write(''.join(row) + '\n')

    def act_man_2_bounded_dfs(self, depth_limit):
        dungeon_copy, monsters_copy, act_man_copy = self.create_copy(self.dungeon, self.monsters, self.act_man)
        frontier = [(copy.deepcopy(act_man_copy.pos),  # pos
                    copy.deepcopy(act_man_copy.score),  # score
                    copy.deepcopy(act_man_copy.fire_attempts),  # fire_attempts
                    dungeon_copy,  # state_dungeon
                    monsters_copy,  # state_monsters
                    act_man_copy,  # state_act_man
                    [],  # actions
                    [self.dungeon])]  # dungeon_state_sequence
        best_sequence = None
        best_dungeon_state_sequence = None
        best_score = 0
        initial_monsters = self.monsters.__len__()
        max_monsters_died = 0
        visited = set()
        while frontier:
            pos, score, fire_attempts, state_dungeon, state_monsters, state_act_man, actions, dungeon_state_sequence = frontier.pop()
            if len(actions) == depth_limit:
                if not state_monsters and len(actions)<=10:  # Victory condition when no monsters left
                    if best_sequence is None or len(actions) == len(best_sequence):
                        best_sequence = actions  # Update the best sequence if it's longer than the current best
                        best_dungeon_state_sequence = dungeon_state_sequence
                        best_score = score
                        return (best_sequence, best_dungeon_state_sequence, best_score)
            
                elif len(actions) == 10:
                    if best_sequence is None or max_monsters_died < initial_monsters - len(state_monsters):
                        best_sequence = actions  # Update the best sequence if it's longer than the current best
                        best_dungeon_state_sequence = dungeon_state_sequence
                        best_score = score
                        max_monsters_died = initial_monsters - len(state_monsters)
                    continue              
                
            if len(actions) > depth_limit:
                continue  # Skip further exploration if depth limit is reached

            # Avoid revisiting the same state
            state = (pos, frozenset([(m.pos, m.is_alive) for m in state_monsters.values()]), fire_attempts, pickle.dumps(state_dungeon), pickle.dumps(state_monsters), pickle.dumps(state_act_man))
            if state in visited:
                continue
            visited.add(state)

            # Explore all possible moves
            for direction in self.move_strategy.deltas_for_numeric_directions:
                dungeon_copy, monsters_copy, act_man_copy = self.create_copy(state_dungeon, state_monsters, state_act_man)
                move_result = self.move_strategy.move(pos, direction, dungeon_copy, act_man_copy)
                if move_result==False:
                    continue
                if move_result['is_alive']:
                    new_pos = move_result['new_pos']
                    # Update monster positions based on Act-Man's move
                    updated_monsters, monster_Score, new_dungeon_copy, new_monsters_copy = self.update_monster_positions(dungeon_copy, monsters_copy, new_pos, act_man_copy)
                    if updated_monsters is not None:  # Only add the state to the queue if Act-Man survives
                        dungeon_copy = new_dungeon_copy
                        monsters_copy = new_monsters_copy
                        new_actions = actions + [direction]
                        new_dungeon_state_sequence = dungeon_state_sequence + [dungeon_copy]
                        frontier.append((new_pos, score - 1 + monster_Score, fire_attempts, dungeon_copy, monsters_copy, act_man_copy, new_actions, new_dungeon_state_sequence))
                        
        # Explore firing in all directions if fire attempts are available
            if fire_attempts > 0:
                for direction in self.fire_strategy.deltas_for_literal_directions:
                    dungeon_copy, monsters_copy, act_man_copy = self.create_copy(state_dungeon, state_monsters, state_act_man)
                    fire_result = self.fire_strategy.fire(pos, direction, dungeon_copy, monsters_copy, act_man_copy)
                    updated_monsters, monster_Score, new_dungeon_copy, new_monsters_copy = self.update_monster_positions(dungeon_copy, monsters_copy, pos, act_man_copy)
                    if updated_monsters is not None: # Only add the state to the queue if Act-Man survives
                        dungeon_copy = new_dungeon_copy
                        monsters_copy = new_monsters_copy
                        new_actions = actions + [direction]
                        new_dungeon_state_sequence = dungeon_state_sequence + [dungeon_copy]
                        frontier.append((pos, score - 20 + fire_result["score"] + monster_Score, fire_attempts - 1, dungeon_copy, monsters_copy, act_man_copy, new_actions, new_dungeon_state_sequence))


    
        return (best_sequence, best_dungeon_state_sequence, best_score) if best_sequence is not None else None
            

    def act_man_2_id_dfs(self):
        depth = 1
        while True:
            result = self.act_man_2_bounded_dfs(depth)
            if result is not None:
                return result  # Return the first sequence found
            depth += 1  # Increase depth and continue searching

    def update_monster_positions(self, dungeon_copy, monsters_copy, act_man_pos, act_man_copy):
        updated_monsters = {}
        monsters_score = 0
        move_result = self.monster_move_strategy.move(dungeon_copy, monsters_copy, act_man_pos, act_man_copy)
        if act_man_copy.is_alive == False:
            return None, 0, move_result[2], move_result[3]
        dungeon_copy = move_result[2]
        monsters_copy = move_result[3]
        return move_result

class MonsterMoveStrategyUsingEuclidDistanceToActMan(MoveStrategy):  # Moving strategy for Monsters considering Shortest Euclidean Distance from Act-Man
    def __init__(self, dungeon, monsters, act_man):
        super().__init__(dungeon, monsters)
        self.act_man = act_man

    def calculate_potential_moves(self, dungeon_copy, monsters, act_man_pos):
        next_moves = {}
        for monster in monsters.values():
            if not monster.is_alive:
                continue  # Skip the monster if it's not alive
            x, y = monster.pos
            monster_type = dungeon_copy[x][y]

            possible_moves = []
            for direction in self.clockwise_numeric_directions:
                delta = self.deltas_for_numeric_directions[direction]
                new_x, new_y = x + delta[0], y + delta[1]
                if self.is_valid_move((x, y), delta[0], delta[1]):
                    possible_moves.append((new_x, new_y, direction))

            if not possible_moves:
                continue

            grouped_moves = {}
            for move in possible_moves:
                distance = self.euclidean_distance((move[0], move[1]), act_man_pos)
                if distance not in grouped_moves:
                    grouped_moves[distance] = []
                grouped_moves[distance].append(move)

            min_distance = min(grouped_moves.keys())
            closest_moves = grouped_moves[min_distance]

            if len(closest_moves) > 1:
                preferred_order = self.clockwise_numeric_directions if monster_type == 'G' else self.anti_clockwise_numeric_directions
                best_move = sorted(closest_moves, key=lambda move: preferred_order.index(move[2]))[0]
            else:
                best_move = closest_moves[0]

            next_moves[monster] = (best_move[0], best_move[1])

        return next_moves

    def print_dungeon(self, dungeon):
        for row in dungeon:
            print(''.join(row))
        print()

    def move(self, dungeon_copy, monsters_copy, act_man_pos, act_man_copy):
        
        next_moves = self.calculate_potential_moves(dungeon_copy, monsters_copy, act_man_copy.pos)
        new_dungeon_copy = copy.deepcopy(dungeon_copy)
        rows = len(new_dungeon_copy)
        cols = len(new_dungeon_copy[0])

        for i in range(rows):
            for j in range(cols):
                if new_dungeon_copy[i][j] in 'DG':
                    new_dungeon_copy[i][j] = ' '

                
        map = {}
        monsters_died=0
        new_monsters={}

        for monster, next_move in next_moves.items():
            map[next_move]=None

        for monster, next_move in next_moves.items():
            if map[next_move]==None:
                map[next_move] = monster
            else: 
                if type(map[next_move]) == type(Monster(None, 'D')):
                    map[next_move]=2
                else: map[next_move]+=1
            if new_dungeon_copy[next_move[0]][next_move[1]] in 'AX':
                new_dungeon_copy[next_move[0]][next_move[1]]='X'
                act_man_copy.score = 0
                act_man_copy.is_alive = False
            if type(map[next_move]) != type(Monster(None, 'D')) and map[next_move]>1:
                new_dungeon_copy[next_move[0]][next_move[1]]='@'

        for next_move, freq in map.items():
            if type(freq) == type(Monster(None, 'D')):
                if new_dungeon_copy[next_move[0]][next_move[1]]!='@':
                    new_dungeon_copy[next_move[0]][next_move[1]]=freq.type
                    new_monsters[next_move]=freq
                    freq.pos=next_move
                else: monsters_died+=1
            else:
                monsters_died+=freq

        dungeon_copy=new_dungeon_copy
        monsters_copy=new_monsters
        self.monsters=new_monsters
        self.dungeon=new_dungeon_copy
        self.act_man=act_man_copy
        act_man_copy.score += monsters_died*5

        return (monsters_copy, monsters_died*5, new_dungeon_copy, new_monsters)


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
        
        self.act_man.move_strategy = IDSActManStrategy(self.dungeon, self.monsters, self.act_man)
        for monster in self.monsters.values():
            monster.move_strategy = MonsterMoveStrategyUsingEuclidDistanceToActMan(self.dungeon, self.monsters, self.act_man)

    def play_game(self, output_file):
        action_sequence, best_dungeon_state_sequence, best_score = self.act_man.move_strategy.act_man_2_id_dfs()
            
        if action_sequence is None:
            print("No valid sequence of actions found.")
            return
        self.dungeon = best_dungeon_state_sequence[(best_dungeon_state_sequence.__len__()) - 1]  # Update the game's dungeon state to the last state in the sequence
        self.act_man.score = best_score  # Update Act-Man's score

        self.act_man.actions = action_sequence  # Update Act-Man's actions

        if not self.act_man.is_alive or not self.monsters:  # Game ends if Act-Man is dead or all monsters are defeated
           return

        self.write_to_file(output_file)
        
    def update_monsters(self):
        for pos, monster in list(self.monsters.items()):
            if self.dungeon[pos[0]][pos[1]] in 'DG':  # Check if the monster is still alive
                monster.move(self.dungeon)
                if not monster.is_alive:  # Check if the monster died after moving
                    del self.monsters[pos]  # Remove the dead monster from the dictionary
                else:
                    # Update the monster's position in the dictionary if it moved
                    if pos != monster.pos:
                        del self.monsters[pos]
                        self.monsters[monster.pos] = monster
        
    def write_to_file(self, output_file):
        with open(output_file, 'w') as file:
            file.write(''.join(self.act_man.actions) + '\n')
            file.write(str(self.act_man.score) + '\n')
            for row in self.dungeon:
                file.write(''.join(row) + '\n')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python <script_name>.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    game = ActManGame(input_file)
    game.play_game(output_file)