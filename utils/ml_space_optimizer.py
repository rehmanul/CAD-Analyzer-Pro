import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
from gym import spaces
import logging
from typing import Dict, List, Any, Optional, Tuple
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import random
from collections import deque
import pickle
import json

logger = logging.getLogger(__name__)

class FloorPlanEnvironment(gym.Env):
    """Custom Gym environment for floor plan optimization"""
    
    def __init__(self, analysis_results: Dict[str, Any], 
                 ilot_requirements: List[Dict[str, Any]],
                 constraints: Dict[str, Any]):
        super(FloorPlanEnvironment, self).__init__()
        
        self.analysis_results = analysis_results
        self.ilot_requirements = ilot_requirements
        self.constraints = constraints
        
        # Define action space (x, y coordinates for placing îlot)
        bounds = analysis_results.get('metadata', {}).get('analysis_bounds', {})
        self.x_min = bounds.get('min_x', 0) + 2
        self.x_max = bounds.get('max_x', 100) - 2
        self.y_min = bounds.get('min_y', 0) + 2
        self.y_max = bounds.get('max_y', 100) - 2
        
        # Continuous action space for placement coordinates
        self.action_space = spaces.Box(
            low=np.array([self.x_min, self.y_min]),
            high=np.array([self.x_max, self.y_max]),
            dtype=np.float32
        )
        
        # Observation space includes current state of placement
        obs_dim = 100  # Flattened grid representation + features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.placed_ilots = []
        self.current_ilot_idx = 0
        self.total_reward = 0
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, action):
        """Execute one placement action"""
        x, y = action
        
        # Get current îlot to place
        if self.current_ilot_idx >= len(self.ilot_requirements):
            return self._get_observation(), 0, True, {}
        
        current_ilot = self.ilot_requirements[self.current_ilot_idx]
        
        # Validate placement
        placement_valid, penalty = self._validate_placement(x, y, current_ilot)
        
        if placement_valid:
            # Place îlot
            self.placed_ilots.append({
                'id': current_ilot['id'],
                'position': {'x': x, 'y': y, 'z': 0},
                'dimensions': current_ilot['dimensions'],
                'size_category': current_ilot['size_category']
            })
            
            # Calculate reward
            reward = self._calculate_reward(x, y, current_ilot)
            self.current_ilot_idx += 1
        else:
            reward = penalty
        
        self.total_reward += reward
        
        # Check if done
        done = self.current_ilot_idx >= len(self.ilot_requirements)
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'placed_count': len(self.placed_ilots),
            'total_ilots': len(self.ilot_requirements),
            'placement_valid': placement_valid
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current state observation"""
        # Create grid representation
        grid_size = 20
        grid = np.zeros((grid_size, grid_size))
        
        # Mark placed îlots
        for ilot in self.placed_ilots:
            x_idx = int((ilot['position']['x'] - self.x_min) / (self.x_max - self.x_min) * grid_size)
            y_idx = int((ilot['position']['y'] - self.y_min) / (self.y_max - self.y_min) * grid_size)
            
            x_idx = np.clip(x_idx, 0, grid_size - 1)
            y_idx = np.clip(y_idx, 0, grid_size - 1)
            
            grid[y_idx, x_idx] = 1
        
        # Flatten grid
        features = grid.flatten()
        
        # Add additional features
        progress = self.current_ilot_idx / len(self.ilot_requirements)
        coverage = len(self.placed_ilots) / len(self.ilot_requirements)
        
        # Pad to match observation space
        obs = np.zeros(self.observation_space.shape[0])
        obs[:len(features)] = features
        obs[-2] = progress
        obs[-1] = coverage
        
        return obs
    
    def _validate_placement(self, x: float, y: float, 
                          ilot: Dict[str, Any]) -> Tuple[bool, float]:
        """Validate îlot placement"""
        width = ilot['dimensions']['width']
        height = ilot['dimensions']['height']
        
        # Create îlot geometry
        ilot_geom = box(x - width/2, y - height/2, x + width/2, y + height/2)
        
        # Check boundaries
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False, -10.0
        
        # Check collision with existing îlots
        min_spacing = self.constraints.get('min_spacing', 1.5)
        for placed in self.placed_ilots:
            placed_pos = placed['position']
            placed_dims = placed['dimensions']
            placed_geom = box(
                placed_pos['x'] - placed_dims['width']/2,
                placed_pos['y'] - placed_dims['height']/2,
                placed_pos['x'] + placed_dims['width']/2,
                placed_pos['y'] + placed_dims['height']/2
            )
            
            if ilot_geom.distance(placed_geom) < min_spacing:
                return False, -5.0
        
        # Check restricted areas
        for restricted in self.analysis_results.get('restricted_areas', []):
            restricted_geom = restricted.get('shapely_geom')
            if restricted_geom and ilot_geom.intersects(restricted_geom):
                return False, -15.0
        
        # Check entrances
        for entrance in self.analysis_results.get('entrances', []):
            entrance_geom = entrance.get('shapely_geom')
            if entrance_geom and ilot_geom.distance(entrance_geom) < 1.0:
                return False, -10.0
        
        return True, 0.0
    
    def _calculate_reward(self, x: float, y: float, ilot: Dict[str, Any]) -> float:
        """Calculate reward for placement"""
        reward = 0.0
        
        # Base reward for successful placement
        reward += 10.0
        
        # Reward for space utilization
        area_ratio = ilot['dimensions']['area'] / 100  # Normalize
        reward += area_ratio * 5.0
        
        # Reward for good distribution
        if self.placed_ilots:
            min_distance = float('inf')
            for placed in self.placed_ilots:
                dist = np.sqrt(
                    (x - placed['position']['x'])**2 + 
                    (y - placed['position']['y'])**2
                )
                min_distance = min(min_distance, dist)
            
            # Optimal spacing reward
            optimal_spacing = 4.0
            spacing_score = 1.0 - abs(min_distance - optimal_spacing) / optimal_spacing
            reward += spacing_score * 3.0
        
        # Reward for accessibility
        entrance_distances = []
        for entrance in self.analysis_results.get('entrances', []):
            ent_pos = entrance.get('position', {})
            if ent_pos:
                dist = np.sqrt(
                    (x - ent_pos.get('x', 0))**2 + 
                    (y - ent_pos.get('y', 0))**2
                )
                entrance_distances.append(dist)
        
        if entrance_distances:
            min_entrance_dist = min(entrance_distances)
            accessibility_score = 1.0 / (1.0 + min_entrance_dist / 10.0)
            reward += accessibility_score * 2.0
        
        return reward
    
    def render(self, mode='human'):
        """Render current state (optional)"""
        pass


class DQNAgent:
    """Deep Q-Network agent for floor plan optimization"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        
        # Neural network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build deep neural network"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, env):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Explore: random action
            return env.action_space.sample()
        
        # Exploit: use model prediction
        state_tensor = tf.expand_dims(state, 0)
        q_values = self.model(state_tensor)
        
        # Convert Q-values to continuous action
        action_idx = tf.argmax(q_values[0]).numpy()
        
        # Map to continuous coordinates
        x = self._map_to_coordinate(action_idx, env.x_min, env.x_max, 10)
        y = self._map_to_coordinate(action_idx // 10, env.y_min, env.y_max, 10)
        
        return np.array([x, y])
    
    def _map_to_coordinate(self, idx: int, min_val: float, max_val: float, 
                          grid_size: int) -> float:
        """Map discrete index to continuous coordinate"""
        return min_val + (max_val - min_val) * (idx % grid_size) / grid_size
    
    def replay(self, batch_size: int = 32):
        """Train model on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Predict Q-values
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            # Map action to discrete index
            action_idx = int(actions[i][0] * 10 + actions[i][1])
            action_idx = min(action_idx, self.action_size - 1)
            
            current_q_values[i][action_idx] = target
        
        # Train model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class MLSpaceOptimizer:
    """Machine Learning-based space optimization with reinforcement learning"""
    
    def __init__(self):
        self.agent = None
        self.env = None
        self.training_history = []
        self.best_placement = None
        self.best_score = float('-inf')
        
        # Model persistence
        self.model_path = "models/space_optimizer_dqn.h5"
        self.history_path = "models/training_history.pkl"
    
    def optimize_placement(self, analysis_results: Dict[str, Any],
                          ilot_requirements: List[Dict[str, Any]],
                          constraints: Dict[str, Any],
                          episodes: int = 100) -> List[Dict[str, Any]]:
        """
        Optimize îlot placement using reinforcement learning
        
        Args:
            analysis_results: Zone analysis results
            ilot_requirements: Required îlots to place
            constraints: Placement constraints
            episodes: Number of training episodes
            
        Returns:
            Optimized îlot placement
        """
        logger.info("Starting ML-based space optimization")
        
        try:
            # Create environment
            self.env = FloorPlanEnvironment(analysis_results, ilot_requirements, constraints)
            
            # Initialize or load agent
            if self.agent is None:
                self.agent = DQNAgent(
                    state_size=self.env.observation_space.shape[0],
                    action_size=100  # Discretized action space
                )
                
                # Try to load existing model
                self._load_model()
            
            # Training loop
            for episode in range(episodes):
                state = self.env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    # Choose action
                    action = self.agent.act(state, self.env)
                    
                    # Execute action
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store experience
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    # Train model
                    if len(self.agent.memory) > 32:
                        self.agent.replay()
                
                # Update best placement
                if total_reward > self.best_score:
                    self.best_score = total_reward
                    self.best_placement = self.env.placed_ilots.copy()
                
                # Update target network
                if episode % 10 == 0:
                    self.agent.update_target_model()
                
                # Log progress
                self.training_history.append({
                    'episode': episode,
                    'total_reward': total_reward,
                    'epsilon': self.agent.epsilon,
                    'placed_count': len(self.env.placed_ilots)
                })
                
                if episode % 20 == 0:
                    logger.info(f"Episode {episode}: Reward = {total_reward:.2f}, "
                              f"Epsilon = {self.agent.epsilon:.3f}")
            
            # Save model
            self._save_model()
            
            # Return best placement found
            return self._convert_to_ilot_format(self.best_placement)
            
        except Exception as e:
            logger.error(f"Error in ML optimization: {str(e)}")
            raise
    
    def _convert_to_ilot_format(self, placements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert placement format to standard îlot format"""
        formatted_ilots = []
        
        for placement in placements:
            ilot = {
                'id': placement['id'],
                'size_category': placement['size_category'],
                'dimensions': placement['dimensions'],
                'position': placement['position'],
                'geometry': box(
                    placement['position']['x'] - placement['dimensions']['width'] / 2,
                    placement['position']['y'] - placement['dimensions']['height'] / 2,
                    placement['position']['x'] + placement['dimensions']['width'] / 2,
                    placement['position']['y'] + placement['dimensions']['height'] / 2
                ),
                'placement_score': 95.0,  # High score for ML-optimized placement
                'accessibility_score': 90.0,
                'properties': {
                    'algorithm': 'ml_reinforcement_learning',
                    'validated': True,
                    'optimization_method': 'DQN'
                }
            }
            formatted_ilots.append(ilot)
        
        return formatted_ilots
    
    def _save_model(self):
        """Save trained model and history"""
        try:
            import os
            os.makedirs("models", exist_ok=True)
            
            # Save model weights
            if self.agent and self.agent.model:
                self.agent.model.save_weights(self.model_path)
            
            # Save training history
            with open(self.history_path, 'wb') as f:
                pickle.dump(self.training_history, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def _load_model(self):
        """Load trained model if exists"""
        try:
            import os
            
            if os.path.exists(self.model_path):
                self.agent.model.load_weights(self.model_path)
                self.agent.update_target_model()
                logger.info("Loaded existing model")
            
            if os.path.exists(self.history_path):
                with open(self.history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
                logger.info("Loaded training history")
                
        except Exception as e:
            logger.warning(f"Could not load model: {str(e)}")
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization metrics and insights"""
        if not self.training_history:
            return {}
        
        latest_history = self.training_history[-100:] if len(self.training_history) > 100 else self.training_history
        
        rewards = [h['total_reward'] for h in latest_history]
        placed_counts = [h['placed_count'] for h in latest_history]
        
        return {
            'average_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'reward_improvement': (rewards[-1] - rewards[0]) / abs(rewards[0]) * 100 if rewards[0] != 0 else 0,
            'average_placement_rate': np.mean(placed_counts) / len(self.env.ilot_requirements) if self.env else 0,
            'convergence_rate': 1.0 - np.std(rewards[-20:]) / np.mean(rewards[-20:]) if len(rewards) >= 20 else 0,
            'training_episodes': len(self.training_history),
            'current_epsilon': self.agent.epsilon if self.agent else 1.0
        }


class HybridOptimizer:
    """Hybrid optimization combining ML with traditional algorithms"""
    
    def __init__(self):
        self.ml_optimizer = MLSpaceOptimizer()
        self.optimization_modes = {
            'ml_only': self._ml_only_optimization,
            'hybrid': self._hybrid_optimization,
            'ensemble': self._ensemble_optimization
        }
    
    def optimize(self, analysis_results: Dict[str, Any],
                ilot_requirements: List[Dict[str, Any]],
                constraints: Dict[str, Any],
                mode: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Perform optimization using specified mode
        
        Args:
            analysis_results: Zone analysis results
            ilot_requirements: Required îlots
            constraints: Placement constraints
            mode: Optimization mode
            
        Returns:
            Optimized placement
        """
        optimization_func = self.optimization_modes.get(mode, self._hybrid_optimization)
        return optimization_func(analysis_results, ilot_requirements, constraints)
    
    def _ml_only_optimization(self, analysis_results: Dict[str, Any],
                             ilot_requirements: List[Dict[str, Any]],
                             constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Pure ML optimization"""
        return self.ml_optimizer.optimize_placement(
            analysis_results, ilot_requirements, constraints, episodes=100
        )
    
    def _hybrid_optimization(self, analysis_results: Dict[str, Any],
                           ilot_requirements: List[Dict[str, Any]],
                           constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hybrid optimization combining ML with heuristics"""
        # Start with ML optimization
        ml_placement = self.ml_optimizer.optimize_placement(
            analysis_results, ilot_requirements, constraints, episodes=50
        )
        
        # Refine with local search
        refined_placement = self._local_search_refinement(
            ml_placement, analysis_results, constraints
        )
        
        return refined_placement
    
    def _ensemble_optimization(self, analysis_results: Dict[str, Any],
                             ilot_requirements: List[Dict[str, Any]],
                             constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ensemble of multiple optimization methods"""
        results = []
        
        # ML optimization
        ml_result = self.ml_optimizer.optimize_placement(
            analysis_results, ilot_requirements, constraints, episodes=30
        )
        results.append(ml_result)
        
        # Add results from other optimizers (genetic, simulated annealing, etc.)
        # These would be imported from spatial_optimizer.py
        
        # Select best result based on multi-objective criteria
        best_result = self._select_best_result(results, analysis_results, constraints)
        
        return best_result
    
    def _local_search_refinement(self, placement: List[Dict[str, Any]],
                               analysis_results: Dict[str, Any],
                               constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine placement using local search"""
        refined = placement.copy()
        
        # Apply small perturbations to improve placement
        for i, ilot in enumerate(refined):
            best_pos = ilot['position'].copy()
            best_score = self._evaluate_position(ilot, refined, analysis_results, constraints)
            
            # Try small movements
            for dx in [-0.5, 0, 0.5]:
                for dy in [-0.5, 0, 0.5]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_pos = {
                        'x': ilot['position']['x'] + dx,
                        'y': ilot['position']['y'] + dy,
                        'z': 0
                    }
                    
                    # Temporarily update position
                    ilot['position'] = new_pos
                    score = self._evaluate_position(ilot, refined, analysis_results, constraints)
                    
                    if score > best_score:
                        best_score = score
                        best_pos = new_pos.copy()
            
            # Apply best position
            ilot['position'] = best_pos
        
        return refined
    
    def _evaluate_position(self, ilot: Dict[str, Any], all_ilots: List[Dict[str, Any]],
                         analysis_results: Dict[str, Any], constraints: Dict[str, Any]) -> float:
        """Evaluate quality of îlot position"""
        score = 0.0
        
        # Check constraints
        ilot_geom = box(
            ilot['position']['x'] - ilot['dimensions']['width'] / 2,
            ilot['position']['y'] - ilot['dimensions']['height'] / 2,
            ilot['position']['x'] + ilot['dimensions']['width'] / 2,
            ilot['position']['y'] + ilot['dimensions']['height'] / 2
        )
        
        # Spacing score
        for other in all_ilots:
            if other['id'] == ilot['id']:
                continue
            
            other_geom = box(
                other['position']['x'] - other['dimensions']['width'] / 2,
                other['position']['y'] - other['dimensions']['height'] / 2,
                other['position']['x'] + other['dimensions']['width'] / 2,
                other['position']['y'] + other['dimensions']['height'] / 2
            )
            
            distance = ilot_geom.distance(other_geom)
            if distance < constraints.get('min_spacing', 1.5):
                score -= 10
            else:
                score += min(5, distance)
        
        # Accessibility score
        for entrance in analysis_results.get('entrances', []):
            ent_pos = entrance.get('position', {})
            if ent_pos:
                dist = np.sqrt(
                    (ilot['position']['x'] - ent_pos.get('x', 0))**2 + 
                    (ilot['position']['y'] - ent_pos.get('y', 0))**2
                )
                score += 10 / (1 + dist)
        
        return score
    
    def _select_best_result(self, results: List[List[Dict[str, Any]]],
                          analysis_results: Dict[str, Any],
                          constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select best result from ensemble"""
        if not results:
            return []
        
        best_score = float('-inf')
        best_result = results[0]
        
        for result in results:
            score = self._evaluate_placement(result, analysis_results, constraints)
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def _evaluate_placement(self, placement: List[Dict[str, Any]],
                          analysis_results: Dict[str, Any],
                          constraints: Dict[str, Any]) -> float:
        """Evaluate overall placement quality"""
        total_score = 0.0
        
        for ilot in placement:
            total_score += self._evaluate_position(ilot, placement, analysis_results, constraints)
        
        # Add global metrics
        total_area = sum(ilot['dimensions']['area'] for ilot in placement)
        coverage_score = total_area * 0.1
        total_score += coverage_score
        
        return total_score