"""
Advanced AI Agent Swarm Intelligence System

Self-organizing agent collectives with quantum-enhanced coordination:
- Distributed problem-solving swarms
- Emergent intelligence from collective behavior
- Quantum-enhanced agent communication
- Dynamic swarm topology adaptation
- Multi-objective swarm optimization
"""

import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Agent roles in the swarm"""
    EXPLORER = "explorer"           # Explores new solution spaces
    EXPLOITER = "exploiter"         # Refines known good solutions
    COORDINATOR = "coordinator"     # Coordinates swarm activities  
    SPECIALIST = "specialist"       # Domain-specific expertise
    MEDIATOR = "mediator"          # Resolves conflicts between agents
    QUANTUM_ORACLE = "quantum_oracle"  # Quantum-enhanced reasoning

class SwarmTopology(Enum):
    """Swarm communication topologies"""
    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    HIERARCHICAL = "hierarchical"
    QUANTUM_ENTANGLED = "quantum_entangled"

class TaskComplexity(Enum):
    """Problem complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ULTRA_COMPLEX = "ultra_complex"

@dataclass
class AgentState:
    """State of an individual agent"""
    agent_id: str
    role: AgentRole
    position: np.ndarray  # Position in solution space
    velocity: np.ndarray  # Velocity vector
    best_solution: Dict[str, Any]
    best_fitness: float
    energy: float  # Agent energy level
    specialization: List[str]
    quantum_coherence: float
    learning_rate: float
    communication_range: float

@dataclass
class SwarmMessage:
    """Message passed between agents"""
    sender_id: str
    receiver_ids: List[str]
    message_type: str
    content: Dict[str, Any]
    quantum_entangled: bool
    timestamp: datetime
    urgency: int  # 1-10 priority

@dataclass
class SwarmObjective:
    """Multi-objective optimization target"""
    objective_id: str
    name: str
    weight: float
    minimize: bool
    fitness_function: str  # Function name or description
    constraints: List[str]

@dataclass
class SwarmPerformanceMetrics:
    """Swarm performance tracking"""
    swarm_id: str
    generation: int
    best_fitness: float
    average_fitness: float
    diversity_index: float
    convergence_rate: float
    quantum_advantage: float
    computational_efficiency: float
    timestamp: datetime

class QuantumAgent:
    """Individual quantum-enhanced agent"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        dimension: int,
        specialization: Optional[List[str]] = None
    ):
        self.state = AgentState(
            agent_id=agent_id,
            role=role,
            position=np.random.uniform(-10, 10, dimension),
            velocity=np.random.uniform(-1, 1, dimension),
            best_solution={},
            best_fitness=float('inf'),
            energy=1.0,
            specialization=specialization or [],
            quantum_coherence=np.random.uniform(0.5, 1.0),
            learning_rate=0.1,
            communication_range=5.0
        )
        
        self.memory = []
        self.social_connections: Set[str] = set()
        self.message_queue: List[SwarmMessage] = []
        self.quantum_state = self._initialize_quantum_state()
        
    def _initialize_quantum_state(self) -> Dict[str, Any]:
        """Initialize quantum state representation"""
        return {
            "superposition_states": np.random.uniform(-1, 1, len(self.state.position)),
            "entanglement_partners": [],
            "measurement_history": [],
            "coherence_time": 100  # Time steps before decoherence
        }
    
    async def update_position(
        self,
        global_best: np.ndarray,
        swarm_center: np.ndarray,
        inertia: float = 0.7,
        cognitive: float = 1.4,
        social: float = 1.4,
        quantum_influence: float = 0.2
    ):
        """Update agent position using quantum-enhanced PSO"""
        
        # Classical PSO components
        r1, r2 = np.random.random(2)
        
        # Quantum enhancement: superposition of multiple trajectories
        quantum_noise = self.quantum_state["superposition_states"][:len(self.state.position)]
        quantum_noise = quantum_noise * quantum_influence * self.state.quantum_coherence
        
        # Update velocity
        self.state.velocity = (
            inertia * self.state.velocity +
            cognitive * r1 * (self.state.position - self.state.position) +  # Personal best
            social * r2 * (global_best - self.state.position) +  # Global best
            quantum_noise  # Quantum influence
        )
        
        # Update position
        self.state.position += self.state.velocity
        
        # Apply quantum tunneling (small probability of jumping to distant regions)
        if np.random.random() < 0.01 * self.state.quantum_coherence:
            tunnel_direction = np.random.uniform(-1, 1, len(self.state.position))
            tunnel_distance = np.random.exponential(2.0)
            self.state.position += tunnel_direction * tunnel_distance
        
        # Update quantum coherence (gradually decays)
        self.state.quantum_coherence *= 0.999
        if self.state.quantum_coherence < 0.1:
            self.state.quantum_coherence = np.random.uniform(0.5, 1.0)  # Quantum refresh
    
    async def evaluate_fitness(
        self,
        fitness_function: Callable[[np.ndarray], float],
        objectives: List[SwarmObjective]
    ) -> float:
        """Evaluate fitness using multi-objective optimization"""
        
        total_fitness = 0.0
        
        for objective in objectives:
            if objective.fitness_function == "main":
                obj_fitness = fitness_function(self.state.position)
            else:
                # Mock additional objectives
                obj_fitness = np.sum(self.state.position ** 2) + np.random.normal(0, 0.1)
            
            if not objective.minimize:
                obj_fitness = -obj_fitness
                
            total_fitness += objective.weight * obj_fitness
        
        # Update personal best
        if total_fitness < self.state.best_fitness:
            self.state.best_fitness = total_fitness
            self.state.best_solution = {
                "position": self.state.position.copy(),
                "fitness": total_fitness,
                "timestamp": datetime.now()
            }
        
        return total_fitness
    
    async def send_message(
        self,
        message: SwarmMessage,
        quantum_entangled: bool = False
    ):
        """Send message to other agents"""
        
        if quantum_entangled:
            # Quantum entangled communication - instantaneous
            message.quantum_entangled = True
            message.content["quantum_state"] = self.quantum_state.copy()
        
        self.message_queue.append(message)
    
    async def process_messages(self) -> List[SwarmMessage]:
        """Process incoming messages"""
        
        processed_messages = []
        
        for message in self.message_queue:
            if self.state.agent_id in message.receiver_ids:
                # Process message based on type
                if message.message_type == "position_update":
                    await self._process_position_update(message)
                elif message.message_type == "solution_share":
                    await self._process_solution_share(message)
                elif message.message_type == "quantum_entanglement":
                    await self._process_quantum_entanglement(message)
                
                processed_messages.append(message)
        
        # Clear processed messages
        self.message_queue = [m for m in self.message_queue if self.state.agent_id not in m.receiver_ids]
        
        return processed_messages
    
    async def _process_position_update(self, message: SwarmMessage):
        """Process position update from other agents"""
        
        sender_position = np.array(message.content["position"])
        sender_fitness = message.content["fitness"]
        
        # Learn from better solutions
        if sender_fitness < self.state.best_fitness:
            learning_influence = self.state.learning_rate * self.state.quantum_coherence
            direction = sender_position - self.state.position
            self.state.velocity += learning_influence * direction
    
    async def _process_solution_share(self, message: SwarmMessage):
        """Process shared solution from other agents"""
        
        shared_solution = message.content["solution"]
        
        # Add to memory for future reference
        self.memory.append({
            "solution": shared_solution,
            "source": message.sender_id,
            "timestamp": message.timestamp,
            "quality": message.content.get("fitness", 0.0)
        })
        
        # Keep memory limited
        if len(self.memory) > 50:
            self.memory = sorted(self.memory, key=lambda x: x["quality"])[:25]
    
    async def _process_quantum_entanglement(self, message: SwarmMessage):
        """Process quantum entanglement request"""
        
        if message.sender_id not in self.quantum_state["entanglement_partners"]:
            self.quantum_state["entanglement_partners"].append(message.sender_id)
            
            # Synchronize quantum states
            sender_quantum_state = message.content["quantum_state"]
            self.quantum_state["superposition_states"] = (
                self.quantum_state["superposition_states"] + 
                sender_quantum_state["superposition_states"]
            ) / 2
    
    def adapt_role(self, swarm_needs: Dict[str, int]) -> AgentRole:
        """Dynamically adapt agent role based on swarm needs"""
        
        current_role = self.state.role
        
        # Check if swarm needs more of certain roles
        role_priorities = {
            AgentRole.EXPLORER: swarm_needs.get("exploration", 0),
            AgentRole.EXPLOITER: swarm_needs.get("exploitation", 0),
            AgentRole.COORDINATOR: swarm_needs.get("coordination", 0),
            AgentRole.SPECIALIST: swarm_needs.get("specialization", 0),
            AgentRole.QUANTUM_ORACLE: swarm_needs.get("quantum_reasoning", 0)
        }
        
        # Agents can change roles based on swarm needs
        if np.random.random() < 0.1:  # 10% chance to change role
            new_role = max(role_priorities.items(), key=lambda x: x[1])[0]
            if new_role != current_role:
                self.state.role = new_role
                logger.info(f"Agent {self.state.agent_id} changed role to {new_role.value}")
                return new_role
        
        return current_role

class SwarmTopologyManager:
    """Manages swarm communication topology"""
    
    def __init__(self, topology: SwarmTopology):
        self.topology = topology
        self.connections: Dict[str, Set[str]] = defaultdict(set)
        self.quantum_entanglements: Dict[str, Set[str]] = defaultdict(set)
        
    def build_topology(self, agent_ids: List[str]):
        """Build communication topology between agents"""
        
        self.connections.clear()
        
        if self.topology == SwarmTopology.FULLY_CONNECTED:
            self._build_fully_connected(agent_ids)
        elif self.topology == SwarmTopology.RING:
            self._build_ring(agent_ids)
        elif self.topology == SwarmTopology.SMALL_WORLD:
            self._build_small_world(agent_ids)
        elif self.topology == SwarmTopology.QUANTUM_ENTANGLED:
            self._build_quantum_entangled(agent_ids)
        else:
            self._build_fully_connected(agent_ids)  # Default
    
    def _build_fully_connected(self, agent_ids: List[str]):
        """Build fully connected topology"""
        for agent_id in agent_ids:
            self.connections[agent_id] = set(agent_ids) - {agent_id}
    
    def _build_ring(self, agent_ids: List[str]):
        """Build ring topology"""
        n = len(agent_ids)
        for i, agent_id in enumerate(agent_ids):
            left_neighbor = agent_ids[(i - 1) % n]
            right_neighbor = agent_ids[(i + 1) % n]
            self.connections[agent_id] = {left_neighbor, right_neighbor}
    
    def _build_small_world(self, agent_ids: List[str], k: int = 4, p: float = 0.3):
        """Build small world topology (Watts-Strogatz)"""
        n = len(agent_ids)
        
        # Start with ring lattice
        for i, agent_id in enumerate(agent_ids):
            for j in range(1, k // 2 + 1):
                left = agent_ids[(i - j) % n]
                right = agent_ids[(i + j) % n]
                self.connections[agent_id].add(left)
                self.connections[agent_id].add(right)
        
        # Rewire with probability p
        for agent_id in agent_ids:
            connections_copy = self.connections[agent_id].copy()
            for connected_id in connections_copy:
                if np.random.random() < p:
                    self.connections[agent_id].remove(connected_id)
                    # Add random connection
                    possible_connections = set(agent_ids) - {agent_id} - self.connections[agent_id]
                    if possible_connections:
                        new_connection = np.random.choice(list(possible_connections))
                        self.connections[agent_id].add(new_connection)
    
    def _build_quantum_entangled(self, agent_ids: List[str]):
        """Build quantum entangled topology"""
        
        # Create quantum entangled pairs
        for i in range(0, len(agent_ids) - 1, 2):
            agent1 = agent_ids[i]
            agent2 = agent_ids[i + 1] if i + 1 < len(agent_ids) else agent_ids[0]
            
            self.quantum_entanglements[agent1].add(agent2)
            self.quantum_entanglements[agent2].add(agent1)
        
        # Add some regular connections
        self._build_small_world(agent_ids, k=3, p=0.2)
    
    def get_neighbors(self, agent_id: str) -> Set[str]:
        """Get neighboring agents for communication"""
        return self.connections.get(agent_id, set())
    
    def get_quantum_entangled(self, agent_id: str) -> Set[str]:
        """Get quantum entangled partners"""
        return self.quantum_entanglements.get(agent_id, set())

class SwarmOrchestrator:
    """Orchestrates the entire swarm intelligence system"""
    
    def __init__(
        self,
        swarm_size: int = 50,
        dimension: int = 10,
        topology: SwarmTopology = SwarmTopology.SMALL_WORLD
    ):
        self.swarm_id = str(uuid.uuid4())
        self.swarm_size = swarm_size
        self.dimension = dimension
        self.topology_manager = SwarmTopologyManager(topology)
        
        self.agents: Dict[str, QuantumAgent] = {}
        self.global_best_solution: Optional[Dict[str, Any]] = None
        self.global_best_fitness = float('inf')
        
        self.objectives: List[SwarmObjective] = []
        self.generation = 0
        self.performance_history: List[SwarmPerformanceMetrics] = []
        
        self.message_broker: List[SwarmMessage] = []
        
    async def initialize_swarm(
        self,
        role_distribution: Optional[Dict[AgentRole, float]] = None
    ):
        """Initialize the swarm with diverse agents"""
        
        if role_distribution is None:
            role_distribution = {
                AgentRole.EXPLORER: 0.3,
                AgentRole.EXPLOITER: 0.3,
                AgentRole.COORDINATOR: 0.1,
                AgentRole.SPECIALIST: 0.2,
                AgentRole.QUANTUM_ORACLE: 0.1
            }
        
        # Create agents with diverse roles
        agent_roles = []
        for role, proportion in role_distribution.items():
            count = int(self.swarm_size * proportion)
            agent_roles.extend([role] * count)
        
        # Fill any remaining slots
        while len(agent_roles) < self.swarm_size:
            agent_roles.append(AgentRole.EXPLORER)
        
        # Shuffle roles
        random.shuffle(agent_roles)
        
        # Create agents
        for i in range(self.swarm_size):
            agent_id = f"agent_{i:03d}"
            specialization = self._assign_specialization(agent_roles[i])
            
            agent = QuantumAgent(
                agent_id=agent_id,
                role=agent_roles[i],
                dimension=self.dimension,
                specialization=specialization
            )
            
            self.agents[agent_id] = agent
        
        # Build communication topology
        agent_ids = list(self.agents.keys())
        self.topology_manager.build_topology(agent_ids)
        
        # Establish social connections
        for agent_id, agent in self.agents.items():
            neighbors = self.topology_manager.get_neighbors(agent_id)
            agent.social_connections = neighbors
        
        logger.info(f"Initialized swarm {self.swarm_id} with {self.swarm_size} agents")
    
    def _assign_specialization(self, role: AgentRole) -> List[str]:
        """Assign specialization based on agent role"""
        
        specializations = {
            AgentRole.EXPLORER: ["global_search", "diversity_maintenance"],
            AgentRole.EXPLOITER: ["local_search", "solution_refinement"],
            AgentRole.COORDINATOR: ["task_allocation", "conflict_resolution"],
            AgentRole.SPECIALIST: ["domain_expertise", "constraint_handling"],
            AgentRole.QUANTUM_ORACLE: ["quantum_computing", "superposition_analysis"]
        }
        
        return specializations.get(role, ["general"])
    
    def add_objective(
        self,
        name: str,
        weight: float = 1.0,
        minimize: bool = True,
        fitness_function: str = "main",
        constraints: Optional[List[str]] = None
    ):
        """Add optimization objective"""
        
        objective = SwarmObjective(
            objective_id=str(uuid.uuid4()),
            name=name,
            weight=weight,
            minimize=minimize,
            fitness_function=fitness_function,
            constraints=constraints or []
        )
        
        self.objectives.append(objective)
    
    async def evolve_generation(
        self,
        fitness_function: Callable[[np.ndarray], float],
        max_iterations: int = 100
    ):
        """Evolve swarm for one generation"""
        
        self.generation += 1
        start_time = datetime.now()
        
        # Evaluate all agent fitness
        fitness_values = []
        for agent in self.agents.values():
            fitness = await agent.evaluate_fitness(fitness_function, self.objectives)
            fitness_values.append(fitness)
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_solution = agent.state.best_solution.copy()
        
        # Calculate swarm center
        positions = np.array([agent.state.position for agent in self.agents.values()])
        swarm_center = np.mean(positions, axis=0)
        
        # Update agent positions
        global_best_position = (
            self.global_best_solution["position"] 
            if self.global_best_solution else swarm_center
        )
        
        update_tasks = []
        for agent in self.agents.values():
            task = agent.update_position(
                global_best_position, 
                swarm_center,
                quantum_influence=0.1 if agent.state.role == AgentRole.QUANTUM_ORACLE else 0.05
            )
            update_tasks.append(task)
        
        await asyncio.gather(*update_tasks)
        
        # Process inter-agent communication
        await self._process_swarm_communication()
        
        # Adapt swarm topology if needed
        if self.generation % 20 == 0:
            await self._adapt_swarm_topology()
        
        # Record performance metrics
        await self._record_performance_metrics(fitness_values, start_time)
        
        logger.info(
            f"Generation {self.generation}: Best fitness = {self.global_best_fitness:.6f}, "
            f"Avg fitness = {np.mean(fitness_values):.6f}"
        )
    
    async def _process_swarm_communication(self):
        """Process communication between agents"""
        
        communication_tasks = []
        
        for agent_id, agent in self.agents.items():
            # Send position updates to neighbors
            neighbors = self.topology_manager.get_neighbors(agent_id)
            
            if neighbors and np.random.random() < 0.3:  # 30% communication probability
                message = SwarmMessage(
                    sender_id=agent_id,
                    receiver_ids=list(neighbors),
                    message_type="position_update",
                    content={
                        "position": agent.state.position.tolist(),
                        "fitness": agent.state.best_fitness,
                        "role": agent.state.role.value
                    },
                    quantum_entangled=False,
                    timestamp=datetime.now(),
                    urgency=5
                )
                
                communication_tasks.append(agent.send_message(message))
            
            # Quantum entangled communication
            quantum_partners = self.topology_manager.get_quantum_entangled(agent_id)
            if quantum_partners and np.random.random() < 0.1:  # 10% quantum communication
                quantum_message = SwarmMessage(
                    sender_id=agent_id,
                    receiver_ids=list(quantum_partners),
                    message_type="quantum_entanglement",
                    content={
                        "quantum_state": agent.quantum_state,
                        "coherence": agent.state.quantum_coherence
                    },
                    quantum_entangled=True,
                    timestamp=datetime.now(),
                    urgency=8
                )
                
                communication_tasks.append(agent.send_message(quantum_message, quantum_entangled=True))
        
        await asyncio.gather(*communication_tasks, return_exceptions=True)
        
        # Process received messages
        processing_tasks = []
        for agent in self.agents.values():
            processing_tasks.append(agent.process_messages())
        
        await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    async def _adapt_swarm_topology(self):
        """Dynamically adapt swarm topology based on performance"""
        
        # Analyze swarm diversity
        positions = np.array([agent.state.position for agent in self.agents.values()])
        diversity = np.std(positions, axis=0).mean()
        
        # If diversity is too low, increase exploration connections
        if diversity < 1.0:
            logger.info("Low diversity detected, adapting topology for more exploration")
            # Add more random long-range connections
            for agent_id in list(self.agents.keys())[::5]:  # Every 5th agent
                other_agents = [aid for aid in self.agents.keys() if aid != agent_id]
                new_connections = np.random.choice(other_agents, 2, replace=False)
                self.topology_manager.connections[agent_id].update(new_connections)
        
        # Adapt agent roles based on performance
        swarm_needs = self._assess_swarm_needs()
        for agent in self.agents.values():
            agent.adapt_role(swarm_needs)
    
    def _assess_swarm_needs(self) -> Dict[str, int]:
        """Assess what types of agents the swarm needs more of"""
        
        # Count current role distribution
        role_counts = defaultdict(int)
        for agent in self.agents.values():
            role_counts[agent.state.role] += 1
        
        # Determine needs based on performance trends
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        needs = {"exploration": 0, "exploitation": 0, "coordination": 0, "specialization": 0, "quantum_reasoning": 0}
        
        if recent_performance:
            avg_convergence = np.mean([p.convergence_rate for p in recent_performance])
            avg_diversity = np.mean([p.diversity_index for p in recent_performance])
            
            if avg_convergence < 0.01:  # Slow convergence
                needs["exploration"] = 3
            if avg_diversity < 0.5:  # Low diversity
                needs["exploration"] = 2
                needs["quantum_reasoning"] = 1
            if avg_convergence > 0.1:  # Fast convergence, need exploitation
                needs["exploitation"] = 2
        
        return needs
    
    async def _record_performance_metrics(self, fitness_values: List[float], start_time: datetime):
        """Record performance metrics for this generation"""
        
        positions = np.array([agent.state.position for agent in self.agents.values()])
        diversity = np.std(positions, axis=0).mean()
        
        # Calculate convergence rate
        convergence_rate = 0.0
        if len(self.performance_history) > 0:
            prev_best = self.performance_history[-1].best_fitness
            convergence_rate = abs(self.global_best_fitness - prev_best) / (abs(prev_best) + 1e-10)
        
        # Calculate quantum advantage
        quantum_agents = [a for a in self.agents.values() if a.state.role == AgentRole.QUANTUM_ORACLE]
        quantum_advantage = np.mean([a.state.quantum_coherence for a in quantum_agents]) if quantum_agents else 0.0
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        metrics = SwarmPerformanceMetrics(
            swarm_id=self.swarm_id,
            generation=self.generation,
            best_fitness=self.global_best_fitness,
            average_fitness=np.mean(fitness_values),
            diversity_index=diversity,
            convergence_rate=convergence_rate,
            quantum_advantage=quantum_advantage,
            computational_efficiency=1.0 / (computation_time + 1e-10),
            timestamp=datetime.now()
        )
        
        self.performance_history.append(metrics)
    
    async def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        max_generations: int = 100,
        target_fitness: Optional[float] = None,
        patience: int = 20
    ) -> Dict[str, Any]:
        """Run complete swarm optimization"""
        
        logger.info(f"Starting swarm optimization for {max_generations} generations")
        
        best_fitness_history = []
        no_improvement_count = 0
        
        for generation in range(max_generations):
            await self.evolve_generation(fitness_function)
            
            best_fitness_history.append(self.global_best_fitness)
            
            # Check for early stopping
            if target_fitness and self.global_best_fitness <= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at generation {generation}")
                break
            
            # Check for stagnation
            if len(best_fitness_history) > patience:
                recent_improvement = best_fitness_history[-patience] - self.global_best_fitness
                if recent_improvement < 1e-6:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        logger.info(f"No improvement for {patience} generations, stopping early")
                        break
                else:
                    no_improvement_count = 0
        
        # Return optimization results
        return {
            "swarm_id": self.swarm_id,
            "best_solution": self.global_best_solution,
            "best_fitness": self.global_best_fitness,
            "generations_completed": self.generation,
            "fitness_history": best_fitness_history,
            "performance_metrics": [asdict(m) for m in self.performance_history[-10:]],
            "final_agent_states": {
                agent_id: {
                    "role": agent.state.role.value,
                    "position": agent.state.position.tolist(),
                    "fitness": agent.state.best_fitness,
                    "quantum_coherence": agent.state.quantum_coherence
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        
        role_distribution = defaultdict(int)
        avg_quantum_coherence = 0.0
        
        for agent in self.agents.values():
            role_distribution[agent.state.role.value] += 1
            avg_quantum_coherence += agent.state.quantum_coherence
        
        avg_quantum_coherence /= len(self.agents)
        
        return {
            "swarm_id": self.swarm_id,
            "generation": self.generation,
            "swarm_size": len(self.agents),
            "best_fitness": self.global_best_fitness,
            "role_distribution": dict(role_distribution),
            "average_quantum_coherence": avg_quantum_coherence,
            "topology": self.topology_manager.topology.value,
            "objectives_count": len(self.objectives)
        }

# Global swarm management
class SwarmIntelligenceManager:
    """Manages multiple swarms for different problems"""
    
    def __init__(self):
        self.active_swarms: Dict[str, SwarmOrchestrator] = {}
        self.swarm_results: Dict[str, Dict[str, Any]] = {}
    
    async def create_swarm(
        self,
        problem_id: str,
        swarm_size: int = 50,
        dimension: int = 10,
        topology: SwarmTopology = SwarmTopology.SMALL_WORLD,
        role_distribution: Optional[Dict[AgentRole, float]] = None
    ) -> str:
        """Create a new swarm for problem solving"""
        
        orchestrator = SwarmOrchestrator(swarm_size, dimension, topology)
        await orchestrator.initialize_swarm(role_distribution)
        
        self.active_swarms[problem_id] = orchestrator
        
        logger.info(f"Created swarm for problem {problem_id}")
        return orchestrator.swarm_id
    
    async def solve_problem(
        self,
        problem_id: str,
        fitness_function: Callable[[np.ndarray], float],
        objectives: List[Dict[str, Any]],
        max_generations: int = 100,
        target_fitness: Optional[float] = None
    ) -> str:
        """Solve optimization problem using swarm intelligence"""
        
        if problem_id not in self.active_swarms:
            await self.create_swarm(problem_id)
        
        swarm = self.active_swarms[problem_id]
        
        # Add objectives
        for obj in objectives:
            swarm.add_objective(**obj)
        
        # Start optimization
        result = await swarm.optimize(
            fitness_function, max_generations, target_fitness
        )
        
        self.swarm_results[problem_id] = result
        
        return result["swarm_id"]
    
    def get_swarm_status(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get status of swarm solving specific problem"""
        
        if problem_id in self.active_swarms:
            return self.active_swarms[problem_id].get_swarm_status()
        
        return None
    
    def get_problem_result(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get result of solved problem"""
        
        return self.swarm_results.get(problem_id)

# Global instance
swarm_intelligence_manager = SwarmIntelligenceManager()