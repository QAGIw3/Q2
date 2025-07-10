import React, { useState, useEffect, useCallback, useMemo } from 'react';
import ReactFlow, { addEdge, Connection, Edge, Node, Position, Handle, MiniMap, Controls, Background } from 'reactflow';
import 'reactflow/dist/style.css';
import './WorkflowVisualizer.css';
import { getWorkflow, approveWorkflowTask, connectToDashboardSocket, disconnectFromDashboardSocket } from '../../services/managerAPI';
import dagre from 'dagre';
import { Box, Paper } from '@mui/material';
import { Workflow, TaskBlock } from '../../services/types';
import TaskDetailPanel from './TaskDetailPanel'; // Import the new panel

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const isHorizontal = direction === 'LR';
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 150, height: 50 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = isHorizontal ? Position.Left : Position.Top;
    node.sourcePosition = isHorizontal ? Position.Right : Position.Bottom;
    node.position = {
      x: nodeWithPosition.x - 150 / 2,
      y: nodeWithPosition.y - 50 / 2,
    };
    return node;
  });

  return { nodes, edges };
};

const workflowToElements = (workflow: any): { nodes: Node[], edges: Edge[] } => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const workflowId = workflow.workflow_id;

    const traverse = (tasks: any[], parentId?: string) => {
        tasks.forEach(task => {
            let nodeType = 'default';
            if (task.type === 'approval') nodeType = 'approval';
            if (task.type === 'conditional') nodeType = 'default';

            nodes.push({
                id: task.task_id,
                data: { 
                    label: `${task.type}: ${task.task_id.substring(0, 8)}`, 
                    status: task.status,
                    message: task.message, // For approval nodes
                    workflowId: workflowId,
                },
                position: { x: 0, y: 0 },
                type: nodeType,
                className: `status-${task.status}`
            });
            if (parentId) {
                edges.push({ id: `e-${parentId}-${task.task_id}`, source: parentId, target: task.task_id, animated: true });
            }
            if (task.dependencies && task.dependencies.length > 0) {
                task.dependencies.forEach((dep: string) => {
                    edges.push({ id: `e-${dep}-${task.task_id}`, source: dep, target: task.task_id });
                });
            }
            if (task.type === 'conditional') {
                task.branches.forEach((branch: any) => {
                    traverse(branch.tasks, task.task_id);
                });
            }
        });
    };

    traverse(workflow.tasks);
    return { nodes, edges };
};

const ApprovalNode = ({ data }: { data: any }) => {
    const handleApproval = async (approved: boolean) => {
        // The data object from ReactFlow node contains what we need
        const { workflowId, label } = data;
        const taskId = label.split(': ')[1];
        try {
            await approveWorkflowTask(workflowId, taskId, approved);
            // The UI will update automatically via the WebSocket message
        } catch (error) {
            console.error("Failed to submit approval:", error);
            alert("Failed to submit approval.");
        }
    };

    return (
        <div className="approval-node">
            <Handle type="target" position={Position.Top} />
            <div>{data.message}</div>
            {data.status === 'pending_approval' && (
                <div className="approval-buttons">
                    <button onClick={() => handleApproval(true)}>Approve</button>
                    <button onClick={() => handleApproval(false)}>Reject</button>
                </div>
            )}
            <Handle type="source" position={Position.Bottom} />
        </div>
    );
};

const nodeTypes = {
    approval: ApprovalNode,
};

interface WorkflowVisualizerProps {
    workflowId: string;
}

const WorkflowVisualizer: React.FC<WorkflowVisualizerProps> = ({ workflowId }) => {
    const [nodes, setNodes] = useState<Node[]>([]);
    const [edges, setEdges] = useState<Edge[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [selectedTask, setSelectedTask] = useState<TaskBlock | null>(null);
    const [isPanelOpen, setIsPanelOpen] = useState(false);

    const findTaskById = (tasks: TaskBlock[], taskId: string): TaskBlock | null => {
        for (const task of tasks) {
            if (task.task_id === taskId) {
                return task;
            }
            // Check inside conditional branches
            if (task.type === 'conditional' && task.branches) {
                for (const branch of task.branches) {
                    const found = findTaskById(branch.tasks, taskId);
                    if (found) return found;
                }
            }
            // Check inside loop blocks
            if (task.type === 'loop' && task.tasks) {
                const found = findTaskById(task.tasks, taskId);
                if (found) return found;
            }
        }
        return null;
    };

    const onNodeClick = (event: React.MouseEvent, node: Node) => {
        const task = findTaskById(workflow.tasks, node.id);
        if (task) {
            setSelectedTask(task);
            setIsPanelOpen(true);
        }
    };

    const handleClosePanel = () => {
        setIsPanelOpen(false);
        setSelectedTask(null);
    };

    const fetchWorkflow = useCallback(async () => {
        if (!workflowId) return;
        try {
            const workflowData = await getWorkflow(workflowId);
            const { nodes: initialNodes, edges: initialEdges } = workflowToElements(workflowData);
            const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
                initialNodes,
                initialEdges
            );
            setNodes(layoutedNodes);
            setEdges(layoutedEdges);
            setError(null);
        } catch (error: any) {
            console.error("Error fetching workflow:", error);
            setError(error.message || "Failed to load workflow.");
        }
    }, [workflowId]);

    useEffect(() => {
        fetchWorkflow();

        connectToDashboardSocket((eventData: any) => {
            if (eventData.workflow_id !== workflowId) return;

            // When a task status updates, or the workflow completes, re-fetch everything
            // This is simpler than patching the state and ensures consistency.
            if (eventData.event_type === 'TASK_STATUS_UPDATE' || eventData.event_type === 'WORKFLOW_COMPLETED') {
                fetchWorkflow();
            }
        });

        return () => {
            disconnectFromDashboardSocket();
        };
    }, [workflowId, fetchWorkflow]);

    const onConnect = (params: Connection | Edge) => setEdges((eds) => addEdge(params, eds));

    if (error) {
        return <div className="error-message">{error}</div>
    }

    return (
        <Paper elevation={3} style={{ height: '70vh', width: '100%' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onConnect={onConnect}
                onNodeClick={onNodeClick}
                fitView
                nodeTypes={nodeTypes}
            >
                <MiniMap />
                <Controls />
                <Background />
            </ReactFlow>
            <TaskDetailPanel
                task={selectedTask}
                open={isPanelOpen}
                onClose={handleClosePanel}
            />
        </Paper>
    );
};

export default WorkflowVisualizer; 