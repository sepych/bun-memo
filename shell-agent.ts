import {ChatOpenAI} from "langchain/chat_models/openai";
import {DynamicStructuredTool} from "langchain/tools";
import {z} from "zod";
import inquirer from "inquirer";
import chalk from "chalk";
import {AgentExecutor, createOpenAIFunctionsAgent} from "langchain/agents";
import {ChatPromptTemplate} from "langchain/prompts";
import type {BaseMessage} from "langchain/schema";
import {AIMessage, HumanMessage} from "@langchain/core/messages";
import {spawnSync} from 'child_process';

const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
});


const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You an agent that can execute shell commands."],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
    ["placeholder", "{agent_scratchpad}"],
]);

const tools = [
    new DynamicStructuredTool({
        name: "execute-shell-command",
        description: "execute a shell command",
        schema: z.object({
            command: z.string().describe("The shell command to execute"),
            workingDirectory: z.string().optional().describe("The working directory to execute the command in"),
        }),
        func: async ({command, workingDirectory}): Promise<string> => {
            let commandToExecute = command;
            if (workingDirectory) {
                commandToExecute = `cd ${workingDirectory} && ${command}`;
            }
            const prompt = await inquirer.prompt([{
                type: 'confirm',
                name: 'confirm',
                message: chalk.yellow('Agent is about to execute a shell ') + chalk.red(commandToExecute) + chalk.yellow(' Proceed?'),
            }]);
            if (!prompt.confirm) {
                return 'An error occurred while executing the command';
            }

            try {
                const promise = new Promise<string>((resolve, reject) => {
                    const options = {
                        cwd: workingDirectory ?? process.cwd(),
                    };
                    // console.log('Executing command:', command)
                    const args = command.split(' ');
                    const mainCommand = args.shift();
                    const bufferSpawnSyncReturns = spawnSync(mainCommand!, args, options);

                    // console.log('Command executed:', bufferSpawnSyncReturns)
                    if (bufferSpawnSyncReturns.error) {
                        console.error(bufferSpawnSyncReturns.error);
                        resolve(bufferSpawnSyncReturns.error.message);
                    } else if (bufferSpawnSyncReturns.stderr.length > 0) {
                        console.error(bufferSpawnSyncReturns.stderr.toString());
                        resolve(bufferSpawnSyncReturns.stderr.toString());
                    } else if (bufferSpawnSyncReturns.status !== 0) {
                        console.error(`Command exited with status ${bufferSpawnSyncReturns.status}`);
                        resolve(`Command exited with status ${bufferSpawnSyncReturns.status}`);
                    } else {
                        // console.log('---', bufferSpawnSyncReturns.stdout.toString());
                        resolve(bufferSpawnSyncReturns.stdout.toString());
                    }
                });
                const result = await promise;
                console.log('--',result);
                return result;
            } catch (e) {
                console.error(e);
                return "An error occurred while executing the command";
            }
        }
    }),
];

const agent = await createOpenAIFunctionsAgent({
    tools: tools,
    llm: llm,
    prompt: prompt,
    streamRunnable: false,
});
const agentExecutor = new AgentExecutor({
    agent,
    tools,
});
const chatHistory: BaseMessage[] = [];

export async function shellAgent(request: string) {
    const res = await agentExecutor.invoke({
        input: request,
        chat_history: chatHistory,
    });
    // console.log(chalk.magenta(JSON.stringify(res)));
    chatHistory.push(new HumanMessage(request));
    chatHistory.push(new AIMessage(res.output));
    return res.output;
}
