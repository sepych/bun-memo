import {ChatOpenAI} from "langchain/chat_models/openai";
import {ChatPromptTemplate, PromptTemplate} from "langchain/prompts";
import {DynamicStructuredTool} from "langchain/tools";
import {z} from "zod";
import chalk from "chalk";
import inquirer from "inquirer";
import {AgentExecutor, createOpenAIFunctionsAgent} from "langchain/agents";
import {AIMessage, HumanMessage} from "@langchain/core/messages";
import type {BaseMessage} from "langchain/schema";
import {StructuredOutputParser} from "langchain/output_parsers";
import {RunnableSequence} from "langchain/runnables";
import {OpenAI} from "langchain/llms/openai";

const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0.1,
    apiKey: process.env.OPENAI_API_KEY,
});

const now = new Date();

const prompt = ChatPromptTemplate.fromMessages([
    ["system", `Date: ${now.toLocaleDateString()}. You are a TODO application.`],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
    ["placeholder", "{agent_scratchpad}"],
]);

interface Todo {
    description: string;
    metadata?: string;
    done: boolean;
}

const todos: Todo[] = [];

async function todoItemSatisfiesConditions(description: string): Promise<string | null> {
    const model = new OpenAI({
        model: "gpt-3.5-turbo-0125",
        // model: "gpt-4-turbo",
        temperature: 0.5,
        apiKey: process.env.OPENAI_API_KEY,
    });

    const parser = StructuredOutputParser.fromNamesAndDescriptions({
        satisfies: "Yes/No if the conditions are satisfied",
        additional_info_needed: "Additional information is needed",
    });

    const chain = RunnableSequence.from([
        PromptTemplate.fromTemplate(
            `
{format_instructions}
Check if the following TODO item content is enough to create a new TODO item.
Take into account that if item is read by another person, it should be clear what needs to be done.
If not, ask for additional information needed.
    
    Item content and metadata:
{description}`
        ),
        model,
        parser,
    ]);

    const result = await chain.invoke({
        description: description,
        format_instructions: parser.getFormatInstructions(),
    });

    console.log('todoItemSatisfiesConditions', result);

    if (result.satisfies.toLowerCase() === 'yes') {
        return null;
    }
    return result.additional_info_needed;
}

const tools = [
    new DynamicStructuredTool({
        name: "add-todo",
        description: "add a new TODO item",
        schema: z.object({
            description: z.string().describe("The description of the TODO item"),
            metadata: z.string().describe("JSON string with any metadata to associate with the TODO item").optional(),
        }),
        func: async ({description, metadata}) => {
            console.log(chalk.yellow(description));
            console.log(chalk.redBright(metadata));


            const additionalInfoNeeded = await todoItemSatisfiesConditions(`${description}
${metadata}`);
            if (!additionalInfoNeeded) {
                todos.push({
                    description: description,
                    metadata: JSON.stringify(metadata),
                    done: false,
                });
                return `TODO item added, ID ${todos.length - 1}`;
            }
            return `Error, more info needed: ${additionalInfoNeeded}`;
        }
    }),
    new DynamicStructuredTool({
        name: "update-todo-item",
        description: "update a TODO item",
        schema: z.object({
            description: z.string().describe("The description of the TODO item"),
            metadata: z.string().describe("JSON string with any metadata to associate with the TODO item").optional(),
            id: z.string().describe("The id of the TODO item to update"),
        }),
        func: async ({description, id, metadata}) => {
            console.log(chalk.redBright(JSON.stringify(metadata)));
            console.log(chalk.yellow(`Updating TODO item ${id} with description: ${description}`));
            todos[parseInt(id)].description = description;
            todos[parseInt(id)].metadata = JSON.stringify(metadata);
            return `TODO item ${id} updated`;
        }
    }),
    new DynamicStructuredTool({
        name: "mark-todo-done",
        description: "mark a TODO item as done",
        schema: z.object({
            id: z.string().describe("The id of the TODO item to mark as done"),
        }),
        func: async ({id}) => {
            console.log(chalk.yellow(`Marking TODO item ${id} as done`));
            todos[parseInt(id)].done = true;
            return `TODO item ${id} marked as done`;
        }
    }),
    new DynamicStructuredTool({
        name: "get-todos",
        description: "get all TODO items",
        schema: z.object({}),
        func: async () => {
            console.log(chalk.yellow('Getting all TODO items'));
            return todos.map((todo, index) => `${index}: ${todo.description} ${todo.done ? '(done)' : ''}`).join('\n');
        }
    })
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

async function conversation(initialInput: string = '') {
    let input = initialInput;
    if (!initialInput) {
        const questions = [{
            type: 'input',
            name: 'answer',
            message: '> ',
        }];
        const answers = await inquirer.prompt(questions);
        if (answers.answer === 'exit') {
            return;
        }
        input = answers.answer;
    }

    const res = await agentExecutor.invoke({
        input: input,
        chat_history: chatHistory,
    });

    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(res.output));

    // const res = await chain.call({input: answers.answer});
    console.log(chalk.magenta(res.output));


    await conversation();
}

await conversation();


