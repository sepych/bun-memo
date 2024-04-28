import { OpenAI } from "langchain/llms/openai";
import {BufferMemory} from "langchain/memory";
import {LLMChain} from "langchain/chains";
import {ChatPromptTemplate, PromptTemplate} from "langchain/prompts";
import {StructuredOutputParser} from "langchain/output_parsers";
import { RunnableSequence } from "langchain/runnables";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";



const model = new OpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0.2,
    apiKey: process.env.OPENAI_API_KEY,
});


const parser = StructuredOutputParser.fromNamesAndDescriptions({
    entity: "extracted entity if any",
    data: "extracted facts and nuances about the entity",
});

const chain = RunnableSequence.from([
    PromptTemplate.fromTemplate(
        `
{format_instructions}
Extract facts and nuances from the text:
{question}`
    ),
    model,
    parser,
]);


const buffer: {
    entity: string;
    data: string;
}[] = [];

export async function memoryAgent(input: string): Promise<string> {
    console.log('memoryAgent', input);
    const result = await chain.invoke({
        question: input,
        format_instructions: parser.getFormatInstructions(),
    });

    buffer.push({
        entity: result.entity,
        data: result.data,
    })

    console.log('memoryAgent', result);
    // console.log('memoryAgent', await memory.loadMemoryVariables({input: 'Show memory variables'}));
    return result.entity + ' ' + result.facts;
}

export async function fetchFromMemory(query: string): Promise<string> {
    const vectorStore = await HNSWLib.load("memory", new OpenAIEmbeddings({
        apiKey: process.env.OPENAI_API_KEY,
    }));
    const result = await vectorStore.similaritySearch(query, 1);
    if (result.length === 0) {
        return 'No results found';
    }

    let response = '';
    for (const item of result) {
        response += `${item.pageContent}\n`;
    }
    console.log('fetchFromMemory', response)
    return response;
}

export async function saveInnerMemory(): Promise<void> {
    if (buffer.length === 0) {
        console.log('No data to save');
        return;
    }

    console.log('saveInnerMemory');
    const mappedText = buffer.map(({entity, data}) => `${entity}: ${data}`);
    const mappedMetadata = buffer.map(({entity}) => ({entity}));


    const vectorStore = await HNSWLib.fromTexts(
        mappedText,
        mappedMetadata,
        new OpenAIEmbeddings({
            apiKey: process.env.OPENAI_API_KEY,
        })
    );

    await vectorStore.save("memory");
    console.log('saved conversation to memory');
}
