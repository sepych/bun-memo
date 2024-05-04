import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import { readdir } from "node:fs/promises";

const embeddings = new OpenAIEmbeddings();

const memoryVectorStore = new MemoryVectorStore(embeddings);

const directory= 'memory';

export async function saveToMemory(content: string, metadata: Record<string, any>) {
    const document = {
        pageContent: content,
        metadata,
    }
    await memoryVectorStore.addDocuments([document])
}

export async function fetchFromMemory(query: string, k: number, similarityThreshold: number = 0.5): Promise<string[] | null> {
    const result = await memoryVectorStore.similaritySearchWithScore(query, k);
    const filtered = result.filter(([, score]) => score > similarityThreshold);
    if (filtered.length === 0) {
        return null;
    }
    return filtered.map(([doc]) => doc.pageContent);
}

export async function saveToDisk() {
    const vectors = memoryVectorStore.memoryVectors;
    const serialized = JSON.stringify(vectors);
    const timestamp = new Date().toISOString();

    await Bun.write(`${directory}/${timestamp}.json`, serialized);
}


export async function loadFromDisk(daysInPast: number = 7) {
    const files = await readdir(directory, { recursive: true });

    const now = new Date();
    const cutoff = new Date(now);
    cutoff.setDate(now.getDate() - daysInPast);

    const vectors = [];
    for (const file of files) {
        // get date iso string from file name
        const isoString = file.split('.json')[0];
        const timestamp = new Date(isoString);
        if (timestamp < cutoff) {
            console.log('skipping', file);
            continue;
        }


        const bunFile = await Bun.file(`${directory}/${file}`);
        const content = await bunFile.text();
        const memoryVectors = JSON.parse(content) as MemoryVector[];
        const embeddingArr = memoryVectors.map(({content, embedding, metadata}) => {
            return embedding;
        });
        const documentArr = memoryVectors.map(({content, embedding, metadata}) => {
            return {
                pageContent: content,
                metadata,
            };
        });

        await memoryVectorStore.addVectors(embeddingArr, documentArr);
    }
}

interface MemoryVector {
    content: string;
    embedding: number[];
    metadata: Record<string, any>;
}

// test
// await saveToMemory('healthy food', {output: 'galbi'});
// await saveToMemory('healthy food', {output: 'schnitzel'});
// await saveToMemory('foo', {output: 'bar'});
// await saveToDisk();

// await loadFromDisk(1);
// const result = await fetchFromMemory('drink', 1, 0.8);
// console.log(result);
