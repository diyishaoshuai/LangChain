import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import * as dotenv from 'dotenv';
import fs from 'fs/promises';
dotenv.config();

// 1. 初始化模型和嵌入模型（使用 OpenAI gpt-3.5-turbo）
const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    model: "gpt-3.5-turbo",
    temperature: 0.2, // RAG任务要求高准确性，降低随机性
    configuration: {
        baseURL: "https://api.chatanywhere.tech/v1",
    },
});

// 嵌入模型：使用 OpenAI text-embedding-3-small
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-3-small",
    configuration: {
        // 走同一个兼容代理，避免直连超时/被墙
        baseURL: "https://api.chatanywhere.tech/v1",
        timeout: 60_000,
    },
});

async function createVectorStoreFromDocument(filePath) {
    console.log(`正在处理文档: ${filePath}`);

    // 2. 文档加载
    let rawDocuments;
    if (filePath.endsWith('.pdf')) {
        const loader = new PDFLoader(filePath);
        rawDocuments = await loader.load();
    } else if (filePath.endsWith('.txt')) {
        const content = await fs.readFile(filePath, "utf-8");
        rawDocuments = [
            new Document({
                pageContent: content,
                metadata: { source: filePath },
            }),
        ];
    } else {
        throw new Error(`不支持的文件格式: ${filePath}`);
    }
    console.log(`原始文档加载完成，共 ${rawDocuments.length} 页/节`);

    // 3. 文本分割（关键步骤！）
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,  // 每个文本块的最大字符数
        chunkOverlap: 200, // 块之间的重叠字符，保持上下文连贯
    });
    const splitDocuments = await textSplitter.splitDocuments(rawDocuments);
    console.log(`分割为 ${splitDocuments.length} 个文本块`);

    // 4. 向量化并存储到向量数据库
    console.log('正在生成向量并存入内存向量库...（本地计算，无需外部DB）');
    const vectors = await embeddings.embedDocuments(splitDocuments.map(d => d.pageContent));
    console.log('✅ 向量生成完成（内存存储）');
    return { vectors, documents: splitDocuments };
}

async function askQuestion(vectorStore, question) {
    console.log(`\n🤔 你的问题: ${question}`);

    // 5. 检索相关文档片段
    // 先对问题生成向量，再做余弦相似度检索
    const questionVec = await embeddings.embedQuery(question);
    const scores = vectorStore.vectors.map((vec, idx) => ({
        idx,
        score: cosineSimilarity(questionVec, vec),
    }));
    const top = scores.sort((a, b) => b.score - a.score).slice(0, 3);
    const relevantDocs = top.map(({ idx, score }) => ({
        ...vectorStore.documents[idx],
        metadata: { ...vectorStore.documents[idx].metadata, score },
    }));
    console.log(`🔍 检索到 ${relevantDocs.length} 个相关片段:`);
    relevantDocs.forEach((doc, i) => {
        console.log(`\n[片段 ${i + 1}] 来源: ${doc.metadata.source || '未知'}，页码: ${doc.metadata.page || 'N/A'}`);
        console.log(`预览: ${doc.pageContent.substring(0, 150)}...`);
    });

    // 6. 构建增强后的提示词
    const contextText = relevantDocs.map(doc => doc.pageContent).join('\n\n---\n\n');
    const augmentedPrompt = `请基于以下提供的上下文信息回答问题。如果上下文信息不足以回答问题，请直接说明。

上下文信息：
${contextText}

问题：${question}

基于上下文的答案：`;

    // 7. 调用模型生成最终答案
    console.log('\n🧠 正在生成答案...');
    const response = await llm.invoke([new HumanMessage(augmentedPrompt)]);

    console.log('\n💡 最终答案：');
    console.log(response.content);
    console.log('\n' + '='.repeat(60));
}

// 计算余弦相似度
function cosineSimilarity(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

async function main() {
    try {
        // 注意：你需要有一个真实的文档文件，例如 `sample.pdf` 或 `notes.txt`
        const filePath = './sample.txt'; // 请将此路径改为你的实际文件路径

        // 检查文件是否存在
        try {
            await fs.access(filePath);
        } catch {
            console.log(`请先在项目根目录创建文件 ${filePath}，并放入一些文本内容。`);
            console.log('示例：创建一个 sample.txt，内容可以是产品说明书、学习笔记等。');
            return;
        }

        // 创建或加载向量存储
        let vectorStore;
        const dbPath = './chroma_db';

        // 这里简化处理：每次都重新生成向量库。实际应用中应有持久化和检查逻辑。
        console.log('=== 开始构建知识库 ===');
        vectorStore = await createVectorStoreFromDocument(filePath);

        // 开始问答循环
        console.log('\n=== 知识库就绪，开始问答 ===（输入 q 退出）\n');

        // 示例问题（在实际应用中，这里可以是一个循环接收用户输入）
        const sampleQuestions = [
            "文档主要讲述了什么内容？",
            "分析感情关系",
            "续写一个章节"
        ];

        for (const question of sampleQuestions) {
            await askQuestion(vectorStore, question);
        }

        console.log('演示结束。你可以修改代码，实现交互式问答。');

    } catch (error) {
        console.error('❌ 程序出错:', error);
    }
}

main();