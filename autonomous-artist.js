class AdvancedNeuralNetwork {
    constructor(layerSizes) {
        this.layers = layerSizes;
        this.weights = [];
        this.biases = [];
        this.attention = new AttentionMechanism(layerSizes);
        this.emotion = new EmotionSystem();
        this.composition = new CompositionRules();
        this.environment = new EnvironmentalAwareness();
        this.memory = new LongTermMemory(100); // Stores 100 previous experiences
        
        // Initialize weights with Xavier/Glorot initialization
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const limit = Math.sqrt(6 / (layerSizes[i] + layerSizes[i + 1]));
            this.weights.push(
                Array(layerSizes[i]).fill().map(() => 
                    Array(layerSizes[i + 1]).fill().map(() => 
                        (Math.random() * 2 - 1) * limit
                    )
                )
            );
            this.biases.push(
                Array(layerSizes[i + 1]).fill().map(() => this.randomGaussian())
            );
        }

        // Initialize style preferences
        this.stylePreferences = {
            complexity: Math.random(),
            harmony: Math.random(),
            contrast: Math.random(),
            rhythm: Math.random()
        };
    }

    // Box-Muller transform for better weight initialization
    randomGaussian() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * 0.1;
    }

    // Activation functions
    relu(x) {
        return Math.max(0, x);
    }

    leakyRelu(x) {
        if (isNaN(x)) return 0;
        if (x > 100) return 100;
        if (x < -100) return -1;
        return x > 0 ? x : 0.01 * x;
    }

    tanh(x) {
        if (isNaN(x)) return 0;
        if (x > 100) return 1;
        if (x < -100) return -1;
        return Math.tanh(x);
    }

    sigmoid(x) {
        if (isNaN(x)) return 0.5;
        if (x > 100) return 1;
        if (x < -100) return 0;
        return 1 / (1 + Math.exp(-x));
    }

    // Forward propagation with different activation functions per layer
    predict(inputs) {
        console.log('Neural network input:', inputs);
        
        // Normalize inputs to be between 0 and 1
        let current = inputs.map(input => {
            if (isNaN(input)) return 0.5;
            return Math.min(1, Math.max(0, input));
        });

        for (let i = 0; i < this.weights.length; i++) {
            const layerOutput = [];
            
            for (let j = 0; j < this.weights[i][0].length; j++) {
                let sum = this.biases[i][j];
                for (let k = 0; k < this.weights[i].length; k++) {
                    // Add check for valid multiplication
                    const weight = this.weights[i][k][j];
                    const input = current[k];
                    if (isNaN(weight) || isNaN(input)) {
                        sum += 0;
                    } else {
                        sum += input * weight;
                    }
                }

                // Use different activation functions for different layers
                let value;
                switch(i) {
                    case 0:
                        value = this.leakyRelu(sum);
                        break;
                    case 1:
                        value = this.tanh(sum);
                        break;
                    default:
                        value = this.sigmoid(sum);
                }

                // Ensure no NaN values and keep within bounds
                if (isNaN(value)) {
                    value = 0.5;
                }
                value = Math.min(1, Math.max(0, value));
                
                layerOutput.push(value);
            }
            current = layerOutput;

            // Log layer outputs for debugging
            console.log(`Layer ${i} output:`, current);
        }

        // Final normalization
        current = current.map(value => {
            if (isNaN(value)) return 0.5;
            return Math.min(1, Math.max(0, value));
        });

        console.log('Neural network final output:', current);

        // Update memory with new input if it's novel enough
        this.updateMemory(inputs, current);

        return current; // Remove enhanceOutput for now as it might be causing issues
    }

    // Memory system for creative decision making
    updateMemory(inputs, output) {
        const novelty = this.calculateNovelty(inputs);
        if (novelty > 0.6) { // Only remember interesting inputs
            this.memory.remember({ input: [...inputs], novelty });
        }
    }

    calculateNovelty(inputs) {
        // Calculate how different this input is from memory
        const avgDifference = this.memory.memories.reduce((sum, mem) => {
            const diff = inputs.reduce((d, input, i) => 
                d + Math.abs(input - mem.input[i]), 0
            ) / inputs.length;
            return sum + diff;
        }, 0) / this.memory.memories.length;

        return Math.min(1, avgDifference * 2);
    }

    enhanceOutput(output) {
        // Use memory to influence decisions
        const avgNovelty = this.memory.memories.reduce((sum, mem) => sum + mem.novelty, 0) / this.memory.memories.length;
        
        // Enhance contrast in output based on novelty
        return output.map(value => {
            const enhanced = value * (1 + avgNovelty * 0.5);
            return Math.min(1, Math.max(0, enhanced));
        });
    }

    // Add some controlled randomness to weights
    mutate(rate = 0.1) {
        this.weights = this.weights.map(layer =>
            layer.map(neuron =>
                neuron.map(weight =>
                    Math.random() < rate ? weight + this.randomGaussian() : weight
                )
            )
        );
    }

    // Add feedback mechanism
    provideFeedback(score) {
        if (score > this.lastScore) {
            // Slightly mutate weights in the successful direction
            this.mutate(0.05);
        } else {
            // Revert to previous weights or mutate less
            this.mutate(0.01);
        }
        this.lastScore = score;
    }

    // Modify memory to influence future generations
    updateMemory(inputs, output, feedback) {
        const novelty = this.calculateNovelty(inputs);
        if (novelty > 0.6 || feedback > 0.7) {
            this.memory.remember({ 
                input: [...inputs], 
                novelty,
                feedback,
                weights: this.weights.map(layer => layer.map(row => [...row]))
            });
            
            // Learn from successful past experiences
            this.learnFromMemory();
        }
    }

    learnFromMemory() {
        const bestMemories = this.memory.memories
            .filter(m => m.feedback > 0.8)
            .slice(-5);

        if (bestMemories.length > 0) {
            // Adjust weights towards successful past configurations
            const targetWeights = bestMemories[Math.floor(Math.random() * bestMemories.length)].weights;
            this.weights = this.weights.map((layer, i) =>
                layer.map((row, j) =>
                    row.map((weight, k) => {
                        const target = targetWeights[i][j][k];
                        return weight + (target - weight) * 0.1;
                    })
                )
            );
        }
    }
}

class AttentionMechanism {
    constructor(layerSizes) {
        this.attentionWeights = new Array(layerSizes.length).fill(1);
        this.focusHistory = [];
    }

    focus(input, context) {
        const timeOfDay = this.getTimeOfDay();
        const seasonalFactor = this.getSeasonalFactor();
        const weatherInfluence = this.getWeatherInfluence();

        return input.map((value, i) => {
            const attention = this.calculateAttention(value, i, {
                timeOfDay,
                seasonalFactor,
                weatherInfluence,
                context
            });
            // Ensure attention stays within reasonable bounds
            const boundedAttention = Math.min(1.5, Math.max(0.5, attention));
            const focusedValue = value * boundedAttention;
            // Ensure output is normalized
            return Math.min(1, Math.max(0, focusedValue));
        });
    }

    calculateAttention(value, index, factors) {
        // Complex attention calculation based on multiple factors
        let attention = 1;
        attention *= Math.sin(factors.timeOfDay * Math.PI * 2) * 0.5 + 0.5;
        attention *= factors.seasonalFactor;
        attention *= factors.weatherInfluence;
        return attention;
    }

    // Time-based influences
    getTimeOfDay() {
        return (new Date().getHours() + new Date().getMinutes() / 60) / 24;
    }

    getSeasonalFactor() {
        const day = Math.floor((new Date() - new Date(new Date().getFullYear(), 0, 0)) / 86400000);
        return Math.sin(day / 365 * Math.PI * 2) * 0.5 + 0.5;
    }

    getWeatherInfluence() {
        // Could be connected to a weather API using DLC or similar
        return Math.random() * 0.5 + 0.5;
    }
}

class EmotionSystem {
    constructor() {
        this.emotions = {
            joy: Math.random(),
            melancholy: Math.random(),
            energy: Math.random(),
            chaos: Math.random(),
            harmony: Math.random()
        };
        this.moodHistory = [];
        this.emotionalMemory = new Map();
        this.baseFrequency = 0.005; // Base frequency for emotional oscillation
    }

    updateEmotions(input, blockHeight) {
        // Update emotions based on various factors with increased dynamism
        this.emotions.joy = this.calculateJoy(input, blockHeight);
        this.emotions.melancholy = this.calculateMelancholy(input, blockHeight);
        this.emotions.energy = this.calculateEnergy(input, blockHeight);
        this.emotions.chaos = this.calculateChaos(input, blockHeight);
        this.emotions.harmony = this.calculateHarmony(input, blockHeight);

        // Add time-based variation
        const timeInfluence = Date.now() * 0.001;
        this.emotions.joy += Math.sin(timeInfluence) * 0.2;
        this.emotions.melancholy += Math.cos(timeInfluence) * 0.2;
        this.emotions.energy += Math.sin(timeInfluence * 1.5) * 0.2;
        this.emotions.chaos += Math.cos(timeInfluence * 2) * 0.2;
        this.emotions.harmony += Math.sin(timeInfluence * 0.5) * 0.2;

        // Normalize emotions to ensure they stay within 0-1 range
        Object.keys(this.emotions).forEach(emotion => {
            this.emotions[emotion] = Math.max(0, Math.min(1, this.emotions[emotion]));
        });

        // Add random spikes to create more dramatic variations
        if (Math.random() < 0.3) { // 30% chance of emotional spike
            const randomEmotion = Object.keys(this.emotions)[Math.floor(Math.random() * Object.keys(this.emotions).length)];
            this.emotions[randomEmotion] = Math.min(1, this.emotions[randomEmotion] + Math.random() * 0.4);
        }

        this.moodHistory.push({
            emotions: {...this.emotions},
            timestamp: Date.now()
        });

        // Maintain mood history
        if (this.moodHistory.length > 100) {
            this.moodHistory.shift();
        }

        return this.emotions;
    }

    calculateJoy(input, blockHeight) {
        const timeInfluence = Math.sin(Date.now() * this.baseFrequency * 1.5);
        const blockInfluence = Math.sin(blockHeight * 0.02);
        const inputAverage = input.reduce((a, b) => a + b, 0) / input.length;
        
        return this.normalizeEmotion(
            (timeInfluence * 0.4 + blockInfluence * 0.3 + inputAverage * 0.3) + Math.random() * 0.3
        );
    }

    calculateMelancholy(input, blockHeight) {
        const timeInfluence = Math.cos(Date.now() * this.baseFrequency * 1.2);
        const blockInfluence = Math.cos(blockHeight * 0.015);
        const inputVariance = this.calculateVariance(input);
        
        return this.normalizeEmotion(
            (timeInfluence * 0.5 + blockInfluence * 0.2 + inputVariance * 0.3) + Math.random() * 0.3
        );
    }

    calculateEnergy(input, blockHeight) {
        const timeInfluence = Math.sin(Date.now() * this.baseFrequency * 2.5);
        const blockInfluence = Math.sin(blockHeight * 0.025);
        const inputMax = Math.max(...input);
        
        return this.normalizeEmotion(
            (timeInfluence * 0.3 + blockInfluence * 0.3 + inputMax * 0.4) + Math.random() * 0.3
        );
    }

    calculateChaos(input, blockHeight) {
        const timeInfluence = Math.sin(Date.now() * this.baseFrequency * 3.5);
        const blockInfluence = Math.tan(blockHeight * 0.01) % 1;
        const inputEntropy = this.calculateEntropy(input);
        
        return this.normalizeEmotion(
            (timeInfluence * 0.3 + blockInfluence * 0.3 + inputEntropy * 0.4) + Math.random() * 0.3
        );
    }

    calculateHarmony(input, blockHeight) {
        const timeInfluence = Math.cos(Date.now() * this.baseFrequency);
        const blockInfluence = Math.sin(blockHeight * 0.012);
        const inputBalance = this.calculateBalance(input);
        const joyInfluence = this.emotions.joy * 0.2;
        const calmInfluence = (1 - this.emotions.chaos) * 0.2;
        
        return this.normalizeEmotion(
            (timeInfluence * 0.2 + blockInfluence * 0.2 + inputBalance * 0.2 + 
             joyInfluence + calmInfluence) + Math.random() * 0.3
        );
    }

    // Utility functions for emotion calculations
    normalizeEmotion(value) {
        return Math.max(0, Math.min(1, (value + 1) / 2));
    }

    calculateVariance(array) {
        const mean = array.reduce((a, b) => a + b, 0) / array.length;
        const squareDiffs = array.map(value => Math.pow(value - mean, 2));
        return squareDiffs.reduce((a, b) => a + b, 0) / array.length;
    }

    calculateEntropy(array) {
        const normalized = array.map(this.normalizeEmotion);
        return normalized.reduce((entropy, value) => {
            if (value === 0) return entropy;
            return entropy - (value * Math.log2(value));
        }, 0) / array.length;
    }

    calculateBalance(array) {
        const max = Math.max(...array);
        const min = Math.min(...array);
        const range = max - min;
        const normalizedValues = array.map(v => (v - min) / (range || 1));
        const balance = 1 - this.calculateVariance(normalizedValues);
        return balance;
    }

    // Emotional memory methods
    rememberEmotionalState(input, emotions) {
        const key = JSON.stringify(input.map(v => Math.round(v * 100) / 100));
        this.emotionalMemory.set(key, {
            emotions: {...emotions},
            timestamp: Date.now()
        });
        
        // Cleanup old memories
        if (this.emotionalMemory.size > 1000) {
            const oldestKey = Array.from(this.emotionalMemory.keys())[0];
            this.emotionalMemory.delete(oldestKey);
        }
    }

    getEmotionalMemory(input) {
        const key = JSON.stringify(input.map(v => Math.round(v * 100) / 100));
        return this.emotionalMemory.get(key);
    }

    getEmotionalTrend() {
        if (this.moodHistory.length < 2) return null;
        
        const recent = this.moodHistory.slice(-10);
        const trends = {};
        
        Object.keys(this.emotions).forEach(emotion => {
            const values = recent.map(m => m.emotions[emotion]);
            const trend = values.slice(1).reduce((acc, val, i) => 
                acc + (val - values[i]), 0) / (values.length - 1);
            trends[emotion] = trend;
        });
        
        return trends;
    }
}

class CompositionRules {
    constructor() {
        this.goldenRatio = (1 + Math.sqrt(5)) / 2;
        this.rules = {
            ruleOfThirds: true,
            goldenSpiral: true,
            dynamicSymmetry: true,
            colorHarmony: true
        };
    }

    applyCompositionRules(elements, emotions) {
        return elements.map(element => {
            element = this.applyGoldenRatio(element);
            element = this.applyRuleOfThirds(element);
            element = this.applyDynamicSymmetry(element, emotions);
            return element;
        });
    }

    applyGoldenRatio(element) {
        // Adjust element positions and sizes based on golden ratio
        return {
            ...element,
            width: element.width * this.goldenRatio,
            height: element.height / this.goldenRatio
        };
    }
}

class EnvironmentalAwareness {
    constructor() {
        this.timeOfDay = 0;
        this.season = 0;
        this.weather = 'clear';
        this.lastUpdate = Date.now();
        this.updateEnvironment();
    }

    async updateEnvironment() {
        // Update time-based factors
        const now = new Date();
        this.timeOfDay = (now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds()) / 86400;
        
        // Calculate season
        const dayOfYear = Math.floor((now - new Date(now.getFullYear(), 0, 0)) / 86400000);
        this.season = (Math.sin(dayOfYear / 365 * Math.PI * 2) + 1) / 2;

        // Could integrate with weather API using DLC
        try {
            this.weather = ['clear', 'cloudy', 'rainy', 'stormy'][Math.floor(Math.random() * 4)];
        } catch (e) {
            console.log('Weather data unavailable');
        }
    }
}

class LongTermMemory {
    constructor(capacity) {
        this.capacity = capacity;
        this.memories = [];
        this.importantEvents = new Map();
        this.patterns = new Set();
    }

    remember(experience) {
        // Add new experience
        this.memories.push({
            ...experience,
            timestamp: Date.now(),
            importance: this.calculateImportance(experience)
        });

        // Maintain capacity
        if (this.memories.length > this.capacity) {
            // Remove least important memory instead of oldest
            const leastImportantIndex = this.findLeastImportantMemoryIndex();
            this.memories.splice(leastImportantIndex, 1);
        }

        // Detect and store patterns
        this.detectPatterns();
    }

    calculateImportance(experience) {
        // Importance calculation based on multiple factors
        let importance = 0;
        importance += experience.novelty || 0;
        importance += experience.emotionalImpact || 0;
        importance += experience.userFeedback || 0;
        return importance;
    }

    detectPatterns() {
        // Analyze recent memories for patterns
        const recentMemories = this.memories.slice(-10);
    }
}

class AutonomousArtist {
    constructor() {
        console.log('Initializing AutonomousArtist...');
        console.log('rough.js available:', typeof rough !== 'undefined');
        
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute("viewBox", "0 0 600 600");
        this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        this.svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        this.svg.style.opacity = '0';
        
        // Initialize neural network and supporting systems
        this.model = new AdvancedNeuralNetwork([4, 32, 64, 32, 16, 8]);
        this.blockHeight = 0;
        this.gradientIndex = 0;
        
        if (typeof rough === 'undefined') {
            console.error('rough.js is not loaded!');
            return;
        }
        
        try {
            console.log('Initializing rough.js...');
            this.rc = rough.svg(this.svg);
            console.log('rough.js initialized successfully');
        } catch (error) {
            console.error('Failed to initialize rough.js:', error);
        }

        this.expressionPalette = {
            rage: ['#FF0000', '#FF2D00', '#FF4500', '#FF1E1E', '#FF3300', '#FF6B6B', '#FF3366', '#FF4444', '#FF5733', '#FF6347'],
            melancholy: ['#4B0082', '#483D8B', '#8A2BE2', '#9932CC', '#7B68EE', '#6A5ACD', '#5D478B', '#8B668B', '#7851A9', '#9370DB'],
            ecstasy: ['#FFA500', '#FF8C00', '#FFD700', '#FF7F00', '#FFA000', '#FFB347', '#FFCF48', '#FFD300', '#FFAA33', '#FFB733'],
            fear: ['#800080', '#8B008B', '#9400D3', '#8B0000', '#4B0082', '#871F78', '#702963', '#660066', '#86608E', '#301934'],
            hope: ['#32CD32', '#228B22', '#006400', '#FF4500', '#FFA500', '#90EE90', '#98FB98', '#3CB371', '#2E8B57', '#00FA9A'],
            serenity: ['#4169E1', '#000080', '#191970', '#483D8B', '#4B0082', '#1E90FF', '#87CEEB', '#B0C4DE', '#6495ED', '#7B68EE'],
            passion: ['#FF0000', '#FF4500', '#FF6347', '#FF1493', '#FF4081', '#FF69B4', '#FF355E', '#FF033E', '#FF91A4', '#FF577F'],
            wisdom: ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#D2691E', '#B8860B', '#DAA520', '#BC8F8F', '#F4A460', '#D2B48C'],
            mystery: ['#483D8B', '#4B0082', '#663399', '#8B008B', '#9400D3', '#9932CC', '#8A2BE2', '#9370DB', '#7B68EE', '#6A5ACD'],
            nature: ['#228B22', '#006400', '#8B4513', '#A0522D', '#CD853F', '#556B2F', '#6B8E23', '#808000', '#3CB371', '#2E8B57']
        };

        this.emotionalSymbols = {
            rage: ['✦', '△', '⚡', '×', '✸'],
            melancholy: ['☾', '☁', '∇', '○', '⚊'],
            ecstasy: ['☀', '⚘', '♢', '⚝', '✯'],
            fear: ['⚈', '▽', '⚊', '⚆', '◊'],
            hope: ['✺', '❋', '☘', '✧', '❊'],
            serenity: ['≋', '∿', '~', '≈', '⌇'],
            passion: ['❤', '♥', '⚘', '✿', '❀'],
            wisdom: ['⚯', '☯', '◉', '⚫', '⚪'],
            mystery: ['✧', '⚝', '✦', '⚹', '✴'],
            nature: ['❀', '✿', '☘', '❁', '✾']
        };
        
        this.speed = 20;
    }

    setSpeed(newSpeed) {
        this.speed = newSpeed;
    }

    adjustDuration(originalDuration) {
        return originalDuration / this.speed;
    }

    calculateRoughness(prediction, emotions, index) {
        const baseRoughness = prediction[index % prediction.length] * 2;
        const emotionalFactor = (emotions.chaos + emotions.energy) / 3;
        const randomVariation = Math.random() * 0.3;
        return (baseRoughness + emotionalFactor * 2 + randomVariation) * 1.5;
    }

    calculateStrokeWidth(emotions, elementType) {
        const baseWidth = 1;
        const emotionalVariation = emotions.chaos * 1.5;
        const randomVariation = Math.random() * 0.5;
        
        switch(elementType) {
            case 'figure':
                return baseWidth * (2 + emotionalVariation + randomVariation);
            case 'detail':
                return baseWidth * (1 + emotionalVariation * 0.5);
            case 'background':
                return baseWidth * (0.5 + emotionalVariation * 0.8);
            case 'gesture':
                return baseWidth * (1.5 + emotionalVariation);
            default:
                return baseWidth;
        }
    }

    renderLandscape(prediction, emotions) {
        while (this.svg.firstChild) {
            this.svg.removeChild(this.svg.firstChild);
        }

        const drawingGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.svg.appendChild(drawingGroup);
        this.currentGroup = drawingGroup;

        const lineSegments = this.generateLineSequence(prediction, emotions);
        this.svg.style.opacity = '1';

        const drawSegment = (index) => {
            if (index >= lineSegments.length) return;

            const segment = lineSegments[index];
            
            // Validate path data before passing to rough.js
            if (!segment || !segment.path || typeof segment.path !== 'string' || !segment.path.trim()) {
                console.warn('Invalid path data detected, skipping segment:', segment);
                requestAnimationFrame(() => drawSegment(index + 1));
                return;
            }

            try {
                const roughPath = this.rc.path(segment.path, {
                    stroke: segment.color || 'black',
                    strokeWidth: segment.width || 1,
                    roughness: this.calculateRoughness(prediction, emotions, index),
                    bowing: emotions.chaos * 1.5,
                    disableMultiStroke: true,
                    fill: segment.fill ? segment.color : undefined,
                    fillStyle: segment.fill ? 'solid' : undefined
                });

                if (!roughPath) {
                    console.warn('Failed to generate rough path, skipping segment');
                    requestAnimationFrame(() => drawSegment(index + 1));
                    return;
                }

                const paths = Array.from(roughPath.querySelectorAll('path'));
                const splitPaths = [];
                
                paths.forEach(path => {
                    const d = path.getAttribute('d');
                    if (!d) return;

                    const subPaths = d.split(/(?=[Mm])/).filter(Boolean);
                    
                    if (subPaths.length > 1) {
                        subPaths.forEach(subPath => {
                            if (!subPath.trim()) return;
                            
                            const newPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                            newPath.setAttribute('d', subPath);
                            newPath.setAttribute('stroke', path.getAttribute('stroke') || segment.color || 'black');
                            newPath.setAttribute('stroke-width', path.getAttribute('stroke-width') || segment.width || '1');
                            newPath.setAttribute('fill', 'none');
                            
                            const length = newPath.getTotalLength();
                            newPath.style.opacity = '1';
                            newPath.style.strokeDasharray = length;
                            newPath.style.strokeDashoffset = length;
                            newPath.style.transition = 'none';
                            
                            splitPaths.push(newPath);
                        });
                        path.remove();
                    } else {
                        const length = path.getTotalLength();
                        path.style.opacity = '1';
                        path.style.strokeDasharray = length;
                        path.style.strokeDashoffset = length;
                        path.style.transition = 'none';
                        splitPaths.push(path);
                    }
                });

                const pathGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                splitPaths.forEach(path => pathGroup.appendChild(path));
                this.currentGroup.appendChild(pathGroup);

                pathGroup.getBoundingClientRect();

                let currentPathIndex = 0;
                const animatePath = () => {
                    if (currentPathIndex >= splitPaths.length) {
                        setTimeout(() => {
                            drawSegment(index + 1);
                        }, segment.pause || 1000);
                        return;
                    }

                    const path = splitPaths[currentPathIndex];
                    path.style.transition = `stroke-dashoffset ${segment.duration}ms ease-in-out`;
                    path.style.strokeDashoffset = '0';

                    currentPathIndex++;
                    setTimeout(animatePath, segment.duration);
                };

                requestAnimationFrame(animatePath);
            } catch (error) {
                console.error('Error rendering path:', error);
                console.warn('Problematic path data:', segment.path);
                requestAnimationFrame(() => drawSegment(index + 1));
            }
        };

        drawSegment(0);
        return this.svg;
    }

    generateLineSequence(prediction, emotions) {
        const sequences = [];
        const centerX = 300;
        const centerY = 300;
        
        const emotionalState = this.determineEmotionalState(emotions);
        const palette = this.expressionPalette[emotionalState];
        
        // 1. Generate wild background first
        sequences.push(...this.generateBackground(prediction, emotions));
        
        // 2. Create groups for the figure elements
        const torsoGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.currentGroup.appendChild(torsoGroup);
        this.fillGroup = torsoGroup;
        sequences.push(...this.generateTorso(centerX, centerY, emotions, palette, prediction));
        
        const headGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.currentGroup.appendChild(headGroup);
        this.fillGroup = headGroup;
        sequences.push(...this.generateHead(centerX, centerY, emotions, palette, prediction));
        
        const armsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.currentGroup.appendChild(armsGroup);
        this.fillGroup = armsGroup;
        sequences.push(...this.generateArms(centerX, centerY, emotions, palette, prediction));
        
        return sequences;
    }

    generateFigure(prediction, emotions) {
        const sequences = [];
        const centerX = 300;
        const centerY = 300;
        
        const emotionalState = this.determineEmotionalState(emotions);
        const palette = this.expressionPalette[emotionalState];
        
        // Generate head first
        sequences.push(...this.generateHead(centerX, centerY, emotions, palette, prediction));
        
        // Generate torso
        sequences.push(...this.generateTorso(centerX, centerY, emotions, palette, prediction));
        
        // Generate arms
        sequences.push(...this.generateArms(centerX, centerY, emotions, palette, prediction));
        
        return sequences;
    }

    generateHead(x, y, emotions, palette, prediction) {
        const sequences = [];
        const size = 150;
        const distortion = emotions.chaos * 10;

        // Variation factors based on prediction
        const chinLength = size * 0.5 * (0.8 + prediction[0] * 0.4);
        const foreheadCurve = size * 0.4 * (0.8 + prediction[1] * 0.4);
        const cheekWidth = size * 0.4 * (0.8 + prediction[2] * 0.4);

        // Create head shape
        const headPath = `
            M ${x-cheekWidth},${y-size*0.4} 
            C ${x-size*0.3},${y-foreheadCurve} ${x-size*0.2},${y-size*0.6} ${x},${y-size*0.6}
            C ${x+size*0.2},${y-size*0.6} ${x+size*0.3},${y-foreheadCurve} ${x+cheekWidth},${y-size*0.4}
            C ${x+size*0.5},${y-size*0.3} ${x+size*0.5},${y-size*0.2} ${x+cheekWidth},${y}
            C ${x+size*0.3},${y+size*0.2} ${x+size*0.2},${y+chinLength} ${x},${y+chinLength}
            C ${x-size*0.2},${y+chinLength} ${x-size*0.3},${y+size*0.2} ${x-cheekWidth},${y}
            C ${x-size*0.5},${y-size*0.2} ${x-size*0.5},${y-size*0.3} ${x-cheekWidth},${y-size*0.4}
            Z`;

        // 1. Add head fill
        sequences.push({
            path: headPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: palette[0],
            width: 0,
            fill: true
        });

        // 2. Add head outlines
        sequences.push({
            path: headPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: '#FFFFFF',
            width: this.calculateStrokeWidth(emotions, 'figure') * 1.4,
            fill: false
        });

        sequences.push({
            path: headPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'figure') * 0.6,
            fill: false
        });

        // 3. Generate varied nose based on emotional state and prediction
        const noseVariation = Math.floor(prediction[3] * 9);
        sequences.push(...this.generateVariedNose(x, y, size * 0.4, emotions, palette, noseVariation));

        // 4. Left eye
        const leftEyePath = `
            M ${x-size*0.3},${y-size*0.1}
            C ${x-size*0.25},${y-size*0.15} ${x-size*0.2},${y-size*0.12} ${x-size*0.15},${y-size*0.1}
            C ${x-size*0.2},${y-size*0.08} ${x-size*0.25},${y-size*0.05} ${x-size*0.3},${y-size*0.1}
            Z`;

        // Add left eye fill and outline
        sequences.push({
            path: leftEyePath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: palette[2],
            width: 0,
            fill: true
        });

        sequences.push({
            path: leftEyePath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'detail')
        });

        // 5. Right eye
        const rightEyePath = `
            M ${x+size*0.15},${y-size*0.1}
            C ${x+size*0.2},${y-size*0.15} ${x+size*0.25},${y-size*0.12} ${x+size*0.3},${y-size*0.1}
            C ${x+size*0.25},${y-size*0.08} ${x+size*0.2},${y-size*0.05} ${x+size*0.15},${y-size*0.1}
            Z`;

        // Add right eye fill and outline
        sequences.push({
            path: rightEyePath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: palette[2],
            width: 0,
            fill: true
        });

        sequences.push({
            path: rightEyePath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'detail')
        });

        // 6. Mouth
        sequences.push(...this.generateExpressiveMouth(x, y, size, emotions, palette, prediction));

        // 7. Add details and texture
        for (let i = 0; i < 5; i++) {
            const angle = (i / 5) * Math.PI * 2;
            const detailX = x + Math.cos(angle) * size * 0.3;
            const detailY = y + Math.sin(angle) * size * 0.3;
            
            sequences.push({
                path: this.generateExpressiveLine(
                    detailX, 
                    detailY,
                    detailX + Math.random() * size * 0.1,
                    detailY + Math.random() * size * 0.1,
                    emotions,
                    angle
                ),
                duration: this.adjustDuration(500),
                pause: this.adjustDuration(100),
                color: palette[Math.floor(Math.random() * palette.length)],
                width: this.calculateStrokeWidth(emotions, 'detail') * 0.5
            });
        }

        // 8. Add hair with style variation
        const hairStyleVariation = Math.floor(prediction[4] * 5);
        sequences.push(...this.generateVerticalHair(x, y - size * 0.35, size, emotions, palette, hairStyleVariation));

        // 9. Add beard with 20-25% chance
        if (Math.random() < 0.25) {
            console.log('Adding beard...');
            const beardSequences = this.generateBeard(x, y + size * 0.2, size, emotions, palette, hairStyleVariation);
            console.log('Beard sequences:', beardSequences);
            sequences.push(...beardSequences);
        }

        return sequences;
    }

    generateTorso(x, y, emotions, palette, prediction) {
        const sequences = [];
        const headSize = 150;
        const torsoWidth = headSize * 1.2;
        const torsoHeight = headSize * 1.8;
        const shoulderWidth = torsoWidth * 1.4;
        
        // Calculate torso start point
        const torsoStartY = y + headSize * 0.4;
        const shoulderSlantLeft = headSize * (0.1 + Math.random() * 0.3);
        const shoulderSlantRight = headSize * (0.1 + Math.random() * 0.3);
        
        // Create torso path
        const torsoPath = `
            M ${x - shoulderWidth/2},${torsoStartY - shoulderSlantLeft}
            C ${x - shoulderWidth/2},${torsoStartY + torsoHeight * 0.1} 
              ${x - torsoWidth/2},${torsoStartY + torsoHeight * 0.2} 
              ${x - torsoWidth/2},${torsoStartY + torsoHeight * 0.3}
            L ${x - torsoWidth/2},${torsoStartY + torsoHeight * 0.8}
            C ${x - torsoWidth/2},${torsoStartY + torsoHeight} 
              ${x},${torsoStartY + torsoHeight} 
              ${x},${torsoStartY + torsoHeight}
            C ${x},${torsoStartY + torsoHeight} 
              ${x + torsoWidth/2},${torsoStartY + torsoHeight} 
              ${x + torsoWidth/2},${torsoStartY + torsoHeight * 0.8}
            L ${x + torsoWidth/2},${torsoStartY + torsoHeight * 0.3}
            C ${x + torsoWidth/2},${torsoStartY + torsoHeight * 0.2} 
              ${x + shoulderWidth/2},${torsoStartY + torsoHeight * 0.1} 
              ${x + shoulderWidth/2},${torsoStartY - shoulderSlantRight}
            Z`;

        // Add torso fill
        sequences.push({
            path: torsoPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: palette[0],
            width: 0,
            fill: true
        });

        // Add torso outlines
        sequences.push({
            path: torsoPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: '#FFFFFF',
            width: this.calculateStrokeWidth(emotions, 'figure') * 1.4,
            fill: false
        });

        sequences.push({
            path: torsoPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'figure') * 0.6,
            fill: false
        });

        return sequences;
    }

    generateArms(x, y, emotions, palette, prediction) {
        const sequences = [];
        const headSize = 150;
        const shoulderY = y + headSize * 0.4;
        const shoulderWidth = headSize * 1.2 * 1.4;
        const armLength = headSize * 1.4;

        // Determine arm position based on prediction and emotions
        const positions = [
            'both_up',
            'both_down',
            'left_up_right_down',
            'left_down_right_up',
            'both_sides',
            'reaching',
            'folded',
            'waving',
            'pointing',
            'dancing',
            'thinking',
            'victory',
            'heart_shape',      // New: Arms forming a heart shape
            'meditation',       // New: Meditation pose
            'dramatic_pose',    // New: Dramatic diagonal pose
            'conducting',       // New: Orchestra conductor pose
            'stretching',      // New: Full stretch pose
            'power_pose',      // New: Superhero power pose
            'graceful_arc',    // New: Ballet-like graceful arc
            'shrugging'        // New: Shrugging shoulders pose
        ];
        const positionIndex = Math.floor(prediction[0] * positions.length);
        const position = positions[positionIndex];

        // Calculate arm angles based on position
        let leftArmAngle, rightArmAngle;
        let leftElbowRatio = 0.5, rightElbowRatio = 0.5;
        
        switch(position) {
            case 'both_up':
                leftArmAngle = Math.PI * 1.7;  // Pointing up and slightly outward
                rightArmAngle = Math.PI * 1.3;  // Pointing up and slightly outward
                break;
            case 'both_down':
                leftArmAngle = Math.PI * 0.3;  // Pointing down and slightly outward
                rightArmAngle = Math.PI * 0.7;  // Pointing down and slightly outward
                break;
            case 'left_up_right_down':
                leftArmAngle = Math.PI * 1.7;   // Left arm up
                rightArmAngle = Math.PI * 0.7;  // Right arm down
                break;
            case 'left_down_right_up':
                leftArmAngle = Math.PI * 0.3;   // Left arm down
                rightArmAngle = Math.PI * 1.3;  // Right arm up
                break;
            case 'both_sides':
                leftArmAngle = Math.PI;         // Left arm straight out
                rightArmAngle = 0;              // Right arm straight out
                break;
            case 'reaching':
                leftArmAngle = Math.PI * 1.5;   // Left arm reaching up
                rightArmAngle = Math.PI * 1.5;  // Right arm reaching up
                leftElbowRatio = 0.7;           // Longer upper arm segment
                rightElbowRatio = 0.7;          // Longer upper arm segment
                break;
            case 'folded':
                leftArmAngle = Math.PI * 0.9;   // Left arm folded
                rightArmAngle = Math.PI * 0.1;  // Right arm folded
                leftElbowRatio = 0.3;           // Shorter upper arm segment
                rightElbowRatio = 0.3;          // Shorter upper arm segment
                break;
            case 'waving':
                leftArmAngle = Math.PI * 1.6;   // Left arm waving high
                rightArmAngle = Math.PI * 0.4;  // Right arm relaxed down
                leftElbowRatio = 0.6;           // Bent elbow for waving
                rightElbowRatio = 0.5;          // Normal right arm
                break;
            case 'pointing':
                leftArmAngle = Math.PI * 0.4;   // Left arm down
                rightArmAngle = Math.PI * 1.2;  // Right arm pointing up and forward
                leftElbowRatio = 0.5;           // Normal left arm
                rightElbowRatio = 0.8;          // Extended pointing arm
                break;
            case 'dancing':
                leftArmAngle = Math.PI * 1.8;   // Left arm up and out
                rightArmAngle = Math.PI * 0.2;  // Right arm down and out
                leftElbowRatio = 0.4;           // Bent dancing pose
                rightElbowRatio = 0.4;          // Bent dancing pose
                break;
            case 'thinking':
                leftArmAngle = Math.PI * 0.3;   // Left arm down
                rightArmAngle = Math.PI * 0.8;  // Right arm bent towards face
                leftElbowRatio = 0.5;           // Normal left arm
                rightElbowRatio = 0.3;          // Sharply bent thinking pose
                break;
            case 'victory':
                leftArmAngle = Math.PI * 1.6;   // Left arm up in V
                rightArmAngle = Math.PI * 1.4;  // Right arm up in V
                leftElbowRatio = 0.8;           // Straight victory pose
                rightElbowRatio = 0.8;          // Straight victory pose
                break;
            case 'heart_shape':
                leftArmAngle = Math.PI * 1.8;    // Left arm up and curved
                rightArmAngle = Math.PI * 1.2;   // Right arm up and curved
                leftElbowRatio = 0.5;            // Bent for heart shape
                rightElbowRatio = 0.5;           // Bent for heart shape
                // Adjust hands to meet at the center
                leftArmAngle += Math.PI/6;       // Curve inward
                rightArmAngle -= Math.PI/6;      // Curve inward
                break;
            case 'meditation':
                leftArmAngle = Math.PI * 0.8;    // Left arm in meditation pose
                rightArmAngle = Math.PI * 0.2;   // Right arm in meditation pose
                leftElbowRatio = 0.4;            // Sharp bend at elbow
                rightElbowRatio = 0.4;           // Sharp bend at elbow
                break;
            case 'dramatic_pose':
                leftArmAngle = Math.PI * 1.9;    // Left arm up dramatically
                rightArmAngle = Math.PI * 0.3;   // Right arm down dramatically
                leftElbowRatio = 0.7;            // Slight bend
                rightElbowRatio = 0.6;           // More bent
                break;
            case 'conducting':
                leftArmAngle = Math.PI * 1.4;    // Left arm raised for conducting
                rightArmAngle = Math.PI * 1.6;   // Right arm raised for conducting
                leftElbowRatio = 0.6;            // Elegant bend
                rightElbowRatio = 0.6;           // Elegant bend
                break;
            case 'stretching':
                leftArmAngle = Math.PI * 1.5;    // Left arm straight up
                rightArmAngle = Math.PI * 1.5;   // Right arm straight up
                leftElbowRatio = 0.9;            // Almost straight
                rightElbowRatio = 0.9;           // Almost straight
                break;
            case 'power_pose':
                leftArmAngle = Math.PI * 1.3;    // Left arm in power pose
                rightArmAngle = Math.PI * 1.7;   // Right arm in power pose
                leftElbowRatio = 0.7;            // Strong pose
                rightElbowRatio = 0.7;           // Strong pose
                break;
            case 'graceful_arc':
                leftArmAngle = Math.PI * 1.7;    // Left arm in graceful arc
                rightArmAngle = Math.PI * 1.3;   // Right arm in graceful arc
                leftElbowRatio = 0.8;            // Gentle curve
                rightElbowRatio = 0.8;           // Gentle curve
                break;
            case 'shrugging':
                leftArmAngle = Math.PI * 1.2;    // Left arm raised in shrug
                rightArmAngle = Math.PI * 1.8;   // Right arm raised in shrug
                leftElbowRatio = 0.3;            // Sharp bend for shrug
                rightElbowRatio = 0.3;           // Sharp bend for shrug
                break;
            default:
                // Random natural position as fallback
                leftArmAngle = Math.PI * (0.3 + Math.random() * 0.4);
                rightArmAngle = Math.PI * (1.3 + Math.random() * 0.4);
        }

        // Add emotional influence to the angles
        const emotionalVariation = (emotions.energy - 0.5) * Math.PI * 0.1;
        leftArmAngle += emotionalVariation;
        rightArmAngle -= emotionalVariation;
        
        // Left arm
        const leftShoulderX = x - shoulderWidth/2;
        const leftArmWidth = headSize * 0.25;
        const leftElbowX = leftShoulderX + Math.cos(leftArmAngle) * armLength * leftElbowRatio;
        const leftElbowY = shoulderY + Math.sin(leftArmAngle) * armLength * leftElbowRatio;
        const leftHandX = leftElbowX + Math.cos(leftArmAngle + Math.PI/6) * armLength * (1 - leftElbowRatio);
        const leftHandY = leftElbowY + Math.sin(leftArmAngle + Math.PI/6) * armLength * (1 - leftElbowRatio);

        // Right arm
        const rightShoulderX = x + shoulderWidth/2;
        const rightArmWidth = headSize * 0.25;
        const rightElbowX = rightShoulderX + Math.cos(rightArmAngle) * armLength * rightElbowRatio;
        const rightElbowY = shoulderY + Math.sin(rightArmAngle) * armLength * rightElbowRatio;
        const rightHandX = rightElbowX + Math.cos(rightArmAngle - Math.PI/6) * armLength * (1 - rightElbowRatio);
        const rightHandY = rightElbowY + Math.sin(rightArmAngle - Math.PI/6) * armLength * (1 - rightElbowRatio);

        // Generate rectangular paths for arms
        const leftArmPath = `
            M ${leftShoulderX} ${shoulderY}
            L ${leftElbowX + leftArmWidth/2} ${leftElbowY}
            L ${leftHandX + leftArmWidth/2} ${leftHandY}
            L ${leftHandX - leftArmWidth/2} ${leftHandY}
            L ${leftElbowX - leftArmWidth/2} ${leftElbowY}
            Z
        `;

        const rightArmPath = `
            M ${rightShoulderX} ${shoulderY}
            L ${rightElbowX + rightArmWidth/2} ${rightElbowY}
            L ${rightHandX + rightArmWidth/2} ${rightHandY}
            L ${rightHandX - rightArmWidth/2} ${rightHandY}
            L ${rightElbowX - rightArmWidth/2} ${rightElbowY}
            Z
        `;

        // 1. Add left arm fill
        sequences.push({
            path: leftArmPath,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(100),
            color: palette[0],
            width: 0,
            fill: true
        });

        // 2. Add left arm outlines
        sequences.push({
            path: leftArmPath,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(100),
            color: '#FFFFFF',
            width: this.calculateStrokeWidth(emotions, 'figure') * 1.4,
            fill: false
        });

        sequences.push({
            path: leftArmPath,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'figure') * 0.6,
            fill: false
        });

        // 3. Add right arm fill
        sequences.push({
            path: rightArmPath,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(100),
            color: palette[0],
            width: 0,
            fill: true
        });

        // 4. Add right arm outlines
        sequences.push({
            path: rightArmPath,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(100),
            color: '#FFFFFF',
            width: this.calculateStrokeWidth(emotions, 'figure') * 1.4,
            fill: false
        });

        sequences.push({
            path: rightArmPath,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'figure') * 0.6,
            fill: false
        });

        // Generate hand paths
        const generateHandPath = (handX, handY, angle, size, isRightHand = false) => {
            // Increase size for right hand by 25%
            const sizeMultiplier = isRightHand ? 1.25 : 1;
            const palmSize = size * 0.2 * sizeMultiplier;
            const fingerLength = size * 0.15 * sizeMultiplier;
            const fingerWidth = size * 0.08 * sizeMultiplier;
            
            // Adjust palm start position based on hand side
            const palmStartX = handX + (isRightHand ? -palmSize * 0.5 : -palmSize * 0.5);
            
            // Create a unified hand path that includes palm and fingers in one shape
            let handPath = `M ${palmStartX} ${handY}`;
            
            // Add palm curve with proper orientation
            handPath += ` C ${palmStartX} ${handY - palmSize * 0.4} 
                           ${handX + (isRightHand ? -palmSize * 0.5 : palmSize * 0.5)} ${handY - palmSize * 0.4} 
                           ${handX + (isRightHand ? -palmSize * 0.5 : palmSize * 0.5)} ${handY}`;
            
            // Add fingers in a fan shape with proper orientation
            for (let i = 0; i < 5; i++) {
                const fingerAngle = angle + (isRightHand ? -(i - 2) : (i - 2)) * Math.PI/10;
                const fingerStartX = handX + Math.cos(angle) * palmSize * (isRightHand ? -0.3 : 0.3);
                const fingerStartY = handY + Math.sin(angle) * palmSize * 0.3;
                const fingerEndX = fingerStartX + Math.cos(fingerAngle) * fingerLength;
                const fingerEndY = fingerStartY + Math.sin(fingerAngle) * fingerLength;
                
                // Add finger as a curved extension with proper orientation
                handPath += ` C ${fingerStartX + Math.cos(fingerAngle) * fingerLength * 0.5} 
                              ${fingerStartY + Math.sin(fingerAngle) * fingerLength * 0.5}
                              ${fingerEndX - Math.cos(fingerAngle) * fingerWidth} 
                              ${fingerEndY - Math.sin(fingerAngle) * fingerWidth}
                              ${fingerEndX} ${fingerEndY}`;
            }
            
            // Close the path back to the palm
            handPath += ' Z';
            
            return handPath;
        };

        // Generate hand paths with proper orientation
        const leftHandPath = generateHandPath(leftHandX, leftHandY, leftArmAngle + Math.PI/6, headSize, false);
        const rightHandPath = generateHandPath(rightHandX, rightHandY, rightArmAngle - Math.PI/6, headSize, true);

        // 5. Add left hand fill
        sequences.push({
            path: leftHandPath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: this.adjustColor(palette[0], 20), // Slightly lighter than body color
            width: 0,
            fill: true
        });

        // 6. Add left hand outlines
        sequences.push({
            path: leftHandPath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: '#FFFFFF',
            width: this.calculateStrokeWidth(emotions, 'figure') * 1.2,
            fill: false
        });

        sequences.push({
            path: leftHandPath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'figure') * 0.5,
            fill: false
        });

        // 7. Add right hand fill
        sequences.push({
            path: rightHandPath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: this.adjustColor(palette[0], 20), // Slightly lighter than body color
            width: 0,
            fill: true
        });

        // 8. Add right hand outlines
        sequences.push({
            path: rightHandPath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: '#FFFFFF',
            width: this.calculateStrokeWidth(emotions, 'figure') * 1.2,
            fill: false
        });

        sequences.push({
            path: rightHandPath,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'figure') * 0.5,
            fill: false
        });

        return sequences;
    }

    generateHand(x, y, angle, size, emotions, palette) {
        const sequences = [];
        const fingerCount = 5;
        const fingerLength = size * 0.8;
        const fingerWidth = size * 0.15;
        const palmSize = size * 0.6;

        // Generate palm
        const palmPath = this.generateDistortedOval(x, y, palmSize, palmSize * 0.8, emotions);
        sequences.push({
            path: palmPath,
            duration: this.adjustDuration(800),
            pause: this.adjustDuration(100),
            color: palette[0],
            width: this.calculateStrokeWidth(emotions, 'detail'),
            fill: true
        });

        // Generate fingers
        for (let i = 0; i < fingerCount; i++) {
            const fingerAngle = angle + (i - 2) * Math.PI/8;
            const fingerStartX = x + Math.cos(angle) * palmSize * 0.5;
            const fingerStartY = y + Math.sin(angle) * palmSize * 0.5;
            const fingerEndX = fingerStartX + Math.cos(fingerAngle) * fingerLength;
            const fingerEndY = fingerStartY + Math.sin(fingerAngle) * fingerLength;

            const fingerPath = `
                M ${fingerStartX - fingerWidth/2} ${fingerStartY}
                L ${fingerEndX - fingerWidth/2} ${fingerEndY}
                L ${fingerEndX + fingerWidth/2} ${fingerEndY}
                L ${fingerStartX + fingerWidth/2} ${fingerStartY}
                Z
            `;

            sequences.push({
                path: fingerPath,
                duration: this.adjustDuration(600),
                pause: this.adjustDuration(50),
                color: palette[0],
                width: this.calculateStrokeWidth(emotions, 'detail'),
                fill: true
            });
        }

        return sequences;
    }

    generateExpressiveLine(x1, y1, x2, y2, emotions, angle = 0) {
        // Validate inputs
        if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
            console.warn('Invalid coordinates in generateExpressiveLine');
            return `M ${0} ${0} L ${1} ${1}`;  // Return a simple fallback path
        }

        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        const distance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
        const controlDistance = distance * 0.5 * (1 + (emotions?.chaos || 0));
        
        const controlX = midX + Math.cos(angle || 0) * controlDistance;
        const controlY = midY + Math.sin(angle || 0) * controlDistance;
        
        return `M ${x1} ${y1} Q ${controlX} ${controlY} ${x2} ${y2}`;
    }

    generateExpressiveMark(x, y, length, angle, emotions) {
        // Ensure valid numeric inputs
        x = Number(x) || 0;
        y = Number(y) || 0;
        length = Math.max(1, Number(length) || 10);
        angle = Number(angle) || 0;
        emotions = emotions || { chaos: 0, energy: 0 };

        // Calculate end points with bounded variation
        const endX = x + Math.cos(angle) * length * (1 + Math.min(0.5, (Math.random() - 0.5) * emotions.chaos));
        const endY = y + Math.sin(angle) * length * (1 + Math.min(0.5, (Math.random() - 0.5) * emotions.chaos));

        // Add multiple control points for more expressive strokes
        const numControls = Math.max(2, Math.min(5, Math.floor(2 + emotions.energy * 3)));
        let path = `M ${Math.round(x * 100) / 100} ${Math.round(y * 100) / 100}`;

        // Create a series of control points for a more expressive line
        for (let i = 1; i <= numControls; i++) {
            const t = i / (numControls + 1);
            const baseX = x + (endX - x) * t;
            const baseY = y + (endY - y) * t;
            
            const perpX = -Math.sin(angle) * length * 0.5;
            const perpY = Math.cos(angle) * length * 0.5;
            const emotionalOffset = Math.min(length * 0.3, (Math.random() - 0.5) * emotions.chaos * length * 0.3);
            
            const controlX = Math.round((baseX + perpX * Math.sin(t * Math.PI) * emotions.energy + 
                           (Math.random() - 0.5) * Math.min(50, emotions.chaos * 50)) * 100) / 100;
            const controlY = Math.round((baseY + perpY * Math.sin(t * Math.PI) * emotions.energy + 
                           (Math.random() - 0.5) * Math.min(50, emotions.chaos * 50)) * 100) / 100;

            if (i === numControls) {
                path += ` Q ${controlX} ${controlY} ${Math.round(endX * 100) / 100} ${Math.round(endY * 100) / 100}`;
            } else {
                const nextT = (i + 1) / (numControls + 1);
                const nextX = Math.round((x + (endX - x) * nextT) * 100) / 100;
                const nextY = Math.round((y + (endY - y) * nextT) * 100) / 100;
                path += ` Q ${controlX} ${controlY} ${nextX} ${nextY}`;
            }
        }

        return path;
    }

    generateDistortedShape(points, emotions) {
        if (!Array.isArray(points) || points.length < 2) {
            return `M 0 0 L 1 1`;
        }

        emotions = emotions || { chaos: 0, energy: 0 };
        const useLinear = emotions.chaos > 0.7;
        const distortionAmount = Math.min(20, 20 * emotions.chaos);
        
        let path = `M ${Math.round(points[0][0] * 100) / 100} ${Math.round(points[0][1] * 100) / 100}`;
        
        for (let i = 1; i < points.length; i++) {
            const point = points[i];
            if (!Array.isArray(point) || point.length < 2) continue;

            const x = Number(point[0]) || 0;
            const y = Number(point[1]) || 0;

            if (useLinear) {
                const dx = (Math.random() - 0.5) * distortionAmount;
                const dy = (Math.random() - 0.5) * distortionAmount;
                path += ` L ${Math.round((x + dx) * 100) / 100} ${Math.round((y + dy) * 100) / 100}`;
            } else {
                const prev = points[i-1];
                const prevX = Number(prev[0]) || 0;
                const prevY = Number(prev[1]) || 0;
                
                const controlX = Math.round((prevX + (x - prevX) * 0.5 + 
                               (Math.random() - 0.5) * distortionAmount) * 100) / 100;
                const controlY = Math.round((prevY + (y - prevY) * 0.5 +
                               (Math.random() - 0.5) * distortionAmount) * 100) / 100;
                
                const energyOffset = Math.min(10, emotions.energy * 10);
                const dx = (Math.random() - 0.5) * energyOffset;
                const dy = (Math.random() - 0.5) * energyOffset;
                
                path += ` Q ${controlX} ${controlY} ${Math.round((x + dx) * 100) / 100} ${Math.round((y + dy) * 100) / 100}`;
            }
        }
        
        if (points.length > 2 && 
            points[0][0] === points[points.length-1][0] && 
            points[0][1] === points[points.length-1][1]) {
            path += ' Z';
        }
        
        return path;
    }

    generateDistortedOval(centerX, centerY, radiusX, radiusY, emotions) {
        // Ensure valid numeric inputs
        centerX = Number(centerX) || 0;
        centerY = Number(centerY) || 0;
        radiusX = Math.max(1, Number(radiusX) || 10);
        radiusY = Math.max(1, Number(radiusY) || 10);
        emotions = emotions || { chaos: 0, energy: 0 };

        const points = [];
        const steps = 24;
        const distortionX = Math.min(radiusX * 0.3, radiusX * emotions.chaos * 0.3);
        const distortionY = Math.min(radiusY * 0.3, radiusY * emotions.chaos * 0.3);
        const energyInfluence = 1 + Math.min(0.4, emotions.energy * 0.4);

        for (let i = 0; i <= steps; i++) {
            const angle = (i / steps) * Math.PI * 2;
            const baseX = Math.cos(angle) * radiusX * energyInfluence;
            const baseY = Math.sin(angle) * radiusY * energyInfluence;
            
            const emotionalDistortionX = (Math.random() - 0.5) * distortionX;
            const emotionalDistortionY = (Math.random() - 0.5) * distortionY;
            
            const waveDistortionX = Math.sin(angle * 3) * distortionX * emotions.chaos;
            const waveDistortionY = Math.cos(angle * 3) * distortionY * emotions.chaos;
            
            const x = Math.round((centerX + baseX + emotionalDistortionX + waveDistortionX) * 100) / 100;
            const y = Math.round((centerY + baseY + emotionalDistortionY + waveDistortionY) * 100) / 100;
            
            points.push([x, y]);
        }

        points.push(points[0]);
        return this.generateDistortedShape(points, emotions);
    }

    getEmotionalColor(emotions, opacity = 1) {
        const state = this.determineEmotionalState(emotions);
        const palette = this.expressionPalette[state];
        const color = palette[Math.floor(Math.random() * palette.length)];
        
        // Convert hex to rgba
        const r = parseInt(color.slice(1,3), 16);
        const g = parseInt(color.slice(3,5), 16);
        const b = parseInt(color.slice(5,7), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }

    determineEmotionalState(emotions) {
        // Calculate emotional intensities with more balanced weights
        const intensities = {
            rage: emotions.chaos * 0.7 + emotions.energy * 0.5 + Math.random() * 0.3,
            melancholy: emotions.melancholy * 0.8 + (1 - emotions.energy) * 0.4 + Math.random() * 0.3,
            ecstasy: emotions.joy * 0.8 + emotions.energy * 0.4 + Math.random() * 0.3,
            fear: emotions.chaos * 0.6 + emotions.melancholy * 0.7 + Math.random() * 0.3,
            hope: emotions.joy * 0.7 + (1 - emotions.chaos) * 0.5 + Math.random() * 0.3,
            serenity: emotions.harmony * 0.6 + (1 - emotions.chaos) * 0.4 + Math.random() * 0.3,
            passion: emotions.energy * 0.9 + emotions.joy * 0.3 + Math.random() * 0.3,
            wisdom: emotions.harmony * 0.5 + emotions.melancholy * 0.4 + (1 - emotions.chaos) * 0.3 + Math.random() * 0.3,
            mystery: emotions.chaos * 0.4 + emotions.melancholy * 0.5 + (1 - emotions.joy) * 0.4 + Math.random() * 0.3,
            nature: emotions.harmony * 0.5 + emotions.joy * 0.4 + (1 - emotions.chaos) * 0.4 + Math.random() * 0.3
        };

        // Add time-based variation to prevent repetitive patterns
        const timeInfluence = Math.sin(Date.now() * 0.001) * 0.2;
        Object.keys(intensities).forEach(emotion => {
            intensities[emotion] += timeInfluence;
        });

        // Find the strongest emotion with added randomization
        let maxIntensity = -1;
        let dominantEmotion = 'hope';  // default state
        const randomBoost = Math.random() * 0.3; // Random boost for variety

        for (const [emotion, intensity] of Object.entries(intensities)) {
            const finalIntensity = intensity + (Math.random() < 0.3 ? randomBoost : 0); // 30% chance of random boost
            if (finalIntensity > maxIntensity) {
                maxIntensity = finalIntensity;
                dominantEmotion = emotion;
            }
        }

        return dominantEmotion;
    }

    async generatePainting() {
        console.log('Generating landscape...');
        await this.getBlockHeight();
        
        await this.model.environment.updateEnvironment();
        
        const seed = this.blockHeight % 1000 / 1000;
        const time = this.model.environment.timeOfDay;
        const season = this.model.environment.season;
        const weather = this.model.environment.weather;
        
        const input = [seed, time, season, Math.random()];
        console.log('Input values:', input);
        
        const focusedInput = this.model.attention.focus(input, {
            blockHeight: this.blockHeight,
            weather: weather
        });
        
        const emotions = this.model.emotion.updateEmotions(focusedInput, this.blockHeight);
        console.log('Emotions:', emotions);
        
        const prediction = this.model.predict(focusedInput);
        console.log('Prediction:', prediction);
        
        const svg = this.renderLandscape(prediction, emotions);
        console.log('SVG generated:', svg.outerHTML);

        const feedback = await this.collectFeedback();
        this.model.provideFeedback(feedback);
        this.model.updateMemory(input, prediction, feedback);

        return svg;
    }

    async getBlockHeight() {
        try {
            const response = await fetch('https://ordinals.com/r/blockheight');
            const height = await response.text();
            this.blockHeight = parseInt(height);
            return this.blockHeight;
        } catch (error) {
            console.error('Error fetching block height:', error);
            return Math.floor(Math.random() * 1000000);
        }
    }

    async collectFeedback() {
        return Math.random();
    }

    start() {
        this.generatePainting().then(svg => {
            const container = document.getElementById('artwork-container');
            if (container) {
                container.innerHTML = '';
                svg.style.width = '100%';
                svg.style.height = '100%';
                container.appendChild(svg);
            } else {
                document.body.appendChild(svg);
            }
        });
    }

    generateVariedNose(x, y, size, emotions, palette, variation) {
        const sequences = [];
        // Calculate direction angle based on emotions and variation
        const baseAngle = (variation % 4) * Math.PI / 2; // 0, 90, 180, or 270 degrees
        const emotionalTilt = (emotions.chaos - 0.5) * Math.PI / 4; // Up to 45 degrees tilt
        const direction = baseAngle + emotionalTilt;

        // Log the nose type being generated
        const noseTypes = [
            'Simple line nose',
            'Curved nose',
            'Hook nose',
            'Aquiline nose'
        ];
        console.log(`Generating nose type: ${noseTypes[variation] || 'Default nose'}`);

        switch(variation) {
            case 0: // Simple line nose with direction
                sequences.push({
                    path: this.generateExpressiveLine(
                        x, y - size/2,
                        x + Math.cos(direction) * size, y + Math.sin(direction) * size,
                        emotions,
                        direction
                    ),
                    duration: this.adjustDuration(800),
                    pause: this.adjustDuration(150),
                    color: '#000000',
                    width: this.calculateStrokeWidth(emotions, 'detail')
                });
                break;
            case 1: // Curved nose with direction
                sequences.push({
                    path: `M ${x} ${y - size/2} 
                           Q ${x + Math.cos(direction) * size * 0.8} ${y} 
                           ${x + Math.cos(direction) * size/2} ${y + Math.sin(direction) * size/2}`,
                    duration: this.adjustDuration(800),
                    pause: this.adjustDuration(150),
                    color: '#000000',
                    width: this.calculateStrokeWidth(emotions, 'detail')
                });
                break;
            case 2: // Hook nose with direction
                sequences.push(...this.generateDirectionalHookNose(x, y, size, direction, emotions, palette));
                break;
            case 3: // Aquiline nose with direction
                sequences.push(...this.generateDirectionalAquilineNose(x, y, size, direction, emotions, palette));
                break;
            default:
                // Default to curved nose as it's one of the most reliable
                sequences.push({
                    path: `M ${x} ${y - size/2} 
                           Q ${x + Math.cos(direction) * size * 0.8} ${y} 
                           ${x + Math.cos(direction) * size/2} ${y + Math.sin(direction) * size/2}`,
                    duration: this.adjustDuration(800),
                    pause: this.adjustDuration(150),
                    color: '#000000',
                    width: this.calculateStrokeWidth(emotions, 'detail')
                });
        }
        return sequences;
    }


    generateDirectionalHookNose(x, y, size, direction, emotions, palette) {
        const sequences = [];
        const hookCurve = size * 0.3 * (1 + emotions.energy * 0.3);
        
        // Rotate the hook based on direction
        const hookAngle = direction + Math.PI/4;
        const hookX = x + Math.cos(hookAngle) * hookCurve;
        const hookY = y + Math.sin(hookAngle) * hookCurve;
        
        sequences.push({
            path: `M ${x} ${y - size/2} 
                   Q ${x + Math.cos(direction) * size * 0.4} ${y} 
                   ${hookX} ${hookY}
                   Q ${x + Math.cos(direction) * size * 0.2} ${y + size/3} 
                   ${x - Math.cos(direction) * size * 0.1} ${y + size/3}`,
            duration: this.adjustDuration(1000),
            pause: this.adjustDuration(200),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'detail')
        });

        return sequences;
    }

    generateDirectionalAquilineNose(x, y, size, direction, emotions, palette) {
        const sequences = [];
        const bridgeWidth = size * 0.3;
        
        // Calculate control points based on direction
        const cp1x = x + Math.cos(direction) * bridgeWidth;
        const cp1y = y + Math.sin(direction) * bridgeWidth;
        const cp2x = x + Math.cos(direction) * bridgeWidth * 1.5;
        const cp2y = y + Math.sin(direction) * bridgeWidth * 1.5;
        
        sequences.push({
            path: `M ${x} ${y - size/2}
                   Q ${cp1x} ${cp1y} ${cp2x} ${cp2y}
                   Q ${x + Math.cos(direction) * size * 0.4} ${y + size/3} 
                   ${x} ${y + size/2}`,
            duration: this.adjustDuration(1200),
            pause: this.adjustDuration(200),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'detail')
        });

        return sequences;
    }

    generateVerticalHair(x, y, size, emotions, palette, styleVariation = 0) {
        const sequences = [];
        
        switch(styleVariation) {
            case 0: // Thin style (Charlie Brown)
                const thinCount = 8 + Math.floor(emotions.energy * 4);
                for (let i = 0; i < thinCount; i++) {
                    const t = i / (thinCount - 1);
                    const archOffset = Math.sin(t * Math.PI) * size * 0.1;
                    const startX = x - size * 0.4 + size * 0.8 * t;
                    const startY = y + archOffset; // Start at head level with arch
                    const length = size * 0.2 * (0.8 + Math.random() * 0.4);
                    const endX = startX + (Math.random() - 0.5) * size * 0.1;
                    const endY = startY - length; // Go up from starting point
                    
                    sequences.push({
                        path: `M ${startX} ${startY} L ${startX} ${endY}`,
                        duration: this.adjustDuration(400),
                        pause: this.adjustDuration(30),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.5
                    });
                }
                break;

            case 1: // Curly top
                const curlyCount = 10 + Math.floor(emotions.energy * 6);
                for (let i = 0; i < curlyCount; i++) {
                    const t = i / (curlyCount - 1);
                    const startX = x - size * 0.4 + size * 0.8 * t;
                    const startY = y; // Start from top of head
                    const length = size * 0.3 * (0.8 + Math.random() * 0.4);
                    const curlRadius = size * 0.05;
                    
                    let curlPath = `M ${startX} ${startY}`;
                    for (let j = 0; j <= 8; j++) {
                        const ct = j / 8;
                        const cx = startX + Math.sin(ct * Math.PI * 4) * curlRadius;
                        const cy = startY - length * ct;
                        curlPath += ` L ${cx} ${cy}`;
                    }
                    
                    sequences.push({
                        path: curlPath,
                        duration: this.adjustDuration(600),
                        pause: this.adjustDuration(40),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.6
                    });
                }
                break;

            case 2: // Thick strands
                const thickCount = 6 + Math.floor(emotions.energy * 4);
                for (let i = 0; i < thickCount; i++) {
                    const t = i / (thickCount - 1);
                    const startX = x - size * 0.4 + size * 0.8 * t;
                    const startY = y; // Start from top of head
                    const length = size * 0.3 * (0.8 + Math.random() * 0.4);
                    const width = size * 0.1;
                    
                    sequences.push({
                        path: `M ${startX - width/2} ${startY} 
                               Q ${startX - width/2} ${startY - length/2} ${startX} ${startY - length}
                               Q ${startX + width/2} ${startY - length/2} ${startX + width/2} ${startY}`,
                        duration: this.adjustDuration(700),
                        pause: this.adjustDuration(50),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 1.2
                    });
                }
                break;

            case 3: // Long on sides
                const centerCount = 6 + Math.floor(emotions.energy * 4);
                const sideCount = 4 + Math.floor(emotions.energy * 3);
                
                // Center strands
                for (let i = 0; i < centerCount; i++) {
                    const t = i / (centerCount - 1);
                    const startX = x - size * 0.3 + size * 0.6 * t;
                    const startY = y; // Start from top of head
                    const length = size * 0.25 * (0.8 + Math.random() * 0.4);
                    
                    sequences.push({
                        path: this.generateExpressiveLine(
                            startX, startY,
                            startX, startY - length,
                            emotions
                        ),
                        duration: this.adjustDuration(500),
                        pause: this.adjustDuration(30),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.7
                    });
                }
                
                // Side strands (left)
                for (let i = 0; i < sideCount; i++) {
                    const t = i / sideCount;
                    const startX = x - size * 0.4;
                    const startY = y + size * 0.2 * t; // Start slightly lower for sides
                    const length = size * 0.4 * (0.8 + Math.random() * 0.4);
                    const angle = Math.PI * 0.7;
                    
                    sequences.push({
                        path: this.generateExpressiveLine(
                            startX, startY,
                            startX - Math.cos(angle) * length,
                            startY - Math.sin(angle) * length,
                            emotions
                        ),
                        duration: this.adjustDuration(600),
                        pause: this.adjustDuration(40),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.8
                    });
                }
                
                // Side strands (right)
                for (let i = 0; i < sideCount; i++) {
                    const t = i / sideCount;
                    const startX = x + size * 0.4;
                    const startY = y + size * 0.2 * t; // Start slightly lower for sides
                    const length = size * 0.4 * (0.8 + Math.random() * 0.4);
                    const angle = Math.PI * 0.3;
                    
                    sequences.push({
                        path: this.generateExpressiveLine(
                            startX, startY,
                            startX + Math.cos(angle) * length,
                            startY - Math.sin(angle) * length,
                            emotions
                        ),
                        duration: this.adjustDuration(600),
                        pause: this.adjustDuration(40),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.8
                    });
                }
                break;

            case 4: // Wavy textured
                const wavyCount = 12 + Math.floor(emotions.energy * 6);
                for (let i = 0; i < wavyCount; i++) {
                    const t = i / (wavyCount - 1);
                    const startX = x - size * 0.4 + size * 0.8 * t;
                    const startY = y; // Start from top of head
                    const length = size * 0.3 * (0.8 + Math.random() * 0.4);
                    const waveAmplitude = size * 0.03;
                    
                    let wavePath = `M ${startX} ${startY}`;
                    for (let j = 0; j <= 8; j++) {
                        const wt = j / 8;
                        const wx = startX + Math.sin(wt * Math.PI * 4) * waveAmplitude * (1 - wt);
                        const wy = startY - length * wt;
                        wavePath += ` L ${wx} ${wy}`;
                    }
                    
                    sequences.push({
                        path: wavePath,
                        duration: this.adjustDuration(500),
                        pause: this.adjustDuration(35),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * (0.6 + Math.random() * 0.4)
                    });
                }
                break;
        }
        
        return sequences;
    }

    generateBeard(x, y, size, emotions, palette, styleVariation = 0) {
        console.log('generateBeard called with:', { x, y, size, emotions, palette, styleVariation });
        const sequences = [];
        // Make beard width 30% of hair width and center it
        const beardWidth = size * 0.8 * 0.3; // 30% of hair width
        // Position slightly lower on the chin
        const beardY = y + size * 0.25; // Moved slightly lower
        
        switch(styleVariation) {
            case 0: // Thin style (Charlie Brown stubble)
                const stubbleCount = 8 + Math.floor(emotions.energy * 4);
                for (let i = 0; i < stubbleCount; i++) {
                    const t = i / (stubbleCount - 1);
                    const startX = x - beardWidth/2 + beardWidth * t;
                    const length = size * 0.1 * (0.8 + Math.random() * 0.4);
                    const angle = Math.PI/2 + (Math.random() - 0.5) * emotions.chaos;
                    const endX = startX + Math.cos(angle) * length;
                    const endY = beardY + Math.sin(angle) * length;
                    
                    sequences.push({
                        path: `M ${startX} ${beardY} L ${endX} ${endY}`,
                        duration: this.adjustDuration(400),
                        pause: this.adjustDuration(30),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.5
                    });
                }
                break;

            case 1: // Curly beard
                const curlCount = 6 + Math.floor(emotions.energy * 3);
                for (let i = 0; i < curlCount; i++) {
                    const t = i / (curlCount - 1);
                    const startX = x - beardWidth/2 + beardWidth * t;
                    const length = size * 0.25 * (0.8 + Math.random() * 0.4);
                    const curlRadius = size * 0.04;
                    
                    let curlPath = `M ${startX} ${beardY}`;
                    for (let j = 0; j <= 8; j++) {
                        const ct = j / 8;
                        const cx = startX + Math.sin(ct * Math.PI * 4) * curlRadius;
                        const cy = beardY + length * ct;
                        curlPath += ` L ${cx} ${cy}`;
                    }
                    
                    sequences.push({
                        path: curlPath,
                        duration: this.adjustDuration(600),
                        pause: this.adjustDuration(40),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.6
                    });
                }
                break;

            case 2: // Thick beard strands
                const strandCount = 4 + Math.floor(emotions.energy * 2);
                for (let i = 0; i < strandCount; i++) {
                    const t = i / (strandCount - 1);
                    const startX = x - beardWidth/2 + beardWidth * t;
                    const length = size * 0.3 * (0.8 + Math.random() * 0.4);
                    const width = size * 0.06;
                    
                    sequences.push({
                        path: `M ${startX - width/2} ${beardY} 
                               Q ${startX - width/2} ${beardY + length/2} ${startX} ${beardY + length}
                               Q ${startX + width/2} ${beardY + length/2} ${startX + width/2} ${beardY}`,
                        duration: this.adjustDuration(700),
                        pause: this.adjustDuration(50),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 1.2
                    });
                }
                break;

            case 3: // Long flowing beard
                const centerStrandCount = 3 + Math.floor(emotions.energy * 2);
                const sideStrandCount = 2;
                
                // Center strands
                for (let i = 0; i < centerStrandCount; i++) {
                    const t = i / (centerStrandCount - 1);
                    const startX = x - beardWidth/3 + beardWidth/1.5 * t;
                    const length = size * 0.35 * (0.8 + Math.random() * 0.4);
                    const angle = Math.PI/2 + (Math.random() - 0.5) * emotions.chaos * 0.2;
                    
                    sequences.push({
                        path: this.generateExpressiveLine(
                            startX, beardY,
                            startX + Math.cos(angle) * length,
                            beardY + Math.sin(angle) * length,
                            emotions
                        ),
                        duration: this.adjustDuration(600),
                        pause: this.adjustDuration(40),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * 0.8
                    });
                }
                
                // Side strands
                for (let side = 0; side < 2; side++) {
                    const baseX = side === 0 ? x - beardWidth/2 : x + beardWidth/2;
                    for (let i = 0; i < sideStrandCount; i++) {
                        const t = i / sideStrandCount;
                        const length = size * 0.4 * (0.8 + Math.random() * 0.4);
                        const angle = Math.PI/2 + (side === 0 ? -0.2 : 0.2) + (Math.random() - 0.5) * emotions.chaos * 0.15;
                        
                        sequences.push({
                            path: this.generateExpressiveLine(
                                baseX, beardY + t * size * 0.1,
                                baseX + Math.cos(angle) * length,
                                beardY + t * size * 0.1 + Math.sin(angle) * length,
                                emotions
                            ),
                            duration: this.adjustDuration(600),
                            pause: this.adjustDuration(40),
                            color: '#000000',
                            width: this.calculateStrokeWidth(emotions, 'detail') * 0.8
                        });
                    }
                }
                break;

            case 4: // Wavy beard
                const wavyCount = 7 + Math.floor(emotions.energy * 3);
                for (let i = 0; i < wavyCount; i++) {
                    const t = i / (wavyCount - 1);
                    const startX = x - beardWidth/2 + beardWidth * t;
                    const length = size * 0.3 * (0.8 + Math.random() * 0.4);
                    const waveAmplitude = size * 0.02;
                    
                    let wavePath = `M ${startX} ${beardY}`;
                    for (let j = 0; j <= 8; j++) {
                        const wt = j / 8;
                        const wx = startX + Math.sin(wt * Math.PI * 4) * waveAmplitude * (1 - wt);
                        const wy = beardY + length * wt;
                        wavePath += ` L ${wx} ${wy}`;
                    }
                    
                    sequences.push({
                        path: wavePath,
                        duration: this.adjustDuration(500),
                        pause: this.adjustDuration(35),
                        color: '#000000',
                        width: this.calculateStrokeWidth(emotions, 'detail') * (0.6 + Math.random() * 0.4)
                    });
                }
                break;
        }
        
        return sequences;
    }

    // Add helper method for color adjustment
    adjustColor(color, amount) {
        const hex = color.replace('#', '');
        const r = Math.max(0, Math.min(255, parseInt(hex.substring(0, 2), 16) + amount));
        const g = Math.max(0, Math.min(255, parseInt(hex.substring(2, 4), 16) + amount));
        const b = Math.max(0, Math.min(255, parseInt(hex.substring(4, 6), 16) + amount));
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }

    generateBackground(prediction, emotions) {
        const sequences = [];
        const width = 600;
        const height = 600;
        
        // Wild neo-expressionist background strokes
        const numStrokes = 8 + Math.floor(emotions.chaos * 8);
        const emotionalState = this.determineEmotionalState(emotions);
        const palette = this.expressionPalette[emotionalState];
        
        // Add large sweeping background strokes
        for (let i = 0; i < numStrokes; i++) {
            const startX = Math.random() * width * 1.2 - width * 0.1;
            const startY = Math.random() * height * 1.2 - height * 0.1;
            const endX = Math.random() * width * 1.2 - width * 0.1;
            const endY = Math.random() * height * 1.2 - height * 0.1;
            const controlX = (startX + endX) / 2 + (Math.random() - 0.5) * width * 0.8;
            const controlY = (startY + endY) / 2 + (Math.random() - 0.5) * height * 0.8;
            
            // Create wild curved strokes
            const path = `M ${startX} ${startY} Q ${controlX} ${controlY} ${endX} ${endY}`;
            
            sequences.push({
                path: path,
                duration: this.adjustDuration(2000),
                pause: this.adjustDuration(200),
                color: this.getEmotionalColor(emotions, 0.4 + Math.random() * 0.3),
                width: 100 + Math.random() * 150,
                fill: false
            });
        }
        
        // Add explosive circular elements
        for (let i = 0; i < 5; i++) {
            const centerX = Math.random() * width;
            const centerY = Math.random() * height;
            const radius = 100 + Math.random() * 200;
            
            // Create explosive circular paths with distortion
            let explosivePath = `M ${centerX} ${centerY}`;
            const points = 12;
            for (let j = 0; j <= points; j++) {
                const angle = (j / points) * Math.PI * 2;
                const distortion = 0.7 + Math.random() * 0.6;
                const px = centerX + Math.cos(angle) * radius * distortion;
                const py = centerY + Math.sin(angle) * radius * distortion;
                explosivePath += ` L ${px} ${py}`;
            }
            explosivePath += ' Z';
            
            sequences.push({
                path: explosivePath,
                duration: this.adjustDuration(1500),
                pause: this.adjustDuration(150),
                color: this.getEmotionalColor(emotions, 0.3 + Math.random() * 0.2),
                width: 80 + Math.random() * 100,
                fill: Math.random() > 0.5
            });
        }
        
        // Add chaotic scribbles
        for (let i = 0; i < 6; i++) {
            let scribblePath = `M ${Math.random() * width} ${Math.random() * height}`;
            const points = 10;
            for (let j = 0; j < points; j++) {
                const x = Math.random() * width;
                const y = Math.random() * height;
                scribblePath += ` L ${x} ${y}`;
            }
            
            sequences.push({
                path: scribblePath,
                duration: this.adjustDuration(1000),
                pause: this.adjustDuration(100),
                color: this.getEmotionalColor(emotions, 0.5 + Math.random() * 0.3),
                width: 120 + Math.random() * 100,
                fill: false
            });
        }
        
        // Add abstract symbols
        const symbols = this.emotionalSymbols[emotionalState];
        for (let i = 0; i < 4; i++) {
            const symbol = symbols[Math.floor(Math.random() * symbols.length)];
            const symbolSize = 200 + Math.random() * 200;
            const x = Math.random() * (width - symbolSize);
            const y = Math.random() * (height - symbolSize);
            
            sequences.push({
                path: `M ${x} ${y} L ${x + symbolSize} ${y + symbolSize}`,
                duration: this.adjustDuration(800),
                pause: this.adjustDuration(100),
                color: this.getEmotionalColor(emotions, 0.6),
                width: 150 + Math.random() * 100,
                fill: false,
                text: symbol
            });
        }
        
        return sequences;
    }

    generateExpressiveMouth(x, y, size, emotions, palette, prediction) {
        const sequences = [];
        const emotionalState = this.determineEmotionalState(emotions);
        const mouthWidth = size * 0.4;
        const mouthHeight = size * 0.2;
        const centerY = y + size * 0.2;
        
        let mouthPath = '';
        
        switch(emotionalState) {
            case 'rage': // Screaming mouth
                mouthPath = `
                    M ${x - mouthWidth/2},${centerY}
                    C ${x - mouthWidth/3},${centerY + mouthHeight * 1.2} 
                      ${x + mouthWidth/3},${centerY + mouthHeight * 1.2} 
                      ${x + mouthWidth/2},${centerY}
                    C ${x + mouthWidth/3},${centerY + mouthHeight * 0.5} 
                      ${x - mouthWidth/3},${centerY + mouthHeight * 0.5} 
                      ${x - mouthWidth/2},${centerY}
                    Z`;
                break;
                
            case 'melancholy': // Sad downturned mouth
                mouthPath = `
                    M ${x - mouthWidth/2},${centerY}
                    C ${x - mouthWidth/3},${centerY} 
                      ${x},${centerY - mouthHeight/2} 
                      ${x},${centerY}
                    C ${x},${centerY + mouthHeight/2} 
                      ${x + mouthWidth/3},${centerY + mouthHeight} 
                      ${x + mouthWidth/2},${centerY + mouthHeight/2}`;
                break;
                
            case 'ecstasy': // Big smile
                mouthPath = `
                    M ${x - mouthWidth/2},${centerY}
                    C ${x - mouthWidth/3},${centerY + mouthHeight} 
                      ${x},${centerY + mouthHeight * 1.5} 
                      ${x},${centerY + mouthHeight}
                    C ${x},${centerY + mouthHeight * 1.5} 
                      ${x + mouthWidth/3},${centerY + mouthHeight} 
                      ${x + mouthWidth/2},${centerY}
                    C ${x + mouthWidth/3},${centerY - mouthHeight * 0.3} 
                      ${x - mouthWidth/3},${centerY - mouthHeight * 0.3} 
                      ${x - mouthWidth/2},${centerY}
                    Z`;
                break;
                
            case 'fear': // Trembling mouth
                const tremble = emotions.chaos * 5;
                mouthPath = `
                    M ${x - mouthWidth/2},${centerY + Math.sin(tremble) * 2}
                    Q ${x - mouthWidth/4},${centerY + Math.cos(tremble) * 3} 
                      ${x},${centerY + Math.sin(tremble) * 2}
                    Q ${x + mouthWidth/4},${centerY + Math.cos(tremble) * 3} 
                      ${x + mouthWidth/2},${centerY + Math.sin(tremble) * 2}`;
                break;
                
            case 'hope': // Gentle smile
                mouthPath = `
                    M ${x - mouthWidth/2},${centerY}
                    C ${x - mouthWidth/3},${centerY + mouthHeight * 0.5} 
                      ${x},${centerY + mouthHeight * 0.8} 
                      ${x},${centerY + mouthHeight * 0.6}
                    C ${x},${centerY + mouthHeight * 0.8} 
                      ${x + mouthWidth/3},${centerY + mouthHeight * 0.5} 
                      ${x + mouthWidth/2},${centerY}`;
                break;
                
            default: // Neutral expression
                mouthPath = `
                    M ${x - mouthWidth/2},${centerY}
                    C ${x - mouthWidth/3},${centerY + mouthHeight * 0.2} 
                      ${x},${centerY + mouthHeight * 0.3} 
                      ${x},${centerY + mouthHeight * 0.2}
                    C ${x},${centerY + mouthHeight * 0.3} 
                      ${x + mouthWidth/3},${centerY + mouthHeight * 0.2} 
                      ${x + mouthWidth/2},${centerY}`;
        }

        // Add mouth fill
        sequences.push({
            path: mouthPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: palette[1],
            width: 0,
            fill: true
        });

        // Add mouth outline
        sequences.push({
            path: mouthPath,
            duration: this.adjustDuration(1500),
            pause: this.adjustDuration(100),
            color: '#000000',
            width: this.calculateStrokeWidth(emotions, 'detail')
        });

        // For screaming mouth (rage), add inner mouth details
        if (emotionalState === 'rage') {
            const innerMouthPath = `
                M ${x - mouthWidth/3},${centerY + mouthHeight * 0.6}
                C ${x - mouthWidth/4},${centerY + mouthHeight} 
                  ${x + mouthWidth/4},${centerY + mouthHeight} 
                  ${x + mouthWidth/3},${centerY + mouthHeight * 0.6}`;
            
            sequences.push({
                path: innerMouthPath,
                duration: this.adjustDuration(800),
                pause: this.adjustDuration(50),
                color: '#000000',
                width: this.calculateStrokeWidth(emotions, 'detail') * 0.8
            });
        }

        return sequences;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (typeof rough === 'undefined') {
        console.error('rough.js is not loaded! Loading it now...');
        const script = document.createElement('script');
        script.src = 'https://ordinals.com/content/c39de9dcaa1a9ba756e14fad9460b9bb18de3a5447fe018bae02a7515ee37c65i0';
        script.onload = () => {
            const artist = new AutonomousArtist();
            artist.start();
        };
        document.head.appendChild(script);
    } else {
        const artist = new AutonomousArtist();
        artist.start();
    }
}); 