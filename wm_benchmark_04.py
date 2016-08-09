import numpy as np
import nengo
import nengo.spa as spa
import pylab

import ctn_benchmark

class Working_Memory(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensions', dimensions = 10)
        self.default('input_scale', input_scale = 0.1086521264298292)
        self.default('n_neurons_per_dim', n_neurons_per_dim = 50)
        
        self.default('intercept_low', intercept_low = -0.4986753102824562)
        self.default('intercept_high', intercept_high = 0.5627377682692168)
        self.default('tau_input', tau_input = 0.00222577425747703)
        self.default('tau_recurrent', tau_recurrent = 0.18286600309425424)
        self.default('tau_reset', tau_reset = 0.10423573561783896)
        self.default('max_rate_high', max_rate_high = 236.700401176162)
        self.default('max_rate_low', max_rate_low = 148.35745250164726)
        
        self.default('sensory_delay', sensory_delay = 0.05)
        self.default('reset_scale', reset_scale = 0.3)
        self.default('filter_threshold', filter_threshold = 0.0)
        
    def model(self, p): 
    
        model = nengo.Network()
        with model:
            vocab = spa.Vocabulary(p.dimensions)
            value = vocab.parse('A').v
            self.value = value
            def stim(t):
                if 0.5 < t - p.sensory_delay < 0.75:
                    return value
                else:
                    return [0]*p.dimensions
            stim = nengo.Node(stim)
            
            a = nengo.Ensemble(n_neurons=p.n_neurons_per_dim * p.dimensions,
                               dimensions=p.dimensions,
                               max_rates=nengo.dists.Uniform(p.max_rate_low, p.max_rate_high),
                               intercepts=nengo.dists.Uniform(p.intercept_low, p.intercept_high))
            
            b = nengo.Ensemble(n_neurons=p.n_neurons_per_dim * p.dimensions,
                               dimensions=p.dimensions,
                               max_rates=nengo.dists.Uniform(p.max_rate_low, p.max_rate_high),
                               intercepts=nengo.dists.Uniform(p.intercept_low, p.intercept_high))
            
            nengo.Connection(stim, a, synapse=None)
            nengo.Connection(a, b, synapse=p.tau_input, transform=p.input_scale)
            nengo.Connection(b, b, synapse=p.tau_recurrent)
            
            def reset(t):
                if t - p.sensory_delay > 1.75:
                    return 1
                else:
                    return 0
            reset_stim = nengo.Node(reset)
            reset_value = vocab.parse('B').v
            reset_value.shape = p.dimensions, 1
            nengo.Connection(reset_stim, b.neurons, transform=np.ones((b.n_neurons, 1))*-p.reset_scale, synapse=p.tau_reset)
            #nengo.Connection(reset_stim, b, transform=reset_value*p.reset_scale, synapse=p.tau_reset)
            self.b = b
            self.p_value = nengo.Probe(b, synapse=0.01)
            self.p_neurons = nengo.Probe(b.neurons)
        return model
            
    def evaluate(self, p, sim, plt):
        sim.run(2.5)
        
        rates = sim.data[self.p_neurons]
        ratesf = nengo.synapses.Lowpass(0.05).filt(rates)
        
        encs = sim.data[self.b].encoders
        similarity = np.dot(encs, self.value)
        items = np.where(similarity>p.filter_threshold)
        N = len(items[0])
        y_axis = np.mean(ratesf[:,items[0]], axis=1)
        pylab.plot(sim.trange(), np.mean(ratesf[:,items[0]], axis=1))
        pylab.axvline(0.5)
        pylab.axvline(0.75)
        pylab.axvline(1.75)
        pylab.show()
        
        act_510= y_axis[510]
        act_1750 = y_axis[1750]
        peak = y_axis[800]
        
        
        return {'N':N, 'act_510':act_510, 'act_1750':act_1750, 'peak':peak}
        
        
#FOR FILTER THRESHOLD = 0.2: BEST NUMBERS ARE
#{'max_rate_high': 197.19119519297644, 'intercept_low': -0.4992744299277808, 'intercept_high': #0.809991052831423, 'input_scale': 0.3833780061255102, 'max_rate_low': 119.17285810399966, 'tau_reset': #0.07956492225081398, 'tau_recurrent': 0.19573723138942678, 'tau_input': 0.013502354155587815}
    
# FOR FILTER THRESHOLD = 0.4, the best numbers are:     
#{'max_rate_high': 236.700401176162, 'intercept_low': -0.4986753102824562, 'intercept_high': #0.5627377682692168, 'input_scale': 0.1086521264298292, 'max_rate_low': 148.35745250164726, 'tau_reset': #0.10423573561783896, 'tau_recurrent': 0.18286600309425424, 'tau_input': 0.00222577425747703}   
