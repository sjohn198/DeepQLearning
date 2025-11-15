from ReplayBuffer import ReplayBuffer

rb = ReplayBuffer(1)
rb.push([1,2,3,4], 1,[1,2,3,4],1,1,1)
rb.sample(1)