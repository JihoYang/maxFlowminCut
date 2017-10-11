/*
#########################################################
#                                                       #
#  IBFSGraph -  Software for solving                    #
#               Maximum s-t Flow / Minimum s-t Cut      #
#               using the IBFS algorithm                #
#                                                       #
#  http://www.cs.tau.ac.il/~sagihed/ibfs/               #
#                                                       #
#  Haim Kaplan (haimk@cs.tau.ac.il)                     #
#  Sagi Hed (sagihed@post.tau.ac.il)                    #
#                                                       #
#########################################################

This software implements the IBFS (Incremental Breadth First Search) maximum flow algorithm from
	"Faster and More Dynamic Maximum Flow
	by Incremental Breadth-First Search"
	Andrew V. Goldberg, Sagi Hed, Haim Kaplan, Pushmeet Kohli,
	Robert E. Tarjan, and Renato F. Werneck
	In Proceedings of the 23rd European conference on Algorithms, ESA'15
	2015
and from
	"Maximum flows by incremental breadth-first search"
	Andrew V. Goldberg, Sagi Hed, Haim Kaplan, Robert E. Tarjan, and Renato F. Werneck.
	In Proceedings of the 19th European conference on Algorithms, ESA'11, pages 457-468.
	ISBN 978-3-642-23718-8
	2011

Copyright Haim Kaplan (haimk@cs.tau.ac.il) and Sagi Hed (sagihed@post.tau.ac.il)

###########
# LICENSE #
###########
This software can be used for research purposes only.
If you use this software for research purposes, you should cite the aforementioned papers
in any resulting publication and appropriately credit it.

If you require another license, please contact the above.

*/


#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "ibfs.h"


#define REMOVE_SIBLING(x, tmp) \
	{ (tmp) = (x)->parent->head->firstSon; \
	if ((tmp) == (x)) { \
		(x)->parent->head->firstSon = (x)->nextPtr; \
	} else { \
		for (; (tmp)->nextPtr != (x); (tmp) = (tmp)->nextPtr); \
		(tmp)->nextPtr = (x)->nextPtr; \
	} }

#define ADD_SIBLING(x, parentNode) \
	{ (x)->nextPtr = (parentNode)->firstSon; \
	(parentNode)->firstSon = (x); \
	}


IBFSGraph::IBFSGraph(IBFSInitMode a_initMode)
:prNodeBuckets(orphan3PassBuckets)
{
	initMode = a_initMode;
	arcIter = NULL;
	incList = NULL;
	incLen = incIteration = 0;
	numNodes = 0;
	uniqOrphansS = uniqOrphansT = 0;
	augTimestamp = 0;
	verbose = IBTEST;
	arcs = arcEnd = NULL;
	nodes = nodeEnd = NULL;
	topLevelS = topLevelT = 0;
	flow = 0;
	memArcs = NULL;
	tmpArcs = NULL;
	tmpEdges = tmpEdgeLast = NULL;
	ptrs = NULL;
	testFlow = 0;
	testExcess = 0;
	file = fileCompiled = NULL;
	fileIsCompiled = false;
	fileHasMore = false;
}


IBFSGraph::~IBFSGraph()
{
	delete []nodes;
	delete []memArcs;
	orphanBuckets.free();
	orphan3PassBuckets.free();
	excessBuckets.free();
	if (file != NULL) fclose(file);
	if (fileCompiled != NULL) fclose(file);
}

void IBFSGraph::initGraph()
{
	if (initMode == IB_INIT_FAST) {
		initGraphFast();
	} else if (initMode == IB_INIT_COMPACT) {
		initGraphCompact();
	}
	topLevelS = topLevelT = 1;
}


void IBFSGraph::initSize(int numNodes, int numEdges)
{
	// compute allocation size
	unsigned long long arcTmpMemsize = (unsigned long long)sizeof(TmpEdge)*(unsigned long long)numEdges;
	unsigned long long arcRealMemsize = (unsigned long long)sizeof(Arc)*(unsigned long long)(numEdges*2);
	unsigned long long nodeMemsize = (unsigned long long)sizeof(Node**)*(unsigned long long)(numNodes*3) +
			(IB_EXCESSES ? ((unsigned long long)sizeof(Node**)*(unsigned long long)(numNodes*2)) : 0);
	unsigned long long arcMemsize = 0;
	if (initMode == IB_INIT_FAST) {
		arcMemsize = arcRealMemsize + arcTmpMemsize;
	} else if (initMode == IB_INIT_COMPACT) {
		arcTmpMemsize += (unsigned long long)sizeof(TmpArc)*(unsigned long long)(numEdges*2);
		arcMemsize = arcTmpMemsize;
	}
	if (arcMemsize < (arcRealMemsize + nodeMemsize)) {
		arcMemsize = (arcRealMemsize + nodeMemsize);
	}

	// alocate arcs
	if (verbose) {
		fprintf(stdout, "c allocating arcs... \t [%lu MB]\n", (unsigned long)arcMemsize/(1<<20));
		fflush(stdout);
	}
	memArcs = new char[arcMemsize];
	memset(memArcs, 0, (unsigned long long)sizeof(char)*arcMemsize);
	if (initMode == IB_INIT_FAST) {
		tmpEdges = (TmpEdge*)(memArcs + arcRealMemsize);
	} else if (initMode == IB_INIT_COMPACT) {
		tmpEdges = (TmpEdge*)(memArcs);
		tmpArcs = (TmpArc*)(memArcs +arcMemsize -(unsigned long long)sizeof(TmpArc)*(unsigned long long)(numEdges*2));
	}
	tmpEdgeLast = tmpEdges; // will advance as edges are added
	arcs = (Arc*)memArcs;
	arcEnd = arcs + numEdges*2;

	// allocate nodes
//	if (verbose) {
//		fprintf(stdout, "c allocating nodes... \t [%lu MB]\n", (unsigned long)sizeof(Node)*(unsigned long)(numNodes+1) / (1<<20));
//		fflush(stdout);
//	}
	this->numNodes = numNodes;
	nodes = new Node[numNodes+1];
	memset(nodes, 0, sizeof(Node)*(numNodes+1));
	nodeEnd = nodes+numNodes;
	active0.init((Node**)(arcEnd));
	activeS1.init((Node**)(arcEnd) + numNodes);
	activeT1.init((Node**)(arcEnd) + (2*numNodes));
	if (IB_EXCESSES) {
		ptrs = (Node**)(arcEnd) + (3*numNodes);
		excessBuckets.init(nodes, ptrs, numNodes);
	}
	orphan3PassBuckets.init(nodes, numNodes);
	orphanBuckets.init(nodes, numNodes);

	// init members
	flow = 0;

	if (verbose) {
		fprintf(stdout, "c sizeof(ptr) = %d bytes\n", (int)sizeof(Node*));
		fprintf(stdout, "c sizeof(node) = %d bytes\n", (int)sizeof(Node));
		fprintf(stdout, "c sizeof(arc) = %d bytes\n", (int)sizeof(Arc));
	}
}


void IBFSGraph::initNodes()
{
	Node *x;
	for (x=nodes; x <= nodeEnd; x++) {
		x->firstArc = (arcs + x->label);
		if (x->excess == 0) {
			x->label = 0;
			continue;
		}
		if (x->excess > 0) {
			x->label = 1;
			activeS1.add(x);
		} else {
			x->label = -1;
			activeT1.add(x);
		}
	}
}

void IBFSGraph::initGraphFast()
{
	Node *x;
	TmpEdge *te;
	Arc *a;

	// calculate start arc offsets and labels for every node
	nodes->firstArc = arcs;
	for (x=nodes; x != nodeEnd; x++) {
		(x+1)->firstArc = x->firstArc + x->label;
		x->label = x->firstArc-arcs;
	}
	nodeEnd->label = arcEnd-arcs;

	// copy arcs
	for (te=tmpEdges; te != tmpEdgeLast; te++) {
		a = (nodes+te->tail)->firstArc;
		a->rev = (nodes+te->head)->firstArc;
		a->head = nodes+te->head;
		a->rCap = te->cap;
		a->isRevResidual = (te->revCap != 0);

		a = (nodes+te->head)->firstArc;
		a->rev = (nodes+te->tail)->firstArc;
		a->head = nodes+te->tail;
		a->rCap = te->revCap;
		a->isRevResidual = (te->cap != 0);

		++((nodes+te->head)->firstArc);
		++((nodes+te->tail)->firstArc);
	}

	initNodes();
}


void IBFSGraph::initGraphCompact()
{
	Node *x;
	Arc *a;
	TmpArc *ta, *taEnd;
	TmpEdge *te;

	// tmpEdges:			edges read
	// node.label:			out degree

	// calculate start arc offsets every node
	nodes->firstArc = (Arc*)(tmpArcs);
	for (x=nodes; x != nodeEnd; x++) {
		(x+1)->firstArc = (Arc*)(((TmpArc*)(x->firstArc)) + x->label);
		x->label = ((TmpArc*)(x->firstArc))-tmpArcs;
	}
	nodeEnd->label = arcEnd-arcs;

	// tmpEdges:				edges read
	// node.label: 				index into arcs array of first out arc
	// node.firstArc-tmpArcs: 	index into arcs array of next out arc to be allocated
	//							(initially the first out arc)

	// copy to temp arcs memory
	if (IB_DEBUG_INIT) {
		IBDEBUG("c initFast copy1");
	}
	for (te=tmpEdges; te != tmpEdgeLast; te++) {
		ta = (TmpArc*)((nodes+te->tail)->firstArc);
		ta->cap = te->cap;
		ta->rev = (TmpArc*)((nodes+te->head)->firstArc);

		ta = (TmpArc*)((nodes+te->head)->firstArc);
		ta->cap = te->revCap;
		ta->rev = (TmpArc*)((nodes+te->tail)->firstArc);

		(nodes+te->tail)->firstArc = (Arc*)(((TmpArc*)((nodes+te->tail)->firstArc))+1);
		(nodes+te->head)->firstArc = (Arc*)(((TmpArc*)((nodes+te->head)->firstArc))+1);
	}

	// tmpEdges:				edges read
	// tmpArcs:					arcs with reverse pointer but no node id
	// node.label: 				index into arcs array of first out arc
	// node.firstArc-tmpArcs: 	index into arcs array of last allocated out arc

	// copy to permanent arcs array, but saving tail instead of head
	if (IB_DEBUG_INIT) {
		IBDEBUG("c initFast copy2");
	}
	a = arcs;
	x = nodes;
	taEnd = (tmpArcs+(arcEnd-arcs));
	for (ta=tmpArcs; ta != taEnd; ta++) {
		while (x->label <= (ta-tmpArcs)) x++;
		a->head = (x-1);
		a->rCap = ta->cap;
		a->rev = arcs + (ta->rev-tmpArcs);
		a++;
	}

	// tmpEdges:				overwritten
	// tmpArcs:					overwritten
	// arcs:					arcs array
	// node.label: 				index into arcs array of first out arc
	// node.firstArc-tmpArcs: 	index into arcs array of last allocated out arc
	// arc.head = tail of arc

	// swap the head and tail pointers and set isRevResidual
	if (IB_DEBUG_INIT) {
		IBDEBUG("c initFast copy3");
	}
	for (a=arcs; a != arcEnd; a++) {
		if (a->rev <= a) continue;
		x = a->head;
		a->head = a->rev->head;
		a->rev->head = x;
		a->isRevResidual = (a->rev->rCap != 0);
		a->rev->isRevResidual = (a->rCap != 0);
	}

	// set firstArc pointers in nodes array
	if (IB_DEBUG_INIT) {
		IBDEBUG("c initFast nodes");
	}
	initNodes();

	// check consistency
	if (IBTEST || IB_DEBUG_INIT) {
		IBDEBUG("c initFast test");
		for (x=nodes; x != nodeEnd; x++) {
			if ((x+1)->firstArc < x->firstArc) {
				fprintf(stderr, "INIT CONSISTENCY: arc pointers descending");
				exit(1);
			}
			for (a=x->firstArc; a !=(x+1)->firstArc; a++) {
				if (a->rev->head != x) {
					fprintf(stderr, "INIT CONSISTENCY: arc head pointer inconsistent");
					exit(1);
				}
				if (a->rev->rev != a) {
					fprintf(stderr, "INIT CONSISTENCY: arc reverse pointer inconsistent");
					exit(1);
				}
			}
		}
	}
}



// @ret: minimum orphan level
template<bool sTree> int IBFSGraph::augmentPath(Node *x, int push)
{
	Node *y;
	Arc *a;
	int orphanMinLevel = (sTree ? topLevelS : topLevelT) + 1;

	augTimestamp++;
	for (; ; x=a->head)
	{
		stats.incPushes();
		if (x->excess) break;
		a = x->parent;
		if (sTree) {
			a->rCap += push;
			a->rev->isRevResidual = 1;
			a->rev->rCap -= push;
		} else {
			a->rev->rCap += push;
			a->isRevResidual = 1;
			a->rCap -= push;
		}

		// saturated?
		if ((sTree ? (a->rev->rCap) : (a->rCap)) == 0)
		{
			if (sTree) a->isRevResidual = 0;
			else a->rev->isRevResidual = 0;
			REMOVE_SIBLING(x,y);
			orphanMinLevel = (sTree ? x->label : -x->label);
			orphanBuckets.add<sTree>(x);
		}
	}
	x->excess += (sTree ? -push : push);
	if (x->excess == 0) {
		orphanMinLevel = (sTree ? x->label : -x->label);
		orphanBuckets.add<sTree>(x);
	}
	flow += push;

	return orphanMinLevel;
}


// @ret: minimum level in which created an orphan
template<bool sTree> int IBFSGraph::augmentExcess(Node *x, int push)
{
	Node *y;
	Arc *a;
	int orphanMinLevel = (sTree ? topLevelS : topLevelT)+1;
	augTimestamp++;

	// start of loop
	//----------------
	// x 		  the current node along the path
	// a		  arc incoming into x
	// push 	  the amount of flow coming into x
	// a->resCap  updated with incoming flow already
	// x->excess  not updated with incoming flow yet
	//
	// end of loop
	//-----------------
	// x 		  the current node along the path
	// a		  arc outgoing from x
	// push 	  the amount of flow coming out of x
	// a->resCap  updated with outgoing flow already
	// x->excess  updated with incoming flow already
	while (sTree ? (x->excess <= 0) : (x->excess >= 0))
	{
		testNode(x);
		stats.incPushes();
		a = x->parent;

		// update excess and find next flow
		if ((sTree ? (a->rev->rCap) : (a->rCap)) < (sTree ? (push-x->excess) : (x->excess+push))) {
			// some excess remains, node is an orphan
			x->excess += (sTree ? (a->rev->rCap - push) : (push-a->rCap));
			push = (sTree ? a->rev->rCap : a->rCap);
		} else {
			// all excess is pushed out, node may or may not be an orphan
			push += (sTree ? -(x->excess) : x->excess);
			x->excess = 0;
		}

		// push flow
		// note: push != 0
		if (sTree) {
			a->rCap += push;
			a->rev->isRevResidual = 1;
			a->rev->rCap -= push;
		} else {
			a->rev->rCap += push;
			a->isRevResidual = 1;
			a->rCap -= push;
		}

		// saturated?
		if ((sTree ? (a->rev->rCap) : (a->rCap)) == 0)
		{
			if (sTree) a->isRevResidual = 0;
			else a->rev->isRevResidual = 0;
			REMOVE_SIBLING(x,y);
			orphanMinLevel = (sTree ? x->label : -x->label);
			orphanBuckets.add<sTree>(x);
			if (x->excess) excessBuckets.incMaxBucket(sTree ? x->label : -x->label);
		}

		// advance
		// a precondition determines that the first node on the path is not in excess buckets
		// so only the next nodes may need to be removed from there
		x = a->head;
		if (sTree ? (x->excess < 0) : (x->excess > 0)) excessBuckets.remove<sTree>(x);
	}

	// update the excess at the root
	if (push <= (sTree ? (x->excess) : -(x->excess))) flow += push;
	else flow += (sTree ? (x->excess) : -(x->excess));
	x->excess += (sTree ? (-push) : push);
	if (sTree ? (x->excess <= 0) : (x->excess >= 0)) {
		orphanMinLevel = (sTree ? x->label : -x->label);
		orphanBuckets.add<sTree>(x);
		if (x->excess) excessBuckets.incMaxBucket(sTree ? x->label : -x->label);
	}

	return orphanMinLevel;
}


template<bool sTree> void IBFSGraph::augmentExcesses()
{
	Node *x;
	int minOrphanLevel;
	int adoptedUpToLevel = excessBuckets.maxBucket;

	if (!excessBuckets.empty())
	for (; excessBuckets.maxBucket != (excessBuckets.minBucket-1); excessBuckets.maxBucket--)
	while ((x=excessBuckets.popFront(excessBuckets.maxBucket)) != NULL)
	{
		minOrphanLevel = augmentExcess<sTree>(x, 0);
		// if we did not create new orphans
		if (adoptedUpToLevel < minOrphanLevel) minOrphanLevel = adoptedUpToLevel;
		adoption<sTree>(minOrphanLevel, false);
		adoptedUpToLevel = excessBuckets.maxBucket;
	}
	excessBuckets.reset();
	if (orphanBuckets.maxBucket != 0) adoption<sTree>(adoptedUpToLevel+1, true);
	// free 3pass orphans
	while ((x=excessBuckets.popFront(0)) != NULL) orphanFree<sTree>(x);
}


void IBFSGraph::augment(Arc *bridge)
{
	Node *x, *y;
	Arc *a;
	int bottleneck, bottleneckT, bottleneckS, minOrphanLevel;
	bool forceBottleneck;
	stats.incAugs();

	// must compute forceBottleneck once, so that it is constant throughout this method
	forceBottleneck = (IB_EXCESSES ? false : true);
	if (IB_BOTTLENECK_ORIG && IB_EXCESSES)
	{
		// limit by end nodes excess
		bottleneck = bridge->rCap;
		if (bridge->head->excess != 0 && -(bridge->head->excess) < bottleneck) {
			bottleneck = -(bridge->head->excess);
		}
		if (bridge->rev->head->excess != 0 && bridge->rev->head->excess < bottleneck) {
			bottleneck = bridge->rev->head->excess;
		}
	}
	else
	{
		bottleneck = bottleneckS = bridge->rCap;
		if (bottleneck != 1) {
			for (x=bridge->rev->head; ; x=a->head)
			{
				if (x->excess) break;
				a = x->parent;
				if (bottleneckS > a->rev->rCap) {
					bottleneckS = a->rev->rCap;
				}
			}
			if (bottleneckS > x->excess) {
				bottleneckS = x->excess;
			}
			if (IB_EXCESSES && x->label != 1) forceBottleneck = true;
			if (x == bridge->rev->head) bottleneck = bottleneckS;
		}

		if (bottleneck != 1) {
			bottleneckT = bridge->rCap;
			for (x=bridge->head; ; x=a->head)
			{
				if (x->excess) break;
				a = x->parent;
				if (bottleneckT > a->rCap) {
					bottleneckT = a->rCap;
				}
			}
			if (bottleneckT > (-x->excess)) {
				bottleneckT = (-x->excess);
			}
			if (IB_EXCESSES && x->label != -1) forceBottleneck = true;
			if (x == bridge->head && bottleneck > bottleneckT) bottleneck = bottleneckT;

			if (forceBottleneck) {
				if (bottleneckS < bottleneckT) bottleneck = bottleneckS;
				else bottleneck = bottleneckT;
			}
		}
	}

	// stats
	if (IBSTATS) {
		int augLen = (-(bridge->head->label)-1 + bridge->rev->head->label-1 + 1);
		stats.addAugLen(augLen);
	}

	// augment connecting arc
	bridge->rev->rCap += bottleneck;
	bridge->isRevResidual = 1;
	bridge->rCap -= bottleneck;
	if (bridge->rCap == 0) {
		bridge->rev->isRevResidual = 0;
	}
	stats.incPushes();
	flow -= bottleneck;

	// augment T
	x = bridge->head;
	if (!IB_EXCESSES || bottleneck == 1 || forceBottleneck) {
		minOrphanLevel = augmentPath<false>(x, bottleneck);
		adoption<false>(minOrphanLevel, true);
	} else if (IB_ADOPTION_PR && !x->excess) {
		x->excess += bottleneck;
		excessBuckets.add<false>(x);
		REMOVE_SIBLING(x,y);
		augmentExcessesDischarge<false>();
	} else {
		minOrphanLevel = augmentExcess<false>(x, bottleneck);
		adoption<false>(minOrphanLevel, false);
		augmentExcesses<false>();
	}

	// augment S
	x = bridge->rev->head;
	if (!IB_EXCESSES || bottleneck == 1 || forceBottleneck) {
		minOrphanLevel = augmentPath<true>(x, bottleneck);
		adoption<true>(minOrphanLevel, true);
	} else if (IB_ADOPTION_PR && !x->excess) {
		x->excess -= bottleneck;
		excessBuckets.add<true>(x);
		REMOVE_SIBLING(x,y);
		augmentExcessesDischarge<true>();
	} else {
		minOrphanLevel = augmentExcess<true>(x, bottleneck);
		adoption<true>(minOrphanLevel, false);
		augmentExcesses<true>();
	}
}




template<bool sTree> void IBFSGraph::adoption(int fromLevel, bool toTop)
{
	Node *x, *y, *z;
	register Arc *a;
	Arc *aEnd;
	int threePassLevel;
	int minLabel, numOrphans, numOrphansUniq;
	int level;

	threePassLevel=0;
	numOrphans=0;
	numOrphansUniq=0;
	for (level = fromLevel;
		level <= orphanBuckets.maxBucket &&
		(!IB_EXCESSES || toTop || threePassLevel || level <= excessBuckets.maxBucket);
		level++)
	while ((x=orphanBuckets.popFront(level)) != NULL)
	{
		testNode(x);
		stats.incOrphans();
		numOrphans++;
		if (x->lastAugTimestamp != augTimestamp) {
			x->lastAugTimestamp = augTimestamp;
			if (sTree) uniqOrphansS++;
			else uniqOrphansT++;
			numOrphansUniq++;
		}
		if (IB_HYBRID_ADOPTION && threePassLevel == 0 && numOrphans >= 3*numOrphansUniq) {
			// switch to 3pass
			threePassLevel = 1;
		}

		//
		// check for same level connection
		//
		if (x->isParentCurr) {
			a = x->parent;
		} else {
			a = x->firstArc;
			x->isParentCurr = 1;
		}
		x->parent = NULL;
		aEnd = (x+1)->firstArc;
		if (x->label != (sTree ? 1 : -1))
		{
			minLabel = x->label - (sTree ? 1 : -1);
			for (; a != aEnd; a++)
			{
				stats.incOrphanArcs1();
				y = a->head;
				if ((sTree ? a->isRevResidual : a->rCap) != 0 && y->label == minLabel)
				{
					x->parent = a;
					ADD_SIBLING(x,y);
					break;
				}
			}
		}
		if (x->parent != NULL) {
			if (IB_EXCESSES && x->excess) excessBuckets.add<sTree>(x);
			continue;
		}

		//
		// on the top level there is no need to relabel
		//
		if (x->label == (sTree ? topLevelS : -topLevelT)) {
			orphanFree<sTree>(x);
			continue;
		}

		//
		// give up on same level - relabel it!
		// (1) create orphan sons
		//
		for (y=x->firstSon; y != NULL; y=z)
		{
			stats.incOrphanArcs3();
			z=y->nextPtr;
			if (IB_EXCESSES && y->excess) excessBuckets.remove<sTree>(y);
			orphanBuckets.add<sTree>(y);
		}
		x->firstSon = NULL;

		//
		// (2) 3pass relabeling: move to buckets structure
		//
		if (threePassLevel) {
			x->label += (sTree ? 1 : -1);
			orphan3PassBuckets.add<sTree>(x);
			if (threePassLevel == 1) {
				threePassLevel = level+1;
			}
			continue;
		}

		//
		// (2) relabel: find the lowest level parent
		//
		minLabel = (sTree ? topLevelS : -topLevelT);
		if (x->label != minLabel) for (a=x->firstArc; a != aEnd; a++)
		{
			stats.incOrphanArcs2();
			y = a->head;
			if ((sTree ? a->isRevResidual : a->rCap) &&
				// y->label != 0 ---> holds implicitly
				(sTree ? (y->label > 0) : (y->label < 0)) &&
				(sTree ? (y->label < minLabel) : (y->label > minLabel)))
			{
				minLabel = y->label;
				x->parent = a;
				if (minLabel == x->label) break;
			}
		}

		//
		// (3) relabel onto new parent
		//
		if (x->parent != NULL) {
			x->label = minLabel + (sTree ? 1 : -1);
			ADD_SIBLING(x, x->parent->head);
			// add to active list of the next growth phase
			if (sTree) {
				if (x->label == topLevelS) activeS1.add(x);
			} else {
				if (x->label == -topLevelT) activeT1.add(x);
			}
			if (IB_EXCESSES && x->excess) excessBuckets.add<sTree>(x);
		} else {
			orphanFree<sTree>(x);
		}
	}
	if (level > orphanBuckets.maxBucket) orphanBuckets.maxBucket=0;

	if (threePassLevel) {
		adoption3Pass<sTree>(threePassLevel);
	}
}

template <bool sTree> void IBFSGraph::adoption3Pass(int minBucket)
{
	Arc *a, *aEnd;
	Node *x, *y;
	int minLabel, destLabel;

	for (int level=minBucket; level <= orphan3PassBuckets.maxBucket; level++)
	while ((x = orphan3PassBuckets.popFront(level)) != NULL)
	{
		testNode(x);
		aEnd = (x+1)->firstArc;

		// pass 2: find lowest level parent
		if (x->parent == NULL) {
			minLabel = (sTree ? topLevelS : -topLevelT);
			destLabel = x->label - (sTree ? 1 : -1);
			for (a=x->firstArc; a != aEnd; a++) {
				stats.incOrphanArcs3();
				y = a->head;
				if ((sTree ? a->isRevResidual : a->rCap) &&
					((sTree ? (y->excess > 0) : (y->excess < 0)) || y->parent != NULL) &&
					(sTree ? (y->label > 0) : (y->label < 0)) &&
					(sTree ? (y->label < minLabel) : (y->label > minLabel)))
				{
					x->parent = a;
					if ((minLabel = y->label) == destLabel) break;
				}
			}
			if (x->parent == NULL) {
				x->label = 0;
				if (IB_EXCESSES && x->excess) excessBuckets.add<sTree>(x);
				continue;
			}
			x->label = minLabel + (sTree ? 1 : -1);
			if (x->label != (sTree ? level : -level)) {
				orphan3PassBuckets.add<sTree>(x);
				continue;
			}
		}

		// pass 3: lower potential sons and/or find first parent
		if (x->label != (sTree ? topLevelS : -topLevelT))
		{
			minLabel = x->label + (sTree ? 1 : -1);
			for (a=x->firstArc; a != aEnd; a++) {
				stats.incOrphanArcs3();
				y = a->head;

				// lower potential sons
				if ((sTree ? a->rCap : a->isRevResidual) &&
					(y->label == 0 ||
					(sTree ? (minLabel < y->label) : (minLabel > y->label))))
				{
					if (y->label != 0) orphan3PassBuckets.remove<sTree>(y);
					else if (IB_EXCESSES && y->excess) excessBuckets.remove<sTree>(y);
					y->label = minLabel;
					y->parent = a->rev;
					orphan3PassBuckets.add<sTree>(y);
				}
			}
		}

		// relabel onto new parent
		ADD_SIBLING(x, x->parent->head);
		x->isParentCurr = 0;
		if (IB_EXCESSES && x->excess) excessBuckets.add<sTree>(x);

		// add to active list of the next growth phase
		if (sTree) {
			if (x->label == topLevelS) activeS1.add(x);
		} else {
			if (x->label == -topLevelT) activeT1.add(x);
		}
	}

	orphan3PassBuckets.maxBucket = 0;
}


template<bool dirS> void IBFSGraph::growth()
{
	Node *x, *y;
	Arc *a, *aEnd;

	for (Node **active=active0.list; active != (active0.list + active0.len); active++)
	{
		// get active node
		x = (*active);
		testNode(x);

		// node no longer at level
		if (x->label != (dirS ? (topLevelS-1): -(topLevelT-1))) {
			continue;
		}

		// grow or augment
		if (dirS) stats.incGrowthS();
		else stats.incGrowthT();
		aEnd = (x+1)->firstArc;
		for (a=x->firstArc; a != aEnd; a++)
		{
			stats.incGrowthArcs();
			if ((dirS ? a->rCap : a->isRevResidual) == 0) continue;
			y = a->head;
			if (y->label == 0)
			{
				// grow node x (attach y)
				testNode(y);
				y->isParentCurr = 0;
				y->label = x->label + (dirS ? 1 : -1);
				y->parent = a->rev;
				ADD_SIBLING(y, x);
				if (dirS) activeS1.add(y);
				else activeT1.add(y);
			}
			else if (dirS ? (y->label < 0) : (y->label > 0))
			{
				// augment
				augment(dirS ? a : (a->rev));
				if (x->label != (dirS ? (topLevelS-1) : -(topLevelT-1))) {
					break;
				}
				if (dirS ? (a->rCap) : (a->isRevResidual)) a--;
			}
		}
	}
	active0.clear();
}

template<bool sTree> void IBFSGraph::augmentIncrements()
{
	Node *x, *y;
	Node **end = incList+incLen;
	int minOrphanLevel = 1<<30;

	for (Node **inc=incList; inc != end; inc++)
	{
		x = (*inc);
		testNode(x);
		if (!x->isIncremental || (sTree ? (x->label < 0) : (x->label > 0))) continue;
		x->isIncremental = 0;
		if (x->label == 0)
		{
			//**** new root from outside the tree
			if (!x->excess) continue;
			x->isParentCurr = 0;
			if (x->excess > 0) {
				x->label = topLevelS;
				activeS1.add(x);
			} else if (x->excess < 0) {
				x->label = -topLevelT;
				activeT1.add(x);
			}
		}
		else if ((sTree ? (x->excess <= 0) : (x->excess >= 0)) &&
				(!x->parent || !(sTree ? x->parent->isRevResidual : x->parent->rCap)))
		{
			//**** new orphan
			if (x->parent) REMOVE_SIBLING(x,y);
			if ((sTree ? x->label : -x->label) < minOrphanLevel) {
				minOrphanLevel = (sTree ? x->label : -x->label);
			}
			orphanBuckets.add<sTree>(x);
			if (x->excess) excessBuckets.incMaxBucket(sTree ? x->label : -x->label);
		}
		else if (sTree ? (x->excess < 0) : (x->excess > 0))
		{
			//**** new deficit/excess to empty
			excessBuckets.add<sTree>(x);
		}
		else if (x->excess && x->parent)
		{
			//**** new root
			REMOVE_SIBLING(x,y);
			x->parent = NULL;
			x->isParentCurr = 0;
		}
	}
	if (orphanBuckets.maxBucket != 0) adoption<sTree>(minOrphanLevel, false);
	if (IB_ADOPTION_PR) augmentExcessesDischarge<sTree>();
	else augmentExcesses<sTree>();
}


int IBFSGraph::computeMaxFlow()
{
	return computeMaxFlow(true, false);
}

int IBFSGraph::computeMaxFlow(bool allowIncrements)
{
	return computeMaxFlow(true, allowIncrements);
}

int IBFSGraph::computeMaxFlow(bool initialDirS, bool allowIncrements)
{
	// incremental?
	if (incIteration >= 1 && incList != NULL) {
		augmentIncrements<true>();
		augmentIncrements<false>();
		incList = NULL;
	}

	// test
	if (IBTEST) {
		testFlow = flow;
		for (Node *x=nodes; x < nodeEnd; x++) {
			if (x->excess > 0) testExcess += x->excess;
		}
	}

	//
	// IBFS
	//
	bool dirS = initialDirS;
	while (true)
	{
		// BFS level
		if (dirS) {
			ActiveList::swapLists(&active0, &activeS1);
			topLevelS++;
		} else {
			ActiveList::swapLists(&active0, &activeT1);
			topLevelT++;
		}
		orphanBuckets.allocate((topLevelS > topLevelT) ? topLevelS : topLevelT);
		orphan3PassBuckets.allocate((topLevelS > topLevelT) ? topLevelS : topLevelT);
		if (IB_EXCESSES) excessBuckets.allocate((topLevelS > topLevelT) ? topLevelS : topLevelT);
		if (dirS) growth<true>();
		else growth<false>();
		if (IBTEST) {
			testTree();
			fprintf(stdout, "dirS=%d aug=%d   S %d / T %d   flow=%d\n",
					dirS, augTimestamp, uniqOrphansS, uniqOrphansT, flow);
			fflush(stdout);
		}

		// switch to next level
		if (!allowIncrements && (activeS1.len == 0 || activeT1.len == 0)) break;
		if (activeS1.len == 0 && activeT1.len == 0) break;
		if (activeT1.len == 0) dirS=true;
		else if (activeS1.len == 0) dirS=false;
		else if (!IB_ALTERNATE_SMART && dirS) dirS = false;
		else if (IB_ALTERNATE_SMART && uniqOrphansT == uniqOrphansS && dirS) dirS=false;
		else if (IB_ALTERNATE_SMART && uniqOrphansT < uniqOrphansS) dirS=false;
		else dirS=true;
	}

	incIteration++;
	return flow;
}


///////////////////////////////////////////////////
// experimental min marginals
///////////////////////////////////////////////////
void IBFSGraph::computeMinMarginals()
{
	int *srcSide;
	Arc *a;
	Node *x;
//	Node **inc, **end;
	stats.reset();

	// compute infinite capacity
	srcSide = new int[nodeEnd-nodes];
//	int *flowDiffs = new int[nodeEnd-nodes];
//	maxDeg = maxExcess = 0;
	for (x=nodes; x != nodeEnd; x++) {
//		if (maxDeg < ((x+1)->firstArc - x->firstArc)) {
//			maxDeg = ((x+1)->firstArc - x->firstArc);
//		}
//		if (x->excess > maxExcess) maxExcess = x->excess;
//		else if (x->excess < -maxExcess) maxExcess = -x->excess;
		srcSide[x-nodes] = isNodeOnSrcSide(x-nodes, 2);
	}
//	if (((double)maxDeg*(double)maxCap + (double)maxExcess + 1) > (double)(1<<30)) {
//		fprintf(stdout, "ERROR: Infinite Capacity is too large %.0f\n",
//				(double)maxDeg*(double)maxCap + (double)maxExcess + 1);
//		exit(1);
//	}
//	infCap = maxDeg*maxCap + maxExcess + 1;

	// compute confidence
//	incChangedList.init(numNodes);
//	Arc *arcsCopy = new Arc[arcEnd-arcs];
//	Node *nodesCopy = new Node[numNodes];

//	for (x=nodes; x != nodeEnd; x++) {
//		if (x->label > 0) {
//			x->firstSon = NULL;
//			x->parent = NULL;
//			x->isParentCurr = 0;
//			if (x->excess == 0) {
//				x->label = 0;
//			} else {
//				x->label = 1;
//			}
//		}
//	}
//	topLevelS=1;

	int flowCopy = flow;
//	int topLevelSCopy = topLevelS;
//	int topLevelTCopy = topLevelT;
//	memcpy(arcsCopy, arcs, sizeof(Arc)*(arcEnd-arcs));
//	memcpy(nodesCopy, nodes, sizeof(Node)*(numNodes));

	int nEmpty=0;
	for (int nodeIndex=0; nodeIndex < (nodeEnd-nodes); nodeIndex++)
	{
		if (srcSide[nodeIndex] == 2) {
//			flowDiffs[nodeIndex] = 0;
			nEmpty++;
			continue;
		}

//		//***** WHILE
//		bool newCutHasSons = true;
//		while (newCutHasSons) {
//		int depth=0;
		int infCap = (srcSide[nodeIndex] ? (nodes[nodeIndex].excess) : (-nodes[nodeIndex].excess));
		for (a=nodes[nodeIndex].firstArc; a != nodes[nodeIndex+1].firstArc; a++) {
			if (srcSide[nodeIndex]) infCap += a->rev->rCap;
			else infCap += a->rCap;
		}
		if (srcSide[nodeIndex]) incNode(nodeIndex, 0, infCap);
		else incNode(nodeIndex, infCap, 0);
		int flowDiff = computeMaxFlow(false, !srcSide[nodeIndex])-flowCopy;
		if (flowDiff == infCap || flowDiff == -infCap) nEmpty++;
//		testTree();

//		srcSide[nodeIndex]|=8;
//		newCutHasSons = false;
//		if (nodes[nodeIndex].firstSon == NULL) nEmpty++;
//		for (x=nodes[nodeIndex].firstSon; false && x != NULL; x=x->nextPtr)
//		{
//			if ((srcSide[x-nodes]&8)==0 && (srcSide[x-nodes]%4)==(srcSide[nodeIndex]%4))
//			{
//				depth++;
//				if (depth >= 2) fprintf(stdout, "DEPTH %d\n", depth);
//				if ((srcSide[nodeIndex]%4)) incNode(nodeIndex, 0, -infCap);
//				else incNode(nodeIndex, -infCap, 0);
//				nodeIndex = x-nodes;
//				newCutHasSons = true;
//				break;
//			}
//		}
//		} //***** END WHILE


		if (IB_MIN_MARGINALS_DEBUG && (incIteration % ((nodeEnd-nodes)/10) == 0)) {
			int minLabelS=topLevelS;
			int minLabelT=topLevelT;
			for (x=nodes; x != nodeEnd; x++) {
				if (x->label > 0 && x->label < minLabelS) minLabelS = x->label;
				else if (x->label < 0 && -x->label < minLabelT) minLabelT = -x->label;
			}
			fprintf(stdout, "%d (%d>%d, %d>%d, %d) ", (int)(100*incIteration/(nodeEnd-nodes)),
					minLabelS, topLevelS, minLabelT, topLevelT, flow);
			fflush(stdout);
		}

//		end=incChangedList.list + incChangedList.len;
//		int maxLabel=0;
//		for (inc=incChangedList.list; inc != end; inc++) {
//			x = *inc;
//			int label = (x->label > 0 ? x->label : (-x->label));
//			if (label > maxLabel) maxLabel=label;
//			(*x) = nodesCopy[(*inc)-nodes];
//			memcpy(x->firstArc, arcsCopy+(x->firstArc-arcs), ((x+1)->firstArc-x->firstArc)*sizeof(Arc));
//		}
//		if (maxLabel > 0) memset(orphanBuckets.buckets, 0, sizeof(Node*)*(maxLabel+1));
//		topLevelS = topLevelSCopy;
//		topLevelT = topLevelTCopy;
//		activeS1.len = 0;
//		activeT1.len = 0;
//		flow = flowCopy;
//		incChangedList.len = 0;

//		IBSTOP;
//		double t = IBSECS;
//		if (t >= 0.01) {
//			fprintf(stdout, "\n iteration %d  time %f\n", incIteration-1, t);
//			fflush(stdout);
//		}

			if (srcSide[nodeIndex]) incNode(nodeIndex, 0, -infCap);
			else incNode(nodeIndex, -infCap, 0);
	}
	fprintf(stdout, "\n");
//	if (IB_MIN_MARGINALS_TEST) {
//		for (int i=0; i < numNodes; i++) {
//			fprintf(stdout, "%d\n", flowDiffs[i]);
//		}
//	}
	fprintf(stdout, "c trivial = %.02f\n", nEmpty / (float)numNodes);
	delete []srcSide;
}



///////////////////////////////////////////////////
// experimental push relabel orphan processing
///////////////////////////////////////////////////
template<bool sTree> void IBFSGraph::augmentExcessesDischarge()
{
	Node *x;
	if (!excessBuckets.empty())
	for (; excessBuckets.maxBucket != (excessBuckets.minBucket-1); excessBuckets.maxBucket--)
	while ((x=excessBuckets.popFront(excessBuckets.maxBucket)) != NULL) {
		augmentDischarge<sTree>(x);
	}
	excessBuckets.reset();
	while ((x=excessBuckets.popFront(0)) != NULL) {
		x->isIncremental = 0;
		orphanBuckets.add<sTree>(x);
		// TODO: add orphan min level optimization here
	}
	augTimestamp++;
	adoption<sTree>(1, true);
}

// @pre: !x->isIncremental && x not in excessBuckets[0] && x not in x->parent sons list
template<bool sTree> void IBFSGraph::augmentDischarge(Node *x)
{
	Node *y, *z;
	int minLabel, push;
	Arc *aEnd = (x+1)->firstArc;
	Arc *a;
	int startLabel = x->label;
	testNode(x);

	// loop
	while (true)
	{
		// push
		if (x->isParentCurr) {
			a = x->parent;
		} else {
			a = x->firstArc;
			x->isParentCurr = 1;
		}
		if (x->label != (sTree ? 1 : -1) && a != NULL)
		{
			minLabel = x->label + (sTree ? -1 : 1);
			for (; a != aEnd; a++)
			{
				// check admissible
				y = a->head;
				if ((sTree ? a->isRevResidual : a->rCap) == 0 || y->label != minLabel) {
					continue;
				}

				// push admissible
				push = (sTree ? (a->rev->rCap) : (a->rCap));
				if (push > (sTree ? (-x->excess) : (x->excess))) {
					push = (sTree ? (-x->excess) : (x->excess));
				}
				x->excess += (sTree ? push : (-push));
				if (sTree) {
					a->rev->rCap -= push;
					a->rCap += push;
					a->rev->isRevResidual = 1;
					a->isRevResidual = (a->rev->rCap ? 1 : 0);
				} else {
					a->rCap -= push;
					a->rev->rCap += push;
					a->rev->isRevResidual = (a->rCap ? 1 : 0);
					a->isRevResidual = 1;
				}

				// add excess
				if (sTree && y->excess > 0) {
					if (y->excess >= push) flow += push;
					else flow += y->excess;
				} else if (!sTree && y->excess < 0) {
					if (-y->excess >= push) flow += push;
					else flow -= y->excess;
				}
				y->excess += (sTree ? (-push) : push);
				if (y->excess == 0 /* implicit && !y->isIncremental && y has no parent */) {
					y->label = 0;
					excessBuckets.add<sTree>(y);
					y->label = minLabel;
					y->isIncremental = 1;
				} else if (sTree ? (y->excess < 0 && y->excess >= -push) : (y->excess > 0 && y->excess <= push)) {
					if (y->isIncremental) {
						y->label = 0;
						excessBuckets.remove<sTree>(y);
						y->label = minLabel;
						y->isIncremental = 0;
					} else if (y->parent != NULL) {
						REMOVE_SIBLING(y,z);
					}
					excessBuckets.add<sTree>(y);
				}
				if (x->excess == 0) {
					x->parent = a;
					if (!(sTree ? a->isRevResidual : a->rCap)) {
						x->label = 0;
						excessBuckets.add<sTree>(x);
						x->label = minLabel + (sTree ? 1 : -1);
						x->isIncremental = 1;
					}
					break;
				}
			}
		}
		if (x->excess == 0) break;

		// make sons orphans
		minLabel = x->label + (sTree ? 1 : -1);
		for (y=x->firstSon; y != NULL; y=z)
		{
			stats.incOrphanArcs3();
			z=y->nextPtr;
			// implicit !y->isIncremental && !y->excess
			y->label = 0;
			excessBuckets.add<sTree>(y);
			y->label = minLabel;
			y->isIncremental = 1;
		}
		x->firstSon = NULL;

		// relabel
		minLabel = (sTree ? topLevelS : -topLevelT);
		x->parent = NULL;
		for (a=x->firstArc; a != aEnd; a++)
		{
			y = a->head;
			if ((sTree ? a->isRevResidual : a->rCap) &&
				// y->label != 0 ---> holds implicitly
				(sTree ? (y->label > 0) : (y->label < 0)) &&
				(sTree ? (y->label < minLabel) : (y->label > minLabel)))
			{
				minLabel = y->label;
				x->parent = a;
				if (minLabel == x->label) break;
			}
		}
		if (x->parent != NULL) {
			x->label = minLabel + (sTree ? 1 : -1);
		} else {
			orphanFree<sTree>(x);
			break;
		}
	}

	// set new parent
	if (x->parent != NULL && !x->isIncremental) ADD_SIBLING(x, x->parent->head);
	if (sTree) {
		if (startLabel != x->label && x->label == topLevelS) activeS1.add(x);
	} else {
		if (startLabel != x->label && x->label == -topLevelT) activeT1.add(x);
	}
}




///////////////////////////////////////////////////
// testing/debugging
///////////////////////////////////////////////////
void IBFSGraph::testTree()
{
	Node *x, *y;
	Arc *a;
	double totalExcess=0;

	for (x=nodes; x != nodeEnd; x++) {
		if (x->label > topLevelS || x->label < -topLevelT) {
			IBDEBUG("ILLEGAL LABEL!");
			testExit();
		}
		if (x->label == 0) {
			if (x->excess) {
				IBDEBUG("EXCESS OUTSIDE!");
				testExit();
			}
			continue;
		}
		bool sTree = (x->label > 0);
		if (sTree ? (x->excess < 0) : (x->excess > 0)) {
			IBDEBUG("EXCESS ON WRONG SIDE!");
			testExit();
		}
		if (sTree && x->excess > 0) totalExcess += x->excess;
		if (!x->excess && x->parent == NULL) {
			IBDEBUG("NO PARENT!");
			testExit();
		}
		if (!x->excess && x->parent->head->label != x->label + (sTree ? -1 : 1)) {
			IBDEBUG("ILLEGAL PARENT!");
			testExit();
		}
		if (x->label == (sTree ? topLevelS : -topLevelT)) {
			int k=0;
			for (; k < (sTree ? activeS1 : activeT1).len; k++) {
				if ((sTree ? activeS1 : activeT1).list[k] == x) break;
			}
			if (k == (sTree ? activeS1 : activeT1).len && incIteration == 1) {
				IBDEBUG("NOT ACTIVE!");
				testExit();
			}
			continue;
		}
		for (y=x->firstSon; y != NULL; y=y->nextPtr) {
			if (y->parent->head != x) {
				IBDEBUG("ILLEGAL SIBLING!");
				testExit();
			}
		}
		for (a=x->firstArc; a != (x+1)->firstArc; a++) {
			if (x->isParentCurr &&
					(sTree ? a->isRevResidual : a->rCap) &&
					(sTree ? (a->head->label > 0) : (a->head->label < 0)) &&
					a->head->label == (sTree ? (x->label-1) : (x->label+1)) &&
					a < x->parent) {
				IBDEBUG("ILLEGAL CURRENT ARC!");
				testExit();
			}
			if (!(sTree ? a->rCap : a->isRevResidual)) continue;
			if (a->head->label > topLevelS || a->head->label < -topLevelT) {
					IBDEBUG("ILLEGAL LABEL!");
					testExit();
			}
			if (a->head->label == 0 || (a->head->parent == NULL && a->head->excess == 0)) {
				IBDEBUG("CROSS OUT NODE!");
				testExit();
			}
			if (sTree ? (a->head->label < 0) : (a->head->label > 0)) {
				IBDEBUG("CROSS NODE!");
				testExit();
			}
			if (sTree ? (a->head->label > (x->label+1)) : (a->head->label < (x->label-1))) {
				IBDEBUG("EXTENDED ARC!");
				testExit();
			}
		}
	}

	if ((int)(testExcess - totalExcess) != (flow - testFlow)) {
//		IBDEBUG("ILLEGAL FLOW!");
//		testExit();
	}
}

void IBFSGraph::testPrint()
{
	int *nums = new int[numNodes];
	memset(nums, 0, sizeof(int)*numNodes);
	for (Node *x=nodes; x != nodeEnd; x++)
	{
		if (x->label >= 0) {
			nums[x->label]++;
		} else {
			nums[numNodes+x->label]++;
		}
	}
	fprintf(stdout, "S = ");
	for (int i=1; i<=topLevelS; i++) {
		fprintf(stdout, "%d ", nums[i]);
	}
	fprintf(stdout, "\nT = ");
	for (int i=1; i<=topLevelT; i++) {
		fprintf(stdout, "%d ", nums[numNodes-i]);
	}
	delete []nums;
	fprintf(stdout,"\n");
	fflush(stdout);
}




///////////////////////////////////////////////////
// push relabel implementation
///////////////////////////////////////////////////
void IBFSGraph::pushRelabelShelve(int fromLevel)
{
	Node *x = NULL;
	for (int bucket=fromLevel; bucket <= prNodeBuckets.maxBucket; bucket++) {
		if (prNodeBuckets.isEmpty(bucket)) continue;
//		if (x == NULL) prNodeShelves.add(prNodeBuckets.buckets[bucket]);
//		else x->nextPtr = prNodeBuckets.buckets[bucket];
//		for (x=prNodeBuckets.buckets[bucket]; x->nextPtr != NULL; x = x->nextPtr) {
//			x->label = -(x->label-fromLevel);
//		}
//		x->label = -(x->label-fromLevel);
		for (x=prNodeBuckets.buckets[bucket]; x != NULL; x = x->nextPtr) x->label = 0;
	}
	int numLevels = prNodeBuckets.maxBucket - fromLevel + 1;
	memset(prNodeBuckets.buckets + fromLevel, 0, sizeof(Node*)*numLevels);
	memset(excessBuckets.buckets + fromLevel, 0, sizeof(Node*)*numLevels);
	prNodeBuckets.maxBucket = fromLevel-1;
	excessBuckets.maxBucket = fromLevel-1;
}

//template <bool sTree> void IBFSGraph::prUnshelve(int fromLevel)
//{
//	Node *next;
//	for (Node *x = prNodeShelves.pop(); x != NULL; x=next) {
//		next = x->nextPtr;
//		x->label = fromLevel - x->label;
//		prNodeBuckets.add<sTree>(x);
//		if (x->excess) excessBuckets.add<sTree>(x);
//	}
//}

void IBFSGraph::pushRelabel()
{
	return pushRelabelDir<false>();
}

template<bool sTree> void IBFSGraph::pushRelabelDir()
{
	Node *x;
	int level;

	// init
	topLevelS = topLevelT = numNodes;
	pushRelabelGlobalUpdate<sTree>();

	// main loop
	int nDischarges = 0;
	for (; excessBuckets.maxBucket >= excessBuckets.minBucket; excessBuckets.maxBucket--)
	while ((x=excessBuckets.popFront(excessBuckets.maxBucket)) != NULL)
	{
		// discharge
		level = excessBuckets.maxBucket; // excessBuckets.maxBucket may change in discharge()
		pushRelabelDischarge<sTree>(x);
		nDischarges++;
		if (prNodeBuckets.maxBucket < level) {
			excessBuckets.allocate(level+2);
			prNodeBuckets.allocate(level+2);
		}

		// global update / gap heuristic
		if (nDischarges % (30*numNodes) == 0) pushRelabelGlobalUpdate<sTree>();
		else if (prNodeBuckets.isEmpty(level)) pushRelabelShelve(level+1);
	}
}

template<bool sTree> void IBFSGraph::pushRelabelGlobalUpdate()
{
	Node *x, *y;
	Arc *a, *aEnd;

	memset(prNodeBuckets.buckets, 0, sizeof(Node*)*(prNodeBuckets.allocLevels+1));
	memset(excessBuckets.buckets, 0, sizeof(Node*)*(excessBuckets.allocLevels+1));
	prNodeBuckets.maxBucket = 1;
	excessBuckets.reset();
	for (x=nodes; x != nodeEnd; x++) {
		x->parent = NULL;
		x->isParentCurr = 0;
		if ((sTree ? (x->excess > 0) : (x->excess < 0))) {
			x->label = (sTree ? 1 : -1);
			prNodeBuckets.add<sTree>(x);
		} else x->label = 0;
	}
	for (int bucket=1; bucket <= prNodeBuckets.maxBucket; bucket++)
	for (x=prNodeBuckets.buckets[bucket]; x != NULL; x = x->nextPtr) {
		aEnd = (x+1)->firstArc;
		for (a=x->firstArc; a != aEnd; a++) {
			if (!(sTree ? a->rCap : a->isRevResidual)) continue;
			y = a->head;
			if (y->parent != NULL || (sTree ? (y->excess > 0) : (y->excess < 0))) continue;
			y->label = (sTree ? (bucket+1) : (-bucket-1));
			prNodeBuckets.add<sTree>(y);
			y->parent = a->rev;
			if (y->excess) excessBuckets.add<sTree>(y);
		}
	}
}

template<bool sTree> void IBFSGraph::pushRelabelDischarge(Node *x)
{
	Node *y;
	int minLabel, push;
	Arc *aEnd = (x+1)->firstArc;
	Arc *a;

	testNode(x);
	prNodeBuckets.remove<sTree>(x);
	while (true)
	{
		// push
		if (x->isParentCurr) {
			a = x->parent;
		} else {
			a = x->firstArc;
			x->isParentCurr = 1;
		}
		if (x->label != (sTree ? 1 : -1))
		{
			minLabel = x->label - (sTree ? 1 : -1);
			for (; a != aEnd; a++)
			{
				// check admissible
				y = a->head;
				if ((sTree ? a->isRevResidual : a->rCap) == 0 || y->label != minLabel) {
					continue;
				}

				// push admissible
				push = (sTree ? (a->rev->rCap) : (a->rCap));
				if (push > (sTree ? (-x->excess) : (x->excess))) {
					push = (sTree ? (-x->excess) : (x->excess));
				}
				x->excess += (sTree ? push : (-push));
				if (sTree) {
					a->rev->rCap -= push;
					a->rCap += push;
					a->rev->isRevResidual = 1;
					a->isRevResidual = (a->rev->rCap ? 1 : 0);
				} else {
					a->rCap -= push;
					a->rev->rCap += push;
					a->rev->isRevResidual = (a->rCap ? 1 : 0);
					a->isRevResidual = 1;
				}

				// add excess
				if (sTree && y->excess > 0) {
					if (y->excess >= push) flow += push;
					else flow += y->excess;
				} else if (!sTree && y->excess < 0) {
					if (-y->excess >= push) flow += push;
					else flow -= y->excess;
				}
				y->excess += (sTree ? (-push) : push);
				if (sTree ? (y->excess < 0 && y->excess >= -push) : (y->excess > 0 && y->excess <= push)) {
					excessBuckets.add<sTree>(y);
				}
				if (x->excess == 0) {
					x->parent = a;
					break;
				}
			}
		}
		if (x->excess == 0) break;

		// relabel
		minLabel = (sTree ? (numNodes-1) : (-numNodes+1));
		x->parent = NULL;
		for (a=x->firstArc; a != aEnd; a++)
		{
			y = a->head;
			if ((sTree ? a->isRevResidual : a->rCap) &&
				// y->label != 0 ---> holds implicitly
				(sTree ? (y->label > 0) : (y->label < 0)) &&
				(sTree ? (y->label < minLabel) : (y->label > minLabel)))
			{
				minLabel = y->label;
				x->parent = a;
				if (minLabel == x->label) break;
			}
		}
		if (x->parent != NULL) {
			x->label = minLabel + (sTree ? 1 : -1);
		} else {
			x->label = 0;
			break;
		}
	}
	if (x->label != 0) prNodeBuckets.add<sTree>(x);
}



///////////////////////////////////////////////////
// file reading
///////////////////////////////////////////////////
bool IBFSGraph::readFromFile(char *filename)
{
	return readFromFile(filename, false);
}
bool IBFSGraph::readFromFileCompile(char *filename)
{
	return readFromFile(filename, true);
}
bool IBFSGraph::readFromFile(char *filename, bool checkCompile)
{
	const int MAX_LINE_LEN = 100;
	char line[MAX_LINE_LEN];
	int declaredNumOfNodes, declaredNumOfEdges, nodeId1, nodeId2;
	int currentNumOfEdges = 0;
	char c, c1, c2, c3;
	int capacity, capacity2;
	int numLines=0;
	// only for compile mode
	const int bufferSize = sizeof(char) + sizeof(int)*4;
	char buffer[bufferSize];

	char *filenameCompiled = new char[strlen(filename) + strlen(".compiled") + 1];
	strcpy(filenameCompiled, filename);
	strcat(filenameCompiled, ".compiled");

	FILE *pFile;
	FILE *pFileCompiled = NULL;
	if (checkCompile) {
		if (fileIsCompiled) {
			delete []filenameCompiled;
			return readCompiled(file);
		} else if (file == NULL && (pFileCompiled = fopen(filenameCompiled, "rb")) != NULL) {
			delete []filenameCompiled;
			return readCompiled(pFileCompiled);
		}
	}
	if (file == NULL) {
		if ((pFile = fopen(filename, "r")) == NULL) {
			fprintf(stdout, "Could not open file %s\n", filename);
			delete []filenameCompiled;
			return false;
		}
		if (checkCompile && (pFileCompiled = fopen(filenameCompiled, "wb")) == NULL) {
			fprintf(stdout, "Could not open file %s\n", filenameCompiled);
			delete []filenameCompiled;
			return false;
		}
	} else {
		pFile = file;
		if (checkCompile) pFileCompiled = fileCompiled;
	}
	delete []filenameCompiled;

	// read from file into temporary structure
	fileHasMore = false;
	while (!fileHasMore && fgets(line, MAX_LINE_LEN, pFile) != NULL)
	{
		numLines++;
		switch (line[0])
	    {
			case 'c':
			case '\n':
			case '\0':
			default:
				break;
			case 'p':
				if (!isInitializedGraph())
				{
					sscanf(line, "%c %c%c%c", &c, &c1, &c2, &c3);
					if (c1=='m' && c2=='a' && c3=='x') {
						sscanf(line, "%c %c%c%c %d %d", &c, &c1, &c2, &c3, &declaredNumOfNodes, &declaredNumOfEdges);
					} else {
						sscanf(line, "%c %d %d", &c, &declaredNumOfNodes, &declaredNumOfEdges);
					}
					initSize(declaredNumOfNodes, declaredNumOfEdges);
					if (checkCompile) {
						fwrite(&declaredNumOfNodes, sizeof(int), 1, pFileCompiled);
						fwrite(&declaredNumOfEdges, sizeof(int), 1, pFileCompiled);
					}
				}
				else
				{
					if (checkCompile) {
						memset(buffer, 0, bufferSize);
						buffer[0] = 'p';
						fwrite(&buffer, 1, bufferSize, pFileCompiled);
					}
					fileHasMore = true;
				}
				break;

			//
			// Read Nodes
			//
			case 'n':
				sscanf(line, "%c %d %d %d ", &c, &nodeId1, &capacity, &capacity2);
				if (capacity == 0 && capacity2 == 0) break;
				if (file == NULL) {
					addNode(nodeId1, capacity, capacity2);
				} else {
					incNode(nodeId1, capacity, capacity2);
				}
				if (checkCompile) {
					buffer[0] = 'n';
					memcpy(buffer+sizeof(char), &nodeId1, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int), &nodeId1, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int), &capacity, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int)+sizeof(int), &capacity2, sizeof(int));
					fwrite(&buffer, 1, bufferSize, pFileCompiled);
				}
				break;

			//
			// Read Arcs
			//
			case 'a':
				sscanf(line, "%c %d %d %d %d", &c,
					&nodeId1, &nodeId2, &capacity, &capacity2);
				if (nodeId1 < 0 ||
					nodeId1 >= (nodeEnd-nodes) ||
					nodeId2 < 0 ||
					nodeId2 >= (nodeEnd-nodes))
				{
					fprintf(stdout, "inconsistent node index %d or %d (Line %d)\n", nodeId1, nodeId2, numLines);
					return false;
				}
				if (file == NULL)
				{
					if (currentNumOfEdges >= declaredNumOfEdges) {
						fprintf(stdout, "inconsistent number of edges (Line %d)\n", numLines);
						return false;
					}
					currentNumOfEdges++;
					addEdge(nodeId1, nodeId2, capacity, capacity2);
				}
				else
				{
					incEdge(nodeId1, nodeId2, capacity, capacity2);
				}
				if (checkCompile) {
					buffer[0] = 'a';
					memcpy(buffer+sizeof(char), &nodeId1, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int), &nodeId2, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int), &capacity, sizeof(int));
					memcpy(buffer+sizeof(char)+sizeof(int)+sizeof(int)+sizeof(int), &capacity2, sizeof(int));
					fwrite(&buffer, 1, bufferSize, pFileCompiled);
				}
				break;
		}
	}
	if (file == NULL && currentNumOfEdges > declaredNumOfEdges) {
		fprintf(stdout, "inconsistent number of edges: differs from declared %d != %d\n",
				currentNumOfEdges, declaredNumOfEdges);
		return false;
	}

	file = pFile;
	fileCompiled = pFileCompiled;
	fileIsCompiled = false;
	if (!fileHasMore) {
		if (checkCompile) {
			buffer[0] = 'x';
			fwrite(&buffer, 1, bufferSize, pFileCompiled);
			fclose(fileCompiled);
			fileCompiled = NULL;
		}
		fclose(file);
		file = NULL;
	}
	return true;
}



bool IBFSGraph::readCompiled(FILE *pFile)
{
	int declaredNumOfNodes, declaredNumOfEdges, nodeId1, nodeId2;
	int capacity, capacity2;
	const int bufferSize = sizeof(char)+sizeof(int)*4;
	char buffer[bufferSize];

	// read from file into htemporary structure
	//fprintf(stdout, "c reading compiled file\n");
	if (!isInitializedGraph()) {
		if (fread(&declaredNumOfNodes, sizeof(int), 1, (pFile)) < 1 ||
				fread(&declaredNumOfEdges, sizeof(int), 1, (pFile)) < 1) {
			fprintf(stdout, "ERROR while reading compiled num nodes/edges, EOF=%d\n", feof(pFile));
			fclose(pFile);
			return false;
		}
		initSize(declaredNumOfNodes, declaredNumOfEdges);
	}
	fileHasMore = false;
	for (int line=0; !fileHasMore && !feof(pFile); line++) {
		if (fread(&buffer, 1, bufferSize, pFile) < bufferSize) {
			fprintf(stdout, "ERROR while reading compiled line %d, EOF=%d\n", line, feof(pFile));
			fclose(pFile);
			return false;
		}
		memcpy(&nodeId1,   buffer+sizeof(char), sizeof(int));
		memcpy(&nodeId2,   buffer+sizeof(char)+sizeof(int), sizeof(int));
		memcpy(&capacity,  buffer+sizeof(char)+sizeof(int)+sizeof(int), sizeof(int));
		memcpy(&capacity2, buffer+sizeof(char)+sizeof(int)+sizeof(int)+sizeof(int), sizeof(int));
		if (buffer[0] == 'x')  break;
		switch(buffer[0])
		{
		case 'n':
			if (capacity == 0 && capacity2 == 0) break;
			if (file == NULL) {
				addNode(nodeId1, capacity, capacity2);
			} else {
				incNode(nodeId1, capacity, capacity2);
			}
			break;
		case 'a':
			if (nodeId1 < 0 ||
				nodeId1 >= (nodeEnd-nodes) ||
				nodeId2 < 0 ||
				nodeId2 >= (nodeEnd-nodes))
			{
				fprintf(stdout, "inconsistent node index in compiled file %d or %d (Line %d)\n", nodeId1, nodeId2, line);
				return false;
			}
			if (file == NULL) {
				addEdge(nodeId1, nodeId2, capacity, capacity2);
			} else {
				incEdge(nodeId1, nodeId2, capacity, capacity2);
			}
			break;
		case 'p':
			fileHasMore = true;
			break;
		}
	}

	file = pFile;
	fileCompiled = NULL;
	fileIsCompiled = true;
	if (!fileHasMore) {
		fclose(file);
		file = NULL;
	}
	return true;
}


