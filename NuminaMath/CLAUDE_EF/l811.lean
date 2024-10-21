import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_properties_l811_81104

/-- Basketball shooting game -/
structure BasketballGame where
  player_a_percentage : ℝ
  player_b_percentage : ℝ
  start_probability : ℝ

/-- The probability that player B takes the second shot -/
noncomputable def prob_b_second_shot (game : BasketballGame) : ℝ :=
  game.start_probability * (1 - game.player_a_percentage) + game.start_probability * game.player_b_percentage

/-- The probability that player A takes the i-th shot -/
noncomputable def prob_a_ith_shot (game : BasketballGame) (i : ℕ) : ℝ :=
  1/3 + (1/6) * (2/5)^(i-1)

/-- The expected number of times player A shoots in the first n shots -/
noncomputable def expected_a_shots (game : BasketballGame) (n : ℕ) : ℝ :=
  (5/18) * (1 - (2/5)^n) + n/3

theorem basketball_game_properties (game : BasketballGame) 
    (h1 : game.player_a_percentage = 0.6)
    (h2 : game.player_b_percentage = 0.8)
    (h3 : game.start_probability = 0.5) :
  prob_b_second_shot game = 0.6 ∧
  (∀ i : ℕ, prob_a_ith_shot game i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, expected_a_shots game n = (5/18) * (1 - (2/5)^n) + n/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_properties_l811_81104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l811_81142

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

/-- Theorem: In a triangle with a = 4, b = 5, and area = 5√3, 
    the length of side c is either √21 or √61 -/
theorem triangle_side_length (t : Triangle) 
    (ha : t.a = 4) 
    (hb : t.b = 5) 
    (harea : area t = 5 * Real.sqrt 3) :
    t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l811_81142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_differences_l811_81171

def school_enrollments : List Nat := [1500, 1650, 2100, 1850, 1400]

theorem enrollment_differences :
  (List.maximum school_enrollments).isSome ∧
  (List.minimum school_enrollments).isSome ∧
  (do
    let max ← List.maximum school_enrollments
    let min ← List.minimum school_enrollments
    pure (max - min)
  ) = some 700 ∧
  (List.minimum (List.filterMap
    (λ pair : Nat × Nat => if pair.1 ≠ pair.2 then some (Int.natAbs (pair.1 - pair.2)) else none)
    (List.join (List.map (λ x => List.map (Prod.mk x) school_enrollments) school_enrollments)))).isSome ∧
  (List.minimum (List.filterMap
    (λ pair : Nat × Nat => if pair.1 ≠ pair.2 then some (Int.natAbs (pair.1 - pair.2)) else none)
    (List.join (List.map (λ x => List.map (Prod.mk x) school_enrollments) school_enrollments)))) = some 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrollment_differences_l811_81171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_group_scores_l811_81128

noncomputable def average_score (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

noncomputable def variance (scores : List ℝ) : ℝ :=
  let μ := average_score scores
  (scores.map (λ x => (x - μ)^2)).sum / scores.length

theorem study_group_scores (x₁ x₂ x₃ : ℝ) :
  let girls_scores := [85, 75]
  let boys_scores := [x₁, x₂, x₃]
  let all_scores := girls_scores ++ boys_scores

  (average_score all_scores = 80) →
  (variance boys_scores = 150) →
  (average_score boys_scores = 80 ∧ variance all_scores = 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_group_scores_l811_81128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_is_2pi_l811_81167

/-- A circle with diameter endpoints on the x-axis and y-axis, tangent to the line x + y - 4 = 0 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  is_valid : center.1 ^ 2 + center.2 ^ 2 = radius ^ 2
  is_tangent : |center.1 + center.2 - 4| / Real.sqrt 2 = radius

/-- The area of a circle -/
noncomputable def circle_area (c : TangentCircle) : ℝ := Real.pi * c.radius ^ 2

/-- The minimum area of a TangentCircle is 2π -/
theorem min_area_is_2pi :
  ∃ (c : TangentCircle), ∀ (c' : TangentCircle), circle_area c ≤ circle_area c' ∧ circle_area c = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_is_2pi_l811_81167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_journey_time_l811_81129

/-- Represents the time taken for a journey on a river -/
structure RiverJourney where
  downstream : ℚ  -- Time taken downstream
  upstream : ℚ    -- Time taken upstream

/-- Calculates the time taken by rafts given a RiverJourney -/
def raftTime (journey : RiverJourney) : ℚ :=
  (journey.downstream * journey.upstream) / (journey.upstream - journey.downstream)

/-- Theorem stating that if a steamboat takes 5 days downstream and 7 days upstream,
    then rafts will take 35 days for the same journey downstream -/
theorem raft_journey_time (journey : RiverJourney) 
        (h1 : journey.downstream = 5)
        (h2 : journey.upstream = 7) : 
        raftTime journey = 35 := by
  sorry

#eval raftTime { downstream := 5, upstream := 7 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raft_journey_time_l811_81129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_120_degrees_l811_81149

noncomputable def angle_between_vectors (e₁ e₂ : ℝ × ℝ) (a b : ℝ × ℝ) : ℝ :=
  let dot_product := (e₁.1 * e₂.1 + e₁.2 * e₂.2)
  let a_vec := (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2)
  let b_vec := (-3 * e₁.1 + 2 * e₂.1, -3 * e₁.2 + 2 * e₂.2)
  let a_dot_b := a_vec.1 * b_vec.1 + a_vec.2 * b_vec.2
  let a_mag := Real.sqrt (a_vec.1 * a_vec.1 + a_vec.2 * a_vec.2)
  let b_mag := Real.sqrt (b_vec.1 * b_vec.1 + b_vec.2 * b_vec.2)
  Real.arccos (a_dot_b / (a_mag * b_mag))

theorem angle_between_vectors_is_120_degrees (e₁ e₂ : ℝ × ℝ) :
  (e₁.1 * e₁.1 + e₁.2 * e₁.2 = 1) →
  (e₂.1 * e₂.1 + e₂.2 * e₂.2 = 1) →
  (e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) →
  angle_between_vectors e₁ e₂ (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2) (-3 * e₁.1 + 2 * e₂.1, -3 * e₁.2 + 2 * e₂.2) = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_120_degrees_l811_81149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_always_holds_l811_81138

theorem inequality_always_holds (a : ℝ) : 
  (∀ x : ℝ, (1/2 : ℝ)^(x^2 - 2*a*x) < (2 : ℝ)^(3*x + a^2)) ↔ a > 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_always_holds_l811_81138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_points_calculation_l811_81185

/-- Calculate points for grocery shopping --/
theorem grocery_points_calculation (beef_price beef_quantity fruits_price fruits_quantity spices_price spices_quantity other_groceries : ℕ) : 
  beef_price = 11 →
  beef_quantity = 3 →
  fruits_price = 4 →
  fruits_quantity = 8 →
  spices_price = 6 →
  spices_quantity = 3 →
  other_groceries = 37 →
  let total_spent := beef_price * beef_quantity + fruits_price * fruits_quantity + spices_price * spices_quantity + other_groceries
  let base_points := (total_spent / 10) * 50
  let bonus_points := if total_spent > 100 then 250 else 0
  base_points + bonus_points = 850 := by
  sorry

#check grocery_points_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_points_calculation_l811_81185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_52_25_l811_81137

/-- Represents the grid and shapes configuration -/
structure GridConfig where
  gridSize : Nat
  smallSquareSize : ℝ
  smallCircleDiameter : ℝ
  largeCircleDiameter : ℝ
  unshadeSquareSide : ℝ
  smallCircleCount : Nat
  largeCircleCount : Nat
  unshadeSquareCount : Nat

/-- Calculates the sum of A, B, and C for the given grid configuration -/
noncomputable def calculateSum (config : GridConfig) : ℝ :=
  let totalArea := (config.gridSize * config.gridSize : ℝ) * config.smallSquareSize^2
  let smallCircleArea := config.smallCircleCount * Real.pi * (config.smallCircleDiameter / 2)^2
  let largeCircleArea := config.largeCircleCount * Real.pi * (config.largeCircleDiameter / 2)^2
  let unshadeSquareArea := config.unshadeSquareCount * config.unshadeSquareSide^2
  let A := totalArea - unshadeSquareArea
  let B := (smallCircleArea + largeCircleArea) / Real.pi
  let C := unshadeSquareArea
  A + B + C

/-- Theorem stating that the sum of A, B, and C equals 52.25 for the given configuration -/
theorem sum_equals_52_25 :
  let config := GridConfig.mk 7 1 1 2 0.5 5 2 4
  calculateSum config = 52.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_52_25_l811_81137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_longer_segment_l811_81141

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a segment MN of length 1, P is the golden section point if MP/NP = φ -/
def isGoldenSectionPoint (MP NP : ℝ) : Prop :=
  MP / NP = φ ∧ MP + NP = 1

theorem golden_section_longer_segment :
  ∀ MP NP : ℝ,
  isGoldenSectionPoint MP NP →
  MP > NP →
  MP = (Real.sqrt 5 - 1) / 2 :=
by
  intros MP NP h1 h2
  sorry

#check golden_section_longer_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_longer_segment_l811_81141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l811_81162

theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) :
  0 < B → B < π →
  a = (Real.sqrt 5 / 2) * b →
  A = 2 * B →
  Real.cos B = Real.sqrt 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l811_81162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l811_81198

theorem tan_alpha_value (α : Real) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) : 
  Real.tan α = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l811_81198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conflict_graph_has_k4_l811_81125

/-- A graph representing conflicts between mafia clans. -/
structure ConflictGraph where
  /-- The set of vertices (clans) in the graph. -/
  vertices : Finset ℕ
  /-- The set of edges (conflicts) in the graph. -/
  edges : Finset (ℕ × ℕ)
  /-- The number of vertices is 20. -/
  vertex_count : vertices.card = 20
  /-- Each vertex has a degree of at least 14. -/
  min_degree : ∀ v, v ∈ vertices → (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 14

/-- A clique is a set of vertices that are all connected to each other. -/
def is_clique (G : ConflictGraph) (clique : Finset ℕ) : Prop :=
  ∀ u v, u ∈ clique → v ∈ clique → u ≠ v → (u, v) ∈ G.edges ∨ (v, u) ∈ G.edges

/-- The main theorem: there always exists a clique of size 4 in the conflict graph. -/
theorem conflict_graph_has_k4 (G : ConflictGraph) : ∃ clique : Finset ℕ, clique.card = 4 ∧ is_clique G clique := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conflict_graph_has_k4_l811_81125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ham_block_cut_masses_l811_81107

/-- Represents a block of ham -/
structure HamBlock where
  length : ℝ
  width : ℝ
  height : ℝ
  mass : ℝ

/-- Represents the parallelogram cross-section after cutting -/
structure ParallelogramCrossSection where
  side1 : ℝ
  side2 : ℝ

/-- Calculates the masses of the two pieces after cutting the ham block -/
noncomputable def calculatePieceMasses (block : HamBlock) (crossSection : ParallelogramCrossSection) : (ℝ × ℝ) :=
  sorry

/-- Theorem stating the masses of the two pieces after cutting the ham block -/
theorem ham_block_cut_masses (block : HamBlock) (crossSection : ParallelogramCrossSection) :
  block.length = 12 ∧ block.width = 12 ∧ block.height = 35 ∧ block.mass = 5 ∧
  crossSection.side1 = 15 ∧ crossSection.side2 = 20 →
  let (mass1, mass2) := calculatePieceMasses block crossSection
  (abs (mass1 - 1.7857) < 0.0001 ∧ abs (mass2 - 3.2143) < 0.0001) ∨
  (abs (mass1 - 3.2143) < 0.0001 ∧ abs (mass2 - 1.7857) < 0.0001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ham_block_cut_masses_l811_81107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l811_81143

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_prop1 : f 1 = 2
axiom f_prop2 : f (-3) = 2

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 2)

-- Theorem to prove
theorem intersection_sum : 
  (intersection_point.fst + intersection_point.snd = 3) ∧ 
  f intersection_point.fst = f (intersection_point.fst - 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l811_81143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_bound_l811_81195

/-- Curve C₂ -/
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

/-- Rectangle vertices -/
noncomputable def A : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def B : ℝ × ℝ := (-Real.sqrt 3, 1)
noncomputable def C : ℝ × ℝ := (-Real.sqrt 3, -1)
noncomputable def D : ℝ × ℝ := (Real.sqrt 3, -1)

/-- Squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of squared distances from a point to rectangle vertices -/
noncomputable def sum_squared_distances (p : ℝ × ℝ) : ℝ :=
  squared_distance p A + squared_distance p B + squared_distance p C + squared_distance p D

theorem sum_squared_distances_bound :
  ∀ θ : ℝ, 20 ≤ sum_squared_distances (C₂ θ) ∧ sum_squared_distances (C₂ θ) ≤ 32 := by
  sorry

#check sum_squared_distances_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_bound_l811_81195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_implies_a_range_l811_81118

noncomputable def f (a x : ℝ) : ℝ := (4 : ℝ)^x + a * (2 : ℝ)^x + 3

theorem two_distinct_roots_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  -4 < a ∧ a < -2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_implies_a_range_l811_81118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periwinkle_position_l811_81115

def periwinkle_sequence : ℕ → ℕ → ℚ × ℚ
  | _, 0 => (0, 1)
  | n, m + 1 => let (x, y) := periwinkle_sequence n m; (2 * y, x + y)

theorem periwinkle_position : 
  periwinkle_sequence 2017 2017 = (2/3 * 2^2016 - 2/3, 1/3 * 2^2017 + 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periwinkle_position_l811_81115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_felicity_gas_usage_l811_81193

-- Define the gas usage for Adhira and Felicity
variable (adhira_gas : ℝ)
variable (felicity_gas : ℝ)

-- State the conditions
axiom felicity_gas_relation : felicity_gas = 4 * adhira_gas - 5
axiom total_gas : felicity_gas + adhira_gas = 30

-- State the theorem to be proved
theorem felicity_gas_usage : felicity_gas = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_felicity_gas_usage_l811_81193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_pension_eligible_l811_81122

/-- Represents a person with an age and eligibility for pension -/
structure Person where
  age : ℕ
  working : Bool
  deriving Repr

/-- Determines if a person is eligible for pension -/
def isPensionEligible (p : Person) : Bool :=
  p.age ≥ 55 && p.working

/-- The name of the additional payment -/
def pensionName : String :=
  "пенсия"

/-- Theorem: Maria Ivanovna is eligible for pension -/
theorem maria_pension_eligible :
    let maria : Person := { age := 55, working := true }
    isPensionEligible maria = true := by
  -- The proof would go here
  sorry

#eval pensionName

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_pension_eligible_l811_81122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_distributions_l811_81103

/-- Represents a card with a number -/
structure Card where
  number : Nat
  h_range : number ≥ 15 ∧ number ≤ 33

/-- Represents a valid distribution of cards -/
structure Distribution where
  vasya : List Card
  petya : List Card
  misha : List Card
  h_all_cards : vasya.length + petya.length + misha.length = 19
  h_non_empty : vasya.length > 0 ∧ petya.length > 0 ∧ misha.length > 0
  h_no_odd_diff : ∀ (l : List Card), l ∈ [vasya, petya, misha] → 
    ∀ (c1 c2 : Card), c1 ∈ l → c2 ∈ l → Even (c1.number - c2.number)

/-- The set of all valid distributions -/
def AllDistributions : Set Distribution :=
  {d : Distribution | True}

/-- Assume the set of all valid distributions is finite -/
instance : Fintype AllDistributions := sorry

/-- The main theorem stating the number of valid distributions -/
theorem num_valid_distributions : Fintype.card AllDistributions = 4596 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_distributions_l811_81103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_problem_l811_81157

/-- The number of chocolate-flavored ice creams initially in the vendor's cart. -/
def C : ℕ := 50

/-- The number of mango-flavored ice creams initially in the vendor's cart. -/
def M : ℕ := 54

/-- The fraction of chocolate-flavored ice creams sold. -/
def chocolate_sold_fraction : ℚ := 3/5

/-- The fraction of mango-flavored ice creams sold. -/
def mango_sold_fraction : ℚ := 2/3

/-- The total number of ice creams not sold. -/
def total_not_sold : ℕ := 38

theorem ice_cream_problem :
  C = 50 ∧
  M = 54 ∧
  (C * (1 - chocolate_sold_fraction) + M * (1 - mango_sold_fraction) : ℚ) = total_not_sold :=
by
  -- Split the goal into three parts
  apply And.intro
  · -- Prove C = 50
    rfl
  apply And.intro
  · -- Prove M = 54
    rfl
  · -- Prove the equation
    sorry -- We'll leave this part as 'sorry' for now

#eval C -- This will output 50
#eval M -- This will output 54

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_problem_l811_81157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l811_81163

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def sum_first_three (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3

def product_first_three (a : ℕ → ℝ) : ℝ :=
  a 1 * a 2 * a 3

def S (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => |a 1|
  | 2 => |a 1| + |a 2|
  | n + 3 => S a (n + 2) + |a (n + 3)|

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : increasing_sequence a) 
  (h3 : sum_first_three a = -3) 
  (h4 : product_first_three a = 8) :
  (∀ n : ℕ, a n = -7 + 3 * (n : ℝ)) ∧
  (∀ n : ℕ, S a n = 
    if n = 1 then 4
    else if n = 2 then 5
    else (3/2) * (n : ℝ)^2 - (11/2) * (n : ℝ) + 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_theorem_l811_81163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l811_81106

/-- A line in 2D space represented by a function f(x, y) = 0 -/
def Line2D (f : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | f p.1 p.2 = 0}

/-- A point in 2D space -/
def Point2D := ℝ × ℝ

/-- Check if a point lies on a line -/
def PointOnLine (p : Point2D) (l : Set (ℝ × ℝ)) : Prop := p ∈ l

/-- Check if two lines are parallel -/
def ParallelLines (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ) (f : ℝ → ℝ → ℝ), k ≠ 0 ∧
    (l1 = Line2D f) ∧ (l2 = Line2D (λ x y => k * f x y))

theorem parallel_line_through_point 
  (f : ℝ → ℝ → ℝ) 
  (p₁ p₂ : Point2D) 
  (h_p1_on_l : PointOnLine p₁ (Line2D f)) 
  (h_p2_off_l : ¬PointOnLine p₂ (Line2D f)) :
  let l := Line2D f
  let l' := Line2D (λ x y => f x y - f p₁.1 p₁.2 - f p₂.1 p₂.2)
  ParallelLines l l' ∧ PointOnLine p₂ l' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_point_l811_81106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_sqrt3_over_6_l811_81178

open MeasureTheory

def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

def region_under_parabola : Set (ℝ × ℝ) :=
  {p | p.1 < p.2^2}

noncomputable def probability_x_less_than_y_squared : ℝ :=
  (volume (rectangle ∩ region_under_parabola)).toReal / (volume rectangle).toReal

theorem probability_is_sqrt3_over_6 :
  probability_x_less_than_y_squared = Real.sqrt 3 / 6 := by
  sorry

#check probability_is_sqrt3_over_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_sqrt3_over_6_l811_81178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_at_three_l811_81191

noncomputable def expression (x : ℝ) : ℝ := 1 / (x + 1 / (x + 1 / (x - 1 / x)))

theorem expression_value_at_three :
  let x : ℝ := 3
  abs (expression x - 0.30337078651685395) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_at_three_l811_81191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_digit_move_l811_81169

theorem no_double_digit_move : ∀ (N : ℕ) (n : ℕ) (a : ℕ) (k : ℕ), 
  (N = a * (10 ^ n) + k) → 
  (1 ≤ a ∧ a ≤ 9) → 
  (k < 10 ^ n) → 
  (10 * k + a ≠ 2 * N) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_digit_move_l811_81169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_area_l811_81155

/-- The curve to which the ellipse is tangent -/
noncomputable def C (x y : ℝ) : ℝ := x^3 - 6*x^2*y + 3*x*y^2 + y^3 + 9*x^2 - 9*x*y + 9*y^2

/-- The ellipse centered at the origin -/
noncomputable def E (x y : ℝ) : ℝ := 4*x^2 - 4*x*y + 4*y^2 - 81

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 81 * Real.pi * Real.sqrt 3 / 4

theorem ellipse_tangent_area : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
  C x₁ y₁ = 0 ∧ C x₂ y₂ = 0 ∧ C x₃ y₃ = 0 ∧
  E x₁ y₁ = 0 ∧ E x₂ y₂ = 0 ∧ E x₃ y₃ = 0 ∧
  (∀ x y, E x y = 0 → C x y = 0 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃)) ∧
  ellipse_area = 81 * Real.pi * Real.sqrt 3 / 4 := by
  sorry

#check ellipse_tangent_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_area_l811_81155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_transformation_l811_81183

/-- Represents the number of "flop" operations needed to transform a deck of n cards from any order to any other order -/
def F (n : ℕ) : ℕ := sorry

/-- A shuffle operation on a deck of cards -/
def shuffle (n : ℕ) (deck : List ℕ) : List ℕ := sorry

theorem deck_transformation (n : ℕ) (initial_deck final_deck : List ℕ) :
  n = 1000 →
  initial_deck.length = n →
  final_deck.length = n →
  ∃ k : ℕ, k ≤ 56 ∧ ∃ shuffled_decks : List (List ℕ),
    shuffled_decks.length = k + 1 ∧
    shuffled_decks.head? = some initial_deck ∧
    shuffled_decks.getLast? = some final_deck ∧
    ∀ i : ℕ, i < k → shuffled_decks[i + 1]? = some (shuffle n shuffled_decks[i]!) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_transformation_l811_81183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_vector_a_length_of_vector_b_l811_81192

-- Define the vectors
def a : Fin 2 → ℝ := ![5, 12]
def b : Fin 2 → ℝ := ![7, -1]

-- Define the length (magnitude) of a vector
noncomputable def vectorLength (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt (v 0 ^ 2 + v 1 ^ 2)

-- Theorem for vector a
theorem length_of_vector_a :
  vectorLength a = 13 := by
  -- Proof goes here
  sorry

-- Theorem for vector b
theorem length_of_vector_b :
  vectorLength b = 5 * Real.sqrt 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_vector_a_length_of_vector_b_l811_81192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l811_81190

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2 / 4) + 2 * x

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | -4 ≤ y ∧ y ≤ Real.sqrt 17} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l811_81190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_problem_l811_81133

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The main theorem -/
theorem fractional_part_problem :
  let a : ℝ := (5 * Real.sqrt 2 + 7) ^ 2017
  a * frac a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_problem_l811_81133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_magic_square_l811_81146

/-- The first nine prime numbers -/
def first_nine_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23]

/-- Definition of a 3x3 magic square -/
def is_magic_square (square : Matrix (Fin 3) (Fin 3) Nat) : Prop :=
  let row_sums := (List.range 3).map (λ i => (List.range 3).foldl (λ acc j => acc + square i j) 0)
  let col_sums := (List.range 3).map (λ j => (List.range 3).foldl (λ acc i => acc + square i j) 0)
  let diag1_sum := (List.range 3).foldl (λ acc i => acc + square i i) 0
  let diag2_sum := (List.range 3).foldl (λ acc i => acc + square i (2 - i)) 0
  let all_sums := row_sums ++ col_sums ++ [diag1_sum, diag2_sum]
  all_sums.all (· = all_sums.head!)

/-- Theorem: It's impossible to create a 3x3 magic square using the first nine prime numbers -/
theorem no_prime_magic_square :
  ¬ ∃ (square : Matrix (Fin 3) (Fin 3) Nat),
    (∀ i j, square i j ∈ first_nine_primes) ∧
    (is_magic_square square) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_magic_square_l811_81146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_sum_l811_81134

def is_valid_arrangement (M : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  ∀ i j, -5 ≤ M i j ∧ M i j ≤ 10

def row_sum (M : Matrix (Fin 4) (Fin 4) ℤ) (i : Fin 4) : ℤ :=
  (Finset.univ.sum fun j => M i j)

def col_sum (M : Matrix (Fin 4) (Fin 4) ℤ) (j : Fin 4) : ℤ :=
  (Finset.univ.sum fun i => M i j)

def main_diag_sum (M : Matrix (Fin 4) (Fin 4) ℤ) : ℤ :=
  (Finset.univ.sum fun i => M i i)

def anti_diag_sum (M : Matrix (Fin 4) (Fin 4) ℤ) : ℤ :=
  (Finset.univ.sum fun i => M i (3 - i))

def all_sums_equal (M : Matrix (Fin 4) (Fin 4) ℤ) (s : ℤ) : Prop :=
  (∀ i, row_sum M i = s) ∧
  (∀ j, col_sum M j = s) ∧
  main_diag_sum M = s ∧
  anti_diag_sum M = s

theorem square_arrangement_sum :
  ∀ M : Matrix (Fin 4) (Fin 4) ℤ,
  is_valid_arrangement M →
  (∃ s, all_sums_equal M s) →
  (∃ s, all_sums_equal M s ∧ s = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_arrangement_sum_l811_81134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l811_81113

-- Define the curves
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, 6 + Real.sqrt 2 * Real.sin θ)

noncomputable def C₂ (φ : ℝ) : ℝ × ℝ := (Real.sqrt 10 * Real.cos φ, Real.sin φ)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_distance_between_curves :
  ∃ (θ φ : ℝ), ∀ (θ' φ' : ℝ),
    distance (C₁ θ) (C₂ φ) ≥ distance (C₁ θ') (C₂ φ') ∧
    distance (C₁ θ) (C₂ φ) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l811_81113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_speed_200m_40s_l811_81108

/-- Calculates the speed of an athlete given distance and time -/
noncomputable def athleteSpeed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: An athlete running 200 meters in 40 seconds has a speed of 5 meters per second -/
theorem athlete_speed_200m_40s :
  athleteSpeed 200 40 = 5 := by
  -- Unfold the definition of athleteSpeed
  unfold athleteSpeed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_speed_200m_40s_l811_81108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_to_cos_l811_81161

theorem sin_translation_to_cos (x : ℝ) : 
  Real.sin (2 * (x - 19 * π / 24) + π / 3) = Real.cos (2 * x + π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_to_cos_l811_81161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l811_81109

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ 0 ∧ x ≠ -2) :
  let f := λ x : ℝ ↦ (3 * x^5 - 12 * x^3 - 7) / (x^2 + 2 * x)
  let F := λ x : ℝ ↦ 3 * x^4 / 4 - 2 * x^3 - 7 / 2 * Real.log (abs x) + 7 / 2 * Real.log (abs (x + 2))
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l811_81109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l811_81127

noncomputable def angle_terminal_point (α : ℝ) : ℝ × ℝ := (-1, 2)

theorem sin_alpha_value (α : ℝ) :
  angle_terminal_point α = (-1, 2) → Real.sin α = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l811_81127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_melt_depth_l811_81114

/-- The height of a cylinder with radius 10 inches, having the same volume as a sphere with radius 3 inches, is 9/25 inches. -/
theorem ice_cream_melt_depth :
  (let sphere_radius : ℝ := 3
   let cylinder_radius : ℝ := 10
   let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
   let cylinder_height := sphere_volume / (Real.pi * cylinder_radius ^ 2)
   cylinder_height) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_melt_depth_l811_81114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l811_81139

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - Real.cos x ^ 2 + 1 / 2

theorem f_properties :
  (∀ x : ℝ, f x = 0 ↔ ∃ k : ℤ, x = k * Real.pi / 2 + Real.pi / 12) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -1 / 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l811_81139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l811_81111

-- Define the sequence a_n
def a : ℕ → ℚ
  | 0 => -1  -- Define for 0 to cover all natural numbers
  | n + 1 => (1 + a n) / (1 - a n)

-- State the theorem
theorem a_2016_equals_negative_one : a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_one_l811_81111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_quartic_equation_l811_81187

theorem solution_to_quartic_equation :
  {z : ℂ | z^4 + 2*z^2 - 8 = 0} = {z : ℂ | z = -Real.sqrt 2 ∨ z = Real.sqrt 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_quartic_equation_l811_81187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_values_l811_81160

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  (∀ a b : ℕ+, Nat.Coprime a.val b.val → f (a * b) = f a * f b) ∧
  (∀ p q : ℕ+, Nat.Prime p.val → Nat.Prime q.val → f (p + q) = f p + f q)

/-- The main theorem -/
theorem special_function_values (f : ℕ+ → ℕ+) (h : special_function f) :
  f 2 = 2 ∧ f 3 = 3 ∧ f 1999 = 1999 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_values_l811_81160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_2a_plus_b_minus_c_l811_81130

theorem square_root_of_2a_plus_b_minus_c (a b c : ℝ) : 
  (5 * a + 2)^(1/3) = 3 →
  ((3 * a + b - 1)^(1/2) = 4 ∨ (3 * a + b - 1)^(1/2) = -4) →
  c = ⌊Real.sqrt 11⌋ →
  ((2 * a + b - c)^(1/2) = 3 ∨ (2 * a + b - c)^(1/2) = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_2a_plus_b_minus_c_l811_81130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_5_t_50_mod_5_actual_l811_81173

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 11  -- Base case for n = 0 (corresponding to t₁)
  | n + 1 => 11^(T n)

-- State the theorem
theorem t_50_mod_5 : T 49 ≡ 1 [MOD 5] := by
  -- Proof steps would go here
  sorry

-- The actual result we want
theorem t_50_mod_5_actual : T 49 ≡ 1 [MOD 5] := by
  -- This is equivalent to t₅₀ ≡ 1 [MOD 5] in the original problem
  exact t_50_mod_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_5_t_50_mod_5_actual_l811_81173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_for_all_cos_positive_even_function_derivative_property_l811_81179

-- Proposition 1
theorem negation_for_all_cos_positive :
  (¬ ∀ x : ℝ, Real.cos x > 0) ↔ (∃ x : ℝ, Real.cos x ≤ 0) :=
sorry

-- Proposition 3
theorem even_function_derivative_property 
  (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) 
  (h_deriv_pos : ∀ x > 0, deriv f x > 0) :
  ∀ x < 0, deriv f x < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_for_all_cos_positive_even_function_derivative_property_l811_81179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwiches_per_customer_factor_of_24_l811_81148

/-- Represents the number of bacon sandwiches ordered by each customer in the group -/
def sandwiches_per_customer : ℕ := sorry

/-- Represents the total number of customers in the group -/
def group_size : ℕ := sorry

/-- The number of local offices -/
def num_offices : ℕ := 3

/-- The number of bacon sandwiches ordered by each office -/
def sandwiches_per_office : ℕ := 10

/-- The total number of bacon sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- Half of the group ordered bacon sandwiches -/
axiom half_group_ordered : group_size / 2 * sandwiches_per_customer = total_sandwiches - num_offices * sandwiches_per_office

theorem sandwiches_per_customer_factor_of_24 : ∃ k : ℕ, sandwiches_per_customer * k = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwiches_per_customer_factor_of_24_l811_81148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_four_l811_81196

theorem exponent_of_four (x y : ℝ) 
  (eq : (5 : ℝ)^(x+1) * (4 : ℝ)^(y-1) = (25 : ℝ)^x * (64 : ℝ)^y) 
  (sum : x + y = 0.5) : 
  y - 1 = -1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_four_l811_81196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_ratio_l811_81153

/-- Given a parabola y² = 2px with p > 0 and focus F at distance 4 from the directrix,
    prove that for points M and N on the parabola satisfying (y₁ - 2y₂)(y₁ + 2y₂) = 48,
    the ratio |MF| / |NF| = 4. -/
theorem parabola_focus_distance_ratio (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  p > 0 →
  y₁^2 = 2*p*x₁ →
  y₂^2 = 2*p*x₂ →
  (y₁ - 2*y₂)*(y₁ + 2*y₂) = 48 →
  p = 4 →
  (x₁ + 2) / (x₂ + 2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_ratio_l811_81153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l811_81184

-- Define the function f(x) as noncomputable due to its dependence on Real.pi
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x)

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f (x - 1) = f (1 - x)) ∧  -- f(x-1) is an even function
  (∀ x, f x ≥ 3) ∧                -- The minimum value of f(x) is 3
  (∀ x, f (x + 2) = f x) :=       -- f(x) has a period of 2
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l811_81184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_inequality_solution_f_odd_l811_81140

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.exp x + 1) + 1

theorem f_monotone_and_inequality_solution :
  (∀ x : ℝ, Monotone f) ∧
  (∀ x : ℝ, f (Real.log 2 * (x^2)) + f (Real.log (Real.sqrt 2) * x - 3) ≤ 0 ↔ 1/8 ≤ x ∧ x ≤ 2) :=
by sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_inequality_solution_f_odd_l811_81140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l811_81154

/-- The trajectory of point Q given the conditions of the problem -/
theorem trajectory_of_Q (P Q : ℝ × ℝ) : 
  (2 * P.1 - P.2 + 3 = 0) →  -- P is on the line 2x - y + 3 = 0
  (∃ k : ℝ, k > 1 ∧ Q = (k * (P.1 + 1) - 1, k * (P.2 - 2) + 2)) →  -- Q is on the extension of PM
  ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (P.1 + 1)^2 + (P.2 - 2)^2) →  -- |PM| = |MQ|
  (2 * Q.1 - Q.2 + 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l811_81154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l811_81197

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x^2 - 2*x - 3)

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f (x + 1) ≥ f (y + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l811_81197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_marked_price_l811_81131

/-- The marked price of a gadget given purchase price, discounts, and desired profit -/
noncomputable def marked_price (purchase_price : ℝ) (initial_discount : ℝ) (desired_profit : ℝ) (final_discount : ℝ) : ℝ :=
  let cost_after_discount := purchase_price * (1 - initial_discount)
  let selling_price := cost_after_discount * (1 + desired_profit)
  selling_price / (1 - final_discount)

/-- Theorem stating the correct marked price for the given conditions -/
theorem correct_marked_price :
  marked_price 36 0.15 0.25 0.1 = 42.5 := by
  -- Unfold the definition of marked_price
  unfold marked_price
  -- Simplify the expression
  simp
  -- Assert the equality (this step would normally be proved, but we use sorry here)
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_marked_price_l811_81131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intercept_l811_81175

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := (x^4 - x^3) / (x - 1)

/-- The derivative of the curve function -/
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_intercept (m : ℝ) :
  f' m = 3 →
  ∃ (b : ℝ), b = -2/3 ∧ f m + 3 * (b - m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intercept_l811_81175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l811_81123

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2) * x^2 - 1

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l811_81123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_in_M_unique_root_bound_difference_l811_81117

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | (∃ x, f x = x) ∧ (∀ x, 0 < deriv f x ∧ deriv f x < 1)}

-- Define the specific function f
noncomputable def f (x : ℝ) : ℝ := x / 2 - (Real.log x) / 2 + 3

-- Theorem I
theorem f_in_M : f ∈ M := by sorry

-- Theorem II
theorem unique_root : ∃! x, f x = x := by sorry

-- Theorem III
theorem bound_difference (f : ℝ → ℝ) (h : f ∈ M) (a b x₁ x₂ x₃ : ℝ) 
  (ha : a < x₁ ∧ x₁ < b) (hb : a < x₂ ∧ x₂ < b) (hc : a < x₃ ∧ x₃ < b)
  (h1 : |x₂ - x₁| < 1) (h2 : |x₃ - x₁| < 1) : 
  |f x₃ - f x₂| < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_in_M_unique_root_bound_difference_l811_81117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_problem_l811_81110

-- Define the positive number x and its square roots
noncomputable def x : ℝ := sorry
def root1 (a : ℝ) : ℝ := 2*a - 3
def root2 (a : ℝ) : ℝ := 5 - a

-- State the theorem
theorem square_roots_problem (h1 : x > 0) (h2 : ∃ a : ℝ, root1 a = Real.sqrt x ∧ root2 a = Real.sqrt x) :
  (∃ a : ℝ, a = -2 ∧ x = 49) ∧ 
  (∃ y : ℝ, y^2 = x + 12*(-2) ∧ (y = 5 ∨ y = -5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_problem_l811_81110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_positivity_l811_81105

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x + 1) - a * Real.log x + a

-- State the theorem
theorem tangent_line_and_positivity (a : ℝ) (h : a > 0) :
  -- Part 1: Equation of tangent line when a = 1
  ((Real.exp 2 - 1) * 1 - f 1 1 - 2 = 0) ∧
  -- Part 2: Condition for f(x) > 0
  (∀ x > 0, f a x > 0) ↔ (0 < a ∧ a < Real.exp 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_positivity_l811_81105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_B_reaches_B_after_C_reaches_A_l811_81120

/-- Represents a robot on a circular track -/
structure Robot where
  position : ℝ  -- Position on the track (0 ≤ position < 2π)
  speed : ℝ     -- Angular speed (radians per second)

/-- The circular track setup -/
structure CircularTrack where
  robotA : Robot
  robotB : Robot
  robotC : Robot

/-- The initial setup of the robots on the track -/
noncomputable def initialSetup : CircularTrack where
  robotA := { position := 0, speed := Real.pi / 12 }
  robotB := { position := 0, speed := -Real.pi / 84 }
  robotC := { position := Real.pi, speed := Real.pi / 28 }

/-- Time when robot A first reaches point B -/
def timeAReachesB : ℝ := 12

/-- Time when robot A catches up with C and meets B -/
def timeMeeting : ℝ := 21

/-- Time for robot C to complete one revolution -/
def timeCRevolution : ℝ := 7

/-- Theorem stating the time for robot B to reach point B after C reaches A -/
theorem time_B_reaches_B_after_C_reaches_A : 
  ∀ (track : CircularTrack),
  track = initialSetup →
  (timeCRevolution * 8 : ℝ) = 56 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_B_reaches_B_after_C_reaches_A_l811_81120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_is_correct_l811_81119

noncomputable def box_length : ℝ := 20
noncomputable def box_width : ℝ := 14
noncomputable def box_height : ℝ := 10
noncomputable def cube_side : ℝ := 4

noncomputable def box_volume : ℝ := box_length * box_width * box_height
noncomputable def removed_volume : ℝ := 8 * (cube_side ^ 3)

noncomputable def volume_removed_percentage : ℝ := (removed_volume / box_volume) * 100

theorem volume_removed_percentage_is_correct :
  ∃ ε > 0, |volume_removed_percentage - 18.29| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_is_correct_l811_81119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_eating_nuts_l811_81152

theorem crow_eating_nuts (total_nuts : ℝ) (h : total_nuts > 0) :
  (1/4 : ℝ) * total_nuts / 10 * 8 = (1/5 : ℝ) * total_nuts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_eating_nuts_l811_81152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_inequality_l811_81181

theorem max_sum_inequality (O sq : ℕ) (hO : O > 0) (hsq : sq > 0)
  (h1 : (O : ℚ) / 11 < 7 / sq) (h2 : (7 : ℚ) / sq < 4 / 5) :
  O + sq ≤ 18 ∧ ∃ (O' sq' : ℕ), O' > 0 ∧ sq' > 0 ∧
    (O' : ℚ) / 11 < 7 / sq' ∧ (7 : ℚ) / sq' < 4 / 5 ∧ O' + sq' = 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_inequality_l811_81181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l811_81189

/-- The area of a triangle formed by two vectors in 2D space -/
noncomputable def triangleArea (a b : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((a.1 * b.2) - (a.2 * b.1))

/-- Theorem: The area of the triangle formed by (0, 0), (3, 2), and (1, 5) is 6.5 -/
theorem triangle_area_example : triangleArea (3, 2) (1, 5) = 6.5 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l811_81189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l811_81116

-- Define the function f and its derivative f'
def f : ℝ → ℝ := sorry
def f' : ℝ → ℝ := sorry

-- Define the conditions
axiom f'_odd : ∀ x, f' (-x) = -f' x
axiom f_neg_two_zero : f (-2) = 0
axiom f_plus_f'_pos : ∀ x > 0, f x + (x / 3) * f' x > 0

-- Define the set of x values where f(x) > 0
def f_positive_set : Set ℝ := {x | f x > 0}

-- State the theorem
theorem f_positive_range : f_positive_set = Set.Ioo (-2) 0 ∪ Set.Ioi 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_l811_81116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l811_81101

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x - Real.pi / 4) + 4 * Real.cos (3 * x)

theorem phase_shift_of_f :
  ∃ (shift : ℝ), shift = Real.pi / 12 ∧
  ∀ (x : ℝ), f (x + shift) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l811_81101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_length_l811_81126

/-- A circle passing through three given points intersects the y-axis, forming a segment of length 4√6 -/
theorem circle_y_axis_intersection_length : 
  ∀ (circle : Set (ℝ × ℝ)) 
    (A B C M N : ℝ × ℝ),
    A = (1, 3) →
    B = (4, 2) →
    C = (1, -7) →
    A ∈ circle →
    B ∈ circle →
    C ∈ circle →
    M ∈ circle →
    N ∈ circle →
    (M.1 = 0 ∧ N.1 = 0) →
    dist M N = 4 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_y_axis_intersection_length_l811_81126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l811_81135

/-- The value of k for which the line y = kx + 2 and the curve x^2/2 + y^2 = 1 have exactly one common point -/
theorem line_curve_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.1^2 / 2 + p.2^2 = 1) ↔ k = Real.sqrt 6 / 2 ∨ k = -Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l811_81135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l811_81102

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1) + 1 / x

-- State the theorem
theorem tangent_line_and_inequality (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  -- Part 1: Tangent line equation
  (∃ (y : ℝ), y = f 1 ∧ y - f 1 = -(1/2) * (x - 1)) ∧
  -- Part 2: Inequality and range of a
  (∀ (a : ℝ), f x > (Real.log x) / (x - 1) + (a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l811_81102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_five_l811_81170

/-- Represents the properties of a river --/
structure River where
  width : ℝ
  flowRate : ℝ
  volumePerMinute : ℝ

/-- Calculates the depth of a river given its properties --/
noncomputable def riverDepth (r : River) : ℝ :=
  r.volumePerMinute / (r.width * (r.flowRate * 1000 / 60))

/-- Theorem stating that a river with given properties has a depth of 5 meters --/
theorem river_depth_is_five :
  let r : River := {
    width := 19,
    flowRate := 4,
    volumePerMinute := 6333.333333333333
  }
  riverDepth r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_depth_is_five_l811_81170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l811_81194

/-- Represents a student in the line -/
inductive Student
| BoyA
| BoyB
| Girl1
| Girl2
| Girl3
deriving BEq, Repr

/-- A valid arrangement of students -/
def Arrangement := List Student

/-- Checks if two girls are adjacent in the arrangement -/
def two_girls_adjacent (arr : Arrangement) : Bool :=
  sorry

/-- Checks if BoyA is not at either end of the arrangement -/
def boyA_not_at_ends (arr : Arrangement) : Bool :=
  sorry

/-- Checks if the arrangement is valid according to the problem conditions -/
def is_valid_arrangement (arr : Arrangement) : Bool :=
  arr.length = 5 ∧
  arr.count Student.BoyA = 1 ∧
  arr.count Student.BoyB = 1 ∧
  arr.count Student.Girl1 + arr.count Student.Girl2 + arr.count Student.Girl3 = 3 ∧
  two_girls_adjacent arr ∧
  boyA_not_at_ends arr

/-- The set of all valid arrangements -/
def valid_arrangements : Set Arrangement :=
  {arr | is_valid_arrangement arr}

theorem valid_arrangements_count :
  (List.filter is_valid_arrangement (List.permutations [Student.BoyA, Student.BoyB, Student.Girl1, Student.Girl2, Student.Girl3])).length = 48 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l811_81194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l811_81168

/-- Parabola intersection with line passing through focus -/
theorem parabola_line_intersection (p : ℝ) (h_p : p > 0) :
  ∃ (A B : ℝ × ℝ),
    (let x1 := A.1
     let y1 := A.2
     let x2 := B.1
     let y2 := B.2
     -- Parabola equation
     x1^2 = 2*p*y1 ∧ x2^2 = 2*p*y2 ∧
     -- Line equation
     y1 = x1 + p/2 ∧ y2 = x2 + p/2 ∧
     -- A and B are distinct points
     x1 < x2 ∧
     -- Area of trapezoid ABCD
     (1/2 : ℝ) * (y1 + y2) * (x2 - x1) = 12 * Real.sqrt 2) →
    p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l811_81168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_from_origin_l811_81165

noncomputable def dog_post : ℝ × ℝ := (6, 8)
def rope_length : ℝ := 15
def wall_y : ℝ := 5
def wall_x_min : ℝ := 0
def wall_x_max : ℝ := 8

noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2))

def is_within_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 ≤ radius ^ 2

theorem greatest_distance_from_origin :
  ∃ (max_dist : ℝ),
    max_dist = Real.sqrt 565 ∧
    ∀ (p : ℝ × ℝ),
      is_within_circle dog_post rope_length p →
      p.2 ≥ wall_y →
      wall_x_min ≤ p.1 ∧ p.1 ≤ wall_x_max →
      distance_from_origin p ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_from_origin_l811_81165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l811_81159

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

-- Define the function g(x) = (f(x) + 1) / x
noncomputable def g (x : ℝ) : ℝ := (f x + 1) / x

-- State the theorem
theorem max_k_value (k : ℤ) : 
  (∀ x : ℝ, x > 1 → (k : ℝ) < g x) ↔ k ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l811_81159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_theorem_l811_81136

/-- Calculates the profit percentage without discount given the profit percentage with discount and the discount rate. -/
noncomputable def profit_without_discount (profit_with_discount : ℝ) (discount_rate : ℝ) : ℝ :=
  (1 + profit_with_discount) / (1 - discount_rate) - 1

/-- Theorem: If a shopkeeper earns a 32% profit after offering a 4% discount,
    then the profit percentage without discount would be 37.5%. -/
theorem shopkeeper_profit_theorem :
  profit_without_discount 0.32 0.04 = 0.375 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval profit_without_discount 0.32 0.04

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_theorem_l811_81136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_on_parabola_l811_81156

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  focus : ℝ × ℝ

/-- Point on the parabola -/
structure ParabolaPoint where
  coords : ℝ × ℝ

/-- Quadrilateral ABCD on the parabola -/
structure Quadrilateral where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  D : ParabolaPoint

/-- Helper function to calculate quadrilateral area -/
noncomputable def quadrilateral_area (q : Quadrilateral) : ℝ := sorry

/-- The theorem statement -/
theorem min_area_quadrilateral_on_parabola 
  (p : Parabola) 
  (q : Quadrilateral) 
  (h1 : q.A.coords ≠ (0, 0))
  (h2 : q.B.coords ≠ (0, 0))
  (h3 : (q.A.coords.1 - p.focus.1) * (q.B.coords.1 - p.focus.1) + 
        (q.A.coords.2 - p.focus.2) * (q.B.coords.2 - p.focus.2) = 0)
  (h4 : ∃ t : ℝ, (q.A.coords.1 + t * (q.C.coords.1 - q.A.coords.1),
                  q.A.coords.2 + t * (q.C.coords.2 - q.A.coords.2)) = p.focus)
  (h5 : ∃ t : ℝ, (q.B.coords.1 + t * (q.D.coords.1 - q.B.coords.1),
                  q.B.coords.2 + t * (q.D.coords.2 - q.B.coords.2)) = p.focus)
  : 
  ∃ (area : ℝ), area ≥ 32 ∧ 
  (∀ (other_area : ℝ), quadrilateral_area q = other_area → area ≤ other_area) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_on_parabola_l811_81156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_A_l811_81177

/-- The set A defined as {x | 1 ≤ x ≤ 5/2} -/
noncomputable def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5/2}

/-- Function f(x) = x^2 + px + q -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- Function g(x) = x + 4/x -/
noncomputable def g (x : ℝ) : ℝ := x + 4/x

/-- The theorem stating the maximum value of f(x) on set A -/
theorem max_value_f_on_A (p q : ℝ) :
  (∃ x₀ ∈ A, ∀ x ∈ A, f p q x ≥ f p q x₀ ∧ g x ≥ g x₀ ∧ f p q x₀ = g x₀) →
  (∃ x ∈ A, ∀ y ∈ A, f p q x ≥ f p q y) →
  (∃ x ∈ A, f p q x = 5) ∧ ∀ y ∈ A, f p q y ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_A_l811_81177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_increasing_on_positive_reals_l811_81166

-- Define the function f(x) = log₂x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_increasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_increasing_on_positive_reals_l811_81166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_columns_first_row_column_increment_ninety_nine_in_fourth_column_l811_81186

/-- Represents the column number (1-indexed) for a given odd number in the arrangement. -/
def columnNumber (n : Nat) : Nat :=
  ((n - 1) % 10 + 1 + 4) % 5 + 1

/-- The arrangement has 5 columns. -/
theorem num_columns (n : Nat) : columnNumber n ≤ 5 := by
  sorry

/-- The first row contains numbers 1, 3, 5, 7, 9. -/
theorem first_row : 
  columnNumber 1 = 1 ∧ 
  columnNumber 3 = 2 ∧ 
  columnNumber 5 = 3 ∧ 
  columnNumber 7 = 4 ∧ 
  columnNumber 9 = 5 := by
  sorry

/-- Numbers in each column increase by 10 when moving down. -/
theorem column_increment (n : Nat) : columnNumber (n + 10) = columnNumber n := by
  sorry

theorem ninety_nine_in_fourth_column : columnNumber 99 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_columns_first_row_column_increment_ninety_nine_in_fourth_column_l811_81186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_implies_a_geq_neg_inv_e_l811_81158

open Real Set

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * exp x
def g (x a : ℝ) : ℝ := -(x + 1)^2 + a

-- State the theorem
theorem f_leq_g_implies_a_geq_neg_inv_e (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Icc (-2) 0 ∧ x₂ ∈ Icc (-2) 0 ∧ f x₂ ≤ g x₁ a) →
  a ≥ -exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_implies_a_geq_neg_inv_e_l811_81158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_of_f_l811_81176

noncomputable def f (x : ℝ) := (1/3) * x - Real.log x

theorem zero_points_of_f :
  (∀ x, x ∈ Set.Ioo (1/Real.exp 1) 1 → f x ≠ 0) ∧
  (∃! x, x ∈ Set.Ioo 1 (Real.exp 1) ∧ f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_of_f_l811_81176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_after_10_years_time_to_reach_1_2_million_l811_81147

-- Define the initial population and growth rate
def initial_population : ℝ := 1
def growth_rate : ℝ := 0.012

-- Define the population function
noncomputable def population (x : ℝ) : ℝ := initial_population * (1 + growth_rate) ^ x

-- Theorem for the population after 10 years
theorem population_after_10_years :
  ∃ ε > 0, |population 10 - 1.127| < ε := by
  sorry

-- Theorem for the time to reach 1.2 million
theorem time_to_reach_1_2_million :
  ∃ t ∈ Set.Icc 15 16, population t = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_after_10_years_time_to_reach_1_2_million_l811_81147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_g_less_than_f_condition_l811_81180

open Real

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Part 1
theorem f_extrema_on_interval :
  let a : ℝ := -1/2
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≤ Real.log 2 - 1) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = Real.log 2 - 1) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = -1/2) :=
by sorry

-- Part 2
theorem g_less_than_f_condition (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (-1) 2, ∃ x₂ > 0, g x₁ < f a x₂) ↔
  a ∈ Set.Ioi (-Real.exp (-6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_g_less_than_f_condition_l811_81180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_inequality_l811_81144

theorem exponent_inequality (m n : ℝ) : (5 : ℝ)^m > (5 : ℝ)^n → m > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_inequality_l811_81144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_minus_2cos_squared_theta_l811_81150

theorem sin_2theta_minus_2cos_squared_theta (θ : ℝ) (h : Real.tan θ = 1/2) :
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_minus_2cos_squared_theta_l811_81150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l811_81124

/-- A function that represents sin(ωx + φ) - cos(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

/-- The number of extreme points of a function in an interval -/
noncomputable def num_extreme_points (g : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

/-- The number of zero points of a function in an interval -/
noncomputable def num_zero_points (g : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem omega_range (ω φ : ℝ) :
  ω > 0 →
  |φ| < π / 2 →
  is_odd_function (f ω φ) →
  num_extreme_points (f ω φ) 0 (2 * π) = 2 →
  num_zero_points (f ω φ) 0 (2 * π) = 1 →
  3 / 4 < ω ∧ ω ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l811_81124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_depth_calculation_l811_81132

/-- Represents a rectangular tank with given dimensions -/
structure Tank where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of the tank to be plastered -/
def surface_area (t : Tank) : ℝ :=
  t.length * t.width + 2 * (t.length * t.depth + t.width * t.depth)

/-- Theorem stating the depth of the tank given the conditions -/
theorem tank_depth_calculation (t : Tank) 
  (h1 : t.length = 25)
  (h2 : t.width = 12)
  (h3 : surface_area t * 0.55 = 409.20) :
  ∃ ε > 0, |t.depth - 6| < ε := by
  sorry

#check tank_depth_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_depth_calculation_l811_81132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l811_81199

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (Real.log x) / x - k / x

theorem function_properties (k : ℝ) :
  (∀ x ≥ 1, x^2 * f x k + 1 / (x + 1) ≥ 0) →
  (∀ x ≥ 1, k ≥ 1/2 * x^2 + (Real.exp 2 - 2) * x - Real.exp x - 7) →
  ((deriv (f · k)) 1 = 10) →
  (∃ x > 0, ∀ y > 0, f y k ≤ f x k) ∧
  (f (Real.exp 10) k = 1 / Real.exp 10) ∧
  (Real.exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l811_81199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_system_l811_81172

theorem unique_solution_system :
  ∃! p : ℝ × ℝ, 
    let (x, y) := p
    Real.sqrt (x^2 + y^2) + Real.sqrt ((x - 4)^2 + (y - 3)^2) = 5 ∧
    3 * x^2 + 4 * x * y = 24 ∧
    x = 2 ∧ y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_system_l811_81172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travelers_checks_worth_l811_81121

/-- Represents the denominations of travelers checks -/
inductive Denomination
  | fifty : Denomination
  | hundred : Denomination

/-- Calculates the value of a check given its denomination -/
def checkValue (d : Denomination) : ℕ :=
  match d with
  | Denomination.fifty => 50
  | Denomination.hundred => 100

/-- Represents a set of travelers checks -/
structure TravelersChecks where
  fifties : ℕ
  hundreds : ℕ
  total_count : ℕ
  h_total : total_count = fifties + hundreds
  h_30 : total_count = 30

/-- Calculates the total worth of a set of travelers checks -/
def totalWorth (checks : TravelersChecks) : ℕ :=
  checks.fifties * checkValue Denomination.fifty +
  checks.hundreds * checkValue Denomination.hundred

/-- Theorem stating the total worth of travelers checks -/
theorem travelers_checks_worth
  (checks : TravelersChecks)
  (h_spend : checks.fifties ≥ 24)
  (h_avg : (totalWorth checks - 24 * checkValue Denomination.fifty) / (checks.total_count - 24) = 100) :
  totalWorth checks = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travelers_checks_worth_l811_81121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_product_ring_properties_l811_81151

/-- A ring where every element is a product of two idempotent elements -/
class IdempotentProductRing (R : Type) extends CommRing R where
  idempotent_product : ∀ (x : R), ∃ (a b : R), a * a = a ∧ b * b = b ∧ x = a * b

/-- Theorem stating that in an IdempotentProductRing, 1 is the only unit and the ring is Boolean -/
theorem idempotent_product_ring_properties (R : Type) [IdempotentProductRing R] : 
  (∀ (x : R), IsUnit x → x = 1) ∧ 
  (∀ (x : R), x * x = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_product_ring_properties_l811_81151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABCD_l811_81164

/-- The parabola y^2 = 4x --/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola --/
def focus : ℝ × ℝ := (1, 0)

/-- A point on the parabola --/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Vector from focus to a point --/
def vector_from_focus (p : PointOnParabola) : ℝ × ℝ :=
  (p.x - focus.1, p.y - focus.2)

/-- Perpendicular vectors --/
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

/-- Area of quadrilateral ABCD --/
noncomputable def area_ABCD (A B C D : PointOnParabola) : ℝ :=
  let AC := Real.sqrt ((C.x - A.x)^2 + (C.y - A.y)^2)
  let BD := Real.sqrt ((D.x - B.x)^2 + (D.y - B.y)^2)
  (1/2) * AC * BD

/-- The origin point --/
def origin : PointOnParabola :=
  { x := 0, y := 0, on_parabola := by simp [parabola] }

/-- The main theorem --/
theorem min_area_ABCD :
  ∀ (A B C D : PointOnParabola),
  A ≠ origin → B ≠ origin →
  perpendicular (vector_from_focus A) (vector_from_focus B) →
  (∃ (k : ℝ), C.y = k * (C.x - 1) ∧ D.y = (-1/k) * (D.x - 1)) →
  area_ABCD A B C D ≥ 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_ABCD_l811_81164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l811_81112

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^4
def curve2 (x y : ℝ) : Prop := x + y^2 = 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- State the theorem
theorem intersection_distance_is_two :
  ∃ p1 p2, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_two_l811_81112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_value_l811_81145

-- Define the slopes of two lines
noncomputable def slope_l1 : ℝ := 2
noncomputable def slope_l2 (a : ℝ) : ℝ := -2 / (a + 1)

-- Define the condition for parallel lines
def parallel_lines (a : ℝ) : Prop := slope_l1 = slope_l2 a

-- Theorem statement
theorem parallel_lines_imply_a_value :
  ∀ a : ℝ, parallel_lines a → a = -2 := by
  intro a h
  -- Proof steps would go here
  sorry

#check parallel_lines_imply_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_value_l811_81145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l811_81188

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + 5 * Real.pi / 6) + 1

/-- Theorem stating the equivalence of f(x) and g(x) -/
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l811_81188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l811_81174

theorem inverse_function_property {α β : Type*} (f : α → β) (f_inv : β → α) (a : α) (b : β) :
  Function.LeftInverse f_inv f →
  Function.RightInverse f_inv f →
  f a = b →
  f_inv b = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l811_81174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cornbread_pieces_l811_81100

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculate the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- The pan dimensions -/
def pan : Dimensions := { length := 20, width := 18 }

/-- The cornbread piece dimensions -/
def piece : Dimensions := { length := 2, width := 2 }

/-- The number of pieces that can be cut from the pan -/
def num_pieces : ℕ := (area pan / area piece).floor.toNat

theorem cornbread_pieces : num_pieces = 90 := by
  -- Unfold definitions
  unfold num_pieces
  unfold area
  -- Simplify the arithmetic
  simp [pan, piece]
  -- The proof is complete
  rfl

#eval num_pieces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cornbread_pieces_l811_81100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BD_length_l811_81182

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom on_circle : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle
axiom BC_equals_DC : dist B C = 4 ∧ dist D C = 4
axiom AE_length : dist A E = 6
axiom BE_DE_integers : ∃ (n m : ℕ), dist B E = n ∧ dist D E = m

-- Define the theorem
theorem BD_length : dist B D = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BD_length_l811_81182
