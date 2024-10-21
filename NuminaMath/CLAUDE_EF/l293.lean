import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_is_one_million_l293_29346

/-- Represents the dimensions of a metal sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the volume of a cube-shaped tank given the number of sheets required and sheet dimensions -/
noncomputable def tankVolume (sheetsRequired : ℝ) (sheetDim : SheetDimensions) : ℝ :=
  let sheetArea := sheetDim.length * sheetDim.width
  let faceArea := (sheetsRequired / 6) * sheetArea
  let sideLength := Real.sqrt faceArea
  sideLength ^ 3 * 1000  -- Convert cubic meters to liters

/-- Theorem: The volume of the tank is 1,000,000 liters -/
theorem tank_volume_is_one_million :
  let sheetsRequired : ℝ := 74.99999999999997
  let sheetDim : SheetDimensions := { length := 4, width := 2 }
  tankVolume sheetsRequired sheetDim = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_is_one_million_l293_29346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l293_29332

/-- Calculates the selling price of an article given its cost price and profit percentage. -/
noncomputable def sellingPrice (costPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  costPrice * (1 + profitPercentage / 100)

/-- Theorem stating that an article with cost price 280 and 30% profit has a selling price of 364. -/
theorem article_selling_price :
  sellingPrice 280 30 = 364 := by
  -- Unfold the definition of sellingPrice
  unfold sellingPrice
  -- Simplify the expression
  simp [mul_add, mul_div_right_comm]
  -- Perform the calculation
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check sellingPrice 280 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l293_29332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_odd_eight_rolls_l293_29370

def standard_die := Finset.range 6

theorem probability_product_odd_eight_rolls :
  let p_odd_single := (standard_die.filter (fun x => x % 2 = 1)).card / standard_die.card
  let p_all_odd := (p_odd_single : ℚ) ^ 8
  p_odd_single = 1 / 2 →
  p_all_odd = (1 : ℚ) / 256 :=
by
  intro h
  simp [h]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_odd_eight_rolls_l293_29370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cyclist_speed_theorem_l293_29318

/-- The speed of the third cyclist given the conditions of the problem -/
noncomputable def third_cyclist_speed (a b : ℝ) : ℝ :=
  (a + 3*b + Real.sqrt (a^2 - 10*a*b + 9*b^2)) / 4

/-- Theorem stating the speed of the third cyclist -/
theorem third_cyclist_speed_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (t₁ t₂ : ℝ), t₁ > 0 ∧ t₂ > 0 ∧ t₂ > t₁ ∧
  (t₁ + 1/6) * a = t₁ * third_cyclist_speed a b ∧
  (t₂ + 1/2) * b = (t₂ + 1/3) * third_cyclist_speed a b := by
  sorry

#check third_cyclist_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cyclist_speed_theorem_l293_29318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l293_29378

noncomputable def a (k : ℝ) (n : ℕ+) : ℝ := (k * n - 3) / (n - 3/2)

theorem sequence_properties (k : ℝ) :
  /- If {a_n} is an arithmetic sequence, then k = 2 -/
  (∀ n : ℕ+, 2 * a k n = a k (n + 1) + a k (n - 1)) → k = 2 ∧

  /- If k ≠ 2, then the maximum and minimum terms are as specified -/
  (k ≠ 2 →
    ((k > 2 →
      (∀ n : ℕ+, a k 1 ≤ a k n) ∧
      (∀ n : ℕ+, a k n ≤ a k 2)) ∧
    (k < 2 →
      (∀ n : ℕ+, a k 2 ≤ a k n) ∧
      (∀ n : ℕ+, a k n ≤ a k 1)))) ∧

  /- If a_n > (k2^n+(-1)^n)/2^n for any n ∈ ℕ*, then 101/48 < k < 13/6 -/
  ((∀ n : ℕ+, a k n > (k * (2 : ℝ)^(n : ℕ) + (-1)^(n : ℕ)) / (2 : ℝ)^(n : ℕ)) →
    101/48 < k ∧ k < 13/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l293_29378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l293_29397

theorem trigonometric_identity (α : ℝ) : 
  (Real.sin (2 * Real.pi - α))^2 + Real.cos (Real.pi + α) * Real.cos (Real.pi - α) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l293_29397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_equal_quotient_and_remainder_l293_29323

theorem division_with_equal_quotient_and_remainder (N : ℕ) : 
  (∃ k : ℕ, 2014 = k * (N + 1) ∧ k < N) ↔ N ∈ ({2013, 1006, 105, 52} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_equal_quotient_and_remainder_l293_29323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_exists_iff_circles_intersect_l293_29317

-- Define the circle S
variable (S : Set (ℝ × ℝ))

-- Define that S is a circle
variable (is_circle_S : IsCircle S)

-- Define the line segment MN
variable (M N : ℝ × ℝ)

-- Define the vector MN
def vector_MN (M N : ℝ × ℝ) : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the translated circle S1
def S1 (S : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Set (ℝ × ℝ) := 
  {p | ∃ q ∈ S, p = (q.1 + (vector_MN M N).1, q.2 + (vector_MN M N).2)}

-- Define the existence of chord AB
def exists_chord (S : Set (ℝ × ℝ)) (M N : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  A ∈ S ∧ B ∈ S ∧ 
  (B.1 - A.1, B.2 - A.2) = vector_MN M N ∧
  (B.1 - A.1) * (N.2 - M.2) = (B.2 - A.2) * (N.1 - M.1)

-- Theorem statement
theorem chord_exists_iff_circles_intersect (S : Set (ℝ × ℝ)) (M N : ℝ × ℝ) :
  (∃ A B, exists_chord S M N A B) ↔ (S ∩ S1 S M N).Nonempty :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_exists_iff_circles_intersect_l293_29317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l293_29303

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (p : Solution)
  (q : Solution)
  (p_volume : ℝ)
  (q_volume : ℝ)

/-- The percentage of solution P in the mixture -/
noncomputable def p_percentage (m : Mixture) : ℝ :=
  m.p_volume / (m.p_volume + m.q_volume) * 100

/-- The percentage of carbonated water in the mixture -/
noncomputable def mixture_carbonated_water_percent (m : Mixture) : ℝ :=
  (m.p.carbonated_water_percent * m.p_volume + m.q.carbonated_water_percent * m.q_volume) /
  (m.p_volume + m.q_volume)

theorem mixture_composition 
  (p : Solution)
  (q : Solution)
  (h_p : p.carbonated_water_percent = 80)
  (h_q : q.carbonated_water_percent = 55)
  (m : Mixture)
  (h_m : m.p = p ∧ m.q = q)
  (h_mixture : mixture_carbonated_water_percent m = 72) :
  p_percentage m = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l293_29303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_fixed_points_l293_29312

/-- Definition of the sequence of functions pₙ(x) -/
def p : ℕ → (ℝ → ℝ)
| 0 => λ x => x^2 - 2
| n + 1 => λ x => p 0 (p n x)

/-- Theorem stating that all fixed points of pₙ(x) are distinct real numbers -/
theorem distinct_fixed_points (n : ℕ) : 
  ∀ x y : ℝ, p n x = x ∧ p n y = y ∧ x ≠ y → x ≠ y :=
by
  sorry

/-- Lemma: For |x| > 2, pₙ(x) > |x| for all natural numbers n -/
lemma p_greater_than_abs (n : ℕ) (x : ℝ) (h : |x| > 2) : p n x > |x| :=
by
  sorry

/-- Lemma: All real roots of pₙ(x) satisfy |x| ≤ 2 -/
lemma roots_bounded (n : ℕ) (x : ℝ) (h : p n x = x) : |x| ≤ 2 :=
by
  sorry

/-- Lemma: The roots of pₙ(x) can be expressed in terms of cosine -/
lemma roots_cosine_form (n : ℕ) :
  ∃ (k : Fin (2^n - 1)) (l : Fin (2^n + 1)),
    (p n (2 * Real.cos (2 * k * Real.pi / (2^n - 1))) = 2 * Real.cos (2 * k * Real.pi / (2^n - 1))) ∧
    (p n (2 * Real.cos (2 * l * Real.pi / (2^n + 1))) = 2 * Real.cos (2 * l * Real.pi / (2^n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_fixed_points_l293_29312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propA_necessary_not_sufficient_l293_29365

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Proposition A: |PA| + |PB| is a constant -/
def propA (A B P : Point) (c : ℝ) : Prop :=
  distance P A + distance P B = c

/-- Proposition B: The trajectory of point P is an ellipse with foci A and B -/
def propB (A B P : Point) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (P.x - A.x)^2 / a^2 + (P.y - A.y)^2 / b^2 = 1 ∧
    distance A B = 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: propA is a necessary but not sufficient condition for propB -/
theorem propA_necessary_not_sufficient (A B : Point) :
  (∃ c : ℝ, ∀ P : Point, propB A B P → propA A B P c) ∧
  (∃ P : Point, ∃ c : ℝ, propA A B P c ∧ ¬propB A B P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propA_necessary_not_sufficient_l293_29365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_independent_set_size_l293_29330

/-- A graph satisfying the given conditions -/
structure FriendshipGraph where
  -- The graph is represented as a function from pairs of vertices to Bool
  edge : Fin 30 → Fin 30 → Bool
  -- Each vertex has degree at most 5
  degree_bound : ∀ v : Fin 30, (Finset.univ.filter (λ u => edge v u)).card ≤ 5
  -- For every subset of 5 vertices, there exists a pair of non-adjacent vertices
  non_adjacent_pair : ∀ S : Finset (Fin 30), S.card = 5 → 
    ∃ u v : Fin 30, u ∈ S ∧ v ∈ S ∧ u ≠ v ∧ ¬(edge u v)

/-- An independent set in the graph -/
def IndependentSet (G : FriendshipGraph) (S : Finset (Fin 30)) : Prop :=
  ∀ u v : Fin 30, u ∈ S → v ∈ S → u ≠ v → ¬(G.edge u v)

/-- The main theorem stating that the maximum size of an independent set is 6 -/
theorem max_independent_set_size (G : FriendshipGraph) :
  (∃ S : Finset (Fin 30), IndependentSet G S ∧ S.card = 6) ∧
  (∀ S : Finset (Fin 30), IndependentSet G S → S.card ≤ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_independent_set_size_l293_29330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_on_interval_l293_29311

noncomputable section

def f : ℝ → ℝ := sorry

def g : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)

axiom f_shifted_even : ∀ x : ℝ, f (x + 2) = f (-(x + 2))

axiom f_eq_g_on_interval : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x = g x

theorem f_periodic_on_interval (n : ℤ) (x : ℝ) 
  (h : -4 * ↑n - 2 ≤ x ∧ x ≤ -4 * ↑n + 2) : 
  f x = g (x + 4 * ↑n) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_on_interval_l293_29311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_journey_completion_l293_29337

/-- Represents a round-trip journey -/
structure RoundTrip where
  outbound : ℝ
  inbound : ℝ
  round_trip_symmetry : outbound = inbound

/-- Calculates the percentage of a round-trip completed -/
noncomputable def completed_percentage (trip : RoundTrip) (outbound_completed : ℝ) (inbound_completed : ℝ) : ℝ :=
  (outbound_completed + inbound_completed) / (trip.outbound + trip.inbound) * 100

/-- Theorem: If a technician completes the outbound journey and 50% of the inbound journey,
    they have completed 75% of the total round-trip -/
theorem technician_journey_completion (trip : RoundTrip) :
  completed_percentage trip trip.outbound (0.5 * trip.inbound) = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_journey_completion_l293_29337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_ratio_theorem_l293_29354

/-- Represents the sock order scenario --/
structure SockOrder where
  yellow_count : ℕ
  red_count : ℕ
  yellow_price : ℝ
  red_price : ℝ

/-- The initial sock order --/
def initial_order (red_count : ℕ) (p : ℝ) : SockOrder := {
  yellow_count := 5,
  red_count := red_count,
  yellow_price := p,
  red_price := 2 * p
}

/-- The exchanged sock order --/
def exchanged_order (initial : SockOrder) : SockOrder := {
  yellow_count := initial.red_count,
  red_count := initial.yellow_count,
  yellow_price := initial.yellow_price,
  red_price := initial.red_price
}

/-- Calculate the total cost of a sock order --/
def total_cost (order : SockOrder) : ℝ :=
  order.yellow_count * order.yellow_price + order.red_count * order.red_price

/-- The theorem to be proved --/
theorem sock_ratio_theorem (red_count : ℕ) (p : ℝ) :
  let initial := initial_order red_count p
  total_cost (exchanged_order initial) = 1.6 * total_cost initial →
  (initial.yellow_count : ℝ) / initial.red_count = 11 / 2 := by
  sorry

#check sock_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_ratio_theorem_l293_29354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_proportional_l293_29344

noncomputable section

-- Define the sets of line segments
def set_A : List ℝ := [2, 2.5, 3, 3.5]
def set_B : List ℝ := [Real.sqrt 3, 3, 3, 4 * Real.sqrt 3]
def set_C : List ℝ := [2, 4, 9, 18]
def set_D : List ℝ := [4, 5, 6, 7]

-- Define proportionality check
def is_proportional (s : List ℝ) : Prop :=
  s.length = 4 ∧ s[0]! * s[3]! = s[1]! * s[2]!

-- Theorem statement
theorem only_set_C_proportional :
  ¬(is_proportional set_A) ∧
  ¬(is_proportional set_B) ∧
  is_proportional set_C ∧
  ¬(is_proportional set_D) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_C_proportional_l293_29344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l293_29375

-- Define the radius of the original circular sheet
noncomputable def original_radius : ℝ := 8

-- Define the number of sectors
def num_sectors : ℕ := 4

-- Define the radius of the cone's base
noncomputable def cone_base_radius : ℝ := original_radius / num_sectors

-- Define the slant height of the cone (equal to the original radius)
noncomputable def slant_height : ℝ := original_radius

-- Theorem statement
theorem cone_height_from_circular_sector :
  ∃ (h : ℝ), h = 2 * Real.sqrt 15 ∧ h^2 + cone_base_radius^2 = slant_height^2 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l293_29375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_84_l293_29340

def scores : List Nat := [65, 69, 78, 84, 92]

def is_integer_average (sublist : List Nat) : Prop :=
  ∀ k : Nat, k ≤ sublist.length → (sublist.take k).sum % k = 0

theorem last_score_is_84 :
  ∃! x : Nat, x ∈ scores ∧ is_integer_average (scores.filter (· ≠ x) ++ [x]) :=
sorry

#check last_score_is_84

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_84_l293_29340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l293_29399

/-- A game where two players take turns removing stones from a pile. -/
structure StoneGame where
  n : ℕ
  initial_stones : ℕ
  turn_order : List (Fin 2)

/-- The possible moves in the game. -/
inductive Move
  | prime_less_than_n (p : ℕ)
  | multiple_of_n (k : ℕ)
  | one_stone

/-- Predicate to check if a move is valid. -/
def valid_move (g : StoneGame) (m : Move) : Prop :=
  match m with
  | Move.prime_less_than_n p => Nat.Prime p ∧ p < g.n
  | Move.multiple_of_n k => ∃ (x : ℕ), k * g.n = x
  | Move.one_stone => True

/-- A strategy is a function that given the current game state, returns a valid move. -/
def Strategy (g : StoneGame) := (stones_remaining : ℕ) → {m : Move // valid_move g m}

/-- Predicate to check if a strategy is winning for the first player. -/
def winning_strategy (g : StoneGame) (s : Strategy g) : Prop :=
  ∀ (opponent_strategy : Strategy g), 
    g.initial_stones > g.n^2 → 
    ∃ (final_move : Move), 
      valid_move g final_move ∧ 
      (final_move = Move.one_stone ∨ 
       ∃ (previous_stones : ℕ), previous_stones > 1 ∧ previous_stones ≤ g.initial_stones)

/-- Theorem stating that there exists a winning strategy for the first player. -/
theorem first_player_winning_strategy (g : StoneGame) : 
  ∃ (s : Strategy g), winning_strategy g s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l293_29399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_points_l293_29314

-- Define the five points
def p1 : ℝ × ℝ := (-2, 1)
def p2 : ℝ × ℝ := (0, 0)
def p3 : ℝ × ℝ := (0, 3)
def p4 : ℝ × ℝ := (4, 0)
def p5 : ℝ × ℝ := (4, 3)

-- Define the set of points
def points : Set (ℝ × ℝ) := {p1, p2, p3, p4, p5}

-- Define the property that no three points are collinear
def no_three_collinear (S : Set (ℝ × ℝ)) : Prop :=
  ∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (a.2 - b.2) * (c.1 - b.1) ≠ (c.2 - b.2) * (a.1 - b.1)

-- Define the conic section
structure ConicSection where
  equation : ℝ → ℝ → Prop
  center : ℝ × ℝ
  axes_parallel : Prop

-- Theorem statement
theorem conic_through_points (cs : ConicSection) 
  (h1 : ∀ p ∈ points, cs.equation p.1 p.2)
  (h2 : no_three_collinear points)
  (h3 : cs.axes_parallel) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    (∀ x y : ℝ, cs.equation x y ↔ 
      ((x - cs.center.1)^2 / a^2 + (y - cs.center.2)^2 / b^2 = 1)) ∧
    b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_points_l293_29314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l293_29385

theorem product_remainder (a b c : ℕ) 
  (ha : a % 6 = 2)
  (hb : b % 7 = 3)
  (hc : c % 8 = 4) :
  (a * b * c) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l293_29385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truthful_monkeys_count_l293_29326

/-- Represents the number of animals of each type -/
def num_animals : ℕ := 100

/-- Represents the total number of groups -/
def num_groups : ℕ := 100

/-- Represents the number of animals in each group -/
def group_size : ℕ := 3

/-- Represents the number of animals that said "Yes" to the tiger question -/
def tiger_yes_responses : ℕ := 138

/-- Represents the number of animals that said "Yes" to the fox question -/
def fox_yes_responses : ℕ := 188

theorem truthful_monkeys_count :
  ∃ x y z : ℕ,
    x + y + z = num_animals ∧
    x ≤ 2 * (num_animals - y) ∧
    y + (num_animals - x - z) = num_animals ∧
    x - y + z = tiger_yes_responses - num_animals ∧
    num_animals - x + y = fox_yes_responses - num_animals ∧
    z = num_animals - y - 26 ∧
    num_animals - x = 76 := by
  sorry

#check truthful_monkeys_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truthful_monkeys_count_l293_29326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_value_l293_29369

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the side length function
noncomputable def sideLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the theorem
theorem max_tan_C_value (t : Triangle) :
  sideLength t.A t.B = 2 →
  (sideLength t.A t.C)^2 - (sideLength t.B t.C)^2 = 6 →
  ∃ (max_tan_C : ℝ), max_tan_C = 2 * Real.sqrt 5 / 5 ∧
    ∀ (tan_C : ℝ), tan_C ≤ max_tan_C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tan_C_value_l293_29369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_squares_l293_29349

/-- Square in 2D plane -/
def Square (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Rectangle in 2D plane -/
def Rectangle (r : Set (ℝ × ℝ)) : Prop := sorry

/-- Area of a 2D shape -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The area of a rectangle containing four squares with specific areas -/
theorem rectangle_area_with_squares (EFGH : Set (ℝ × ℝ)) 
  (s1 s2 s3 s4 : Set (ℝ × ℝ)) :
  (∃ (a b c d : ℝ), 
    Square s1 ∧ area s1 = 4 ∧
    Square s2 ∧ area s2 = 4 ∧
    Square s3 ∧ area s3 = 4 ∧
    Square s4 ∧ area s4 = 1 ∧
    Rectangle EFGH ∧
    s1 ⊆ EFGH ∧ s2 ⊆ EFGH ∧ s3 ⊆ EFGH ∧ s4 ⊆ EFGH ∧
    Disjoint s1 s2 ∧ Disjoint s1 s3 ∧ Disjoint s1 s4 ∧
    Disjoint s2 s3 ∧ Disjoint s2 s4 ∧
    Disjoint s3 s4) →
  area EFGH = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_with_squares_l293_29349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l293_29341

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then 2 * x else -x^2

-- Define the proposed inverse function
noncomputable def f_inv (x : ℝ) : ℝ := if x ≥ 0 then x / 2 else -Real.sqrt (-x)

-- State the theorem
theorem f_inverse_correct : 
  (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_correct_l293_29341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_three_sufficient_not_necessary_l293_29300

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2) ∨ v2 = (k * v1.1, k * v1.2)

/-- The condition "λ = 3" is sufficient but not necessary for the vectors to be parallel -/
theorem lambda_three_sufficient_not_necessary (l : ℝ) :
  (l = 3 → are_parallel (3, l) (l - 1, 2)) ∧
  ¬(are_parallel (3, l) (l - 1, 2) → l = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_three_sufficient_not_necessary_l293_29300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l293_29387

-- Define the sets A, B, and C
def A : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 1}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem set_operations_and_range :
  (A ∪ B = {x | x > -1}) ∧
  ((Set.univ \ A) ∩ B = {x | -1 < x ∧ x ≤ 0}) ∧
  (∀ a : ℝ, B ∪ C a = C a → a ≥ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_range_l293_29387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l293_29398

/-- The speed of a train given its length, time to pass a man, and the man's speed in the opposite direction -/
theorem train_speed (train_length : Real) (passing_time : Real) (man_speed : Real) :
  train_length = 110 →
  passing_time = 5.999520038396929 →
  man_speed = 6 →
  (train_length / passing_time * 3600 / 1000) + man_speed = 72.00001499862502 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l293_29398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_chords_twenty_points_l293_29302

/-- The number of ways to draw n non-intersecting chords connecting 2n points on a circle -/
noncomputable def chord_arrangements (n : ℕ) : ℚ := (1 : ℚ) / (n + 1 : ℚ) * (Nat.choose (2 * n) n)

/-- The theorem stating the number of ways to draw 10 non-intersecting chords connecting 20 points on a circle -/
theorem ten_chords_twenty_points : ⌊chord_arrangements 10⌋ = 16796 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_chords_twenty_points_l293_29302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_downstream_l293_29331

/-- Calculates the distance traveled downstream given the boat speed in still water,
    current speed, and time taken. -/
noncomputable def distance_downstream (boat_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + current_speed) * (1000 / 3600) * time

/-- Theorem stating that given the specific conditions, the distance covered is approximately 59.995 meters. -/
theorem distance_covered_downstream :
  let boat_speed := (15 : ℝ)
  let current_speed := (3 : ℝ)
  let time := (11.999040076793857 : ℝ)
  abs (distance_downstream boat_speed current_speed time - 59.995200383969285) < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_downstream_l293_29331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_DEF_l293_29352

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  2 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) / (a + b + c)

/-- Theorem: The radius of the inscribed circle in triangle DEF is 2√14 -/
theorem inscribed_circle_radius_DEF :
  inscribed_circle_radius 26 15 17 = 2 * Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_DEF_l293_29352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_84_l293_29358

theorem divisible_by_84 (p : ℕ) : ∃ k : ℤ, (4 : ℤ)^(2*p) - (3 : ℤ)^(2*p) - 7 = 84 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_84_l293_29358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l293_29322

/-- A power function that passes through the point (2, √2/2) -/
noncomputable def f : ℝ → ℝ :=
  fun x => x^(-(1/2 : ℝ))

theorem power_function_through_point (h : f 2 = Real.sqrt 2 / 2) :
  f 9 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l293_29322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l293_29366

def is_monic_quadratic (p : Polynomial ℝ) : Prop :=
  (p.degree = 2) ∧ (p.leadingCoeff = 1)

def has_complex_root (p : Polynomial ℝ) (z : ℂ) : Prop :=
  p.eval₂ Complex.ofReal z = 0

theorem monic_quadratic_with_complex_root :
  ∃! p : Polynomial ℝ, is_monic_quadratic p ∧ has_complex_root p (Complex.mk 2 (-3)) :=
by sorry

#check monic_quadratic_with_complex_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l293_29366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_triangle_from_orthocenters_l293_29342

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The feet of altitudes of a triangle --/
structure AltitudeFeet where
  A₁ : Point
  B₁ : Point
  C₁ : Point

/-- Orthocenter of a triangle --/
noncomputable def orthocenter (t : Triangle) : Point :=
  sorry

/-- Check if three points are collinear --/
def collinear (p q r : Point) : Prop :=
  sorry

/-- The statement of the theorem --/
theorem construct_triangle_from_orthocenters 
  (A₁ B₁ C₁ : Point) 
  (H_AB₁C₁ H_BC₁A₁ H_CA₁B₁ : Point) : 
  ∃! (ABC : Triangle), 
    let feet := AltitudeFeet.mk A₁ B₁ C₁
    orthocenter (Triangle.mk ABC.A B₁ C₁) = H_AB₁C₁ ∧
    orthocenter (Triangle.mk ABC.B C₁ A₁) = H_BC₁A₁ ∧
    orthocenter (Triangle.mk ABC.C A₁ B₁) = H_CA₁B₁ ∧
    ¬collinear ABC.A ABC.B ABC.C :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_triangle_from_orthocenters_l293_29342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_injective_implies_a_less_than_two_l293_29338

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else a*x - 1

theorem f_not_injective_implies_a_less_than_two (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_injective_implies_a_less_than_two_l293_29338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_RSQT_l293_29315

/-- A triangle with vertices P, Q, and R -/
structure Triangle (P Q R : ℝ × ℝ) : Type :=
  (isIsosceles : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - R.1)^2 + (P.2 - R.2)^2)

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle P Q R) : ℝ := sorry

/-- A trapezoid with vertices R, S, Q, and T -/
structure Trapezoid (R S Q T : ℝ × ℝ) : Type

/-- The area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid R S Q T) : ℝ := sorry

/-- Theorem stating that the area of trapezoid RSQT is 60 -/
theorem area_of_trapezoid_RSQT 
  (P Q R : ℝ × ℝ)
  (PQR : Triangle P Q R)
  (R S Q T : ℝ × ℝ)
  (RSQT : Trapezoid R S Q T)
  (allTrianglesSimilar : ∀ (P' Q' R' : ℝ × ℝ) (t : Triangle P' Q' R'), ∃ k : ℝ, triangleArea t = k * triangleArea PQR)
  (smallestTriangles : ℕ)
  (smallestTriangleArea : ℝ)
  (h1 : smallestTriangles = 9)
  (h2 : smallestTriangleArea = 3)
  (h3 : triangleArea PQR = 75)
  : trapezoidArea RSQT = 60 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_RSQT_l293_29315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_value_l293_29383

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_function_m_value 
  (a b c : ℝ) 
  (h1 : quadratic_function a b c 0 = 2)
  (h2 : quadratic_function a b c 1 = 1)
  (h3 : quadratic_function a b c 2 = 2)
  (h4 : quadratic_function a b c 3 = 5)
  (h5 : quadratic_function a b c 4 = 10) :
  quadratic_function a b c (-1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_value_l293_29383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_sixteen_pairs_l293_29339

/-- Definition of remainder for real division -/
noncomputable def remainder (p q : ℝ) : ℝ :=
  p - q * ⌊p / q⌋

/-- Checks if a pair (a, b) satisfies the conditions -/
def satisfiesConditions (a b : ℕ) : Prop :=
  a ≤ 20 ∧ b ≤ 20 ∧
  (remainder (a * Real.sqrt 2 + b * Real.sqrt 3) (Real.sqrt 2) +
   remainder (a * Real.sqrt 2 + b * Real.sqrt 3) (Real.sqrt 3) = Real.sqrt 2)

/-- The main theorem stating that there are exactly 16 pairs satisfying the conditions -/
theorem exactly_sixteen_pairs :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 16 ∧ ∀ (p : ℕ × ℕ), p ∈ s ↔ satisfiesConditions p.1 p.2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_sixteen_pairs_l293_29339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l293_29307

theorem trigonometric_identity (α : Real) 
  (h1 : Real.tan α + (1 / Real.tan α) = 10 / 3)
  (h2 : α ∈ Set.Ioo (π / 4) (π / 2)) :
  Real.sin (2 * α + π / 4) + 2 * Real.cos (π / 4) * (Real.cos α) ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l293_29307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_australia_modern_newzealand_modern_not_all_oceania_modern_l293_29345

-- Define the types of countries
inductive CountryType
| Developed
| Developing

-- Define the types of population growth patterns
inductive GrowthPattern
| Modern
| Traditional

-- Define a country or region
structure Country where
  name : String
  type : CountryType
  growthPattern : GrowthPattern

-- Define Oceania as a set of countries
def Oceania : Set Country := sorry

-- Define Australia and New Zealand
def Australia : Country := ⟨"Australia", CountryType.Developed, GrowthPattern.Modern⟩
def NewZealand : Country := ⟨"New Zealand", CountryType.Developed, GrowthPattern.Modern⟩

-- State that Australia and New Zealand are in Oceania
axiom australia_in_oceania : Australia ∈ Oceania
axiom newzealand_in_oceania : NewZealand ∈ Oceania

-- State that Australia and New Zealand have modern growth patterns
theorem australia_modern : Australia.growthPattern = GrowthPattern.Modern := rfl
theorem newzealand_modern : NewZealand.growthPattern = GrowthPattern.Modern := rfl

-- State that there exists at least one country in Oceania with traditional growth pattern
axiom exists_traditional_in_oceania : ∃ c ∈ Oceania, c.growthPattern = GrowthPattern.Traditional

-- Theorem: Not all countries in Oceania have modern growth patterns
theorem not_all_oceania_modern :
  ¬(∀ c ∈ Oceania, c.growthPattern = GrowthPattern.Modern) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_australia_modern_newzealand_modern_not_all_oceania_modern_l293_29345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_condition_l293_29390

/-- Two vectors form an obtuse angle if their dot product is negative -/
def is_obtuse_angle (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) < 0

/-- The theorem statement -/
theorem obtuse_angle_condition (lambda : ℝ) :
  is_obtuse_angle (2, -4) (-1, lambda) ↔ lambda > -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_condition_l293_29390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_Q_in_second_quadrant_l293_29367

-- Define the point type
def Point := ℝ × ℝ

-- Define the quadrant function
noncomputable def quadrant (p : Point) : ℕ :=
  match p with
  | (x, y) =>
    if x > 0 ∧ y > 0 then 1
    else if x < 0 ∧ y > 0 then 2
    else if x < 0 ∧ y < 0 then 3
    else if x > 0 ∧ y < 0 then 4
    else 0  -- On axis or at origin

-- Theorem statement
theorem point_Q_in_second_quadrant (a : ℝ) :
  quadrant (3, -a) = 1 → quadrant (a, -a + 2) = 2 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_Q_in_second_quadrant_l293_29367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l293_29348

-- Define our set
def S : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define the property of containing 6
def contains_six (subset : Finset Nat) : Prop := 6 ∈ subset

-- Prove that contains_six is decidable
instance : DecidablePred contains_six := fun s => Finset.decidableMem 6 s

-- State the theorem
theorem subsets_containing_six :
  (Finset.filter contains_six (Finset.powerset S)).card = 2^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_containing_six_l293_29348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_closed_l293_29362

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 2  -- Added case for n = 0
  | 1 => 2
  | (n + 2) => a (n + 1) + (n + 2) + 1

/-- The proposed closed form for a_n -/
def a_closed (n : ℕ) : ℚ := (n^2 + n + 2) / 2

/-- Theorem stating that the recursive and closed forms are equivalent -/
theorem a_equals_a_closed : ∀ n : ℕ, a n = a_closed n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_closed_l293_29362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_range_l293_29305

/-- The angle between asymptotes of a hyperbola with given eccentricity range -/
theorem hyperbola_asymptote_angle_range (a b : ℝ) (e : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (e = Real.sqrt (a^2 + b^2) / a) →         -- Definition of eccentricity
  (Real.sqrt 2 ≤ e ∧ e ≤ 2) →               -- Given eccentricity range
  (∃ θ : ℝ, θ = 2 * Real.arctan (b / a) ∧ π / 2 ≤ θ ∧ θ ≤ 2 * π / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_range_l293_29305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_zero_and_sum_of_roots_l293_29355

/-- A monic quartic polynomial with specific complex roots -/
noncomputable def Q (x : ℂ) : ℂ := x^4 + Q.a*x^3 + Q.b*x^2 + Q.c*x + Q.d
  where
  Q.a : ℂ := sorry
  Q.b : ℂ := sorry
  Q.c : ℂ := sorry
  Q.d : ℂ := sorry

/-- The angle α, where 0 < α < π/6 -/
noncomputable def α : ℝ := sorry

/-- Assumption that 0 < α < π/6 -/
axiom α_range : 0 < α ∧ α < Real.pi/6

/-- The roots of the polynomial Q -/
noncomputable def roots : List ℂ := [
  Complex.exp (Complex.I * α),
  Complex.exp (-Complex.I * α),
  Complex.exp (Complex.I * (α + Real.pi/6)),
  Complex.exp (-Complex.I * (α + Real.pi/6))
]

/-- Q is a monic quartic polynomial with real coefficients -/
axiom Q_monic_real : ∀ x, Q x = (x - roots[0]!) * (x - roots[1]!) * (x - roots[2]!) * (x - roots[3]!)

/-- The area of the quadrilateral formed by the roots is equal to a quarter of Q(0) -/
axiom area_condition : (Complex.abs (roots[0]! - roots[2]!) * Complex.abs (roots[1]! - roots[3]!)) / 2 = Q 0 / 4

theorem Q_zero_and_sum_of_roots :
  Q 0 = 1 ∧ roots.sum = 2 * (Real.cos α + Real.cos (α + Real.pi/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_zero_and_sum_of_roots_l293_29355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incandescent_switched_on_percentage_l293_29381

/-- Represents the ratio of incandescent to fluorescent bulbs -/
noncomputable def bulb_ratio : ℝ := 1 / 4

/-- Percentage of incandescent bulbs that are switched on -/
noncomputable def incandescent_on_ratio : ℝ := 0.3

/-- Percentage of fluorescent bulbs that are switched on -/
noncomputable def fluorescent_on_ratio : ℝ := 0.8

/-- Percentage of all bulbs that are switched on -/
noncomputable def total_on_ratio : ℝ := 0.7

/-- Theorem stating the percentage of switched-on bulbs that are incandescent -/
theorem incandescent_switched_on_percentage :
  let total_bulbs := 1 + bulb_ratio
  let incandescent_on := incandescent_on_ratio * 1
  let fluorescent_on := fluorescent_on_ratio * bulb_ratio
  let total_on := incandescent_on + fluorescent_on
  ∀ ε > 0, |incandescent_on / total_on - 0.0857| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incandescent_switched_on_percentage_l293_29381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_variables_related_l293_29316

-- Define the variables
variable (taxi_fare : ℝ → ℝ)
variable (distance_traveled : ℝ)
variable (house_size : ℝ → ℝ)
variable (house_price : ℝ)
variable (height : ℝ → ℝ)
variable (weight : ℝ)
variable (iron_size : ℝ → ℝ)
variable (iron_mass : ℝ)

-- Define the concept of related variables
def are_related (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ f x = k * x

-- Theorem stating that all given pairs of variables are related
theorem all_variables_related : 
  are_related taxi_fare distance_traveled ∧
  are_related house_size house_price ∧
  are_related height weight ∧
  are_related iron_size iron_mass :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_variables_related_l293_29316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_one_equals_nine_l293_29373

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 3 * ((fun y ↦ (y + 9) / 4) x)^2 - 4 * ((fun y ↦ (y + 9) / 4) x) + 5

-- Theorem statement
theorem g_of_negative_one_equals_nine : g (-1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_one_equals_nine_l293_29373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_ratio_l293_29396

-- Define the curves
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the tangent line
def tangent_line (x₀ y₀ m : ℝ) (x : ℝ) : ℝ := m * (x - x₀) + y₀

theorem tangent_line_ratio : 
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → 
  (∃ (m b : ℝ), 
    (∀ x, tangent_line x₁ (f x₁) (2 * x₁) x = m * x + b) ∧
    (∀ x, tangent_line x₂ (g x₂) (3 * x₂^2) x = m * x + b)) →
  x₁ / x₂ = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_ratio_l293_29396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_third_point_l293_29359

/-- The golden ratio, approximately 0.618 --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- The 0.618 method for selecting trial points --/
noncomputable def select_trial_point (a b : ℝ) : ℝ := a + φ * (b - a)

/-- The third trial point in the 0.618 method --/
noncomputable def third_trial_point (a b x₁ x₂ : ℝ) : ℝ :=
  if x₁ > x₂ 
  then b - φ * (b - x₁)
  else a + b - (b - φ * (b - a))

theorem golden_section_search_third_point 
  (a b x₁ x₂ : ℝ) 
  (h_interval : a = 2 ∧ b = 4) 
  (h_x₁ : x₁ = select_trial_point a b ∨ x₁ = select_trial_point b a)
  (h_x₂ : x₂ = select_trial_point a b ∨ x₂ = select_trial_point b a)
  (h_x₁_better : (x₁ > x₂ → x₁ = select_trial_point a b) ∧ 
                 (x₁ < x₂ → x₁ = select_trial_point b a)) :
  third_trial_point a b x₁ x₂ = 3.528 ∨ 
  third_trial_point a b x₁ x₂ = 2.472 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_third_point_l293_29359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l293_29386

theorem complex_number_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (Complex.mk x y) = (Complex.mk 3 1) / (Complex.mk 1 (-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l293_29386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_lowest_degree_polynomial_l293_29357

noncomputable def degree (p : ℕ → ℤ) : ℕ := sorry

theorem unique_lowest_degree_polynomial (f : ℕ → ℤ) : 
  (∀ n : ℕ, f n = n^2 + n + 1) →
  f 0 = 1 ∧ f 1 = 3 ∧ f 2 = 7 ∧ f 3 = 13 ∧ f 4 = 21 ∧
  (∀ g : ℕ → ℤ, g 0 = 1 ∧ g 1 = 3 ∧ g 2 = 7 ∧ g 3 = 13 ∧ g 4 = 21 →
    (∃ p : ℕ → ℤ, ∀ n : ℕ, g n = p n) →
    (∃ d : ℕ, ∀ n : ℕ, f n = p n ∧ degree p ≤ d) →
    degree p ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_lowest_degree_polynomial_l293_29357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_l293_29304

/-- The number of permutations of (n-1) elements when one element is fixed -/
def fixedElementPermutations (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem relay_team_orders :
  let total_members : ℕ := 5
  let fixed_member_position : ℕ := 5
  fixedElementPermutations total_members = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_l293_29304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_house_number_l293_29384

def house_numbers (a₁ : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + 2 * i)

theorem fifth_house_number (a₁ : ℕ) (n : ℕ) :
  (house_numbers a₁ n).sum = 117 →
  n ≥ 5 →
  (house_numbers a₁ n).get? 4 = some 13 :=
by
  sorry

#eval house_numbers 5 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_house_number_l293_29384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l293_29379

-- Define the triangle and its properties
def triangle (a b c A B C : ℝ) : Prop :=
  b = 2 ∧ 
  B = Real.pi / 3 ∧ 
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C ∧
  0 < A ∧ A < Real.pi / 2 ∧
  0 < B ∧ B < Real.pi / 2 ∧
  0 < C ∧ C < Real.pi / 2

-- State the theorem
theorem triangle_area (a b c A B C : ℝ) 
  (h : triangle a b c A B C) : 
  (1 / 2) * b * b * Real.sin B = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l293_29379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_final_ellipse_equation_l293_29393

/-- The equation of an ellipse with semi-major axis 5 and semi-minor axis b -/
def ellipse_equation (x y b : ℝ) : Prop := x^2 / 25 + y^2 / b^2 = 1

/-- The major axis of the ellipse -/
def major_axis : ℝ := 10

/-- The minor axis of the ellipse -/
def minor_axis (b : ℝ) : ℝ := 2 * b

/-- The focal length of the ellipse -/
noncomputable def focal_length (b : ℝ) : ℝ := 2 * Real.sqrt (25 - b^2)

/-- The arithmetic sequence property -/
def arithmetic_sequence (b : ℝ) : Prop :=
  minor_axis b + focal_length b = major_axis

theorem ellipse_b_value (b : ℝ) (h1 : 0 < b) (h2 : b < 5) 
  (h3 : arithmetic_sequence b) : b = 4 := by
  sorry

/-- The final equation of the ellipse -/
theorem final_ellipse_equation (x y : ℝ) : 
  ellipse_equation x y 4 ↔ x^2 / 25 + y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_b_value_final_ellipse_equation_l293_29393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l293_29306

def point_A (x : ℝ) : ℝ × ℝ × ℝ := (x, 5 - x, 2 * x - 1)

def point_B (x : ℝ) : ℝ × ℝ × ℝ := (1, x + 2, 2 - x)

def vector_AB (x : ℝ) : ℝ × ℝ × ℝ := 
  ((point_B x).1 - (point_A x).1,
   (point_B x).2.1 - (point_A x).2.1,
   (point_B x).2.2 - (point_A x).2.2)

noncomputable def length_AB (x : ℝ) : ℝ :=
  Real.sqrt ((vector_AB x).1^2 + (vector_AB x).2.1^2 + (vector_AB x).2.2^2)

theorem min_length_AB :
  ∃ (x : ℝ), ∀ (y : ℝ), length_AB x ≤ length_AB y ∧ x = 8/7 := by
  sorry

#eval vector_AB 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l293_29306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_two_possibilities_l293_29329

noncomputable def area_triangle (a b c : ℝ) : ℝ := (1 / 2) * b * c * Real.sin a

theorem triangle_area_two_possibilities (A B C : ℝ) (angleC : A + B + C = Real.pi) 
  (side_AC : Real.sqrt 3 * 3 = side_AC) (side_AB : side_AB = 3) (angle_C : C = Real.pi / 6) :
  (area_triangle A B C = (9 * Real.sqrt 3) / 2) ∨ 
  (area_triangle A B C = (9 * Real.sqrt 3) / 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_two_possibilities_l293_29329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l293_29351

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 2|

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -2 then x + 2 else -x - 2

-- Theorem stating that f and g are equal for all real x
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l293_29351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ball_probability_l293_29343

/-- Represents the outcome of drawing two balls from the urn -/
inductive DrawOutcome
  | SameColor
  | DifferentColors

/-- Represents the color of the last remaining ball in the urn -/
inductive FinalBallColor
  | White
  | Black

/-- The probability that the last remaining ball is white given the initial number of white balls -/
def probability_white_ball (n : ℕ) : ℝ :=
  if n % 2 = 1 then 1 else 0

/-- The main theorem stating the probability of the last ball being white -/
theorem last_ball_probability (n m : ℕ) :
  probability_white_ball n = 
    if n % 2 = 1 then 1 else 0 :=
by sorry

/-- The parity of white balls remains constant throughout the process -/
lemma white_ball_parity_invariant (n m : ℕ) (outcome : DrawOutcome) :
  (n % 2 = n % 2) :=
by sorry

/-- The final ball color depends on the initial parity of white balls -/
lemma final_ball_color (n m : ℕ) :
  (n % 2 = 1) ↔ (FinalBallColor.White = FinalBallColor.White) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_ball_probability_l293_29343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_points_condition_l293_29328

/-- A function f(x) = √(x - m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (x - m)

/-- The inverse of f -/
def f_inv (m : ℝ) (x : ℝ) : ℝ := x^2 + m

/-- Theorem stating the condition for common points -/
theorem common_points_condition (m : ℝ) :
  (∃ x : ℝ, f m x = f_inv m x) ↔ m ≤ 1/4 := by
  sorry

#check common_points_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_points_condition_l293_29328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_implies_a_equals_three_l293_29301

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * (1 + Real.log x)

-- Theorem statement
theorem derivative_at_one_implies_a_equals_three 
  (a : ℝ) (h : f' a 1 = 3) : a = 3 := by
  have h1 : f' a 1 = a * (1 + Real.log 1) := rfl
  have h2 : Real.log 1 = 0 := Real.log_one
  rw [h2] at h1
  rw [h1] at h
  simp at h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_implies_a_equals_three_l293_29301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_min_c_achievable_l293_29377

/-- Represents the piecewise linear function f(x) = |x - a| + |x - b| + |x - c| -/
def f (a b c x : ℤ) : ℤ := abs (x - a) + abs (x - b) + abs (x - c)

/-- Represents the system of equations -/
def has_unique_solution (a b c : ℤ) : Prop :=
  ∃! x y : ℤ, 2 * x + y = 2027 ∧ y = f a b c x

/-- The main theorem -/
theorem min_c_value (a b c : ℤ) : 
  a < b → b < c → has_unique_solution a b c → c ≥ 1014 := by
  sorry

/-- The minimum value of c is actually achievable -/
theorem min_c_achievable : 
  ∃ a b : ℤ, a < b ∧ b < 1014 ∧ has_unique_solution a b 1014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_min_c_achievable_l293_29377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l293_29334

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a + 4 / a) → 
  a ∈ Set.Ioi 0 ∪ {2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l293_29334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l293_29389

/-- The sum of the first n terms of the sequence a_n = 1 / (n * (n + 1)) -/
def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (fun k => 1 / ((k + 1 : ℚ) * (k + 2)))

/-- The sequence a_n = 1 / (n * (n + 1)) -/
def a (n : ℕ+) : ℚ := 1 / ((n : ℚ) * (n + 1))

theorem sum_of_sequence (n : ℕ+) : S n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_l293_29389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circle_relation_l293_29382

theorem equilateral_triangle_circle_relation (R : ℝ) : 
  (let circle_radius : ℝ := 1.5;
   let circle_circumference : ℝ := 2 * Real.pi * circle_radius;
   let triangle_side_length : ℝ := circle_circumference / 3;
   let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side_length ^ 2;
   triangle_area = R * Real.pi ^ 2) → R = Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circle_relation_l293_29382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_specific_floor_l293_29380

/-- Calculates the shaded area of a floor with specified dimensions and tile patterns -/
noncomputable def shadedAreaOfFloor (floorLength floorWidth tileSize quarterCircleRadius centerCircleRadius : ℝ) : ℝ :=
  let numTiles := (floorLength / tileSize) * (floorWidth / tileSize)
  let tileArea := tileSize ^ 2
  let quarterCircleArea := Real.pi * quarterCircleRadius ^ 2 / 4
  let centerCircleArea := Real.pi * centerCircleRadius ^ 2
  let whiteAreaPerTile := 4 * quarterCircleArea + centerCircleArea
  let shadedAreaPerTile := tileArea - whiteAreaPerTile
  numTiles * shadedAreaPerTile

/-- The shaded area of the floor with given specifications is 105.75 - 38.25π square feet -/
theorem shaded_area_of_specific_floor :
  shadedAreaOfFloor 12 15 2 1 (1/4) = 105.75 - 38.25 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_specific_floor_l293_29380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_function_l293_29376

noncomputable def f (x : ℝ) : ℝ := 2 * Real.tan (2 * x - Real.pi / 4)

theorem symmetry_center_of_tan_function :
  ∀ k : ℤ, ∃ x y : ℝ, 
    (x = k * Real.pi / 4 + Real.pi / 8 ∧ y = 0) ∧
    (∀ t : ℝ, f (x + t) = f (x - t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_function_l293_29376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l293_29372

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1/3) ^ x

-- Define the interval
def interval : Set ℝ := Set.Ioo 0 (50 * Real.pi)

-- Define the solution set
def solution_set : Set ℝ := {x ∈ interval | f x = g x}

-- Theorem statement
theorem number_of_solutions : ∃ (S : Finset ℝ), S.card = 50 ∧ ∀ x ∈ S, x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l293_29372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_imply_a_value_l293_29321

/-- Given points A, B, and C in a rectangular coordinate system,
    if AB is perpendicular to AC, then a = -1 -/
theorem perpendicular_vectors_imply_a_value
  (a : ℝ)
  (A B C : ℝ × ℝ × ℝ)
  (hA : A = (1, 0, 2))
  (hB : B = (2, 1, 0))
  (hC : C = (0, a, 1))
  (h_perp : ((B.1 - A.1) * (C.1 - A.1) + (B.2.1 - A.2.1) * (C.2.1 - A.2.1) + (B.2.2 - A.2.2) * (C.2.2 - A.2.2)) = 0) :
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_imply_a_value_l293_29321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adBlockerEffectiveness_l293_29336

/-- The effectiveness of AdBlocker A -/
noncomputable def adBlockerA : ℝ := 0.75

/-- The effectiveness of AdBlocker B -/
noncomputable def adBlockerB : ℝ := 0.85

/-- The effectiveness of AdBlocker C -/
noncomputable def adBlockerC : ℝ := 0.95

/-- The percentage of interesting ads among those not blocked -/
noncomputable def interestingUnblockedPercentage : ℝ := 0.15

/-- The percentage of ads that are not interesting and don't get blocked by any AdBlocker -/
noncomputable def notInterestingUnblockedPercentage : ℝ := 0.159375 / 100

theorem adBlockerEffectiveness :
  (1 - adBlockerA) * (1 - adBlockerB) * (1 - adBlockerC) * (1 - interestingUnblockedPercentage) = notInterestingUnblockedPercentage := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adBlockerEffectiveness_l293_29336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_f_minimum_value_property_l293_29371

def f (x : ℝ) := |x - 1| + |x + 2|

theorem f_inequality_solution_set :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 := by sorry

theorem f_minimum_value_property (a : ℝ) :
  (∀ x, f x > a^2 - 2*a) ↔ a ∈ Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_f_minimum_value_property_l293_29371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l293_29353

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the point of tangency
def x₀ : ℝ := 0

-- State the theorem
theorem tangent_line_at_zero : 
  ∃ (m b : ℝ), ∀ x, (f x₀ + m * (x - x₀) = m * x + b) ∧ (m = 1 ∧ b = 1) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l293_29353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_quadratic_equation_cos_2x_quadratic_equation_specific_l293_29364

theorem cos_2x_quadratic_equation (a b c x : ℝ) :
  a * (Real.cos x)^2 + b * Real.cos x + c = 0 →
  (a^2 / 4) * (Real.cos (2 * x))^2 + ((a^2 + 2 * a * c - b^2) / 2) * Real.cos (2 * x) +
  ((a^2 + 4 * a * c + 4 * c^2 - 2 * b^2) / 4) = 0 := by
  sorry

-- Application to the specific case
theorem cos_2x_quadratic_equation_specific :
  (4 : ℝ) * (Real.cos x)^2 + 2 * Real.cos x - 1 = 0 →
  (Real.cos (2 * x))^2 + 2 * Real.cos (2 * x) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_quadratic_equation_cos_2x_quadratic_equation_specific_l293_29364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l293_29325

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := (5 * x + 12 * y + 20 = 0) ∨ (x + 4 = 0)

-- Define the point (-4, 0)
def point_on_l : ℝ × ℝ := (-4, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_equation :
  ∃ (A B : ℝ × ℝ),
    my_circle A.1 A.2 ∧
    my_circle B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    line_l point_on_l.1 point_on_l.2 ∧
    distance A B = 8 :=
by
  sorry

#check line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l293_29325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l293_29310

noncomputable def f (x : ℝ) : ℝ := (3 - x) / (4 * x + 5)

theorem unattainable_value (x : ℝ) (h : x ≠ -5/4) : f x ≠ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l293_29310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l293_29395

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k / (x + 1)

-- State the theorem
theorem function_properties :
  -- Part 1: Monotonicity condition
  (∃ k : ℝ, ∀ x y : ℝ, 0 < x ∧ x < y → f k x < f k y) ↔ 
  (∃ k : ℝ, k ≤ 4) ∧
  -- Part 2: Inequality for x ∈ (1, 2)
  ∀ x : ℝ, 1 < x ∧ x < 2 → 
    (2 - x) * Real.exp (2 * (x - 1/x)) - 2 * x^2 + x < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l293_29395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_longest_diagonal_l293_29368

/-- The length of the longest diagonal in a regular heptagon with side length a -/
noncomputable def longest_diagonal (a : ℝ) : ℝ := 2 * a / Real.sqrt 5

/-- Theorem: In a regular heptagon with side length a, the length of the longest diagonal is (2a / √5) -/
theorem regular_heptagon_longest_diagonal (a : ℝ) (h : a > 0) :
  longest_diagonal a = 2 * a * Real.cos (3 * Real.pi / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_longest_diagonal_l293_29368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l293_29320

noncomputable section

-- Define the circle and its properties
def circle_radius : ℝ := 7
def central_angle : ℝ := 240

-- Define the perimeter function
noncomputable def perimeter_shaded_region (r : ℝ) (angle : ℝ) : ℝ :=
  2 * r + (angle / 360) * 2 * Real.pi * r

-- Theorem statement
theorem shaded_region_perimeter :
  perimeter_shaded_region circle_radius central_angle = 14 + (35 / 3) * Real.pi := by
  -- Unfold the definition of perimeter_shaded_region
  unfold perimeter_shaded_region
  -- Simplify the expression
  simp [circle_radius, central_angle]
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l293_29320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l293_29356

def custom_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 1 / (1 - a n)

theorem sequence_property (a : ℕ → ℚ) (h : custom_sequence a) (h8 : a 8 = 2) : a 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l293_29356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l293_29363

/-- The diameter of each cylindrical pipe in centimeters -/
def pipe_diameter : ℝ := 8

/-- The number of pipes in each crate, excluding the top layer -/
def pipes_per_crate : ℕ := 160

/-- The number of pipes in a full horizontal row -/
def pipes_per_row : ℕ := 10

/-- The number of full rows in each crate -/
def full_rows : ℕ := 16

/-- The height of Crate A's packing in centimeters -/
def height_crate_a : ℝ := full_rows * pipe_diameter + pipe_diameter

/-- The vertical spacing between pipe centers in Crate B -/
noncomputable def vertical_spacing_b : ℝ := pipe_diameter * Real.sqrt 3 / 2

/-- The height of Crate B's packing in centimeters -/
noncomputable def height_crate_b : ℝ := 2 * pipe_diameter + (full_rows - 1) * vertical_spacing_b

/-- The positive difference in total heights between Crate A and Crate B -/
noncomputable def height_difference : ℝ := |height_crate_a - height_crate_b|

theorem crate_height_difference :
  height_difference = 120 - 60 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l293_29363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_property_l293_29388

-- Define the function f and its inverse g
variable {α : Type*} [Group α] [AddGroup α]
variable (f : α → α) (g : α → α)

-- Define the property of f
variable (h : ∀ a b, f (a * b) = f a + f b)

-- Define g as the inverse of f
variable (inv : ∀ x, g (f x) = x)
variable (inv' : ∀ y, f (g y) = y)

-- The theorem to prove
theorem inverse_property (a b : α) : g (a + b) = g a * g b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_property_l293_29388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l293_29319

/-- Given two lines l₁ and l₂ in the real plane, defined by their equations
    l₁: ax + y + 1 = 0 and l₂: 2x + (a+1)y + 3 = 0, where a is a real number,
    prove that a = 1 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem parallel_lines_condition (a : ℝ) : 
  let l₁ := {p : ℝ × ℝ | a * p.1 + p.2 + 1 = 0}
  let l₂ := {p : ℝ × ℝ | 2 * p.1 + (a + 1) * p.2 + 3 = 0}
  (a = 1 → Set.EqOn (fun p => a * p.1 + p.2) (fun p => 2 * p.1 + (a + 1) * p.2) (Set.univ : Set (ℝ × ℝ))) ∧
  ¬(Set.EqOn (fun p => a * p.1 + p.2) (fun p => 2 * p.1 + (a + 1) * p.2) (Set.univ : Set (ℝ × ℝ)) → a = 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l293_29319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l293_29313

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_foci : c = 6
  h_relation : a^2 + b^2 = c^2

/-- The angle between the asymptotes of the hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := Real.arctan (2 * h.b / h.a)

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The theorem stating the properties of the hyperbola -/
theorem hyperbola_properties (h : Hyperbola) 
  (h_angle : asymptote_angle h = π/3) :
  ((h.a = 3*Real.sqrt 3 ∧ h.b = 3) ∨ (h.a = 3 ∧ h.b = 3*Real.sqrt 3)) ∧
  ((eccentricity h = 2*Real.sqrt 3/3) ∨ (eccentricity h = 2)) := by
  sorry

#check hyperbola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l293_29313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l293_29391

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1/2) :
  (20 * p^3 * (1-p)^3 = 5/32) → p = 1/2 - (Real.sqrt (16 - 8 * Real.rpow 5 (1/3))) / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_probability_l293_29391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l293_29335

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (k : ℕ) : ℤ :=
  k * (2 * a₁ + (k - 1) * d) / 2

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (k : ℕ) :
  a 1 = 1 ∧ a 3 = -3 →
  (∀ n : ℕ, a n = 3 - 2 * n) ∧
  (sum_arithmetic_sequence 1 (-2) k = -35 → k = 7) := by
  sorry

#check arithmetic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l293_29335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_plus_minus_one_l293_29324

noncomputable section

-- Define the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (3 + Real.sqrt 5 * Real.cos θ, Real.sqrt 5 * Real.sin θ)

-- Define the line l
def line_l (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem line_slope_is_plus_minus_one (α : ℝ) :
  ∃ (θ₁ θ₂ t₁ t₂ : ℝ),
    let A := curve_C θ₁
    let B := curve_C θ₂
    line_l α t₁ = A ∧
    line_l α t₂ = B ∧
    distance A B = 2 * Real.sqrt 3 →
    (Real.sin α) / (Real.cos α) = 1 ∨ (Real.sin α) / (Real.cos α) = -1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_is_plus_minus_one_l293_29324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l293_29333

noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 3)

theorem function_properties (A ω : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x : ℝ, f A ω (x + Real.pi) = f A ω x) →
  (f A ω (Real.pi / 3) = Real.sqrt 3 / 2) →
  (∃ α : ℝ, 0 < α ∧ α < Real.pi ∧ f A ω α + Real.sqrt 3 * f A ω (α - Real.pi / 2) = 1) →
  ((∀ x : ℝ, f A ω x = Real.sin (x + Real.pi / 3)) ∧
   (∃ α : ℝ, (α = Real.pi / 6 ∨ α = 5 * Real.pi / 6) ∧
              0 < α ∧ α < Real.pi ∧
              f A ω α + Real.sqrt 3 * f A ω (α - Real.pi / 2) = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l293_29333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_specific_lines_l293_29361

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines and x-axis from x=0 to x=8 -/
noncomputable def areaBetweenLines (l1 l2 : Line) : ℝ :=
  sorry

theorem area_between_specific_lines :
  let line1 : Line := ⟨0, 3, 9, 0⟩
  let line2 : Line := ⟨2, 6, 8, 1⟩
  areaBetweenLines line1 line2 = 7 := by
  sorry

#check area_between_specific_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_specific_lines_l293_29361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_NaBrO3_approx_l293_29308

noncomputable section

/-- Molar mass of sodium in g/mol -/
def molar_mass_Na : ℝ := 22.99

/-- Molar mass of bromine in g/mol -/
def molar_mass_Br : ℝ := 79.90

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of NaBrO3 in g/mol -/
def molar_mass_NaBrO3 : ℝ := molar_mass_Na + molar_mass_Br + 3 * molar_mass_O

/-- Mass of oxygen in one mole of NaBrO3 in g -/
def mass_O_in_NaBrO3 : ℝ := 3 * molar_mass_O

/-- Mass percentage of oxygen in NaBrO3 -/
def mass_percentage_O_in_NaBrO3 : ℝ := (mass_O_in_NaBrO3 / molar_mass_NaBrO3) * 100

theorem mass_percentage_O_in_NaBrO3_approx :
  ∃ ε > 0, |mass_percentage_O_in_NaBrO3 - 31.81| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_NaBrO3_approx_l293_29308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l293_29327

def expression (x : ℝ) : ℝ :=
  5 * (x - 2 * x^3) - 4 * (2 * x^2 - 3 * x^3 + x^6) + 3 * (3 * x^2 - x^10)

theorem coefficient_of_x_squared :
  deriv (deriv expression) 0 / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l293_29327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_playing_theater_l293_29347

theorem boys_playing_theater (total_friends : ℕ) 
  (girl_ratio : ℚ) (theater_ratio : ℚ) : ℕ :=
  let total_friends := 12
  let girl_ratio := 2/3
  let theater_ratio := 3/4
  let boys_playing_theater := 3
  boys_playing_theater

#check boys_playing_theater

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_playing_theater_l293_29347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_for_two_extrema_inequality_for_extrema_points_l293_29360

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

theorem tangent_line_at_one (h : f 1 1 = -1/2) :
  ∀ x, f 1 x = -1/2 := by sorry

theorem range_of_a_for_two_extrema (a : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ (∀ x, g a x ≤ g a x₁ ∨ g a x ≤ g a x₂)) →
  (0 < a ∧ a < Real.exp (-1)) := by sorry

theorem inequality_for_extrema_points (a : ℝ)
  (h : ∃ x₁ x₂, x₁ ≠ x₂ ∧ (∀ x, g a x ≤ g a x₁ ∨ g a x ≤ g a x₂)) :
  ∀ x₁ x₂, x₁ ≠ x₂ → (∀ x, g a x ≤ g a x₁ ∨ g a x ≤ g a x₂) →
  1 / Real.log x₁ + 1 / Real.log x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_for_two_extrema_inequality_for_extrema_points_l293_29360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_amount_l293_29309

/-- Represents the amount of a substance in moles -/
def Moles : Type := ℝ

/-- Represents the chemical reaction between Methane and Benzene to form Toluene and Hydrogen -/
structure Reaction where
  methane : Moles
  benzene : Moles
  toluene : Moles
  hydrogen : Moles

/-- The reaction is balanced when the input and output moles are equal -/
def is_balanced (r : Reaction) : Prop :=
  r.methane = r.benzene ∧ r.methane = r.toluene ∧ r.methane = r.hydrogen

/-- The theorem stating the amount of Benzene needed for the given reaction -/
theorem benzene_amount (r : Reaction) 
  (h1 : is_balanced r) 
  (h2 : r.methane = (2 : ℝ)) 
  (h3 : r.toluene = (2 : ℝ)) 
  (h4 : r.hydrogen = (2 : ℝ)) : 
  r.benzene = (2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_benzene_amount_l293_29309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l293_29350

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(i) for i from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) odd -/
def a : ℕ := sorry

/-- The number of positive integers n ≤ 1000 with S(n) even -/
def b : ℕ := sorry

theorem divisor_sum_parity_difference : |Int.ofNat a - Int.ofNat b| = 61 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l293_29350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_sides_l293_29374

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem min_product_of_sides (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 2 * sin A + sin B = 2 * sin C * cos B)
  (h_area : (Real.sqrt 3 / 2) * c = (1 / 2) * a * b * sin C) :
  a * b ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_sides_l293_29374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l293_29394

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)

theorem omega_value (ω : ℝ) (h1 : 0 < ω) (h2 : ω < Real.pi) 
  (h3 : ∀ x, f ω (2 + x) = f ω (2 - x)) : 
  ω = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l293_29394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l293_29392

noncomputable def f (x : ℝ) := Real.sin (Real.pi / 3 + 4 * x) + Real.cos (4 * x - Real.pi / 6)

theorem f_properties :
  let period : ℝ := Real.pi / 2
  let increasing_interval (k : ℤ) := Set.Icc (-5 * Real.pi / 24 + k * Real.pi / 2) (Real.pi / 24 + k * Real.pi / 2)
  let decreasing_interval (k : ℤ) := Set.Icc (Real.pi / 24 + k * Real.pi / 2) (7 * Real.pi / 24 + k * Real.pi / 2)
  let domain : Set ℝ := Set.Icc 0 (Real.pi / 4)
  (∀ x, f (x + period) = f x) ∧ 
  (∀ p, p > 0 → (∀ x, f (x + p) = f x) → p ≥ period) ∧
  (∀ k : ℤ, StrictMonoOn f (increasing_interval k)) ∧
  (∀ k : ℤ, StrictAntiOn f (decreasing_interval k)) ∧
  (∃ x ∈ domain, f x = 2 ∧ ∀ y ∈ domain, f y ≤ f x) ∧
  (∃ x ∈ domain, f x = -Real.sqrt 3 ∧ ∀ y ∈ domain, f y ≥ f x) ∧
  (∀ x ∈ domain, f x ≤ 2 ∧ f x ≥ -Real.sqrt 3) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l293_29392
