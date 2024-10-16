import Mathlib

namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3201_320196

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 10) ∧
  (Real.log p / Real.log 8 = Real.log (p^2 + q) / Real.log 20) →
  p^2 / q = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3201_320196


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l3201_320125

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l3201_320125


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l3201_320110

/-- 
Given two arithmetic progressions:
1) {5, 9, 13, 17, ...} with common difference 4
2) {4, 12, 20, 28, ...} with common difference 8
This theorem states that their largest common value less than 1000 is 993.
-/
theorem largest_common_value_less_than_1000 :
  let seq1 := fun n : ℕ => 5 + 4 * n
  let seq2 := fun n : ℕ => 4 + 8 * n
  ∃ (k1 k2 : ℕ), seq1 k1 = seq2 k2 ∧ 
                 seq1 k1 < 1000 ∧
                 ∀ (m1 m2 : ℕ), seq1 m1 = seq2 m2 → seq1 m1 < 1000 → seq1 m1 ≤ seq1 k1 ∧
                 seq1 k1 = 993 :=
by sorry


end NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l3201_320110


namespace NUMINAMATH_CALUDE_money_distribution_l3201_320188

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 700)
  (ac_sum : A + C = 300)
  (bc_sum : B + C = 600) :
  C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3201_320188


namespace NUMINAMATH_CALUDE_prob_ace_king_queen_value_l3201_320194

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing Ace, King, Queen in order without replacement -/
def prob_ace_king_queen : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  NumKings / (StandardDeck - 1) *
  NumQueens / (StandardDeck - 2)

theorem prob_ace_king_queen_value :
  prob_ace_king_queen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_queen_value_l3201_320194


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3201_320149

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3201_320149


namespace NUMINAMATH_CALUDE_side_c_length_l3201_320165

-- Define the triangle ABC
def triangle_ABC (A B C a b c : Real) : Prop :=
  -- Angles sum to 180°
  A + B + C = Real.pi ∧
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Sine law
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = c / Real.sin C

-- Theorem statement
theorem side_c_length :
  ∀ (A B C a b c : Real),
    triangle_ABC A B C a b c →
    A = Real.pi / 6 →  -- 30°
    B = 7 * Real.pi / 12 →  -- 105°
    a = 2 →
    c = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_side_c_length_l3201_320165


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3201_320154

def I : Set Nat := {1, 2, 3, 4}
def S : Set Nat := {1, 3}
def T : Set Nat := {4}

theorem complement_union_theorem :
  (I \ S) ∪ T = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3201_320154


namespace NUMINAMATH_CALUDE_classroom_size_l3201_320187

theorem classroom_size :
  ∀ (initial_students : ℕ),
  (0.4 * initial_students : ℝ) = (0.32 * (initial_students + 5) : ℝ) →
  initial_students = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_size_l3201_320187


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l3201_320158

/-- A parabola with equation x = ay² + by + c, vertex at (3, -6), and passing through (2, -4) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_condition : 3 = a * (-6)^2 + b * (-6) + c
  point_condition : 2 = a * (-4)^2 + b * (-4) + c

/-- The sum of coefficients a, b, and c for the given parabola is -25/4 -/
theorem parabola_coefficient_sum (p : Parabola) : p.a + p.b + p.c = -25/4 := by
  sorry

#check parabola_coefficient_sum

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l3201_320158


namespace NUMINAMATH_CALUDE_division_properties_l3201_320166

theorem division_properties (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (¬ (a ∣ b^2 ↔ a ∣ b)) ∧ (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_division_properties_l3201_320166


namespace NUMINAMATH_CALUDE_eesha_late_arrival_l3201_320115

/-- Eesha's commute problem -/
theorem eesha_late_arrival (usual_time : ℕ) (late_start : ℕ) (speed_reduction : ℚ) : 
  usual_time = 60 → late_start = 30 → speed_reduction = 1/4 →
  (usual_time : ℚ) / (1 - speed_reduction) + late_start - usual_time = 15 := by
  sorry

#check eesha_late_arrival

end NUMINAMATH_CALUDE_eesha_late_arrival_l3201_320115


namespace NUMINAMATH_CALUDE_yellow_packs_count_l3201_320148

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := sorry

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

theorem yellow_packs_count : yellow_packs = 4 := by
  have h1 : red_packs * balls_per_pack = yellow_packs * balls_per_pack + 18 := by sorry
  sorry

end NUMINAMATH_CALUDE_yellow_packs_count_l3201_320148


namespace NUMINAMATH_CALUDE_unique_function_is_identity_l3201_320199

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that f(mn) = f(m)f(n) for all positive integers m and n -/
def IsMultiplicative (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

/-- The property that f^(n^k)(n) = n for all positive integers n -/
def SatisfiesExpProperty (f : PositiveIntFunction) (k : ℕ+) : Prop :=
  ∀ n : ℕ+, (f^[n^k.val]) n = n

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := id

theorem unique_function_is_identity (k : ℕ+) :
  ∃! f : PositiveIntFunction, IsMultiplicative f ∧ SatisfiesExpProperty f k →
  f = identityFunction :=
sorry

end NUMINAMATH_CALUDE_unique_function_is_identity_l3201_320199


namespace NUMINAMATH_CALUDE_cakes_served_total_l3201_320190

/-- The number of cakes served over two days in a restaurant -/
theorem cakes_served_total (lunch_today : ℕ) (dinner_today : ℕ) (yesterday : ℕ)
  (h1 : lunch_today = 5)
  (h2 : dinner_today = 6)
  (h3 : yesterday = 3) :
  lunch_today + dinner_today + yesterday = 14 :=
by sorry

end NUMINAMATH_CALUDE_cakes_served_total_l3201_320190


namespace NUMINAMATH_CALUDE_solution_product_l3201_320128

theorem solution_product (p q : ℝ) : 
  (p - 6) * (3 * p + 12) = p^2 + 2 * p - 72 →
  (q - 6) * (3 * q + 12) = q^2 + 2 * q - 72 →
  p ≠ q →
  (p + 2) * (q + 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_product_l3201_320128


namespace NUMINAMATH_CALUDE_impossibleTiling_l3201_320153

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Function to check if a modified chessboard can be tiled with dominoes -/
def canTileChessboard (board : ModifiedChessboard) (tile : Domino) : Prop :=
  board.size = 8 ∧
  board.cornersRemoved = 2 ∧
  tile.length = 2 ∧
  tile.width = 1 ∧
  ∃ (tiling : Nat), False  -- This represents the impossibility of tiling

/-- Theorem stating that it's impossible to tile the modified chessboard with 2x1 dominoes -/
theorem impossibleTiling (board : ModifiedChessboard) (tile : Domino) :
  ¬(canTileChessboard board tile) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleTiling_l3201_320153


namespace NUMINAMATH_CALUDE_toy_production_rate_l3201_320123

/-- Represents the toy production in a factory --/
structure ToyFactory where
  weekly_production : ℕ
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ

/-- Calculates the hourly toy production rate --/
def hourly_production_rate (factory : ToyFactory) : ℚ :=
  let total_hours := factory.monday_hours + factory.tuesday_hours + factory.wednesday_hours + factory.thursday_hours
  factory.weekly_production / total_hours

/-- Theorem stating the hourly production rate for the given factory --/
theorem toy_production_rate (factory : ToyFactory) 
  (h1 : factory.weekly_production = 20500)
  (h2 : factory.monday_hours = 8)
  (h3 : factory.tuesday_hours = 7)
  (h4 : factory.wednesday_hours = 9)
  (h5 : factory.thursday_hours = 6) :
  ∃ (ε : ℚ), abs (hourly_production_rate factory - 683.33) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_toy_production_rate_l3201_320123


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l3201_320117

theorem tan_half_product_squared (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l3201_320117


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l3201_320184

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ,
    n > 1 ∧
    ¬ Prime n ∧
    (∀ p : ℕ, Prime p → p < 20 → ¬ p ∣ n) ∧
    (∀ m : ℕ, m > 1 → ¬ Prime m → (∀ q : ℕ, Prime q → q < 20 → ¬ q ∣ m) → m ≥ n) ∧
    500 < n ∧
    n ≤ 600 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l3201_320184


namespace NUMINAMATH_CALUDE_negation_equivalence_l3201_320186

theorem negation_equivalence : 
  (¬(∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ 
  (∀ x : ℝ, x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3201_320186


namespace NUMINAMATH_CALUDE_complex_number_relation_l3201_320104

theorem complex_number_relation (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 8)
  (eq5 : s + t + u = 4) :
  s * t * u = 10 := by
sorry


end NUMINAMATH_CALUDE_complex_number_relation_l3201_320104


namespace NUMINAMATH_CALUDE_inequality_proof_l3201_320146

theorem inequality_proof (x y z : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_bound : x < 2) (hy_bound : y < 2) (hz_bound : z < 2)
  (h_sum : x^2 + y^2 + z^2 = 3) : 
  (3/2 : ℝ) < (1+y^2)/(x+2) + (1+z^2)/(y+2) + (1+x^2)/(z+2) ∧ 
  (1+y^2)/(x+2) + (1+z^2)/(y+2) + (1+x^2)/(z+2) < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3201_320146


namespace NUMINAMATH_CALUDE_train_crossing_time_l3201_320135

/-- Proves that a train crossing a platform of its own length in 60 seconds
    will take 30 seconds to cross a signal pole. -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ)
  (h1 : train_length = 420)
  (h2 : platform_length = train_length)
  (h3 : platform_crossing_time = 60) :
  train_length / ((train_length + platform_length) / platform_crossing_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3201_320135


namespace NUMINAMATH_CALUDE_at_most_one_true_l3201_320103

theorem at_most_one_true (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_true_l3201_320103


namespace NUMINAMATH_CALUDE_simplify_expression_l3201_320156

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b - 4) - 2*b^2 = 9*b^3 + 4*b^2 - 12*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3201_320156


namespace NUMINAMATH_CALUDE_area_formula_correct_perimeter_formula_correct_l3201_320109

/-- Represents a figure composed of a square and a rectangle -/
structure CompositeFigure where
  a : ℝ
  h : a > 0

namespace CompositeFigure

/-- The area of the composite figure -/
def area (f : CompositeFigure) : ℝ := f.a^2 + 1.5 * f.a

/-- The perimeter of the composite figure -/
def perimeter (f : CompositeFigure) : ℝ := 4 * f.a + 3

/-- Theorem stating that the area formula is correct -/
theorem area_formula_correct (f : CompositeFigure) : 
  area f = f.a^2 + 1.5 * f.a := by sorry

/-- Theorem stating that the perimeter formula is correct -/
theorem perimeter_formula_correct (f : CompositeFigure) : 
  perimeter f = 4 * f.a + 3 := by sorry

end CompositeFigure

end NUMINAMATH_CALUDE_area_formula_correct_perimeter_formula_correct_l3201_320109


namespace NUMINAMATH_CALUDE_woman_birth_year_l3201_320118

/-- A woman born in the latter half of the nineteenth century was y years old in the year y^2. -/
theorem woman_birth_year (y : ℕ) (h1 : 1850 ≤ y^2 - y) (h2 : y^2 - y < 1900) (h3 : y^2 = y + 1892) : 
  y^2 - y = 1892 := by
  sorry

end NUMINAMATH_CALUDE_woman_birth_year_l3201_320118


namespace NUMINAMATH_CALUDE_total_colored_pencils_l3201_320136

/-- The number of colored pencils each person has -/
structure ColoredPencils where
  cheryl : ℕ
  cyrus : ℕ
  madeline : ℕ
  daniel : ℕ

/-- The conditions of the colored pencils problem -/
def colored_pencils_conditions (p : ColoredPencils) : Prop :=
  p.cheryl = 3 * p.cyrus ∧
  p.madeline = 63 ∧
  p.madeline * 2 = p.cheryl ∧
  p.daniel = ((p.cheryl + p.cyrus + p.madeline) * 25 + 99) / 100

/-- The theorem stating the total number of colored pencils -/
theorem total_colored_pencils (p : ColoredPencils) 
  (h : colored_pencils_conditions p) : 
  p.cheryl + p.cyrus + p.madeline + p.daniel = 289 := by
  sorry

end NUMINAMATH_CALUDE_total_colored_pencils_l3201_320136


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3201_320107

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 5 →
  n + k = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3201_320107


namespace NUMINAMATH_CALUDE_evaluate_g_l3201_320177

/-- The function g(x) = 3x^2 - 5x + 7 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

/-- Theorem: 3g(5) + 4g(-2) = 287 -/
theorem evaluate_g : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l3201_320177


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3201_320193

/-- The line equation kx - y + 1 - 3k = 0 passes through the point (3, 1) for all k. -/
theorem line_passes_through_point :
  ∀ (k : ℝ), k * 3 - 1 + 1 - 3 * k = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3201_320193


namespace NUMINAMATH_CALUDE_female_population_l3201_320124

theorem female_population (total_population : ℕ) (num_parts : ℕ) (female_parts : ℕ) : 
  total_population = 720 →
  num_parts = 4 →
  female_parts = 2 →
  (total_population / num_parts) * female_parts = 360 :=
by sorry

end NUMINAMATH_CALUDE_female_population_l3201_320124


namespace NUMINAMATH_CALUDE_impossibleRectangle_l3201_320167

/-- Represents the counts of sticks of each length -/
structure StickCounts where
  one_cm : Nat
  two_cm : Nat
  three_cm : Nat
  four_cm : Nat

/-- Calculates the total length of all sticks -/
def totalLength (counts : StickCounts) : Nat :=
  counts.one_cm * 1 + counts.two_cm * 2 + counts.three_cm * 3 + counts.four_cm * 4

/-- Theorem stating that it's impossible to form a rectangle with the given sticks -/
theorem impossibleRectangle (counts : StickCounts) 
  (h1 : counts.one_cm = 4)
  (h2 : counts.two_cm = 4)
  (h3 : counts.three_cm = 7)
  (h4 : counts.four_cm = 5)
  (h5 : totalLength counts = 53) :
  ¬∃ (a b : Nat), a + b = (totalLength counts) / 2 := by
  sorry

#eval totalLength { one_cm := 4, two_cm := 4, three_cm := 7, four_cm := 5 }

end NUMINAMATH_CALUDE_impossibleRectangle_l3201_320167


namespace NUMINAMATH_CALUDE_evaluate_expression_l3201_320143

theorem evaluate_expression : -(18 / 3 * 8 - 72 + 4 * 8) = 8 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3201_320143


namespace NUMINAMATH_CALUDE_range_of_m_l3201_320120

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - a*x + 2

theorem range_of_m (a : ℝ) (x₀ : ℝ) :
  (∀ a ∈ Set.Icc (-2) 0, ∃ x₀ ∈ Set.Ioc 0 1, 
    f x₀ a > a^2 + 3*a + 2 - 2*m*(Real.exp a)*(a+1)) →
  m ∈ Set.Icc (-1/2) (5*(Real.exp 2)/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3201_320120


namespace NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l3201_320147

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def primesBetween1And15 : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15 ∧ isPrime n}

theorem short_bingo_first_column_possibilities :
  Nat.factorial 6 / Nat.factorial 1 = 720 :=
sorry

end NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l3201_320147


namespace NUMINAMATH_CALUDE_subscription_ratio_l3201_320130

/-- Represents the number of magazine subscriptions sold to different people --/
structure Subscriptions where
  parents : ℕ
  grandfather : ℕ
  nextDoorNeighbor : ℕ
  otherNeighbor : ℕ

/-- Calculates the total earnings from selling subscriptions --/
def totalEarnings (s : Subscriptions) (pricePerSubscription : ℕ) : ℕ :=
  (s.parents + s.grandfather + s.nextDoorNeighbor + s.otherNeighbor) * pricePerSubscription

/-- Theorem stating the ratio of subscriptions sold to other neighbor vs next-door neighbor --/
theorem subscription_ratio (s : Subscriptions) (pricePerSubscription totalEarned : ℕ) :
  s.parents = 4 →
  s.grandfather = 1 →
  s.nextDoorNeighbor = 2 →
  pricePerSubscription = 5 →
  totalEarnings s pricePerSubscription = totalEarned →
  totalEarned = 55 →
  s.otherNeighbor = 2 * s.nextDoorNeighbor :=
by sorry


end NUMINAMATH_CALUDE_subscription_ratio_l3201_320130


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_primes_l3201_320164

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

theorem remainder_of_sum_of_primes :
  (3 * (List.sum (List.take 7 first_eight_primes))) % (List.get! first_eight_primes 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_primes_l3201_320164


namespace NUMINAMATH_CALUDE_polyhedron_exists_l3201_320150

-- Define a custom type for vertices
inductive Vertex : Type
  | A | B | C | D | E | F | G | H

-- Define an edge as a pair of vertices
def Edge : Type := Vertex × Vertex

-- Define the list of edges
def edgeList : List Edge :=
  [(Vertex.A, Vertex.B), (Vertex.A, Vertex.C), (Vertex.B, Vertex.C),
   (Vertex.B, Vertex.D), (Vertex.C, Vertex.D), (Vertex.D, Vertex.E),
   (Vertex.E, Vertex.F), (Vertex.E, Vertex.G), (Vertex.F, Vertex.G),
   (Vertex.F, Vertex.H), (Vertex.G, Vertex.H), (Vertex.A, Vertex.H)]

-- Define a polyhedron as a list of edges
def Polyhedron : Type := List Edge

-- Theorem: There exists a polyhedron with the given list of edges
theorem polyhedron_exists : ∃ (p : Polyhedron), p = edgeList := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_exists_l3201_320150


namespace NUMINAMATH_CALUDE_math_books_count_l3201_320182

theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℚ) (total_price : ℚ) :
  total_books = 80 →
  math_book_price = 4 →
  history_book_price = 5 →
  total_price = 368 →
  ∃ (math_books : ℕ), 
    math_books ≤ total_books ∧
    math_book_price * math_books + history_book_price * (total_books - math_books) = total_price ∧
    math_books = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_math_books_count_l3201_320182


namespace NUMINAMATH_CALUDE_susie_large_rooms_l3201_320100

/-- Represents the number of rooms of each size in Susie's house. -/
structure RoomCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the time needed to vacuum each type of room. -/
structure VacuumTimes where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total vacuuming time for all rooms. -/
def totalVacuumTime (counts : RoomCounts) (times : VacuumTimes) : Nat :=
  counts.small * times.small + counts.medium * times.medium + counts.large * times.large

/-- The theorem stating that given the conditions, Susie has 2 large rooms. -/
theorem susie_large_rooms : 
  ∀ (counts : RoomCounts) (times : VacuumTimes),
    counts.small = 4 →
    counts.medium = 3 →
    times.small = 15 →
    times.medium = 25 →
    times.large = 35 →
    totalVacuumTime counts times = 225 →
    counts.large = 2 := by
  sorry

end NUMINAMATH_CALUDE_susie_large_rooms_l3201_320100


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3201_320152

theorem polygon_diagonals (n : ℕ) (h : n > 0) : 
  3 * n = n * (n - 3) / 2 ↔ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3201_320152


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3201_320121

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3201_320121


namespace NUMINAMATH_CALUDE_one_and_two_thirds_problem_l3201_320131

theorem one_and_two_thirds_problem (x : ℝ) : (5 / 3 : ℝ) * x = 36 → x = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_problem_l3201_320131


namespace NUMINAMATH_CALUDE_whatsapp_messages_theorem_l3201_320160

/-- The number of messages sent on Monday in a Whatsapp group -/
def monday_messages : ℕ := sorry

/-- The number of messages sent on Tuesday in a Whatsapp group -/
def tuesday_messages : ℕ := 200

/-- The number of messages sent on Wednesday in a Whatsapp group -/
def wednesday_messages : ℕ := tuesday_messages + 300

/-- The number of messages sent on Thursday in a Whatsapp group -/
def thursday_messages : ℕ := 2 * wednesday_messages

/-- The total number of messages sent over four days in a Whatsapp group -/
def total_messages : ℕ := 2000

theorem whatsapp_messages_theorem :
  monday_messages + tuesday_messages + wednesday_messages + thursday_messages = total_messages ∧
  monday_messages = 300 := by sorry

end NUMINAMATH_CALUDE_whatsapp_messages_theorem_l3201_320160


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3201_320113

theorem circle_diameter_ratio (D C : Real) (h1 : D = 20) 
  (h2 : C > 0) (h3 : C < D) 
  (h4 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4) : 
  C = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3201_320113


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2050_l3201_320170

theorem sum_of_tens_and_units_digits_of_9_pow_2050 :
  ∃ (n : ℕ), 9^2050 = 10000 * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_pow_2050_l3201_320170


namespace NUMINAMATH_CALUDE_cole_drive_to_work_time_l3201_320173

/-- The time taken for Cole to drive to work, given his speeds and total round trip time -/
theorem cole_drive_to_work_time 
  (speed_to_work : ℝ) 
  (speed_from_work : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_to_work = 75) 
  (h2 : speed_from_work = 105) 
  (h3 : total_time = 1) : 
  ∃ (distance : ℝ), 
    distance / speed_to_work + distance / speed_from_work = total_time ∧ 
    (distance / speed_to_work) * 60 = 35 :=
by sorry

end NUMINAMATH_CALUDE_cole_drive_to_work_time_l3201_320173


namespace NUMINAMATH_CALUDE_set_inclusion_theorem_l3201_320189

def A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

theorem set_inclusion_theorem :
  (∀ x ∈ B, x ∈ A 1) ∧
  (∀ a : ℝ, (∀ x ∈ A a, x ∈ B) ↔ a < -8 ∨ a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_theorem_l3201_320189


namespace NUMINAMATH_CALUDE_x_value_proof_l3201_320155

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (144 : ℝ)^(1/7) :=
by sorry

end NUMINAMATH_CALUDE_x_value_proof_l3201_320155


namespace NUMINAMATH_CALUDE_ceiling_minus_half_integer_l3201_320101

theorem ceiling_minus_half_integer (n : ℤ) : 
  let x : ℝ := n + 1/2
  ⌈x⌉ - x = 1/2 := by sorry

end NUMINAMATH_CALUDE_ceiling_minus_half_integer_l3201_320101


namespace NUMINAMATH_CALUDE_train_final_speed_l3201_320183

/-- Given a train with the following properties:
  * Length: 360 meters
  * Initial velocity: 0 m/s (starts from rest)
  * Acceleration: 1 m/s²
  * Time to cross a man on the platform: 20 seconds
Prove that the final speed of the train is 20 m/s. -/
theorem train_final_speed
  (length : ℝ)
  (initial_velocity : ℝ)
  (acceleration : ℝ)
  (time : ℝ)
  (h1 : length = 360)
  (h2 : initial_velocity = 0)
  (h3 : acceleration = 1)
  (h4 : time = 20)
  : initial_velocity + acceleration * time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_final_speed_l3201_320183


namespace NUMINAMATH_CALUDE_complex_equal_modulus_unequal_square_exists_l3201_320112

theorem complex_equal_modulus_unequal_square_exists : 
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equal_modulus_unequal_square_exists_l3201_320112


namespace NUMINAMATH_CALUDE_flour_recipe_total_l3201_320157

/-- The amount of flour required for Mary's cake recipe -/
def flour_recipe (flour_added : ℕ) (flour_to_add : ℕ) : ℕ :=
  flour_added + flour_to_add

/-- Theorem: The total amount of flour required by the recipe is equal to 
    the sum of the flour already added and the flour still to be added -/
theorem flour_recipe_total (flour_added flour_to_add : ℕ) :
  flour_recipe flour_added flour_to_add = flour_added + flour_to_add :=
by
  sorry

#eval flour_recipe 3 6  -- Should evaluate to 9

end NUMINAMATH_CALUDE_flour_recipe_total_l3201_320157


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l3201_320127

/-- Given a mixture of pure water and salt solution, find the volume of salt solution needed. -/
theorem salt_solution_mixture (x : ℝ) : 
  (0.15 * (1 + x) = 0.45 * x) → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l3201_320127


namespace NUMINAMATH_CALUDE_female_students_count_l3201_320145

/-- Represents the class configuration described in the problem -/
structure ClassConfiguration where
  total_students : Nat
  male_students : Nat
  (total_ge_male : total_students ≥ male_students)

/-- The number of students called by the kth student -/
def students_called (k : Nat) : Nat := k + 2

/-- The theorem statement -/
theorem female_students_count (c : ClassConfiguration) 
  (h1 : c.total_students = 42)
  (h2 : ∀ k, k ≤ c.male_students → students_called k ≤ c.total_students)
  (h3 : students_called c.male_students = c.total_students / 2) :
  c.total_students - c.male_students = 23 := by
  sorry


end NUMINAMATH_CALUDE_female_students_count_l3201_320145


namespace NUMINAMATH_CALUDE_abs_complex_value_l3201_320159

theorem abs_complex_value : Complex.abs (-3 - (9/4)*Complex.I) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_abs_complex_value_l3201_320159


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l3201_320172

theorem range_of_2a_minus_b (a b : ℝ) (ha : -1 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 4) :
  (∀ x, 2 * a - b ≤ x → x ≤ 4) ∧ (∀ y, -6 ≤ y → y ≤ 2 * a - b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l3201_320172


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3201_320114

-- Problem 1
theorem factorization_problem_1 (y : ℝ) : y^3 - y^2 + (1/4)*y = y*(y - 1/2)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (m n : ℝ) : m^4 - n^4 = (m - n)*(m + n)*(m^2 + n^2) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3201_320114


namespace NUMINAMATH_CALUDE_not_coprime_sum_equal_l3201_320105

/-- For any two natural numbers a and b, if a+n and b+n are not coprime for all natural numbers n, then a = b. -/
theorem not_coprime_sum_equal (a b : ℕ) 
  (h : ∀ n : ℕ, ¬ Nat.Coprime (a + n) (b + n)) : 
  a = b := by
  sorry

end NUMINAMATH_CALUDE_not_coprime_sum_equal_l3201_320105


namespace NUMINAMATH_CALUDE_fiona_cleaning_time_l3201_320191

/-- Given a total cleaning time and the fraction of time one person spends cleaning,
    calculate the time the other person spends cleaning in minutes. -/
def cleaning_time (total_hours : ℝ) (first_person_fraction : ℝ) : ℝ :=
  (total_hours * (1 - first_person_fraction)) * 60

/-- Theorem: When the total cleaning time is 8 hours and one person cleans for 1/4 of the time,
    the other person cleans for 360 minutes. -/
theorem fiona_cleaning_time :
  cleaning_time 8 (1/4) = 360 := by
  sorry

end NUMINAMATH_CALUDE_fiona_cleaning_time_l3201_320191


namespace NUMINAMATH_CALUDE_road_vehicles_l3201_320116

/-- Given a road with the specified conditions, prove the total number of vehicles -/
theorem road_vehicles (lanes : Nat) (trucks_per_lane : Nat) (cars_multiplier : Nat) : 
  lanes = 4 → 
  trucks_per_lane = 60 → 
  cars_multiplier = 2 →
  (lanes * trucks_per_lane + lanes * cars_multiplier * lanes * trucks_per_lane) = 2160 := by
  sorry

#check road_vehicles

end NUMINAMATH_CALUDE_road_vehicles_l3201_320116


namespace NUMINAMATH_CALUDE_circle_intersection_l3201_320185

/-- The number of intersection points between two circles -/
def intersectionPoints (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℕ :=
  sorry

/-- Theorem: The circle centered at (0, 3) with radius 3 and the circle centered at (5, 0) with radius 5 intersect at 4 points -/
theorem circle_intersection :
  intersectionPoints (0, 3) (5, 0) 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_l3201_320185


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3201_320129

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3201_320129


namespace NUMINAMATH_CALUDE_team_omega_score_l3201_320139

/-- Given a basketball match between Team Alpha and Team Omega where:
  - The total points scored by both teams is 60
  - Team Alpha won by a margin of 12 points
  This theorem proves that Team Omega scored 24 points. -/
theorem team_omega_score (total_points : ℕ) (margin : ℕ) 
  (h1 : total_points = 60) 
  (h2 : margin = 12) : 
  (total_points - margin) / 2 = 24 := by
  sorry

#check team_omega_score

end NUMINAMATH_CALUDE_team_omega_score_l3201_320139


namespace NUMINAMATH_CALUDE_combine_terms_power_l3201_320138

/-- Given that two terms can be combined, prove that m^n = 8 -/
theorem combine_terms_power (a b c m n : ℕ) : 
  (∃ k : ℚ, k * a^m * b^3 * c^4 = -3 * a^2 * b^n * c^4) → m^n = 8 := by
sorry

end NUMINAMATH_CALUDE_combine_terms_power_l3201_320138


namespace NUMINAMATH_CALUDE_zoe_chocolate_sales_l3201_320176

/-- Given a box of chocolate bars, calculate the money made from selling a certain number of bars. -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Prove that Zoe made $42 by selling all but 6 bars from a box of 13 bars, each costing $6. -/
theorem zoe_chocolate_sales : money_made 13 6 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_zoe_chocolate_sales_l3201_320176


namespace NUMINAMATH_CALUDE_initial_friends_correct_l3201_320119

/-- The number of friends James had initially -/
def initial_friends : ℕ := 20

/-- The number of friends James lost due to an argument -/
def friends_lost : ℕ := 2

/-- The number of new friends James made -/
def new_friends : ℕ := 1

/-- The number of friends James has now -/
def current_friends : ℕ := 19

/-- Theorem stating that the initial number of friends is correct given the conditions -/
theorem initial_friends_correct :
  initial_friends = current_friends + friends_lost - new_friends :=
by sorry

end NUMINAMATH_CALUDE_initial_friends_correct_l3201_320119


namespace NUMINAMATH_CALUDE_journey_length_l3201_320174

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total +        -- First part (dirt road)
  30 +                         -- Second part (highway)
  (1 / 7 : ℚ) * total =        -- Third part (city street)
  total →                      -- Sum of all parts equals total
  total = 840 / 17 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l3201_320174


namespace NUMINAMATH_CALUDE_tan_2alpha_l3201_320179

theorem tan_2alpha (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 5) :
  Real.tan (2 * α) = -4/7 := by sorry

end NUMINAMATH_CALUDE_tan_2alpha_l3201_320179


namespace NUMINAMATH_CALUDE_machine_working_time_l3201_320175

theorem machine_working_time 
  (total_shirts : ℕ) 
  (production_rate : ℕ) 
  (num_malfunctions : ℕ) 
  (malfunction_fix_time : ℕ) 
  (h1 : total_shirts = 360)
  (h2 : production_rate = 4)
  (h3 : num_malfunctions = 2)
  (h4 : malfunction_fix_time = 5) :
  (total_shirts / production_rate) + (num_malfunctions * malfunction_fix_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_l3201_320175


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3201_320163

/-- Time for a train to pass a jogger given their speeds, train length, and initial distance -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 210) 
  (h4 : initial_distance = 240) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3201_320163


namespace NUMINAMATH_CALUDE_solution_x_equals_two_l3201_320142

theorem solution_x_equals_two : 
  let x : ℝ := 2
  3 * x - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_x_equals_two_l3201_320142


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3201_320102

-- Define the systems of equations
def system1 (x y : ℚ) : Prop :=
  x - y = 3 ∧ 3 * x - 8 * y = 14

def system2 (x y : ℚ) : Prop :=
  3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33

-- Theorem for the first system
theorem solution_system1 :
  ∃ x y : ℚ, system1 x y ∧ x = 2 ∧ y = -1 :=
by sorry

-- Theorem for the second system
theorem solution_system2 :
  ∃ x y : ℚ, system2 x y ∧ x = 6 ∧ y = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3201_320102


namespace NUMINAMATH_CALUDE_employee_payment_l3201_320181

theorem employee_payment (total_payment x y z : ℝ) : 
  total_payment = 1000 →
  x = 1.2 * y →
  z = 0.8 * y →
  x + z = 600 →
  y = 300 := by sorry

end NUMINAMATH_CALUDE_employee_payment_l3201_320181


namespace NUMINAMATH_CALUDE_max_value_linear_program_l3201_320111

/-- Given a set of linear constraints, prove that the maximum value of the objective function is 2 -/
theorem max_value_linear_program :
  ∀ x y : ℝ,
  x + y ≥ 1 →
  2 * x - y ≤ 0 →
  3 * x - 2 * y + 2 ≥ 0 →
  (∀ x' y' : ℝ,
    x' + y' ≥ 1 →
    2 * x' - y' ≤ 0 →
    3 * x' - 2 * y' + 2 ≥ 0 →
    3 * x - y ≥ 3 * x' - y') →
  3 * x - y = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_linear_program_l3201_320111


namespace NUMINAMATH_CALUDE_average_of_9_15_N_l3201_320141

theorem average_of_9_15_N (N : ℝ) (h : 12 < N ∧ N < 22) :
  let avg := (9 + 15 + N) / 3
  avg = 12 ∨ avg = 15 := by
sorry

end NUMINAMATH_CALUDE_average_of_9_15_N_l3201_320141


namespace NUMINAMATH_CALUDE_admission_price_is_12_l3201_320180

/-- The admission price for the aqua park. -/
def admission_price : ℝ := sorry

/-- The price of the tour. -/
def tour_price : ℝ := 6

/-- The number of people in the first group (who take the tour). -/
def group1_size : ℕ := 10

/-- The number of people in the second group (who only pay admission). -/
def group2_size : ℕ := 5

/-- The total earnings of the aqua park. -/
def total_earnings : ℝ := 240

/-- Theorem stating that the admission price is $12 given the conditions. -/
theorem admission_price_is_12 :
  (group1_size : ℝ) * (admission_price + tour_price) + (group2_size : ℝ) * admission_price = total_earnings →
  admission_price = 12 := by
  sorry

end NUMINAMATH_CALUDE_admission_price_is_12_l3201_320180


namespace NUMINAMATH_CALUDE_angle_function_value_l3201_320108

open Real

/-- Given an angle α in the third quadrant and f(α) = (cos(π/2 + α) * cos(π - α)) / sin(π + α),
    if cos(α - 3π/2) = 1/5, then f(α) = 2√6/5 -/
theorem angle_function_value (α : ℝ) :
  π < α ∧ α < 3*π/2 →
  cos (α - 3*π/2) = 1/5 →
  (cos (π/2 + α) * cos (π - α)) / sin (π + α) = 2*Real.sqrt 6/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_function_value_l3201_320108


namespace NUMINAMATH_CALUDE_divisor_power_expression_l3201_320162

theorem divisor_power_expression (k : ℕ) : 
  (30 ^ k : ℕ) ∣ 929260 → 3 ^ k - k ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_expression_l3201_320162


namespace NUMINAMATH_CALUDE_dinner_task_assignments_l3201_320168

theorem dinner_task_assignments (n : ℕ) (h : n = 5) : 
  (n.choose 2) * ((n - 2).choose 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_dinner_task_assignments_l3201_320168


namespace NUMINAMATH_CALUDE_book_selling_price_l3201_320178

-- Define the cost price and profit rate
def cost_price : ℝ := 50
def profit_rate : ℝ := 0.20

-- Define the selling price function
def selling_price (cost : ℝ) (rate : ℝ) : ℝ :=
  cost * (1 + rate)

-- Theorem statement
theorem book_selling_price :
  selling_price cost_price profit_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_l3201_320178


namespace NUMINAMATH_CALUDE_bus_children_count_l3201_320134

/-- The number of children on a bus before and after a bus stop. -/
theorem bus_children_count (after : ℕ) (difference : ℕ) (before : ℕ) 
  (h1 : after = 18)
  (h2 : difference = 23)
  (h3 : before = after + difference) :
  before = 41 := by
  sorry

end NUMINAMATH_CALUDE_bus_children_count_l3201_320134


namespace NUMINAMATH_CALUDE_factor_expression_l3201_320151

theorem factor_expression (x : ℝ) : 3*x*(x-5) + 7*(x-5) - 2*(x-5) = (3*x+5)*(x-5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3201_320151


namespace NUMINAMATH_CALUDE_red_cars_count_l3201_320192

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 90 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / ratio_black * black_cars = 33 := by
sorry

end NUMINAMATH_CALUDE_red_cars_count_l3201_320192


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3201_320161

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem f_monotonicity_and_extrema :
  (∀ x y, -2 < x → x < y → f x < f y) ∧
  (∀ x y, x < y → y < -2 → f y < f x) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, f x ≤ f 0) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, -1 / Real.exp 2 ≤ f x) ∧
  f 0 = 1 ∧
  f (-2) = -1 / Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3201_320161


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3201_320140

theorem roots_sum_of_squares (a b : ℝ) : 
  a^2 - a - 2023 = 0 → b^2 - b - 2023 = 0 → a^2 + b^2 = 4047 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3201_320140


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3201_320195

/-- Proves that adding 3 liters of water to 11 liters of a 42% alcohol solution 
    results in a new mixture with 33% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 11 ∧ 
  initial_concentration = 0.42 ∧ 
  added_water = 3 ∧ 
  final_concentration = 0.33 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l3201_320195


namespace NUMINAMATH_CALUDE_max_triples_1955_l3201_320122

/-- The maximum number of triples that can be chosen from a set of points,
    such that each pair of triples has one point in common. -/
def max_triples (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2)) / 4

/-- Theorem stating that for 1955 points, the maximum number of triples
    that can be chosen such that each pair of triples has one point in
    common is 977. -/
theorem max_triples_1955 :
  max_triples 1955 = 977 := by
  sorry


end NUMINAMATH_CALUDE_max_triples_1955_l3201_320122


namespace NUMINAMATH_CALUDE_second_number_is_068_l3201_320171

/-- Represents a random number table as a list of natural numbers -/
def RandomNumberTable : List ℕ := [84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76]

/-- The total number of restaurants -/
def TotalRestaurants : ℕ := 200

/-- The number of restaurants to be selected -/
def SelectedRestaurants : ℕ := 5

/-- The starting column in the random number table -/
def StartColumn : ℕ := 5

/-- Function to select numbers from the random number table -/
def selectNumbers (table : List ℕ) (start : ℕ) (count : ℕ) : List ℕ :=
  (table.drop start).take count

/-- Theorem stating that the second selected number is 068 -/
theorem second_number_is_068 : 
  (selectNumbers RandomNumberTable StartColumn SelectedRestaurants)[1] = 68 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_068_l3201_320171


namespace NUMINAMATH_CALUDE_reflection_F_to_H_l3201_320126

/-- Represents the possible shapes in the problem -/
inductive Shape
  | F
  | E
  | H
  | Other

/-- Represents the types of reflections -/
inductive Reflection
  | Vertical
  | Horizontal

/-- Applies a reflection to a shape -/
def applyReflection (s : Shape) (r : Reflection) : Shape :=
  match s, r with
  | Shape.F, Reflection.Vertical => Shape.E
  | Shape.E, Reflection.Horizontal => Shape.H
  | _, _ => Shape.Other

/-- Theorem stating that applying vertical then horizontal reflection to F results in H -/
theorem reflection_F_to_H :
  applyReflection (applyReflection Shape.F Reflection.Vertical) Reflection.Horizontal = Shape.H :=
by sorry

end NUMINAMATH_CALUDE_reflection_F_to_H_l3201_320126


namespace NUMINAMATH_CALUDE_f_symmetry_g_zero_l3201_320169

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define g as the derivative of f
def g : ℝ → ℝ := f'

-- State the conditions
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorems to be proved
theorem f_symmetry : f (-1) = f 4 := by sorry

theorem g_zero : g (-1/2) = 0 := by sorry

end NUMINAMATH_CALUDE_f_symmetry_g_zero_l3201_320169


namespace NUMINAMATH_CALUDE_pipe_length_theorem_l3201_320198

theorem pipe_length_theorem (shorter_piece longer_piece total_length : ℝ) :
  longer_piece = 2 * shorter_piece →
  longer_piece = 118 →
  total_length = shorter_piece + longer_piece →
  total_length = 177 := by
  sorry

end NUMINAMATH_CALUDE_pipe_length_theorem_l3201_320198


namespace NUMINAMATH_CALUDE_no_integer_solution_l3201_320197

theorem no_integer_solution : ¬∃ (a b : ℤ), a * b * (a + b) = 20182017 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3201_320197


namespace NUMINAMATH_CALUDE_problem_solution_l3201_320137

theorem problem_solution (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : a + b > 0) :
  (a / b > -1) ∧ (abs a < abs b) ∧ ((a - 1) * (b - 1) < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3201_320137


namespace NUMINAMATH_CALUDE_square_of_1023_l3201_320133

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l3201_320133


namespace NUMINAMATH_CALUDE_laptop_price_l3201_320132

def original_price : ℝ → Prop :=
  λ x => (0.855 * x - 50) - (0.88 * x - 20) = 30

theorem laptop_price : ∃ x : ℝ, original_price x ∧ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l3201_320132


namespace NUMINAMATH_CALUDE_spinner_probability_l3201_320144

def spinner_A : Finset ℕ := {1, 2, 3}
def spinner_B : Finset ℕ := {2, 3, 4}

def is_multiple_of_four (n : ℕ) : Bool :=
  n % 4 = 0

def total_outcomes : ℕ :=
  (spinner_A.card) * (spinner_B.card)

def favorable_outcomes : ℕ :=
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (1 + b))).card +
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (2 + b))).card +
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (3 + b))).card

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3201_320144


namespace NUMINAMATH_CALUDE_bus_capacity_fraction_l3201_320106

/-- The capacity of the train in number of people -/
def train_capacity : ℕ := 120

/-- The combined capacity of the two buses in number of people -/
def combined_bus_capacity : ℕ := 40

/-- The fraction of the train's capacity that each bus can hold -/
def bus_fraction : ℚ := 1 / 6

theorem bus_capacity_fraction :
  bus_fraction = combined_bus_capacity / (2 * train_capacity) :=
sorry

end NUMINAMATH_CALUDE_bus_capacity_fraction_l3201_320106
