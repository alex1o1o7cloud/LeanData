import Mathlib

namespace NUMINAMATH_CALUDE_quarter_circle_arcs_sum_limit_l823_82313

/-- The sum of the lengths of quarter-circle arcs approaches πR/2 as n approaches infinity -/
theorem quarter_circle_arcs_sum_limit (R : ℝ) (h : R > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * R / (2 * n)) - π * R / 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_arcs_sum_limit_l823_82313


namespace NUMINAMATH_CALUDE_completing_square_result_l823_82329

theorem completing_square_result (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → ((x - 3)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l823_82329


namespace NUMINAMATH_CALUDE_sum_of_positive_reals_l823_82336

theorem sum_of_positive_reals (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_sq_xy : x^2 + y^2 = 2500)
  (sum_sq_zw : z^2 + w^2 = 2500)
  (prod_xz : x * z = 1200)
  (prod_yw : y * w = 1200) :
  x + y + z + w = 140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_reals_l823_82336


namespace NUMINAMATH_CALUDE_team_cost_comparison_l823_82316

/-- The cost calculation for Team A and Team B based on the number of people and ticket price --/
def cost_comparison (n : ℕ+) (x : ℝ) : Prop :=
  let cost_A := x + (3/4) * x * (n - 1)
  let cost_B := (4/5) * x * n
  (n = 5 → cost_A = cost_B) ∧
  (n > 5 → cost_A < cost_B) ∧
  (n < 5 → cost_A > cost_B)

/-- Theorem stating the cost comparison between Team A and Team B --/
theorem team_cost_comparison (n : ℕ+) (x : ℝ) (hx : x > 0) :
  cost_comparison n x := by
  sorry

end NUMINAMATH_CALUDE_team_cost_comparison_l823_82316


namespace NUMINAMATH_CALUDE_square_difference_l823_82376

theorem square_difference (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l823_82376


namespace NUMINAMATH_CALUDE_remainder_divisibility_l823_82362

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 20) → (∃ m : ℤ, N = 13 * m + 7) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l823_82362


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l823_82308

theorem quadratic_equivalence : ∀ x : ℝ, x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l823_82308


namespace NUMINAMATH_CALUDE_correct_operation_l823_82353

theorem correct_operation (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = -x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l823_82353


namespace NUMINAMATH_CALUDE_line_intersection_xz_plane_l823_82343

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_intersection_xz_plane (p₁ p₂ : ℝ × ℝ × ℝ) (intersection : ℝ × ℝ × ℝ) : 
  p₁ = (2, 3, 2) → 
  p₂ = (6, -1, 7) → 
  intersection.2 = 0 → 
  ∃ t : ℝ, intersection = (2 + 4*t, 3 - 4*t, 2 + 5*t) ∧ 
        intersection = (5, 0, 23/4) := by
  sorry

#check line_intersection_xz_plane

end NUMINAMATH_CALUDE_line_intersection_xz_plane_l823_82343


namespace NUMINAMATH_CALUDE_batsman_score_l823_82342

theorem batsman_score (T : ℝ) : 
  (5 * 4 + 5 * 6 : ℝ) + (2/3) * T = T → T = 150 := by sorry

end NUMINAMATH_CALUDE_batsman_score_l823_82342


namespace NUMINAMATH_CALUDE_fruit_purchase_problem_l823_82398

/-- Fruit purchase problem -/
theorem fruit_purchase_problem (x y : ℝ) :
  let apple_weight : ℝ := 2
  let orange_weight : ℝ := 5 * apple_weight
  let total_weight : ℝ := apple_weight + orange_weight
  let total_cost : ℝ := x * apple_weight + y * orange_weight
  (orange_weight = 10 ∧ total_cost = 2*x + 10*y) ∧ total_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_problem_l823_82398


namespace NUMINAMATH_CALUDE_decimal_23_equals_binary_10111_l823_82325

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_23_equals_binary_10111 :
  to_binary 23 = [true, true, true, false, true] ∧
  from_binary [true, true, true, false, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_decimal_23_equals_binary_10111_l823_82325


namespace NUMINAMATH_CALUDE_min_value_7x_5y_min_value_achieved_min_value_is_7_plus_2sqrt6_l823_82300

theorem min_value_7x_5y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (2 * x + y) + 4 / (x + y) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (2 * a + b) + 4 / (a + b) = 2 → 7 * x + 5 * y ≤ 7 * a + 5 * b :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (2 * x + y) + 4 / (x + y) = 2) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (2 * a + b) + 4 / (a + b) = 2 ∧ 7 * a + 5 * b = 7 + 2 * Real.sqrt 6 :=
by sorry

theorem min_value_is_7_plus_2sqrt6 :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / (2 * x + y) + 4 / (x + y) = 2 ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (2 * a + b) + 4 / (a + b) = 2 → 7 * x + 5 * y ≤ 7 * a + 5 * b) ∧
  7 * x + 5 * y = 7 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_7x_5y_min_value_achieved_min_value_is_7_plus_2sqrt6_l823_82300


namespace NUMINAMATH_CALUDE_score_ordering_l823_82377

-- Define the set of people
inductive Person : Type
| K : Person  -- Kaleana
| Q : Person  -- Quay
| M : Person  -- Marty
| S : Person  -- Shana

-- Define a function to represent the score of each person
variable (score : Person → ℕ)

-- Define the conditions
axiom quay_thought : score Person.Q = score Person.K
axiom marty_thought : score Person.M > score Person.K
axiom shana_thought : score Person.S < score Person.K

-- Define the theorem to prove
theorem score_ordering :
  score Person.S < score Person.Q ∧ score Person.Q < score Person.M :=
sorry

end NUMINAMATH_CALUDE_score_ordering_l823_82377


namespace NUMINAMATH_CALUDE_triangle_city_population_l823_82332

theorem triangle_city_population : ∃ (x y z : ℕ+), 
  x^2 + 50 = y^2 + 1 ∧ 
  y^2 + 351 = z^2 ∧ 
  x^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_triangle_city_population_l823_82332


namespace NUMINAMATH_CALUDE_domain_intersection_subset_l823_82379

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def C (m : ℝ) : Set ℝ := {x | 3*x < 2*m - 1}

-- State the theorem
theorem domain_intersection_subset (m : ℝ) : 
  (A ∩ B) ⊆ C m → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_domain_intersection_subset_l823_82379


namespace NUMINAMATH_CALUDE_circle_center_and_sum_l823_82315

/-- The equation of a circle in the form x² + y² = ax + by + c -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

def circle_equation : CircleEquation :=
  { a := 6, b := -10, c := 9 }

theorem circle_center_and_sum (eq : CircleEquation) :
  ∃ (center : CircleCenter), 
    center.x = eq.a / 2 ∧
    center.y = -eq.b / 2 ∧
    center.x + center.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_sum_l823_82315


namespace NUMINAMATH_CALUDE_is_valid_factorization_l823_82320

/-- Proves that x^2 - 2x + 1 = (x - 1)^2 is a valid factorization -/
theorem is_valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_is_valid_factorization_l823_82320


namespace NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l823_82314

theorem square_difference_divided (a b : ℕ) (h : a > b) : 
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (725^2 - 675^2) / 25 = 2800 :=
by 
  have h : 725 > 675 := by sorry
  have key := square_difference_divided 725 675 h
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_problem_solution_l823_82314


namespace NUMINAMATH_CALUDE_wire_cutting_l823_82341

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 49 →
  ratio = 2 / 5 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l823_82341


namespace NUMINAMATH_CALUDE_students_in_both_sports_l823_82319

theorem students_in_both_sports (total : ℕ) (baseball : ℕ) (hockey : ℕ) 
  (h1 : total = 36) (h2 : baseball = 25) (h3 : hockey = 19) :
  baseball + hockey - total = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_sports_l823_82319


namespace NUMINAMATH_CALUDE_flour_calculation_l823_82358

/-- Given a recipe for cookies, calculate the amount of each type of flour needed when doubling the recipe and using two types of flour. -/
theorem flour_calculation (original_cookies : ℕ) (original_flour : ℚ) (new_cookies : ℕ) :
  original_cookies > 0 →
  original_flour > 0 →
  new_cookies = 2 * original_cookies →
  ∃ (flour_each : ℚ),
    flour_each = original_flour ∧
    flour_each * 2 = new_cookies / original_cookies * original_flour :=
by sorry

end NUMINAMATH_CALUDE_flour_calculation_l823_82358


namespace NUMINAMATH_CALUDE_majorization_iff_transformable_l823_82317

/-- Represents a triplet of real numbers -/
structure Triplet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines majorization relation between two triplets -/
def majorizes (α β : Triplet) : Prop :=
  α.a ≥ β.a ∧ α.a + α.b ≥ β.a + β.b ∧ α.a + α.b + α.c = β.a + β.b + β.c

/-- Represents the allowed operations on triplets -/
inductive Operation
  | op1 : Operation  -- (k, j, i) ↔ (k-1, j+1, i)
  | op2 : Operation  -- (k, j, i) ↔ (k-1, j, i+1)
  | op3 : Operation  -- (k, j, i) ↔ (k, j-1, i+1)

/-- Applies an operation to a triplet -/
def applyOperation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.op1 => ⟨t.a - 1, t.b + 1, t.c⟩
  | Operation.op2 => ⟨t.a - 1, t.b, t.c + 1⟩
  | Operation.op3 => ⟨t.a, t.b - 1, t.c + 1⟩

/-- Checks if one triplet can be obtained from another using allowed operations -/
def canObtain (α β : Triplet) : Prop :=
  ∃ (ops : List Operation), β = ops.foldl applyOperation α

/-- Main theorem: Majorization is equivalent to ability to transform using allowed operations -/
theorem majorization_iff_transformable (α β : Triplet) :
  majorizes α β ↔ canObtain α β := by sorry

end NUMINAMATH_CALUDE_majorization_iff_transformable_l823_82317


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l823_82365

/-- Given a geometric sequence {a_n} with a_3 * a_7 = 8 and a_4 + a_6 = 6, prove that a_2 + a_8 = 9 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_prod : a 3 * a 7 = 8) (h_sum : a 4 + a 6 = 6) : 
  a 2 + a 8 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l823_82365


namespace NUMINAMATH_CALUDE_worker_delay_l823_82391

/-- Proves that reducing speed to 5/6 of normal results in a 12-minute delay -/
theorem worker_delay (usual_time : ℝ) (speed_ratio : ℝ) 
  (h1 : usual_time = 60)
  (h2 : speed_ratio = 5 / 6) : 
  (usual_time / speed_ratio) - usual_time = 12 := by
  sorry

#check worker_delay

end NUMINAMATH_CALUDE_worker_delay_l823_82391


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l823_82369

theorem nested_fraction_simplification :
  2 + 3 / (4 + 5 / 6) = 76 / 29 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l823_82369


namespace NUMINAMATH_CALUDE_tenth_term_is_18_l823_82378

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 = 2 ∧ 
  a 3 = 4

/-- The 10th term of the arithmetic sequence is 18 -/
theorem tenth_term_is_18 (a : ℕ → ℝ) (h : arithmetic_sequence a) : 
  a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_18_l823_82378


namespace NUMINAMATH_CALUDE_point_coordinates_l823_82349

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance from a point to x-axis and y-axis
def distToXAxis (p : Point2D) : ℝ := |p.y|
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- Define the set of possible coordinates
def possibleCoordinates : Set Point2D :=
  {⟨2, 1⟩, ⟨2, -1⟩, ⟨-2, 1⟩, ⟨-2, -1⟩}

-- Theorem statement
theorem point_coordinates (M : Point2D) :
  distToXAxis M = 1 ∧ distToYAxis M = 2 → M ∈ possibleCoordinates := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l823_82349


namespace NUMINAMATH_CALUDE_max_fibonacci_match_l823_82387

/-- A sequence that matches the Fibonacci sequence for a given number of terms -/
def MatchesFibonacci (t : ℕ → ℝ) (start : ℕ) (count : ℕ) : Prop :=
  ∀ k, k < count → t (start + k + 2) = t (start + k + 1) + t (start + k)

/-- The quadratic sequence defined by A, B, and C -/
def QuadraticSequence (A B C : ℝ) (n : ℕ) : ℝ :=
  A * (n : ℝ)^2 + B * (n : ℝ) + C

/-- The theorem stating the maximum number of consecutive Fibonacci terms -/
theorem max_fibonacci_match (A B C : ℝ) (h : A ≠ 0) :
  (∃ start, MatchesFibonacci (QuadraticSequence A B C) start 4) ∧
  (∀ start count, count > 4 → ¬MatchesFibonacci (QuadraticSequence A B C) start count) ∧
  ((A = 1/2 ∧ B = -1/2 ∧ C = 2) ∨ (A = 1/2 ∧ B = 1/2 ∧ C = 2)) :=
sorry

end NUMINAMATH_CALUDE_max_fibonacci_match_l823_82387


namespace NUMINAMATH_CALUDE_irrational_identification_l823_82309

theorem irrational_identification :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (5 : ℚ)^(1/3) = a / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (9 : ℚ)^(1/2) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-8/3 : ℚ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (60.25 : ℚ) = a / b) :=
by sorry

end NUMINAMATH_CALUDE_irrational_identification_l823_82309


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l823_82302

theorem inverse_proportion_ratio (c₁ c₂ d₁ d₂ : ℝ) : 
  c₁ ≠ 0 → c₂ ≠ 0 → d₁ ≠ 0 → d₂ ≠ 0 →
  (∃ k : ℝ, ∀ c d, c * d = k) →
  c₁ * d₁ = c₂ * d₂ →
  c₁ / c₂ = 3 / 4 →
  d₁ / d₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l823_82302


namespace NUMINAMATH_CALUDE_infinite_binary_sequences_and_powerset_cardinality_l823_82363

/-- The type of infinite binary sequences -/
def InfiniteBinarySequence := ℕ → Fin 2

/-- The cardinality of the continuum -/
def ContinuumCardinality := Cardinal.mk (Set ℝ)

theorem infinite_binary_sequences_and_powerset_cardinality :
  (Cardinal.mk (Set InfiniteBinarySequence) = ContinuumCardinality) ∧
  (Cardinal.mk (Set (Set ℕ)) = ContinuumCardinality) := by
  sorry

end NUMINAMATH_CALUDE_infinite_binary_sequences_and_powerset_cardinality_l823_82363


namespace NUMINAMATH_CALUDE_students_wanting_fruit_l823_82386

theorem students_wanting_fruit (red_apples green_apples extra_apples : ℕ) :
  red_apples = 43 →
  green_apples = 32 →
  extra_apples = 73 →
  (red_apples + green_apples + extra_apples) - (red_apples + green_apples) = extra_apples :=
by sorry

end NUMINAMATH_CALUDE_students_wanting_fruit_l823_82386


namespace NUMINAMATH_CALUDE_range_of_a_l823_82322

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l823_82322


namespace NUMINAMATH_CALUDE_intersection_and_trajectory_l823_82340

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define the line l passing through the origin
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the intersection of line l and circle C₁
def intersection (k x y : ℝ) : Prop := C₁ x y ∧ line_l k x y

-- Define the range of k for intersection
def k_range (k : ℝ) : Prop := -2 * Real.sqrt 5 / 5 ≤ k ∧ k ≤ 2 * Real.sqrt 5 / 5

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = 9/4 ∧ 5/3 < x ∧ x ≤ 3

-- Theorem statement
theorem intersection_and_trajectory :
  (∀ k, k_range k ↔ ∃ x y, intersection k x y) ∧
  (∀ k x₁ y₁ x₂ y₂,
    intersection k x₁ y₁ ∧ intersection k x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) →
    trajectory_M ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_trajectory_l823_82340


namespace NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l823_82394

/-- Given a function f(x) = ax^5 + bx^3 + sin(x) - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_of_two_equals_negative_twenty_six 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + Real.sin x - 8) 
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_f_of_two_equals_negative_twenty_six_l823_82394


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l823_82373

/-- Given that -x^2 + bx - 4 < 0 only when x ∈ (-∞, 0) ∪ (4, ∞), prove that b = 4 -/
theorem quadratic_inequality_roots (b : ℝ) 
  (h : ∀ x : ℝ, (-x^2 + b*x - 4 < 0) ↔ (x < 0 ∨ x > 4)) : 
  b = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l823_82373


namespace NUMINAMATH_CALUDE_discount_percentage_l823_82361

/-- The percentage discount for buying 3 pairs of shorts at once, given the regular price and savings -/
theorem discount_percentage
  (regular_price : ℚ)  -- Regular price of one pair of shorts
  (total_savings : ℚ)  -- Total savings when buying 3 pairs at once
  (h1 : regular_price = 10)  -- Each pair costs $10 normally
  (h2 : total_savings = 3)   -- Saving $3 by buying 3 pairs at once
  : (total_savings / (3 * regular_price)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l823_82361


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l823_82326

/-- Given an inverse proportion function y = k/x passing through (2, -6), prove k = -12 -/
theorem inverse_proportion_k_value : ∀ k : ℝ, 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f 2 = -6) → 
  k = -12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l823_82326


namespace NUMINAMATH_CALUDE_floor_product_l823_82371

theorem floor_product : ⌊(21.7 : ℝ)⌋ * ⌊(-21.7 : ℝ)⌋ = -462 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_l823_82371


namespace NUMINAMATH_CALUDE_soccer_league_games_l823_82380

theorem soccer_league_games (C D : ℕ) : 
  (3 * C = 4 * (C - (C / 4))) →  -- Team C has won 3/4 of its games
  (2 * (C + 6) = 3 * ((C + 6) - ((C + 6) / 3))) →  -- Team D has won 2/3 of its games
  (C + 6 = D) →  -- Team D has played 6 more games than team C
  (C = 12) :=  -- Prove that team C has played 12 games
by sorry

end NUMINAMATH_CALUDE_soccer_league_games_l823_82380


namespace NUMINAMATH_CALUDE_max_students_in_dance_l823_82356

theorem max_students_in_dance (x : ℕ) : 
  x < 100 ∧ 
  x % 8 = 5 ∧ 
  x % 5 = 3 →
  x ≤ 93 ∧ 
  ∃ y : ℕ, y = 93 ∧ 
    y < 100 ∧ 
    y % 8 = 5 ∧ 
    y % 5 = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_students_in_dance_l823_82356


namespace NUMINAMATH_CALUDE_rice_containers_l823_82301

theorem rice_containers (total_weight : ℚ) (container_capacity : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 25 / 4 →
  container_capacity = 25 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce / container_capacity : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_containers_l823_82301


namespace NUMINAMATH_CALUDE_f_min_at_neg_three_p_half_l823_82359

/-- The function f(x) = x^2 + 3px + 2p^2 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + 3*p*x + 2*p^2

/-- Theorem: The minimum of f(x) occurs at x = -3p/2 when p > 0 -/
theorem f_min_at_neg_three_p_half (p : ℝ) (h : p > 0) :
  ∀ x : ℝ, f p (-3*p/2) ≤ f p x :=
sorry

end NUMINAMATH_CALUDE_f_min_at_neg_three_p_half_l823_82359


namespace NUMINAMATH_CALUDE_solve_for_y_l823_82347

theorem solve_for_y (x y : ℝ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l823_82347


namespace NUMINAMATH_CALUDE_ryan_english_time_l823_82372

/-- The time Ryan spends on learning English, given the total time spent on learning
    English and Chinese, and the time spent on learning Chinese. -/
def time_learning_english (total_time : ℝ) (chinese_time : ℝ) : ℝ :=
  total_time - chinese_time

/-- Theorem stating that Ryan spends 2 hours learning English -/
theorem ryan_english_time :
  time_learning_english 3 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_time_l823_82372


namespace NUMINAMATH_CALUDE_immigrant_count_l823_82397

/-- The number of people born in the country last year -/
def births : ℕ := 90171

/-- The total number of new people who began living in the country last year -/
def total_new_people : ℕ := 106491

/-- The number of immigrants to the country last year -/
def immigrants : ℕ := total_new_people - births

theorem immigrant_count : immigrants = 16320 := by
  sorry

end NUMINAMATH_CALUDE_immigrant_count_l823_82397


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l823_82395

theorem fraction_equation_solution (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  (2 / x) - (1 / y) = (3 / z) → z = (2 * y - x) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l823_82395


namespace NUMINAMATH_CALUDE_jones_clothing_count_l823_82384

def pants_count : ℕ := 40
def shirts_per_pants : ℕ := 6
def ties_per_pants : ℕ := 5
def socks_per_shirt : ℕ := 3

def total_clothing : ℕ := 
  pants_count + 
  (pants_count * shirts_per_pants) + 
  (pants_count * ties_per_pants) + 
  (pants_count * shirts_per_pants * socks_per_shirt)

theorem jones_clothing_count : total_clothing = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jones_clothing_count_l823_82384


namespace NUMINAMATH_CALUDE_unique_friendship_configs_l823_82354

/-- Represents a friendship configuration in a group of 8 people --/
structure FriendshipConfig :=
  (num_friends : Nat)
  (valid : num_friends = 0 ∨ num_friends = 1 ∨ num_friends = 6)

/-- Counts the number of unique friendship configurations --/
def count_unique_configs : Nat :=
  sorry

/-- Theorem stating that the number of unique friendship configurations is 37 --/
theorem unique_friendship_configs :
  count_unique_configs = 37 :=
sorry

end NUMINAMATH_CALUDE_unique_friendship_configs_l823_82354


namespace NUMINAMATH_CALUDE_no_square_subdivision_l823_82312

theorem no_square_subdivision : ¬ ∃ (s : ℝ) (n : ℕ), 
  s > 0 ∧ n > 0 ∧ 
  ∃ (a : ℝ), a > 0 ∧ 
  s * s = n * (1/2 * a * a * Real.sqrt 3) ∧
  s = a * Real.sqrt 3 ∨ s = 2 * a ∨ s = 3 * a :=
sorry

end NUMINAMATH_CALUDE_no_square_subdivision_l823_82312


namespace NUMINAMATH_CALUDE_unique_tuple_l823_82324

def satisfies_condition (a : Fin 9 → ℕ+) : Prop :=
  ∀ i j k l, i < j → j < k → k ≤ 9 → l ≠ i → l ≠ j → l ≠ k → l ≤ 9 →
    a i + a j + a k + a l = 100

theorem unique_tuple : ∃! a : Fin 9 → ℕ+, satisfies_condition a := by
  sorry

end NUMINAMATH_CALUDE_unique_tuple_l823_82324


namespace NUMINAMATH_CALUDE_car_speed_proof_l823_82374

/-- Proves that a car traveling at 400 km/h takes 9 seconds to travel 1 kilometer,
    given that it takes 5 seconds longer than traveling 1 kilometer at 900 km/h. -/
theorem car_speed_proof (v : ℝ) (h1 : v > 0) :
  (1 / v) * 3600 = 9 ↔ v = 400 ∧ (1 / 900) * 3600 + 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_proof_l823_82374


namespace NUMINAMATH_CALUDE_forty_five_million_scientific_notation_l823_82392

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem forty_five_million_scientific_notation :
  toScientificNotation 45000000 = ScientificNotation.mk 4.5 7 sorry := by sorry

end NUMINAMATH_CALUDE_forty_five_million_scientific_notation_l823_82392


namespace NUMINAMATH_CALUDE_snack_cost_theorem_l823_82337

/-- The total cost of snacks bought by Robert and Teddy -/
def total_cost (pizza_price : ℕ) (pizza_quantity : ℕ) (drink_price : ℕ) (robert_drink_quantity : ℕ) (hamburger_price : ℕ) (hamburger_quantity : ℕ) (teddy_drink_quantity : ℕ) : ℕ :=
  pizza_price * pizza_quantity + 
  drink_price * robert_drink_quantity + 
  hamburger_price * hamburger_quantity + 
  drink_price * teddy_drink_quantity

theorem snack_cost_theorem : 
  total_cost 10 5 2 10 3 6 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_snack_cost_theorem_l823_82337


namespace NUMINAMATH_CALUDE_system_solution_l823_82330

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0 ∧
   3 * x^2 * y^2 + y^4 = 84) ↔
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l823_82330


namespace NUMINAMATH_CALUDE_game_cost_l823_82350

theorem game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_price : ℕ) (game_cost : ℕ) : 
  initial_money = 57 → 
  num_toys = 5 → 
  toy_price = 6 → 
  initial_money = game_cost + (num_toys * toy_price) → 
  game_cost = 27 := by
sorry

end NUMINAMATH_CALUDE_game_cost_l823_82350


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l823_82366

theorem hyperbola_standard_form (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 2 * a = 8) (h4 : (a^2 + b^2) / a^2 = (5/4)^2) :
  a = 4 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l823_82366


namespace NUMINAMATH_CALUDE_tire_cost_theorem_l823_82327

/-- Calculates the total cost of tires with given prices, discounts, and taxes -/
def totalTireCost (allTerrainPrice : ℝ) (allTerrainDiscount : ℝ) (allTerrainTax : ℝ)
                  (sparePrice : ℝ) (spareDiscount : ℝ) (spareTax : ℝ) : ℝ :=
  let allTerrainDiscountedPrice := allTerrainPrice * (1 - allTerrainDiscount)
  let allTerrainFinalPrice := allTerrainDiscountedPrice * (1 + allTerrainTax)
  let allTerrainTotal := 4 * allTerrainFinalPrice

  let spareDiscountedPrice := sparePrice * (1 - spareDiscount)
  let spareFinalPrice := spareDiscountedPrice * (1 + spareTax)

  allTerrainTotal + spareFinalPrice

/-- The total cost of tires is $291.20 -/
theorem tire_cost_theorem :
  totalTireCost 60 0.15 0.08 75 0.10 0.05 = 291.20 := by
  sorry

end NUMINAMATH_CALUDE_tire_cost_theorem_l823_82327


namespace NUMINAMATH_CALUDE_goldfish_pond_problem_l823_82321

theorem goldfish_pond_problem :
  ∀ (x : ℕ),
  (x > 0) →
  (3 * x / 7 : ℚ) + (4 * x / 7 : ℚ) = x →
  (5 * x / 8 : ℚ) + (3 * x / 8 : ℚ) = x →
  (5 * x / 8 : ℚ) - (3 * x / 7 : ℚ) = 33 →
  x = 168 := by
sorry

end NUMINAMATH_CALUDE_goldfish_pond_problem_l823_82321


namespace NUMINAMATH_CALUDE_negative_half_to_fourth_power_l823_82383

theorem negative_half_to_fourth_power :
  (-1/2 : ℚ)^4 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_to_fourth_power_l823_82383


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_and_inradius_l823_82335

theorem right_triangle_arithmetic_progression_and_inradius (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem
    a = 3*d ∧ 
    b = 4*d ∧ 
    c = 5*d ∧ 
    (a + b - c) / 2 = d  -- Inradius formula
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_and_inradius_l823_82335


namespace NUMINAMATH_CALUDE_sin_330_degrees_l823_82311

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l823_82311


namespace NUMINAMATH_CALUDE_expression_evaluation_l823_82339

theorem expression_evaluation (x : ℝ) (h1 : x^2 - 3*x + 2 = 0) (h2 : x ≠ 2) :
  (x^2 / (x - 2) - x - 2) / (4*x / (x^2 - 4)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l823_82339


namespace NUMINAMATH_CALUDE_waiter_tables_l823_82304

/-- Proves that a waiter with 40 customers and tables of 5 women and 3 men each has 5 tables. -/
theorem waiter_tables (total_customers : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  total_customers = 40 →
  women_per_table = 5 →
  men_per_table = 3 →
  total_customers = (women_per_table + men_per_table) * 5 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tables_l823_82304


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l823_82345

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition
  (a : ℕ → ℝ) (m p q : ℕ) (h : arithmetic_sequence a) :
  (∀ m p q : ℕ, p + q = 2 * m → a p + a q = 2 * a m) ∧
  (∃ m p q : ℕ, a p + a q = 2 * a m ∧ p + q ≠ 2 * m) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l823_82345


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l823_82360

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l823_82360


namespace NUMINAMATH_CALUDE_fraction_product_l823_82303

theorem fraction_product : (2 : ℚ) / 9 * (5 : ℚ) / 11 = 10 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l823_82303


namespace NUMINAMATH_CALUDE_shorter_side_is_ten_l823_82318

/-- A rectangular room with given perimeter and area -/
structure Room where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 30
  area_eq : length * width = 200

/-- The shorter side of the room is 10 feet -/
theorem shorter_side_is_ten (room : Room) : min room.length room.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_shorter_side_is_ten_l823_82318


namespace NUMINAMATH_CALUDE_hexagon_smallest_angle_l823_82310

-- Define a hexagon with angles in arithmetic progression
def hexagon_angles (x : ℝ) : List ℝ := [x, x + 10, x + 20, x + 30, x + 40, x + 50]

-- Theorem statement
theorem hexagon_smallest_angle :
  ∃ (x : ℝ), 
    (List.sum (hexagon_angles x) = 720) ∧ 
    (∀ (angle : ℝ), angle ∈ hexagon_angles x → angle ≥ x) ∧
    x = 95 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_smallest_angle_l823_82310


namespace NUMINAMATH_CALUDE_fifth_root_inequality_l823_82328

theorem fifth_root_inequality (x y : ℝ) : x < y → x^(1/5) > y^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_inequality_l823_82328


namespace NUMINAMATH_CALUDE_bucket_size_calculation_l823_82382

/-- Given a leak rate and maximum time away, calculate the required bucket size -/
theorem bucket_size_calculation (leak_rate : ℝ) (max_time : ℝ) 
  (h1 : leak_rate = 1.5)
  (h2 : max_time = 12)
  (h3 : leak_rate > 0)
  (h4 : max_time > 0) :
  2 * (leak_rate * max_time) = 36 :=
by sorry

end NUMINAMATH_CALUDE_bucket_size_calculation_l823_82382


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l823_82306

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l823_82306


namespace NUMINAMATH_CALUDE_no_adjacent_x_probability_l823_82396

-- Define the number of X tiles and O tiles
def num_x : ℕ := 4
def num_o : ℕ := 3

-- Define the total number of tiles
def total_tiles : ℕ := num_x + num_o

-- Function to calculate the number of ways to arrange tiles
def arrange_tiles (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the number of valid arrangements (no adjacent X tiles)
def valid_arrangements : ℕ := 1

-- Theorem statement
theorem no_adjacent_x_probability :
  (valid_arrangements : ℚ) / (arrange_tiles total_tiles num_x) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_x_probability_l823_82396


namespace NUMINAMATH_CALUDE_cube_root_fifth_power_sixth_l823_82364

theorem cube_root_fifth_power_sixth : (((5 ^ (1/2)) ^ 4) ^ (1/3)) ^ 6 = 625 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fifth_power_sixth_l823_82364


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l823_82399

def power_product : ℕ := 2^2010 * 5^2008 * 7

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_power_product : sum_of_digits power_product = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l823_82399


namespace NUMINAMATH_CALUDE_no_family_of_lines_exist_l823_82333

theorem no_family_of_lines_exist :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧ 
    (∀ n, k n * k (n + 1) ≥ 0) ∧ 
    (∀ n, k n ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_family_of_lines_exist_l823_82333


namespace NUMINAMATH_CALUDE_trig_identities_l823_82370

theorem trig_identities (α : Real) (h : Real.tan α = 2) : 
  ((Real.sin (Real.pi - α) + Real.cos (α - Real.pi/2) - Real.cos (3*Real.pi + α)) / 
   (Real.cos (Real.pi/2 + α) - Real.sin (2*Real.pi + α) + 2*Real.sin (α - Real.pi/2)) = -5/6) ∧ 
  (Real.cos (2*α) + Real.sin α * Real.cos α = -1/5) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l823_82370


namespace NUMINAMATH_CALUDE_max_value_with_remainder_l823_82334

theorem max_value_with_remainder (A B : ℕ) (h1 : A ≠ B) (h2 : A = 17 * 25 + B) : 
  (∀ C : ℕ, C < 17 → B ≤ C) → A = 441 :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_remainder_l823_82334


namespace NUMINAMATH_CALUDE_action_figure_cost_l823_82352

theorem action_figure_cost 
  (current : ℕ) 
  (total : ℕ) 
  (cost : ℚ) : 
  current = 3 → 
  total = 8 → 
  cost = 30 → 
  (cost / (total - current) : ℚ) = 6 := by
sorry

end NUMINAMATH_CALUDE_action_figure_cost_l823_82352


namespace NUMINAMATH_CALUDE_geometric_sequence_converse_l823_82375

/-- The converse of a proposition "If P, then Q" is "If Q, then P" -/
def converse_of (P Q : Prop) : Prop :=
  Q → P

/-- Three real numbers form a geometric sequence if the middle term 
    is the geometric mean of the other two -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- The proposition "If a, b, c form a geometric sequence, then b^2 = ac" 
    and its converse -/
theorem geometric_sequence_converse :
  converse_of (is_geometric_sequence a b c) (b^2 = a * c) =
  (b^2 = a * c → is_geometric_sequence a b c) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_converse_l823_82375


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_50_l823_82388

/-- The speed of a train given travel time and alternative speed scenario -/
theorem train_speed (travel_time : ℝ) (alt_time : ℝ) (alt_speed : ℝ) : ℝ :=
  let distance := alt_speed * alt_time
  distance / travel_time

/-- Proof that the train's speed is 50 mph given the specified conditions -/
theorem train_speed_is_50 :
  train_speed 4 2 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_50_l823_82388


namespace NUMINAMATH_CALUDE_intersection_equality_l823_82346

theorem intersection_equality (m : ℝ) : 
  let A : Set ℝ := {0, 1, 2}
  let B : Set ℝ := {1, m}
  A ∩ B = B → m = 0 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_l823_82346


namespace NUMINAMATH_CALUDE_jake_peaches_l823_82368

theorem jake_peaches (steven_peaches jill_peaches jake_peaches : ℕ) : 
  jake_peaches + 7 = steven_peaches → 
  steven_peaches = jill_peaches + 14 → 
  steven_peaches = 15 → 
  jake_peaches = 8 := by
sorry

end NUMINAMATH_CALUDE_jake_peaches_l823_82368


namespace NUMINAMATH_CALUDE_debbys_flour_amount_l823_82381

/-- Proves that Debby's total flour amount is correct given her initial amount and purchase. -/
theorem debbys_flour_amount (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 12 → bought = 4 → total = initial + bought → total = 16 := by
  sorry

end NUMINAMATH_CALUDE_debbys_flour_amount_l823_82381


namespace NUMINAMATH_CALUDE_leo_mira_sum_difference_l823_82338

def leo_sum : ℕ := (List.range 50).map (· + 1) |>.sum

def digit_replace (n : ℕ) : ℕ :=
  let s := toString n
  let s' := s.replace "2" "1" |>.replace "3" "0"
  s'.toNat!

def mira_sum : ℕ := (List.range 50).map (· + 1 |> digit_replace) |>.sum

theorem leo_mira_sum_difference : leo_sum - mira_sum = 420 := by
  sorry

end NUMINAMATH_CALUDE_leo_mira_sum_difference_l823_82338


namespace NUMINAMATH_CALUDE_initial_gifts_count_l823_82390

/-- The number of gifts sent to the orphanage -/
def gifts_sent : ℕ := 66

/-- The number of gifts left under the tree -/
def gifts_left : ℕ := 11

/-- The initial number of gifts -/
def initial_gifts : ℕ := gifts_sent + gifts_left

theorem initial_gifts_count : initial_gifts = 77 := by
  sorry

end NUMINAMATH_CALUDE_initial_gifts_count_l823_82390


namespace NUMINAMATH_CALUDE_project_completion_time_l823_82307

/-- Given a project requiring 1500 hours and a daily work schedule of 15 hours,
    prove that the number of days needed to complete the project is 100. -/
theorem project_completion_time (project_hours : ℕ) (daily_hours : ℕ) :
  project_hours = 1500 →
  daily_hours = 15 →
  project_hours / daily_hours = 100 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l823_82307


namespace NUMINAMATH_CALUDE_money_distribution_l823_82367

theorem money_distribution (raquel sam nataly tom : ℚ) : 
  raquel = 40 →
  nataly = 3 * raquel →
  nataly = (5/3) * sam →
  tom = (1/4) * nataly →
  tom + raquel + nataly + sam = 262 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l823_82367


namespace NUMINAMATH_CALUDE_transformation_result_l823_82331

/-- Reflect a point about the line y = x -/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Rotate a point 180° counterclockwise around (2,3) -/
def rotate_180_around_2_3 (p : ℝ × ℝ) : ℝ × ℝ :=
  (4 - p.1, 6 - p.2)

/-- The final position after transformations -/
def final_position : ℝ × ℝ := (-2, -1)

theorem transformation_result (m n : ℝ) : 
  rotate_180_around_2_3 (reflect_about_y_eq_x (m, n)) = final_position → n - m = -1 := by
  sorry

#check transformation_result

end NUMINAMATH_CALUDE_transformation_result_l823_82331


namespace NUMINAMATH_CALUDE_hyperbola_equation_l823_82393

/-- The standard equation of a hyperbola with the same foci as a given ellipse and passing through a specific point -/
theorem hyperbola_equation (e : Real → Real → Prop) (p : Real × Real) :
  (∀ x y, e x y ↔ x^2 / 9 + y^2 / 5 = 1) →
  p = (Real.sqrt 2, Real.sqrt 3) →
  ∃ h : Real → Real → Prop,
    (∀ x y, h x y ↔ x^2 - y^2 / 3 = 1) ∧
    (∀ c : Real, (∃ x, e x 0 ∧ x^2 = c^2) ↔ (∃ x, h x 0 ∧ x^2 = c^2)) ∧
    h p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l823_82393


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l823_82305

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem perpendicular_lines_sum (a b c : ℝ) : 
  perpendicular (-a/4) (2/5) →
  point_on_line 1 c a 4 (-2) →
  point_on_line 1 c 2 (-5) b →
  a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l823_82305


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l823_82351

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l823_82351


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l823_82344

theorem sum_of_possible_x_values (x : ℝ) (h : |x - 12| = 100) : 
  ∃ (x₁ x₂ : ℝ), |x₁ - 12| = 100 ∧ |x₂ - 12| = 100 ∧ x₁ + x₂ = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l823_82344


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l823_82348

/-- Conversion of rectangular coordinates (8, 2√6) to polar coordinates (r, θ) -/
theorem rect_to_polar_conversion :
  ∃ (r θ : ℝ), 
    r = 2 * Real.sqrt 22 ∧ 
    Real.tan θ = Real.sqrt 6 / 4 ∧ 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_conversion_l823_82348


namespace NUMINAMATH_CALUDE_sequence_properties_l823_82389

/-- Given a sequence and its partial sum satisfying certain conditions, 
    prove that it's geometric and find the range of t when the sum converges to 1 -/
theorem sequence_properties (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (t : ℝ) 
    (h1 : ∀ n : ℕ+, S n = 1 + t * a n) 
    (h2 : t ≠ 1) (h3 : t ≠ 0) :
  (∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = r * a n) ∧ 
  (∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |S n - 1| < ε) → 
  (t < 1/2 ∧ t ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l823_82389


namespace NUMINAMATH_CALUDE_pizza_lovers_count_l823_82357

theorem pizza_lovers_count (total pupils_like_burgers pupils_like_both : ℕ) 
  (h1 : total = 200)
  (h2 : pupils_like_burgers = 115)
  (h3 : pupils_like_both = 40)
  : ∃ pupils_like_pizza : ℕ, 
    pupils_like_pizza + pupils_like_burgers - pupils_like_both = total ∧ 
    pupils_like_pizza = 125 :=
by sorry

end NUMINAMATH_CALUDE_pizza_lovers_count_l823_82357


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l823_82323

/-- The probability that all three quitters are from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (n : ℕ) (k : ℕ) (q : ℕ) : 
  n = 20 → -- Total number of contestants
  k = 10 → -- Number of contestants in each tribe
  q = 3 →  -- Number of quitters
  (n = 2 * k) → -- Two equally sized tribes
  (Fintype.card {s : Finset (Fin n) // s.card = q ∧ (∀ i ∈ s, i < k) } +
   Fintype.card {s : Finset (Fin n) // s.card = q ∧ (∀ i ∈ s, k ≤ i) }) /
  Fintype.card {s : Finset (Fin n) // s.card = q} = 20 / 95 :=
by sorry


end NUMINAMATH_CALUDE_survivor_quitters_probability_l823_82323


namespace NUMINAMATH_CALUDE_fraction_calculation_l823_82355

theorem fraction_calculation : (5 / 3 : ℚ) ^ 3 * (2 / 5 : ℚ) = 50 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l823_82355


namespace NUMINAMATH_CALUDE_train_length_l823_82385

/-- Calculates the length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 5 → speed_kmh * (1000 / 3600) * time_s = 100 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l823_82385
