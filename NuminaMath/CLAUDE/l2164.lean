import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2164_216425

/-- Given an arithmetic sequence with first term 11 and common difference -3,
    prove that its 8th term is -10. -/
theorem arithmetic_sequence_eighth_term :
  let a : ℕ → ℤ := fun n => 11 - 3 * (n - 1)
  a 8 = -10 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2164_216425


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2164_216455

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 2 →
  perimeter = 2 * (width + length) →
  perimeter = 54 →
  area = width * length →
  area = 162 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2164_216455


namespace NUMINAMATH_CALUDE_cannot_reach_54_from_12_l2164_216423

def Operation := Nat → Nat

def isValidOperation (op : Operation) : Prop :=
  ∀ n, (op n = 2 * n) ∨ (op n = 3 * n) ∨ (op n = n / 2) ∨ (op n = n / 3)

def applyOperations (ops : List Operation) (start : Nat) : Nat :=
  ops.foldl (λ acc op => op acc) start

theorem cannot_reach_54_from_12 :
  ¬ ∃ (ops : List Operation),
    (ops.length = 60) ∧
    (∀ op ∈ ops, isValidOperation op) ∧
    (applyOperations ops 12 = 54) :=
sorry

end NUMINAMATH_CALUDE_cannot_reach_54_from_12_l2164_216423


namespace NUMINAMATH_CALUDE_stating_mooncake_packing_solution_l2164_216466

/-- Represents the number of mooncakes in a large bag -/
def large_bag : ℕ := 9

/-- Represents the number of mooncakes in a small package -/
def small_package : ℕ := 4

/-- Represents the total number of mooncakes -/
def total_mooncakes : ℕ := 35

/-- 
Theorem stating that there exist non-negative integers x and y 
such that 9x + 4y = 35, and x + y is minimized
-/
theorem mooncake_packing_solution :
  ∃ x y : ℕ, large_bag * x + small_package * y = total_mooncakes ∧
  ∀ a b : ℕ, large_bag * a + small_package * b = total_mooncakes → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_stating_mooncake_packing_solution_l2164_216466


namespace NUMINAMATH_CALUDE_order_of_6_l2164_216453

def f (x : ℕ) : ℕ := x^2 % 13

def is_periodic (f : ℕ → ℕ) (x : ℕ) (period : ℕ) : Prop :=
  ∀ n, f^[n + period] x = f^[n] x

theorem order_of_6 (h : is_periodic f 6 72) :
  ∀ k, 0 < k → k < 72 → ¬ is_periodic f 6 k :=
sorry

end NUMINAMATH_CALUDE_order_of_6_l2164_216453


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2164_216498

/-- 
Given a quadratic equation x^2 + bx + 4 = 0 with two equal real roots,
prove that b = 4 or b = -4.
-/
theorem quadratic_equal_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + b*y + 4 = 0 → y = x) → 
  b = 4 ∨ b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2164_216498


namespace NUMINAMATH_CALUDE_parliament_vote_ratio_l2164_216487

theorem parliament_vote_ratio (V : ℝ) (X : ℝ) (q : ℝ) : 
  V > 0 →
  X = 7/10 * V →
  (6/50 * V + q * X) / (9/50 * V + (1 - q) * X) = 3/2 →
  q = 24/35 := by
  sorry

end NUMINAMATH_CALUDE_parliament_vote_ratio_l2164_216487


namespace NUMINAMATH_CALUDE_problem_l2164_216444

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem problem (a b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n, a n ≠ 0) →
  2 * a 3 - a 1 ^ 2 = 0 →
  a 1 = d →
  b 13 = a 2 →
  b 1 = a 1 →
  b 6 * b 8 = 72 := by
sorry

end NUMINAMATH_CALUDE_problem_l2164_216444


namespace NUMINAMATH_CALUDE_part1_part2_l2164_216454

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part 1
theorem part1 (x : ℝ) :
  f 4 x = 7 → -3 ≤ x ∧ x ≤ 4 :=
by sorry

-- Part 2
theorem part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2}) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_part1_part2_l2164_216454


namespace NUMINAMATH_CALUDE_square_with_tens_digit_7_l2164_216490

/-- A square number with tens digit 7 has units digit 6 -/
theorem square_with_tens_digit_7 (n : ℕ) :
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_with_tens_digit_7_l2164_216490


namespace NUMINAMATH_CALUDE_square_with_trees_theorem_l2164_216447

/-- Represents a square with trees at its vertices -/
structure SquareWithTrees where
  side_length : ℝ
  height_A : ℝ
  height_B : ℝ
  height_C : ℝ
  height_D : ℝ

/-- Checks if there exists a point equidistant from all tree tops -/
def has_equidistant_point (s : SquareWithTrees) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ x < s.side_length ∧ 0 < y ∧ y < s.side_length ∧
  (s.height_A^2 + x^2 + y^2 = s.height_B^2 + (s.side_length - x)^2 + y^2) ∧
  (s.height_A^2 + x^2 + y^2 = s.height_C^2 + (s.side_length - x)^2 + (s.side_length - y)^2) ∧
  (s.height_A^2 + x^2 + y^2 = s.height_D^2 + x^2 + (s.side_length - y)^2)

/-- The main theorem about the square with trees -/
theorem square_with_trees_theorem (s : SquareWithTrees) 
  (h1 : s.height_A = 7)
  (h2 : s.height_B = 13)
  (h3 : s.height_C = 17)
  (h4 : has_equidistant_point s) :
  s.side_length > Real.sqrt 120 ∧ s.height_D = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_with_trees_theorem_l2164_216447


namespace NUMINAMATH_CALUDE_picnic_theorem_l2164_216442

-- Define the propositions
variable (P : Prop) -- "The picnic on Sunday will be held"
variable (Q : Prop) -- "The weather is fair on Sunday"

-- State the given condition
axiom given_statement : (¬P → ¬Q)

-- State the theorem to be proved
theorem picnic_theorem : Q → P := by sorry

end NUMINAMATH_CALUDE_picnic_theorem_l2164_216442


namespace NUMINAMATH_CALUDE_dans_eggs_l2164_216448

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Dan bought -/
def dans_dozens : ℕ := 9

/-- Theorem: Dan bought 108 eggs -/
theorem dans_eggs : dans_dozens * eggs_per_dozen = 108 := by
  sorry

end NUMINAMATH_CALUDE_dans_eggs_l2164_216448


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2164_216416

theorem geometric_sequence_middle_term (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 15 * r = b ∧ b * r = 1) → b = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2164_216416


namespace NUMINAMATH_CALUDE_ab_value_l2164_216485

theorem ab_value (a b : ℕ+) (h : a^2 + 3*b = 33) : a*b = 24 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2164_216485


namespace NUMINAMATH_CALUDE_max_d_value_l2164_216404

def a (n : ℕ+) : ℕ := 99 + 2 * n ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 11) ∧ (∀ (n : ℕ+), d n ≤ 11) :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2164_216404


namespace NUMINAMATH_CALUDE_shoes_cost_theorem_l2164_216412

theorem shoes_cost_theorem (cost_first_pair : ℝ) (percentage_increase : ℝ) : 
  cost_first_pair = 22 →
  percentage_increase = 50 →
  let cost_second_pair := cost_first_pair * (1 + percentage_increase / 100)
  let total_cost := cost_first_pair + cost_second_pair
  total_cost = 55 := by
sorry

end NUMINAMATH_CALUDE_shoes_cost_theorem_l2164_216412


namespace NUMINAMATH_CALUDE_einstein_fundraising_goal_l2164_216492

theorem einstein_fundraising_goal (pizza_price : ℝ) (fries_price : ℝ) (soda_price : ℝ)
  (pizza_sold : ℕ) (fries_sold : ℕ) (soda_sold : ℕ) (additional_needed : ℝ) :
  pizza_price = 12 →
  fries_price = 0.30 →
  soda_price = 2 →
  pizza_sold = 15 →
  fries_sold = 40 →
  soda_sold = 25 →
  additional_needed = 258 →
  pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold + additional_needed = 500 := by
  sorry

end NUMINAMATH_CALUDE_einstein_fundraising_goal_l2164_216492


namespace NUMINAMATH_CALUDE_proposition_equivalences_and_set_equality_l2164_216469

-- Define the proposition
def P (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def Q (x : ℝ) : Prop := x = 1 ∨ x = 2

-- Define the sets P and S
def setP : Set ℝ := {x | -1 < x ∧ x < 3}
def setS (a : ℝ) : Set ℝ := {x | x^2 + (a+1)*x + a < 0}

theorem proposition_equivalences_and_set_equality :
  (∀ x, Q x → P x) ∧
  (∀ x, ¬(P x) → ¬(Q x)) ∧
  (∀ x, ¬(Q x) → ¬(P x)) ∧
  ∃ a, setP = setS a ∧ a = -3 := by sorry

end NUMINAMATH_CALUDE_proposition_equivalences_and_set_equality_l2164_216469


namespace NUMINAMATH_CALUDE_complement_A_union_B_equals_interval_intersection_A_B_empty_t_range_l2164_216405

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * 3 * x + 10

-- Define set A
def A : Set ℝ := {x | f x > 0}

-- Define set B
def B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1}

-- Theorem for part (I)
theorem complement_A_union_B_equals_interval :
  (Aᶜ ∪ B 1) = {x | -3 ≤ x ∧ x ≤ 2} :=
sorry

-- Theorem for part (II)
theorem intersection_A_B_empty_t_range (t : ℝ) :
  (A ∩ B t = ∅) ↔ (-2 ≤ t ∧ t ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_complement_A_union_B_equals_interval_intersection_A_B_empty_t_range_l2164_216405


namespace NUMINAMATH_CALUDE_function_inequality_l2164_216486

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) (a : ℝ) (ha : a > 0) : 
  f a < Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2164_216486


namespace NUMINAMATH_CALUDE_same_color_probability_l2164_216438

theorem same_color_probability
  (total_balls : ℕ)
  (prob_white : ℝ)
  (h_total : total_balls = 4)
  (h_prob : prob_white = 1/2) :
  let white_balls := (total_balls : ℝ) * prob_white
  let black_balls := total_balls - white_balls
  (white_balls * (white_balls - 1) + black_balls * (black_balls - 1)) / (total_balls * (total_balls - 1)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l2164_216438


namespace NUMINAMATH_CALUDE_sin_double_angle_with_tan_two_l2164_216432

theorem sin_double_angle_with_tan_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_with_tan_two_l2164_216432


namespace NUMINAMATH_CALUDE_expression_simplification_l2164_216496

theorem expression_simplification (x : ℝ) (h : x = -3) :
  (x - 3) * (x + 4) - (x - x^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2164_216496


namespace NUMINAMATH_CALUDE_average_mpg_calculation_l2164_216428

theorem average_mpg_calculation (initial_reading final_reading : ℕ) (fuel_used : ℕ) :
  initial_reading = 56200 →
  final_reading = 57150 →
  fuel_used = 50 →
  (final_reading - initial_reading : ℚ) / fuel_used = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_mpg_calculation_l2164_216428


namespace NUMINAMATH_CALUDE_f_odd_implies_a_zero_necessary_not_sufficient_l2164_216426

noncomputable def f (a x : ℝ) : ℝ := 1 / (x - 1) + a / (x + a - 1) + 1 / (x + 1)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem f_odd_implies_a_zero_necessary_not_sufficient :
  (∃ a : ℝ, is_odd_function (f a)) ∧
  (∀ a : ℝ, is_odd_function (f a) → a = 0 ∨ a = 1) ∧
  (∃ a : ℝ, a ≠ 0 ∧ is_odd_function (f a)) :=
sorry

end NUMINAMATH_CALUDE_f_odd_implies_a_zero_necessary_not_sufficient_l2164_216426


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l2164_216470

/-- The probability of drawing two white balls without replacement from a box containing 8 white balls and 7 black balls is 4/15. -/
theorem two_white_balls_probability :
  let total_balls : ℕ := 8 + 7
  let white_balls : ℕ := 8
  let black_balls : ℕ := 7
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l2164_216470


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2164_216440

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 10}
def N : Set ℝ := {x | x < -4/3 ∨ x > 3}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2164_216440


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2164_216482

/-- A circle tangent to coordinate axes and hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : ℝ      -- hypotenuse length of the 45-45-90 triangle

/-- The circle is tangent to both axes and the hypotenuse --/
def is_tangent (c : TangentCircle) : Prop :=
  c.O.1 = c.r ∧ c.O.2 = c.r ∧ c.O.1 + c.O.2 + c.r = c.h

theorem tangent_circle_radius (c : TangentCircle) 
  (h_hypotenuse : c.h = 2 * Real.sqrt 2)
  (h_tangent : is_tangent c) : 
  c.r = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2164_216482


namespace NUMINAMATH_CALUDE_floor_sum_example_l2164_216401

theorem floor_sum_example : ⌊(24.8 : ℝ)⌋ + ⌊(-24.8 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2164_216401


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2164_216445

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 12 ∧ 1 / a₀ + 1 / b₀ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2164_216445


namespace NUMINAMATH_CALUDE_calculation_proof_l2164_216478

theorem calculation_proof : ((5 + 7 + 3) * 2 - 4) / 2 - 5 / 2 = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2164_216478


namespace NUMINAMATH_CALUDE_gcd_n4_plus_16_and_n_plus_3_l2164_216497

theorem gcd_n4_plus_16_and_n_plus_3 (n : ℕ) (h1 : n > 9) (h2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_16_and_n_plus_3_l2164_216497


namespace NUMINAMATH_CALUDE_equation_solution_l2164_216452

theorem equation_solution : 
  ∃ (S : Set ℝ), S = {x : ℝ | (x + 2)^4 + x^4 = 82} ∧ S = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2164_216452


namespace NUMINAMATH_CALUDE_integer_between_sqrt_twelve_l2164_216403

theorem integer_between_sqrt_twelve : ∃ (m : ℤ), m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_twelve_l2164_216403


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2164_216430

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≥ 1}

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- Theorem statement
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -1 ∨ x = 1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2164_216430


namespace NUMINAMATH_CALUDE_set_equivalence_l2164_216450

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem set_equivalence : {x : ℝ | x ≥ 1} = (Set.univ : Set ℝ) \ (M ∪ N) := by sorry

end NUMINAMATH_CALUDE_set_equivalence_l2164_216450


namespace NUMINAMATH_CALUDE_halfway_fraction_l2164_216483

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2164_216483


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2164_216420

theorem hemisphere_surface_area (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  3 * Real.pi * r^2 = 972 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2164_216420


namespace NUMINAMATH_CALUDE_event_probability_l2164_216476

theorem event_probability (P_A P_A_and_B P_A_or_B : ℝ) 
  (h1 : P_A = 0.4)
  (h2 : P_A_and_B = 0.25)
  (h3 : P_A_or_B = 0.8) :
  ∃ P_B : ℝ, P_B = 0.65 ∧ P_A_or_B = P_A + P_B - P_A_and_B :=
by
  sorry

end NUMINAMATH_CALUDE_event_probability_l2164_216476


namespace NUMINAMATH_CALUDE_quadratic_intercepts_l2164_216463

/-- Given a quadratic function y = x^2 + bx - 3 that passes through the point (3,0),
    prove that b = -2 and the other x-intercept is at (-1,0) -/
theorem quadratic_intercepts (b : ℝ) : 
  (3^2 + 3*b - 3 = 0) → 
  (b = -2 ∧ (-1)^2 + (-1)*b - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intercepts_l2164_216463


namespace NUMINAMATH_CALUDE_mary_snake_observation_l2164_216443

/-- Given the number of breeding balls, snakes per ball, and total snakes observed,
    calculate the number of additional pairs of snakes. -/
def additional_snake_pairs (breeding_balls : ℕ) (snakes_per_ball : ℕ) (total_snakes : ℕ) : ℕ :=
  ((total_snakes - breeding_balls * snakes_per_ball) / 2)

/-- Theorem stating that given 3 breeding balls with 8 snakes each,
    and a total of 36 snakes observed, the number of additional pairs of snakes is 6. -/
theorem mary_snake_observation :
  additional_snake_pairs 3 8 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_snake_observation_l2164_216443


namespace NUMINAMATH_CALUDE_min_value_implies_t_l2164_216417

-- Define the function f
def f (x t : ℝ) : ℝ := |x - t| + |5 - x|

-- State the theorem
theorem min_value_implies_t (t : ℝ) : 
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x t ≥ m) → t = 2 ∨ t = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_t_l2164_216417


namespace NUMINAMATH_CALUDE_binomial_20_4_l2164_216459

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l2164_216459


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2164_216409

theorem book_arrangement_theorem :
  let n : ℕ := 7  -- number of books
  let k : ℕ := 3  -- number of shelves
  let arrangements := (n - 1).choose (k - 1) * n.factorial
  arrangements = 75600 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2164_216409


namespace NUMINAMATH_CALUDE_product_pricing_l2164_216456

/-- Given three products A, B, and C with unknown prices, prove that if 2A + 3B + 1C costs 295 yuan
    and 4A + 3B + 5C costs 425 yuan, then 1A + 1B + 1C costs 120 yuan. -/
theorem product_pricing (a b c : ℝ) 
    (h1 : 2*a + 3*b + c = 295)
    (h2 : 4*a + 3*b + 5*c = 425) : 
  a + b + c = 120 := by
sorry

end NUMINAMATH_CALUDE_product_pricing_l2164_216456


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l2164_216495

/-- Returns the last four digits of a number in base 10 -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- Checks if a number satisfies the conditions of the problem -/
def satisfiesConditions (n : ℕ) : Prop :=
  (n > 0) ∧ (lastFourDigits n = lastFourDigits (n^2)) ∧ ((n - 2) % 7 = 0)

theorem smallest_satisfying_number :
  satisfiesConditions 625 ∧ ∀ n < 625, ¬(satisfiesConditions n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l2164_216495


namespace NUMINAMATH_CALUDE_three_points_determine_plane_l2164_216429

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space using the general equation ax + by + cz + d = 0
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if two planes are perpendicular
def perpendicularPlanes (p1 p2 : Plane) : Prop :=
  p1.a * p2.a + p1.b * p2.b + p1.c * p2.c = 0

-- Function to check if a point lies on a plane
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Theorem statement
theorem three_points_determine_plane 
  (p1 p2 p3 : Plane) 
  (point1 point2 point3 : Point3D) : 
  perpendicularPlanes p1 p2 ∧ 
  perpendicularPlanes p2 p3 ∧ 
  perpendicularPlanes p3 p1 ∧ 
  pointOnPlane point1 p1 ∧ 
  pointOnPlane point2 p2 ∧ 
  pointOnPlane point3 p3 → 
  ∃! (resultPlane : Plane), 
    pointOnPlane point1 resultPlane ∧ 
    pointOnPlane point2 resultPlane ∧ 
    pointOnPlane point3 resultPlane :=
by
  sorry

end NUMINAMATH_CALUDE_three_points_determine_plane_l2164_216429


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2164_216414

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The number to be converted to scientific notation -/
def original_number : ℕ := 189130000000

/-- Function to convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem scientific_notation_correct :
  let sn := to_scientific_notation original_number
  sn.coefficient = 1.8913 ∧ sn.exponent = 11 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2164_216414


namespace NUMINAMATH_CALUDE_max_distance_complex_l2164_216427

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z + 1 - Complex.I) = 1) :
  ∃ (max_val : ℝ), max_val = 3 ∧ ∀ w, Complex.abs (w + 1 - Complex.I) = 1 →
    Complex.abs (w - 1 - Complex.I) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2164_216427


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l2164_216479

-- Define the conditions p and q
def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Define not_p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, not_p x → q x) ∧ 
  ¬(∀ x, q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l2164_216479


namespace NUMINAMATH_CALUDE_sqrt_product_equals_120_sqrt_3_l2164_216488

theorem sqrt_product_equals_120_sqrt_3 : 
  Real.sqrt 75 * Real.sqrt 48 * Real.sqrt 12 = 120 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_120_sqrt_3_l2164_216488


namespace NUMINAMATH_CALUDE_expression_equality_l2164_216439

theorem expression_equality : 
  (1 / 3) ^ 2000 * 27 ^ 669 + Real.sin (60 * π / 180) * Real.tan (60 * π / 180) + (2009 + Real.sin (25 * π / 180)) ^ 0 = 2 + 29 / 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2164_216439


namespace NUMINAMATH_CALUDE_not_magical_2099_l2164_216491

/-- A year is magical if there exists a month and day such that their sum equals the last two digits of the year. -/
def isMagicalYear (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month + day = year % 100

/-- 2099 is not a magical year. -/
theorem not_magical_2099 : ¬ isMagicalYear 2099 := by
  sorry

#check not_magical_2099

end NUMINAMATH_CALUDE_not_magical_2099_l2164_216491


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l2164_216418

theorem factorization_of_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l2164_216418


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l2164_216446

theorem initial_markup_percentage (initial_price : ℝ) (price_increase : ℝ) : 
  initial_price = 36 →
  price_increase = 4 →
  (initial_price + price_increase) / (initial_price - initial_price * 0.8) = 2 →
  (initial_price - (initial_price - initial_price * 0.8)) / (initial_price - initial_price * 0.8) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l2164_216446


namespace NUMINAMATH_CALUDE_inequality_solution_l2164_216434

theorem inequality_solution (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2164_216434


namespace NUMINAMATH_CALUDE_value_of_y_l2164_216464

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2164_216464


namespace NUMINAMATH_CALUDE_max_constant_inequality_l2164_216419

theorem max_constant_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∀ k : ℝ, (∀ a b c d : ℝ, 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 →
    a^2*b + b^2*c + c^2*d + d^2*a + 4 ≥ k*(a^3 + b^3 + c^3 + d^3)) → k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l2164_216419


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2164_216424

def initial_price : ℝ := 50
def first_year_increase : ℝ := 2  -- 200% increase
def second_year_decrease : ℝ := 0.5  -- 50% decrease

def final_price : ℝ :=
  initial_price * (1 + first_year_increase) * second_year_decrease

theorem stock_price_calculation :
  final_price = 75 := by sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2164_216424


namespace NUMINAMATH_CALUDE_additional_charging_time_l2164_216421

/-- Represents the charging characteristics of a mobile battery -/
structure BatteryCharging where
  initial_charge_time : ℕ  -- Time to reach 20% charge in minutes
  initial_charge_percent : ℕ  -- Initial charge percentage
  total_charge_time : ℕ  -- Total time to reach P% charge in minutes

/-- Theorem stating the additional charging time -/
theorem additional_charging_time (b : BatteryCharging) 
  (h1 : b.initial_charge_time = 60)  -- 1 hour = 60 minutes
  (h2 : b.initial_charge_percent = 20)
  (h3 : b.total_charge_time = b.initial_charge_time + 150) :
  b.total_charge_time - b.initial_charge_time = 150 := by
  sorry

#check additional_charging_time

end NUMINAMATH_CALUDE_additional_charging_time_l2164_216421


namespace NUMINAMATH_CALUDE_white_square_area_main_white_square_area_l2164_216489

-- Define the cube's side length
def cubeSide : ℝ := 12

-- Define the total amount of blue paint
def totalBluePaint : ℝ := 432

-- Define the number of faces on a cube
def numFaces : ℕ := 6

-- Theorem statement
theorem white_square_area (cubeSide : ℝ) (totalBluePaint : ℝ) (numFaces : ℕ) :
  cubeSide > 0 →
  totalBluePaint > 0 →
  numFaces = 6 →
  let totalSurfaceArea := numFaces * cubeSide * cubeSide
  let bluePaintPerFace := totalBluePaint / numFaces
  let whiteSquareArea := cubeSide * cubeSide - bluePaintPerFace
  whiteSquareArea = 72 := by
  sorry

-- Main theorem using the defined constants
theorem main_white_square_area : 
  let totalSurfaceArea := numFaces * cubeSide * cubeSide
  let bluePaintPerFace := totalBluePaint / numFaces
  let whiteSquareArea := cubeSide * cubeSide - bluePaintPerFace
  whiteSquareArea = 72 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_main_white_square_area_l2164_216489


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2164_216481

-- Define the quadratic function
def f (a x : ℝ) := (a - 2) * x^2 + 2 * (a - 2) * x - 4

-- State the theorem
theorem quadratic_inequality_range :
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Set.Ioc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2164_216481


namespace NUMINAMATH_CALUDE_bill_purchase_percentage_bill_specific_problem_l2164_216484

/-- The problem of determining the percentage by which Bill could have purchased a product for less -/
theorem bill_purchase_percentage (original_profit_rate : ℝ) (new_profit_rate : ℝ) 
  (original_selling_price : ℝ) (additional_profit : ℝ) : ℝ :=
  let original_cost := original_selling_price / (1 + original_profit_rate)
  let new_selling_price := original_selling_price + additional_profit
  let percentage_less := 1 - (new_selling_price / ((1 + new_profit_rate) * original_cost))
  percentage_less * 100

/-- Proof of the specific problem instance -/
theorem bill_specific_problem : 
  bill_purchase_percentage 0.1 0.3 549.9999999999995 35 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_purchase_percentage_bill_specific_problem_l2164_216484


namespace NUMINAMATH_CALUDE_price_difference_year_l2164_216499

def price_P (n : ℕ) : ℚ := 420/100 + 40/100 * n
def price_Q (n : ℕ) : ℚ := 630/100 + 15/100 * n

theorem price_difference_year : 
  ∃ n : ℕ, price_P n = price_Q n + 40/100 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_price_difference_year_l2164_216499


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2164_216449

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) < x + 2) → ((x + 1) / 2 < x) → (1 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2164_216449


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2164_216474

theorem inverse_proportion_problem (x y : ℝ → ℝ) (k : ℝ) :
  (∀ t, x t * y t = k) →  -- x and y are inversely proportional
  x 15 = 3 →              -- x = 3 when y = 15
  y 15 = 15 →             -- y = 15 when x = 3
  y (-30) = -30 →         -- y = -30
  x (-30) = -3/2 :=       -- x = -3/2 when y = -30
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2164_216474


namespace NUMINAMATH_CALUDE_mirasol_spending_l2164_216407

/-- Mirasol's spending problem -/
theorem mirasol_spending (initial_amount : ℕ) (coffee_cost : ℕ) (remaining_amount : ℕ) 
  (tumbler_cost : ℕ) :
  initial_amount = 50 →
  coffee_cost = 10 →
  remaining_amount = 10 →
  initial_amount = coffee_cost + tumbler_cost + remaining_amount →
  tumbler_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_mirasol_spending_l2164_216407


namespace NUMINAMATH_CALUDE_exam_results_l2164_216451

theorem exam_results (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_hindi = 0.25 * total)
  (h2 : failed_both = 0.4 * total)
  (h3 : passed_both = 0.8 * total) :
  ∃ failed_english : ℝ, failed_english = 0.35 * total :=
by
  sorry

end NUMINAMATH_CALUDE_exam_results_l2164_216451


namespace NUMINAMATH_CALUDE_white_ball_from_first_urn_l2164_216441

/-- Represents an urn with black and white balls -/
structure Urn :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of choosing an urn -/
def urn_prob : ℚ := 1/2

/-- Calculate the probability of drawing a white ball from an urn -/
def white_ball_prob (u : Urn) : ℚ :=
  u.white / (u.black + u.white)

/-- The theorem to prove -/
theorem white_ball_from_first_urn 
  (urn1 : Urn)
  (urn2 : Urn)
  (h1 : urn1 = ⟨3, 7⟩)
  (h2 : urn2 = ⟨4, 6⟩)
  : (urn_prob * white_ball_prob urn1) / 
    (urn_prob * white_ball_prob urn1 + urn_prob * white_ball_prob urn2) = 7/13 :=
sorry

end NUMINAMATH_CALUDE_white_ball_from_first_urn_l2164_216441


namespace NUMINAMATH_CALUDE_largest_consecutive_non_prime_under_50_l2164_216467

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_consecutive_non_prime_under_50 (a b c d e f : ℕ) :
  a < 100 ∧ b < 100 ∧ c < 100 ∧ d < 100 ∧ e < 100 ∧ f < 100 →  -- two-digit integers
  a < 50 ∧ b < 50 ∧ c < 50 ∧ d < 50 ∧ e < 50 ∧ f < 50 →  -- less than 50
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧ f = e + 1 →  -- consecutive
  ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ 
  ¬(is_prime d) ∧ ¬(is_prime e) ∧ ¬(is_prime f) →  -- not prime
  f = 37 :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_non_prime_under_50_l2164_216467


namespace NUMINAMATH_CALUDE_workers_total_earning_l2164_216462

/-- Calculates the total earning of three workers given their daily wages and work days -/
def total_earning (daily_wage_a daily_wage_b daily_wage_c : ℚ) 
  (days_a days_b days_c : ℕ) : ℚ :=
  daily_wage_a * days_a + daily_wage_b * days_b + daily_wage_c * days_c

/-- The total earning of three workers with given conditions -/
theorem workers_total_earning : 
  ∃ (daily_wage_a daily_wage_b daily_wage_c : ℚ),
    -- Daily wages ratio is 3:4:5
    daily_wage_a / daily_wage_b = 3 / 4 ∧
    daily_wage_b / daily_wage_c = 4 / 5 ∧
    -- Daily wage of c is Rs. 115
    daily_wage_c = 115 ∧
    -- Total earning calculation
    total_earning daily_wage_a daily_wage_b daily_wage_c 6 9 4 = 1702 := by
  sorry

end NUMINAMATH_CALUDE_workers_total_earning_l2164_216462


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2164_216406

theorem complex_equation_solution (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 56 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2164_216406


namespace NUMINAMATH_CALUDE_moon_radius_scientific_notation_l2164_216477

/-- The radius of the moon in meters -/
def moon_radius : ℝ := 1738000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the moon's radius is correctly expressed in scientific notation -/
theorem moon_radius_scientific_notation :
  ∃ (sn : ScientificNotation), moon_radius = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end NUMINAMATH_CALUDE_moon_radius_scientific_notation_l2164_216477


namespace NUMINAMATH_CALUDE_solution_existence_l2164_216402

theorem solution_existence (k : ℕ+) :
  (∃ x y : ℕ+, x * (x + k) = y * (y + 1)) ↔ (k = 1 ∨ k ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l2164_216402


namespace NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l2164_216410

theorem dice_sum_divisibility_probability (n : ℕ) (a b c : ℕ) 
  (h1 : a + b + c = n) 
  (h2 : 0 ≤ a ∧ a ≤ n) 
  (h3 : 0 ≤ b ∧ b ≤ n) 
  (h4 : 0 ≤ c ∧ c ≤ n) :
  (a^3 + b^3 + c^3 + 6*a*b*c : ℚ) / (n^3 : ℚ) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_divisibility_probability_l2164_216410


namespace NUMINAMATH_CALUDE_milk_cartons_sold_l2164_216494

theorem milk_cartons_sold (regular : ℕ) (chocolate : ℕ) : 
  regular = 3 →
  chocolate = 7 * regular →
  regular + chocolate = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_cartons_sold_l2164_216494


namespace NUMINAMATH_CALUDE_candy_given_to_haley_l2164_216458

def initial_candy : ℕ := 15
def remaining_candy : ℕ := 9

theorem candy_given_to_haley : initial_candy - remaining_candy = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_given_to_haley_l2164_216458


namespace NUMINAMATH_CALUDE_square_side_length_l2164_216468

theorem square_side_length (perimeter : ℚ) (h : perimeter = 12 / 25) :
  perimeter / 4 = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2164_216468


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2164_216436

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (·*·) 1

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  (∀ m : ℕ, m > 0 → a m * a (m + 2) = 2 * a (m + 1)) →
  SequenceProduct a (2 * m + 1) = 128 →
  m = 3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2164_216436


namespace NUMINAMATH_CALUDE_eliza_dress_ironing_time_l2164_216473

/-- Represents the time in minutes it takes Eliza to iron a dress -/
def dress_ironing_time : ℕ := sorry

/-- Represents the time in minutes it takes Eliza to iron a blouse -/
def blouse_ironing_time : ℕ := 15

/-- Represents the total time in minutes Eliza spends ironing blouses -/
def total_blouse_ironing_time : ℕ := 2 * 60

/-- Represents the total time in minutes Eliza spends ironing dresses -/
def total_dress_ironing_time : ℕ := 3 * 60

/-- Represents the total number of clothes Eliza ironed -/
def total_clothes : ℕ := 17

theorem eliza_dress_ironing_time :
  (total_blouse_ironing_time / blouse_ironing_time) +
  (total_dress_ironing_time / dress_ironing_time) = total_clothes →
  dress_ironing_time = 20 := by sorry

end NUMINAMATH_CALUDE_eliza_dress_ironing_time_l2164_216473


namespace NUMINAMATH_CALUDE_carpet_for_room_l2164_216460

/-- Calculates the minimum number of whole square yards of carpet needed for a rectangular room with overlap -/
def carpet_needed (length width overlap : ℕ) : ℕ :=
  let adjusted_length := length + 2 * overlap
  let adjusted_width := width + 2 * overlap
  let area := adjusted_length * adjusted_width
  (area + 8) / 9  -- Adding 8 before division by 9 to round up

theorem carpet_for_room : carpet_needed 15 9 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_carpet_for_room_l2164_216460


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2164_216408

/-- A quadratic function f(x) = x^2 + px + qx, where p and q are positive constants -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q*x

/-- The theorem stating that the minimum of f occurs at x = -(p+q)/2 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), x_min = -(p + q) / 2 ∧ 
  ∀ (x : ℝ), f p q x_min ≤ f p q x :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2164_216408


namespace NUMINAMATH_CALUDE_circle_polar_to_cartesian_and_area_l2164_216411

/-- Given a circle C with polar equation p = 2cosθ, this theorem proves that
    its Cartesian equation is x² - 2x + y² = 0 and its area is π. -/
theorem circle_polar_to_cartesian_and_area :
  ∀ (p θ x y : ℝ),
  (p = 2 * Real.cos θ) →                  -- Polar equation
  (x = p * Real.cos θ ∧ y = p * Real.sin θ) →  -- Polar to Cartesian conversion
  (x^2 - 2*x + y^2 = 0) ∧                 -- Cartesian equation
  (Real.pi = (Real.pi : ℝ)) :=            -- Area (π)
by sorry

end NUMINAMATH_CALUDE_circle_polar_to_cartesian_and_area_l2164_216411


namespace NUMINAMATH_CALUDE_f_equal_range_l2164_216422

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/2) * x + (3/2) else Real.log x

theorem f_equal_range (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  n - m ∈ Set.Icc (5 - 2 * Real.log 2) (Real.exp 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_f_equal_range_l2164_216422


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2164_216415

-- Define the two lines
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 4

-- Define the intersection point
def intersection_point : ℚ × ℚ := (-12/7, 22/7)

-- Theorem statement
theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2164_216415


namespace NUMINAMATH_CALUDE_waiter_problem_l2164_216475

/-- Given a waiter's section with initial customers, some leaving customers, and a number of tables,
    calculate the number of people at each table after the customers left. -/
def people_per_table (initial_customers leaving_customers tables : ℕ) : ℕ :=
  (initial_customers - leaving_customers) / tables

/-- Theorem stating that with 44 initial customers, 12 leaving customers, and 4 tables,
    the number of people at each table after the customers left is 8. -/
theorem waiter_problem :
  people_per_table 44 12 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l2164_216475


namespace NUMINAMATH_CALUDE_andrews_payment_l2164_216493

/-- Calculates the total amount paid for fruits with discounts and taxes -/
def totalAmountPaid (grapeQuantity grapePrice mangoQuantity mangoPrice appleQuantity applePrice orangeQuantity orangePrice discountRate taxRate : ℝ) : ℝ :=
  let grapeCost := grapeQuantity * grapePrice
  let mangoCost := mangoQuantity * mangoPrice
  let appleCost := appleQuantity * applePrice
  let orangeCost := orangeQuantity * orangePrice
  let grapeMangoCost := grapeCost + mangoCost
  let discountAmount := discountRate * grapeMangoCost
  let discountedGrapeMangoCost := grapeMangoCost - discountAmount
  let totalCostBeforeTax := discountedGrapeMangoCost + appleCost + orangeCost
  let taxAmount := taxRate * totalCostBeforeTax
  totalCostBeforeTax + taxAmount

/-- Theorem stating that Andrew's total payment is $1306.41 -/
theorem andrews_payment :
  totalAmountPaid 7 68 9 48 5 55 4 38 0.1 0.05 = 1306.41 := by
  sorry

end NUMINAMATH_CALUDE_andrews_payment_l2164_216493


namespace NUMINAMATH_CALUDE_mailbox_distance_l2164_216400

/-- Represents Jeffrey's walking pattern and the total steps taken -/
structure WalkingPattern where
  forward_steps : ℕ
  backward_steps : ℕ
  total_steps : ℕ

/-- Calculates the effective distance covered given a walking pattern -/
def effectiveDistance (pattern : WalkingPattern) : ℕ :=
  let cycle := pattern.forward_steps + pattern.backward_steps
  let effective_steps_per_cycle := pattern.forward_steps - pattern.backward_steps
  (pattern.total_steps / cycle) * effective_steps_per_cycle

/-- Theorem: Given Jeffrey's walking pattern and total steps, the distance to the mailbox is 110 steps -/
theorem mailbox_distance (pattern : WalkingPattern) 
  (h1 : pattern.forward_steps = 3)
  (h2 : pattern.backward_steps = 2)
  (h3 : pattern.total_steps = 330) :
  effectiveDistance pattern = 110 := by
  sorry

end NUMINAMATH_CALUDE_mailbox_distance_l2164_216400


namespace NUMINAMATH_CALUDE_basement_water_pump_time_l2164_216457

/-- Proves that it takes 450 minutes to pump out water from a basement given specific conditions -/
theorem basement_water_pump_time : 
  let basement_length : ℝ := 30
  let basement_width : ℝ := 40
  let water_depth_inches : ℝ := 24
  let num_pumps : ℕ := 4
  let pump_rate : ℝ := 10  -- gallons per minute
  let cubic_foot_to_gallon : ℝ := 7.5
  let inches_per_foot : ℝ := 12

  let water_depth_feet : ℝ := water_depth_inches / inches_per_foot
  let water_volume_cubic_feet : ℝ := basement_length * basement_width * water_depth_feet
  let water_volume_gallons : ℝ := water_volume_cubic_feet * cubic_foot_to_gallon
  let total_pump_rate : ℝ := pump_rate * num_pumps
  let pump_time_minutes : ℝ := water_volume_gallons / total_pump_rate

  pump_time_minutes = 450 := by sorry

end NUMINAMATH_CALUDE_basement_water_pump_time_l2164_216457


namespace NUMINAMATH_CALUDE_washing_machine_loads_l2164_216433

theorem washing_machine_loads (machine_capacity : ℕ) (total_clothes : ℕ) : 
  machine_capacity = 5 → total_clothes = 53 → 
  (total_clothes + machine_capacity - 1) / machine_capacity = 11 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_loads_l2164_216433


namespace NUMINAMATH_CALUDE_product_of_numbers_l2164_216435

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2164_216435


namespace NUMINAMATH_CALUDE_finger_multiplication_rule_l2164_216461

theorem finger_multiplication_rule (n : ℕ) (h : 1 ≤ n ∧ n ≤ 9) : 9 * n = 10 * (n - 1) + (10 - n) := by
  sorry

end NUMINAMATH_CALUDE_finger_multiplication_rule_l2164_216461


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_zero_not_odd_units_digit_zero_smallest_non_odd_units_digit_l2164_216437

def Digit : Type := { n : ℕ // n < 10 }

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def unitsDigit (n : ℤ) : Digit := 
  ⟨n.natAbs % 10, by sorry⟩

theorem smallest_non_odd_units_digit : 
  ∀ d : Digit, d.val > 0 → ∃ n : ℤ, isOdd n ∧ unitsDigit n = d :=
by sorry

theorem zero_not_odd_units_digit :
  ∀ n : ℤ, isOdd n → unitsDigit n ≠ ⟨0, by sorry⟩ :=
by sorry

theorem zero_smallest_non_odd_units_digit :
  (∀ d : Digit, d.val < 0 → ∃ n : ℤ, isOdd n ∧ unitsDigit n = d) ∧
  (∀ n : ℤ, isOdd n → unitsDigit n ≠ ⟨0, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_zero_not_odd_units_digit_zero_smallest_non_odd_units_digit_l2164_216437


namespace NUMINAMATH_CALUDE_spring_length_at_6kg_l2164_216480

/-- Represents the relationship between weight and spring length -/
def spring_length (initial_length : ℝ) (stretch_rate : ℝ) (weight : ℝ) : ℝ :=
  initial_length + stretch_rate * weight

/-- Theorem stating that a spring with initial length 8 cm and stretch rate 0.5 cm/kg 
    will have a length of 11 cm when a 6 kg weight is hung -/
theorem spring_length_at_6kg 
  (initial_length : ℝ) (stretch_rate : ℝ) (weight : ℝ)
  (h1 : initial_length = 8)
  (h2 : stretch_rate = 0.5)
  (h3 : weight = 6) :
  spring_length initial_length stretch_rate weight = 11 := by
  sorry

end NUMINAMATH_CALUDE_spring_length_at_6kg_l2164_216480


namespace NUMINAMATH_CALUDE_inverted_sand_height_is_25_l2164_216413

/-- Represents the container with frustum and cylinder components -/
structure Container where
  radius : ℝ
  frustumHeight : ℝ
  cylinderHeight : ℝ
  cylinderFillHeight : ℝ

/-- Calculates the total height of sand when the container is inverted -/
def invertedSandHeight (c : Container) : ℝ :=
  c.frustumHeight + c.cylinderFillHeight

/-- Theorem stating the height of sand when the container is inverted -/
theorem inverted_sand_height_is_25 (c : Container) 
  (h_radius : c.radius = 12)
  (h_frustum_height : c.frustumHeight = 20)
  (h_cylinder_height : c.cylinderHeight = 20)
  (h_cylinder_fill : c.cylinderFillHeight = 5) :
  invertedSandHeight c = 25 := by
  sorry

#check inverted_sand_height_is_25

end NUMINAMATH_CALUDE_inverted_sand_height_is_25_l2164_216413


namespace NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_condition_l2164_216431

theorem complex_inequality (z : ℂ) (h : Complex.abs z ≥ 1) :
  (Complex.abs (2 * z - 1))^5 / (25 * Real.sqrt 5) ≥ (Complex.abs (z - 1))^4 / 4 :=
by sorry

theorem complex_inequality_equality_condition (z : ℂ) :
  (Complex.abs (2 * z - 1))^5 / (25 * Real.sqrt 5) = (Complex.abs (z - 1))^4 / 4 ↔
  z = Complex.I ∨ z = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_condition_l2164_216431


namespace NUMINAMATH_CALUDE_courtyard_paving_l2164_216471

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1

theorem courtyard_paving :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l2164_216471


namespace NUMINAMATH_CALUDE_b_current_age_l2164_216472

/-- Given two people A and B, where in 10 years A will be twice as old as B was 10 years ago,
    and A is currently 7 years older than B, prove that B's current age is 37 years. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) → 
  (a = b + 7) → 
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_b_current_age_l2164_216472


namespace NUMINAMATH_CALUDE_unique_number_l2164_216465

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (d : ℕ), d < 10 ∧
    (n - d * 10000 + n) = 54321 ∨
    (n - d * 1000 + n) = 54321 ∨
    (n - d * 100 + n) = 54321 ∨
    (n - d * 10 + n) = 54321 ∨
    (n - d + n) = 54321

theorem unique_number : ∀ n : ℕ, is_valid_number n ↔ n = 49383 := by sorry

end NUMINAMATH_CALUDE_unique_number_l2164_216465
