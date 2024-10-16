import Mathlib

namespace NUMINAMATH_CALUDE_log_power_sum_l2740_274039

theorem log_power_sum (c d : ℝ) (hc : c = Real.log 16) (hd : d = Real.log 25) :
  (9 : ℝ) ^ (c / d) + (4 : ℝ) ^ (d / c) = 4421 / 625 := by
  sorry

end NUMINAMATH_CALUDE_log_power_sum_l2740_274039


namespace NUMINAMATH_CALUDE_not_divisible_by_15_l2740_274029

theorem not_divisible_by_15 : ∀ a : ℤ, ¬(15 ∣ (a^2 + a + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_15_l2740_274029


namespace NUMINAMATH_CALUDE_infinitely_many_common_terms_l2740_274070

/-- Sequence a_n defined by the recurrence relation -/
def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

/-- Sequence b_n defined by the recurrence relation -/
def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

/-- There exist infinitely many pairs of natural numbers (n, m) such that a_n = b_m -/
theorem infinitely_many_common_terms : ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ a n = b m := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_common_terms_l2740_274070


namespace NUMINAMATH_CALUDE_angle_value_for_point_on_terminal_side_l2740_274059

open Real

theorem angle_value_for_point_on_terminal_side :
  ∀ θ : ℝ,
  0 ≤ θ ∧ θ < 2 * π →
  (∃ P : ℝ × ℝ, 
    P.1 = sin (3 * π / 4) ∧ 
    P.2 = cos (3 * π / 4) ∧ 
    P.1 = sin θ ∧ 
    P.2 = cos θ) →
  θ = 7 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_value_for_point_on_terminal_side_l2740_274059


namespace NUMINAMATH_CALUDE_mary_received_more_than_mike_l2740_274073

/-- Represents the profit distribution in a partnership --/
def profit_distribution (mary_investment mike_investment total_profit : ℚ) : ℚ :=
  let equal_share := (1/3) * total_profit / 2
  let ratio_share := (2/3) * total_profit
  let mary_ratio := mary_investment / (mary_investment + mike_investment)
  let mike_ratio := mike_investment / (mary_investment + mike_investment)
  let mary_total := equal_share + mary_ratio * ratio_share
  let mike_total := equal_share + mike_ratio * ratio_share
  mary_total - mike_total

/-- Theorem stating that Mary received $800 more than Mike --/
theorem mary_received_more_than_mike :
  profit_distribution 700 300 3000 = 800 := by
  sorry


end NUMINAMATH_CALUDE_mary_received_more_than_mike_l2740_274073


namespace NUMINAMATH_CALUDE_unique_solution_l2740_274032

def U : Set ℤ := {-2, 3, 4, 5}

def M (p q : ℝ) : Set ℤ := {x ∈ U | (x : ℝ)^2 + p * x + q = 0}

theorem unique_solution :
  ∃! (p q : ℝ), (U \ M p q : Set ℤ) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2740_274032


namespace NUMINAMATH_CALUDE_T_property_M_remainder_l2740_274088

/-- A sequence of positive integers where each number has exactly 9 ones in its binary representation -/
def T : ℕ → ℕ := sorry

/-- The 500th number in the sequence T -/
def M : ℕ := T 500

/-- Predicate to check if a natural number has exactly 9 ones in its binary representation -/
def has_nine_ones (n : ℕ) : Prop := sorry

theorem T_property (n : ℕ) : has_nine_ones (T n) := sorry

theorem M_remainder : M % 500 = 191 := sorry

end NUMINAMATH_CALUDE_T_property_M_remainder_l2740_274088


namespace NUMINAMATH_CALUDE_star_equal_is_four_lines_l2740_274052

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Define the union of four lines
def four_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

-- Theorem statement
theorem star_equal_is_four_lines : star_equal_set = four_lines := by
  sorry

end NUMINAMATH_CALUDE_star_equal_is_four_lines_l2740_274052


namespace NUMINAMATH_CALUDE_largest_non_sum_36_composite_l2740_274064

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k

def is_sum_of_multiple_36_and_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_sum_36_composite : 
  (∀ n > 209, is_sum_of_multiple_36_and_composite n) ∧
  ¬is_sum_of_multiple_36_and_composite 209 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_36_composite_l2740_274064


namespace NUMINAMATH_CALUDE_tangent_line_range_l2740_274009

/-- Given k > 0, if a line can always be drawn through the point (3, 1) to be tangent
    to the circle (x-2k)^2 + (y-k)^2 = k, then k ∈ (0, 1) ∪ (2, +∞) -/
theorem tangent_line_range (k : ℝ) (h_pos : k > 0) 
  (h_tangent : ∀ (x y : ℝ), (x - 2*k)^2 + (y - k)^2 = k → 
    ∃ (m b : ℝ), y = m*x + b ∧ (3 - 2*k)^2 + (1 - k)^2 ≥ k) :
  k ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_range_l2740_274009


namespace NUMINAMATH_CALUDE_multiplication_problem_l2740_274062

theorem multiplication_problem : ∃ x : ℕ, 72516 * x = 724797420 ∧ x = 10001 := by sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2740_274062


namespace NUMINAMATH_CALUDE_probability_all_selected_l2740_274087

/-- The probability of Ram being selected -/
def p_ram : ℚ := 6/7

/-- The initial probability of Ravi being selected -/
def p_ravi_initial : ℚ := 1/5

/-- The probability of Ravi being selected given Ram is selected -/
def p_ravi_given_ram : ℚ := 2/5

/-- The initial probability of Rajesh being selected -/
def p_rajesh_initial : ℚ := 2/3

/-- The probability of Rajesh being selected given Ravi is selected -/
def p_rajesh_given_ravi : ℚ := 1/2

/-- The theorem stating the probability of all three brothers being selected -/
theorem probability_all_selected : 
  p_ram * p_ravi_given_ram * p_rajesh_given_ravi = 6/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_selected_l2740_274087


namespace NUMINAMATH_CALUDE_factory_working_days_l2740_274080

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 4340

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 2170

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days : working_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_factory_working_days_l2740_274080


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_max_negative_l2740_274093

def f (x : ℝ) := |x - 1| + |x + 1|

theorem f_inequality_solution (x : ℝ) :
  f x ≤ 4 ↔ x ∈ Set.Icc (-2) 2 :=
sorry

theorem f_max_negative (b : ℝ) (hb : b ≠ 0) :
  (∀ x, f x ≥ (|2*b + 1| + |1 - b|) / |b|) →
  (∃ x, x < 0 ∧ f x ≥ (|2*b + 1| + |1 - b|) / |b| ∧
    ∀ y, y < 0 → f y ≥ (|2*b + 1| + |1 - b|) / |b| → y ≤ x) →
  x = -1.5 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_max_negative_l2740_274093


namespace NUMINAMATH_CALUDE_variance_scaling_l2740_274042

variable {n : ℕ}
variable (a : Fin n → ℝ)

/-- The variance of a dataset -/
def variance (x : Fin n → ℝ) : ℝ := sorry

/-- The scaled dataset where each element is multiplied by 2 -/
def scaled_data (x : Fin n → ℝ) : Fin n → ℝ := λ i => 2 * x i

theorem variance_scaling (h : variance a = 4) : 
  variance (scaled_data a) = 16 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l2740_274042


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2740_274050

theorem trigonometric_identities :
  -- Part 1
  ¬∃x : ℝ, x = Real.sin (-14 / 3 * π) + Real.cos (20 / 3 * π) + Real.tan (-53 / 6 * π) ∧
  -- Part 2
  Real.tan (675 * π / 180) - Real.sin (-330 * π / 180) - Real.cos (960 * π / 180) = 0 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2740_274050


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2740_274079

/-- Given vectors a and b in ℝ², where a is parallel to b, prove their dot product is -5 -/
theorem parallel_vectors_dot_product (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, x - 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∃ (k : ℝ), a = k • b) →
  (a 0 * b 0 + a 1 * b 1 = -5) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l2740_274079


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2740_274074

theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (h : w < z) :
  let l := z - w
  2 * (l + w) = 2 * z :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2740_274074


namespace NUMINAMATH_CALUDE_clock_hand_overlaps_l2740_274035

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_hand : ℝ)
  (minute_hand : ℝ)

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of overlaps in a 12-hour period -/
def overlaps_per_half_day : ℕ := 11

/-- Calculates the number of times the hour and minute hands overlap in a day -/
def overlaps_per_day (c : Clock) : ℕ :=
  2 * overlaps_per_half_day

/-- Theorem: The number of times the hour and minute hands of a clock overlap in a 24-hour day is 22 -/
theorem clock_hand_overlaps :
  ∀ c : Clock, overlaps_per_day c = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_hand_overlaps_l2740_274035


namespace NUMINAMATH_CALUDE_angle_330_equivalent_to_negative_30_l2740_274026

/-- Two angles have the same terminal side if they are equivalent modulo 360° -/
def same_terminal_side (a b : ℝ) : Prop := a % 360 = b % 360

/-- The problem statement -/
theorem angle_330_equivalent_to_negative_30 :
  same_terminal_side 330 (-30) := by sorry

end NUMINAMATH_CALUDE_angle_330_equivalent_to_negative_30_l2740_274026


namespace NUMINAMATH_CALUDE_weighted_cauchy_schwarz_l2740_274084

theorem weighted_cauchy_schwarz (p q x y : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q < 1) : 
  (p * x + q * y)^2 ≤ p * x^2 + q * y^2 := by
  sorry

end NUMINAMATH_CALUDE_weighted_cauchy_schwarz_l2740_274084


namespace NUMINAMATH_CALUDE_lcm_ten_times_gcd_characterization_l2740_274028

theorem lcm_ten_times_gcd_characterization (a b : ℕ+) :
  Nat.lcm a b = 10 * Nat.gcd a b ↔
  (∃ d : ℕ+, (a = d ∧ b = 10 * d) ∨
             (a = 2 * d ∧ b = 5 * d) ∨
             (a = 5 * d ∧ b = 2 * d) ∨
             (a = 10 * d ∧ b = d)) :=
by sorry

end NUMINAMATH_CALUDE_lcm_ten_times_gcd_characterization_l2740_274028


namespace NUMINAMATH_CALUDE_matrix_homomorphism_implies_equal_dim_l2740_274051

-- Define the set of valid dimensions
def ValidDim : Set ℕ := {2, 3}

-- Define the property of the bijective function
def IsMatrixHomomorphism {n p : ℕ} (f : Matrix (Fin n) (Fin n) ℂ → Matrix (Fin p) (Fin p) ℂ) : Prop :=
  ∀ X Y : Matrix (Fin n) (Fin n) ℂ, f (X * Y) = f X * f Y

-- The main theorem
theorem matrix_homomorphism_implies_equal_dim (n p : ℕ) 
  (hn : n ∈ ValidDim) (hp : p ∈ ValidDim) :
  (∃ f : Matrix (Fin n) (Fin n) ℂ → Matrix (Fin p) (Fin p) ℂ, 
    Function.Bijective f ∧ IsMatrixHomomorphism f) → n = p := by
  sorry

end NUMINAMATH_CALUDE_matrix_homomorphism_implies_equal_dim_l2740_274051


namespace NUMINAMATH_CALUDE_remaining_balloons_l2740_274030

theorem remaining_balloons (fred_balloons sam_balloons destroyed_balloons : ℝ)
  (h1 : fred_balloons = 10.0)
  (h2 : sam_balloons = 46.0)
  (h3 : destroyed_balloons = 16.0) :
  fred_balloons + sam_balloons - destroyed_balloons = 40.0 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l2740_274030


namespace NUMINAMATH_CALUDE_nails_to_buy_l2740_274021

theorem nails_to_buy (tom_nails : ℝ) (toolshed_nails : ℝ) (drawer_nail : ℝ) (neighbor_nails : ℝ) (total_needed : ℝ) :
  tom_nails = 247 →
  toolshed_nails = 144 →
  drawer_nail = 0.5 →
  neighbor_nails = 58.75 →
  total_needed = 625.25 →
  total_needed - (tom_nails + toolshed_nails + drawer_nail + neighbor_nails) = 175 := by
  sorry

end NUMINAMATH_CALUDE_nails_to_buy_l2740_274021


namespace NUMINAMATH_CALUDE_octagon_sectors_area_l2740_274077

/-- The area of the region inside a regular octagon with side length 8 but outside
    the circular sectors with radius 4 centered at each vertex. -/
theorem octagon_sectors_area : 
  let side_length : ℝ := 8
  let sector_radius : ℝ := 4
  let octagon_area : ℝ := 8 * (1/2 * side_length^2 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2))
  let sectors_area : ℝ := 8 * (π * sector_radius^2 / 8)
  octagon_area - sectors_area = 256 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2) - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_octagon_sectors_area_l2740_274077


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l2740_274040

theorem square_sum_equals_one (a b : ℝ) 
  (h : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1) : 
  a^2 + b^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l2740_274040


namespace NUMINAMATH_CALUDE_junior_score_l2740_274056

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (class_avg : ℝ) (senior_avg : ℝ) (junior_score : ℝ) :
  junior_ratio = 0.2 →
  senior_ratio = 0.8 →
  junior_ratio + senior_ratio = 1 →
  class_avg = 75 →
  senior_avg = 72 →
  class_avg * n = senior_avg * (senior_ratio * n) + junior_score * (junior_ratio * n) →
  junior_score = 87 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l2740_274056


namespace NUMINAMATH_CALUDE_set_operations_l2740_274047

def A : Set ℤ := {x : ℤ | |x| < 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5}

theorem set_operations :
  (B ∩ C = {3}) ∧
  (B ∪ C = {1, 2, 3, 4, 5}) ∧
  (A ∪ (B ∩ C) = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}) ∧
  (A ∩ (A \ (B ∪ C)) = {-5, -4, -3, -2, -1, 0}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l2740_274047


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l2740_274025

/-- The probability of flipping exactly k tails in n flips of an unfair coin -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 3 tails in 8 flips of an unfair coin with 2/3 probability of tails -/
theorem unfair_coin_flip_probability : 
  binomial_probability 8 3 (2/3) = 448/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l2740_274025


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2740_274072

theorem sum_of_two_numbers (a b : ℕ) : a = 22 ∧ b = a - 10 → a + b = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2740_274072


namespace NUMINAMATH_CALUDE_equation_solution_l2740_274096

theorem equation_solution (x : ℝ) : 
  (Real.cos (2 * x / 5) - Real.cos (2 * Real.pi / 15))^2 + 
  (Real.sin (2 * x / 3) - Real.sin (4 * Real.pi / 9))^2 = 0 ↔ 
  ∃ t : ℤ, x = 29 * Real.pi / 3 + 15 * Real.pi * (t : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2740_274096


namespace NUMINAMATH_CALUDE_coal_relationship_warehouse_b_coal_amount_l2740_274063

/-- The amount of coal in warehouse A in tons -/
def warehouse_a_coal : ℝ := 130

/-- The amount of coal in warehouse B in tons -/
def warehouse_b_coal : ℝ := 150

/-- Theorem stating the relationship between coal in warehouses A and B -/
theorem coal_relationship : warehouse_a_coal = 0.8 * warehouse_b_coal + 10 := by
  sorry

/-- Theorem proving the amount of coal in warehouse B -/
theorem warehouse_b_coal_amount : warehouse_b_coal = 150 := by
  sorry

end NUMINAMATH_CALUDE_coal_relationship_warehouse_b_coal_amount_l2740_274063


namespace NUMINAMATH_CALUDE_product_invariance_l2740_274002

theorem product_invariance (a b : ℝ) (h : a * b = 300) :
  (6 * a) * (b / 6) = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_invariance_l2740_274002


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2740_274014

/-- Given a quadratic function y = 2x^2 + px + q, 
    prove that q = 10 + p^2/8 when the minimum value of y is 10 -/
theorem quadratic_minimum (p : ℝ) :
  ∃ (q : ℝ), (∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) ∧
             (∃ x₀ : ℝ, 2 * x₀^2 + p * x₀ + q = 10) →
  q = 10 + p^2 / 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2740_274014


namespace NUMINAMATH_CALUDE_people_visited_neither_l2740_274081

theorem people_visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 100 →
  iceland = 55 →
  norway = 43 →
  both = 61 →
  total - (iceland + norway - both) = 63 := by
  sorry

end NUMINAMATH_CALUDE_people_visited_neither_l2740_274081


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l2740_274034

/-- The polynomial function P(x) = x^4 + 2x^3 - 2x^2 - 4x - 5 -/
def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 2*x^2 - 4*x - 5

/-- A function is a perfect square if there exists a real function g such that f(x) = g(x)^2 for all x -/
def is_perfect_square (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = (g x)^2

theorem polynomial_perfect_square :
  ∀ x : ℝ, is_perfect_square P ↔ (x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l2740_274034


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l2740_274043

theorem set_equality_implies_difference (a b : ℝ) :
  ({0, b/a, b} : Set ℝ) = ({1, a+b, a} : Set ℝ) →
  b - a = 2 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l2740_274043


namespace NUMINAMATH_CALUDE_choir_size_proof_l2740_274004

theorem choir_size_proof : Nat.lcm (Nat.lcm 9 10) 11 = 990 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_proof_l2740_274004


namespace NUMINAMATH_CALUDE_percentage_correct_second_question_l2740_274048

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered the second question correctly. -/
theorem percentage_correct_second_question
  (total : ℝ) -- Total number of students
  (first_correct : ℝ) -- Number of students who answered the first question correctly
  (both_correct : ℝ) -- Number of students who answered both questions correctly
  (neither_correct : ℝ) -- Number of students who answered neither question correctly
  (h1 : first_correct = 0.75 * total) -- 75% answered the first question correctly
  (h2 : both_correct = 0.25 * total) -- 25% answered both questions correctly
  (h3 : neither_correct = 0.2 * total) -- 20% answered neither question correctly
  : (total - neither_correct - (first_correct - both_correct)) / total = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_percentage_correct_second_question_l2740_274048


namespace NUMINAMATH_CALUDE_pond_length_l2740_274038

theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) : 
  field_length = 48 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 18 →
  Real.sqrt pond_area = 8 := by
sorry

end NUMINAMATH_CALUDE_pond_length_l2740_274038


namespace NUMINAMATH_CALUDE_prob_not_snowing_l2740_274017

theorem prob_not_snowing (p_snow : ℚ) (h : p_snow = 1/4) : 1 - p_snow = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_snowing_l2740_274017


namespace NUMINAMATH_CALUDE_units_digit_13_times_41_l2740_274069

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The theorem stating that the units digit of 13 · 41 is 3 -/
theorem units_digit_13_times_41 : unitsDigit (13 * 41) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_13_times_41_l2740_274069


namespace NUMINAMATH_CALUDE_fraction_addition_l2740_274036

theorem fraction_addition : (11 : ℚ) / 12 + 7 / 15 = 83 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2740_274036


namespace NUMINAMATH_CALUDE_set_operations_l2740_274031

def A : Set ℕ := {x | x > 0 ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem set_operations :
  (A ∩ C = {3, 4, 5, 6, 7}) ∧
  ((A \ B) = {5, 6, 7, 8, 9, 10}) ∧
  ((A \ (B ∪ C)) = {8, 9, 10}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l2740_274031


namespace NUMINAMATH_CALUDE_johann_oranges_l2740_274067

def orange_problem (initial_oranges eaten_oranges stolen_fraction returned_oranges : ℕ) : Prop :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen := (remaining_after_eating / 2 : ℕ)
  let final_count := remaining_after_eating - stolen + returned_oranges
  final_count = 30

theorem johann_oranges :
  orange_problem 60 10 2 5 := by sorry

end NUMINAMATH_CALUDE_johann_oranges_l2740_274067


namespace NUMINAMATH_CALUDE_parallelogram_area_l2740_274010

-- Define the lines
def L1 (x y : ℝ) : Prop := y = 2
def L2 (x y : ℝ) : Prop := y = -2
def L3 (x y : ℝ) : Prop := 4 * x + 7 * y - 10 = 0
def L4 (x y : ℝ) : Prop := 4 * x + 7 * y + 20 = 0

-- Define the vertices of the parallelogram
def A : ℝ × ℝ := (-1.5, -2)
def B : ℝ × ℝ := (6, -2)
def C : ℝ × ℝ := (-1, 2)
def D : ℝ × ℝ := (-8.5, 2)

-- State the theorem
theorem parallelogram_area : 
  (A.1 = -1.5 ∧ A.2 = -2) →
  (B.1 = 6 ∧ B.2 = -2) →
  (C.1 = -1 ∧ C.2 = 2) →
  (D.1 = -8.5 ∧ D.2 = 2) →
  L1 C.1 C.2 →
  L1 D.1 D.2 →
  L2 A.1 A.2 →
  L2 B.1 B.2 →
  L3 A.1 A.2 →
  L3 C.1 C.2 →
  L4 B.1 B.2 →
  L4 D.1 D.2 →
  (B.1 - A.1) * (C.2 - A.2) = 30 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2740_274010


namespace NUMINAMATH_CALUDE_coins_per_pile_l2740_274022

theorem coins_per_pile (total_piles : ℕ) (total_coins : ℕ) (h1 : total_piles = 10) (h2 : total_coins = 30) :
  ∃ (coins_per_pile : ℕ), coins_per_pile * total_piles = total_coins ∧ coins_per_pile = 3 :=
sorry

end NUMINAMATH_CALUDE_coins_per_pile_l2740_274022


namespace NUMINAMATH_CALUDE_revenue_change_l2740_274018

theorem revenue_change 
  (price_increase : ℝ) 
  (sales_decrease : ℝ) 
  (price_increase_percent : price_increase = 30) 
  (sales_decrease_percent : sales_decrease = 20) : 
  (1 + price_increase / 100) * (1 - sales_decrease / 100) - 1 = 0.04 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l2740_274018


namespace NUMINAMATH_CALUDE_angle_equality_in_triangle_l2740_274037

/-- Given an acute triangle ABC with its circumcircle, tangents at A and B intersecting at D,
    and M as the midpoint of AB, prove that ∠ACM = ∠BCD. -/
theorem angle_equality_in_triangle (A B C D M : ℂ) : 
  -- A, B, C are on the unit circle (representing the circumcircle)
  Complex.abs A = 1 ∧ Complex.abs B = 1 ∧ Complex.abs C = 1 →
  -- Triangle ABC is acute
  (0 < Real.cos (Complex.arg (B - A) - Complex.arg (C - A))) ∧
  (0 < Real.cos (Complex.arg (C - B) - Complex.arg (A - B))) ∧
  (0 < Real.cos (Complex.arg (A - C) - Complex.arg (B - C))) →
  -- D is the intersection of tangents at A and B
  D = (2 * A * B) / (A + B) →
  -- M is the midpoint of AB
  M = (A + B) / 2 →
  -- Conclusion: ∠ACM = ∠BCD
  Complex.arg ((M - C) / (A - C)) = Complex.arg ((B - C) / (D - C)) := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_in_triangle_l2740_274037


namespace NUMINAMATH_CALUDE_smallest_valid_six_digit_number_l2740_274046

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2

def append_three_digits (n : ℕ) (m : ℕ) : ℕ :=
  n * 1000 + m

theorem smallest_valid_six_digit_number :
  ∃ (n m : ℕ),
    is_valid_three_digit n ∧
    m < 1000 ∧
    let six_digit := append_three_digits n m
    six_digit = 122040 ∧
    six_digit % 4 = 0 ∧
    six_digit % 5 = 0 ∧
    six_digit % 6 = 0 ∧
    ∀ (n' m' : ℕ),
      is_valid_three_digit n' ∧
      m' < 1000 ∧
      let six_digit' := append_three_digits n' m'
      six_digit' % 4 = 0 ∧
      six_digit' % 5 = 0 ∧
      six_digit' % 6 = 0 →
      six_digit ≤ six_digit' :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_six_digit_number_l2740_274046


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2740_274066

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_34 : ℚ := 34 / 99

theorem product_of_repeating_decimals :
  repeating_decimal_12 * repeating_decimal_34 = 136 / 3267 :=
by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2740_274066


namespace NUMINAMATH_CALUDE_bakers_new_cakes_l2740_274099

/-- Baker's cake problem -/
theorem bakers_new_cakes 
  (initial_cakes : ℕ) 
  (sold_cakes : ℕ) 
  (difference : ℕ) 
  (h1 : initial_cakes = 13) 
  (h2 : sold_cakes = 91) 
  (h3 : difference = 63) : 
  sold_cakes + difference = 154 := by
  sorry

end NUMINAMATH_CALUDE_bakers_new_cakes_l2740_274099


namespace NUMINAMATH_CALUDE_line_equation_sum_l2740_274098

/-- Proves that for a line passing through (1, -2) and (4, 7), m + b = -2 --/
theorem line_equation_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → 
    ((x = 1 ∧ y = -2) ∨ (x = 4 ∧ y = 7))) → 
  m + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_sum_l2740_274098


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2740_274085

theorem magnitude_of_z (z : ℂ) (h : z^2 = 24 - 32*I) : Complex.abs z = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2740_274085


namespace NUMINAMATH_CALUDE_prime_equation_l2740_274020

theorem prime_equation (a b : ℕ) : 
  Prime a → Prime b → a^11 + b = 2089 → 49*b - a = 2007 := by sorry

end NUMINAMATH_CALUDE_prime_equation_l2740_274020


namespace NUMINAMATH_CALUDE_radii_product_l2740_274006

/-- Two circles C₁ and C₂ with centers (2, 2) and (-1, -1) respectively, 
    radii r₁ and r₂ (both positive), that are tangent to each other, 
    and have an external common tangent line with a slope of 7. -/
structure TangentCircles where
  r₁ : ℝ
  r₂ : ℝ
  h₁ : r₁ > 0
  h₂ : r₂ > 0
  h_tangent : (r₁ + r₂)^2 = (2 - (-1))^2 + (2 - (-1))^2  -- Distance between centers equals sum of radii
  h_slope : ∃ t : ℝ, (7 * 2 - 2 + t)^2 / 50 = r₁^2 ∧ (7 * (-1) - (-1) + t)^2 / 50 = r₂^2

/-- The product of the radii of two tangent circles with the given properties is 72/25. -/
theorem radii_product (c : TangentCircles) : c.r₁ * c.r₂ = 72 / 25 := by
  sorry

end NUMINAMATH_CALUDE_radii_product_l2740_274006


namespace NUMINAMATH_CALUDE_water_cup_pricing_equation_l2740_274083

/-- Represents the pricing of a Huashan brand water cup -/
def water_cup_pricing (x : ℝ) : Prop :=
  let first_discount := x - 5
  let second_discount := 0.8 * first_discount
  second_discount = 60

/-- The equation representing the water cup pricing after discounts -/
theorem water_cup_pricing_equation (x : ℝ) :
  water_cup_pricing x ↔ 0.8 * (x - 5) = 60 := by sorry

end NUMINAMATH_CALUDE_water_cup_pricing_equation_l2740_274083


namespace NUMINAMATH_CALUDE_fabian_shopping_cost_l2740_274024

/-- The cost of Fabian's shopping trip -/
def shopping_cost (apple_price : ℝ) (walnut_price : ℝ) (apple_quantity : ℝ) (sugar_quantity : ℝ) (walnut_quantity : ℝ) : ℝ :=
  let sugar_price := apple_price - 1
  apple_price * apple_quantity + sugar_price * sugar_quantity + walnut_price * walnut_quantity

/-- Theorem: The total cost of Fabian's shopping is $16 -/
theorem fabian_shopping_cost : 
  shopping_cost 2 6 5 3 0.5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fabian_shopping_cost_l2740_274024


namespace NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l2740_274019

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (α β : Plane) (a : Line)
  (h_a_in_α : line_in_plane a α) :
  (∀ α β a, plane_parallel α β → line_plane_parallel a β) ∧
  (∃ α β a, line_plane_parallel a β ∧ ¬ plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_sufficient_not_necessary_l2740_274019


namespace NUMINAMATH_CALUDE_bus_boarding_problem_l2740_274012

theorem bus_boarding_problem (total_rows : Nat) (seats_per_row : Nat) 
  (initial_boarding : Nat) (first_stop_exit : Nat) (second_stop_boarding : Nat) 
  (second_stop_exit : Nat) (final_empty_seats : Nat) :
  let total_seats := total_rows * seats_per_row
  let empty_seats_after_start := total_seats - initial_boarding
  let first_stop_boarding := total_seats - empty_seats_after_start + first_stop_exit - 
    (total_seats - (empty_seats_after_start - (second_stop_boarding - second_stop_exit) - final_empty_seats))
  total_rows = 23 →
  seats_per_row = 4 →
  initial_boarding = 16 →
  first_stop_exit = 3 →
  second_stop_boarding = 17 →
  second_stop_exit = 10 →
  final_empty_seats = 57 →
  first_stop_boarding = 15 := by
    sorry

#check bus_boarding_problem

end NUMINAMATH_CALUDE_bus_boarding_problem_l2740_274012


namespace NUMINAMATH_CALUDE_orchid_count_l2740_274045

/-- Time in minutes to paint each type of flower or vine -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def vine_time : ℕ := 2

/-- Number of each type of flower or vine painted -/
def lily_count : ℕ := 17
def rose_count : ℕ := 10
def vine_count : ℕ := 20

/-- Total time spent painting -/
def total_time : ℕ := 213

/-- Theorem stating the number of orchids painted -/
theorem orchid_count : 
  ∃ (x : ℕ), x * orchid_time = total_time - (lily_count * lily_time + rose_count * rose_time + vine_count * vine_time) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_orchid_count_l2740_274045


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2740_274090

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem f_decreasing_on_interval : 
  ∀ x ∈ Set.Ioo (-2 : ℝ) 1, (deriv f) x < 0 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2740_274090


namespace NUMINAMATH_CALUDE_divisibility_of_linear_combination_l2740_274008

theorem divisibility_of_linear_combination (a b c : ℕ+) : 
  ∃ (r s : ℕ+), (Nat.gcd r s = 1) ∧ (∃ k : ℤ, (a : ℤ) * (r : ℤ) + (b : ℤ) * (s : ℤ) = k * (c : ℤ)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_linear_combination_l2740_274008


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2740_274058

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k r : ℝ) (hr : r ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = a * ((x - h)^2 / r^2) + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2740_274058


namespace NUMINAMATH_CALUDE_wig_cost_calculation_l2740_274011

-- Define the given conditions
def total_plays : ℕ := 3
def acts_per_play : ℕ := 5
def wigs_per_act : ℕ := 2
def dropped_play_sale : ℚ := 4
def total_spent : ℚ := 110

-- Define the theorem
theorem wig_cost_calculation :
  let wigs_per_play := acts_per_play * wigs_per_act
  let total_wigs := total_plays * wigs_per_play
  let remaining_wigs := total_wigs - wigs_per_play
  let cost_per_wig := total_spent / remaining_wigs
  cost_per_wig = 5.5 := by sorry

end NUMINAMATH_CALUDE_wig_cost_calculation_l2740_274011


namespace NUMINAMATH_CALUDE_traffic_light_theorem_l2740_274097

/-- Represents the probability of different traffic light combinations -/
structure TrafficLightProbabilities where
  p1 : ℝ  -- Both lights green
  p2 : ℝ  -- First green, second red
  p3 : ℝ  -- First red, second green
  p4 : ℝ  -- Both lights red

/-- The conditions of the traffic light problem -/
def traffic_light_conditions (p : TrafficLightProbabilities) : Prop :=
  0 ≤ p.p1 ∧ 0 ≤ p.p2 ∧ 0 ≤ p.p3 ∧ 0 ≤ p.p4 ∧  -- Probabilities are non-negative
  p.p1 + p.p2 + p.p3 + p.p4 = 1 ∧  -- Sum of probabilities is 1
  p.p1 + p.p2 = 2/3 ∧  -- First light is green 2/3 of the time
  p.p1 + p.p3 = 2/3 ∧  -- Second light is green 2/3 of the time
  p.p1 / (p.p1 + p.p2) = 3/4  -- Given first is green, second is green 3/4 of the time

/-- The theorem to be proved -/
theorem traffic_light_theorem (p : TrafficLightProbabilities) 
  (h : traffic_light_conditions p) : 
  p.p4 / (p.p3 + p.p4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_theorem_l2740_274097


namespace NUMINAMATH_CALUDE_h_derivative_l2740_274092

/-- Given f = 5, g = 4g', and h(x) = (f + 2) / x, prove that h'(x) = 5/16 -/
theorem h_derivative (f g g' : ℝ) (h : ℝ → ℝ) :
  f = 5 →
  g = 4 * g' →
  (∀ x, h x = (f + 2) / x) →
  ∀ x, deriv h x = 5 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_h_derivative_l2740_274092


namespace NUMINAMATH_CALUDE_stratified_sampling_group_a_l2740_274065

/-- Calculates the number of cities to be selected from a group in stratified sampling -/
def stratifiedSampleSize (totalCities : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℚ :=
  (groupSize : ℚ) * (sampleSize : ℚ) / (totalCities : ℚ)

/-- Theorem: In a stratified sampling of 6 cities from 24 total cities, 
    where 4 cities belong to group A, 1 city should be selected from group A -/
theorem stratified_sampling_group_a : 
  stratifiedSampleSize 24 4 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_a_l2740_274065


namespace NUMINAMATH_CALUDE_part_one_part_two_l2740_274016

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 3
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Part I
theorem part_one :
  let S := {x : ℝ | (p x ∨ q x 2) ∧ ¬(p x ∧ q x 2)}
  S = {x : ℝ | -4 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} :=
sorry

-- Part II
theorem part_two :
  let T := {m : ℝ | m > 0 ∧ {x : ℝ | p x} ⊃ {x : ℝ | q x m} ∧ {x : ℝ | p x} ≠ {x : ℝ | q x m}}
  T = {m : ℝ | 0 < m ∧ m ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2740_274016


namespace NUMINAMATH_CALUDE_movie_and_popcorn_expense_l2740_274049

/-- The fraction of allowance spent on movie ticket and popcorn -/
theorem movie_and_popcorn_expense (B : ℝ) (m p : ℝ) 
  (hm : m = (1/4) * (B - p)) 
  (hp : p = (1/10) * (B - m)) : 
  (m + p) / B = 4/13 := by
  sorry

end NUMINAMATH_CALUDE_movie_and_popcorn_expense_l2740_274049


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l2740_274000

theorem slope_angle_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y + 5 = 0 →
  Real.arctan (-Real.sqrt 3 / 3) = 150 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l2740_274000


namespace NUMINAMATH_CALUDE_xy_equality_l2740_274003

theorem xy_equality (x y : ℝ) : 4 * x * y - 3 * x * y = x * y := by
  sorry

end NUMINAMATH_CALUDE_xy_equality_l2740_274003


namespace NUMINAMATH_CALUDE_arc_length_for_60_degrees_l2740_274086

/-- Given a circle with radius 10 cm and a central angle of 60°, 
    the length of the corresponding arc is 10π/3 cm. -/
theorem arc_length_for_60_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 10 → θ = 60 * π / 180 → l = r * θ → l = 10 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_60_degrees_l2740_274086


namespace NUMINAMATH_CALUDE_round_robin_equation_l2740_274005

/-- Represents a round-robin tournament -/
structure RoundRobinTournament where
  teams : ℕ
  total_games : ℕ
  games_formula : total_games = teams * (teams - 1) / 2

/-- Theorem: In a round-robin tournament with 45 total games, the equation x(x-1) = 2 * 45 holds true -/
theorem round_robin_equation (t : RoundRobinTournament) (h : t.total_games = 45) :
  t.teams * (t.teams - 1) = 2 * 45 := by
  sorry


end NUMINAMATH_CALUDE_round_robin_equation_l2740_274005


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2740_274094

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : Nat), 
    Prime p ∧ Prime q ∧ 
    p ≤ 50 ∧ q ≤ 50 ∧
    (∀ r, Prime r ∧ r ≤ 50 → p ≤ r) ∧
    (∀ r, Prime r ∧ r ≤ 50 → r ≤ q) ∧
    p + q = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2740_274094


namespace NUMINAMATH_CALUDE_negation_equivalence_l2740_274095

theorem negation_equivalence (a : ℝ) : (¬(a < 0)) ↔ (¬(a^2 > a)) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2740_274095


namespace NUMINAMATH_CALUDE_even_function_decreasing_interval_l2740_274076

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem even_function_decreasing_interval (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) →
  (∃ a : ℝ, ∀ x y : ℝ, x < y ∧ y ≤ 0 → f k x > f k y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f k x < f k y) :=
sorry

end NUMINAMATH_CALUDE_even_function_decreasing_interval_l2740_274076


namespace NUMINAMATH_CALUDE_f_value_plus_derivative_at_pi_half_l2740_274055

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_value_plus_derivative_at_pi_half (π : ℝ) (h : π > 0) :
  f π + (deriv f) (π / 2) = -3 / π :=
sorry

end NUMINAMATH_CALUDE_f_value_plus_derivative_at_pi_half_l2740_274055


namespace NUMINAMATH_CALUDE_inequality_proof_l2740_274061

theorem inequality_proof (a : ℝ) (h : a ≠ -1) :
  (1 + a^3) / ((1 + a)^3) ≥ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2740_274061


namespace NUMINAMATH_CALUDE_sqrt_of_one_plus_three_l2740_274057

theorem sqrt_of_one_plus_three : Real.sqrt (1 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_one_plus_three_l2740_274057


namespace NUMINAMATH_CALUDE_probability_of_odd_product_l2740_274075

def range_start : ℕ := 4
def range_end : ℕ := 16

def count_integers : ℕ := range_end - range_start + 1
def count_odd_integers : ℕ := (range_end - range_start + 1) / 2

def total_combinations : ℕ := count_integers.choose 3
def odd_combinations : ℕ := count_odd_integers.choose 3

theorem probability_of_odd_product :
  (odd_combinations : ℚ) / total_combinations = 10 / 143 :=
sorry

end NUMINAMATH_CALUDE_probability_of_odd_product_l2740_274075


namespace NUMINAMATH_CALUDE_ship_optimal_speed_and_cost_l2740_274091

/-- The optimal speed and cost for a ship's journey -/
theorem ship_optimal_speed_and_cost (distance : ℝ) (fuel_cost_coeff : ℝ) (fixed_cost : ℝ)
  (h_distance : distance = 100)
  (h_fuel_cost : fuel_cost_coeff = 0.005)
  (h_fixed_cost : fixed_cost = 80) :
  ∃ (optimal_speed : ℝ) (min_cost : ℝ),
    optimal_speed = 20 ∧
    min_cost = 600 ∧
    ∀ (v : ℝ), v > 0 →
      distance / v * (fuel_cost_coeff * v^3 + fixed_cost) ≥ min_cost :=
by sorry

end NUMINAMATH_CALUDE_ship_optimal_speed_and_cost_l2740_274091


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2740_274041

/-- The focal length of an ellipse with equation x^2/25 + y^2/16 = 1 is 6 -/
theorem ellipse_focal_length : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 - b^2)
  2 * c = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l2740_274041


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2740_274023

theorem arithmetic_sequence_length
  (a₁ : ℤ)
  (aₙ : ℤ)
  (d : ℤ)
  (h1 : a₁ = -3)
  (h2 : aₙ = 45)
  (h3 : d = 4)
  (h4 : aₙ = a₁ + (n - 1) * d) :
  n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2740_274023


namespace NUMINAMATH_CALUDE_wire_ratio_l2740_274033

/-- Given a wire of length 21 cm cut into two pieces, where the shorter piece is 5.999999999999998 cm long,
    prove that the ratio of the shorter piece to the longer piece is 2:5. -/
theorem wire_ratio (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 21 →
  shorter_length = 5.999999999999998 →
  let longer_length := total_length - shorter_length
  shorter_length / longer_length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_l2740_274033


namespace NUMINAMATH_CALUDE_amanda_walk_distance_l2740_274007

/-- Amanda's walk to Kimberly's house -/
theorem amanda_walk_distance :
  let initial_speed : ℝ := 2
  let time_before_break : ℝ := 1.5
  let break_duration : ℝ := 0.5
  let speed_after_break : ℝ := 3
  let total_time : ℝ := 3.5
  let distance_before_break := initial_speed * time_before_break
  let time_after_break := total_time - break_duration - time_before_break
  let distance_after_break := speed_after_break * time_after_break
  let total_distance := distance_before_break + distance_after_break
  total_distance = 7.5 := by sorry

end NUMINAMATH_CALUDE_amanda_walk_distance_l2740_274007


namespace NUMINAMATH_CALUDE_fraction_zero_l2740_274060

theorem fraction_zero (x : ℝ) (h : x ≠ 0) : (x + 1) / x = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l2740_274060


namespace NUMINAMATH_CALUDE_cos_difference_from_sum_of_sin_and_cos_l2740_274027

theorem cos_difference_from_sum_of_sin_and_cos 
  (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 5/4) : 
  Real.cos (A - B) = 13/32 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_from_sum_of_sin_and_cos_l2740_274027


namespace NUMINAMATH_CALUDE_find_m_value_l2740_274001

/-- Given two functions f and g, prove that m equals 10/7 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 - 3*x + m) →
  (∀ x, g x = x^2 - 3*x + 5*m) →
  3 * f 5 = 2 * g 5 →
  m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l2740_274001


namespace NUMINAMATH_CALUDE_probability_even_sum_and_same_number_l2740_274071

/-- A fair six-sided die -/
def Die : Type := Fin 6

/-- The outcome of rolling two dice -/
def RollOutcome : Type := Die × Die

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : Nat := 36

/-- Predicate for checking if a roll outcome has an even sum -/
def hasEvenSum (roll : RollOutcome) : Prop :=
  (roll.1.val + 1 + roll.2.val + 1) % 2 = 0

/-- Predicate for checking if both dice show the same number -/
def hasSameNumber (roll : RollOutcome) : Prop :=
  roll.1 = roll.2

/-- The set of favorable outcomes (even sum and same number) -/
def favorableOutcomes : Finset RollOutcome :=
  sorry

/-- The number of favorable outcomes -/
def numFavorableOutcomes : Nat :=
  favorableOutcomes.card

theorem probability_even_sum_and_same_number :
  (numFavorableOutcomes : ℚ) / totalOutcomes = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_probability_even_sum_and_same_number_l2740_274071


namespace NUMINAMATH_CALUDE_terminal_side_of_negative_400_degrees_l2740_274044

/-- The quadrant of an angle in degrees -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Normalizes an angle to the range [0, 360) -/
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

/-- Determines the quadrant of a normalized angle -/
def quadrantOfNormalizedAngle (angle : Int) : Quadrant :=
  if 0 ≤ angle ∧ angle < 90 then Quadrant.first
  else if 90 ≤ angle ∧ angle < 180 then Quadrant.second
  else if 180 ≤ angle ∧ angle < 270 then Quadrant.third
  else Quadrant.fourth

/-- Determines the quadrant of any angle -/
def quadrantOfAngle (angle : Int) : Quadrant :=
  quadrantOfNormalizedAngle (normalizeAngle angle)

theorem terminal_side_of_negative_400_degrees :
  quadrantOfAngle (-400) = Quadrant.fourth := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_of_negative_400_degrees_l2740_274044


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2740_274078

theorem arithmetic_operations : 
  ((-9) + ((-4) * 5) = -29) ∧ 
  ((6 * (-2)) / (2/3) = -18) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2740_274078


namespace NUMINAMATH_CALUDE_square_diagonal_l2740_274082

theorem square_diagonal (perimeter : ℝ) (h : perimeter = 28) :
  let side := perimeter / 4
  let diagonal := Real.sqrt (2 * side ^ 2)
  diagonal = 7 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_diagonal_l2740_274082


namespace NUMINAMATH_CALUDE_grid_whitening_l2740_274053

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 9x9 grid of cells -/
def Grid := Fin 9 → Fin 9 → Color

/-- Represents a corner shape operation -/
structure CornerOperation where
  row : Fin 9
  col : Fin 9
  orientation : Fin 4

/-- Applies a corner operation to a grid -/
def applyOperation (g : Grid) (op : CornerOperation) : Grid :=
  sorry

/-- Checks if all cells in the grid are white -/
def allWhite (g : Grid) : Prop :=
  ∀ (i j : Fin 9), g i j = Color.White

/-- Main theorem: Any grid can be made all white with finite operations -/
theorem grid_whitening (g : Grid) :
  ∃ (ops : List CornerOperation), allWhite (ops.foldl applyOperation g) :=
  sorry

end NUMINAMATH_CALUDE_grid_whitening_l2740_274053


namespace NUMINAMATH_CALUDE_no_roots_in_interval_l2740_274054

-- Define the function f(x) = x^3 + x^2 - 2x - 1
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 1

-- State the theorem
theorem no_roots_in_interval :
  (Continuous f) →
  (f 0 < 0) →
  (f 1 < 0) →
  ∀ x ∈ Set.Ioo 0 1, f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_roots_in_interval_l2740_274054


namespace NUMINAMATH_CALUDE_weekly_earnings_theorem_l2740_274013

/-- Represents the shop's T-shirt sales and operating conditions -/
structure ShopConditions where
  women_tshirt_interval : ℕ := 30 -- minutes between women's T-shirt sales
  women_tshirt_price : ℕ := 18 -- price of women's T-shirt
  men_tshirt_interval : ℕ := 40 -- minutes between men's T-shirt sales
  men_tshirt_price : ℕ := 15 -- price of men's T-shirt
  daily_operating_minutes : ℕ := 720 -- minutes of operation per day (12 hours)
  days_per_week : ℕ := 7 -- number of operating days per week

/-- Calculates the weekly earnings from T-shirt sales given the shop conditions -/
def calculate_weekly_earnings (conditions : ShopConditions) : ℕ :=
  let women_daily_sales := conditions.daily_operating_minutes / conditions.women_tshirt_interval
  let men_daily_sales := conditions.daily_operating_minutes / conditions.men_tshirt_interval
  let daily_earnings := women_daily_sales * conditions.women_tshirt_price +
                        men_daily_sales * conditions.men_tshirt_price
  daily_earnings * conditions.days_per_week

/-- Theorem stating that the weekly earnings from T-shirt sales is $4914 -/
theorem weekly_earnings_theorem (shop : ShopConditions) :
  calculate_weekly_earnings shop = 4914 := by
  sorry


end NUMINAMATH_CALUDE_weekly_earnings_theorem_l2740_274013


namespace NUMINAMATH_CALUDE_polynomial_real_root_exists_l2740_274068

theorem polynomial_real_root_exists (b : ℝ) : ∃ x : ℝ, x^3 + b*x^2 - 4*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_exists_l2740_274068


namespace NUMINAMATH_CALUDE_irrational_sum_product_theorem_l2740_274089

theorem irrational_sum_product_theorem (a : ℝ) (h : Irrational a) :
  ∃ (b b' : ℝ), Irrational b ∧ Irrational b' ∧
    (¬ Irrational (a + b)) ∧
    (¬ Irrational (a * b')) ∧
    (Irrational (a * b)) ∧
    (Irrational (a + b')) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sum_product_theorem_l2740_274089


namespace NUMINAMATH_CALUDE_tan_product_values_l2740_274015

theorem tan_product_values (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 1 / 2 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_values_l2740_274015
