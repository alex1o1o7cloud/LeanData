import Mathlib

namespace NUMINAMATH_GPT_sum_of_prism_features_l1968_196837

theorem sum_of_prism_features : (12 + 8 + 6 = 26) := by
  sorry

end NUMINAMATH_GPT_sum_of_prism_features_l1968_196837


namespace NUMINAMATH_GPT_function_relationship_minimize_total_cost_l1968_196891

noncomputable def y (a x : ℕ) : ℕ :=
6400 * x + 50 * a + 100 * a^2 / (x - 1)

theorem function_relationship (a : ℕ) (hx : 2 ≤ x) : 
  y a x = 6400 * x + 50 * a + 100 * a^2 / (x - 1) :=
by sorry

theorem minimize_total_cost (a : ℕ) (hx : 2 ≤ x) (ha : a = 56) : 
  y a x ≥ 1650 * a + 6400 ∧ (x = 8) :=
by sorry

end NUMINAMATH_GPT_function_relationship_minimize_total_cost_l1968_196891


namespace NUMINAMATH_GPT_min_value_d_l1968_196842

theorem min_value_d (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (unique_solution : ∃! x y : ℤ, 2 * x + y = 2007 ∧ y = (abs (x - a) + abs (x - b) + abs (x - c) + abs (x - d))) :
  d = 504 :=
sorry

end NUMINAMATH_GPT_min_value_d_l1968_196842


namespace NUMINAMATH_GPT_sum_of_coefficients_factors_l1968_196889

theorem sum_of_coefficients_factors :
  ∃ (a b c d e : ℤ), 
    (343 * (x : ℤ)^3 + 125 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 51) :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_factors_l1968_196889


namespace NUMINAMATH_GPT_sum_of_possible_values_of_x_l1968_196861

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_x_l1968_196861


namespace NUMINAMATH_GPT_part1_part2_l1968_196815

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem part1 (x : ℝ) : f x ≥ 1 ↔ (x ≤ -5/2 ∨ x ≥ 3/2) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1968_196815


namespace NUMINAMATH_GPT_medal_award_count_l1968_196848

theorem medal_award_count :
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  no_canadians_get_medals + one_canadian_gets_medal = 480 :=
by
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  show no_canadians_get_medals + one_canadian_gets_medal = 480
  -- here should be the steps skipped
  sorry

end NUMINAMATH_GPT_medal_award_count_l1968_196848


namespace NUMINAMATH_GPT_base_8_addition_l1968_196845

-- Definitions
def five_base_8 : ℕ := 5
def thirteen_base_8 : ℕ := 1 * 8 + 3 -- equivalent of (13)_8 in base 10

-- Theorem statement
theorem base_8_addition :
  (five_base_8 + thirteen_base_8) = 2 * 8 + 0 :=
sorry

end NUMINAMATH_GPT_base_8_addition_l1968_196845


namespace NUMINAMATH_GPT_equalize_vertex_values_impossible_l1968_196875

theorem equalize_vertex_values_impossible 
  (n : ℕ) (h₁ : 2 ≤ n) 
  (vertex_values : Fin n → ℤ) 
  (h₂ : ∃! i : Fin n, vertex_values i = 1 ∧ ∀ j ≠ i, vertex_values j = 0) 
  (k : ℕ) (hk : k ∣ n) :
  ¬ (∃ c : ℤ, ∀ v : Fin n, vertex_values v = c) := 
sorry

end NUMINAMATH_GPT_equalize_vertex_values_impossible_l1968_196875


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l1968_196854

theorem sum_of_other_endpoint_coordinates
  (x₁ y₁ x₂ y₂ : ℝ)
  (hx : (x₁ + x₂) / 2 = 5)
  (hy : (y₁ + y₂) / 2 = -8)
  (endpt1 : x₁ = 7)
  (endpt2 : y₁ = -2) :
  x₂ + y₂ = -11 :=
sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l1968_196854


namespace NUMINAMATH_GPT_wine_problem_l1968_196810

theorem wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + (1 / 3) * y = 33) : x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by
  sorry

end NUMINAMATH_GPT_wine_problem_l1968_196810


namespace NUMINAMATH_GPT_minimum_words_to_learn_l1968_196892

-- Definition of the problem
def total_words : ℕ := 600
def required_percentage : ℕ := 90

-- Lean statement of the problem
theorem minimum_words_to_learn : ∃ x : ℕ, (x / total_words : ℚ) = required_percentage / 100 ∧ x = 540 :=
sorry

end NUMINAMATH_GPT_minimum_words_to_learn_l1968_196892


namespace NUMINAMATH_GPT_percentage_of_third_number_l1968_196833

theorem percentage_of_third_number (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A / C = 0.06 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_third_number_l1968_196833


namespace NUMINAMATH_GPT_max_average_growth_rate_l1968_196893

theorem max_average_growth_rate 
  (P1 P2 : ℝ) (M : ℝ)
  (h1 : P1 + P2 = M) : 
  (1 + (M / 2))^2 ≥ (1 + P1) * (1 + P2) := 
by
  -- AM-GM Inequality application and other mathematical steps go here.
  sorry

end NUMINAMATH_GPT_max_average_growth_rate_l1968_196893


namespace NUMINAMATH_GPT_village_population_l1968_196872

theorem village_population (initial_population: ℕ) (died_percent left_percent: ℕ) (remaining_population current_population: ℕ)
    (h1: initial_population = 6324)
    (h2: died_percent = 10)
    (h3: left_percent = 20)
    (h4: remaining_population = initial_population - (initial_population * died_percent / 100))
    (h5: current_population = remaining_population - (remaining_population * left_percent / 100)):
  current_population = 4554 :=
  by
    sorry

end NUMINAMATH_GPT_village_population_l1968_196872


namespace NUMINAMATH_GPT_charles_picked_50_pears_l1968_196831

variable (P B S : ℕ)

theorem charles_picked_50_pears 
  (cond1 : S = B + 10)
  (cond2 : B = 3 * P)
  (cond3 : S = 160) : 
  P = 50 := by
  sorry

end NUMINAMATH_GPT_charles_picked_50_pears_l1968_196831


namespace NUMINAMATH_GPT_expand_expression_l1968_196821

theorem expand_expression (x y : ℝ) : 
  (2 * x + 3) * (5 * y + 7) = 10 * x * y + 14 * x + 15 * y + 21 := 
by sorry

end NUMINAMATH_GPT_expand_expression_l1968_196821


namespace NUMINAMATH_GPT_no_valid_bases_l1968_196880

theorem no_valid_bases
  (x y : ℕ)
  (h1 : 4 * x + 9 = 4 * y + 1)
  (h2 : 4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9)
  (hx : x > 1)
  (hy : y > 1)
  : false :=
by
  sorry

end NUMINAMATH_GPT_no_valid_bases_l1968_196880


namespace NUMINAMATH_GPT_Zhu_Zaiyu_problem_l1968_196885

theorem Zhu_Zaiyu_problem
  (f : ℕ → ℝ) 
  (q : ℝ)
  (h_geom_seq : ∀ n, f (n+1) = q * f n)
  (h_octave : f 13 = 2 * f 1) :
  (f 7) / (f 3) = 2^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_Zhu_Zaiyu_problem_l1968_196885


namespace NUMINAMATH_GPT_total_weight_correct_l1968_196822

variable (c1 c2 w2 c : Float)

def total_weight (c1 c2 w2 c : Float) (W x : Float) :=
  (c1 * x + c2 * w2) / (x + w2) = c ∧ W = x + w2

theorem total_weight_correct :
  total_weight 9 8 12 8.40 20 8 :=
by sorry

end NUMINAMATH_GPT_total_weight_correct_l1968_196822


namespace NUMINAMATH_GPT_proof_problem_l1968_196811

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define the set M
def M : Set Nat := {2, 4}

-- Define the set N
def N : Set Nat := {0, 4}

-- Define the union of sets M and N
def M_union_N : Set Nat := M ∪ N

-- Define the complement of M ∪ N in U
def complement_U (s : Set Nat) : Set Nat := U \ s

-- State the theorem
theorem proof_problem : complement_U M_union_N = {1, 3} := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1968_196811


namespace NUMINAMATH_GPT_smallest_non_factor_product_of_100_l1968_196800

/-- Let a and b be distinct positive integers that are factors of 100. 
    The smallest value of their product which is not a factor of 100 is 8. -/
theorem smallest_non_factor_product_of_100 (a b : ℕ) (hab : a ≠ b) (ha : a ∣ 100) (hb : b ∣ 100) (hprod : ¬ (a * b ∣ 100)) : a * b = 8 :=
sorry

end NUMINAMATH_GPT_smallest_non_factor_product_of_100_l1968_196800


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l1968_196812

def point : ℝ × ℝ := (1, 0)

def given_line (x y : ℝ) : Prop := x - y + 2 = 0

def is_perpendicular_to (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y → l2 (y - x) (-x - y + 2)

def target_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem perpendicular_line_through_point (l1 : ℝ → ℝ → Prop) (p : ℝ × ℝ) :
  given_line = l1 ∧ p = point →
  (∃ l2 : ℝ → ℝ → Prop, is_perpendicular_to l1 l2 ∧ l2 p.1 p.2) →
  target_line p.1 p.2 :=
by
  intro hp hl2
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l1968_196812


namespace NUMINAMATH_GPT_problem_solution_l1968_196803

noncomputable def time_min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * (Real.cos α) / (2 * c * (1 - Real.sin α))

noncomputable def min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * Real.sqrt ((1 - (Real.sin α)) / 2)

theorem problem_solution (α : ℝ) (c : ℝ) (a : ℝ) 
  (α_30 : α = Real.pi / 6) (c_50 : c = 50) (a_50sqrt3 : a = 50 * Real.sqrt 3) :
  (time_min_distance c α a = 1.5) ∧ (min_distance c α a = 25 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1968_196803


namespace NUMINAMATH_GPT_fraction_equivalence_l1968_196823

variable {m n p q : ℚ}

theorem fraction_equivalence
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_equivalence_l1968_196823


namespace NUMINAMATH_GPT_bah_to_yah_conversion_l1968_196829

theorem bah_to_yah_conversion :
  (10 : ℝ) * (1500 * (3/5) * (10/16)) / 16 = 562.5 := by
sorry

end NUMINAMATH_GPT_bah_to_yah_conversion_l1968_196829


namespace NUMINAMATH_GPT_production_rate_l1968_196819

theorem production_rate (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (H : x * x * 2 * x = 2 * x^3) :
  y * y * 3 * y = 3 * y^3 := by
  sorry

end NUMINAMATH_GPT_production_rate_l1968_196819


namespace NUMINAMATH_GPT_defective_probability_l1968_196820

theorem defective_probability {total_switches checked_switches defective_checked : ℕ}
  (h1 : total_switches = 2000)
  (h2 : checked_switches = 100)
  (h3 : defective_checked = 10) :
  (defective_checked : ℚ) / checked_switches = 1 / 10 :=
sorry

end NUMINAMATH_GPT_defective_probability_l1968_196820


namespace NUMINAMATH_GPT_proportionality_intersect_calculation_l1968_196886

variables {x1 x2 y1 y2 : ℝ}

/-- Proof that (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15,
    given specific conditions on x1, x2, y1, and y2. -/
theorem proportionality_intersect_calculation
  (h1 : y1 = 5 / x1) 
  (h2 : y2 = 5 / x2)
  (h3 : x1 * y1 = 5)
  (h4 : x2 * y2 = 5)
  (h5 : x1 = -x2)
  (h6 : y1 = -y2) :
  (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15 := 
sorry

end NUMINAMATH_GPT_proportionality_intersect_calculation_l1968_196886


namespace NUMINAMATH_GPT_area_of_rectangle_l1968_196888

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1968_196888


namespace NUMINAMATH_GPT_sequence_k_value_l1968_196855

theorem sequence_k_value {k : ℕ} (h : 9 < (2 * k - 8) ∧ (2 * k - 8) < 12) 
  (Sn : ℕ → ℤ) (hSn : ∀ n, Sn n = n^2 - 7*n) 
  : k = 9 :=
by
  sorry

end NUMINAMATH_GPT_sequence_k_value_l1968_196855


namespace NUMINAMATH_GPT_expression_value_l1968_196840

theorem expression_value (a b : ℚ) (h₁ : a = -1/2) (h₂ : b = 3/2) : -a - 2 * b^2 + 3 * a * b = -25/4 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1968_196840


namespace NUMINAMATH_GPT_rewrite_subtraction_rewrite_division_l1968_196843

theorem rewrite_subtraction : -8 - 5 = -8 + (-5) :=
by sorry

theorem rewrite_division : (1/2) / (-2) = (1/2) * (-1/2) :=
by sorry

end NUMINAMATH_GPT_rewrite_subtraction_rewrite_division_l1968_196843


namespace NUMINAMATH_GPT_difference_of_two_smallest_integers_divisors_l1968_196825

theorem difference_of_two_smallest_integers_divisors (n m : ℕ) (h₁ : n > 1) (h₂ : m > 1) 
(h₃ : n % 2 = 1) (h₄ : n % 3 = 1) (h₅ : n % 4 = 1) (h₆ : n % 5 = 1) 
(h₇ : n % 6 = 1) (h₈ : n % 7 = 1) (h₉ : n % 8 = 1) (h₁₀ : n % 9 = 1) 
(h₁₁ : n % 10 = 1) (h₃' : m % 2 = 1) (h₄' : m % 3 = 1) (h₅' : m % 4 = 1) 
(h₆' : m % 5 = 1) (h₇' : m % 6 = 1) (h₈' : m % 7 = 1) (h₉' : m % 8 = 1) 
(h₁₀' : m % 9 = 1) (h₁₁' : m % 10 = 1): m - n = 2520 :=
sorry

end NUMINAMATH_GPT_difference_of_two_smallest_integers_divisors_l1968_196825


namespace NUMINAMATH_GPT_bushes_for_60_zucchinis_l1968_196882

/-- 
Given:
1. Each blueberry bush yields twelve containers of blueberries.
2. Four containers of blueberries can be traded for three pumpkins.
3. Six pumpkins can be traded for five zucchinis.

Prove that eight bushes are needed to harvest 60 zucchinis.
-/
theorem bushes_for_60_zucchinis (bush_to_containers : ℕ) (containers_to_pumpkins : ℕ) (pumpkins_to_zucchinis : ℕ) :
  (bush_to_containers = 12) → (containers_to_pumpkins = 4) → (pumpkins_to_zucchinis = 6) →
  ∃ bushes_needed, bushes_needed = 8 ∧ (60 * pumpkins_to_zucchinis / 5 * containers_to_pumpkins / 3 / bush_to_containers) = bushes_needed :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_bushes_for_60_zucchinis_l1968_196882


namespace NUMINAMATH_GPT_number_of_adults_l1968_196818

theorem number_of_adults (total_apples : ℕ) (children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) (h : total_apples = 450) (h1 : children = 33) (h2 : apples_per_child = 10) (h3 : apples_per_adult = 3) :
  total_apples - (children * apples_per_child) = 120 →
  (total_apples - (children * apples_per_child)) / apples_per_adult = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_adults_l1968_196818


namespace NUMINAMATH_GPT_original_price_l1968_196806

theorem original_price (P : ℝ) (h : 0.75 * (0.75 * P) = 17) : P = 30.22 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1968_196806


namespace NUMINAMATH_GPT_marbles_difference_l1968_196835

theorem marbles_difference {red_marbles blue_marbles : ℕ} 
  (h₁ : red_marbles = 288) (bags_red : ℕ) (h₂ : bags_red = 12) 
  (h₃ : blue_marbles = 243) (bags_blue : ℕ) (h₄ : bags_blue = 9) :
  (blue_marbles / bags_blue) - (red_marbles / bags_red) = 3 :=
by
  sorry

end NUMINAMATH_GPT_marbles_difference_l1968_196835


namespace NUMINAMATH_GPT_reach_any_composite_from_4_l1968_196879

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

def can_reach (A : ℕ) : Prop :=
  ∀ n : ℕ, is_composite n → ∃ seq : ℕ → ℕ, seq 0 = A ∧ seq (n + 1) - seq n ∣ seq n ∧ seq (n + 1) ≠ seq n ∧ seq (n + 1) ≠ 1 ∧ seq (n + 1) = n

theorem reach_any_composite_from_4 : can_reach 4 :=
  sorry

end NUMINAMATH_GPT_reach_any_composite_from_4_l1968_196879


namespace NUMINAMATH_GPT_m_ge_1_l1968_196895

open Set

theorem m_ge_1 (m : ℝ) :
  (∀ x, x ∈ {x | x ≤ 1} ∩ {x | ¬ (x ≤ m)} → False) → m ≥ 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_m_ge_1_l1968_196895


namespace NUMINAMATH_GPT_temperature_difference_l1968_196863

variable (high_temp : ℝ) (low_temp : ℝ)

theorem temperature_difference (h1 : high_temp = 15) (h2 : low_temp = 7) : high_temp - low_temp = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_temperature_difference_l1968_196863


namespace NUMINAMATH_GPT_count_valid_rods_l1968_196887

def isValidRodLength (d : ℕ) : Prop :=
  5 ≤ d ∧ d < 27

def countValidRodLengths (lower upper : ℕ) : ℕ :=
  upper - lower + 1

theorem count_valid_rods :
  let valid_rods_count := countValidRodLengths 5 26
  valid_rods_count = 22 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_rods_l1968_196887


namespace NUMINAMATH_GPT_max_intersections_circle_pentagon_l1968_196807

theorem max_intersections_circle_pentagon : 
  ∃ (circle : Set Point) (pentagon : List (Set Point)),
    (∀ (side : Set Point), side ∈ pentagon → ∃ p1 p2 : Point, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2) ∧
    pentagon.length = 5 →
    (∃ n : ℕ, n = 10) :=
by
  sorry

end NUMINAMATH_GPT_max_intersections_circle_pentagon_l1968_196807


namespace NUMINAMATH_GPT_part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l1968_196824

noncomputable def a (n : ℕ) : ℚ := 1 / (n : ℚ)

noncomputable def S (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ k => a (k + 1))

noncomputable def f (n : ℕ) : ℚ :=
  if n = 1 then S 2
  else S (2 * n) - S (n - 1)

theorem part1_f1 : f 1 = 3 / 2 := by sorry

theorem part1_f2 : f 2 = 13 / 12 := by sorry

theorem part1_f3 : f 3 = 19 / 20 := by sorry

theorem part2_f_gt_1_for_n_1_2 (n : ℕ) (h₁ : n = 1 ∨ n = 2) : f n > 1 := by sorry

theorem part2_f_lt_1_for_n_ge_3 (n : ℕ) (h₁ : n ≥ 3) : f n < 1 := by sorry

end NUMINAMATH_GPT_part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l1968_196824


namespace NUMINAMATH_GPT_min_tablets_to_extract_l1968_196805

noncomputable def min_tablets_needed : ℕ :=
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  worst_case + required_A -- 14 + 18 + 20 + 3 = 55

theorem min_tablets_to_extract : min_tablets_needed = 55 :=
by {
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  have h : worst_case + required_A = 55 := by decide
  exact h
}

end NUMINAMATH_GPT_min_tablets_to_extract_l1968_196805


namespace NUMINAMATH_GPT_car_speed_l1968_196870

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end NUMINAMATH_GPT_car_speed_l1968_196870


namespace NUMINAMATH_GPT_journey_distance_l1968_196834

theorem journey_distance :
  ∃ D T : ℝ,
    D = 100 * T ∧
    D = 80 * (T + 1/3) ∧
    D = 400 / 3 :=
by
  sorry

end NUMINAMATH_GPT_journey_distance_l1968_196834


namespace NUMINAMATH_GPT_Cindy_coins_l1968_196847

theorem Cindy_coins (n : ℕ) (h1 : ∃ X Y : ℕ, n = X * Y ∧ Y > 1 ∧ Y < n) (h2 : ∀ Y, Y > 1 ∧ Y < n → ¬Y ∣ n → False) : n = 65536 :=
by
  sorry

end NUMINAMATH_GPT_Cindy_coins_l1968_196847


namespace NUMINAMATH_GPT_each_person_tip_l1968_196878

-- Definitions based on the conditions
def julie_cost : ℝ := 10
def letitia_cost : ℝ := 20
def anton_cost : ℝ := 30
def tip_rate : ℝ := 0.2

-- Theorem statement
theorem each_person_tip (total_cost := julie_cost + letitia_cost + anton_cost)
 (total_tip := total_cost * tip_rate) :
 (total_tip / 3) = 4 := by
  sorry

end NUMINAMATH_GPT_each_person_tip_l1968_196878


namespace NUMINAMATH_GPT_product_implication_l1968_196871

theorem product_implication (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a * b > 1) : a > 1 ∨ b > 1 :=
sorry

end NUMINAMATH_GPT_product_implication_l1968_196871


namespace NUMINAMATH_GPT_neces_not_suff_cond_l1968_196873

theorem neces_not_suff_cond (a : ℝ) (h : a ≠ 0) : (1 / a < 1) → (a > 1) :=
sorry

end NUMINAMATH_GPT_neces_not_suff_cond_l1968_196873


namespace NUMINAMATH_GPT_largest_digit_B_divisible_by_4_l1968_196850

theorem largest_digit_B_divisible_by_4 :
  ∃ (B : ℕ), B ≤ 9 ∧ ∀ B', (B' ≤ 9 ∧ (4 * 10^5 + B' * 10^4 + 5 * 10^3 + 7 * 10^2 + 8 * 10 + 4) % 4 = 0) → B' ≤ B :=
by
  sorry

end NUMINAMATH_GPT_largest_digit_B_divisible_by_4_l1968_196850


namespace NUMINAMATH_GPT_probability_three_specific_cards_l1968_196846

noncomputable def deck_size : ℕ := 52
noncomputable def num_suits : ℕ := 4
noncomputable def cards_per_suit : ℕ := 13
noncomputable def p_king_spades : ℚ := 1 / deck_size
noncomputable def p_10_hearts : ℚ := 1 / (deck_size - 1)
noncomputable def p_queen : ℚ := 4 / (deck_size - 2)

theorem probability_three_specific_cards :
  (p_king_spades * p_10_hearts * p_queen) = 1 / 33150 := 
sorry

end NUMINAMATH_GPT_probability_three_specific_cards_l1968_196846


namespace NUMINAMATH_GPT_genevieve_initial_amount_l1968_196844

def cost_per_kg : ℕ := 8
def kg_bought : ℕ := 250
def short_amount : ℕ := 400
def total_cost : ℕ := kg_bought * cost_per_kg
def initial_amount := total_cost - short_amount

theorem genevieve_initial_amount : initial_amount = 1600 := by
  unfold initial_amount total_cost cost_per_kg kg_bought short_amount
  sorry

end NUMINAMATH_GPT_genevieve_initial_amount_l1968_196844


namespace NUMINAMATH_GPT_school_days_per_week_l1968_196868

-- Definitions based on the conditions given
def paper_per_class_per_day : ℕ := 200
def total_paper_per_week : ℕ := 9000
def number_of_classes : ℕ := 9

-- The theorem stating the main claim to prove
theorem school_days_per_week :
  total_paper_per_week / (paper_per_class_per_day * number_of_classes) = 5 :=
  by
  sorry

end NUMINAMATH_GPT_school_days_per_week_l1968_196868


namespace NUMINAMATH_GPT_remainder_9_pow_2023_div_50_l1968_196899

theorem remainder_9_pow_2023_div_50 : (9 ^ 2023) % 50 = 41 := by
  sorry

end NUMINAMATH_GPT_remainder_9_pow_2023_div_50_l1968_196899


namespace NUMINAMATH_GPT_right_triangle_legs_l1968_196896

theorem right_triangle_legs (a b : ℝ) (r R : ℝ) (hypotenuse : ℝ) (h_ab : a + b = 14) (h_c : hypotenuse = 10)
  (h_leg: a * b = a + b + 10) (h_Pythag : a^2 + b^2 = hypotenuse^2) 
  (h_inradius : r = 2) (h_circumradius : R = 5) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_l1968_196896


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1968_196808

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (x^2 - 2 * x < 0) ↔ (0 < x ∧ x < 2) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1968_196808


namespace NUMINAMATH_GPT_negation_of_universal_prop_l1968_196851

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l1968_196851


namespace NUMINAMATH_GPT_value_of_y_l1968_196814

theorem value_of_y (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : y = 9 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_y_l1968_196814


namespace NUMINAMATH_GPT_xy_equals_nine_l1968_196860

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end NUMINAMATH_GPT_xy_equals_nine_l1968_196860


namespace NUMINAMATH_GPT_product_of_possible_b_values_l1968_196884

theorem product_of_possible_b_values (b : ℝ) :
  (∀ (y1 y2 x1 x2 : ℝ), y1 = -1 ∧ y2 = 3 ∧ x1 = 2 ∧ (x2 = b) ∧ (y2 - y1 = 4) → 
   (b = 2 + 4 ∨ b = 2 - 4)) → 
  (b = 6 ∨ b = -2) → (b = 6) ∧ (b = -2) → 6 * -2 = -12 :=
sorry

end NUMINAMATH_GPT_product_of_possible_b_values_l1968_196884


namespace NUMINAMATH_GPT_ball_radius_l1968_196839

theorem ball_radius (x r : ℝ) (h1 : x^2 + 256 = r^2) (h2 : r = x + 16) : r = 16 :=
by
  sorry

end NUMINAMATH_GPT_ball_radius_l1968_196839


namespace NUMINAMATH_GPT_true_proposition_l1968_196827

def p : Prop := ∃ x₀ : ℝ, x₀^2 < x₀
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem true_proposition : p ∧ q :=
by 
  sorry

end NUMINAMATH_GPT_true_proposition_l1968_196827


namespace NUMINAMATH_GPT_fuelA_amount_l1968_196853

def tankCapacity : ℝ := 200
def ethanolInFuelA : ℝ := 0.12
def ethanolInFuelB : ℝ := 0.16
def totalEthanol : ℝ := 30
def limitedFuelA : ℝ := 100
def limitedFuelB : ℝ := 150

theorem fuelA_amount : ∃ (x : ℝ), 
  (x ≤ limitedFuelA ∧ x ≥ 0) ∧ 
  ((tankCapacity - x) ≤ limitedFuelB ∧ (tankCapacity - x) ≥ 0) ∧ 
  (ethanolInFuelA * x + ethanolInFuelB * (tankCapacity - x)) = totalEthanol ∧ 
  x = 50 := 
by
  sorry

end NUMINAMATH_GPT_fuelA_amount_l1968_196853


namespace NUMINAMATH_GPT_second_solution_volume_l1968_196809

theorem second_solution_volume
  (V : ℝ)
  (h1 : 0.20 * 6 + 0.60 * V = 0.36 * (6 + V)) : 
  V = 4 :=
sorry

end NUMINAMATH_GPT_second_solution_volume_l1968_196809


namespace NUMINAMATH_GPT_power_addition_l1968_196826

theorem power_addition :
  (-2 : ℤ) ^ 2009 + (-2 : ℤ) ^ 2010 = 2 ^ 2009 :=
by
  sorry

end NUMINAMATH_GPT_power_addition_l1968_196826


namespace NUMINAMATH_GPT_number_of_paths_l1968_196801

-- Define the conditions of the problem
def grid_width : ℕ := 7
def grid_height : ℕ := 6
def diagonal_steps : ℕ := 2

-- Define the main proof statement
theorem number_of_paths (width height diag : ℕ) 
  (Nhyp : width = grid_width ∧ height = grid_height ∧ diag = diagonal_steps) : 
  ∃ (paths : ℕ), paths = 6930 := 
sorry

end NUMINAMATH_GPT_number_of_paths_l1968_196801


namespace NUMINAMATH_GPT_find_m_and_y_range_l1968_196836

open Set

noncomputable def y (m x : ℝ) := (6 + 2 * m) * x^2 - 5 * x^((abs (m + 2))) + 3 

theorem find_m_and_y_range :
  (∃ m : ℝ, (∀ x : ℝ, y m x = (6 + 2*m) * x^2 - 5*x^((abs (m+2))) + 3) ∧ (∀ x : ℝ, y m x = -5 * x + 3 → m = -3)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 → y (-3) x ∈ Icc (-22 : ℝ) (8 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_find_m_and_y_range_l1968_196836


namespace NUMINAMATH_GPT_correct_statement_l1968_196856

theorem correct_statement : -3 > -5 := 
by {
  sorry
}

end NUMINAMATH_GPT_correct_statement_l1968_196856


namespace NUMINAMATH_GPT_sum_of_coordinates_of_D_l1968_196802

-- Definition of points M, C and D
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨4, 7⟩
def C : Point := ⟨6, 2⟩

-- Conditions that M is the midpoint of segment CD
def isMidpoint (M C D : Point) : Prop :=
  ((C.x + D.x) / 2 = M.x) ∧
  ((C.y + D.y) / 2 = M.y)

-- Definition for the sum of the coordinates of a point
def sumOfCoordinates (P : Point) : ℝ :=
  P.x + P.y

-- The main theorem stating the sum of the coordinates of D is 14 given the conditions
theorem sum_of_coordinates_of_D :
  ∃ D : Point, isMidpoint M C D ∧ sumOfCoordinates D = 14 := 
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_D_l1968_196802


namespace NUMINAMATH_GPT_parametric_plane_equiv_l1968_196838

/-- Define the parametric form of the plane -/
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + s - t, 2 - s, 3 - 2*s + 2*t)

/-- Define the equation of the plane in standard form -/
def plane_equation (x y z : ℝ) : Prop :=
  2 * x + z - 5 = 0

/-- The theorem stating that the parametric form corresponds to the given plane equation -/
theorem parametric_plane_equiv :
  ∃ x y z s t,
    (x, y, z) = parametric_plane s t ∧ plane_equation x y z :=
by
  sorry

end NUMINAMATH_GPT_parametric_plane_equiv_l1968_196838


namespace NUMINAMATH_GPT_truthful_dwarfs_count_l1968_196841

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end NUMINAMATH_GPT_truthful_dwarfs_count_l1968_196841


namespace NUMINAMATH_GPT_problem_l1968_196874

variable (a b c d : ℕ)

theorem problem (h1 : a + b = 12) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 6 :=
sorry

end NUMINAMATH_GPT_problem_l1968_196874


namespace NUMINAMATH_GPT_break_even_point_l1968_196813

/-- Conditions of the problem -/
def fixed_costs : ℝ := 10410
def variable_cost_per_unit : ℝ := 2.65
def selling_price_per_unit : ℝ := 20

/-- The mathematically equivalent proof problem / statement -/
theorem break_even_point :
  fixed_costs / (selling_price_per_unit - variable_cost_per_unit) = 600 := 
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_break_even_point_l1968_196813


namespace NUMINAMATH_GPT_part1_part2_part3_l1968_196816

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1968_196816


namespace NUMINAMATH_GPT_range_of_g_l1968_196828

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : Set.Icc (-1.1071) 1.1071 = Set.image g (Set.Icc (-1:ℝ) 1) := by
  sorry

end NUMINAMATH_GPT_range_of_g_l1968_196828


namespace NUMINAMATH_GPT_sum_f_1_to_10_l1968_196883

-- Define the function f with the properties given.

def f (x : ℝ) : ℝ := sorry

-- Specify the conditions of the problem
local notation "R" => ℝ

axiom odd_function : ∀ (x : R), f (-x) = -f (x)
axiom periodicity : ∀ (x : R), f (x + 3) = f (x)
axiom f_neg1 : f (-1) = 1

-- State the theorem to be proved
theorem sum_f_1_to_10 : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry
end NUMINAMATH_GPT_sum_f_1_to_10_l1968_196883


namespace NUMINAMATH_GPT_cubic_root_sum_l1968_196898

-- Assume we have three roots a, b, and c of the polynomial x^3 - 3x - 2 = 0
variables {a b c : ℝ}

-- Using Vieta's formulas for the polynomial x^3 - 3x - 2 = 0
axiom Vieta1 : a + b + c = 0
axiom Vieta2 : a * b + a * c + b * c = -3
axiom Vieta3 : a * b * c = -2

-- The proof that the given expression evaluates to 9
theorem cubic_root_sum:
  a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_cubic_root_sum_l1968_196898


namespace NUMINAMATH_GPT_Haley_initial_trees_l1968_196864

theorem Haley_initial_trees (T : ℕ) (h1 : T - 4 ≥ 0) (h2 : (T - 4) + 5 = 10): T = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Haley_initial_trees_l1968_196864


namespace NUMINAMATH_GPT_quadratic_solutions_l1968_196859

theorem quadratic_solutions (x : ℝ) :
  (4 * x^2 - 6 * x = 0) ↔ (x = 0) ∨ (x = 3 / 2) :=
sorry

end NUMINAMATH_GPT_quadratic_solutions_l1968_196859


namespace NUMINAMATH_GPT_amount_paid_l1968_196876

def cost_cat_toy : ℝ := 8.77
def cost_cage : ℝ := 10.97
def change_received : ℝ := 0.26

theorem amount_paid : (cost_cat_toy + cost_cage + change_received) = 20.00 := by
  sorry

end NUMINAMATH_GPT_amount_paid_l1968_196876


namespace NUMINAMATH_GPT_Mike_books_l1968_196862

theorem Mike_books
  (initial_books : ℝ)
  (books_sold : ℝ)
  (books_gifts : ℝ) 
  (books_bought : ℝ)
  (h_initial : initial_books = 51.5)
  (h_sold : books_sold = 45.75)
  (h_gifts : books_gifts = 12.25)
  (h_bought : books_bought = 3.5):
  initial_books - books_sold + books_gifts + books_bought = 21.5 := 
sorry

end NUMINAMATH_GPT_Mike_books_l1968_196862


namespace NUMINAMATH_GPT_segment_length_l1968_196817

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem segment_length (x : ℝ) 
  (h : |x - cbrt 27| = 5) : (abs ((cbrt 27 + 5) - (cbrt 27 - 5)) = 10) :=
by
  sorry

end NUMINAMATH_GPT_segment_length_l1968_196817


namespace NUMINAMATH_GPT_sum_ab_system_1_l1968_196877

theorem sum_ab_system_1 {a b : ℝ} 
  (h1 : a^3 - a^2 + a - 5 = 0) 
  (h2 : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sum_ab_system_1_l1968_196877


namespace NUMINAMATH_GPT_point_outside_circle_l1968_196849

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) : a^2 + b^2 > 1 :=
by
  sorry

end NUMINAMATH_GPT_point_outside_circle_l1968_196849


namespace NUMINAMATH_GPT_martin_ratio_of_fruits_eaten_l1968_196857

theorem martin_ratio_of_fruits_eaten
    (initial_fruits : ℕ)
    (current_oranges : ℕ)
    (current_oranges_twice_limes : current_oranges = 2 * (current_oranges / 2))
    (initial_fruits_count : initial_fruits = 150)
    (current_oranges_count : current_oranges = 50) :
    (initial_fruits - (current_oranges + (current_oranges / 2))) / initial_fruits = 1 / 2 := 
by
    sorry

end NUMINAMATH_GPT_martin_ratio_of_fruits_eaten_l1968_196857


namespace NUMINAMATH_GPT_somu_current_age_l1968_196869

variable (S F : ℕ)

theorem somu_current_age
  (h1 : S = F / 3)
  (h2 : S - 10 = (F - 10) / 5) :
  S = 20 := by
  sorry

end NUMINAMATH_GPT_somu_current_age_l1968_196869


namespace NUMINAMATH_GPT_correct_operation_l1968_196804

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1968_196804


namespace NUMINAMATH_GPT_calendar_sum_l1968_196897

theorem calendar_sum (n : ℕ) : 
    n + (n + 7) + (n + 14) = 3 * n + 21 :=
by sorry

end NUMINAMATH_GPT_calendar_sum_l1968_196897


namespace NUMINAMATH_GPT_pamela_spilled_sugar_l1968_196832

theorem pamela_spilled_sugar 
  (original_amount : ℝ)
  (amount_left : ℝ)
  (h1 : original_amount = 9.8)
  (h2 : amount_left = 4.6)
  : original_amount - amount_left = 5.2 :=
by 
  sorry

end NUMINAMATH_GPT_pamela_spilled_sugar_l1968_196832


namespace NUMINAMATH_GPT_smallest_x_value_l1968_196852

theorem smallest_x_value : ∃ x : ℤ, ∃ y : ℤ, (xy + 7 * x + 6 * y = -8) ∧ x = -40 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_value_l1968_196852


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1968_196894

-- Definitions
def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

-- Problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ)
    (h_geom : is_geom_seq a q)
    (h1 : a 3 * a 7 = 8)
    (h2 : a 4 + a 6 = 6) :
    a 2 + a 8 = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1968_196894


namespace NUMINAMATH_GPT_k_gt_4_l1968_196881

theorem k_gt_4 {x y k : ℝ} (h1 : 2 * x + y = 2 * k - 1) (h2 : x + 2 * y = -4) (h3 : x + y > 1) : k > 4 :=
by
  -- This 'sorry' serves as a placeholder for the actual proof steps
  sorry

end NUMINAMATH_GPT_k_gt_4_l1968_196881


namespace NUMINAMATH_GPT_vinny_final_weight_l1968_196867

theorem vinny_final_weight :
  let initial_weight := 300
  let first_month_loss := 20
  let second_month_loss := first_month_loss / 2
  let third_month_loss := second_month_loss / 2
  let fourth_month_loss := third_month_loss / 2
  let fifth_month_loss := 12
  let total_loss := first_month_loss + second_month_loss + third_month_loss + fourth_month_loss + fifth_month_loss
  let final_weight := initial_weight - total_loss
  final_weight = 250.5 :=
by
  sorry

end NUMINAMATH_GPT_vinny_final_weight_l1968_196867


namespace NUMINAMATH_GPT_right_triangle_ratio_l1968_196858

theorem right_triangle_ratio (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : a^2 + b^2 = c^2) (r s : ℝ) (h3 : r = a^2 / c) (h4 : s = b^2 / c) : 
  r / s = 9 / 16 := by
 sorry

end NUMINAMATH_GPT_right_triangle_ratio_l1968_196858


namespace NUMINAMATH_GPT_compute_div_square_of_negatives_l1968_196830

theorem compute_div_square_of_negatives : (-128)^2 / (-64)^2 = 4 := by
  sorry

end NUMINAMATH_GPT_compute_div_square_of_negatives_l1968_196830


namespace NUMINAMATH_GPT_manager_hourly_wage_l1968_196866

open Real

theorem manager_hourly_wage (M D C : ℝ) 
  (hD : D = M / 2)
  (hC : C = 1.20 * D)
  (hC_manager : C = M - 3.40) :
  M = 8.50 :=
by
  sorry

end NUMINAMATH_GPT_manager_hourly_wage_l1968_196866


namespace NUMINAMATH_GPT_gcd_lcm_sum_l1968_196865

-- Define the numbers and their prime factorizations
def a := 120
def b := 4620
def a_prime_factors := (2, 3) -- 2^3
def b_prime_factors := (2, 2) -- 2^2

-- Define gcd and lcm based on the problem statement
def gcd_ab := 60
def lcm_ab := 4620

-- The statement to be proved
theorem gcd_lcm_sum : gcd a b + lcm a b = 4680 :=
by sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l1968_196865


namespace NUMINAMATH_GPT_maya_total_pages_l1968_196890

def books_first_week : ℕ := 5
def pages_per_book_first_week : ℕ := 300
def books_second_week := books_first_week * 2
def pages_per_book_second_week : ℕ := 350
def books_third_week := books_first_week * 3
def pages_per_book_third_week : ℕ := 400

def total_pages_first_week : ℕ := books_first_week * pages_per_book_first_week
def total_pages_second_week : ℕ := books_second_week * pages_per_book_second_week
def total_pages_third_week : ℕ := books_third_week * pages_per_book_third_week

def total_pages_maya_read : ℕ := total_pages_first_week + total_pages_second_week + total_pages_third_week

theorem maya_total_pages : total_pages_maya_read = 11000 := by
  sorry

end NUMINAMATH_GPT_maya_total_pages_l1968_196890
