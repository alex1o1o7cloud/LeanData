import Mathlib

namespace NUMINAMATH_GPT_intersection_eq_expected_result_l1140_114087

def M := { x : ℝ | x - 2 > 0 }
def N := { x : ℝ | (x - 3) * (x - 1) < 0 }
def expected_result := { x : ℝ | 2 < x ∧ x < 3 }

theorem intersection_eq_expected_result : M ∩ N = expected_result := 
by
  sorry

end NUMINAMATH_GPT_intersection_eq_expected_result_l1140_114087


namespace NUMINAMATH_GPT_weight_of_new_student_l1140_114042

theorem weight_of_new_student (W : ℝ) (x : ℝ) (h1 : 5 * W - 92 + x = 5 * (W - 4)) : x = 72 :=
sorry

end NUMINAMATH_GPT_weight_of_new_student_l1140_114042


namespace NUMINAMATH_GPT_space_taken_by_files_l1140_114017

-- Definitions/Conditions
def total_space : ℕ := 28
def space_left : ℕ := 2

-- Statement of the theorem
theorem space_taken_by_files : total_space - space_left = 26 := by sorry

end NUMINAMATH_GPT_space_taken_by_files_l1140_114017


namespace NUMINAMATH_GPT_initial_number_is_12_l1140_114020

theorem initial_number_is_12 {x : ℤ} (h : ∃ k : ℤ, x + 17 = 29 * k) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_is_12_l1140_114020


namespace NUMINAMATH_GPT_find_h_at_2_l1140_114078

noncomputable def h (x : ℝ) : ℝ := x^4 + 2 * x^3 - 12 * x^2 - 14 * x + 24

lemma poly_value_at_minus_2 : h (-2) = -4 := by
  sorry

lemma poly_value_at_1 : h 1 = -1 := by
  sorry

lemma poly_value_at_minus_4 : h (-4) = -16 := by
  sorry

lemma poly_value_at_3 : h 3 = -9 := by
  sorry

theorem find_h_at_2 : h 2 = -20 := by
  sorry

end NUMINAMATH_GPT_find_h_at_2_l1140_114078


namespace NUMINAMATH_GPT_last_two_digits_of_100_factorial_l1140_114004

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_nonzero_digits (n : ℕ) : ℕ := sorry

theorem last_two_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 24 :=
sorry

end NUMINAMATH_GPT_last_two_digits_of_100_factorial_l1140_114004


namespace NUMINAMATH_GPT_locus_of_centers_l1140_114058

-- Statement of the problem
theorem locus_of_centers :
  ∀ (a b : ℝ),
    ((∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (3 - r)^2))) ↔ (4 * a^2 + 4 * b^2 - 25 = 0) := by
  sorry

end NUMINAMATH_GPT_locus_of_centers_l1140_114058


namespace NUMINAMATH_GPT_sum_xyz_eq_neg7_l1140_114012

theorem sum_xyz_eq_neg7 (x y z : ℝ)
  (h1 : x = y + z + 2)
  (h2 : y = z + x + 1)
  (h3 : z = x + y + 4) :
  x + y + z = -7 :=
by
  sorry

end NUMINAMATH_GPT_sum_xyz_eq_neg7_l1140_114012


namespace NUMINAMATH_GPT_problem_l1140_114049

variables {S T : ℕ → ℕ} {a b : ℕ → ℕ}

-- Conditions
-- S_n and T_n are sums of first n terms of arithmetic sequences {a_n} and {b_n}, respectively.
axiom sum_S : ∀ n, S n = n * (n + 1) / 2  -- Example: sum from 1 to n
axiom sum_T : ∀ n, T n = n * (n + 1) / 2  -- Example: sum from 1 to n

-- For any positive integer n, (S_n / T_n = (5n - 3) / (2n + 1))
axiom condition : ∀ n > 0, (S n : ℚ) / T n = (5 * n - 3 : ℚ) / (2 * n + 1)

-- Theorem to prove
theorem problem : (a 20 : ℚ) / (b 7) = 64 / 9 :=
sorry

end NUMINAMATH_GPT_problem_l1140_114049


namespace NUMINAMATH_GPT_valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l1140_114083

-- Definition: City is divided by roads, and there are initial and additional currency exchange points

structure City := 
(exchange_points : ℕ)   -- Number of exchange points in the city
(parts : ℕ)             -- Number of parts the city is divided into

-- Given: Initial conditions with one existing exchange point and divided parts
def initialCity : City :=
{ exchange_points := 1, parts := 2 }

-- Function to add exchange points in the city
def addExchangePoints (c : City) (new_points : ℕ) : City :=
{ exchange_points := c.exchange_points + new_points, parts := c.parts }

-- Function to verify that each part has exactly two exchange points
def isValidConfiguration (c : City) : Prop :=
c.exchange_points = 2 * c.parts

-- Theorem: Prove that each configuration of new points is valid
theorem valid_first_configuration : 
  isValidConfiguration (addExchangePoints initialCity 3) := 
sorry

theorem valid_second_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_third_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_fourth_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

end NUMINAMATH_GPT_valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l1140_114083


namespace NUMINAMATH_GPT_work_rate_calculate_l1140_114073

theorem work_rate_calculate (A_time B_time C_time total_time: ℕ) 
  (hA : A_time = 4) 
  (hB : B_time = 8)
  (hTotal : total_time = 2) : 
  C_time = 8 :=
by
  sorry

end NUMINAMATH_GPT_work_rate_calculate_l1140_114073


namespace NUMINAMATH_GPT_min_value_x_plus_2_div_x_l1140_114091

theorem min_value_x_plus_2_div_x (x : ℝ) (hx : x > 0) : x + 2 / x ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_2_div_x_l1140_114091


namespace NUMINAMATH_GPT_roots_of_transformed_quadratic_l1140_114006

theorem roots_of_transformed_quadratic (a b c d x : ℝ) :
  (∀ x, (x - a) * (x - b) - x = 0 → x = c ∨ x = d) →
  (x - c) * (x - d) + x = 0 → x = a ∨ x = b :=
by
  sorry

end NUMINAMATH_GPT_roots_of_transformed_quadratic_l1140_114006


namespace NUMINAMATH_GPT_contrapositive_is_false_l1140_114040

-- Define the property of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, a = k • b

-- Define the property of vectors having the same direction
def same_direction (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, k > 0 ∧ a = k • b

-- Original proposition in Lean statement
def original_proposition (a b : ℝ × ℝ) : Prop :=
  collinear a b → same_direction a b

-- Contrapositive of the original proposition
def contrapositive_proposition (a b : ℝ × ℝ) : Prop :=
  ¬ same_direction a b → ¬ collinear a b

-- The proof goal that the contrapositive is false
theorem contrapositive_is_false (a b : ℝ × ℝ) :
  (contrapositive_proposition a b) = false :=
sorry

end NUMINAMATH_GPT_contrapositive_is_false_l1140_114040


namespace NUMINAMATH_GPT_danny_chemistry_marks_l1140_114046

theorem danny_chemistry_marks 
  (eng marks_physics marks_biology math : ℕ)
  (average: ℕ) 
  (total_marks: ℕ) 
  (chemistry: ℕ) 
  (h_eng : eng = 76) 
  (h_math : math = 65) 
  (h_phys : marks_physics = 82) 
  (h_bio : marks_biology = 75) 
  (h_avg : average = 73) 
  (h_total : total_marks = average * 5) : 
  chemistry = total_marks - (eng + math + marks_physics + marks_biology) :=
by
  sorry

end NUMINAMATH_GPT_danny_chemistry_marks_l1140_114046


namespace NUMINAMATH_GPT_probability_at_least_one_female_l1140_114084

open Nat

theorem probability_at_least_one_female :
  let males := 2
  let females := 3
  let total_students := males + females
  let select := 2
  let total_ways := choose total_students select
  let ways_at_least_one_female : ℕ := (choose females 1) * (choose males 1) + choose females 2
  (ways_at_least_one_female / total_ways : ℚ) = 9 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_female_l1140_114084


namespace NUMINAMATH_GPT_count_five_digit_multiples_of_5_l1140_114026

-- Define the range of five-digit positive integers
def lower_bound : ℕ := 10000
def upper_bound : ℕ := 99999

-- Define the divisor
def divisor : ℕ := 5

-- Define the count of multiples of 5 in the range
def count_multiples_of_5 : ℕ :=
  (upper_bound / divisor) - (lower_bound / divisor) + 1

-- The main statement: The number of five-digit multiples of 5 is 18000
theorem count_five_digit_multiples_of_5 : count_multiples_of_5 = 18000 :=
  sorry

end NUMINAMATH_GPT_count_five_digit_multiples_of_5_l1140_114026


namespace NUMINAMATH_GPT_sqrt_domain_l1140_114063

theorem sqrt_domain (x : ℝ) : x - 5 ≥ 0 ↔ x ≥ 5 :=
by sorry

end NUMINAMATH_GPT_sqrt_domain_l1140_114063


namespace NUMINAMATH_GPT_number_of_customers_left_l1140_114013

theorem number_of_customers_left (x : ℕ) (h : 14 - x + 39 = 50) : x = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_customers_left_l1140_114013


namespace NUMINAMATH_GPT_bees_leg_count_l1140_114051

-- Define the number of legs per bee
def legsPerBee : Nat := 6

-- Define the number of bees
def numberOfBees : Nat := 8

-- Calculate the total number of legs for 8 bees
def totalLegsForEightBees : Nat := 48

-- The theorem statement
theorem bees_leg_count : (legsPerBee * numberOfBees) = totalLegsForEightBees := 
by
  -- Skipping the proof by using sorry
  sorry

end NUMINAMATH_GPT_bees_leg_count_l1140_114051


namespace NUMINAMATH_GPT_number_of_distinct_gardens_l1140_114055

def is_adjacent (i1 j1 i2 j2 : ℕ) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨ 
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

def is_garden (M : ℕ → ℕ → ℕ) (m n : ℕ) : Prop :=
  ∀ i j i' j', (i < m ∧ j < n ∧ i' < m ∧ j' < n ∧ is_adjacent i j i' j') → 
    ((M i j = M i' j') ∨ (M i j = M i' j' + 1) ∨ (M i j + 1 = M i' j')) ∧
  ∀ i j, (i < m ∧ j < n ∧ 
    (∀ (i' j'), is_adjacent i j i' j' → (M i j ≤ M i' j'))) → M i j = 0

theorem number_of_distinct_gardens (m n : ℕ) : 
  ∃ (count : ℕ), count = 2 ^ (m * n) - 1 :=
sorry

end NUMINAMATH_GPT_number_of_distinct_gardens_l1140_114055


namespace NUMINAMATH_GPT_billy_video_count_l1140_114008

theorem billy_video_count 
  (generate_suggestions : ℕ) 
  (rounds : ℕ) 
  (videos_in_total : ℕ)
  (H1 : generate_suggestions = 15)
  (H2 : rounds = 5)
  (H3 : videos_in_total = generate_suggestions * rounds + 1) : 
  videos_in_total = 76 := 
by
  sorry

end NUMINAMATH_GPT_billy_video_count_l1140_114008


namespace NUMINAMATH_GPT_taller_tree_height_l1140_114005

-- Given conditions
variables (h : ℕ) (ratio_cond : (h - 20) * 7 = h * 5)

-- Proof goal
theorem taller_tree_height : h = 70 :=
sorry

end NUMINAMATH_GPT_taller_tree_height_l1140_114005


namespace NUMINAMATH_GPT_license_plates_count_l1140_114047

def numConsonantsExcludingY : Nat := 19
def numVowelsIncludingY : Nat := 6
def numConsonantsIncludingY : Nat := 21
def numEvenDigits : Nat := 5

theorem license_plates_count : 
  numConsonantsExcludingY * numVowelsIncludingY * numConsonantsIncludingY * numEvenDigits = 11970 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1140_114047


namespace NUMINAMATH_GPT_percentage_students_taking_music_l1140_114019

theorem percentage_students_taking_music
  (total_students : ℕ)
  (students_take_dance : ℕ)
  (students_take_art : ℕ)
  (students_take_music : ℕ)
  (percentage_students_taking_music : ℕ) :
  total_students = 400 →
  students_take_dance = 120 →
  students_take_art = 200 →
  students_take_music = total_students - students_take_dance - students_take_art →
  percentage_students_taking_music = (students_take_music * 100) / total_students →
  percentage_students_taking_music = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_students_taking_music_l1140_114019


namespace NUMINAMATH_GPT_geom_seq_a5_a6_eq_180_l1140_114060

theorem geom_seq_a5_a6_eq_180 (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n+1) = a n * q)
  (cond1 : a 1 + a 2 = 20)
  (cond2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 180 :=
sorry

end NUMINAMATH_GPT_geom_seq_a5_a6_eq_180_l1140_114060


namespace NUMINAMATH_GPT_maple_tree_taller_than_pine_tree_l1140_114090

def improper_fraction (a b : ℕ) : ℚ := a + (b : ℚ) / 4
def mixed_number_to_improper_fraction (n m : ℕ) : ℚ := improper_fraction n m

def pine_tree_height : ℚ := mixed_number_to_improper_fraction 12 1
def maple_tree_height : ℚ := mixed_number_to_improper_fraction 18 3

theorem maple_tree_taller_than_pine_tree :
  maple_tree_height - pine_tree_height = 6 + 1 / 2 :=
by sorry

end NUMINAMATH_GPT_maple_tree_taller_than_pine_tree_l1140_114090


namespace NUMINAMATH_GPT_sum_of_ages_l1140_114079

variable (S F : ℕ)

-- Conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F + 6 = 2 * (S + 6)

-- Theorem Statement
theorem sum_of_ages (h1 : condition1 S F) (h2 : condition2 S F) : S + 6 + (F + 6) = 36 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1140_114079


namespace NUMINAMATH_GPT_rhombus_area_l1140_114085

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) : 
  (d1 * d2) / 2 = 157.5 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1140_114085


namespace NUMINAMATH_GPT_problem1_extr_vals_l1140_114077

-- Definitions from conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

theorem problem1_extr_vals :
  ∃ a b : ℝ, a = g (1/3) ∧ b = g 1 ∧ a = 31/27 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_extr_vals_l1140_114077


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1140_114015

variable (a b m n : ℝ)

theorem simplify_expr1 : 2 * a - 6 * b - 3 * a + 9 * b = -a + 3 * b := by
  sorry

theorem simplify_expr2 : 2 * (3 * m^2 - m * n) - m * n + m^2 = 7 * m^2 - 3 * m * n := by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1140_114015


namespace NUMINAMATH_GPT_tangent_line_to_curve_l1140_114059

section TangentLine

variables {x m : ℝ}

theorem tangent_line_to_curve (x0 : ℝ) :
  (∀ x : ℝ, x > 0 → y = x * Real.log x) →
  (∀ x : ℝ, y = 2 * x + m) →
  (x0 > 0) →
  (x0 * Real.log x0 = 2 * x0 + m) →
  m = -Real.exp 1 :=
by
  sorry

end TangentLine

end NUMINAMATH_GPT_tangent_line_to_curve_l1140_114059


namespace NUMINAMATH_GPT_salary_increase_after_five_years_l1140_114096

theorem salary_increase_after_five_years :
  ∀ (S : ℝ), (S * (1.15)^5 - S) / S * 100 = 101.14 := by
sorry

end NUMINAMATH_GPT_salary_increase_after_five_years_l1140_114096


namespace NUMINAMATH_GPT_distance_of_intersections_l1140_114011

theorem distance_of_intersections 
  (t : ℝ)
  (x := (2 - t) * (Real.sin (Real.pi / 6)))
  (y := (-1 + t) * (Real.sin (Real.pi / 6)))
  (curve : x = y)
  (circle : x^2 + y^2 = 8) :
  ∃ (B C : ℝ × ℝ), dist B C = Real.sqrt 30 := 
by
  sorry

end NUMINAMATH_GPT_distance_of_intersections_l1140_114011


namespace NUMINAMATH_GPT_find_x_l1140_114066

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 :=
by sorry

end NUMINAMATH_GPT_find_x_l1140_114066


namespace NUMINAMATH_GPT_probability_at_least_one_die_shows_three_l1140_114028

noncomputable def probability_at_least_one_three : ℚ :=
  (15 : ℚ) / 64

theorem probability_at_least_one_die_shows_three :
  ∃ (p : ℚ), p = probability_at_least_one_three :=
by
  use (15 : ℚ) / 64
  sorry

end NUMINAMATH_GPT_probability_at_least_one_die_shows_three_l1140_114028


namespace NUMINAMATH_GPT_find_value_l1140_114070

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom explicit_form : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem statement
theorem find_value : f (-5/2) = -1/2 :=
by
  -- Here would be the place to start the proof based on the above axioms
  sorry

end NUMINAMATH_GPT_find_value_l1140_114070


namespace NUMINAMATH_GPT_annual_population_increase_l1140_114010

theorem annual_population_increase (P₀ P₂ : ℝ) (r : ℝ) 
  (h0 : P₀ = 12000) 
  (h2 : P₂ = 18451.2) 
  (h_eq : P₂ = P₀ * (1 + r / 100)^2) :
  r = 24 :=
by
  sorry

end NUMINAMATH_GPT_annual_population_increase_l1140_114010


namespace NUMINAMATH_GPT_matches_in_each_box_l1140_114056

noncomputable def matches_per_box (dozens_boxes : ℕ) (total_matches : ℕ) : ℕ :=
  total_matches / (dozens_boxes * 12)

theorem matches_in_each_box :
  matches_per_box 5 1200 = 20 :=
by
  sorry

end NUMINAMATH_GPT_matches_in_each_box_l1140_114056


namespace NUMINAMATH_GPT_algebra_statements_correct_l1140_114082

theorem algebra_statements_correct (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (ac < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (ab > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
sorry

end NUMINAMATH_GPT_algebra_statements_correct_l1140_114082


namespace NUMINAMATH_GPT_complement_union_complement_l1140_114054

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- The proof problem
theorem complement_union_complement : (U \ (M ∪ N)) = {1, 6} := by
  sorry

end NUMINAMATH_GPT_complement_union_complement_l1140_114054


namespace NUMINAMATH_GPT_determine_cans_l1140_114030

-- Definitions based on the conditions
def num_cans_total : ℕ := 140
def volume_large (y : ℝ) : ℝ := y + 2.5
def total_volume_large (x : ℕ) (y : ℝ) : ℝ := ↑x * volume_large y
def total_volume_small (x : ℕ) (y : ℝ) : ℝ := ↑(num_cans_total - x) * y

-- Proof statement
theorem determine_cans (x : ℕ) (y : ℝ) 
    (h1 : total_volume_large x y = 60)
    (h2 : total_volume_small x y = 60) : 
    x = 20 ∧ num_cans_total - x = 120 := 
by
  sorry

end NUMINAMATH_GPT_determine_cans_l1140_114030


namespace NUMINAMATH_GPT_inequality_proof_l1140_114034

theorem inequality_proof (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1140_114034


namespace NUMINAMATH_GPT_max_value_abc_l1140_114031

theorem max_value_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
(h_sum : a + b + c = 3) : 
  a^2 * b^3 * c^4 ≤ 2048 / 19683 :=
sorry

end NUMINAMATH_GPT_max_value_abc_l1140_114031


namespace NUMINAMATH_GPT_cubic_function_not_monotonically_increasing_l1140_114088

theorem cubic_function_not_monotonically_increasing (b : ℝ) :
  ¬(∀ x y : ℝ, x ≤ y → (1/3)*x^3 + b*x^2 + (b+2)*x + 3 ≤ (1/3)*y^3 + b*y^2 + (b+2)*y + 3) ↔ b ∈ (Set.Iio (-1) ∪ Set.Ioi 2) :=
by sorry

end NUMINAMATH_GPT_cubic_function_not_monotonically_increasing_l1140_114088


namespace NUMINAMATH_GPT_proportion_solution_l1140_114057

theorem proportion_solution (x : ℝ) : (x ≠ 0) → (1 / 3 = 5 / (3 * x)) → x = 5 :=
by
  intro hnx hproportion
  sorry

end NUMINAMATH_GPT_proportion_solution_l1140_114057


namespace NUMINAMATH_GPT_smallest_norwegian_is_1344_l1140_114065

def is_norwegian (n : ℕ) : Prop :=
  ∃ d1 d2 d3 : ℕ, n > 0 ∧ d1 < d2 ∧ d2 < d3 ∧ d1 * d2 * d3 = n ∧ d1 + d2 + d3 = 2022

theorem smallest_norwegian_is_1344 : ∀ m : ℕ, (is_norwegian m) → m ≥ 1344 :=
by
  sorry

end NUMINAMATH_GPT_smallest_norwegian_is_1344_l1140_114065


namespace NUMINAMATH_GPT_probability_blue_face_l1140_114094

-- Define the total number of faces and the number of blue faces
def total_faces : ℕ := 4 + 2 + 6
def blue_faces : ℕ := 6

-- Calculate the probability of a blue face being up when rolled
theorem probability_blue_face :
  (blue_faces : ℚ) / total_faces = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_blue_face_l1140_114094


namespace NUMINAMATH_GPT_triangle_trig_problems_l1140_114025

open Real

-- Define the main theorem
theorem triangle_trig_problems (A B C a b c : ℝ) (h1: b ≠ 0) 
  (h2: cos A - 2 * cos C ≠ 0) 
  (h3 : (cos A - 2 * cos C) / cos B = (2 * c - a) / b) 
  (h4 : cos B = 1/4)
  (h5 : b = 2) :
  (sin C / sin A = 2) ∧ 
  (2 * a * c * sqrt 15 / 4 = sqrt 15 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_trig_problems_l1140_114025


namespace NUMINAMATH_GPT_eq1_eq2_eq3_l1140_114093

theorem eq1 (x : ℝ) : (x - 2)^2 - 5 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by 
  intro h
  sorry

theorem eq2 (x : ℝ) : x^2 + 4 * x = -3 → x = -1 ∨ x = -3 := 
by 
  intro h
  sorry
  
theorem eq3 (x : ℝ) : 4 * x * (x - 2) = x - 2 → x = 2 ∨ x = 1/4 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_eq1_eq2_eq3_l1140_114093


namespace NUMINAMATH_GPT_string_cuts_l1140_114071

theorem string_cuts (L S : ℕ) (h_diff : L - S = 48) (h_sum : L + S = 64) : 
  (L / S) = 7 :=
by
  sorry

end NUMINAMATH_GPT_string_cuts_l1140_114071


namespace NUMINAMATH_GPT_quadratic_discriminant_eq_l1140_114009

theorem quadratic_discriminant_eq (a b c n : ℤ) (h_eq : a = 3) (h_b : b = -8) (h_c : c = -5)
  (h_discriminant : b^2 - 4 * a * c = n) : n = 124 := 
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_eq_l1140_114009


namespace NUMINAMATH_GPT_chord_length_l1140_114007

theorem chord_length (r d AB : ℝ) (hr : r = 5) (hd : d = 4) : AB = 6 :=
by
  -- Given
  -- r = radius = 5
  -- d = distance from center to chord = 4

  -- prove AB = 6
  sorry

end NUMINAMATH_GPT_chord_length_l1140_114007


namespace NUMINAMATH_GPT_employed_males_percentage_l1140_114052

variables {p : ℕ} -- total population
variables {employed_p : ℕ} {employed_females_p : ℕ}

-- 60 percent of the population is employed
def employed_population (p : ℕ) : ℕ := 60 * p / 100

-- 20 percent of the employed people are females
def employed_females (employed : ℕ) : ℕ := 20 * employed / 100

-- The question we're solving:
theorem employed_males_percentage (h1 : employed_p = employed_population p)
  (h2 : employed_females_p = employed_females employed_p)
  : (employed_p - employed_females_p) * 100 / p = 48 :=
by
  sorry

end NUMINAMATH_GPT_employed_males_percentage_l1140_114052


namespace NUMINAMATH_GPT_least_integer_remainder_l1140_114050

theorem least_integer_remainder (n : ℕ) 
  (h₁ : n > 1)
  (h₂ : n % 5 = 2)
  (h₃ : n % 6 = 2)
  (h₄ : n % 7 = 2)
  (h₅ : n % 8 = 2)
  (h₆ : n % 10 = 2): 
  n = 842 := 
by
  sorry

end NUMINAMATH_GPT_least_integer_remainder_l1140_114050


namespace NUMINAMATH_GPT_smallest_n_for_sum_condition_l1140_114075

theorem smallest_n_for_sum_condition :
  ∃ n, n ≥ 4 ∧ (∀ S : Finset ℤ, S.card = n → ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a + b - c - d) % 20 = 0) ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_sum_condition_l1140_114075


namespace NUMINAMATH_GPT_initial_distance_between_jack_and_christina_l1140_114002

theorem initial_distance_between_jack_and_christina
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_total_distance : ℝ)
  (meeting_time : ℝ)
  (combined_speed : ℝ) :
  jack_speed = 5 ∧
  christina_speed = 3 ∧
  lindy_speed = 9 ∧
  lindy_total_distance = 270 ∧
  meeting_time = lindy_total_distance / lindy_speed ∧
  combined_speed = jack_speed + christina_speed →
  meeting_time = 30 ∧
  combined_speed = 8 →
  (combined_speed * meeting_time) = 240 :=
by
  sorry

end NUMINAMATH_GPT_initial_distance_between_jack_and_christina_l1140_114002


namespace NUMINAMATH_GPT_find_pairs_of_positive_integers_l1140_114021

theorem find_pairs_of_positive_integers (x y : ℕ) (h : x > 0 ∧ y > 0) (h_eq : x + y + x * y = 2006) :
  (x, y) = (2, 668) ∨ (x, y) = (668, 2) ∨ (x, y) = (8, 222) ∨ (x, y) = (222, 8) :=
sorry

end NUMINAMATH_GPT_find_pairs_of_positive_integers_l1140_114021


namespace NUMINAMATH_GPT_arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l1140_114041

open Real

theorem arctan_sum_lt_pi_div_two_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  arctan x + arctan y < (π / 2) ↔ x * y < 1 :=
sorry

theorem arctan_sum_lt_pi_iff (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  arctan x + arctan y + arctan z < π ↔ x * y * z < x + y + z :=
sorry

end NUMINAMATH_GPT_arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l1140_114041


namespace NUMINAMATH_GPT_factorial_trailing_zeros_l1140_114081

theorem factorial_trailing_zeros :
  ∃ (S : Finset ℕ), (∀ m ∈ S, 1 ≤ m ∧ m ≤ 30) ∧ (S.card = 24) ∧ (∀ m ∈ S, 
    ∃ n : ℕ, ∃ k : ℕ,  n ≥ k * 5 ∧ n ≤ (k + 1) * 5 - 1 ∧ 
      m = (n / 5) + (n / 25) + (n / 125) ∧ ((n / 5) % 5 = 0)) :=
sorry

end NUMINAMATH_GPT_factorial_trailing_zeros_l1140_114081


namespace NUMINAMATH_GPT_find_g7_l1140_114027

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_value : g 6 = 7

theorem find_g7 : g 7 = 49 / 6 := by
  sorry

end NUMINAMATH_GPT_find_g7_l1140_114027


namespace NUMINAMATH_GPT_percentage_land_mr_william_l1140_114072

noncomputable def tax_rate_arable := 0.01
noncomputable def tax_rate_orchard := 0.02
noncomputable def tax_rate_pasture := 0.005

noncomputable def subsidy_arable := 100
noncomputable def subsidy_orchard := 50
noncomputable def subsidy_pasture := 20

noncomputable def total_tax_village := 3840
noncomputable def tax_mr_william := 480

theorem percentage_land_mr_william : 
  (tax_mr_william / total_tax_village : ℝ) * 100 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_land_mr_william_l1140_114072


namespace NUMINAMATH_GPT_new_circumference_of_circle_l1140_114035

theorem new_circumference_of_circle (w h : ℝ) (d_multiplier : ℝ) 
  (h_w : w = 7) (h_h : h = 24) (h_d_multiplier : d_multiplier = 1.5) : 
  (π * (d_multiplier * (Real.sqrt (w^2 + h^2)))) = 37.5 * π :=
by
  sorry

end NUMINAMATH_GPT_new_circumference_of_circle_l1140_114035


namespace NUMINAMATH_GPT_total_students_end_of_year_l1140_114062

def M := 50
def E (M : ℕ) := 4 * M - 3
def H (E : ℕ) := 2 * E

def E_end (E : ℕ) := E + (E / 10)
def M_end (M : ℕ) := M - (M / 20)
def H_end (H : ℕ) := H + ((7 * H) / 100)

def total_end (E_end M_end H_end : ℕ) := E_end + M_end + H_end

theorem total_students_end_of_year : 
  total_end (E_end (E M)) (M_end M) (H_end (H (E M))) = 687 := sorry

end NUMINAMATH_GPT_total_students_end_of_year_l1140_114062


namespace NUMINAMATH_GPT_factorial_expression_calculation_l1140_114038

theorem factorial_expression_calculation :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 6) - 6 * (Nat.factorial 5) = 7920 :=
by
  sorry

end NUMINAMATH_GPT_factorial_expression_calculation_l1140_114038


namespace NUMINAMATH_GPT_count_valid_N_l1140_114089

theorem count_valid_N : ∃ (N : ℕ), N = 1174 ∧ ∀ (n : ℕ), (1 ≤ n ∧ n < 2000) → ∃ (x : ℝ), x ^ (⌊x⌋ + 1) = n :=
by
  sorry

end NUMINAMATH_GPT_count_valid_N_l1140_114089


namespace NUMINAMATH_GPT_range_of_a_l1140_114039

theorem range_of_a (a : ℝ) : (∀ x > 1, x^2 ≥ a) ↔ (a ≤ 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1140_114039


namespace NUMINAMATH_GPT_candy_store_revenue_l1140_114024

def fudge_revenue : ℝ := 20 * 2.50
def truffles_revenue : ℝ := 5 * 12 * 1.50
def pretzels_revenue : ℝ := 3 * 12 * 2.00
def total_revenue : ℝ := fudge_revenue + truffles_revenue + pretzels_revenue

theorem candy_store_revenue :
  total_revenue = 212.00 :=
sorry

end NUMINAMATH_GPT_candy_store_revenue_l1140_114024


namespace NUMINAMATH_GPT_zero_x_intersections_l1140_114044

theorem zero_x_intersections 
  (a b c : ℝ) 
  (h_geom_seq : b^2 = a * c) 
  (h_ac_pos : a * c > 0) : 
  ∀ x : ℝ, ¬(ax^2 + bx + c = 0) := 
by 
  sorry

end NUMINAMATH_GPT_zero_x_intersections_l1140_114044


namespace NUMINAMATH_GPT_find_y_l1140_114033

-- Definition of the modified magic square
variable (a b c d e y : ℕ)

-- Conditions from the modified magic square problem
axiom h1 : y + 5 + c = 120 + a + c
axiom h2 : y + (y - 115) + e = 120 + b + e
axiom h3 : y + 25 + 120 = 5 + (y - 115) + (2*y - 235)

-- The statement to prove
theorem find_y : y = 245 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1140_114033


namespace NUMINAMATH_GPT_age_weight_not_proportional_l1140_114067

theorem age_weight_not_proportional (age weight : ℕ) : ¬(∃ k, ∀ (a w : ℕ), w = k * a → age / weight = k) :=
by
  sorry

end NUMINAMATH_GPT_age_weight_not_proportional_l1140_114067


namespace NUMINAMATH_GPT_initial_number_of_kids_l1140_114061

theorem initial_number_of_kids (joined kids_total initial : ℕ) (h1 : joined = 22) (h2 : kids_total = 36) (h3 : kids_total = initial + joined) : initial = 14 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_number_of_kids_l1140_114061


namespace NUMINAMATH_GPT_andy_solves_49_problems_l1140_114018

theorem andy_solves_49_problems : ∀ (a b : ℕ), a = 78 → b = 125 → b - a + 1 = 49 :=
by
  introv ha hb
  rw [ha, hb]
  norm_num
  sorry

end NUMINAMATH_GPT_andy_solves_49_problems_l1140_114018


namespace NUMINAMATH_GPT_cos_value_of_2alpha_plus_5pi_over_12_l1140_114023

theorem cos_value_of_2alpha_plus_5pi_over_12
  (α : ℝ) (h1 : Real.pi / 2 < α ∧ α < Real.pi)
  (h2 : Real.sin (α + Real.pi / 3) = -4 / 5) :
  Real.cos (2 * α + 5 * Real.pi / 12) = 17 * Real.sqrt 2 / 50 :=
by 
  sorry

end NUMINAMATH_GPT_cos_value_of_2alpha_plus_5pi_over_12_l1140_114023


namespace NUMINAMATH_GPT_prove_y_minus_x_l1140_114032

theorem prove_y_minus_x (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 7 / 8) : y - x = 100 / 3 := 
by
  sorry

end NUMINAMATH_GPT_prove_y_minus_x_l1140_114032


namespace NUMINAMATH_GPT_solve_fraction_eq_zero_l1140_114016

theorem solve_fraction_eq_zero (x : ℝ) (h : x ≠ 0) : 
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_zero_l1140_114016


namespace NUMINAMATH_GPT_symmetric_point_l1140_114069

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def plane_eq (M : Point3D) : Prop :=
  2 * M.x - 4 * M.y - 4 * M.z - 13 = 0

-- Given Point M
def M : Point3D := { x := 3, y := -3, z := -1 }

-- Symmetric Point M'
def M' : Point3D := { x := 2, y := -1, z := 1 }

theorem symmetric_point (H : plane_eq M) : plane_eq M' ∧ 
  (M'.x = 2 * (3 + 2 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.x) ∧ 
  (M'.y = 2 * (-3 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.y) ∧ 
  (M'.z = 2 * (-1 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.z) :=
sorry

end NUMINAMATH_GPT_symmetric_point_l1140_114069


namespace NUMINAMATH_GPT_sufficient_condition_above_2c_l1140_114064

theorem sufficient_condition_above_2c (a b c : ℝ) (h1 : a > c) (h2 : b > c) : a + b > 2 * c :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_above_2c_l1140_114064


namespace NUMINAMATH_GPT_sum_of_coefficients_of_factorized_polynomial_l1140_114076

theorem sum_of_coefficients_of_factorized_polynomial : 
  ∃ (a b c d e : ℕ), 
    (216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 36) :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_factorized_polynomial_l1140_114076


namespace NUMINAMATH_GPT_largest_of_three_l1140_114000

theorem largest_of_three (a b c : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 4) : max a (max b c) = 8 := 
sorry

end NUMINAMATH_GPT_largest_of_three_l1140_114000


namespace NUMINAMATH_GPT_total_mangoes_l1140_114068

-- Definitions of the entities involved
variables (Alexis Dilan Ashley Ben : ℚ)

-- Conditions given in the problem
def condition1 : Prop := Alexis = 4 * (Dilan + Ashley) ∧ Alexis = 60
def condition2 : Prop := Ashley = 2 * Dilan
def condition3 : Prop := Ben = (1/2) * (Dilan + Ashley)

-- The theorem we want to prove: total mangoes is 82.5
theorem total_mangoes (Alexis Dilan Ashley Ben : ℚ)
  (h1 : condition1 Alexis Dilan Ashley)
  (h2 : condition2 Dilan Ashley)
  (h3 : condition3 Dilan Ashley Ben) :
  Alexis + Dilan + Ashley + Ben = 82.5 :=
sorry

end NUMINAMATH_GPT_total_mangoes_l1140_114068


namespace NUMINAMATH_GPT_domain_tan_3x_sub_pi_over_4_l1140_114003

noncomputable def domain_of_f : Set ℝ :=
  {x : ℝ | ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4}

theorem domain_tan_3x_sub_pi_over_4 :
  ∀ x : ℝ, x ∈ domain_of_f ↔ ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_domain_tan_3x_sub_pi_over_4_l1140_114003


namespace NUMINAMATH_GPT_candy_bar_cost_l1140_114099

theorem candy_bar_cost :
  ∀ (members : ℕ) (avg_candy_bars : ℕ) (total_earnings : ℝ), 
  members = 20 →
  avg_candy_bars = 8 →
  total_earnings = 80 →
  total_earnings / (members * avg_candy_bars) = 0.50 :=
by
  intros members avg_candy_bars total_earnings h_mem h_avg h_earn
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l1140_114099


namespace NUMINAMATH_GPT_car_distance_traveled_l1140_114029

theorem car_distance_traveled (d : ℝ)
  (h_avg_speed : 84.70588235294117 = 320 / ((d / 90) + (d / 80))) :
  d = 160 :=
by
  sorry

end NUMINAMATH_GPT_car_distance_traveled_l1140_114029


namespace NUMINAMATH_GPT_decimal_to_fraction_sum_l1140_114043

def recurring_decimal_fraction_sum : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ gcd a b = 1 ∧ (a / b : ℚ) = (0.345345345 : ℚ) ∧ a + b = 226

theorem decimal_to_fraction_sum :
  recurring_decimal_fraction_sum :=
sorry

end NUMINAMATH_GPT_decimal_to_fraction_sum_l1140_114043


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1140_114098

-- The sequence S_n and its given condition
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2 * n

-- Definitions for a_1, a_2, and a_3 based on S_n conditions
theorem problem_1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 14 :=
sorry

-- Definition of sequence b_n and its property of being geometric
def b (n : ℕ) (a : ℕ → ℕ) : ℕ := a n + 2

theorem problem_2 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n ≥ 1, b n a = 2 * b (n - 1) a :=
sorry

-- The sum of the first n terms of the sequence {na_n}, denoted by T_n
def T (n : ℕ) (a : ℕ → ℕ) : ℕ := (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1)

theorem problem_3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n, T n a = (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1140_114098


namespace NUMINAMATH_GPT_total_blocks_to_ride_l1140_114036

-- Constants representing the problem conditions
def rotations_per_block : ℕ := 200
def initial_rotations : ℕ := 600
def additional_rotations : ℕ := 1000

-- Main statement asserting the total number of blocks Greg wants to ride
theorem total_blocks_to_ride : 
  (initial_rotations / rotations_per_block) + (additional_rotations / rotations_per_block) = 8 := 
  by 
    sorry

end NUMINAMATH_GPT_total_blocks_to_ride_l1140_114036


namespace NUMINAMATH_GPT_circle_radius_l1140_114014

theorem circle_radius (A : ℝ) (k : ℝ) (r : ℝ) (h : A = k * π * r^2) (hA : A = 225 * π) (hk : k = 4) : 
  r = 7.5 :=
by 
  sorry

end NUMINAMATH_GPT_circle_radius_l1140_114014


namespace NUMINAMATH_GPT_planned_pencils_is_49_l1140_114080

def pencils_planned (x : ℕ) : ℕ := x
def pencils_bought (x : ℕ) : ℕ := x + 12
def total_pencils_bought (x : ℕ) : ℕ := 61

theorem planned_pencils_is_49 (x : ℕ) :
  pencils_bought (pencils_planned x) = total_pencils_bought x → x = 49 :=
sorry

end NUMINAMATH_GPT_planned_pencils_is_49_l1140_114080


namespace NUMINAMATH_GPT_sandy_took_200_l1140_114092

variable (X : ℝ)

/-- Given that Sandy had $140 left after spending 30% of the money she took for shopping,
we want to prove that Sandy took $200 for shopping. -/
theorem sandy_took_200 (h : 0.70 * X = 140) : X = 200 :=
by
  sorry

end NUMINAMATH_GPT_sandy_took_200_l1140_114092


namespace NUMINAMATH_GPT_range_of_a_l1140_114022

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2 * x - 1| + |x + 1| > a) ↔ a < 3 / 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1140_114022


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1140_114045

theorem necessary_but_not_sufficient_condition
    {a b : ℕ} :
    (¬ (a = 1) ∨ ¬ (b = 2)) ↔ (a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2) :=
by
    sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1140_114045


namespace NUMINAMATH_GPT_problem_relationship_l1140_114048

theorem problem_relationship (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_relationship_l1140_114048


namespace NUMINAMATH_GPT_track_time_is_80_l1140_114097

noncomputable def time_to_complete_track
  (a b : ℕ) 
  (meetings : a = 15 ∧ b = 25) : ℕ :=
a + b

theorem track_time_is_80 (a b : ℕ) (meetings : a = 15 ∧ b = 25) : time_to_complete_track a b meetings = 80 := by
  sorry

end NUMINAMATH_GPT_track_time_is_80_l1140_114097


namespace NUMINAMATH_GPT_sum_of_angles_is_540_l1140_114095

variables (angle1 angle2 angle3 angle4 angle5 angle6 angle7 : ℝ)

theorem sum_of_angles_is_540
  (h : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540 :=
sorry

end NUMINAMATH_GPT_sum_of_angles_is_540_l1140_114095


namespace NUMINAMATH_GPT_find_m_l1140_114074

noncomputable def f (x : ℝ) := 4 * x^2 - 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -14 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1140_114074


namespace NUMINAMATH_GPT_simplify_fraction_l1140_114086

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1140_114086


namespace NUMINAMATH_GPT_next_podcast_length_l1140_114053

theorem next_podcast_length 
  (drive_hours : ℕ := 6)
  (podcast1_minutes : ℕ := 45)
  (podcast2_minutes : ℕ := 90) -- Since twice the first podcast (45 * 2)
  (podcast3_minutes : ℕ := 105) -- 1 hour 45 minutes (60 + 45)
  (podcast4_minutes : ℕ := 60) -- 1 hour 
  (minutes_per_hour : ℕ := 60)
  : (drive_hours * minutes_per_hour - (podcast1_minutes + podcast2_minutes + podcast3_minutes + podcast4_minutes)) / minutes_per_hour = 1 :=
by
  sorry

end NUMINAMATH_GPT_next_podcast_length_l1140_114053


namespace NUMINAMATH_GPT_spell_casting_contest_orders_l1140_114037

-- Definition for factorial
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem statement: number of ways to order 4 contestants is 4!
theorem spell_casting_contest_orders : factorial 4 = 24 := by
  sorry

end NUMINAMATH_GPT_spell_casting_contest_orders_l1140_114037


namespace NUMINAMATH_GPT_find_lunch_days_l1140_114001

variable (x y : ℕ) -- School days for School A and School B
def P_A := x / 2 -- Aliyah packs lunch half the time
def P_B := y / 4 -- Becky packs lunch a quarter of the time
def P_C := y / 2 -- Charlie packs lunch half the time

theorem find_lunch_days (x y : ℕ) :
  P_A x = x / 2 ∧
  P_B y = y / 4 ∧
  P_C y = y / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_lunch_days_l1140_114001
