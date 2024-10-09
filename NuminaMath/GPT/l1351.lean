import Mathlib

namespace dozen_Pokemon_cards_per_friend_l1351_135154

theorem dozen_Pokemon_cards_per_friend
  (total_cards : ℕ) (num_friends : ℕ) (cards_per_dozen : ℕ)
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : cards_per_dozen = 12) :
  (total_cards / num_friends) / cards_per_dozen = 9 := 
sorry

end dozen_Pokemon_cards_per_friend_l1351_135154


namespace find_x_l1351_135143

theorem find_x (x : ℝ) (A B : Set ℝ) (hA : A = {1, 4, x}) (hB : B = {1, x^2}) (h_inter : A ∩ B = B) : x = -2 ∨ x = 2 ∨ x = 0 :=
sorry

end find_x_l1351_135143


namespace rectangle_problem_l1351_135156

theorem rectangle_problem (x : ℝ) (h1 : 4 * x = l) (h2 : x + 7 = w) (h3 : l * w = 2 * (2 * l + 2 * w)) : x = 1 := 
by {
  sorry
}

end rectangle_problem_l1351_135156


namespace ratio_problem_l1351_135133

theorem ratio_problem (X : ℕ) :
  (18 : ℕ) * 360 = 9 * X → X = 720 :=
by
  intro h
  sorry

end ratio_problem_l1351_135133


namespace value_of_expression_l1351_135120

theorem value_of_expression :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) :=
by
  -- Proof goes here
  sorry

end value_of_expression_l1351_135120


namespace inverse_property_l1351_135129

-- Given conditions
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variable (hf_injective : Function.Injective f)
variable (hf_surjective : Function.Surjective f)
variable (h_inverse : ∀ y : ℝ, f (f_inv y) = y)
variable (hf_property : ∀ x : ℝ, f (-x) + f (x) = 3)

-- The proof goal
theorem inverse_property (x : ℝ) : (f_inv (x - 1) + f_inv (4 - x)) = 0 :=
by
  sorry

end inverse_property_l1351_135129


namespace number_of_pairs_lcm_600_l1351_135135

theorem number_of_pairs_lcm_600 :
  ∃ n, n = 53 ∧ (∀ m n : ℕ, (m ≤ n ∧ m > 0 ∧ n > 0 ∧ Nat.lcm m n = 600) ↔ n = 53) := sorry

end number_of_pairs_lcm_600_l1351_135135


namespace gcd_372_684_l1351_135169

theorem gcd_372_684 : Nat.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l1351_135169


namespace systematic_sampling_l1351_135173

-- Definitions for the class of 50 students numbered from 1 to 50, sampling interval, and starting number.
def students : Set ℕ := {n | n ∈ Finset.range 50 ∧ n ≥ 1}
def sampling_interval : ℕ := 10
def start : ℕ := 6

-- The main theorem stating that the selected students' numbers are as given.
theorem systematic_sampling : ∃ (selected : List ℕ), selected = [6, 16, 26, 36, 46] ∧ 
  ∀ x ∈ selected, x ∈ students := 
  sorry

end systematic_sampling_l1351_135173


namespace trees_died_in_typhoon_l1351_135107

-- Define the total number of trees, survived trees, and died trees
def total_trees : ℕ := 14

def survived_trees (S : ℕ) : ℕ := S

def died_trees (S : ℕ) : ℕ := S + 4

-- The Lean statement that formalizes the proof problem
theorem trees_died_in_typhoon : ∃ S : ℕ, survived_trees S + died_trees S = total_trees ∧ died_trees S = 9 :=
by
  -- Provide a placeholder for the proof
  sorry

end trees_died_in_typhoon_l1351_135107


namespace solve_for_x_l1351_135122

theorem solve_for_x 
    (x : ℝ) 
    (h : (4 * x - 2) / (5 * x - 5) = 3 / 4) 
    : x = -7 :=
sorry

end solve_for_x_l1351_135122


namespace flag_yellow_area_percentage_l1351_135130

theorem flag_yellow_area_percentage (s w : ℝ) (h_flag_area : s > 0)
  (h_width_positive : w > 0) (h_cross_area : 4 * s * w - 3 * w^2 = 0.49 * s^2) :
  (w^2 / s^2) * 100 = 12.25 :=
by
  sorry

end flag_yellow_area_percentage_l1351_135130


namespace perimeter_of_region_l1351_135117

theorem perimeter_of_region : 
  let side := 1
  let diameter := side
  let radius := diameter / 2
  let full_circumference := 2 * Real.pi * radius
  let arc_length := (3 / 4) * full_circumference
  let total_arcs := 4
  let perimeter := total_arcs * arc_length
  perimeter = 3 * Real.pi :=
by 
  sorry

end perimeter_of_region_l1351_135117


namespace small_supermarkets_sample_count_l1351_135174

def large := 300
def medium := 600
def small := 2100
def sample_size := 100
def total := large + medium + small

theorem small_supermarkets_sample_count :
  small * (sample_size / total) = 70 := by
  sorry

end small_supermarkets_sample_count_l1351_135174


namespace combined_distance_l1351_135138

theorem combined_distance (t1 t2 : ℕ) (s1 s2 : ℝ)
  (h1 : t1 = 30) (h2 : s1 = 9.5) (h3 : t2 = 45) (h4 : s2 = 8.3)
  : (s1 * t1 + s2 * t2) = 658.5 := 
by
  sorry

end combined_distance_l1351_135138


namespace smallest_rel_prime_120_l1351_135189

theorem smallest_rel_prime_120 : ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 120 = 1 ∧ ∀ y, y > 1 ∧ Nat.gcd y 120 = 1 → x ≤ y :=
by
  use 7
  sorry

end smallest_rel_prime_120_l1351_135189


namespace t_sum_max_min_l1351_135139

noncomputable def t_max (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry
noncomputable def t_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry

theorem t_sum_max_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) :
  t_max a b h + t_min a b h = 16 / 7 := sorry

end t_sum_max_min_l1351_135139


namespace solution_l1351_135197

noncomputable def polynomial (x m : ℝ) := 3 * x^2 - 5 * x + m

theorem solution (m : ℝ) : (∃ a : ℝ, a = 2 ∧ polynomial a m = 0) -> m = -2 := by
  sorry

end solution_l1351_135197


namespace bank_robbery_car_l1351_135166

def car_statement (make color : String) : Prop :=
  (make = "Buick" ∨ color = "blue") ∧
  (make = "Chrysler" ∨ color = "black") ∧
  (make = "Ford" ∨ color ≠ "blue")

theorem bank_robbery_car : ∃ make color : String, car_statement make color ∧ make = "Buick" ∧ color = "black" :=
by
  sorry

end bank_robbery_car_l1351_135166


namespace axis_of_symmetry_l1351_135185

-- Given conditions
variables {b c : ℝ}
axiom eq_roots : ∃ (x1 x2 : ℝ), (x1 = -1 ∧ x2 = 2) ∧ (x1 + x2 = -b) ∧ (x1 * x2 = c)

-- Question translation to Lean statement
theorem axis_of_symmetry : 
  ∀ b c, 
  (∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ x1 + x2 = -b ∧ x1 * x2 = c) 
  → -b / 2 = 1 / 2 := 
by 
  sorry

end axis_of_symmetry_l1351_135185


namespace no_two_distinct_real_roots_l1351_135171

-- Definitions of the conditions and question in Lean 4
theorem no_two_distinct_real_roots (a : ℝ) (h : a ≥ 1) : ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*x1 + a = 0) ∧ (x2^2 - 2*x2 + a = 0) :=
sorry

end no_two_distinct_real_roots_l1351_135171


namespace sharpener_difference_l1351_135136

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end sharpener_difference_l1351_135136


namespace solve_part_a_solve_part_b_l1351_135106

-- Part (a)
theorem solve_part_a (x : ℝ) (h1 : 36 * x^2 - 1 = (6 * x + 1) * (6 * x - 1)) :
  (3 / (1 - 6 * x) = 2 / (6 * x + 1) - (8 + 9 * x) / (36 * x^2 - 1)) ↔ x = 1 / 3 :=
sorry

-- Part (b)
theorem solve_part_b (z : ℝ) (h2 : 1 - z^2 = (1 + z) * (1 - z)) :
  (3 / (1 - z^2) = 2 / (1 + z)^2 - 5 / (1 - z)^2) ↔ z = -3 / 7 :=
sorry

end solve_part_a_solve_part_b_l1351_135106


namespace large_block_volume_l1351_135194

theorem large_block_volume (W D L : ℝ) (h1 : W * D * L = 3) : 
  (2 * W) * (2 * D) * (3 * L) = 36 := 
by 
  sorry

end large_block_volume_l1351_135194


namespace part_I_part_II_l1351_135192

-- Condition definitions:
def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

-- Part I: Prove m = 1
theorem part_I (m : ℝ) : (∀ x : ℝ, f (x + 2) m ≥ 0) ↔ m = 1 :=
by
  sorry

-- Part II: Prove a + 2b + 3c ≥ 9
theorem part_II (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end part_I_part_II_l1351_135192


namespace number_of_students_l1351_135179

def total_students (a b : ℕ) : ℕ :=
  a + b

variables (a b : ℕ)

theorem number_of_students (h : 48 * a + 45 * b = 972) : total_students a b = 21 :=
by
  sorry

end number_of_students_l1351_135179


namespace like_terms_sum_l1351_135127

theorem like_terms_sum (m n : ℕ) (h1 : 6 * x ^ 5 * y ^ (2 * n) = 6 * x ^ m * y ^ 4) : m + n = 7 := by
  sorry

end like_terms_sum_l1351_135127


namespace Dan_tshirts_total_l1351_135115

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end Dan_tshirts_total_l1351_135115


namespace meal_preppers_activity_setters_count_l1351_135112

-- Definitions for the problem conditions
def num_friends : ℕ := 6
def num_meal_preppers : ℕ := 3

-- Statement of the theorem
theorem meal_preppers_activity_setters_count :
  (num_friends.choose num_meal_preppers) = 20 :=
by
  -- Proof would go here
  sorry

end meal_preppers_activity_setters_count_l1351_135112


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l1351_135113

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l1351_135113


namespace count_true_statements_l1351_135164

theorem count_true_statements (a b c d : ℝ) : 
  (∃ (H1 : a ≠ b) (H2 : c ≠ d), a + c = b + d) →
  ((a ≠ b) ∧ (c ≠ d) → a + c ≠ b + d) = false ∧ 
  ((a + c ≠ b + d) → (a ≠ b) ∧ (c ≠ d)) = false ∧ 
  (∃ (H3 : a = b) (H4 : c = d), a + c ≠ b + d) = false ∧ 
  ((a + c = b + d) → (a = b) ∨ (c = d)) = false → 
  number_of_true_statements = 0 := 
by
  sorry

end count_true_statements_l1351_135164


namespace percentage_of_a_added_to_get_x_l1351_135163

variable (a b x m : ℝ) (P : ℝ) (k : ℝ)
variable (h1 : a / b = 4 / 5)
variable (h2 : x = a * (1 + P / 100))
variable (h3 : m = b * 0.2)
variable (h4 : m / x = 0.14285714285714285)

theorem percentage_of_a_added_to_get_x :
  P = 75 :=
by
  sorry

end percentage_of_a_added_to_get_x_l1351_135163


namespace right_triangle_exists_l1351_135190

theorem right_triangle_exists :
  (3^2 + 4^2 = 5^2) ∧ ¬(2^2 + 3^2 = 4^2) ∧ ¬(4^2 + 6^2 = 7^2) ∧ ¬(5^2 + 11^2 = 12^2) :=
by
  sorry

end right_triangle_exists_l1351_135190


namespace top_layer_lamps_l1351_135178

theorem top_layer_lamps (a : ℕ) :
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a = 381) → a = 3 := 
by
  intro h
  sorry

end top_layer_lamps_l1351_135178


namespace find_values_of_M_l1351_135116

theorem find_values_of_M :
  ∃ M : ℕ, 
    (M = 81 ∨ M = 92) ∧ 
    (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ M = 10 * a + b ∧
     (∃ k : ℕ, k ^ 3 = 9 * (a - b) ∧ k > 0)) :=
sorry

end find_values_of_M_l1351_135116


namespace monthly_salary_l1351_135162

variables (S : ℝ) (savings : ℝ) (new_expenses : ℝ)

theorem monthly_salary (h1 : savings = 0.20 * S)
                      (h2 : new_expenses = 0.96 * S)
                      (h3 : S = 200 + new_expenses) :
                      S = 5000 :=
by
  sorry

end monthly_salary_l1351_135162


namespace larger_number_is_8_l1351_135180

-- Define the conditions
def is_twice (x y : ℕ) : Prop := x = 2 * y
def product_is_40 (x y : ℕ) : Prop := x * y = 40
def sum_is_14 (x y : ℕ) : Prop := x + y = 14

-- The proof statement
theorem larger_number_is_8 (x y : ℕ) (h1 : is_twice x y) (h2 : product_is_40 x y) (h3 : sum_is_14 x y) : x = 8 :=
  sorry

end larger_number_is_8_l1351_135180


namespace percentage_of_uninsured_part_time_l1351_135146

noncomputable def number_of_employees := 330
noncomputable def uninsured_employees := 104
noncomputable def part_time_employees := 54
noncomputable def probability_neither := 0.5606060606060606

theorem percentage_of_uninsured_part_time:
  (13 / 104) * 100 = 12.5 := 
by 
  -- Here you can assume proof steps would occur/assertions to align with the solution found
  sorry

end percentage_of_uninsured_part_time_l1351_135146


namespace parallel_lines_m_values_l1351_135151

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end parallel_lines_m_values_l1351_135151


namespace base3_last_two_digits_l1351_135188

open Nat

theorem base3_last_two_digits (a b c : ℕ) (h1 : a = 2005) (h2 : b = 2003) (h3 : c = 2004) :
  (2005 ^ (2003 ^ 2004 + 3) % 81) = 11 :=
by
  sorry

end base3_last_two_digits_l1351_135188


namespace gcd_14568_78452_l1351_135182

theorem gcd_14568_78452 : Nat.gcd 14568 78452 = 4 :=
sorry

end gcd_14568_78452_l1351_135182


namespace max_value_of_a_plus_b_l1351_135186

theorem max_value_of_a_plus_b (a b : ℕ) 
  (h : 5 * a + 19 * b = 213) : a + b ≤ 37 :=
  sorry

end max_value_of_a_plus_b_l1351_135186


namespace smallest_four_digit_divisible_by_53_l1351_135123

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l1351_135123


namespace contrapositive_of_odd_even_l1351_135142

-- Definitions as conditions
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main statement
theorem contrapositive_of_odd_even :
  (∀ a b : ℕ, is_odd a ∧ is_odd b → is_even (a + b)) →
  (∀ a b : ℕ, ¬ is_even (a + b) → ¬ (is_odd a ∧ is_odd b)) := 
by
  intros h a b h1
  sorry

end contrapositive_of_odd_even_l1351_135142


namespace product_equation_l1351_135102

/-- Given two numbers x and y such that x + y = 20 and x - y = 4,
    the product of three times the larger number and the smaller number is 288. -/
theorem product_equation (x y : ℕ) (h1 : x + y = 20) (h2 : x - y = 4) (h3 : x > y) : 3 * x * y = 288 := 
sorry

end product_equation_l1351_135102


namespace smallest_w_l1351_135161

theorem smallest_w (w : ℕ) (h1 : 2^4 ∣ 1452 * w) (h2 : 3^3 ∣ 1452 * w) (h3 : 13^3 ∣ 1452 * w) : w = 79132 :=
by
  sorry

end smallest_w_l1351_135161


namespace max_ahn_achieve_max_ahn_achieve_attained_l1351_135103

def is_two_digit_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ahn_achieve :
  ∀ (n : ℕ), is_two_digit_integer n → 3 * (300 - n) ≤ 870 := 
by sorry

theorem max_ahn_achieve_attained :
  3 * (300 - 10) = 870 := 
by norm_num

end max_ahn_achieve_max_ahn_achieve_attained_l1351_135103


namespace determine_digits_l1351_135158

theorem determine_digits (h t u : ℕ) (hu: h > u) (h_subtr: t = h - 5) (unit_result: u = 3) : (h = 9 ∧ t = 4 ∧ u = 3) := by
  sorry

end determine_digits_l1351_135158


namespace quadratic_pairs_square_diff_exists_l1351_135170

open Nat Polynomial

theorem quadratic_pairs_square_diff_exists (P : Polynomial ℤ) (u v w a b n : ℤ) (n_pos : 0 < n)
    (hp : ∃ (u v w : ℤ), P = C u * X ^ 2 + C v * X + C w)
    (h_ab : P.eval a - P.eval b = n^2) : ∃ k > 10^6, ∃ m : ℕ, ∃ c d : ℤ, (c - d = a - b + 2 * k) ∧ 
    (P.eval c - P.eval d = n^2 * m ^ 2) :=
by
  sorry

end quadratic_pairs_square_diff_exists_l1351_135170


namespace chord_equation_l1351_135108

-- Definitions and conditions
def parabola (x y : ℝ) := y^2 = 8 * x
def point_Q := (4, 1)

-- Statement to prove
theorem chord_equation :
  ∃ (m : ℝ) (c : ℝ), m = 4 ∧ c = -15 ∧
    ∀ (x y : ℝ), (parabola x y ∧ x + y = 8 ∧ y + y = 2) →
      4 * x - y = 15 :=
by
  sorry -- Proof elided

end chord_equation_l1351_135108


namespace arithmetic_square_root_of_9_l1351_135105

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end arithmetic_square_root_of_9_l1351_135105


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l1351_135148

section ProbabilityQuiz

variable (total_questions : ℕ) (mc_questions : ℕ) (tf_questions : ℕ)

def prob_A_mc_and_B_tf (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  (mc_questions * tf_questions : ℚ) / (total_questions * (total_questions - 1))

def prob_at_least_one_mc (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  1 - ((tf_questions * (tf_questions - 1) : ℚ) / (total_questions * (total_questions - 1)))

theorem probability_A_mc_and_B_tf :
  prob_A_mc_and_B_tf 10 6 4 = 4 / 15 := by
  sorry

theorem probability_at_least_one_mc :
  prob_at_least_one_mc 10 6 4 = 13 / 15 := by
  sorry

end ProbabilityQuiz

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l1351_135148


namespace arithmetic_sequence_third_term_l1351_135160

theorem arithmetic_sequence_third_term 
    (a d : ℝ) 
    (h1 : a = 2)
    (h2 : (a + d) + (a + 3 * d) = 10) : 
    a + 2 * d = 5 := 
by
  sorry

end arithmetic_sequence_third_term_l1351_135160


namespace weight_gain_difference_l1351_135157

theorem weight_gain_difference :
  let orlando_gain := 5
  let jose_gain := 2 * orlando_gain + 2
  let total_gain := 20
  let fernando_gain := total_gain - (orlando_gain + jose_gain)
  let half_jose_gain := jose_gain / 2
  half_jose_gain - fernando_gain = 3 :=
by
  sorry

end weight_gain_difference_l1351_135157


namespace factorize_expr_l1351_135125

theorem factorize_expr (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l1351_135125


namespace largest_number_l1351_135109

-- Define the set elements with b = -3
def neg_5b (b : ℤ) : ℤ := -5 * b
def pos_3b (b : ℤ) : ℤ := 3 * b
def frac_30_b (b : ℤ) : ℤ := 30 / b
def b_sq (b : ℤ) : ℤ := b * b

-- Prove that when b = -3, the largest element in the set {-5b, 3b, 30/b, b^2, 2} is 15
theorem largest_number (b : ℤ) (h : b = -3) : max (max (max (max (neg_5b b) (pos_3b b)) (frac_30_b b)) (b_sq b)) 2 = 15 :=
by {
  sorry
}

end largest_number_l1351_135109


namespace remainder_when_divided_by_13_is_11_l1351_135144

theorem remainder_when_divided_by_13_is_11 
  (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : 
  349 % 13 = 11 := 
by 
  sorry

end remainder_when_divided_by_13_is_11_l1351_135144


namespace find_temp_friday_l1351_135104

-- Definitions for conditions
variables (M T W Th F : ℝ)

-- Condition 1: Average temperature for Monday to Thursday is 48 degrees
def avg_temp_mon_thu : Prop := (M + T + W + Th) / 4 = 48

-- Condition 2: Average temperature for Tuesday to Friday is 46 degrees
def avg_temp_tue_fri : Prop := (T + W + Th + F) / 4 = 46

-- Condition 3: Temperature on Monday is 39 degrees
def temp_monday : Prop := M = 39

-- Theorem: Temperature on Friday is 31 degrees
theorem find_temp_friday (h1 : avg_temp_mon_thu M T W Th)
                         (h2 : avg_temp_tue_fri T W Th F)
                         (h3 : temp_monday M) :
  F = 31 :=
sorry

end find_temp_friday_l1351_135104


namespace koala_fiber_consumption_l1351_135111

theorem koala_fiber_consumption (x : ℝ) (h : 0.40 * x = 8) : x = 20 :=
sorry

end koala_fiber_consumption_l1351_135111


namespace percentage_of_whole_is_10_l1351_135124

def part : ℝ := 0.01
def whole : ℝ := 0.1

theorem percentage_of_whole_is_10 : (part / whole) * 100 = 10 := by
  sorry

end percentage_of_whole_is_10_l1351_135124


namespace janet_wait_time_l1351_135152

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l1351_135152


namespace solid_views_same_shape_and_size_l1351_135184

theorem solid_views_same_shape_and_size (solid : Type) (sphere triangular_pyramid cube cylinder : solid)
  (views_same_shape_and_size : solid → Bool) : 
  views_same_shape_and_size cylinder = false :=
sorry

end solid_views_same_shape_and_size_l1351_135184


namespace count_ordered_pairs_l1351_135149

theorem count_ordered_pairs : 
  ∃ n : ℕ, n = 136 ∧ 
  ∀ a b : ℝ, 
    (∃ x y : ℤ, a * x + b * y = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) → n = 136 := 
sorry

end count_ordered_pairs_l1351_135149


namespace greatest_divisor_arithmetic_sequence_sum_l1351_135153

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l1351_135153


namespace percentage_error_calc_l1351_135134

theorem percentage_error_calc (x : ℝ) (h : x ≠ 0) : 
  let correct_result := x * (5 / 3)
  let incorrect_result := x * (3 / 5)
  let percentage_error := (correct_result - incorrect_result) / correct_result * 100
  percentage_error = 64 := by
  sorry

end percentage_error_calc_l1351_135134


namespace central_angle_of_sector_l1351_135199

theorem central_angle_of_sector (r A : ℝ) (h₁ : r = 4) (h₂ : A = 4) :
  (1 / 2) * r^2 * (1 / 4) = A :=
by
  sorry

end central_angle_of_sector_l1351_135199


namespace neither_sufficient_nor_necessary_l1351_135145

-- Definitions based on given conditions
def propA (a b : ℕ) : Prop := a + b ≠ 4
def propB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem statement (proof not required)
theorem neither_sufficient_nor_necessary (a b : ℕ) :
  ¬ (propA a b → propB a b) ∧ ¬ (propB a b → propA a b) := 
sorry

end neither_sufficient_nor_necessary_l1351_135145


namespace find_number_l1351_135165

theorem find_number
  (x : ℝ)
  (h : 0.90 * x = 0.50 * 1080) :
  x = 600 :=
by
  sorry

end find_number_l1351_135165


namespace average_speed_for_trip_l1351_135181

theorem average_speed_for_trip (t₁ t₂ : ℝ) (v₁ v₂ : ℝ) (total_time : ℝ) 
  (h₁ : t₁ = 6) 
  (h₂ : v₁ = 30) 
  (h₃ : t₂ = 2) 
  (h₄ : v₂ = 46) 
  (h₅ : total_time = t₁ + t₂) 
  (h₆ : total_time = 8) :
  ((v₁ * t₁ + v₂ * t₂) / total_time) = 34 := 
  by 
    sorry

end average_speed_for_trip_l1351_135181


namespace regular_hexagon_area_l1351_135193

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l1351_135193


namespace last_two_digits_square_l1351_135132

theorem last_two_digits_square (n : ℕ) (hnz : (n % 10 ≠ 0) ∧ ((n ^ 2) % 100 = n % 10 * 11)): ((n ^ 2) % 100 = 44) :=
sorry

end last_two_digits_square_l1351_135132


namespace factor_2310_two_digit_numbers_l1351_135137

theorem factor_2310_two_digit_numbers :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2310 ∧ ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c * d = 2310 → (c = a ∧ d = b) ∨ (c = b ∧ d = a) :=
by {
  sorry
}

end factor_2310_two_digit_numbers_l1351_135137


namespace find_f_2009_l1351_135175

-- Defining the function f and specifying the conditions
variable (f : ℝ → ℝ)
axiom h1 : f 3 = -Real.sqrt 3
axiom h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x

-- Proving the desired statement
theorem find_f_2009 : f 2009 = 2 + Real.sqrt 3 :=
sorry

end find_f_2009_l1351_135175


namespace lemonade_water_l1351_135177

theorem lemonade_water (L S W : ℝ) (h1 : S = 1.5 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 18 :=
by
  sorry

end lemonade_water_l1351_135177


namespace average_speed_is_55_l1351_135118

theorem average_speed_is_55 
  (initial_reading : ℕ) (final_reading : ℕ) (time_hours : ℕ)
  (H1 : initial_reading = 15951) 
  (H2 : final_reading = 16061)
  (H3 : time_hours = 2) : 
  (final_reading - initial_reading) / time_hours = 55 :=
by
  sorry

end average_speed_is_55_l1351_135118


namespace chrom_replication_not_in_prophase_I_l1351_135140

-- Definitions for the conditions
def chrom_replication (stage : String) : Prop := 
  stage = "Interphase"

def chrom_shortening_thickening (stage : String) : Prop := 
  stage = "Prophase I"

def pairing_homologous_chromosomes (stage : String) : Prop := 
  stage = "Prophase I"

def crossing_over (stage : String) : Prop :=
  stage = "Prophase I"

-- Stating the theorem
theorem chrom_replication_not_in_prophase_I :
  chrom_replication "Interphase" ∧ 
  chrom_shortening_thickening "Prophase I" ∧ 
  pairing_homologous_chromosomes "Prophase I" ∧ 
  crossing_over "Prophase I" → 
  ¬ chrom_replication "Prophase I" := 
by
  sorry

end chrom_replication_not_in_prophase_I_l1351_135140


namespace cost_of_green_shirts_l1351_135128

noncomputable def total_cost_kindergarten : ℝ := 101 * 5.8
noncomputable def total_cost_first_grade : ℝ := 113 * 5
noncomputable def total_cost_second_grade : ℝ := 107 * 5.6
noncomputable def total_cost_all_but_third : ℝ := total_cost_kindergarten + total_cost_first_grade + total_cost_second_grade
noncomputable def total_third_grade : ℝ := 2317 - total_cost_all_but_third
noncomputable def cost_per_third_grade_shirt : ℝ := total_third_grade / 108

theorem cost_of_green_shirts : cost_per_third_grade_shirt = 5.25 := sorry

end cost_of_green_shirts_l1351_135128


namespace find_m_correct_l1351_135131

noncomputable def find_m (Q : Point) (B : List Point) (m : ℝ) : Prop :=
  let circle_area := 4 * Real.pi
  let radius := 2
  let area_sector_B1B2 := Real.pi / 3
  let area_region_B1B2 := 1 / 8
  let area_triangle_B1B2 := area_sector_B1B2 - area_region_B1B2 * circle_area
  let area_sector_B4B5 := Real.pi / 3
  let area_region_B4B5 := 1 / 10
  let area_triangle_B4B5 := area_sector_B4B5 - area_region_B4B5 * circle_area
  let area_sector_B9B10 := Real.pi / 3
  let area_region_B9B10 := 4 / 15 - Real.sqrt 3 / m
  let area_triangle_B9B10 := area_sector_B9B10 - area_region_B9B10 * circle_area
  m = 3

theorem find_m_correct (Q : Point) (B : List Point) : find_m Q B 3 :=
by
  unfold find_m
  sorry

end find_m_correct_l1351_135131


namespace num_integers_div_10_or_12_l1351_135176

-- Define the problem in Lean
theorem num_integers_div_10_or_12 (N : ℕ) : (1 ≤ N ∧ N ≤ 2007) ∧ (N % 10 = 0 ∨ N % 12 = 0) ↔ ∃ k, k = 334 := by
  sorry

end num_integers_div_10_or_12_l1351_135176


namespace mark_min_correct_problems_l1351_135168

noncomputable def mark_score (x : ℕ) : ℤ :=
  8 * x - 21

theorem mark_min_correct_problems (x : ℕ) :
  (4 * 2) + mark_score x ≥ 120 ↔ x ≥ 17 :=
by
  sorry

end mark_min_correct_problems_l1351_135168


namespace smallest_num_rectangles_l1351_135167

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l1351_135167


namespace Indians_drink_tea_is_zero_l1351_135198

-- Definitions based on given conditions and questions
variable (total_people : Nat)
variable (total_drink_tea : Nat)
variable (total_drink_coffee : Nat)
variable (answer_do_you_drink_coffee : Nat)
variable (answer_are_you_a_turk : Nat)
variable (answer_is_it_raining : Nat)
variable (Indians_drink_tea : Nat)
variable (Indians_drink_coffee : Nat)
variable (Turks_drink_coffee : Nat)
variable (Turks_drink_tea : Nat)

-- The given facts and conditions
axiom hx1 : total_people = 55
axiom hx2 : answer_do_you_drink_coffee = 44
axiom hx3 : answer_are_you_a_turk = 33
axiom hx4 : answer_is_it_raining = 22
axiom hx5 : Indians_drink_tea + Indians_drink_coffee + Turks_drink_coffee + Turks_drink_tea = total_people
axiom hx6 : Indians_drink_coffee + Turks_drink_coffee = answer_do_you_drink_coffee
axiom hx7 : Indians_drink_coffee + Turks_drink_tea = answer_are_you_a_turk
axiom hx8 : Indians_drink_tea + Turks_drink_coffee = answer_is_it_raining

-- Prove that the number of Indians drinking tea is 0
theorem Indians_drink_tea_is_zero : Indians_drink_tea = 0 :=
by {
    sorry
}

end Indians_drink_tea_is_zero_l1351_135198


namespace tangent_line_eqn_at_one_l1351_135183

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_eqn_at_one :
  let k := (Real.exp 1)
  let p := (1, Real.exp 1)
  ∃ m b : ℝ, (m = k) ∧ (b = p.2 - m * p.1) ∧ (∀ x, f x = y → y = m * x + b) :=
sorry

end tangent_line_eqn_at_one_l1351_135183


namespace find_number_l1351_135187

theorem find_number (x : ℤ) (h : 16 * x = 32) : x = 2 :=
sorry

end find_number_l1351_135187


namespace find_original_number_l1351_135119

variable (x : ℝ)

def tripled := 3 * x
def doubled := 2 * tripled
def subtracted := doubled - 9
def trebled := 3 * subtracted

theorem find_original_number (h : trebled = 90) : x = 6.5 := by
  sorry

end find_original_number_l1351_135119


namespace simplify_expression_l1351_135141

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 5)) + (1 / (1 - Real.sqrt 5)) = 1 := 
by sorry

end simplify_expression_l1351_135141


namespace quadratic_least_value_l1351_135150

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end quadratic_least_value_l1351_135150


namespace fraction_of_jenny_bounce_distance_l1351_135196

-- Definitions for the problem conditions
def jenny_initial_distance := 18
def jenny_bounce_fraction (f : ℚ) : ℚ := 18 * f
def jenny_total_distance (f : ℚ) : ℚ := jenny_initial_distance + jenny_bounce_fraction f

def mark_initial_distance := 15
def mark_bounce_distance := 2 * mark_initial_distance
def mark_total_distance : ℚ := mark_initial_distance + mark_bounce_distance

def distance_difference := 21

-- The theorem to prove
theorem fraction_of_jenny_bounce_distance (f : ℚ) :
  mark_total_distance = jenny_total_distance f + distance_difference →
  f = 1 / 3 :=
by
  sorry

end fraction_of_jenny_bounce_distance_l1351_135196


namespace dilation_complex_l1351_135110

theorem dilation_complex :
  let c := (1 : ℂ) - (2 : ℂ) * I
  let k := 3
  let z := -1 + I
  (k * (z - c) + c = -5 + 7 * I) :=
by
  sorry

end dilation_complex_l1351_135110


namespace find_numbers_between_1000_and_4000_l1351_135114

theorem find_numbers_between_1000_and_4000 :
  ∃ (x : ℤ), 1000 ≤ x ∧ x ≤ 4000 ∧
             (x % 11 = 2) ∧
             (x % 13 = 12) ∧
             (x % 19 = 18) ∧
             (x = 1234 ∨ x = 3951) :=
sorry

end find_numbers_between_1000_and_4000_l1351_135114


namespace min_sum_of_bases_l1351_135195

theorem min_sum_of_bases (a b : ℕ) (h : 3 * a + 5 = 4 * b + 2) : a + b = 13 :=
sorry

end min_sum_of_bases_l1351_135195


namespace solution_l1351_135147

def question (x : ℝ) : Prop := (x - 5) / ((x - 3) ^ 2) < 0

theorem solution :
  {x : ℝ | question x} = {x : ℝ | x < 3} ∪ {x : ℝ | 3 < x ∧ x < 5} :=
by {
  sorry
}

end solution_l1351_135147


namespace wrongly_entered_mark_l1351_135121

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ (correct_mark avg_increase pupils : ℝ), 
  correct_mark = 45 ∧ avg_increase = 0.5 ∧ pupils = 80 ∧
  (avg_increase * pupils = (x - correct_mark)) →
  x = 85) :=
by 
  intro correct_mark avg_increase pupils
  rintro ⟨hc, ha, hp, h⟩
  sorry

end wrongly_entered_mark_l1351_135121


namespace inequality_relations_l1351_135159

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 125 ^ (1 / 6)
noncomputable def c : ℝ := Real.log 7 / Real.log (1 / 6)

theorem inequality_relations :
  c < a ∧ a < b := 
by 
  sorry

end inequality_relations_l1351_135159


namespace number_of_teams_l1351_135126

theorem number_of_teams (n : ℕ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → (games_played : ℕ) = 4) 
  (h2 : ∀ (i j : ℕ), i ≠ j → (count : ℕ) = 760) : 
  n = 20 := 
by 
  sorry

end number_of_teams_l1351_135126


namespace speed_of_second_part_of_trip_l1351_135155

-- Given conditions
def total_distance : Real := 50
def first_part_distance : Real := 25
def first_part_speed : Real := 66
def average_speed : Real := 44.00000000000001

-- The statement we want to prove
theorem speed_of_second_part_of_trip :
  ∃ second_part_speed : Real, second_part_speed = 33 :=
by
  sorry

end speed_of_second_part_of_trip_l1351_135155


namespace sqrt3_f_pi6_lt_f_pi3_l1351_135101

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_derivative_tan_lt (x : ℝ) (h : 0 < x ∧ x < π / 2) : f x < (deriv f x) * tan x

theorem sqrt3_f_pi6_lt_f_pi3 :
  sqrt 3 * f (π / 6) < f (π / 3) :=
by
  sorry

end sqrt3_f_pi6_lt_f_pi3_l1351_135101


namespace smith_trip_times_same_l1351_135172

theorem smith_trip_times_same (v : ℝ) (hv : v > 0) : 
  let t1 := 80 / v 
  let t2 := 160 / (2 * v) 
  t1 = t2 :=
by
  sorry

end smith_trip_times_same_l1351_135172


namespace find_a_cubed_minus_b_cubed_l1351_135100

theorem find_a_cubed_minus_b_cubed (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) : a^3 - b^3 = 486 := 
by 
  sorry

end find_a_cubed_minus_b_cubed_l1351_135100


namespace percent_brandA_in_mix_l1351_135191

theorem percent_brandA_in_mix (x : Real) :
  (0.60 * x + 0.35 * (100 - x) = 50) → x = 60 :=
by
  intro h
  sorry

end percent_brandA_in_mix_l1351_135191
