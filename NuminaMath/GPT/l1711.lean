import Mathlib

namespace NUMINAMATH_GPT_cos_17_pi_over_6_l1711_171127

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * 180 / Real.pi

theorem cos_17_pi_over_6 : Real.cos (17 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_cos_17_pi_over_6_l1711_171127


namespace NUMINAMATH_GPT_train_speed_correct_l1711_171134

def train_length : ℝ := 100
def crossing_time : ℝ := 12
def expected_speed : ℝ := 8.33

theorem train_speed_correct : (train_length / crossing_time) = expected_speed :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1711_171134


namespace NUMINAMATH_GPT_division_pow_zero_l1711_171198

theorem division_pow_zero (a b : ℝ) (hb : b ≠ 0) : ((a / b) ^ 0 = (1 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_division_pow_zero_l1711_171198


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1711_171106

theorem ellipse_eccentricity (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2) : (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1711_171106


namespace NUMINAMATH_GPT_sufficient_y_wages_l1711_171107

noncomputable def days_sufficient_for_y_wages (Wx Wy : ℝ) (total_money : ℝ) : ℝ :=
  total_money / Wy

theorem sufficient_y_wages
  (Wx Wy : ℝ)
  (H1 : ∀(D : ℝ), total_money = D * Wx → D = 36 )
  (H2 : total_money = 20 * (Wx + Wy)) :
  days_sufficient_for_y_wages Wx Wy total_money = 45 := by
  sorry

end NUMINAMATH_GPT_sufficient_y_wages_l1711_171107


namespace NUMINAMATH_GPT_find_a_l1711_171141

def F (a b c : ℤ) : ℤ := a * b^2 + c

theorem find_a (a : ℤ) (h : F a 3 (-1) = F a 5 (-3)) : a = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_find_a_l1711_171141


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1711_171150

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : 2 * (X^2) - 8 * X + 6 = 0) : 
  (-b / a) = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1711_171150


namespace NUMINAMATH_GPT_books_difference_l1711_171136

theorem books_difference (maddie_books luisa_books amy_books total_books : ℕ) 
  (h1 : maddie_books = 15) 
  (h2 : luisa_books = 18) 
  (h3 : amy_books = 6) 
  (h4 : total_books = amy_books + luisa_books) :
  total_books - maddie_books = 9 := 
sorry

end NUMINAMATH_GPT_books_difference_l1711_171136


namespace NUMINAMATH_GPT_reduce_repeating_decimal_l1711_171103

noncomputable def repeating_decimal_to_fraction (a : ℚ) (n : ℕ) : ℚ :=
  a + (n / 99)

theorem reduce_repeating_decimal : repeating_decimal_to_fraction 2 7 = 205 / 99 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_reduce_repeating_decimal_l1711_171103


namespace NUMINAMATH_GPT_no_valid_road_network_l1711_171170

theorem no_valid_road_network
  (k_A k_B k_C : ℕ)
  (h_kA : k_A ≥ 2)
  (h_kB : k_B ≥ 2)
  (h_kC : k_C ≥ 2) :
  ¬ ∃ (t : ℕ) (d : ℕ → ℕ), t ≥ 7 ∧ 
    (∀ i j, i ≠ j → d i ≠ d j) ∧
    (∀ i, i < 4 * (k_A + k_B + k_C) + 4 → d i = i + 1) :=
sorry

end NUMINAMATH_GPT_no_valid_road_network_l1711_171170


namespace NUMINAMATH_GPT_delete_middle_divides_l1711_171105

def digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := n / 10000
  let b := (n % 10000) / 1000
  let c := (n % 1000) / 100
  let d := (n % 100) / 10
  let e := n % 10
  (a, b, c, d, e)

def delete_middle_digit (n : ℕ) : ℕ :=
  let (a, b, c, d, e) := digits n
  1000 * a + 100 * b + 10 * d + e

theorem delete_middle_divides (n : ℕ) (hn : 10000 ≤ n ∧ n < 100000) :
  (delete_middle_digit n) ∣ n :=
sorry

end NUMINAMATH_GPT_delete_middle_divides_l1711_171105


namespace NUMINAMATH_GPT_correct_result_l1711_171157

theorem correct_result (x : ℕ) (h: (325 - x) * 5 = 1500) : 325 - x * 5 = 200 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_correct_result_l1711_171157


namespace NUMINAMATH_GPT_least_pos_int_satisfies_conditions_l1711_171115

theorem least_pos_int_satisfies_conditions :
  ∃ x : ℕ, x > 0 ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  x = 419 :=
by
  sorry

end NUMINAMATH_GPT_least_pos_int_satisfies_conditions_l1711_171115


namespace NUMINAMATH_GPT_pow137_mod8_l1711_171153

theorem pow137_mod8 : (5 ^ 137) % 8 = 5 := by
  -- Use the provided conditions
  have h1: 5 % 8 = 5 := by norm_num
  have h2: (5 ^ 2) % 8 = 1 := by norm_num
  sorry

end NUMINAMATH_GPT_pow137_mod8_l1711_171153


namespace NUMINAMATH_GPT_mean_tasks_b_l1711_171174

variable (a b : ℕ)
variable (m_a m_b : ℕ)
variable (h1 : a + b = 260)
variable (h2 : a = 3 * b / 10 + b)
variable (h3 : m_a = 80)
variable (h4 : m_b = 12 * m_a / 10)

theorem mean_tasks_b :
  m_b = 96 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_mean_tasks_b_l1711_171174


namespace NUMINAMATH_GPT_prime_factorization_of_expression_l1711_171152

theorem prime_factorization_of_expression (p n : ℕ) (hp : Nat.Prime p) (hdiv : p^2 ∣ 2^(p-1) - 1) : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) ∧ 
  a ∣ (p-1) ∧ b ∣ (p! + 2^n) ∧ c ∣ (p! + 2^n) := 
sorry

end NUMINAMATH_GPT_prime_factorization_of_expression_l1711_171152


namespace NUMINAMATH_GPT_complex_division_example_l1711_171147

theorem complex_division_example : (2 : ℂ) / (I * (3 - I)) = (1 - 3 * I) / 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_complex_division_example_l1711_171147


namespace NUMINAMATH_GPT_dave_winfield_home_runs_correct_l1711_171167

def dave_winfield_home_runs (W : ℕ) : Prop :=
  755 = 2 * W - 175

theorem dave_winfield_home_runs_correct : dave_winfield_home_runs 465 :=
by
  -- The proof is omitted as requested
  sorry

end NUMINAMATH_GPT_dave_winfield_home_runs_correct_l1711_171167


namespace NUMINAMATH_GPT_male_listeners_l1711_171171

structure Survey :=
  (males_dont_listen : Nat)
  (females_listen : Nat)
  (total_listeners : Nat)
  (total_dont_listen : Nat)

def number_of_females_dont_listen (s : Survey) : Nat :=
  s.total_dont_listen - s.males_dont_listen

def number_of_males_listen (s : Survey) : Nat :=
  s.total_listeners - s.females_listen

theorem male_listeners (s : Survey) (h : s = { males_dont_listen := 85, females_listen := 75, total_listeners := 180, total_dont_listen := 160 }) :
  number_of_males_listen s = 105 :=
by
  sorry

end NUMINAMATH_GPT_male_listeners_l1711_171171


namespace NUMINAMATH_GPT_transformation_power_of_two_l1711_171145

theorem transformation_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ s : ℕ, 2 ^ s ≥ n :=
by sorry

end NUMINAMATH_GPT_transformation_power_of_two_l1711_171145


namespace NUMINAMATH_GPT_cards_probability_ratio_l1711_171135

theorem cards_probability_ratio :
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  q / p = 44 :=
by
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  have : q / p = 44 := sorry
  exact this

end NUMINAMATH_GPT_cards_probability_ratio_l1711_171135


namespace NUMINAMATH_GPT_power_sum_l1711_171160

theorem power_sum : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end NUMINAMATH_GPT_power_sum_l1711_171160


namespace NUMINAMATH_GPT_proof_value_of_expression_l1711_171120

theorem proof_value_of_expression (a b c d m : ℝ) 
  (h1: a + b = 0)
  (h2: c * d = 1)
  (h3: |m| = 4) : 
  m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end NUMINAMATH_GPT_proof_value_of_expression_l1711_171120


namespace NUMINAMATH_GPT_B_completes_work_in_n_days_l1711_171114

-- Define the conditions
def can_complete_work_A_in_d_days (d : ℕ) : Prop := d = 15
def fraction_of_work_left_after_working_together (t : ℕ) (fraction : ℝ) : Prop :=
  t = 5 ∧ fraction = 0.41666666666666663

-- Define the theorem to be proven
theorem B_completes_work_in_n_days (d t : ℕ) (fraction : ℝ) (x : ℕ) 
  (hA : can_complete_work_A_in_d_days d) 
  (hB : fraction_of_work_left_after_working_together t fraction) : x = 20 :=
sorry

end NUMINAMATH_GPT_B_completes_work_in_n_days_l1711_171114


namespace NUMINAMATH_GPT_fraction_of_number_l1711_171186

theorem fraction_of_number (x f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_number_l1711_171186


namespace NUMINAMATH_GPT_scientific_notation_650000_l1711_171182

theorem scientific_notation_650000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 650000 = a * 10 ^ n ∧ a = 6.5 ∧ n = 5 :=
  sorry

end NUMINAMATH_GPT_scientific_notation_650000_l1711_171182


namespace NUMINAMATH_GPT_owen_turtles_l1711_171137

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end NUMINAMATH_GPT_owen_turtles_l1711_171137


namespace NUMINAMATH_GPT_equal_distribution_l1711_171158

theorem equal_distribution (total_cookies bags : ℕ) (h_total : total_cookies = 14) (h_bags : bags = 7) : total_cookies / bags = 2 := by
  sorry

end NUMINAMATH_GPT_equal_distribution_l1711_171158


namespace NUMINAMATH_GPT_annika_total_east_hike_distance_l1711_171184

def annika_flat_rate : ℝ := 10 -- minutes per kilometer on flat terrain
def annika_initial_distance : ℝ := 2.75 -- kilometers already hiked east
def total_time : ℝ := 45 -- minutes
def uphill_rate : ℝ := 15 -- minutes per kilometer on uphill
def downhill_rate : ℝ := 5 -- minutes per kilometer on downhill
def uphill_distance : ℝ := 0.5 -- kilometer of uphill section
def downhill_distance : ℝ := 0.5 -- kilometer of downhill section

theorem annika_total_east_hike_distance :
  let total_uphill_time := uphill_distance * uphill_rate
  let total_downhill_time := downhill_distance * downhill_rate
  let time_for_uphill_and_downhill := total_uphill_time + total_downhill_time
  let time_available_for_outward_hike := total_time / 2
  let remaining_time_after_up_down := time_available_for_outward_hike - time_for_uphill_and_downhill
  let additional_flat_distance := remaining_time_after_up_down / annika_flat_rate
  (annika_initial_distance + additional_flat_distance) = 4 :=
by
  sorry

end NUMINAMATH_GPT_annika_total_east_hike_distance_l1711_171184


namespace NUMINAMATH_GPT_sqrt_pow_simplification_l1711_171187

theorem sqrt_pow_simplification :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * (5 ^ (3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_pow_simplification_l1711_171187


namespace NUMINAMATH_GPT_subtraction_of_negatives_l1711_171131

theorem subtraction_of_negatives :
  -2 - (-3) = 1 := 
by
  sorry

end NUMINAMATH_GPT_subtraction_of_negatives_l1711_171131


namespace NUMINAMATH_GPT_elijah_total_cards_l1711_171196

-- Define the conditions
def num_decks : ℕ := 6
def cards_per_deck : ℕ := 52

-- The main statement that we need to prove
theorem elijah_total_cards : num_decks * cards_per_deck = 312 := by
  -- We skip the proof
  sorry

end NUMINAMATH_GPT_elijah_total_cards_l1711_171196


namespace NUMINAMATH_GPT_min_n_constant_term_exists_l1711_171122

theorem min_n_constant_term_exists (n : ℕ) (h : 0 < n) :
  (∃ r : ℕ, (2 * n = 3 * r) ∧ n > 0) ↔ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_n_constant_term_exists_l1711_171122


namespace NUMINAMATH_GPT_remainder_of_product_of_odd_primes_mod_32_l1711_171156

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end NUMINAMATH_GPT_remainder_of_product_of_odd_primes_mod_32_l1711_171156


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1711_171142

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1711_171142


namespace NUMINAMATH_GPT_seeds_per_can_l1711_171154

theorem seeds_per_can (total_seeds : ℕ) (num_cans : ℕ) (h1 : total_seeds = 54) (h2 : num_cans = 9) : total_seeds / num_cans = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_seeds_per_can_l1711_171154


namespace NUMINAMATH_GPT_find_actual_weights_l1711_171117

noncomputable def melon_weight : ℝ := 4.5
noncomputable def watermelon_weight : ℝ := 3.5
noncomputable def scale_error : ℝ := 0.5

def weight_bounds (actual_weight measured_weight error_margin : ℝ) :=
  (measured_weight - error_margin ≤ actual_weight) ∧ (actual_weight ≤ measured_weight + error_margin)

theorem find_actual_weights (x y : ℝ) 
  (melon_measured : x = 4)
  (watermelon_measured : y = 3)
  (combined_measured : x + y = 8.5)
  (hx : weight_bounds melon_weight x scale_error)
  (hy : weight_bounds watermelon_weight y scale_error)
  (h_combined : weight_bounds (melon_weight + watermelon_weight) (x + y) (2 * scale_error)) :
  x = melon_weight ∧ y = watermelon_weight := 
sorry

end NUMINAMATH_GPT_find_actual_weights_l1711_171117


namespace NUMINAMATH_GPT_exists_multiple_of_power_of_two_non_zero_digits_l1711_171177

open Nat

theorem exists_multiple_of_power_of_two_non_zero_digits (k : ℕ) (h : 0 < k) : 
  ∃ m : ℕ, (2^k ∣ m) ∧ (∀ d ∈ digits 10 m, d ≠ 0) :=
sorry

end NUMINAMATH_GPT_exists_multiple_of_power_of_two_non_zero_digits_l1711_171177


namespace NUMINAMATH_GPT_bones_in_beef_l1711_171102

def price_of_beef_with_bones : ℝ := 78
def price_of_boneless_beef : ℝ := 90
def price_of_bones : ℝ := 15
def fraction_of_bones_in_kg : ℝ := 0.16
def grams_per_kg : ℝ := 1000

theorem bones_in_beef :
  (fraction_of_bones_in_kg * grams_per_kg = 160) :=
by
  sorry

end NUMINAMATH_GPT_bones_in_beef_l1711_171102


namespace NUMINAMATH_GPT_anne_initial_sweettarts_l1711_171199

variable (x : ℕ)
variable (num_friends : ℕ := 3)
variable (sweettarts_per_friend : ℕ := 5)
variable (total_sweettarts_given : ℕ := num_friends * sweettarts_per_friend)

theorem anne_initial_sweettarts 
  (h1 : ∀ person, person < num_friends → sweettarts_per_friend = 5)
  (h2 : total_sweettarts_given = 15) : 
  total_sweettarts_given = 15 := 
by 
  sorry

end NUMINAMATH_GPT_anne_initial_sweettarts_l1711_171199


namespace NUMINAMATH_GPT_fg_of_3_eq_neg5_l1711_171100

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Lean statement to prove question == answer
theorem fg_of_3_eq_neg5 : f (g 3) = -5 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_neg5_l1711_171100


namespace NUMINAMATH_GPT_box_height_correct_l1711_171125

noncomputable def box_height : ℕ :=
  8

theorem box_height_correct (box_width box_length block_height block_width block_length : ℕ) (num_blocks : ℕ) :
  box_width = 10 ∧
  box_length = 12 ∧
  block_height = 3 ∧
  block_width = 2 ∧
  block_length = 4 ∧
  num_blocks = 40 →
  (num_blocks * block_height * block_width * block_length) /
  (box_width * box_length) = box_height :=
  by
  sorry

end NUMINAMATH_GPT_box_height_correct_l1711_171125


namespace NUMINAMATH_GPT_eggs_left_in_jar_l1711_171130

variable (initial_eggs : ℝ) (removed_eggs : ℝ)

theorem eggs_left_in_jar (h1 : initial_eggs = 35.3) (h2 : removed_eggs = 4.5) :
  initial_eggs - removed_eggs = 30.8 :=
by
  sorry

end NUMINAMATH_GPT_eggs_left_in_jar_l1711_171130


namespace NUMINAMATH_GPT_part1_part2_part3_l1711_171189

variable {x : ℝ}

def A := {x : ℝ | x^2 + 3 * x - 4 > 0}
def B := {x : ℝ | x^2 - x - 6 < 0}
def C_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem part1 : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 3} := sorry

theorem part2 : (C_R (A ∩ B)) = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := sorry

theorem part3 : (A ∪ (C_R B)) = {x : ℝ | x ≤ -2 ∨ x > 1} := sorry

end NUMINAMATH_GPT_part1_part2_part3_l1711_171189


namespace NUMINAMATH_GPT_solution_to_inequality_l1711_171151

-- Define the combination function C(n, k)
def combination (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the permutation function A(n, k)
def permutation (n k : ℕ) : ℕ :=
  n.factorial / (n - k).factorial

-- State the final theorem
theorem solution_to_inequality : 
  ∀ x : ℕ, (combination 5 x + permutation x 3 < 30) ↔ (x = 3 ∨ x = 4) :=
by
  -- The actual proof is not required as per the instructions
  sorry

end NUMINAMATH_GPT_solution_to_inequality_l1711_171151


namespace NUMINAMATH_GPT_nat_perfect_square_l1711_171146

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end NUMINAMATH_GPT_nat_perfect_square_l1711_171146


namespace NUMINAMATH_GPT_find_x_l1711_171193

theorem find_x (x : ℝ) (h : 1 - 1 / (1 - x) = 1 / (1 - x)) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1711_171193


namespace NUMINAMATH_GPT_race_course_length_proof_l1711_171194

def race_course_length (L : ℝ) (v_A v_B : ℝ) : Prop :=
  v_A = 4 * v_B ∧ (L / v_A = (L - 66) / v_B) → L = 88

theorem race_course_length_proof (v_A v_B : ℝ) : race_course_length 88 v_A v_B :=
by 
  intros
  sorry

end NUMINAMATH_GPT_race_course_length_proof_l1711_171194


namespace NUMINAMATH_GPT_min_sum_reciprocals_of_roots_l1711_171166

theorem min_sum_reciprocals_of_roots (k : ℝ) 
  (h_roots_positive : ∀ x : ℝ, (x^2 - k * x + k + 3 = 0) → 0 < x) :
  (k ≥ 6) → 
  ∀ x1 x2 : ℝ, (x1*x2 = k + 3) ∧ (x1 + x2 = k) ∧ (x1 > 0) ∧ (x2 > 0) → 
  (1 / x1 + 1 / x2) = 2 / 3 :=
by 
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_min_sum_reciprocals_of_roots_l1711_171166


namespace NUMINAMATH_GPT_angle_E_measure_l1711_171118

-- Definition of degrees for each angle in the quadrilateral
def angle_measure (E F G H : ℝ) : Prop :=
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360

-- Prove the measure of angle E
theorem angle_E_measure (E F G H : ℝ) (h : angle_measure E F G H) : E = 360 * (4 / 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_E_measure_l1711_171118


namespace NUMINAMATH_GPT_find_point_on_parabola_l1711_171108

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def positive_y (y : ℝ) : Prop := y > 0
def distance_to_focus (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = (5/2)^2 

theorem find_point_on_parabola (x y : ℝ) :
  parabola x y ∧ positive_y y ∧ distance_to_focus x y → (x = 1 ∧ y = Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_find_point_on_parabola_l1711_171108


namespace NUMINAMATH_GPT_conic_sections_l1711_171165

theorem conic_sections (x y : ℝ) (h : y^4 - 6 * x^4 = 3 * y^2 - 2) :
  (∃ a b : ℝ, y^2 = a + b * x^2) ∨ (∃ c d : ℝ, y^2 = c - d * x^2) :=
sorry

end NUMINAMATH_GPT_conic_sections_l1711_171165


namespace NUMINAMATH_GPT_total_food_per_day_l1711_171148

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_food_per_day_l1711_171148


namespace NUMINAMATH_GPT_trig_identity_l1711_171140

noncomputable def trig_expr := 
  4.34 * (Real.cos (28 * Real.pi / 180) * Real.cos (56 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) + 
  (Real.cos (2 * Real.pi / 180) * Real.cos (4 * Real.pi / 180) / Real.sin (28 * Real.pi / 180))

theorem trig_identity : 
  trig_expr = (Real.sqrt 3 * Real.sin (38 * Real.pi / 180)) / (4 * Real.sin (2 * Real.pi / 180) * Real.sin (28 * Real.pi / 180)) :=
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l1711_171140


namespace NUMINAMATH_GPT_value_of_z_l1711_171143

theorem value_of_z (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x * y - 9) : z = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_z_l1711_171143


namespace NUMINAMATH_GPT_children_count_l1711_171132

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end NUMINAMATH_GPT_children_count_l1711_171132


namespace NUMINAMATH_GPT_find_ending_number_l1711_171192

theorem find_ending_number (N : ℕ) :
  (∃ k : ℕ, N = 3 * k) ∧ (∀ x,  40 < x ∧ x ≤ N → x % 3 = 0) ∧ (∃ avg, avg = (N + 42) / 2 ∧ avg = 60) → N = 78 :=
by
  sorry

end NUMINAMATH_GPT_find_ending_number_l1711_171192


namespace NUMINAMATH_GPT_total_people_at_fair_l1711_171168

theorem total_people_at_fair (num_children : ℕ) (num_adults : ℕ) 
  (children_attended : num_children = 700) 
  (adults_attended : num_adults = 1500) : 
  num_children + num_adults = 2200 := by
  sorry

end NUMINAMATH_GPT_total_people_at_fair_l1711_171168


namespace NUMINAMATH_GPT_crease_length_l1711_171176

theorem crease_length (A B C : ℝ) (h1 : A = 5) (h2 : B = 12) (h3 : C = 13) : ∃ D, D = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_crease_length_l1711_171176


namespace NUMINAMATH_GPT_problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l1711_171164

variable (a b : ℝ)

theorem problem_statement_part1 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a + 2 / b) ≥ 9 := sorry

theorem problem_statement_part2 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (2 ^ a + 4 ^ b) ≥ 2 * Real.sqrt 2 := sorry

theorem problem_statement_part3 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a * b) ≤ (1 / 8) := sorry

theorem problem_statement_part4 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2) ≥ (1 / 5) := sorry

end NUMINAMATH_GPT_problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l1711_171164


namespace NUMINAMATH_GPT_horizontal_asymptote_l1711_171128

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 7 * x^3 + 10 * x^2 + 6 * x + 4) / (4 * x^4 + 3 * x^3 + 9 * x^2 + 4 * x + 2)

theorem horizontal_asymptote :
  ∃ L : ℝ, (∀ ε > 0, ∃ M > 0, ∀ x > M, |rational_function x - L| < ε) → L = 15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_horizontal_asymptote_l1711_171128


namespace NUMINAMATH_GPT_least_three_digit_multiple_of_3_4_5_l1711_171161

def is_multiple_of (a b : ℕ) : Prop := b % a = 0

theorem least_three_digit_multiple_of_3_4_5 : 
  ∃ n : ℕ, is_multiple_of 3 n ∧ is_multiple_of 4 n ∧ is_multiple_of 5 n ∧ 100 ≤ n ∧ n < 1000 ∧ (∀ m : ℕ, is_multiple_of 3 m ∧ is_multiple_of 4 m ∧ is_multiple_of 5 m ∧ 100 ≤ m ∧ m < 1000 → n ≤ m) ∧ n = 120 :=
by
  sorry

end NUMINAMATH_GPT_least_three_digit_multiple_of_3_4_5_l1711_171161


namespace NUMINAMATH_GPT_ratio_of_democrats_l1711_171119

variable (F M D_F D_M : ℕ)

theorem ratio_of_democrats (h1 : F + M = 750)
    (h2 : D_F = 1 / 2 * F)
    (h3 : D_F = 125)
    (h4 : D_M = 1 / 4 * M) :
    (D_F + D_M) / 750 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_democrats_l1711_171119


namespace NUMINAMATH_GPT_part1_solution_set_l1711_171123

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part1_solution_set : {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_GPT_part1_solution_set_l1711_171123


namespace NUMINAMATH_GPT_units_produced_today_eq_90_l1711_171159

-- Define the average production and number of past days
def average_past_production (n : ℕ) (past_avg : ℕ) : ℕ :=
  n * past_avg

def average_total_production (n : ℕ) (current_avg : ℕ) : ℕ :=
  (n + 1) * current_avg

def units_produced_today (n : ℕ) (past_avg : ℕ) (current_avg : ℕ) : ℕ :=
  average_total_production n current_avg - average_past_production n past_avg

-- Given conditions
def n := 5
def past_avg := 60
def current_avg := 65

-- Statement to prove
theorem units_produced_today_eq_90 : units_produced_today n past_avg current_avg = 90 :=
by
  -- Declare which parts need proving
  sorry

end NUMINAMATH_GPT_units_produced_today_eq_90_l1711_171159


namespace NUMINAMATH_GPT_simplify_expression_l1711_171124

variable (a : ℝ)

theorem simplify_expression (h₁ : a ≠ -3) (h₂ : a ≠ 1) :
  (1 - 4/(a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1711_171124


namespace NUMINAMATH_GPT_triangle_inequality_a2_lt_ab_ac_l1711_171190

theorem triangle_inequality_a2_lt_ab_ac {a b c : ℝ} (h1 : a < b + c) (h2 : 0 < a) : a^2 < a * b + a * c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_a2_lt_ab_ac_l1711_171190


namespace NUMINAMATH_GPT_find_density_of_gold_l1711_171163

theorem find_density_of_gold
  (side_length : ℝ)
  (gold_cost_per_gram : ℝ)
  (sale_factor : ℝ)
  (profit : ℝ)
  (density_of_gold : ℝ) :
  side_length = 6 →
  gold_cost_per_gram = 60 →
  sale_factor = 1.5 →
  profit = 123120 →
  density_of_gold = 19 :=
sorry

end NUMINAMATH_GPT_find_density_of_gold_l1711_171163


namespace NUMINAMATH_GPT_sum_of_integers_between_neg20_5_and_10_5_l1711_171179

noncomputable def sum_arithmetic_series (a l n : ℤ) : ℤ :=
  n * (a + l) / 2

theorem sum_of_integers_between_neg20_5_and_10_5 :
  (sum_arithmetic_series (-20) 10 31) = -155 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_between_neg20_5_and_10_5_l1711_171179


namespace NUMINAMATH_GPT_find_a_value_l1711_171191

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l1711_171191


namespace NUMINAMATH_GPT_diving_classes_on_weekdays_l1711_171197

theorem diving_classes_on_weekdays 
  (x : ℕ) 
  (weekend_classes_per_day : ℕ := 4)
  (people_per_class : ℕ := 5)
  (total_people_3_weeks : ℕ := 270)
  (weekend_days : ℕ := 2)
  (total_weeks : ℕ := 3)
  (weekend_total_classes : ℕ := weekend_classes_per_day * weekend_days * total_weeks) 
  (total_people_weekends : ℕ := weekend_total_classes * people_per_class) 
  (total_people_weekdays : ℕ := total_people_3_weeks - total_people_weekends)
  (weekday_classes_needed : ℕ := total_people_weekdays / people_per_class)
  (weekly_weekday_classes : ℕ := weekday_classes_needed / total_weeks)
  (h : weekly_weekday_classes = x)
  : x = 10 := sorry

end NUMINAMATH_GPT_diving_classes_on_weekdays_l1711_171197


namespace NUMINAMATH_GPT_non_officers_count_l1711_171112

theorem non_officers_count 
    (avg_salary_employees : ℝ) 
    (avg_salary_officers : ℝ) 
    (avg_salary_non_officers : ℝ) 
    (num_officers : ℕ) : 
    avg_salary_employees = 120 ∧ avg_salary_officers = 470 ∧ avg_salary_non_officers = 110 ∧ num_officers = 15 → 
    ∃ N : ℕ, N = 525 ∧ 
    (num_officers * avg_salary_officers + N * avg_salary_non_officers) / (num_officers + N) = avg_salary_employees := 
by 
    sorry

end NUMINAMATH_GPT_non_officers_count_l1711_171112


namespace NUMINAMATH_GPT_braden_total_amount_after_winning_l1711_171195

noncomputable def initial_amount := 400
noncomputable def multiplier := 2

def total_amount_after_winning (initial: ℕ) (mult: ℕ) : ℕ := initial + (mult * initial)

theorem braden_total_amount_after_winning : total_amount_after_winning initial_amount multiplier = 1200 := by
  sorry

end NUMINAMATH_GPT_braden_total_amount_after_winning_l1711_171195


namespace NUMINAMATH_GPT_greatest_ab_sum_l1711_171155

theorem greatest_ab_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) :
  a + b = Real.sqrt 220 ∨ a + b = -Real.sqrt 220 :=
sorry

end NUMINAMATH_GPT_greatest_ab_sum_l1711_171155


namespace NUMINAMATH_GPT_meaningful_expression_iff_l1711_171185

theorem meaningful_expression_iff (x : ℝ) : (∃ y, y = (2 : ℝ) / (2*x - 1)) ↔ x ≠ (1 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_iff_l1711_171185


namespace NUMINAMATH_GPT_area_ratio_of_circles_l1711_171149

-- Define the circles and lengths of arcs
variables {R_C R_D : ℝ} (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D))

-- Theorem proving the ratio of the areas
theorem area_ratio_of_circles (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 := sorry

end NUMINAMATH_GPT_area_ratio_of_circles_l1711_171149


namespace NUMINAMATH_GPT_dice_probability_sum_18_l1711_171110

theorem dice_probability_sum_18 : 
  (∃ d1 d2 d3 : ℕ, 1 ≤ d1 ∧ d1 ≤ 8 ∧ 1 ≤ d2 ∧ d2 ≤ 8 ∧ 1 ≤ d3 ∧ d3 ≤ 8 ∧ d1 + d2 + d3 = 18) →
  (1/8 : ℚ) * (1/8) * (1/8) * 9 = 9 / 512 :=
by 
  sorry

end NUMINAMATH_GPT_dice_probability_sum_18_l1711_171110


namespace NUMINAMATH_GPT_prod_mod7_eq_zero_l1711_171111

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_prod_mod7_eq_zero_l1711_171111


namespace NUMINAMATH_GPT_translate_triangle_l1711_171169

theorem translate_triangle (A B C A' : (ℝ × ℝ)) (hx_A : A = (2, 1)) (hx_B : B = (4, 3)) 
  (hx_C : C = (0, 2)) (hx_A' : A' = (-1, 5)) : 
  ∃ C' : (ℝ × ℝ), C' = (-3, 6) :=
by 
  sorry

end NUMINAMATH_GPT_translate_triangle_l1711_171169


namespace NUMINAMATH_GPT_number_of_dimes_paid_l1711_171181

theorem number_of_dimes_paid (cost_in_dollars : ℕ) (value_of_dime_in_cents : ℕ) (value_of_dollar_in_cents : ℕ) 
  (h_cost : cost_in_dollars = 9) (h_dime : value_of_dime_in_cents = 10) (h_dollar : value_of_dollar_in_cents = 100) : 
  (cost_in_dollars * value_of_dollar_in_cents) / value_of_dime_in_cents = 90 := by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_number_of_dimes_paid_l1711_171181


namespace NUMINAMATH_GPT_triangles_from_decagon_l1711_171183

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end NUMINAMATH_GPT_triangles_from_decagon_l1711_171183


namespace NUMINAMATH_GPT_target_avg_weekly_income_l1711_171116

-- Define the weekly incomes for the past 5 weeks
def past_incomes : List ℤ := [406, 413, 420, 436, 395]

-- Define the average income over the next 2 weeks
def avg_income_next_two_weeks : ℤ := 365

-- Define the target average weekly income over the 7-week period
theorem target_avg_weekly_income : 
  ((past_incomes.sum + 2 * avg_income_next_two_weeks) / 7 = 400) :=
sorry

end NUMINAMATH_GPT_target_avg_weekly_income_l1711_171116


namespace NUMINAMATH_GPT_programs_produce_same_result_l1711_171139

-- Define Program A's computation
def programA_sum : ℕ := (List.range (1000 + 1)).sum -- Sum of numbers from 0 to 1000

-- Define Program B's computation
def programB_sum : ℕ := (List.range (1000 + 1)).reverse.sum -- Sum of numbers from 1000 down to 0

theorem programs_produce_same_result : programA_sum = programB_sum :=
  sorry

end NUMINAMATH_GPT_programs_produce_same_result_l1711_171139


namespace NUMINAMATH_GPT_roots_of_quadratic_l1711_171173

open Real

theorem roots_of_quadratic (r s : ℝ) (h1 : r + s = 2 * sqrt 3) (h2 : r * s = 2) :
  r^6 + s^6 = 3104 :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1711_171173


namespace NUMINAMATH_GPT_john_minimum_pizzas_l1711_171104

theorem john_minimum_pizzas (car_cost bag_cost earnings_per_pizza gas_cost p : ℕ) 
  (h_car : car_cost = 6000)
  (h_bag : bag_cost = 200)
  (h_earnings : earnings_per_pizza = 12)
  (h_gas : gas_cost = 4)
  (h_p : 8 * p >= car_cost + bag_cost) : p >= 775 := 
sorry

end NUMINAMATH_GPT_john_minimum_pizzas_l1711_171104


namespace NUMINAMATH_GPT_volunteers_distribution_l1711_171138

theorem volunteers_distribution:
  let num_volunteers := 5
  let group_distribution := (2, 2, 1)
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end NUMINAMATH_GPT_volunteers_distribution_l1711_171138


namespace NUMINAMATH_GPT_quadratic_equation_only_option_B_l1711_171113

theorem quadratic_equation_only_option_B (a b c : ℝ) (x : ℝ):
  (a ≠ 0 → (a * x^2 + b * x + c = 0)) ∧              -- Option A
  (3 * (x + 1)^2 = 2 * (x - 2) ↔ 3 * x^2 + 4 * x + 7 = 0) ∧  -- Option B
  (1 / x^2 + 1 = x^2 + 1 → False) ∧         -- Option C
  (1 / x^2 + 1 / x - 2 = 0 → False) →       -- Option D
  -- Option B is the only quadratic equation.
  (3 * (x + 1)^2 = 2 * (x - 2)) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_only_option_B_l1711_171113


namespace NUMINAMATH_GPT_find_b_l1711_171129

theorem find_b (b : ℝ) (x : ℝ) (hx : x^2 + b * x - 45 = 0) (h_root : x = -5) : b = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1711_171129


namespace NUMINAMATH_GPT_exists_small_triangle_l1711_171109

-- Definitions and conditions based on the identified problem points
def square_side_length : ℝ := 1
def total_points : ℕ := 53
def vertex_points : ℕ := 4
def interior_points : ℕ := 49
def total_area : ℝ := square_side_length ^ 2
def max_triangle_area : ℝ := 0.01

-- The main theorem statement
theorem exists_small_triangle
  (sq_side : ℝ := square_side_length)
  (total_pts : ℕ := total_points)
  (vertex_pts : ℕ := vertex_points)
  (interior_pts : ℕ := interior_points)
  (total_ar : ℝ := total_area)
  (max_area : ℝ := max_triangle_area)
  (h_side : sq_side = 1)
  (h_pts : total_pts = 53)
  (h_vertex : vertex_pts = 4)
  (h_interior : interior_pts = 49)
  (h_total_area : total_ar = 1) :
  ∃ (t : ℝ), t ≤ max_area :=
sorry

end NUMINAMATH_GPT_exists_small_triangle_l1711_171109


namespace NUMINAMATH_GPT_line_through_point_and_isosceles_triangle_l1711_171172

def is_line_eq (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

def is_isosceles_right_triangle_with_axes (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0

theorem line_through_point_and_isosceles_triangle (a b c : ℝ) (hx : ℝ) (hy : ℝ) :
  is_line_eq a b c hx hy ∧ is_isosceles_right_triangle_with_axes a b → 
  ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 1 ∧ b = -1 ∧ c = -1)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_and_isosceles_triangle_l1711_171172


namespace NUMINAMATH_GPT_additional_hours_q_l1711_171133

variable (P Q : ℝ)

theorem additional_hours_q (h1 : P = 1.5 * Q) 
                           (h2 : P = Q + 8) 
                           (h3 : 480 / P = 20):
  (480 / Q) - (480 / P) = 10 :=
by
  sorry

end NUMINAMATH_GPT_additional_hours_q_l1711_171133


namespace NUMINAMATH_GPT_correct_cost_per_piece_l1711_171175

-- Definitions for the given conditions
def totalPaid : ℝ := 20700
def reimbursement : ℝ := 600
def numberOfPieces : ℝ := 150
def correctTotal := totalPaid - reimbursement

-- Theorem stating the correct cost per piece of furniture
theorem correct_cost_per_piece : correctTotal / numberOfPieces = 134 := 
by
  sorry

end NUMINAMATH_GPT_correct_cost_per_piece_l1711_171175


namespace NUMINAMATH_GPT_absolute_value_inequality_l1711_171178

theorem absolute_value_inequality (x : ℝ) : 
  (|3 * x + 1| > 2) ↔ (x > 1/3 ∨ x < -1) := by
  sorry

end NUMINAMATH_GPT_absolute_value_inequality_l1711_171178


namespace NUMINAMATH_GPT_fraction_problem_l1711_171101

theorem fraction_problem (x : ℝ) (h₁ : x * 180 = 18) (h₂ : x < 0.15) : x = 1/10 :=
by sorry

end NUMINAMATH_GPT_fraction_problem_l1711_171101


namespace NUMINAMATH_GPT_abs_diff_of_solutions_l1711_171144

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end NUMINAMATH_GPT_abs_diff_of_solutions_l1711_171144


namespace NUMINAMATH_GPT_population_present_l1711_171180

variable (P : ℝ)

theorem population_present (h1 : P * 0.90 = 450) : P = 500 :=
by
  sorry

end NUMINAMATH_GPT_population_present_l1711_171180


namespace NUMINAMATH_GPT_find_x_l1711_171126

theorem find_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  -- proof is not required, so we insert sorry
  sorry

end NUMINAMATH_GPT_find_x_l1711_171126


namespace NUMINAMATH_GPT_multiple_of_15_bounds_and_difference_l1711_171162

theorem multiple_of_15_bounds_and_difference :
  ∃ (n : ℕ), 15 * n ≤ 2016 ∧ 2016 < 15 * (n + 1) ∧ (15 * (n + 1) - 2016) = 9 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_15_bounds_and_difference_l1711_171162


namespace NUMINAMATH_GPT_right_triangle_inscribed_circle_inequality_l1711_171188

theorem right_triangle_inscribed_circle_inequality 
  {a b c r : ℝ} (h : a^2 + b^2 = c^2) (hr : r = (a + b - c) / 2) : 
  r ≤ (c / 2) * (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_GPT_right_triangle_inscribed_circle_inequality_l1711_171188


namespace NUMINAMATH_GPT_find_multiple_l1711_171121

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_eq : m * n - 15 = 2 * n + 10) : m = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l1711_171121
