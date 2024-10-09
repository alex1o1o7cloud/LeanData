import Mathlib

namespace projection_matrix_exists_l2394_239499

noncomputable def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, (20 : ℚ) / 49], ![c, (29 : ℚ) / 49]]

theorem projection_matrix_exists :
  ∃ (a c : ℚ), P a c * P a c = P a c ∧ a = (20 : ℚ) / 49 ∧ c = (29 : ℚ) / 49 := 
by
  use ((20 : ℚ) / 49), ((29 : ℚ) / 49)
  simp [P]
  sorry

end projection_matrix_exists_l2394_239499


namespace ratio_alcohol_to_water_l2394_239420

-- Definitions of volume fractions for alcohol and water
def alcohol_volume_fraction : ℚ := 1 / 7
def water_volume_fraction : ℚ := 2 / 7

-- The theorem stating the ratio of alcohol to water volumes
theorem ratio_alcohol_to_water : (alcohol_volume_fraction / water_volume_fraction) = 1 / 2 :=
by sorry

end ratio_alcohol_to_water_l2394_239420


namespace calculate_three_times_neg_two_l2394_239427

-- Define the multiplication of a positive and a negative number resulting in a negative number
def multiply_positive_negative (a b : Int) (ha : a > 0) (hb : b < 0) : Int :=
  a * b

-- Define the absolute value multiplication
def absolute_value_multiplication (a b : Int) : Int :=
  abs a * abs b

-- The theorem that verifies the calculation
theorem calculate_three_times_neg_two : 3 * (-2) = -6 :=
by
  -- Using the given conditions to conclude the result
  sorry

end calculate_three_times_neg_two_l2394_239427


namespace geometric_sequence_ratio_l2394_239454

variables {a b c q : ℝ}

theorem geometric_sequence_ratio (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sequence : ∃ q : ℝ, (a + b + c) * q = b + c - a ∧
                         (a + b + c) * q^2 = c + a - b ∧
                         (a + b + c) * q^3 = a + b - c) :
  q^3 + q^2 + q = 1 := 
sorry

end geometric_sequence_ratio_l2394_239454


namespace polar_coordinates_standard_representation_l2394_239485

theorem polar_coordinates_standard_representation :
  ∀ (r θ : ℝ), (r, θ) = (-4, 5 * Real.pi / 6) → (∃ (r' θ' : ℝ), r' > 0 ∧ (r', θ') = (4, 11 * Real.pi / 6))
:= by
  sorry

end polar_coordinates_standard_representation_l2394_239485


namespace option_C_correct_inequality_l2394_239473

theorem option_C_correct_inequality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) :=
sorry

end option_C_correct_inequality_l2394_239473


namespace pencils_sold_l2394_239413

theorem pencils_sold (C S : ℝ) (n : ℝ) 
  (h1 : 12 * C = n * S) (h2 : S = 1.5 * C) : n = 8 := by
  sorry

end pencils_sold_l2394_239413


namespace problem_conditions_l2394_239405

theorem problem_conditions (a b c x : ℝ) :
  (∀ x, ax^2 + bx + c ≥ 0 ↔ (x ≤ -3 ∨ x ≥ 4)) →
  (a > 0) ∧
  (∀ x, bx + c > 0 → x > -12 = false) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ (x < -1/4 ∨ x > 1/3)) ∧
  (a + b + c ≤ 0) :=
by
  sorry

end problem_conditions_l2394_239405


namespace initial_ratio_is_four_five_l2394_239494

variable (M W : ℕ)

axiom initial_conditions :
  (M + 2 = 14) ∧ (2 * (W - 3) = 24)

theorem initial_ratio_is_four_five 
  (h : M + 2 = 14) 
  (k : 2 * (W - 3) = 24) : M / W = 4 / 5 :=
by
  sorry

end initial_ratio_is_four_five_l2394_239494


namespace parity_of_function_parity_neither_odd_nor_even_l2394_239467

def f (x p : ℝ) : ℝ := x * |x| + p * x^2

theorem parity_of_function (p : ℝ) :
  (∀ x : ℝ, f x p = - f (-x) p) ↔ p = 0 :=
by
  sorry

theorem parity_neither_odd_nor_even (p : ℝ) :
  (∀ x : ℝ, f x p ≠ f (-x) p) ∧ (∀ x : ℝ, f x p ≠ - f (-x) p) ↔ p ≠ 0 :=
by
  sorry

end parity_of_function_parity_neither_odd_nor_even_l2394_239467


namespace tan_alpha_in_second_quadrant_l2394_239462

theorem tan_alpha_in_second_quadrant (α : ℝ) (h₁ : π / 2 < α ∧ α < π) (hsin : Real.sin α = 5 / 13) :
    Real.tan α = -5 / 12 :=
sorry

end tan_alpha_in_second_quadrant_l2394_239462


namespace fraction_of_orange_juice_in_large_container_l2394_239458

def total_capacity := 800 -- mL for each pitcher
def orange_juice_first_pitcher := total_capacity / 2 -- 400 mL
def orange_juice_second_pitcher := total_capacity / 4 -- 200 mL
def total_orange_juice := orange_juice_first_pitcher + orange_juice_second_pitcher -- 600 mL
def total_volume := total_capacity + total_capacity -- 1600 mL

theorem fraction_of_orange_juice_in_large_container :
  (total_orange_juice / total_volume) = 3 / 8 :=
by
  sorry

end fraction_of_orange_juice_in_large_container_l2394_239458


namespace min_abc_sum_l2394_239492

theorem min_abc_sum (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 2010) : 
  a + b + c ≥ 78 := 
sorry

end min_abc_sum_l2394_239492


namespace sum_of_three_numbers_l2394_239438

theorem sum_of_three_numbers (a b c : ℕ)
    (h1 : a + b = 35)
    (h2 : b + c = 40)
    (h3 : c + a = 45) :
    a + b + c = 60 := 
  by sorry

end sum_of_three_numbers_l2394_239438


namespace tan_eq_sin3x_solutions_l2394_239431

open Real

theorem tan_eq_sin3x_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ tan x = sin (3 * x)) ∧ s.card = 6 :=
sorry

end tan_eq_sin3x_solutions_l2394_239431


namespace extreme_value_a_range_l2394_239479

theorem extreme_value_a_range (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1 < x ∧ x < Real.exp 1 ∧ x + a * Real.log x + 1 + a / x = 0)) →
  -Real.exp 1 < a ∧ a < -1 / Real.exp 1 :=
by sorry

end extreme_value_a_range_l2394_239479


namespace point_on_x_axis_l2394_239484

theorem point_on_x_axis : ∃ p, (p = (-2, 0) ∧ p.snd = 0) ∧
  ((p ≠ (0, 2)) ∧ (p ≠ (-2, -3)) ∧ (p ≠ (-1, -2))) :=
by
  sorry

end point_on_x_axis_l2394_239484


namespace probability_of_yellow_ball_is_correct_l2394_239471

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end probability_of_yellow_ball_is_correct_l2394_239471


namespace part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l2394_239486

theorem part_a_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2022 → ∃ k : ℕ, k = 65 :=
sorry

theorem part_b_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2023 → ∃ k : ℕ, k = 65 :=
sorry

end part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l2394_239486


namespace margie_change_l2394_239429

theorem margie_change (n_sold n_cost n_paid : ℕ) (h1 : n_sold = 3) (h2 : n_cost = 50) (h3 : n_paid = 500) : 
  n_paid - (n_sold * n_cost) = 350 := by
  sorry

end margie_change_l2394_239429


namespace tracey_initial_candies_l2394_239417

theorem tracey_initial_candies (x : ℕ) :
  (x % 4 = 0) ∧ (104 ≤ x) ∧ (x ≤ 112) ∧
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ 6 ∧ (x / 2 - 40 - k = 10)) →
  (x = 108 ∨ x = 112) :=
by
  sorry

end tracey_initial_candies_l2394_239417


namespace insurance_coverage_is_80_percent_l2394_239493

-- Definitions and conditions
def MRI_cost : ℕ := 1200
def doctor_hourly_fee : ℕ := 300
def doctor_examination_time : ℕ := 30  -- in minutes
def seen_fee : ℕ := 150
def amount_paid_by_tim : ℕ := 300

-- The total cost calculation
def total_cost : ℕ := MRI_cost + (doctor_hourly_fee * doctor_examination_time / 60) + seen_fee

-- The amount covered by insurance
def amount_covered_by_insurance : ℕ := total_cost - amount_paid_by_tim

-- The percentage of coverage by insurance
def insurance_coverage_percentage : ℕ := (amount_covered_by_insurance * 100) / total_cost

theorem insurance_coverage_is_80_percent : insurance_coverage_percentage = 80 := by
  sorry

end insurance_coverage_is_80_percent_l2394_239493


namespace cycle_original_cost_l2394_239430

theorem cycle_original_cost (SP : ℝ) (gain : ℝ) (CP : ℝ) (h₁ : SP = 2000) (h₂ : gain = 1) (h₃ : SP = CP * (1 + gain)) : CP = 1000 :=
by
  sorry

end cycle_original_cost_l2394_239430


namespace eight_people_lineup_two_windows_l2394_239472

theorem eight_people_lineup_two_windows :
  (2 ^ 8) * (Nat.factorial 8) = 10321920 := by
  sorry

end eight_people_lineup_two_windows_l2394_239472


namespace number_of_negative_x_l2394_239446

theorem number_of_negative_x (n : ℤ) (hn : 1 ≤ n ∧ n * n < 200) : 
  ∃ m ≥ 1, m = 14 := sorry

end number_of_negative_x_l2394_239446


namespace ratio_of_sphere_radii_l2394_239447

noncomputable def ratio_of_radius (V_large : ℝ) (percentage : ℝ) : ℝ :=
  let V_small := (percentage / 100) * V_large
  let ratio := (V_small / V_large) ^ (1/3)
  ratio

theorem ratio_of_sphere_radii : 
  ratio_of_radius (450 * Real.pi) 27.04 = 0.646 := 
  by
  sorry

end ratio_of_sphere_radii_l2394_239447


namespace geometric_sequence_sum_inequality_l2394_239463

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geom : ∀ k, a (k + 1) = a k * q)
  (h_pos : ∀ k ≤ 7, a k > 0)
  (h_q_ne_one : q ≠ 1) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_sequence_sum_inequality_l2394_239463


namespace sum_proper_divisors_81_l2394_239453

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l2394_239453


namespace send_messages_ways_l2394_239436

theorem send_messages_ways : (3^4 = 81) :=
by
  sorry

end send_messages_ways_l2394_239436


namespace sin_transformation_identity_l2394_239482

theorem sin_transformation_identity 
  (θ : ℝ) 
  (h : Real.cos (π / 12 - θ) = 1 / 3) : 
  Real.sin (2 * θ + π / 3) = -7 / 9 := 
by 
  sorry

end sin_transformation_identity_l2394_239482


namespace sum_of_xyz_l2394_239470

theorem sum_of_xyz (x y z : ℝ) (h : (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0) : x + y + z = 12 :=
sorry

end sum_of_xyz_l2394_239470


namespace yoki_cans_collected_l2394_239481

theorem yoki_cans_collected (total_cans LaDonna_cans Prikya_cans Avi_cans : ℕ) (half_Avi_cans Yoki_cans : ℕ) 
    (h1 : total_cans = 85) 
    (h2 : LaDonna_cans = 25) 
    (h3 : Prikya_cans = 2 * LaDonna_cans - 3) 
    (h4 : Avi_cans = 8) 
    (h5 : half_Avi_cans = Avi_cans / 2) 
    (h6 : total_cans = LaDonna_cans + Prikya_cans + half_Avi_cans + Yoki_cans) :
    Yoki_cans = 9 := sorry

end yoki_cans_collected_l2394_239481


namespace parabola_properties_l2394_239400

theorem parabola_properties (m : ℝ) :
  (∀ P : ℝ × ℝ, P = (m, 1) ∧ (P.1 ^ 2 = 4 * P.2) →
    ((∃ y : ℝ, y = -1) ∧ (dist P (0, 1) = 2))) :=
by
  sorry

end parabola_properties_l2394_239400


namespace class_C_payment_l2394_239448

-- Definitions based on conditions
variables (x y z : ℤ) (total_C : ℤ)

-- Given conditions
def condition_A : Prop := 3 * x + 7 * y + z = 14
def condition_B : Prop := 4 * x + 10 * y + z = 16
def condition_C : Prop := 3 * (x + y + z) = total_C

-- The theorem to prove
theorem class_C_payment (hA : condition_A x y z) (hB : condition_B x y z) : total_C = 30 :=
sorry

end class_C_payment_l2394_239448


namespace b_divisible_by_8_l2394_239489

variable (b : ℕ) (n : ℕ)
variable (hb_even : b % 2 = 0) (hb_pos : b > 0) (hn_gt1 : n > 1)
variable (h_square : ∃ k : ℕ, k^2 = (b^n - 1) / (b - 1))

theorem b_divisible_by_8 : b % 8 = 0 :=
by
  sorry

end b_divisible_by_8_l2394_239489


namespace algebraic_expression_value_l2394_239437

-- Define the conditions
variables (x y : ℝ)
-- Condition 1: x - y = 5
def cond1 : Prop := x - y = 5
-- Condition 2: xy = -3
def cond2 : Prop := x * y = -3

-- Define the statement to be proved
theorem algebraic_expression_value :
  cond1 x y → cond2 x y → x^2 * y - x * y^2 = -15 :=
by
  intros h1 h2
  sorry

end algebraic_expression_value_l2394_239437


namespace arith_seq_sum_l2394_239410

-- We start by defining what it means for a sequence to be arithmetic
def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

-- We are given that a_2 = 5 and a_6 = 33 for an arithmetic sequence
variable (a : ℕ → ℤ)
variable (h_arith : is_arith_seq a)
variable (h1 : a 2 = 5)
variable (h2 : a 6 = 33)

-- The statement we want to prove
theorem arith_seq_sum (a : ℕ → ℤ) (h_arith : is_arith_seq a) (h1 : a 2 = 5) (h2 : a 6 = 33) :
  (a 3 + a 5) = 38 :=
  sorry

end arith_seq_sum_l2394_239410


namespace solve_for_n_l2394_239496

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 34) : n = 7 :=
by
  sorry

end solve_for_n_l2394_239496


namespace opposite_of_expression_l2394_239419

theorem opposite_of_expression : 
  let expr := 1 - (3 : ℝ)^(1/3)
  (-1 + (3 : ℝ)^(1/3)) = (3 : ℝ)^(1/3) - 1 :=
by 
  let expr := 1 - (3 : ℝ)^(1/3)
  sorry

end opposite_of_expression_l2394_239419


namespace James_total_tabs_l2394_239495

theorem James_total_tabs (browsers windows tabs additional_tabs : ℕ) 
  (h_browsers : browsers = 4)
  (h_windows : windows = 5)
  (h_tabs : tabs = 12)
  (h_additional_tabs : additional_tabs = 3) : 
  browsers * (windows * (tabs + additional_tabs)) = 300 := by
  -- Proof goes here
  sorry

end James_total_tabs_l2394_239495


namespace no_valid_sum_seventeen_l2394_239441

def std_die (n : ℕ) : Prop := n ∈ [1, 2, 3, 4, 5, 6]

def valid_dice (a b c d : ℕ) : Prop := std_die a ∧ std_die b ∧ std_die c ∧ std_die d

def sum_dice (a b c d : ℕ) : ℕ := a + b + c + d

def prod_dice (a b c d : ℕ) : ℕ := a * b * c * d

theorem no_valid_sum_seventeen (a b c d : ℕ) (h_valid : valid_dice a b c d) (h_prod : prod_dice a b c d = 360) : sum_dice a b c d ≠ 17 :=
sorry

end no_valid_sum_seventeen_l2394_239441


namespace cost_of_history_book_l2394_239477

theorem cost_of_history_book (total_books : ℕ) (cost_math_book : ℕ) (total_price : ℕ) (num_math_books : ℕ) (num_history_books : ℕ) (cost_history_book : ℕ) 
    (h_books_total : total_books = 90)
    (h_cost_math : cost_math_book = 4)
    (h_total_price : total_price = 396)
    (h_num_math_books : num_math_books = 54)
    (h_num_total_books : num_math_books + num_history_books = total_books)
    (h_total_cost : num_math_books * cost_math_book + num_history_books * cost_history_book = total_price) : cost_history_book = 5 := by 
  sorry

end cost_of_history_book_l2394_239477


namespace count_3digit_numbers_div_by_13_l2394_239408

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l2394_239408


namespace simplify_and_evaluate_l2394_239414

def a : Int := 1
def b : Int := -2

theorem simplify_and_evaluate :
  ((a * b - 3 * a^2) - 2 * b^2 - 5 * a * b - (a^2 - 2 * a * b)) = -8 := by
  sorry

end simplify_and_evaluate_l2394_239414


namespace proof_problem_l2394_239444

theorem proof_problem (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a > b) (h5 : a^2 - a * c + b * c = 7) :
  a - c = 0 ∨ a - c = 1 :=
 sorry

end proof_problem_l2394_239444


namespace simplify_fraction_l2394_239480

def expr1 : ℚ := 3
def expr2 : ℚ := 2
def expr3 : ℚ := 3
def expr4 : ℚ := 4
def expected : ℚ := 12 / 5

theorem simplify_fraction : (expr1 / (expr2 - (expr3 / expr4))) = expected := by
  sorry

end simplify_fraction_l2394_239480


namespace quadratic_root_shift_l2394_239442

theorem quadratic_root_shift (r s : ℝ)
    (hr : 2 * r^2 - 8 * r + 6 = 0)
    (hs : 2 * s^2 - 8 * s + 6 = 0)
    (h_sum_roots : r + s = 4)
    (h_prod_roots : r * s = 3)
    (b : ℝ) (c : ℝ)
    (h_b : b = - (r - 3 + s - 3))
    (h_c : c = (r - 3) * (s - 3)) : c = 0 :=
  by sorry

end quadratic_root_shift_l2394_239442


namespace average_population_increase_l2394_239421

-- Conditions
def population_2000 : ℕ := 450000
def population_2005 : ℕ := 467000
def years : ℕ := 5

-- Theorem statement
theorem average_population_increase :
  (population_2005 - population_2000) / years = 3400 := by
  sorry

end average_population_increase_l2394_239421


namespace quadratic_root_condition_l2394_239474

theorem quadratic_root_condition (a : ℝ) :
  (4 * Real.sqrt 2) = 3 * Real.sqrt (3 - 2 * a) → a = 1 / 2 :=
by
  sorry

end quadratic_root_condition_l2394_239474


namespace number_of_4_letter_words_with_vowel_l2394_239407

def is_vowel (c : Char) : Bool :=
c = 'A' ∨ c = 'E'

def count_4letter_words_with_vowels : Nat :=
  let total_words := 5^4
  let words_without_vowels := 3^4
  total_words - words_without_vowels

theorem number_of_4_letter_words_with_vowel :
  count_4letter_words_with_vowels = 544 :=
by
  -- proof goes here
  sorry

end number_of_4_letter_words_with_vowel_l2394_239407


namespace best_sampling_method_l2394_239412

theorem best_sampling_method :
  let elderly := 27
  let middle_aged := 54
  let young := 81
  let total_population := elderly + middle_aged + young
  let sample_size := 36
  let sampling_methods := ["simple random sampling", "systematic sampling", "stratified sampling"]
  stratified_sampling
:=
by
  sorry

end best_sampling_method_l2394_239412


namespace green_more_than_blue_l2394_239459

-- Define the conditions
variables (B Y G n : ℕ)
def ratio_condition := 3 * n = B ∧ 7 * n = Y ∧ 8 * n = G
def total_disks_condition := B + Y + G = 72

-- State the theorem
theorem green_more_than_blue (B Y G n : ℕ) 
  (h_ratio : ratio_condition B Y G n) 
  (h_total : total_disks_condition B Y G) 
  : G - B = 20 := 
sorry

end green_more_than_blue_l2394_239459


namespace avg_highway_mpg_l2394_239416

noncomputable def highway_mpg (total_distance : ℕ) (fuel : ℕ) : ℝ :=
  total_distance / fuel
  
theorem avg_highway_mpg :
  highway_mpg 305 25 = 12.2 :=
by
  sorry

end avg_highway_mpg_l2394_239416


namespace intersection_range_l2394_239445

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem intersection_range :
  {m : ℝ | ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m} = Set.Ioo (-3 : ℝ) 1 :=
by
  sorry

end intersection_range_l2394_239445


namespace sum_of_N_values_eq_neg_one_l2394_239403

theorem sum_of_N_values_eq_neg_one (R : ℝ) :
  ∀ (N : ℝ), N ≠ 0 ∧ (N + N^2 - 5 / N = R) →
  (∃ N₁ N₂ N₃ : ℝ, N₁ + N₂ + N₃ = -1 ∧ N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ N₃ ≠ 0) :=
by
  sorry

end sum_of_N_values_eq_neg_one_l2394_239403


namespace total_valid_votes_l2394_239422

theorem total_valid_votes (V : ℕ) (h1 : 0.70 * (V: ℝ) - 0.30 * (V: ℝ) = 184) : V = 460 :=
by sorry

end total_valid_votes_l2394_239422


namespace seven_not_spheric_spheric_power_spheric_l2394_239487

def is_spheric (r : ℚ) : Prop := ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := 
sorry

theorem spheric_power_spheric (r : ℚ) (n : ℕ) (h : is_spheric r) (hn : n > 1) : is_spheric (r ^ n) := 
sorry

end seven_not_spheric_spheric_power_spheric_l2394_239487


namespace find_x_value_l2394_239491

-- Define vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the condition that a + b is parallel to 2a - b
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (2 * a.1 - b.1) = k * (a.1 + b.1) ∧ (2 * a.2 - b.2) = k * (a.2 + b.2)

-- Problem statement: Prove that x = -4
theorem find_x_value : ∀ (x : ℝ),
  parallel_vectors vector_a (vector_b x) → x = -4 :=
by
  sorry

end find_x_value_l2394_239491


namespace discount_difference_l2394_239449

def single_discount (original: ℝ) (discount: ℝ) : ℝ :=
  original * (1 - discount)

def successive_discount (original: ℝ) (first_discount: ℝ) (second_discount: ℝ) : ℝ :=
  original * (1 - first_discount) * (1 - second_discount)

theorem discount_difference : 
  let original := 12000
  let single_disc := 0.30
  let first_disc := 0.20
  let second_disc := 0.10
  single_discount original single_disc - successive_discount original first_disc second_disc = 240 := 
by sorry

end discount_difference_l2394_239449


namespace circle_radius_order_l2394_239418

theorem circle_radius_order 
  (rA: ℝ) (rA_condition: rA = 2)
  (CB: ℝ) (CB_condition: CB = 10 * Real.pi)
  (AC: ℝ) (AC_condition: AC = 16 * Real.pi) :
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  rA < rC ∧ rC < rB :=
by 
  sorry

end circle_radius_order_l2394_239418


namespace sum_first_12_terms_geom_seq_l2394_239425

def geometric_sequence_periodic (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem sum_first_12_terms_geom_seq :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 2 ∧
    a 3 = 4 ∧
    geometric_sequence_periodic a 8 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_first_12_terms_geom_seq_l2394_239425


namespace simplify_expression_l2394_239401

theorem simplify_expression :
  (∃ (x : Real), x = 3 * (Real.sqrt 3 + Real.sqrt 7) / (4 * Real.sqrt (3 + Real.sqrt 5)) ∧ 
    x = Real.sqrt (224 - 22 * Real.sqrt 105) / 8) := sorry

end simplify_expression_l2394_239401


namespace range_of_a_l2394_239415

theorem range_of_a (h : ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) : a > -1 :=
sorry

end range_of_a_l2394_239415


namespace russian_needed_goals_equals_tunisian_scored_goals_l2394_239469

-- Define the total goals required by each team
def russian_goals := 9
def tunisian_goals := 5

-- Statement: there exists a moment where the Russian remaining goals equal the Tunisian scored goals
theorem russian_needed_goals_equals_tunisian_scored_goals :
  ∃ n : ℕ, n ≤ russian_goals ∧ (russian_goals - n) = (tunisian_goals) := by
  sorry

end russian_needed_goals_equals_tunisian_scored_goals_l2394_239469


namespace gcd_is_12_l2394_239488

noncomputable def gcd_problem (b : ℤ) : Prop :=
  b % 2027 = 0 → Int.gcd (b^2 + 7*b + 18) (b + 6) = 12

-- Now, let's state the theorem
theorem gcd_is_12 (b : ℤ) : gcd_problem b :=
  sorry

end gcd_is_12_l2394_239488


namespace find_r_l2394_239406

variable (a b c r : ℝ)

theorem find_r (h1 : a * (b - c) / (b * (c - a)) = r)
               (h2 : b * (c - a) / (c * (b - a)) = r)
               (h3 : r > 0) : 
               r = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_r_l2394_239406


namespace defective_percentage_is_correct_l2394_239450

noncomputable def percentage_defective (defective : ℕ) (total : ℝ) : ℝ := 
  (defective / total) * 100

theorem defective_percentage_is_correct : 
  percentage_defective 2 3333.3333333333335 = 0.06000600060006 :=
by
  sorry

end defective_percentage_is_correct_l2394_239450


namespace total_journey_distance_l2394_239451

theorem total_journey_distance : 
  ∃ D : ℝ, 
    (∀ (T : ℝ), T = 10) →
    ((D/2) / 21 + (D/2) / 24 = 10) →
    D = 224 := 
by
  sorry

end total_journey_distance_l2394_239451


namespace find_third_number_l2394_239435

-- Given conditions
variable (A B C : ℕ)
variable (LCM HCF : ℕ)
variable (h1 : A = 36)
variable (h2 : B = 44)
variable (h3 : LCM = 792)
variable (h4 : HCF = 12)
variable (h5 : A * B * C = LCM * HCF)

-- Desired proof
theorem find_third_number : C = 6 :=
by
  sorry

end find_third_number_l2394_239435


namespace min_value_reciprocal_l2394_239433

theorem min_value_reciprocal (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  3 ≤ (1/a) + (1/b) + (1/c) :=
sorry

end min_value_reciprocal_l2394_239433


namespace group_size_l2394_239426

theorem group_size (n : ℕ) (T : ℕ) (h1 : T = 14 * n) (h2 : T + 32 = 16 * (n + 1)) : n = 8 :=
by
  -- We skip the proof steps
  sorry

end group_size_l2394_239426


namespace algebraic_expression_value_l2394_239466

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -1) : 6 + 2 * x - 4 * y = 4 := by
  sorry

end algebraic_expression_value_l2394_239466


namespace line_through_points_on_parabola_l2394_239404

theorem line_through_points_on_parabola 
  (x1 y1 x2 y2 : ℝ)
  (h_parabola_A : y1^2 = 4 * x1)
  (h_parabola_B : y2^2 = 4 * x2)
  (h_midpoint : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  ∃ (m b : ℝ), m = 1 ∧ b = 2 ∧ (∀ x y : ℝ, y = m * x + b ↔ x - y = 0) :=
sorry

end line_through_points_on_parabola_l2394_239404


namespace sin_alpha_plus_pi_over_2_l2394_239478

theorem sin_alpha_plus_pi_over_2 
  (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -4 / 3) :
  Real.sin (α + Real.pi / 2) = -3 / 5 :=
by
  sorry

end sin_alpha_plus_pi_over_2_l2394_239478


namespace rearrange_to_rectangle_l2394_239424

-- Definition of a geometric figure and operations
structure Figure where
  parts : List (List (ℤ × ℤ)) -- List of parts represented by lists of coordinates

def is_cut_into_three_parts (fig : Figure) : Prop :=
  fig.parts.length = 3

def can_be_rearranged_to_form_rectangle (fig : Figure) : Prop := sorry

-- Initial given figure
variable (initial_figure : Figure)

-- Conditions
axiom figure_can_be_cut : is_cut_into_three_parts initial_figure
axiom cuts_not_along_grid_lines : True -- Replace with appropriate geometric operation when image is known
axiom parts_can_be_flipped : True -- Replace with operation allowing part flipping

-- Theorem to prove
theorem rearrange_to_rectangle : 
  is_cut_into_three_parts initial_figure →
  can_be_rearranged_to_form_rectangle initial_figure := 
sorry

end rearrange_to_rectangle_l2394_239424


namespace polynomial_solution_exists_l2394_239483

open Real

theorem polynomial_solution_exists
    (P : ℝ → ℝ → ℝ)
    (hP : ∃ (f : ℝ → ℝ), ∀ x y : ℝ, P x y = f (x + y) - f x - f y) :
  ∃ (q : ℝ → ℝ), ∀ x y : ℝ, P x y = q (x + y) - q x - q y := sorry

end polynomial_solution_exists_l2394_239483


namespace students_in_each_group_l2394_239468

theorem students_in_each_group (num_boys : ℕ) (num_girls : ℕ) (num_groups : ℕ) 
  (h_boys : num_boys = 26) (h_girls : num_girls = 46) (h_groups : num_groups = 8) : 
  (num_boys + num_girls) / num_groups = 9 := 
by 
  sorry

end students_in_each_group_l2394_239468


namespace cos_210_eq_neg_sqrt3_div_2_l2394_239475

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l2394_239475


namespace james_vegetable_consumption_l2394_239439

def vegetable_consumption_weekdays (asparagus broccoli cauliflower spinach : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + spinach

def vegetable_consumption_weekend (asparagus broccoli cauliflower other_veg : ℚ) : ℚ :=
  asparagus + broccoli + cauliflower + other_veg

def total_vegetable_consumption (
  wd_asparagus wd_broccoli wd_cauliflower wd_spinach : ℚ)
  (sat_asparagus sat_broccoli sat_cauliflower sat_other : ℚ)
  (sun_asparagus sun_broccoli sun_cauliflower sun_other : ℚ) : ℚ :=
  5 * vegetable_consumption_weekdays wd_asparagus wd_broccoli wd_cauliflower wd_spinach +
  vegetable_consumption_weekend sat_asparagus sat_broccoli sat_cauliflower sat_other +
  vegetable_consumption_weekend sun_asparagus sun_broccoli sun_cauliflower sun_other

theorem james_vegetable_consumption :
  total_vegetable_consumption 0.5 0.75 0.875 0.5 0.3 0.4 0.6 1 0.3 0.4 0.6 0.5 = 17.225 :=
sorry

end james_vegetable_consumption_l2394_239439


namespace find_number_l2394_239456

variable (N : ℝ)

theorem find_number (h : (5 / 6) * N = (5 / 16) * N + 50) : N = 96 := 
by 
  sorry

end find_number_l2394_239456


namespace problem_statement_l2394_239432

def g (x : ℝ) : ℝ := x ^ 3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem problem_statement : f (g 3) = 53 :=
by
  sorry

end problem_statement_l2394_239432


namespace even_odd_product_zero_l2394_239411

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem even_odd_product_zero (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : is_even f) (hg : is_odd g) : ∀ x, f (-x) * g (-x) + f x * g x = 0 :=
by
  intro x
  have h₁ := hf x
  have h₂ := hg x
  sorry

end even_odd_product_zero_l2394_239411


namespace polar_equation_is_circle_of_radius_five_l2394_239440

theorem polar_equation_is_circle_of_radius_five :
  ∀ θ : ℝ, (3 * Real.sin θ + 4 * Real.cos θ) ^ 2 = 25 :=
by
  sorry

end polar_equation_is_circle_of_radius_five_l2394_239440


namespace difference_between_numbers_l2394_239443

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 20000) (h2 : b = 2 * a + 6) (h3 : 9 ∣ a) : b - a = 6670 :=
by
  sorry

end difference_between_numbers_l2394_239443


namespace determine_x_l2394_239428

theorem determine_x (x : ℕ) 
  (hx1 : x % 6 = 0) 
  (hx2 : x^2 > 196) 
  (hx3 : x < 30) : 
  x = 18 ∨ x = 24 := 
sorry

end determine_x_l2394_239428


namespace age_of_youngest_boy_l2394_239461

theorem age_of_youngest_boy (average_age : ℕ) (age_proportion : ℕ → ℕ) 
  (h1 : average_age = 120) 
  (h2 : ∀ x, age_proportion x = 2 * x ∨ age_proportion x = 6 * x ∨ age_proportion x = 8 * x)
  (total_age : ℕ) 
  (h3 : total_age = 3 * average_age) :
  ∃ x, age_proportion x = 2 * x ∧ 2 * x * (3 * average_age / total_age) = 45 :=
by {
  sorry
}

end age_of_youngest_boy_l2394_239461


namespace ratio_rectangle_to_semicircles_area_l2394_239498

theorem ratio_rectangle_to_semicircles_area (AB AD : ℝ) (h1 : AB = 40) (h2 : AD / AB = 3 / 2) : 
  (AB * AD) / (2 * (π * (AB / 2)^2)) = 6 / π :=
by
  -- here we process the proof
  sorry

end ratio_rectangle_to_semicircles_area_l2394_239498


namespace range_of_a_l2394_239455

/--
Given the parabola \(x^2 = y\), points \(A\) and \(B\) are on the parabola and located on both sides of the y-axis,
and the line \(AB\) intersects the y-axis at point \((0, a)\). If \(\angle AOB\) is an acute angle (where \(O\) is the origin),
then the real number \(a\) is greater than 1.
-/
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) : (x1^2 = x2^2) → (x1 * x2 = -a) → ((-a + a^2) > 0) → (1 < a) :=
by 
  sorry

end range_of_a_l2394_239455


namespace boys_without_pencils_l2394_239476

variable (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)

theorem boys_without_pencils
  (h1 : total_boys = 18)
  (h2 : students_with_pencils = 25)
  (h3 : girls_with_pencils = 15)
  (h4 : total_students = 30) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by
  sorry

end boys_without_pencils_l2394_239476


namespace find_fraction_l2394_239452

-- Let's define the conditions
variables (F N : ℝ)
axiom condition1 : (1 / 4) * (1 / 3) * F * N = 15
axiom condition2 : 0.4 * N = 180

-- theorem to prove the fraction F
theorem find_fraction : F = 2 / 5 :=
by
  -- proof steps would go here, but we're adding sorry to skip the proof.
  sorry

end find_fraction_l2394_239452


namespace floor_sum_correct_l2394_239423

def floor_sum_1_to_24 := 
  let sum := (3 * 1) + (5 * 2) + (7 * 3) + (9 * 4)
  sum

theorem floor_sum_correct : floor_sum_1_to_24 = 70 := by
  sorry

end floor_sum_correct_l2394_239423


namespace abs_inequality_solution_l2394_239497

theorem abs_inequality_solution {a : ℝ} (h : ∀ x : ℝ, |2 - x| + |x + 1| ≥ a) : a ≤ 3 :=
sorry

end abs_inequality_solution_l2394_239497


namespace express_in_standard_form_l2394_239460

theorem express_in_standard_form (x : ℝ) : x^2 - 6 * x = (x - 3)^2 - 9 :=
by
  sorry

end express_in_standard_form_l2394_239460


namespace spherical_to_rectangular_coordinates_l2394_239402

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ),
    ρ = 15 →
    θ = 5 * Real.pi / 6 →
    φ = Real.pi / 3 →
    x = ρ * Real.sin φ * Real.cos θ →
    y = ρ * Real.sin φ * Real.sin θ →
    z = ρ * Real.cos φ →
    x = -45 / 4 ∧ y = -15 * Real.sqrt 3 / 4 ∧ z = 7.5 := 
by
  intro ρ θ φ x y z
  intro hρ hθ hφ hx hy hz
  rw [hρ, hθ, hφ] at *
  rw [hx, hy, hz]
  sorry

end spherical_to_rectangular_coordinates_l2394_239402


namespace last_digit_inverse_power_two_l2394_239409

theorem last_digit_inverse_power_two :
  let n := 12
  let x := 5^n
  let y := 10^n
  (x % 10 = 5) →
  ((1 / (2^n)) * (5^n) / (5^n) == (5^n) / (10^n)) →
  (y % 10 = 0) →
  ((1 / (2^n)) % 10 = 5) :=
by
  intros n x y h1 h2 h3
  sorry

end last_digit_inverse_power_two_l2394_239409


namespace pie_eating_contest_l2394_239434

theorem pie_eating_contest :
  let a := 5 / 6
  let b := 7 / 8
  let c := 2 / 3
  let max_pie := max a (max b c)
  let min_pie := min a (min b c)
  max_pie - min_pie = 5 / 24 :=
by
  sorry

end pie_eating_contest_l2394_239434


namespace three_digit_numbers_square_ends_in_1001_l2394_239464

theorem three_digit_numbers_square_ends_in_1001 (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ n^2 % 10000 = 1001 → n = 501 ∨ n = 749 :=
by
  intro h
  sorry

end three_digit_numbers_square_ends_in_1001_l2394_239464


namespace wayne_needs_30_more_blocks_l2394_239465

def initial_blocks : ℕ := 9
def additional_blocks : ℕ := 6
def total_blocks : ℕ := initial_blocks + additional_blocks
def triple_total : ℕ := 3 * total_blocks

theorem wayne_needs_30_more_blocks :
  triple_total - total_blocks = 30 := by
  sorry

end wayne_needs_30_more_blocks_l2394_239465


namespace compare_points_on_line_l2394_239490

theorem compare_points_on_line (m n : ℝ) 
  (hA : ∃ (x : ℝ), x = -3 ∧ m = -2 * x + 1) 
  (hB : ∃ (x : ℝ), x = 2 ∧ n = -2 * x + 1) : 
  m > n :=
by sorry

end compare_points_on_line_l2394_239490


namespace min_length_BC_l2394_239457

theorem min_length_BC (A B C D : Type) (AB AC DC BD BC : ℝ) :
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → (BC > AC - AB) ∧ (BC > BD - DC) → BC ≥ 15 :=
by
  intros hAB hAC hDC hBD hIneq
  sorry

end min_length_BC_l2394_239457
