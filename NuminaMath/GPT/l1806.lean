import Mathlib

namespace NUMINAMATH_GPT_gcd_245_1001_l1806_180691

-- Definitions based on the given conditions

def fact245 : ℕ := 5 * 7^2
def fact1001 : ℕ := 7 * 11 * 13

-- Lean 4 statement of the proof problem
theorem gcd_245_1001 : Nat.gcd fact245 fact1001 = 7 :=
by
  -- Add the prime factorizations as assumptions
  have h1: fact245 = 245 := by sorry
  have h2: fact1001 = 1001 := by sorry
  -- The goal is to prove the GCD
  sorry

end NUMINAMATH_GPT_gcd_245_1001_l1806_180691


namespace NUMINAMATH_GPT_point_divides_segment_l1806_180646

theorem point_divides_segment (x₁ y₁ x₂ y₂ m n : ℝ) (h₁ : (x₁, y₁) = (3, 7)) (h₂ : (x₂, y₂) = (5, 1)) (h₃ : m = 1) (h₄ : n = 3) :
  ( (m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n) ) = (3.5, 5.5) :=
by
  sorry

end NUMINAMATH_GPT_point_divides_segment_l1806_180646


namespace NUMINAMATH_GPT_count_integers_six_times_sum_of_digits_l1806_180681

theorem count_integers_six_times_sum_of_digits (n : ℕ) (h : n < 1000) 
    (digit_sum : ℕ → ℕ)
    (digit_sum_correct : ∀ (n : ℕ), digit_sum n = (n % 10) + ((n / 10) % 10) + (n / 100)) :
    ∃! n, n < 1000 ∧ n = 6 * digit_sum n :=
sorry

end NUMINAMATH_GPT_count_integers_six_times_sum_of_digits_l1806_180681


namespace NUMINAMATH_GPT_diane_15_cents_arrangement_l1806_180680

def stamps : List (ℕ × ℕ) := 
  [(1, 1), 
   (2, 2), 
   (3, 3), 
   (4, 4), 
   (5, 5), 
   (6, 6), 
   (7, 7), 
   (8, 8), 
   (9, 9), 
   (10, 10), 
   (11, 11), 
   (12, 12)]

def number_of_arrangements (value : ℕ) (stamps : List (ℕ × ℕ)) : ℕ := sorry

theorem diane_15_cents_arrangement : number_of_arrangements 15 stamps = 32 := 
sorry

end NUMINAMATH_GPT_diane_15_cents_arrangement_l1806_180680


namespace NUMINAMATH_GPT_harvest_apples_l1806_180617

def sacks_per_section : ℕ := 45
def sections : ℕ := 8
def total_sacks_per_day : ℕ := 360

theorem harvest_apples : sacks_per_section * sections = total_sacks_per_day := by
  sorry

end NUMINAMATH_GPT_harvest_apples_l1806_180617


namespace NUMINAMATH_GPT_four_faucets_fill_time_correct_l1806_180683

-- Define the parameters given in the conditions
def three_faucets_rate (volume : ℕ) (time : ℕ) := volume / time
def one_faucet_rate (rate : ℕ) := rate / 3
def four_faucets_rate (rate : ℕ) := 4 * rate
def fill_time (volume : ℕ) (rate : ℕ) := volume / rate

-- Given problem parameters
def volume_large_tub : ℕ := 100
def time_large_tub : ℕ := 6
def volume_small_tub : ℕ := 50

-- Theorem to be proven
theorem four_faucets_fill_time_correct :
  fill_time volume_small_tub (four_faucets_rate (one_faucet_rate (three_faucets_rate volume_large_tub time_large_tub))) * 60 = 135 :=
sorry

end NUMINAMATH_GPT_four_faucets_fill_time_correct_l1806_180683


namespace NUMINAMATH_GPT_total_pigs_indeterminate_l1806_180605

noncomputable def average_weight := 15
def underweight_threshold := 16
def max_underweight_pigs := 4

theorem total_pigs_indeterminate :
  ∃ (P U : ℕ), U ≤ max_underweight_pigs ∧ (average_weight = 15) → P = P :=
sorry

end NUMINAMATH_GPT_total_pigs_indeterminate_l1806_180605


namespace NUMINAMATH_GPT_student_marks_l1806_180626

variable (x : ℕ)
variable (passing_marks : ℕ)
variable (max_marks : ℕ := 400)
variable (fail_by : ℕ := 14)

theorem student_marks :
  (passing_marks = 36 * max_marks / 100) →
  (x + fail_by = passing_marks) →
  x = 130 :=
by sorry

end NUMINAMATH_GPT_student_marks_l1806_180626


namespace NUMINAMATH_GPT_solve_for_y_l1806_180669

theorem solve_for_y : ∃ (y : ℚ), y + 2 - 2 / 3 = 4 * y - (y + 2) ∧ y = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1806_180669


namespace NUMINAMATH_GPT_factorization_identity_l1806_180698

theorem factorization_identity (a b : ℝ) : 
  -a^3 + 12 * a^2 * b - 36 * a * b^2 = -a * (a - 6 * b)^2 :=
by 
  sorry

end NUMINAMATH_GPT_factorization_identity_l1806_180698


namespace NUMINAMATH_GPT_customOp_eval_l1806_180696

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- State the theorem we need to prove
theorem customOp_eval : customOp 4 (-1) = -4 :=
  by
    sorry

end NUMINAMATH_GPT_customOp_eval_l1806_180696


namespace NUMINAMATH_GPT_cube_volume_and_surface_area_l1806_180662

theorem cube_volume_and_surface_area (e : ℕ) (h : 12 * e = 72) :
  (e^3 = 216) ∧ (6 * e^2 = 216) := by
  sorry

end NUMINAMATH_GPT_cube_volume_and_surface_area_l1806_180662


namespace NUMINAMATH_GPT_find_p_value_l1806_180627

noncomputable def solve_p (m p : ℕ) :=
  (1^m / 5^m) * (1^16 / 4^16) = 1 / (2 * p^31)

theorem find_p_value (m p : ℕ) (hm : m = 31) :
  solve_p m p ↔ p = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_p_value_l1806_180627


namespace NUMINAMATH_GPT_cary_strips_ivy_l1806_180635

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end NUMINAMATH_GPT_cary_strips_ivy_l1806_180635


namespace NUMINAMATH_GPT_initial_rate_of_interest_l1806_180609

theorem initial_rate_of_interest (P : ℝ) (R : ℝ) 
  (h1 : 1680 = (P * R * 5) / 100) 
  (h2 : 1680 = (P * 5 * 4) / 100) : 
  R = 4 := 
by 
  sorry

end NUMINAMATH_GPT_initial_rate_of_interest_l1806_180609


namespace NUMINAMATH_GPT_side_length_square_l1806_180658

theorem side_length_square (A : ℝ) (s : ℝ) (h1 : A = 30) (h2 : A = s^2) : 5 < s ∧ s < 6 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_side_length_square_l1806_180658


namespace NUMINAMATH_GPT_determine_delta_l1806_180619

theorem determine_delta (r1 r2 r3 r4 r5 r6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ) (O Δ : ℕ) 
  (h_sums_rows : r1 + r2 + r3 + r4 + r5 + r6 = 190)
  (h_row1 : r1 = 29) (h_row2 : r2 = 33) (h_row3 : r3 = 33) 
  (h_row4 : r4 = 32) (h_row5 : r5 = 32) (h_row6 : r6 = 31)
  (h_sums_cols : c1 + c2 + c3 + c4 + c5 + c6 = 190)
  (h_col1 : c1 = 29) (h_col2 : c2 = 33) (h_col3 : c3 = 33) 
  (h_col4 : c4 = 32) (h_col5 : c5 = 32) (h_col6 : c6 = 31)
  (h_O : O = 6) : 
  Δ = 4 :=
by 
  sorry

end NUMINAMATH_GPT_determine_delta_l1806_180619


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1806_180688

theorem sum_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 220) 
  (h2 : x * y = 52) : 
  x + y = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1806_180688


namespace NUMINAMATH_GPT_determine_sequence_parameters_l1806_180614

variables {n : ℕ} {d q : ℝ} (h1 : 1 + (n-1) * d = 81) (h2 : 1 * q^(n-1) = 81) (h3 : q / d = 0.15)

theorem determine_sequence_parameters : n = 5 ∧ d = 20 ∧ q = 3 :=
by {
  -- Assumptions:
  -- h1: Arithmetic sequence, a1 = 1, an = 81
  -- h2: Geometric sequence, b1 = 1, bn = 81
  -- h3: q / d = 0.15
  -- Goal: n = 5, d = 20, q = 3
  sorry
}

end NUMINAMATH_GPT_determine_sequence_parameters_l1806_180614


namespace NUMINAMATH_GPT_min_value_expression_l1806_180638

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1806_180638


namespace NUMINAMATH_GPT_fractions_order_l1806_180645

theorem fractions_order :
  (25 / 21 < 23 / 19) ∧ (23 / 19 < 21 / 17) :=
by {
  sorry
}

end NUMINAMATH_GPT_fractions_order_l1806_180645


namespace NUMINAMATH_GPT_abs_difference_of_squares_l1806_180642

theorem abs_difference_of_squares : abs ((102: ℤ) ^ 2 - (98: ℤ) ^ 2) = 800 := by
  sorry

end NUMINAMATH_GPT_abs_difference_of_squares_l1806_180642


namespace NUMINAMATH_GPT_correct_money_calculation_l1806_180664

structure BootSale :=
(initial_money : ℕ)
(price_per_boot : ℕ)
(total_taken : ℕ)
(total_returned : ℕ)
(money_spent : ℕ)
(remaining_money_to_return : ℕ)

theorem correct_money_calculation (bs : BootSale) :
  bs.initial_money = 25 →
  bs.price_per_boot = 12 →
  bs.total_taken = 25 →
  bs.total_returned = 5 →
  bs.money_spent = 3 →
  bs.remaining_money_to_return = 2 →
  bs.total_taken - bs.total_returned + bs.money_spent = 23 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_correct_money_calculation_l1806_180664


namespace NUMINAMATH_GPT_height_on_fifth_bounce_l1806_180697

-- Define initial conditions
def initial_height : ℝ := 96
def initial_efficiency : ℝ := 0.5
def efficiency_decrease : ℝ := 0.05
def air_resistance_loss : ℝ := 0.02

-- Recursive function to compute the height after each bounce
def bounce_height (height : ℝ) (efficiency : ℝ) : ℝ :=
  let height_after_bounce := height * efficiency
  height_after_bounce - (height_after_bounce * air_resistance_loss)

-- Function to compute the bounce efficiency after each bounce
def bounce_efficiency (initial_efficiency : ℝ) (n : ℕ) : ℝ :=
  initial_efficiency - n * efficiency_decrease

-- Function to calculate the height after n-th bounce
def height_after_n_bounces (n : ℕ) : ℝ :=
  match n with
  | 0     => initial_height
  | n + 1 => bounce_height (height_after_n_bounces n) (bounce_efficiency initial_efficiency n)

-- Lean statement to prove the problem
theorem height_on_fifth_bounce :
  height_after_n_bounces 5 = 0.82003694685696 := by
  sorry

end NUMINAMATH_GPT_height_on_fifth_bounce_l1806_180697


namespace NUMINAMATH_GPT_polygon_length_l1806_180690

noncomputable def DE : ℝ := 3
noncomputable def EF : ℝ := 6
noncomputable def DE_plus_EF : ℝ := DE + EF

theorem polygon_length 
  (area_ABCDEF : ℝ)
  (AB BC FA : ℝ)
  (A B C D E F : ℝ × ℝ) :
  area_ABCDEF = 60 →
  AB = 10 →
  BC = 7 →
  FA = 6 →
  A = (0, 10) →
  B = (10, 10) →
  C = (10, 0) →
  D = (6, 0) →
  E = (6, 3) →
  F = (0, 3) →
  DE_plus_EF = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_polygon_length_l1806_180690


namespace NUMINAMATH_GPT_tan_problem_l1806_180601

noncomputable def problem : ℝ :=
  (Real.tan (20 * Real.pi / 180) + Real.tan (40 * Real.pi / 180) + Real.tan (120 * Real.pi / 180)) / 
  (Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180))

theorem tan_problem : problem = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_problem_l1806_180601


namespace NUMINAMATH_GPT_positive_number_is_nine_l1806_180629

theorem positive_number_is_nine (x : ℝ) (n : ℝ) (hx : x > 0) (hn : n > 0)
  (sqrt1 : x^2 = n) (sqrt2 : (x - 6)^2 = n) : 
  n = 9 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_is_nine_l1806_180629


namespace NUMINAMATH_GPT_max_difference_in_masses_of_two_flour_bags_l1806_180684

theorem max_difference_in_masses_of_two_flour_bags :
  ∀ (x y : ℝ), (24.8 ≤ x ∧ x ≤ 25.2) → (24.8 ≤ y ∧ y ≤ 25.2) → |x - y| ≤ 0.4 :=
by
  sorry

end NUMINAMATH_GPT_max_difference_in_masses_of_two_flour_bags_l1806_180684


namespace NUMINAMATH_GPT_part_I_part_II_part_III_l1806_180670

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2

-- Part (Ⅰ)
theorem part_I (x : ℝ) : (0 < x) → (f 1 x < f 1 (x+1)) := sorry

-- Part (Ⅱ)
theorem part_II (f_has_two_distinct_extreme_values : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ (f a x = f a y))) : 0 < a ∧ a < 1 := sorry

-- Part (Ⅲ)
theorem part_III (f_has_two_distinct_zeros : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) : 0 < a ∧ a < (2 / Real.exp 1) := sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_l1806_180670


namespace NUMINAMATH_GPT_positive_difference_x_coordinates_lines_l1806_180668

theorem positive_difference_x_coordinates_lines :
  let l := fun x : ℝ => -2 * x + 4
  let m := fun x : ℝ => - (1 / 5) * x + 1
  let x_l := (- (10 - 4) / 2)
  let x_m := (- (10 - 1) * 5)
  abs (x_l - x_m) = 42 := by
  sorry

end NUMINAMATH_GPT_positive_difference_x_coordinates_lines_l1806_180668


namespace NUMINAMATH_GPT_probability_at_least_9_heads_in_12_flips_l1806_180644

theorem probability_at_least_9_heads_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := favorable_outcomes / total_outcomes
  probability = 299 / 4096 := 
by
  sorry

end NUMINAMATH_GPT_probability_at_least_9_heads_in_12_flips_l1806_180644


namespace NUMINAMATH_GPT_three_times_x_greater_than_four_l1806_180621

theorem three_times_x_greater_than_four (x : ℝ) : 3 * x > 4 := by
  sorry

end NUMINAMATH_GPT_three_times_x_greater_than_four_l1806_180621


namespace NUMINAMATH_GPT_pineapple_cost_l1806_180604

variables (P W : ℕ)

theorem pineapple_cost (h1 : 2 * P + 5 * W = 38) : P = 14 :=
sorry

end NUMINAMATH_GPT_pineapple_cost_l1806_180604


namespace NUMINAMATH_GPT_find_p_q_d_l1806_180695

noncomputable def cubic_polynomial_real_root (p q d : ℕ) (x : ℝ) : Prop :=
  27 * x^3 - 12 * x^2 - 4 * x - 1 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / d ∧
  p > 0 ∧ q > 0 ∧ d > 0

theorem find_p_q_d :
  ∃ (p q d : ℕ), cubic_polynomial_real_root p q d 1 ∧ p + q + d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_d_l1806_180695


namespace NUMINAMATH_GPT_max_value_of_function_l1806_180634

theorem max_value_of_function (x : ℝ) (h : x < 1 / 2) : 
  ∃ y, y = 2 * x + 1 / (2 * x - 1) ∧ y ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_function_l1806_180634


namespace NUMINAMATH_GPT_shaded_areas_equal_l1806_180656

theorem shaded_areas_equal (φ : ℝ) (hφ1 : 0 < φ) (hφ2 : φ < π / 4) : 
  (Real.tan φ) = 2 * φ :=
sorry

end NUMINAMATH_GPT_shaded_areas_equal_l1806_180656


namespace NUMINAMATH_GPT_merchant_markup_l1806_180666

theorem merchant_markup (C : ℝ) (M : ℝ) (h1 : (1 + M / 100 - 0.40 * (1 + M / 100)) * C = 1.05 * C) : 
  M = 75 := sorry

end NUMINAMATH_GPT_merchant_markup_l1806_180666


namespace NUMINAMATH_GPT_total_profit_from_selling_30_necklaces_l1806_180623

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_from_selling_30_necklaces_l1806_180623


namespace NUMINAMATH_GPT_Victoria_money_left_l1806_180657

noncomputable def Victoria_initial_money : ℝ := 10000
noncomputable def jacket_price : ℝ := 250
noncomputable def trousers_price : ℝ := 180
noncomputable def purse_price : ℝ := 450
noncomputable def jackets_bought : ℕ := 8
noncomputable def trousers_bought : ℕ := 15
noncomputable def purses_bought : ℕ := 4
noncomputable def discount_rate : ℝ := 0.15
noncomputable def dinner_bill_inclusive : ℝ := 552.50
noncomputable def dinner_service_charge_rate : ℝ := 0.15

theorem Victoria_money_left : 
  Victoria_initial_money - 
  ((jackets_bought * jacket_price + trousers_bought * trousers_price) * (1 - discount_rate) + 
   purses_bought * purse_price + 
   dinner_bill_inclusive / (1 + dinner_service_charge_rate)) = 3725 := 
by 
  sorry

end NUMINAMATH_GPT_Victoria_money_left_l1806_180657


namespace NUMINAMATH_GPT_power_of_two_l1806_180637

theorem power_of_two (Number : ℕ) (h1 : Number = 128) (h2 : Number * (1/4 : ℝ) = 2^5) :
  ∃ power : ℕ, 2^power = 128 := 
by
  use 7
  sorry

end NUMINAMATH_GPT_power_of_two_l1806_180637


namespace NUMINAMATH_GPT_total_fruit_weight_l1806_180653

-- Definitions for the conditions
def mario_ounces : ℕ := 8
def lydia_ounces : ℕ := 24
def nicolai_pounds : ℕ := 6
def ounces_per_pound : ℕ := 16

-- Theorem statement
theorem total_fruit_weight : 
  ((mario_ounces / ounces_per_pound : ℚ) + 
   (lydia_ounces / ounces_per_pound : ℚ) + 
   (nicolai_pounds : ℚ)) = 8 := 
sorry

end NUMINAMATH_GPT_total_fruit_weight_l1806_180653


namespace NUMINAMATH_GPT_smallest_positive_period_maximum_f_B_l1806_180661

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 2

theorem smallest_positive_period (x : ℝ) : 
  (∀ T, (f (x + T) = f x) → (T ≥ 0) → T = Real.pi) := 
sorry

variable {a b c : ℝ}

lemma cos_law_cos_B (h : b^2 = a * c) : 
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  (1 / 2) ≤ Real.cos B ∧ Real.cos B < 1 := 
sorry

theorem maximum_f_B (h : b^2 = a * c) :
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  f B ≤ 1 := 
sorry

end NUMINAMATH_GPT_smallest_positive_period_maximum_f_B_l1806_180661


namespace NUMINAMATH_GPT_largest_integer_value_x_l1806_180636

theorem largest_integer_value_x : ∀ (x : ℤ), (5 - 4 * x > 17) → x ≤ -4 := sorry

end NUMINAMATH_GPT_largest_integer_value_x_l1806_180636


namespace NUMINAMATH_GPT_sum_of_solutions_l1806_180607

theorem sum_of_solutions : 
  let a := 1
  let b := -7
  let c := -30
  (a * x^2 + b * x + c = 0) → ((-b / a) = 7) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1806_180607


namespace NUMINAMATH_GPT_find_a_sq_plus_b_sq_l1806_180600

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 48
axiom h2 : a * b = 156

theorem find_a_sq_plus_b_sq : a^2 + b^2 = 1992 :=
by sorry

end NUMINAMATH_GPT_find_a_sq_plus_b_sq_l1806_180600


namespace NUMINAMATH_GPT_A_worked_alone_after_B_left_l1806_180659

/-- A and B can together finish a work in 40 days. They worked together for 10 days and then B left.
    A alone can finish the job in 80 days. We need to find out how many days did A work alone after B left. -/
theorem A_worked_alone_after_B_left
  (W : ℝ)
  (A_work_rate : ℝ := W / 80)
  (B_work_rate : ℝ := W / 80)
  (AB_work_rate : ℝ := W / 40)
  (work_done_together_in_10_days : ℝ := 10 * (W / 40))
  (remaining_work : ℝ := W - work_done_together_in_10_days)
  (A_rate_alone : ℝ := W / 80) :
  ∃ D : ℝ, D * (W / 80) = remaining_work → D = 60 :=
by
  sorry

end NUMINAMATH_GPT_A_worked_alone_after_B_left_l1806_180659


namespace NUMINAMATH_GPT_last_three_digits_of_5_power_15000_l1806_180613

theorem last_three_digits_of_5_power_15000:
  (5^15000) % 1000 = 1 % 1000 :=
by
  have h : 5^500 % 1000 = 1 % 1000 := by sorry
  sorry

end NUMINAMATH_GPT_last_three_digits_of_5_power_15000_l1806_180613


namespace NUMINAMATH_GPT_cricket_runs_l1806_180674

theorem cricket_runs (A B C : ℕ) (h1 : A / B = 1 / 3) (h2 : B / C = 1 / 5) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Skipping proof details
  sorry

end NUMINAMATH_GPT_cricket_runs_l1806_180674


namespace NUMINAMATH_GPT_pairings_count_l1806_180625

-- Define the problem's conditions explicitly
def number_of_bowls : Nat := 6
def number_of_glasses : Nat := 6

-- The theorem stating that the number of pairings is 36
theorem pairings_count : number_of_bowls * number_of_glasses = 36 := by
  sorry

end NUMINAMATH_GPT_pairings_count_l1806_180625


namespace NUMINAMATH_GPT_sum_of_rel_prime_greater_than_one_l1806_180633

theorem sum_of_rel_prime_greater_than_one (a : ℕ) (h : a > 6) : 
  ∃ b c : ℕ, a = b + c ∧ b > 1 ∧ c > 1 ∧ Nat.gcd b c = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_rel_prime_greater_than_one_l1806_180633


namespace NUMINAMATH_GPT_part1_part2_part3_l1806_180648

def pointM (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Part 1
theorem part1 (m : ℝ) (h : 2 * m + 3 = 0) : pointM m = (-5 / 2, 0) :=
  sorry

-- Part 2
theorem part2 (m : ℝ) (h : 2 * m + 3 = -1) : pointM m = (-3, -1) :=
  sorry

-- Part 3
theorem part3 (m : ℝ) (h1 : |m - 1| = 2) : pointM m = (2, 9) ∨ pointM m = (-2, 1) :=
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l1806_180648


namespace NUMINAMATH_GPT_ways_to_write_1800_as_sum_of_twos_and_threes_l1806_180611

theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  ∃ (n : ℕ), n = 301 ∧ ∀ (x y : ℕ), 2 * x + 3 * y = 1800 → ∃ (a : ℕ), (x, y) = (3 * a, 300 - a) :=
sorry

end NUMINAMATH_GPT_ways_to_write_1800_as_sum_of_twos_and_threes_l1806_180611


namespace NUMINAMATH_GPT_martin_speed_first_half_l1806_180616

variable (v : ℝ) -- speed during the first half of the trip

theorem martin_speed_first_half
    (trip_duration : ℝ := 8)              -- The trip lasted 8 hours
    (speed_second_half : ℝ := 85)          -- Speed during the second half of the trip
    (total_distance : ℝ := 620)            -- Total distance traveled
    (time_each_half : ℝ := trip_duration / 2) -- Each half of the trip took half of the total time
    (distance_second_half : ℝ := speed_second_half * time_each_half)
    (distance_first_half : ℝ := total_distance - distance_second_half) :
    v = distance_first_half / time_each_half :=
by
  sorry

end NUMINAMATH_GPT_martin_speed_first_half_l1806_180616


namespace NUMINAMATH_GPT_lines_perpendicular_l1806_180675

-- Definition of lines and their relationships
def Line : Type := ℝ × ℝ × ℝ → Prop

variables (a b c : Line)

-- Condition 1: a is perpendicular to b
axiom perp (a b : Line) : Prop
-- Condition 2: b is parallel to c
axiom parallel (b c : Line) : Prop

-- Theorem to prove: 
theorem lines_perpendicular (h1 : perp a b) (h2 : parallel b c) : perp a c :=
sorry

end NUMINAMATH_GPT_lines_perpendicular_l1806_180675


namespace NUMINAMATH_GPT_sum_integers_neg50_to_60_l1806_180632

theorem sum_integers_neg50_to_60 : 
  (Finset.sum (Finset.Icc (-50 : ℤ) 60) id) = 555 := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sum_integers_neg50_to_60_l1806_180632


namespace NUMINAMATH_GPT_combined_number_of_fasteners_l1806_180630

def lorenzo_full_cans_total_fasteners
  (thumbtacks_cans : ℕ)
  (pushpins_cans : ℕ)
  (staples_cans : ℕ)
  (thumbtacks_per_board : ℕ)
  (pushpins_per_board : ℕ)
  (staples_per_board : ℕ)
  (boards_tested : ℕ)
  (thumbtacks_remaining : ℕ)
  (pushpins_remaining : ℕ)
  (staples_remaining : ℕ) :
  ℕ :=
  let thumbtacks_used := thumbtacks_per_board * boards_tested
  let pushpins_used := pushpins_per_board * boards_tested
  let staples_used := staples_per_board * boards_tested
  let thumbtacks_per_can := thumbtacks_used + thumbtacks_remaining
  let pushpins_per_can := pushpins_used + pushpins_remaining
  let staples_per_can := staples_used + staples_remaining
  let total_thumbtacks := thumbtacks_per_can * thumbtacks_cans
  let total_pushpins := pushpins_per_can * pushpins_cans
  let total_staples := staples_per_can * staples_cans
  total_thumbtacks + total_pushpins + total_staples

theorem combined_number_of_fasteners :
  lorenzo_full_cans_total_fasteners 5 3 2 3 2 4 150 45 35 25 = 4730 :=
  by
  sorry

end NUMINAMATH_GPT_combined_number_of_fasteners_l1806_180630


namespace NUMINAMATH_GPT_exists_x0_lt_l1806_180628

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d
noncomputable def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem exists_x0_lt {a b c d p q r s : ℝ} (h1 : r < s) (h2 : s - r > 2)
  (h3 : ∀ x, r < x ∧ x < s → P x a b c d < 0 ∧ Q x p q < 0)
  (h4 : ∀ x, x < r ∨ x > s → P x a b c d >= 0 ∧ Q x p q >= 0) :
  ∃ x0, r < x0 ∧ x0 < s ∧ P x0 a b c d < Q x0 p q :=
sorry

end NUMINAMATH_GPT_exists_x0_lt_l1806_180628


namespace NUMINAMATH_GPT_solution_l1806_180699

def money_problem (x y : ℝ) : Prop :=
  (x + y / 2 = 50) ∧ (y + 2 * x / 3 = 50)

theorem solution :
  ∃ x y : ℝ, money_problem x y ∧ x = 37.5 ∧ y = 25 :=
by
  use 37.5, 25
  sorry

end NUMINAMATH_GPT_solution_l1806_180699


namespace NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l1806_180687

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l1806_180687


namespace NUMINAMATH_GPT_wrap_XL_boxes_per_roll_l1806_180678

-- Conditions
def rolls_per_shirt_box : ℕ := 5
def num_shirt_boxes : ℕ := 20
def num_XL_boxes : ℕ := 12
def cost_per_roll : ℕ := 4
def total_cost : ℕ := 32

-- Prove that one roll of wrapping paper can wrap 3 XL boxes
theorem wrap_XL_boxes_per_roll : (num_XL_boxes / ((total_cost / cost_per_roll) - (num_shirt_boxes / rolls_per_shirt_box))) = 3 := 
sorry

end NUMINAMATH_GPT_wrap_XL_boxes_per_roll_l1806_180678


namespace NUMINAMATH_GPT_farm_width_l1806_180615

theorem farm_width (L W : ℕ) (h1 : 2 * (L + W) = 46) (h2 : W = L + 7) : W = 15 :=
by
  sorry

end NUMINAMATH_GPT_farm_width_l1806_180615


namespace NUMINAMATH_GPT_ratio_of_cube_volumes_l1806_180665

theorem ratio_of_cube_volumes (a b : ℕ) (ha : a = 10) (hb : b = 25) :
  (a^3 : ℚ) / (b^3 : ℚ) = 8 / 125 := by
  sorry

end NUMINAMATH_GPT_ratio_of_cube_volumes_l1806_180665


namespace NUMINAMATH_GPT_remainder_of_150_div_k_l1806_180622

theorem remainder_of_150_div_k (k : ℕ) (hk : k > 0) (h1 : 90 % (k^2) = 10) :
  150 % k = 2 := 
sorry

end NUMINAMATH_GPT_remainder_of_150_div_k_l1806_180622


namespace NUMINAMATH_GPT_all_positive_integers_are_clever_l1806_180606

theorem all_positive_integers_are_clever : ∀ n : ℕ, 0 < n → ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (a^2 - b^2) / (c^2 + d^2) := 
by
  intros n h_pos
  sorry

end NUMINAMATH_GPT_all_positive_integers_are_clever_l1806_180606


namespace NUMINAMATH_GPT_find_S10_l1806_180671

def sequence_sums (S : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 3 * S n - S (n + 1) - 1)

theorem find_S10 (S a : ℕ → ℚ) (h : sequence_sums S a) : S 10 = 513 / 2 :=
  sorry

end NUMINAMATH_GPT_find_S10_l1806_180671


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1806_180663

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 2) → ((x + 1) * (x - 2) > 0) ∧ ¬(∀ y, (y + 1) * (y - 2) > 0 → y > 2) := 
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1806_180663


namespace NUMINAMATH_GPT_least_number_subtracted_l1806_180620

theorem least_number_subtracted (x : ℕ) (y : ℕ) (h : 2590 - x = y) : 
  y % 9 = 6 ∧ y % 11 = 6 ∧ y % 13 = 6 → x = 10 := 
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l1806_180620


namespace NUMINAMATH_GPT_additional_cost_per_person_l1806_180692

-- Define the initial conditions and variables used in the problem
def base_cost := 1700
def discount_per_person := 50
def car_wash_earnings := 500
def initial_friends := 6
def final_friends := initial_friends - 1

-- Calculate initial cost per person with all friends
def discounted_base_cost_initial := base_cost - (initial_friends * discount_per_person)
def total_cost_after_car_wash_initial := discounted_base_cost_initial - car_wash_earnings
def cost_per_person_initial := total_cost_after_car_wash_initial / initial_friends

-- Calculate final cost per person after Brad leaves
def discounted_base_cost_final := base_cost - (final_friends * discount_per_person)
def total_cost_after_car_wash_final := discounted_base_cost_final - car_wash_earnings
def cost_per_person_final := total_cost_after_car_wash_final / final_friends

-- Proving the amount each friend has to pay more after Brad leaves
theorem additional_cost_per_person : cost_per_person_final - cost_per_person_initial = 40 := 
by
  sorry

end NUMINAMATH_GPT_additional_cost_per_person_l1806_180692


namespace NUMINAMATH_GPT_sum_of_squared_distances_range_l1806_180610

theorem sum_of_squared_distances_range
  (φ : ℝ)
  (x : ℝ := 2 * Real.cos φ)
  (y : ℝ := 3 * Real.sin φ)
  (A : ℝ × ℝ := (1, Real.sqrt 3))
  (B : ℝ × ℝ := (-Real.sqrt 3, 1))
  (C : ℝ × ℝ := (-1, -Real.sqrt 3))
  (D : ℝ × ℝ := (Real.sqrt 3, -1))
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2)
  (PD := (x - D.1)^2 + (y - D.2)^2) :
  32 ≤ PA + PB + PC + PD ∧ PA + PB + PC + PD ≤ 52 :=
  by sorry

end NUMINAMATH_GPT_sum_of_squared_distances_range_l1806_180610


namespace NUMINAMATH_GPT_pieces_not_chewed_l1806_180654

theorem pieces_not_chewed : 
  (8 * 7 - 54) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_pieces_not_chewed_l1806_180654


namespace NUMINAMATH_GPT_gcd_sequence_property_l1806_180673

theorem gcd_sequence_property (a : ℕ → ℕ) (m n : ℕ) (h : ∀ m n, m > n → Nat.gcd (a m) (a n) = Nat.gcd (a (m - n)) (a n)) : 
  Nat.gcd (a m) (a n) = a (Nat.gcd m n) :=
by
  sorry

end NUMINAMATH_GPT_gcd_sequence_property_l1806_180673


namespace NUMINAMATH_GPT_mean_value_of_interior_angles_of_quadrilateral_l1806_180602

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end NUMINAMATH_GPT_mean_value_of_interior_angles_of_quadrilateral_l1806_180602


namespace NUMINAMATH_GPT_negation_of_universal_l1806_180603

variable {f g : ℝ → ℝ}

theorem negation_of_universal :
  ¬ (∀ x : ℝ, f x * g x ≠ 0) ↔ ∃ x₀ : ℝ, f x₀ = 0 ∨ g x₀ = 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l1806_180603


namespace NUMINAMATH_GPT_equal_costs_at_45_students_l1806_180693

def ticket_cost_option1 (x : ℕ) : ℝ :=
  x * 30 * 0.8

def ticket_cost_option2 (x : ℕ) : ℝ :=
  (x - 5) * 30 * 0.9

theorem equal_costs_at_45_students : ∀ x : ℕ, ticket_cost_option1 x = ticket_cost_option2 x ↔ x = 45 := 
by
  intro x
  sorry

end NUMINAMATH_GPT_equal_costs_at_45_students_l1806_180693


namespace NUMINAMATH_GPT_Ella_food_each_day_l1806_180647

variable {E : ℕ} -- Define E as the number of pounds of food Ella eats each day

def food_dog_eats (E : ℕ) : ℕ := 4 * E -- Definition of food the dog eats each day

def total_food_eaten_in_10_days (E : ℕ) : ℕ := 10 * E + 10 * (food_dog_eats E) -- Total food (Ella and dog) in 10 days

theorem Ella_food_each_day : total_food_eaten_in_10_days E = 1000 → E = 20 :=
by
  intros h -- Assume the given condition
  sorry -- Skip the actual proof

end NUMINAMATH_GPT_Ella_food_each_day_l1806_180647


namespace NUMINAMATH_GPT_correct_statement_l1806_180643

noncomputable def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  let rec b_aux (m : ℕ) :=
    match m with
    | 0     => 0
    | m + 1 => 1 + 1 / (α m + b_aux m)
  b_aux n

theorem correct_statement (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α :=
by sorry

end NUMINAMATH_GPT_correct_statement_l1806_180643


namespace NUMINAMATH_GPT_power_equal_20mn_l1806_180624

theorem power_equal_20mn (m n : ℕ) (P Q : ℕ) (hP : P = 2^m) (hQ : Q = 5^n) : 
  P^(2 * n) * Q^m = (20^(m * n)) :=
by
  sorry

end NUMINAMATH_GPT_power_equal_20mn_l1806_180624


namespace NUMINAMATH_GPT_largest_tile_size_l1806_180649

theorem largest_tile_size
  (length width : ℕ)
  (H1 : length = 378)
  (H2 : width = 595) :
  Nat.gcd length width = 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_tile_size_l1806_180649


namespace NUMINAMATH_GPT_fish_worth_rice_l1806_180651

variables (f l r : ℝ)

-- Conditions based on the problem statement
def fish_for_bread : Prop := 3 * f = 2 * l
def bread_for_rice : Prop := l = 4 * r

-- Statement to be proven
theorem fish_worth_rice (h₁ : fish_for_bread f l) (h₂ : bread_for_rice l r) : f = (8 / 3) * r :=
  sorry

end NUMINAMATH_GPT_fish_worth_rice_l1806_180651


namespace NUMINAMATH_GPT_two_faucets_fill_60_gallons_l1806_180631

def four_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  4 * (tub_volume / time_minutes) = 120 / 5

def two_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  2 * (tub_volume / time_minutes) = 60 / time_minutes

theorem two_faucets_fill_60_gallons :
  (four_faucets_fill 120 5) → ∃ t: ℕ, two_faucets_fill 60 t ∧ t = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_faucets_fill_60_gallons_l1806_180631


namespace NUMINAMATH_GPT_positive_difference_abs_eq_15_l1806_180689

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_abs_eq_15_l1806_180689


namespace NUMINAMATH_GPT_rectangle_area_correct_l1806_180677

-- Definitions of side lengths
def sideOne : ℝ := 5.9
def sideTwo : ℝ := 3

-- Definition of the area calculation for a rectangle
def rectangleArea (a b : ℝ) : ℝ :=
  a * b

-- The main theorem stating the area is as calculated
theorem rectangle_area_correct :
  rectangleArea sideOne sideTwo = 17.7 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_correct_l1806_180677


namespace NUMINAMATH_GPT_cupcakes_sold_l1806_180686

theorem cupcakes_sold (initial additional final sold : ℕ) (h1 : initial = 14) (h2 : additional = 17) (h3 : final = 25) :
  initial + additional - final = sold :=
by
  sorry

end NUMINAMATH_GPT_cupcakes_sold_l1806_180686


namespace NUMINAMATH_GPT_sum_ages_in_five_years_l1806_180652

theorem sum_ages_in_five_years (L J : ℕ) (hL : L = 13) (h_relation : L = 2 * J + 3) : 
  (L + 5) + (J + 5) = 28 := 
by 
  sorry

end NUMINAMATH_GPT_sum_ages_in_five_years_l1806_180652


namespace NUMINAMATH_GPT_total_number_of_red_and_white_jelly_beans_in_fishbowl_l1806_180655

def number_of_red_jelly_beans_in_bag := 24
def number_of_white_jelly_beans_in_bag := 18
def number_of_bags := 3

theorem total_number_of_red_and_white_jelly_beans_in_fishbowl :
  number_of_red_jelly_beans_in_bag * number_of_bags + number_of_white_jelly_beans_in_bag * number_of_bags = 126 := by
  sorry

end NUMINAMATH_GPT_total_number_of_red_and_white_jelly_beans_in_fishbowl_l1806_180655


namespace NUMINAMATH_GPT_lion_king_cost_l1806_180650

theorem lion_king_cost
  (LK_earned : ℕ := 200) -- The Lion King earned 200 million
  (LK_profit : ℕ := 190) -- The Lion King profit calculated from half of Star Wars' profit
  (SW_cost : ℕ := 25)    -- Star Wars cost 25 million
  (SW_earned : ℕ := 405) -- Star Wars earned 405 million
  (SW_profit : SW_earned - SW_cost = 380) -- Star Wars profit
  (LK_profit_from_SW : LK_profit = 1/2 * (SW_earned - SW_cost)) -- The Lion King profit calculation
  (LK_cost : ℕ := LK_earned - LK_profit) -- The Lion King cost calculation
  : LK_cost = 10 := 
sorry

end NUMINAMATH_GPT_lion_king_cost_l1806_180650


namespace NUMINAMATH_GPT_find_x_l1806_180641

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end NUMINAMATH_GPT_find_x_l1806_180641


namespace NUMINAMATH_GPT_find_a7_l1806_180672

variable {a : ℕ → ℕ}  -- Define the geometric sequence as a function from natural numbers to natural numbers.
variable (h_geo_seq : ∀ (n k : ℕ), a n ^ 2 = a (n - k) * a (n + k)) -- property of geometric sequences
variable (h_a3 : a 3 = 2) -- given a₃ = 2
variable (h_a5 : a 5 = 8) -- given a₅ = 8

theorem find_a7 : a 7 = 32 :=
by
  sorry

end NUMINAMATH_GPT_find_a7_l1806_180672


namespace NUMINAMATH_GPT_find_numbers_l1806_180679

theorem find_numbers (A B C D : ℚ) 
  (h1 : A + B = 44)
  (h2 : 5 * A = 6 * B)
  (h3 : C = 2 * (A - B))
  (h4 : D = (A + B + C) / 3 + 3) :
  A = 24 ∧ B = 20 ∧ C = 8 ∧ D = 61 / 3 := 
  by 
    sorry

end NUMINAMATH_GPT_find_numbers_l1806_180679


namespace NUMINAMATH_GPT_min_n_satisfies_inequality_l1806_180685

theorem min_n_satisfies_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ n * (x^4 + y^4 + z^4)) ∧ (n = 3) :=
by
  sorry

end NUMINAMATH_GPT_min_n_satisfies_inequality_l1806_180685


namespace NUMINAMATH_GPT_two_pow_gt_twice_n_plus_one_l1806_180660

theorem two_pow_gt_twice_n_plus_one (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
sorry

end NUMINAMATH_GPT_two_pow_gt_twice_n_plus_one_l1806_180660


namespace NUMINAMATH_GPT_one_positive_real_solution_l1806_180639

theorem one_positive_real_solution : 
    ∃! x : ℝ, 0 < x ∧ (x ^ 10 + 7 * x ^ 9 + 14 * x ^ 8 + 1729 * x ^ 7 - 1379 * x ^ 6 = 0) :=
sorry

end NUMINAMATH_GPT_one_positive_real_solution_l1806_180639


namespace NUMINAMATH_GPT_decimal_0_0_1_7_eq_rational_l1806_180608

noncomputable def infinite_loop_decimal_to_rational_series (a : ℚ) (r : ℚ) : ℚ :=
  a / (1 - r)

theorem decimal_0_0_1_7_eq_rational :
  infinite_loop_decimal_to_rational_series (17 / 1000) (1 / 100) = 17 / 990 :=
by
  sorry

end NUMINAMATH_GPT_decimal_0_0_1_7_eq_rational_l1806_180608


namespace NUMINAMATH_GPT_number_of_streams_l1806_180676

theorem number_of_streams (S A B C D : Type) (f : S → A) (f1 : A → B) :
  (∀ (x : ℕ), x = 1000 → 
  (x * 375 / 1000 = 375 ∧ x * 625 / 1000 = 625) ∧ 
  (S ≠ C ∧ S ≠ D ∧ C ≠ D)) →
  -- Introduce some conditions to represent the described transition process
  -- Specifically the conditions mentioning the lakes and transitions 
  ∀ (transition_count : ℕ), 
    (transition_count = 4) →
    ∃ (number_of_streams : ℕ), number_of_streams = 3 := 
sorry

end NUMINAMATH_GPT_number_of_streams_l1806_180676


namespace NUMINAMATH_GPT_man_l1806_180694

theorem man's_salary 
  (food_fraction : ℚ := 1/5) 
  (rent_fraction : ℚ := 1/10) 
  (clothes_fraction : ℚ := 3/5) 
  (remaining_money : ℚ := 15000) 
  (S : ℚ) :
  (S * (1 - (food_fraction + rent_fraction + clothes_fraction)) = remaining_money) →
  S = 150000 := 
by
  intros h1
  sorry

end NUMINAMATH_GPT_man_l1806_180694


namespace NUMINAMATH_GPT_evaluate_expression_l1806_180618

theorem evaluate_expression (x y z : ℝ) (hx : x ≠ 3) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 3) / (7 - z) * (y - 5) / (3 - x) * (z - 7) / (5 - y) = -1 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1806_180618


namespace NUMINAMATH_GPT_contractor_absent_days_l1806_180682

theorem contractor_absent_days (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 25 * x - 7.5 * y = 685) : 
  y = 2 :=
by
  sorry

end NUMINAMATH_GPT_contractor_absent_days_l1806_180682


namespace NUMINAMATH_GPT_Proof_l1806_180612

-- Definitions for the conditions
def Snakes : Type := {s : Fin 20 // s < 20}
def Purple (s : Snakes) : Prop := s.val < 6
def Happy (s : Snakes) : Prop := s.val >= 6 ∧ s.val < 14
def CanAdd (s : Snakes) : Prop := ∃ h ∈ Finset.Ico 6 14, h = s.val
def CanSubtract (s : Snakes) : Prop := ¬Purple s

-- Conditions extraction
axiom SomeHappyCanAdd : ∃ s : Snakes, Happy s ∧ CanAdd s
axiom NoPurpleCanSubtract : ∀ s : Snakes, Purple s → ¬CanSubtract s
axiom CantSubtractCantAdd : ∀ s : Snakes, ¬CanSubtract s → ¬CanAdd s

-- Theorem statement depending on conditions
theorem Proof :
    (∀ s : Snakes, CanSubtract s → ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬CanSubtract s) :=
by {
  sorry -- Proof required here
}

end NUMINAMATH_GPT_Proof_l1806_180612


namespace NUMINAMATH_GPT_product_of_roots_eq_neg_14_l1806_180667

theorem product_of_roots_eq_neg_14 :
  ∀ (x : ℝ), 25 * x^2 + 60 * x - 350 = 0 → ((-350) / 25) = -14 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_product_of_roots_eq_neg_14_l1806_180667


namespace NUMINAMATH_GPT_total_area_of_hexagon_is_693_l1806_180640

-- Conditions
def hexagon_side1_length := 3
def hexagon_side2_length := 2
def angle_between_length3_sides := 120
def all_internal_triangles_are_equilateral := true
def number_of_triangles := 6

-- Define the problem statement
theorem total_area_of_hexagon_is_693 
  (a1 : hexagon_side1_length = 3)
  (a2 : hexagon_side2_length = 2)
  (a3 : angle_between_length3_sides = 120)
  (a4 : all_internal_triangles_are_equilateral = true)
  (a5 : number_of_triangles = 6) :
  total_area_of_hexagon = 693 :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_hexagon_is_693_l1806_180640
