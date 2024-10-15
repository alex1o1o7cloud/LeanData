import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_property_l1780_178051

variable {a_n : ℕ → ℝ}

theorem geometric_sequence_property (h1 : ∀ m n p q : ℕ, m + n = p + q → a_n m * a_n n = a_n p * a_n q)
    (h2 : a_n 4 * a_n 5 * a_n 6 = 27) : a_n 1 * a_n 9 = 9 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l1780_178051


namespace NUMINAMATH_GPT_not_divisible_l1780_178091

-- Defining the necessary conditions
variable (m : ℕ)

theorem not_divisible (m : ℕ) : ¬ (1000^m - 1 ∣ 1978^m - 1) :=
sorry

end NUMINAMATH_GPT_not_divisible_l1780_178091


namespace NUMINAMATH_GPT_factor_polynomial_l1780_178029

theorem factor_polynomial (z : ℝ) : (70 * z ^ 20 + 154 * z ^ 40 + 224 * z ^ 60) = 14 * z ^ 20 * (5 + 11 * z ^ 20 + 16 * z ^ 40) := 
sorry

end NUMINAMATH_GPT_factor_polynomial_l1780_178029


namespace NUMINAMATH_GPT_pencil_eraser_cost_l1780_178027

/-- Oscar buys 13 pencils and 3 erasers for 100 cents. A pencil costs more than an eraser, 
    and both items cost a whole number of cents. 
    We need to prove that the total cost of one pencil and one eraser is 10 cents. -/
theorem pencil_eraser_cost (p e : ℕ) (h1 : 13 * p + 3 * e = 100) (h2 : p > e) : p + e = 10 :=
sorry

end NUMINAMATH_GPT_pencil_eraser_cost_l1780_178027


namespace NUMINAMATH_GPT_troy_needs_more_money_to_buy_computer_l1780_178023

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end NUMINAMATH_GPT_troy_needs_more_money_to_buy_computer_l1780_178023


namespace NUMINAMATH_GPT_perp_lines_implies_values_l1780_178048

variable (a : ℝ)

def line1_perpendicular (a : ℝ) : Prop :=
  (1 - a) * (2 * a + 3) + a * (a - 1) = 0

theorem perp_lines_implies_values (h : line1_perpendicular a) :
  a = 1 ∨ a = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_perp_lines_implies_values_l1780_178048


namespace NUMINAMATH_GPT_price_of_each_cake_is_correct_l1780_178084

-- Define the conditions
def total_flour : ℕ := 6
def flour_for_cakes : ℕ := 4
def flour_per_cake : ℚ := 0.5
def remaining_flour := total_flour - flour_for_cakes
def flour_per_cupcake : ℚ := 1 / 5
def total_earnings : ℚ := 30
def cupcake_price : ℚ := 1

-- Number of cakes and cupcakes
def number_of_cakes := flour_for_cakes / flour_per_cake
def number_of_cupcakes := remaining_flour / flour_per_cupcake

-- Earnings from cupcakes
def earnings_from_cupcakes := number_of_cupcakes * cupcake_price

-- Earnings from cakes
def earnings_from_cakes := total_earnings - earnings_from_cupcakes

-- Price per cake
def price_per_cake := earnings_from_cakes / number_of_cakes

-- Final statement to prove
theorem price_of_each_cake_is_correct : price_per_cake = 2.50 := by
  sorry

end NUMINAMATH_GPT_price_of_each_cake_is_correct_l1780_178084


namespace NUMINAMATH_GPT_bill_difference_l1780_178026

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end NUMINAMATH_GPT_bill_difference_l1780_178026


namespace NUMINAMATH_GPT_goshawk_eurasian_reserve_l1780_178088

theorem goshawk_eurasian_reserve (B : ℝ)
  (h1 : 0.30 * B + 0.28 * B + K * 0.28 * B = 0.65 * B)
  : K = 0.25 :=
by sorry

end NUMINAMATH_GPT_goshawk_eurasian_reserve_l1780_178088


namespace NUMINAMATH_GPT_third_divisor_l1780_178012

/-- 
Given that the new number after subtracting 7 from 3,381 leaves a remainder of 8 when divided by 9 
and 11, prove that the third divisor that also leaves a remainder of 8 is 17.
-/
theorem third_divisor (x : ℕ) (h1 : x = 3381 - 7)
                      (h2 : x % 9 = 8)
                      (h3 : x % 11 = 8) :
  ∃ (d : ℕ), d = 17 ∧ x % d = 8 := sorry

end NUMINAMATH_GPT_third_divisor_l1780_178012


namespace NUMINAMATH_GPT_power_difference_of_squares_l1780_178053

theorem power_difference_of_squares : (((7^2 - 3^2) : ℤ)^4) = 2560000 := by
  sorry

end NUMINAMATH_GPT_power_difference_of_squares_l1780_178053


namespace NUMINAMATH_GPT_range_of_decreasing_function_l1780_178099

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_decreasing_function (a : ℝ) :
  (∀ x : ℝ, x < 3 → (deriv (f a) x) ≤ 0) ↔ 0 ≤ a ∧ a ≤ 3/4 := 
sorry

end NUMINAMATH_GPT_range_of_decreasing_function_l1780_178099


namespace NUMINAMATH_GPT_find_number_of_students_l1780_178087

theorem find_number_of_students 
    (N T : ℕ) 
    (h1 : T = 80 * N)
    (h2 : (T - 350) / (N - 5) = 90) : 
    N = 10 :=
sorry

end NUMINAMATH_GPT_find_number_of_students_l1780_178087


namespace NUMINAMATH_GPT_john_boxes_l1780_178015

theorem john_boxes
  (stan_boxes : ℕ)
  (joseph_boxes : ℕ)
  (jules_boxes : ℕ)
  (john_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_boxes = stan_boxes - 80 * stan_boxes / 100)
  (h3 : jules_boxes = joseph_boxes + 5)
  (h4 : john_boxes = jules_boxes + 20 * jules_boxes / 100) :
  john_boxes = 30 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_john_boxes_l1780_178015


namespace NUMINAMATH_GPT_house_to_car_ratio_l1780_178042

-- Define conditions
def cost_per_night := 4000
def nights_at_hotel := 2
def cost_of_car := 30000
def total_value_of_treats := 158000

-- Prove that the ratio of the value of the house to the value of the car is 4:1
theorem house_to_car_ratio : 
  (total_value_of_treats - (nights_at_hotel * cost_per_night + cost_of_car)) / cost_of_car = 4 := by
  sorry

end NUMINAMATH_GPT_house_to_car_ratio_l1780_178042


namespace NUMINAMATH_GPT_intersection_A_B_l1780_178040

open Set

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 1}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_A_B_l1780_178040


namespace NUMINAMATH_GPT_find_three_power_l1780_178057

theorem find_three_power (m n : ℕ) (h₁: 3^m = 4) (h₂: 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end NUMINAMATH_GPT_find_three_power_l1780_178057


namespace NUMINAMATH_GPT_old_manufacturing_cost_l1780_178081

theorem old_manufacturing_cost (P : ℝ) :
  (50 : ℝ) = P * 0.50 →
  (0.65 : ℝ) * P = 65 :=
by
  intros hp₁
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_old_manufacturing_cost_l1780_178081


namespace NUMINAMATH_GPT_coprime_divisors_imply_product_divisor_l1780_178068

theorem coprime_divisors_imply_product_divisor 
  (a b n : ℕ) (h_coprime : Nat.gcd a b = 1)
  (h_a_div_n : a ∣ n) (h_b_div_n : b ∣ n) : a * b ∣ n :=
by
  sorry

end NUMINAMATH_GPT_coprime_divisors_imply_product_divisor_l1780_178068


namespace NUMINAMATH_GPT_max_large_sculptures_l1780_178063

theorem max_large_sculptures (x y : ℕ) (h1 : 1 * x = x) 
  (h2 : 3 * y = y + y + y) 
  (h3 : ∃ n, n = (x + y) / 2) 
  (h4 : x + 3 * y + (x + y) / 2 ≤ 30) 
  (h5 : x > y) : 
  y ≤ 4 := 
sorry

end NUMINAMATH_GPT_max_large_sculptures_l1780_178063


namespace NUMINAMATH_GPT_min_focal_length_of_hyperbola_l1780_178000

theorem min_focal_length_of_hyperbola
  (a b k : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_area : k * b = 8) :
  2 * Real.sqrt (a^2 + b^2) = 8 :=
sorry -- proof to be completed

end NUMINAMATH_GPT_min_focal_length_of_hyperbola_l1780_178000


namespace NUMINAMATH_GPT_total_students_in_faculty_l1780_178050

theorem total_students_in_faculty (N A B : ℕ) (hN : N = 230) (hA : A = 423) (hB : B = 134)
  (h80_percent : (N + A - B) = 80 / 100 * T) : T = 649 := 
by
  sorry

end NUMINAMATH_GPT_total_students_in_faculty_l1780_178050


namespace NUMINAMATH_GPT_sin_lower_bound_lt_l1780_178007

theorem sin_lower_bound_lt (a : ℝ) (h : ∃ x : ℝ, Real.sin x < a) : a > -1 :=
sorry

end NUMINAMATH_GPT_sin_lower_bound_lt_l1780_178007


namespace NUMINAMATH_GPT_sixteen_k_plus_eight_not_perfect_square_l1780_178060

theorem sixteen_k_plus_eight_not_perfect_square (k : ℕ) (hk : 0 < k) : ¬ ∃ m : ℕ, (16 * k + 8) = m * m := sorry

end NUMINAMATH_GPT_sixteen_k_plus_eight_not_perfect_square_l1780_178060


namespace NUMINAMATH_GPT_value_of_A_l1780_178069

theorem value_of_A
  (A B C D E F G H I J : ℕ)
  (h_diff : ∀ x y : ℕ, x ≠ y → x ≠ y)
  (h_decreasing_ABC : A > B ∧ B > C)
  (h_decreasing_DEF : D > E ∧ E > F)
  (h_decreasing_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_odd_DEF : D % 2 = 1 ∧ E % 2 = 1 ∧ F % 2 = 1 ∧ E = D - 2 ∧ F = E - 2)
  (h_consecutive_even_GHIJ : G % 2 = 0 ∧ H % 2 = 0 ∧ I % 2 = 0 ∧ J % 2 = 0 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) : 
  A = 8 :=
sorry

end NUMINAMATH_GPT_value_of_A_l1780_178069


namespace NUMINAMATH_GPT_custom_op_evaluation_l1780_178067

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : custom_op 6 5 - custom_op 5 6 = -4 := by
  sorry

end NUMINAMATH_GPT_custom_op_evaluation_l1780_178067


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1780_178078

variable {a_n : ℕ → ℕ} -- the arithmetic sequence

-- Define condition
def condition (a : ℕ → ℕ) : Prop :=
  a 1 + a 5 + a 9 = 18

-- The sum of the first n terms of arithmetic sequence is S_n
def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

-- The goal is to prove that S 9 = 54
theorem arithmetic_sequence_sum (h : condition a_n) : S 9 a_n = 54 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1780_178078


namespace NUMINAMATH_GPT_basket_weight_l1780_178039

variable (B P : ℕ)

theorem basket_weight (h1 : B + P = 62) (h2 : B + P / 2 = 34) : B = 6 :=
by
  sorry

end NUMINAMATH_GPT_basket_weight_l1780_178039


namespace NUMINAMATH_GPT_problem_a_problem_b_l1780_178072

theorem problem_a (p : ℕ) (hp : Nat.Prime p) : 
  (∃ x : ℕ, (7^(p-1) - 1) = p * x^2) ↔ p = 3 := 
by
  sorry

theorem problem_b (p : ℕ) (hp : Nat.Prime p) : 
  ¬ ∃ x : ℕ, (11^(p-1) - 1) = p * x^2 := 
by
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_l1780_178072


namespace NUMINAMATH_GPT_area_of_smallest_square_l1780_178009

-- Define a circle with a given radius
def radius : ℝ := 7

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the side length of the smallest square that can contain the circle
def side_length : ℝ := diameter

-- Define the area of the square as the side length squared
def area_of_square : ℝ := side_length ^ 2

-- State the theorem: the area of the smallest square that contains a circle of radius 7 is 196
theorem area_of_smallest_square : area_of_square = 196 := by
    sorry

end NUMINAMATH_GPT_area_of_smallest_square_l1780_178009


namespace NUMINAMATH_GPT_remainder_of_n_plus_3255_l1780_178006

theorem remainder_of_n_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_n_plus_3255_l1780_178006


namespace NUMINAMATH_GPT_worker_savings_l1780_178032

theorem worker_savings (P : ℝ) (f : ℝ) (h : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  have h1 : 12 * f * P = 4 * (1 - f) * P := h
  have h2 : P ≠ 0 := sorry  -- P should not be 0 for the worker to have a meaningful income.
  field_simp [h2] at h1
  linarith

end NUMINAMATH_GPT_worker_savings_l1780_178032


namespace NUMINAMATH_GPT_number_of_students_playing_soccer_l1780_178074

-- Definitions of the conditions
def total_students : ℕ := 500
def total_boys : ℕ := 350
def percent_boys_playing_soccer : ℚ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- To be proved
theorem number_of_students_playing_soccer :
  ∃ (S : ℕ), S = 250 ∧ 0.14 * (S : ℚ) = 35 :=
sorry

end NUMINAMATH_GPT_number_of_students_playing_soccer_l1780_178074


namespace NUMINAMATH_GPT_eden_initial_bears_l1780_178013

theorem eden_initial_bears (d_total : ℕ) (d_favorite : ℕ) (sisters : ℕ) (eden_after : ℕ) (each_share : ℕ)
  (h1 : d_total = 20)
  (h2 : d_favorite = 8)
  (h3 : sisters = 3)
  (h4 : eden_after = 14)
  (h5 : each_share = (d_total - d_favorite) / sisters)
  : (eden_after - each_share) = 10 :=
by
  sorry

end NUMINAMATH_GPT_eden_initial_bears_l1780_178013


namespace NUMINAMATH_GPT_geometric_sum_n_is_4_l1780_178083

theorem geometric_sum_n_is_4 
  (a r : ℚ) (n : ℕ) (S_n : ℚ) 
  (h1 : a = 1) 
  (h2 : r = 1 / 4) 
  (h3 : S_n = (a * (1 - r^n)) / (1 - r)) 
  (h4 : S_n = 85 / 64) : 
  n = 4 := 
sorry

end NUMINAMATH_GPT_geometric_sum_n_is_4_l1780_178083


namespace NUMINAMATH_GPT_sum_midpoints_x_coordinates_is_15_l1780_178096

theorem sum_midpoints_x_coordinates_is_15 :
  ∀ (a b : ℝ), a + 2 * b = 15 → 
  (a + 2 * b) = 15 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_sum_midpoints_x_coordinates_is_15_l1780_178096


namespace NUMINAMATH_GPT_simplify_expression_l1780_178098

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1780_178098


namespace NUMINAMATH_GPT_gcd_lcm_product_eq_prod_l1780_178034

theorem gcd_lcm_product_eq_prod (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
sorry

end NUMINAMATH_GPT_gcd_lcm_product_eq_prod_l1780_178034


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1780_178035

noncomputable def area_triangle (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C
noncomputable def area_quadrilateral (e f φ : ℝ) : ℝ := (1/2) * e * f * Real.sin φ

theorem problem_1 (a b C : ℝ) (hC : Real.sin C ≤ 1) : 
  area_triangle a b C ≤ (a^2 + b^2) / 4 :=
sorry

theorem problem_2 (e f φ : ℝ) (hφ : Real.sin φ ≤ 1) : 
  area_quadrilateral e f φ ≤ (e^2 + f^2) / 4 :=
sorry

theorem problem_3 (a b C c d D : ℝ) 
  (hC : Real.sin C ≤ 1) 
  (hD : Real.sin D ≤ 1) :
  area_triangle a b C + area_triangle c d D ≤ (a^2 + b^2 + c^2 + d^2) / 4 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1780_178035


namespace NUMINAMATH_GPT_solve_laundry_problem_l1780_178010

def laundry_problem : Prop :=
  let total_weight := 20
  let clothes_weight := 5
  let detergent_per_scoop := 0.02
  let initial_detergent := 2 * detergent_per_scoop
  let optimal_ratio := 0.004
  let additional_detergent := 0.02
  let additional_water := 14.94
  let total_detergent := initial_detergent + additional_detergent
  let final_amount := clothes_weight + initial_detergent + additional_detergent + additional_water
  final_amount = total_weight ∧ total_detergent / (total_weight - clothes_weight) = optimal_ratio

theorem solve_laundry_problem : laundry_problem :=
by 
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_solve_laundry_problem_l1780_178010


namespace NUMINAMATH_GPT_value_of_a_l1780_178044

theorem value_of_a (a : ℝ) : (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  intro h
  have h1 : 2 = a - 1 := sorry
  have h2 : 4 = a + 1 := sorry
  have h3 : a = 3 := sorry
  exact h3

end NUMINAMATH_GPT_value_of_a_l1780_178044


namespace NUMINAMATH_GPT_range_of_a_l1780_178090

theorem range_of_a (a : ℝ) (h : Real.sqrt ((2 * a - 1)^2) = 1 - 2 * a) : a ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1780_178090


namespace NUMINAMATH_GPT_chimps_moved_l1780_178041

theorem chimps_moved (total_chimps : ℕ) (chimps_staying : ℕ) (chimps_moved : ℕ) 
  (h_total : total_chimps = 45)
  (h_staying : chimps_staying = 27) :
  chimps_moved = 18 :=
by
  sorry

end NUMINAMATH_GPT_chimps_moved_l1780_178041


namespace NUMINAMATH_GPT_find_current_l1780_178017

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end NUMINAMATH_GPT_find_current_l1780_178017


namespace NUMINAMATH_GPT_man_l1780_178025

theorem man's_present_age (P : ℝ) 
  (h1 : P = (4/5) * P + 10)
  (h2 : P = (3/2.5) * P - 10) :
  P = 50 :=
sorry

end NUMINAMATH_GPT_man_l1780_178025


namespace NUMINAMATH_GPT_sixteenth_answer_is_three_l1780_178008

theorem sixteenth_answer_is_three (total_members : ℕ)
  (answers_1 answers_2 answers_3 : ℕ) 
  (h_total : total_members = 16) 
  (h_answers_1 : answers_1 = 6) 
  (h_answers_2 : answers_2 = 6) 
  (h_answers_3 : answers_3 = 3) :
  ∃ answer : ℕ, answer = 3 ∧ (answers_1 + answers_2 + answers_3 + 1 = total_members) :=
sorry

end NUMINAMATH_GPT_sixteenth_answer_is_three_l1780_178008


namespace NUMINAMATH_GPT_chord_eq_line_l1780_178077

theorem chord_eq_line (x y : ℝ)
  (h_ellipse : (x^2) / 16 + (y^2) / 4 = 1)
  (h_midpoint : ∃ x1 y1 x2 y2 : ℝ, 
    ((x1^2) / 16 + (y1^2) / 4 = 1) ∧ 
    ((x2^2) / 16 + (y2^2) / 4 = 1) ∧ 
    (x1 + x2) / 2 = 2 ∧ 
    (y1 + y2) / 2 = 1) :
  x + 2 * y - 4 = 0 :=
sorry

end NUMINAMATH_GPT_chord_eq_line_l1780_178077


namespace NUMINAMATH_GPT_best_fit_slope_eq_l1780_178056

theorem best_fit_slope_eq :
  let x1 := 150
  let y1 := 2
  let x2 := 160
  let y2 := 3
  let x3 := 170
  let y3 := 4
  (x2 - x1 = 10 ∧ x3 - x2 = 10) →
  let slope := (x1 - x2) * (y1 - y2) + (x3 - x2) * (y3 - y2) / (x1 - x2)^2 + (x3 - x2)^2
  slope = 1 / 10 :=
sorry

end NUMINAMATH_GPT_best_fit_slope_eq_l1780_178056


namespace NUMINAMATH_GPT_bottles_produced_l1780_178073

/-- 
14 machines produce 2520 bottles in 4 minutes, given that 6 machines produce 270 bottles per minute. 
-/
theorem bottles_produced (rate_6_machines : Nat) (bottles_per_minute : Nat) 
  (rate_one_machine : Nat) (rate_14_machines : Nat) (total_production : Nat) : 
  rate_6_machines = 6 ∧ bottles_per_minute = 270 ∧ rate_one_machine = bottles_per_minute / rate_6_machines 
  ∧ rate_14_machines = 14 * rate_one_machine ∧ total_production = rate_14_machines * 4 → 
  total_production = 2520 :=
sorry

end NUMINAMATH_GPT_bottles_produced_l1780_178073


namespace NUMINAMATH_GPT_triangle_perimeter_l1780_178016

theorem triangle_perimeter (r AP PB x : ℕ) (h_r : r = 14) (h_AP : AP = 20) (h_PB : PB = 30) (h_BC_gt_AC : ∃ BC AC : ℝ, BC > AC)
: ∃ s : ℕ, s = (25 + x) → 2 * s = 50 + 2 * x :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1780_178016


namespace NUMINAMATH_GPT_ivan_total_money_l1780_178028

-- Define the value of a dime in cents
def value_of_dime : ℕ := 10

-- Define the value of a penny in cents
def value_of_penny : ℕ := 1

-- Define the number of dimes per piggy bank
def dimes_per_piggy_bank : ℕ := 50

-- Define the number of pennies per piggy bank
def pennies_per_piggy_bank : ℕ := 100

-- Define the number of piggy banks
def number_of_piggy_banks : ℕ := 2

-- Define the total value in dollars
noncomputable def total_value_in_dollars : ℕ := 
  (dimes_per_piggy_bank * value_of_dime + pennies_per_piggy_bank * value_of_penny) * number_of_piggy_banks / 100

theorem ivan_total_money : total_value_in_dollars = 12 := by
  sorry

end NUMINAMATH_GPT_ivan_total_money_l1780_178028


namespace NUMINAMATH_GPT_sector_area_correct_l1780_178066

-- Definitions based on the conditions
def sector_perimeter := 16 -- cm
def central_angle := 2 -- radians
def radius := 4 -- The radius computed from perimeter condition

-- Lean 4 statement to prove the equivalent math problem
theorem sector_area_correct : ∃ (s : ℝ), 
  (∀ (r : ℝ), (2 * r + r * central_angle = sector_perimeter → r = 4) → 
  (s = (1 / 2) * central_angle * (radius) ^ 2) → 
  s = 16) :=
by 
  sorry

end NUMINAMATH_GPT_sector_area_correct_l1780_178066


namespace NUMINAMATH_GPT_dropped_test_score_l1780_178004

theorem dropped_test_score (A B C D : ℕ) 
  (h1 : A + B + C + D = 280) 
  (h2 : A + B + C = 225) : 
  D = 55 := 
by sorry

end NUMINAMATH_GPT_dropped_test_score_l1780_178004


namespace NUMINAMATH_GPT_problem1_problem2_l1780_178059

-- Given conditions
variables (x y : ℝ)

-- Problem 1: Prove that ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = -xy
theorem problem1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = - (x * y) :=
sorry

-- Problem 2: Prove that (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2
theorem problem2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1780_178059


namespace NUMINAMATH_GPT_find_k_l1780_178036

-- Define the matrix M
def M (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 3], ![0, 4, -k], ![3, -1, 2]]

-- Define the problem statement
theorem find_k (k : ℝ) (h : Matrix.det (M k) = -20) : k = 0 := by
  sorry

end NUMINAMATH_GPT_find_k_l1780_178036


namespace NUMINAMATH_GPT_isosceles_triangle_height_ratio_l1780_178011

theorem isosceles_triangle_height_ratio (b1 h1 b2 h2 : ℝ) 
  (A1 : ℝ := 1/2 * b1 * h1) (A2 : ℝ := 1/2 * b2 * h2)
  (area_ratio : A1 / A2 = 16 / 49)
  (similar : b1 / b2 = h1 / h2) : 
  h1 / h2 = 4 / 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_height_ratio_l1780_178011


namespace NUMINAMATH_GPT_find_square_l1780_178076

theorem find_square (s : ℕ) : 
    (7863 / 13 = 604 + (s / 13)) → s = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_square_l1780_178076


namespace NUMINAMATH_GPT_avg_production_last_5_days_l1780_178022

theorem avg_production_last_5_days
  (avg_first_25_days : ℕ)
  (total_days : ℕ)
  (avg_entire_month : ℕ)
  (h1 : avg_first_25_days = 60)
  (h2 : total_days = 30)
  (h3 : avg_entire_month = 58) : 
  (total_days * avg_entire_month - 25 * avg_first_25_days) / 5 = 48 := 
by
  sorry

end NUMINAMATH_GPT_avg_production_last_5_days_l1780_178022


namespace NUMINAMATH_GPT_sum_of_last_two_digits_l1780_178019

theorem sum_of_last_two_digits (a b : ℕ) (ha: a = 6) (hb: b = 10) :
  ((a^15 + b^15) % 100) = 0 :=
by
  -- ha, hb represent conditions given.
  sorry

end NUMINAMATH_GPT_sum_of_last_two_digits_l1780_178019


namespace NUMINAMATH_GPT_find_rectangle_length_l1780_178037

theorem find_rectangle_length (L W : ℕ) (h_area : L * W = 300) (h_perimeter : 2 * L + 2 * W = 70) : L = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_rectangle_length_l1780_178037


namespace NUMINAMATH_GPT_paint_time_l1780_178031

theorem paint_time (n1 t1 n2 : ℕ) (k : ℕ) (h : n1 * t1 = k) (h1 : 5 * 4 = k) (h2 : n2 = 6) : (k / n2) = 10 / 3 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_paint_time_l1780_178031


namespace NUMINAMATH_GPT_city_population_l1780_178058

theorem city_population (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 :=
by
  sorry

end NUMINAMATH_GPT_city_population_l1780_178058


namespace NUMINAMATH_GPT_area_of_triangle_l1780_178038

theorem area_of_triangle (x : ℝ) :
  let t1_area := 16
  let t2_area := 25
  let t3_area := 64
  let total_area_factor := t1_area + t2_area + t3_area
  let side_factor := 17 * 17
  ΔABC_area = side_factor * total_area_factor :=
by {
  -- Placeholder to complete the proof
  sorry
}

end NUMINAMATH_GPT_area_of_triangle_l1780_178038


namespace NUMINAMATH_GPT_stamps_in_album_l1780_178065

theorem stamps_in_album (n : ℕ) : 
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ 
  n % 6 = 5 ∧ n % 7 = 6 ∧ n % 8 = 7 ∧ n % 9 = 8 ∧ 
  n % 10 = 9 ∧ n < 3000 → n = 2519 :=
by
  sorry

end NUMINAMATH_GPT_stamps_in_album_l1780_178065


namespace NUMINAMATH_GPT_option_a_correct_l1780_178055

-- Define the variables as real numbers
variables {a b : ℝ}

-- Define the main theorem to prove
theorem option_a_correct : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  -- start the proof block
  sorry

end NUMINAMATH_GPT_option_a_correct_l1780_178055


namespace NUMINAMATH_GPT_distance_to_cheaper_gas_station_l1780_178086

-- Define the conditions
def miles_per_gallon : ℕ := 3
def initial_gallons : ℕ := 12
def additional_gallons : ℕ := 18

-- Define the question and proof statement
theorem distance_to_cheaper_gas_station : 
  (initial_gallons + additional_gallons) * miles_per_gallon = 90 := by
  sorry

end NUMINAMATH_GPT_distance_to_cheaper_gas_station_l1780_178086


namespace NUMINAMATH_GPT_find_first_number_l1780_178024

theorem find_first_number
  (x y : ℝ)
  (h1 : y = 3.0)
  (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end NUMINAMATH_GPT_find_first_number_l1780_178024


namespace NUMINAMATH_GPT_mail_cars_in_train_l1780_178075

theorem mail_cars_in_train (n : ℕ) (hn : n % 2 = 0) (hfront : 1 ≤ n ∧ n ≤ 20)
  (hclose : ∀ i, 1 ≤ i ∧ i < n → (∃ j, i < j ∧ j ≤ 20))
  (hlast : 4 * n ≤ 20)
  (hconn : ∀ k, (k = 4 ∨ k = 5 ∨ k = 15 ∨ k = 16) → 
                  (∃ j, j = k + 1 ∨ j = k - 1)) :
  ∃ (i : ℕ) (j : ℕ), i = 4 ∧ j = 16 :=
by
  sorry

end NUMINAMATH_GPT_mail_cars_in_train_l1780_178075


namespace NUMINAMATH_GPT_parabola_equation_l1780_178070

theorem parabola_equation (a b c : ℝ) (h1 : a^2 = 3) (h2 : b^2 = 1) (h3 : c^2 = a^2 + b^2) : 
  (c = 2) → (vertex = 0) → (focus = 2) → ∀ x y, y^2 = 16 * x := 
by 
  sorry

end NUMINAMATH_GPT_parabola_equation_l1780_178070


namespace NUMINAMATH_GPT_expand_polynomial_l1780_178020

theorem expand_polynomial (x : ℝ) :
    (5*x^2 + 3*x - 7) * (4*x^3) = 20*x^5 + 12*x^4 - 28*x^3 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1780_178020


namespace NUMINAMATH_GPT_photograph_perimeter_l1780_178049

theorem photograph_perimeter (w l m : ℕ) 
  (h1 : (w + 4) * (l + 4) = m)
  (h2 : (w + 8) * (l + 8) = m + 94) :
  2 * (w + l) = 23 := 
by
  sorry

end NUMINAMATH_GPT_photograph_perimeter_l1780_178049


namespace NUMINAMATH_GPT_increasing_interval_of_f_l1780_178046

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2)

theorem increasing_interval_of_f :
  f x = (1/2)^(x^2 - 2) →
  ∀ x, f (x) ≤ f (x + 0.0001) :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_of_f_l1780_178046


namespace NUMINAMATH_GPT_flea_returns_to_0_l1780_178092

noncomputable def flea_return_probability (p : ℝ) : ℝ :=
if p = 1 then 0 else 1

theorem flea_returns_to_0 (p : ℝ) : 
  flea_return_probability p = (if p = 1 then 0 else 1) :=
by
  sorry

end NUMINAMATH_GPT_flea_returns_to_0_l1780_178092


namespace NUMINAMATH_GPT_solve_inequality_l1780_178002

theorem solve_inequality (a : ℝ) : 
  (a = 0 → {x : ℝ | x ≥ -1} = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
  (a ≠ 0 → 
    ((a > 0 → { x : ℝ | -1 ≤ x ∧ x ≤ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (-2 < a ∧ a < 0 → { x : ℝ | x ≤ 2 / a } ∪ { x : ℝ | -1 ≤ x }  = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a < -2 → { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a = -2 → { x : ℝ | True } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 })
)) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1780_178002


namespace NUMINAMATH_GPT_intersection_sets_m_n_l1780_178094

theorem intersection_sets_m_n :
  let M := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
  let N := { x : ℝ | x > 0 }
  M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_sets_m_n_l1780_178094


namespace NUMINAMATH_GPT_minimal_rooms_l1780_178021

-- Definitions
def numTourists := 100

def roomsAvailable (n k : Nat) : Prop :=
  ∀ k_even : k % 2 = 0, 
    ∃ m : Nat, k = 2 * m ∧ n = 100 * (m + 1) ∨
    ∀ k_odd : k % 2 = 1, k = 2 * m + 1 ∧ n = 100 * (m + 1) + 1

-- Proof statement
theorem minimal_rooms (k n : Nat) : roomsAvailable n k :=
by 
  -- The proof is provided in the solution steps
  sorry

end NUMINAMATH_GPT_minimal_rooms_l1780_178021


namespace NUMINAMATH_GPT_Keiko_speed_is_pi_div_3_l1780_178089

noncomputable def Keiko_avg_speed {r : ℝ} (v : ℝ → ℝ) (pi : ℝ) : ℝ :=
let C1 := 2 * pi * (r + 6) - 2 * pi * r
let t1 := 36
let v1 := C1 / t1

let C2 := 2 * pi * (r + 8) - 2 * pi * r
let t2 := 48
let v2 := C2 / t2

if v r = v1 ∧ v r = v2 then (v1 + v2) / 2 else 0

theorem Keiko_speed_is_pi_div_3 (pi : ℝ) (r : ℝ) (v : ℝ → ℝ) :
  v r = π / 3 ∧ (forall t1 t2 C1 C2, C1 / t1 = π / 3 ∧ C2 / t2 = π / 3 → 
  (C1/t1 + C2/t2)/2 = π / 3) :=
sorry

end NUMINAMATH_GPT_Keiko_speed_is_pi_div_3_l1780_178089


namespace NUMINAMATH_GPT_sum_eq_3_or_7_l1780_178054

theorem sum_eq_3_or_7 {x y z : ℝ} 
  (h1 : x + y / z = 2)
  (h2 : y + z / x = 2)
  (h3 : z + x / y = 2) : 
  x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_eq_3_or_7_l1780_178054


namespace NUMINAMATH_GPT_minimize_maximum_absolute_value_expression_l1780_178061

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end NUMINAMATH_GPT_minimize_maximum_absolute_value_expression_l1780_178061


namespace NUMINAMATH_GPT_parking_garage_savings_l1780_178064

theorem parking_garage_savings :
  let weekly_cost := 10
  let monthly_cost := 35
  let weeks_per_year := 52
  let months_per_year := 12
  let annual_weekly_cost := weekly_cost * weeks_per_year
  let annual_monthly_cost := monthly_cost * months_per_year
  let annual_savings := annual_weekly_cost - annual_monthly_cost
  annual_savings = 100 := 
by
  sorry

end NUMINAMATH_GPT_parking_garage_savings_l1780_178064


namespace NUMINAMATH_GPT_trigonometric_identity_solution_l1780_178014

open Real

theorem trigonometric_identity_solution (k n l : ℤ) (x : ℝ) 
  (h : 2 * cos x ≠ sin x) : 
  (sin x ^ 3 + cos x ^ 3) / (2 * cos x - sin x) = cos (2 * x) ↔
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (π / 4) * (4 * n - 1)) ∨
  (∃ l : ℤ, x = arctan (1 / 2) + π * l) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_solution_l1780_178014


namespace NUMINAMATH_GPT_incorrect_proposition_b_l1780_178085

axiom plane (α β : Type) : Prop
axiom line (m n : Type) : Prop
axiom parallel (a b : Type) : Prop
axiom perpendicular (a b : Type) : Prop
axiom intersection (α β : Type) (n : Type) : Prop
axiom contained (a b : Type) : Prop

theorem incorrect_proposition_b (α β m n : Type)
  (hαβ_plane : plane α β)
  (hmn_line : line m n)
  (h_parallel_m_α : parallel m α)
  (h_intersection : intersection α β n) :
  ¬ parallel m n :=
sorry

end NUMINAMATH_GPT_incorrect_proposition_b_l1780_178085


namespace NUMINAMATH_GPT_parabola_distance_l1780_178097

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_parabola_distance_l1780_178097


namespace NUMINAMATH_GPT_find_common_ratio_l1780_178047

noncomputable def geom_series_common_ratio (q : ℝ) : Prop :=
  ∃ (a1 : ℝ), a1 > 0 ∧ (a1 * q^2 = 18) ∧ (a1 * (1 + q + q^2) = 26)

theorem find_common_ratio (q : ℝ) :
  geom_series_common_ratio q → q = 3 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l1780_178047


namespace NUMINAMATH_GPT_customers_left_l1780_178001

theorem customers_left (initial_customers remaining_tables people_per_table customers_left : ℕ)
    (h_initial : initial_customers = 62)
    (h_tables : remaining_tables = 5)
    (h_people : people_per_table = 9)
    (h_left : customers_left = initial_customers - remaining_tables * people_per_table) : 
    customers_left = 17 := 
    by 
        -- Provide the proof here 
        sorry

end NUMINAMATH_GPT_customers_left_l1780_178001


namespace NUMINAMATH_GPT_find_f_of_7_over_3_l1780_178003

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the odd function f

-- Hypothesis: f is an odd function
axiom odd_function (x : ℝ) : f (-x) = -f x

-- Hypothesis: f(1 + x) = f(-x) for all x in ℝ
axiom functional_equation (x : ℝ) : f (1 + x) = f (-x)

-- Hypothesis: f(-1/3) = 1/3
axiom initial_condition : f (-1 / 3) = 1 / 3

-- The statement we need to prove
theorem find_f_of_7_over_3 : f (7 / 3) = - (1 / 3) :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_find_f_of_7_over_3_l1780_178003


namespace NUMINAMATH_GPT_non_basalt_rocks_total_eq_l1780_178082

def total_rocks_in_box_A : ℕ := 57
def basalt_rocks_in_box_A : ℕ := 25

def total_rocks_in_box_B : ℕ := 49
def basalt_rocks_in_box_B : ℕ := 19

def non_basalt_rocks_in_box_A : ℕ := total_rocks_in_box_A - basalt_rocks_in_box_A
def non_basalt_rocks_in_box_B : ℕ := total_rocks_in_box_B - basalt_rocks_in_box_B

def total_non_basalt_rocks : ℕ := non_basalt_rocks_in_box_A + non_basalt_rocks_in_box_B

theorem non_basalt_rocks_total_eq : total_non_basalt_rocks = 62 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_non_basalt_rocks_total_eq_l1780_178082


namespace NUMINAMATH_GPT_product_of_roots_eq_25_l1780_178095

theorem product_of_roots_eq_25 (t : ℝ) (h : t^2 - 10 * t + 25 = 0) : t * t = 25 :=
sorry

end NUMINAMATH_GPT_product_of_roots_eq_25_l1780_178095


namespace NUMINAMATH_GPT_find_number_l1780_178080

theorem find_number (x : ℝ) (h : x = (1 / 3) * x + 120) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1780_178080


namespace NUMINAMATH_GPT_terminal_side_in_second_quadrant_l1780_178062

theorem terminal_side_in_second_quadrant (α : ℝ) 
  (hcos : Real.cos α = -1/5) 
  (hsin : Real.sin α = 2 * Real.sqrt 6 / 5) : 
  (π / 2 < α ∧ α < π) :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_in_second_quadrant_l1780_178062


namespace NUMINAMATH_GPT_analyze_quadratic_function_l1780_178005

variable (x : ℝ)

def quadratic_function : ℝ → ℝ := λ x => x^2 - 4 * x + 6

theorem analyze_quadratic_function :
  (∃ y : ℝ, quadratic_function y = (x - 2)^2 + 2) ∧
  (∃ x0 : ℝ, quadratic_function x0 = (x0 - 2)^2 + 2 ∧ x0 = 2 ∧ (∀ y : ℝ, quadratic_function y ≥ 2)) :=
by
  sorry

end NUMINAMATH_GPT_analyze_quadratic_function_l1780_178005


namespace NUMINAMATH_GPT_relationship_S_T_l1780_178071

-- Definitions based on the given conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n

def seq_b (n : ℕ) : ℕ :=
  2 ^ (n - 1) + 1

def S (n : ℕ) : ℕ :=
  (n * (n + 1))

def T (n : ℕ) : ℕ :=
  (2^n) + n - 1

-- The conjecture and proofs
theorem relationship_S_T (n : ℕ) : 
  if n = 1 then T n = S n
  else if (2 ≤ n ∧ n < 5) then T n < S n
  else n ≥ 5 → T n > S n :=
by sorry

end NUMINAMATH_GPT_relationship_S_T_l1780_178071


namespace NUMINAMATH_GPT_find_deleted_files_l1780_178030

def original_files : Nat := 21
def remaining_files : Nat := 7
def deleted_files : Nat := 14

theorem find_deleted_files : original_files - remaining_files = deleted_files := by
  sorry

end NUMINAMATH_GPT_find_deleted_files_l1780_178030


namespace NUMINAMATH_GPT_dog_total_distance_l1780_178093

-- Define the conditions
def distance_between_A_and_B : ℝ := 100
def speed_A : ℝ := 6
def speed_B : ℝ := 4
def speed_dog : ℝ := 10

-- Define the statement we want to prove
theorem dog_total_distance : ∀ t : ℝ, (speed_A + speed_B) * t = distance_between_A_and_B → speed_dog * t = 100 :=
by
  intro t
  intro h
  sorry

end NUMINAMATH_GPT_dog_total_distance_l1780_178093


namespace NUMINAMATH_GPT_value_of_N_l1780_178043

theorem value_of_N (N : ℕ): 6 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 7.5 ↔ N = 25 ∨ N = 26 ∨ N = 27 ∨ N = 28 ∨ N = 29 := 
by
  sorry

end NUMINAMATH_GPT_value_of_N_l1780_178043


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_53_l1780_178052

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_53_l1780_178052


namespace NUMINAMATH_GPT_paint_required_for_small_statues_l1780_178018

-- Constants definition
def pint_per_8ft_statue : ℕ := 1
def height_original_statue : ℕ := 8
def height_small_statue : ℕ := 2
def number_of_small_statues : ℕ := 400

-- Theorem statement
theorem paint_required_for_small_statues :
  pint_per_8ft_statue = 1 →
  height_original_statue = 8 →
  height_small_statue = 2 →
  number_of_small_statues = 400 →
  (number_of_small_statues * (pint_per_8ft_statue * (height_small_statue / height_original_statue)^2)) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_paint_required_for_small_statues_l1780_178018


namespace NUMINAMATH_GPT_percentage_A_is_22_l1780_178033

noncomputable def percentage_A_in_mixture : ℝ :=
  (0.8 * 0.20 + 0.2 * 0.30) * 100

theorem percentage_A_is_22 :
  percentage_A_in_mixture = 22 := 
by
  sorry

end NUMINAMATH_GPT_percentage_A_is_22_l1780_178033


namespace NUMINAMATH_GPT_minimum_nine_points_distance_l1780_178079

theorem minimum_nine_points_distance (n : ℕ) : 
  (∀ (p : Fin n → ℝ × ℝ),
    (∀ i, ∃! (four_points : List (Fin n)), 
      List.length four_points = 4 ∧ (∀ j ∈ four_points, dist (p i) (p j) = 1)))
    ↔ n = 9 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_nine_points_distance_l1780_178079


namespace NUMINAMATH_GPT_bethany_age_l1780_178045

theorem bethany_age : ∀ (B S R : ℕ),
  (B - 3 = 2 * (S - 3)) →
  (B - 3 = R - 3 + 4) →
  (S + 5 = 16) →
  (R + 5 = 21) →
  B = 19 :=
by
  intros B S R h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_bethany_age_l1780_178045
