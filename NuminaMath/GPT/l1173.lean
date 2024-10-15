import Mathlib

namespace NUMINAMATH_GPT_reduction_percentage_price_increase_l1173_117312

-- Proof Problem 1: Reduction Percentage
theorem reduction_percentage (a : ℝ) (h₁ : (50 * (1 - a)^2 = 32)) : a = 0.2 := by
  sorry

-- Proof Problem 2: Price Increase for Daily Profit
theorem price_increase 
  (x : ℝ)
  (profit_per_kg : ℝ := 10)
  (initial_sales : ℕ := 500)
  (sales_decrease_per_unit : ℝ := 20)
  (required_profit : ℝ := 6000)
  (h₁ : (10 + x) * (initial_sales - sales_decrease_per_unit * x) = required_profit) : 
  x = 5 := by
  sorry

end NUMINAMATH_GPT_reduction_percentage_price_increase_l1173_117312


namespace NUMINAMATH_GPT_part1_part2_l1173_117384

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ := Real.tan ((x / 2) - (Real.pi / 3))

-- Part (1)
theorem part1 : f (5 * Real.pi / 2) = Real.sqrt 3 - 2 :=
by
  sorry

-- Part (2)
theorem part2 (k : ℤ) : { x : ℝ | f x ≤ Real.sqrt 3 } = 
  {x | ∃ (k : ℤ), 2 * k * Real.pi - Real.pi / 3 < x ∧ x ≤ 2 * k * Real.pi + 4 * Real.pi / 3} :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_part1_part2_l1173_117384


namespace NUMINAMATH_GPT_length_BC_fraction_AD_l1173_117333

theorem length_BC_fraction_AD {A B C D : Type} {AB BD AC CD AD BC : ℕ} 
  (h1 : AB = 4 * BD) (h2 : AC = 9 * CD) (h3 : AD = AB + BD) (h4 : AD = AC + CD)
  (h5 : B ≠ A) (h6 : C ≠ A) (h7 : A ≠ D) : BC = AD / 10 :=
by
  sorry

end NUMINAMATH_GPT_length_BC_fraction_AD_l1173_117333


namespace NUMINAMATH_GPT_area_of_smaller_part_l1173_117382

noncomputable def average (a b : ℝ) : ℝ :=
  (a + b) / 2

theorem area_of_smaller_part:
  ∃ A B : ℝ, A + B = 900 ∧ (B - A) = (1 / 5) * average A B ∧ A = 405 :=
by
  sorry

end NUMINAMATH_GPT_area_of_smaller_part_l1173_117382


namespace NUMINAMATH_GPT_hyperbola_asymptote_b_l1173_117371

theorem hyperbola_asymptote_b {b : ℝ} (hb : b > 0) :
  (∀ x y : ℝ, x^2 - (y^2 / b^2) = 1 → (y = 2 * x)) → b = 2 := by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_b_l1173_117371


namespace NUMINAMATH_GPT_repeating_decimal_sum_correct_l1173_117359

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  x + y - z

theorem repeating_decimal_sum_correct :
  repeating_decimal_sum = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_correct_l1173_117359


namespace NUMINAMATH_GPT_inequality_am_gm_l1173_117388

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1173_117388


namespace NUMINAMATH_GPT_product_of_roots_l1173_117387

theorem product_of_roots : ∃ (x : ℕ), x = 45 ∧ (∃ a b c : ℕ, a ^ 3 = 27 ∧ b ^ 4 = 81 ∧ c ^ 2 = 25 ∧ x = a * b * c) := 
sorry

end NUMINAMATH_GPT_product_of_roots_l1173_117387


namespace NUMINAMATH_GPT_camp_cedar_counselors_l1173_117390

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) 
(counselors_for_boys : ℕ) (counselors_for_girls : ℕ) 
(total_counselors : ℕ) 
(h1 : boys = 80)
(h2 : girls = 6 * boys - 40)
(h3 : counselors_for_boys = boys / 5)
(h4 : counselors_for_girls = (girls + 11) / 12)  -- +11 to account for rounding up
(h5 : total_counselors = counselors_for_boys + counselors_for_girls) : 
total_counselors = 53 :=
by
  sorry

end NUMINAMATH_GPT_camp_cedar_counselors_l1173_117390


namespace NUMINAMATH_GPT_digits_divisible_by_101_l1173_117326

theorem digits_divisible_by_101 :
  ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ (2013 * 100 + 10 * x + y) % 101 = 0 ∧ x = 9 ∧ y = 4 := by
  sorry

end NUMINAMATH_GPT_digits_divisible_by_101_l1173_117326


namespace NUMINAMATH_GPT_inequality_holds_l1173_117394

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_holds (h_cont : Continuous f) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, 2 * f x - (deriv f x) > 0) : 
  f 1 > (f 2) / (Real.exp 2) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1173_117394


namespace NUMINAMATH_GPT_machine_a_produces_6_sprockets_per_hour_l1173_117314

theorem machine_a_produces_6_sprockets_per_hour : 
  ∀ (A G T : ℝ), 
  (660 = A * (T + 10)) → 
  (660 = G * T) → 
  (G = 1.10 * A) → 
  A = 6 := 
by
  intros A G T h1 h2 h3
  sorry

end NUMINAMATH_GPT_machine_a_produces_6_sprockets_per_hour_l1173_117314


namespace NUMINAMATH_GPT_sum_of_three_consecutive_divisible_by_three_l1173_117308

theorem sum_of_three_consecutive_divisible_by_three (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2)) = 3 * k := by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_divisible_by_three_l1173_117308


namespace NUMINAMATH_GPT_sum_of_remainders_l1173_117335

theorem sum_of_remainders (n : ℤ) (h : n % 12 = 5) :
  (n % 4) + (n % 3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1173_117335


namespace NUMINAMATH_GPT_oil_drop_probability_l1173_117360

theorem oil_drop_probability :
  let r_circle := 1 -- radius of the circle in cm
  let side_square := 0.5 -- side length of the square in cm
  let area_circle := π * r_circle^2
  let area_square := side_square * side_square
  (area_square / area_circle) = 1 / (4 * π) :=
by
  sorry

end NUMINAMATH_GPT_oil_drop_probability_l1173_117360


namespace NUMINAMATH_GPT_covering_percentage_77_l1173_117330

-- Definition section for conditions
def radius_of_circle (r a : ℝ) := 2 * r * Real.pi = 4 * a
def center_coincide (a b : ℝ) := a = b

-- Theorem to be proven
theorem covering_percentage_77
  (r a : ℝ)
  (h_radius: radius_of_circle r a)
  (h_center: center_coincide 0 0) : 
  (r^2 * Real.pi - 0.7248 * r^2) / (r^2 * Real.pi) * 100 = 77 := by
  sorry

end NUMINAMATH_GPT_covering_percentage_77_l1173_117330


namespace NUMINAMATH_GPT_digit_product_equality_l1173_117306

theorem digit_product_equality :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    a * (10 * b + c) * (100 * d + 10 * e + f) = (1000 * g + 100 * h + 10 * i + j) :=
sorry

end NUMINAMATH_GPT_digit_product_equality_l1173_117306


namespace NUMINAMATH_GPT_intersection_complement_eq_empty_l1173_117366

open Set

variable {α : Type*} (M N U: Set α)

theorem intersection_complement_eq_empty (h : M ⊆ N) : M ∩ (compl N) = ∅ :=
sorry

end NUMINAMATH_GPT_intersection_complement_eq_empty_l1173_117366


namespace NUMINAMATH_GPT_difference_approx_l1173_117317

-- Let L be the larger number and S be the smaller number
variables (L S : ℝ)

-- Conditions given:
-- 1. L is approximately 1542.857
def approx_L : Prop := abs (L - 1542.857) < 1

-- 2. When L is divided by S, quotient is 8 and remainder is 15
def division_condition : Prop := L = 8 * S + 15

-- The theorem stating the difference L - S is approximately 1351.874
theorem difference_approx (hL : approx_L L) (hdiv : division_condition L S) :
  abs ((L - S) - 1351.874) < 1 :=
sorry

#check difference_approx

end NUMINAMATH_GPT_difference_approx_l1173_117317


namespace NUMINAMATH_GPT_paco_more_cookies_l1173_117349

def paco_cookies_difference
  (initial_cookies : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_given : ℕ) : ℕ :=
  cookies_eaten - cookies_given

theorem paco_more_cookies 
  (initial_cookies : ℕ)
  (cookies_eaten : ℕ)
  (cookies_given : ℕ)
  (h1 : initial_cookies = 17)
  (h2 : cookies_eaten = 14)
  (h3 : cookies_given = 13) :
  paco_cookies_difference initial_cookies cookies_eaten cookies_given = 1 :=
by
  rw [h2, h3]
  exact rfl

end NUMINAMATH_GPT_paco_more_cookies_l1173_117349


namespace NUMINAMATH_GPT_set_intersection_l1173_117300

-- Definitions of sets M and N
def M : Set ℤ := {-1, 1, 2}
def N : Set ℤ := {1, 2, 3}

-- The statement to prove that M ∩ N = {1, 2}
theorem set_intersection :
  M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_GPT_set_intersection_l1173_117300


namespace NUMINAMATH_GPT_calculate_arithmetic_expression_l1173_117386

noncomputable def arithmetic_sum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem calculate_arithmetic_expression :
  3 * (arithmetic_sum 71 2 99) = 3825 :=
by
  sorry

end NUMINAMATH_GPT_calculate_arithmetic_expression_l1173_117386


namespace NUMINAMATH_GPT_coupon1_best_discount_l1173_117339

noncomputable def listed_prices : List ℝ := [159.95, 179.95, 199.95, 219.95, 239.95]

theorem coupon1_best_discount (x : ℝ) (h₁ : x ∈ listed_prices) (h₂ : x > 120) :
  0.15 * x > 25 ∧ 0.15 * x > 0.20 * (x - 120) ↔ 
  x = 179.95 ∨ x = 199.95 ∨ x = 219.95 ∨ x = 239.95 :=
sorry

end NUMINAMATH_GPT_coupon1_best_discount_l1173_117339


namespace NUMINAMATH_GPT_hex_351_is_849_l1173_117302

noncomputable def hex_to_decimal : ℕ := 1 * 16^0 + 5 * 16^1 + 3 * 16^2

-- The following statement is the core of the proof problem
theorem hex_351_is_849 : hex_to_decimal = 849 := by
  -- Here the proof steps would normally go
  sorry

end NUMINAMATH_GPT_hex_351_is_849_l1173_117302


namespace NUMINAMATH_GPT_largest_d_for_range_l1173_117397

theorem largest_d_for_range (d : ℝ) : (∃ x : ℝ, x^2 - 6*x + d = 2) ↔ d ≤ 11 := 
by
  sorry

end NUMINAMATH_GPT_largest_d_for_range_l1173_117397


namespace NUMINAMATH_GPT_smallest_triangle_perimeter_l1173_117336

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end NUMINAMATH_GPT_smallest_triangle_perimeter_l1173_117336


namespace NUMINAMATH_GPT_product_of_x1_to_x13_is_zero_l1173_117361

theorem product_of_x1_to_x13_is_zero
  (a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ)
  (h1 : a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) * (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2 : a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) * (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 :=
sorry

end NUMINAMATH_GPT_product_of_x1_to_x13_is_zero_l1173_117361


namespace NUMINAMATH_GPT_percent_defective_units_shipped_for_sale_l1173_117327

theorem percent_defective_units_shipped_for_sale 
  (P : ℝ) -- total number of units produced
  (h_defective : 0.06 * P = d) -- 6 percent of units are defective
  (h_shipped : 0.0024 * P = s) -- 0.24 percent of units are defective units shipped for sale
  : (s / d) * 100 = 4 :=
by
  sorry

end NUMINAMATH_GPT_percent_defective_units_shipped_for_sale_l1173_117327


namespace NUMINAMATH_GPT_simplify_evaluate_expression_l1173_117372

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (2 / (x + 1) + 1 / (x - 2)) / (x - 1) / (x - 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expression_l1173_117372


namespace NUMINAMATH_GPT_find_equation_of_tangent_line_perpendicular_l1173_117329

noncomputable def tangent_line_perpendicular_to_curve (a b : ℝ) : Prop :=
  (∃ (P : ℝ × ℝ), P = (-1, -3) ∧ 2 * P.1 - 6 * P.2 + 1 = 0 ∧ P.2 = P.1^3 + 5 * P.1^2 - 5) ∧
  (-3) = 3 * (-1)^2 + 6 * (-1)

theorem find_equation_of_tangent_line_perpendicular :
  tangent_line_perpendicular_to_curve (-1) (-3) →
  ∀ x y : ℝ, 3 * x + y + 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_equation_of_tangent_line_perpendicular_l1173_117329


namespace NUMINAMATH_GPT_max_omega_l1173_117370

theorem max_omega (ω : ℕ) (T : ℝ) (h₁ : T = 2 * Real.pi / ω) (h₂ : 1 < T) (h₃ : T < 3) : ω = 6 :=
sorry

end NUMINAMATH_GPT_max_omega_l1173_117370


namespace NUMINAMATH_GPT_abs_ineq_one_abs_ineq_two_l1173_117309

-- First proof problem: |x-1| + |x+3| < 6 implies -4 < x < 2
theorem abs_ineq_one (x : ℝ) : |x - 1| + |x + 3| < 6 → -4 < x ∧ x < 2 :=
by
  sorry

-- Second proof problem: 1 < |3x-2| < 4 implies -2/3 ≤ x < 1/3 or 1 < x ≤ 2
theorem abs_ineq_two (x : ℝ) : 1 < |3 * x - 2| ∧ |3 * x - 2| < 4 → (-2/3) ≤ x ∧ x < (1/3) ∨ 1 < x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_one_abs_ineq_two_l1173_117309


namespace NUMINAMATH_GPT_sum_f_positive_l1173_117367

noncomputable def f (x : ℝ) : ℝ := (x ^ 3) / (Real.cos x)

theorem sum_f_positive 
  (x1 x2 x3 : ℝ)
  (hdom1 : abs x1 < Real.pi / 2)
  (hdom2 : abs x2 < Real.pi / 2)
  (hdom3 : abs x3 < Real.pi / 2)
  (hx1x2 : x1 + x2 > 0)
  (hx2x3 : x2 + x3 > 0)
  (hx1x3 : x1 + x3 > 0) :
  f x1 + f x2 + f x3 > 0 :=
sorry

end NUMINAMATH_GPT_sum_f_positive_l1173_117367


namespace NUMINAMATH_GPT_gcd_between_35_and_7_l1173_117304

theorem gcd_between_35_and_7 {n : ℕ} (h1 : 65 < n) (h2 : n < 75) (h3 : gcd 35 n = 7) : n = 70 := 
sorry

end NUMINAMATH_GPT_gcd_between_35_and_7_l1173_117304


namespace NUMINAMATH_GPT_non_zero_x_satisfies_equation_l1173_117368

theorem non_zero_x_satisfies_equation :
  ∃ (x : ℝ), (x ≠ 0) ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_non_zero_x_satisfies_equation_l1173_117368


namespace NUMINAMATH_GPT_bear_pies_l1173_117383

-- Lean definitions model:

variables (v_M v_B u_M u_B : ℝ)
variables (M_raspberries B_raspberries : ℝ)
variables (P_M P_B : ℝ)

-- Given conditions
axiom v_B_eq_6v_M : v_B = 6 * v_M
axiom u_B_eq_3u_M : u_B = 3 * u_M
axiom B_raspberries_eq_2M_raspberries : B_raspberries = 2 * M_raspberries
axiom P_sum : P_B + P_M = 60
axiom P_B_eq_9P_M : P_B = 9 * P_M

-- The theorem to prove
theorem bear_pies : P_B = 54 :=
sorry

end NUMINAMATH_GPT_bear_pies_l1173_117383


namespace NUMINAMATH_GPT_closest_to_fraction_l1173_117322

theorem closest_to_fraction (options : List ℝ) (h1 : options = [2000, 1500, 200, 2500, 3000]) :
  ∃ closest : ℝ, closest ∈ options ∧ closest = 2000 :=
by
  sorry

end NUMINAMATH_GPT_closest_to_fraction_l1173_117322


namespace NUMINAMATH_GPT_cos_neg_45_eq_one_over_sqrt_two_l1173_117389

theorem cos_neg_45_eq_one_over_sqrt_two : Real.cos (-(45 : ℝ)) = 1 / Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_neg_45_eq_one_over_sqrt_two_l1173_117389


namespace NUMINAMATH_GPT_plates_to_remove_l1173_117396

-- Definitions based on the problem conditions
def number_of_plates : ℕ := 38
def weight_per_plate : ℕ := 10
def acceptable_weight : ℕ := 320

-- Theorem to prove
theorem plates_to_remove (initial_weight := number_of_plates * weight_per_plate) 
  (excess_weight := initial_weight - acceptable_weight) 
  (plates_to_remove := excess_weight / weight_per_plate) :
  plates_to_remove = 6 :=
by
  sorry

end NUMINAMATH_GPT_plates_to_remove_l1173_117396


namespace NUMINAMATH_GPT_solve_for_x_l1173_117374

theorem solve_for_x (x : ℝ) :
  (2 * x - 30) / 3 = (5 - 3 * x) / 4 + 1 → x = 147 / 17 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1173_117374


namespace NUMINAMATH_GPT_minimum_value_function_l1173_117381

theorem minimum_value_function (x : ℝ) (h : x > -1) : 
  (∃ y, y = (x^2 + 7 * x + 10) / (x + 1) ∧ y ≥ 9) :=
sorry

end NUMINAMATH_GPT_minimum_value_function_l1173_117381


namespace NUMINAMATH_GPT_power_function_evaluation_l1173_117344

theorem power_function_evaluation (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 4 = 2) : f 16 = 4 :=
by
  sorry

end NUMINAMATH_GPT_power_function_evaluation_l1173_117344


namespace NUMINAMATH_GPT_airplane_seat_count_l1173_117320

theorem airplane_seat_count (s : ℝ) 
  (h1 : 30 + 0.2 * s + 0.75 * s = s) : 
  s = 600 :=
sorry

end NUMINAMATH_GPT_airplane_seat_count_l1173_117320


namespace NUMINAMATH_GPT_flags_left_l1173_117385

theorem flags_left (interval circumference : ℕ) (total_flags : ℕ) (h1 : interval = 20) (h2 : circumference = 200) (h3 : total_flags = 12) : 
  total_flags - (circumference / interval) = 2 := 
by 
  -- Using the conditions h1, h2, h3
  sorry

end NUMINAMATH_GPT_flags_left_l1173_117385


namespace NUMINAMATH_GPT_product_of_numbers_l1173_117343

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := 
sorry

end NUMINAMATH_GPT_product_of_numbers_l1173_117343


namespace NUMINAMATH_GPT_min_value_of_f_l1173_117311

open Real

noncomputable def f (x : ℝ) := x + 1 / (x - 2)

theorem min_value_of_f : ∃ x : ℝ, x > 2 ∧ ∀ y : ℝ, y > 2 → f y ≥ f 3 := by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1173_117311


namespace NUMINAMATH_GPT_cost_price_correct_l1173_117342

noncomputable def cost_price_per_meter (selling_price_per_meter : ℝ) (total_meters : ℝ) (loss_per_meter : ℝ) :=
  (selling_price_per_meter * total_meters + loss_per_meter * total_meters) / total_meters

theorem cost_price_correct :
  cost_price_per_meter 18000 500 5 = 41 :=
by 
  sorry

end NUMINAMATH_GPT_cost_price_correct_l1173_117342


namespace NUMINAMATH_GPT_jack_marathon_time_l1173_117313

theorem jack_marathon_time :
  ∀ {marathon_distance : ℝ} {jill_time : ℝ} {speed_ratio : ℝ},
    marathon_distance = 40 → 
    jill_time = 4 → 
    speed_ratio = 0.888888888888889 → 
    (marathon_distance / (speed_ratio * (marathon_distance / jill_time))) = 4.5 :=
by
  intros marathon_distance jill_time speed_ratio h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_jack_marathon_time_l1173_117313


namespace NUMINAMATH_GPT_decimal_to_base_five_l1173_117315

theorem decimal_to_base_five : 
  (2 * 5^3 + 1 * 5^1 + 0 * 5^2 + 0 * 5^0 = 255) := 
by
  sorry

end NUMINAMATH_GPT_decimal_to_base_five_l1173_117315


namespace NUMINAMATH_GPT_choir_members_minimum_l1173_117398

theorem choir_members_minimum (n : Nat) (h9 : n % 9 = 0) (h10 : n % 10 = 0) (h11 : n % 11 = 0) (h14 : n % 14 = 0) : n = 6930 :=
sorry

end NUMINAMATH_GPT_choir_members_minimum_l1173_117398


namespace NUMINAMATH_GPT_sum_of_four_squares_l1173_117376

theorem sum_of_four_squares (a b c : ℕ) 
    (h1 : 2 * a + b + c = 27)
    (h2 : 2 * b + a + c = 25)
    (h3 : 3 * c + a = 39) : 4 * c = 44 := 
  sorry

end NUMINAMATH_GPT_sum_of_four_squares_l1173_117376


namespace NUMINAMATH_GPT_compute_value_of_expression_l1173_117319

theorem compute_value_of_expression (p q : ℝ) (h1 : 3 * p^2 - 7 * p + 1 = 0) (h2 : 3 * q^2 - 7 * q + 1 = 0) :
  (9 * p^3 - 9 * q^3) / (p - q) = 46 :=
sorry

end NUMINAMATH_GPT_compute_value_of_expression_l1173_117319


namespace NUMINAMATH_GPT_lcm_of_54_and_198_l1173_117352

theorem lcm_of_54_and_198 : Nat.lcm 54 198 = 594 :=
by
  have fact1 : 54 = 2 ^ 1 * 3 ^ 3 := by norm_num
  have fact2 : 198 = 2 ^ 1 * 3 ^ 2 * 11 ^ 1 := by norm_num
  have lcm_prime : Nat.lcm 54 198 = 594 := by
    sorry -- Proof skipped
  exact lcm_prime

end NUMINAMATH_GPT_lcm_of_54_and_198_l1173_117352


namespace NUMINAMATH_GPT_second_derivative_parametric_l1173_117305

noncomputable def x (t : ℝ) := Real.sqrt (t - 1)
noncomputable def y (t : ℝ) := 1 / Real.sqrt t

noncomputable def y_xx (t : ℝ) := (2 * t - 3) * Real.sqrt t / t^3

theorem second_derivative_parametric :
  ∀ t, y_xx t = (2 * t - 3) * Real.sqrt t / t^3 := sorry

end NUMINAMATH_GPT_second_derivative_parametric_l1173_117305


namespace NUMINAMATH_GPT_number_of_cubes_with_three_faces_painted_l1173_117355

-- Definitions of conditions
def large_cube_side_length : ℕ := 4
def total_smaller_cubes := large_cube_side_length ^ 3

-- Prove the number of smaller cubes with at least 3 faces painted is 8
theorem number_of_cubes_with_three_faces_painted :
  (∃ (n : ℕ), n = 8) :=
by
  -- Conditions recall
  have side_length := large_cube_side_length
  have total_cubes := total_smaller_cubes
  
  -- Recall that the cube is composed by smaller cubes with painted faces.
  have painted_faces_condition : (∀ (cube : ℕ), cube = 8) := sorry
  
  exact ⟨8, painted_faces_condition 8⟩

end NUMINAMATH_GPT_number_of_cubes_with_three_faces_painted_l1173_117355


namespace NUMINAMATH_GPT_angle_measure_l1173_117303

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1173_117303


namespace NUMINAMATH_GPT_sally_balloons_l1173_117307

theorem sally_balloons (F S : ℕ) (h1 : F = 3 * S) (h2 : F = 18) : S = 6 :=
by sorry

end NUMINAMATH_GPT_sally_balloons_l1173_117307


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l1173_117334

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l1173_117334


namespace NUMINAMATH_GPT_sampling_methods_correct_l1173_117301

-- Define the conditions given in the problem.
def total_students := 200
def method_1_is_simple_random := true
def method_2_is_systematic := true

-- The proof problem statement, no proof is required.
theorem sampling_methods_correct :
  (method_1_is_simple_random = true) ∧
  (method_2_is_systematic = true) :=
by
  -- using conditions defined above, we state the theorem we need to prove
  sorry

end NUMINAMATH_GPT_sampling_methods_correct_l1173_117301


namespace NUMINAMATH_GPT_subset_singleton_zero_A_l1173_117379

def A : Set ℝ := {x | x > -3}

theorem subset_singleton_zero_A : {0} ⊆ A := 
by
  sorry  -- Proof is not required

end NUMINAMATH_GPT_subset_singleton_zero_A_l1173_117379


namespace NUMINAMATH_GPT_provisions_last_60_days_l1173_117363

/-
A garrison of 1000 men has provisions for a certain number of days.
At the end of 15 days, a reinforcement of 1250 arrives, and it is now found that the provisions will last only for 20 days more.
Prove that the provisions were supposed to last initially for 60 days.
-/

def initial_provisions (D : ℕ) : Prop :=
  let initial_garrison := 1000
  let reinforcement_garrison := 1250
  let days_spent := 15
  let remaining_days := 20
  initial_garrison * (D - days_spent) = (initial_garrison + reinforcement_garrison) * remaining_days

theorem provisions_last_60_days (D : ℕ) : initial_provisions D → D = 60 := by
  sorry

end NUMINAMATH_GPT_provisions_last_60_days_l1173_117363


namespace NUMINAMATH_GPT_m₁_m₂_relationship_l1173_117331

-- Defining the conditions
variables {Point Line : Type}
variables (intersect : Line → Line → Prop)
variables (coplanar : Line → Line → Prop)

-- Assumption that lines l₁ and l₂ are non-coplanar.
variables {l₁ l₂ : Line} (h_non_coplanar : ¬ coplanar l₁ l₂)

-- Assuming m₁ and m₂ both intersect with l₁ and l₂.
variables {m₁ m₂ : Line}
variables (h_intersect_m₁_l₁ : intersect m₁ l₁)
variables (h_intersect_m₁_l₂ : intersect m₁ l₂)
variables (h_intersect_m₂_l₁ : intersect m₂ l₁)
variables (h_intersect_m₂_l₂ : intersect m₂ l₂)

-- Statement to prove that m₁ and m₂ are either intersecting or non-coplanar.
theorem m₁_m₂_relationship :
  (¬ coplanar m₁ m₂) ∨ (∃ p : Point, (intersect m₁ m₂ ∧ intersect m₂ m₁)) :=
sorry

end NUMINAMATH_GPT_m₁_m₂_relationship_l1173_117331


namespace NUMINAMATH_GPT_evaluate_magnitude_of_product_l1173_117364

theorem evaluate_magnitude_of_product :
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  Complex.abs (z1 * z2) = 4 * Real.sqrt 43 := by
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  suffices Complex.abs z1 * Complex.abs z2 = 4 * Real.sqrt 43 by sorry
  sorry

end NUMINAMATH_GPT_evaluate_magnitude_of_product_l1173_117364


namespace NUMINAMATH_GPT_range_of_quadratic_function_l1173_117325

theorem range_of_quadratic_function :
  ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), -x^2 - 4 * x + 1 ∈ Set.Icc (-11) (5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_quadratic_function_l1173_117325


namespace NUMINAMATH_GPT_more_chickens_than_chicks_l1173_117345

-- Let's define the given conditions
def total : Nat := 821
def chicks : Nat := 267

-- The statement we need to prove
theorem more_chickens_than_chicks : (total - chicks) - chicks = 287 :=
by
  -- This is needed for the proof and not part of conditions
  -- Add sorry as a placeholder for proof steps 
  sorry

end NUMINAMATH_GPT_more_chickens_than_chicks_l1173_117345


namespace NUMINAMATH_GPT_triangle_angle_sum_l1173_117338

theorem triangle_angle_sum {x : ℝ} (h : 60 + 5 * x + 3 * x = 180) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l1173_117338


namespace NUMINAMATH_GPT_butterfly_count_l1173_117346

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end NUMINAMATH_GPT_butterfly_count_l1173_117346


namespace NUMINAMATH_GPT_field_length_proof_l1173_117391

noncomputable def field_width (w : ℝ) : Prop := w > 0

def pond_side_length : ℝ := 7

def pond_area : ℝ := pond_side_length * pond_side_length

def field_length (w l : ℝ) : Prop := l = 2 * w

def field_area (w l : ℝ) : ℝ := l * w

def pond_area_condition (w l : ℝ) : Prop :=
  pond_area = (1 / 8) * field_area w l

theorem field_length_proof {w l : ℝ} (hw : field_width w)
                           (hl : field_length w l)
                           (hpond : pond_area_condition w l) :
  l = 28 := by
  sorry

end NUMINAMATH_GPT_field_length_proof_l1173_117391


namespace NUMINAMATH_GPT_perimeter_remaining_shape_l1173_117378

theorem perimeter_remaining_shape (length width square1 square2 : ℝ) 
  (H_len : length = 50) (H_width : width = 20) 
  (H_sq1 : square1 = 12) (H_sq2 : square2 = 4) : 
  2 * (length + width) + 4 * (square1 + square2) = 204 :=
by 
  rw [H_len, H_width, H_sq1, H_sq2]
  sorry

end NUMINAMATH_GPT_perimeter_remaining_shape_l1173_117378


namespace NUMINAMATH_GPT_max_value_y_l1173_117341

open Real

theorem max_value_y (x : ℝ) (h : -1 < x ∧ x < 1) : 
  ∃ y_max, y_max = 0 ∧ ∀ y, y = x / (x - 1) + x → y ≤ y_max :=
by
  have y : ℝ := x / (x - 1) + x
  use 0
  sorry

end NUMINAMATH_GPT_max_value_y_l1173_117341


namespace NUMINAMATH_GPT_number_of_music_files_l1173_117373

-- The conditions given in the problem
variable {M : ℕ} -- M is a natural number representing the initial number of music files

-- Conditions: Initial state and changes
def initial_video_files : ℕ := 21
def files_deleted : ℕ := 23
def remaining_files : ℕ := 2

-- Statement of the theorem
theorem number_of_music_files (h : M + initial_video_files - files_deleted = remaining_files) : M = 4 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_music_files_l1173_117373


namespace NUMINAMATH_GPT_find_d_minus_r_l1173_117323

theorem find_d_minus_r :
  ∃ d r : ℕ, 1 < d ∧ 1223 % d = r ∧ 1625 % d = r ∧ 2513 % d = r ∧ d - r = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_d_minus_r_l1173_117323


namespace NUMINAMATH_GPT_expression_value_l1173_117340

theorem expression_value : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1173_117340


namespace NUMINAMATH_GPT_negation_of_p_l1173_117310

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.exp x > Real.log x

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.exp x ≤ Real.log x

-- The statement we want to prove
theorem negation_of_p : ¬p ↔ neg_p :=
by sorry

end NUMINAMATH_GPT_negation_of_p_l1173_117310


namespace NUMINAMATH_GPT_max_sum_cubes_l1173_117348

theorem max_sum_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_sum_cubes_l1173_117348


namespace NUMINAMATH_GPT_geometric_sequence_sum_product_l1173_117365

theorem geometric_sequence_sum_product {a b c : ℝ} : 
  a + b + c = 14 → 
  a * b * c = 64 → 
  (a = 8 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_product_l1173_117365


namespace NUMINAMATH_GPT_determine_d_l1173_117392

-- Given conditions
def equation (d x : ℝ) : Prop := 3 * (5 + d * x) = 15 * x + 15

-- Proof statement
theorem determine_d (d : ℝ) : (∀ x : ℝ, equation d x) ↔ d = 5 :=
by
  sorry

end NUMINAMATH_GPT_determine_d_l1173_117392


namespace NUMINAMATH_GPT_point_p_final_position_l1173_117324

theorem point_p_final_position :
  let P_start := -2
  let P_right := P_start + 5
  let P_final := P_right - 4
  P_final = -1 :=
by
  sorry

end NUMINAMATH_GPT_point_p_final_position_l1173_117324


namespace NUMINAMATH_GPT_max_sum_consecutive_integers_less_360_l1173_117399

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_consecutive_integers_less_360_l1173_117399


namespace NUMINAMATH_GPT_function_monotone_increasing_l1173_117337

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - log x

theorem function_monotone_increasing : ∀ x, 1 ≤ x → (0 < x) → (1 / 2) * x^2 - log x = f x → (∀ y, 1 ≤ y → (0 < y) → (f y ≤ f x)) :=
sorry

end NUMINAMATH_GPT_function_monotone_increasing_l1173_117337


namespace NUMINAMATH_GPT_guests_not_eating_brownies_ala_mode_l1173_117393

theorem guests_not_eating_brownies_ala_mode (total_brownies : ℕ) (eaten_brownies : ℕ) (eaten_scoops : ℕ)
    (scoops_per_serving : ℕ) (scoops_per_tub : ℕ) (tubs_eaten : ℕ) : 
    total_brownies = 32 → eaten_brownies = 28 → eaten_scoops = 48 → scoops_per_serving = 2 → scoops_per_tub = 8 → tubs_eaten = 6 → (eaten_scoops - eaten_brownies * scoops_per_serving) / scoops_per_serving = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_guests_not_eating_brownies_ala_mode_l1173_117393


namespace NUMINAMATH_GPT_greatest_savings_by_choosing_boat_l1173_117362

/-- Given the transportation costs:
     - plane cost: $600.00
     - boat cost: $254.00
     - helicopter cost: $850.00
    Prove that the greatest amount of money saved by choosing the boat over the other options is $596.00. -/
theorem greatest_savings_by_choosing_boat :
  let plane_cost := 600
  let boat_cost := 254
  let helicopter_cost := 850
  max (plane_cost - boat_cost) (helicopter_cost - boat_cost) = 596 :=
by
  sorry

end NUMINAMATH_GPT_greatest_savings_by_choosing_boat_l1173_117362


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1173_117369

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y, 3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0) →
  (∀ x y, y * y = 9 * (x * x / (a * a) - 1)) →
  a = 2 :=
by
  intros asymptote_constr hyp
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1173_117369


namespace NUMINAMATH_GPT_distance_between_lines_l1173_117356

/-- The graph of the function y = x^2 + ax + b is drawn on a board.
Let the parabola intersect the horizontal lines y = s and y = t at points A, B and C, D respectively,
with A B = 5 and C D = 11. Then the distance between the lines y = s and y = t is 24. -/
theorem distance_between_lines 
  (a b s t : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + a * x1 + b = s) ∧ (x2^2 + a * x2 + b = s) ∧ |x1 - x2| = 5)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ (x3^2 + a * x3 + b = t) ∧ (x4^2 + a * x4 + b = t) ∧ |x3 - x4| = 11) :
  |t - s| = 24 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_lines_l1173_117356


namespace NUMINAMATH_GPT_number_of_girls_in_group_l1173_117350

-- Define the given conditions
def total_students : ℕ := 20
def prob_of_selecting_girl : ℚ := 2/5

-- State the lean problem for the proof
theorem number_of_girls_in_group : (total_students : ℚ) * prob_of_selecting_girl = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_group_l1173_117350


namespace NUMINAMATH_GPT_min_cuts_to_one_meter_pieces_l1173_117357

theorem min_cuts_to_one_meter_pieces (x y : ℕ) (hx : x + y = 30) (hl : 3 * x + 4 * y = 100) : (2 * x + 3 * y) = 70 := 
by sorry

end NUMINAMATH_GPT_min_cuts_to_one_meter_pieces_l1173_117357


namespace NUMINAMATH_GPT_zachary_pushups_l1173_117332

theorem zachary_pushups (C P : ℕ) (h1 : C = 14) (h2 : P + C = 67) : P = 53 :=
by
  rw [h1] at h2
  linarith

end NUMINAMATH_GPT_zachary_pushups_l1173_117332


namespace NUMINAMATH_GPT_sum_gcd_lcm_eq_4851_l1173_117351

theorem sum_gcd_lcm_eq_4851 (a b : ℕ) (ha : a = 231) (hb : b = 4620) :
  Nat.gcd a b + Nat.lcm a b = 4851 :=
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_eq_4851_l1173_117351


namespace NUMINAMATH_GPT_non_integer_x_and_y_impossible_l1173_117354

theorem non_integer_x_and_y_impossible 
  (x y : ℚ) (m n : ℤ) 
  (h1 : 5 * x + 7 * y = m)
  (h2 : 7 * x + 10 * y = n) : 
  ∃ (x y : ℤ), 5 * x + 7 * y = m ∧ 7 * x + 10 * y = n := 
sorry

end NUMINAMATH_GPT_non_integer_x_and_y_impossible_l1173_117354


namespace NUMINAMATH_GPT_parabola_y_intercepts_l1173_117395

theorem parabola_y_intercepts : 
  (∀ y : ℝ, 3 * y^2 - 6 * y + 1 = 0) → (∃ y1 y2 : ℝ, y1 ≠ y2) :=
by sorry

end NUMINAMATH_GPT_parabola_y_intercepts_l1173_117395


namespace NUMINAMATH_GPT_remainder_of_sum_is_zero_l1173_117347

-- Define the properties of m and n according to the conditions of the problem
def m : ℕ := 2 * 1004 ^ 2
def n : ℕ := 2007 * 1003

-- State the theorem that proves the remainder of (m + n) divided by 1004 is 0
theorem remainder_of_sum_is_zero : (m + n) % 1004 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_is_zero_l1173_117347


namespace NUMINAMATH_GPT_gcd_decomposition_l1173_117318

open Polynomial

noncomputable def f : Polynomial ℚ := 4 * X ^ 4 - 2 * X ^ 3 - 16 * X ^ 2 + 5 * X + 9
noncomputable def g : Polynomial ℚ := 2 * X ^ 3 - X ^ 2 - 5 * X + 4

theorem gcd_decomposition :
  ∃ (u v : Polynomial ℚ), u * f + v * g = X - 1 :=
sorry

end NUMINAMATH_GPT_gcd_decomposition_l1173_117318


namespace NUMINAMATH_GPT_range_of_m_l1173_117328

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1173_117328


namespace NUMINAMATH_GPT_calculate_exponentiation_l1173_117377

theorem calculate_exponentiation : (64^(0.375) * 64^(0.125) = 8) :=
by sorry

end NUMINAMATH_GPT_calculate_exponentiation_l1173_117377


namespace NUMINAMATH_GPT_fraction_sum_in_simplest_form_l1173_117375

theorem fraction_sum_in_simplest_form :
  ∃ a b : ℕ, a + b = 11407 ∧ 0.425875 = a / (b : ℝ) ∧ Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_in_simplest_form_l1173_117375


namespace NUMINAMATH_GPT_smallest_x_division_remainder_l1173_117353

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end NUMINAMATH_GPT_smallest_x_division_remainder_l1173_117353


namespace NUMINAMATH_GPT_sum_of_six_terms_l1173_117380

variable {a : ℕ → ℝ} {q : ℝ}

/-- Given conditions:
* a is a decreasing geometric sequence with ratio q
-/
def is_decreasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem sum_of_six_terms
  (h_geo : is_decreasing_geometric_sequence a q)
  (h_decreasing : 0 < q ∧ q < 1)
  (h_a1 : 0 < a 1)
  (h_a1a3 : a 1 * a 3 = 1)
  (h_a2a4 : a 2 + a 4 = 5 / 4) :
  (a 1 * (1 - q^6) / (1 - q)) = 63 / 16 := by
  sorry

end NUMINAMATH_GPT_sum_of_six_terms_l1173_117380


namespace NUMINAMATH_GPT_jill_spent_10_percent_on_food_l1173_117358

theorem jill_spent_10_percent_on_food 
  (T : ℝ)                         
  (h1 : 0.60 * T = 0.60 * T)    -- 60% on clothing
  (h2 : 0.30 * T = 0.30 * T)    -- 30% on other items
  (h3 : 0.04 * (0.60 * T) = 0.024 * T)  -- 4% tax on clothing
  (h4 : 0.08 * (0.30 * T) = 0.024 * T)  -- 8% tax on other items
  (h5 : 0.048 * T = (0.024 * T + 0.024 * T)) -- total tax is 4.8%
  : 0.10 * T = (T - (0.60*T + 0.30*T)) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_jill_spent_10_percent_on_food_l1173_117358


namespace NUMINAMATH_GPT_simplify_expression_l1173_117321

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1173_117321


namespace NUMINAMATH_GPT_increasing_ω_l1173_117316

noncomputable def f (ω x : ℝ) : ℝ := (1 / 2) * (Real.sin ((ω * x) / 2)) * (Real.cos ((ω * x) / 2))

theorem increasing_ω (ω : ℝ) (hω : 0 < ω) :
  (∀ x y, - (Real.pi / 3) ≤ x → x ≤ y → y ≤ (Real.pi / 4) → f ω x ≤ f ω y)
  ↔ 0 < ω ∧ ω ≤ (3 / 2) :=
sorry

end NUMINAMATH_GPT_increasing_ω_l1173_117316
