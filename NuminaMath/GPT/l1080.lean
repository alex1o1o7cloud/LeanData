import Mathlib

namespace NUMINAMATH_GPT_greatest_sum_solution_l1080_108054

theorem greatest_sum_solution (x y : ℤ) (h : x^2 + y^2 = 20) : 
  x + y ≤ 6 :=
sorry

end NUMINAMATH_GPT_greatest_sum_solution_l1080_108054


namespace NUMINAMATH_GPT_find_g_l1080_108011

noncomputable def g : ℝ → ℝ := sorry

theorem find_g :
  (g 1 = 2) ∧ (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) ↔ (∀ x : ℝ, g x = 2 * (4^x - 3^x)) := 
by
  sorry

end NUMINAMATH_GPT_find_g_l1080_108011


namespace NUMINAMATH_GPT_intersection_A_B_l1080_108098

def setA (x : ℝ) : Prop := 3 * x + 2 > 0
def setB (x : ℝ) : Prop := (x + 1) * (x - 3) > 0
def A : Set ℝ := { x | setA x }
def B : Set ℝ := { x | setB x }

theorem intersection_A_B : A ∩ B = { x | 3 < x } := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1080_108098


namespace NUMINAMATH_GPT_molecular_weight_CaO_l1080_108026

theorem molecular_weight_CaO (m : ℕ -> ℝ) (h : m 7 = 392) : m 1 = 56 :=
sorry

end NUMINAMATH_GPT_molecular_weight_CaO_l1080_108026


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l1080_108038

-- Defining the main hypothesis
theorem quadratic_distinct_real_roots (k : ℝ) :
  (k < 4 / 3) ∧ (k ≠ 1) ↔ (∀ x : ℂ, ((k-1) * x^2 - 2 * x + 3 = 0) → ∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ ((k-1) * x₁ ^ 2 - 2 * x₁ + 3 = 0) ∧ ((k-1) * x₂ ^ 2 - 2 * x₂ + 3 = 0)) := by
sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l1080_108038


namespace NUMINAMATH_GPT_doug_initial_marbles_l1080_108090

theorem doug_initial_marbles (E D : ℕ) (H1 : E = D + 5) (H2 : E = 27) : D = 22 :=
by
  -- proof provided here would infer the correct answer from the given conditions
  sorry

end NUMINAMATH_GPT_doug_initial_marbles_l1080_108090


namespace NUMINAMATH_GPT_value_of_b_l1080_108064

theorem value_of_b :
  (∃ b : ℝ, (1 / Real.log b / Real.log 3 + 1 / Real.log b / Real.log 4 + 1 / Real.log b / Real.log 5 = 1) → b = 60) :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l1080_108064


namespace NUMINAMATH_GPT_adult_elephant_weekly_bananas_l1080_108043

theorem adult_elephant_weekly_bananas (daily_bananas : Nat) (days_in_week : Nat) (H1 : daily_bananas = 90) (H2 : days_in_week = 7) :
  daily_bananas * days_in_week = 630 :=
by
  sorry

end NUMINAMATH_GPT_adult_elephant_weekly_bananas_l1080_108043


namespace NUMINAMATH_GPT_convert_108_kmph_to_mps_l1080_108075

-- Definitions and assumptions
def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * (1000 / 3600)

-- Theorem statement
theorem convert_108_kmph_to_mps : kmph_to_mps 108 = 30 := 
by
  sorry

end NUMINAMATH_GPT_convert_108_kmph_to_mps_l1080_108075


namespace NUMINAMATH_GPT_dvaneft_shares_percentage_range_l1080_108047

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_dvaneft_shares_percentage_range_l1080_108047


namespace NUMINAMATH_GPT_saline_solution_mixture_l1080_108010

theorem saline_solution_mixture 
  (x : ℝ) 
  (h₁ : 20 + 0.1 * x = 0.25 * (50 + x)) 
  : x = 50 := 
by 
  sorry

end NUMINAMATH_GPT_saline_solution_mixture_l1080_108010


namespace NUMINAMATH_GPT_greg_experienced_less_rain_l1080_108069

theorem greg_experienced_less_rain (rain_day1 rain_day2 rain_day3 rain_house : ℕ) 
  (h1 : rain_day1 = 3) 
  (h2 : rain_day2 = 6) 
  (h3 : rain_day3 = 5) 
  (h4 : rain_house = 26) :
  rain_house - (rain_day1 + rain_day2 + rain_day3) = 12 :=
by
  sorry

end NUMINAMATH_GPT_greg_experienced_less_rain_l1080_108069


namespace NUMINAMATH_GPT_find_k_range_for_two_roots_l1080_108003

noncomputable def f (k x : ℝ) : ℝ := (Real.log x / x) - k * x

theorem find_k_range_for_two_roots :
  ∃ k_min k_max : ℝ, k_min = (2 / (Real.exp 4)) ∧ k_max = (1 / (2 * Real.exp 1)) ∧
  ∀ k : ℝ, (k_min ≤ k ∧ k < k_max) ↔
    ∃ x1 x2 : ℝ, 
    (1 / Real.exp 1) ≤ x1 ∧ x1 ≤ Real.exp 2 ∧ 
    (1 / Real.exp 1) ≤ x2 ∧ x2 ≤ Real.exp 2 ∧ 
    f k x1 = 0 ∧ f k x2 = 0 ∧ 
    x1 ≠ x2 :=
sorry

end NUMINAMATH_GPT_find_k_range_for_two_roots_l1080_108003


namespace NUMINAMATH_GPT_third_racer_sent_time_l1080_108015

theorem third_racer_sent_time (a : ℝ) (t t1 : ℝ) :
  t1 = 1.5 * t → 
  (1.25 * a) * (t1 - (1 / 2)) = 1.5 * a * t → 
  t = 5 / 3 → 
  (t1 - t) * 60 = 50 :=
by 
  intro h_t1_eq h_second_eq h_t_value
  rw [h_t1_eq] at h_second_eq
  have t_correct : t = 5 / 3 := h_t_value
  sorry

end NUMINAMATH_GPT_third_racer_sent_time_l1080_108015


namespace NUMINAMATH_GPT_longest_side_of_rectangle_l1080_108078

theorem longest_side_of_rectangle (l w : ℕ) 
  (h1 : 2 * l + 2 * w = 240) 
  (h2 : l * w = 1920) : 
  l = 101 ∨ w = 101 :=
sorry

end NUMINAMATH_GPT_longest_side_of_rectangle_l1080_108078


namespace NUMINAMATH_GPT_div_by_3_implies_one_div_by_3_l1080_108014

theorem div_by_3_implies_one_div_by_3 (a b : ℕ) (h_ab : 3 ∣ (a * b)) (h_na : ¬ 3 ∣ a) (h_nb : ¬ 3 ∣ b) : false :=
sorry

end NUMINAMATH_GPT_div_by_3_implies_one_div_by_3_l1080_108014


namespace NUMINAMATH_GPT_find_dimes_l1080_108080

-- Definitions for the conditions
def total_dollars : ℕ := 13
def dollar_bills_1 : ℕ := 2
def dollar_bills_5 : ℕ := 1
def quarters : ℕ := 13
def nickels : ℕ := 8
def pennies : ℕ := 35
def value_dollar_bill_1 : ℝ := 1.0
def value_dollar_bill_5 : ℝ := 5.0
def value_quarter : ℝ := 0.25
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_dime : ℝ := 0.10

-- Theorem statement
theorem find_dimes (total_dollars dollar_bills_1 dollar_bills_5 quarters nickels pennies : ℕ)
  (value_dollar_bill_1 value_dollar_bill_5 value_quarter value_nickel value_penny value_dime : ℝ) :
  (2 * value_dollar_bill_1 + 1 * value_dollar_bill_5 + 13 * value_quarter + 8 * value_nickel + 35 * value_penny) + 
  (20 * value_dime) = ↑total_dollars :=
sorry

end NUMINAMATH_GPT_find_dimes_l1080_108080


namespace NUMINAMATH_GPT_geometric_sequence_S28_l1080_108048

noncomputable def geom_sequence_sum (S : ℕ → ℝ) (a : ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, S (n * (n + 1) / 2) = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S28 {S : ℕ → ℝ} (a r : ℝ)
  (h1 : geom_sequence_sum S a r)
  (h2 : S 14 = 3)
  (h3 : 3 * S 7 = 3) :
  S 28 = 15 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S28_l1080_108048


namespace NUMINAMATH_GPT_find_valid_n_l1080_108029

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_valid_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : is_square (n^2 + (n + 1)^2)) :
  n = 3 ∨ n = 20 ∨ n = 119 :=
by
  sorry

end NUMINAMATH_GPT_find_valid_n_l1080_108029


namespace NUMINAMATH_GPT_solve_equation_solve_inequality_system_l1080_108065

theorem solve_equation :
  ∃ x, 2 * x^2 - 4 * x - 1 = 0 ∧ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
sorry

theorem solve_inequality_system : 
  ∀ x, (2 * x + 3 > 1 → -1 < x) ∧
       (x - 2 ≤ (1 / 2) * (x + 2) → x ≤ 6) ∧ 
       (2 * x + 3 > 1 ∧ x - 2 ≤ (1 / 2) * (x + 2) ↔ (-1 < x ∧ x ≤ 6)) :=
sorry

end NUMINAMATH_GPT_solve_equation_solve_inequality_system_l1080_108065


namespace NUMINAMATH_GPT_inequality_not_always_true_l1080_108081

theorem inequality_not_always_true {a b c : ℝ}
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : c ≠ 0) : ¬ ∀ c : ℝ, (a / c > b / c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_not_always_true_l1080_108081


namespace NUMINAMATH_GPT_altitude_of_triangle_l1080_108022

theorem altitude_of_triangle
  (a b c : ℝ)
  (h₁ : a = 13)
  (h₂ : b = 15)
  (h₃ : c = 22)
  (h₄ : a + b > c)
  (h₅ : a + c > b)
  (h₆ : b + c > a) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := (2 * A) / c
  h = (30 * Real.sqrt 10) / 11 :=
by
  sorry

end NUMINAMATH_GPT_altitude_of_triangle_l1080_108022


namespace NUMINAMATH_GPT_expression_evaluation_l1080_108062

def evaluate_expression : ℝ := (-1) ^ 51 + 3 ^ (2^3 + 5^2 - 7^2)

theorem expression_evaluation :
  evaluate_expression = -1 + (1 / 43046721) :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1080_108062


namespace NUMINAMATH_GPT_ratio_of_area_of_inscribed_circle_to_triangle_l1080_108023

theorem ratio_of_area_of_inscribed_circle_to_triangle (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  (π * r) / s = (5 * π * r) / (12 * h) :=
by
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  sorry

end NUMINAMATH_GPT_ratio_of_area_of_inscribed_circle_to_triangle_l1080_108023


namespace NUMINAMATH_GPT_net_hourly_rate_correct_l1080_108007

noncomputable def net_hourly_rate
    (hours : ℕ) 
    (speed : ℕ) 
    (fuel_efficiency : ℕ) 
    (earnings_per_mile : ℝ) 
    (cost_per_gallon : ℝ) 
    (distance := speed * hours) 
    (gasoline_used := distance / fuel_efficiency) 
    (earnings := earnings_per_mile * distance) 
    (cost_of_gasoline := cost_per_gallon * gasoline_used) 
    (net_earnings := earnings - cost_of_gasoline) : ℝ :=
  net_earnings / hours

theorem net_hourly_rate_correct : 
  net_hourly_rate 3 45 25 0.6 1.8 = 23.76 := 
by 
  unfold net_hourly_rate
  norm_num
  sorry

end NUMINAMATH_GPT_net_hourly_rate_correct_l1080_108007


namespace NUMINAMATH_GPT_power_of_three_l1080_108093

theorem power_of_three (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
by
  sorry

end NUMINAMATH_GPT_power_of_three_l1080_108093


namespace NUMINAMATH_GPT_triangle_area_l1080_108058

-- Define the lines and the x-axis
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := 1 - 5 * x
noncomputable def x_axis (x : ℝ) : ℝ := 0

-- Define intersection points
noncomputable def intersect_x_axis1 : ℝ × ℝ := (-1 / 2, 0)
noncomputable def intersect_x_axis2 : ℝ × ℝ := (1 / 5, 0)
noncomputable def intersect_lines : ℝ × ℝ := (0, 1)

-- State the theorem for the area of the triangle
theorem triangle_area : 
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  (1 / 2) * d * h = 7 / 20 := 
by
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  sorry

end NUMINAMATH_GPT_triangle_area_l1080_108058


namespace NUMINAMATH_GPT_no_real_y_for_two_equations_l1080_108087

theorem no_real_y_for_two_equations:
  ¬ ∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3 * y + 30 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_y_for_two_equations_l1080_108087


namespace NUMINAMATH_GPT_expand_product_l1080_108036

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by sorry

end NUMINAMATH_GPT_expand_product_l1080_108036


namespace NUMINAMATH_GPT_find_number_of_children_l1080_108020

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end NUMINAMATH_GPT_find_number_of_children_l1080_108020


namespace NUMINAMATH_GPT_volume_of_sphere_l1080_108039

theorem volume_of_sphere (r : ℝ) (h : r = 3) : (4 / 3) * π * r ^ 3 = 36 * π := 
by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_l1080_108039


namespace NUMINAMATH_GPT_distance_of_point_P_to_base_AB_l1080_108021

theorem distance_of_point_P_to_base_AB :
  ∀ (P : ℝ) (A B C : ℝ → ℝ)
    (h : ∀ (x : ℝ), A x = B x)
    (altitude : ℝ)
    (area_ratio : ℝ),
  altitude = 6 →
  area_ratio = 1 / 3 →
  (∃ d : ℝ, d = 6 - (2 / 3) * 6 ∧ d = 2) := 
  sorry

end NUMINAMATH_GPT_distance_of_point_P_to_base_AB_l1080_108021


namespace NUMINAMATH_GPT_valid_interval_for_a_l1080_108061

theorem valid_interval_for_a (a : ℝ) :
  (6 - 3 * a > 0) ∧ (a > 0) ∧ (3 * a^2 + a - 2 ≥ 0) ↔ (2 / 3 ≤ a ∧ a < 2 ∧ a ≠ 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_valid_interval_for_a_l1080_108061


namespace NUMINAMATH_GPT_fraction_cube_l1080_108028

theorem fraction_cube (a b : ℚ) (h : (a / b) ^ 3 = 15625 / 1000000) : a / b = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_cube_l1080_108028


namespace NUMINAMATH_GPT_mans_rate_in_still_water_l1080_108044

theorem mans_rate_in_still_water (Vm Vs : ℝ) (h1 : Vm + Vs = 14) (h2 : Vm - Vs = 4) : Vm = 9 :=
by
  sorry

end NUMINAMATH_GPT_mans_rate_in_still_water_l1080_108044


namespace NUMINAMATH_GPT_division_addition_l1080_108012

theorem division_addition :
  (-150 + 50) / (-50) = 2 := by
  sorry

end NUMINAMATH_GPT_division_addition_l1080_108012


namespace NUMINAMATH_GPT_basketball_competition_l1080_108016

theorem basketball_competition:
  (∃ x : ℕ, (0 ≤ x) ∧ (x ≤ 12) ∧ (3 * x - (12 - x) ≥ 28)) := by
  sorry

end NUMINAMATH_GPT_basketball_competition_l1080_108016


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1080_108084

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 2 - 2 * I) : z.im = -2 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l1080_108084


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1080_108057

theorem sale_in_fifth_month 
    (a1 a2 a3 a4 a6 : ℕ) 
    (avg_sale : ℕ)
    (H_avg : avg_sale = 8500)
    (H_a1 : a1 = 8435) 
    (H_a2 : a2 = 8927) 
    (H_a3 : a3 = 8855) 
    (H_a4 : a4 = 9230) 
    (H_a6 : a6 = 6991) : 
    ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sale ∧ a5 = 8562 := 
by
    sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1080_108057


namespace NUMINAMATH_GPT_percentage_of_dogs_l1080_108092

theorem percentage_of_dogs (total_pets : ℕ) (percent_cats : ℕ) (bunnies : ℕ) 
  (h1 : total_pets = 36) (h2 : percent_cats = 50) (h3 : bunnies = 9) : 
  ((total_pets - ((percent_cats * total_pets) / 100) - bunnies) / total_pets * 100) = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_of_dogs_l1080_108092


namespace NUMINAMATH_GPT_mary_added_peanuts_l1080_108042

-- Defining the initial number of peanuts
def initial_peanuts : ℕ := 4

-- Defining the final number of peanuts
def total_peanuts : ℕ := 10

-- Defining the number of peanuts added by Mary
def peanuts_added : ℕ := total_peanuts - initial_peanuts

-- The proof problem is to show that Mary added 6 peanuts
theorem mary_added_peanuts : peanuts_added = 6 :=
by
  -- We leave the proof part as a sorry as per instruction
  sorry

end NUMINAMATH_GPT_mary_added_peanuts_l1080_108042


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1080_108033

theorem trajectory_of_midpoint (x y : ℝ) (A B : ℝ × ℝ) 
  (hB : B = (4, 0)) (hA_on_circle : (A.1)^2 + (A.2)^2 = 4)
  (hM : ((x, y) = ( (A.1 + B.1)/2, (A.2 + B.2)/2))) :
  (x - 2)^2 + y^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1080_108033


namespace NUMINAMATH_GPT_range_of_a_l1080_108000

def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end NUMINAMATH_GPT_range_of_a_l1080_108000


namespace NUMINAMATH_GPT_money_left_after_shopping_l1080_108018

-- Definitions based on conditions
def initial_amount : ℝ := 5000
def percentage_spent : ℝ := 0.30
def amount_spent : ℝ := percentage_spent * initial_amount
def remaining_amount : ℝ := initial_amount - amount_spent

-- Theorem statement based on the question and correct answer
theorem money_left_after_shopping : remaining_amount = 3500 :=
by
  sorry

end NUMINAMATH_GPT_money_left_after_shopping_l1080_108018


namespace NUMINAMATH_GPT_quadratic_function_equal_values_l1080_108088

theorem quadratic_function_equal_values (a m n : ℝ) (h : a ≠ 0) (hmn : a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) : m + n = 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_equal_values_l1080_108088


namespace NUMINAMATH_GPT_intersection_points_l1080_108068

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 4
noncomputable def line (x : ℝ) : ℝ := -x + 2

theorem intersection_points :
  (parabola (-1 / 3) = line (-1 / 3) ∧ parabola (-2) = line (-2)) ∧
  (parabola (-1 / 3) = 7 / 3) ∧ (parabola (-2) = 4) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l1080_108068


namespace NUMINAMATH_GPT_comparison_of_f_values_l1080_108076

noncomputable def f (x : ℝ) := Real.cos x - x

theorem comparison_of_f_values :
  f (8 * Real.pi / 9) > f Real.pi ∧ f Real.pi > f (10 * Real.pi / 9) :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_f_values_l1080_108076


namespace NUMINAMATH_GPT_inequality_solution_l1080_108046

theorem inequality_solution :
  ∀ x : ℝ, (5 / 24 + |x - 11 / 48| < 5 / 16 ↔ (1 / 8 < x ∧ x < 1 / 3)) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_inequality_solution_l1080_108046


namespace NUMINAMATH_GPT_usual_time_56_l1080_108004

theorem usual_time_56 (S : ℝ) (T : ℝ) (h : (T + 24) * S = T * (0.7 * S)) : T = 56 :=
by sorry

end NUMINAMATH_GPT_usual_time_56_l1080_108004


namespace NUMINAMATH_GPT_remainder_polynomial_l1080_108037

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def r (x : ℝ) : ℝ := x^2 + x

theorem remainder_polynomial (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (p 2 = 6) ∧ (p 4 = 20) ∧ (p 6 = 42) →
  (r 2 = 2^2 + 2) ∧ (r 4 = 4^2 + 4) ∧ (r 6 = 6^2 + 6) :=
sorry

end NUMINAMATH_GPT_remainder_polynomial_l1080_108037


namespace NUMINAMATH_GPT_range_of_x_l1080_108074

variable (x y : ℝ)

def op (x y : ℝ) := x * (1 - y)

theorem range_of_x (h : op (x - 1) (x + 2) < 0) : x < -1 ∨ 1 < x :=
by
  dsimp [op] at h
  sorry

end NUMINAMATH_GPT_range_of_x_l1080_108074


namespace NUMINAMATH_GPT_trigonometric_identity_l1080_108013

theorem trigonometric_identity (θ : ℝ) (h : 2 * (Real.cos θ) + (Real.sin θ) = 0) :
  Real.cos (2 * θ) + 1/2 * Real.sin (2 * θ) = -1 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1080_108013


namespace NUMINAMATH_GPT_height_difference_percentage_l1080_108060

theorem height_difference_percentage (q p : ℝ) (h : p = 0.6 * q) : (q - p) / p * 100 = 66.67 := 
by
  sorry

end NUMINAMATH_GPT_height_difference_percentage_l1080_108060


namespace NUMINAMATH_GPT_trajectory_of_point_l1080_108051

theorem trajectory_of_point (x y : ℝ)
  (h1 : (x - 1)^2 + (y - 1)^2 = ((3 * x + y - 4)^2) / 10) :
  x - 3 * y + 2 = 0 :=
sorry

end NUMINAMATH_GPT_trajectory_of_point_l1080_108051


namespace NUMINAMATH_GPT_solve_cubic_root_eq_l1080_108006

theorem solve_cubic_root_eq (x : ℝ) (h : (5 - x)^(1/3) = 4) : x = -59 := 
by
  sorry

end NUMINAMATH_GPT_solve_cubic_root_eq_l1080_108006


namespace NUMINAMATH_GPT_solution_set_of_inequalities_l1080_108071

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequalities_l1080_108071


namespace NUMINAMATH_GPT_expected_value_is_6_5_l1080_108019

noncomputable def expected_value_12_sided_die : ℚ :=
  (1 / 12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

theorem expected_value_is_6_5 : expected_value_12_sided_die = 6.5 := 
by
  sorry

end NUMINAMATH_GPT_expected_value_is_6_5_l1080_108019


namespace NUMINAMATH_GPT_min_perimeter_lateral_face_l1080_108053

theorem min_perimeter_lateral_face (x h : ℝ) (V : ℝ) (P : ℝ): 
  (x > 0) → (h > 0) → (V = 4) → (V = x^2 * h) → 
  (∀ y : ℝ, y > 0 → 2*y + 2 * (4 / y^2) ≥ P) → P = 6 := 
by
  intro x_pos h_pos volume_eq volume_expr min_condition
  sorry

end NUMINAMATH_GPT_min_perimeter_lateral_face_l1080_108053


namespace NUMINAMATH_GPT_gpa_at_least_3_5_l1080_108005

noncomputable def prob_gpa_at_least_3_5 : ℚ :=
  let p_A_eng := 1 / 3
  let p_B_eng := 1 / 5
  let p_C_eng := 7 / 15 -- 1 - p_A_eng - p_B_eng
  
  let p_A_hist := 1 / 5
  let p_B_hist := 1 / 4
  let p_C_hist := 11 / 20 -- 1 - p_A_hist - p_B_hist

  let prob_two_As := p_A_eng * p_A_hist
  let prob_A_eng_B_hist := p_A_eng * p_B_hist
  let prob_A_hist_B_eng := p_A_hist * p_B_eng
  let prob_two_Bs := p_B_eng * p_B_hist

  let total_prob := prob_two_As + prob_A_eng_B_hist + prob_A_hist_B_eng + prob_two_Bs
  total_prob

theorem gpa_at_least_3_5 : prob_gpa_at_least_3_5 = 6 / 25 := by {
  sorry
}

end NUMINAMATH_GPT_gpa_at_least_3_5_l1080_108005


namespace NUMINAMATH_GPT_nth_equation_l1080_108066

theorem nth_equation (n : ℕ) : 
  n^2 + (n + 1)^2 = (n * (n + 1) + 1)^2 - (n * (n + 1))^2 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l1080_108066


namespace NUMINAMATH_GPT_Tom_search_cost_l1080_108040

theorem Tom_search_cost (first_5_days_rate: ℕ) (first_5_days: ℕ) (remaining_days_rate: ℕ) (total_days: ℕ) : 
  first_5_days_rate = 100 → 
  first_5_days = 5 → 
  remaining_days_rate = 60 → 
  total_days = 10 → 
  (first_5_days * first_5_days_rate + (total_days - first_5_days) * remaining_days_rate) = 800 := 
by 
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_Tom_search_cost_l1080_108040


namespace NUMINAMATH_GPT_calculate_pow_zero_l1080_108041

theorem calculate_pow_zero: (2023 - Real.pi) ≠ 0 → (2023 - Real.pi)^0 = 1 := by
  -- Proof
  sorry

end NUMINAMATH_GPT_calculate_pow_zero_l1080_108041


namespace NUMINAMATH_GPT_initial_action_figures_correct_l1080_108082

def initial_action_figures (x : ℕ) : Prop :=
  x + 11 - 10 = 8

theorem initial_action_figures_correct :
  ∃ x : ℕ, initial_action_figures x ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_initial_action_figures_correct_l1080_108082


namespace NUMINAMATH_GPT_no_real_solutions_l1080_108034

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3 * x + 8)^2 + 4 = -2 * |x| :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1080_108034


namespace NUMINAMATH_GPT_general_equation_of_curve_l1080_108045

variable (θ x y : ℝ)

theorem general_equation_of_curve
  (h1 : x = Real.cos θ - 1)
  (h2 : y = Real.sin θ + 1) :
  (x + 1)^2 + (y - 1)^2 = 1 := sorry

end NUMINAMATH_GPT_general_equation_of_curve_l1080_108045


namespace NUMINAMATH_GPT_polar_coordinates_of_2_neg2_l1080_108035

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (ρ, θ)

theorem polar_coordinates_of_2_neg2 :
  polar_coordinates 2 (-2) = (2 * Real.sqrt 2, -Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_polar_coordinates_of_2_neg2_l1080_108035


namespace NUMINAMATH_GPT_ratio_sheep_horses_l1080_108050

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_sheep_horses_l1080_108050


namespace NUMINAMATH_GPT_max_a_if_monotonically_increasing_l1080_108025

noncomputable def f (x a : ℝ) : ℝ := x^3 + Real.exp x - a * x

theorem max_a_if_monotonically_increasing (a : ℝ) : 
  (∀ x, 0 ≤ x → 3 * x^2 + Real.exp x - a ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_max_a_if_monotonically_increasing_l1080_108025


namespace NUMINAMATH_GPT_largest_constant_inequality_l1080_108072

theorem largest_constant_inequality :
  ∃ C, (∀ x y z : ℝ, x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧ (C = Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_largest_constant_inequality_l1080_108072


namespace NUMINAMATH_GPT_value_of_expression_l1080_108067

theorem value_of_expression (x y : ℤ) (h1 : x = 1) (h2 : y = 630) : 
  2019 * x - 3 * y - 9 = 120 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1080_108067


namespace NUMINAMATH_GPT_parallel_lines_sufficient_condition_l1080_108097

theorem parallel_lines_sufficient_condition :
  ∀ a : ℝ, (a^2 - a) = 2 → (a = 2 ∨ a = -1) :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_parallel_lines_sufficient_condition_l1080_108097


namespace NUMINAMATH_GPT_brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l1080_108024

def march_13_2007_day_of_week : String := "Tuesday"

def days_until_brothers_birthday : Nat := 2000

def start_date := (2007, 3, 13)  -- (year, month, day)

def days_per_week := 7

def carlos_initial_age := 7

def day_of_week_after_n_days (start_day : String) (n : Nat) : String :=
  match n % 7 with
  | 0 => "Tuesday"
  | 1 => "Wednesday"
  | 2 => "Thursday"
  | 3 => "Friday"
  | 4 => "Saturday"
  | 5 => "Sunday"
  | 6 => "Monday"
  | _ => "Unknown" -- This case should never happen

def carlos_age_after_n_days (initial_age : Nat) (n : Nat) : Nat :=
  initial_age + n / 365

theorem brother_15th_birthday_day_of_week : 
  day_of_week_after_n_days march_13_2007_day_of_week days_until_brothers_birthday = "Sunday" := 
by sorry

theorem carlos_age_on_brothers_15th_birthday :
  carlos_age_after_n_days carlos_initial_age days_until_brothers_birthday = 12 :=
by sorry

end NUMINAMATH_GPT_brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l1080_108024


namespace NUMINAMATH_GPT_tan_15pi_over_4_is_neg1_l1080_108094

noncomputable def tan_15pi_over_4 : ℝ :=
  Real.tan (15 * Real.pi / 4)

theorem tan_15pi_over_4_is_neg1 :
  tan_15pi_over_4 = -1 :=
sorry

end NUMINAMATH_GPT_tan_15pi_over_4_is_neg1_l1080_108094


namespace NUMINAMATH_GPT_line_slope_l1080_108002

theorem line_slope (x y : ℝ) : 3 * y - (1 / 2) * x = 9 → ∃ m, m = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_line_slope_l1080_108002


namespace NUMINAMATH_GPT_find_b_if_lines_parallel_l1080_108009

theorem find_b_if_lines_parallel (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → y = 3 * x + b) ∧
  (∀ x y : ℝ, y + 2 = (b + 9) * x → y = (b + 9) * x - 2) →
  3 = b + 9 →
  b = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_if_lines_parallel_l1080_108009


namespace NUMINAMATH_GPT_tan_product_identity_l1080_108030

theorem tan_product_identity : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 3)) = 4 + 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_identity_l1080_108030


namespace NUMINAMATH_GPT_inequality_holds_l1080_108059

theorem inequality_holds (a b c : ℝ) 
  (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1080_108059


namespace NUMINAMATH_GPT_linear_regression_change_l1080_108085

theorem linear_regression_change : ∀ (x : ℝ), ∀ (y : ℝ), 
  y = 2 - 3.5 * x → (y - (2 - 3.5 * (x + 1))) = 3.5 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_linear_regression_change_l1080_108085


namespace NUMINAMATH_GPT_terminating_fraction_count_l1080_108077

theorem terminating_fraction_count : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 299 ∧ (∃ k, n = 3 * k)) ∧ 
  (∃ (count : ℕ), count = 99) :=
by
  sorry

end NUMINAMATH_GPT_terminating_fraction_count_l1080_108077


namespace NUMINAMATH_GPT_find_A_l1080_108052

variable (x A B C : ℝ)

theorem find_A :
  (∃ A B C : ℝ, (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → 
  (1 / (x^3 + 2 * x^2 - 19 * x - 30) = 
  (A / (x + 3)) + (B / (x - 2)) + (C / (x - 2)^2)) ∧ 
  A = 1 / 25)) :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1080_108052


namespace NUMINAMATH_GPT_NaCl_moles_formed_l1080_108086

-- Definitions for the conditions
def NaOH_moles : ℕ := 2
def Cl2_moles : ℕ := 1

-- Chemical reaction of NaOH and Cl2 resulting in NaCl and H2O
def reaction (n_NaOH n_Cl2 : ℕ) : ℕ :=
  if n_NaOH = 2 ∧ n_Cl2 = 1 then 2 else 0

-- Statement to be proved
theorem NaCl_moles_formed : reaction NaOH_moles Cl2_moles = 2 :=
by
  sorry

end NUMINAMATH_GPT_NaCl_moles_formed_l1080_108086


namespace NUMINAMATH_GPT_toms_friend_decks_l1080_108049

theorem toms_friend_decks
  (cost_per_deck : ℕ)
  (tom_decks : ℕ)
  (total_spent : ℕ)
  (h1 : cost_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : total_spent = 64) :
  (total_spent - tom_decks * cost_per_deck) / cost_per_deck = 5 := by
  sorry

end NUMINAMATH_GPT_toms_friend_decks_l1080_108049


namespace NUMINAMATH_GPT_external_tangent_b_value_l1080_108095

theorem external_tangent_b_value:
  ∀ {C1 C2 : ℝ × ℝ} (r1 r2 : ℝ) (m b : ℝ),
  C1 = (3, -2) ∧ r1 = 3 ∧ 
  C2 = (15, 8) ∧ r2 = 8 ∧
  m = (60 / 11) →
  (∃ b, y = m * x + b ∧ b = 720 / 11) :=
by 
  sorry

end NUMINAMATH_GPT_external_tangent_b_value_l1080_108095


namespace NUMINAMATH_GPT_commutative_not_associative_l1080_108056

variable (k : ℝ) (h_k : 0 < k)

noncomputable def star (x y : ℝ) : ℝ := (x * y + k) / (x + y + k)

theorem commutative (x y : ℝ) (h_x : 0 < x) (h_y : 0 < y) :
  star k x y = star k y x :=
by sorry

theorem not_associative (x y z : ℝ) (h_x : 0 < x) (h_y : 0 < y) (h_z : 0 < z) :
  ¬(star k (star k x y) z = star k x (star k y z)) :=
by sorry

end NUMINAMATH_GPT_commutative_not_associative_l1080_108056


namespace NUMINAMATH_GPT_exists_multiple_with_equal_digit_sum_l1080_108073

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_with_equal_digit_sum (k : ℕ) (h : k > 0) : 
  ∃ n : ℕ, (n % k = 0) ∧ (sum_of_digits n = sum_of_digits (n * n)) :=
sorry

end NUMINAMATH_GPT_exists_multiple_with_equal_digit_sum_l1080_108073


namespace NUMINAMATH_GPT_rotated_clockwise_120_correct_l1080_108083

-- Problem setup definitions
structure ShapePosition :=
  (triangle : Point)
  (smaller_circle : Point)
  (square : Point)

-- Conditions for the initial positions of the shapes
variable (initial : ShapePosition)

def rotated_positions (initial: ShapePosition) : ShapePosition :=
  { 
    triangle := initial.smaller_circle,
    smaller_circle := initial.square,
    square := initial.triangle 
  }

-- Problem statement: show that after a 120° clockwise rotation, 
-- the shapes move to the specified new positions.
theorem rotated_clockwise_120_correct (initial : ShapePosition) 
  (after_rotation : ShapePosition) :
  after_rotation = rotated_positions initial := 
sorry

end NUMINAMATH_GPT_rotated_clockwise_120_correct_l1080_108083


namespace NUMINAMATH_GPT_ratio_fourth_to_sixth_l1080_108063

-- Definitions from the conditions
def fourth_level_students := 40
def sixth_level_students := 40
def seventh_level_students := 2 * fourth_level_students

-- Statement to prove
theorem ratio_fourth_to_sixth : 
  fourth_level_students / sixth_level_students = 1 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_ratio_fourth_to_sixth_l1080_108063


namespace NUMINAMATH_GPT_cos_three_theta_l1080_108096

theorem cos_three_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end NUMINAMATH_GPT_cos_three_theta_l1080_108096


namespace NUMINAMATH_GPT_integer_not_in_range_l1080_108079

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end NUMINAMATH_GPT_integer_not_in_range_l1080_108079


namespace NUMINAMATH_GPT_regular_polygon_sides_l1080_108089

theorem regular_polygon_sides (n : ℕ) (h : ∀ (x : ℕ), x = 180 * (n - 2) / n → x = 144) :
  n = 10 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1080_108089


namespace NUMINAMATH_GPT_line_graph_displays_trend_l1080_108091

-- Define the types of statistical graphs
inductive StatisticalGraph : Type
| barGraph : StatisticalGraph
| lineGraph : StatisticalGraph
| pieChart : StatisticalGraph
| histogram : StatisticalGraph

-- Define the property of displaying trends over time
def displaysTrend (g : StatisticalGraph) : Prop := 
  g = StatisticalGraph.lineGraph

-- Theorem to prove that the type of statistical graph that displays the trend of data is the line graph
theorem line_graph_displays_trend : displaysTrend StatisticalGraph.lineGraph :=
sorry

end NUMINAMATH_GPT_line_graph_displays_trend_l1080_108091


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1080_108032

theorem quadratic_inequality_solution :
  {x : ℝ | 2*x^2 - 3*x - 2 ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1080_108032


namespace NUMINAMATH_GPT_range_of_a_l1080_108027

noncomputable def g (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x + 3

theorem range_of_a 
  (h_mono_inc : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → g a x1 ≤ g a x2)
  (h_nonneg : ∀ x : ℝ, -1 < x ∧ x < 1 → 0 ≤ g a x) :
  (-2 : ℝ) ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1080_108027


namespace NUMINAMATH_GPT_x5_plus_y5_l1080_108031

theorem x5_plus_y5 (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 252 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_x5_plus_y5_l1080_108031


namespace NUMINAMATH_GPT_avg_speed_round_trip_l1080_108017

-- Definitions for the conditions
def speed_P_to_Q : ℝ := 80
def distance (D : ℝ) : ℝ := D
def speed_increase_percentage : ℝ := 0.1
def speed_Q_to_P : ℝ := speed_P_to_Q * (1 + speed_increase_percentage)

-- Average speed calculation function
noncomputable def average_speed (D : ℝ) : ℝ := 
  let total_distance := 2 * D
  let time_P_to_Q := D / speed_P_to_Q
  let time_Q_to_P := D / speed_Q_to_P
  let total_time := time_P_to_Q + time_Q_to_P
  total_distance / total_time

-- Theorem: Average speed for the round trip is 83.81 km/hr
theorem avg_speed_round_trip (D : ℝ) : average_speed D = 83.81 := 
by 
  -- Dummy proof placeholder
  sorry

end NUMINAMATH_GPT_avg_speed_round_trip_l1080_108017


namespace NUMINAMATH_GPT_top_card_is_5_or_king_l1080_108070

-- Define the number of cards in a deck
def total_cards : ℕ := 52

-- Define the number of 5s in a deck
def number_of_5s : ℕ := 4

-- Define the number of Kings in a deck
def number_of_kings : ℕ := 4

-- Define the number of favorable outcomes (cards that are either 5 or King)
def favorable_outcomes : ℕ := number_of_5s + number_of_kings

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_cards

-- Theorem: The probability that the top card is either a 5 or a King is 2/13
theorem top_card_is_5_or_king (h_total_cards : total_cards = 52)
    (h_number_of_5s : number_of_5s = 4)
    (h_number_of_kings : number_of_kings = 4) :
    probability = 2 / 13 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_top_card_is_5_or_king_l1080_108070


namespace NUMINAMATH_GPT_james_nickels_l1080_108008

theorem james_nickels (p n : ℕ) (h₁ : p + n = 50) (h₂ : p + 5 * n = 150) : n = 25 :=
by
  -- Skipping the proof since only the statement is required
  sorry

end NUMINAMATH_GPT_james_nickels_l1080_108008


namespace NUMINAMATH_GPT_circle_equation_l1080_108001

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1080_108001


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1080_108055

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : 1/x = 3 * (1/y)) : 
  x + y = 16 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1080_108055


namespace NUMINAMATH_GPT_jesse_mia_total_miles_per_week_l1080_108099

noncomputable def jesse_miles_per_day_first_three := 2 / 3
noncomputable def jesse_miles_day_four := 10
noncomputable def mia_miles_per_day_first_four := 3
noncomputable def average_final_three_days := 6

theorem jesse_mia_total_miles_per_week :
  let jesse_total_first_four_days := 3 * jesse_miles_per_day_first_three + jesse_miles_day_four
  let mia_total_first_four_days := 4 * mia_miles_per_day_first_four
  let total_miles_needed_final_three_days := 3 * average_final_three_days * 2
  jesse_total_first_four_days + total_miles_needed_final_three_days = 48 ∧
  mia_total_first_four_days + total_miles_needed_final_three_days = 48 :=
by
  sorry

end NUMINAMATH_GPT_jesse_mia_total_miles_per_week_l1080_108099
