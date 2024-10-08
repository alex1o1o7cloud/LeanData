import Mathlib

namespace least_common_multiple_1008_672_l67_67136

theorem least_common_multiple_1008_672 : Nat.lcm 1008 672 = 2016 := by
  -- Add the prime factorizations and show the LCM calculation
  have h1 : 1008 = 2^4 * 3^2 * 7 := by sorry
  have h2 : 672 = 2^5 * 3 * 7 := by sorry
  -- Utilize the factorizations to compute LCM
  have calc1 : Nat.lcm (2^4 * 3^2 * 7) (2^5 * 3 * 7) = 2^5 * 3^2 * 7 := by sorry
  -- Show the calculation of 2^5 * 3^2 * 7
  have calc2 : 2^5 * 3^2 * 7 = 2016 := by sorry
  -- Therefore, LCM of 1008 and 672 is 2016
  exact calc2

end least_common_multiple_1008_672_l67_67136


namespace percentage_saved_l67_67750

theorem percentage_saved (amount_saved : ℝ) (amount_spent : ℝ) (h1 : amount_saved = 5) (h2 : amount_spent = 45) : 
  (amount_saved / (amount_spent + amount_saved)) * 100 = 10 :=
by 
  sorry

end percentage_saved_l67_67750


namespace sum_of_six_smallest_multiples_of_12_l67_67732

-- Define the six smallest distinct positive integer multiples of 12
def multiples_of_12 : List ℕ := [12, 24, 36, 48, 60, 72]

-- Define their sum
def sum_of_multiples : ℕ := multiples_of_12.sum

-- The proof statement
theorem sum_of_six_smallest_multiples_of_12 : sum_of_multiples = 252 := 
by
  sorry

end sum_of_six_smallest_multiples_of_12_l67_67732


namespace find_a6_l67_67185

open Nat

noncomputable def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def a2 := 4
def a4 := 2

theorem find_a6 (a1 d : ℤ) (h_a2 : arith_seq a1 d 2 = a2) (h_a4 : arith_seq a1 d 4 = a4) : 
  arith_seq a1 d 6 = 0 := by
  sorry

end find_a6_l67_67185


namespace paula_go_kart_rides_l67_67022

theorem paula_go_kart_rides
  (g : ℕ)
  (ticket_cost_go_karts : ℕ := 4 * g)
  (ticket_cost_bumper_cars : ℕ := 20)
  (total_tickets : ℕ := 24) :
  ticket_cost_go_karts + ticket_cost_bumper_cars = total_tickets → g = 1 :=
by {
  sorry
}

end paula_go_kart_rides_l67_67022


namespace sin_2theta_in_third_quadrant_l67_67687

open Real

variables (θ : ℝ)

/-- \theta is an angle in the third quadrant.
Given that \(\sin^{4}\theta + \cos^{4}\theta = \frac{5}{9}\), 
prove that \(\sin 2\theta = \frac{2\sqrt{2}}{3}\). --/
theorem sin_2theta_in_third_quadrant (h_theta_third_quadrant : π < θ ∧ θ < 3 * π / 2)
(h_cond : sin θ ^ 4 + cos θ ^ 4 = 5 / 9) : sin (2 * θ) = 2 * sqrt 2 / 3 :=
sorry

end sin_2theta_in_third_quadrant_l67_67687


namespace profit_percentage_l67_67327

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 75) : 
  ((selling_price - cost_price) / cost_price) * 100 = 25 := by
  sorry

end profit_percentage_l67_67327


namespace sum_remainder_div_9_l67_67449

theorem sum_remainder_div_9 : 
  let S := (20 / 2) * (1 + 20)
  S % 9 = 3 := 
by
  -- use let S to simplify the proof
  let S := (20 / 2) * (1 + 20)
  -- sum of first 20 natural numbers
  have H1 : S = 210 := by sorry
  -- division and remainder result
  have H2 : 210 % 9 = 3 := by sorry
  -- combine both results to conclude 
  exact H2

end sum_remainder_div_9_l67_67449


namespace solve_sqrt_equation_l67_67085

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end solve_sqrt_equation_l67_67085


namespace triangle_angle_sum_l67_67724

theorem triangle_angle_sum (a : ℝ) (x : ℝ) :
  0 < 2 * a + 20 ∧ 0 < 3 * a - 15 ∧ 0 < 175 - 5 * a ∧
  2 * a + 20 + 3 * a - 15 + x = 180 → 
  x = 175 - 5 * a ∧ max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
sorry

end triangle_angle_sum_l67_67724


namespace general_term_formula_for_sequence_l67_67681

theorem general_term_formula_for_sequence (a b : ℕ → ℝ) 
  (h1 : ∀ n, 2 * b n = a n + a (n + 1)) 
  (h2 : ∀ n, (a (n + 1))^2 = b n * b (n + 1)) 
  (h3 : a 1 = 1) 
  (h4 : a 2 = 3) :
  ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_formula_for_sequence_l67_67681


namespace find_x_l67_67134

theorem find_x (x : ℝ) (h : 0.5 * x = 0.05 * 500 - 20) : x = 10 :=
by
  sorry

end find_x_l67_67134


namespace kati_age_l67_67701

/-- Define the age of Kati using the given conditions -/
theorem kati_age (kati_age : ℕ) (brother_age kati_birthdays : ℕ) 
  (h1 : kati_age = kati_birthdays) 
  (h2 : kati_age + brother_age = 111) 
  (h3 : kati_birthdays = kati_age) : 
  kati_age = 18 :=
by
  sorry

end kati_age_l67_67701


namespace j_h_five_l67_67261

-- Define the functions h and j
def h (x : ℤ) : ℤ := 4 * x + 5
def j (x : ℤ) : ℤ := 6 * x - 11

-- State the theorem to prove j(h(5)) = 139
theorem j_h_five : j (h 5) = 139 := by
  sorry

end j_h_five_l67_67261


namespace car_mpg_city_l67_67104

theorem car_mpg_city
  (h c T : ℝ)
  (h1 : h * T = 480)
  (h2 : c * T = 336)
  (h3 : c = h - 6) :
  c = 14 :=
by
  sorry

end car_mpg_city_l67_67104


namespace point_in_second_quadrant_iff_l67_67037

theorem point_in_second_quadrant_iff (a : ℝ) : (a - 2 < 0) ↔ (a < 2) :=
by
  sorry

end point_in_second_quadrant_iff_l67_67037


namespace quarter_circle_area_ratio_l67_67534

theorem quarter_circle_area_ratio (R : ℝ) (hR : 0 < R) :
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  (2 * (AXC + BYD) / O = 1 / 8) := 
by
  let O := pi * R^2
  let AXC := pi * (R/2)^2 / 4
  let BYD := pi * (R/2)^2 / 4
  sorry

end quarter_circle_area_ratio_l67_67534


namespace math_problem_l67_67519

theorem math_problem : 2 + 5 * 4 - 6 + 3 = 19 := by
  sorry

end math_problem_l67_67519


namespace outdoor_tables_count_l67_67820

theorem outdoor_tables_count (num_indoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) : ℕ :=
  let num_outdoor_tables := (total_chairs - (num_indoor_tables * chairs_per_indoor_table)) / chairs_per_outdoor_table
  num_outdoor_tables

example (h₁ : num_indoor_tables = 9)
        (h₂ : chairs_per_indoor_table = 10)
        (h₃ : chairs_per_outdoor_table = 3)
        (h₄ : total_chairs = 123) :
        outdoor_tables_count 9 10 3 123 = 11 :=
by
  -- Only the statement has to be provided; proof steps are not needed
  sorry

end outdoor_tables_count_l67_67820


namespace min_value_of_2a_b_c_l67_67453

-- Given conditions
variables (a b c : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (h : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3)

-- Question to prove
theorem min_value_of_2a_b_c : 2 * a + b + c = 2 * Real.sqrt 3 + 2 :=
sorry

end min_value_of_2a_b_c_l67_67453


namespace base_three_to_base_ten_l67_67426

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end base_three_to_base_ten_l67_67426


namespace percentage_spent_on_household_items_eq_50_l67_67275

-- Definitions for the conditions in the problem
def MonthlyIncome : ℝ := 90000
def ClothesPercentage : ℝ := 0.25
def MedicinesPercentage : ℝ := 0.15
def Savings : ℝ := 9000

-- Definition of the statement where we need to calculate the percentage spent on household items
theorem percentage_spent_on_household_items_eq_50 :
  let ClothesExpense := ClothesPercentage * MonthlyIncome
  let MedicinesExpense := MedicinesPercentage * MonthlyIncome
  let TotalExpense := ClothesExpense + MedicinesExpense + Savings
  let HouseholdItemsExpense := MonthlyIncome - TotalExpense
  let TotalIncome := MonthlyIncome
  (HouseholdItemsExpense / TotalIncome) * 100 = 50 :=
by
  sorry

end percentage_spent_on_household_items_eq_50_l67_67275


namespace original_fraction_is_two_thirds_l67_67955

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end original_fraction_is_two_thirds_l67_67955


namespace circuit_disconnected_scenarios_l67_67078

def num_scenarios_solder_points_fall_off (n : Nat) : Nat :=
  2 ^ n - 1

theorem circuit_disconnected_scenarios : num_scenarios_solder_points_fall_off 6 = 63 :=
by
  sorry

end circuit_disconnected_scenarios_l67_67078


namespace james_lifting_heavy_after_39_days_l67_67162

noncomputable def JamesInjuryHealingTime : Nat := 3
noncomputable def HealingTimeFactor : Nat := 5
noncomputable def WaitingTimeAfterHealing : Nat := 3
noncomputable def AdditionalWaitingTimeWeeks : Nat := 3

theorem james_lifting_heavy_after_39_days :
  let healing_time := JamesInjuryHealingTime * HealingTimeFactor
  let total_time_before_workout := healing_time + WaitingTimeAfterHealing
  let additional_waiting_time_days := AdditionalWaitingTimeWeeks * 7
  let total_time_before_lifting_heavy := total_time_before_workout + additional_waiting_time_days
  total_time_before_lifting_heavy = 39 := by
  sorry

end james_lifting_heavy_after_39_days_l67_67162


namespace implication_equivalence_l67_67478

variable (P Q : Prop)

theorem implication_equivalence :
  (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by sorry

end implication_equivalence_l67_67478


namespace sqrt_three_cubes_l67_67373

theorem sqrt_three_cubes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := 
  sorry

end sqrt_three_cubes_l67_67373


namespace sandwich_percentage_not_vegetables_l67_67618

noncomputable def percentage_not_vegetables (total_weight : ℝ) (vegetable_weight : ℝ) : ℝ :=
  (total_weight - vegetable_weight) / total_weight * 100

theorem sandwich_percentage_not_vegetables :
  percentage_not_vegetables 180 50 = 72.22 :=
by
  sorry

end sandwich_percentage_not_vegetables_l67_67618


namespace possible_values_of_C_l67_67088

theorem possible_values_of_C {a b C : ℤ} :
  (C = a * (a - 5) ∧ C = b * (b - 8)) ↔ (C = 0 ∨ C = 84) :=
sorry

end possible_values_of_C_l67_67088


namespace total_snowfall_yardley_l67_67362

theorem total_snowfall_yardley (a b c d : ℝ) (ha : a = 0.12) (hb : b = 0.24) (hc : c = 0.5) (hd : d = 0.36) :
  a + b + c + d = 1.22 :=
by
  sorry

end total_snowfall_yardley_l67_67362


namespace train_average_speed_with_stoppages_l67_67011

theorem train_average_speed_with_stoppages (D : ℝ) :
  let speed_without_stoppages := 200
  let stoppage_time_per_hour_in_hours := 12 / 60.0
  let effective_running_time := 1 - stoppage_time_per_hour_in_hours
  let speed_with_stoppages := effective_running_time * speed_without_stoppages
  speed_with_stoppages = 160 := by
  sorry

end train_average_speed_with_stoppages_l67_67011


namespace sequence_values_l67_67649

theorem sequence_values (x y z : ℚ) :
  (∀ n : ℕ, x = 1 ∧ y = 9 / 8 ∧ z = 5 / 4) :=
by
  sorry

end sequence_values_l67_67649


namespace balls_sold_eq_13_l67_67299

-- Let SP be the selling price, CP be the cost price per ball, and loss be the loss incurred.
def SP : ℕ := 720
def CP : ℕ := 90
def loss : ℕ := 5 * CP
def total_CP (n : ℕ) : ℕ := n * CP

-- Given the conditions:
axiom loss_eq : loss = 5 * CP
axiom ball_CP_value : CP = 90
axiom selling_price_value : SP = 720

-- Loss is defined as total cost price minus selling price
def calculated_loss (n : ℕ) : ℕ := total_CP n - SP

-- The proof statement:
theorem balls_sold_eq_13 (n : ℕ) (h1 : calculated_loss n = loss) : n = 13 :=
by sorry

end balls_sold_eq_13_l67_67299


namespace sum_squares_not_perfect_square_l67_67849

theorem sum_squares_not_perfect_square (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) : ¬ ∃ a : ℤ, x + y + z = a^2 :=
sorry

end sum_squares_not_perfect_square_l67_67849


namespace math_equivalence_example_l67_67550

theorem math_equivalence_example :
  ((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2 = 494.09014144 := 
by
  sorry

end math_equivalence_example_l67_67550


namespace greatest_product_of_two_integers_with_sum_300_l67_67685

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l67_67685


namespace frac_sum_eq_one_l67_67521

variable {x y : ℝ}

theorem frac_sum_eq_one (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x) + (1 / y) = 1 :=
by sorry

end frac_sum_eq_one_l67_67521


namespace octahedron_tetrahedron_volume_ratio_l67_67847

theorem octahedron_tetrahedron_volume_ratio (a : ℝ) :
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3
  V_o / V_t = 1 :=
by 
  -- Definitions from conditions
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3

  -- Proof omitted
  -- Proof goes here
  sorry

end octahedron_tetrahedron_volume_ratio_l67_67847


namespace dot_product_correct_l67_67948

-- Define the vectors as given conditions
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (1, -2)

-- State the theorem to prove the dot product
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = -4 := by
  -- Proof steps go here
  sorry

end dot_product_correct_l67_67948


namespace inverse_function_ratio_l67_67265

noncomputable def g (x : ℚ) : ℚ := (3 * x + 2) / (2 * x - 5)

noncomputable def g_inv (x : ℚ) : ℚ := (-5 * x + 2) / (-2 * x + 3)

theorem inverse_function_ratio :
  ∀ x : ℚ, g (g_inv x) = x ∧ (∃ a b c d : ℚ, a = -5 ∧ b = 2 ∧ c = -2 ∧ d = 3 ∧ a / c = 2.5) :=
by
  sorry

end inverse_function_ratio_l67_67265


namespace bonnie_roark_wire_ratio_l67_67399

theorem bonnie_roark_wire_ratio :
  let bonnie_wire_length := 12 * 8
  let bonnie_cube_volume := 8 ^ 3
  let roark_cube_volume := 2
  let roark_edge_length := 1.5
  let roark_cube_edge_count := 12
  let num_roark_cubes := bonnie_cube_volume / roark_cube_volume
  let roark_wire_per_cube := roark_cube_edge_count * roark_edge_length
  let roark_total_wire := num_roark_cubes * roark_wire_per_cube
  bonnie_wire_length / roark_total_wire = 1 / 48 :=
  by
  sorry

end bonnie_roark_wire_ratio_l67_67399


namespace correct_transformation_l67_67213

theorem correct_transformation (m : ℤ) (h : 2 * m - 1 = 3) : 2 * m = 4 :=
by
  sorry

end correct_transformation_l67_67213


namespace isabel_reading_homework_pages_l67_67772

-- Definitions for the given problem
def num_math_pages := 2
def problems_per_page := 5
def total_problems := 30

-- Calculation based on conditions
def math_problems := num_math_pages * problems_per_page
def reading_problems := total_problems - math_problems

-- The statement to be proven
theorem isabel_reading_homework_pages : (reading_problems / problems_per_page) = 4 :=
by
  -- The proof would go here.
  sorry

end isabel_reading_homework_pages_l67_67772


namespace find_m_l67_67579

-- We define the universal set U, the set A with an unknown m, and the complement of A in U.
def U : Set ℕ := {1, 2, 3}
def A (m : ℕ) : Set ℕ := {1, m}
def complement_U_A (m : ℕ) : Set ℕ := U \ A m

-- The main theorem where we need to prove m = 3 given the conditions.
theorem find_m (m : ℕ) (hU : U = {1, 2, 3})
  (hA : ∀ m, A m = {1, m})
  (h_complement : complement_U_A m = {2}) : m = 3 := sorry

end find_m_l67_67579


namespace mean_score_74_9_l67_67045

/-- 
In a class of 100 students, the score distribution is as follows:
- 10 students scored 100%
- 15 students scored 90%
- 20 students scored 80%
- 30 students scored 70%
- 20 students scored 60%
- 4 students scored 50%
- 1 student scored 40%

Prove that the mean percentage score of the class is 74.9.
-/
theorem mean_score_74_9 : 
  let scores := [100, 90, 80, 70, 60, 50, 40]
  let counts := [10, 15, 20, 30, 20, 4, 1]
  let total_students := 100
  let total_score := 1000 + 1350 + 1600 + 2100 + 1200 + 200 + 40
  (total_score / total_students : ℝ) = 74.9 :=
by {
  -- The detailed proof steps are omitted with sorry.
  sorry
}

end mean_score_74_9_l67_67045


namespace c_share_l67_67726

theorem c_share (A B C : ℝ) 
  (h1 : A = (1 / 2) * B)
  (h2 : B = (1 / 2) * C)
  (h3 : A + B + C = 392) : 
  C = 224 :=
by
  sorry

end c_share_l67_67726


namespace total_cost_of_mangoes_l67_67846

-- Definition of prices per dozen in one box
def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Number of dozens per box (constant for all boxes)
def dozens_per_box : ℕ := 10

-- Number of boxes
def number_of_boxes : ℕ := 36

-- Calculate the total cost of mangoes in all boxes
theorem total_cost_of_mangoes :
  (prices_per_dozen.sum * number_of_boxes = 3060) := by
  -- Proof goes here
  sorry

end total_cost_of_mangoes_l67_67846


namespace infinite_sorted_subsequence_l67_67479

theorem infinite_sorted_subsequence : 
  ∀ (warriors : ℕ → ℕ), (∀ n, ∃ m, m > n ∧ warriors m < warriors n) 
  ∨ (∃ k, warriors k = 0) → 
  ∃ (remaining : ℕ → ℕ), (∀ i j, i < j → remaining i > remaining j) :=
by
  intros warriors h
  sorry

end infinite_sorted_subsequence_l67_67479


namespace smallest_three_digit_candy_number_l67_67067

theorem smallest_three_digit_candy_number (n : ℕ) (hn1 : 100 ≤ n) (hn2 : n ≤ 999)
    (h1 : (n + 6) % 9 = 0) (h2 : (n - 9) % 6 = 0) : n = 111 := by
  sorry

end smallest_three_digit_candy_number_l67_67067


namespace quadratic_inequality_range_a_l67_67728

theorem quadratic_inequality_range_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end quadratic_inequality_range_a_l67_67728


namespace divisor_inequality_l67_67777

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end divisor_inequality_l67_67777


namespace range_of_2x_minus_y_l67_67077

theorem range_of_2x_minus_y (x y : ℝ) (hx : 0 < x ∧ x < 4) (hy : 0 < y ∧ y < 6) : -6 < 2 * x - y ∧ 2 * x - y < 8 := 
sorry

end range_of_2x_minus_y_l67_67077


namespace area_triangle_sum_l67_67668

theorem area_triangle_sum (AB : ℝ) (angle_BAC angle_ABC angle_ACB angle_EDC : ℝ) 
  (h_AB : AB = 1) (h_angle_BAC : angle_BAC = 70) (h_angle_ABC : angle_ABC = 50) 
  (h_angle_ACB : angle_ACB = 60) (h_angle_EDC : angle_EDC = 80) :
  let area_triangle := (1/2) * AB * (Real.sin angle_70 / Real.sin angle_60) * (Real.sin angle_60) 
  let area_CDE := (1/2) * (Real.sin angle_80)
  area_triangle + 2 * area_CDE = (Real.sin angle_70 + Real.sin angle_80) / 2 :=
sorry

end area_triangle_sum_l67_67668


namespace problem1_problem2_l67_67828

namespace MathProofs

theorem problem1 : (-3 - (-8) + (-6) + 10) = 9 :=
by
  sorry

theorem problem2 : (-12 * ((1 : ℚ) / 6 - (1 : ℚ) / 3 - 3 / 4)) = 11 :=
by
  sorry

end MathProofs

end problem1_problem2_l67_67828


namespace my_problem_l67_67753

theorem my_problem (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := 
sorry

end my_problem_l67_67753


namespace necessary_but_not_sufficient_l67_67613

-- Define the sets A, B, and C
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

-- The set A ∪ B in terms of Lean
def A_union_B : Set ℝ := A ∪ B

-- State the necessary and sufficient conditions
theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x ∈ A_union_B → x ∈ C) ∧ ¬ (∀ x : ℝ, x ∈ C → x ∈ A_union_B) :=
sorry

end necessary_but_not_sufficient_l67_67613


namespace find_natural_number_with_common_divisor_l67_67671

def commonDivisor (a b : ℕ) (d : ℕ) : Prop :=
  d > 1 ∧ d ∣ a ∧ d ∣ b

theorem find_natural_number_with_common_divisor :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k ≤ 20 →
    ∃ d : ℕ, commonDivisor (n + k) 30030 d) ∧ n = 9440 :=
by
  sorry

end find_natural_number_with_common_divisor_l67_67671


namespace gain_percent_l67_67747

def cycle_gain_percent (cp sp : ℕ) : ℚ :=
  (sp - cp) / cp * 100

theorem gain_percent {cp sp : ℕ} (h1 : cp = 1500) (h2 : sp = 1620) : cycle_gain_percent cp sp = 8 := by
  sorry

end gain_percent_l67_67747


namespace students_in_section_A_l67_67599

theorem students_in_section_A (x : ℕ) (h1 : (40 : ℝ) * x + 44 * 35 = 37.25 * (x + 44)) : x = 36 :=
by
  sorry

end students_in_section_A_l67_67599


namespace initial_distance_is_18_l67_67219

-- Step a) Conditions and Definitions
def distance_covered (v t d : ℝ) : Prop := 
  d = v * t

def increased_speed_time (v t d : ℝ) : Prop := 
  d = (v + 1) * (3 / 4 * t)

def decreased_speed_time (v t d : ℝ) : Prop := 
  d = (v - 1) * (t + 3)

-- Step c) Mathematically Equivalent Proof Problem
theorem initial_distance_is_18 (v t d : ℝ) 
  (h1 : distance_covered v t d) 
  (h2 : increased_speed_time v t d) 
  (h3 : decreased_speed_time v t d) : 
  d = 18 :=
sorry

end initial_distance_is_18_l67_67219


namespace tangent_line_at_point_l67_67860

theorem tangent_line_at_point :
  ∀ (x y : ℝ) (h : y = x^3 - 2 * x + 1),
    ∃ (m b : ℝ), (1, 0) = (x, y) → (m = 1) ∧ (b = -1) ∧ (∀ (z : ℝ), z = m * x + b) := sorry

end tangent_line_at_point_l67_67860


namespace smallest_multiple_l67_67075

theorem smallest_multiple (x : ℕ) (h1 : x % 24 = 0) (h2 : x % 36 = 0) (h3 : x % 20 ≠ 0) :
  x = 72 :=
by
  sorry

end smallest_multiple_l67_67075


namespace uncovered_area_is_8_l67_67017

-- Conditions
def shoebox_height : ℕ := 4
def shoebox_width : ℕ := 6
def block_side_length : ℕ := 4

-- Theorem to prove
theorem uncovered_area_is_8
  (sh_height : ℕ := shoebox_height)
  (sh_width : ℕ := shoebox_width)
  (bl_length : ℕ := block_side_length)
  : sh_height * sh_width - bl_length * bl_length = 8 :=
by {
  -- Placeholder for proof; we are not proving it as per instructions.
  sorry
}

end uncovered_area_is_8_l67_67017


namespace find_y_l67_67872

theorem find_y (x y : ℝ) (h1 : 2 * x - 3 * y = 24) (h2 : x + 2 * y = 15) : y = 6 / 7 :=
by sorry

end find_y_l67_67872


namespace range_of_k_l67_67682

theorem range_of_k {k : ℝ} : (∀ x : ℝ, x < 0 → (k - 2)/x > 0) ∧ (∀ x : ℝ, x > 0 → (k - 2)/x < 0) → k < 2 := 
by
  sorry

end range_of_k_l67_67682


namespace smallest_positive_sum_l67_67577

structure ArithmeticSequence :=
  (a_n : ℕ → ℤ)  -- The sequence is an integer sequence
  (d : ℤ)        -- The common difference of the sequence

def sum_of_first_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a_n 1 + seq.a_n n)) / 2  -- Sum of first n terms

def condition (seq : ArithmeticSequence) : Prop :=
  (seq.a_n 11 < -1 * seq.a_n 10)

theorem smallest_positive_sum (seq : ArithmeticSequence) (H : condition seq) :
  ∃ n, sum_of_first_n seq n > 0 ∧ ∀ m < n, sum_of_first_n seq m ≤ 0 → n = 19 :=
sorry

end smallest_positive_sum_l67_67577


namespace initial_number_306_l67_67417

theorem initial_number_306 (x : ℝ) : 
  (x / 34) * 15 + 270 = 405 → x = 306 :=
by
  intro h
  sorry

end initial_number_306_l67_67417


namespace volume_of_soup_in_hemisphere_half_height_l67_67139

theorem volume_of_soup_in_hemisphere_half_height 
  (V_hemisphere : ℝ)
  (hV_hemisphere : V_hemisphere = 8)
  (V_cap : ℝ) :
  V_cap = 2.5 :=
sorry

end volume_of_soup_in_hemisphere_half_height_l67_67139


namespace brick_laying_days_l67_67403

theorem brick_laying_days (a m n d : ℕ) (hm : 0 < m) (hn : 0 < n) (hd : 0 < d) :
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  (a * rate_M * (d * total_days) + 2 * a * rate_N * (d * total_days)) = (a + 2 * a) :=
by
  -- Definitions from the problem conditions
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  have h0 : a * rate_M * (d * total_days) = a := sorry
  have h1 : 2 * a * rate_N * (d * total_days) = 2 * a := sorry
  exact sorry

end brick_laying_days_l67_67403


namespace max_value_2019m_2020n_l67_67258

theorem max_value_2019m_2020n (m n : ℤ) (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) :
  (∀ (m' n' : ℤ), (0 ≤ m' - n') → (m' - n' ≤ 1) → (2 ≤ m' + n') → (m' + n' ≤ 4) → (m - 2 * n ≥ m' - 2 * n')) →
  2019 * m + 2020 * n = 2019 :=
by
  sorry

end max_value_2019m_2020n_l67_67258


namespace y_intercept_of_line_l67_67291

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l67_67291


namespace different_signs_abs_value_larger_l67_67443

variable {a b : ℝ}

theorem different_signs_abs_value_larger (h1 : a + b < 0) (h2 : ab < 0) : 
  (a > 0 ∧ b < 0 ∧ |a| < |b|) ∨ (a < 0 ∧ b > 0 ∧ |b| < |a|) :=
sorry

end different_signs_abs_value_larger_l67_67443


namespace geom_series_sum_l67_67700

def a : ℚ := 1 / 3
def r : ℚ := 2 / 3
def n : ℕ := 9

def S_n (a r : ℚ) (n : ℕ) := a * (1 - r^n) / (1 - r)

theorem geom_series_sum :
  S_n a r n = 19171 / 19683 := by
    sorry

end geom_series_sum_l67_67700


namespace solve_problem1_solve_problem2_l67_67510

-- Problem 1
theorem solve_problem1 (x : ℚ) : (3 * x - 1) ^ 2 = 9 ↔ x = 4 / 3 ∨ x = -2 / 3 := 
by sorry

-- Problem 2
theorem solve_problem2 (x : ℚ) : x * (2 * x - 4) = (2 - x) ^ 2 ↔ x = 2 ∨ x = -2 :=
by sorry

end solve_problem1_solve_problem2_l67_67510


namespace total_area_l67_67617

variable (A : ℝ)

-- Defining the conditions
def first_carpet : Prop := 0.55 * A = 36
def second_carpet : Prop := 0.25 * A = A * 0.25
def third_carpet : Prop := 0.15 * A = 18 + 6
def remaining_floor : Prop := 0.05 * A + 0.55 * A + 0.25 * A + 0.15 * A = A

-- Main theorem to prove the total area
theorem total_area : first_carpet A → second_carpet A → third_carpet A → remaining_floor A → A = 65.45 :=
by
  sorry

end total_area_l67_67617


namespace find_positive_integer_cube_root_divisible_by_21_l67_67259

theorem find_positive_integer_cube_root_divisible_by_21 (m : ℕ) (h1: m = 735) :
  m % 21 = 0 ∧ 9 < (m : ℝ)^(1/3) ∧ (m : ℝ)^(1/3) < 9.1 :=
by {
  sorry
}

end find_positive_integer_cube_root_divisible_by_21_l67_67259


namespace solve_equation_l67_67149

theorem solve_equation (x : ℚ) (h : x ≠ 1) : (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by 
  sorry

end solve_equation_l67_67149


namespace num_solutions_even_pairs_l67_67243

theorem num_solutions_even_pairs : ∃ n : ℕ, n = 25 ∧ ∀ (x y : ℕ),
  x % 2 = 0 ∧ y % 2 = 0 ∧ 4 * x + 6 * y = 600 → n = 25 :=
by
  sorry

end num_solutions_even_pairs_l67_67243


namespace profit_percentage_l67_67415

theorem profit_percentage (CP SP : ℝ) (h₁ : CP = 400) (h₂ : SP = 560) : 
  ((SP - CP) / CP) * 100 = 40 := by 
  sorry

end profit_percentage_l67_67415


namespace algebra_inequality_l67_67456

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end algebra_inequality_l67_67456


namespace mailing_ways_l67_67346

-- Definitions based on the problem conditions
def countWays (letters mailboxes : ℕ) : ℕ := mailboxes^letters

-- The theorem to prove the mathematically equivalent proof problem
theorem mailing_ways (letters mailboxes : ℕ) (h_letters : letters = 3) (h_mailboxes : mailboxes = 4) : countWays letters mailboxes = 4^3 := 
by
  rw [h_letters, h_mailboxes]
  rfl

end mailing_ways_l67_67346


namespace negation_of_forall_x_squared_nonnegative_l67_67924

theorem negation_of_forall_x_squared_nonnegative :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_forall_x_squared_nonnegative_l67_67924


namespace comparison_abc_l67_67118

noncomputable def a : ℝ := (Real.exp 1 + 2) / Real.log (Real.exp 1 + 2)
noncomputable def b : ℝ := 2 / Real.log 2
noncomputable def c : ℝ := (Real.exp 1)^2 / (4 - Real.log 4)

theorem comparison_abc : c < b ∧ b < a :=
by {
  sorry
}

end comparison_abc_l67_67118


namespace parabola_constant_c_l67_67925

theorem parabola_constant_c (b c : ℝ): 
  (∀ x : ℝ, y = x^2 + b * x + c) ∧ 
  (10 = 2^2 + b * 2 + c) ∧ 
  (31 = 4^2 + b * 4 + c) → 
  c = -3 :=
by
  sorry

end parabola_constant_c_l67_67925


namespace solve_for_x_l67_67514

theorem solve_for_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 :=
by
  sorry

end solve_for_x_l67_67514


namespace double_counted_toddlers_l67_67176

def number_of_toddlers := 21
def missed_toddlers := 3
def billed_count := 26

theorem double_counted_toddlers : 
  ∃ (D : ℕ), (number_of_toddlers + D - missed_toddlers = billed_count) ∧ D = 8 :=
by
  sorry

end double_counted_toddlers_l67_67176


namespace sqrt_square_eq_17_l67_67611

theorem sqrt_square_eq_17 :
  (Real.sqrt 17) ^ 2 = 17 :=
sorry

end sqrt_square_eq_17_l67_67611


namespace concert_tickets_l67_67223

theorem concert_tickets : ∃ (A B : ℕ), 8 * A + 425 * B = 3000000 ∧ A + B = 4500 ∧ A = 2900 := by
  sorry

end concert_tickets_l67_67223


namespace ELMO_value_l67_67657

def digits := {n : ℕ // n < 10}

variables (L E T M O : digits)

-- Conditions
axiom h1 : L.val ≠ 0
axiom h2 : O.val = 0
axiom h3 : (1000 * L.val + 100 * E.val + 10 * E.val + T.val) + (100 * L.val + 10 * M.val + T.val) = 1000 * T.val + L.val

-- Conclusion
theorem ELMO_value : E.val * 1000 + L.val * 100 + M.val * 10 + O.val = 1880 :=
sorry

end ELMO_value_l67_67657


namespace g_of_12_l67_67138

def g (n : ℕ) : ℕ := n^2 - n + 23

theorem g_of_12 : g 12 = 155 :=
by
  sorry

end g_of_12_l67_67138


namespace remainder_not_power_of_4_l67_67959

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end remainder_not_power_of_4_l67_67959


namespace sam_age_l67_67146

-- Definitions
variables (B J S : ℕ)
axiom H1 : B = 2 * J
axiom H2 : B + J = 60
axiom H3 : S = (B + J) / 2

-- Problem statement
theorem sam_age : S = 30 :=
sorry

end sam_age_l67_67146


namespace min_phi_l67_67237

theorem min_phi
  (ϕ : ℝ) (hϕ : ϕ > 0)
  (h_symm : ∃ k : ℤ, 2 * (π / 6) - 2 * ϕ = k * π + π / 2) :
  ϕ = 5 * π / 12 :=
sorry

end min_phi_l67_67237


namespace three_consecutive_odd_numbers_l67_67300

theorem three_consecutive_odd_numbers (x : ℤ) (h : x - 2 + x + x + 2 = 27) : 
  (x + 2, x, x - 2) = (11, 9, 7) :=
by
  sorry

end three_consecutive_odd_numbers_l67_67300


namespace find_a_minus_b_l67_67322

theorem find_a_minus_b
  (a b : ℝ)
  (f g h h_inv : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = -4 * x + 3)
  (hh : ∀ x, h x = f (g x))
  (hinv : ∀ x, h_inv x = 2 * x + 6)
  (h_comp : ∀ x, h x = (x - 6) / 2) :
  a - b = 5 / 2 :=
sorry

end find_a_minus_b_l67_67322


namespace problem1_problem2_l67_67557

variable (t : ℝ)

-- Problem 1
theorem problem1 (h : (4:ℝ) - 8 * t + 16 < 0) : t > 5 / 2 :=
sorry

-- Problem 2
theorem problem2 (hp: 4 - t > t - 2) (hq : t - 2 > 0) (hdisjoint : (∃ (p : Prop) (q : Prop), (p ∨ q) ∧ ¬(p ∧ q))):
  (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) :=
sorry


end problem1_problem2_l67_67557


namespace rotate_parabola_180deg_l67_67094

theorem rotate_parabola_180deg (x y : ℝ) :
  (∀ x, y = 2 * x^2 - 12 * x + 16) →
  (∀ x, y = -2 * x^2 + 12 * x - 20) :=
sorry

end rotate_parabola_180deg_l67_67094


namespace final_price_correct_l67_67036

noncomputable def original_price : ℝ := 49.99
noncomputable def first_discount : ℝ := 0.10
noncomputable def second_discount : ℝ := 0.20

theorem final_price_correct :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount = 36.00 := by
    -- The proof would go here
    sorry

end final_price_correct_l67_67036


namespace solution_set_of_inequalities_l67_67295

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l67_67295


namespace solve_a_range_m_l67_67201

def f (x : ℝ) (a : ℝ) : ℝ := |x - a|

theorem solve_a :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ↔ (2 = 2) :=
by {
  sorry
}

theorem range_m :
  (∀ x : ℝ, f (3 * x) 2 + f (x + 3) 2 ≥ m) ↔ (m ≤ 5 / 3) :=
by {
  sorry
}

end solve_a_range_m_l67_67201


namespace rachel_total_apples_l67_67010

noncomputable def totalRemainingApples (X : ℕ) : ℕ :=
  let remainingFirstFour := 10 + 40 + 15 + 22
  let remainingOtherTrees := 48 * X
  remainingFirstFour + remainingOtherTrees

theorem rachel_total_apples (X : ℕ) :
  totalRemainingApples X = 87 + 48 * X :=
by
  sorry

end rachel_total_apples_l67_67010


namespace range_of_m_l67_67080

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, (m + 1) * x^2 ≥ 0) : m > -1 :=
by
  sorry

end range_of_m_l67_67080


namespace no_integer_solution_for_conditions_l67_67059

theorem no_integer_solution_for_conditions :
  ¬∃ (x : ℤ), 
    (18 + x = 2 * (5 + x)) ∧
    (18 + x = 3 * (2 + x)) ∧
    ((18 + x) + (5 + x) + (2 + x) = 50) :=
by
  sorry

end no_integer_solution_for_conditions_l67_67059


namespace quadratic_k_value_l67_67755

theorem quadratic_k_value (a b c k : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : 4 * b * b - k * a * c = 0): 
  k = 16 / 3 :=
by
  sorry

end quadratic_k_value_l67_67755


namespace johns_weekly_earnings_after_raise_l67_67939

theorem johns_weekly_earnings_after_raise 
  (original_weekly_earnings : ℕ) 
  (percentage_increase : ℝ) 
  (new_weekly_earnings : ℕ)
  (h1 : original_weekly_earnings = 60)
  (h2 : percentage_increase = 0.16666666666666664) :
  new_weekly_earnings = 70 :=
sorry

end johns_weekly_earnings_after_raise_l67_67939


namespace problem1_problem2_l67_67157

-- Problem 1:
theorem problem1 (P : ℝ × ℝ) (hP : P = (-1, 0)) :
  (∃ m b, (∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0)) ∧ ∀ x y, (x = -1 → y = 0 → y = m * x + b)) → 
  ∃ m b, ∀ x y, y = m * x + b ↔ (3 * x - y + 3 = 0) :=
sorry

-- Problem 2:
theorem problem2 (L1 : ℝ → ℝ → Prop) (hL1 : ∀ x y, L1 x y ↔ 3 * x + 4 * y - 12 = 0) (d : ℝ) (hd : d = 7) :
  (∃ c, ∀ x y, (3 * x + 4 * y + c = 0 ∨ 3 * x + 4 * y - 47 = 0) ↔ L1 x y ∧ d = 7) :=
sorry

end problem1_problem2_l67_67157


namespace expected_defective_chips_in_60000_l67_67517

def shipmentS1 := (2, 5000)
def shipmentS2 := (4, 12000)
def shipmentS3 := (2, 15000)
def shipmentS4 := (4, 16000)

def total_defective_chips := shipmentS1.1 + shipmentS2.1 + shipmentS3.1 + shipmentS4.1
def total_chips := shipmentS1.2 + shipmentS2.2 + shipmentS3.2 + shipmentS4.2

def defective_ratio := total_defective_chips / total_chips
def shipment60000 := 60000

def expected_defectives (ratio : ℝ) (total_chips : ℝ) := ratio * total_chips

theorem expected_defective_chips_in_60000 :
  expected_defectives defective_ratio shipment60000 = 15 :=
by
  sorry

end expected_defective_chips_in_60000_l67_67517


namespace lana_total_spending_l67_67913

theorem lana_total_spending (ticket_price : ℕ) (tickets_friends : ℕ) (tickets_extra : ℕ)
  (H1 : ticket_price = 6)
  (H2 : tickets_friends = 8)
  (H3 : tickets_extra = 2) :
  ticket_price * (tickets_friends + tickets_extra) = 60 :=
by
  sorry

end lana_total_spending_l67_67913


namespace find_unknown_polynomial_l67_67152

theorem find_unknown_polynomial (m : ℤ) : 
  ∃ q : ℤ, (q + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1) → q = 2 * m^2 + 3 * m - 4 :=
by {
  sorry
}

end find_unknown_polynomial_l67_67152


namespace minimum_sum_of_x_and_y_l67_67718

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 4 * y = x * y

theorem minimum_sum_of_x_and_y (x y : ℝ) (h : conditions x y) : x + y ≥ 9 := by
  sorry

end minimum_sum_of_x_and_y_l67_67718


namespace product_of_last_two_digits_l67_67767

theorem product_of_last_two_digits (A B : ℕ) (h₁ : A + B = 17) (h₂ : 4 ∣ (10 * A + B)) :
  A * B = 72 := sorry

end product_of_last_two_digits_l67_67767


namespace range_of_a_l67_67458

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l67_67458


namespace hexagon_perimeter_l67_67233

def side_length : ℕ := 10
def num_sides : ℕ := 6

theorem hexagon_perimeter : num_sides * side_length = 60 := by
  sorry

end hexagon_perimeter_l67_67233


namespace find_number_l67_67079

theorem find_number (x : ℝ) (h : 0.50 * x = 48 + 180) : x = 456 :=
sorry

end find_number_l67_67079


namespace sum_of_interior_angles_of_polygon_l67_67987

theorem sum_of_interior_angles_of_polygon (n : ℕ) (h : n - 3 = 3) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l67_67987


namespace quadratic_inequality_solution_set_l67_67184

theorem quadratic_inequality_solution_set (x : ℝ) : (x + 3) * (2 - x) < 0 ↔ x < -3 ∨ x > 2 := 
sorry

end quadratic_inequality_solution_set_l67_67184


namespace minimum_x_y_sum_l67_67917

theorem minimum_x_y_sum (x y : ℕ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 1 / 15) : x + y = 64 :=
  sorry

end minimum_x_y_sum_l67_67917


namespace binary_modulo_eight_l67_67608

theorem binary_modulo_eight : (0b1110101101101 : ℕ) % 8 = 5 := 
by {
  -- This is where the proof would go.
  sorry
}

end binary_modulo_eight_l67_67608


namespace probability_correct_l67_67251

-- Define the set and the probability calculation
def set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Function to check if the difference condition holds
def valid_triplet (a b c: ℕ) : Prop := a < b ∧ b < c ∧ c - a = 4

-- Total number of ways to pick 3 numbers and ways that fit the condition
noncomputable def total_ways : ℕ := Nat.choose 9 3
noncomputable def valid_ways : ℕ := 5 * 2

-- Calculate the probability
noncomputable def probability : ℚ := valid_ways / total_ways

-- The theorem statement
theorem probability_correct : probability = 5 / 42 := by sorry

end probability_correct_l67_67251


namespace sum_max_min_expr_l67_67175

theorem sum_max_min_expr (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) : 
    let expr := (x / |x|) + (|y| / y) - (|x * y| / (x * y))
    max (max expr (expr)) (min expr expr) = -2 :=
sorry

end sum_max_min_expr_l67_67175


namespace ratio_sum_pqr_uvw_l67_67546

theorem ratio_sum_pqr_uvw (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end ratio_sum_pqr_uvw_l67_67546


namespace gmat_test_statistics_l67_67083

theorem gmat_test_statistics 
    (p1 : ℝ) (p2 : ℝ) (p12 : ℝ) (neither : ℝ) (S : ℝ) 
    (h1 : p1 = 0.85)
    (h2 : p12 = 0.60) 
    (h3 : neither = 0.05) :
    0.25 + S = 0.95 → S = 0.70 :=
by
  sorry

end gmat_test_statistics_l67_67083


namespace inequality_solution_l67_67007

theorem inequality_solution (x : ℝ) : |x - 3| + |x - 5| ≥ 4 → x ≥ 6 ∨ x ≤ 2 :=
by
  sorry

end inequality_solution_l67_67007


namespace x_squared_plus_y_squared_l67_67347

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^3 = 8) (h2 : x * y = 5) : 
  x^2 + y^2 = -6 := by
  sorry

end x_squared_plus_y_squared_l67_67347


namespace dot_product_of_a_and_c_is_4_l67_67506

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b : vector := (-3, 2)

def three_a : vector := (3 * 1, 3 * -2)
def two_b_minus_a : vector := (2 * -3 - 1, 2 * 2 - -2)

def c : vector := (-(-three_a.fst + two_b_minus_a.fst), -(-three_a.snd + two_b_minus_a.snd))

def dot_product (u v : vector) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem dot_product_of_a_and_c_is_4 : dot_product a c = 4 := 
by
  sorry

end dot_product_of_a_and_c_is_4_l67_67506


namespace roger_owes_correct_amount_l67_67054

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end roger_owes_correct_amount_l67_67054


namespace largest_subset_size_with_property_l67_67512

def no_four_times_property (S : Finset ℕ) : Prop := 
  ∀ {x y}, x ∈ S → y ∈ S → x = 4 * y → False

noncomputable def max_subset_size : ℕ := 145

theorem largest_subset_size_with_property :
  ∃ (S : Finset ℕ), (∀ x ∈ S, x ≤ 150) ∧ no_four_times_property S ∧ S.card = max_subset_size :=
sorry

end largest_subset_size_with_property_l67_67512


namespace totalPieces_l67_67725

   -- Definitions given by the conditions
   def packagesGum := 21
   def packagesCandy := 45
   def packagesMints := 30
   def piecesPerGumPackage := 9
   def piecesPerCandyPackage := 12
   def piecesPerMintPackage := 8

   -- Define the total pieces of gum, candy, and mints
   def totalPiecesGum := packagesGum * piecesPerGumPackage
   def totalPiecesCandy := packagesCandy * piecesPerCandyPackage
   def totalPiecesMints := packagesMints * piecesPerMintPackage

   -- The mathematical statement to prove
   theorem totalPieces :
     totalPiecesGum + totalPiecesCandy + totalPiecesMints = 969 :=
   by
     -- Proof is skipped
     sorry
   
end totalPieces_l67_67725


namespace closest_point_in_plane_l67_67727

noncomputable def closest_point (x y z : ℚ) : Prop :=
  ∃ (t : ℚ), 
    x = 2 + 2 * t ∧ 
    y = 3 - 3 * t ∧ 
    z = 1 + 4 * t ∧ 
    2 * (2 + 2 * t) - 3 * (3 - 3 * t) + 4 * (1 + 4 * t) = 40

theorem closest_point_in_plane :
  closest_point (92 / 29) (16 / 29) (145 / 29) :=
by
  sorry

end closest_point_in_plane_l67_67727


namespace mary_donated_books_l67_67270

theorem mary_donated_books 
  (s : ℕ) (b_c : ℕ) (b_b : ℕ) (b_y : ℕ) (g_d : ℕ) (g_m : ℕ) (e : ℕ) (s_s : ℕ) 
  (total : ℕ) (out_books : ℕ) (d : ℕ)
  (h1 : s = 72)
  (h2 : b_c = 12)
  (h3 : b_b = 5)
  (h4 : b_y = 2)
  (h5 : g_d = 1)
  (h6 : g_m = 4)
  (h7 : e = 81)
  (h8 : s_s = 3)
  (ht : total = s + b_c + b_b + b_y + g_d + g_m)
  (ho : out_books = total - e)
  (hd : d = out_books - s_s) :
  d = 12 :=
by { sorry }

end mary_donated_books_l67_67270


namespace find_d_l67_67325

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ k : Nat, k > 1 → k < n → n % k ≠ 0)

def less_than_10_primes (n : Nat) : Prop :=
  n < 10 ∧ is_prime n

theorem find_d (d e f : Nat) (hd : less_than_10_primes d) (he : less_than_10_primes e) (hf : less_than_10_primes f) :
  d + e = f → d < e → d = 2 :=
by
  sorry

end find_d_l67_67325


namespace correct_calculation_l67_67423

theorem correct_calculation (a b : ℝ) : (3 * a * b) ^ 2 = 9 * a ^ 2 * b ^ 2 :=
by
  sorry

end correct_calculation_l67_67423


namespace gcd_891_810_l67_67766

theorem gcd_891_810 : Nat.gcd 891 810 = 81 := 
by
  sorry

end gcd_891_810_l67_67766


namespace cube_partition_exists_l67_67241

theorem cube_partition_exists : ∃ (n_0 : ℕ), (0 < n_0) ∧ (∀ (n : ℕ), n ≥ n_0 → ∃ k : ℕ, n = k) := sorry

end cube_partition_exists_l67_67241


namespace weight_of_second_triangle_l67_67771

theorem weight_of_second_triangle :
  let side_len1 := 4
  let density1 := 0.9
  let weight1 := 10.8
  let side_len2 := 6
  let density2 := 1.2
  let weight2 := 18.7
  let area1 := (side_len1 ^ 2 * Real.sqrt 3) / 4
  let area2 := (side_len2 ^ 2 * Real.sqrt 3) / 4
  let calc_weight1 := area1 * density1
  let calc_weight2 := area2 * density2
  calc_weight1 = weight1 → calc_weight2 = weight2 := 
by
  intros
  -- Proof logic goes here
  sorry

end weight_of_second_triangle_l67_67771


namespace fraction_of_single_men_l67_67012

theorem fraction_of_single_men :
  ∀ (total_faculty : ℕ) (women_percentage : ℝ) (married_percentage : ℝ) (married_men_ratio : ℝ),
    women_percentage = 0.7 → married_percentage = 0.4 → married_men_ratio = 2/3 →
    (total_faculty * (1 - women_percentage)) * (1 - married_men_ratio) / 
    (total_faculty * (1 - women_percentage)) = 1/3 :=
by 
  intros total_faculty women_percentage married_percentage married_men_ratio h_women h_married h_men_marry
  sorry

end fraction_of_single_men_l67_67012


namespace wire_cut_square_octagon_area_l67_67707

theorem wire_cut_square_octagon_area (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (equal_area : (a / 4)^2 = (2 * (b / 8)^2 * (1 + Real.sqrt 2))) : 
  a / b = Real.sqrt ((1 + Real.sqrt 2) / 2) := 
  sorry

end wire_cut_square_octagon_area_l67_67707


namespace general_inequality_l67_67108

theorem general_inequality (x : ℝ) (n : ℕ) (h_pos_x : x > 0) (h_pos_n : 0 < n) : 
  x + n^n / x^n ≥ n + 1 := by 
  sorry

end general_inequality_l67_67108


namespace penny_identified_whales_l67_67870

theorem penny_identified_whales (sharks eels total : ℕ)
  (h_sharks : sharks = 35)
  (h_eels   : eels = 15)
  (h_total  : total = 55) :
  total - (sharks + eels) = 5 :=
by
  sorry

end penny_identified_whales_l67_67870


namespace greatest_b_value_l67_67318

def equation_has_integer_solutions (b : ℕ) : Prop :=
  ∃ (x : ℤ), x * (x + b) = -20

theorem greatest_b_value : ∃ (b : ℕ), b = 21 ∧ equation_has_integer_solutions b :=
by
  sorry

end greatest_b_value_l67_67318


namespace many_people_sharing_car_l67_67804

theorem many_people_sharing_car (x y : ℤ) 
  (h1 : 3 * (y - 2) = x) 
  (h2 : 2 * y + 9 = x) : 
  3 * (y - 2) = 2 * y + 9 := 
by
  -- by assumption h1 and h2, we already have the setup, refute/validate consistency
  sorry

end many_people_sharing_car_l67_67804


namespace pentagon_coloring_valid_l67_67934

-- Define the colors
inductive Color
| Red
| Blue

-- Define the vertices as a type
inductive Vertex
| A | B | C | D | E

open Vertex Color

-- Define an edge as a pair of vertices
def Edge := Vertex × Vertex

-- Define the coloring function
def color : Edge → Color := sorry

-- Define the pentagon
def pentagon_edges : List Edge :=
  [(A, B), (B, C), (C, D), (D, E), (E, A), (A, C), (A, D), (A, E), (B, D), (B, E), (C, E)]

-- Define the condition for a valid triangle coloring
def valid_triangle_coloring (e1 e2 e3 : Edge) : Prop :=
  (color e1 = Red ∧ (color e2 = Blue ∨ color e3 = Blue)) ∨
  (color e2 = Red ∧ (color e1 = Blue ∨ color e3 = Blue)) ∨
  (color e3 = Red ∧ (color e1 = Blue ∨ color e2 = Blue))

-- Define the condition for all triangles formed by the vertices of the pentagon
def all_triangles_valid : Prop :=
  ∀ v1 v2 v3 : Vertex,
    v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 →
    valid_triangle_coloring (v1, v2) (v2, v3) (v1, v3)

-- Statement: Prove that there are 12 valid ways to color the pentagon
theorem pentagon_coloring_valid : (∃ (coloring : Edge → Color), all_triangles_valid) :=
  sorry

end pentagon_coloring_valid_l67_67934


namespace solve_equation_l67_67656

variable (a b c : ℝ)

theorem solve_equation (h : (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1) : 
  a * c = 36 * b :=
by 
  -- Proof goes here
  sorry

end solve_equation_l67_67656


namespace minimize_sum_of_squares_l67_67051

theorem minimize_sum_of_squares (x1 x2 x3 : ℝ) (hpos1 : 0 < x1) (hpos2 : 0 < x2) (hpos3 : 0 < x3)
  (h_eq : x1 + 3 * x2 + 5 * x3 = 100) : x1^2 + x2^2 + x3^2 = 2000 / 7 := 
sorry

end minimize_sum_of_squares_l67_67051


namespace total_students_in_classrooms_l67_67008

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end total_students_in_classrooms_l67_67008


namespace ordered_pair_solution_l67_67474

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end ordered_pair_solution_l67_67474


namespace isosceles_triangle_base_angles_l67_67165

theorem isosceles_triangle_base_angles 
  (α β : ℝ) -- α and β are the base angles
  (h : α = β)
  (height leg : ℝ)
  (h_height_leg : height = leg / 2) : 
  α = 75 ∨ α = 15 :=
by
  sorry

end isosceles_triangle_base_angles_l67_67165


namespace problem_proof_l67_67427

-- Define the geometric sequence and vectors conditions
variables (a : ℕ → ℝ) (q : ℝ)
variables (h1 : ∀ n, a (n + 1) = q * a n)
variables (h2 : a 2 = a 2)
variables (h3 : a 3 = q * a 2)
variables (h4 : 3 * a 2 = 2 * a 3)

-- Statement to prove
theorem problem_proof:
  (a 2 + a 4) / (a 3 + a 5) = 2 / 3 :=
  sorry

end problem_proof_l67_67427


namespace total_amount_spent_l67_67224

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = (1/2) * B
def condition2 : Prop := B = D + 15

-- Proof statement
theorem total_amount_spent (h1 : condition1 B D) (h2 : condition2 B D) : B + D = 45 := by
  sorry

end total_amount_spent_l67_67224


namespace remainder_of_sum_of_5_consecutive_numbers_mod_9_l67_67342

theorem remainder_of_sum_of_5_consecutive_numbers_mod_9 :
  (9154 + 9155 + 9156 + 9157 + 9158) % 9 = 1 :=
by
  sorry

end remainder_of_sum_of_5_consecutive_numbers_mod_9_l67_67342


namespace opposite_numbers_abs_eq_l67_67674

theorem opposite_numbers_abs_eq (a : ℚ) : abs a = abs (-a) :=
by
  sorry

end opposite_numbers_abs_eq_l67_67674


namespace num_coloring_l67_67056

-- Define the set of numbers to be colored
def numbers_to_color : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the set of colors
inductive Color
| red
| green
| blue

-- Define proper divisors for the numbers in the list
def proper_divisors (n : ℕ) : List ℕ :=
  match n with
  | 2 => []
  | 3 => []
  | 4 => [2]
  | 5 => []
  | 6 => [2, 3]
  | 7 => []
  | 8 => [2, 4]
  | 9 => [3]
  | _ => []

-- The proof statement
theorem num_coloring (h : ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, n ≠ d) :
  ∃ f : ℕ → Color, ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, f n ≠ f d :=
  sorry

end num_coloring_l67_67056


namespace solve_abs_inequality_l67_67582

theorem solve_abs_inequality (x : ℝ) : (|x + 3| + |x - 4| < 8) ↔ (4 ≤ x ∧ x < 4.5) := sorry

end solve_abs_inequality_l67_67582


namespace sum_gt_two_l67_67481

noncomputable def f (x : ℝ) : ℝ := ((x - 1) * Real.log x) / x

theorem sum_gt_two (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ > 2 := 
sorry

end sum_gt_two_l67_67481


namespace chemistry_more_than_physics_l67_67523

theorem chemistry_more_than_physics
  (M P C : ℕ)
  (h1 : M + P = 60)
  (h2 : (M + C) / 2 = 35) :
  ∃ x : ℕ, C = P + x ∧ x = 10 := 
by
  sorry

end chemistry_more_than_physics_l67_67523


namespace average_of_measurements_l67_67637

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80.0, 79.6, 80.5]

theorem average_of_measurements : (measurements.sum / measurements.length) = 80 := by sorry

end average_of_measurements_l67_67637


namespace apples_count_l67_67793

def mangoes_oranges_apples_ratio (mangoes oranges apples : Nat) : Prop :=
  mangoes / 10 = oranges / 2 ∧ mangoes / 10 = apples / 3

theorem apples_count (mangoes oranges apples : Nat) (h_ratio : mangoes_oranges_apples_ratio mangoes oranges apples) (h_mangoes : mangoes = 120) : apples = 36 :=
by
  sorry

end apples_count_l67_67793


namespace find_fraction_l67_67641

-- Define the initial amount, the amount spent on pads, and the remaining amount
def initial_amount := 150
def spent_on_pads := 50
def remaining := 25

-- Define the fraction she spent on hockey skates
def fraction_spent_on_skates (f : ℚ) : Prop :=
  let spent_on_skates := initial_amount - remaining - spent_on_pads
  (spent_on_skates / initial_amount) = f

theorem find_fraction : fraction_spent_on_skates (1 / 2) :=
by
  -- Proof steps go here
  sorry

end find_fraction_l67_67641


namespace math_problem_l67_67028

theorem math_problem :
  (-1)^2024 + (-10) / (1/2) * 2 + (2 - (-3)^3) = -10 := by
  sorry

end math_problem_l67_67028


namespace max_value_of_expression_l67_67821

theorem max_value_of_expression (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  2 * Real.sqrt (abc / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by
  sorry

end max_value_of_expression_l67_67821


namespace ratio_eq_23_over_28_l67_67944

theorem ratio_eq_23_over_28 (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) : 
  a / b = 23 / 28 := 
sorry

end ratio_eq_23_over_28_l67_67944


namespace product_of_square_and_neighbor_is_divisible_by_12_l67_67960

theorem product_of_square_and_neighbor_is_divisible_by_12 (n : ℤ) : 12 ∣ (n^2 * (n - 1) * (n + 1)) :=
sorry

end product_of_square_and_neighbor_is_divisible_by_12_l67_67960


namespace apple_multiple_l67_67503

theorem apple_multiple (K Ka : ℕ) (M : ℕ) 
  (h1 : K + Ka = 340)
  (h2 : Ka = M * K + 10)
  (h3 : Ka = 274) : 
  M = 4 := 
by
  sorry

end apple_multiple_l67_67503


namespace a_2_pow_100_value_l67_67328

theorem a_2_pow_100_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (2 * n) = 3 * n * a n) :
  a (2^100) = 2^4852 * 3^4950 :=
by
  sorry

end a_2_pow_100_value_l67_67328


namespace initial_student_count_l67_67332

theorem initial_student_count
  (n : ℕ)
  (T : ℝ)
  (h1 : T = 60.5 * (n : ℝ))
  (h2 : T - 8 = 64 * ((n - 1) : ℝ))
  : n = 16 :=
sorry

end initial_student_count_l67_67332


namespace derivative_of_log_base2_inv_x_l67_67014

noncomputable def my_function (x : ℝ) : ℝ := (Real.log x⁻¹) / (Real.log 2)

theorem derivative_of_log_base2_inv_x : 
  ∀ x : ℝ, x > 0 → deriv my_function x = -1 / (x * Real.log 2) :=
by
  intros x hx
  sorry

end derivative_of_log_base2_inv_x_l67_67014


namespace proof_problem_l67_67539

-- Define the conditions based on Classmate A and Classmate B's statements
def classmateA_statement (x y : ℝ) : Prop := 6 * x = 5 * y
def classmateB_statement (x y : ℝ) : Prop := x = 2 * y - 40

-- Define the system of equations derived from the statements
def system_of_equations (x y : ℝ) : Prop := (6 * x = 5 * y) ∧ (x = 2 * y - 40)

-- Proof goal: Prove the system of equations if classmate statements hold
theorem proof_problem (x y : ℝ) :
  classmateA_statement x y ∧ classmateB_statement x y → system_of_equations x y :=
by
  sorry

end proof_problem_l67_67539


namespace longest_side_is_48_l67_67168

noncomputable def longest_side_of_triangle (a b c : ℝ) (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : ℝ :=
  a

theorem longest_side_is_48 {a b c : ℝ} (ha : a/b = 1/2) (hb : b/c = 1/3) (hc : c/a = 1/4) (hp : a + b + c = 104) : 
  longest_side_of_triangle a b c ha hb hc hp = 48 :=
sorry

end longest_side_is_48_l67_67168


namespace prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l67_67379

theorem prop_P_subset_q_when_m_eq_1 :
  ∀ x : ℝ, ∀ m : ℝ, m = 1 → (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) ↔ (x ∈ {x | 0 ≤ x ∧ x ≤ 2}) := 
by sorry

theorem range_m_for_necessity_and_not_sufficiency :
  ∀ m : ℝ, (∀ x : ℝ, (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) → (x ∈ {x | 1 - m ≤ x ∧ x ≤ 1 + m})) ↔ (m ≥ 9) := 
by sorry

end prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l67_67379


namespace ratio_Bill_to_Bob_l67_67800

-- Define the shares
def Bill_share : ℕ := 300
def Bob_share : ℕ := 900

-- The theorem statement
theorem ratio_Bill_to_Bob : Bill_share / Bob_share = 1 / 3 := by
  sorry

end ratio_Bill_to_Bob_l67_67800


namespace simplify_expression_l67_67106

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end simplify_expression_l67_67106


namespace votes_candidate_X_l67_67545

theorem votes_candidate_X (X Y Z : ℕ) (h1 : X = (3 / 2 : ℚ) * Y) (h2 : Y = (3 / 5 : ℚ) * Z) (h3 : Z = 25000) : X = 22500 :=
by
  sorry

end votes_candidate_X_l67_67545


namespace value_of_expression_l67_67650

theorem value_of_expression (x y : ℝ) (h₀ : x = Real.sqrt 2 + 1) (h₁ : y = Real.sqrt 2 - 1) : 
  (x + y) * (x - y) = 4 * Real.sqrt 2 :=
by
  sorry

end value_of_expression_l67_67650


namespace percentage_differences_equal_l67_67402

noncomputable def calculation1 : ℝ := 0.60 * 50
noncomputable def calculation2 : ℝ := 0.30 * 30
noncomputable def calculation3 : ℝ := 0.45 * 90
noncomputable def calculation4 : ℝ := 0.20 * 40

noncomputable def diff1 : ℝ := abs (calculation1 - calculation2)
noncomputable def diff2 : ℝ := abs (calculation2 - calculation3)
noncomputable def diff3 : ℝ := abs (calculation3 - calculation4)
noncomputable def largest_diff1 : ℝ := max diff1 (max diff2 diff3)

noncomputable def calculation5 : ℝ := 0.40 * 120
noncomputable def calculation6 : ℝ := 0.25 * 80
noncomputable def calculation7 : ℝ := 0.35 * 150
noncomputable def calculation8 : ℝ := 0.55 * 60

noncomputable def diff4 : ℝ := abs (calculation5 - calculation6)
noncomputable def diff5 : ℝ := abs (calculation6 - calculation7)
noncomputable def diff6 : ℝ := abs (calculation7 - calculation8)
noncomputable def largest_diff2 : ℝ := max diff4 (max diff5 diff6)

theorem percentage_differences_equal :
  largest_diff1 = largest_diff2 :=
sorry

end percentage_differences_equal_l67_67402


namespace negation_of_proposition_l67_67507

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℝ, 2^x_0 < x_0^2) ↔ (∀ x : ℝ, 2^x ≥ x^2) :=
by sorry

end negation_of_proposition_l67_67507


namespace fill_tank_time_l67_67639

-- Define the rates of filling and draining
def rateA : ℕ := 200 -- Pipe A fills at 200 liters per minute
def rateB : ℕ := 50  -- Pipe B fills at 50 liters per minute
def rateC : ℕ := 25  -- Pipe C drains at 25 liters per minute

-- Define the times each pipe is open
def timeA : ℕ := 1   -- Pipe A is open for 1 minute
def timeB : ℕ := 2   -- Pipe B is open for 2 minutes
def timeC : ℕ := 2   -- Pipe C is open for 2 minutes

-- Define the capacity of the tank
def tankCapacity : ℕ := 1000

-- Prove the total time to fill the tank is 20 minutes
theorem fill_tank_time : 
  (tankCapacity * ((timeA * rateA + timeB * rateB) - (timeC * rateC)) * 5) = 20 :=
sorry

end fill_tank_time_l67_67639


namespace two_times_sum_of_squares_l67_67135

theorem two_times_sum_of_squares (P a b : ℤ) (h : P = a^2 + b^2) : 
  ∃ x y : ℤ, 2 * P = x^2 + y^2 := 
by 
  sorry

end two_times_sum_of_squares_l67_67135


namespace jake_watched_friday_l67_67981

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jake_watched_friday_l67_67981


namespace sale_in_fifth_month_l67_67994

def sale_first_month : ℝ := 3435
def sale_second_month : ℝ := 3927
def sale_third_month : ℝ := 3855
def sale_fourth_month : ℝ := 4230
def required_avg_sale : ℝ := 3500
def sale_sixth_month : ℝ := 1991

theorem sale_in_fifth_month :
  (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + s + sale_sixth_month) / 6 = required_avg_sale ->
  s = 3562 :=
by
  sorry

end sale_in_fifth_month_l67_67994


namespace arithmetic_sequence_common_difference_l67_67633

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_variance : (1/5) * ((a 1 - (a 3)) ^ 2 + (a 2 - (a 3)) ^ 2 + (a 3 - (a 3)) ^ 2 + (a 4 - (a 3)) ^ 2 + (a 5 - (a 3)) ^ 2) = 8) :
  d = 2 ∨ d = -2 := 
sorry

end arithmetic_sequence_common_difference_l67_67633


namespace final_pens_count_l67_67875

-- Define the initial number of pens and subsequent operations
def initial_pens : ℕ := 7
def pens_after_mike (initial : ℕ) : ℕ := initial + 22
def pens_after_cindy (pens : ℕ) : ℕ := pens * 2
def pens_after_sharon (pens : ℕ) : ℕ := pens - 19

-- Prove that the final number of pens is 39
theorem final_pens_count : pens_after_sharon (pens_after_cindy (pens_after_mike initial_pens)) = 39 := 
sorry

end final_pens_count_l67_67875


namespace smallest_n_l67_67294

theorem smallest_n (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 2012) :
  ∃ m n : ℕ, a.factorial * b.factorial * c.factorial = m * 10 ^ n ∧ ¬ (10 ∣ m) ∧ n = 501 :=
by
  sorry

end smallest_n_l67_67294


namespace sugar_amount_indeterminate_l67_67063

-- Define the variables and conditions
variable (cups_of_flour_needed : ℕ) (cups_of_sugar_needed : ℕ)
variable (cups_of_flour_put_in : ℕ) (cups_of_flour_to_add : ℕ)

-- Conditions
axiom H1 : cups_of_flour_needed = 8
axiom H2 : cups_of_flour_put_in = 4
axiom H3 : cups_of_flour_to_add = 4

-- Problem statement: Prove that the amount of sugar cannot be determined
theorem sugar_amount_indeterminate (h : cups_of_sugar_needed > 0) :
  cups_of_flour_needed = 8 → cups_of_flour_put_in = 4 → cups_of_flour_to_add = 4 → cups_of_sugar_needed > 0 :=
by
  intros
  sorry

end sugar_amount_indeterminate_l67_67063


namespace numerator_multiple_of_prime_l67_67385

theorem numerator_multiple_of_prime (n : ℕ) (hp : Nat.Prime (3 * n + 1)) :
  (2 * n - 1) % (3 * n + 1) = 0 :=
sorry

end numerator_multiple_of_prime_l67_67385


namespace surface_area_to_lateral_surface_ratio_cone_l67_67745

noncomputable def cone_surface_lateral_area_ratio : Prop :=
  let radius : ℝ := 1
  let theta : ℝ := (2 * Real.pi) / 3
  let lateral_surface_area := Real.pi * radius^2 * (theta / (2 * Real.pi))
  let base_radius := (2 * Real.pi * radius * (theta / (2 * Real.pi))) / (2 * Real.pi)
  let base_area := Real.pi * base_radius^2
  let surface_area := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = (4 / 3)

theorem surface_area_to_lateral_surface_ratio_cone :
  cone_surface_lateral_area_ratio :=
  by
  sorry

end surface_area_to_lateral_surface_ratio_cone_l67_67745


namespace length_RS_l67_67783

open Real

-- Given definitions and conditions
def PQ : ℝ := 10
def PR : ℝ := 10
def QR : ℝ := 5
def PS : ℝ := 13

-- Prove the length of RS
theorem length_RS : ∃ (RS : ℝ), RS = 6.17362 := by
  sorry

end length_RS_l67_67783


namespace seashell_count_l67_67312

def initialSeashells : Nat := 5
def givenSeashells : Nat := 2
def remainingSeashells : Nat := initialSeashells - givenSeashells

theorem seashell_count : remainingSeashells = 3 := by
  sorry

end seashell_count_l67_67312


namespace inequality_solution_l67_67505

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) ∨ 7 < x) →
  (1 / (x - 1)) - (4 / (x - 2)) + (4 / (x - 3)) - (1 / (x - 4)) < 1 / 30 :=
by
  sorry

end inequality_solution_l67_67505


namespace brokerage_percentage_l67_67610

theorem brokerage_percentage (cash_realized amount_before : ℝ) (h1 : cash_realized = 105.25) (h2 : amount_before = 105) :
  |((amount_before - cash_realized) / amount_before) * 100| = 0.2381 := by
sorry

end brokerage_percentage_l67_67610


namespace find_a_l67_67268

open Real

noncomputable def valid_solutions (a b : ℝ) : Prop :=
  a + 2 / b = 17 ∧ b + 2 / a = 1 / 3

theorem find_a (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : valid_solutions a b) :
  a = 6 ∨ a = 17 :=
by sorry

end find_a_l67_67268


namespace continuity_at_x0_l67_67066

def f (x : ℝ) : ℝ := -5 * x ^ 2 - 8
def x0 : ℝ := 2

theorem continuity_at_x0 :
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ (∀ (x : ℝ), abs (x - x0) < δ → abs (f x - f x0) < ε) :=
by
  sorry

end continuity_at_x0_l67_67066


namespace spinner_final_direction_l67_67466

theorem spinner_final_direction 
  (initial_direction : ℕ) -- 0 for north, 1 for east, 2 for south, 3 for west
  (clockwise_revolutions : ℚ)
  (counterclockwise_revolutions : ℚ)
  (net_revolutions : ℚ) -- derived via net movement calculation
  (final_position : ℕ) -- correct position after net movement
  : initial_direction = 3 → clockwise_revolutions = 9/4 → counterclockwise_revolutions = 15/4 → final_position = 1 :=
by
  sorry

end spinner_final_direction_l67_67466


namespace integer_points_on_segment_l67_67195

open Int

def is_integer_point (x y : ℝ) : Prop := ∃ (a b : ℤ), x = a ∧ y = b

def f (n : ℕ) : ℕ := 
  if 3 ∣ n then 2
  else 0

theorem integer_points_on_segment (n : ℕ) (hn : 0 < n) :
  (f n) = if 3 ∣ n then 2 else 0 := 
  sorry

end integer_points_on_segment_l67_67195


namespace count_integers_satisfying_conditions_l67_67619

theorem count_integers_satisfying_conditions :
  (∃ (s : Finset ℤ), s.card = 3 ∧
  ∀ x : ℤ, x ∈ s ↔ (-5 ≤ x ∧ x ≤ -3)) :=
by {
  sorry
}

end count_integers_satisfying_conditions_l67_67619


namespace auditorium_seats_l67_67731

variable (S : ℕ)

theorem auditorium_seats (h1 : 2 * S / 5 + S / 10 + 250 = S) : S = 500 :=
by
  sorry

end auditorium_seats_l67_67731


namespace part1_assoc_eq_part2_k_range_part3_m_range_l67_67068

-- Part 1
theorem part1_assoc_eq (x : ℝ) :
  (2 * (x + 1) - x = -3 ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  ((x+1)/3 - 1 = x ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  (2 * x - 7 = 0 ∧ (-4 < x ∧ x ≤ 4)) :=
sorry

-- Part 2
theorem part2_k_range (k : ℝ) :
  (∀ (x : ℝ), (x = (k + 6) / 2) → -5 < x ∧ x ≤ -3) ↔ (-16 < k) ∧ (k ≤ -12) :=
sorry 

-- Part 3
theorem part3_m_range (m : ℝ) :
  (∀ (x : ℝ), (x = 6 * m - 5) → (0 < x) ∧ (x ≤ 3 * m + 1) ∧ (1 ≤ x) ∧ (x ≤ 3)) ↔ (5/6 < m) ∧ (m < 1) :=
sorry

end part1_assoc_eq_part2_k_range_part3_m_range_l67_67068


namespace no_real_y_for_common_solution_l67_67247

theorem no_real_y_for_common_solution :
  ∀ (x y : ℝ), x^2 + y^2 = 25 → x^2 + 3 * y = 45 → false :=
by 
sorry

end no_real_y_for_common_solution_l67_67247


namespace solve_for_a_l67_67308

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem solve_for_a : 
  (∃ a : ℝ, f (f 0 a) a = 4 * a) → (a = 2) :=
by
  sorry

end solve_for_a_l67_67308


namespace sum_of_integers_eq_28_24_23_l67_67215

theorem sum_of_integers_eq_28_24_23 
  (a b : ℕ) 
  (h1 : a * b + a + b = 143)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 30)
  (h4 : b < 30) 
  : a + b = 28 ∨ a + b = 24 ∨ a + b = 23 :=
by
  sorry

end sum_of_integers_eq_28_24_23_l67_67215


namespace calculate_expression_l67_67277

theorem calculate_expression :
  4 + ((-2)^2) * 2 + (-36) / 4 = 3 := by
  sorry

end calculate_expression_l67_67277


namespace complete_square_l67_67207

theorem complete_square (x : ℝ) : x^2 + 4*x + 1 = 0 -> (x + 2)^2 = 3 :=
by sorry

end complete_square_l67_67207


namespace gcd_390_455_546_l67_67475

theorem gcd_390_455_546 : Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
by
  sorry    -- this indicates the proof is not included

end gcd_390_455_546_l67_67475


namespace find_k_l67_67864

def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 6
def g (k x : ℝ) : ℝ := 2 * x^2 - k * x + 2

theorem find_k (k : ℝ) : 
  f 5 - g k 5 = 15 -> k = -15.8 :=
by
  intro h
  sorry

end find_k_l67_67864


namespace reduce_fraction_l67_67638

-- Defining a structure for a fraction
structure Fraction where
  num : ℕ
  denom : ℕ
  deriving Repr

-- The original fraction
def originalFraction : Fraction :=
  { num := 368, denom := 598 }

-- The reduced fraction
def reducedFraction : Fraction :=
  { num := 184, denom := 299 }

-- The statement of our theorem
theorem reduce_fraction :
  ∃ (d : ℕ), d > 0 ∧ (originalFraction.num / d = reducedFraction.num) ∧ (originalFraction.denom / d = reducedFraction.denom) := by
  sorry

end reduce_fraction_l67_67638


namespace iron_balls_count_l67_67536

-- Conditions
def length_bar := 12  -- in cm
def width_bar := 8    -- in cm
def height_bar := 6   -- in cm
def num_bars := 10
def volume_iron_ball := 8  -- in cubic cm

-- Calculate the volume of one iron bar
def volume_one_bar := length_bar * width_bar * height_bar

-- Calculate the total volume of the ten iron bars
def total_volume := volume_one_bar * num_bars

-- Calculate the number of iron balls
def num_iron_balls := total_volume / volume_iron_ball

-- The proof statement
theorem iron_balls_count : num_iron_balls = 720 := by
  sorry

end iron_balls_count_l67_67536


namespace sequence_match_l67_67967

-- Define the sequence sum S_n
def S_n (n : ℕ) : ℕ := 2^(n + 1) - 1

-- Define the sequence a_n based on the problem statement
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 2^n

-- The theorem stating that sequence a_n satisfies the given sum condition S_n
theorem sequence_match (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end sequence_match_l67_67967


namespace parallel_lines_slope_l67_67309

theorem parallel_lines_slope (a : ℝ) :
  (∃ b : ℝ, ( ∀ x y : ℝ, a*x - 5*y - 9 = 0 → b*x - 3*y - 10 = 0) → a = 10/3) :=
sorry

end parallel_lines_slope_l67_67309


namespace midpoint_uniqueness_l67_67013

-- Define a finite set of points in the plane
axiom S : Finset (ℝ × ℝ)

-- Define what it means for P to be the midpoint of a segment
def is_midpoint (P A A' : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + A'.1) / 2 ∧ P.2 = (A.2 + A'.2) / 2

-- Statement of the problem
theorem midpoint_uniqueness (P Q : ℝ × ℝ) :
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint P A A') →
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint Q A A') →
  P = Q :=
sorry

end midpoint_uniqueness_l67_67013


namespace stratified_sampling_male_students_l67_67231

theorem stratified_sampling_male_students (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 900) (h2 : female_students = 0) (h3 : sample_size = 45) : 
  ((total_students - female_students) * sample_size / total_students) = 25 := 
by {
  sorry
}

end stratified_sampling_male_students_l67_67231


namespace square_section_dimensions_l67_67052

theorem square_section_dimensions (x length : ℕ) :
  (250 ≤ x^2 + x * length ∧ x^2 + x * length ≤ 300) ∧ (25 ≤ length ∧ length ≤ 30) →
  (x = 7 ∨ x = 8) :=
  by
    sorry

end square_section_dimensions_l67_67052


namespace abs_neg_implies_nonpositive_l67_67076

theorem abs_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  sorry

end abs_neg_implies_nonpositive_l67_67076


namespace complete_the_square_example_l67_67044

theorem complete_the_square_example : ∀ x m n : ℝ, (x^2 - 12 * x + 33 = 0) → 
  (x + m)^2 = n → m = -6 ∧ n = 3 :=
by
  sorry

end complete_the_square_example_l67_67044


namespace range_of_a_l67_67949

def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3
def g (a x : ℝ) : ℝ := x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(f a x < 0 ∧ g a x < 0)) ↔ a ∈ Set.Icc (-3 : ℝ) 6 :=
sorry

end range_of_a_l67_67949


namespace problem1_problem2_l67_67573

variable (x y : ℝ)

theorem problem1 :
  x^4 * x^3 * x - (x^4)^2 + (-2 * x)^3 * x^5 = -8 * x^8 :=
by sorry

theorem problem2 :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end problem1_problem2_l67_67573


namespace problem_part1_problem_part2_l67_67331

noncomputable def f (m x : ℝ) := Real.log (m * x) - x + 1
noncomputable def g (m x : ℝ) := (x - 1) * Real.exp x - m * x

theorem problem_part1 (m : ℝ) (h : m > 0) (hf : ∀ x, f m x ≤ 0) : m = 1 :=
sorry

theorem problem_part2 (m : ℝ) (h : m > 0) :
  ∃ x₀, (∀ x, g m x ≤ g m x₀) ∧ (1 / 2 * Real.log (m + 1) < x₀ ∧ x₀ < m) :=
sorry

end problem_part1_problem_part2_l67_67331


namespace select_16_genuine_coins_l67_67966

theorem select_16_genuine_coins (coins : Finset ℕ) (h_coins_count : coins.card = 40) 
  (counterfeit : Finset ℕ) (h_counterfeit_count : counterfeit.card = 3)
  (h_counterfeit_lighter : ∀ c ∈ counterfeit, ∀ g ∈ (coins \ counterfeit), c < g) :
  ∃ genuine : Finset ℕ, genuine.card = 16 ∧ 
    (∀ h1 h2 h3 : Finset ℕ, h1.card = 20 → h2.card = 10 → h3.card = 8 →
      ((h1 ⊆ coins ∧ h2 ⊆ h1 ∧ h3 ⊆ (h1 \ counterfeit)) ∨
       (h1 ⊆ coins ∧ h2 ⊆ (h1 \ counterfeit) ∧ h3 ⊆ (h2 \ counterfeit))) →
      genuine ⊆ coins \ counterfeit) :=
sorry

end select_16_genuine_coins_l67_67966


namespace determinant_of_matrixA_l67_67857

variable (x : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_of_matrixA : Matrix.det (matrixA x) = 8 * x + 8 := by
  sorry

end determinant_of_matrixA_l67_67857


namespace minimum_value_of_x_minus_y_l67_67842

variable (x y : ℝ)
open Real

theorem minimum_value_of_x_minus_y (hx : x > 0) (hy : y < 0) 
  (h : (1 / (x + 2)) + (1 / (1 - y)) = 1 / 6) : 
  x - y = 21 :=
sorry

end minimum_value_of_x_minus_y_l67_67842


namespace tan_alpha_neg_seven_l67_67477

noncomputable def tan_alpha (α : ℝ) := Real.tan α

theorem tan_alpha_neg_seven {α : ℝ} 
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.cos α ^ 2 + Real.sin (Real.pi + 2 * α) = 3 / 10) : 
  tan_alpha α = -7 := 
sorry

end tan_alpha_neg_seven_l67_67477


namespace circle_diameter_l67_67483

theorem circle_diameter (A : ℝ) (hA : A = 25 * π) (r : ℝ) (h : A = π * r^2) : 2 * r = 10 := by
  sorry

end circle_diameter_l67_67483


namespace present_age_of_son_l67_67852

variable (S M : ℕ)

-- Conditions
def age_difference : Prop := M = S + 40
def age_relation_in_seven_years : Prop := M + 7 = 3 * (S + 7)

-- Theorem to prove
theorem present_age_of_son : age_difference S M → age_relation_in_seven_years S M → S = 13 := by
  sorry

end present_age_of_son_l67_67852


namespace discounted_price_is_correct_l67_67740

def original_price_of_cork (C : ℝ) : Prop :=
  C + (C + 2.00) = 2.10

def discounted_price_of_cork (C : ℝ) : ℝ :=
  C - (C * 0.12)

theorem discounted_price_is_correct :
  ∃ C : ℝ, original_price_of_cork C ∧ discounted_price_of_cork C = 0.044 :=
by
  sorry

end discounted_price_is_correct_l67_67740


namespace no_solution_k_eq_7_l67_67074

-- Define the condition that x should not be equal to 4 and 8
def condition (x : ℝ) : Prop := x ≠ 4 ∧ x ≠ 8

-- Define the equation
def equation (x k : ℝ) : Prop := (x - 3) / (x - 4) = (x - k) / (x - 8)

-- Prove that for the equation to have no solution, k must be 7
theorem no_solution_k_eq_7 : (∀ x, condition x → ¬ equation x 7) ↔ (∃ k, k = 7) :=
by
  sorry

end no_solution_k_eq_7_l67_67074


namespace odd_function_evaluation_l67_67335

theorem odd_function_evaluation (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) (h : f (-3) = -2) : f 3 + f 0 = 2 :=
by 
  sorry

end odd_function_evaluation_l67_67335


namespace sophia_fraction_of_pie_l67_67604

theorem sophia_fraction_of_pie
  (weight_fridge : ℕ) (weight_eaten : ℕ)
  (h1 : weight_fridge = 1200)
  (h2 : weight_eaten = 240) :
  (weight_eaten : ℚ) / ((weight_fridge + weight_eaten : ℚ)) = (1 / 6) :=
by
  sorry

end sophia_fraction_of_pie_l67_67604


namespace solve_for_m_l67_67193

def f (x : ℝ) (m : ℝ) := x^3 - m * x + 3

def f_prime (x : ℝ) (m : ℝ) := 3 * x^2 - m

theorem solve_for_m (m : ℝ) : f_prime 1 m = 0 → m = 3 :=
by
  sorry

end solve_for_m_l67_67193


namespace intersection_point_l67_67430

theorem intersection_point : ∃ (x y : ℝ), y = 3 - x ∧ y = 3 * x - 5 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l67_67430


namespace grooming_time_l67_67779

theorem grooming_time (time_per_dog : ℕ) (num_dogs : ℕ) (days : ℕ) (minutes_per_hour : ℕ) :
  time_per_dog = 20 →
  num_dogs = 2 →
  days = 30 →
  minutes_per_hour = 60 →
  (time_per_dog * num_dogs * days) / minutes_per_hour = 20 := 
by
  intros
  exact sorry

end grooming_time_l67_67779


namespace tan_cos_sin_fraction_l67_67673

theorem tan_cos_sin_fraction (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 := 
by
  sorry

end tan_cos_sin_fraction_l67_67673


namespace series_satisfies_l67_67500

noncomputable def series (x : ℝ) : ℝ :=
  let S₁ := 1 / (1 + x^2)
  let S₂ := x / (1 + x^2)
  (S₁ - S₂)

theorem series_satisfies (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  x = series x ↔ x^3 + 2 * x - 1 = 0 :=
by 
  -- Proof outline:
  -- 1. Calculate the series S as a function of x
  -- 2. Equate series x to x and simplify to derive the polynomial equation
  sorry

end series_satisfies_l67_67500


namespace combined_books_total_l67_67978

def keith_books : ℕ := 20
def jason_books : ℕ := 21
def amanda_books : ℕ := 15
def sophie_books : ℕ := 30

def total_books := keith_books + jason_books + amanda_books + sophie_books

theorem combined_books_total : total_books = 86 := 
by sorry

end combined_books_total_l67_67978


namespace value_of_a_is_3_l67_67956

def symmetric_about_x1 (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| + |x - a| = |2 - x + 1| + |2 - x - a|

theorem value_of_a_is_3 : symmetric_about_x1 3 :=
sorry

end value_of_a_is_3_l67_67956


namespace infinitely_many_MTRP_numbers_l67_67891

def sum_of_digits (n : ℕ) : ℕ := 
n.digits 10 |>.sum

def is_MTRP_number (m n : ℕ) : Prop :=
  n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_MTRP_numbers (m : ℕ) : 
  ∀ N : ℕ, ∃ n > N, is_MTRP_number m n :=
by sorry

end infinitely_many_MTRP_numbers_l67_67891


namespace sarah_rye_flour_l67_67047

-- Definitions
variables (b c p t r : ℕ)

-- Conditions
def condition1 : Prop := b = 10
def condition2 : Prop := c = 3
def condition3 : Prop := p = 2
def condition4 : Prop := t = 20

-- Proposition to prove
theorem sarah_rye_flour : condition1 b → condition2 c → condition3 p → condition4 t → r = t - (b + c + p) → r = 5 :=
by
  intros h1 h2 h3 h4 hr
  rw [h1, h2, h3, h4] at hr
  exact hr

end sarah_rye_flour_l67_67047


namespace prime_in_choices_l67_67182

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def twenty := 20
def twenty_one := 21
def twenty_three := 23
def twenty_five := 25
def twenty_seven := 27

theorem prime_in_choices :
  is_prime twenty_three ∧ ¬ is_prime twenty ∧ ¬ is_prime twenty_one ∧ ¬ is_prime twenty_five ∧ ¬ is_prime twenty_seven :=
by
  sorry

end prime_in_choices_l67_67182


namespace parallelogram_base_length_l67_67986

theorem parallelogram_base_length (b : ℝ) (A : ℝ) (h : ℝ)
  (H1 : A = 288) 
  (H2 : h = 2 * b) 
  (H3 : A = b * h) : 
  b = 12 := 
by 
  sorry

end parallelogram_base_length_l67_67986


namespace total_amount_shared_l67_67357

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end total_amount_shared_l67_67357


namespace total_pencils_l67_67580

theorem total_pencils (pencils_per_box : ℕ) (friends : ℕ) (total_pencils : ℕ) : 
  pencils_per_box = 7 ∧ friends = 5 → total_pencils = pencils_per_box + friends * pencils_per_box → total_pencils = 42 :=
by
  intros h1 h2
  sorry

end total_pencils_l67_67580


namespace highest_number_of_years_of_service_l67_67095

theorem highest_number_of_years_of_service
  (years_of_service : Fin 8 → ℕ)
  (h_range : ∃ L, ∃ H, H - L = 14)
  (h_second_highest : ∃ second_highest, second_highest = 16) :
  ∃ highest, highest = 17 := by
  sorry

end highest_number_of_years_of_service_l67_67095


namespace sleeves_add_correct_weight_l67_67647

variable (R W_r W_s S : ℝ)

-- Conditions
def raw_squat : Prop := R = 600
def wraps_add_25_percent : Prop := W_r = R + 0.25 * R
def wraps_vs_sleeves_difference : Prop := W_r = W_s + 120

-- To Prove
theorem sleeves_add_correct_weight (h1 : raw_squat R) (h2 : wraps_add_25_percent R W_r) (h3 : wraps_vs_sleeves_difference W_r W_s) : S = 30 :=
by
  sorry

end sleeves_add_correct_weight_l67_67647


namespace identical_remainders_l67_67629

theorem identical_remainders (a : Fin 11 → Fin 11) (h_perm : ∀ n, ∃ m, a m = n) :
  ∃ (i j : Fin 11), i ≠ j ∧ (i * a i) % 11 = (j * a j) % 11 :=
by 
  sorry

end identical_remainders_l67_67629


namespace range_of_set_l67_67140

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end range_of_set_l67_67140


namespace alex_growth_rate_l67_67947

noncomputable def growth_rate_per_hour_hanging_upside_down
  (current_height : ℝ)
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) : Prop :=
  current_height + 12 * normal_growth_per_month + 12 * hanging_hours_per_month * answer = required_height

theorem alex_growth_rate 
  (current_height : ℝ) 
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) :
  current_height = 48 → 
  required_height = 54 → 
  normal_growth_per_month = 1/3 → 
  hanging_hours_per_month = 2 → 
  growth_rate_per_hour_hanging_upside_down current_height required_height normal_growth_per_month hanging_hours_per_month answer ↔ answer = 1/12 :=
by sorry

end alex_growth_rate_l67_67947


namespace number_of_ways_l67_67158

theorem number_of_ways (h_walk : ℕ) (h_drive : ℕ) (h_eq1 : h_walk = 3) (h_eq2 : h_drive = 4) : h_walk + h_drive = 7 :=
by 
  sorry

end number_of_ways_l67_67158


namespace percentage_music_students_l67_67501

variables (total_students : ℕ) (dance_students : ℕ) (art_students : ℕ)
  (music_students : ℕ) (music_percentage : ℚ)

def students_music : ℕ := total_students - (dance_students + art_students)
def percentage_students_music : ℚ := (students_music total_students dance_students art_students : ℚ) / (total_students : ℚ) * 100

theorem percentage_music_students (h1 : total_students = 400)
                                  (h2 : dance_students = 120)
                                  (h3 : art_students = 200) :
  percentage_students_music total_students dance_students art_students = 20 := by {
  sorry
}

end percentage_music_students_l67_67501


namespace derivative_at_2_l67_67684

def f (x : ℝ) : ℝ := (x + 3) * (x + 2) * (x + 1) * x * (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_2 : (deriv f 2) = -120 :=
by
  sorry

end derivative_at_2_l67_67684


namespace positive_operation_l67_67628

def operation_a := 1 + (-2)
def operation_b := 1 - (-2)
def operation_c := 1 * (-2)
def operation_d := 1 / (-2)

theorem positive_operation : operation_b > 0 ∧ 
  (operation_a <= 0) ∧ (operation_c <= 0) ∧ (operation_d <= 0) := by
  sorry

end positive_operation_l67_67628


namespace find_n_in_range_and_modulus_l67_67692

theorem find_n_in_range_and_modulus :
  ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ (-200) % 21 = n % 21 → n = 10 := by
  sorry

end find_n_in_range_and_modulus_l67_67692


namespace angle_DEF_EDF_proof_l67_67161

theorem angle_DEF_EDF_proof (angle_DOE : ℝ) (angle_EOD : ℝ) 
  (h1 : angle_DOE = 130) (h2 : angle_EOD = 90) :
  let angle_DEF := 45
  let angle_EDF := 45
  angle_DEF = 45 ∧ angle_EDF = 45 :=
by
  sorry

end angle_DEF_EDF_proof_l67_67161


namespace find_roots_l67_67763

theorem find_roots : ∀ z : ℂ, (z^2 + 2 * z = 3 - 4 * I) → (z = 1 - I ∨ z = -3 + I) :=
by
  intro z
  intro h
  sorry

end find_roots_l67_67763


namespace total_length_of_intervals_l67_67100

theorem total_length_of_intervals :
  (∀ (x : ℝ), |x| < 1 → Real.tan (Real.log x / Real.log 5) < 0) →
  ∃ (length : ℝ), length = (2 * (5 ^ (Real.pi / 2))) / (1 + (5 ^ (Real.pi / 2))) :=
sorry

end total_length_of_intervals_l67_67100


namespace scaling_transformation_l67_67062

theorem scaling_transformation (a b : ℝ) :
  (∀ x y : ℝ, (y = 1 - x → y' = b * (1 - x))
    → (y' = b - b * x)) 
  ∧
  (∀ x' y' : ℝ, (y = (2 / 3) * x' + 2)
    → (y' = (2 / 3) * (a * x) + 2))
  → a = 3 ∧ b = 2 := by
  sorry

end scaling_transformation_l67_67062


namespace slope_of_line_n_l67_67464

noncomputable def tan_double_angle (t : ℝ) : ℝ := (2 * t) / (1 - t^2)

theorem slope_of_line_n :
  let slope_m := 6
  let alpha := Real.arctan slope_m
  let slope_n := tan_double_angle slope_m
  slope_n = -12 / 35 :=
by
  sorry

end slope_of_line_n_l67_67464


namespace problem_circumscribing_sphere_surface_area_l67_67979

noncomputable def surface_area_of_circumscribing_sphere (a b c : ℕ) :=
  let R := (Real.sqrt (a^2 + b^2 + c^2)) / 2
  4 * Real.pi * R^2

theorem problem_circumscribing_sphere_surface_area
  (a b c : ℕ)
  (ha : (1 / 2 : ℝ) * a * b = 4)
  (hb : (1 / 2 : ℝ) * b * c = 6)
  (hc : (1 / 2: ℝ) * a * c = 12) : 
  surface_area_of_circumscribing_sphere a b c = 56 * Real.pi := 
sorry

end problem_circumscribing_sphere_surface_area_l67_67979


namespace repetend_five_seventeen_l67_67558

noncomputable def repetend_of_fraction (n : ℕ) (d : ℕ) : ℕ := sorry

theorem repetend_five_seventeen : repetend_of_fraction 5 17 = 294117647058823529 := sorry

end repetend_five_seventeen_l67_67558


namespace increasing_on_interval_l67_67695

open Real

noncomputable def f (x a b : ℝ) := abs (x^2 - 2*a*x + b)

theorem increasing_on_interval {a b : ℝ} (h : a^2 - b ≤ 0) :
  ∀ ⦃x1 x2⦄, a ≤ x1 → x1 ≤ x2 → f x1 a b ≤ f x2 a b := sorry

end increasing_on_interval_l67_67695


namespace tickets_system_l67_67982

variable (x y : ℕ)

theorem tickets_system (h1 : x + y = 20) (h2 : 2800 * x + 6400 * y = 74000) :
  (x + y = 20) ∧ (2800 * x + 6400 * y = 74000) :=
by {
  exact (And.intro h1 h2)
}

end tickets_system_l67_67982


namespace percentage_not_drop_l67_67387

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l67_67387


namespace min_both_attendees_l67_67775

-- Defining the parameters and conditions
variable (n : ℕ) -- total number of attendees
variable (glasses name_tags both : ℕ) -- attendees wearing glasses, name tags, and both

-- Conditions provided in the problem
def wearing_glasses_condition (n : ℕ) (glasses : ℕ) : Prop := glasses = n / 3
def wearing_name_tags_condition (n : ℕ) (name_tags : ℕ) : Prop := name_tags = n / 2
def total_attendees_condition (n : ℕ) : Prop := n = 6

-- Theorem to prove the minimum attendees wearing both glasses and name tags is 1
theorem min_both_attendees (n glasses name_tags both : ℕ) (h1 : wearing_glasses_condition n glasses) 
  (h2 : wearing_name_tags_condition n name_tags) (h3 : total_attendees_condition n) : 
  both = 1 :=
sorry

end min_both_attendees_l67_67775


namespace expression_value_l67_67159

theorem expression_value (x : ℤ) (h : x = 2) : (2 * x + 5)^3 = 729 := by
  sorry

end expression_value_l67_67159


namespace anna_gets_more_candy_l67_67131

theorem anna_gets_more_candy :
  let anna_pieces_per_house := 14
  let anna_houses := 60
  let billy_pieces_per_house := 11
  let billy_houses := 75
  let anna_total := anna_pieces_per_house * anna_houses
  let billy_total := billy_pieces_per_house * billy_houses
  anna_total - billy_total = 15 := by
    let anna_pieces_per_house := 14
    let anna_houses := 60
    let billy_pieces_per_house := 11
    let billy_houses := 75
    let anna_total := anna_pieces_per_house * anna_houses
    let billy_total := billy_pieces_per_house * billy_houses
    have h1 : anna_total = 14 * 60 := rfl
    have h2 : billy_total = 11 * 75 := rfl
    sorry

end anna_gets_more_candy_l67_67131


namespace jane_doe_investment_l67_67554

theorem jane_doe_investment (total_investment mutual_funds real_estate : ℝ)
  (h1 : total_investment = 250000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total_investment = mutual_funds + real_estate) :
  real_estate = 187500 :=
by
  sorry

end jane_doe_investment_l67_67554


namespace find_somu_age_l67_67689

noncomputable def somu_age (S F : ℕ) : Prop :=
  S = (1/3 : ℝ) * F ∧ S - 6 = (1/5 : ℝ) * (F - 6)

theorem find_somu_age {S F : ℕ} (h : somu_age S F) : S = 12 :=
by sorry

end find_somu_age_l67_67689


namespace max_distance_between_P_and_Q_l67_67450

-- Definitions of the circle and ellipse
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + (P.2 - 6)^2 = 2
def is_on_ellipse (Q : ℝ × ℝ) : Prop := (Q.1^2) / 10 + Q.2^2 = 1

-- The maximum distance between any point on the circle and any point on the ellipse
theorem max_distance_between_P_and_Q :
  ∃ P Q : ℝ × ℝ, is_on_circle P ∧ is_on_ellipse Q ∧ dist P Q = 6 * Real.sqrt 2 :=
sorry

end max_distance_between_P_and_Q_l67_67450


namespace negation_exists_equation_l67_67435

theorem negation_exists_equation (P : ℝ → Prop) :
  (∃ x > 0, x^2 + 3 * x - 5 = 0) → ¬ (∃ x > 0, x^2 + 3 * x - 5 = 0) = ∀ x > 0, x^2 + 3 * x - 5 ≠ 0 :=
by sorry

end negation_exists_equation_l67_67435


namespace units_digit_fraction_l67_67192

theorem units_digit_fraction :
  (30 * 31 * 32 * 33 * 34 * 35) % 10000 % 10 = 4 :=
by
  -- Placeholder for actual proof
  sorry

end units_digit_fraction_l67_67192


namespace lens_discount_l67_67319

theorem lens_discount :
  ∃ (P : ℚ), ∀ (D : ℚ),
    (300 - D = 240) →
    (P = (D / 300) * 100) →
    P = 20 :=
by
  sorry

end lens_discount_l67_67319


namespace gcd_of_polynomial_l67_67043

def multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem gcd_of_polynomial (b : ℕ) (h : multiple_of b 456) :
  Nat.gcd (4 * b^3 + b^2 + 6 * b + 152) b = 152 := sorry

end gcd_of_polynomial_l67_67043


namespace garbage_classification_competition_l67_67989

theorem garbage_classification_competition :
  let boy_rate_seventh := 0.4
  let boy_rate_eighth := 0.5
  let girl_rate_seventh := 0.6
  let girl_rate_eighth := 0.7
  let combined_boy_rate := (boy_rate_seventh + boy_rate_eighth) / 2
  let combined_girl_rate := (girl_rate_seventh + girl_rate_eighth) / 2
  boy_rate_seventh < boy_rate_eighth ∧ combined_boy_rate < combined_girl_rate :=
by {
  sorry
}

end garbage_classification_competition_l67_67989


namespace opera_house_earnings_l67_67129

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end opera_house_earnings_l67_67129


namespace volume_first_cube_l67_67831

theorem volume_first_cube (a b : ℝ) (h_ratio : a = 3 * b) (h_volume : b^3 = 8) : a^3 = 216 :=
by
  sorry

end volume_first_cube_l67_67831


namespace min_value_of_x_plus_y_l67_67298

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y = x * y) : x + y ≥ 9 :=
by
  sorry

end min_value_of_x_plus_y_l67_67298


namespace missing_water_calculation_l67_67292

def max_capacity : ℝ := 350000
def loss_rate1 : ℝ := 32000
def time1 : ℝ := 5
def loss_rate2 : ℝ := 10000
def time2 : ℝ := 10
def fill_rate : ℝ := 40000
def fill_time : ℝ := 3

theorem missing_water_calculation :
  350000 - ((350000 - (32000 * 5 + 10000 * 10)) + 40000 * 3) = 140000 :=
by
  sorry

end missing_water_calculation_l67_67292


namespace arrange_descending_order_l67_67603

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem arrange_descending_order : c > a ∧ a > b := by
  sorry

end arrange_descending_order_l67_67603


namespace gcd_of_differences_l67_67060

theorem gcd_of_differences (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 1351) : 
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a) = 4 :=
by
  sorry

end gcd_of_differences_l67_67060


namespace find_fahrenheit_l67_67171

variable (F : ℝ)
variable (C : ℝ)

theorem find_fahrenheit (h : C = 40) (h' : C = 5 / 9 * (F - 32)) : F = 104 := by
  sorry

end find_fahrenheit_l67_67171


namespace T_description_l67_67285

def is_single_point {x y : ℝ} : Prop := (x = 2) ∧ (y = 11)

theorem T_description :
  ∀ (T : Set (ℝ × ℝ)),
  (∀ x y : ℝ, 
    (T (x, y) ↔ 
    ((5 = x + 3 ∧ 5 = y - 6) ∨ 
     (5 = x + 3 ∧ x + 3 = y - 6) ∨ 
     (5 = y - 6 ∧ x + 3 = y - 6)) ∧ 
    ((x = 2) ∧ (y = 11))
    )
  ) →
  (T = { (2, 11) }) :=
by
  sorry

end T_description_l67_67285


namespace fraction_of_is_l67_67805

theorem fraction_of_is (a b c d e : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) (h5 : e = 8/27) :
  (a / b) = e * (c / d) := 
sorry

end fraction_of_is_l67_67805


namespace original_ratio_white_yellow_l67_67469

-- Define the given conditions
variables (W Y : ℕ)
axiom total_balls : W + Y = 64
axiom erroneous_dispatch : W = 8 * (Y + 20) / 13

-- The theorem we need to prove
theorem original_ratio_white_yellow (W Y : ℕ) (h1 : W + Y = 64) (h2 : W = 8 * (Y + 20) / 13) : W = Y :=
by sorry

end original_ratio_white_yellow_l67_67469


namespace converse_false_inverse_false_l67_67837

-- Definitions of the conditions
def is_rhombus (Q : Type) : Prop := -- definition of a rhombus
  sorry

def is_parallelogram (Q : Type) : Prop := -- definition of a parallelogram
  sorry

variable {Q : Type}

-- Initial statement: If a quadrilateral is a rhombus, then it is a parallelogram.
axiom initial_statement : is_rhombus Q → is_parallelogram Q

-- Goals: Prove both the converse and inverse are false
theorem converse_false : ¬ ((is_parallelogram Q) → (is_rhombus Q)) :=
sorry

theorem inverse_false : ¬ (¬ (is_rhombus Q) → ¬ (is_parallelogram Q)) :=
    sorry

end converse_false_inverse_false_l67_67837


namespace rectangle_area_l67_67209

theorem rectangle_area (w : ℝ) (h : ℝ) (area : ℝ) 
  (h1 : w = 5)
  (h2 : h = 2 * w) :
  area = h * w := by
  sorry

end rectangle_area_l67_67209


namespace unique_rectangle_dimensions_l67_67748

theorem unique_rectangle_dimensions (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < a ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = a * b / 4 :=
sorry

end unique_rectangle_dimensions_l67_67748


namespace stratified_sampling_example_l67_67406

theorem stratified_sampling_example 
  (N : ℕ) (S : ℕ) (D : ℕ) 
  (hN : N = 1000) (hS : S = 50) (hD : D = 200) : 
  D * (S : ℝ) / (N : ℝ) = 10 := 
by
  sorry

end stratified_sampling_example_l67_67406


namespace find_natural_number_n_l67_67632

theorem find_natural_number_n (n x y : ℕ) (h1 : n + 195 = x^3) (h2 : n - 274 = y^3) : 
  n = 2002 :=
by
  sorry

end find_natural_number_n_l67_67632


namespace class_average_weight_l67_67352

theorem class_average_weight :
  (24 * 40 + 16 * 35 + 18 * 42 + 22 * 38) / (24 + 16 + 18 + 22) = 38.9 :=
by
  -- skipped proof
  sorry

end class_average_weight_l67_67352


namespace exp_decreasing_iff_a_in_interval_l67_67124

theorem exp_decreasing_iff_a_in_interval (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 - a)^x > (2 - a)^y) ↔ 1 < a ∧ a < 2 :=
by 
  sorry

end exp_decreasing_iff_a_in_interval_l67_67124


namespace angle_P_in_quadrilateral_l67_67592

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end angle_P_in_quadrilateral_l67_67592


namespace major_snow_shadow_length_l67_67125

theorem major_snow_shadow_length :
  ∃ (a1 d : ℝ), 
  (3 * a1 + 12 * d = 16.5) ∧ 
  (12 * a1 + 66 * d = 84) ∧
  (a1 + 11 * d = 12.5) := 
sorry

end major_snow_shadow_length_l67_67125


namespace winning_candidate_percentage_is_57_l67_67572

def candidate_votes : List ℕ := [1136, 7636, 11628]

def total_votes : ℕ := candidate_votes.sum

def winning_votes : ℕ := candidate_votes.maximum?.getD 0

def winning_percentage (votes : ℕ) (total : ℕ) : ℚ :=
  (votes * 100) / total

theorem winning_candidate_percentage_is_57 :
  winning_percentage winning_votes total_votes = 57 := by
  sorry

end winning_candidate_percentage_is_57_l67_67572


namespace solve_inequality_l67_67433

theorem solve_inequality (x : ℝ) :
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 :=
  sorry

end solve_inequality_l67_67433


namespace candy_total_l67_67563

theorem candy_total (chocolate_boxes caramel_boxes mint_boxes berry_boxes : ℕ)
  (chocolate_pieces caramel_pieces mint_pieces berry_pieces : ℕ)
  (h_chocolate : chocolate_boxes = 7)
  (h_caramel : caramel_boxes = 3)
  (h_mint : mint_boxes = 5)
  (h_berry : berry_boxes = 4)
  (p_chocolate : chocolate_pieces = 8)
  (p_caramel : caramel_pieces = 8)
  (p_mint : mint_pieces = 10)
  (p_berry : berry_pieces = 12) :
  (chocolate_boxes * chocolate_pieces + caramel_boxes * caramel_pieces + mint_boxes * mint_pieces + berry_boxes * berry_pieces) = 178 := by
  sorry

end candy_total_l67_67563


namespace age_ratio_in_4_years_l67_67206

-- Definitions based on conditions
def Age6YearsAgoVimal := 12
def Age6YearsAgoSaroj := 10
def CurrentAgeSaroj := 16
def CurrentAgeVimal := Age6YearsAgoVimal + 6

-- Lean statement to prove the problem
theorem age_ratio_in_4_years (x : ℕ) 
  (h_ratio : (CurrentAgeVimal + x) / (CurrentAgeSaroj + x) = 11 / 10) :
  x = 4 := 
sorry

end age_ratio_in_4_years_l67_67206


namespace Bret_catches_12_frogs_l67_67759

-- Conditions from the problem
def frogs_caught_by_Alster : Nat := 2
def frogs_caught_by_Quinn : Nat := 2 * frogs_caught_by_Alster
def frogs_caught_by_Bret : Nat := 3 * frogs_caught_by_Quinn

-- Statement of the theorem to be proved
theorem Bret_catches_12_frogs : frogs_caught_by_Bret = 12 :=
by
  sorry

end Bret_catches_12_frogs_l67_67759


namespace share_difference_l67_67776

theorem share_difference (x : ℕ) (p q r : ℕ) 
  (h1 : 3 * x = p) 
  (h2 : 7 * x = q) 
  (h3 : 12 * x = r) 
  (h4 : q - p = 2800) : 
  r - q = 3500 := by {
  sorry
}

end share_difference_l67_67776


namespace number_of_terms_in_arithmetic_sequence_l67_67197

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l67_67197


namespace part1_part2_l67_67424

theorem part1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (2 / (1 + a)) + (2 / (1 + b)) + (2 / (1 + c)) :=
sorry

end part1_part2_l67_67424


namespace arithmetic_seq_third_term_l67_67537

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end arithmetic_seq_third_term_l67_67537


namespace correct_arrangements_count_l67_67438

def valid_arrangements_count : Nat :=
  let houses := ['O', 'R', 'B', 'Y', 'G']
  let arrangements := houses.permutations
  let valid_arr := arrangements.filter (fun a =>
    let o_idx := a.indexOf 'O'
    let r_idx := a.indexOf 'R'
    let b_idx := a.indexOf 'B'
    let y_idx := a.indexOf 'Y'
    let constraints_met :=
      o_idx < r_idx ∧       -- O before R
      b_idx < y_idx ∧       -- B before Y
      (b_idx + 1 != y_idx) ∧ -- B not next to Y
      (r_idx + 1 != b_idx) ∧ -- R not next to B
      (b_idx + 1 != r_idx)   -- symmetrical R not next to B

    constraints_met)
  valid_arr.length

theorem correct_arrangements_count : valid_arrangements_count = 5 :=
  by
    -- To be filled with proof steps.
    sorry

end correct_arrangements_count_l67_67438


namespace parallel_lines_m_values_l67_67810

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 ↔ mx + 3 * y - 2 = 0) → (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_values_l67_67810


namespace ticket_costs_l67_67046

-- Define the conditions
def cost_per_ticket : ℕ := 44
def number_of_tickets : ℕ := 7

-- Define the total cost calculation
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Prove that given the conditions, the total cost is 308
theorem ticket_costs :
  total_cost = 308 :=
by
  -- Proof steps here
  sorry

end ticket_costs_l67_67046


namespace geometric_sequence_sixth_term_l67_67150

/-- 
The statement: 
The first term of a geometric sequence is 1000, and the 8th term is 125. Prove that the positive,
real value for the 6th term is 31.25.
-/
theorem geometric_sequence_sixth_term :
  ∀ (a1 a8 a6 : ℝ) (r : ℝ),
    a1 = 1000 →
    a8 = 125 →
    a8 = a1 * r^7 →
    a6 = a1 * r^5 →
    a6 = 31.25 :=
by
  intros a1 a8 a6 r h1 h2 h3 h4
  sorry

end geometric_sequence_sixth_term_l67_67150


namespace ratio_Andrea_Jude_l67_67221

-- Definitions
def number_of_tickets := 100
def tickets_left := 40
def tickets_sold := number_of_tickets - tickets_left

def Jude_tickets := 16
def Sandra_tickets := 4 + 1/2 * Jude_tickets
def Andrea_tickets := tickets_sold - (Jude_tickets + Sandra_tickets)

-- Assertion that needs proof
theorem ratio_Andrea_Jude : 
  (Andrea_tickets / Jude_tickets) = 2 := by
  sorry

end ratio_Andrea_Jude_l67_67221


namespace arithmetic_sequence_problem_l67_67540

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem 
  (h : is_arithmetic_sequence a)
  (h_cond : a 2 + 2 * a 6 + a 10 = 120) :
  a 3 + a 9 = 60 :=
sorry

end arithmetic_sequence_problem_l67_67540


namespace Marla_colors_green_squares_l67_67909

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end Marla_colors_green_squares_l67_67909


namespace _l67_67238

noncomputable def urn_marble_theorem (r w b g y : Nat) : Prop :=
  let n := r + w + b + g + y
  ∃ k : Nat, 
  (k * r * (r-1) * (r-2) * (r-3) * (r-4) / 120 = w * r * (r-1) * (r-2) * (r-3) / 24)
  ∧ (w * r * (r-1) * (r-2) * (r-3) / 24 = w * b * r * (r-1) * (r-2) / 6)
  ∧ (w * b * r * (r-1) * (r-2) / 6 = w * b * g * r * (r-1) / 2)
  ∧ (w * b * g * r * (r-1) / 2 = w * b * g * r * y)
  ∧ n = 55

example : ∃ (r w b g y : Nat), urn_marble_theorem r w b g y := sorry

end _l67_67238


namespace puppies_in_each_cage_l67_67307

theorem puppies_in_each_cage (initial_puppies sold_puppies cages : ℕ)
  (h_initial : initial_puppies = 18)
  (h_sold : sold_puppies = 3)
  (h_cages : cages = 3) :
  (initial_puppies - sold_puppies) / cages = 5 :=
by
  sorry

end puppies_in_each_cage_l67_67307


namespace sum_of_perimeters_l67_67876

theorem sum_of_perimeters (a : ℝ) : 
    ∑' n : ℕ, (3 * a) * (1/3)^n = 9 * a / 2 :=
by sorry

end sum_of_perimeters_l67_67876


namespace sum_series_eq_one_l67_67844

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, (2^n + 1) / (3^(2^n) + 1)

theorem sum_series_eq_one : sum_series = 1 := 
by 
  sorry

end sum_series_eq_one_l67_67844


namespace abs_diff_of_m_and_n_l67_67520

theorem abs_diff_of_m_and_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 :=
sorry

end abs_diff_of_m_and_n_l67_67520


namespace inequality_proof_l67_67743

variable {a b c : ℝ}

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
by
  sorry

end inequality_proof_l67_67743


namespace reciprocal_2023_l67_67880

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l67_67880


namespace quadratic_inequality_solution_set_l67_67112

/- Given a quadratic function with specific roots and coefficients, prove a quadratic inequality. -/
theorem quadratic_inequality_solution_set :
  ∀ (a b : ℝ),
    (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 + a*x + b < 0) →
    a = -3 →
    b = 2 →
    ∀ x : ℝ, (x < 1/2 ∨ x > 1) ↔ (2*x^2 - 3*x + 1 > 0) :=
by
  intros a b h cond_a cond_b x
  sorry

end quadratic_inequality_solution_set_l67_67112


namespace taxi_ride_cost_l67_67910

noncomputable def fixed_cost : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def distance_traveled : ℝ := 8

theorem taxi_ride_cost :
  fixed_cost + (cost_per_mile * distance_traveled) = 4.40 := by
  sorry

end taxi_ride_cost_l67_67910


namespace vector_magnitude_problem_l67_67894

open Real

noncomputable def magnitude (x : ℝ × ℝ) : ℝ := sqrt (x.1 ^ 2 + x.2 ^ 2)

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h_a : a = (1, 3))
  (h_perp : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) :
  magnitude b = sqrt 10 := 
sorry

end vector_magnitude_problem_l67_67894


namespace production_rate_l67_67281

theorem production_rate (minutes: ℕ) (machines1 machines2 paperclips1 paperclips2 : ℕ)
  (h1 : minutes = 1) (h2 : machines1 = 8) (h3 : machines2 = 18) (h4 : paperclips1 = 560) 
  (h5 : paperclips2 = (paperclips1 / machines1) * machines2 * minutes) : 
  paperclips2 = 7560 :=
by
  sorry

end production_rate_l67_67281


namespace zhao_estimate_larger_l67_67977

theorem zhao_estimate_larger (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2 * ε) > x - y :=
by
  sorry

end zhao_estimate_larger_l67_67977


namespace average_primes_30_50_l67_67035

/-- The theorem statement for proving the average of all prime numbers between 30 and 50 is 39.8 -/
theorem average_primes_30_50 : (31 + 37 + 41 + 43 + 47) / 5 = 39.8 :=
  by
  sorry

end average_primes_30_50_l67_67035


namespace combined_pre_tax_and_pre_tip_cost_l67_67188

theorem combined_pre_tax_and_pre_tip_cost (x y : ℝ) 
  (hx : 1.28 * x = 35.20) 
  (hy : 1.19 * y = 22.00) : 
  x + y = 46 := 
by
  sorry

end combined_pre_tax_and_pre_tip_cost_l67_67188


namespace dan_bought_18_stickers_l67_67126

variable (S D : ℕ)

-- Given conditions
def stickers_initially_same : Prop := S = S -- Cindy and Dan have the same number of stickers initially
def cindy_used_15_stickers : Prop := true -- Cindy used 15 of her stickers
def dan_bought_D_stickers : Prop := true -- Dan bought D stickers
def dan_has_33_more_stickers_than_cindy : Prop := (S + D) = (S - 15 + 33)

-- Question: Prove that the number of stickers Dan bought is 18
theorem dan_bought_18_stickers (h1 : stickers_initially_same S)
                               (h2 : cindy_used_15_stickers)
                               (h3 : dan_bought_D_stickers)
                               (h4 : dan_has_33_more_stickers_than_cindy S D) : D = 18 :=
sorry

end dan_bought_18_stickers_l67_67126


namespace number_of_dogs_l67_67250

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l67_67250


namespace value_of_x_minus_y_l67_67962

theorem value_of_x_minus_y (x y : ℚ) 
    (h₁ : 3 * x - 5 * y = 5) 
    (h₂ : x / (x + y) = 5 / 7) : x - y = 3 := by
  sorry

end value_of_x_minus_y_l67_67962


namespace find_a_from_quadratic_inequality_l67_67578

theorem find_a_from_quadratic_inequality :
  ∀ (a : ℝ), (∀ x : ℝ, (x > - (1 / 2)) ∧ (x < 1 / 3) → a * x^2 - 2 * x + 2 > 0) → a = -12 :=
by
  intros a h
  have h1 := h (-1 / 2)
  have h2 := h (1 / 3)
  sorry

end find_a_from_quadratic_inequality_l67_67578


namespace total_gain_loss_is_correct_l67_67121

noncomputable def total_gain_loss_percentage 
    (cost1 cost2 cost3 : ℝ) 
    (gain1 gain2 gain3 : ℝ) : ℝ :=
  let total_cost := cost1 + cost2 + cost3
  let gain_amount1 := cost1 * gain1
  let loss_amount2 := cost2 * gain2
  let gain_amount3 := cost3 * gain3
  let net_gain_loss := (gain_amount1 + gain_amount3) - loss_amount2
  (net_gain_loss / total_cost) * 100

theorem total_gain_loss_is_correct :
  total_gain_loss_percentage 
    675958 995320 837492 0.11 (-0.11) 0.15 = 3.608 := 
sorry

end total_gain_loss_is_correct_l67_67121


namespace melanie_picked_plums_l67_67834

variable (picked_plums : ℕ)
variable (given_plums : ℕ := 3)
variable (total_plums : ℕ := 10)

theorem melanie_picked_plums :
  picked_plums + given_plums = total_plums → picked_plums = 7 := by
  sorry

end melanie_picked_plums_l67_67834


namespace joy_sixth_time_is_87_seconds_l67_67680

def sixth_time (times : List ℝ) (new_median : ℝ) : ℝ :=
  let sorted_times := times |>.insertNth 2 (2 * new_median - times.nthLe 2 sorry)
  2 * new_median - times.nthLe 2 sorry

theorem joy_sixth_time_is_87_seconds (times : List ℝ) (new_median : ℝ) :
  times = [82, 85, 93, 95, 99] → new_median = 90 →
  sixth_time times new_median = 87 :=
by
  intros h_times h_median
  rw [h_times]
  rw [h_median]
  sorry

end joy_sixth_time_is_87_seconds_l67_67680


namespace probability_neither_A_nor_B_l67_67004

noncomputable def pA : ℝ := 0.25
noncomputable def pB : ℝ := 0.35
noncomputable def pA_and_B : ℝ := 0.15

theorem probability_neither_A_nor_B :
  1 - (pA + pB - pA_and_B) = 0.55 :=
by
  simp [pA, pB, pA_and_B]
  norm_num
  sorry

end probability_neither_A_nor_B_l67_67004


namespace like_apple_orange_mango_l67_67227

theorem like_apple_orange_mango (A B C: ℕ) 
  (h1: A = 40) 
  (h2: B = 7) 
  (h3: C = 10) 
  (total: ℕ) 
  (h_total: total = 47) 
: ∃ x: ℕ, 40 + (10 - x) + x = 47 ∧ x = 3 := 
by 
  sorry

end like_apple_orange_mango_l67_67227


namespace resort_total_cost_l67_67480

noncomputable def first_cabin_cost (P : ℝ) := P
noncomputable def second_cabin_cost (P : ℝ) := (1/2) * P
noncomputable def third_cabin_cost (P : ℝ) := (1/6) * P
noncomputable def land_cost (P : ℝ) := 4 * P
noncomputable def pool_cost (P : ℝ) := P

theorem resort_total_cost (P : ℝ) (h : P = 22500) :
  first_cabin_cost P + pool_cost P + second_cabin_cost P + third_cabin_cost P + land_cost P = 150000 :=
by
  sorry

end resort_total_cost_l67_67480


namespace count_1000_pointed_stars_l67_67148

/--
A regular n-pointed star is defined by:
1. The points P_1, P_2, ..., P_n are coplanar and no three of them are collinear.
2. Each of the n line segments intersects at least one other segment at a point other than an endpoint.
3. All of the angles at P_1, P_2, ..., P_n are congruent.
4. All of the n line segments P_2P_3, ..., P_nP_1 are congruent.
5. The path P_1P_2, P_2P_3, ..., P_nP_1 turns counterclockwise at an angle of less than 180 degrees at each vertex.

There are no regular 3-pointed, 4-pointed, or 6-pointed stars.
All regular 5-pointed stars are similar.
There are two non-similar regular 7-pointed stars.

Prove that the number of non-similar regular 1000-pointed stars is 199.
-/
theorem count_1000_pointed_stars : ∀ (n : ℕ), n = 1000 → 
  -- Points P_1, P_2, ..., P_1000 are coplanar, no three are collinear.
  -- Each of the 1000 segments intersects at least one other segment not at an endpoint.
  -- Angles at P_1, P_2, ..., P_1000 are congruent.
  -- Line segments P_2P_3, ..., P_1000P_1 are congruent.
  -- Path P_1P_2, P_2P_3, ..., P_1000P_1 turns counterclockwise at < 180 degrees each.
  -- No 3-pointed, 4-pointed, or 6-pointed regular stars.
  -- All regular 5-pointed stars are similar.
  -- There are two non-similar regular 7-pointed stars.
  -- Proven: The number of non-similar regular 1000-pointed stars is 199.
  n = 1000 ∧ (∀ m : ℕ, 1 ≤ m ∧ m < 1000 → (gcd m 1000 = 1 → (m ≠ 1 ∧ m ≠ 999))) → 
    -- Because 1000 = 2^3 * 5^3 and we exclude 1 and 999.
    (2 * 5 * 2 * 5 * 2 * 5) / 2 - 1 - 1 / 2 = 199 :=
by
  -- Pseudo-proof steps for the problem.
  sorry

end count_1000_pointed_stars_l67_67148


namespace units_digit_of_product_l67_67670

/-
Problem: What is the units digit of the product of the first three even positive composite numbers?
Conditions: 
- The first three even positive composite numbers are 4, 6, and 8.
Proof: Prove that the units digit of their product is 2.
-/

def even_positive_composite_numbers := [4, 6, 8]
def product := List.foldl (· * ·) 1 even_positive_composite_numbers
def units_digit (n : Nat) := n % 10

theorem units_digit_of_product : units_digit product = 2 := by
  sorry

end units_digit_of_product_l67_67670


namespace solve_x_division_l67_67113

theorem solve_x_division :
  ∀ x : ℝ, (3 / x + 4 / x / (8 / x) = 1.5) → x = 3 := 
by
  intro x
  intro h
  sorry

end solve_x_division_l67_67113


namespace cube_surface_area_correct_l67_67394

def edge_length : ℝ := 11

def cube_surface_area (e : ℝ) : ℝ := 6 * e^2

theorem cube_surface_area_correct : cube_surface_area edge_length = 726 := by
  sorry

end cube_surface_area_correct_l67_67394


namespace intersecting_graphs_value_l67_67502

theorem intersecting_graphs_value (a b c d : ℝ) 
  (h1 : 5 = -|2 - a| + b) 
  (h2 : 3 = -|8 - a| + b) 
  (h3 : 5 = |2 - c| + d) 
  (h4 : 3 = |8 - c| + d) : 
  a + c = 10 :=
sorry

end intersecting_graphs_value_l67_67502


namespace savings_if_together_l67_67494

def price_per_window : ℕ := 150

def discount_offer (n : ℕ) : ℕ := n - n / 7

def cost (n : ℕ) : ℕ := price_per_window * discount_offer n

def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def separate_cost : ℕ := cost alice_windows + cost bob_windows

def total_windows : ℕ := alice_windows + bob_windows

def together_cost : ℕ := cost total_windows

def savings : ℕ := separate_cost - together_cost

theorem savings_if_together : savings = 150 := by
  sorry

end savings_if_together_l67_67494


namespace find_F_l67_67230

theorem find_F (F C : ℝ) (h1 : C = 4/7 * (F - 40)) (h2 : C = 28) : F = 89 := 
by
  sorry

end find_F_l67_67230


namespace minimize_y_l67_67654

variables (a b k : ℝ)

def y (x : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y : ∃ x : ℝ, y a b k x = y a b k ( (6 * a + 2 * b - k) / 8 ) :=
  sorry

end minimize_y_l67_67654


namespace find_m_l67_67975

theorem find_m (m : ℝ) (a b : ℝ × ℝ) (k : ℝ) (ha : a = (1, 1)) (hb : b = (m, 2)) 
  (h_parallel : 2 • a + b = k • a) : m = 2 :=
sorry

end find_m_l67_67975


namespace geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l67_67721

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ} (hq : 0 < q) (hq2 : q ≠ 1)

-- ① If $a_{1}=1$ and the common ratio is $\frac{1}{2}$, then $S_{n} < 2$;
theorem geom_seq_sum_lt_two (h₁ : a 1 = 1) (hq_half : q = 1 / 2) (n : ℕ) : S n < 2 := sorry

-- ② The sequence $\{a_{n}^{2}\}$ must be a geometric sequence
theorem geom_seq_squared (h_geom : ∀ n, a (n + 1) = q * a n) : ∃ r : ℝ, ∀ n, a n ^ 2 = r ^ n := sorry

-- ④ For any positive integer $n$, $a{}_{n}^{2}+a{}_{n+2}^{2}\geqslant 2a{}_{n+1}^{2}$
theorem geom_seq_square_inequality (h_geom : ∀ n, a (n + 1) = q * a n) (n : ℕ) (hn : 0 < n) : 
  a n ^ 2 + a (n + 2) ^ 2 ≥ 2 * a (n + 1) ^ 2 := sorry

end geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l67_67721


namespace smoothie_cost_l67_67995

-- Definitions of costs and amounts paid.
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def amount_paid : ℕ := 20
def change_received : ℕ := 11

-- Define the total cost of the order and the known costs.
def total_order_cost : ℕ := amount_paid - change_received
def known_costs : ℕ := hamburger_cost + onion_rings_cost

-- State the problem: the cost of the smoothie.
theorem smoothie_cost : total_order_cost - known_costs = 3 :=
by 
  sorry

end smoothie_cost_l67_67995


namespace reverse_digits_multiplication_l67_67317

theorem reverse_digits_multiplication (a b : ℕ) (h₁ : a < 10) (h₂ : b < 10) : 
  (10 * a + b) * (10 * b + a) = 101 * a * b + 10 * (a^2 + b^2) :=
by 
  sorry

end reverse_digits_multiplication_l67_67317


namespace john_safety_percentage_l67_67313

def bench_max_weight : ℕ := 1000
def john_weight : ℕ := 250
def weight_on_bar : ℕ := 550
def total_weight := john_weight + weight_on_bar
def percentage_of_max_weight := (total_weight * 100) / bench_max_weight
def percentage_under_max_weight := 100 - percentage_of_max_weight

theorem john_safety_percentage : percentage_under_max_weight = 20 := by
  sorry

end john_safety_percentage_l67_67313


namespace smallest_difference_of_sides_l67_67143

/-- Triangle PQR has a perimeter of 2021 units. The sides have lengths that are integer values with PQ < QR ≤ PR. 
The smallest possible value of QR - PQ is 1. -/
theorem smallest_difference_of_sides :
  ∃ (PQ QR PR : ℕ), PQ < QR ∧ QR ≤ PR ∧ PQ + QR + PR = 2021 ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ QR + PR > PQ ∧ QR - PQ = 1 :=
sorry

end smallest_difference_of_sides_l67_67143


namespace natasha_destination_distance_l67_67799

theorem natasha_destination_distance
  (over_speed : ℕ)
  (time : ℕ)
  (speed_limit : ℕ)
  (actual_speed : ℕ)
  (distance : ℕ) :
  (over_speed = 10) →
  (time = 1) →
  (speed_limit = 50) →
  (actual_speed = speed_limit + over_speed) →
  (distance = actual_speed * time) →
  (distance = 60) :=
by
  sorry

end natasha_destination_distance_l67_67799


namespace pasture_feeding_l67_67899

-- The definitions corresponding to the given conditions
def portion_per_cow_per_day := 1

def food_needed (cows : ℕ) (days : ℕ) : ℕ := cows * days

def growth_rate (food10for20 : ℕ) (food15for10 : ℕ) (days10_20 : ℕ) : ℕ :=
  (food10for20 - food15for10) / days10_20

def food_growth_rate := growth_rate (food_needed 10 20) (food_needed 15 10) 10

def new_grass_feed_cows_per_day := food_growth_rate / portion_per_cow_per_day

def original_grass := (food_needed 10 20) - (food_growth_rate * 20)

def days_to_feed_30_cows := original_grass / (30 - new_grass_feed_cows_per_day)

-- The statement we want to prove
theorem pasture_feeding :
  new_grass_feed_cows_per_day = 5 ∧ days_to_feed_30_cows = 4 := by
  sorry

end pasture_feeding_l67_67899


namespace Person3IsTriussian_l67_67282

def IsTriussian (person : ℕ) : Prop := if person = 3 then True else False

def Person1Statement : Prop := ∀ i j k : ℕ, i = 1 → j = 2 → k = 3 → (IsTriussian i = (IsTriussian j ∧ IsTriussian k) ∨ (¬IsTriussian j ∧ ¬IsTriussian k))

def Person2Statement : Prop := ∀ i j : ℕ, i = 2 → j = 3 → (IsTriussian j = False)

def Person3Statement : Prop := ∀ i j : ℕ, i = 3 → j = 1 → (IsTriussian j = False)

theorem Person3IsTriussian : (Person1Statement ∧ Person2Statement ∧ Person3Statement) → IsTriussian 3 :=
by 
  sorry

end Person3IsTriussian_l67_67282


namespace quadratic_inequality_solution_l67_67815

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end quadratic_inequality_solution_l67_67815


namespace find_T_b_minus_T_neg_b_l67_67738

noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

theorem find_T_b_minus_T_neg_b (b : ℝ) (h1 : -1 < b ∧ b < 1) (h2 : T b * T (-b) = 3240) (h3 : 1 - b^2 = 100 / 810) :
  T b - T (-b) = 324 * b :=
by
  sorry

end find_T_b_minus_T_neg_b_l67_67738


namespace new_game_cost_l67_67485

theorem new_game_cost (G : ℕ) (h_initial_money : 83 = G + 9 * 4) : G = 47 := by
  sorry

end new_game_cost_l67_67485


namespace find_x_plus_y_l67_67672

theorem find_x_plus_y (x y : Real) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 2009 + Real.pi / 2 :=
by
  sorry

end find_x_plus_y_l67_67672


namespace product_of_real_roots_l67_67323

theorem product_of_real_roots (x1 x2 : ℝ) (h1 : x1^2 - 6 * x1 + 8 = 0) (h2 : x2^2 - 6 * x2 + 8 = 0) :
  x1 * x2 = 8 := 
sorry

end product_of_real_roots_l67_67323


namespace smallest_three_digit_multiple_of_17_l67_67009

theorem smallest_three_digit_multiple_of_17 : ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ ∃ (k : ℕ), n = 17 * k ∧ n = 102 :=
sorry

end smallest_three_digit_multiple_of_17_l67_67009


namespace unique_sequence_l67_67465

theorem unique_sequence (a : ℕ → ℕ) (h_distinct: ∀ m n, a m = a n → m = n)
    (h_divisible: ∀ n, a n % a (a n) = 0) : ∀ n, a n = n :=
by
  -- proof goes here
  sorry

end unique_sequence_l67_67465


namespace intersection_of_P_and_Q_l67_67789
-- Import the entire math library

-- Define the conditions for sets P and Q
def P := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | (x - 1)^2 ≤ 4}

-- Define the theorem to prove that P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3}
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3} :=
by
  -- Placeholder for the proof
  sorry

end intersection_of_P_and_Q_l67_67789


namespace square_side_length_l67_67235

variable (s : ℕ)
variable (P A : ℕ)

theorem square_side_length (h1 : P = 52) (h2 : A = 169) (h3 : P = 4 * s) (h4 : A = s * s) : s = 13 :=
sorry

end square_side_length_l67_67235


namespace all_numbers_equal_l67_67964

theorem all_numbers_equal
  (n : ℕ)
  (h n_eq_20 : n = 20)
  (a : ℕ → ℝ)
  (h_avg : ∀ i : ℕ, i < n → a i = (a ((i+n-1) % n) + a ((i+1) % n)) / 2) :
  ∀ i j : ℕ, i < n → j < n → a i = a j :=
by {
  -- Proof steps go here.
  sorry
}

end all_numbers_equal_l67_67964


namespace range_of_m_l67_67431

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + m > 0) ↔ 0 < m ∧ m < 4 :=
by sorry

end range_of_m_l67_67431


namespace box_dimensions_l67_67730

theorem box_dimensions (x y z : ℝ) (h1 : x * y * z = 160) 
  (h2 : y * z = 80) (h3 : x * z = 40) (h4 : x * y = 32) : 
  x = 4 ∧ y = 8 ∧ z = 10 :=
by
  -- Placeholder for the actual proof steps
  sorry

end box_dimensions_l67_67730


namespace survivor_probability_l67_67907

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem survivor_probability :
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  probability = 20 / 95 :=
by
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  have : probability = 20 / 95 := sorry
  exact this

end survivor_probability_l67_67907


namespace trigonometric_identity_l67_67395

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (π / 4 + θ) = 3) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
sorry

end trigonometric_identity_l67_67395


namespace point_A_inside_circle_O_l67_67400

-- Definitions based on conditions in the problem
def radius := 5 -- in cm
def distance_to_center := 4 -- in cm

-- The theorem to be proven
theorem point_A_inside_circle_O (r d : ℝ) (hr : r = 5) (hd : d = 4) (h : r > d) : true :=
by {
  sorry
}

end point_A_inside_circle_O_l67_67400


namespace trapezoid_PQRS_PQ_squared_l67_67416

theorem trapezoid_PQRS_PQ_squared
  (PR PS PQ : ℝ)
  (cond1 : PR = 13)
  (cond2 : PS = 17)
  (h : PQ^2 + PR^2 = PS^2) :
  PQ^2 = 120 :=
by
  rw [cond1, cond2] at h
  sorry

end trapezoid_PQRS_PQ_squared_l67_67416


namespace tangent_line_to_curve_at_Mpi_l67_67538

noncomputable def tangent_line_eq_at_point (x : ℝ) (y : ℝ) : Prop :=
  y = (Real.sin x) / x

theorem tangent_line_to_curve_at_Mpi :
  (∀ x y, tangent_line_eq_at_point x y →
    (∃ (m : ℝ), m = -1 / π) →
    (∀ x1 y1 (hx : x1 = π) (hy : y1 = 0), x + π * y - π = 0)) :=
by
  sorry

end tangent_line_to_curve_at_Mpi_l67_67538


namespace factorial_last_nonzero_digit_non_periodic_l67_67884

def last_nonzero_digit (n : ℕ) : ℕ :=
  -- function to compute last nonzero digit of n!
  sorry

def sequence_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a n = a (n + T)

theorem factorial_last_nonzero_digit_non_periodic : ¬ ∃ T, sequence_periodic last_nonzero_digit T :=
  sorry

end factorial_last_nonzero_digit_non_periodic_l67_67884


namespace equation_relationship_linear_l67_67923

theorem equation_relationship_linear 
  (x y : ℕ)
  (h1 : (x, y) = (0, 200) ∨ (x, y) = (1, 160) ∨ (x, y) = (2, 120) ∨ (x, y) = (3, 80) ∨ (x, y) = (4, 40)) :
  y = 200 - 40 * x :=
  sorry

end equation_relationship_linear_l67_67923


namespace shoe_store_sale_l67_67867

theorem shoe_store_sale (total_sneakers : ℕ) (total_sandals : ℕ) (total_shoes : ℕ) (total_boots : ℕ) 
  (h1 : total_sneakers = 2) 
  (h2 : total_sandals = 4) 
  (h3 : total_shoes = 17) 
  (h4 : total_boots = total_shoes - (total_sneakers + total_sandals)) : 
  total_boots = 11 :=
by
  rw [h1, h2, h3] at h4
  exact h4
-- sorry

end shoe_store_sale_l67_67867


namespace parabola_directrix_l67_67216

theorem parabola_directrix (x : ℝ) : 
  (6 * x^2 + 5 = y) → (y = 6 * x^2 + 5) → (y = 6 * 0^2 + 5) → (y = (119 : ℝ) / 24) := 
sorry

end parabola_directrix_l67_67216


namespace largest_possible_perimeter_l67_67297

theorem largest_possible_perimeter :
  ∃ (l w : ℕ), 8 * l + 8 * w = l * w - 1 ∧ 2 * l + 2 * w = 164 :=
sorry

end largest_possible_perimeter_l67_67297


namespace ratio_first_term_l67_67887

theorem ratio_first_term (x : ℝ) (h1 : 60 / 100 = x / 25) : x = 15 := 
sorry

end ratio_first_term_l67_67887


namespace nuts_in_mason_car_l67_67722

-- Define the constants for the rates of stockpiling
def busy_squirrel_rate := 30 -- nuts per day
def sleepy_squirrel_rate := 20 -- nuts per day
def days := 40 -- number of days
def num_busy_squirrels := 2 -- number of busy squirrels
def num_sleepy_squirrels := 1 -- number of sleepy squirrels

-- Define the total number of nuts
def total_nuts_in_mason_car : ℕ :=
  (num_busy_squirrels * busy_squirrel_rate * days) +
  (num_sleepy_squirrels * sleepy_squirrel_rate * days)

theorem nuts_in_mason_car :
  total_nuts_in_mason_car = 3200 :=
sorry

end nuts_in_mason_car_l67_67722


namespace precision_of_21_658_billion_is_hundred_million_l67_67593

theorem precision_of_21_658_billion_is_hundred_million :
  (21.658 : ℝ) * 10^9 % (10^8) = 0 :=
by
  sorry

end precision_of_21_658_billion_is_hundred_million_l67_67593


namespace find_a_for_even_function_l67_67918

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l67_67918


namespace installation_cost_l67_67607

-- Definitions
variables (LP : ℝ) (P : ℝ := 16500) (D : ℝ := 0.2) (T : ℝ := 125) (SP : ℝ := 23100) (I : ℝ)

-- Conditions
def purchase_price := P = (1 - D) * LP
def selling_price := SP = 1.1 * LP
def total_cost := P + T + I = SP

-- Proof Statement
theorem installation_cost : I = 6350 :=
  by
    -- sorry is used to skip the proof
    sorry

end installation_cost_l67_67607


namespace problem_statement_l67_67110

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 1988 a b α β = 3) : f 2013 a b α β = 5 :=
by 
  sorry

end problem_statement_l67_67110


namespace gcd_of_three_numbers_l67_67916

theorem gcd_of_three_numbers : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end gcd_of_three_numbers_l67_67916


namespace trigonometric_identity_simplification_l67_67445

open Real

theorem trigonometric_identity_simplification (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  (sqrt (1 - 2 * sin (3 * π - θ) * sin (π / 2 + θ)) = cos θ - sin θ) :=
sorry

end trigonometric_identity_simplification_l67_67445


namespace simplify_expression_l67_67049

variable (a b : ℝ)

theorem simplify_expression : a + (3 * a - 3 * b) - (a - 2 * b) = 3 * a - b := 
by 
  sorry

end simplify_expression_l67_67049


namespace min_accommodation_cost_l67_67529

theorem min_accommodation_cost :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 3 * x + 2 * y + z = 50 ∧ 100 * 3 * x + 150 * 2 * y + 200 * z = 5500 :=
by
  sorry

end min_accommodation_cost_l67_67529


namespace eq_satisfied_for_all_y_l67_67359

theorem eq_satisfied_for_all_y (x : ℝ) : 
  (∀ y: ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by
  sorry

end eq_satisfied_for_all_y_l67_67359


namespace find_exponent_l67_67601

theorem find_exponent (y : ℕ) (b : ℕ) (h_b : b = 2)
  (h : 1 / 8 * 2 ^ 40 = b ^ y) : y = 37 :=
by
  sorry

end find_exponent_l67_67601


namespace find_a_and_b_l67_67174

-- Given conditions
def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_and_b (a b : ℝ) :
  (∀ x, ∀ y, tangent_line x y → y = b ∧ x = 0) ∧
  (∀ x, ∀ y, y = curve x a b) →
  a = 1 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l67_67174


namespace four_consecutive_integers_divisible_by_24_l67_67353

noncomputable def product_of_consecutive_integers (n : ℤ) : ℤ :=
  n * (n + 1) * (n + 2) * (n + 3)

theorem four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ product_of_consecutive_integers n :=
by
  sorry

end four_consecutive_integers_divisible_by_24_l67_67353


namespace swap_tens_units_digits_l67_67552

theorem swap_tens_units_digits (x a b : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : a = x / 10) (h4 : b = x % 10) :
  10 * b + a = (x % 10) * 10 + (x / 10) :=
by
  sorry

end swap_tens_units_digits_l67_67552


namespace parallel_lines_condition_l67_67544

theorem parallel_lines_condition (m n : ℝ) :
  (∃x y, (m * x + y - n = 0) ∧ (x + m * y + 1 = 0)) →
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1) :=
by
  sorry

end parallel_lines_condition_l67_67544


namespace range_of_a_l67_67904

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
by sorry

end range_of_a_l67_67904


namespace proof_5x_plus_4_l67_67702

variable (x : ℝ)

-- Given condition
def condition := 5 * x - 8 = 15 * x + 12

-- Required proof
theorem proof_5x_plus_4 (h : condition x) : 5 * (x + 4) = 10 :=
by {
  sorry
}

end proof_5x_plus_4_l67_67702


namespace length_width_difference_l67_67688

theorem length_width_difference
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 768) :
  l - w = 24 * Real.sqrt 3 :=
by
  sorry

end length_width_difference_l67_67688


namespace range_of_a_l67_67070

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) : (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) ↔ (a < 0 ∨ (1 / 4 < a ∧ a < 4)) :=
by
  sorry

end range_of_a_l67_67070


namespace eq1_eq2_eq3_eq4_l67_67690

theorem eq1 (x : ℚ) : 3 * x^2 - 32 * x - 48 = 0 ↔ (x = 12 ∨ x = -4/3) := sorry

theorem eq2 (x : ℚ) : 4 * x^2 + x - 3 = 0 ↔ (x = 3/4 ∨ x = -1) := sorry

theorem eq3 (x : ℚ) : (3 * x + 1)^2 - 4 = 0 ↔ (x = 1/3 ∨ x = -1) := sorry

theorem eq4 (x : ℚ) : 9 * (x - 2)^2 = 4 * (x + 1)^2 ↔ (x = 8 ∨ x = 4/5) := sorry

end eq1_eq2_eq3_eq4_l67_67690


namespace total_Pokemon_cards_l67_67377

def j : Nat := 6
def o : Nat := j + 2
def r : Nat := 3 * o
def t : Nat := j + o + r

theorem total_Pokemon_cards : t = 38 := by 
  sorry

end total_Pokemon_cards_l67_67377


namespace repeatable_transformation_l67_67304

theorem repeatable_transformation (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
  (2 * c > a + b) ∧ (2 * a > b + c) ∧ (2 * b > c + a) := 
sorry

end repeatable_transformation_l67_67304


namespace integer_pairs_m_n_l67_67344

theorem integer_pairs_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (cond1 : ∃ k1 : ℕ, k1 * m = 3 * n ^ 2)
  (cond2 : ∃ k2 : ℕ, k2 ^ 2 = n ^ 2 + m) :
  ∃ a : ℕ, m = 3 * a ^ 2 ∧ n = a :=
by
  sorry

end integer_pairs_m_n_l67_67344


namespace max_value_of_expression_l67_67922

theorem max_value_of_expression (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 :=
sorry

end max_value_of_expression_l67_67922


namespace range_exp3_eq_l67_67050

noncomputable def exp3 (x : ℝ) : ℝ := 3^x

theorem range_exp3_eq (x : ℝ) : Set.range (exp3) = Set.Ioi 0 :=
sorry

end range_exp3_eq_l67_67050


namespace harold_august_tips_fraction_l67_67961

noncomputable def tips_fraction : ℚ :=
  let A : ℚ := sorry -- average monthly tips for March to July and September
  let august_tips := 6 * A -- Tips for August
  let total_tips := 6 * A + 6 * A -- Total tips for all months worked
  august_tips / total_tips

theorem harold_august_tips_fraction :
  tips_fraction = 1 / 2 :=
by
  sorry

end harold_august_tips_fraction_l67_67961


namespace sandy_correct_sums_l67_67993

theorem sandy_correct_sums :
  ∃ c i : ℤ,
  c + i = 40 ∧
  4 * c - 3 * i = 72 ∧
  c = 27 :=
by 
  sorry

end sandy_correct_sums_l67_67993


namespace original_price_of_trouser_l67_67786

-- Define conditions
def sale_price : ℝ := 20
def discount : ℝ := 0.80

-- Define what the proof aims to show
theorem original_price_of_trouser (P : ℝ) (h : sale_price = P * (1 - discount)) : P = 100 :=
sorry

end original_price_of_trouser_l67_67786


namespace exists_int_less_than_sqrt_twenty_three_l67_67283

theorem exists_int_less_than_sqrt_twenty_three : ∃ n : ℤ, n < Real.sqrt 23 := 
  sorry

end exists_int_less_than_sqrt_twenty_three_l67_67283


namespace simplify_expression_l67_67249

theorem simplify_expression (x : ℝ) : 3 * x + 5 * x ^ 2 + 2 - (9 - 4 * x - 5 * x ^ 2) = 10 * x ^ 2 + 7 * x - 7 :=
by
  sorry

end simplify_expression_l67_67249


namespace find_BD_in_triangle_l67_67588

theorem find_BD_in_triangle (A B C D : Type)
  (distance_AC : Float) (distance_BC : Float)
  (distance_AD : Float) (distance_CD : Float)
  (hAC : distance_AC = 10)
  (hBC : distance_BC = 10)
  (hAD : distance_AD = 12)
  (hCD : distance_CD = 5) :
  ∃ (BD : Float), BD = 6.85435 :=
by 
  sorry

end find_BD_in_triangle_l67_67588


namespace arcsin_add_arccos_eq_pi_div_two_l67_67073

open Real

theorem arcsin_add_arccos_eq_pi_div_two (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  arcsin x + arccos x = (π / 2) :=
sorry

end arcsin_add_arccos_eq_pi_div_two_l67_67073


namespace Dan_gave_Sara_limes_l67_67968

theorem Dan_gave_Sara_limes : 
  ∀ (original_limes now_limes given_limes : ℕ),
  original_limes = 9 →
  now_limes = 5 →
  given_limes = original_limes - now_limes →
  given_limes = 4 :=
by
  intros original_limes now_limes given_limes h1 h2 h3
  sorry

end Dan_gave_Sara_limes_l67_67968


namespace compute_result_l67_67612

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end compute_result_l67_67612


namespace avg_last_four_is_63_75_l67_67001

noncomputable def average_of_list (l : List ℝ) : ℝ :=
  l.sum / l.length

variable (l : List ℝ)
variable (h_lenl : l.length = 7)
variable (h_avg7 : average_of_list l = 60)
variable (h_l3 : List ℝ := l.take 3)
variable (h_l4 : List ℝ := l.drop 3)
variable (h_avg3 : average_of_list h_l3 = 55)

theorem avg_last_four_is_63_75 : average_of_list h_l4 = 63.75 :=
by
  sorry

end avg_last_four_is_63_75_l67_67001


namespace arithmetic_sequence_sum_l67_67418

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h_arithmetic : ∀ n, a (n+1) = a n + d)
  (h_pos_diff : d > 0)
  (h_sum_3 : a 0 + a 1 + a 2 = 15)
  (h_prod_3 : a 0 * a 1 * a 2 = 80) :
  a 10 + a 11 + a 12 = 105 :=
sorry

end arithmetic_sequence_sum_l67_67418


namespace no_solution_inequalities_l67_67915

theorem no_solution_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ (x > 3 ∧ x < a)) ↔ (a ≤ 3) :=
by
  sorry

end no_solution_inequalities_l67_67915


namespace weight_of_sparrow_l67_67293

variable (a b : ℝ)

-- Define the conditions as Lean statements
-- 1. Six sparrows and seven swallows are balanced
def balanced_initial : Prop :=
  6 * b = 7 * a

-- 2. Sparrows are heavier than swallows
def sparrows_heavier : Prop :=
  b > a

-- 3. If one sparrow and one swallow are exchanged, the balance is maintained
def balanced_after_exchange : Prop :=
  5 * b + a = 6 * a + b

-- The theorem to prove the weight of one sparrow in terms of the weight of one swallow
theorem weight_of_sparrow (h1 : balanced_initial a b) (h2 : sparrows_heavier a b) (h3 : balanced_after_exchange a b) : 
  b = (5 / 4) * a :=
sorry

end weight_of_sparrow_l67_67293


namespace solve_abs_inequality_l67_67509

theorem solve_abs_inequality (x : ℝ) : 
  (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := 
by sorry

end solve_abs_inequality_l67_67509


namespace watch_current_price_l67_67413

-- Definitions based on conditions
def original_price : ℝ := 15
def first_reduction_rate : ℝ := 0.25
def second_reduction_rate : ℝ := 0.40

-- The price after the first reduction
def first_reduced_price : ℝ := original_price * (1 - first_reduction_rate)

-- The price after the second reduction
def final_price : ℝ := first_reduced_price * (1 - second_reduction_rate)

-- The theorem that needs to be proved
theorem watch_current_price : final_price = 6.75 :=
by
  -- Proof goes here
  sorry

end watch_current_price_l67_67413


namespace find_a_b_l67_67881

noncomputable def f (x : ℝ) (a b : ℝ) := x^3 + a * x + b

theorem find_a_b 
  (a b : ℝ) 
  (h_tangent : ∀ x y, y = 2 * x - 5 → y = f 1 a b - 3) 
  : a = -1 ∧ b = -3 :=
by 
{
  sorry
}

end find_a_b_l67_67881


namespace find_solution_set_l67_67274

-- Define the problem
def absolute_value_equation_solution_set (x : ℝ) : Prop :=
  |x - 2| + |2 * x - 3| = |3 * x - 5|

-- Define the expected solution set
def solution_set (x : ℝ) : Prop :=
  x ≤ 3 / 2 ∨ 2 ≤ x

-- The proof problem statement
theorem find_solution_set :
  ∀ x : ℝ, absolute_value_equation_solution_set x ↔ solution_set x :=
sorry -- No proof required, so we use 'sorry' to skip the proof

end find_solution_set_l67_67274


namespace bananas_more_than_pears_l67_67428

theorem bananas_more_than_pears (A P B : ℕ) (h1 : P = A + 2) (h2 : A + P + B = 19) (h3 : B = 9) : B - P = 3 :=
  by
  sorry

end bananas_more_than_pears_l67_67428


namespace evan_runs_200_more_feet_l67_67039

def street_width : ℕ := 25
def block_side : ℕ := 500

def emily_path : ℕ := 4 * block_side
def evan_path : ℕ := 4 * (block_side + 2 * street_width)

theorem evan_runs_200_more_feet : evan_path - emily_path = 200 := by
  sorry

end evan_runs_200_more_feet_l67_67039


namespace problem_1_problem_2_l67_67963

universe u

/-- Assume the universal set U is the set of real numbers -/
def U : Set ℝ := Set.univ

/-- Define set A -/
def A : Set ℝ := {x : ℝ | x ≥ 1}

/-- Define set B -/
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

/-- Prove the intersection of A and B -/
theorem problem_1 : (A ∩ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

/-- Prove the complement of the union of A and B -/
theorem problem_2 : (U \ (A ∪ B)) = {x : ℝ | x < -1} :=
sorry

end problem_1_problem_2_l67_67963


namespace savings_percentage_l67_67114

variable {I S : ℝ}
variable (h1 : 1.30 * I - 2 * S + I - S = 2 * (I - S))

theorem savings_percentage (h : 1.30 * I - 2 * S + I - S = 2 * (I - S)) : S = 0.30 * I :=
  by
    sorry

end savings_percentage_l67_67114


namespace track_length_l67_67334

variable {x : ℕ}

-- Conditions
def runs_distance_jacob (x : ℕ) := 120
def runs_distance_liz (x : ℕ) := (x / 2 - 120)

def runs_second_meeting_jacob (x : ℕ) := x + 120 -- Jacob's total distance by second meeting
def runs_second_meeting_liz (x : ℕ) := (x / 2 + 60) -- Liz's total distance by second meeting

-- The relationship is simplified into the final correct answer
theorem track_length (h1 : 120 / (x / 2 - 120) = (x / 2 + 60) / 180) :
  x = 340 := 
sorry

end track_length_l67_67334


namespace find_a_l67_67096

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 2

theorem find_a (a : ℝ) (h : (3 * a * (-1 : ℝ)^2) = 3) : a = 1 :=
by
  sorry

end find_a_l67_67096


namespace intervals_of_monotonic_increase_max_area_acute_triangle_l67_67490

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (sin x, (sqrt 3 / 2) * (sin x - cos x))

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (cos x, sin x + cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2

-- Problem 1: Proving the intervals of monotonic increase for the function f(x)
theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  ∀ x₁ x₂ : ℝ, (k * π - π / 12 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ k * π + 5 * π / 12) → f x₁ ≤ f x₂ :=
sorry

-- Problem 2: Proving the maximum area of triangle ABC
theorem max_area_acute_triangle (A : ℝ) (a b c : ℝ) :
  (f A = 1 / 2) → (a = sqrt 2) →
  ∀ S : ℝ, S ≤ (1 + sqrt 2) / 2 :=
sorry

end intervals_of_monotonic_increase_max_area_acute_triangle_l67_67490


namespace fifth_pyTriple_is_correct_l67_67153

-- Definitions based on conditions from part (a)
def pyTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := b + 1
  (a, b, c)

-- Question: Prove that the 5th Pythagorean triple is (11, 60, 61)
theorem fifth_pyTriple_is_correct : pyTriple 5 = (11, 60, 61) :=
  by
    -- Skip the proof
    sorry

end fifth_pyTriple_is_correct_l67_67153


namespace cube_faces_l67_67735

theorem cube_faces : ∀ (c : {s : Type | ∃ (x y z : ℝ), s = ({ (x0, y0, z0) : ℝ × ℝ × ℝ | x0 ≤ x ∧ y0 ≤ y ∧ z0 ≤ z}) }), 
  ∃ (f : ℕ), f = 6 :=
by 
  -- proof would be written here
  sorry

end cube_faces_l67_67735


namespace inequality_interval_l67_67386

def differentiable_on_R (f : ℝ → ℝ) : Prop := Differentiable ℝ f
def strictly_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

theorem inequality_interval (f : ℝ → ℝ)
  (h_diff : differentiable_on_R f)
  (h_cond : ∀ x : ℝ, f x > deriv f x)
  (h_init : f 0 = 1) :
  ∀ x : ℝ, (x > 0) ↔ (f x / Real.exp x < 1) := 
by
  sorry

end inequality_interval_l67_67386


namespace proof_problem_l67_67825

theorem proof_problem (k m : ℕ) (hk_pos : 0 < k) (hm_pos : 0 < m) (hkm : k > m)
  (hdiv : (k * m * (k ^ 2 - m ^ 2)) ∣ (k ^ 3 - m ^ 3)) :
  (k - m) ^ 3 > 3 * k * m :=
sorry

end proof_problem_l67_67825


namespace intersection_A_C_U_B_l67_67988

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_A_C_U_B_l67_67988


namespace Victor_bought_6_decks_l67_67447

theorem Victor_bought_6_decks (V : ℕ) (h1 : 2 * 8 + 8 * V = 64) : V = 6 := by
  sorry

end Victor_bought_6_decks_l67_67447


namespace cost_to_plant_flowers_l67_67155

theorem cost_to_plant_flowers :
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  cost_flowers + cost_clay_pot + cost_soil = 45 := 
by
  let cost_flowers := 9
  let cost_clay_pot := cost_flowers + 20
  let cost_soil := cost_flowers - 2
  show cost_flowers + cost_clay_pot + cost_soil = 45
  sorry

end cost_to_plant_flowers_l67_67155


namespace find_shirt_cost_l67_67693

def cost_each_shirt (x : ℝ) : Prop :=
  let total_purchase_price := x + 5 + 30 + 14
  let shipping_cost := if total_purchase_price > 50 then 0.2 * total_purchase_price else 5
  let total_bill := total_purchase_price + shipping_cost
  total_bill = 102

theorem find_shirt_cost (x : ℝ) (h : cost_each_shirt x) : x = 36 :=
sorry

end find_shirt_cost_l67_67693


namespace arithmetic_series_sum_l67_67677

theorem arithmetic_series_sum : 
  ∀ (a d a_n : ℤ), 
  a = -48 → d = 2 → a_n = 0 → 
  ∃ n S : ℤ, 
  a + (n - 1) * d = a_n ∧ 
  S = n * (a + a_n) / 2 ∧ 
  S = -600 :=
by
  intros a d a_n ha hd han
  have h₁ : a = -48 := ha
  have h₂ : d = 2 := hd
  have h₃ : a_n = 0 := han
  sorry

end arithmetic_series_sum_l67_67677


namespace perfect_square_trinomial_k_l67_67058

theorem perfect_square_trinomial_k (k : ℤ) : 
  (∀ x : ℝ, x^2 - k*x + 64 = (x + 8)^2 ∨ x^2 - k*x + 64 = (x - 8)^2) → 
  (k = 16 ∨ k = -16) :=
by
  sorry

end perfect_square_trinomial_k_l67_67058


namespace ages_of_father_and_daughter_l67_67099

variable (F D : ℕ)

-- Conditions
def condition1 : Prop := F = 4 * D
def condition2 : Prop := F + 20 = 2 * (D + 20)

-- Main statement
theorem ages_of_father_and_daughter (h1 : condition1 F D) (h2 : condition2 F D) : D = 10 ∧ F = 40 := by
  sorry

end ages_of_father_and_daughter_l67_67099


namespace max_distance_circle_ellipse_l67_67791

theorem max_distance_circle_ellipse:
  (∀ P Q : ℝ × ℝ, 
     (P.1^2 + (P.2 - 3)^2 = 1 / 4) → 
     (Q.1^2 + 4 * Q.2^2 = 4) → 
     ∃ Q_max : ℝ × ℝ, 
         Q_max = (0, -1) ∧ 
         (∀ P : ℝ × ℝ, P.1^2 + (P.2 - 3)^2 = 1 / 4 →
         |dist P Q_max| = 9 / 2)) := 
sorry

end max_distance_circle_ellipse_l67_67791


namespace math_problem_l67_67234
-- Import necessary modules

-- Define the condition as a hypothesis and state the theorem
theorem math_problem (x : ℝ) (h : 8 * x - 6 = 10) : 50 * (1 / x) + 150 = 175 :=
sorry

end math_problem_l67_67234


namespace gain_percent_l67_67109

variable (C S : ℝ)
variable (h : 65 * C = 50 * S)

theorem gain_percent (h : 65 * C = 50 * S) : (S - C) / C * 100 = 30 :=
by
  sorry

end gain_percent_l67_67109


namespace no_int_solutions_for_cubic_eqn_l67_67303

theorem no_int_solutions_for_cubic_eqn :
  ¬ ∃ (m n : ℤ), m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_int_solutions_for_cubic_eqn_l67_67303


namespace Ruth_math_class_percentage_l67_67640

theorem Ruth_math_class_percentage :
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  (hours_math_week / total_school_hours_week) * 100 = 25 := 
by 
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  -- skip the proof here
  sorry

end Ruth_math_class_percentage_l67_67640


namespace quotient_of_division_l67_67645

theorem quotient_of_division (Q : ℤ) (h1 : 172 = (17 * Q) + 2) : Q = 10 :=
sorry

end quotient_of_division_l67_67645


namespace eight_pow_91_gt_seven_pow_92_l67_67339

theorem eight_pow_91_gt_seven_pow_92 : 8^91 > 7^92 :=
  sorry

end eight_pow_91_gt_seven_pow_92_l67_67339


namespace complex_multiplication_l67_67969

theorem complex_multiplication :
  ∀ (i : ℂ), i^2 = -1 → (1 - i) * i = 1 + i :=
by
  sorry

end complex_multiplication_l67_67969


namespace best_model_is_model1_l67_67404

noncomputable def model_best_fitting (R1 R2 R3 R4 : ℝ) :=
  R1 = 0.975 ∧ R2 = 0.79 ∧ R3 = 0.55 ∧ R4 = 0.25

theorem best_model_is_model1 (R1 R2 R3 R4 : ℝ) (h : model_best_fitting R1 R2 R3 R4) :
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  cases h with
  | intro h1 h_rest =>
    cases h_rest with
    | intro h2 h_rest2 =>
      cases h_rest2 with
      | intro h3 h4 =>
        sorry

end best_model_is_model1_l67_67404


namespace sum_of_fourth_powers_l67_67636

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1 / 2 :=
by sorry

end sum_of_fourth_powers_l67_67636


namespace origin_inside_ellipse_iff_abs_k_range_l67_67784

theorem origin_inside_ellipse_iff_abs_k_range (k : ℝ) :
  (k^2 * 0^2 + 0^2 - 4 * k * 0 + 2 * k * 0 + k^2 - 1 < 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end origin_inside_ellipse_iff_abs_k_range_l67_67784


namespace required_hemispherical_containers_l67_67976

noncomputable def initial_volume : ℝ := 10940
noncomputable def initial_temperature : ℝ := 20
noncomputable def final_temperature : ℝ := 25
noncomputable def expansion_coefficient : ℝ := 0.002
noncomputable def container_volume : ℝ := 4
noncomputable def usable_capacity : ℝ := 0.8

noncomputable def volume_expansion : ℝ := initial_volume * (final_temperature - initial_temperature) * expansion_coefficient
noncomputable def final_volume : ℝ := initial_volume + volume_expansion
noncomputable def usable_volume_per_container : ℝ := container_volume * usable_capacity
noncomputable def number_of_containers_needed : ℝ := final_volume / usable_volume_per_container

theorem required_hemispherical_containers : ⌈number_of_containers_needed⌉ = 3453 :=
by 
  sorry

end required_hemispherical_containers_l67_67976


namespace max_participants_l67_67729

structure MeetingRoom where
  rows : ℕ
  cols : ℕ
  seating : ℕ → ℕ → Bool -- A function indicating if a seat (i, j) is occupied (true) or not (false)
  row_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating i (j+1) → seating i (j+2) → False
  col_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating (i+1) j → seating (i+2) j → False

theorem max_participants {room : MeetingRoom} (h : room.rows = 4 ∧ room.cols = 4) : 
  (∃ n : ℕ, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → n < 12) ∧
            (∀ m, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → m < 12) → m ≤ 11)) :=
  sorry

end max_participants_l67_67729


namespace sequence_bk_bl_sum_l67_67069

theorem sequence_bk_bl_sum (b : ℕ → ℕ) (m : ℕ) 
  (h_pairwise_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (h_b0 : b 0 = 0)
  (h_b_lt_2n : ∀ n, 0 < n → b n < 2 * n) :
  ∃ k ℓ : ℕ, b k + b ℓ = m := 
  sorry

end sequence_bk_bl_sum_l67_67069


namespace not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l67_67945

theorem not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles : 
  ¬ ∃ (rectangles : ℕ × ℕ), rectangles.1 = 1 ∧ rectangles.2 = 7 ∧ rectangles.1 * 4 + rectangles.2 * 3 = 25 :=
by
  sorry

end not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l67_67945


namespace max_value_of_fraction_l67_67119

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l67_67119


namespace side_length_of_square_l67_67130

theorem side_length_of_square (A : ℝ) (h : A = 81) : ∃ s : ℝ, s^2 = A ∧ s = 9 :=
by
  sorry

end side_length_of_square_l67_67130


namespace Phil_earns_per_hour_l67_67444

-- Definitions based on the conditions in the problem
def Mike_hourly_rate : ℝ := 14
def Phil_hourly_rate : ℝ := Mike_hourly_rate - (0.5 * Mike_hourly_rate)

-- Mathematical assertion to prove
theorem Phil_earns_per_hour : Phil_hourly_rate = 7 :=
by 
  sorry

end Phil_earns_per_hour_l67_67444


namespace perpendicular_lines_slope_eq_l67_67374

theorem perpendicular_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
               2 * x + m * y - 6 = 0 → 
               (1 / 2) * (-2 / m) = -1) →
  m = 1 := 
by sorry

end perpendicular_lines_slope_eq_l67_67374


namespace triangle_angles_arithmetic_progression_l67_67757

theorem triangle_angles_arithmetic_progression (α β γ : ℝ) (a c : ℝ) :
  (α < β) ∧ (β < γ) ∧ (α + β + γ = 180) ∧
  (∃ x : ℝ, β = α + x ∧ γ = β + x) ∧
  (a = c / 2) → 
  (α = 30) ∧ (β = 60) ∧ (γ = 90) :=
by
  intros h
  sorry

end triangle_angles_arithmetic_progression_l67_67757


namespace n_sum_of_two_squares_l67_67662

theorem n_sum_of_two_squares (n : ℤ) (m : ℤ) (hn_gt_2 : n > 2) (hn2_eq_diff_cubes : n^2 = (m+1)^3 - m^3) : 
  ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

end n_sum_of_two_squares_l67_67662


namespace count_two_digit_remainders_l67_67531

theorem count_two_digit_remainders : 
  ∃ n, (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 7 = 3 ↔ (∃ k, 1 ≤ k ∧ k ≤ 13 ∧ x = 7*k + 3)) ∧ n = 13 :=
by
  sorry

end count_two_digit_remainders_l67_67531


namespace geometric_sequence_third_term_l67_67625

-- Define the problem statement in Lean 4
theorem geometric_sequence_third_term :
  ∃ r : ℝ, (a = 1024) ∧ (a_5 = 128) ∧ (a_5 = a * r^4) ∧ 
  (a_3 = a * r^2) ∧ (a_3 = 256) :=
sorry

end geometric_sequence_third_term_l67_67625


namespace total_pages_in_book_l67_67926

theorem total_pages_in_book 
    (pages_read : ℕ) (pages_left : ℕ) 
    (h₁ : pages_read = 11) 
    (h₂ : pages_left = 6) : 
    pages_read + pages_left = 17 := 
by 
    sorry

end total_pages_in_book_l67_67926


namespace tangent_line_to_parabola_l67_67245

noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

def derivative_parabola (x : ℝ) : ℝ := 8 * x

def tangent_line_eq (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_to_parabola (x : ℝ) (hx : x = 1) (hy : parabola x = 4) :
    tangent_line_eq 1 4 :=
by 
  -- Sorry to skip the detailed proof, but it should follow the steps outlined in the solution.
  sorry

end tangent_line_to_parabola_l67_67245


namespace lateral_surface_area_cone_l67_67127

theorem lateral_surface_area_cone (r l : ℝ) (h₀ : r = 6) (h₁ : l = 10) : π * r * l = 60 * π := by 
  sorry

end lateral_surface_area_cone_l67_67127


namespace bob_distance_when_meet_l67_67396

def distance_xy : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def time_start_diff : ℝ := 1

theorem bob_distance_when_meet : ∃ t : ℝ, yolanda_rate * t + bob_rate * (t - time_start_diff) = distance_xy ∧ bob_rate * (t - time_start_diff) = 4 :=
by
  sorry

end bob_distance_when_meet_l67_67396


namespace total_volume_of_all_cubes_l67_67398

def volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume_of_cubes (num_cubes : ℕ) (side_length : ℕ) : ℕ :=
  num_cubes * volume side_length

theorem total_volume_of_all_cubes :
  total_volume_of_cubes 3 3 + total_volume_of_cubes 4 4 = 337 :=
by
  sorry

end total_volume_of_all_cubes_l67_67398


namespace algebra_problem_l67_67025

theorem algebra_problem 
  (a : ℝ) 
  (h : a^3 + 3 * a^2 + 3 * a + 2 = 0) :
  (a + 1) ^ 2008 + (a + 1) ^ 2009 + (a + 1) ^ 2010 = 1 :=
by 
  sorry

end algebra_problem_l67_67025


namespace infinite_solutions_if_one_exists_l67_67561

namespace RationalSolutions

def has_rational_solution (a b : ℚ) : Prop :=
  ∃ (x y : ℚ), a * x^2 + b * y^2 = 1

def infinite_rational_solutions (a b : ℚ) : Prop :=
  ∀ (x₀ y₀ : ℚ), (a * x₀^2 + b * y₀^2 = 1) → ∃ (f : ℕ → ℚ × ℚ), ∀ n : ℕ, a * (f n).1^2 + b * (f n).2^2 = 1 ∧ (f 0 = (x₀, y₀)) ∧ ∀ m n : ℕ, m ≠ n → (f m) ≠ (f n)

theorem infinite_solutions_if_one_exists (a b : ℚ) (h : has_rational_solution a b) : infinite_rational_solutions a b :=
  sorry

end RationalSolutions

end infinite_solutions_if_one_exists_l67_67561


namespace arithmetic_sequence_problem_l67_67664

noncomputable def a_n (n : ℕ) : ℚ := 1 + (n - 1) / 2

noncomputable def S_n (n : ℕ) : ℚ := n * (n + 3) / 4

theorem arithmetic_sequence_problem :
  -- Given
  (∀ n, ∃ d, a_n n = a_1 + (n - 1) * d) →
  (a_n 7 = 4) →
  (a_n 19 = 2 * a_n 9) →
  -- Prove
  (∀ n, a_n n = (n + 1) / 2) ∧ (∀ n, S_n n = n * (n + 3) / 4) :=
by
  sorry

end arithmetic_sequence_problem_l67_67664


namespace factorize_expression_l67_67024

theorem factorize_expression (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := 
by
  sorry

end factorize_expression_l67_67024


namespace remaining_battery_life_l67_67739

theorem remaining_battery_life :
  let capacity1 := 60
  let capacity2 := 80
  let capacity3 := 120
  let used1 := capacity1 * (3 / 4 : ℚ)
  let used2 := capacity2 * (1 / 2 : ℚ)
  let used3 := capacity3 * (2 / 3 : ℚ)
  let remaining1 := capacity1 - used1 - 2
  let remaining2 := capacity2 - used2 - 2
  let remaining3 := capacity3 - used3 - 2
  remaining1 + remaining2 + remaining3 = 89 := 
by
  sorry

end remaining_battery_life_l67_67739


namespace sum_of_squares_diagonals_of_rhombus_l67_67935

theorem sum_of_squares_diagonals_of_rhombus (d1 d2 : ℝ) (h : (d1 / 2)^2 + (d2 / 2)^2 = 4) : d1^2 + d2^2 = 16 :=
sorry

end sum_of_squares_diagonals_of_rhombus_l67_67935


namespace cost_of_student_ticket_l67_67311

theorem cost_of_student_ticket
  (cost_adult : ℤ)
  (total_tickets : ℤ)
  (total_revenue : ℤ)
  (adult_tickets : ℤ)
  (student_tickets : ℤ)
  (H1 : cost_adult = 6)
  (H2 : total_tickets = 846)
  (H3 : total_revenue = 3846)
  (H4 : adult_tickets = 410)
  (H5 : student_tickets = 436)
  : (total_revenue = adult_tickets * cost_adult + student_tickets * (318 / 100)) :=
by
  -- mathematical proof steps would go here
  sorry

end cost_of_student_ticket_l67_67311


namespace logan_usual_cartons_l67_67198

theorem logan_usual_cartons 
  (C : ℕ)
  (h1 : ∀ cartons, (∀ jars : ℕ, jars = 20 * cartons) → jars = 20 * C)
  (h2 : ∀ cartons, cartons = C - 20)
  (h3 : ∀ damaged_jars, (∀ cartons : ℕ, cartons = 5) → damaged_jars = 3 * 5)
  (h4 : ∀ completely_damaged_jars, completely_damaged_jars = 20)
  (h5 : ∀ good_jars, good_jars = 565) :
  C = 50 :=
by
  sorry

end logan_usual_cartons_l67_67198


namespace dons_profit_l67_67101

-- Definitions from the conditions
def bundles_jamie_bought := 20
def bundles_jamie_sold := 15
def profit_jamie := 60

def bundles_linda_bought := 34
def bundles_linda_sold := 24
def profit_linda := 69

def bundles_don_bought := 40
def bundles_don_sold := 36

-- Variables representing the unknown prices
variables (b s : ℝ)

-- Conditions written as equalities
axiom eq_jamie : bundles_jamie_sold * s - bundles_jamie_bought * b = profit_jamie
axiom eq_linda : bundles_linda_sold * s - bundles_linda_bought * b = profit_linda

-- Statement to prove Don's profit
theorem dons_profit : bundles_don_sold * s - bundles_don_bought * b = 252 :=
by {
  sorry -- proof goes here
}

end dons_profit_l67_67101


namespace curve_crossing_point_l67_67646

theorem curve_crossing_point :
  (∃ t : ℝ, (t^2 - 4 = 2) ∧ (t^3 - 6 * t + 4 = 4)) ∧
  (∃ t' : ℝ, t ≠ t' ∧ (t'^2 - 4 = 2) ∧ (t'^3 - 6 * t' + 4 = 4)) :=
sorry

end curve_crossing_point_l67_67646


namespace cos_of_acute_angle_l67_67451

theorem cos_of_acute_angle (θ : ℝ) (hθ1 : 0 < θ ∧ θ < π / 2) (hθ2 : Real.sin θ = 1 / 3) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 :=
by
  -- The proof steps will be filled here
  sorry

end cos_of_acute_angle_l67_67451


namespace part1_part2_l67_67115

-- Part 1: Expression simplification
theorem part1 (a : ℝ) : (a - 3)^2 + a * (4 - a) = -2 * a + 9 := 
by
  sorry

-- Part 2: Solution set of inequalities
theorem part2 (x : ℝ) : 
  (3 * x - 5 < x + 1) ∧ (2 * (2 * x - 1) ≥ 3 * x - 4) ↔ (-2 ≤ x ∧ x < 3) := 
by
  sorry

end part1_part2_l67_67115


namespace fish_left_in_tank_l67_67021

theorem fish_left_in_tank (initial_fish : ℕ) (fish_taken_out : ℕ) (fish_left : ℕ) 
  (h1 : initial_fish = 19) (h2 : fish_taken_out = 16) : fish_left = initial_fish - fish_taken_out :=
by
  simp [h1, h2]
  sorry

end fish_left_in_tank_l67_67021


namespace rectangular_field_area_l67_67760

theorem rectangular_field_area
  (x : ℝ) 
  (length := 3 * x) 
  (breadth := 4 * x) 
  (perimeter := 2 * (length + breadth))
  (cost_per_meter : ℝ := 0.25) 
  (total_cost : ℝ := 87.5) 
  (paise_per_rupee : ℝ := 100)
  (perimeter_eq_cost : 14 * x * cost_per_meter * paise_per_rupee = total_cost * paise_per_rupee) :
  (length * breadth = 7500) := 
by
  -- proof omitted
  sorry

end rectangular_field_area_l67_67760


namespace trig_problem_1_trig_problem_2_l67_67774

noncomputable def trig_expr_1 : ℝ :=
  Real.cos (-11 * Real.pi / 6) + Real.sin (12 * Real.pi / 5) * Real.tan (6 * Real.pi)

noncomputable def trig_expr_2 : ℝ :=
  Real.sin (420 * Real.pi / 180) * Real.cos (750 * Real.pi / 180) +
  Real.sin (-330 * Real.pi / 180) * Real.cos (-660 * Real.pi / 180)

theorem trig_problem_1 : trig_expr_1 = Real.sqrt 3 / 2 :=
by
  sorry

theorem trig_problem_2 : trig_expr_2 = 1 :=
by
  sorry

end trig_problem_1_trig_problem_2_l67_67774


namespace train_speed_l67_67895

theorem train_speed
  (length_of_train : ℝ) 
  (time_to_cross : ℝ) 
  (train_length_is_140 : length_of_train = 140)
  (time_is_6 : time_to_cross = 6) :
  (length_of_train / time_to_cross) = 23.33 :=
sorry

end train_speed_l67_67895


namespace sheep_to_cow_ratio_l67_67147

theorem sheep_to_cow_ratio : 
  ∀ (cows sheep : ℕ) (cow_water sheep_water : ℕ),
  cows = 40 →
  cow_water = 80 →
  sheep_water = cow_water / 4 →
  7 * (cows * cow_water + sheep * sheep_water) = 78400 →
  sheep / cows = 10 :=
by
  intros cows sheep cow_water sheep_water hcows hcow_water hsheep_water htotal
  sorry

end sheep_to_cow_ratio_l67_67147


namespace janice_initial_sentences_l67_67208

theorem janice_initial_sentences :
  ∀ (initial_sentences total_sentences erased_sentences: ℕ)
    (typed_rate before_break_minutes additional_minutes after_meeting_minutes: ℕ),
  typed_rate = 6 →
  before_break_minutes = 20 →
  additional_minutes = 15 →
  after_meeting_minutes = 18 →
  erased_sentences = 40 →
  total_sentences = 536 →
  (total_sentences - (before_break_minutes * typed_rate + (before_break_minutes + additional_minutes) * typed_rate + after_meeting_minutes * typed_rate - erased_sentences)) = initial_sentences →
  initial_sentences = 138 :=
by
  intros initial_sentences total_sentences erased_sentences typed_rate before_break_minutes additional_minutes after_meeting_minutes
  intros h_rate h_before h_additional h_after_meeting h_erased h_total h_eqn
  rw [h_rate, h_before, h_additional, h_after_meeting, h_erased, h_total] at h_eqn
  linarith

end janice_initial_sentences_l67_67208


namespace range_m_l67_67167

noncomputable def circle_c (x y : ℝ) : Prop := (x - 4) ^ 2 + (y - 3) ^ 2 = 4

def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

theorem range_m (m : ℝ) (P : ℝ × ℝ) :
  circle_c P.1 P.2 ∧ m > 0 ∧ (∃ (a b : ℝ), P = (a, b) ∧ (a + m) * (a - m) + b ^ 2 = 0) → m ∈ Set.Icc 3 7 :=
sorry

end range_m_l67_67167


namespace intersection_complement_l67_67361

open Set

/-- The universal set U as the set of all real numbers -/
def U : Set ℝ := @univ ℝ

/-- The set M -/
def M : Set ℝ := {-1, 0, 1}

/-- The set N defined by the equation x^2 + x = 0 -/
def N : Set ℝ := {x | x^2 + x = 0}

/-- The complement of set N in the universal set U -/
def C_U_N : Set ℝ := {x ∈ U | x ≠ -1 ∧ x ≠ 0}

theorem intersection_complement :
  M ∩ C_U_N = {1} :=
by
  sorry

end intersection_complement_l67_67361


namespace necessary_but_not_sufficient_l67_67930

theorem necessary_but_not_sufficient (a b x y : ℤ) (ha : 0 < a) (hb : 0 < b) (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  (x > a ∧ y > b) := sorry

end necessary_but_not_sufficient_l67_67930


namespace emily_stickers_l67_67833

theorem emily_stickers:
  ∃ S : ℕ, (S % 4 = 2) ∧
           (S % 6 = 2) ∧
           (S % 9 = 2) ∧
           (S % 10 = 2) ∧
           (S > 2) ∧
           (S = 182) :=
  sorry

end emily_stickers_l67_67833


namespace tortoise_age_l67_67111

-- Definitions based on the given problem conditions
variables (a b c : ℕ)

-- The conditions as provided in the problem
def condition1 (a b : ℕ) : Prop := a / 4 = 2 * a - b
def condition2 (b c : ℕ) : Prop := b / 7 = 2 * b - c
def condition3 (a b c : ℕ) : Prop := a + b + c = 264

-- The main theorem to prove
theorem tortoise_age (a b c : ℕ) (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 a b c) : b = 77 :=
sorry

end tortoise_age_l67_67111


namespace cost_price_of_watch_l67_67041

theorem cost_price_of_watch (CP : ℝ) (h1 : SP1 = CP * 0.64) (h2 : SP2 = CP * 1.04) (h3 : SP2 = SP1 + 140) : CP = 350 :=
by
  sorry

end cost_price_of_watch_l67_67041


namespace greatest_common_divisor_of_98_and_n_l67_67919

theorem greatest_common_divisor_of_98_and_n (n : ℕ) (h1 : ∃ (d : Finset ℕ),  d = {1, 7, 49} ∧ ∀ x ∈ d, x ∣ 98 ∧ x ∣ n) :
  ∃ (g : ℕ), g = 49 :=
by
  sorry

end greatest_common_divisor_of_98_and_n_l67_67919


namespace remainder_is_cx_plus_d_l67_67495

-- Given a polynomial Q, assume the following conditions
variables {Q : ℕ → ℚ}

-- Conditions
axiom condition1 : Q 15 = 12
axiom condition2 : Q 10 = 4

theorem remainder_is_cx_plus_d : 
  ∃ c d, (c = 8 / 5) ∧ (d = -12) ∧ 
          ∀ x, Q x % ((x - 10) * (x - 15)) = c * x + d :=
by
  sorry

end remainder_is_cx_plus_d_l67_67495


namespace job_completion_l67_67459

theorem job_completion (A_rate D_rate : ℝ) (h₁ : A_rate = 1 / 12) (h₂ : A_rate + D_rate = 1 / 4) : D_rate = 1 / 6 := 
by 
  sorry

end job_completion_l67_67459


namespace raja_monthly_income_l67_67796

noncomputable def monthly_income (household_percentage clothes_percentage medicines_percentage savings : ℝ) : ℝ :=
  let spending_percentage := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage := 1 - spending_percentage
  savings / savings_percentage

theorem raja_monthly_income :
  monthly_income 0.35 0.20 0.05 15000 = 37500 :=
by
  sorry

end raja_monthly_income_l67_67796


namespace a_values_l67_67998

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem a_values (a : ℝ) : A a ∩ B a = {x} → (a = 0 ∧ x = 1) ∨ (a = -2 ∧ x = 5) := sorry

end a_values_l67_67998


namespace wheel_center_travel_distance_l67_67000

theorem wheel_center_travel_distance (radius : ℝ) (revolutions : ℝ) (flat_surface : Prop) 
  (h_radius : radius = 2) (h_revolutions : revolutions = 2) : 
  radius * 2 * π * revolutions = 8 * π :=
by
  rw [h_radius, h_revolutions]
  simp [mul_assoc, mul_comm]
  sorry

end wheel_center_travel_distance_l67_67000


namespace odd_divisors_l67_67018

-- Define p_1, p_2, p_3 as distinct prime numbers greater than 3
variables {p_1 p_2 p_3 : ℕ}
-- Define k, a, b, c as positive integers
variables {n k a b c : ℕ}

-- The conditions
def distinct_primes (p_1 p_2 p_3 : ℕ) : Prop :=
  p_1 > 3 ∧ p_2 > 3 ∧ p_3 > 3 ∧ p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_2 ≠ p_3

def is_n (n k p_1 p_2 p_3 a b c : ℕ) : Prop :=
  n = 2^k * p_1^a * p_2^b * p_3^c

def conditions (a b c : ℕ) : Prop :=
  a + b > c ∧ 1 ≤ b ∧ b ≤ c

-- The main statement
theorem odd_divisors
  (h_prime : distinct_primes p_1 p_2 p_3)
  (h_n : is_n n k p_1 p_2 p_3 a b c)
  (h_cond : conditions a b c) : 
  ∃ d : ℕ, d = (a + 1) * (b + 1) * (c + 1) :=
by sorry

end odd_divisors_l67_67018


namespace mark_asphalt_total_cost_l67_67434

noncomputable def total_cost (road_length : ℕ) (road_width : ℕ) (area_per_truckload : ℕ) (cost_per_truckload : ℕ) (sales_tax_rate : ℚ) : ℚ :=
  let total_area := road_length * road_width
  let num_truckloads := total_area / area_per_truckload
  let cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := cost_before_tax * sales_tax_rate
  let total_cost := cost_before_tax + sales_tax
  total_cost

theorem mark_asphalt_total_cost :
  total_cost 2000 20 800 75 0.2 = 4500 := 
by sorry

end mark_asphalt_total_cost_l67_67434


namespace system_of_equations_solution_l67_67623

variable {x y : ℝ}

theorem system_of_equations_solution
  (h1 : x^2 + x * y * Real.sqrt (x * y) + y^2 = 25)
  (h2 : x^2 - x * y * Real.sqrt (x * y) + y^2 = 9) :
  (x, y) = (1, 4) ∨ (x, y) = (4, 1) ∨ (x, y) = (-1, -4) ∨ (x, y) = (-4, -1) :=
by
  sorry

end system_of_equations_solution_l67_67623


namespace inscribed_circle_radius_l67_67366

theorem inscribed_circle_radius (r : ℝ) (radius : ℝ) (angle_deg : ℝ): 
  radius = 6 ∧ angle_deg = 120 ∧ (∀ θ : ℝ, θ = 60) → r = 3 := 
by
  sorry

end inscribed_circle_radius_l67_67366


namespace arithmetic_sequence_term_l67_67733

theorem arithmetic_sequence_term (a d n : ℕ) (h₀ : a = 1) (h₁ : d = 3) (h₂ : a + (n - 1) * d = 6019) :
  n = 2007 :=
sorry

end arithmetic_sequence_term_l67_67733


namespace trains_meet_distance_from_delhi_l67_67785

-- Define the speeds of the trains as constants
def speed_bombay_express : ℕ := 60  -- kmph
def speed_rajdhani_express : ℕ := 80  -- kmph

-- Define the time difference in hours between the departures of the two trains
def time_difference : ℕ := 2  -- hours

-- Define the distance the Bombay Express travels before the Rajdhani Express starts
def distance_head_start : ℕ := speed_bombay_express * time_difference

-- Define the relative speed between the two trains
def relative_speed : ℕ := speed_rajdhani_express - speed_bombay_express

-- Define the time taken for the Rajdhani Express to catch up with the Bombay Express
def time_to_meet : ℕ := distance_head_start / relative_speed

-- The final meeting distance from Delhi for the Rajdhani Express
def meeting_distance : ℕ := speed_rajdhani_express * time_to_meet

-- Theorem stating the solution to the problem
theorem trains_meet_distance_from_delhi : meeting_distance = 480 :=
by sorry  -- proof is omitted

end trains_meet_distance_from_delhi_l67_67785


namespace probability_one_each_l67_67889

-- Define the counts of letters
def total_letters : ℕ := 11
def cybil_count : ℕ := 5
def ronda_count : ℕ := 5
def andy_initial_count : ℕ := 1

-- Define the probability calculation
def probability_one_from_cybil_and_one_from_ronda : ℚ :=
  (cybil_count / total_letters) * (ronda_count / (total_letters - 1)) +
  (ronda_count / total_letters) * (cybil_count / (total_letters - 1))

theorem probability_one_each (total_letters cybil_count ronda_count andy_initial_count : ℕ) :
  probability_one_from_cybil_and_one_from_ronda = 5 / 11 := sorry

end probability_one_each_l67_67889


namespace lieutenant_age_l67_67081

variables (n x : ℕ) 

-- Condition 1: Number of soldiers is the same in both formations
def total_soldiers_initial (n : ℕ) : ℕ := n * (n + 5)
def total_soldiers_new (n x : ℕ) : ℕ := x * (n + 9)

-- Condition 2: The number of soldiers is the same 
-- and Condition 3: Equations relating n and x
theorem lieutenant_age (n x : ℕ) (h1: total_soldiers_initial n = total_soldiers_new n x) (h2 : x = 24) : 
  x = 24 :=
by {
  sorry
}

end lieutenant_age_l67_67081


namespace sqrt_expression_evaluation_l67_67397

theorem sqrt_expression_evaluation : 
  (Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2) := 
by
  sorry

end sqrt_expression_evaluation_l67_67397


namespace smallest_percent_increase_is_100_l67_67906

-- The values for each question
def prize_values : List ℕ := [150, 300, 450, 900, 1800, 3600, 7200, 14400, 28800, 57600, 115200, 230400, 460800, 921600, 1843200]

-- Definition of percent increase calculation
def percent_increase (old new : ℕ) : ℕ :=
  ((new - old : ℕ) * 100) / old

-- Lean theorem statement
theorem smallest_percent_increase_is_100 :
  percent_increase (prize_values.get! 5) (prize_values.get! 6) = 100 ∧
  percent_increase (prize_values.get! 7) (prize_values.get! 8) = 100 ∧
  percent_increase (prize_values.get! 9) (prize_values.get! 10) = 100 ∧
  percent_increase (prize_values.get! 10) (prize_values.get! 11) = 100 ∧
  percent_increase (prize_values.get! 13) (prize_values.get! 14) = 100 :=
by
  sorry

end smallest_percent_increase_is_100_l67_67906


namespace intersection_sets_l67_67446

theorem intersection_sets :
  let M := {x : ℝ | 0 < x} 
  let N := {y : ℝ | 1 ≤ y}
  M ∩ N = {z : ℝ | 1 ≤ z} :=
by
  -- Proof goes here
  sorry

end intersection_sets_l67_67446


namespace b_is_dk_squared_l67_67272

theorem b_is_dk_squared (a b : ℤ) (h : ∃ r1 r2 r3 : ℤ, (r1 * r2 * r3 = b) ∧ (r1 + r2 + r3 = a) ∧ (r1 * r2 + r1 * r3 + r2 * r3 = 0))
  : ∃ d k : ℤ, (b = d * k^2) ∧ (d ∣ a) := 
sorry

end b_is_dk_squared_l67_67272


namespace polynomial_value_at_minus_two_l67_67574

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_minus_two :
  f (-2) = -1 :=
by sorry

end polynomial_value_at_minus_two_l67_67574


namespace corridor_perimeter_l67_67868

theorem corridor_perimeter
  (P1 P2 : ℕ)
  (h₁ : P1 = 16)
  (h₂ : P2 = 24) : 
  2 * ((P2 / 4 + (P1 + P2) / 4) + (P2 / 4) - (P1 / 4)) = 40 :=
by {
  -- The proof can be filled here
  sorry
}

end corridor_perimeter_l67_67868


namespace radius_of_inscribed_circle_is_three_fourths_l67_67827

noncomputable def circle_diameter : ℝ := Real.sqrt 12

noncomputable def radius_of_new_inscribed_circle : ℝ :=
  let R := circle_diameter / 2
  let s := R * Real.sqrt 3
  let h := s * Real.sqrt 3 / 2
  let a := Real.sqrt (h^2 - (h/2)^2)
  a * Real.sqrt 3 / 6

theorem radius_of_inscribed_circle_is_three_fourths :
  radius_of_new_inscribed_circle = 3 / 4 := sorry

end radius_of_inscribed_circle_is_three_fourths_l67_67827


namespace value_of_abc_l67_67123

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c
noncomputable def f_inv (a b c x : ℝ) := c * x^2 + b * x + a

-- The main theorem statement
theorem value_of_abc (a b c : ℝ) (h : ∀ x : ℝ, f a b c (f_inv a b c x) = x) : a + b + c = 1 :=
sorry

end value_of_abc_l67_67123


namespace new_parabola_after_shift_l67_67349

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the transformation functions for shifting the parabola
def shift_left (x : ℝ) (shift : ℝ) : ℝ := x + shift
def shift_down (y : ℝ) (shift : ℝ) : ℝ := y - shift

-- Prove the transformation yields the correct new parabola equation
theorem new_parabola_after_shift : 
  (∀ x : ℝ, (shift_down (original_parabola (shift_left x 2)) 3) = (x + 2)^2 - 2) :=
by
  sorry

end new_parabola_after_shift_l67_67349


namespace direct_variation_y_value_l67_67156

theorem direct_variation_y_value (x y k : ℝ) (h1 : y = k * x) (h2 : ∀ x, x = 5 → y = 10) 
                                 (h3 : ∀ x, x < 0 → k = 4) (hx : x = -6) : y = -24 :=
sorry

end direct_variation_y_value_l67_67156


namespace base_length_first_tri_sail_l67_67666

-- Define the areas of the sails
def area_rect_sail : ℕ := 5 * 8
def area_second_tri_sail : ℕ := (4 * 6) / 2

-- Total canvas area needed
def total_canvas_area_needed : ℕ := 58

-- Calculate the total area so far (rectangular sail + second triangular sail)
def total_area_so_far : ℕ := area_rect_sail + area_second_tri_sail

-- Define the height of the first triangular sail
def height_first_tri_sail : ℕ := 4

-- Define the area needed for the first triangular sail
def area_first_tri_sail : ℕ := total_canvas_area_needed - total_area_so_far

-- Prove that the base length of the first triangular sail is 3 inches
theorem base_length_first_tri_sail : ∃ base : ℕ, base = 3 ∧ (base * height_first_tri_sail) / 2 = area_first_tri_sail := by
  use 3
  have h1 : (3 * 4) / 2 = 6 := by sorry -- This is a placeholder for actual calculation
  exact ⟨rfl, h1⟩

end base_length_first_tri_sail_l67_67666


namespace distance_from_plate_to_bottom_edge_l67_67218

theorem distance_from_plate_to_bottom_edge :
    ∀ (d : ℕ), 10 + 63 = 20 + d → d = 53 :=
by
  intros d h
  sorry

end distance_from_plate_to_bottom_edge_l67_67218


namespace original_example_intended_l67_67770

theorem original_example_intended (x : ℝ) : (3 * x - 4 = x / 3 + 4) → x = 3 :=
by
  sorry

end original_example_intended_l67_67770


namespace train_vs_airplane_passenger_capacity_l67_67530

theorem train_vs_airplane_passenger_capacity :
  (60 * 16) - (366 * 2) = 228 := by
sorry

end train_vs_airplane_passenger_capacity_l67_67530


namespace sacks_required_in_4_weeks_l67_67971

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end sacks_required_in_4_weeks_l67_67971


namespace find_z_plus_1_over_y_l67_67284

theorem find_z_plus_1_over_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 20) : 
  z + 1 / y = 29 / 139 := 
by 
  sorry

end find_z_plus_1_over_y_l67_67284


namespace harry_ron_difference_l67_67713

-- Define the amounts each individual paid
def harry_paid : ℕ := 150
def ron_paid : ℕ := 180
def hermione_paid : ℕ := 210

-- Define the total amount
def total_paid : ℕ := harry_paid + ron_paid + hermione_paid

-- Define the amount each should have paid
def equal_share : ℕ := total_paid / 3

-- Define the amount Harry owes to Hermione
def harry_owes : ℕ := equal_share - harry_paid

-- Define the amount Ron owes to Hermione
def ron_owes : ℕ := equal_share - ron_paid

-- Define the difference between what Harry and Ron owe Hermione
def difference : ℕ := harry_owes - ron_owes

-- Prove that the difference is 30
theorem harry_ron_difference : difference = 30 := by
  sorry

end harry_ron_difference_l67_67713


namespace fraction_percent_l67_67812

theorem fraction_percent (x : ℝ) (h : x > 0) : ((x / 10 + x / 25) / x) * 100 = 14 :=
by
  sorry

end fraction_percent_l67_67812


namespace range_of_m_for_nonempty_solution_set_l67_67678

theorem range_of_m_for_nonempty_solution_set :
  {m : ℝ | ∃ x : ℝ, m * x^2 - m * x + 1 < 0} = {m : ℝ | m < 0} ∪ {m : ℝ | m > 4} :=
by sorry

end range_of_m_for_nonempty_solution_set_l67_67678


namespace max_x_y_given_condition_l67_67794

theorem max_x_y_given_condition (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 1/x + 1/y = 5) : x + y ≤ 4 :=
sorry

end max_x_y_given_condition_l67_67794


namespace water_cost_is_1_l67_67871

-- Define the conditions
def cost_cola : ℝ := 3
def cost_juice : ℝ := 1.5
def bottles_sold_cola : ℝ := 15
def bottles_sold_juice : ℝ := 12
def bottles_sold_water : ℝ := 25
def total_revenue : ℝ := 88

-- Compute the revenue from cola and juice
def revenue_cola : ℝ := bottles_sold_cola * cost_cola
def revenue_juice : ℝ := bottles_sold_juice * cost_juice

-- Define a proof that the cost of a bottle of water is $1
theorem water_cost_is_1 : (total_revenue - revenue_cola - revenue_juice) / bottles_sold_water = 1 :=
by
  -- Proof is omitted
  sorry

end water_cost_is_1_l67_67871


namespace binom_25_7_l67_67614

theorem binom_25_7 :
  (Nat.choose 23 5 = 33649) →
  (Nat.choose 23 6 = 42504) →
  (Nat.choose 23 7 = 33649) →
  Nat.choose 25 7 = 152306 :=
by
  intros h1 h2 h3
  sorry

end binom_25_7_l67_67614


namespace rectangle_perimeter_is_70_l67_67492

-- Define the length and width of the rectangle
def length : ℕ := 19
def width : ℕ := 16

-- Define the perimeter function for a rectangle
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem statement asserting that the perimeter of the given rectangle is 70 cm
theorem rectangle_perimeter_is_70 :
  perimeter length width = 70 := 
sorry

end rectangle_perimeter_is_70_l67_67492


namespace action_figure_price_l67_67371

theorem action_figure_price (x : ℝ) (h1 : 2 + 4 * x = 30) : x = 7 :=
by
  -- The proof is provided here
  sorry

end action_figure_price_l67_67371


namespace main_theorem_l67_67222

noncomputable def exists_coprime_integers (a b p : ℤ) : Prop :=
  ∃ (m n : ℤ), Int.gcd m n = 1 ∧ p ∣ (a * m + b * n)

theorem main_theorem (a b p : ℤ) : exists_coprime_integers a b p := 
  sorry

end main_theorem_l67_67222


namespace same_terminal_angle_l67_67790

theorem same_terminal_angle (k : ℤ) :
  ∃ α : ℝ, α = k * 360 + 40 :=
by
  sorry

end same_terminal_angle_l67_67790


namespace train_speed_kmph_l67_67276

noncomputable def train_length : ℝ := 200
noncomputable def crossing_time : ℝ := 3.3330666879982935

theorem train_speed_kmph : (train_length / crossing_time) * 3.6 = 216.00072 := by
  sorry

end train_speed_kmph_l67_67276


namespace find_first_number_l67_67419

variable {A B C D : ℕ}

theorem find_first_number (h1 : A + B + C = 60) (h2 : B + C + D = 45) (h3 : D = 18) : A = 33 := 
  sorry

end find_first_number_l67_67419


namespace log_order_l67_67033

theorem log_order (a b c : ℝ) (h_a : a = Real.log 6 / Real.log 2) 
  (h_b : b = Real.log 15 / Real.log 5) (h_c : c = Real.log 21 / Real.log 7) : 
  a > b ∧ b > c := by sorry

end log_order_l67_67033


namespace total_employee_costs_in_February_l67_67354

def weekly_earnings (hours_per_week : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_week * rate_per_hour

def monthly_earnings 
  (hours_per_week : ℕ) 
  (rate_per_hour : ℕ) 
  (weeks_worked : ℕ) 
  (bonus_deduction : ℕ := 0) 
  : ℕ :=
  weeks_worked * weekly_earnings hours_per_week rate_per_hour + bonus_deduction

theorem total_employee_costs_in_February 
  (hours_Fiona : ℕ := 40) (rate_Fiona : ℕ := 20) (weeks_worked_Fiona : ℕ := 3)
  (hours_John : ℕ := 30) (rate_John : ℕ := 22) (overtime_hours_John : ℕ := 10)
  (hours_Jeremy : ℕ := 25) (rate_Jeremy : ℕ := 18) (bonus_Jeremy : ℕ := 200)
  (hours_Katie : ℕ := 35) (rate_Katie : ℕ := 21) (deduction_Katie : ℕ := 150)
  (hours_Matt : ℕ := 28) (rate_Matt : ℕ := 19) : 
  monthly_earnings hours_Fiona rate_Fiona weeks_worked_Fiona 
  + monthly_earnings hours_John rate_John 4 
    + overtime_hours_John * (rate_John * 3 / 2)
  + monthly_earnings hours_Jeremy rate_Jeremy 4 bonus_Jeremy
  + monthly_earnings hours_Katie rate_Katie 4 - deduction_Katie
  + monthly_earnings hours_Matt rate_Matt 4 = 13278 := 
by sorry

end total_employee_costs_in_February_l67_67354


namespace intersection_equivalence_l67_67782

open Set

noncomputable def U : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def M : Set ℤ := {-1, 0, 1}
noncomputable def N : Set ℤ := {x | x * x - x - 2 = 0}
noncomputable def complement_M_in_U : Set ℤ := U \ M

theorem intersection_equivalence : (complement_M_in_U ∩ N) = {2} := 
by
  sorry

end intersection_equivalence_l67_67782


namespace smallest_prime_sum_l67_67908

theorem smallest_prime_sum (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (H1 : Prime (a + b + c + d))
  (H2 : Prime (a + b)) (H3 : Prime (a + c)) (H4 : Prime (a + d)) (H5 : Prime (b + c)) (H6 : Prime (b + d)) (H7 : Prime (c + d))
  (H8 : Prime (a + b + c)) (H9 : Prime (a + b + d)) (H10 : Prime (a + c + d)) (H11 : Prime (b + c + d))
  : a + b + c + d = 31 :=
sorry

end smallest_prime_sum_l67_67908


namespace difference_of_squares_is_39_l67_67711

theorem difference_of_squares_is_39 (L S : ℕ) (h1 : L = 8) (h2 : L - S = 3) : L^2 - S^2 = 39 :=
by
  sorry

end difference_of_squares_is_39_l67_67711


namespace unique_solution_iff_t_eq_quarter_l67_67363

variable {x y t : ℝ}

theorem unique_solution_iff_t_eq_quarter : (∃! (x y : ℝ), (x ≥ y^2 + t * y ∧ y^2 + t * y ≥ x^2 + t)) ↔ t = 1 / 4 :=
by
  sorry

end unique_solution_iff_t_eq_quarter_l67_67363


namespace cube_surface_area_l67_67856

-- Define the edge length of the cube.
def edge_length (a : ℝ) : ℝ := 6 * a

-- Define the surface area of a cube given the edge length.
def surface_area (e : ℝ) : ℝ := 6 * (e * e)

-- The theorem to prove.
theorem cube_surface_area (a : ℝ) : surface_area (edge_length a) = 216 * (a * a) := 
  sorry

end cube_surface_area_l67_67856


namespace marble_count_l67_67746

theorem marble_count (r g b : ℕ) (h1 : g + b = 6) (h2 : r + b = 8) (h3 : r + g = 4) : r + g + b = 9 :=
sorry

end marble_count_l67_67746


namespace square_side_length_increase_l67_67027

variables {a x : ℝ}

theorem square_side_length_increase 
  (h : (a * (1 + x / 100) * 1.8)^2 = (1 + 159.20000000000002 / 100) * (a^2 + (a * (1 + x / 100))^2)) : 
  x = 100 :=
by sorry

end square_side_length_increase_l67_67027


namespace complex_abs_of_sqrt_l67_67212

variable (z : ℂ)

theorem complex_abs_of_sqrt (h : z^2 = 16 - 30 * Complex.I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_of_sqrt_l67_67212


namespace smallest_y_in_geometric_sequence_l67_67840

theorem smallest_y_in_geometric_sequence (x y z r : ℕ) (h1 : y = x * r) (h2 : z = x * r^2) (h3 : xyz = 125) : y = 5 :=
by sorry

end smallest_y_in_geometric_sequence_l67_67840


namespace probability_sibling_pair_l67_67996

-- Define the necessary constants for the problem.
def B : ℕ := 500 -- Number of business students
def L : ℕ := 800 -- Number of law students
def S : ℕ := 30  -- Number of sibling pairs

-- State the theorem representing the mathematical proof problem
theorem probability_sibling_pair :
  (S : ℝ) / (B * L) = 0.000075 := sorry

end probability_sibling_pair_l67_67996


namespace minimum_width_l67_67333

theorem minimum_width (w : ℝ) (h_area : w * (w + 15) ≥ 200) : w ≥ 10 :=
by
  sorry

end minimum_width_l67_67333


namespace positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l67_67765

-- Definitions for the conditions
def eq1 (x y : ℝ) := x + 2 * y = 6
def eq2 (x y m : ℝ) := x - 2 * y + m * x + 5 = 0

-- Theorem for part (1)
theorem positive_integer_solutions :
  {x y : ℕ} → eq1 x y → (x = 4 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

-- Theorem for part (2)
theorem value_of_m_when_sum_is_zero (x y : ℝ) (h : x + y = 0) :
  eq1 x y → ∃ m : ℝ, eq2 x y m → m = -13/6 :=
sorry

-- Theorem for part (3)
theorem fixed_solution (m : ℝ) : eq2 0 2.5 m :=
sorry

-- Theorem for part (4)
theorem integer_values_of_m (x : ℤ) :
  (∃ y : ℤ, eq1 x y ∧ ∃ m : ℤ, eq2 x y m) → m = -1 ∨ m = -3 :=
sorry

end positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l67_67765


namespace fraction_is_square_l67_67269

theorem fraction_is_square (a b : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) 
  (hdiv : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_square_l67_67269


namespace women_in_luxury_suites_count_l67_67098

noncomputable def passengers : ℕ := 300
noncomputable def percentage_women : ℝ := 70 / 100
noncomputable def percentage_luxury : ℝ := 15 / 100

noncomputable def women_on_ship : ℝ := passengers * percentage_women
noncomputable def women_in_luxury_suites : ℝ := women_on_ship * percentage_luxury

theorem women_in_luxury_suites_count : 
  round women_in_luxury_suites = 32 :=
by sorry

end women_in_luxury_suites_count_l67_67098


namespace find_num_yoYos_l67_67498

variables (x y z w : ℕ)

def stuffed_animals_frisbees_puzzles := x + y + w = 80
def total_prizes := x + y + z + w + 180 + 60
def cars_and_robots := 180 + 60 = x + y + z + w + 15

theorem find_num_yoYos 
(h1 : stuffed_animals_frisbees_puzzles x y w)
(h2 : total_prizes = 300)
(h3 : cars_and_robots x y z w) : z = 145 :=
sorry

end find_num_yoYos_l67_67498


namespace bridget_block_collection_l67_67442

-- Defining the number of groups and blocks per group.
def num_groups : ℕ := 82
def blocks_per_group : ℕ := 10

-- Defining the total number of blocks calculation.
def total_blocks : ℕ := num_groups * blocks_per_group

-- Theorem stating the total number of blocks is 820.
theorem bridget_block_collection : total_blocks = 820 :=
  by
  sorry

end bridget_block_collection_l67_67442


namespace simplify_fraction_l67_67429

theorem simplify_fraction (h1 : 90 = 2 * 3^2 * 5) (h2 : 150 = 2 * 3 * 5^2) : (90 / 150 : ℚ) = 3 / 5 := by
  sorry

end simplify_fraction_l67_67429


namespace box_made_by_Bellini_or_son_l67_67057

-- Definitions of the conditions
variable (B : Prop) -- Bellini made the box
variable (S : Prop) -- Bellini's son made the box
variable (inscription_true : Prop) -- The inscription "I made this box" is truthful

-- The problem statement in Lean: Prove that B or S given the inscription is true
theorem box_made_by_Bellini_or_son (B S inscription_true : Prop) (h1 : inscription_true → (B ∨ S)) : B ∨ S :=
by
  sorry

end box_made_by_Bellini_or_son_l67_67057


namespace circumscribed_circle_area_l67_67315

noncomputable def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area (s : ℝ) (hs : s = 15) : circle_area (circumradius s) = 75 * Real.pi :=
by
  sorry

end circumscribed_circle_area_l67_67315


namespace margarets_mean_score_l67_67751

noncomputable def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

open List

theorem margarets_mean_score :
  let scores := [86, 88, 91, 93, 95, 97, 99, 100]
  let cyprians_mean := 92
  let num_scores := 8
  let cyprians_scores := 4
  let margarets_scores := num_scores - cyprians_scores
  (scores.sum - cyprians_scores * cyprians_mean) / margarets_scores = 95.25 :=
by
  sorry

end margarets_mean_score_l67_67751


namespace part1_part2_l67_67092

open Set

variable {α : Type*} [LinearOrderedField α]

def A : Set α := { x | abs (x - 1) ≤ 1 }
def B (a : α) : Set α := { x | x ≥ a }

theorem part1 {x : α} : x ∈ (A ∩ B 1) ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 {a : α} : (A ⊆ B a) ↔ a ≤ 0 := by
  sorry

end part1_part2_l67_67092


namespace pancakes_needed_l67_67715

def short_stack_pancakes : ℕ := 3
def big_stack_pancakes : ℕ := 5
def short_stack_customers : ℕ := 9
def big_stack_customers : ℕ := 6

theorem pancakes_needed : (short_stack_customers * short_stack_pancakes + big_stack_customers * big_stack_pancakes) = 57 :=
by
  sorry

end pancakes_needed_l67_67715


namespace other_number_l67_67879

theorem other_number (a b : ℝ) (h : a = 0.650) (h2 : a = b + 0.525) : b = 0.125 :=
sorry

end other_number_l67_67879


namespace perimeter_ratio_l67_67851

variables (K T k R r : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (r = 2 * T / K)
def condition2 : Prop := (2 * T = R * k)

-- The statement we want to prove
theorem perimeter_ratio :
  condition1 K T r →
  condition2 T R k →
  K / k = R / r :=
by
  intros h1 h2
  sorry

end perimeter_ratio_l67_67851


namespace math_problem_l67_67819

theorem math_problem (x : ℝ) (h : x = 0.18 * 4750) : 1.5 * x = 1282.5 :=
by
  sorry

end math_problem_l67_67819


namespace july_birth_percentage_l67_67897

theorem july_birth_percentage (total : ℕ) (july : ℕ) (h1 : total = 150) (h2 : july = 18) : (july : ℚ) / total * 100 = 12 := sorry

end july_birth_percentage_l67_67897


namespace smallest_number_divisible_by_618_3648_60_l67_67631

theorem smallest_number_divisible_by_618_3648_60 :
  ∃ n : ℕ, (∀ m, (m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0 → m = 1038239) :=
by
  use 1038239
  sorry

end smallest_number_divisible_by_618_3648_60_l67_67631


namespace max_value_f_max_value_f_at_13_l67_67719

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_f : ∀ x : ℝ, f x ≤ 1 / 3 := by
  sorry

theorem max_value_f_at_13 : ∃ x : ℝ, f x = 1 / 3 := by
  sorry

end max_value_f_max_value_f_at_13_l67_67719


namespace find_b_l67_67023

theorem find_b (a b c : ℝ) (h1 : a + b + c = 150) (h2 : a + 10 = c^2) (h3 : b - 5 = c^2) : 
  b = (1322 - 2 * Real.sqrt 1241) / 16 := 
by 
  sorry

end find_b_l67_67023


namespace other_number_is_286_l67_67888

theorem other_number_is_286 (a b hcf lcm : ℕ) (h_hcf : hcf = 26) (h_lcm : lcm = 2310) (h_one_num : a = 210) 
  (rel : lcm * hcf = a * b) : b = 286 :=
by
  sorry

end other_number_is_286_l67_67888


namespace xiao_ming_valid_paths_final_valid_paths_l67_67467

-- Definitions from conditions
def paths_segments := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
def initial_paths := 256
def invalid_paths := 64

-- Theorem statement
theorem xiao_ming_valid_paths : initial_paths - invalid_paths = 192 :=
by sorry

theorem final_valid_paths : 192 * 2 = 384 :=
by sorry

end xiao_ming_valid_paths_final_valid_paths_l67_67467


namespace polygon_interior_angles_sum_l67_67885

theorem polygon_interior_angles_sum (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := 
by sorry

end polygon_interior_angles_sum_l67_67885


namespace inequality_solution_l67_67566

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

theorem inequality_solution :
  -3 < x ∧ x < -1 ↔ f (x^2 - 3) < f (x - 1) :=
sorry

end inequality_solution_l67_67566


namespace unique_solution_of_system_l67_67191

theorem unique_solution_of_system :
  ∀ (a : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) →
  ((a = 1 ∧ ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃ x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0)) :=
by
  sorry

end unique_solution_of_system_l67_67191


namespace sun_volume_exceeds_moon_volume_by_387_cubed_l67_67358

/-- Given Sun's distance to Earth is 387 times greater than Moon's distance to Earth. 
Given diameters:
- Sun's diameter: D_s
- Moon's diameter: D_m
Formula for volume of a sphere: V = (4/3) * pi * R^3
Derive that the Sun's volume exceeds the Moon's volume by 387^3 times. -/
theorem sun_volume_exceeds_moon_volume_by_387_cubed
  (D_s D_m : ℝ)
  (h : D_s = 387 * D_m) :
  (4/3) * Real.pi * (D_s / 2)^3 = 387^3 * (4/3) * Real.pi * (D_m / 2)^3 := by
  sorry

end sun_volume_exceeds_moon_volume_by_387_cubed_l67_67358


namespace sqrt_computation_l67_67659

theorem sqrt_computation : 
  Real.sqrt ((35 * 34 * 33 * 32) + Nat.factorial 4) = 1114 := by
sorry

end sqrt_computation_l67_67659


namespace geometric_sequence_fifth_term_l67_67055

theorem geometric_sequence_fifth_term 
    (a₁ : ℕ) (a₄ : ℕ) (r : ℕ) (a₅ : ℕ)
    (h₁ : a₁ = 3) (h₂ : a₄ = 240) 
    (h₃ : a₄ = a₁ * r^3) 
    (h₄ : a₅ = a₁ * r^4) : 
    a₅ = 768 :=
by
  sorry

end geometric_sequence_fifth_term_l67_67055


namespace dot_product_conditioned_l67_67527

variables (a b : ℝ×ℝ)

def condition1 : Prop := 2 • a + b = (1, 6)
def condition2 : Prop := a + 2 • b = (-4, 9)
def dot_product (u v : ℝ×ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_conditioned :
  condition1 a b ∧ condition2 a b → dot_product a b = -2 :=
by
  sorry

end dot_product_conditioned_l67_67527


namespace focus_of_parabola_x_squared_eq_neg_4_y_l67_67957

theorem focus_of_parabola_x_squared_eq_neg_4_y:
  (∃ F : ℝ × ℝ, (F = (0, -1)) ∧ (∀ x y : ℝ, x^2 = -4 * y → F = (0, y + 1))) :=
sorry

end focus_of_parabola_x_squared_eq_neg_4_y_l67_67957


namespace sampling_method_is_stratified_l67_67375

-- Given conditions
def unit_population : ℕ := 500 + 1000 + 800
def elderly_ratio : ℕ := 5
def middle_aged_ratio : ℕ := 10
def young_ratio : ℕ := 8
def total_selected : ℕ := 230

-- Prove that the sampling method used is stratified sampling
theorem sampling_method_is_stratified :
  (500 + 1000 + 800 = unit_population) ∧
  (total_selected = 230) ∧
  (500 * 230 / unit_population = elderly_ratio) ∧
  (1000 * 230 / unit_population = middle_aged_ratio) ∧
  (800 * 230 / unit_population = young_ratio) →
  sampling_method = stratified_sampling :=
by
  sorry

end sampling_method_is_stratified_l67_67375


namespace goods_train_length_l67_67861

noncomputable def length_of_goods_train (speed_first_train_kmph speed_goods_train_kmph time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := speed_first_train_kmph + speed_goods_train_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5.0 / 18.0)
  relative_speed_mps * (time_seconds : ℝ)

theorem goods_train_length
  (speed_first_train_kmph : ℕ) (speed_goods_train_kmph : ℕ) (time_seconds : ℕ) 
  (h1 : speed_first_train_kmph = 50)
  (h2 : speed_goods_train_kmph = 62)
  (h3 : time_seconds = 9) :
  length_of_goods_train speed_first_train_kmph speed_goods_train_kmph time_seconds = 280 :=
  sorry

end goods_train_length_l67_67861


namespace fraction_after_adding_liters_l67_67120

-- Given conditions
variables (c w : ℕ)
variables (h1 : w = c / 3)
variables (h2 : (w + 5) / c = 2 / 5)

-- The proof statement
theorem fraction_after_adding_liters (h1 : w = c / 3) (h2 : (w + 5) / c = 2 / 5) : 
  (w + 9) / c = 34 / 75 :=
sorry -- Proof omitted

end fraction_after_adding_liters_l67_67120


namespace fido_yard_area_reach_l67_67015

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end fido_yard_area_reach_l67_67015


namespace zoe_calories_l67_67393

theorem zoe_calories 
  (s : ℕ) (y : ℕ) (c_s : ℕ) (c_y : ℕ)
  (s_eq : s = 12) (y_eq : y = 6) (cs_eq : c_s = 4) (cy_eq : c_y = 17) :
  s * c_s + y * c_y = 150 :=
by
  sorry

end zoe_calories_l67_67393


namespace tricycle_wheel_count_l67_67808

theorem tricycle_wheel_count (bicycles wheels_per_bicycle tricycles total_wheels : ℕ)
  (h1 : bicycles = 16)
  (h2 : wheels_per_bicycle = 2)
  (h3 : tricycles = 7)
  (h4 : total_wheels = 53)
  (h5 : total_wheels = (bicycles * wheels_per_bicycle) + (tricycles * (3 : ℕ))) : 
  (3 : ℕ) = 3 := by
  sorry

end tricycle_wheel_count_l67_67808


namespace factor_quadratic_l67_67877

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l67_67877


namespace profit_percent_approx_l67_67410

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 30
noncomputable def selling_price : ℝ := 300

noncomputable def cost_price : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit / cost_price) * 100

theorem profit_percent_approx :
  purchase_price = 225 ∧ 
  overhead_expenses = 30 ∧ 
  selling_price = 300 → 
  abs (profit_percent - 17.65) < 0.01 := 
by 
  -- Proof omitted
  sorry

end profit_percent_approx_l67_67410


namespace inradius_circumradius_le_height_l67_67336

theorem inradius_circumradius_le_height
    {α β γ : ℝ}
    (hα : 0 < α ∧ α ≤ 90)
    (hβ : 0 < β ∧ β ≤ 90)
    (hγ : 0 < γ ∧ γ ≤ 90)
    (α_ge_β : α ≥ β)
    (β_ge_γ : β ≥ γ)
    {r R h : ℝ} :
  r + R ≤ h := 
sorry

end inradius_circumradius_le_height_l67_67336


namespace sum_of_abs_arithmetic_sequence_l67_67376

theorem sum_of_abs_arithmetic_sequence {a_n : ℕ → ℤ} {S_n : ℕ → ℤ} 
  (hS3 : S_n 3 = 21) (hS9 : S_n 9 = 9) :
  ∃ (T_n : ℕ → ℤ), 
    (∀ (n : ℕ), n ≤ 5 → T_n n = -n^2 + 10 * n) ∧
    (∀ (n : ℕ), n ≥ 6 → T_n n = n^2 - 10 * n + 50) :=
sorry

end sum_of_abs_arithmetic_sequence_l67_67376


namespace total_buttons_l67_67927

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end total_buttons_l67_67927


namespace remainder_correct_l67_67321

noncomputable def p : Polynomial ℝ := Polynomial.C 3 * Polynomial.X^5 + Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X + Polynomial.C 8
noncomputable def d : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) ^ 2
noncomputable def r : Polynomial ℝ := Polynomial.C 16 * Polynomial.X - Polynomial.C 8 

theorem remainder_correct : (p % d) = r := by sorry

end remainder_correct_l67_67321


namespace harriet_trip_time_l67_67869

theorem harriet_trip_time
  (speed_AB : ℕ := 100)
  (speed_BA : ℕ := 150)
  (total_trip_time : ℕ := 5)
  (time_threshold : ℕ := 180) :
  let D := (speed_AB * speed_BA * total_trip_time) / (speed_AB + speed_BA)
  let time_AB := D / speed_AB
  let time_AB_min := time_AB * 60
  time_AB_min = time_threshold :=
by
  sorry

end harriet_trip_time_l67_67869


namespace solve_car_production_l67_67164

def car_production_problem : Prop :=
  ∃ (NorthAmericaCars : ℕ) (TotalCars : ℕ) (EuropeCars : ℕ),
    NorthAmericaCars = 3884 ∧
    TotalCars = 6755 ∧
    EuropeCars = TotalCars - NorthAmericaCars ∧
    EuropeCars = 2871

theorem solve_car_production : car_production_problem := by
  sorry

end solve_car_production_l67_67164


namespace compute_expression_l67_67736

/-- Definitions of parts of the expression --/
def expr1 := 6 ^ 2
def expr2 := 4 * 5
def expr3 := 2 ^ 3
def expr4 := 4 ^ 2 / 2

/-- Main statement to prove --/
theorem compute_expression : expr1 + expr2 - expr3 + expr4 = 56 := 
by
  sorry

end compute_expression_l67_67736


namespace complement_union_l67_67570

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end complement_union_l67_67570


namespace employee_n_salary_l67_67665

theorem employee_n_salary (m n : ℝ) (h1 : m = 1.2 * n) (h2 : m + n = 594) :
  n = 270 :=
sorry

end employee_n_salary_l67_67665


namespace largest_possible_value_l67_67985

variable (a b : ℝ)

theorem largest_possible_value (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) :
  2 * a + b ≤ 5 :=
sorry

end largest_possible_value_l67_67985


namespace circle_value_a_l67_67390

noncomputable def represents_circle (a : ℝ) (x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

theorem circle_value_a {a : ℝ} (h : ∀ x y : ℝ, represents_circle a x y) :
  a = -1 :=
by
  sorry

end circle_value_a_l67_67390


namespace puppy_food_total_correct_l67_67205

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end puppy_food_total_correct_l67_67205


namespace smallest_solution_l67_67756

theorem smallest_solution (x : ℝ) (h : x * |x| = 3 * x - 2) : 
  x = 1 ∨ x = 2 ∨ x = (-(3 + Real.sqrt 17)) / 2 :=
by
  sorry

end smallest_solution_l67_67756


namespace find_a_l67_67217

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x ^ 2) - x)

theorem find_a (a : ℝ) :
  (∀ (x : ℝ), f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end find_a_l67_67217


namespace infinite_series_sum_l67_67661

noncomputable def inf_series (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), if n = 1 then 1 / (b * a)
  else if n % 2 = 0 then 1 / ((↑(n - 1) * a - b) * (↑n * a - b))
  else 1 / ((↑(n - 1) * a + b) * (↑n * a - b))

theorem infinite_series_sum (a b : ℝ) 
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) :
  inf_series a b = 1 / (a * b) :=
sorry

end infinite_series_sum_l67_67661


namespace nandan_gain_l67_67621

theorem nandan_gain (x t : ℝ) (nandan_gain krishan_gain total_gain : ℝ)
  (h1 : krishan_gain = 12 * x * t)
  (h2 : nandan_gain = x * t)
  (h3 : total_gain = nandan_gain + krishan_gain)
  (h4 : total_gain = 78000) :
  nandan_gain = 6000 :=
by
  -- Proof goes here
  sorry

end nandan_gain_l67_67621


namespace intersection_eq_l67_67596

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def M : Set ℝ := { x | -1/2 < x ∧ x < 1/2 }
def N : Set ℝ := { x | 0 ≤ x ∧ x * x ≤ x }

theorem intersection_eq :
  M ∩ N = { x | 0 ≤ x ∧ x < 1/2 } := by
  sorry

end intersection_eq_l67_67596


namespace factorize_polynomial_l67_67504

theorem factorize_polynomial (x y : ℝ) :
  3 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 3 * (x + y) ^ 2 :=
by
  sorry

end factorize_polynomial_l67_67504


namespace ratio_A_B_l67_67535

noncomputable def A : ℝ := ∑' n : ℕ, if n % 4 = 0 then 0 else 1 / (n:ℝ) ^ 2
noncomputable def B : ℝ := ∑' k : ℕ, (-1)^(k+1) / (4 * (k:ℝ)) ^ 2

theorem ratio_A_B : A / B = 32 := by
  -- proof here
  sorry

end ratio_A_B_l67_67535


namespace compare_logarithms_l67_67768

theorem compare_logarithms (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3) 
                           (h2 : b = (Real.log 2 / Real.log 3)^2) 
                           (h3 : c = Real.log (2/3) / Real.log 4) : c < b ∧ b < a :=
by
  sorry

end compare_logarithms_l67_67768


namespace solution_set_of_inequality_l67_67938

variable {a x : ℝ}

theorem solution_set_of_inequality (h : 2 * a + 1 < 0) : 
  {x : ℝ | x^2 - 4 * a * x - 5 * a^2 > 0} = {x | x < 5 * a ∨ x > -a} := by
  sorry

end solution_set_of_inequality_l67_67938


namespace square_side_length_is_10_l67_67086

-- Define the side lengths of the original squares
def side_length1 : ℝ := 8
def side_length2 : ℝ := 6

-- Define the areas of the original squares
def area1 : ℝ := side_length1^2
def area2 : ℝ := side_length2^2

-- Define the total area of the combined squares
def total_area : ℝ := area1 + area2

-- Define the side length of the new square
def side_length_new_square : ℝ := 10

-- Theorem statement to prove that the side length of the new square is 10 cm
theorem square_side_length_is_10 : side_length_new_square^2 = total_area := by
  sorry

end square_side_length_is_10_l67_67086


namespace decks_left_is_3_l67_67798

-- Given conditions
def price_per_deck := 2
def total_decks_start := 5
def money_earned := 4

-- The number of decks sold
def decks_sold := money_earned / price_per_deck

-- The number of decks left
def decks_left := total_decks_start - decks_sold

-- The theorem to prove 
theorem decks_left_is_3 : decks_left = 3 :=
by
  -- Here we put the steps to prove
  sorry

end decks_left_is_3_l67_67798


namespace line_tangent_to_curve_iff_a_zero_l67_67635

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def l (x a : ℝ) := 2 * x + a

theorem line_tangent_to_curve_iff_a_zero (a : ℝ) :
  (∃ x₀ : ℝ, deriv f x₀ = 2 ∧ f x₀ = l x₀ a) → a = 0 :=
sorry

end line_tangent_to_curve_iff_a_zero_l67_67635


namespace second_smallest_packs_hot_dogs_l67_67883

theorem second_smallest_packs_hot_dogs (n : ℕ) :
  (∃ k : ℕ, n = 5 * k + 3) →
  n > 0 →
  ∃ m : ℕ, m < n ∧ (∃ k2 : ℕ, m = 5 * k2 + 3) →
  n = 8 :=
by
  sorry

end second_smallest_packs_hot_dogs_l67_67883


namespace exists_polynomial_P_l67_67145

open Int Nat

/-- Define a predicate for a value is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Define the polynomial P(x, y, z) --/
noncomputable def P (x y z : ℕ) : ℤ := 
  (1 - 2013 * (z - 1) * (z - 2)) * 
  ((x + y - 1) * (x + y - 1) + 2 * y - 2 + z)

/-- The main theorem to prove --/
theorem exists_polynomial_P :
  ∃ (P : ℕ → ℕ → ℕ → ℤ), 
  (∀ n : ℕ, (¬ is_square n) ↔ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) := 
sorry

end exists_polynomial_P_l67_67145


namespace average_donation_proof_l67_67090

noncomputable def average_donation (total_people : ℝ) (donated_200 : ℝ) (donated_100 : ℝ) (donated_50 : ℝ) : ℝ :=
  let proportion_200 := donated_200 / total_people
  let proportion_100 := donated_100 / total_people
  let proportion_50 := donated_50 / total_people
  let total_donation := (200 * proportion_200) + (100 * proportion_100) + (50 * proportion_50)
  total_donation

theorem average_donation_proof 
  (total_people : ℝ)
  (donated_200 donated_100 donated_50 : ℝ)
  (h1 : proportion_200 = 1 / 10)
  (h2 : proportion_100 = 3 / 4)
  (h3 : proportion_50 = 1 - proportion_200 - proportion_100) :
  average_donation total_people donated_200 donated_100 donated_50 = 102.5 :=
  by 
    sorry

end average_donation_proof_l67_67090


namespace find_smallest_c_plus_d_l67_67902

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end find_smallest_c_plus_d_l67_67902


namespace value_of_expression_l67_67714

theorem value_of_expression (a b : ℝ) (h : a + b = 4) : a^2 + 2 * a * b + b^2 = 16 := by
  sorry

end value_of_expression_l67_67714


namespace domain_f_l67_67229

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5 * x + 6)

theorem domain_f :
  {x : ℝ | f x ≠ f x} = {x : ℝ | (x < 2) ∨ (2 < x ∧ x < 3) ∨ (3 < x)} :=
by sorry

end domain_f_l67_67229


namespace alley_width_l67_67364

theorem alley_width (ℓ : ℝ) (m : ℝ) (n : ℝ): ℓ * (1 / 2 + Real.cos (70 * Real.pi / 180)) = ℓ * (Real.cos (60 * Real.pi / 180)) + ℓ * (Real.cos (70 * Real.pi / 180)) := by
  sorry

end alley_width_l67_67364


namespace reflection_coordinates_l67_67626

-- Define the original coordinates of point M
def original_point : (ℝ × ℝ) := (3, -4)

-- Define the function to reflect a point across the x-axis
def reflect_across_x_axis (p: ℝ × ℝ) : (ℝ × ℝ) :=
  (p.1, -p.2)

-- State the theorem to prove the coordinates after reflection
theorem reflection_coordinates :
  reflect_across_x_axis original_point = (3, 4) :=
by
  sorry

end reflection_coordinates_l67_67626


namespace no_valid_base_l67_67487

theorem no_valid_base (b : ℤ) (n : ℤ) : b^2 + 2*b + 2 ≠ n^2 := by
  sorry

end no_valid_base_l67_67487


namespace consecutive_product_not_mth_power_l67_67065

theorem consecutive_product_not_mth_power (n m k : ℕ) :
  ¬ ∃ k, (n - 1) * n * (n + 1) = k^m := 
sorry

end consecutive_product_not_mth_power_l67_67065


namespace combined_value_of_cookies_sold_l67_67609

theorem combined_value_of_cookies_sold:
  ∀ (total_boxes : ℝ) (plain_boxes : ℝ) (price_plain : ℝ) (price_choco : ℝ),
    total_boxes = 1585 →
    plain_boxes = 793.125 →
    price_plain = 0.75 →
    price_choco = 1.25 →
    (plain_boxes * price_plain + (total_boxes - plain_boxes) * price_choco) = 1584.6875 :=
by
  intros total_boxes plain_boxes price_plain price_choco
  intro h1 h2 h3 h4
  sorry

end combined_value_of_cookies_sold_l67_67609


namespace total_cost_of_color_drawing_l67_67228

def cost_bwch_drawing : ℕ := 160
def bwch_to_color_cost_multiplier : ℝ := 1.5

theorem total_cost_of_color_drawing 
  (cost_bwch : ℕ)
  (bwch_to_color_mult : ℝ)
  (h₁ : cost_bwch = 160)
  (h₂ : bwch_to_color_mult = 1.5) :
  cost_bwch * bwch_to_color_mult = 240 := 
  by
    sorry

end total_cost_of_color_drawing_l67_67228


namespace citrus_grove_total_orchards_l67_67103

theorem citrus_grove_total_orchards (lemons_orchards oranges_orchards grapefruits_orchards limes_orchards total_orchards : ℕ) 
  (h1 : lemons_orchards = 8) 
  (h2 : oranges_orchards = lemons_orchards / 2) 
  (h3 : grapefruits_orchards = 2) 
  (h4 : limes_orchards = grapefruits_orchards) 
  (h5 : total_orchards = lemons_orchards + oranges_orchards + grapefruits_orchards + limes_orchards) : 
  total_orchards = 16 :=
by 
  sorry

end citrus_grove_total_orchards_l67_67103


namespace magnitude_vec_sum_l67_67448

open Real

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

theorem magnitude_vec_sum
    (a b : ℝ × ℝ)
    (h_angle : ∃ θ, θ = 150 * (π / 180) ∧ cos θ = cos (5 * π / 6))
    (h_norm_a : ‖a‖ = sqrt 3)
    (h_norm_b : ‖b‖ = 2) :
  ‖(2 * a.1 + b.1, 2 * a.2 + b.2)‖ = 2 :=
  by
  sorry

end magnitude_vec_sum_l67_67448


namespace tom_gave_8_boxes_l67_67031

-- Define the given conditions and the question in terms of variables
variables (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (boxes_given : ℕ)

-- Specify the actual values for the given problem
def tom_initial_pieces := total_boxes * pieces_per_box
def pieces_given := tom_initial_pieces - pieces_left
def calculated_boxes_given := pieces_given / pieces_per_box

-- Prove the number of boxes Tom gave to his little brother
theorem tom_gave_8_boxes
  (h1 : total_boxes = 14)
  (h2 : pieces_per_box = 3)
  (h3 : pieces_left = 18)
  (h4 : calculated_boxes_given = boxes_given) :
  boxes_given = 8 :=
by
  sorry

end tom_gave_8_boxes_l67_67031


namespace trajectory_of_moving_circle_l67_67896

noncomputable def ellipse_trajectory_eq (x y : ℝ) : Prop :=
  (x^2)/25 + (y^2)/9 = 1

theorem trajectory_of_moving_circle
  (x y : ℝ)
  (A : ℝ × ℝ)
  (C : ℝ × ℝ)
  (radius_C : ℝ)
  (hC : (x + 4)^2 + y^2 = 100)
  (hA : A = (4, 0))
  (radius_C_eq : radius_C = 10) :
  ellipse_trajectory_eq x y :=
sorry

end trajectory_of_moving_circle_l67_67896


namespace total_worth_all_crayons_l67_67267

def cost_of_crayons (packs: ℕ) (cost_per_pack: ℝ) : ℝ := packs * cost_per_pack

def discounted_cost (cost: ℝ) (discount_rate: ℝ) : ℝ := cost * (1 - discount_rate)

def tax_amount (cost: ℝ) (tax_rate: ℝ) : ℝ := cost * tax_rate

theorem total_worth_all_crayons : 
  let cost_per_pack := 2.5
  let discount_rate := 0.15
  let tax_rate := 0.07
  let packs_already_have := 4
  let packs_to_buy := 2
  let cost_two_packs := cost_of_crayons packs_to_buy cost_per_pack
  let discounted_two_packs := discounted_cost cost_two_packs discount_rate
  let tax_two_packs := tax_amount cost_two_packs tax_rate
  let total_cost_two_packs := discounted_two_packs + tax_two_packs
  let cost_four_packs := cost_of_crayons packs_already_have cost_per_pack
  cost_four_packs + total_cost_two_packs = 14.60 := 
by 
  sorry

end total_worth_all_crayons_l67_67267


namespace sequence_sum_S15_S22_S31_l67_67892

def sequence_sum (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | m + 1 => sequence_sum m + (-1)^m * (3 * (m + 1) - 1)

theorem sequence_sum_S15_S22_S31 :
  sequence_sum 15 + sequence_sum 22 - sequence_sum 31 = -57 := 
sorry

end sequence_sum_S15_S22_S31_l67_67892


namespace solve_inequality_l67_67454

theorem solve_inequality (a x : ℝ) (h : a < 0) :
  (56 * x^2 + a * x - a^2 < 0) ↔ (a / 8 < x ∧ x < -a / 7) :=
by
  sorry

end solve_inequality_l67_67454


namespace theo_cookies_eaten_in_9_months_l67_67974

-- Define the basic variable values as per the conditions
def cookiesPerTime : Nat := 25
def timesPerDay : Nat := 5
def daysPerMonth : Nat := 27
def numMonths : Nat := 9

-- Define the total number of cookies Theo can eat in 9 months
def totalCookiesIn9Months : Nat :=
  cookiesPerTime * timesPerDay * daysPerMonth * numMonths

-- The theorem stating the answer
theorem theo_cookies_eaten_in_9_months :
  totalCookiesIn9Months = 30375 := by
  -- Proof will go here
  sorry

end theo_cookies_eaten_in_9_months_l67_67974


namespace pure_imaginary_real_part_zero_l67_67663

-- Define the condition that the complex number a + i is a pure imaginary number.
def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.I * b

-- Define the complex number a + i.
def z (a : ℝ) : ℂ := a + Complex.I

-- The theorem states that if z is pure imaginary, then a = 0.
theorem pure_imaginary_real_part_zero (a : ℝ) (h : isPureImaginary (z a)) : a = 0 :=
by
  sorry

end pure_imaginary_real_part_zero_l67_67663


namespace jackson_weeks_of_school_l67_67210

def jackson_sandwich_per_week : ℕ := 2

def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2
def total_missed_sandwiches : ℕ := missed_wednesdays + missed_fridays

def total_sandwiches_eaten : ℕ := 69

def total_sandwiches_without_missing : ℕ := total_sandwiches_eaten + total_missed_sandwiches

def calculate_weeks_of_school (total_sandwiches : ℕ) (sandwiches_per_week : ℕ) : ℕ :=
total_sandwiches / sandwiches_per_week

theorem jackson_weeks_of_school : calculate_weeks_of_school total_sandwiches_without_missing jackson_sandwich_per_week = 36 :=
by
  sorry

end jackson_weeks_of_school_l67_67210


namespace find_x_l67_67568

def binop (a b : ℤ) : ℤ := a * b + a + b + 2

theorem find_x :
  ∃ x : ℤ, binop x 3 = 1 ∧ x = -1 :=
by
  sorry

end find_x_l67_67568


namespace orchard_trees_l67_67822

theorem orchard_trees (n : ℕ) (hn : n^2 + 146 = 7890) : 
    n^2 + 146 + 31 = 89^2 := by
  sorry

end orchard_trees_l67_67822


namespace proof_n_value_l67_67543

theorem proof_n_value (n : ℕ) (h : (9^n) * (9^n) * (9^n) * (9^n) * (9^n) = 81^5) : n = 2 :=
by
  sorry

end proof_n_value_l67_67543


namespace sum_of_ages_l67_67180

theorem sum_of_ages (a b c : ℕ) (twin : a = b) (product : a * b * c = 256) : a + b + c = 20 := by
  sorry

end sum_of_ages_l67_67180


namespace work_hours_l67_67351

-- Let h be the number of hours worked
def hours_worked (total_paid part_cost hourly_rate : ℕ) : ℕ :=
  (total_paid - part_cost) / hourly_rate

-- Given conditions
def total_paid : ℕ := 300
def part_cost : ℕ := 150
def hourly_rate : ℕ := 75

-- The statement to be proved
theorem work_hours :
  hours_worked total_paid part_cost hourly_rate = 2 :=
by
  -- Provide the proof here
  sorry

end work_hours_l67_67351


namespace find_set_B_l67_67841

set_option pp.all true

variable (A : Set ℤ) (B : Set ℤ)

theorem find_set_B (hA : A = {-2, 0, 1, 3})
                    (hB : B = {x | -x ∈ A ∧ 1 - x ∉ A}) :
  B = {-3, -1, 2} :=
by
  sorry

end find_set_B_l67_67841


namespace math_pages_l67_67952

def total_pages := 7
def reading_pages := 2

theorem math_pages : total_pages - reading_pages = 5 := by
  sorry

end math_pages_l67_67952


namespace reduced_price_per_kg_l67_67624

theorem reduced_price_per_kg (P R : ℝ) (Q : ℝ)
  (h1 : R = 0.80 * P)
  (h2 : Q * P = 1500)
  (h3 : (Q + 10) * R = 1500) : R = 30 :=
by
  sorry

end reduced_price_per_kg_l67_67624


namespace triangular_weight_60_l67_67830

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end triangular_weight_60_l67_67830


namespace number_of_new_books_l67_67703

-- Defining the given conditions
def adventure_books : ℕ := 24
def mystery_books : ℕ := 37
def used_books : ℕ := 18

-- Defining the total books and new books
def total_books : ℕ := adventure_books + mystery_books
def new_books : ℕ := total_books - used_books

-- Proving the number of new books
theorem number_of_new_books : new_books = 43 := by
  -- Here we need to show that the calculated number of new books equals 43
  sorry

end number_of_new_books_l67_67703


namespace AJHSMETL_19892_reappears_on_line_40_l67_67584
-- Import the entire Mathlib library

-- Define the conditions
def cycleLengthLetters : ℕ := 8
def cycleLengthDigits : ℕ := 5
def lcm_cycles : ℕ := Nat.lcm cycleLengthLetters cycleLengthDigits

-- Problem statement with proof to be filled in later
theorem AJHSMETL_19892_reappears_on_line_40 :
  lcm_cycles = 40 := 
by
  sorry

end AJHSMETL_19892_reappears_on_line_40_l67_67584


namespace range_of_f_l67_67658

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * Real.sin x + 6

theorem range_of_f :
  ∀ (x : ℝ), Real.sin x ≠ 2 → 
  (1 ≤ f x ∧ f x ≤ 11) :=
by 
  sorry

end range_of_f_l67_67658


namespace scientific_notation_of_32000000_l67_67723

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end scientific_notation_of_32000000_l67_67723


namespace value_of_x_minus_2y_l67_67710

theorem value_of_x_minus_2y (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 :=
sorry

end value_of_x_minus_2y_l67_67710


namespace gcd_1987_2025_l67_67698

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end gcd_1987_2025_l67_67698


namespace find_common_ratio_sum_arithmetic_sequence_l67_67422

-- Conditions
variable {a : ℕ → ℝ}   -- a_n is a numeric sequence
variable (S : ℕ → ℝ)   -- S_n is the sum of the first n terms
variable {q : ℝ}       -- q is the common ratio
variable (k : ℕ)

-- Given: a_n is a geometric sequence with common ratio q, q ≠ 1, q ≠ 0
variable (h_geometric : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (h_q_ne_zero : q ≠ 0)

-- Given: S_n = a_1 * (1 - q^n) / (1 - q) when q ≠ 1 and q ≠ 0
variable (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))

-- Given: a_5, a_3, a_4 form an arithmetic sequence, so 2a_3 = a_5 + a_4
variable (h_arithmetic : 2 * a 3 = a 5 + a 4)

-- Prove part 1: common ratio q is -2
theorem find_common_ratio (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : 2 * a 3 = a 5 + a 4) 
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0) : q = -2 :=
sorry

-- Prove part 2: S_(k+2), S_k, S_(k+1) form an arithmetic sequence
theorem sum_arithmetic_sequence (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0)
  (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))
  (k : ℕ) : S (k + 2) + S k = 2 * S (k + 1) :=
sorry

end find_common_ratio_sum_arithmetic_sequence_l67_67422


namespace base_height_calculation_l67_67183

noncomputable def height_of_sculpture : ℚ := 2 + 5/6 -- 2 feet 10 inches in feet
noncomputable def total_height : ℚ := 3.5
noncomputable def height_of_base : ℚ := 2/3

theorem base_height_calculation (h1 : height_of_sculpture = 17/6) (h2 : total_height = 21/6):
  height_of_base = total_height - height_of_sculpture := by
  sorry

end base_height_calculation_l67_67183


namespace business_proof_l67_67151

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end business_proof_l67_67151


namespace engineers_crimson_meet_in_tournament_l67_67564

noncomputable def probability_engineers_crimson_meet : ℝ := 
  1 - Real.exp (-1)

theorem engineers_crimson_meet_in_tournament :
  (∃ (n : ℕ), n = 128) → 
  (∀ (i : ℕ), i < 128 → (∀ (j : ℕ), j < 128 → i ≠ j → ∃ (p : ℝ), p = probability_engineers_crimson_meet)) :=
sorry

end engineers_crimson_meet_in_tournament_l67_67564


namespace expected_value_twelve_sided_die_l67_67792

/-- A twelve-sided die has its faces numbered from 1 to 12.
    Prove that the expected value of a roll of this die is 6.5. -/
theorem expected_value_twelve_sided_die : 
  (1 / 12 : ℚ) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_twelve_sided_die_l67_67792


namespace line_intersects_plane_at_angle_l67_67942

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end line_intersects_plane_at_angle_l67_67942


namespace distance_traveled_is_6000_l67_67709

-- Define the conditions and the question in Lean 4
def footprints_per_meter_Pogo := 4
def footprints_per_meter_Grimzi := 3 / 6
def combined_total_footprints := 27000

theorem distance_traveled_is_6000 (D : ℕ) :
  footprints_per_meter_Pogo * D + footprints_per_meter_Grimzi * D = combined_total_footprints →
  D = 6000 :=
by
  sorry

end distance_traveled_is_6000_l67_67709


namespace number_of_mappings_n_elements_l67_67105

theorem number_of_mappings_n_elements
  (A : Type) [Fintype A] [DecidableEq A] (n : ℕ) (h : 3 ≤ n) (f : A → A)
  (H1 : ∀ x : A, ∃ c : A, ∀ (i : ℕ), i ≥ n - 2 → f^[i] x = c)
  (H2 : ∃ x₁ x₂ : A, f^[n] x₁ ≠ f^[n] x₂) :
  ∃ m : ℕ, m = (2 * n - 5) * (n.factorial) / 2 :=
sorry

end number_of_mappings_n_elements_l67_67105


namespace cookies_with_flour_l67_67921

theorem cookies_with_flour (x: ℕ) (c1: ℕ) (c2: ℕ) (h: c1 = 18 ∧ c2 = 2 ∧ x = 9 * 5):
  x = 45 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end cookies_with_flour_l67_67921


namespace determine_radius_of_semicircle_l67_67144

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem determine_radius_of_semicircle :
  radius_of_semicircle 32.392033717615696 = 6.3 :=
by
  sorry

end determine_radius_of_semicircle_l67_67144


namespace maria_uses_666_blocks_l67_67972

theorem maria_uses_666_blocks :
  let original_volume := 15 * 12 * 7
  let interior_length := 15 - 2 * 1.5
  let interior_width := 12 - 2 * 1.5
  let interior_height := 7 - 1.5
  let interior_volume := interior_length * interior_width * interior_height
  let blocks_volume := original_volume - interior_volume
  blocks_volume = 666 :=
by
  sorry

end maria_uses_666_blocks_l67_67972


namespace negation_forall_pos_l67_67754

theorem negation_forall_pos (h : ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) :
  ∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0 :=
sorry

end negation_forall_pos_l67_67754


namespace fangfang_travel_time_l67_67854

theorem fangfang_travel_time (time_1_to_5 : ℕ) (start_floor end_floor : ℕ) (floors_1_to_5 : ℕ) (floors_2_to_7 : ℕ) :
  time_1_to_5 = 40 →
  floors_1_to_5 = 5 - 1 →
  floors_2_to_7 = 7 - 2 →
  end_floor = 7 →
  start_floor = 2 →
  (end_floor - start_floor) * (time_1_to_5 / floors_1_to_5) = 50 :=
by 
  sorry

end fangfang_travel_time_l67_67854


namespace days_vacuuming_l67_67606

theorem days_vacuuming (V : ℕ) (h1 : ∀ V, 130 = 30 * V + 40) : V = 3 :=
by
    have eq1 : 130 = 30 * V + 40 := h1 V
    sorry

end days_vacuuming_l67_67606


namespace triangle_expression_negative_l67_67886

theorem triangle_expression_negative {a b c : ℝ} (habc : a > 0 ∧ b > 0 ∧ c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a^2 + b^2 - c^2 - 2 * a * b < 0 :=
sorry

end triangle_expression_negative_l67_67886


namespace sum_constants_l67_67392

def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

theorem sum_constants (a b c : ℝ) (h : ∀ x : ℝ, -4 * x^2 + 20 * x - 88 = a * (x + b)^2 + c) : 
  a + b + c = -70.5 :=
sorry

end sum_constants_l67_67392


namespace sum_first_six_terms_l67_67137

variable (a1 q : ℤ)
variable (n : ℕ)

noncomputable def geometric_sum (a1 q : ℤ) (n : ℕ) : ℤ :=
  a1 * (1 - q^n) / (1 - q)

theorem sum_first_six_terms :
  geometric_sum (-1) 2 6 = 63 :=
sorry

end sum_first_six_terms_l67_67137


namespace area_of_square_field_l67_67863

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area of the square based on the side length
def square_area (side : ℝ) : ℝ := side * side

-- The theorem stating the area of a square with side length 15 meters
theorem area_of_square_field : square_area side_length = 225 := 
by 
  sorry

end area_of_square_field_l67_67863


namespace area_of_figure_l67_67676

noncomputable def area_enclosed : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi / 3), 2 * Real.sin x

theorem area_of_figure :
  area_enclosed = 3 := by
  sorry

end area_of_figure_l67_67676


namespace find_missing_number_l67_67330

-- Define the known values
def numbers : List ℕ := [1, 22, 24, 25, 26, 27, 2]
def specified_mean : ℕ := 20
def total_counts : ℕ := 8

-- The theorem statement
theorem find_missing_number : (∀ (x : ℕ), (List.sum (x :: numbers) = specified_mean * total_counts) → x = 33) :=
by
  sorry

end find_missing_number_l67_67330


namespace train_speed_is_60_kmph_l67_67203

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l67_67203


namespace melted_ice_cream_depth_l67_67511

noncomputable def ice_cream_depth : ℝ :=
  let r1 := 3 -- radius of the sphere
  let r2 := 10 -- radius of the cylinder
  let V_sphere := (4/3) * Real.pi * r1^3 -- volume of the sphere
  let V_cylinder h := Real.pi * r2^2 * h -- volume of the cylinder
  V_sphere / (Real.pi * r2^2)

theorem melted_ice_cream_depth :
  ice_cream_depth = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l67_67511


namespace boat_distance_downstream_l67_67813

theorem boat_distance_downstream (speed_boat_still: ℕ) (speed_stream: ℕ) (time: ℕ)
    (h1: speed_boat_still = 25)
    (h2: speed_stream = 5)
    (h3: time = 4) :
    (speed_boat_still + speed_stream) * time = 120 := 
sorry

end boat_distance_downstream_l67_67813


namespace flowers_given_l67_67006

theorem flowers_given (initial_flowers total_flowers flowers_given : ℕ) 
  (h1 : initial_flowers = 67) 
  (h2 : total_flowers = 90) 
  (h3 : total_flowers = initial_flowers + flowers_given) : 
  flowers_given = 23 :=
by {
  sorry
}

end flowers_given_l67_67006


namespace gumballs_per_pair_of_earrings_l67_67946

theorem gumballs_per_pair_of_earrings : 
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  (total_gumballs / total_earrings) = 9 :=
by
  -- Definitions
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  -- Theorem statement
  sorry

end gumballs_per_pair_of_earrings_l67_67946


namespace find_train_parameters_l67_67194

-- Definitions based on the problem statement
def bridge_length : ℕ := 1000
def time_total : ℕ := 60
def time_on_bridge : ℕ := 40
def speed_train (x : ℕ) := (40 * x = bridge_length)
def length_train (x y : ℕ) := (60 * x = bridge_length + y)

-- Stating the problem to be proved
theorem find_train_parameters (x y : ℕ) (h₁ : speed_train x) (h₂ : length_train x y) :
  x = 20 ∧ y = 200 :=
sorry

end find_train_parameters_l67_67194


namespace initial_saltwater_amount_l67_67488

variable (x y : ℝ)
variable (h1 : 0.04 * x = (x - y) * 0.1)
variable (h2 : ((x - y) * 0.1 + 300 * 0.04) / (x - y + 300) = 0.064)

theorem initial_saltwater_amount : x = 500 :=
by
  sorry

end initial_saltwater_amount_l67_67488


namespace find_largest_integer_l67_67667

theorem find_largest_integer : ∃ (x : ℤ), x < 120 ∧ x % 8 = 7 ∧ x = 119 := 
by
  use 119
  sorry

end find_largest_integer_l67_67667


namespace correct_choice_D_l67_67651

variable (a b : Line) (α : Plane)

-- Definitions for the conditions
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry  -- Definition of perpendicular
def is_parallel_line (l1 l2 : Line) : Prop := sorry  -- Definition of parallel lines
def is_parallel_plane (l : Line) (p : Plane) : Prop := sorry  -- Definition of line parallel to plane
def is_subset (l : Line) (p : Plane) : Prop := sorry  -- Definition of line being in a plane

-- The statement of the problem
theorem correct_choice_D :
  (is_parallel_plane a α) ∧ (is_subset b α) → (is_parallel_plane a α) := 
by 
  sorry

end correct_choice_D_l67_67651


namespace circle_equation_bisects_l67_67694

-- Define the given conditions
def circle1_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 1
def circle2_eq (x y : ℝ) : Prop := (x - 6)^2 + (y + 6)^2 = 9

-- Define the goal equation
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 = 81

-- The statement of the problem
theorem circle_equation_bisects (a r : ℝ) (h1 : ∀ x y, circle1_eq x y → circleC_eq x y) (h2 : ∀ x y, circle2_eq x y → circleC_eq x y):
  circleC_eq (a * r) 0 := sorry

end circle_equation_bisects_l67_67694


namespace arithmetic_mean_34_58_l67_67758

theorem arithmetic_mean_34_58 :
  (3 / 4 : ℚ) + (5 / 8 : ℚ) / 2 = 11 / 16 := sorry

end arithmetic_mean_34_58_l67_67758


namespace jessica_quarters_l67_67737

theorem jessica_quarters (initial_quarters borrowed_quarters remaining_quarters : ℕ)
  (h1 : initial_quarters = 8)
  (h2 : borrowed_quarters = 3) :
  remaining_quarters = initial_quarters - borrowed_quarters → remaining_quarters = 5 :=
by
  intro h3
  rw [h1, h2] at h3
  exact h3

end jessica_quarters_l67_67737


namespace problem1_problem2_l67_67211

theorem problem1 (a b : ℝ) : ((a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4) = a^8 * b^8 := 
by sorry

theorem problem2 (x : ℝ) : ((3 * x^3)^2 * x^5 - (-x^2)^6 / x) = 8 * x^11 :=
by sorry

end problem1_problem2_l67_67211


namespace smallest_three_digit_number_with_property_l67_67401

theorem smallest_three_digit_number_with_property :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ (∃ (n : ℕ), 317 ≤ n ∧ n ≤ 999 ∧ 1001 * a + 1 = n^2) ∧ a = 183 :=
by
  sorry

end smallest_three_digit_number_with_property_l67_67401


namespace ratio_of_surface_areas_of_spheres_l67_67958

theorem ratio_of_surface_areas_of_spheres (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) : 
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 9 := by
  sorry

end ratio_of_surface_areas_of_spheres_l67_67958


namespace chinese_carriage_problem_l67_67302

theorem chinese_carriage_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) :=
sorry

end chinese_carriage_problem_l67_67302


namespace no_infinite_non_constant_arithmetic_progression_with_powers_l67_67341

theorem no_infinite_non_constant_arithmetic_progression_with_powers (a b : ℕ) (b_ge_2 : b ≥ 2) : 
  ¬ ∃ (f : ℕ → ℕ) (d : ℕ), (∀ n : ℕ, f n = (a^(b + n*d)) ∧ b ≥ 2) := sorry

end no_infinite_non_constant_arithmetic_progression_with_powers_l67_67341


namespace john_overall_profit_l67_67585

-- Definitions based on conditions
def cost_grinder : ℕ := 15000
def cost_mobile : ℕ := 8000
def loss_percentage_grinder : ℚ := 4 / 100
def profit_percentage_mobile : ℚ := 15 / 100

-- Calculations based on the conditions
def loss_amount_grinder := cost_grinder * loss_percentage_grinder
def selling_price_grinder := cost_grinder - loss_amount_grinder
def profit_amount_mobile := cost_mobile * profit_percentage_mobile
def selling_price_mobile := cost_mobile + profit_amount_mobile
def total_cost_price := cost_grinder + cost_mobile
def total_selling_price := selling_price_grinder + selling_price_mobile

-- Overall profit calculation
def overall_profit := total_selling_price - total_cost_price

-- Proof statement to prove the overall profit
theorem john_overall_profit : overall_profit = 600 := by
  sorry

end john_overall_profit_l67_67585


namespace quadratic_roots_r6_s6_l67_67240

theorem quadratic_roots_r6_s6 (r s : ℝ) (h1 : r + s = 3 * Real.sqrt 2) (h2 : r * s = 4) : r^6 + s^6 = 648 := by
  sorry

end quadratic_roots_r6_s6_l67_67240


namespace pedoe_inequality_l67_67179

variables {a b c a' b' c' Δ Δ' : ℝ} {A A' : ℝ}

theorem pedoe_inequality :
  a' ^ 2 * (-a ^ 2 + b ^ 2 + c ^ 2) +
  b' ^ 2 * (a ^ 2 - b ^ 2 + c ^ 2) +
  c' ^ 2 * (a ^ 2 + b ^ 2 - c ^ 2) -
  16 * Δ * Δ' =
  2 * (b * c' - b' * c) ^ 2 +
  8 * b * b' * c * c' * (Real.sin ((A - A') / 2)) ^ 2 := sorry

end pedoe_inequality_l67_67179


namespace problem1_problem2_l67_67082

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

variable (a : ℝ) (x : ℝ)

-- Problem 1: Proving intersection of sets when a = 2
theorem problem1 (ha : a = 2) : (A a ∩ B a) = {x | 4 < x ∧ x < 5} :=
sorry

-- Problem 2: Proving the range of a for which B is a subset of A
theorem problem2 : {a | B a ⊆ A a} = {a | (1 < a ∧ a ≤ 3) ∨ a = -1} :=
sorry

end problem1_problem2_l67_67082


namespace base_8_subtraction_l67_67984

theorem base_8_subtraction : 
  let x := 0o1234   -- 1234 in base 8
  let y := 0o765    -- 765 in base 8
  let result := 0o225 -- 225 in base 8
  x - y = result := by sorry

end base_8_subtraction_l67_67984


namespace part_one_part_two_part_three_l67_67372

def numberOfWaysToPlaceBallsInBoxes : ℕ :=
  4 ^ 4

def numberOfWaysOneBoxEmpty : ℕ :=
  Nat.choose 4 2 * (Nat.factorial 4 / Nat.factorial 1)

def numberOfWaysTwoBoxesEmpty : ℕ :=
  (Nat.choose 4 1 * (Nat.factorial 4 / Nat.factorial 2)) + (Nat.choose 4 2 * (Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)))

theorem part_one : numberOfWaysToPlaceBallsInBoxes = 256 := by
  sorry

theorem part_two : numberOfWaysOneBoxEmpty = 144 := by
  sorry

theorem part_three : numberOfWaysTwoBoxesEmpty = 120 := by
  sorry

end part_one_part_two_part_three_l67_67372


namespace cubics_product_equals_1_over_1003_l67_67571

theorem cubics_product_equals_1_over_1003
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2007)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2006)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2007)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2006)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2007)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2006) :
  (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end cubics_product_equals_1_over_1003_l67_67571


namespace circle_equation_l67_67749

theorem circle_equation :
  ∃ (a : ℝ), (y - a)^2 + x^2 = 1 ∧ (1 - 0)^2 + (2 - a)^2 = 1 ∧
  ∀ a, (1 - 0)^2 + (2 - a)^2 = 1 → a = 2 →
  x^2 + (y - 2)^2 = 1 := by sorry

end circle_equation_l67_67749


namespace cuboid_first_dimension_l67_67562

theorem cuboid_first_dimension (x : ℕ)
  (h₁ : ∃ n : ℕ, n = 24) 
  (h₂ : ∃ a b c d e f g : ℕ, x = a ∧ 9 = b ∧ 12 = c ∧ a * b * c = d * e * f ∧ g = Nat.gcd b c ∧ f = (g^3) ∧ e = (n * f) ∧ d = 648) : 
  x = 6 :=
by
  sorry

end cuboid_first_dimension_l67_67562


namespace incorrect_judgment_D_l67_67788

theorem incorrect_judgment_D (p q : Prop) (hp : p = (2 + 3 = 5)) (hq : q = (5 < 4)) : 
  ¬((p ∧ q) ∧ (p ∨ q)) := by 
    sorry

end incorrect_judgment_D_l67_67788


namespace perimeter_of_square_l67_67301

theorem perimeter_of_square (s : ℝ) (h : s^2 = s * Real.sqrt 2) (h_ne_zero : s ≠ 0) :
    4 * s = 4 * Real.sqrt 2 := by
  sorry

end perimeter_of_square_l67_67301


namespace scientific_notation_coronavirus_diameter_l67_67439

theorem scientific_notation_coronavirus_diameter : 0.00000011 = 1.1 * 10^(-7) :=
by {
  sorry
}

end scientific_notation_coronavirus_diameter_l67_67439


namespace toy_factory_max_profit_l67_67326

theorem toy_factory_max_profit :
  ∃ x y : ℕ,    -- x: number of bears, y: number of cats
  15 * x + 10 * y ≤ 450 ∧    -- labor hours constraint
  20 * x + 5 * y ≤ 400 ∧     -- raw materials constraint
  80 * x + 45 * y = 2200 :=  -- total selling price
by
  sorry

end toy_factory_max_profit_l67_67326


namespace roots_in_interval_l67_67324

def P (x : ℝ) : ℝ := x^2014 - 100 * x + 1

theorem roots_in_interval : 
  ∀ x : ℝ, P x = 0 → (1/100) ≤ x ∧ x ≤ 100^(1 / 2013) := 
  sorry

end roots_in_interval_l67_67324


namespace xyz_squared_sum_l67_67525

theorem xyz_squared_sum (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) :
  x^2 + y^2 + z^2 = 2 := 
by 
  sorry

end xyz_squared_sum_l67_67525


namespace line_x_intercept_l67_67965

theorem line_x_intercept {x1 y1 x2 y2 : ℝ} (h : (x1, y1) = (4, 6)) (h2 : (x2, y2) = (8, 2)) :
  ∃ x : ℝ, (y1 - y2) / (x1 - x2) * x + 6 - ((y1 - y2) / (x1 - x2)) * 4 = 0 ∧ x = 10 :=
by
  sorry

end line_x_intercept_l67_67965


namespace find_second_equation_value_l67_67409

theorem find_second_equation_value:
  (∃ x y : ℝ, 2 * x + y = 26 ∧ (x + y) / 3 = 4) →
  (∃ x y : ℝ, 2 * x + y = 26 ∧ x + 2 * y = 10) :=
by
  sorry

end find_second_equation_value_l67_67409


namespace smallest_value_of_n_l67_67204

theorem smallest_value_of_n 
  (n : ℕ) 
  (h1 : ∀ θ : ℝ, θ = (n - 2) * 180 / n) 
  (h2 : ∀ α : ℝ, α = 360 / n) 
  (h3 : 28 = 180 / n) :
  n = 45 :=
sorry

end smallest_value_of_n_l67_67204


namespace number_of_dogs_l67_67581

def legs_in_pool : ℕ := 24
def human_legs : ℕ := 4
def legs_per_dog : ℕ := 4

theorem number_of_dogs : (legs_in_pool - human_legs) / legs_per_dog = 5 :=
by
  sorry

end number_of_dogs_l67_67581


namespace maria_initial_carrots_l67_67675

theorem maria_initial_carrots (C : ℕ) (h : C - 11 + 15 = 52) : C = 48 :=
by
  sorry

end maria_initial_carrots_l67_67675


namespace parking_garage_capacity_l67_67278

open Nat

-- Definitions from the conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9
def initial_parked_cars : Nat := 100

-- The proof statement
theorem parking_garage_capacity : 
  (first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces - initial_parked_cars) = 299 := 
  by 
    sorry

end parking_garage_capacity_l67_67278


namespace increasing_function_condition_l67_67355

theorem increasing_function_condition (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k - 6) * x1 + (2 * k + 1) < (2 * k - 6) * x2 + (2 * k + 1)) ↔ (k > 3) :=
by
  -- To prove the statement, we would need to prove it in both directions.
  sorry

end increasing_function_condition_l67_67355


namespace eccentricity_of_given_hyperbola_l67_67648

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : b = 2 * a) : ℝ :=
  Real.sqrt (1 + (b * b) / (a * a))

theorem eccentricity_of_given_hyperbola (a b : ℝ) 
  (h_hyperbola : b = 2 * a)
  (h_asymptote : ∃ k, k = 2 ∧ ∀ x, y = k * x → ((y * a) = (b * x))) :
  hyperbola_eccentricity a b h_hyperbola = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_given_hyperbola_l67_67648


namespace petya_addition_mistake_l67_67950

theorem petya_addition_mistake:
  ∃ (x y c : ℕ), x + y = 12345 ∧ (10 * x + c) + y = 44444 ∧ x = 3566 ∧ y = 8779 ∧ c = 5 := by
  sorry

end petya_addition_mistake_l67_67950


namespace div_ad_bc_by_k_l67_67412

theorem div_ad_bc_by_k 
  (a b c d l k m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n) : 
  k ∣ (a * d - b * c) :=
sorry

end div_ad_bc_by_k_l67_67412


namespace sum_tens_ones_digit_of_7_pow_11_l67_67652

/--
The sum of the tens digit and the ones digit of (3+4)^{11} is 7.
-/
theorem sum_tens_ones_digit_of_7_pow_11 : 
  let number := (3 + 4)^11
  let tens_digit := (number / 10) % 10
  let ones_digit := number % 10
  tens_digit + ones_digit = 7 :=
by
  sorry

end sum_tens_ones_digit_of_7_pow_11_l67_67652


namespace mutually_exclusive_not_necessarily_complementary_l67_67850

-- Define what it means for events to be mutually exclusive
def mutually_exclusive (E1 E2 : Prop) : Prop :=
  ¬ (E1 ∧ E2)

-- Define what it means for events to be complementary
def complementary (E1 E2 : Prop) : Prop :=
  (E1 ∨ E2) ∧ ¬ (E1 ∧ E2) ∧ (¬ E1 ∨ ¬ E2)

theorem mutually_exclusive_not_necessarily_complementary :
  ∀ E1 E2 : Prop, mutually_exclusive E1 E2 → ¬ complementary E1 E2 :=
sorry

end mutually_exclusive_not_necessarily_complementary_l67_67850


namespace herd_compuation_l67_67555

theorem herd_compuation (a b c : ℕ) (total_animals total_payment : ℕ) 
  (H1 : total_animals = a + b + 10 * c) 
  (H2 : total_payment = 20 * a + 10 * b + 10 * c) 
  (H3 : total_animals = 100) 
  (H4 : total_payment = 200) :
  a = 1 ∧ b = 9 ∧ 10 * c = 90 :=
by
  sorry

end herd_compuation_l67_67555


namespace horner_evaluation_at_2_l67_67734

def f (x : ℤ) : ℤ := 3 * x^5 - 2 * x^4 + 2 * x^3 - 4 * x^2 - 7

theorem horner_evaluation_at_2 : f 2 = 16 :=
by {
  sorry
}

end horner_evaluation_at_2_l67_67734


namespace oliver_earnings_l67_67600

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l67_67600


namespace largest_n_cube_condition_l67_67244

theorem largest_n_cube_condition :
  ∃ n : ℕ, (n^3 + 4 * n^2 - 15 * n - 18 = k^3) ∧ ∀ m : ℕ, (m^3 + 4 * m^2 - 15 * m - 18 = k^3 → m ≤ n) → n = 19 :=
by
  sorry

end largest_n_cube_condition_l67_67244


namespace problem_solution_l67_67797

theorem problem_solution (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f ((x - y) ^ 2) = f x ^ 2 - 2 * x * f y + y ^ 2) :
    ∃ n s : ℕ, 
    (n = 2) ∧ 
    (s = 3) ∧
    (n * s = 6) :=
sorry

end problem_solution_l67_67797


namespace margaret_mean_score_l67_67644

noncomputable def cyprian_scores : List ℕ := [82, 85, 89, 91, 95, 97]
noncomputable def cyprian_mean : ℕ := 88

theorem margaret_mean_score :
  let total_sum := List.sum cyprian_scores
  let cyprian_sum := cyprian_mean * 3
  let margaret_sum := total_sum - cyprian_sum
  let margaret_mean := (margaret_sum : ℚ) / 3
  margaret_mean = 91.66666666666667 := 
by 
  -- Definitions used in conditions, skipping steps.
  sorry

end margaret_mean_score_l67_67644


namespace sector_area_l67_67855

theorem sector_area (r α l S : ℝ) (h1 : l + 2 * r = 8) (h2 : α = 2) (h3 : l = α * r) :
  S = 4 :=
by
  -- Let the radius be 2 as a condition derived from h1 and h2
  have r := 2
  -- Substitute and compute to find S
  have S_calculated := (1 / 2 * α * r * r)
  sorry

end sector_area_l67_67855


namespace ratio_of_adults_to_children_closest_to_one_l67_67893

theorem ratio_of_adults_to_children_closest_to_one (a c : ℕ) 
  (h₁ : 25 * a + 12 * c = 1950) 
  (h₂ : a ≥ 1) 
  (h₃ : c ≥ 1) : (a : ℚ) / (c : ℚ) = 27 / 25 := 
by 
  sorry

end ratio_of_adults_to_children_closest_to_one_l67_67893


namespace cone_volume_l67_67199

theorem cone_volume (p q : ℕ) (a α : ℝ) :
  V = (2 * π * a^3) / (3 * (Real.sin (2 * α)) * (Real.cos (180 * q / (p + q)))^2 * (Real.cos α)) :=
sorry

end cone_volume_l67_67199


namespace number_of_clown_mobiles_l67_67356

def num_clown_mobiles (total_clowns clowns_per_mobile : ℕ) : ℕ :=
  total_clowns / clowns_per_mobile

theorem number_of_clown_mobiles :
  num_clown_mobiles 140 28 = 5 :=
by
  sorry

end number_of_clown_mobiles_l67_67356


namespace apples_to_mangos_equivalent_l67_67020

-- Definitions and conditions
def apples_worth_mangos (a b : ℝ) : Prop := (5 / 4) * 16 * a = 10 * b

-- Theorem statement
theorem apples_to_mangos_equivalent : 
  ∀ (a b : ℝ), apples_worth_mangos a b → (3 / 4) * 12 * a = 4.5 * b :=
by
  intro a b
  intro h
  sorry

end apples_to_mangos_equivalent_l67_67020


namespace speed_of_current_l67_67489

theorem speed_of_current (upstream_time : ℝ) (downstream_time : ℝ) :
    upstream_time = 25 / 60 ∧ downstream_time = 12 / 60 →
    ( (60 / downstream_time - 60 / upstream_time) / 2 ) = 1.3 :=
by
  -- Introduce the conditions
  intro h
  -- Simplify using given facts
  have h1 := h.1
  have h2 := h.2
  -- Calcuation of the speed of current
  sorry

end speed_of_current_l67_67489


namespace math_problem_l67_67823

theorem math_problem (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 :=
sorry

end math_problem_l67_67823


namespace solution_to_fraction_l67_67141

theorem solution_to_fraction (x : ℝ) (h_fraction : (x^2 - 4) / (x + 4) = 0) (h_denom : x ≠ -4) : x = 2 ∨ x = -2 :=
sorry

end solution_to_fraction_l67_67141


namespace gemstone_necklaces_sold_correct_l67_67817

-- Define the conditions
def bead_necklaces_sold : Nat := 4
def necklace_cost : Nat := 3
def total_earnings : Nat := 21
def bead_necklaces_earnings : Nat := bead_necklaces_sold * necklace_cost
def gemstone_necklaces_earnings : Nat := total_earnings - bead_necklaces_earnings
def gemstone_necklaces_sold : Nat := gemstone_necklaces_earnings / necklace_cost

-- Theorem to prove the number of gem stone necklaces sold
theorem gemstone_necklaces_sold_correct :
  gemstone_necklaces_sold = 3 :=
by
  -- Proof omitted
  sorry

end gemstone_necklaces_sold_correct_l67_67817


namespace rope_length_after_100_cuts_l67_67252

noncomputable def rope_cut (initial_length : ℝ) (num_cuts : ℕ) (cut_fraction : ℝ) : ℝ :=
  initial_length * (1 - cut_fraction) ^ num_cuts

theorem rope_length_after_100_cuts :
  rope_cut 1 100 (3 / 4) = (1 / 4) ^ 100 :=
by
  sorry

end rope_length_after_100_cuts_l67_67252


namespace AM_GM_inequality_example_l67_67189

theorem AM_GM_inequality_example (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) : 
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_inequality_example_l67_67189


namespace no_nat_nums_satisfying_l67_67920

theorem no_nat_nums_satisfying (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k :=
by
  sorry

end no_nat_nums_satisfying_l67_67920


namespace triangle_inequality_area_equality_condition_l67_67643

theorem triangle_inequality_area (a b c S : ℝ) (h_area : S = (a * b * Real.sin (Real.arccos ((a*a + b*b - c*c) / (2*a*b)))) / 2) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by
  sorry

theorem equality_condition (a b c : ℝ) (h_eq : a = b ∧ b = c) : 
  a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * (a^2 * (Real.sqrt 3 / 4)) :=
by
  sorry

end triangle_inequality_area_equality_condition_l67_67643


namespace cricket_team_members_eq_11_l67_67271

-- Definitions based on conditions:
def captain_age : ℕ := 26
def wicket_keeper_age : ℕ := 31
def avg_age_whole_team : ℕ := 24
def avg_age_remaining_players : ℕ := 23

-- Definition of n based on the problem conditions
def number_of_members (n : ℕ) : Prop :=
  n * avg_age_whole_team = (n - 2) * avg_age_remaining_players + (captain_age + wicket_keeper_age)

-- The proof statement:
theorem cricket_team_members_eq_11 : ∃ n, number_of_members n ∧ n = 11 := 
by
  use 11
  unfold number_of_members
  sorry

end cricket_team_members_eq_11_l67_67271


namespace correct_divisor_l67_67838

noncomputable def dividend := 12 * 35

theorem correct_divisor (x : ℕ) : (x * 20 = dividend) → x = 21 :=
sorry

end correct_divisor_l67_67838


namespace expression_is_integer_l67_67773

theorem expression_is_integer (n : ℤ) : (∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k) := 
sorry

end expression_is_integer_l67_67773


namespace area_of_circle_with_given_circumference_l67_67473

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end area_of_circle_with_given_circumference_l67_67473


namespace girls_boys_ratio_l67_67005

theorem girls_boys_ratio (G B : ℕ) (h1 : G + B = 100) (h2 : 0.20 * (G : ℝ) + 0.10 * (B : ℝ) = 15) : G / B = 1 :=
by
  -- Proof steps are omitted
  sorry

end girls_boys_ratio_l67_67005


namespace binomial_10_3_eq_120_l67_67116

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l67_67116


namespace similar_triangles_l67_67890

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end similar_triangles_l67_67890


namespace inequality_problem_l67_67542

theorem inequality_problem
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by sorry

end inequality_problem_l67_67542


namespace problem_27_integer_greater_than_B_over_pi_l67_67053

noncomputable def B : ℕ := 22

theorem problem_27_integer_greater_than_B_over_pi :
  Nat.ceil (B / Real.pi) = 8 := sorry

end problem_27_integer_greater_than_B_over_pi_l67_67053


namespace number_of_members_l67_67858

theorem number_of_members (n : ℕ) (h : n^2 = 5929) : n = 77 :=
sorry

end number_of_members_l67_67858


namespace Lincoln_High_School_max_principals_l67_67122

def max_principals (total_years : ℕ) (term_length : ℕ) (max_principals_count : ℕ) : Prop :=
  ∀ (period : ℕ), period = total_years → 
                  term_length = 4 → 
                  max_principals_count = 3

theorem Lincoln_High_School_max_principals 
  (total_years term_length max_principals_count : ℕ) :
  max_principals total_years term_length max_principals_count :=
by 
  intros period h1 h2
  have h3 : period = 10 := sorry
  have h4 : term_length = 4 := sorry
  have h5 : max_principals_count = 3 := sorry
  sorry

end Lincoln_High_School_max_principals_l67_67122


namespace pirates_treasure_l67_67615

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l67_67615


namespace second_group_students_l67_67452

theorem second_group_students (S : ℕ) : 
    (1200 / 40) = 9 + S + 11 → S = 10 :=
by sorry

end second_group_students_l67_67452


namespace cubic_identity_l67_67370

theorem cubic_identity (x : ℝ) (hx : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end cubic_identity_l67_67370


namespace gcd_of_all_elements_in_B_is_2_l67_67954

-- Define the set B as the set of all numbers that can be represented as the sum of four consecutive positive integers.
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2 ∧ x > 0}

-- Translate the question to a Lean statement.
theorem gcd_of_all_elements_in_B_is_2 : ∀ n ∈ B, gcd n 2 = 2 := 
by
  sorry

end gcd_of_all_elements_in_B_is_2_l67_67954


namespace simplify_expression_l67_67491

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 - 11 * x + 13 * x^2 - 15 + 17 * x + 19 * x^2 = 25 * x^2 + x - 3 := 
by
  sorry

end simplify_expression_l67_67491


namespace age_difference_l67_67655

variable (a b c d : ℕ)
variable (h1 : a + b = b + c + 11)
variable (h2 : a + c = c + d + 15)
variable (h3 : b + d = 36)
variable (h4 : a * 2 = 3 * d)

theorem age_difference :
  a - b = 39 :=
by
  sorry

end age_difference_l67_67655


namespace probability_left_red_off_second_blue_on_right_blue_on_l67_67898

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_on : ℕ := 4
def position := Fin total_lamps
def lamp_state := {state // state < (total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)}

def valid_configuration (leftmost : position) (second_left : position) (rightmost : position) (s : lamp_state) : Prop :=
(leftmost.1 = 1 ∧ second_left.1 = 2 ∧ rightmost.1 = 8) ∧ (s.1 =  (((total_lamps - 3).choose 3) * ((total_lamps - 3).choose 2)))

theorem probability_left_red_off_second_blue_on_right_blue_on :
  ∀ (leftmost second_left rightmost : position) (s : lamp_state),
  valid_configuration leftmost second_left rightmost s ->
  ((total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)) = 49 :=
sorry

end probability_left_red_off_second_blue_on_right_blue_on_l67_67898


namespace ram_salary_percentage_more_l67_67305

theorem ram_salary_percentage_more (R r : ℝ) (h : r = 0.8 * R) :
  ((R - r) / r) * 100 = 25 := 
sorry

end ram_salary_percentage_more_l67_67305


namespace eliza_height_l67_67524

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end eliza_height_l67_67524


namespace P_subsetneq_Q_l67_67463

def P : Set ℝ := { x : ℝ | x > 1 }
def Q : Set ℝ := { x : ℝ | x^2 - x > 0 }

theorem P_subsetneq_Q : P ⊂ Q :=
by
  sorry

end P_subsetneq_Q_l67_67463


namespace no_partition_square_isosceles_10deg_l67_67929

theorem no_partition_square_isosceles_10deg :
  ¬ ∃ (P : ℝ → ℝ → Prop), 
    (∀ x y, P x y → ((x = y) ∨ ((10 * x + 10 * y + 160 * (180 - x - y)) = 9 * 10))) ∧
    (∀ x y, P x 90 → P x y) ∧
    (P 90 90 → False) :=
by
  sorry

end no_partition_square_isosceles_10deg_l67_67929


namespace polynomial_solution_l67_67421

theorem polynomial_solution (f : ℝ → ℝ) (x : ℝ) (h : f (x^2 + 2) = x^4 + 6 * x^2 + 4) : 
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  sorry

end polynomial_solution_l67_67421


namespace min_value_fraction_l67_67752

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) : 
  ∃x : ℝ, (x = (1/a + 2/b)) ∧ (∀y : ℝ, (y = (1/a + 2/b)) → y ≥ 8) :=
by
  sorry

end min_value_fraction_l67_67752


namespace negation_of_implication_iff_l67_67497

variable (a : ℝ)

theorem negation_of_implication_iff (p : a > 1 → a^2 > 1) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) :=
by sorry

end negation_of_implication_iff_l67_67497


namespace quotient_of_powers_l67_67983

theorem quotient_of_powers:
  (50 : ℕ) = 2 * 5^2 →
  (25 : ℕ) = 5^2 →
  (50^50 / 25^25 : ℕ) = 100^25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end quotient_of_powers_l67_67983


namespace derivative_f_at_1_l67_67200

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * Real.sin x

theorem derivative_f_at_1 : (deriv f 1) = 2 + 2 * Real.cos 1 := 
sorry

end derivative_f_at_1_l67_67200


namespace arithmetic_sequences_integer_ratio_count_l67_67253

theorem arithmetic_sequences_integer_ratio_count 
  (a_n b_n : ℕ → ℕ)
  (A_n B_n : ℕ → ℕ)
  (h₁ : ∀ n, A_n n = n * (a_n 1 + a_n (2 * n - 1)) / 2)
  (h₂ : ∀ n, B_n n = n * (b_n 1 + b_n (2 * n - 1)) / 2)
  (h₃ : ∀ n, A_n n / B_n n = (7 * n + 41) / (n + 3)) :
  ∃ (cnt : ℕ), cnt = 3 ∧ ∀ n, (∃ k, n = 1 + 3 * k) → (a_n n) / (b_n n) = 7 + (10 / (n + 1)) :=
by
  sorry

end arithmetic_sequences_integer_ratio_count_l67_67253


namespace part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l67_67970

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + 1
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

-- (Ⅰ) Equation of the tangent line to y = f(x) at x = 1
theorem part1_tangent_line_at_1 : ∀ x, (f 1 + (1 / 1) * (x - 1)) = x - 1 := sorry

-- (Ⅱ) Intervals where F(x) is monotonic
theorem part2_monotonic_intervals (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, F x a > 0) ∧ 
  (a > 0 → (∀ x > 0, x < (1 / a) → F x a > 0) ∧ (∀ x > 1 / a, F x a < 0)) := sorry

-- (Ⅲ) Range of a for which f(x) is below g(x) for all x > 0
theorem part3_range_of_a (a : ℝ) : (∀ x > 0, f x < g x a) ↔ a ∈ Set.Ioi (Real.exp (-2)) := sorry

end part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l67_67970


namespace jessica_games_attended_l67_67824

def total_games : ℕ := 6
def games_missed_by_jessica : ℕ := 4

theorem jessica_games_attended : total_games - games_missed_by_jessica = 2 := by
  sorry

end jessica_games_attended_l67_67824


namespace inverse_function_value_l67_67432

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ y : ℝ, f (3^y) = y) : f 3 = 1 :=
sorry

end inverse_function_value_l67_67432


namespace find_n_l67_67933

theorem find_n : ∀ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) → (n = -7 / 2) := by
  intro n h
  sorry

end find_n_l67_67933


namespace inequality_subtraction_l67_67163

variable (a b : ℝ)

-- Given conditions
axiom nonzero_a : a ≠ 0 
axiom nonzero_b : b ≠ 0 
axiom a_lt_b : a < b 

-- Proof statement
theorem inequality_subtraction : a - 3 < b - 3 := 
by 
  sorry

end inequality_subtraction_l67_67163


namespace non_receivers_after_2020_candies_l67_67594

noncomputable def count_non_receivers (k n : ℕ) : ℕ := 
sorry

theorem non_receivers_after_2020_candies :
  count_non_receivers 73 2020 = 36 :=
sorry

end non_receivers_after_2020_candies_l67_67594


namespace gain_percentage_l67_67953

theorem gain_percentage (C S : ℝ) (h : 80 * C = 25 * S) : 220 = ((S - C) / C) * 100 :=
by sorry

end gain_percentage_l67_67953


namespace token_exchange_l67_67440

def booth1 (r : ℕ) (x : ℕ) : ℕ × ℕ × ℕ := (r - 3 * x, 2 * x, x)
def booth2 (b : ℕ) (y : ℕ) : ℕ × ℕ × ℕ := (y, b - 4 * y, y)

theorem token_exchange (x y : ℕ) (h1 : 100 - 3 * x + y = 2) (h2 : 50 + x - 4 * y = 3) :
  x + y = 58 :=
sorry

end token_exchange_l67_67440


namespace f_at_neg_8_5_pi_eq_pi_div_2_l67_67560

def f (x : Real) : Real := sorry

axiom functional_eqn (x : Real) : f (x + (3 * Real.pi / 2)) = -1 / f x
axiom f_interval (x : Real) (h : x ∈ Set.Icc (-Real.pi) Real.pi) : f x = x * Real.sin x

theorem f_at_neg_8_5_pi_eq_pi_div_2 : f (-8.5 * Real.pi) = Real.pi / 2 := 
  sorry

end f_at_neg_8_5_pi_eq_pi_div_2_l67_67560


namespace find_larger_number_l67_67034

theorem find_larger_number (x y : ℕ) (h1 : x + y = 55) (h2 : x - y = 15) : x = 35 := by 
  -- proof will go here
  sorry

end find_larger_number_l67_67034


namespace bread_loaves_l67_67762

theorem bread_loaves (loaf_cost : ℝ) (pb_cost : ℝ) (total_money : ℝ) (leftover_money : ℝ) : ℝ :=
  let spent_money := total_money - leftover_money
  let remaining_money := spent_money - pb_cost
  remaining_money / loaf_cost

example : bread_loaves 2.25 2 14 5.25 = 3 := by
  sorry

end bread_loaves_l67_67762


namespace total_thread_needed_l67_67256

def keychain_length : Nat := 12
def friends_in_classes : Nat := 10
def multiplier_for_club_friends : Nat := 2
def thread_per_class_friend : Nat := 16
def thread_per_club_friend : Nat := 20

theorem total_thread_needed :
  10 * thread_per_class_friend + (10 * multiplier_for_club_friends) * thread_per_club_friend = 560 := by
  sorry

end total_thread_needed_l67_67256


namespace natural_number_with_property_l67_67117

theorem natural_number_with_property :
  ∃ n a b c : ℕ, (n = 10 * a + b) ∧ (100 * a + 10 * c + b = 6 * n) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (n = 18) :=
sorry

end natural_number_with_property_l67_67117


namespace Ellipse_area_constant_l67_67461

-- Definitions of given conditions and problem setup
def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse_equation 1 (Real.sqrt 3 / 2) a b

def eccentricity (c a : ℝ) : Prop :=
  c / a = Real.sqrt 3 / 2

def moving_points_on_ellipse (a b x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_equation x₁ y₁ a b ∧ ellipse_equation x₂ y₂ a b

def slopes_condition (k₁ k₂ : ℝ) : Prop :=
  k₁ * k₂ = -1/4

def area_OMN := 1

-- Main theorem statement
theorem Ellipse_area_constant
(a b : ℝ) 
(h_ellipse : point_on_ellipse a b)
(h_eccentricity : eccentricity (Real.sqrt 3 / 2 * a) a)
(M N : ℝ × ℝ) 
(h_points : moving_points_on_ellipse a b M.1 M.2 N.1 N.2)
(k₁ k₂ : ℝ) 
(h_slopes : slopes_condition k₁ k₂) : 
a^2 = 4 ∧ b^2 = 1 ∧ area_OMN = 1 := 
sorry

end Ellipse_area_constant_l67_67461


namespace correct_remove_parentheses_l67_67019

theorem correct_remove_parentheses (a b c d : ℝ) :
  (a - (5 * b - (2 * c - 1)) = a - 5 * b + 2 * c - 1) :=
by sorry

end correct_remove_parentheses_l67_67019


namespace max_min_diff_c_l67_67591

theorem max_min_diff_c (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  ∃ c_max c_min, 
  (∀ c', (a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18) → c_min ≤ c' ∧ c' ≤ c_max) 
  ∧ (c_max - c_min = 6) :=
  sorry

end max_min_diff_c_l67_67591


namespace sample_size_is_40_l67_67597

theorem sample_size_is_40 (total_students : ℕ) (sample_students : ℕ) (h1 : total_students = 240) (h2 : sample_students = 40) : sample_students = 40 :=
by
  sorry

end sample_size_is_40_l67_67597


namespace ascorbic_acid_oxygen_mass_percentage_l67_67744

noncomputable def mass_percentage_oxygen_in_ascorbic_acid : Float := 54.49

theorem ascorbic_acid_oxygen_mass_percentage :
  let C_mass := 12.01
  let H_mass := 1.01
  let O_mass := 16.00
  let ascorbic_acid_formula := (6, 8, 6) -- (number of C, number of H, number of O)
  let total_mass := 6 * C_mass + 8 * H_mass + 6 * O_mass
  let O_mass_total := 6 * O_mass
  mass_percentage_oxygen_in_ascorbic_acid = (O_mass_total / total_mass) * 100 := by
  sorry

end ascorbic_acid_oxygen_mass_percentage_l67_67744


namespace line_product_l67_67780

theorem line_product (b m : Int) (h_b : b = -2) (h_m : m = 3) : m * b = -6 :=
by
  rw [h_b, h_m]
  norm_num

end line_product_l67_67780


namespace line_passes_through_fixed_point_l67_67911

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * (1) + 3 * k = 0 :=
by
  intro k
  sorry

end line_passes_through_fixed_point_l67_67911


namespace jill_total_trip_duration_is_101_l67_67002

def first_bus_wait_time : Nat := 12
def first_bus_ride_time : Nat := 30
def first_bus_delay_time : Nat := 5

def walk_time_to_train : Nat := 10
def train_wait_time : Nat := 8
def train_ride_time : Nat := 20
def train_delay_time : Nat := 3

def second_bus_wait_time : Nat := 20
def second_bus_ride_time : Nat := 6

def route_b_combined_time := (second_bus_wait_time + second_bus_ride_time) / 2

def total_trip_duration : Nat := 
  first_bus_wait_time + first_bus_ride_time + first_bus_delay_time +
  walk_time_to_train + train_wait_time + train_ride_time + train_delay_time +
  route_b_combined_time

theorem jill_total_trip_duration_is_101 : total_trip_duration = 101 := by
  sorry

end jill_total_trip_duration_is_101_l67_67002


namespace probability_A_inter_B_l67_67811

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 5
def set_B (x : ℝ) : Prop := (x-2)/(3-x) > 0

def A_inter_B (x : ℝ) : Prop := set_A x ∧ set_B x

theorem probability_A_inter_B :
  let length_A := 5 - (-1)
  let length_A_inter_B := 3 - 2 
  length_A > 0 ∧ length_A_inter_B > 0 →
  length_A_inter_B / length_A = 1 / 6 :=
by
  intro h
  sorry

end probability_A_inter_B_l67_67811


namespace no_partition_of_integers_l67_67262

theorem no_partition_of_integers (A B C : Set ℕ) :
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (∀ a b, a ∈ A ∧ b ∈ B → (a^2 - a * b + b^2) ∈ C) ∧
  (∀ a b, a ∈ B ∧ b ∈ C → (a^2 - a * b + b^2) ∈ A) ∧
  (∀ a b, a ∈ C ∧ b ∈ A → (a^2 - a * b + b^2) ∈ B) →
  False := 
sorry

end no_partition_of_integers_l67_67262


namespace necessary_but_not_sufficient_l67_67533

variable {a b c : ℝ}

theorem necessary_but_not_sufficient (h1 : b^2 - 4 * a * c ≥ 0) (h2 : a * c > 0) (h3 : a * b < 0) : 
  ¬∀ r1 r2 : ℝ, (r1 = (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) ∧ (r2 = (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)) → r1 > 0 ∧ r2 > 0 :=
sorry

end necessary_but_not_sufficient_l67_67533


namespace work_rate_problem_l67_67484

theorem work_rate_problem
  (W : ℕ) -- total work
  (A_rate : ℕ) -- A's work rate in days
  (B_rate : ℕ) -- B's work rate in days
  (x : ℕ) -- days A worked alone
  (total_days : ℕ) -- days A and B worked together
  (hA : A_rate = 12) -- A can do the work in 12 days
  (hB : B_rate = 6) -- B can do the work in 6 days
  (hx : total_days = 3) -- remaining days they together work
  : x = 3 := 
by
  sorry

end work_rate_problem_l67_67484


namespace metres_sold_is_200_l67_67569

-- Define the conditions
def loss_per_metre : ℕ := 6
def cost_price_per_metre : ℕ := 66
def total_selling_price : ℕ := 12000

-- Define the selling price per metre based on the conditions
def selling_price_per_metre := cost_price_per_metre - loss_per_metre

-- Define the number of metres sold
def metres_sold : ℕ := total_selling_price / selling_price_per_metre

-- Proof statement: Check if the number of metres sold equals 200
theorem metres_sold_is_200 : metres_sold = 200 :=
  by
  sorry

end metres_sold_is_200_l67_67569


namespace origin_moves_distance_l67_67214

noncomputable def origin_distance_moved : ℝ :=
  let B := (3, 1)
  let B' := (7, 9)
  let k := 1.5
  let center_of_dilation := (-1, -3)
  let d0 := Real.sqrt ((-1)^2 + (-3)^2)
  let d1 := k * d0
  d1 - d0

theorem origin_moves_distance :
  origin_distance_moved = 0.5 * Real.sqrt 10 :=
by 
  sorry

end origin_moves_distance_l67_67214


namespace correlation_index_l67_67630

-- Define the conditions given in the problem
def height_explains_weight_variation : Prop :=
  ∃ R : ℝ, R^2 = 0.64

-- State the main conjecture (actual proof omitted for simplicity)
theorem correlation_index (R : ℝ) (h : height_explains_weight_variation) : R^2 = 0.64 := by
  sorry

end correlation_index_l67_67630


namespace find_second_number_l67_67040

theorem find_second_number (x : ℕ) :
  22030 = (555 + x) * 2 * (x - 555) + 30 → 
  x = 564 :=
by
  intro h
  sorry

end find_second_number_l67_67040


namespace possible_to_select_three_numbers_l67_67778

theorem possible_to_select_three_numbers (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i < j → a i < a j) (h_bound : ∀ i, a i < 2 * n) :
  ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ a i + a j = a k := sorry

end possible_to_select_three_numbers_l67_67778


namespace age_of_B_l67_67616

-- Define the ages based on the conditions
def A (x : ℕ) : ℕ := 2 * x + 2
def B (x : ℕ) : ℕ := 2 * x
def C (x : ℕ) : ℕ := x

-- The main statement to be proved
theorem age_of_B (x : ℕ) (h : A x + B x + C x = 72) : B 14 = 28 :=
by
  -- we need the proof here but we will put sorry for now
  sorry

end age_of_B_l67_67616


namespace correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l67_67460

theorem correct_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

-- Incorrect options for clarity
theorem incorrect_division (a : ℝ) : a^6 / a^2 ≠ a^3 :=
by sorry

theorem incorrect_multiplication (a : ℝ) : a^2 * a^3 ≠ a^6 :=
by sorry

theorem incorrect_addition (a : ℝ) : (a^2 + a^3) ≠ a^5 :=
by sorry

end correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l67_67460


namespace circle_radius_l67_67905

theorem circle_radius 
  {XA XB XC r : ℝ}
  (h1 : XA = 3)
  (h2 : XB = 5)
  (h3 : XC = 1)
  (hx : XA * XB = XC * r)
  (hh : 2 * r = CD) :
  r = 8 :=
by
  sorry

end circle_radius_l67_67905


namespace company_bought_oil_l67_67526

-- Define the conditions
def tank_capacity : ℕ := 32
def oil_in_tank : ℕ := 24

-- Formulate the proof problem
theorem company_bought_oil : oil_in_tank = 24 := by
  sorry

end company_bought_oil_l67_67526


namespace third_side_of_triangle_l67_67548

theorem third_side_of_triangle (a b : ℝ) (γ : ℝ) (x : ℝ) 
  (ha : a = 6) (hb : b = 2 * Real.sqrt 7) (hγ : γ = Real.pi / 3) :
  x = 2 ∨ x = 4 :=
by 
  sorry

end third_side_of_triangle_l67_67548


namespace range_of_a_l67_67345

theorem range_of_a (x a : ℝ) (p : (x + 1)^2 > 4) (q : x > a) 
  (h : (¬((x + 1)^2 > 4)) → (¬(x > a)))
  (sufficient_but_not_necessary : (¬((x + 1)^2 > 4)) → (¬(x > a))) : a ≥ 1 :=
sorry

end range_of_a_l67_67345


namespace evaluate_expression_l67_67769

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-2) = 85 :=
by
  sorry

end evaluate_expression_l67_67769


namespace students_diff_l67_67553

-- Define the conditions
def M : ℕ := 457
def B : ℕ := 394

-- Prove the final answer
theorem students_diff : M - B = 63 := by
  -- The proof is omitted here with a sorry placeholder
  sorry

end students_diff_l67_67553


namespace university_theater_ticket_sales_l67_67257

theorem university_theater_ticket_sales (total_tickets : ℕ) (adult_price : ℕ) (senior_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) (h2 : adult_price = 21) (h3 : senior_price = 15) (h4 : senior_tickets = 327) : 
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 8748 :=
by 
  -- Proof skipped
  sorry

end university_theater_ticket_sales_l67_67257


namespace chocolate_cost_l67_67634

def cost_of_chocolates (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

theorem chocolate_cost : cost_of_chocolates 30 8 450 = 120 :=
by
  -- The proof is not needed per the instructions
  sorry

end chocolate_cost_l67_67634


namespace mass_of_man_proof_l67_67839

def volume_displaced (L B h : ℝ) : ℝ :=
  L * B * h

def mass_of_man (V ρ : ℝ) : ℝ :=
  ρ * V

theorem mass_of_man_proof :
  ∀ (L B h ρ : ℝ), L = 9 → B = 3 → h = 0.01 → ρ = 1000 →
  mass_of_man (volume_displaced L B h) ρ = 270 :=
by
  intros L B h ρ L_eq B_eq h_eq ρ_eq
  rw [L_eq, B_eq, h_eq, ρ_eq]
  unfold volume_displaced
  unfold mass_of_man
  simp
  sorry

end mass_of_man_proof_l67_67839


namespace boolean_logic_problem_l67_67102

theorem boolean_logic_problem (p q : Prop) (h₁ : ¬(p ∧ q)) (h₂ : ¬(¬p)) : ¬q :=
by {
  sorry
}

end boolean_logic_problem_l67_67102


namespace generalized_inequality_l67_67389

theorem generalized_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) : x^n + (n : ℝ) / x > n + 1 := 
sorry

end generalized_inequality_l67_67389


namespace union_of_S_and_T_l67_67107

-- Declare sets S and T
def S : Set ℕ := {3, 4, 5}
def T : Set ℕ := {4, 7, 8}

-- Statement about their union
theorem union_of_S_and_T : S ∪ T = {3, 4, 5, 7, 8} :=
sorry

end union_of_S_and_T_l67_67107


namespace triangle_perimeter_is_26_l67_67627

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end triangle_perimeter_is_26_l67_67627


namespace ratio_senior_junior_l67_67878

theorem ratio_senior_junior
  (J S : ℕ)
  (h1 : ∃ k : ℕ, S = k * J)
  (h2 : (3 / 8) * S + (1 / 4) * J = (1 / 3) * (S + J)) :
  S = 2 * J :=
by
  -- The proof is to be provided
  sorry

end ratio_senior_junior_l67_67878


namespace yoongi_has_fewer_apples_l67_67518

-- Define the number of apples Jungkook originally has and receives more.
def jungkook_original_apples := 6
def jungkook_received_apples := 3

-- Calculate the total number of apples Jungkook has.
def jungkook_total_apples := jungkook_original_apples + jungkook_received_apples

-- Define the number of apples Yoongi has.
def yoongi_apples := 4

-- State that Yoongi has fewer apples than Jungkook.
theorem yoongi_has_fewer_apples : yoongi_apples < jungkook_total_apples := by
  sorry

end yoongi_has_fewer_apples_l67_67518


namespace total_spent_is_195_l67_67266

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def boots_discounted_cost : ℝ := boots_original_cost * (1 - boots_discount)
def total_cost : ℝ := hoodie_cost + flashlight_cost + boots_discounted_cost

theorem total_spent_is_195 : total_cost = 195 := by
  sorry

end total_spent_is_195_l67_67266


namespace determine_h_l67_67260

theorem determine_h (h : ℝ) : (∃ x : ℝ, x = 3 ∧ x^3 - 2 * h * x + 15 = 0) → h = 7 :=
by
  intro hx
  sorry

end determine_h_l67_67260


namespace unique_solution_l67_67089

theorem unique_solution (a n : ℕ) (h₁ : 0 < a) (h₂ : 0 < n) (h₃ : 3^n = a^2 - 16) : a = 5 ∧ n = 2 :=
by
sorry

end unique_solution_l67_67089


namespace volume_after_increase_l67_67384

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l67_67384


namespace min_value_x_plus_one_over_x_plus_two_l67_67360

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1 / (x + 2) ∧ y ≥ 0 := 
sorry

end min_value_x_plus_one_over_x_plus_two_l67_67360


namespace find_prime_triplet_l67_67472

theorem find_prime_triplet (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ↔ (p, q, r) = (5, 3, 19) :=
by
  sorry

end find_prime_triplet_l67_67472


namespace frac_calc_l67_67980

theorem frac_calc : (2 / 9) * (5 / 11) + 1 / 3 = 43 / 99 :=
by sorry

end frac_calc_l67_67980


namespace original_cost_of_remaining_shirt_l67_67541

theorem original_cost_of_remaining_shirt 
  (total_original_cost : ℝ) 
  (shirts_on_discount : ℕ) 
  (original_cost_per_discounted_shirt : ℝ) 
  (discount : ℝ) 
  (current_total_cost : ℝ) : 
  total_original_cost = 100 → 
  shirts_on_discount = 3 → 
  original_cost_per_discounted_shirt = 25 → 
  discount = 0.4 → 
  current_total_cost = 85 → 
  ∃ (remaining_shirts : ℕ) (original_cost_per_remaining_shirt : ℝ), 
    remaining_shirts = 2 ∧ 
    original_cost_per_remaining_shirt = 12.5 :=
by 
  sorry

end original_cost_of_remaining_shirt_l67_67541


namespace symmetric_point_coordinates_l67_67232

noncomputable def symmetric_with_respect_to_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (-x, y, -z)

theorem symmetric_point_coordinates : symmetric_with_respect_to_y_axis (-2, 1, 4) = (2, 1, -4) :=
by sorry

end symmetric_point_coordinates_l67_67232


namespace prescription_duration_l67_67436

theorem prescription_duration (D : ℕ) (h1 : (2 * D) * (1 / 5) = 12) : D = 30 :=
by
  sorry

end prescription_duration_l67_67436


namespace people_left_after_first_stop_l67_67329

def initial_people_on_train : ℕ := 48
def people_got_off_train : ℕ := 17

theorem people_left_after_first_stop : (initial_people_on_train - people_got_off_train) = 31 := by
  sorry

end people_left_after_first_stop_l67_67329


namespace union_eq_interval_l67_67496

def A := { x : ℝ | 1 < x ∧ x < 4 }
def B := { x : ℝ | (x - 3) * (x + 1) ≤ 0 }

theorem union_eq_interval : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 4 } :=
by
  sorry

end union_eq_interval_l67_67496


namespace distance_D_to_plane_l67_67169

-- Given conditions about the distances from points A, B, and C to plane M
variables (a b c : ℝ)

-- Formalizing the distance from vertex D to plane M
theorem distance_D_to_plane (a b c : ℝ) : 
  ∃ d : ℝ, d = |a + b + c| ∨ d = |a + b - c| ∨ d = |a - b + c| ∨ d = |-a + b + c| ∨ 
                    d = |a - b - c| ∨ d = |-a - b + c| ∨ d = |-a + b - c| ∨ d = |-a - b - c| := sorry

end distance_D_to_plane_l67_67169


namespace percentage_increase_l67_67381

def x (y: ℝ) : ℝ := 1.25 * y
def z : ℝ := 250
def total_amount (x y z : ℝ) : ℝ := x + y + z

theorem percentage_increase (y: ℝ) : (total_amount (x y) y z = 925) → ((y - z) / z) * 100 = 20 := by
  sorry

end percentage_increase_l67_67381


namespace janet_earnings_per_hour_l67_67943

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end janet_earnings_per_hour_l67_67943


namespace total_seeds_eaten_l67_67310

-- Definitions and conditions
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds
def first_four_players_seeds : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds
def average_seeds : ℕ := first_four_players_seeds / 4
def fifth_player_seeds : ℕ := average_seeds

-- Statement to prove
theorem total_seeds_eaten :
  first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds + fifth_player_seeds = 475 :=
by {
  sorry
}

end total_seeds_eaten_l67_67310


namespace cost_of_rope_l67_67493

theorem cost_of_rope : 
  ∀ (total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost : ℝ),
  total_money = 200 ∧
  sheet_cost = 42 ∧
  propane_burner_cost = 14 ∧
  helium_cost_per_ounce = 1.50 ∧
  helium_per_foot = 113 ∧
  max_height = 9492 ∧
  rope_cost = total_money - (sheet_cost + propane_burner_cost + (max_height / helium_per_foot) * helium_cost_per_ounce) →
  rope_cost = 18 :=
by
  intros total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost
  rintro ⟨h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max, h_rope⟩
  rw [h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max] at h_rope
  simp only [inv_mul_eq_iff_eq_mul, div_eq_mul_inv] at h_rope
  norm_num at h_rope
  sorry

end cost_of_rope_l67_67493


namespace ninth_observation_l67_67288

theorem ninth_observation (avg1 : ℝ) (avg2 : ℝ) (n1 n2 : ℝ) 
  (sum1 : n1 * avg1 = 120) 
  (sum2 : n2 * avg2 = 117) 
  (avg_decrease : avg1 - avg2 = 2) 
  (obs_count_change : n1 + 1 = n2) 
  : n2 * avg2 - n1 * avg1 = -3 :=
by
  sorry

end ninth_observation_l67_67288


namespace altitude_line_equation_equal_distance_lines_l67_67236

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- The equation of the line for the altitude from A to BC
theorem altitude_line_equation :
  ∃ (a b c : ℝ), 2 * a - 3 * b + 14 = 0 :=
sorry

-- The equations of the line passing through B such that the distances from A and C are equal
theorem equal_distance_lines :
  ∃ (a b c : ℝ), (7 * a - 6 * b + 4 = 0) ∧ (3 * a + 2 * b - 44 = 0) :=
sorry

end altitude_line_equation_equal_distance_lines_l67_67236


namespace exposed_surface_area_equals_42_l67_67350

-- Define the structure and exposed surface area calculations.
def surface_area_of_sculpture (layers : List Nat) : Nat :=
  (layers.headD 0 * 5) +  -- Top layer (5 faces exposed)
  (layers.getD 1 0 * 3 + layers.getD 1 0) +  -- Second layer
  (layers.getD 2 0 * 1 + layers.getD 2 0) +  -- Third layer
  (layers.getD 3 0 * 1) -- Bottom layer

-- Define the conditions
def number_of_layers : List Nat := [1, 4, 9, 6]

-- State the theorem
theorem exposed_surface_area_equals_42 :
  surface_area_of_sculpture number_of_layers = 42 :=
by
  sorry

end exposed_surface_area_equals_42_l67_67350


namespace complex_expression_evaluation_l67_67559

theorem complex_expression_evaluation : (i : ℂ) * (1 + i : ℂ)^2 = -2 := 
by
  sorry

end complex_expression_evaluation_l67_67559


namespace smallest_solution_floor_eq_l67_67590

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l67_67590


namespace find_second_offset_l67_67132

variable (d : ℕ) (o₁ : ℕ) (A : ℕ)

theorem find_second_offset (hd : d = 20) (ho₁ : o₁ = 5) (hA : A = 90) : ∃ (o₂ : ℕ), o₂ = 4 :=
by
  sorry

end find_second_offset_l67_67132


namespace a5_gt_b5_l67_67835

variables {a_n b_n : ℕ → ℝ}
variables {a1 b1 a3 b3 : ℝ}
variables {q : ℝ} {d : ℝ}

/-- Given conditions -/
axiom h1 : a1 = b1
axiom h2 : a1 > 0
axiom h3 : a3 = b3
axiom h4 : a3 = a1 * q^2
axiom h5 : b3 = a1 + 2 * d
axiom h6 : a1 ≠ a3

/-- Prove that a_5 is greater than b_5 -/
theorem a5_gt_b5 : a1 * q^4 > a1 + 4 * d :=
by sorry

end a5_gt_b5_l67_67835


namespace ratio_of_lengths_l67_67900

theorem ratio_of_lengths (l1 l2 l3 : ℝ)
    (h1 : l2 = (1/2) * (l1 + l3))
    (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
    l1 / l3 = 7 / 5 := by
  sorry

end ratio_of_lengths_l67_67900


namespace even_product_probability_l67_67567

def number_on_first_spinner := [3, 6, 5, 10, 15]
def number_on_second_spinner := [7, 6, 11, 12, 13, 14]

noncomputable def probability_even_product : ℚ :=
  1 - (3 / 5) * (3 / 6)

theorem even_product_probability :
  probability_even_product = 7 / 10 :=
by
  sorry

end even_product_probability_l67_67567


namespace elvins_first_month_bill_l67_67515

theorem elvins_first_month_bill (F C : ℝ) 
  (h1 : F + C = 52)
  (h2 : F + 2 * C = 76) : 
  F + C = 52 :=
by
  sorry

end elvins_first_month_bill_l67_67515


namespace ticket_distribution_l67_67071

noncomputable def num_dist_methods (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : ℕ := sorry

theorem ticket_distribution :
  num_dist_methods 18 5 6 7 10 = 140 := sorry

end ticket_distribution_l67_67071


namespace Lance_workdays_per_week_l67_67225

theorem Lance_workdays_per_week (weekly_hours hourly_wage daily_earnings : ℕ) 
  (h1 : weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63) :
  weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end Lance_workdays_per_week_l67_67225


namespace Danai_can_buy_more_decorations_l67_67653

theorem Danai_can_buy_more_decorations :
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let pumpkins := 24 -- 2 times the number of spiderwebs
  let cauldron := 1
  let planned_total := 83
  let budget_left := 10
  let current_decorations := skulls + broomsticks + spiderwebs + pumpkins + cauldron
  current_decorations = 53 → -- 12 + 4 + 12 + 24 + 1
  let additional_decorations_needed := planned_total - current_decorations
  additional_decorations_needed = 30 → -- 83 - 53
  (additional_decorations_needed - budget_left) = 20 → -- 30 - 10
  True := -- proving the statement
sorry

end Danai_can_buy_more_decorations_l67_67653


namespace noodles_initial_count_l67_67242

theorem noodles_initial_count (noodles_given : ℕ) (noodles_now : ℕ) (initial_noodles : ℕ) 
  (h_given : noodles_given = 12) (h_now : noodles_now = 54) (h_initial_noodles : initial_noodles = noodles_now + noodles_given) : 
  initial_noodles = 66 :=
by 
  rw [h_now, h_given] at h_initial_noodles
  exact h_initial_noodles

-- Adding 'sorry' since the solution steps are not required

end noodles_initial_count_l67_67242


namespace sum_of_roots_of_polynomials_l67_67220

theorem sum_of_roots_of_polynomials :
  ∃ (a b : ℝ), (a^4 - 16 * a^3 + 40 * a^2 - 50 * a + 25 = 0) ∧ (b^4 - 24 * b^3 + 216 * b^2 - 720 * b + 625 = 0) ∧ (a + b = 7 ∨ a + b = 3) :=
by 
  sorry

end sum_of_roots_of_polynomials_l67_67220


namespace square_side_length_l67_67425

theorem square_side_length (x : ℝ) (h : x ^ 2 = 4 * 3) : x = 2 * Real.sqrt 3 :=
by sorry

end square_side_length_l67_67425


namespace intersection_is_correct_l67_67316

-- Defining sets A and B
def setA : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Target intersection set
def setIntersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

-- Theorem to be proved
theorem intersection_is_correct : (setA ∩ setB) = setIntersection :=
by
  -- Proof steps will go here
  sorry

end intersection_is_correct_l67_67316


namespace rectangular_prism_volume_l67_67239

theorem rectangular_prism_volume
  (l w h : ℝ)
  (face1 : l * w = 6)
  (face2 : w * h = 8)
  (face3 : l * h = 12) : l * w * h = 24 := sorry

end rectangular_prism_volume_l67_67239


namespace three_buses_interval_l67_67853

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l67_67853


namespace sum_of_roots_l67_67186

theorem sum_of_roots (a b : ℝ) (h1 : a^2 - 4*a - 2023 = 0) (h2 : b^2 - 4*b - 2023 = 0) : a + b = 4 :=
sorry

end sum_of_roots_l67_67186


namespace Jill_has_5_peaches_l67_67576

-- Define the variables and their relationships
variables (S Jl Jk : ℕ)

-- Declare the conditions as assumptions
axiom Steven_has_14_peaches : S = 14
axiom Jake_has_6_fewer_peaches_than_Steven : Jk = S - 6
axiom Jake_has_3_more_peaches_than_Jill : Jk = Jl + 3

-- Define the theorem to prove Jill has 5 peaches
theorem Jill_has_5_peaches (S Jk Jl : ℕ) 
  (h1 : S = 14) 
  (h2 : Jk = S - 6)
  (h3 : Jk = Jl + 3) : 
  Jl = 5 := 
by
  sorry

end Jill_has_5_peaches_l67_67576


namespace angle_relationship_l67_67712

variables {VU VW : ℝ} {x y z : ℝ} (h1 : VU = VW) 
          (angle_UXZ : ℝ) (angle_VYZ : ℝ) (angle_VZX : ℝ)
          (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z)

theorem angle_relationship (h1 : VU = VW) (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z) : 
    x = (y - z) / 2 := 
by 
    sorry

end angle_relationship_l67_67712


namespace part1_part2_l67_67029

noncomputable def f (x a : ℝ) := (x + 1) * Real.log x - a * (x - 1)

theorem part1 : (∀ x a : ℝ, (x + 1) * Real.log x - a * (x - 1) = x - 1 → a = 1) := 
by sorry

theorem part2 (x : ℝ) (h : 1 < x ∧ x < 2) : 
  ( 1 / Real.log x - 1 / Real.log (x - 1) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end part1_part2_l67_67029


namespace single_elimination_matches_l67_67516

theorem single_elimination_matches (n : ℕ) (h : n = 512) :
  ∃ (m : ℕ), m = n - 1 ∧ m = 511 :=
by
  sorry

end single_elimination_matches_l67_67516


namespace systematic_sampling_first_segment_l67_67679

theorem systematic_sampling_first_segment:
  ∀ (total_students sample_size segment_size 
     drawn_16th drawn_first : ℕ),
  total_students = 160 →
  sample_size = 20 →
  segment_size = 8 →
  drawn_16th = 125 →
  drawn_16th = drawn_first + segment_size * (16 - 1) →
  drawn_first = 5 :=
by
  intros total_students sample_size segment_size drawn_16th drawn_first
         htots hsamp hseg hdrw16 heq
  sorry

end systematic_sampling_first_segment_l67_67679


namespace total_wax_required_l67_67476

/-- Given conditions: -/
def wax_already_have : ℕ := 331
def wax_needed_more : ℕ := 22

/-- Prove the question (the total amount of wax required) -/
theorem total_wax_required :
  (wax_already_have + wax_needed_more) = 353 := by
  sorry

end total_wax_required_l67_67476


namespace original_selling_price_l67_67836

-- Definitions and conditions
def cost_price (CP : ℝ) := CP
def profit (CP : ℝ) := 1.25 * CP
def loss (CP : ℝ) := 0.75 * CP
def loss_price (CP : ℝ) := 600

-- Main theorem statement
theorem original_selling_price (CP : ℝ) (h1 : loss CP = loss_price CP) : profit CP = 1000 :=
by
  -- Note: adding the proof that CP = 800 and then profit CP = 1000 would be here.
  sorry

end original_selling_price_l67_67836


namespace shaded_area_correct_l67_67845

-- Define the side lengths of the squares
def side_length_large_square : ℕ := 14
def side_length_small_square : ℕ := 10

-- Define the areas of the squares
def area_large_square : ℕ := side_length_large_square * side_length_large_square
def area_small_square : ℕ := side_length_small_square * side_length_small_square

-- Define the area of the shaded regions
def area_shaded_regions : ℕ := area_large_square - area_small_square

-- State the theorem
theorem shaded_area_correct : area_shaded_regions = 49 := by
  sorry

end shaded_area_correct_l67_67845


namespace smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l67_67246

theorem smallest_positive_four_digit_integer_equivalent_to_3_mod_4 : 
  ∃ n : ℤ, n ≥ 1000 ∧ n % 4 = 3 ∧ n = 1003 := 
by {
  sorry
}

end smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l67_67246


namespace total_area_of_sheet_l67_67411

theorem total_area_of_sheet (x : ℕ) (h1 : 4 * x - x = 2208) : x + 4 * x = 3680 := 
sorry

end total_area_of_sheet_l67_67411


namespace percent_alcohol_new_solution_l67_67513

theorem percent_alcohol_new_solution :
  let original_volume := 40
  let original_percent_alcohol := 5
  let added_alcohol := 2.5
  let added_water := 7.5
  let original_alcohol := original_volume * (original_percent_alcohol / 100)
  let total_alcohol := original_alcohol + added_alcohol
  let new_total_volume := original_volume + added_alcohol + added_water
  (total_alcohol / new_total_volume) * 100 = 9 :=
by
  sorry

end percent_alcohol_new_solution_l67_67513


namespace sally_needs_8_napkins_l67_67741

theorem sally_needs_8_napkins :
  let tablecloth_length := 102
  let tablecloth_width := 54
  let napkin_length := 6
  let napkin_width := 7
  let total_material_needed := 5844
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let material_needed_for_napkins := total_material_needed - tablecloth_area
  let number_of_napkins := material_needed_for_napkins / napkin_area
  number_of_napkins = 8 :=
by
  sorry

end sally_needs_8_napkins_l67_67741


namespace part1_part2_l67_67807

-- Definitions for sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < -5 ∨ x > 1}

-- Prove (1): A ∪ B
theorem part1 : A ∪ B = {x : ℝ | x < -5 ∨ x > -3} :=
by
  sorry

-- Prove (2): A ∩ (ℝ \ B)
theorem part2 : A ∩ (Set.compl B) = {x : ℝ | -3 < x ∧ x ≤ 1} :=
by
  sorry

end part1_part2_l67_67807


namespace length_of_each_side_is_25_nails_l67_67802

-- Definitions based on the conditions
def nails_per_side := 25
def total_nails := 96

-- The theorem stating the equivalent mathematical problem
theorem length_of_each_side_is_25_nails
  (n : ℕ) (h1 : n = nails_per_side * 4 - 4)
  (h2 : total_nails = 96):
  n = nails_per_side :=
by
  sorry

end length_of_each_side_is_25_nails_l67_67802


namespace problem_equivalence_l67_67809

theorem problem_equivalence : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end problem_equivalence_l67_67809


namespace union_complement_l67_67337

open Set Real

def P : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def Q : Set ℝ := { x | x^2 - 4 < 0 }

theorem union_complement :
  P ∪ (compl Q) = (Iic (-2)) ∪ Ici 1 :=
by
  sorry

end union_complement_l67_67337


namespace solution_set_of_inequality_l67_67669

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) / (x + 3) ≥ 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_inequality_l67_67669


namespace find_c_l67_67829

-- Define the problem conditions and statement

variables (a b c : ℝ) (A B C : ℝ)
variable (cos_C : ℝ)
variable (sin_A sin_B : ℝ)

-- Given conditions
axiom h1 : a = 2
axiom h2 : cos_C = -1/4
axiom h3 : 3 * sin_A = 2 * sin_B
axiom sine_rule : sin_A / a = sin_B / b

-- Using sine rule to derive relation between a and b
axiom h4 : 3 * a = 2 * b

-- Cosine rule axiom
axiom cosine_rule : c^2 = a^2 + b^2 - 2 * a * b * cos_C

-- Prove c = 4
theorem find_c : c = 4 :=
by
  sorry

end find_c_l67_67829


namespace greatest_integer_e_minus_5_l67_67602

theorem greatest_integer_e_minus_5 (e : ℝ) (h : 2 < e ∧ e < 3) : ⌊e - 5⌋ = -3 :=
by
  sorry

end greatest_integer_e_minus_5_l67_67602


namespace directrix_of_parabola_l67_67383

theorem directrix_of_parabola (y x : ℝ) (h : y = 4 * x^2) : y = - (1 / 16) :=
sorry

end directrix_of_parabola_l67_67383


namespace trig_function_value_l67_67289

noncomputable def f : ℝ → ℝ := sorry

theorem trig_function_value:
  (∀ x, f (Real.cos x) = Real.cos (3 * x)) →
  f (Real.sin (Real.pi / 6)) = -1 :=
by
  intro h
  sorry

end trig_function_value_l67_67289


namespace avg_xy_l67_67441

theorem avg_xy (x y : ℝ) (h : (4 + 6.5 + 8 + x + y) / 5 = 18) : (x + y) / 2 = 35.75 :=
by
  sorry

end avg_xy_l67_67441


namespace geom_seq_product_l67_67091

-- Given conditions
variables (a : ℕ → ℝ)
variable (r : ℝ)
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom a1_eq_1 : a 1 = 1
axiom a10_eq_3 : a 10 = 3

-- Proof goal
theorem geom_seq_product : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 :=  
sorry

end geom_seq_product_l67_67091


namespace find_x_81_9_729_l67_67620

theorem find_x_81_9_729
  (x : ℝ)
  (h : (81 : ℝ)^(x-2) / (9 : ℝ)^(x-2) = (729 : ℝ)^(2*x-1)) :
  x = 1/5 :=
sorry

end find_x_81_9_729_l67_67620


namespace smaller_number_l67_67320

theorem smaller_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : y = 15 :=
sorry

end smaller_number_l67_67320


namespace problem_difference_l67_67437

-- Define the sum of first n natural numbers
def sumFirstN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Define the rounding rule to the nearest multiple of 5
def roundToNearest5 (x : ℕ) : ℕ :=
  match x % 5 with
  | 0 => x
  | 1 => x - 1
  | 2 => x - 2
  | 3 => x + 2
  | 4 => x + 1
  | _ => x  -- This case is theoretically unreachable

-- Define the sum of the first n natural numbers after rounding to nearest 5
def sumRoundedFirstN (n : ℕ) : ℕ :=
  (List.range (n + 1)).map roundToNearest5 |>.sum

theorem problem_difference : sumFirstN 120 - sumRoundedFirstN 120 = 6900 := by
  sorry

end problem_difference_l67_67437


namespace remainder_div_5_l67_67142

theorem remainder_div_5 (n : ℕ): (∃ k : ℤ, n = 10 * k + 7) → (∃ m : ℤ, n = 5 * m + 2) :=
by
  sorry

end remainder_div_5_l67_67142


namespace quadratic_two_real_roots_find_m_l67_67990

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l67_67990


namespace proof_C_ST_l67_67287

-- Definitions for sets and their operations
def A1 : Set ℕ := {0, 1}
def A2 : Set ℕ := {1, 2}
def S : Set ℕ := A1 ∪ A2
def T : Set ℕ := A1 ∩ A2
def C_ST : Set ℕ := S \ T

theorem proof_C_ST : 
  C_ST = {0, 2} := 
by 
  sorry

end proof_C_ST_l67_67287


namespace line_passes_through_fixed_point_l67_67471

variable {a b : ℝ}

theorem line_passes_through_fixed_point : 
  (∀ (x y : ℝ), a + 2 * b = 1 ∧ ax + 3 * y + b = 0 → (x, y) = (1/2, -1/6)) :=
by
  sorry

end line_passes_through_fixed_point_l67_67471


namespace journey_ratio_l67_67030

/-- Given a full-circle journey broken into parts,
  including paths through the Zoo Park (Z), the Circus (C), and the Park (P), 
  prove that the journey avoiding the Zoo Park is 11 times shorter. -/
theorem journey_ratio (Z C P : ℝ) (h1 : C = (3 / 4) * Z) 
                      (h2 : P = (1 / 4) * Z) : 
  Z = 11 * P := 
sorry

end journey_ratio_l67_67030


namespace rationalize_sqrt_l67_67691

theorem rationalize_sqrt (h : Real.sqrt 35 ≠ 0) : 35 / Real.sqrt 35 = Real.sqrt 35 := 
by 
sorry

end rationalize_sqrt_l67_67691


namespace rectangle_x_is_18_l67_67882

-- Definitions for the conditions
def rectangle (a b x : ℕ) : Prop := 
  (a = 2 * b) ∧
  (x = 2 * (a + b)) ∧
  (x = a * b)

-- Theorem to prove the equivalence of the conditions and the answer \( x = 18 \)
theorem rectangle_x_is_18 : ∀ a b x : ℕ, rectangle a b x → x = 18 :=
by
  sorry

end rectangle_x_is_18_l67_67882


namespace rectangle_perimeter_l67_67862

theorem rectangle_perimeter (z w : ℝ) (h : z > w) :
  (2 * ((z - w) + w)) = 2 * z := by
  sorry

end rectangle_perimeter_l67_67862


namespace no_positive_reals_satisfy_equations_l67_67937

theorem no_positive_reals_satisfy_equations :
  ¬ ∃ (a b c d : ℝ), (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧
  (a / b + b / c + c / d + d / a = 6) ∧ (b / a + c / b + d / c + a / d = 32) :=
by sorry

end no_positive_reals_satisfy_equations_l67_67937


namespace sum_of_squares_l67_67931

/-- 
Given two real numbers x and y, if their product is 120 and their sum is 23, 
then the sum of their squares is 289.
-/
theorem sum_of_squares (x y : ℝ) (h₁ : x * y = 120) (h₂ : x + y = 23) :
  x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l67_67931


namespace multiplier_for_difference_l67_67408

variable (x y k : ℕ)
variable (h1 : x + y = 81)
variable (h2 : x^2 - y^2 = k * (x - y))
variable (h3 : x ≠ y)

theorem multiplier_for_difference : k = 81 := 
by
  sorry

end multiplier_for_difference_l67_67408


namespace solution_inequality_l67_67248

theorem solution_inequality {x : ℝ} : x - 1 > 0 ↔ x > 1 := 
by
  sorry

end solution_inequality_l67_67248


namespace Total_Cookies_is_135_l67_67589

-- Define the number of cookies in each pack
def PackA_Cookies : ℕ := 15
def PackB_Cookies : ℕ := 30
def PackC_Cookies : ℕ := 45

-- Define the number of packs bought by Paul and Paula
def Paul_PackA_Count : ℕ := 1
def Paul_PackB_Count : ℕ := 2
def Paula_PackA_Count : ℕ := 1
def Paula_PackC_Count : ℕ := 1

-- Calculate total cookies for Paul
def Paul_Cookies : ℕ := (Paul_PackA_Count * PackA_Cookies) + (Paul_PackB_Count * PackB_Cookies)

-- Calculate total cookies for Paula
def Paula_Cookies : ℕ := (Paula_PackA_Count * PackA_Cookies) + (Paula_PackC_Count * PackC_Cookies)

-- Calculate total cookies for Paul and Paula together
def Total_Cookies : ℕ := Paul_Cookies + Paula_Cookies

theorem Total_Cookies_is_135 : Total_Cookies = 135 := by
  sorry

end Total_Cookies_is_135_l67_67589


namespace min_value_of_quadratic_l67_67405

theorem min_value_of_quadratic : ∀ x : ℝ, (x^2 + 6*x + 5) ≥ -4 :=
by 
  sorry

end min_value_of_quadratic_l67_67405


namespace smallest_integer_value_of_x_l67_67787

theorem smallest_integer_value_of_x (x : ℤ) (h : 7 + 3 * x < 26) : x = 6 :=
sorry

end smallest_integer_value_of_x_l67_67787


namespace geom_seq_min_value_l67_67286

noncomputable def minimum_sum (m n : ℕ) (a : ℕ → ℝ) : ℝ :=
  if (a 7 = a 6 + 2 * a 5) ∧ (a m * a n = 16 * (a 1) ^ 2) ∧ (m > 0) ∧ (n > 0) then
    (1 / m) + (4 / n)
  else
    0

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n, a m * a n = 16 * (a 1) ^ 2 ∧ m > 0 ∧ n > 0) →
  (minimum_sum m n a = 3 / 2) := sorry

end geom_seq_min_value_l67_67286


namespace sum_of_abs_of_coefficients_l67_67468

theorem sum_of_abs_of_coefficients :
  ∃ a_0 a_2 a_4 a_1 a_3 a_5 : ℤ, 
    ((2*x - 1)^5 + (x + 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) ∧
    (|a_0| + |a_2| + |a_4| = 110) :=
by
  sorry

end sum_of_abs_of_coefficients_l67_67468


namespace solve_system_l67_67177

theorem solve_system :
  ∀ (a1 a2 c1 c2 x y : ℝ),
  (a1 * 5 + 10 = c1) →
  (a2 * 5 + 10 = c2) →
  (a1 * x + 2 * y = a1 - c1) →
  (a2 * x + 2 * y = a2 - c2) →
  (x = -4) ∧ (y = -5) := by
  intros a1 a2 c1 c2 x y h1 h2 h3 h4
  sorry

end solve_system_l67_67177


namespace domain_of_f_parity_of_f_range_of_f_l67_67457

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a - Real.log (1 - x) / Real.log a

variables {a x : ℝ}

-- The properties derived:
theorem domain_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (-1 < x ∧ x < 1) ↔ ∃ y, f a x = y :=
sorry

theorem parity_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  f a (-x) = -f a x :=
sorry

theorem range_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (f a x > 0 ↔ (a > 1 ∧ 0 < x ∧ x < 1) ∨ (0 < a ∧ a < 1 ∧ -1 < x ∧ x < 0)) :=
sorry

end domain_of_f_parity_of_f_range_of_f_l67_67457


namespace minimum_final_percentage_to_pass_l67_67565

-- Conditions
def problem_sets : ℝ := 100
def midterm_worth : ℝ := 100
def final_worth : ℝ := 300
def perfect_problem_sets_score : ℝ := 100
def midterm1_score : ℝ := 0.60 * midterm_worth
def midterm2_score : ℝ := 0.70 * midterm_worth
def midterm3_score : ℝ := 0.80 * midterm_worth
def passing_percentage : ℝ := 0.70

-- Derived Values
def total_points_available : ℝ := problem_sets + 3 * midterm_worth + final_worth
def required_points_to_pass : ℝ := passing_percentage * total_points_available
def total_points_before_final : ℝ := perfect_problem_sets_score + midterm1_score + midterm2_score + midterm3_score
def points_needed_from_final : ℝ := required_points_to_pass - total_points_before_final

-- Proof Statement
theorem minimum_final_percentage_to_pass : 
  ∃ (final_score : ℝ), (final_score / final_worth * 100) ≥ 60 :=
by
  -- Calculations for proof
  let required_final_percentage := (points_needed_from_final / final_worth) * 100
  -- We need to show that the required percentage is at least 60%
  have : required_final_percentage = 60 := sorry
  exact Exists.intro 180 sorry

end minimum_final_percentage_to_pass_l67_67565


namespace minimize_relative_waiting_time_l67_67061

-- Definitions of task times in seconds
def task_U : ℕ := 10
def task_V : ℕ := 120
def task_W : ℕ := 900

-- Definition of relative waiting time given a sequence of task execution times
def relative_waiting_time (times : List ℕ) : ℚ :=
  (times.head! : ℚ) / (times.head! : ℚ) + 
  (times.head! + times.tail.head! : ℚ) / (times.tail.head! : ℚ) + 
  (times.head! + times.tail.head! + times.tail.tail.head! : ℚ) / (times.tail.tail.head! : ℚ)

-- Sequences
def sequence_A : List ℕ := [task_U, task_V, task_W]
def sequence_B : List ℕ := [task_V, task_W, task_U]
def sequence_C : List ℕ := [task_W, task_U, task_V]
def sequence_D : List ℕ := [task_U, task_W, task_V]

-- Sum of relative waiting times for each sequence
def S_A := relative_waiting_time sequence_A
def S_B := relative_waiting_time sequence_B
def S_C := relative_waiting_time sequence_C
def S_D := relative_waiting_time sequence_D

-- Theorem to prove that sequence A has the minimum sum of relative waiting times
theorem minimize_relative_waiting_time : S_A < S_B ∧ S_A < S_C ∧ S_A < S_D := 
  by sorry

end minimize_relative_waiting_time_l67_67061


namespace total_sets_needed_l67_67026

-- Conditions
variable (n : ℕ)

-- Theorem statement
theorem total_sets_needed : 3 * n = 3 * n :=
by sorry

end total_sets_needed_l67_67026


namespace value_of_k_l67_67873

noncomputable def roots_in_ratio_equation {k : ℝ} (h : k ≠ 0) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ 
  (r₁ / r₂ = 3) ∧ 
  (r₁ + r₂ = -8) ∧ 
  (r₁ * r₂ = k)

theorem value_of_k (k : ℝ) (h : k ≠ 0) (hr : roots_in_ratio_equation h) : k = 12 :=
sorry

end value_of_k_l67_67873


namespace choose_correct_graph_l67_67048

noncomputable def appropriate_graph : String :=
  let bar_graph := "Bar graph"
  let pie_chart := "Pie chart"
  let line_graph := "Line graph"
  let freq_dist_graph := "Frequency distribution graph"
  
  if (bar_graph = "Bar graph") ∧ (pie_chart = "Pie chart") ∧ (line_graph = "Line graph") ∧ (freq_dist_graph = "Frequency distribution graph") then
    "Line graph"
  else
    sorry

theorem choose_correct_graph :
  appropriate_graph = "Line graph" :=
by
  sorry

end choose_correct_graph_l67_67048


namespace pieces_present_l67_67951

-- Define the pieces and their counts in a standard chess set
def total_pieces := 32
def missing_pieces := 12
def missing_kings := 1
def missing_queens := 2
def missing_knights := 3
def missing_pawns := 6

-- The theorem statement that we need to prove
theorem pieces_present : 
  (total_pieces - (missing_kings + missing_queens + missing_knights + missing_pawns)) = 20 :=
by
  sorry

end pieces_present_l67_67951


namespace chord_on_ellipse_midpoint_l67_67154

theorem chord_on_ellipse_midpoint :
  ∀ (A B : ℝ × ℝ)
    (hx1 : (A.1^2) / 2 + A.2^2 = 1)
    (hx2 : (B.1^2) / 2 + B.2^2 = 1)
    (mid : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 1/2),
  ∃ (k : ℝ), ∀ (x y : ℝ), y - 1/2 = k * (x - 1/2) ↔ 2 * x + 4 * y = 3 := 
sorry

end chord_on_ellipse_midpoint_l67_67154


namespace raine_steps_l67_67254

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end raine_steps_l67_67254


namespace parametric_inclination_l67_67587

noncomputable def angle_of_inclination (x y : ℝ) : ℝ := 50

theorem parametric_inclination (t : ℝ) (x y : ℝ) :
  x = t * Real.sin 40 → y = -1 + t * Real.cos 40 → angle_of_inclination x y = 50 :=
by
  intros hx hy
  -- This is where the proof would go, but we skip it.
  sorry

end parametric_inclination_l67_67587


namespace non_coincident_angles_l67_67528

theorem non_coincident_angles : ¬ ∃ k : ℤ, 1050 - (-300) = k * 360 := by
  sorry

end non_coincident_angles_l67_67528


namespace point_in_fourth_quadrant_l67_67832

def point (x y : ℝ) := (x, y)
def x_positive (p : ℝ × ℝ) : Prop := p.1 > 0
def y_negative (p : ℝ × ℝ) : Prop := p.2 < 0
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := x_positive p ∧ y_negative p

theorem point_in_fourth_quadrant : in_fourth_quadrant (2, -4) :=
by
  -- The proof states that (2, -4) is in the fourth quadrant.
  sorry

end point_in_fourth_quadrant_l67_67832


namespace tangent_line_equation_l67_67499

theorem tangent_line_equation (a : ℝ) (h : a ≠ 0) :
  (∃ b : ℝ, b = 2 ∧ (∀ x : ℝ, y = a * x^2) ∧ y - a = b * (x - 1)) → 
  ∃ (x y : ℝ), 2 * x - y - 1 = 0 :=
by
  sorry

end tangent_line_equation_l67_67499


namespace range_of_k_l67_67928

theorem range_of_k (x y k : ℝ) 
  (h1 : 2 * x + y = k + 1) 
  (h2 : x + 2 * y = 2) 
  (h3 : x + y < 0) : 
  k < -3 :=
sorry

end range_of_k_l67_67928


namespace opposite_of_five_l67_67781

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l67_67781


namespace most_lines_of_symmetry_l67_67486

def regular_pentagon_lines_of_symmetry : ℕ := 5
def kite_lines_of_symmetry : ℕ := 1
def regular_hexagon_lines_of_symmetry : ℕ := 6
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def scalene_triangle_lines_of_symmetry : ℕ := 0

theorem most_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry = max
    (max (max (max regular_pentagon_lines_of_symmetry kite_lines_of_symmetry)
              regular_hexagon_lines_of_symmetry)
        isosceles_triangle_lines_of_symmetry)
    scalene_triangle_lines_of_symmetry :=
sorry

end most_lines_of_symmetry_l67_67486


namespace black_cards_taken_out_l67_67642

theorem black_cards_taken_out (initial_black : ℕ) (remaining_black : ℕ) (total_cards : ℕ) (black_cards_per_deck : ℕ) :
  total_cards = 52 → black_cards_per_deck = 26 →
  initial_black = black_cards_per_deck → remaining_black = 22 →
  initial_black - remaining_black = 4 := by
  intros
  sorry

end black_cards_taken_out_l67_67642


namespace domain_condition_l67_67874

variable (k : ℝ)
def quadratic_expression (x : ℝ) : ℝ := k * x^2 - 4 * k * x + k + 8

theorem domain_condition (k : ℝ) : (∀ x : ℝ, quadratic_expression k x > 0) ↔ (0 ≤ k ∧ k < 8/3) :=
sorry

end domain_condition_l67_67874


namespace fraction_of_Charlie_circumference_l67_67992

/-- Definitions for the problem conditions -/
def Jack_head_circumference : ℕ := 12
def Charlie_head_circumference : ℕ := 9 + Jack_head_circumference / 2
def Bill_head_circumference : ℕ := 10

/-- Statement of the theorem to be proved -/
theorem fraction_of_Charlie_circumference :
  Bill_head_circumference / Charlie_head_circumference = 2 / 3 :=
sorry

end fraction_of_Charlie_circumference_l67_67992


namespace number_exceeds_20_percent_by_40_eq_50_l67_67720

theorem number_exceeds_20_percent_by_40_eq_50 (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 := by
  sorry

end number_exceeds_20_percent_by_40_eq_50_l67_67720


namespace right_triangle_ratio_l67_67848

theorem right_triangle_ratio (a b c r s : ℝ) (h_right_angle : (a:ℝ)^2 + (b:ℝ)^2 = c^2)
  (h_perpendicular : ∀ h : ℝ, c = r + s)
  (h_ratio_ab : a / b = 2 / 5)
  (h_geometry_r : r = a^2 / c)
  (h_geometry_s : s = b^2 / c) :
  r / s = 4 / 25 :=
sorry

end right_triangle_ratio_l67_67848


namespace smallest_n_with_290_trailing_zeros_in_factorial_l67_67508

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 5^2) + (n / 5^3) + (n / 5^4) + (n / 5^5) + (n / 5^6) -- sum until the division becomes zero

theorem smallest_n_with_290_trailing_zeros_in_factorial : 
  ∀ (n : ℕ), n >= 1170 ↔ trailing_zeros n >= 290 ∧ trailing_zeros (n-1) < 290 := 
by { sorry }

end smallest_n_with_290_trailing_zeros_in_factorial_l67_67508


namespace find_incorrect_expression_l67_67914

variable {x y : ℚ}

theorem find_incorrect_expression
  (h : x / y = 5 / 6) :
  ¬ (
    (x + 3 * y) / x = 23 / 5
  ) := by
  sorry

end find_incorrect_expression_l67_67914


namespace quadratic_expression_rewrite_l67_67843

theorem quadratic_expression_rewrite (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 :=
sorry

end quadratic_expression_rewrite_l67_67843


namespace rational_xyz_squared_l67_67940

theorem rational_xyz_squared
  (x y z : ℝ)
  (hx : ∃ r1 : ℚ, x + y * z = r1)
  (hy : ∃ r2 : ℚ, y + z * x = r2)
  (hz : ∃ r3 : ℚ, z + x * y = r3)
  (hxy : x^2 + y^2 = 1) :
  ∃ r4 : ℚ, x * y * z^2 = r4 := 
sorry

end rational_xyz_squared_l67_67940


namespace find_P_Q_R_l67_67704

theorem find_P_Q_R :
  ∃ P Q R : ℝ, (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → 
    (5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2)) 
    ∧ P = 5 ∧ Q = -5 ∧ R = -5 :=
by
  sorry

end find_P_Q_R_l67_67704


namespace village_population_decrease_rate_l67_67660

theorem village_population_decrease_rate :
  ∃ (R : ℝ), 15 * R = 18000 :=
by
  sorry

end village_population_decrease_rate_l67_67660


namespace fraction_evaluation_l67_67583

theorem fraction_evaluation :
  (1/5 - 1/7) / (3/8 + 2/9) = 144/1505 := 
  by 
    sorry

end fraction_evaluation_l67_67583


namespace least_number_div_condition_l67_67388

theorem least_number_div_condition (m : ℕ) : 
  (∃ k r : ℕ, m = 34 * k + r ∧ m = 5 * (r + 8) ∧ r < 34) → m = 162 := 
by
  sorry

end least_number_div_condition_l67_67388


namespace count_restricted_arrangements_l67_67818

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l67_67818


namespace length_AP_eq_sqrt2_l67_67742

/-- In square ABCD with side length 2, a circle ω with center at (1, 0)
    and radius 1 is inscribed. The circle intersects CD at point M,
    and line AM intersects ω at a point P different from M.
    Prove that the length of AP is √2. -/
theorem length_AP_eq_sqrt2 :
  let A := (0, 2)
  let M := (2, 0)
  let P : ℝ × ℝ := (1, 1)
  dist A P = Real.sqrt 2 :=
by
  sorry

end length_AP_eq_sqrt2_l67_67742


namespace factorization_correct_l67_67462

theorem factorization_correct :
    (∀ (x y : ℝ), x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y)) :=
by
  intro x y
  sorry

end factorization_correct_l67_67462


namespace second_grade_girls_l67_67128

theorem second_grade_girls (G : ℕ) 
  (h1 : ∃ boys_2nd : ℕ, boys_2nd = 20)
  (h2 : ∃ students_3rd : ℕ, students_3rd = 2 * (20 + G))
  (h3 : 20 + G + (2 * (20 + G)) = 93) :
  G = 11 :=
by
  sorry

end second_grade_girls_l67_67128


namespace smallest_distance_proof_l67_67133

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  Complex.abs (z - w)

theorem smallest_distance_proof (z w : ℂ) 
  (h1 : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (h2 : Complex.abs (w - (-5 + 6*Complex.I)) = 4) :
  smallest_distance z w ≥ Real.sqrt 149 - 6 :=
by
  sorry

end smallest_distance_proof_l67_67133


namespace range_of_m_for_real_roots_value_of_m_for_specific_roots_l67_67716

open Real

variable {m x : ℝ}

def quadratic (m : ℝ) (x : ℝ) := x^2 + 2*(m-1)*x + m^2 + 2 = 0
  
theorem range_of_m_for_real_roots (h : ∃ x : ℝ, quadratic m x) : m ≤ -1/2 :=
sorry

theorem value_of_m_for_specific_roots
  (h : quadratic m x)
  (Hroots : ∃ x1 x2 : ℝ, quadratic m x1 ∧ quadratic m x2 ∧ (x1 - x2)^2 = 18 - x1 * x2) :
  m = -2 :=
sorry

end range_of_m_for_real_roots_value_of_m_for_specific_roots_l67_67716


namespace correct_exponentiation_l67_67340

theorem correct_exponentiation (x : ℝ) : x^2 * x^3 = x^5 :=
by sorry

end correct_exponentiation_l67_67340


namespace simplify_cbrt_8000_eq_21_l67_67595

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l67_67595


namespace parallel_lines_l67_67280

theorem parallel_lines (m : ℝ) 
  (h : 3 * (m - 2) + m * (m + 2) = 0) 
  : m = 1 ∨ m = -6 := 
by 
  sorry

end parallel_lines_l67_67280


namespace functional_equation_solution_l67_67991

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution : 
  (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) 
  → ∃ c : ℝ, (c = 0 ∨ (1 ≤ c ∧ c < 2)) ∧ (∀ x : ℝ, f x = c) :=
by
  intro h
  sorry

end functional_equation_solution_l67_67991


namespace fred_change_received_l67_67226

theorem fred_change_received :
  let ticket_price := 5.92
  let ticket_count := 2
  let borrowed_movie_price := 6.79
  let amount_paid := 20.00
  let total_cost := (ticket_price * ticket_count) + borrowed_movie_price
  let change := amount_paid - total_cost
  change = 1.37 :=
by
  sorry

end fred_change_received_l67_67226


namespace quoted_price_correct_l67_67202

noncomputable def after_tax_yield (yield : ℝ) (tax_rate : ℝ) : ℝ :=
  yield * (1 - tax_rate)

noncomputable def real_yield (after_tax_yield : ℝ) (inflation_rate : ℝ) : ℝ :=
  after_tax_yield - inflation_rate

noncomputable def quoted_price (dividend_rate : ℝ) (real_yield : ℝ) (commission_rate : ℝ) : ℝ :=
  real_yield / (dividend_rate / (1 + commission_rate))

theorem quoted_price_correct :
  quoted_price 0.16 (real_yield (after_tax_yield 0.08 0.15) 0.03) 0.02 = 24.23 :=
by
  -- This is the proof statement. Since the task does not require us to prove it, we use 'sorry'.
  sorry

end quoted_price_correct_l67_67202


namespace rectangle_side_length_relation_l67_67072

variable (x y : ℝ)

-- Condition: The area of the rectangle is 10
def is_rectangle_area_10 (x y : ℝ) : Prop := x * y = 10

-- Theorem: Given the area condition, express y in terms of x
theorem rectangle_side_length_relation (h : is_rectangle_area_10 x y) : y = 10 / x :=
sorry

end rectangle_side_length_relation_l67_67072


namespace complex_addition_l67_67279

theorem complex_addition :
  (⟨6, -5⟩ : ℂ) + (⟨3, 2⟩ : ℂ) = ⟨9, -3⟩ := 
sorry

end complex_addition_l67_67279


namespace mark_weekly_leftover_l67_67365

def initial_hourly_wage := 40
def raise_percentage := 5 / 100
def daily_hours := 8
def weekly_days := 5
def old_weekly_bills := 600
def personal_trainer_cost := 100

def new_hourly_wage := initial_hourly_wage * (1 + raise_percentage)
def weekly_hours := daily_hours * weekly_days
def weekly_earnings := new_hourly_wage * weekly_hours
def new_weekly_expenses := old_weekly_bills + personal_trainer_cost
def leftover_per_week := weekly_earnings - new_weekly_expenses

theorem mark_weekly_leftover : leftover_per_week = 980 := by
  sorry

end mark_weekly_leftover_l67_67365


namespace cost_price_per_meter_l67_67084

-- Definitions based on the conditions given in the problem
def meters_of_cloth : ℕ := 45
def selling_price : ℕ := 4500
def profit_per_meter : ℕ := 12

-- Statement to prove
theorem cost_price_per_meter :
  (selling_price - (profit_per_meter * meters_of_cloth)) / meters_of_cloth = 88 :=
by
  sorry

end cost_price_per_meter_l67_67084


namespace evaluate_expression_l67_67420

-- Define the terms a and b
def a : ℕ := 2023
def b : ℕ := 2024

-- The given expression
def expression : ℤ := (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b)

-- The theorem to prove
theorem evaluate_expression : expression = ↑a := 
by sorry

end evaluate_expression_l67_67420


namespace simplify_expression_l67_67859

theorem simplify_expression (x : ℝ) : 
  (12 * x ^ 12 - 3 * x ^ 10 + 5 * x ^ 9) + (-1 * x ^ 12 + 2 * x ^ 10 + x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  11 * x ^ 12 - x ^ 10 + 6 * x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
by
  sorry

end simplify_expression_l67_67859


namespace find_m_and_other_root_l67_67391

theorem find_m_and_other_root (m x_2 : ℝ) :
  (∃ (x_1 : ℝ), x_1 = -1 ∧ x_1^2 + m * x_1 - 5 = 0) →
  m = -4 ∧ ∃ (x_2 : ℝ), x_2 = 5 ∧ x_2^2 + m * x_2 - 5 = 0 :=
by
  sorry

end find_m_and_other_root_l67_67391


namespace number_of_boys_l67_67973

-- Definitions of the conditions
def total_members (B G : ℕ) : Prop := B + G = 26
def meeting_attendance (B G : ℕ) : Prop := (1 / 2 : ℚ) * G + B = 16

-- Theorem statement
theorem number_of_boys (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : B = 6 := by
  sorry

end number_of_boys_l67_67973


namespace find_BP_l67_67549

theorem find_BP
  (A B C D P : Type) 
  (AP PC BP DP : ℝ)
  (hAP : AP = 8) 
  (hPC : PC = 1)
  (hBD : BD = 6)
  (hBP_less_DP : BP < DP) 
  (hPower_of_Point : AP * PC = BP * DP)
  : BP = 2 := 
by {
  sorry
}

end find_BP_l67_67549


namespace savings_in_july_l67_67196

-- Definitions based on the conditions
def savings_june : ℕ := 27
def savings_august : ℕ := 21
def expenses_books : ℕ := 5
def expenses_shoes : ℕ := 17
def final_amount_left : ℕ := 40

-- Main theorem stating the problem
theorem savings_in_july (J : ℕ) : 
  savings_june + J + savings_august - (expenses_books + expenses_shoes) = final_amount_left → 
  J = 14 :=
by
  sorry

end savings_in_july_l67_67196


namespace engineer_walk_duration_l67_67264

variables (D : ℕ) (S : ℕ) (v : ℕ) (t : ℕ) (t1 : ℕ)

-- Stating the conditions
-- The time car normally takes to travel distance D
-- Speed (S) times the time (t) equals distance (D)
axiom speed_distance_relation : S * t = D

-- Engineer arrives at station at 7:00 AM and walks towards the car
-- They meet at t1 minutes past 7:00 AM, and the car covers part of the distance
-- Engineer reaches factory 20 minutes earlier than usual
-- Therefore, the car now meets the engineer covering less distance and time
axiom car_meets_engineer : S * t1 + v * t1 = D

-- The total travel time to the factory is reduced by 20 minutes
axiom travel_time_reduction : t - t1 = (t - 20 / 60)

-- Mathematically equivalent proof problem
theorem engineer_walk_duration : t1 = 50 := by
  sorry

end engineer_walk_duration_l67_67264


namespace maria_ends_up_with_22_towels_l67_67686

-- Define the number of green towels Maria bought
def green_towels : Nat := 35

-- Define the number of white towels Maria bought
def white_towels : Nat := 21

-- Define the number of towels Maria gave to her mother
def given_towels : Nat := 34

-- Total towels Maria initially bought
def total_towels := green_towels + white_towels

-- Towels Maria ended up with
def remaining_towels := total_towels - given_towels

theorem maria_ends_up_with_22_towels :
  remaining_towels = 22 :=
by
  sorry

end maria_ends_up_with_22_towels_l67_67686


namespace complement_U_A_inter_B_eq_l67_67556

open Set

-- Definitions
def U : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def A : Set ℤ := {-2, 0, 2, 4}
def B : Set ℤ := {-2, 0, 4, 6, 8}

-- Complement of A in U
def complement_U_A : Set ℤ := U \ A

-- Proof Problem
theorem complement_U_A_inter_B_eq : complement_U_A ∩ B = {6, 8} := by
  sorry

end complement_U_A_inter_B_eq_l67_67556


namespace aluminum_weight_proportional_l67_67551

noncomputable def area_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * side_length * Real.sqrt 3) / 4

theorem aluminum_weight_proportional (weight1 weight2 : ℝ) 
  (side_length1 side_length2 : ℝ)
  (h_density_thickness : ∀ s t, area_equilateral_triangle s * weight1 = area_equilateral_triangle t * weight2)
  (h_weight1 : weight1 = 20)
  (h_side_length1 : side_length1 = 2)
  (h_side_length2 : side_length2 = 4) : 
  weight2 = 80 :=
by
  sorry

end aluminum_weight_proportional_l67_67551


namespace find_m_n_sum_product_l67_67575

noncomputable def sum_product_of_roots (m n : ℝ) : Prop :=
  (m^2 - 4*m - 12 = 0) ∧ (n^2 - 4*n - 12 = 0) 

theorem find_m_n_sum_product (m n : ℝ) (h : sum_product_of_roots m n) :
  m + n + m * n = -8 :=
by 
  sorry

end find_m_n_sum_product_l67_67575


namespace find_k_l67_67708

def distances (S x y k : ℝ) := (S - x * 0.75) * x / (x + y) + 0.75 * x = S * x / (x + y) - 18 ∧
                              S * x / (x + y) - (S - y / 3) * x / (x + y) = k

theorem find_k (S x y k : ℝ) (h₁ : x * y / (x + y) = 24) (h₂ : k = 24 / 3)
  : k = 8 :=
by 
  -- We need to fill in the proof steps here
  sorry

end find_k_l67_67708


namespace division_addition_problem_l67_67598

-- Define the terms used in the problem
def ten : ℕ := 10
def one_fifth : ℚ := 1 / 5
def six : ℕ := 6

-- Define the math problem
theorem division_addition_problem :
  (ten / one_fifth : ℚ) + six = 56 :=
by sorry

end division_addition_problem_l67_67598


namespace female_rainbow_trout_l67_67314

-- Define the conditions given in the problem
variables (F_s M_s M_r F_r T : ℕ)
variables (h1 : F_s + M_s = 645)
variables (h2 : M_s = 2 * F_s + 45)
variables (h3 : 4 * M_r = 3 * F_s)
variables (h4 : 20 * M_r = 3 * T)
variables (h5 : T = 645 + F_r + M_r)

theorem female_rainbow_trout :
  F_r = 205 :=
by
  sorry

end female_rainbow_trout_l67_67314


namespace probability_one_card_each_l67_67042

-- Define the total number of cards
def total_cards := 12

-- Define the number of cards from Adrian
def adrian_cards := 7

-- Define the number of cards from Bella
def bella_cards := 5

-- Calculate the probability of one card from each cousin when selecting two cards without replacement
theorem probability_one_card_each :
  (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
  (bella_cards / total_cards) * (adrian_cards / (total_cards - 1)) =
  35 / 66 := sorry

end probability_one_card_each_l67_67042


namespace toothpicks_in_300th_stage_l67_67306

/-- 
Prove that the number of toothpicks needed for the 300th stage is 1201, given:
1. The first stage has 5 toothpicks.
2. Each subsequent stage adds 4 toothpicks to the previous stage.
-/
theorem toothpicks_in_300th_stage :
  let a_1 := 5
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 300 = 1201 := by
  sorry

end toothpicks_in_300th_stage_l67_67306


namespace line_length_l67_67814

theorem line_length (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := 
by
  sorry

end line_length_l67_67814


namespace coins_fit_in_new_box_l67_67699

-- Definitions
def diameters_bound (d : ℕ) : Prop :=
  d ≤ 10

def box_fits (length width : ℕ) (fits : Prop) : Prop :=
  fits

-- Conditions
axiom coins_diameter_bound : ∀ (d : ℕ), diameters_bound d
axiom original_box_fits : box_fits 30 70 True

-- Proof statement
theorem coins_fit_in_new_box : box_fits 40 60 True :=
sorry

end coins_fit_in_new_box_l67_67699


namespace range_of_a_exists_distinct_x1_x2_eq_f_l67_67380

noncomputable
def f (a x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem range_of_a_exists_distinct_x1_x2_eq_f :
  { a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2 } = 
  { a : ℝ | (a > (2 / 3)) ∨ (a ≤ 0) } :=
sorry

end range_of_a_exists_distinct_x1_x2_eq_f_l67_67380


namespace find_f_prime_at_2_l67_67178

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 + 2 * x * f' 2 - Real.log x

theorem find_f_prime_at_2 (f' : ℝ → ℝ) (h : ∀ x, deriv (f f') x = f' x) :
  f' 2 = -7 / 2 :=
by
  have H := h 2
  sorry

end find_f_prime_at_2_l67_67178


namespace find_constant_c_and_t_l67_67016

noncomputable def exists_constant_c_and_t (c : ℝ) (t : ℝ) : Prop :=
∀ (x1 x2 m : ℝ), (x1^2 - m * x1 - c = 0) ∧ (x2^2 - m * x2 - c = 0) →
  (t = 1 / ((1 + m^2) * x1^2) + 1 / ((1 + m^2) * x2^2))

theorem find_constant_c_and_t : ∃ (c t : ℝ), exists_constant_c_and_t c t ∧ c = 2 ∧ t = 3 / 2 :=
sorry

end find_constant_c_and_t_l67_67016


namespace car_total_travel_time_l67_67290

def T_NZ : ℕ := 60

def T_NR : ℕ := 8 / 10 * T_NZ -- 80% of T_NZ

def T_ZV : ℕ := 3 / 4 * T_NR -- 75% of T_NR

theorem car_total_travel_time :
  T_NZ + T_NR + T_ZV = 144 := by
  sorry

end car_total_travel_time_l67_67290


namespace angle_in_quadrant_l67_67273

-- Define the problem statement as a theorem to prove
theorem angle_in_quadrant (α : ℝ) (k : ℤ) 
  (hα : 2 * (k:ℝ) * Real.pi + Real.pi < α ∧ α < 2 * (k:ℝ) * Real.pi + 3 * Real.pi / 2) :
  (k:ℝ) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k:ℝ) * Real.pi + 3 * Real.pi / 4 := 
sorry

end angle_in_quadrant_l67_67273


namespace inequality_example_l67_67901

variable (a b : ℝ)

theorem inequality_example (h1 : a > 1/2) (h2 : b > 1/2) : a + 2 * b - 5 * a * b < 1/4 :=
by
  sorry

end inequality_example_l67_67901


namespace jacket_purchase_price_l67_67187

theorem jacket_purchase_price (S D P : ℝ) 
  (h1 : S = P + 0.30 * S)
  (h2 : D = 0.80 * S)
  (h3 : 6.000000000000007 = D - P) :
  P = 42 :=
by
  sorry

end jacket_purchase_price_l67_67187


namespace most_cost_effective_80_oranges_l67_67173

noncomputable def cost_of_oranges (p1 p2 p3 : ℕ) (q1 q2 q3 : ℕ) : ℕ :=
  let cost_per_orange_p1 := p1 / q1
  let cost_per_orange_p2 := p2 / q2
  let cost_per_orange_p3 := p3 / q3
  if cost_per_orange_p3 ≤ cost_per_orange_p2 ∧ cost_per_orange_p3 ≤ cost_per_orange_p1 then
    (80 / q3) * p3
  else if cost_per_orange_p2 ≤ cost_per_orange_p1 then
    (80 / q2) * p2
  else
    (80 / q1) * p1

theorem most_cost_effective_80_oranges :
  cost_of_oranges 35 45 95 6 9 20 = 380 :=
by sorry

end most_cost_effective_80_oranges_l67_67173


namespace incorrect_inequality_l67_67586

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by
  sorry

end incorrect_inequality_l67_67586


namespace length_of_brick_proof_l67_67368

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end length_of_brick_proof_l67_67368


namespace percentage_puppies_greater_profit_l67_67683

/-- A dog breeder wants to know what percentage of puppies he can sell for a greater profit.
    Puppies with more than 4 spots sell for more money. The last litter had 10 puppies; 
    6 had 5 spots, 3 had 4 spots, and 1 had 2 spots.
    We need to prove that the percentage of puppies that can be sold for more profit is 60%. -/
theorem percentage_puppies_greater_profit
  (total_puppies : ℕ := 10)
  (puppies_with_5_spots : ℕ := 6)
  (puppies_with_4_spots : ℕ := 3)
  (puppies_with_2_spots : ℕ := 1)
  (puppies_with_more_than_4_spots := puppies_with_5_spots) :
  (puppies_with_more_than_4_spots : ℝ) / (total_puppies : ℝ) * 100 = 60 :=
by
  sorry

end percentage_puppies_greater_profit_l67_67683


namespace pqrs_predicate_l67_67093

noncomputable def P (a b c : ℝ) := a + b - c
noncomputable def Q (a b c : ℝ) := b + c - a
noncomputable def R (a b c : ℝ) := c + a - b

theorem pqrs_predicate (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c) * (Q a b c) * (R a b c) > 0 ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end pqrs_predicate_l67_67093


namespace find_number_l67_67801

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 99) : x = 4400 :=
sorry

end find_number_l67_67801


namespace no_positive_int_squares_l67_67903

theorem no_positive_int_squares (n : ℕ) (h_pos : 0 < n) :
  ¬ (∃ a b c : ℕ, a ^ 2 = 2 * n ^ 2 + 1 ∧ b ^ 2 = 3 * n ^ 2 + 1 ∧ c ^ 2 = 6 * n ^ 2 + 1) := by
  sorry

end no_positive_int_squares_l67_67903


namespace imaginary_part_of_z_l67_67166

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 := 
by
  sorry

end imaginary_part_of_z_l67_67166


namespace max_n_value_l67_67343

-- Define the arithmetic sequence
variable {a : ℕ → ℤ} (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)

-- Given conditions
variable (h1 : a 1 + a 3 + a 5 = 105)
variable (h2 : a 2 + a 4 + a 6 = 99)

-- Goal: Prove the maximum integer value of n is 10
theorem max_n_value (n : ℕ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 3 + a 5 = 105) (h2 : a 2 + a 4 + a 6 = 99) : n ≤ 10 → 
  (∀ m, (0 < m ∧ m ≤ n) → a (2 * m) ≥ 0) → n = 10 := 
sorry

end max_n_value_l67_67343


namespace find_p_q_l67_67367

theorem find_p_q (p q : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 + p * x + q)
  (h_min : ∀ x, x = q → f x = (p + q)^2) : 
  (p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1 / 2) :=
by
  sorry

end find_p_q_l67_67367


namespace solve_equation_l67_67622

theorem solve_equation (x : ℝ) : x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1 / 2 := 
by {
  sorry -- placeholder for the proof
}

end solve_equation_l67_67622


namespace perimeter_of_similar_triangle_l67_67087

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

theorem perimeter_of_similar_triangle (a b c : ℕ) (d e f : ℕ) 
  (h1 : is_isosceles a b c) (h2 : min a b = 15) (h3 : min (min a b) c = 15)
  (h4 : d = 75) (h5 : (d / 15) = e / b) (h6 : f = e) :
  d + e + f = 375 :=
by
  sorry

end perimeter_of_similar_triangle_l67_67087


namespace rest_days_in_1200_days_l67_67932

noncomputable def rest_days_coinciding (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 6) else 0

theorem rest_days_in_1200_days :
  rest_days_coinciding 1200 = 200 :=
by
  sorry

end rest_days_in_1200_days_l67_67932


namespace max_a_for_no_lattice_point_l67_67696

theorem max_a_for_no_lattice_point (a : ℝ) (hm : ∀ m : ℝ, 1 / 2 < m ∧ m < a → ¬ ∃ x y : ℤ, 0 < x ∧ x ≤ 200 ∧ y = m * x + 3) : 
  a = 101 / 201 :=
sorry

end max_a_for_no_lattice_point_l67_67696


namespace inequality_proof_l67_67407

variable (m n : ℝ)

theorem inequality_proof (hm : m < 0) (hn : n > 0) (h_sum : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by
  -- introduction and proof commands would go here, but we use sorry to indicate the proof is omitted
  sorry

end inequality_proof_l67_67407


namespace largest_fraction_l67_67997

theorem largest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 5)
                                          (h2 : f2 = 3 / 6)
                                          (h3 : f3 = 5 / 10)
                                          (h4 : f4 = 7 / 15)
                                          (h5 : f5 = 8 / 20) : 
  (f2 = 1 / 2 ∨ f3 = 1 / 2) ∧ (f2 ≥ f1 ∧ f2 ≥ f4 ∧ f2 ≥ f5) ∧ (f3 ≥ f1 ∧ f3 ≥ f4 ∧ f3 ≥ f5) := 
by
  sorry

end largest_fraction_l67_67997


namespace operation_on_b_l67_67826

theorem operation_on_b (t b0 b1 : ℝ) (h : t * b1^4 = 16 * t * b0^4) : b1 = 2 * b0 :=
by
  sorry

end operation_on_b_l67_67826


namespace probability_neither_red_nor_purple_l67_67170

theorem probability_neither_red_nor_purple 
    (total_balls : ℕ)
    (white_balls : ℕ) 
    (green_balls : ℕ) 
    (yellow_balls : ℕ) 
    (red_balls : ℕ) 
    (purple_balls : ℕ) 
    (h_total : total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls)
    (h_counts : white_balls = 50 ∧ green_balls = 30 ∧ yellow_balls = 8 ∧ red_balls = 9 ∧ purple_balls = 3):
    (88 : ℚ) / 100 = 0.88 :=
by
  sorry

end probability_neither_red_nor_purple_l67_67170


namespace monotonic_sufficient_not_necessary_maximum_l67_67866

noncomputable def f (x : ℝ) : ℝ := sorry  -- Placeholder for the function f
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop := (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)
def has_max_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∃ M, ∀ x, a ≤ x → x ≤ b → f x ≤ M

theorem monotonic_sufficient_not_necessary_maximum : 
  ∀ f : ℝ → ℝ,
  ∀ a b : ℝ,
  a ≤ b →
  monotonic_on f a b → 
  has_max_on f a b :=
sorry  -- Proof is omitted

end monotonic_sufficient_not_necessary_maximum_l67_67866


namespace not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l67_67382

theorem not_right_triangle_sqrt_3_sqrt_4_sqrt_5 :
  ¬ (Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2 :=
by
  -- Start constructing the proof here
  sorry

end not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l67_67382


namespace general_term_of_sequence_l67_67263

theorem general_term_of_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_pos_a : ∀ n, 0 < a n)
  (h_pos_b : ∀ n, 0 < b n)
  (h_arith : ∀ n, 2 * b n = a n + a (n + 1))
  (h_geom : ∀ n, (a (n + 1))^2 = b n * b (n + 1))
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3)
  : ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_of_sequence_l67_67263


namespace choir_members_max_l67_67470

theorem choir_members_max (x r m : ℕ) 
  (h1 : r * x + 3 = m)
  (h2 : (r - 3) * (x + 2) = m) 
  (h3 : m < 150) : 
  m = 759 :=
sorry

end choir_members_max_l67_67470


namespace point_A_outside_circle_l67_67605

noncomputable def circle_radius := 6
noncomputable def distance_OA := 8

theorem point_A_outside_circle : distance_OA > circle_radius :=
by
  -- Solution will go here
  sorry

end point_A_outside_circle_l67_67605


namespace power_function_convex_upwards_l67_67338

noncomputable def f (x : ℝ) : ℝ :=
  x ^ (4 / 5)

theorem power_function_convex_upwards (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 :=
sorry

end power_function_convex_upwards_l67_67338


namespace negation_of_universal_proposition_l67_67414

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5 * x = 4) ↔ (∃ x : ℝ, x^2 + 5 * x ≠ 4) :=
by
  sorry

end negation_of_universal_proposition_l67_67414


namespace max_extra_packages_l67_67181

/-- Max's delivery performance --/
def max_daily_packages : Nat := 35

/-- (1) Max delivered the maximum number of packages on two days --/
def max_2_days : Nat := 2 * max_daily_packages

/-- (2) On two other days, Max unloaded a total of 50 packages --/
def two_days_50 : Nat := 50

/-- (3) On one day, Max delivered one-seventh of the maximum possible daily performance --/
def one_seventh_day : Nat := max_daily_packages / 7

/-- (4) On the last two days, the sum of packages was four-fifths of the maximum daily performance --/
def last_2_days : Nat := 2 * (4 * max_daily_packages / 5)

/-- (5) Total packages delivered in the week --/
def total_delivered : Nat := max_2_days + two_days_50 + one_seventh_day + last_2_days

/-- (6) Total possible packages in a week if worked at maximum performance --/
def total_possible : Nat := 7 * max_daily_packages

/-- (7) Difference between total possible and total delivered packages --/
def difference : Nat := total_possible - total_delivered

/-- Proof problem: Prove the difference is 64 --/
theorem max_extra_packages : difference = 64 := by
  sorry

end max_extra_packages_l67_67181


namespace positive_n_for_modulus_eq_l67_67378

theorem positive_n_for_modulus_eq (n : ℕ) (h_pos : 0 < n) (h_eq : Complex.abs (5 + (n : ℂ) * Complex.I) = 5 * Real.sqrt 26) : n = 25 :=
by
  sorry

end positive_n_for_modulus_eq_l67_67378


namespace jason_current_money_l67_67348

/-- Definition of initial amounts and earnings -/
def fred_initial : ℕ := 49
def jason_initial : ℕ := 3
def fred_current : ℕ := 112
def jason_earned : ℕ := 60

/-- The main theorem -/
theorem jason_current_money : jason_initial + jason_earned = 63 := 
by
  -- proof omitted for this example
  sorry

end jason_current_money_l67_67348


namespace find_num_oranges_l67_67172

def num_oranges (O : ℝ) (x : ℕ) : Prop :=
  6 * 0.21 + O * (x : ℝ) = 1.77 ∧ 2 * 0.21 + 5 * O = 1.27
  ∧ 0.21 = 0.21

theorem find_num_oranges (O : ℝ) (x : ℕ) (h : num_oranges O x) : x = 3 :=
  sorry

end find_num_oranges_l67_67172


namespace people_going_to_zoo_l67_67032

theorem people_going_to_zoo (buses people_per_bus total_people : ℕ) 
  (h1 : buses = 3) 
  (h2 : people_per_bus = 73) 
  (h3 : total_people = buses * people_per_bus) : 
  total_people = 219 := by
  rw [h1, h2] at h3
  exact h3

end people_going_to_zoo_l67_67032


namespace degree_to_radian_60_eq_pi_div_3_l67_67522

theorem degree_to_radian_60_eq_pi_div_3 (pi : ℝ) (deg : ℝ) 
  (h : 180 * deg = pi) : 60 * deg = pi / 3 := 
by
  sorry

end degree_to_radian_60_eq_pi_div_3_l67_67522


namespace total_marbles_l67_67160

def mary_marbles := 9
def joan_marbles := 3
def john_marbles := 7

theorem total_marbles :
  mary_marbles + joan_marbles + john_marbles = 19 :=
by
  sorry

end total_marbles_l67_67160


namespace value_of_expression_l67_67999

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end value_of_expression_l67_67999


namespace vans_capacity_l67_67936

def students : ℕ := 33
def adults : ℕ := 9
def vans : ℕ := 6

def total_people : ℕ := students + adults
def people_per_van : ℕ := total_people / vans

theorem vans_capacity : people_per_van = 7 := by
  sorry

end vans_capacity_l67_67936


namespace unique_A_value_l67_67190

theorem unique_A_value (A : ℝ) (x1 x2 : ℂ) (hx1_ne : x1 ≠ x2) :
  (x1 * (x1 + 1) = A) ∧ (x2 * (x2 + 1) = A) ∧ (A * x1^4 + 3 * x1^3 + 5 * x1 = x2^4 + 3 * x2^3 + 5 * x2) 
  → A = -7 := by
  sorry

end unique_A_value_l67_67190


namespace equation_of_AB_l67_67941

-- Definitions based on the conditions
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 3

def midpoint_M (p : ℝ × ℝ) : Prop :=
  p = (1, 0)

-- The theorem to be proved
theorem equation_of_AB (x y : ℝ) (M : ℝ × ℝ) :
  circle_C x y ∧ midpoint_M M → x - y = 1 :=
by
  sorry

end equation_of_AB_l67_67941


namespace factorization_correct_l67_67816

noncomputable def factor_polynomial : Polynomial ℝ :=
  Polynomial.X^6 - 64

theorem factorization_correct : 
  factor_polynomial = 
  (Polynomial.X - 2) * 
  (Polynomial.X + 2) * 
  (Polynomial.X^4 + 4 * Polynomial.X^2 + 16) :=
by
  sorry

end factorization_correct_l67_67816


namespace shaded_area_proof_l67_67764

noncomputable def total_shaded_area (side_length: ℝ) (large_square_ratio: ℝ) (small_square_ratio: ℝ): ℝ := 
  let S := side_length / large_square_ratio
  let T := S / small_square_ratio
  let large_square_area := S ^ 2
  let small_square_area := T ^ 2
  large_square_area + 12 * small_square_area

theorem shaded_area_proof
  (h1: ∀ side_length, side_length = 15)
  (h2: ∀ large_square_ratio, large_square_ratio = 5)
  (h3: ∀ small_square_ratio, small_square_ratio = 4)
  : total_shaded_area 15 5 4 = 15.75 :=
by
  sorry

end shaded_area_proof_l67_67764


namespace simplify_and_rationalize_l67_67064

theorem simplify_and_rationalize :
  ( (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 10 / Real.sqrt 15) *
    (Real.sqrt 12 / Real.sqrt 20) = 1 / 3 ) :=
by
  sorry

end simplify_and_rationalize_l67_67064


namespace small_square_perimeter_l67_67038

-- Condition Definitions
def perimeter_difference := 17
def side_length_of_square (x : ℝ) := 2 * x = perimeter_difference

-- Theorem Statement
theorem small_square_perimeter (x : ℝ) (h : side_length_of_square x) : 4 * x = 34 :=
by
  sorry

end small_square_perimeter_l67_67038


namespace altitudes_order_l67_67003

variable {A a b c h_a h_b h_c : ℝ}

-- Conditions
axiom area_eq : A = (1/2) * a * h_a
axiom area_eq_b : A = (1/2) * b * h_b
axiom area_eq_c : A = (1/2) * c * h_c
axiom sides_order : a > b ∧ b > c

-- Conclusion
theorem altitudes_order : h_a < h_b ∧ h_b < h_c :=
by
  sorry

end altitudes_order_l67_67003


namespace probability_ratio_l67_67255

theorem probability_ratio :
  let draws := 4
  let total_slips := 40
  let numbers := 10
  let slips_per_number := 4
  let p := 10 / (Nat.choose total_slips draws)
  let q := (Nat.choose numbers 2) * (Nat.choose slips_per_number 2) * (Nat.choose slips_per_number 2) / (Nat.choose total_slips draws)
  p ≠ 0 →
  (q / p) = 162 :=
by
  sorry

end probability_ratio_l67_67255


namespace orchids_to_roses_ratio_l67_67369

noncomputable def total_centerpieces : ℕ := 6
noncomputable def roses_per_centerpiece : ℕ := 8
noncomputable def lilies_per_centerpiece : ℕ := 6
noncomputable def total_budget : ℕ := 2700
noncomputable def cost_per_flower : ℕ := 15
noncomputable def total_flowers : ℕ := total_budget / cost_per_flower

noncomputable def total_roses : ℕ := total_centerpieces * roses_per_centerpiece
noncomputable def total_lilies : ℕ := total_centerpieces * lilies_per_centerpiece
noncomputable def total_roses_and_lilies : ℕ := total_roses + total_lilies
noncomputable def total_orchids : ℕ := total_flowers - total_roses_and_lilies
noncomputable def orchids_per_centerpiece : ℕ := total_orchids / total_centerpieces

theorem orchids_to_roses_ratio : orchids_per_centerpiece / roses_per_centerpiece = 2 :=
by
  sorry

end orchids_to_roses_ratio_l67_67369


namespace find_n_solution_l67_67482

theorem find_n_solution (n : ℚ) (h : (2 / (n+2)) + (4 / (n+2)) + (n / (n+2)) = 4) : n = -2 / 3 := 
by 
  sorry

end find_n_solution_l67_67482


namespace total_length_of_board_l67_67532

theorem total_length_of_board (x y : ℝ) (h1 : y = 2 * x) (h2 : y = 46) : x + y = 69 :=
by
  sorry

end total_length_of_board_l67_67532


namespace remainder_76_pow_77_mod_7_l67_67455

theorem remainder_76_pow_77_mod_7 : (76 ^ 77) % 7 = 6 := 
by 
  sorry 

end remainder_76_pow_77_mod_7_l67_67455


namespace m_add_n_equals_19_l67_67795

theorem m_add_n_equals_19 (n m : ℕ) (A_n_m : ℕ) (C_n_m : ℕ) (h1 : A_n_m = 272) (h2 : C_n_m = 136) :
  m + n = 19 :=
by
  sorry

end m_add_n_equals_19_l67_67795


namespace area_of_parallelogram_l67_67097

variable (b : ℕ)
variable (h : ℕ)
variable (A : ℕ)

-- Condition: The height is twice the base.
def height_twice_base := h = 2 * b

-- Condition: The base is 9.
def base_is_9 := b = 9

-- Condition: The area of the parallelogram is base times height.
def area_formula := A = b * h

-- Question: Prove that the area of the parallelogram is 162.
theorem area_of_parallelogram 
  (h_twice : height_twice_base h b) 
  (b_val : base_is_9 b) 
  (area_form : area_formula A b h): A = 162 := 
sorry

end area_of_parallelogram_l67_67097


namespace total_bricks_used_l67_67547

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end total_bricks_used_l67_67547


namespace value_of_2a_plus_b_l67_67706

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def is_tangent_perpendicular (a b : ℝ) : Prop :=
  let f' := (fun x => (1 : ℝ) / x - a)
  let slope_perpendicular_line := - (1/3 : ℝ)
  f' 1 * slope_perpendicular_line = -1 

def point_on_function (a b : ℝ) : Prop :=
  f a 1 = b

theorem value_of_2a_plus_b (a b : ℝ) 
  (h_tangent_perpendicular : is_tangent_perpendicular a b)
  (h_point_on_function : point_on_function a b) : 
  2 * a + b = -2 := sorry

end value_of_2a_plus_b_l67_67706


namespace positive_numbers_l67_67806

theorem positive_numbers 
    (a b c : ℝ) 
    (h1 : a + b + c > 0) 
    (h2 : ab + bc + ca > 0) 
    (h3 : abc > 0) 
    : a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end positive_numbers_l67_67806


namespace relationship_of_y_values_l67_67296

def parabola_y (x : ℝ) (c : ℝ) : ℝ :=
  2 * (x + 1)^2 + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  y1 = parabola_y (-2) c →
  y2 = parabola_y 1 c →
  y3 = parabola_y 2 c →
  y3 > y2 ∧ y2 > y1 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_values_l67_67296


namespace diff_of_two_numbers_l67_67865

theorem diff_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end diff_of_two_numbers_l67_67865


namespace children_play_time_equal_l67_67803

-- Definitions based on the conditions in the problem
def totalChildren := 7
def totalPlayingTime := 140
def playersAtATime := 2

-- The statement to be proved
theorem children_play_time_equal :
  (playersAtATime * totalPlayingTime) / totalChildren = 40 := by
sorry

end children_play_time_equal_l67_67803


namespace find_triangle_height_l67_67697

-- Given conditions
def triangle_area : ℝ := 960
def base : ℝ := 48

-- The problem is to find the height such that 960 = (1/2) * 48 * height
theorem find_triangle_height (height : ℝ) 
  (h_area : triangle_area = (1/2) * base * height) : height = 40 := by
  sorry

end find_triangle_height_l67_67697


namespace reach_one_from_any_non_zero_l67_67705

-- Define the game rules as functions
def remove_units_digit (n : ℕ) : ℕ :=
  n / 10

def multiply_by_two (n : ℕ) : ℕ :=
  n * 2

-- Lemma: Prove that starting from 45, we can reach 1 using the game rules.
lemma reach_one_from_45 : ∃ f : ℕ → ℕ, f 45 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Lemma: Prove that starting from 345, we can reach 1 using the game rules.
lemma reach_one_from_345 : ∃ f : ℕ → ℕ, f 345 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Theorem: Prove that any non-zero natural number can be reduced to 1 using the game rules.
theorem reach_one_from_any_non_zero (n : ℕ) (h : n ≠ 0) : ∃ f : ℕ → ℕ, f n = 1 :=
by {
  sorry
}

end reach_one_from_any_non_zero_l67_67705


namespace possible_values_for_a_l67_67761

def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + 4 = 0}

theorem possible_values_for_a (a : ℝ) : (B a).Nonempty ∧ B a ⊆ A ↔ a = 4 :=
sorry

end possible_values_for_a_l67_67761


namespace toll_for_18_wheel_truck_l67_67912

-- Define the number of wheels and axles conditions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def number_of_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the toll calculation formula
def toll (x : ℕ) : ℝ := 1.50 + 1.50 * (x - 2)

-- Lean theorem statement asserting that the toll for the given truck is 6 dollars
theorem toll_for_18_wheel_truck : toll number_of_axles = 6 := by
  -- Skipping the actual proof using sorry
  sorry

end toll_for_18_wheel_truck_l67_67912


namespace multiplicative_magic_square_h_sum_l67_67717

theorem multiplicative_magic_square_h_sum :
  ∃ (h_vals : List ℕ), 
  (∀ h ∈ h_vals, ∃ (e : ℕ), e > 0 ∧ 25 * e = h ∧ 
    ∃ (b c d f g : ℕ), 
    75 * b * c = d * e * f ∧ 
    d * e * f = g * h * 3 ∧ 
    g * h * 3 = c * f * 3 ∧ 
    c * f * 3 = 75 * e * g
  ) ∧ h_vals.sum = 150 :=
by { sorry }

end multiplicative_magic_square_h_sum_l67_67717
