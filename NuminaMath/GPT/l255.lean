import Mathlib

namespace NUMINAMATH_GPT_fractional_part_of_water_after_replacements_l255_25512

theorem fractional_part_of_water_after_replacements :
  let total_quarts := 25
  let removed_quarts := 5
  (1 - removed_quarts / (total_quarts : ℚ))^3 = 64 / 125 :=
by
  sorry

end NUMINAMATH_GPT_fractional_part_of_water_after_replacements_l255_25512


namespace NUMINAMATH_GPT_op_two_four_l255_25578

def op (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem op_two_four : op 2 4 = 18 := by
  sorry

end NUMINAMATH_GPT_op_two_four_l255_25578


namespace NUMINAMATH_GPT_smallest_positive_integer_k_l255_25540

theorem smallest_positive_integer_k:
  ∀ T : ℕ, ∀ n : ℕ, (T = n * (n + 1) / 2) → ∃ m : ℕ, 81 * T + 10 = m * (m + 1) / 2 :=
by
  intro T n h
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_k_l255_25540


namespace NUMINAMATH_GPT_deposit_paid_l255_25566

variable (P : ℝ) (Deposit Remaining : ℝ)

-- Define the conditions
def deposit_condition : Prop := Deposit = 0.10 * P
def remaining_condition : Prop := Remaining = 0.90 * P
def remaining_amount_given : Prop := Remaining = 1170

-- The goal to prove: the deposit paid is $130
theorem deposit_paid (h₁ : deposit_condition P Deposit) (h₂ : remaining_condition P Remaining) (h₃ : remaining_amount_given Remaining) : 
  Deposit = 130 :=
  sorry

end NUMINAMATH_GPT_deposit_paid_l255_25566


namespace NUMINAMATH_GPT_average_rainfall_per_hour_eq_l255_25555

-- Define the conditions
def february_days_non_leap_year : ℕ := 28
def hours_per_day : ℕ := 24
def total_rainfall_in_inches : ℕ := 280
def total_hours_in_february : ℕ := february_days_non_leap_year * hours_per_day

-- Define the goal
theorem average_rainfall_per_hour_eq :
  total_rainfall_in_inches / total_hours_in_february = 5 / 12 :=
sorry

end NUMINAMATH_GPT_average_rainfall_per_hour_eq_l255_25555


namespace NUMINAMATH_GPT_false_propositions_l255_25527

open Classical

theorem false_propositions :
  ¬ (∀ x : ℝ, x^2 + 3 < 0) ∧ ¬ (∀ x : ℕ, x^2 > 1) ∧ (∃ x : ℤ, x^5 < 1) ∧ ¬ (∃ x : ℚ, x^2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_false_propositions_l255_25527


namespace NUMINAMATH_GPT_stock_comparison_l255_25508

-- Quantities of the first year depreciation or growth rates
def initial_investment : ℝ := 200.0
def dd_first_year_growth : ℝ := 1.10
def ee_first_year_decline : ℝ := 0.85
def ff_first_year_growth : ℝ := 1.05

-- Quantities of the second year depreciation or growth rates
def dd_second_year_growth : ℝ := 1.05
def ee_second_year_growth : ℝ := 1.15
def ff_second_year_decline : ℝ := 0.90

-- Mathematical expression to determine final values after first year
def dd_after_first_year := initial_investment * dd_first_year_growth
def ee_after_first_year := initial_investment * ee_first_year_decline
def ff_after_first_year := initial_investment * ff_first_year_growth

-- Mathematical expression to determine final values after second year
def dd_final := dd_after_first_year * dd_second_year_growth
def ee_final := ee_after_first_year * ee_second_year_growth
def ff_final := ff_after_first_year * ff_second_year_decline

-- Theorem representing the final comparison
theorem stock_comparison : ff_final < ee_final ∧ ee_final < dd_final :=
by {
  -- Here we would provide the proof, but as per instruction we'll place sorry
  sorry
}

end NUMINAMATH_GPT_stock_comparison_l255_25508


namespace NUMINAMATH_GPT_strange_number_l255_25565

theorem strange_number (x : ℤ) (h : (x - 7) * 7 = (x - 11) * 11) : x = 18 :=
sorry

end NUMINAMATH_GPT_strange_number_l255_25565


namespace NUMINAMATH_GPT_find_range_m_l255_25587

def p (m : ℝ) : Prop := m > 2 ∨ m < -2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_m (h₁ : ¬ p m) (h₂ : q m) : (1 : ℝ) < m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_GPT_find_range_m_l255_25587


namespace NUMINAMATH_GPT_odd_and_periodic_40_l255_25510

noncomputable def f : ℝ → ℝ := sorry

theorem odd_and_periodic_40
  (h₁ : ∀ x : ℝ, f (10 + x) = f (10 - x))
  (h₂ : ∀ x : ℝ, f (20 - x) = -f (20 + x)) :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (x + 40) = f (x)) :=
by
  sorry

end NUMINAMATH_GPT_odd_and_periodic_40_l255_25510


namespace NUMINAMATH_GPT_probability_not_orange_not_white_l255_25539

theorem probability_not_orange_not_white (num_orange num_black num_white : ℕ)
    (h_orange : num_orange = 8) (h_black : num_black = 7) (h_white : num_white = 6) :
    (num_black : ℚ) / (num_orange + num_black + num_white : ℚ) = 1 / 3 :=
  by
    -- Solution will be here.
    sorry

end NUMINAMATH_GPT_probability_not_orange_not_white_l255_25539


namespace NUMINAMATH_GPT_profit_without_discount_l255_25564

theorem profit_without_discount (CP SP MP : ℝ) (discountRate profitRate : ℝ)
  (h1 : CP = 100)
  (h2 : discountRate = 0.05)
  (h3 : profitRate = 0.235)
  (h4 : SP = CP * (1 + profitRate))
  (h5 : MP = SP / (1 - discountRate)) :
  (((MP - CP) / CP) * 100) = 30 := 
sorry

end NUMINAMATH_GPT_profit_without_discount_l255_25564


namespace NUMINAMATH_GPT_evaluate_expression_l255_25504

theorem evaluate_expression (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : 4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l255_25504


namespace NUMINAMATH_GPT_sum_converges_to_one_l255_25574

noncomputable def series_sum (n: ℕ) : ℝ :=
  if n ≥ 2 then (6 * n^3 - 2 * n^2 - 2 * n + 1) / (n^6 - 2 * n^5 + 2 * n^4 - n^3 + n^2 - 2 * n)
  else 0

theorem sum_converges_to_one : 
  (∑' n, series_sum n) = 1 := by
  sorry

end NUMINAMATH_GPT_sum_converges_to_one_l255_25574


namespace NUMINAMATH_GPT_rachel_total_apples_l255_25597

noncomputable def totalRemainingApples (X : ℕ) : ℕ :=
  let remainingFirstFour := 10 + 40 + 15 + 22
  let remainingOtherTrees := 48 * X
  remainingFirstFour + remainingOtherTrees

theorem rachel_total_apples (X : ℕ) :
  totalRemainingApples X = 87 + 48 * X :=
by
  sorry

end NUMINAMATH_GPT_rachel_total_apples_l255_25597


namespace NUMINAMATH_GPT_cycle_cost_price_l255_25561

theorem cycle_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1360) 
  (h2 : loss_percentage = 0.15) :
  SP = (1 - loss_percentage) * C → C = 1600 :=
by
  sorry

end NUMINAMATH_GPT_cycle_cost_price_l255_25561


namespace NUMINAMATH_GPT_samuel_apples_left_l255_25537

def bonnieApples : ℕ := 8
def extraApples : ℕ := 20
def samuelTotalApples : ℕ := bonnieApples + extraApples
def samuelAte : ℕ := samuelTotalApples / 2
def samuelRemainingAfterEating : ℕ := samuelTotalApples - samuelAte
def samuelUsedForPie : ℕ := samuelRemainingAfterEating / 7
def samuelFinalRemaining : ℕ := samuelRemainingAfterEating - samuelUsedForPie

theorem samuel_apples_left :
  samuelFinalRemaining = 12 := by
  sorry

end NUMINAMATH_GPT_samuel_apples_left_l255_25537


namespace NUMINAMATH_GPT_math_problem_l255_25589

theorem math_problem :
  (-1)^2024 + (-10) / (1/2) * 2 + (2 - (-3)^3) = -10 := by
  sorry

end NUMINAMATH_GPT_math_problem_l255_25589


namespace NUMINAMATH_GPT_temperature_at_tian_du_peak_height_of_mountain_peak_l255_25533

-- Problem 1: Temperature at the top of Tian Du Peak
theorem temperature_at_tian_du_peak
  (height : ℝ) (drop_rate : ℝ) (initial_temp : ℝ)
  (H : height = 1800) (D : drop_rate = 0.6) (I : initial_temp = 18) :
  (initial_temp - (height / 100 * drop_rate)) = 7.2 :=
by
  sorry

-- Problem 2: Height of the mountain peak
theorem height_of_mountain_peak
  (drop_rate : ℝ) (foot_temp top_temp : ℝ)
  (D : drop_rate = 0.6) (F : foot_temp = 10) (T : top_temp = -8) :
  (foot_temp - top_temp) / drop_rate * 100 = 3000 :=
by
  sorry

end NUMINAMATH_GPT_temperature_at_tian_du_peak_height_of_mountain_peak_l255_25533


namespace NUMINAMATH_GPT_problem_solved_prob_l255_25570

theorem problem_solved_prob (pA pB : ℝ) (HA : pA = 1 / 3) (HB : pB = 4 / 5) :
  ((1 - (1 - pA) * (1 - pB)) = 13 / 15) :=
by
  sorry

end NUMINAMATH_GPT_problem_solved_prob_l255_25570


namespace NUMINAMATH_GPT_largest_possible_number_of_markers_l255_25549

theorem largest_possible_number_of_markers (n_m n_c : ℕ) 
  (h_m : n_m = 72) (h_c : n_c = 48) : Nat.gcd n_m n_c = 24 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_number_of_markers_l255_25549


namespace NUMINAMATH_GPT_students_play_both_football_and_cricket_l255_25534

theorem students_play_both_football_and_cricket :
  ∀ (total F C N both : ℕ),
  total = 460 →
  F = 325 →
  C = 175 →
  N = 50 →
  total - N = F + C - both →
  both = 90 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_play_both_football_and_cricket_l255_25534


namespace NUMINAMATH_GPT_min_cars_needed_l255_25569

theorem min_cars_needed (h1 : ∀ d ∈ Finset.range 7, ∃ s : Finset ℕ, s.card = 2 ∧ (∃ n : ℕ, 7 * (n - 10) ≥ 2 * n)) : 
  ∃ n, n ≥ 14 :=
by
  sorry

end NUMINAMATH_GPT_min_cars_needed_l255_25569


namespace NUMINAMATH_GPT_lcm_inequality_l255_25547

theorem lcm_inequality
  (a b c d e : ℤ)
  (h1 : 1 ≤ a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (1 : ℚ) / Int.lcm a b + (1 : ℚ) / Int.lcm b c + 
  (1 : ℚ) / Int.lcm c d + (1 : ℚ) / Int.lcm d e ≤ (15 : ℚ) / 16 := by
  sorry

end NUMINAMATH_GPT_lcm_inequality_l255_25547


namespace NUMINAMATH_GPT_find_y_l255_25524

theorem find_y (x y : ℤ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end NUMINAMATH_GPT_find_y_l255_25524


namespace NUMINAMATH_GPT_phone_number_C_value_l255_25543

/-- 
In a phone number formatted as ABC-DEF-GHIJ, each letter symbolizes a distinct digit.
Digits in each section ABC, DEF, and GHIJ are in ascending order i.e., A < B < C, D < E < F, and G < H < I < J.
Moreover, D, E, F are consecutive odd digits, and G, H, I, J are consecutive even digits.
Also, A + B + C = 15. Prove that the value of C is 9. 
-/
theorem phone_number_C_value :
  ∃ (A B C D E F G H I J : ℕ), 
  A < B ∧ B < C ∧ D < E ∧ E < F ∧ G < H ∧ H < I ∧ I < J ∧
  (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1) ∧
  (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0) ∧
  (E = D + 2) ∧ (F = D + 4) ∧ (H = G + 2) ∧ (I = G + 4) ∧ (J = G + 6) ∧
  A + B + C = 15 ∧
  C = 9 := by 
  sorry

end NUMINAMATH_GPT_phone_number_C_value_l255_25543


namespace NUMINAMATH_GPT_power_ordering_l255_25522

theorem power_ordering (a b c : ℝ) : 
  (a = 2^30) → (b = 6^10) → (c = 3^20) → (a < b) ∧ (b < c) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  have h1 : 6^10 = (3 * 2)^10 := by sorry
  have h2 : 3^20 = (3^10)^2 := by sorry
  have h3 : 2^30 = (2^10)^3 := by sorry
  sorry

end NUMINAMATH_GPT_power_ordering_l255_25522


namespace NUMINAMATH_GPT_work_rate_D_time_A_B_D_time_D_l255_25528

def workRate (person : String) : ℚ :=
  if person = "A" then 1/12 else
  if person = "B" then 1/6 else
  if person = "A_D" then 1/4 else
  0

theorem work_rate_D : workRate "A_D" - workRate "A" = 1/6 := by
  sorry

theorem time_A_B_D : (1 / (workRate "A" + workRate "B" + (workRate "A_D" - workRate "A"))) = 2.4 := by
  sorry
  
theorem time_D : (1 / (workRate "A_D" - workRate "A")) = 6 := by
  sorry

end NUMINAMATH_GPT_work_rate_D_time_A_B_D_time_D_l255_25528


namespace NUMINAMATH_GPT_tangent_line_at_pi_l255_25572

noncomputable def tangent_equation (x : ℝ) : ℝ := x * Real.sin x

theorem tangent_line_at_pi :
  let f := tangent_equation
  let f' := fun x => Real.sin x + x * Real.cos x
  let x : ℝ := Real.pi
  let y : ℝ := f x
  let slope : ℝ := f' x
  y + slope * x - Real.pi^2 = 0 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_tangent_line_at_pi_l255_25572


namespace NUMINAMATH_GPT_less_than_half_l255_25529

theorem less_than_half (a b c : ℝ) (h₁ : a = 43.2) (h₂ : b = 0.5) (h₃ : c = 42.7) : a - b = c := by
  sorry

end NUMINAMATH_GPT_less_than_half_l255_25529


namespace NUMINAMATH_GPT_find_eighth_number_l255_25519

-- Define the given problem with the conditions
noncomputable def sum_of_sixteen_numbers := 16 * 55
noncomputable def sum_of_first_eight_numbers := 8 * 60
noncomputable def sum_of_last_eight_numbers := 8 * 45
noncomputable def sum_of_last_nine_numbers := 9 * 50
noncomputable def sum_of_first_ten_numbers := 10 * 62

-- Define what we want to prove
theorem find_eighth_number :
  (exists (x : ℕ), x = 90) →
  sum_of_first_eight_numbers = 480 →
  sum_of_last_eight_numbers = 360 →
  sum_of_last_nine_numbers = 450 →
  sum_of_first_ten_numbers = 620 →
  sum_of_sixteen_numbers = 880 →
  x = 90 :=
by sorry

end NUMINAMATH_GPT_find_eighth_number_l255_25519


namespace NUMINAMATH_GPT_fish_left_in_tank_l255_25591

theorem fish_left_in_tank (initial_fish : ℕ) (fish_taken_out : ℕ) (fish_left : ℕ) 
  (h1 : initial_fish = 19) (h2 : fish_taken_out = 16) : fish_left = initial_fish - fish_taken_out :=
by
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_fish_left_in_tank_l255_25591


namespace NUMINAMATH_GPT_zucchini_pounds_l255_25545

theorem zucchini_pounds :
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let quarts := 4
  let cost_per_quart := 10.00
  let total_cost := quarts * cost_per_quart
  let cost_of_eggplants := eggplants_pounds * eggplants_cost_per_pound
  let cost_of_tomatoes := tomatoes_pounds * tomatoes_cost_per_pound
  let cost_of_onions := onions_pounds * onions_cost_per_pound
  let cost_of_basil := basil_pounds * (basil_cost_per_half_pound * 2)
  let other_ingredients_cost := cost_of_eggplants + cost_of_tomatoes + cost_of_onions + cost_of_basil
  let cost_of_zucchini := total_cost - other_ingredients_cost
  let zucchini_cost_per_pound := 2.00
  let pounds_of_zucchini := cost_of_zucchini / zucchini_cost_per_pound
  pounds_of_zucchini = 4 :=
by
  sorry

end NUMINAMATH_GPT_zucchini_pounds_l255_25545


namespace NUMINAMATH_GPT_red_given_red_l255_25541

def p_i (i : ℕ) : ℚ := sorry
axiom lights_probs_eq : p_i 1 + p_i 2 = 2 / 3
axiom lights_probs_eq2 : p_i 1 + p_i 3 = 2 / 3
axiom green_given_green : p_i 1 / (p_i 1 + p_i 2) = 3 / 4
axiom total_prob : p_i 1 + p_i 2 + p_i 3 + p_i 4 = 1

theorem red_given_red : (p_i 4 / (p_i 3 + p_i 4)) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_red_given_red_l255_25541


namespace NUMINAMATH_GPT_solve_for_ab_l255_25550

theorem solve_for_ab (a b : ℤ) 
  (h1 : a + 3 * b = 27) 
  (h2 : 5 * a + 4 * b = 47) : 
  a + b = 11 :=
sorry

end NUMINAMATH_GPT_solve_for_ab_l255_25550


namespace NUMINAMATH_GPT_paula_go_kart_rides_l255_25592

theorem paula_go_kart_rides
  (g : ℕ)
  (ticket_cost_go_karts : ℕ := 4 * g)
  (ticket_cost_bumper_cars : ℕ := 20)
  (total_tickets : ℕ := 24) :
  ticket_cost_go_karts + ticket_cost_bumper_cars = total_tickets → g = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_paula_go_kart_rides_l255_25592


namespace NUMINAMATH_GPT_people_going_to_zoo_l255_25582

theorem people_going_to_zoo (buses people_per_bus total_people : ℕ) 
  (h1 : buses = 3) 
  (h2 : people_per_bus = 73) 
  (h3 : total_people = buses * people_per_bus) : 
  total_people = 219 := by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_people_going_to_zoo_l255_25582


namespace NUMINAMATH_GPT_least_even_integer_square_l255_25516

theorem least_even_integer_square (E : ℕ) (h_even : E % 2 = 0) (h_square : ∃ (I : ℕ), 300 * E = I^2) : E = 6 ∧ ∃ (I : ℕ), I = 30 ∧ 300 * E = I^2 :=
sorry

end NUMINAMATH_GPT_least_even_integer_square_l255_25516


namespace NUMINAMATH_GPT_area_of_region_W_l255_25535

structure Rhombus (P Q R T : Type) :=
  (side_length : ℝ)
  (angle_Q : ℝ)

def Region_W
  (P Q R T : Type)
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) : ℝ :=
6.25

theorem area_of_region_W
  {P Q R T : Type}
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) :
  Region_W P Q R T r h_side h_angle = 6.25 :=
sorry

end NUMINAMATH_GPT_area_of_region_W_l255_25535


namespace NUMINAMATH_GPT_find_a_minus_b_l255_25513

-- Definitions based on conditions
def eq1 (a b : Int) : Prop := 2 * b + a = 5
def eq2 (a b : Int) : Prop := a * b = -12

-- Statement of the problem
theorem find_a_minus_b (a b : Int) (h1 : eq1 a b) (h2 : eq2 a b) : a - b = -7 := 
sorry

end NUMINAMATH_GPT_find_a_minus_b_l255_25513


namespace NUMINAMATH_GPT_triangle_perimeter_l255_25525

/-- In a triangle ABC, where sides a, b, c are opposite to angles A, B, C respectively.
Given the area of the triangle = 15 * sqrt 3 / 4, 
angle A = 60 degrees and 5 * sin B = 3 * sin C,
prove that the perimeter of triangle ABC is 8 + sqrt 19. -/
theorem triangle_perimeter
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A = 60)
  (h_area : (1 / 2) * b * c * (Real.sin (A / (180 / Real.pi))) = 15 * Real.sqrt 3 / 4)
  (h_sin : 5 * Real.sin B = 3 * Real.sin C) :
  a + b + c = 8 + Real.sqrt 19 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l255_25525


namespace NUMINAMATH_GPT_altitudes_order_l255_25595

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

end NUMINAMATH_GPT_altitudes_order_l255_25595


namespace NUMINAMATH_GPT_slices_per_friend_l255_25559

theorem slices_per_friend (total_slices friends : ℕ) (h1 : total_slices = 16) (h2 : friends = 4) : (total_slices / friends) = 4 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_friend_l255_25559


namespace NUMINAMATH_GPT_pastries_eaten_l255_25548

theorem pastries_eaten (total_p: ℕ)
  (hare_fraction: ℚ)
  (dormouse_fraction: ℚ)
  (hare_eaten: ℕ)
  (remaining_after_hare: ℕ)
  (dormouse_eaten: ℕ)
  (final_remaining: ℕ) 
  (hatter_with_left: ℕ) :
  (final_remaining = hatter_with_left) -> hare_fraction = 5 / 16 -> dormouse_fraction = 7 / 11 -> hatter_with_left = 8 -> total_p = 32 -> 
  (total_p = hare_eaten + remaining_after_hare) -> (remaining_after_hare - dormouse_eaten = hatter_with_left) -> (hare_eaten = 10) ∧ (dormouse_eaten = 14) := 
by {
  sorry
}

end NUMINAMATH_GPT_pastries_eaten_l255_25548


namespace NUMINAMATH_GPT_bobbie_letters_to_remove_l255_25507

-- Definitions of the conditions
def samanthaLastNameLength := 7
def bobbieLastNameLength := samanthaLastNameLength + 3
def jamieLastNameLength := 4
def targetBobbieLastNameLength := 2 * jamieLastNameLength

-- Question: How many letters does Bobbie need to take off to have a last name twice the length of Jamie's?
theorem bobbie_letters_to_remove : 
  bobbieLastNameLength - targetBobbieLastNameLength = 2 := by 
  sorry

end NUMINAMATH_GPT_bobbie_letters_to_remove_l255_25507


namespace NUMINAMATH_GPT_balloons_division_correct_l255_25515

def number_of_balloons_per_school (yellow blue more_black num_schools: ℕ) : ℕ :=
  let black := yellow + more_black
  let total := yellow + blue + black
  total / num_schools

theorem balloons_division_correct :
  number_of_balloons_per_school 3414 5238 1762 15 = 921 := 
by
  sorry

end NUMINAMATH_GPT_balloons_division_correct_l255_25515


namespace NUMINAMATH_GPT_smallest_solution_x_abs_x_eq_3x_plus_2_l255_25505

theorem smallest_solution_x_abs_x_eq_3x_plus_2 : ∃ x : ℝ, (x * abs x = 3 * x + 2) ∧ (∀ y : ℝ, (y * abs y = 3 * y + 2) → x ≤ y) ∧ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_x_abs_x_eq_3x_plus_2_l255_25505


namespace NUMINAMATH_GPT_factor_diff_of_squares_l255_25530

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end NUMINAMATH_GPT_factor_diff_of_squares_l255_25530


namespace NUMINAMATH_GPT_dennis_total_cost_l255_25518

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end NUMINAMATH_GPT_dennis_total_cost_l255_25518


namespace NUMINAMATH_GPT_price_reduction_l255_25567

theorem price_reduction (x : ℝ) : 
  188 * (1 - x) ^ 2 = 108 :=
sorry

end NUMINAMATH_GPT_price_reduction_l255_25567


namespace NUMINAMATH_GPT_increase_by_percentage_l255_25511

theorem increase_by_percentage (a b : ℝ) (percentage : ℝ) (final : ℝ) : b = a * percentage → final = a + b → final = 437.5 :=
by
  sorry

end NUMINAMATH_GPT_increase_by_percentage_l255_25511


namespace NUMINAMATH_GPT_regular_tetrahedron_height_eq_4r_l255_25509

noncomputable def equilateral_triangle_inscribed_circle_height (r : ℝ) : ℝ :=
3 * r

noncomputable def regular_tetrahedron_inscribed_sphere_height (r : ℝ) : ℝ :=
4 * r

theorem regular_tetrahedron_height_eq_4r (r : ℝ) :
  regular_tetrahedron_inscribed_sphere_height r = 4 * r :=
by
  unfold regular_tetrahedron_inscribed_sphere_height
  sorry

end NUMINAMATH_GPT_regular_tetrahedron_height_eq_4r_l255_25509


namespace NUMINAMATH_GPT_jill_total_trip_duration_is_101_l255_25581

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

end NUMINAMATH_GPT_jill_total_trip_duration_is_101_l255_25581


namespace NUMINAMATH_GPT_boys_difference_twice_girls_l255_25521

theorem boys_difference_twice_girls :
  ∀ (total_students girls boys : ℕ),
  total_students = 68 →
  girls = 28 →
  boys = total_students - girls →
  2 * girls - boys = 16 :=
by
  intros total_students girls boys h1 h2 h3
  sorry

end NUMINAMATH_GPT_boys_difference_twice_girls_l255_25521


namespace NUMINAMATH_GPT_quadratic_root_expression_value_l255_25552

theorem quadratic_root_expression_value (a : ℝ) 
  (h : a^2 - 2 * a - 3 = 0) : 2 * a^2 - 4 * a + 1 = 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_expression_value_l255_25552


namespace NUMINAMATH_GPT_inverse_function_value_l255_25542

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3

theorem inverse_function_value :
  f 3 = 51 :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_value_l255_25542


namespace NUMINAMATH_GPT_diamond_evaluation_l255_25554

-- Define the diamond operation as a function using the given table
def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4 | (1, 2) => 1 | (1, 3) => 3 | (1, 4) => 2
  | (2, 1) => 1 | (2, 2) => 3 | (2, 3) => 2 | (2, 4) => 4
  | (3, 1) => 3 | (3, 2) => 2 | (3, 3) => 4 | (3, 4) => 1
  | (4, 1) => 2 | (4, 2) => 4 | (4, 3) => 1 | (4, 4) => 3
  | (_, _) => 0  -- default case (should not occur)

-- State the proof problem
theorem diamond_evaluation : diamond (diamond 3 1) (diamond 4 2) = 1 := by
  sorry

end NUMINAMATH_GPT_diamond_evaluation_l255_25554


namespace NUMINAMATH_GPT_area_of_hexagon_correct_l255_25568

variable (α β γ : ℝ) (S : ℝ) (r R : ℝ)
variable (AB BC AC : ℝ)
variable (A' B' C' : ℝ)

noncomputable def area_of_hexagon (AB BC AC : ℝ) (R : ℝ) (S : ℝ) (r : ℝ) : ℝ :=
  2 * (S / (r * r))

theorem area_of_hexagon_correct
  (hAB : AB = 13) (hBC : BC = 14) (hAC : AC = 15)
  (hR : R = 65 / 8) (hS : S = 1344 / 65) :
  area_of_hexagon AB BC AC R S r = 2 * (S / (r * r)) :=
sorry

end NUMINAMATH_GPT_area_of_hexagon_correct_l255_25568


namespace NUMINAMATH_GPT_proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l255_25502

variable (a b : ℝ)

-- Condition
def condition : Prop :=
  (1 / a) - (1 / b) = 1 / (a + b)

-- Proof statement
theorem proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab (h : condition a b) :
  (1 / a^2) - (1 / b^2) = 1 / (a * b) :=
sorry

end NUMINAMATH_GPT_proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l255_25502


namespace NUMINAMATH_GPT_point_D_number_l255_25576

theorem point_D_number (x : ℝ) :
    (5 + 8 - 10 + x = -5 - 8 + 10 - x) ↔ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_point_D_number_l255_25576


namespace NUMINAMATH_GPT_x_and_y_complete_work_in_12_days_l255_25503

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end NUMINAMATH_GPT_x_and_y_complete_work_in_12_days_l255_25503


namespace NUMINAMATH_GPT_ternary_to_decimal_l255_25523

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end NUMINAMATH_GPT_ternary_to_decimal_l255_25523


namespace NUMINAMATH_GPT_travel_time_l255_25573

-- Definitions: 
def speed := 20 -- speed in km/hr
def distance := 160 -- distance in km

-- Proof statement: 
theorem travel_time (s : ℕ) (d : ℕ) (h1 : s = speed) (h2 : d = distance) : 
  d / s = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_travel_time_l255_25573


namespace NUMINAMATH_GPT_total_pages_is_1200_l255_25583

theorem total_pages_is_1200 (A B : ℕ) (h1 : 24 * (A + B) = 60 * A) (h2 : B = A + 10) : (60 * A) = 1200 := by
  sorry

end NUMINAMATH_GPT_total_pages_is_1200_l255_25583


namespace NUMINAMATH_GPT_sally_rum_l255_25517

theorem sally_rum (x : ℕ) (h₁ : 3 * x = x + 12 + 8) : x = 10 := by
  sorry

end NUMINAMATH_GPT_sally_rum_l255_25517


namespace NUMINAMATH_GPT_complement_intersection_l255_25558

def setM : Set ℝ := { x | 2 / x < 1 }
def setN : Set ℝ := { y | ∃ x, y = Real.sqrt (x - 1) }

theorem complement_intersection 
  (R : Set ℝ) : ((R \ setM) ∩ setN = { y | 0 ≤ y ∧ y ≤ 2 }) :=
  sorry

end NUMINAMATH_GPT_complement_intersection_l255_25558


namespace NUMINAMATH_GPT_smallest_AAB_l255_25563

theorem smallest_AAB (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) 
  (h : 10 * A + B = (110 * A + B) / 7) : 110 * A + B = 996 :=
by
  sorry

end NUMINAMATH_GPT_smallest_AAB_l255_25563


namespace NUMINAMATH_GPT_twice_a_plus_one_non_negative_l255_25526

theorem twice_a_plus_one_non_negative (a : ℝ) : 2 * a + 1 ≥ 0 :=
sorry

end NUMINAMATH_GPT_twice_a_plus_one_non_negative_l255_25526


namespace NUMINAMATH_GPT_roots_cubic_reciprocal_sum_l255_25556

theorem roots_cubic_reciprocal_sum (a b c : ℝ) 
(h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 27) (h₃ : a * b * c = 18) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 13 / 24 :=
by
  sorry

end NUMINAMATH_GPT_roots_cubic_reciprocal_sum_l255_25556


namespace NUMINAMATH_GPT_find_number_l255_25546

theorem find_number (x : ℝ) (h : 1345 - x / 20.04 = 1295) : x = 1002 :=
sorry

end NUMINAMATH_GPT_find_number_l255_25546


namespace NUMINAMATH_GPT_ratio_A_B_correct_l255_25579

-- Define the shares of A, B, and C
def A_share := 372
def B_share := 93
def C_share := 62

-- Total amount distributed
def total_share := A_share + B_share + C_share

-- The ratio of A's share to B's share
def ratio_A_to_B := A_share / B_share

theorem ratio_A_B_correct : 
  total_share = 527 ∧ 
  ¬(B_share = (1 / 4) * C_share) ∧ 
  ratio_A_to_B = 4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_A_B_correct_l255_25579


namespace NUMINAMATH_GPT_max_gcd_of_polynomials_l255_25551

def max_gcd (a b : ℤ) : ℤ :=
  let g := Nat.gcd a.natAbs b.natAbs
  Int.ofNat g

theorem max_gcd_of_polynomials :
  ∃ n : ℕ, (n > 0) → max_gcd (14 * ↑n + 5) (9 * ↑n + 2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_of_polynomials_l255_25551


namespace NUMINAMATH_GPT_is_periodic_l255_25520

noncomputable def f : ℝ → ℝ := sorry

axiom domain (x : ℝ) : true
axiom not_eq_neg1_and_not_eq_0 (x : ℝ) : f x ≠ -1 ∧ f x ≠ 0
axiom functional_eq (x y : ℝ) : f (x - y) = - (f x / (1 + f y))

theorem is_periodic : ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end NUMINAMATH_GPT_is_periodic_l255_25520


namespace NUMINAMATH_GPT_handshaking_pairs_l255_25553

-- Definition of the problem: Given 8 people, pair them up uniquely and count the ways modulo 1000
theorem handshaking_pairs (N : ℕ) (H : N=105) : (N % 1000) = 105 :=
by {
  -- The proof is omitted.
  sorry
}

end NUMINAMATH_GPT_handshaking_pairs_l255_25553


namespace NUMINAMATH_GPT_inequality_proof_l255_25500

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)) + (bc / (b + c)) + (ca / (c + a)) ≤ (3 * (ab + bc + ca)) / (2 * (a + b + c)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l255_25500


namespace NUMINAMATH_GPT_train_average_speed_with_stoppages_l255_25598

theorem train_average_speed_with_stoppages (D : ℝ) :
  let speed_without_stoppages := 200
  let stoppage_time_per_hour_in_hours := 12 / 60.0
  let effective_running_time := 1 - stoppage_time_per_hour_in_hours
  let speed_with_stoppages := effective_running_time * speed_without_stoppages
  speed_with_stoppages = 160 := by
  sorry

end NUMINAMATH_GPT_train_average_speed_with_stoppages_l255_25598


namespace NUMINAMATH_GPT_solution_of_ab_l255_25536

theorem solution_of_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (ax^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3)) : 
  a * b = 24 := 
sorry

end NUMINAMATH_GPT_solution_of_ab_l255_25536


namespace NUMINAMATH_GPT_probability_neither_A_nor_B_l255_25596

noncomputable def pA : ℝ := 0.25
noncomputable def pB : ℝ := 0.35
noncomputable def pA_and_B : ℝ := 0.15

theorem probability_neither_A_nor_B :
  1 - (pA + pB - pA_and_B) = 0.55 :=
by
  simp [pA, pB, pA_and_B]
  norm_num
  sorry

end NUMINAMATH_GPT_probability_neither_A_nor_B_l255_25596


namespace NUMINAMATH_GPT_circle_center_coordinates_l255_25538

theorem circle_center_coordinates (b c p q : ℝ) 
    (h_circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * p * x - 2 * q * y + 2 * q - 1 = 0) 
    (h_quad_roots : ∀ x : ℝ, x^2 + b * x + c = 0) 
    (h_condition : b^2 - 4 * c ≥ 0) : 
    (p = -b / 2) ∧ (q = (1 + c) / 2) := 
sorry

end NUMINAMATH_GPT_circle_center_coordinates_l255_25538


namespace NUMINAMATH_GPT_double_neg_five_eq_five_l255_25575

theorem double_neg_five_eq_five : -(-5) = 5 := 
sorry

end NUMINAMATH_GPT_double_neg_five_eq_five_l255_25575


namespace NUMINAMATH_GPT_max_flags_l255_25544

theorem max_flags (n : ℕ) (h1 : ∀ k, n = 9 * k) (h2 : n ≤ 200)
  (h3 : ∃ m, n = 9 * m + k ∧ k ≤ 2 ∧ k + 1 ≠ 0 ∧ k - 2 ≠ 0) : n = 198 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_flags_l255_25544


namespace NUMINAMATH_GPT_data_division_into_groups_l255_25586

-- Conditions
def data_set_size : Nat := 90
def max_value : Nat := 141
def min_value : Nat := 40
def class_width : Nat := 10

-- Proof statement
theorem data_division_into_groups : (max_value - min_value) / class_width + 1 = 11 :=
by
  sorry

end NUMINAMATH_GPT_data_division_into_groups_l255_25586


namespace NUMINAMATH_GPT_rhombus_side_length_l255_25571

variables (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r)

theorem rhombus_side_length (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r) :
  ∃ s : ℝ, s = 2 * r / Real.sin α :=
sorry

end NUMINAMATH_GPT_rhombus_side_length_l255_25571


namespace NUMINAMATH_GPT_supplementary_angle_difference_l255_25501

theorem supplementary_angle_difference (a b : ℝ) (h1 : a + b = 180) (h2 : 5 * b = 3 * a) : abs (a - b) = 45 :=
  sorry

end NUMINAMATH_GPT_supplementary_angle_difference_l255_25501


namespace NUMINAMATH_GPT_find_b_l255_25593

theorem find_b (a b c : ℝ) (h1 : a + b + c = 150) (h2 : a + 10 = c^2) (h3 : b - 5 = c^2) : 
  b = (1322 - 2 * Real.sqrt 1241) / 16 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l255_25593


namespace NUMINAMATH_GPT_count_neither_3_nor_4_l255_25594

def is_multiple_of_3_or_4 (n : Nat) : Bool := (n % 3 = 0) ∨ (n % 4 = 0)

def three_digit_numbers := List.range' 100 900 -- Generates a list from 100 to 999 (inclusive)

def count_multiples_of_3_or_4 : Nat := three_digit_numbers.filter is_multiple_of_3_or_4 |>.length

def count_total := 900 -- Since three-digit numbers range from 100 to 999

theorem count_neither_3_nor_4 : count_total - count_multiples_of_3_or_4 = 450 := by
  sorry

end NUMINAMATH_GPT_count_neither_3_nor_4_l255_25594


namespace NUMINAMATH_GPT_negative_number_is_d_l255_25588

def a : Int := -(-2)
def b : Int := abs (-2)
def c : Int := (-2) ^ 2
def d : Int := (-2) ^ 3

theorem negative_number_is_d : d < 0 :=
  by
  sorry

end NUMINAMATH_GPT_negative_number_is_d_l255_25588


namespace NUMINAMATH_GPT_intersection_M_N_l255_25590

def M : Set ℤ := { x | x^2 > 1 }
def N : Set ℤ := { -2, -1, 0, 1, 2 }

theorem intersection_M_N : (M ∩ N) = { -2, 2 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l255_25590


namespace NUMINAMATH_GPT_gain_in_meters_l255_25506

noncomputable def cost_price : ℝ := sorry
noncomputable def selling_price : ℝ := 1.5 * cost_price
noncomputable def total_cost_price : ℝ := 30 * cost_price
noncomputable def total_selling_price : ℝ := 30 * selling_price
noncomputable def gain : ℝ := total_selling_price - total_cost_price

theorem gain_in_meters (S C : ℝ) (h_S : S = 1.5 * C) (h_gain : gain = 15 * C) :
  15 * C / S = 10 := by
  sorry

end NUMINAMATH_GPT_gain_in_meters_l255_25506


namespace NUMINAMATH_GPT_stockholm_to_malmo_distance_l255_25514
-- Import the necessary library

-- Define the parameters for the problem.
def map_distance : ℕ := 120 -- distance in cm
def scale_factor : ℕ := 12 -- km per cm

-- The hypothesis for the map distance and the scale factor
axiom map_distance_hyp : map_distance = 120
axiom scale_factor_hyp : scale_factor = 12

-- Define the real distance function
def real_distance (d : ℕ) (s : ℕ) : ℕ := d * s

-- The problem statement: Prove that the real distance between the two city centers is 1440 km
theorem stockholm_to_malmo_distance : real_distance map_distance scale_factor = 1440 :=
by
  rw [map_distance_hyp, scale_factor_hyp]
  sorry

end NUMINAMATH_GPT_stockholm_to_malmo_distance_l255_25514


namespace NUMINAMATH_GPT_suresh_work_hours_l255_25560

variable (x : ℕ) -- Number of hours Suresh worked

theorem suresh_work_hours :
  (1/15 : ℝ) * x + (4 * (1/10 : ℝ)) = 1 -> x = 9 :=
by
  sorry

end NUMINAMATH_GPT_suresh_work_hours_l255_25560


namespace NUMINAMATH_GPT_journey_ratio_l255_25584

/-- Given a full-circle journey broken into parts,
  including paths through the Zoo Park (Z), the Circus (C), and the Park (P), 
  prove that the journey avoiding the Zoo Park is 11 times shorter. -/
theorem journey_ratio (Z C P : ℝ) (h1 : C = (3 / 4) * Z) 
                      (h2 : P = (1 / 4) * Z) : 
  Z = 11 * P := 
sorry

end NUMINAMATH_GPT_journey_ratio_l255_25584


namespace NUMINAMATH_GPT_number_of_distinct_linear_recurrences_l255_25580

open BigOperators

/-
  Let p be a prime positive integer.
  Define a mod-p recurrence of degree n to be a sequence {a_k}_{k >= 0} of numbers modulo p 
  satisfying a relation of the form:

  ai+n = c_n-1 ai+n-1 + ... + c_1 ai+1 + c_0 ai
  for all i >= 0, where c_0, c_1, ..., c_n-1 are integers and c_0 not equivalent to 0 mod p.
  Compute the number of distinct linear recurrences of degree at most n in terms of p and n.
-/
theorem number_of_distinct_linear_recurrences (p n : ℕ) (hp : Nat.Prime p) : 
  ∃ d : ℕ, 
    (∀ {a : ℕ → ℕ} {c : ℕ → ℕ} (h : ∀ i, a (i + n) = ∑ j in Finset.range n, c j * a (i + j))
     (hc0 : c 0 ≠ 0), 
      d = (1 - n * (p - 1) / (p + 1) + p^2 * (p^(2 * n) - 1) / (p + 1)^2 : ℚ)) :=
  sorry

end NUMINAMATH_GPT_number_of_distinct_linear_recurrences_l255_25580


namespace NUMINAMATH_GPT_stewart_farm_horseFood_l255_25557

variable (sheep horses horseFoodPerHorse : ℕ)
variable (ratio_sh_to_hs : ℕ × ℕ)
variable (totalHorseFood : ℕ)

noncomputable def horse_food_per_day (sheep : ℕ) (ratio_sh_to_hs : ℕ × ℕ) (totalHorseFood : ℕ) : ℕ :=
  let horses := (sheep * ratio_sh_to_hs.2) / ratio_sh_to_hs.1
  totalHorseFood / horses

theorem stewart_farm_horseFood (h_ratio : ratio_sh_to_hs = (4, 7))
                                (h_sheep : sheep = 32)
                                (h_total : totalHorseFood = 12880) :
    horse_food_per_day sheep ratio_sh_to_hs totalHorseFood = 230 := by
  sorry

end NUMINAMATH_GPT_stewart_farm_horseFood_l255_25557


namespace NUMINAMATH_GPT_natural_number_195_is_solution_l255_25577

-- Define the conditions
def is_odd_digit (n : ℕ) : Prop :=
  n > 0 ∧ n % 2 = 1

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, n / 10 ^ d % 10 < 10 → is_odd_digit (n / 10 ^ d % 10)

-- Define the proof problem
theorem natural_number_195_is_solution :
  195 < 200 ∧ all_digits_odd 195 ∧ (∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 195) :=
by
  sorry

end NUMINAMATH_GPT_natural_number_195_is_solution_l255_25577


namespace NUMINAMATH_GPT_square_side_length_increase_l255_25599

variables {a x : ℝ}

theorem square_side_length_increase 
  (h : (a * (1 + x / 100) * 1.8)^2 = (1 + 159.20000000000002 / 100) * (a^2 + (a * (1 + x / 100))^2)) : 
  x = 100 :=
by sorry

end NUMINAMATH_GPT_square_side_length_increase_l255_25599


namespace NUMINAMATH_GPT_tom_gave_8_boxes_l255_25585

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

end NUMINAMATH_GPT_tom_gave_8_boxes_l255_25585


namespace NUMINAMATH_GPT_total_amount_sold_l255_25562

theorem total_amount_sold (metres_sold : ℕ) (loss_per_metre cost_price_per_metre : ℕ) 
  (h1 : metres_sold = 600) (h2 : loss_per_metre = 5) (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre - loss_per_metre) * metres_sold = 18000 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_sold_l255_25562


namespace NUMINAMATH_GPT_B_share_in_profit_l255_25532

theorem B_share_in_profit (A B C : ℝ) (total_profit : ℝ) 
    (h1 : A = 3 * B)
    (h2 : B = (2/3) * C)
    (h3 : total_profit = 6600) :
    (B / (A + B + C)) * total_profit = 1200 := 
by
  sorry

end NUMINAMATH_GPT_B_share_in_profit_l255_25532


namespace NUMINAMATH_GPT_induction_step_l255_25531

theorem induction_step (x y : ℕ) (k : ℕ) (odd_k : k % 2 = 1) 
  (hk : (x + y) ∣ (x^k + y^k)) : (x + y) ∣ (x^(k+2) + y^(k+2)) :=
sorry

end NUMINAMATH_GPT_induction_step_l255_25531
