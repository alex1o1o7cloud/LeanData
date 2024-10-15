import Mathlib

namespace NUMINAMATH_GPT_a_gt_b_l1030_103017

theorem a_gt_b (n : ℕ) (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hn_ge_two : n ≥ 2)
  (ha_eq : a^n = a + 1) (hb_eq : b^(2*n) = b + 3 * a) : a > b :=
by
  sorry

end NUMINAMATH_GPT_a_gt_b_l1030_103017


namespace NUMINAMATH_GPT_proof_problem_l1030_103061

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1030_103061


namespace NUMINAMATH_GPT_cylinder_radius_l1030_103016

theorem cylinder_radius
  (r h : ℝ) (S : ℝ) (h_cylinder : h = 8) (S_surface : S = 130 * Real.pi)
  (surface_area_eq : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_l1030_103016


namespace NUMINAMATH_GPT_parabola_x_coordinate_l1030_103021

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p, 0)

theorem parabola_x_coordinate
  (M : ℝ × ℝ)
  (h_parabola : (M.2)^2 = 4 * M.1)
  (h_distance : dist M (parabola_focus 2) = 3) :
  M.1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_x_coordinate_l1030_103021


namespace NUMINAMATH_GPT_distance_halfway_along_orbit_l1030_103000

-- Define the conditions
variables (perihelion aphelion : ℝ) (perihelion_dist : perihelion = 3) (aphelion_dist : aphelion = 15)

-- State the theorem
theorem distance_halfway_along_orbit : 
  ∃ d, d = (perihelion + aphelion) / 2 ∧ d = 9 :=
by
  sorry

end NUMINAMATH_GPT_distance_halfway_along_orbit_l1030_103000


namespace NUMINAMATH_GPT_inequality_l1030_103083

theorem inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 :=
sorry

end NUMINAMATH_GPT_inequality_l1030_103083


namespace NUMINAMATH_GPT_simplify_expression_l1030_103022

theorem simplify_expression (a b c x : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  ( ( (x + a)^4 ) / ( (a - b) * (a - c) ) 
  + ( (x + b)^4 ) / ( (b - a) * (b - c) ) 
  + ( (x + c)^4 ) / ( (c - a) * (c - b) ) ) = a + b + c + 4 * x := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1030_103022


namespace NUMINAMATH_GPT_vector_equivalence_l1030_103036

-- Define the vectors a and b
noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def vector_b : ℝ × ℝ := (-1, 1)

-- Define the operation 3a - b
noncomputable def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
  (3 * a.1 - b.1, 3 * a.2 - b.2)

-- State that for given vectors a and b, the result of the operation equals (4, 2)
theorem vector_equivalence : vector_operation vector_a vector_b = (4, 2) :=
  sorry

end NUMINAMATH_GPT_vector_equivalence_l1030_103036


namespace NUMINAMATH_GPT_ellipse_intersects_x_axis_at_four_l1030_103052

theorem ellipse_intersects_x_axis_at_four
    (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 0))
    (h2 : f2 = (4, 0))
    (h3 : ∃ P : ℝ × ℝ, P = (1, 0) ∧ (dist P f1 + dist P f2 = 4)) :
  ∃ Q : ℝ × ℝ, Q = (4, 0) ∧ (dist Q f1 + dist Q f2 = 4) :=
sorry

end NUMINAMATH_GPT_ellipse_intersects_x_axis_at_four_l1030_103052


namespace NUMINAMATH_GPT_terminating_decimals_l1030_103078

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end NUMINAMATH_GPT_terminating_decimals_l1030_103078


namespace NUMINAMATH_GPT_valid_n_values_l1030_103024

theorem valid_n_values :
  {n : ℕ | ∀ a : ℕ, a^(n+1) ≡ a [MOD n]} = {1, 2, 6, 42, 1806} :=
sorry

end NUMINAMATH_GPT_valid_n_values_l1030_103024


namespace NUMINAMATH_GPT_parabola_directrix_l1030_103032

theorem parabola_directrix {x y : ℝ} (h : y^2 = 6 * x) : x = -3 / 2 := 
sorry

end NUMINAMATH_GPT_parabola_directrix_l1030_103032


namespace NUMINAMATH_GPT_dividend_is_176_l1030_103034

theorem dividend_is_176 (divisor quotient remainder : ℕ) (h1 : divisor = 19) (h2 : quotient = 9) (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end NUMINAMATH_GPT_dividend_is_176_l1030_103034


namespace NUMINAMATH_GPT_sum_first_five_arithmetic_l1030_103012

theorem sum_first_five_arithmetic (a : ℕ → ℝ) (h₁ : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h₂ : a 1 = -1) (h₃ : a 3 = -5) :
  (a 0 + a 1 + a 2 + a 3 + a 4) = -15 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_five_arithmetic_l1030_103012


namespace NUMINAMATH_GPT_meal_cost_per_person_l1030_103077

/-
Problem Statement:
Prove that the cost per meal is $3 given the conditions:
- There are 2 adults and 5 children.
- The total bill is $21.
-/

theorem meal_cost_per_person (total_adults : ℕ) (total_children : ℕ) (total_bill : ℝ) 
(total_people : ℕ) (cost_per_meal : ℝ) : 
total_adults = 2 → total_children = 5 → total_bill = 21 → total_people = total_adults + total_children →
cost_per_meal = total_bill / total_people → 
cost_per_meal = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_meal_cost_per_person_l1030_103077


namespace NUMINAMATH_GPT_ali_peter_fish_ratio_l1030_103051

theorem ali_peter_fish_ratio (P J A : ℕ) (h1 : J = P + 1) (h2 : A = 12) (h3 : A + P + J = 25) : A / P = 2 :=
by
  -- Step-by-step simplifications will follow here in the actual proof.
  sorry

end NUMINAMATH_GPT_ali_peter_fish_ratio_l1030_103051


namespace NUMINAMATH_GPT_milk_savings_l1030_103044

theorem milk_savings :
  let cost_for_two_packs : ℝ := 2.50
  let cost_per_pack_individual : ℝ := 1.30
  let num_packs_per_set := 2
  let num_sets := 10
  let cost_per_pack_set := cost_for_two_packs / num_packs_per_set
  let savings_per_pack := cost_per_pack_individual - cost_per_pack_set
  let total_packs := num_sets * num_packs_per_set
  let total_savings := savings_per_pack * total_packs
  total_savings = 1 :=
by
  sorry

end NUMINAMATH_GPT_milk_savings_l1030_103044


namespace NUMINAMATH_GPT_rotated_square_height_l1030_103018

noncomputable def height_of_B (side_length : ℝ) (rotation_angle : ℝ) : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let vertical_component := diagonal * Real.sin rotation_angle
  vertical_component

theorem rotated_square_height :
  height_of_B 1 (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rotated_square_height_l1030_103018


namespace NUMINAMATH_GPT_f_2006_eq_1_l1030_103027

noncomputable def f : ℤ → ℤ := sorry
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3 : ∀ x : ℤ, f (3 * (x + 1)) = f (3 * x + 1)
axiom f_at_1 : f 1 = -1

theorem f_2006_eq_1 : f 2006 = 1 := by
  sorry

end NUMINAMATH_GPT_f_2006_eq_1_l1030_103027


namespace NUMINAMATH_GPT_difference_of_square_of_non_divisible_by_3_l1030_103072

theorem difference_of_square_of_non_divisible_by_3 (n : ℕ) (h : ¬ (n % 3 = 0)) : (n^2 - 1) % 3 = 0 :=
sorry

end NUMINAMATH_GPT_difference_of_square_of_non_divisible_by_3_l1030_103072


namespace NUMINAMATH_GPT_find_c_l1030_103049

theorem find_c (a b c : ℝ) (h1 : ∃ a, ∃ b, ∃ c, 
              ∀ y, (∀ x, (x = a * (y-1)^2 + 4) ↔ (x = -2 → y = 3)) ∧
              (∀ y, x = a * y^2 + b * y + c)) : c = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_c_l1030_103049


namespace NUMINAMATH_GPT_ratio_of_doctors_to_lawyers_l1030_103043

/--
Given the average age of a group consisting of doctors and lawyers is 47,
the average age of doctors is 45,
and the average age of lawyers is 55,
prove that the ratio of the number of doctors to the number of lawyers is 4:1.
-/
theorem ratio_of_doctors_to_lawyers
  (d l : ℕ) -- numbers of doctors and lawyers
  (avg_group_age : ℝ := 47)
  (avg_doctors_age : ℝ := 45)
  (avg_lawyers_age : ℝ := 55)
  (h : (45 * d + 55 * l) / (d + l) = 47) :
  d = 4 * l :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_doctors_to_lawyers_l1030_103043


namespace NUMINAMATH_GPT_train_speed_in_kmh_l1030_103074

def length_of_train : ℝ := 600
def length_of_overbridge : ℝ := 100
def time_to_cross_overbridge : ℝ := 70

theorem train_speed_in_kmh :
  (length_of_train + length_of_overbridge) / time_to_cross_overbridge * 3.6 = 36 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l1030_103074


namespace NUMINAMATH_GPT_two_x_plus_y_equals_7_l1030_103070

noncomputable def proof_problem (x y A : ℝ) : ℝ :=
  if (2 * x + y = A ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) then A else 0

theorem two_x_plus_y_equals_7 (x y : ℝ) : 
  (2 * x + y = proof_problem x y 7) ↔
  (2 * x + y = 7 ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) :=
by sorry

end NUMINAMATH_GPT_two_x_plus_y_equals_7_l1030_103070


namespace NUMINAMATH_GPT_molecular_weight_correct_l1030_103063

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms
def num_N : ℕ := 2
def num_O : ℕ := 3

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 76.02

-- The theorem to prove
theorem molecular_weight_correct :
  (num_N * atomic_weight_N + num_O * atomic_weight_O) = expected_molecular_weight := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l1030_103063


namespace NUMINAMATH_GPT_Deepak_age_l1030_103033

variable (R D : ℕ)

theorem Deepak_age 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26) : D = 15 := 
sorry

end NUMINAMATH_GPT_Deepak_age_l1030_103033


namespace NUMINAMATH_GPT_add_100ml_water_l1030_103089

theorem add_100ml_water 
    (current_volume : ℕ) 
    (current_water_percentage : ℝ) 
    (desired_water_percentage : ℝ) 
    (current_water_volume : ℝ) 
    (x : ℝ) :
    current_volume = 300 →
    current_water_percentage = 0.60 →
    desired_water_percentage = 0.70 →
    current_water_volume = 0.60 * 300 →
    180 + x = 0.70 * (300 + x) →
    x = 100 := 
sorry

end NUMINAMATH_GPT_add_100ml_water_l1030_103089


namespace NUMINAMATH_GPT_members_play_both_l1030_103005

-- Define the conditions
variables (N B T neither : ℕ)
variables (B_union_T B_and_T : ℕ)

-- Assume the given conditions
axiom hN : N = 42
axiom hB : B = 20
axiom hT : T = 23
axiom hNeither : neither = 6
axiom hB_union_T : B_union_T = N - neither

-- State the problem: Prove that B_and_T = 7
theorem members_play_both (N B T neither B_union_T B_and_T : ℕ) 
  (hN : N = 42) 
  (hB : B = 20) 
  (hT : T = 23) 
  (hNeither : neither = 6) 
  (hB_union_T : B_union_T = N - neither) 
  (hInclusionExclusion : B_union_T = B + T - B_and_T) :
  B_and_T = 7 := sorry

end NUMINAMATH_GPT_members_play_both_l1030_103005


namespace NUMINAMATH_GPT_perfect_squares_l1030_103009

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end NUMINAMATH_GPT_perfect_squares_l1030_103009


namespace NUMINAMATH_GPT_john_total_cost_l1030_103001

-- The total cost John incurs to rent a car, buy gas, and drive 320 miles
def total_cost (rental_cost gas_cost_per_gallon cost_per_mile miles driven_gallons : ℝ): ℝ :=
  rental_cost + (gas_cost_per_gallon * driven_gallons) + (cost_per_mile * miles)

theorem john_total_cost :
  let rental_cost := 150
  let gallons := 8
  let gas_cost_per_gallon := 3.50
  let cost_per_mile := 0.50
  let miles := 320
  total_cost rental_cost gas_cost_per_gallon cost_per_mile miles gallons = 338 := 
by
  -- The detailed proof is skipped here
  sorry

end NUMINAMATH_GPT_john_total_cost_l1030_103001


namespace NUMINAMATH_GPT_company_profit_is_correct_l1030_103065

structure CompanyInfo where
  num_employees : ℕ
  shirts_per_employee_per_day : ℕ
  hours_per_shift : ℕ
  wage_per_hour : ℕ
  bonus_per_shirt : ℕ
  price_per_shirt : ℕ
  nonemployee_expenses_per_day : ℕ

def daily_profit (info : CompanyInfo) : ℤ :=
  let total_shirts_per_day := info.num_employees * info.shirts_per_employee_per_day
  let total_revenue := total_shirts_per_day * info.price_per_shirt
  let daily_wage_per_employee := info.wage_per_hour * info.hours_per_shift
  let total_daily_wage := daily_wage_per_employee * info.num_employees
  let daily_bonus_per_employee := info.bonus_per_shirt * info.shirts_per_employee_per_day
  let total_daily_bonus := daily_bonus_per_employee * info.num_employees
  let total_labor_cost := total_daily_wage + total_daily_bonus
  let total_expenses := total_labor_cost + info.nonemployee_expenses_per_day
  total_revenue - total_expenses

theorem company_profit_is_correct (info : CompanyInfo) (h : 
  info.num_employees = 20 ∧
  info.shirts_per_employee_per_day = 20 ∧
  info.hours_per_shift = 8 ∧
  info.wage_per_hour = 12 ∧
  info.bonus_per_shirt = 5 ∧
  info.price_per_shirt = 35 ∧
  info.nonemployee_expenses_per_day = 1000
) : daily_profit info = 9080 := 
by
  sorry

end NUMINAMATH_GPT_company_profit_is_correct_l1030_103065


namespace NUMINAMATH_GPT_eddie_age_l1030_103031

theorem eddie_age (Becky_age Irene_age Eddie_age : ℕ)
  (h1 : Becky_age * 2 = Irene_age)
  (h2 : Irene_age = 46)
  (h3 : Eddie_age = 4 * Becky_age) :
  Eddie_age = 92 := by
  sorry

end NUMINAMATH_GPT_eddie_age_l1030_103031


namespace NUMINAMATH_GPT_tan_half_angle_second_quadrant_l1030_103084

variables (θ : ℝ) (k : ℤ)
open Real

theorem tan_half_angle_second_quadrant (h : (π / 2) + 2 * k * π < θ ∧ θ < π + 2 * k * π) : 
  tan (θ / 2) > 1 := 
sorry

end NUMINAMATH_GPT_tan_half_angle_second_quadrant_l1030_103084


namespace NUMINAMATH_GPT_log_expression_equals_four_l1030_103040

/-- 
  Given the expression as: x = \log_3 (81 + \log_3 (81 + \log_3 (81 + \cdots))), 
  we need to prove that x = 4
  provided that x = \log_3 (81 + x), i.e., 3^x = x + 81.
  And given that the value of x is positive.
-/
theorem log_expression_equals_four
  (x : ℝ)
  (h1 : x = Real.log 81 / Real.log 3 + Real.log (81 + x) / Real.log 3): 
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_equals_four_l1030_103040


namespace NUMINAMATH_GPT_max_bus_capacity_l1030_103019

-- Definitions and conditions
def left_side_regular_seats := 12
def left_side_priority_seats := 3
def right_side_regular_seats := 9
def right_side_priority_seats := 2
def right_side_wheelchair_space := 1
def regular_seat_capacity := 3
def priority_seat_capacity := 2
def back_row_seat_capacity := 7
def standing_capacity := 14

-- Definition of total bus capacity
def total_bus_capacity : ℕ :=
  (left_side_regular_seats * regular_seat_capacity) + 
  (left_side_priority_seats * priority_seat_capacity) + 
  (right_side_regular_seats * regular_seat_capacity) + 
  (right_side_priority_seats * priority_seat_capacity) + 
  back_row_seat_capacity + 
  standing_capacity

-- Theorem to prove
theorem max_bus_capacity : total_bus_capacity = 94 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_max_bus_capacity_l1030_103019


namespace NUMINAMATH_GPT_find_total_money_l1030_103042

theorem find_total_money
  (d x T : ℝ)
  (h1 : d = 5 / 17)
  (h2 : x = 35)
  (h3 : d * T = x) :
  T = 119 :=
by sorry

end NUMINAMATH_GPT_find_total_money_l1030_103042


namespace NUMINAMATH_GPT_more_cats_than_spinsters_l1030_103071

theorem more_cats_than_spinsters :
  ∀ (S C : ℕ), (S = 18) → (2 * C = 9 * S) → (C - S = 63) :=
by
  intros S C hS hRatio
  sorry

end NUMINAMATH_GPT_more_cats_than_spinsters_l1030_103071


namespace NUMINAMATH_GPT_round_robin_teams_l1030_103068

theorem round_robin_teams (x : ℕ) (h : x ≠ 0) :
  (x * (x - 1)) / 2 = 15 → ∃ n : ℕ, x = n :=
by
  sorry

end NUMINAMATH_GPT_round_robin_teams_l1030_103068


namespace NUMINAMATH_GPT_ratio_of_roses_l1030_103010

theorem ratio_of_roses (total_flowers tulips carnations roses : ℕ) 
  (h1 : total_flowers = 40) 
  (h2 : tulips = 10) 
  (h3 : carnations = 14) 
  (h4 : roses = total_flowers - (tulips + carnations)) :
  roses / total_flowers = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_roses_l1030_103010


namespace NUMINAMATH_GPT_planted_fraction_l1030_103096

theorem planted_fraction (length width radius : ℝ) (h_field : length * width = 24)
  (h_circle : π * radius^2 = π) : (24 - π) / 24 = (24 - π) / 24 :=
by
  -- all proofs are skipped
  sorry

end NUMINAMATH_GPT_planted_fraction_l1030_103096


namespace NUMINAMATH_GPT_arccos_half_eq_pi_div_three_l1030_103066

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end NUMINAMATH_GPT_arccos_half_eq_pi_div_three_l1030_103066


namespace NUMINAMATH_GPT_min_positive_period_f_max_value_f_decreasing_intervals_g_l1030_103058

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem min_positive_period_f : 
  ∃ (p : ℝ), p > 0 ∧ (∀ x : ℝ, f (x + 2*Real.pi) = f x) :=
sorry

theorem max_value_f : 
  ∃ (M : ℝ), (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = 2 :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (-x)

theorem decreasing_intervals_g :
  ∀ (k : ℤ), ∀ x : ℝ, (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
  ∀ (h : x ≤ Real.pi * 2 * (↑k+1)), g x ≥ g (x + Real.pi) :=
sorry

end NUMINAMATH_GPT_min_positive_period_f_max_value_f_decreasing_intervals_g_l1030_103058


namespace NUMINAMATH_GPT_packs_of_sugar_l1030_103045

theorem packs_of_sugar (cost_apples_per_kg cost_walnuts_per_kg cost_apples total : ℝ) (weight_apples weight_walnuts : ℝ) (less_sugar_by_1 : ℝ) (packs : ℕ) :
  cost_apples_per_kg = 2 →
  cost_walnuts_per_kg = 6 →
  cost_apples = weight_apples * cost_apples_per_kg →
  weight_apples = 5 →
  weight_walnuts = 0.5 →
  less_sugar_by_1 = 1 →
  total = 16 →
  packs = (total - (weight_apples * cost_apples_per_kg + weight_walnuts * cost_walnuts_per_kg)) / (cost_apples_per_kg - less_sugar_by_1) →
  packs = 3 :=
by
  sorry

end NUMINAMATH_GPT_packs_of_sugar_l1030_103045


namespace NUMINAMATH_GPT_distinct_integers_no_perfect_square_product_l1030_103030

theorem distinct_integers_no_perfect_square_product
  (k : ℕ) (hk : 0 < k) :
  ∀ a b : ℕ, k^2 < a ∧ a < (k+1)^2 → k^2 < b ∧ b < (k+1)^2 → a ≠ b → ¬∃ m : ℕ, a * b = m^2 :=
by sorry

end NUMINAMATH_GPT_distinct_integers_no_perfect_square_product_l1030_103030


namespace NUMINAMATH_GPT_dividend_is_correct_l1030_103064

theorem dividend_is_correct :
  ∃ (R D Q V: ℕ), R = 6 ∧ D = 5 * Q ∧ D = 3 * R + 2 ∧ V = D * Q + R ∧ V = 86 :=
by
  sorry

end NUMINAMATH_GPT_dividend_is_correct_l1030_103064


namespace NUMINAMATH_GPT_sum_of_final_two_numbers_l1030_103057

noncomputable def final_sum (X m n : ℚ) : ℚ :=
  3 * m + 3 * n - 14

theorem sum_of_final_two_numbers (X m n : ℚ) 
  (h1 : m + n = X) :
  final_sum X m n = 3 * X - 14 :=
  sorry

end NUMINAMATH_GPT_sum_of_final_two_numbers_l1030_103057


namespace NUMINAMATH_GPT_max_sum_of_two_integers_l1030_103048

theorem max_sum_of_two_integers (x : ℕ) (h : x + 2 * x < 100) : x + 2 * x = 99 :=
sorry

end NUMINAMATH_GPT_max_sum_of_two_integers_l1030_103048


namespace NUMINAMATH_GPT_angle_BCA_measure_l1030_103056

theorem angle_BCA_measure
  (A B C : Type)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_BAC : ℝ)
  (h1 : angle_ABC = 90)
  (h2 : angle_BAC = 2 * angle_BCA) :
  angle_BCA = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_BCA_measure_l1030_103056


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_l1030_103020

theorem quadratic_has_two_real_roots (a b c : ℝ) (h : a * c < 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * (x1^2) + b * x1 + c = 0 ∧ a * (x2^2) + b * x2 + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_l1030_103020


namespace NUMINAMATH_GPT_correct_calculation_l1030_103087

theorem correct_calculation (a : ℝ) : a^4 / a = a^3 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l1030_103087


namespace NUMINAMATH_GPT_right_triangle_legs_sum_l1030_103041

theorem right_triangle_legs_sum (x : ℕ) (hx1 : x * x + (x + 1) * (x + 1) = 41 * 41) : x + (x + 1) = 59 :=
by sorry

end NUMINAMATH_GPT_right_triangle_legs_sum_l1030_103041


namespace NUMINAMATH_GPT_households_used_both_brands_l1030_103090

theorem households_used_both_brands (X : ℕ) : 
  (80 + 60 + X + 3 * X = 260) → X = 30 :=
by
  sorry

end NUMINAMATH_GPT_households_used_both_brands_l1030_103090


namespace NUMINAMATH_GPT_fraction_division_problem_l1030_103028

theorem fraction_division_problem :
  (-1/42 : ℚ) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 :=
by
  -- Skipping the proof step as per the instructions
  sorry

end NUMINAMATH_GPT_fraction_division_problem_l1030_103028


namespace NUMINAMATH_GPT_unit_digit_smaller_by_four_l1030_103050

theorem unit_digit_smaller_by_four (x : ℤ) : x^2 + (x + 4)^2 = 10 * (x + 4) + x - 4 :=
by
  sorry

end NUMINAMATH_GPT_unit_digit_smaller_by_four_l1030_103050


namespace NUMINAMATH_GPT_find_certain_number_l1030_103075

-- Definitions of conditions from the problem
def greatest_number : ℕ := 10
def divided_1442_by_greatest_number_leaves_remainder := (1442 % greatest_number = 12)
def certain_number_mod_greatest_number (x : ℕ) := (x % greatest_number = 6)

-- Theorem statement
theorem find_certain_number (x : ℕ) (h1 : greatest_number = 10)
  (h2 : 1442 % greatest_number = 12)
  (h3 : certain_number_mod_greatest_number x) : x = 1446 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l1030_103075


namespace NUMINAMATH_GPT_power_division_correct_l1030_103037

theorem power_division_correct :
  (∀ x : ℝ, x^4 / x = x^3) ∧ 
  ¬(∀ x : ℝ, 3 * x^2 * 4 * x^2 = 12 * x^2) ∧
  ¬(∀ x : ℝ, (x - 1) * (x - 1) = x^2 - 1) ∧
  ¬(∀ x : ℝ, (x^5)^2 = x^7) := 
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_power_division_correct_l1030_103037


namespace NUMINAMATH_GPT_rectangle_perimeter_l1030_103004

variables (L W : ℕ)

-- conditions
def conditions : Prop :=
  L - 4 = W + 3 ∧
  (L - 4) * (W + 3) = L * W

-- prove the solution
theorem rectangle_perimeter (h : conditions L W) : 2 * L + 2 * W = 50 := sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1030_103004


namespace NUMINAMATH_GPT_beijing_time_conversion_l1030_103002

-- Define the conversion conditions
def new_clock_hours_in_day : Nat := 10
def new_clock_minutes_per_hour : Nat := 100
def new_clock_time_at_5_beijing_time : Nat := 12 * 60  -- converting 12 noon to minutes


-- Define the problem to prove the corresponding Beijing time 
theorem beijing_time_conversion :
  new_clock_minutes_per_hour * 5 = 500 → 
  new_clock_time_at_5_beijing_time = 720 →
  (720 + 175 * 1.44) = 4 * 60 + 12 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_beijing_time_conversion_l1030_103002


namespace NUMINAMATH_GPT_harkamal_mangoes_l1030_103059

theorem harkamal_mangoes (m : ℕ) (h1: 8 * 70 = 560) (h2 : m * 50 + 560 = 1010) : m = 9 :=
by
  sorry

end NUMINAMATH_GPT_harkamal_mangoes_l1030_103059


namespace NUMINAMATH_GPT_dolphins_scored_15_l1030_103080

theorem dolphins_scored_15 (s d : ℤ) 
  (h1 : s + d = 48) 
  (h2 : s - d = 18) : 
  d = 15 := 
sorry

end NUMINAMATH_GPT_dolphins_scored_15_l1030_103080


namespace NUMINAMATH_GPT_final_position_l1030_103029

structure Position where
  base : ℝ × ℝ
  stem : ℝ × ℝ

def rotate180 (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectX (pos : Position) : Position :=
  { base := (pos.base.1, -pos.base.2),
    stem := (pos.stem.1, -pos.stem.2) }

def rotateHalfTurn (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectY (pos : Position) : Position :=
  { base := (-pos.base.1, pos.base.2),
    stem := (-pos.stem.1, pos.stem.2) }

theorem final_position : 
  let initial_pos := Position.mk (1, 0) (0, 1)
  let pos1 := rotate180 initial_pos
  let pos2 := reflectX pos1
  let pos3 := rotateHalfTurn pos2
  let final_pos := reflectY pos3
  final_pos = { base := (-1, 0), stem := (0, -1) } :=
by
  sorry

end NUMINAMATH_GPT_final_position_l1030_103029


namespace NUMINAMATH_GPT_pre_images_of_one_l1030_103093

def f (x : ℝ) := x^3 - x + 1

theorem pre_images_of_one : {x : ℝ | f x = 1} = {-1, 0, 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_pre_images_of_one_l1030_103093


namespace NUMINAMATH_GPT_calc_val_l1030_103099

theorem calc_val : 
  (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 :=
by 
  -- Calculation proof
  sorry

end NUMINAMATH_GPT_calc_val_l1030_103099


namespace NUMINAMATH_GPT_final_percentage_acid_l1030_103086

theorem final_percentage_acid (initial_volume : ℝ) (initial_percentage : ℝ)
(removal_volume : ℝ) (final_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 12 → 
  initial_percentage = 0.40 → 
  removal_volume = 4 →
  final_volume = initial_volume - removal_volume →
  final_percentage = (initial_percentage * initial_volume) / final_volume * 100 →
  final_percentage = 60 := by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_final_percentage_acid_l1030_103086


namespace NUMINAMATH_GPT_sum_of_ages_is_l1030_103015

-- Define the ages of the triplets and twins
def age_triplet (x : ℕ) := x
def age_twin (x : ℕ) := x - 3

-- Define the total age sum
def total_age_sum (x : ℕ) := 3 * age_triplet x + 2 * age_twin x

-- State the theorem
theorem sum_of_ages_is (x : ℕ) (h : total_age_sum x = 89) : ∃ x : ℕ, total_age_sum x = 89 := 
sorry

end NUMINAMATH_GPT_sum_of_ages_is_l1030_103015


namespace NUMINAMATH_GPT_cameron_list_length_l1030_103023

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end NUMINAMATH_GPT_cameron_list_length_l1030_103023


namespace NUMINAMATH_GPT_daily_serving_size_l1030_103069

-- Definitions based on problem conditions
def days : ℕ := 180
def capsules_per_bottle : ℕ := 60
def bottles : ℕ := 6
def total_capsules : ℕ := bottles * capsules_per_bottle

-- Theorem statement to prove the daily serving size
theorem daily_serving_size :
  total_capsules / days = 2 := by
  sorry

end NUMINAMATH_GPT_daily_serving_size_l1030_103069


namespace NUMINAMATH_GPT_complex_exp_form_pow_four_l1030_103046

theorem complex_exp_form_pow_four :
  let θ := 30 * Real.pi / 180
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  let z := 3 * (cos_θ + Complex.I * sin_θ)
  z ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_complex_exp_form_pow_four_l1030_103046


namespace NUMINAMATH_GPT_S21_sum_is_4641_l1030_103094

-- Define the conditions and the sum of the nth group
def first_number_in_group (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

def last_number_in_group (n : ℕ) : ℕ :=
  first_number_in_group n + (n - 1)

def sum_of_group (n : ℕ) : ℕ :=
  n * (first_number_in_group n + last_number_in_group n) / 2

-- The theorem to prove
theorem S21_sum_is_4641 : sum_of_group 21 = 4641 := by
  sorry

end NUMINAMATH_GPT_S21_sum_is_4641_l1030_103094


namespace NUMINAMATH_GPT_function_graph_intersection_l1030_103062

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end NUMINAMATH_GPT_function_graph_intersection_l1030_103062


namespace NUMINAMATH_GPT_roots_of_unity_polynomial_l1030_103054

theorem roots_of_unity_polynomial (c d : ℤ) (z : ℂ) (hz : z^3 = 1) :
  (z^3 + c * z + d = 0) → (z = 1) :=
sorry

end NUMINAMATH_GPT_roots_of_unity_polynomial_l1030_103054


namespace NUMINAMATH_GPT_non_adjacent_placements_l1030_103098

theorem non_adjacent_placements (n : ℕ) : 
  let total_ways := n^2 * (n^2 - 1)
  let adjacent_ways := 2 * n^2 - 2 * n
  (total_ways - adjacent_ways) = n^4 - 3 * n^2 + 2 * n :=
by
  -- Proof is sorted out
  sorry

end NUMINAMATH_GPT_non_adjacent_placements_l1030_103098


namespace NUMINAMATH_GPT_find_k_eq_3_l1030_103047

theorem find_k_eq_3 (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3) → k = 3 :=
by sorry

end NUMINAMATH_GPT_find_k_eq_3_l1030_103047


namespace NUMINAMATH_GPT_airfare_price_for_BD_l1030_103014

theorem airfare_price_for_BD (AB AC AD CD BC : ℝ) (hAB : AB = 2000) (hAC : AC = 1600) (hAD : AD = 2500) 
    (hCD : CD = 900) (hBC : BC = 1200) (proportional_pricing : ∀ x y : ℝ, x * (y / x) = y) : 
    ∃ BD : ℝ, BD = 1500 :=
by
  sorry

end NUMINAMATH_GPT_airfare_price_for_BD_l1030_103014


namespace NUMINAMATH_GPT_total_toys_is_correct_l1030_103097

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_toys_is_correct_l1030_103097


namespace NUMINAMATH_GPT_abhinav_annual_salary_l1030_103053

def RamMontlySalary : ℝ := 25600
def ShyamMontlySalary (A : ℝ) := 2 * A
def AbhinavAnnualSalary (A : ℝ) := 12 * A

theorem abhinav_annual_salary (A : ℝ) : 
  0.10 * RamMontlySalary = 0.08 * ShyamMontlySalary A → 
  AbhinavAnnualSalary A = 192000 :=
by
  sorry

end NUMINAMATH_GPT_abhinav_annual_salary_l1030_103053


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1030_103055

noncomputable def f (x : Real) : Real := (1 + Real.sin (2 * x)) / 2
noncomputable def a : Real := f (Real.log 5)
noncomputable def b : Real := f (Real.log (1 / 5))

theorem sum_of_a_and_b : a + b = 1 := by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1030_103055


namespace NUMINAMATH_GPT_inequality_proof_l1030_103039

theorem inequality_proof {x y z : ℝ}
  (h1 : x + 2 * y + 4 * z ≥ 3)
  (h2 : y - 3 * x + 2 * z ≥ 5) :
  y - x + 2 * z ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1030_103039


namespace NUMINAMATH_GPT_sequence_formula_l1030_103008

open Nat

noncomputable def S : ℕ → ℤ
| n => n^2 - 2 * n + 2

noncomputable def a : ℕ → ℤ
| 0 => 1  -- note that in Lean, sequence indexing starts from 0
| (n+1) => 2*(n+1) - 3

theorem sequence_formula (n : ℕ) : 
  a n = if n = 0 then 1 else 2*n - 3 := by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1030_103008


namespace NUMINAMATH_GPT_minimum_employees_needed_l1030_103076

def min_new_employees (water_pollution: ℕ) (air_pollution: ℕ) (both: ℕ) : ℕ :=
  119 + 34

theorem minimum_employees_needed : min_new_employees 98 89 34 = 153 := 
  by
  sorry

end NUMINAMATH_GPT_minimum_employees_needed_l1030_103076


namespace NUMINAMATH_GPT_ratio_of_ages_l1030_103025

theorem ratio_of_ages (sandy_future_age : ℕ) (sandy_years_future : ℕ) (molly_current_age : ℕ)
  (h1 : sandy_future_age = 42) (h2 : sandy_years_future = 6) (h3 : molly_current_age = 27) :
  (sandy_future_age - sandy_years_future) / gcd (sandy_future_age - sandy_years_future) molly_current_age = 
    4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1030_103025


namespace NUMINAMATH_GPT_inequality_proof_l1030_103003

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) +
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) +
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1030_103003


namespace NUMINAMATH_GPT_probability_sum_18_l1030_103095

def total_outcomes := 100

def successful_pairs := [(8, 10), (9, 9), (10, 8)]

def num_successful_outcomes := successful_pairs.length

theorem probability_sum_18 : (num_successful_outcomes / total_outcomes : ℚ) = 3 / 100 := 
by
  -- The actual proof should go here
  sorry

end NUMINAMATH_GPT_probability_sum_18_l1030_103095


namespace NUMINAMATH_GPT_pies_left_l1030_103082

theorem pies_left (pies_per_batch : ℕ) (batches : ℕ) (dropped : ℕ) (total_pies : ℕ) (pies_left : ℕ)
  (h1 : pies_per_batch = 5)
  (h2 : batches = 7)
  (h3 : dropped = 8)
  (h4 : total_pies = pies_per_batch * batches)
  (h5 : pies_left = total_pies - dropped) :
  pies_left = 27 := by
  sorry

end NUMINAMATH_GPT_pies_left_l1030_103082


namespace NUMINAMATH_GPT_probability_of_selecting_one_defective_l1030_103011

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_one_defective_l1030_103011


namespace NUMINAMATH_GPT_length_of_AP_l1030_103092

noncomputable def square_side_length : ℝ := 8
noncomputable def rect_width : ℝ := 12
noncomputable def rect_height : ℝ := 8

axiom AD_perpendicular_WX : true
axiom shaded_area_half_WXYZ : true

theorem length_of_AP (AP : ℝ) (shaded_area : ℝ)
  (h1 : shaded_area = (rect_width * rect_height) / 2)
  (h2 : shaded_area = (square_side_length - AP) * square_side_length)
  : AP = 2 := by
  sorry

end NUMINAMATH_GPT_length_of_AP_l1030_103092


namespace NUMINAMATH_GPT_total_water_filled_jars_l1030_103007

theorem total_water_filled_jars :
  ∃ x : ℕ, 
    16 * (1/4) + 12 * (1/2) + 8 * 1 + 4 * 2 + x * 3 = 56 ∧
    16 + 12 + 8 + 4 + x = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_water_filled_jars_l1030_103007


namespace NUMINAMATH_GPT_group_members_count_l1030_103026

theorem group_members_count (n: ℕ) (total_paise: ℕ) (condition1: total_paise = 3249) :
  (n * n = total_paise) → n = 57 :=
by
  sorry

end NUMINAMATH_GPT_group_members_count_l1030_103026


namespace NUMINAMATH_GPT_fifth_grade_total_students_l1030_103067

-- Define the conditions given in the problem
def total_boys : ℕ := 350
def total_playing_soccer : ℕ := 250
def percentage_boys_playing_soccer : ℝ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- Define the total number of students
def total_students : ℕ := 500

-- Prove that the total number of students is 500
theorem fifth_grade_total_students 
  (H1 : total_boys = 350) 
  (H2 : total_playing_soccer = 250) 
  (H3 : percentage_boys_playing_soccer = 0.86) 
  (H4 : girls_not_playing_soccer = 115) :
  total_students = 500 := 
sorry

end NUMINAMATH_GPT_fifth_grade_total_students_l1030_103067


namespace NUMINAMATH_GPT_geometric_sequence_sum_first_five_terms_l1030_103091

theorem geometric_sequence_sum_first_five_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 30)
  (h_geom : ∀ n, a (n + 1) = a n * q) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_first_five_terms_l1030_103091


namespace NUMINAMATH_GPT_expand_expression_l1030_103035

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 :=
  sorry

end NUMINAMATH_GPT_expand_expression_l1030_103035


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1030_103038

noncomputable def has_negative_root (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x < 0

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ x < 0) ↔ (a < 0) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1030_103038


namespace NUMINAMATH_GPT_sphere_radius_l1030_103079

theorem sphere_radius (r_A r_B : ℝ) (h₁ : r_A = 40) (h₂ : (4 * π * r_A^2) / (4 * π * r_B^2) = 16) : r_B = 20 :=
  sorry

end NUMINAMATH_GPT_sphere_radius_l1030_103079


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1030_103060

-- System 1
theorem system1_solution (x y : ℝ) 
  (h1 : y = 2 * x - 3)
  (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := 
by
  sorry

-- System 2
theorem system2_solution (x y : ℝ) 
  (h1 : x + 2 * y = 3)
  (h2 : 2 * x - 4 * y = -10) : 
  x = -1 ∧ y = 2 := 
by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1030_103060


namespace NUMINAMATH_GPT_max_product_of_two_integers_sum_2000_l1030_103088

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end NUMINAMATH_GPT_max_product_of_two_integers_sum_2000_l1030_103088


namespace NUMINAMATH_GPT_parametric_equation_solution_l1030_103073

noncomputable def solve_parametric_equation (a b : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) : ℝ :=
  (5 / (a - 2 * b))

theorem parametric_equation_solution (a b x : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) 
  (h : (a * x - 3) / (b * x + 1) = 2) : 
  x = solve_parametric_equation a b ha2b ha3b :=
sorry

end NUMINAMATH_GPT_parametric_equation_solution_l1030_103073


namespace NUMINAMATH_GPT_sequence_count_l1030_103081

def num_sequences (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem sequence_count :
  let x := 490
  let y := 510
  let a : (n : ℕ) → ℕ := fun n => if n = 0 then 0 else if n = 1000 then 2020 else sorry
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1000 → (a (k + 1) - a k = 1 ∨ a (k + 1) - a k = 3)) →
  (∃ binomial_coeff, binomial_coeff = num_sequences 1000 490) :=
by sorry

end NUMINAMATH_GPT_sequence_count_l1030_103081


namespace NUMINAMATH_GPT_relation_of_exponents_l1030_103006

theorem relation_of_exponents
  (a b c d : ℝ)
  (x y p z : ℝ)
  (h1 : a^x = c)
  (h2 : b^p = c)
  (h3 : b^y = d)
  (h4 : a^z = d) :
  py = xz :=
sorry

end NUMINAMATH_GPT_relation_of_exponents_l1030_103006


namespace NUMINAMATH_GPT_pumpkin_weight_difference_l1030_103013

theorem pumpkin_weight_difference (Brad: ℕ) (Jessica: ℕ) (Betty: ℕ) 
    (h1 : Brad = 54) 
    (h2 : Jessica = Brad / 2) 
    (h3 : Betty = Jessica * 4) 
    : (Betty - Jessica) = 81 := 
by
  sorry

end NUMINAMATH_GPT_pumpkin_weight_difference_l1030_103013


namespace NUMINAMATH_GPT_z_is_negative_y_intercept_l1030_103085

-- Define the objective function as an assumption or condition
def objective_function (x y z : ℝ) : Prop := z = 3 * x - y

-- Define what we need to prove: z is the negative of the y-intercept 
def negative_y_intercept (x y z : ℝ) : Prop := ∃ m b, (y = m * x + b) ∧ m = 3 ∧ b = -z

-- The theorem we need to prove
theorem z_is_negative_y_intercept (x y z : ℝ) (h : objective_function x y z) : negative_y_intercept x y z :=
  sorry

end NUMINAMATH_GPT_z_is_negative_y_intercept_l1030_103085
