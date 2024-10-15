import Mathlib

namespace NUMINAMATH_GPT_calculate_expression_l1925_192549

theorem calculate_expression (y : ℝ) (hy : y ≠ 0) : 
  (18 * y^3) * (4 * y^2) * (1/(2 * y)^3) = 9 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1925_192549


namespace NUMINAMATH_GPT_area_times_breadth_l1925_192580

theorem area_times_breadth (b l A : ℕ) (h1 : b = 11) (h2 : l - b = 10) (h3 : A = l * b) : A / b = 21 := 
by
  sorry

end NUMINAMATH_GPT_area_times_breadth_l1925_192580


namespace NUMINAMATH_GPT_find_x_rational_l1925_192562

theorem find_x_rational (x : ℝ) (h1 : ∃ (a : ℚ), x + Real.sqrt 3 = a)
  (h2 : ∃ (b : ℚ), x^2 + Real.sqrt 3 = b) :
  x = (1 / 2 : ℝ) - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_x_rational_l1925_192562


namespace NUMINAMATH_GPT_angle_A_range_l1925_192594

open Real

theorem angle_A_range (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) (h3 : 0 < A ∧ A < π) : 
  π / 2 < A ∧ A < 3 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_range_l1925_192594


namespace NUMINAMATH_GPT_lawn_width_is_60_l1925_192566

theorem lawn_width_is_60
  (length : ℕ)
  (width : ℕ)
  (road_width : ℕ)
  (cost_per_sq_meter : ℕ)
  (total_cost : ℕ)
  (area_of_lawn : ℕ)
  (total_area_of_roads : ℕ)
  (intersection_area : ℕ)
  (area_cost_relation : total_area_of_roads * cost_per_sq_meter = total_cost)
  (intersection_included : (road_width * length + road_width * width - intersection_area) = total_area_of_roads)
  (length_eq : length = 80)
  (road_width_eq : road_width = 10)
  (cost_eq : cost_per_sq_meter = 2)
  (total_cost_eq : total_cost = 2600)
  (intersection_area_eq : intersection_area = road_width * road_width)
  : width = 60 :=
by
  sorry

end NUMINAMATH_GPT_lawn_width_is_60_l1925_192566


namespace NUMINAMATH_GPT_initial_owls_l1925_192533

theorem initial_owls (n_0 : ℕ) (h : n_0 + 2 = 5) : n_0 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_initial_owls_l1925_192533


namespace NUMINAMATH_GPT_quadratic_decreasing_l1925_192517

-- Define the quadratic function and the condition a < 0
def quadratic_function (a x : ℝ) := a * x^2 - 2 * a * x + 1

-- Define the main theorem to be proven
theorem quadratic_decreasing (a m : ℝ) (ha : a < 0) : 
  (∀ x, x > m → quadratic_function a x < quadratic_function a (x+1)) ↔ m ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_decreasing_l1925_192517


namespace NUMINAMATH_GPT_calculation_identity_l1925_192515

theorem calculation_identity :
  (3.14 - 1)^0 * (-1 / 4)^(-2) = 16 := by
  sorry

end NUMINAMATH_GPT_calculation_identity_l1925_192515


namespace NUMINAMATH_GPT_coterminal_angle_in_radians_l1925_192505

theorem coterminal_angle_in_radians (d : ℝ) (h : d = 2010) : 
  ∃ r : ℝ, r = -5 * Real.pi / 6 ∧ (∃ k : ℤ, d = r * 180 / Real.pi + k * 360) :=
by sorry

end NUMINAMATH_GPT_coterminal_angle_in_radians_l1925_192505


namespace NUMINAMATH_GPT_pollution_index_minimum_l1925_192529

noncomputable def pollution_index (k a b : ℝ) (x : ℝ) : ℝ :=
  k * (a / (x ^ 2) + b / ((18 - x) ^ 2))

theorem pollution_index_minimum (k : ℝ) (h₀ : 0 < k) (h₁ : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 18) :
  ∀ a b x : ℝ, a = 1 → x = 6 → pollution_index k a b x = pollution_index k 1 8 6 :=
by
  intros a b x ha hx
  rw [ha, hx, pollution_index]
  sorry

end NUMINAMATH_GPT_pollution_index_minimum_l1925_192529


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1925_192567

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def decreasing_interval (a b : ℝ) := 
  ∀ x : ℝ, a < x ∧ x < b → deriv f x < 0

theorem monotonic_decreasing_interval : decreasing_interval 0 1 :=
sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1925_192567


namespace NUMINAMATH_GPT_volume_of_large_ball_l1925_192509

theorem volume_of_large_ball (r : ℝ) (V_small : ℝ) (h1 : 1 = r / (2 * r)) (h2 : V_small = (4 / 3) * Real.pi * r^3) : 
  8 * V_small = 288 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_large_ball_l1925_192509


namespace NUMINAMATH_GPT_yoongi_average_score_l1925_192507

/-- 
Yoongi's average score on the English test taken in August and September was 86, and his English test score in October was 98. 
Prove that the average score of the English test for 3 months is 90.
-/
theorem yoongi_average_score 
  (avg_aug_sep : ℕ)
  (score_oct : ℕ)
  (hp1 : avg_aug_sep = 86)
  (hp2 : score_oct = 98) :
  ((avg_aug_sep * 2 + score_oct) / 3) = 90 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_average_score_l1925_192507


namespace NUMINAMATH_GPT_percentage_less_than_l1925_192502

variable (x y : ℝ)
variable (H : y = 1.4 * x)

theorem percentage_less_than :
  ((y - x) / y) * 100 = 28.57 := by
  sorry

end NUMINAMATH_GPT_percentage_less_than_l1925_192502


namespace NUMINAMATH_GPT_garden_perimeter_is_24_l1925_192526

def perimeter_of_garden(a b c x: ℕ) (h1: a + b + c = 3) : ℕ :=
  3 + 5 + a + x + b + 4 + c + 4 + 5 - x

theorem garden_perimeter_is_24 (a b c x : ℕ) (h1 : a + b + c = 3) :
  perimeter_of_garden a b c x h1 = 24 :=
  by
  sorry

end NUMINAMATH_GPT_garden_perimeter_is_24_l1925_192526


namespace NUMINAMATH_GPT_question1_question2_l1925_192531

-- Definitions based on the conditions
def f (x m : ℝ) : ℝ := x^2 + 4*x + m

theorem question1 (m : ℝ) (h1 : m ≠ 0) (h2 : 16 - 4 * m > 0) : m < 4 :=
  sorry

theorem question2 (m : ℝ) (hx : ∀ x : ℝ, f x m = 0 → f (-x - 4) m = 0) 
  (h_circ : ∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1) ∨ (x = -4 ∧ y = 1)) :
  (∀ (x y : ℝ), x^2 + y^2 + 4*x - (m + 1) * y + m = 0 → (x = 0 ∧ y = 1)) ∨ (∀ (x y : ℝ), (x = -4 ∧ y = 1)) :=
  sorry

end NUMINAMATH_GPT_question1_question2_l1925_192531


namespace NUMINAMATH_GPT_find_f_one_third_l1925_192543

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def satisfies_condition (f : ℝ → ℝ) : Prop :=
∀ x, f (2 - x) = f x

noncomputable def f (x : ℝ) : ℝ := if (2 ≤ x ∧ x ≤ 3) then Real.log (x - 1) / Real.log 2 else 0

theorem find_f_one_third (h_odd : is_odd_function f) (h_condition : satisfies_condition f) :
  f (1 / 3) = Real.log 3 / Real.log 2 - 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_one_third_l1925_192543


namespace NUMINAMATH_GPT_suff_not_necessary_condition_l1925_192542

noncomputable def p : ℝ := 1
noncomputable def q (x : ℝ) : Prop := x^3 - 2 * x + 1 = 0

theorem suff_not_necessary_condition :
  (∀ x, x = p → q x) ∧ (∃ x, q x ∧ x ≠ p) :=
by
  sorry

end NUMINAMATH_GPT_suff_not_necessary_condition_l1925_192542


namespace NUMINAMATH_GPT_max_wins_l1925_192519

theorem max_wins (Chloe_wins Max_wins : ℕ) (h1 : Chloe_wins = 24) (h2 : 8 * Max_wins = 3 * Chloe_wins) : Max_wins = 9 := by
  sorry

end NUMINAMATH_GPT_max_wins_l1925_192519


namespace NUMINAMATH_GPT_difference_english_math_l1925_192574

/-- There are 30 students who pass in English and 20 students who pass in Math. -/
axiom passes_in_english : ℕ
axiom passes_in_math : ℕ
axiom both_subjects : ℕ
axiom only_english : ℕ
axiom only_math : ℕ

/-- Definitions based on the problem conditions -/
axiom number_passes_in_english : only_english + both_subjects = 30
axiom number_passes_in_math : only_math + both_subjects = 20

/-- The difference between the number of students who pass only in English
    and the number of students who pass only in Math is 10. -/
theorem difference_english_math : only_english - only_math = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_english_math_l1925_192574


namespace NUMINAMATH_GPT_find_x_ge_0_l1925_192508

-- Defining the condition and the proof problem
theorem find_x_ge_0 :
  {x : ℝ | (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0} = {x : ℝ | 0 ≤ x} :=
by
  sorry -- proof steps not included

end NUMINAMATH_GPT_find_x_ge_0_l1925_192508


namespace NUMINAMATH_GPT_find_S_l1925_192521

noncomputable def A := { x : ℝ | x^2 - 7 * x + 10 ≤ 0 }
noncomputable def B (a b : ℝ) := { x : ℝ | x^2 + a * x + b < 0 }
def A_inter_B_is_empty (a b : ℝ) := A ∩ B a b = ∅
def A_union_B_condition := { x : ℝ | x - 3 < 4 ∧ 4 ≤ 2 * x }

theorem find_S :
  A ∪ B (-12) 35 = { x : ℝ | 2 ≤ x ∧ x < 7 } →
  A ∩ B (-12) 35 = ∅ →
  { x : ℝ | x = -12 + 35 } = { 23 } :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_S_l1925_192521


namespace NUMINAMATH_GPT_train_length_l1925_192532

theorem train_length
  (S : ℝ)  -- speed of the train in meters per second
  (L : ℝ)  -- length of the train in meters
  (h1 : L = S * 20)
  (h2 : L + 500 = S * 40) :
  L = 500 := 
sorry

end NUMINAMATH_GPT_train_length_l1925_192532


namespace NUMINAMATH_GPT_Iain_pennies_problem_l1925_192592

theorem Iain_pennies_problem :
  ∀ (P : ℝ), 200 - 30 = 170 →
             170 - (P / 100) * 170 = 136 →
             P = 20 :=
by
  intros P h1 h2
  sorry

end NUMINAMATH_GPT_Iain_pennies_problem_l1925_192592


namespace NUMINAMATH_GPT_basket_A_apples_count_l1925_192579

-- Conditions
def total_baskets : ℕ := 5
def avg_fruits_per_basket : ℕ := 25
def fruits_in_B : ℕ := 30
def fruits_in_C : ℕ := 20
def fruits_in_D : ℕ := 25
def fruits_in_E : ℕ := 35

-- Calculation of total number of fruits
def total_fruits : ℕ := total_baskets * avg_fruits_per_basket
def other_baskets_fruits : ℕ := fruits_in_B + fruits_in_C + fruits_in_D + fruits_in_E

-- Question and Proof Goal
theorem basket_A_apples_count : total_fruits - other_baskets_fruits = 15 := by
  sorry

end NUMINAMATH_GPT_basket_A_apples_count_l1925_192579


namespace NUMINAMATH_GPT_Seokhyung_drank_the_most_l1925_192575

-- Define the conditions
def Mina_Amount := 0.6
def Seokhyung_Amount := 1.5
def Songhwa_Amount := Seokhyung_Amount - 0.6

-- Statement to prove that Seokhyung drank the most cola
theorem Seokhyung_drank_the_most : Seokhyung_Amount > Mina_Amount ∧ Seokhyung_Amount > Songhwa_Amount :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_Seokhyung_drank_the_most_l1925_192575


namespace NUMINAMATH_GPT_lcm_20_45_75_eq_900_l1925_192557

theorem lcm_20_45_75_eq_900 : Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
by sorry

end NUMINAMATH_GPT_lcm_20_45_75_eq_900_l1925_192557


namespace NUMINAMATH_GPT_equation_of_circle_C_equation_of_line_l_l1925_192576

-- Condition: The center of the circle lies on the line y = x + 1.
def center_on_line (a b : ℝ) : Prop :=
  b = a + 1

-- Condition: The circle is tangent to the x-axis.
def tangent_to_x_axis (a b r : ℝ) : Prop :=
  r = b

-- Condition: Point P(-5, -2) lies on the circle.
def point_on_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Condition: Point Q(-4, -5) lies outside the circle.
def point_outside_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 > r^2

-- Proof (1): Find the equation of the circle.
theorem equation_of_circle_C :
  ∃ (a b r : ℝ), center_on_line a b ∧ tangent_to_x_axis a b r ∧ point_on_circle a b r (-5) (-2) ∧ point_outside_circle a b r (-4) (-5) ∧ (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 3)^2 + (y + 2)^2 = 4) :=
sorry

-- Proof (2): Find the equation of the line l.
theorem equation_of_line_l (a b r : ℝ) (ha : center_on_line a b) (hb : tangent_to_x_axis a b r) (hc : point_on_circle a b r (-5) (-2)) (hd : point_outside_circle a b r (-4) (-5)) :
  ∃ (k : ℝ), ∀ x y, ((k = 0 ∧ x = -2) ∨ (k ≠ 0 ∧ y + 4 = -3/4 * (x + 2))) ↔ ((x = -2) ∨ (3 * x + 4 * y + 22 = 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_circle_C_equation_of_line_l_l1925_192576


namespace NUMINAMATH_GPT_g_x_equation_g_3_value_l1925_192555

noncomputable def g : ℝ → ℝ := sorry

theorem g_x_equation (x : ℝ) (hx : x ≠ 1/2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x := sorry

theorem g_3_value : g 3 = 31 / 8 :=
by
  -- Use the provided functional equation and specific input values to derive g(3)
  sorry

end NUMINAMATH_GPT_g_x_equation_g_3_value_l1925_192555


namespace NUMINAMATH_GPT_water_usage_difference_l1925_192522

theorem water_usage_difference (C X : ℕ)
    (h1 : C = 111000)
    (h2 : C = 3 * X)
    (days : ℕ) (h3 : days = 365) :
    (C * days - X * days) = 26910000 := by
  sorry

end NUMINAMATH_GPT_water_usage_difference_l1925_192522


namespace NUMINAMATH_GPT_meal_arrangement_exactly_two_correct_l1925_192589

noncomputable def meal_arrangement_count : ℕ :=
  let total_people := 13
  let meal_types := ["B", "B", "B", "B", "C", "C", "C", "F", "F", "F", "V", "V", "V"]
  let choose_2_people := (total_people.choose 2)
  let derangement_7 := 1854  -- Derangement of BBCCCVVV
  let derangement_9 := 133496  -- Derangement of BBCCFFFVV
  choose_2_people * (derangement_7 + derangement_9)

theorem meal_arrangement_exactly_two_correct : meal_arrangement_count = 10482600 := by
  sorry

end NUMINAMATH_GPT_meal_arrangement_exactly_two_correct_l1925_192589


namespace NUMINAMATH_GPT_smallest_class_size_l1925_192577

theorem smallest_class_size (n : ℕ) (x : ℕ) (h1 : n > 50) (h2 : n = 4 * x + 2) : n = 54 :=
by
  sorry

end NUMINAMATH_GPT_smallest_class_size_l1925_192577


namespace NUMINAMATH_GPT_intersection_of_N_and_not_R_M_l1925_192506

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def Not_R_M : Set ℝ := {x | x ≤ 2}

theorem intersection_of_N_and_not_R_M : 
  N ∩ Not_R_M = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_N_and_not_R_M_l1925_192506


namespace NUMINAMATH_GPT_cosine_identity_l1925_192528

theorem cosine_identity (alpha : ℝ) (h1 : -180 < alpha ∧ alpha < -90)
  (cos_75_alpha : Real.cos (75 * Real.pi / 180 + alpha) = 1 / 3) :
  Real.cos (15 * Real.pi / 180 - alpha) = -2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_GPT_cosine_identity_l1925_192528


namespace NUMINAMATH_GPT_cricket_target_runs_l1925_192573

-- Define the conditions
def first_20_overs_run_rate : ℝ := 4.2
def remaining_30_overs_run_rate : ℝ := 8
def overs_20 : ℤ := 20
def overs_30 : ℤ := 30

-- State the proof problem
theorem cricket_target_runs : 
  (first_20_overs_run_rate * (overs_20 : ℝ)) + (remaining_30_overs_run_rate * (overs_30 : ℝ)) = 324 :=
by
  sorry

end NUMINAMATH_GPT_cricket_target_runs_l1925_192573


namespace NUMINAMATH_GPT_area_ratio_none_of_these_l1925_192587

theorem area_ratio_none_of_these (h r a : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) (a_pos : 0 < a) (h_square_a_square : h^2 > a^2) :
  ¬ (∃ ratio, ratio = (π * r / (h + r)) ∨
               ratio = (π * r^2 / (a + h)) ∨
               ratio = (π * a * r / (h + 2 * r)) ∨
               ratio = (π * r / (a + r))) :=
by sorry

end NUMINAMATH_GPT_area_ratio_none_of_these_l1925_192587


namespace NUMINAMATH_GPT_cost_of_perfume_l1925_192530

-- Definitions and Constants
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def neighbors_yards_mowed : ℕ := 4
def charge_per_yard : ℕ := 5
def dogs_walked : ℕ := 6
def charge_per_dog : ℕ := 2
def additional_amount_needed : ℕ := 6

-- Theorem Statement
theorem cost_of_perfume :
  let christian_earnings := neighbors_yards_mowed * charge_per_yard
  let sue_earnings := dogs_walked * charge_per_dog
  let christian_savings := christian_initial_savings + christian_earnings
  let sue_savings := sue_initial_savings + sue_earnings
  let total_savings := christian_savings + sue_savings
  total_savings + additional_amount_needed = 50 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_perfume_l1925_192530


namespace NUMINAMATH_GPT_original_number_l1925_192558

theorem original_number (N : ℕ) :
  (∃ k m n : ℕ, N - 6 = 5 * k + 3 ∧ N - 6 = 11 * m + 3 ∧ N - 6 = 13 * n + 3) → N = 724 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l1925_192558


namespace NUMINAMATH_GPT_total_games_l1925_192510

variable (G R : ℕ)

axiom cond1 : 85 + (1/2 : ℚ) * R = (0.70 : ℚ) * G
axiom cond2 : G = 100 + R

theorem total_games : G = 175 := by
  sorry

end NUMINAMATH_GPT_total_games_l1925_192510


namespace NUMINAMATH_GPT_wrench_turns_bolt_l1925_192598

theorem wrench_turns_bolt (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (Real.sqrt 3 / Real.sqrt 2 < b / a) ∧ (b / a ≤ 3 - Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_wrench_turns_bolt_l1925_192598


namespace NUMINAMATH_GPT_growth_rate_yield_per_acre_l1925_192591

theorem growth_rate_yield_per_acre (x : ℝ) (a_i y_i y_f : ℝ) (h1 : a_i = 5) (h2 : y_i = 10000) (h3 : y_f = 30000) 
  (h4 : y_f = 5 * (1 + 2 * x) * (y_i / a_i) * (1 + x)) : x = 0.5 := 
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_growth_rate_yield_per_acre_l1925_192591


namespace NUMINAMATH_GPT_min_buildings_20x20_min_buildings_50x90_l1925_192546

structure CityGrid where
  width : ℕ
  height : ℕ

noncomputable def renovationLaw (grid : CityGrid) : ℕ :=
  if grid.width = 20 ∧ grid.height = 20 then 25
  else if grid.width = 50 ∧ grid.height = 90 then 282
  else sorry -- handle other cases if needed

-- Theorem statements for the proof
theorem min_buildings_20x20 : renovationLaw { width := 20, height := 20 } = 25 := by
  sorry

theorem min_buildings_50x90 : renovationLaw { width := 50, height := 90 } = 282 := by
  sorry

end NUMINAMATH_GPT_min_buildings_20x20_min_buildings_50x90_l1925_192546


namespace NUMINAMATH_GPT_calc_pow_expression_l1925_192572

theorem calc_pow_expression : (27^3 * 9^2) / 3^15 = 1 / 9 := 
by sorry

end NUMINAMATH_GPT_calc_pow_expression_l1925_192572


namespace NUMINAMATH_GPT_permute_rows_to_columns_l1925_192523

open Function

-- Define the problem
theorem permute_rows_to_columns {α : Type*} [Fintype α] [DecidableEq α] (n : ℕ)
  (table : Fin n → Fin n → α)
  (h_distinct_rows : ∀ i : Fin n, ∀ j₁ j₂ : Fin n, j₁ ≠ j₂ → table i j₁ ≠ table i j₂) :
  ∃ (p : Fin n → Fin n → Fin n), ∀ j : Fin n, ∀ i₁ i₂ : Fin n, i₁ ≠ i₂ →
    table i₁ (p i₁ j) ≠ table i₂ (p i₂ j) := 
sorry

end NUMINAMATH_GPT_permute_rows_to_columns_l1925_192523


namespace NUMINAMATH_GPT_cost_of_pen_l1925_192599

theorem cost_of_pen :
  ∃ p q : ℚ, (3 * p + 4 * q = 264) ∧ (4 * p + 2 * q = 230) ∧ (p = 39.2) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pen_l1925_192599


namespace NUMINAMATH_GPT_shadow_of_cube_l1925_192503

theorem shadow_of_cube (x : ℝ) (h_edge : ∀ c : ℝ, c = 2) (h_shadow_area : ∀ a : ℝ, a = 200 + 4) :
  ⌊1000 * x⌋ = 12280 :=
by
  sorry

end NUMINAMATH_GPT_shadow_of_cube_l1925_192503


namespace NUMINAMATH_GPT_tens_digit_of_6_pow_19_l1925_192554

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tens_digit_of_6_pow_19 : tens_digit (6 ^ 19) = 9 := 
by 
  sorry

end NUMINAMATH_GPT_tens_digit_of_6_pow_19_l1925_192554


namespace NUMINAMATH_GPT_eugene_payment_correct_l1925_192552

theorem eugene_payment_correct :
  let t_price := 20
  let p_price := 80
  let s_price := 150
  let discount_rate := 0.1
  let t_quantity := 4
  let p_quantity := 3
  let s_quantity := 2
  let t_cost := t_quantity * t_price
  let p_cost := p_quantity * p_price
  let s_cost := s_quantity * s_price
  let total_cost := t_cost + p_cost + s_cost
  let discount := discount_rate * total_cost
  let final_cost := total_cost - discount
  final_cost = 558 :=
by
  sorry

end NUMINAMATH_GPT_eugene_payment_correct_l1925_192552


namespace NUMINAMATH_GPT_wuyang_volleyball_team_members_l1925_192578

theorem wuyang_volleyball_team_members :
  (Finset.filter Nat.Prime (Finset.range 50)).card = 15 :=
by
  sorry

end NUMINAMATH_GPT_wuyang_volleyball_team_members_l1925_192578


namespace NUMINAMATH_GPT_number_of_shelves_l1925_192581

/-- Adam could fit 11 action figures on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- Adam's shelves could hold a total of 44 action figures -/
def total_action_figures_on_shelves : ℕ := 44

/-- Prove the number of shelves in Adam's room -/
theorem number_of_shelves:
  total_action_figures_on_shelves / action_figures_per_shelf = 4 := 
by {
    sorry
}

end NUMINAMATH_GPT_number_of_shelves_l1925_192581


namespace NUMINAMATH_GPT_finding_f_of_neg_half_l1925_192544

def f (x : ℝ) : ℝ := sorry

theorem finding_f_of_neg_half : f (-1/2) = Real.pi / 3 :=
by
  -- Given function definition condition: f (cos x) = x / 2 for 0 ≤ x ≤ π
  -- f should be defined on ℝ -> ℝ such that this condition holds;
  -- Applying this condition should verify our theorem.
  sorry

end NUMINAMATH_GPT_finding_f_of_neg_half_l1925_192544


namespace NUMINAMATH_GPT_solve_fraction_zero_l1925_192537

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 25) / (x + 5) = 0) (h2 : x ≠ -5) : x = 5 :=
sorry

end NUMINAMATH_GPT_solve_fraction_zero_l1925_192537


namespace NUMINAMATH_GPT_compare_binary_digits_l1925_192541

def numDigits_base2 (n : ℕ) : ℕ :=
  (Nat.log2 n) + 1

theorem compare_binary_digits :
  numDigits_base2 1600 - numDigits_base2 400 = 2 := by
  sorry

end NUMINAMATH_GPT_compare_binary_digits_l1925_192541


namespace NUMINAMATH_GPT_brandon_textbooks_weight_l1925_192590

-- Define the weights of Jon's textbooks
def jon_textbooks : List ℕ := [2, 8, 5, 9]

-- Define the weight ratio between Jon's and Brandon's textbooks
def weight_ratio : ℕ := 3

-- Define the total weight of Jon's textbooks
def weight_jon : ℕ := jon_textbooks.sum

-- Define the weight of Brandon's textbooks to be proven
def weight_brandon : ℕ := weight_jon / weight_ratio

-- The theorem to be proven
theorem brandon_textbooks_weight : weight_brandon = 8 :=
by sorry

end NUMINAMATH_GPT_brandon_textbooks_weight_l1925_192590


namespace NUMINAMATH_GPT_simplify_expression_l1925_192571

theorem simplify_expression (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1925_192571


namespace NUMINAMATH_GPT_distance_with_tide_60_min_l1925_192569

variable (v_m v_t : ℝ)

axiom man_with_tide : (v_m + v_t) = 5
axiom man_against_tide : (v_m - v_t) = 4

theorem distance_with_tide_60_min : (v_m + v_t) = 5 := by
  sorry

end NUMINAMATH_GPT_distance_with_tide_60_min_l1925_192569


namespace NUMINAMATH_GPT_find_x_l1925_192524

theorem find_x (a b c d x : ℕ) 
  (h1 : x = a + 7) 
  (h2 : a = b + 12) 
  (h3 : b = c + 15) 
  (h4 : c = d + 25) 
  (h5 : d = 95) : 
  x = 154 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1925_192524


namespace NUMINAMATH_GPT_expr_value_l1925_192593

-- Define the given expression
def expr : ℕ := 11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3

-- Assert the proof goal
theorem expr_value : expr = 21 := by
  sorry

end NUMINAMATH_GPT_expr_value_l1925_192593


namespace NUMINAMATH_GPT_minimum_m_n_1978_l1925_192585

-- Define the conditions given in the problem
variables (m n : ℕ) (h1 : n > m) (h2 : m > 1)
-- Define the condition that the last three digits of 1978^m and 1978^n are identical
def same_last_three_digits (a b : ℕ) : Prop :=
  (a % 1000 = b % 1000)

-- Define the problem statement: under the conditions, prove that m + n = 106 when minimized
theorem minimum_m_n_1978 (h : same_last_three_digits (1978^m) (1978^n)) : m + n = 106 :=
sorry   -- Proof will be provided here

end NUMINAMATH_GPT_minimum_m_n_1978_l1925_192585


namespace NUMINAMATH_GPT_unique_real_function_l1925_192551

theorem unique_real_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, (f (x * y) / 2 + f (x * z) / 2 - f x * f (y * z)) ≥ 1 / 4) →
  (∀ x : ℝ, f x = 1 / 2) :=
by
  intro h
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_unique_real_function_l1925_192551


namespace NUMINAMATH_GPT_tarantulas_per_egg_sac_l1925_192545

-- Condition: Each tarantula has 8 legs
def legs_per_tarantula : ℕ := 8

-- Condition: There are 32000 baby tarantula legs
def total_legs : ℕ := 32000

-- Condition: Number of egg sacs is one less than 5
def number_of_egg_sacs : ℕ := 5 - 1

-- Calculated: Number of tarantulas in total
def total_tarantulas : ℕ := total_legs / legs_per_tarantula

-- Proof Statement: Number of tarantulas per egg sac
theorem tarantulas_per_egg_sac : total_tarantulas / number_of_egg_sacs = 1000 := by
  sorry

end NUMINAMATH_GPT_tarantulas_per_egg_sac_l1925_192545


namespace NUMINAMATH_GPT_hess_law_delta_H298_l1925_192534

def standardEnthalpyNa2O : ℝ := -416 -- kJ/mol
def standardEnthalpyH2O : ℝ := -286 -- kJ/mol
def standardEnthalpyNaOH : ℝ := -427.8 -- kJ/mol
def deltaH298 : ℝ := 2 * standardEnthalpyNaOH - (standardEnthalpyNa2O + standardEnthalpyH2O) 

theorem hess_law_delta_H298 : deltaH298 = -153.6 := by
  sorry

end NUMINAMATH_GPT_hess_law_delta_H298_l1925_192534


namespace NUMINAMATH_GPT_inequality_of_ab_l1925_192595

theorem inequality_of_ab (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b ∧ b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_ab_l1925_192595


namespace NUMINAMATH_GPT_fraction_spent_on_furniture_l1925_192540

theorem fraction_spent_on_furniture (original_savings : ℝ) (cost_of_tv : ℝ) (f : ℝ)
  (h1 : original_savings = 1800) 
  (h2 : cost_of_tv = 450) 
  (h3 : f * original_savings + cost_of_tv = original_savings) :
  f = 3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_spent_on_furniture_l1925_192540


namespace NUMINAMATH_GPT_total_number_of_vehicles_l1925_192504

theorem total_number_of_vehicles 
  (lanes : ℕ) 
  (trucks_per_lane : ℕ) 
  (buses_per_lane : ℕ) 
  (cars_per_lane : ℕ := 2 * lanes * trucks_per_lane) 
  (motorcycles_per_lane : ℕ := 3 * buses_per_lane)
  (total_trucks : ℕ := lanes * trucks_per_lane)
  (total_cars : ℕ := lanes * cars_per_lane)
  (total_buses : ℕ := lanes * buses_per_lane)
  (total_motorcycles : ℕ := lanes * motorcycles_per_lane)
  (total_vehicles : ℕ := total_trucks + total_cars + total_buses + total_motorcycles)
  (hlanes : lanes = 4) 
  (htrucks : trucks_per_lane = 60) 
  (hbuses : buses_per_lane = 40) :
  total_vehicles = 2800 := sorry

end NUMINAMATH_GPT_total_number_of_vehicles_l1925_192504


namespace NUMINAMATH_GPT_number_of_non_congruent_triangles_perimeter_18_l1925_192596

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end NUMINAMATH_GPT_number_of_non_congruent_triangles_perimeter_18_l1925_192596


namespace NUMINAMATH_GPT_sec_neg_450_undefined_l1925_192559

theorem sec_neg_450_undefined : ¬ ∃ x, x = 1 / Real.cos (-450 * Real.pi / 180) :=
by
  -- Proof skipped using 'sorry'
  sorry

end NUMINAMATH_GPT_sec_neg_450_undefined_l1925_192559


namespace NUMINAMATH_GPT_quadratic_minimum_eq_one_l1925_192583

variable (p q : ℝ)

theorem quadratic_minimum_eq_one (hq : q = 1 + p^2 / 18) : 
  ∃ x : ℝ, 3 * x^2 + p * x + q = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_minimum_eq_one_l1925_192583


namespace NUMINAMATH_GPT_exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l1925_192547

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem exists_a_f_has_two_zeros (a : ℝ) :
  (0 < a ∧ a < 2) ∨ (-2 < a ∧ a < 0) → ∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧ f x₁ a = 0) ∧
  (0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧ f x₂ a = 0) ∧ x₁ ≠ x₂ := sorry

theorem range_of_a_for_f_eq_g :
  ∀ a : ℝ, a ∈ Set.Icc (-2 : ℝ) (3 : ℝ) →
  ∃ x₁ : ℝ, x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ f x₁ a = g 2 ∧
  ∃ x₂ : ℝ, x₂ ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧ f x₁ a = g x₂ := sorry

end NUMINAMATH_GPT_exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l1925_192547


namespace NUMINAMATH_GPT_find_n_l1925_192511

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_lcm1 : Nat.lcm 40 n = 120) (h_lcm2 : Nat.lcm n 45 = 180) : n = 12 :=
sorry

end NUMINAMATH_GPT_find_n_l1925_192511


namespace NUMINAMATH_GPT_gmat_test_takers_correctly_l1925_192514

variable (A B : ℝ)
variable (intersection union : ℝ)

theorem gmat_test_takers_correctly :
  B = 0.8 ∧ intersection = 0.7 ∧ union = 0.95 → A = 0.85 :=
by 
  sorry

end NUMINAMATH_GPT_gmat_test_takers_correctly_l1925_192514


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1925_192520

theorem triangle_is_isosceles (A B C a b c : ℝ) (h_sin : Real.sin (A + B) = 2 * Real.sin A * Real.cos B)
  (h_sine_rule : 2 * a * Real.cos B = c)
  (h_cosine_rule : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)) : a = b :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1925_192520


namespace NUMINAMATH_GPT_find_fourth_vertex_of_square_l1925_192550

-- Given the vertices of the square as complex numbers
def vertex1 : ℂ := 1 + 2 * Complex.I
def vertex2 : ℂ := -2 + Complex.I
def vertex3 : ℂ := -1 - 2 * Complex.I

-- The fourth vertex (to be proved)
def vertex4 : ℂ := 2 - Complex.I

-- The mathematically equivalent proof problem statement
theorem find_fourth_vertex_of_square :
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  -- Define vectors from the vertices
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4
  vector_ab = vector_dc :=
by {
  -- Definitions already provided above
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4

  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_find_fourth_vertex_of_square_l1925_192550


namespace NUMINAMATH_GPT_system_solution_l1925_192548

theorem system_solution (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧ 
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4) ∧ (y = -1) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l1925_192548


namespace NUMINAMATH_GPT_contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l1925_192560

variable {A B : Prop}

def contrary (A : Prop) : Prop := A ∧ ¬A
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B)

theorem contrary_implies_mutually_exclusive (A : Prop) : contrary A → mutually_exclusive A (¬A) :=
by sorry

theorem contrary_sufficient_but_not_necessary (A B : Prop) :
  (∃ (A : Prop), contrary A) → mutually_exclusive A B →
  (∃ (A : Prop), contrary A ∧ mutually_exclusive A B) :=
by sorry

end NUMINAMATH_GPT_contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l1925_192560


namespace NUMINAMATH_GPT_complement_A_is_interval_l1925_192597

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def compl_U_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem complement_A_is_interval : compl_U_A = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_complement_A_is_interval_l1925_192597


namespace NUMINAMATH_GPT_generated_surface_l1925_192501

theorem generated_surface (L : ℝ → ℝ → ℝ → Prop)
  (H1 : ∀ x y z, L x y z → y = z) 
  (H2 : ∀ t, L (t^2 / 2) t 0) 
  (H3 : ∀ s, L (s^2 / 3) 0 s) : 
  ∀ y z, ∃ x, L x y z → x = (y - z) * (y / 2 - z / 3) :=
by
  sorry

end NUMINAMATH_GPT_generated_surface_l1925_192501


namespace NUMINAMATH_GPT_trapezoid_area_l1925_192512

theorem trapezoid_area
  (A B C D : ℝ)
  (BC AD AC : ℝ)
  (radius circle_center : ℝ)
  (h : ℝ)
  (angleBAD angleADC : ℝ)
  (tangency : Bool) :
  BC = 13 → 
  angleBAD = 2 * angleADC →
  radius = 5 →
  tangency = true →
  1/2 * (BC + AD) * h = 157.5 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1925_192512


namespace NUMINAMATH_GPT_find_k_l1925_192539

variables {α : Type*} [CommRing α]

theorem find_k (a b c : α) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c :=
by sorry

end NUMINAMATH_GPT_find_k_l1925_192539


namespace NUMINAMATH_GPT_simplify_expr1_l1925_192516

theorem simplify_expr1 (m n : ℝ) :
  (2 * m + n) ^ 2 - (4 * m + 3 * n) * (m - n) = 8 * m * n + 4 * n ^ 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr1_l1925_192516


namespace NUMINAMATH_GPT_ways_to_get_off_the_bus_l1925_192556

-- Define the number of passengers and stops
def numPassengers : ℕ := 10
def numStops : ℕ := 5

-- Define the theorem that states the number of ways for passengers to get off
theorem ways_to_get_off_the_bus : (numStops^numPassengers) = 5^10 :=
by sorry

end NUMINAMATH_GPT_ways_to_get_off_the_bus_l1925_192556


namespace NUMINAMATH_GPT_positive_whole_numbers_with_cube_roots_less_than_15_l1925_192535

theorem positive_whole_numbers_with_cube_roots_less_than_15 :
  ∃ n : ℕ, n = 3374 ∧ ∀ x : ℕ, (x > 0 ∧ x < 3375) → x <= n :=
by sorry

end NUMINAMATH_GPT_positive_whole_numbers_with_cube_roots_less_than_15_l1925_192535


namespace NUMINAMATH_GPT_min_value_of_A_l1925_192525

noncomputable def A (a b c : ℝ) : ℝ :=
  (a^3 + b^3) / (8 * a * b + 9 - c^2) +
  (b^3 + c^3) / (8 * b * c + 9 - a^2) +
  (c^3 + a^3) / (8 * c * a + 9 - b^2)

theorem min_value_of_A (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  A a b c = 3 / 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_A_l1925_192525


namespace NUMINAMATH_GPT_arielle_age_l1925_192561

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end NUMINAMATH_GPT_arielle_age_l1925_192561


namespace NUMINAMATH_GPT_smallest_whole_number_l1925_192500

theorem smallest_whole_number (a : ℕ) : 
  (a % 4 = 1) ∧ (a % 3 = 1) ∧ (a % 5 = 2) → a = 37 :=
by
  intros
  sorry

end NUMINAMATH_GPT_smallest_whole_number_l1925_192500


namespace NUMINAMATH_GPT_possible_values_of_m_l1925_192568

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem possible_values_of_m (m : ℝ) : (∀ x, S x m → P x) ↔ (m = -1 ∨ m = 1 ∨ m = 3) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l1925_192568


namespace NUMINAMATH_GPT_tom_age_ratio_l1925_192565

variable (T M : ℕ)
variable (h1 : T = T) -- Tom's age is equal to the sum of the ages of his four children
variable (h2 : T - M = 3 * (T - 4 * M)) -- M years ago, Tom's age was three times the sum of his children's ages then

theorem tom_age_ratio : (T / M) = 11 / 2 := 
by
  sorry

end NUMINAMATH_GPT_tom_age_ratio_l1925_192565


namespace NUMINAMATH_GPT_Martha_time_spent_l1925_192553

theorem Martha_time_spent
  (x : ℕ)
  (h1 : 6 * x = 6 * x) -- Time spent on hold with Comcast is 6 times the time spent turning router off and on again
  (h2 : 3 * x = 3 * x) -- Time spent yelling at the customer service rep is half of time spent on hold, which is still 3x
  (h3 : x + 6 * x + 3 * x = 100) -- Total time spent is 100 minutes
  : x = 10 := 
by
  -- skip the proof steps
  sorry

end NUMINAMATH_GPT_Martha_time_spent_l1925_192553


namespace NUMINAMATH_GPT_evaluate_expression_l1925_192588

theorem evaluate_expression (x : ℕ) (h : x = 3) : 5^3 - 2^x * 3 + 4^2 = 117 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1925_192588


namespace NUMINAMATH_GPT_time_after_hours_l1925_192536

def current_time := 9
def total_hours := 2023
def clock_cycle := 12

theorem time_after_hours : (current_time + total_hours) % clock_cycle = 8 := by
  sorry

end NUMINAMATH_GPT_time_after_hours_l1925_192536


namespace NUMINAMATH_GPT_unattainable_y_value_l1925_192563

theorem unattainable_y_value (y : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : ¬ (y = -1 / 3) :=
by {
  -- The proof is omitted for now. 
  -- We're only constructing the outline with necessary imports and conditions.
  sorry
}

end NUMINAMATH_GPT_unattainable_y_value_l1925_192563


namespace NUMINAMATH_GPT_marbles_percentage_l1925_192570

def solid_color_other_than_yellow (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) : ℚ :=
  solid_color_percent - solid_yellow_percent

theorem marbles_percentage (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) :
  solid_color_percent = 90 / 100 →
  solid_yellow_percent = 5 / 100 →
  solid_color_other_than_yellow total_marbles solid_color_percent solid_yellow_percent = 85 / 100 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_marbles_percentage_l1925_192570


namespace NUMINAMATH_GPT_equal_share_each_shopper_l1925_192513

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_equal_share_each_shopper_l1925_192513


namespace NUMINAMATH_GPT_time_for_D_to_complete_job_l1925_192584

-- Definitions for conditions
def A_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 4

-- We need to find D_rate
def D_rate : ℚ := combined_rate - A_rate

-- Now we state the theorem
theorem time_for_D_to_complete_job :
  D_rate = 1 / 12 :=
by
  /-
  We want to show that given the conditions:
  1. A_rate = 1 / 6
  2. A_rate + D_rate = 1 / 4
  it results in D_rate = 1 / 12.
  -/
  sorry

end NUMINAMATH_GPT_time_for_D_to_complete_job_l1925_192584


namespace NUMINAMATH_GPT_triangle_problem_l1925_192527

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (hb : 0 < B ∧ B < Real.pi)
  (hc : 0 < C ∧ C < Real.pi)
  (ha : 0 < A ∧ A < Real.pi)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides : a > b)
  (h_perimeter : a + b + c = 20)
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_eq : a * (Real.sqrt 3 * Real.tan B - 1) = (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C)) :
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := sorry

end NUMINAMATH_GPT_triangle_problem_l1925_192527


namespace NUMINAMATH_GPT_ping_pong_ball_probability_l1925_192582

noncomputable def multiple_of_6_9_or_both_probability : ℚ :=
  let total_numbers := 72
  let multiples_of_6 := 12
  let multiples_of_9 := 8
  let multiples_of_both := 4
  (multiples_of_6 + multiples_of_9 - multiples_of_both) / total_numbers

theorem ping_pong_ball_probability :
  multiple_of_6_9_or_both_probability = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ping_pong_ball_probability_l1925_192582


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l1925_192538

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb 0.5 (x^2 + 2 * x - 3)

theorem monotonic_increasing_interval :
  ∀ x, f x = Real.logb 0.5 (x^2 + 2 * x - 3) → 
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < -3 ∧ x₂ < -3 → f x₁ ≤ f x₂) :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l1925_192538


namespace NUMINAMATH_GPT_regression_line_equation_l1925_192586

-- Define the conditions in the problem
def slope_of_regression_line : ℝ := 1.23
def center_of_sample_points : ℝ × ℝ := (4, 5)

-- The proof problem to show that the equation of the regression line is y = 1.23x + 0.08
theorem regression_line_equation :
  ∃ b : ℝ, (∀ x y : ℝ, (y = slope_of_regression_line * x + b) 
  → (4, 5) = (x, y)) → b = 0.08 :=
sorry

end NUMINAMATH_GPT_regression_line_equation_l1925_192586


namespace NUMINAMATH_GPT_minimum_a1_a2_sum_l1925_192518

theorem minimum_a1_a2_sum (a : ℕ → ℕ)
  (h : ∀ n ≥ 1, a (n + 2) = (a n + 2017) / (1 + a (n + 1)))
  (positive_terms : ∀ n, a n > 0) :
  a 1 + a 2 = 2018 :=
sorry

end NUMINAMATH_GPT_minimum_a1_a2_sum_l1925_192518


namespace NUMINAMATH_GPT_cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l1925_192564

theorem cos_beta_of_tan_alpha_and_sin_alpha_plus_beta 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 3) (h_sin_alpha_beta : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 10 / 10 := 
sorry

end NUMINAMATH_GPT_cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l1925_192564
