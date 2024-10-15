import Mathlib

namespace NUMINAMATH_GPT_false_conjunction_l640_64054

theorem false_conjunction (p q : Prop) (h : ¬(p ∧ q)) : ¬ (¬p ∧ ¬q) := sorry

end NUMINAMATH_GPT_false_conjunction_l640_64054


namespace NUMINAMATH_GPT_value_of_f_of_1_plus_g_of_2_l640_64009

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := x + 1

theorem value_of_f_of_1_plus_g_of_2 : f (1 + g 2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_of_1_plus_g_of_2_l640_64009


namespace NUMINAMATH_GPT_combined_weight_chihuahua_pitbull_greatdane_l640_64086

noncomputable def chihuahua_pitbull_greatdane_combined_weight (C P G : ℕ) : ℕ :=
  C + P + G

theorem combined_weight_chihuahua_pitbull_greatdane :
  ∀ (C P G : ℕ), P = 3 * C → G = 3 * P + 10 → G = 307 → chihuahua_pitbull_greatdane_combined_weight C P G = 439 :=
by
  intros C P G h1 h2 h3
  sorry

end NUMINAMATH_GPT_combined_weight_chihuahua_pitbull_greatdane_l640_64086


namespace NUMINAMATH_GPT_capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l640_64049

-- Step 1: Define the capacities of type A and B cars
def typeACarCapacity := 3
def typeBCarCapacity := 4

-- Step 2: Prove transportation capacities x and y
theorem capacities_correct (x y: ℕ) (h1 : 3 * x + 2 * y = 17) (h2 : 2 * x + 3 * y = 18) :
    x = typeACarCapacity ∧ y = typeBCarCapacity :=
by
  sorry

-- Step 3: Define a rental plan to transport 35 tons
theorem rental_plan_exists (a b : ℕ) : 3 * a + 4 * b = 35 :=
by
  sorry

-- Step 4: Prove the minimal cost solution
def typeACarCost := 300
def typeBCarCost := 320

def rentalCost (a b : ℕ) : ℕ := a * typeACarCost + b * typeBCarCost

theorem minimal_rental_cost_exists :
    ∃ a b, 3 * a + 4 * b = 35 ∧ rentalCost a b = 2860 :=
by
  sorry

end NUMINAMATH_GPT_capacities_correct_rental_plan_exists_minimal_rental_cost_exists_l640_64049


namespace NUMINAMATH_GPT_quadratic_eq_roots_quadratic_eq_range_l640_64033

theorem quadratic_eq_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0 ∧ x1 + 3 * x2 = 2 * m + 8) →
  (m = -1 ∨ m = -2) :=
sorry

theorem quadratic_eq_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0) →
  m ≤ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_eq_roots_quadratic_eq_range_l640_64033


namespace NUMINAMATH_GPT_banana_pie_angle_l640_64015

theorem banana_pie_angle
  (total_students : ℕ := 48)
  (chocolate_students : ℕ := 15)
  (apple_students : ℕ := 10)
  (blueberry_students : ℕ := 9)
  (remaining_students := total_students - (chocolate_students + apple_students + blueberry_students))
  (banana_students := remaining_students / 2) :
  (banana_students : ℝ) / total_students * 360 = 52.5 :=
by
  sorry

end NUMINAMATH_GPT_banana_pie_angle_l640_64015


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l640_64082

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 = 0 → 
  (k ≠ 0 ∧ ((-2)^2 - 4 * k * (-1) > 0))) ↔ (k > -1 ∧ k ≠ 0) := 
sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l640_64082


namespace NUMINAMATH_GPT_new_problem_l640_64031

theorem new_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 3 * y) / (3 * x - y) = 16 / 13 := 
by
  sorry

end NUMINAMATH_GPT_new_problem_l640_64031


namespace NUMINAMATH_GPT_functional_relationship_max_daily_profit_price_reduction_1200_profit_l640_64039

noncomputable def y : ℝ → ℝ := λ x => -2 * x^2 + 60 * x + 800

theorem functional_relationship :
  ∀ x : ℝ, y x = (40 - x) * (20 + 2 * x) := 
by
  intro x
  sorry

theorem max_daily_profit :
  y 15 = 1250 :=
by
  sorry

theorem price_reduction_1200_profit :
  ∀ x : ℝ, y x = 1200 → x = 10 ∨ x = 20 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_functional_relationship_max_daily_profit_price_reduction_1200_profit_l640_64039


namespace NUMINAMATH_GPT_roots_ellipse_condition_l640_64090

theorem roots_ellipse_condition (m n : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1^2 - m*x1 + n = 0 ∧ x2^2 - m*x2 + n = 0) 
  ↔ (m > 0 ∧ n > 0 ∧ m ≠ n) :=
sorry

end NUMINAMATH_GPT_roots_ellipse_condition_l640_64090


namespace NUMINAMATH_GPT_Montoya_budget_spent_on_food_l640_64066

-- Define the fractions spent on groceries and going out to eat
def groceries_fraction : ℝ := 0.6
def eating_out_fraction : ℝ := 0.2

-- Define the total fraction spent on food
def total_food_fraction (g : ℝ) (e : ℝ) : ℝ := g + e

-- The theorem to prove
theorem Montoya_budget_spent_on_food : total_food_fraction groceries_fraction eating_out_fraction = 0.8 := 
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_Montoya_budget_spent_on_food_l640_64066


namespace NUMINAMATH_GPT_family_ages_l640_64019

-- Define the conditions
variables (D M S F : ℕ)

-- Condition 1: In the year 2000, the mother was 4 times the daughter's age.
axiom mother_age : M = 4 * D

-- Condition 2: In the year 2000, the father was 6 times the son's age.
axiom father_age : F = 6 * S

-- Condition 3: The son is 1.5 times the age of the daughter.
axiom son_age_ratio : S = 3 * D / 2

-- Condition 4: In the year 2010, the father became twice the mother's age.
axiom father_mother_2010 : F + 10 = 2 * (M + 10)

-- Condition 5: The age gap between the mother and father has always been the same.
axiom age_gap_constant : F - M = (F + 10) - (M + 10)

-- Define the theorem
theorem family_ages :
  D = 10 ∧ S = 15 ∧ M = 40 ∧ F = 90 ∧ (F - M = 50) := sorry

end NUMINAMATH_GPT_family_ages_l640_64019


namespace NUMINAMATH_GPT_intersection_A_B_l640_64080

-- Defining sets A and B based on the given conditions.
def A : Set ℝ := {x | ∃ y, y = Real.log x ∧ x > 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Stating the theorem that A ∩ B = {x | 0 < x ∧ x < 3}.
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l640_64080


namespace NUMINAMATH_GPT_incorrect_judgment_l640_64042

variable (p q : Prop)
variable (hyp_p : p = (3 + 3 = 5))
variable (hyp_q : q = (5 > 2))

theorem incorrect_judgment : 
  (¬ (p ∧ q) ∧ ¬p) = false :=
by
  sorry

end NUMINAMATH_GPT_incorrect_judgment_l640_64042


namespace NUMINAMATH_GPT_add_three_digits_l640_64098

theorem add_three_digits (x : ℕ) :
  (x = 152 ∨ x = 656) →
  (523000 + x) % 504 = 0 := 
by
  sorry

end NUMINAMATH_GPT_add_three_digits_l640_64098


namespace NUMINAMATH_GPT_tylenol_mg_per_tablet_l640_64053

noncomputable def dose_intervals : ℕ := 3  -- Mark takes Tylenol 3 times
noncomputable def total_mg : ℕ := 3000     -- Total intake in milligrams
noncomputable def tablets_per_dose : ℕ := 2  -- Number of tablets per dose

noncomputable def tablet_mg : ℕ :=
  total_mg / dose_intervals / tablets_per_dose

theorem tylenol_mg_per_tablet : tablet_mg = 500 := by
  sorry

end NUMINAMATH_GPT_tylenol_mg_per_tablet_l640_64053


namespace NUMINAMATH_GPT_least_number_divisible_l640_64024

theorem least_number_divisible (n : ℕ) :
  ((∀ d ∈ [24, 32, 36, 54, 72, 81, 100], (n + 21) % d = 0) ↔ n = 64779) :=
sorry

end NUMINAMATH_GPT_least_number_divisible_l640_64024


namespace NUMINAMATH_GPT_two_pow_gt_square_for_n_ge_5_l640_64070

theorem two_pow_gt_square_for_n_ge_5 (n : ℕ) (hn : n ≥ 5) : 2^n > n^2 :=
sorry

end NUMINAMATH_GPT_two_pow_gt_square_for_n_ge_5_l640_64070


namespace NUMINAMATH_GPT_geom_seq_inc_condition_l640_64034

theorem geom_seq_inc_condition (a₁ a₂ q : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ = a₁ * q) :
  (a₁^2 < a₂^2) ↔ 
  (∀ n m : ℕ, n < m → (a₁ * q^n) < (a₁ * q^m) ∨ ((a₁ * q^n) = (a₁ * q^m) ∧ q = 1)) :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_inc_condition_l640_64034


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l640_64067

theorem necessary_but_not_sufficient (a c : ℝ) : 
  (c ≠ 0) → (∀ (x y : ℝ), ax^2 + y^2 = c → (c = 0 → false) ∧ (c ≠ 0 → (∃ x y : ℝ, ax^2 + y^2 = c))) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l640_64067


namespace NUMINAMATH_GPT_number_of_stadiums_to_visit_l640_64013

def average_cost_per_stadium : ℕ := 900
def annual_savings : ℕ := 1500
def years_saving : ℕ := 18

theorem number_of_stadiums_to_visit (c : ℕ) (s : ℕ) (n : ℕ) (h1 : c = average_cost_per_stadium) (h2 : s = annual_savings) (h3 : n = years_saving) : n * s / c = 30 := 
by 
  rw [h1, h2, h3]
  exact sorry

end NUMINAMATH_GPT_number_of_stadiums_to_visit_l640_64013


namespace NUMINAMATH_GPT_distance_from_point_to_y_axis_l640_64056

/-- Proof that the distance from point P(-4, 3) to the y-axis is 4. -/
theorem distance_from_point_to_y_axis {P : ℝ × ℝ} (hP : P = (-4, 3)) : |P.1| = 4 :=
by {
   -- The proof will depend on the properties of absolute value
   -- and the given condition about the coordinates of P.
   sorry
}

end NUMINAMATH_GPT_distance_from_point_to_y_axis_l640_64056


namespace NUMINAMATH_GPT_average_age_increase_l640_64097

theorem average_age_increase (average_age_students : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ)
                             (h1 : average_age_students = 26) (h2 : num_students = 25) (h3 : teacher_age = 52)
                             (h4 : new_avg_age = (650 + teacher_age) / (num_students + 1))
                             (h5 : 650 = average_age_students * num_students) :
  new_avg_age - average_age_students = 1 := 
by
  sorry

end NUMINAMATH_GPT_average_age_increase_l640_64097


namespace NUMINAMATH_GPT_fraction_apple_juice_in_mixture_l640_64000

theorem fraction_apple_juice_in_mixture :
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let fraction_juice_pitcher1 := (1 : ℚ) / 4
  let fraction_juice_pitcher2 := (3 : ℚ) / 8
  let apple_juice_pitcher1 := pitcher1_capacity * fraction_juice_pitcher1
  let apple_juice_pitcher2 := pitcher2_capacity * fraction_juice_pitcher2
  let total_apple_juice := apple_juice_pitcher1 + apple_juice_pitcher2
  let total_capacity := pitcher1_capacity + pitcher2_capacity
  (total_apple_juice / total_capacity = 31 / 104) :=
by
  sorry

end NUMINAMATH_GPT_fraction_apple_juice_in_mixture_l640_64000


namespace NUMINAMATH_GPT_min_value_at_2_l640_64012

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_at_2 : ∃ x : ℝ, f x = 2 :=
sorry

end NUMINAMATH_GPT_min_value_at_2_l640_64012


namespace NUMINAMATH_GPT_manolo_face_mask_time_l640_64044
variable (x : ℕ)
def time_to_make_mask_first_hour := x
def face_masks_made_first_hour := 60 / x
def face_masks_made_next_three_hours := 180 / 6
def total_face_masks_in_four_hours := face_masks_made_first_hour + face_masks_made_next_three_hours

theorem manolo_face_mask_time : 
  total_face_masks_in_four_hours x = 45 ↔ x = 4 := sorry

end NUMINAMATH_GPT_manolo_face_mask_time_l640_64044


namespace NUMINAMATH_GPT_g_five_eq_one_l640_64046

noncomputable def g : ℝ → ℝ := sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_nonzero : ∀ x : ℝ, g x ≠ 0

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_five_eq_one_l640_64046


namespace NUMINAMATH_GPT_total_cats_in_academy_l640_64058

theorem total_cats_in_academy (cats_jump cats_jump_fetch cats_fetch cats_fetch_spin cats_spin cats_jump_spin cats_all_three cats_none: ℕ)
  (h_jump: cats_jump = 60)
  (h_jump_fetch: cats_jump_fetch = 20)
  (h_fetch: cats_fetch = 35)
  (h_fetch_spin: cats_fetch_spin = 15)
  (h_spin: cats_spin = 40)
  (h_jump_spin: cats_jump_spin = 22)
  (h_all_three: cats_all_three = 11)
  (h_none: cats_none = 10) :
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none = 99 :=
by
  calc 
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none 
  = 11 + (20 - 11) + (15 - 11) + (22 - 11) + (60 - (9 + 11 + 11)) + (35 - (9 + 4 + 11)) + (40 - (11 + 4 + 11)) + 10 
  := by sorry
  _ = 99 := by sorry

end NUMINAMATH_GPT_total_cats_in_academy_l640_64058


namespace NUMINAMATH_GPT_renne_savings_ratio_l640_64064

theorem renne_savings_ratio (ME CV N : ℕ) (h_ME : ME = 4000) (h_CV : CV = 16000) (h_N : N = 8) :
  (CV / N : ℕ) / ME = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_renne_savings_ratio_l640_64064


namespace NUMINAMATH_GPT_smallest_number_of_three_l640_64030

theorem smallest_number_of_three (x : ℕ) (h1 : x = 18)
  (h2 : ∀ y z : ℕ, y = 4 * x ∧ z = 2 * y)
  (h3 : (x + 4 * x + 8 * x) / 3 = 78)
  : x = 18 := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_three_l640_64030


namespace NUMINAMATH_GPT_system_of_equations_unique_solution_l640_64032

theorem system_of_equations_unique_solution :
  (∃ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7) →
  (∀ (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + y = 7 →
    x = 26 / 5 ∧ y = 9 / 5) := 
by {
  -- Proof to be provided
  sorry
}

end NUMINAMATH_GPT_system_of_equations_unique_solution_l640_64032


namespace NUMINAMATH_GPT_ratio_of_area_to_perimeter_l640_64096

noncomputable def altitude_of_equilateral_triangle (s : ℝ) : ℝ :=
  s * (Real.sqrt 3 / 2)

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  1 / 2 * s * altitude_of_equilateral_triangle s

noncomputable def perimeter_of_equilateral_triangle (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_area_to_perimeter (s : ℝ) (h : s = 10) :
    (area_of_equilateral_triangle s) / (perimeter_of_equilateral_triangle s) = 5 * Real.sqrt 3 / 6 :=
  by
  rw [h]
  sorry

end NUMINAMATH_GPT_ratio_of_area_to_perimeter_l640_64096


namespace NUMINAMATH_GPT_local_min_at_neg_one_l640_64035

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_min_at_neg_one : 
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≥ f (-1) := by
  sorry

end NUMINAMATH_GPT_local_min_at_neg_one_l640_64035


namespace NUMINAMATH_GPT_seven_digit_palindromes_l640_64026

def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

theorem seven_digit_palindromes : 
  (∃ l : List ℕ, l = [1, 1, 4, 4, 4, 6, 6] ∧ 
  ∃ pl : List ℕ, pl.length = 7 ∧ is_palindrome pl ∧ 
  ∀ d, d ∈ pl → d ∈ l) →
  ∃! n, n = 12 :=
by
  sorry

end NUMINAMATH_GPT_seven_digit_palindromes_l640_64026


namespace NUMINAMATH_GPT_school_students_count_l640_64038

def students_in_school (c n : ℕ) : ℕ := n * c

theorem school_students_count
  (c n : ℕ)
  (h1 : n * c = (n - 6) * (c + 5))
  (h2 : n * c = (n - 16) * (c + 20)) :
  students_in_school c n = 900 :=
by
  sorry

end NUMINAMATH_GPT_school_students_count_l640_64038


namespace NUMINAMATH_GPT_cos_diff_half_l640_64050

theorem cos_diff_half (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1 / 2)
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) :
  Real.cos (α - β) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_diff_half_l640_64050


namespace NUMINAMATH_GPT_leak_empty_time_l640_64020

variable (inlet_rate : ℕ := 6) -- litres per minute
variable (total_capacity : ℕ := 12960) -- litres
variable (empty_time_with_inlet_open : ℕ := 12) -- hours

def inlet_rate_per_hour := inlet_rate * 60 -- litres per hour
def net_emptying_rate := total_capacity / empty_time_with_inlet_open -- litres per hour
def leak_rate := net_emptying_rate + inlet_rate_per_hour -- litres per hour

theorem leak_empty_time : total_capacity / leak_rate = 9 := by
  sorry

end NUMINAMATH_GPT_leak_empty_time_l640_64020


namespace NUMINAMATH_GPT_total_distance_of_trail_l640_64068

theorem total_distance_of_trail (a b c d e : ℕ) 
    (h1 : a + b + c = 30) 
    (h2 : b + d = 30) 
    (h3 : d + e = 28) 
    (h4 : a + d = 34) : 
    a + b + c + d + e = 58 := 
sorry

end NUMINAMATH_GPT_total_distance_of_trail_l640_64068


namespace NUMINAMATH_GPT_arithmetic_series_sum_l640_64093

theorem arithmetic_series_sum (k : ℤ) : 
  let a₁ := k^2 + k + 1 
  let n := 2 * k + 3 
  let d := 1 
  let aₙ := a₁ + (n - 1) * d 
  let S_n := n / 2 * (a₁ + aₙ)
  S_n = 2 * k^3 + 7 * k^2 + 10 * k + 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_series_sum_l640_64093


namespace NUMINAMATH_GPT_sophie_hours_needed_l640_64073

-- Sophie needs 206 hours to finish the analysis of all bones.
theorem sophie_hours_needed (num_bones : ℕ) (time_per_bone : ℕ) (total_hours : ℕ) (h1 : num_bones = 206) (h2 : time_per_bone = 1) : 
  total_hours = num_bones * time_per_bone :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_sophie_hours_needed_l640_64073


namespace NUMINAMATH_GPT_joe_average_speed_l640_64051

noncomputable def average_speed (total_distance total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem joe_average_speed :
  let distance1 := 420
  let speed1 := 60
  let distance2 := 120
  let speed2 := 40
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  average_speed total_distance total_time = 54 := by
sorry

end NUMINAMATH_GPT_joe_average_speed_l640_64051


namespace NUMINAMATH_GPT_fx_solution_l640_64089

theorem fx_solution (f : ℝ → ℝ) (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1)
  (h_assumption : f (1 / x) = x / (1 - x)) : f x = 1 / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fx_solution_l640_64089


namespace NUMINAMATH_GPT_solve_for_t_l640_64045

theorem solve_for_t (s t : ℝ) (h1 : 12 * s + 8 * t = 160) (h2 : s = t^2 + 2) :
  t = (Real.sqrt 103 - 1) / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_t_l640_64045


namespace NUMINAMATH_GPT_smallest_n_7770_l640_64077

theorem smallest_n_7770 (n : ℕ) 
  (h1 : ∀ d ∈ n.digits 10, d = 0 ∨ d = 7)
  (h2 : 15 ∣ n) : 
  n = 7770 := 
sorry

end NUMINAMATH_GPT_smallest_n_7770_l640_64077


namespace NUMINAMATH_GPT_P_plus_Q_l640_64037

theorem P_plus_Q (P Q : ℝ) (h : ∀ x, x ≠ 3 → (P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3))) : P + Q = 46 :=
sorry

end NUMINAMATH_GPT_P_plus_Q_l640_64037


namespace NUMINAMATH_GPT_bobby_candy_left_l640_64007

def initial_candy := 22
def eaten_candy1 := 9
def eaten_candy2 := 5

theorem bobby_candy_left : initial_candy - eaten_candy1 - eaten_candy2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_bobby_candy_left_l640_64007


namespace NUMINAMATH_GPT_inequality_proof_l640_64094

theorem inequality_proof
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 1) :
  (a^2 + b^2 + c^2) * ((a / (b + c)) + (b / (a + c)) + (c / (a + b))) ≥ 1/2 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l640_64094


namespace NUMINAMATH_GPT_range_of_a_l640_64059

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x + 3

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Ioo (a-1) (a+1), 4*x - 1/x = 0) ↔ 1 ≤ a ∧ a < 3/2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l640_64059


namespace NUMINAMATH_GPT_math_problem_l640_64083

open Real

theorem math_problem
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a - b + c = 0) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
   a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
   b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 := by
  sorry

end NUMINAMATH_GPT_math_problem_l640_64083


namespace NUMINAMATH_GPT_ceil_floor_diff_l640_64071

theorem ceil_floor_diff (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ - ⌊x⌋ = 1 := 
by 
  sorry

end NUMINAMATH_GPT_ceil_floor_diff_l640_64071


namespace NUMINAMATH_GPT_stephen_speed_second_third_l640_64025

theorem stephen_speed_second_third
  (first_third_speed : ℝ)
  (last_third_speed : ℝ)
  (total_distance : ℝ)
  (travel_time : ℝ)
  (time_in_hours : ℝ)
  (h1 : first_third_speed = 16)
  (h2 : last_third_speed = 20)
  (h3 : total_distance = 12)
  (h4 : travel_time = 15)
  (h5 : time_in_hours = travel_time / 60) :
  time_in_hours * (total_distance - (first_third_speed * time_in_hours + last_third_speed * time_in_hours)) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_stephen_speed_second_third_l640_64025


namespace NUMINAMATH_GPT_length_of_shortest_side_30_60_90_l640_64052

theorem length_of_shortest_side_30_60_90 (x : ℝ) : 
  (∃ x : ℝ, (2 * x = 15)) → x = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_shortest_side_30_60_90_l640_64052


namespace NUMINAMATH_GPT_correct_representations_l640_64029

open Set

theorem correct_representations : 
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  (¬S1 ∧ ¬S2 ∧ S3 ∧ S4) :=
by
  let S1 := {2, 3} ≠ ({3, 2} : Set ℕ)
  let S2 := ({(x, y) | x + y = 1} : Set (ℕ × ℕ)) = {y | ∃ x, x + y = 1}
  let S3 := ({x | x > 1} : Set ℕ) = {y | y > 1}
  let S4 := ({x | ∃ y, x + y = 1} : Set ℕ) = {y | ∃ x, x + y = 1}
  exact sorry

end NUMINAMATH_GPT_correct_representations_l640_64029


namespace NUMINAMATH_GPT_find_page_added_twice_l640_64036

theorem find_page_added_twice (m p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ m) (h3 : (m * (m + 1)) / 2 + p = 2550) : p = 6 :=
sorry

end NUMINAMATH_GPT_find_page_added_twice_l640_64036


namespace NUMINAMATH_GPT_corresponding_angles_equal_l640_64092

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end NUMINAMATH_GPT_corresponding_angles_equal_l640_64092


namespace NUMINAMATH_GPT_tom_purchases_mangoes_l640_64043

theorem tom_purchases_mangoes (m : ℕ) (h1 : 8 * 70 + m * 65 = 1145) : m = 9 :=
by
  sorry

end NUMINAMATH_GPT_tom_purchases_mangoes_l640_64043


namespace NUMINAMATH_GPT_triangle_perimeter_l640_64018

/-
  A square piece of paper with side length 2 has vertices A, B, C, and D. 
  The paper is folded such that vertex A meets edge BC at point A', 
  and A'C = 1/2. Prove that the perimeter of triangle A'BD is (3 + sqrt(17))/2 + 2sqrt(2).
-/
theorem triangle_perimeter
  (A B C D A' : ℝ × ℝ)
  (side_length : ℝ)
  (BC_length : ℝ)
  (CA'_length : ℝ)
  (BA'_length : ℝ)
  (BD_length : ℝ)
  (DA'_length : ℝ)
  (perimeter_correct : ℝ) :
  side_length = 2 ∧
  BC_length = 2 ∧
  CA'_length = 1/2 ∧
  BA'_length = 3/2 ∧
  BD_length = 2 * Real.sqrt 2 ∧
  DA'_length = Real.sqrt 17 / 2 →
  perimeter_correct = (3 + Real.sqrt 17) / 2 + 2 * Real.sqrt 2 →
  (side_length ≠ 0 ∧ BC_length = side_length ∧ 
   CA'_length ≠ 0 ∧ BA'_length ≠ 0 ∧ 
   BD_length ≠ 0 ∧ DA'_length ≠ 0) →
  (BA'_length + BD_length + DA'_length = perimeter_correct) :=
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l640_64018


namespace NUMINAMATH_GPT_inequality_add_one_l640_64022

variable {α : Type*} [LinearOrderedField α]

theorem inequality_add_one {a b : α} (h : a > b) : a + 1 > b + 1 :=
sorry

end NUMINAMATH_GPT_inequality_add_one_l640_64022


namespace NUMINAMATH_GPT_min_a_squared_plus_b_squared_l640_64011

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_a_squared_plus_b_squared_l640_64011


namespace NUMINAMATH_GPT_problem_inequality_l640_64072

theorem problem_inequality (a b x y : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l640_64072


namespace NUMINAMATH_GPT_faster_train_cross_time_l640_64084

noncomputable def time_to_cross (speed_fast_kmph : ℝ) (speed_slow_kmph : ℝ) (length_fast_m : ℝ) : ℝ :=
  let speed_diff_kmph := speed_fast_kmph - speed_slow_kmph
  let speed_diff_mps := (speed_diff_kmph * 1000) / 3600
  length_fast_m / speed_diff_mps

theorem faster_train_cross_time :
  time_to_cross 72 36 120 = 12 :=
by
  sorry

end NUMINAMATH_GPT_faster_train_cross_time_l640_64084


namespace NUMINAMATH_GPT_chessboard_max_squares_l640_64061

def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

theorem chessboard_max_squares (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) : max_squares 1000 1000 = 1998 := 
by
  -- This is the theorem statement representing the maximum number of squares chosen
  -- in a 1000 x 1000 chessboard without having exactly three of them with two in the same row
  -- and two in the same column.
  sorry

end NUMINAMATH_GPT_chessboard_max_squares_l640_64061


namespace NUMINAMATH_GPT_triangle_angle_sum_33_75_l640_64074

theorem triangle_angle_sum_33_75 (x : ℝ) 
  (h₁ : 45 + 3 * x + x = 180) : 
  x = 33.75 :=
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_33_75_l640_64074


namespace NUMINAMATH_GPT_largest_fraction_l640_64063

variable {a b c d e f g h : ℝ}
variable {w x y z : ℝ}

/-- Given real numbers w, x, y, z such that w < x < y < z,
    the fraction z/w represents the largest value among the given fractions. -/
theorem largest_fraction (hwx : w < x) (hxy : x < y) (hyz : y < z) :
  (z / w) > (x / w) ∧ (z / w) > (y / x) ∧ (z / w) > (y / w) ∧ (z / w) > (z / x) :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l640_64063


namespace NUMINAMATH_GPT_find_m_and_star_l640_64078

-- Definitions from conditions
def star (x y m : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

-- Given conditions
def given_star (x y : ℚ) (m : ℚ) : Prop := star x y m = 2 / 5

-- Target: Proving m = 1 and 2 * 6 = 6 / 7 given the conditions
theorem find_m_and_star :
  ∀ m : ℚ, 
  (given_star 1 2 m) → 
  (m = 1 ∧ star 2 6 m = 6 / 7) := 
sorry

end NUMINAMATH_GPT_find_m_and_star_l640_64078


namespace NUMINAMATH_GPT_sin_15_add_sin_75_l640_64060

theorem sin_15_add_sin_75 : 
  Real.sin (15 * Real.pi / 180) + Real.sin (75 * Real.pi / 180) = Real.sqrt 6 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_15_add_sin_75_l640_64060


namespace NUMINAMATH_GPT_birds_on_branch_l640_64099

theorem birds_on_branch (initial_parrots remaining_parrots remaining_crows total_birds : ℕ) (h₁ : initial_parrots = 7) (h₂ : remaining_parrots = 2) (h₃ : remaining_crows = 1) (h₄ : initial_parrots - remaining_parrots = total_birds - remaining_crows - initial_parrots) : total_birds = 13 :=
sorry

end NUMINAMATH_GPT_birds_on_branch_l640_64099


namespace NUMINAMATH_GPT_point_not_on_graph_and_others_on_l640_64010

theorem point_not_on_graph_and_others_on (y : ℝ → ℝ) (h₁ : ∀ x, y x = x / (x - 1))
  : ¬ (1 = (1 : ℝ) / ((1 : ℝ) - 1)) 
  ∧ (2 = (2 : ℝ) / ((2 : ℝ) - 1)) 
  ∧ ((-1 : ℝ) = (1/2 : ℝ) / ((1/2 : ℝ) - 1)) 
  ∧ (0 = (0 : ℝ) / ((0 : ℝ) - 1)) 
  ∧ (3/2 = (3 : ℝ) / ((3 : ℝ) - 1)) := 
sorry

end NUMINAMATH_GPT_point_not_on_graph_and_others_on_l640_64010


namespace NUMINAMATH_GPT_athleteA_time_to_complete_race_l640_64008

theorem athleteA_time_to_complete_race
    (v : ℝ)
    (t : ℝ)
    (h1 : v = 1000 / t)
    (h2 : v = 948 / (t + 18)) :
    t = 18000 / 52 := by
  sorry

end NUMINAMATH_GPT_athleteA_time_to_complete_race_l640_64008


namespace NUMINAMATH_GPT_sqrt_12_same_type_sqrt_3_l640_64028

-- We define that two square roots are of the same type if one is a multiple of the other
def same_type (a b : ℝ) : Prop := ∃ k : ℝ, b = k * a

-- We need to show that sqrt(12) is of the same type as sqrt(3), and check options
theorem sqrt_12_same_type_sqrt_3 : same_type (Real.sqrt 3) (Real.sqrt 12) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 8) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 18) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 6) :=
by
  sorry -- Proof is omitted


end NUMINAMATH_GPT_sqrt_12_same_type_sqrt_3_l640_64028


namespace NUMINAMATH_GPT_lucy_reads_sixty_pages_l640_64087

-- Define the number of pages Carter, Lucy, and Oliver can read in an hour.
def pages_carter : ℕ := 30
def pages_oliver : ℕ := 40

-- Carter reads half as many pages as Lucy.
def reads_half_as_much_as (a b : ℕ) : Prop := a = b / 2

-- Lucy reads more pages than Oliver.
def reads_more_than (a b : ℕ) : Prop := a > b

-- The goal is to show that Lucy can read 60 pages in an hour.
theorem lucy_reads_sixty_pages (pages_lucy : ℕ) (h1 : reads_half_as_much_as pages_carter pages_lucy)
  (h2 : reads_more_than pages_lucy pages_oliver) : pages_lucy = 60 :=
sorry

end NUMINAMATH_GPT_lucy_reads_sixty_pages_l640_64087


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_l640_64065

open Set Real

theorem problem1_part1 (a : ℝ) (h1: a = 5) :
  let A := { x : ℝ | (x - 6) * (x - 2 * a - 5) > 0 }
  let B := { x : ℝ | (a ^ 2 + 2 - x) * (2 * a - x) < 0 }
  A ∩ B = { x | 15 < x ∧ x < 27 } := sorry

theorem problem1_part2 (a : ℝ) (h2: a > 1 / 2) :
  let A := { x : ℝ | x < 6 ∨ x > 2 * a + 5 }
  let B := { x : ℝ | 2 * a < x ∧ x < a ^ 2 + 2 }
  (∀ x, x ∈ A → x ∈ B) ∧ ¬ (∀ x, x ∈ B → x ∈ A) → (1 / 2 < a ∧ a ≤ 2) := sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_l640_64065


namespace NUMINAMATH_GPT_area_of_side_face_of_box_l640_64021

theorem area_of_side_face_of_box:
  ∃ (l w h : ℝ), (w * h = (1/2) * (l * w)) ∧
                 (l * w = 1.5 * (l * h)) ∧
                 (l * w * h = 3000) ∧
                 ((l * h) = 200) :=
sorry

end NUMINAMATH_GPT_area_of_side_face_of_box_l640_64021


namespace NUMINAMATH_GPT_problem_solution_l640_64014

def arithmetic_sequence (a_1 : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a_1 + (n - 1) * d

def sum_of_terms (a_1 : ℕ) (a_n : ℕ) (n : ℕ) : ℕ :=
  n * (a_1 + a_n) / 2

theorem problem_solution 
  (a_1 : ℕ) (d : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a_1 = 2)
  (h2 : S_2 = arithmetic_sequence a_1 d 3):
  a_2 = 4 ∧ S_10 = 110 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l640_64014


namespace NUMINAMATH_GPT_gopi_servant_salary_l640_64006

theorem gopi_servant_salary (S : ℝ) (h1 : 9 / 12 * S + 110 = 150) : S = 200 :=
by
  sorry

end NUMINAMATH_GPT_gopi_servant_salary_l640_64006


namespace NUMINAMATH_GPT_sum_of_specific_terms_l640_64062

theorem sum_of_specific_terms 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h1 : S 3 = 9) 
  (h2 : S 6 = 36) 
  (h3 : ∀ n, S n = n * (a 1) + d * n * (n - 1) / 2) :
  a 7 + a 8 + a 9 = 45 := 
sorry

end NUMINAMATH_GPT_sum_of_specific_terms_l640_64062


namespace NUMINAMATH_GPT_value_two_sd_below_mean_l640_64041

theorem value_two_sd_below_mean (mean : ℝ) (std_dev : ℝ) (h_mean : mean = 17.5) (h_std_dev : std_dev = 2.5) : 
  mean - 2 * std_dev = 12.5 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_value_two_sd_below_mean_l640_64041


namespace NUMINAMATH_GPT_remainder_102_104_plus_6_div_9_l640_64081

theorem remainder_102_104_plus_6_div_9 :
  ((102 * 104 + 6) % 9) = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_102_104_plus_6_div_9_l640_64081


namespace NUMINAMATH_GPT_m_minus_n_eq_six_l640_64040

theorem m_minus_n_eq_six (m n : ℝ) (h : ∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) : m - n = 6 := by
  sorry

end NUMINAMATH_GPT_m_minus_n_eq_six_l640_64040


namespace NUMINAMATH_GPT_area_diff_of_rectangle_l640_64079

theorem area_diff_of_rectangle (a : ℝ) : 
  let length_increased := 1.40 * a
  let breadth_increased := 1.30 * a
  let original_area := a * a
  let new_area := length_increased * breadth_increased
  (new_area - original_area) = 0.82 * (a * a) :=
by 
sorry

end NUMINAMATH_GPT_area_diff_of_rectangle_l640_64079


namespace NUMINAMATH_GPT_length_of_chord_EF_l640_64027

theorem length_of_chord_EF 
  (rO rN rP : ℝ)
  (AB BC CD : ℝ)
  (AG_EF_intersec_E AG_EF_intersec_F : ℝ)
  (EF : ℝ)
  (cond1 : rO = 10)
  (cond2 : rN = 20)
  (cond3 : rP = 30)
  (cond4 : AB = 2 * rO)
  (cond5 : BC = 2 * rN)
  (cond6 : CD = 2 * rP)
  (cond7 : EF = 6 * Real.sqrt (24 + 2/3)) :
  EF = 6 * Real.sqrt 24.6666 := sorry

end NUMINAMATH_GPT_length_of_chord_EF_l640_64027


namespace NUMINAMATH_GPT_Jose_played_football_l640_64002

theorem Jose_played_football :
  ∀ (total_hours : ℝ) (basketball_minutes : ℕ) (minutes_per_hour : ℕ), total_hours = 1.5 → basketball_minutes = 60 →
  (total_hours * minutes_per_hour - basketball_minutes = 30) :=
by
  intros total_hours basketball_minutes minutes_per_hour h1 h2
  sorry

end NUMINAMATH_GPT_Jose_played_football_l640_64002


namespace NUMINAMATH_GPT_strawberries_in_each_handful_l640_64048

theorem strawberries_in_each_handful (x : ℕ) (h : (x - 1) * (75 / x) = 60) : x = 5 :=
sorry

end NUMINAMATH_GPT_strawberries_in_each_handful_l640_64048


namespace NUMINAMATH_GPT_sum_of_three_consecutive_even_l640_64016

theorem sum_of_three_consecutive_even (a1 a2 a3 : ℤ) (h1 : a1 % 2 = 0) (h2 : a2 = a1 + 2) (h3 : a3 = a1 + 4) (h4 : a1 + a3 = 128) : a1 + a2 + a3 = 192 :=
sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_even_l640_64016


namespace NUMINAMATH_GPT_ab_leq_one_l640_64088

theorem ab_leq_one (a b : ℝ) (h : (a + b) * (a + b + a + b) = 9) : a * b ≤ 1 := 
  sorry

end NUMINAMATH_GPT_ab_leq_one_l640_64088


namespace NUMINAMATH_GPT_positive_solution_sqrt_a_sub_b_l640_64075

theorem positive_solution_sqrt_a_sub_b (a b : ℕ) (x : ℝ) 
  (h_eq : x^2 + 14 * x = 32) 
  (h_form : x = Real.sqrt a - b) 
  (h_pos_nat : a > 0 ∧ b > 0) : 
  a + b = 88 := 
by
  sorry

end NUMINAMATH_GPT_positive_solution_sqrt_a_sub_b_l640_64075


namespace NUMINAMATH_GPT_initial_pigeons_l640_64085

theorem initial_pigeons (n : ℕ) (h : n + 1 = 2) : n = 1 := 
sorry

end NUMINAMATH_GPT_initial_pigeons_l640_64085


namespace NUMINAMATH_GPT_calculate_expression_l640_64076

theorem calculate_expression :
  |-2*Real.sqrt 3| - (1 - Real.pi)^0 + 2*Real.cos (Real.pi / 6) + (1 / 4)^(-1 : ℤ) = 3 * Real.sqrt 3 + 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l640_64076


namespace NUMINAMATH_GPT_distance_covered_at_40_kmph_l640_64001

theorem distance_covered_at_40_kmph (x : ℝ) (h : 0 ≤ x ∧ x ≤ 250) 
  (total_distance : x + (250 - x) = 250) 
  (total_time : x / 40 + (250 - x) / 60 = 5.5) : 
  x = 160 :=
sorry

end NUMINAMATH_GPT_distance_covered_at_40_kmph_l640_64001


namespace NUMINAMATH_GPT_jackie_eligible_for_free_shipping_l640_64017

def shampoo_cost : ℝ := 2 * 12.50
def conditioner_cost : ℝ := 3 * 15.00
def face_cream_cost : ℝ := 20.00  -- Considering the buy-one-get-one-free deal

def subtotal : ℝ := shampoo_cost + conditioner_cost + face_cream_cost
def discount : ℝ := 0.10 * subtotal
def total_after_discount : ℝ := subtotal - discount

theorem jackie_eligible_for_free_shipping : total_after_discount >= 75 := by
  sorry

end NUMINAMATH_GPT_jackie_eligible_for_free_shipping_l640_64017


namespace NUMINAMATH_GPT_parallel_line_through_point_l640_64095

theorem parallel_line_through_point (x y c : ℝ) (h1 : c = -1) :
  ∃ c, (x-2*y+c = 0 ∧ x = 1 ∧ y = 0) ∧ ∃ k b, k = 1 ∧ b = -2 ∧ k*x-2*y+b=0 → c = -1 := by
  sorry

end NUMINAMATH_GPT_parallel_line_through_point_l640_64095


namespace NUMINAMATH_GPT_original_number_is_fraction_l640_64057

theorem original_number_is_fraction (x : ℚ) (h : 1 + (1 / x) = 9 / 4) : x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_fraction_l640_64057


namespace NUMINAMATH_GPT_jack_walking_rate_l640_64005

variables (distance : ℝ) (time_hours : ℝ)
#check distance  -- ℝ (real number)
#check time_hours  -- ℝ (real number)

-- Define the conditions
def jack_distance : Prop := distance = 9
def jack_time : Prop := time_hours = 1 + 15 / 60

-- Define the statement to prove
theorem jack_walking_rate (h1 : jack_distance distance) (h2 : jack_time time_hours) :
  (distance / time_hours) = 7.2 :=
sorry

end NUMINAMATH_GPT_jack_walking_rate_l640_64005


namespace NUMINAMATH_GPT_solve_xy_l640_64003

theorem solve_xy (x y : ℝ) (hx: x ≠ 0) (hxy: x + y ≠ 0) : 
  (x + y) / x = 2 * y / (x + y) + 1 → (x = y ∨ x = -3 * y) := 
by 
  intros h 
  sorry

end NUMINAMATH_GPT_solve_xy_l640_64003


namespace NUMINAMATH_GPT_relationship_of_AT_l640_64055

def S : ℝ := 300
def PC : ℝ := S + 500
def total_cost : ℝ := 2200

theorem relationship_of_AT (AT : ℝ) 
  (h1: S + PC + AT = total_cost) : 
  AT = S + PC - 400 :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_AT_l640_64055


namespace NUMINAMATH_GPT_range_of_a_l640_64004

-- Defining the function f : ℝ → ℝ
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x + a * Real.log x

-- Main theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (f a x1 = 0 ∧ f a x2 = 0)) → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l640_64004


namespace NUMINAMATH_GPT_complement_of_A_in_U_l640_64023

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}
def complement : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement = {1, 3, 5} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l640_64023


namespace NUMINAMATH_GPT_slope_of_line_eq_neg_four_thirds_l640_64069

variable {x y : ℝ}
variable (p₁ p₂ : ℝ × ℝ) (h₁ : 3 / p₁.1 + 4 / p₁.2 = 0) (h₂ : 3 / p₂.1 + 4 / p₂.2 = 0)

theorem slope_of_line_eq_neg_four_thirds 
  (hneq : p₁.1 ≠ p₂.1):
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1) = -4 / 3 := 
sorry

end NUMINAMATH_GPT_slope_of_line_eq_neg_four_thirds_l640_64069


namespace NUMINAMATH_GPT_max_pieces_in_8x8_grid_l640_64091

theorem max_pieces_in_8x8_grid : 
  ∃ m n : ℕ, (m = 8) ∧ (n = 9) ∧ 
  (∀ H V : ℕ, (H ≤ n) → (V ≤ n) → 
   (H + V + 1 ≤ 16)) := sorry

end NUMINAMATH_GPT_max_pieces_in_8x8_grid_l640_64091


namespace NUMINAMATH_GPT_net_gain_is_88837_50_l640_64047

def initial_home_value : ℝ := 500000
def first_sale_price : ℝ := 1.15 * initial_home_value
def first_purchase_price : ℝ := 0.95 * first_sale_price
def second_sale_price : ℝ := 1.1 * first_purchase_price
def second_purchase_price : ℝ := 0.9 * second_sale_price

def total_sales : ℝ := first_sale_price + second_sale_price
def total_purchases : ℝ := first_purchase_price + second_purchase_price
def net_gain_for_A : ℝ := total_sales - total_purchases

theorem net_gain_is_88837_50 : net_gain_for_A = 88837.50 := by
  -- proof steps would go here, but they are omitted per instructions
  sorry

end NUMINAMATH_GPT_net_gain_is_88837_50_l640_64047
