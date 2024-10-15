import Mathlib

namespace NUMINAMATH_GPT_total_pamphlets_correct_l1658_165895

def mike_initial_speed := 600
def mike_initial_hours := 9
def mike_break_hours := 2
def leo_relative_hours := 1 / 3
def leo_relative_speed := 2

def total_pamphlets (mike_initial_speed mike_initial_hours mike_break_hours leo_relative_hours leo_relative_speed : ℕ) : ℕ :=
  let mike_pamphlets_before_break := mike_initial_speed * mike_initial_hours
  let mike_speed_after_break := mike_initial_speed / 3
  let mike_pamphlets_after_break := mike_speed_after_break * mike_break_hours
  let total_mike_pamphlets := mike_pamphlets_before_break + mike_pamphlets_after_break

  let leo_hours := mike_initial_hours * leo_relative_hours
  let leo_speed := mike_initial_speed * leo_relative_speed
  let leo_pamphlets := leo_hours * leo_speed

  total_mike_pamphlets + leo_pamphlets

theorem total_pamphlets_correct : total_pamphlets 600 9 2 (1 / 3 : ℕ) 2 = 9400 := 
by 
  sorry

end NUMINAMATH_GPT_total_pamphlets_correct_l1658_165895


namespace NUMINAMATH_GPT_zoe_total_songs_l1658_165843

-- Define the number of country albums Zoe bought
def country_albums : Nat := 3

-- Define the number of pop albums Zoe bought
def pop_albums : Nat := 5

-- Define the number of songs per album
def songs_per_album : Nat := 3

-- Define the total number of albums
def total_albums : Nat := country_albums + pop_albums

-- Define the total number of songs
def total_songs : Nat := total_albums * songs_per_album

-- Theorem statement asserting the total number of songs
theorem zoe_total_songs : total_songs = 24 := by
  -- Proof will be inserted here (currently skipped)
  sorry

end NUMINAMATH_GPT_zoe_total_songs_l1658_165843


namespace NUMINAMATH_GPT_Carrie_pays_94_l1658_165893

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end NUMINAMATH_GPT_Carrie_pays_94_l1658_165893


namespace NUMINAMATH_GPT_data_plan_comparison_l1658_165828

theorem data_plan_comparison : ∃ (m : ℕ), 500 < m :=
by
  let cost_plan_x (m : ℕ) : ℕ := 15 * m
  let cost_plan_y (m : ℕ) : ℕ := 2500 + 10 * m
  use 501
  have h : 500 < 501 := by norm_num
  exact h

end NUMINAMATH_GPT_data_plan_comparison_l1658_165828


namespace NUMINAMATH_GPT_solve_for_x_l1658_165861

theorem solve_for_x (x : ℝ) (h : 8 / x + 6 = 8) : x = 4 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1658_165861


namespace NUMINAMATH_GPT_gcd_4830_3289_l1658_165811

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 :=
by sorry

end NUMINAMATH_GPT_gcd_4830_3289_l1658_165811


namespace NUMINAMATH_GPT_sum_possible_values_q_l1658_165878

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end NUMINAMATH_GPT_sum_possible_values_q_l1658_165878


namespace NUMINAMATH_GPT_condition_is_sufficient_but_not_necessary_l1658_165824

variable (P Q : Prop)

theorem condition_is_sufficient_but_not_necessary :
    (P → Q) ∧ ¬(Q → P) :=
sorry

end NUMINAMATH_GPT_condition_is_sufficient_but_not_necessary_l1658_165824


namespace NUMINAMATH_GPT_max_value_quadratic_l1658_165894

theorem max_value_quadratic : ∃ x : ℝ, -9 * x^2 + 27 * x + 15 = 35.25 :=
sorry

end NUMINAMATH_GPT_max_value_quadratic_l1658_165894


namespace NUMINAMATH_GPT_intersection_A_B_l1658_165834

open Set

def A : Set ℤ := {x : ℤ | ∃ y : ℝ, y = Real.sqrt (1 - (x : ℝ)^2)}
def B : Set ℤ := {y : ℤ | ∃ x : ℤ, x ∈ A ∧ y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_A_B_l1658_165834


namespace NUMINAMATH_GPT_ratio_average_speed_l1658_165838

-- Define the conditions based on given distances and times
def distanceAB : ℕ := 600
def timeAB : ℕ := 3

def distanceBD : ℕ := 540
def timeBD : ℕ := 2

def distanceAC : ℕ := 460
def timeAC : ℕ := 4

def distanceCE : ℕ := 380
def timeCE : ℕ := 3

-- Define the total distances and times for Eddy and Freddy
def distanceEddy : ℕ := distanceAB + distanceBD
def timeEddy : ℕ := timeAB + timeBD

def distanceFreddy : ℕ := distanceAC + distanceCE
def timeFreddy : ℕ := timeAC + timeCE

-- Define the average speeds for Eddy and Freddy
def averageSpeedEddy : ℚ := distanceEddy / timeEddy
def averageSpeedFreddy : ℚ := distanceFreddy / timeFreddy

-- Prove the ratio of their average speeds is 19:10
theorem ratio_average_speed (h1 : distanceAB = 600) (h2 : timeAB = 3) 
                           (h3 : distanceBD = 540) (h4 : timeBD = 2)
                           (h5 : distanceAC = 460) (h6 : timeAC = 4) 
                           (h7 : distanceCE = 380) (h8 : timeCE = 3):
  averageSpeedEddy / averageSpeedFreddy = 19 / 10 := by sorry

end NUMINAMATH_GPT_ratio_average_speed_l1658_165838


namespace NUMINAMATH_GPT_flowers_died_l1658_165872

theorem flowers_died : 
  let initial_flowers := 2 * 5
  let grown_flowers := initial_flowers + 20
  let harvested_flowers := 5 * 4
  grown_flowers - harvested_flowers = 10 :=
by
  sorry

end NUMINAMATH_GPT_flowers_died_l1658_165872


namespace NUMINAMATH_GPT_lumber_price_increase_l1658_165849

noncomputable def percentage_increase_in_lumber_cost : ℝ :=
  let original_cost_lumber := 450
  let cost_nails := 30
  let cost_fabric := 80
  let original_total_cost := original_cost_lumber + cost_nails + cost_fabric
  let increase_in_total_cost := 97
  let new_total_cost := original_total_cost + increase_in_total_cost
  let unchanged_cost := cost_nails + cost_fabric
  let new_cost_lumber := new_total_cost - unchanged_cost
  let increase_lumber_cost := new_cost_lumber - original_cost_lumber
  (increase_lumber_cost / original_cost_lumber) * 100

theorem lumber_price_increase :
  percentage_increase_in_lumber_cost = 21.56 := by
  sorry

end NUMINAMATH_GPT_lumber_price_increase_l1658_165849


namespace NUMINAMATH_GPT_team_CB_days_worked_together_l1658_165876

def projectA := 1 -- Project A is 1 unit of work
def projectB := 5 / 4 -- Project B is 1.25 units of work
def work_rate_A := 1 / 20 -- Team A's work rate
def work_rate_B := 1 / 24 -- Team B's work rate
def work_rate_C := 1 / 30 -- Team C's work rate

noncomputable def combined_rate_without_C := work_rate_B + work_rate_C

noncomputable def combined_total_work := projectA + projectB

noncomputable def days_for_combined_work := combined_total_work / combined_rate_without_C

-- Statement to prove the number of days team C and team B worked together
theorem team_CB_days_worked_together : 
  days_for_combined_work = 15 := 
  sorry

end NUMINAMATH_GPT_team_CB_days_worked_together_l1658_165876


namespace NUMINAMATH_GPT_integer_solutions_l1658_165841

theorem integer_solutions (x y : ℤ) : 
  (x^2 + x = y^4 + y^3 + y^2 + y) ↔ 
  (x, y) = (0, -1) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (5, 2) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l1658_165841


namespace NUMINAMATH_GPT_problem_solution_l1658_165846

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = 7 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1658_165846


namespace NUMINAMATH_GPT_find_A_l1658_165826

theorem find_A (A M C : Nat) (h1 : (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050) (h2 : A < 10) (h3 : M < 10) (h4 : C < 10) : A = 2 := by
  sorry

end NUMINAMATH_GPT_find_A_l1658_165826


namespace NUMINAMATH_GPT_find_constants_exist_l1658_165829

theorem find_constants_exist :
  ∃ A B C, (∀ x, 4 * x / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2)
  ∧ (A = 5) ∧ (B = -5) ∧ (C = -6) := 
sorry

end NUMINAMATH_GPT_find_constants_exist_l1658_165829


namespace NUMINAMATH_GPT_polynomial_p0_l1658_165871

theorem polynomial_p0 :
  ∃ p : ℕ → ℚ, (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (3^n)) ∧ (p 0 = 1093) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_p0_l1658_165871


namespace NUMINAMATH_GPT_tim_total_trip_time_l1658_165832

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end NUMINAMATH_GPT_tim_total_trip_time_l1658_165832


namespace NUMINAMATH_GPT_students_present_in_class_l1658_165847

noncomputable def num_students : ℕ := 100
noncomputable def percent_boys : ℝ := 0.55
noncomputable def percent_girls : ℝ := 0.45
noncomputable def absent_boys_percent : ℝ := 0.16
noncomputable def absent_girls_percent : ℝ := 0.12

theorem students_present_in_class :
  let num_boys := percent_boys * num_students
  let num_girls := percent_girls * num_students
  let absent_boys := absent_boys_percent * num_boys
  let absent_girls := absent_girls_percent * num_girls
  let present_boys := num_boys - absent_boys
  let present_girls := num_girls - absent_girls
  present_boys + present_girls = 86 :=
by
  sorry

end NUMINAMATH_GPT_students_present_in_class_l1658_165847


namespace NUMINAMATH_GPT_locus_of_Y_right_angled_triangle_l1658_165865

-- Conditions definitions
variables {A B C : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (b c m : ℝ) -- Coordinates and slopes related to the problem
variables (x : ℝ) -- Independent variable for the locus line

-- The problem statement
theorem locus_of_Y_right_angled_triangle 
  (A_right_angle : ∀ (α β : ℝ), α * β = 0) 
  (perpendicular_lines : b ≠ m * c) 
  (no_coincide : (b^2 * m - 2 * b * c - c^2 * m) ≠ 0) :
  ∃ (y : ℝ), y = (2 * b * c * (b * m - c) - x * (b^2 + 2 * b * c * m - c^2)) / (b^2 * m - 2 * b * c - c^2 * m) := 
sorry

end NUMINAMATH_GPT_locus_of_Y_right_angled_triangle_l1658_165865


namespace NUMINAMATH_GPT_total_boxes_is_27_l1658_165839

-- Defining the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Prove that the total number of boxes is as expected
theorem total_boxes_is_27 : stops * boxes_per_stop = 27 := by
  sorry

end NUMINAMATH_GPT_total_boxes_is_27_l1658_165839


namespace NUMINAMATH_GPT_pqr_problem_l1658_165816

noncomputable def pqr_abs (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : ℝ :=
|p * q * r|

theorem pqr_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : pqr_abs p q r h1 h2 h3 h4 h5 h6 h7 h8 = 2 := 
sorry

end NUMINAMATH_GPT_pqr_problem_l1658_165816


namespace NUMINAMATH_GPT_advertisement_revenue_l1658_165801

theorem advertisement_revenue
  (cost_per_program : ℝ)
  (num_programs : ℕ)
  (selling_price_per_program : ℝ)
  (desired_profit : ℝ)
  (total_cost_production : ℝ)
  (total_revenue_sales : ℝ)
  (total_revenue_needed : ℝ)
  (revenue_from_advertisements : ℝ) :
  cost_per_program = 0.70 →
  num_programs = 35000 →
  selling_price_per_program = 0.50 →
  desired_profit = 8000 →
  total_cost_production = cost_per_program * num_programs →
  total_revenue_sales = selling_price_per_program * num_programs →
  total_revenue_needed = total_cost_production + desired_profit →
  revenue_from_advertisements = total_revenue_needed - total_revenue_sales →
  revenue_from_advertisements = 15000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_advertisement_revenue_l1658_165801


namespace NUMINAMATH_GPT_washing_machine_capacity_l1658_165877

def num_shirts : Nat := 19
def num_sweaters : Nat := 8
def num_loads : Nat := 3

theorem washing_machine_capacity :
  (num_shirts + num_sweaters) / num_loads = 9 := by
  sorry

end NUMINAMATH_GPT_washing_machine_capacity_l1658_165877


namespace NUMINAMATH_GPT_haley_small_gardens_l1658_165844

theorem haley_small_gardens (total_seeds seeds_in_big_garden seeds_per_small_garden : ℕ) (h1 : total_seeds = 56) (h2 : seeds_in_big_garden = 35) (h3 : seeds_per_small_garden = 3) :
  (total_seeds - seeds_in_big_garden) / seeds_per_small_garden = 7 :=
by
  sorry

end NUMINAMATH_GPT_haley_small_gardens_l1658_165844


namespace NUMINAMATH_GPT_sum_S10_equals_10_div_21_l1658_165804

def a (n : ℕ) : ℚ := 1 / (4 * n^2 - 1)
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_S10_equals_10_div_21 : S 10 = 10 / 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_S10_equals_10_div_21_l1658_165804


namespace NUMINAMATH_GPT_coefficient_of_term_free_of_x_l1658_165810

theorem coefficient_of_term_free_of_x 
  (n : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → n = 10) 
  (h2 : (n.choose 4 / n.choose 2) = 14 / 3) : 
  ∃ (c : ℚ), c = 5 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_term_free_of_x_l1658_165810


namespace NUMINAMATH_GPT_min_frac_sum_min_frac_sum_achieved_l1658_165862

theorem min_frac_sum (a b : ℝ) (h₁ : 2 * a + 3 * b = 6) (h₂ : 0 < a) (h₃ : 0 < b) :
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

theorem min_frac_sum_achieved :
  (2 / (6 / 5) + 3 / (6 / 5)) = 25 / 6 :=
by sorry


end NUMINAMATH_GPT_min_frac_sum_min_frac_sum_achieved_l1658_165862


namespace NUMINAMATH_GPT_Grant_made_total_l1658_165813

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end NUMINAMATH_GPT_Grant_made_total_l1658_165813


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1658_165874

theorem sum_of_squares_of_roots 
  (r s t : ℝ) 
  (hr : y^3 - 8 * y^2 + 9 * y - 2 = 0) 
  (hs : y ≥ 0) 
  (ht : y ≥ 0):
  r^2 + s^2 + t^2 = 46 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1658_165874


namespace NUMINAMATH_GPT_parabola_properties_and_intersection_l1658_165822

-- Definition of the parabola C: y^2 = -4x
def parabola_C (x y : ℝ) : Prop := y^2 = -4 * x

-- Focus of the parabola
def focus_C : ℝ × ℝ := (-1, 0)

-- Equation of the directrix
def directrix_C (x: ℝ): Prop := x = 1

-- Distance from the focus to the directrix
def distance_focus_to_directrix : ℝ := 2

-- Line l passing through P(1, 2) with slope k
def line_l (k x y : ℝ) : Prop := y = k * x - k + 2

-- Main theorem statement
theorem parabola_properties_and_intersection (k: ℝ) :
  (focus_C = (-1, 0)) ∧
  (∀ x, directrix_C x ↔ x = 1) ∧
  (distance_focus_to_directrix = 2) ∧
  ((k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) →
    ∃ x y, parabola_C x y ∧ line_l k x y ∧
    (∀ x' y', parabola_C x' y' ∧ line_l k x' y' → x = x' ∧ y = y')) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) →
    ∃ x y x' y', x ≠ x' ∧ y ≠ y' ∧
    parabola_C x y ∧ line_l k x y ∧
    parabola_C x' y' ∧ line_l k x' y') ∧
  ((k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) →
    ∀ x y, ¬(parabola_C x y ∧ line_l k x y)) :=
by sorry

end NUMINAMATH_GPT_parabola_properties_and_intersection_l1658_165822


namespace NUMINAMATH_GPT_multiply_add_distribute_l1658_165836

theorem multiply_add_distribute :
  42 * 25 + 58 * 42 = 3486 := by
  sorry

end NUMINAMATH_GPT_multiply_add_distribute_l1658_165836


namespace NUMINAMATH_GPT_trapezoid_perimeter_l1658_165821

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD BC DA : ℝ)
  (AB_parallel_CD : AB = CD)
  (BC_eq_DA : BC = 13)
  (DA_eq_BC : DA = 13)
  (sum_AB_CD : AB + CD = 24)

-- Define the problem's conditions as Lean definitions
def trapezoidABCD : IsoscelesTrapezoid ℝ ℝ ℝ ℝ :=
{
  AB := 12,
  CD := 12,
  BC := 13,
  DA := 13,
  AB_parallel_CD := by sorry,
  BC_eq_DA := by sorry,
  DA_eq_BC := by sorry,
  sum_AB_CD := by sorry,
}

-- State the theorem we want to prove
theorem trapezoid_perimeter (trapezoid : IsoscelesTrapezoid ℝ ℝ ℝ ℝ) : 
  trapezoid.AB + trapezoid.BC + trapezoid.CD + trapezoid.DA = 50 :=
by sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l1658_165821


namespace NUMINAMATH_GPT_coplanar_vectors_set_B_l1658_165890

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (a b c : V)

theorem coplanar_vectors_set_B
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ • (2 • a + b) + k₂ • (a + b + c) = 7 • a + 5 • b + 3 • c :=
by { sorry }

end NUMINAMATH_GPT_coplanar_vectors_set_B_l1658_165890


namespace NUMINAMATH_GPT_min_value_of_expression_l1658_165898

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 4) :
  (9 / x + 1 / y + 25 / z) ≥ 20.25 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1658_165898


namespace NUMINAMATH_GPT_normal_complaints_calculation_l1658_165879

-- Define the normal number of complaints
def normal_complaints (C : ℕ) : ℕ := C

-- Define the complaints when short-staffed
def short_staffed_complaints (C : ℕ) : ℕ := (4 * C) / 3

-- Define the complaints when both conditions are met
def both_conditions_complaints (C : ℕ) : ℕ := (4 * C) / 3 + (4 * C) / 15

-- Main statement to prove
theorem normal_complaints_calculation (C : ℕ) (h : 3 * (both_conditions_complaints C) = 576) : C = 120 :=
by sorry

end NUMINAMATH_GPT_normal_complaints_calculation_l1658_165879


namespace NUMINAMATH_GPT_probability_of_third_round_expected_value_of_X_variance_of_X_l1658_165805

-- Define the probabilities for passing each round
def P_A : ℚ := 2 / 3
def P_B : ℚ := 3 / 4
def P_C : ℚ := 4 / 5

-- Prove the probability of reaching the third round
theorem probability_of_third_round :
  P_A * P_B = 1 / 2 := sorry

-- Define the probability distribution
def P_X (x : ℕ) : ℚ :=
  if x = 1 then 1 / 3 
  else if x = 2 then 1 / 6
  else if x = 3 then 1 / 2
  else 0

-- Expected value
def EX : ℚ := 1 * (1 / 3) + 2 * (1 / 6) + 3 * (1 / 2)

theorem expected_value_of_X :
  EX = 13 / 6 := sorry

-- E(X^2) computation
def EX2 : ℚ := 1^2 * (1 / 3) + 2^2 * (1 / 6) + 3^2 * (1 / 2)

-- Variance
def variance_X : ℚ := EX2 - EX^2

theorem variance_of_X :
  variance_X = 41 / 36 := sorry

end NUMINAMATH_GPT_probability_of_third_round_expected_value_of_X_variance_of_X_l1658_165805


namespace NUMINAMATH_GPT_john_paid_more_than_jane_l1658_165851

theorem john_paid_more_than_jane :
    let original_price : ℝ := 40.00
    let discount_percentage : ℝ := 0.10
    let tip_percentage : ℝ := 0.15
    let discounted_price : ℝ := original_price - (discount_percentage * original_price)
    let john_tip : ℝ := tip_percentage * original_price
    let john_total : ℝ := discounted_price + john_tip
    let jane_tip : ℝ := tip_percentage * discounted_price
    let jane_total : ℝ := discounted_price + jane_tip
    let difference : ℝ := john_total - jane_total
    difference = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_john_paid_more_than_jane_l1658_165851


namespace NUMINAMATH_GPT_min_total_cost_of_tank_l1658_165819

theorem min_total_cost_of_tank (V D c₁ c₂ : ℝ) (hV : V = 0.18) (hD : D = 0.5)
  (hc₁ : c₁ = 400) (hc₂ : c₂ = 100) : 
  ∃ x : ℝ, x > 0 ∧ (y = c₂*D*(2*x + 0.72/x) + c₁*0.36) ∧ y = 264 := 
sorry

end NUMINAMATH_GPT_min_total_cost_of_tank_l1658_165819


namespace NUMINAMATH_GPT_cheapest_pie_cost_is_18_l1658_165859

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end NUMINAMATH_GPT_cheapest_pie_cost_is_18_l1658_165859


namespace NUMINAMATH_GPT_smallest_possible_perimeter_l1658_165860

theorem smallest_possible_perimeter (a : ℕ) (h : a > 2) (h_triangle : a < a + (a + 1) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a) :
  3 * a + 3 = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_perimeter_l1658_165860


namespace NUMINAMATH_GPT_Toms_walking_speed_l1658_165881

theorem Toms_walking_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (run_distance : ℝ)
  (run_speed : ℝ)
  (walk_distance : ℝ)
  (walk_time : ℝ)
  (walk_speed : ℝ)
  (h1 : total_distance = 1800)
  (h2 : total_time ≤ 20)
  (h3 : run_distance = 600)
  (h4 : run_speed = 210)
  (h5 : total_distance = run_distance + walk_distance)
  (h6 : total_time = walk_time + run_distance / run_speed)
  (h7 : walk_speed = walk_distance / walk_time) :
  walk_speed ≤ 70 := sorry

end NUMINAMATH_GPT_Toms_walking_speed_l1658_165881


namespace NUMINAMATH_GPT_trig_identity_l1658_165807

theorem trig_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 :=
sorry

end NUMINAMATH_GPT_trig_identity_l1658_165807


namespace NUMINAMATH_GPT_largest_k_divides_3n_plus_1_l1658_165863

theorem largest_k_divides_3n_plus_1 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, k = 2 ∧ n % 2 = 1 ∧ 2^k ∣ 3^n + 1 ∨ k = 1 ∧ n % 2 = 0 ∧ 2^k ∣ 3^n + 1 :=
sorry

end NUMINAMATH_GPT_largest_k_divides_3n_plus_1_l1658_165863


namespace NUMINAMATH_GPT_xiaobo_probability_not_home_l1658_165853

theorem xiaobo_probability_not_home :
  let r1 := 1 / 2
  let r2 := 1 / 4
  let area_circle := Real.pi
  let area_greater_r1 := area_circle * (1 - r1^2)
  let area_less_r2 := area_circle * r2^2
  let area_favorable := area_greater_r1 + area_less_r2
  let probability_not_home := area_favorable / area_circle
  probability_not_home = 13 / 16 := by
  sorry

end NUMINAMATH_GPT_xiaobo_probability_not_home_l1658_165853


namespace NUMINAMATH_GPT_harrison_grade_levels_l1658_165815

theorem harrison_grade_levels
  (total_students : ℕ)
  (percent_moving : ℚ)
  (advanced_class_size : ℕ)
  (num_normal_classes : ℕ)
  (normal_class_size : ℕ)
  (students_moving : ℕ)
  (students_per_grade_level : ℕ)
  (grade_levels : ℕ) :
  total_students = 1590 →
  percent_moving = 40 / 100 →
  advanced_class_size = 20 →
  num_normal_classes = 6 →
  normal_class_size = 32 →
  students_moving = total_students * percent_moving →
  students_per_grade_level = advanced_class_size + num_normal_classes * normal_class_size →
  grade_levels = students_moving / students_per_grade_level →
  grade_levels = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_harrison_grade_levels_l1658_165815


namespace NUMINAMATH_GPT_greatest_four_digit_divisible_by_6_l1658_165880

-- Define a variable to represent a four-digit number
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a variable to represent divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define a variable to represent divisibility by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- State the theorem to prove that 9996 is the greatest four-digit number divisible by 6
theorem greatest_four_digit_divisible_by_6 : 
  (∀ n : ℕ, is_four_digit_number n → divisible_by_6 n → n ≤ 9996) ∧ (is_four_digit_number 9996 ∧ divisible_by_6 9996) :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_greatest_four_digit_divisible_by_6_l1658_165880


namespace NUMINAMATH_GPT_cycle_selling_price_l1658_165812

theorem cycle_selling_price
(C : ℝ := 1900)  -- Cost price of the cycle
(Lp : ℝ := 18)  -- Loss percentage
(S : ℝ := 1558) -- Expected selling price
: (S = C - (Lp / 100) * C) :=
by 
  sorry

end NUMINAMATH_GPT_cycle_selling_price_l1658_165812


namespace NUMINAMATH_GPT_David_total_swim_time_l1658_165869

theorem David_total_swim_time :
  let t_freestyle := 48
  let t_backstroke := t_freestyle + 4
  let t_butterfly := t_backstroke + 3
  let t_breaststroke := t_butterfly + 2
  t_freestyle + t_backstroke + t_butterfly + t_breaststroke = 212 :=
by
  sorry

end NUMINAMATH_GPT_David_total_swim_time_l1658_165869


namespace NUMINAMATH_GPT_phi_value_for_unique_symmetry_center_l1658_165899

theorem phi_value_for_unique_symmetry_center :
  ∃ (φ : ℝ), (0 < φ ∧ φ < π / 2) ∧
  (φ = π / 12 ∨ φ = π / 6 ∨ φ = π / 3 ∨ φ = 5 * π / 12) ∧
  ((∃ x : ℝ, 2 * x + φ = π ∧ π / 6 < x ∧ x < π / 3) ↔ φ = 5 * π / 12) :=
  sorry

end NUMINAMATH_GPT_phi_value_for_unique_symmetry_center_l1658_165899


namespace NUMINAMATH_GPT_circumscribed_sphere_radius_l1658_165820

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt (6 + Real.sqrt 20)) / 8

theorem circumscribed_sphere_radius (a : ℝ) :
  radius_of_circumscribed_sphere a = a * (Real.sqrt (6 + Real.sqrt 20)) / 8 :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_sphere_radius_l1658_165820


namespace NUMINAMATH_GPT_robbers_can_divide_loot_equally_l1658_165827

theorem robbers_can_divide_loot_equally (coins : List ℕ) (h1 : (coins.sum % 2 = 0)) 
    (h2 : ∀ k, (k % 2 = 1 ∧ 1 ≤ k ∧ k ≤ 2017) → k ∈ coins) :
  ∃ (subset1 subset2 : List ℕ), subset1 ∪ subset2 = coins ∧ subset1.sum = subset2.sum :=
by
  sorry

end NUMINAMATH_GPT_robbers_can_divide_loot_equally_l1658_165827


namespace NUMINAMATH_GPT_freshmen_sophomores_without_pets_l1658_165886

theorem freshmen_sophomores_without_pets : 
  let total_students := 400
  let percentage_freshmen_sophomores := 0.50
  let percentage_with_pets := 1/5
  let freshmen_sophomores := percentage_freshmen_sophomores * total_students
  160 = (freshmen_sophomores - (percentage_with_pets * freshmen_sophomores)) :=
by
  sorry

end NUMINAMATH_GPT_freshmen_sophomores_without_pets_l1658_165886


namespace NUMINAMATH_GPT_general_term_a_n_l1658_165845

theorem general_term_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (2/3) * a n + 1/3) :
  ∀ n, a n = (-2)^(n-1) :=
sorry

end NUMINAMATH_GPT_general_term_a_n_l1658_165845


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1658_165858

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40 * z + 340 ≤ 4 ↔ 12 ≤ z ∧ z ≤ 28 := by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1658_165858


namespace NUMINAMATH_GPT_jenny_jellybeans_original_l1658_165833

theorem jenny_jellybeans_original (x : ℝ) 
  (h : 0.75^3 * x = 45) : x = 107 := 
sorry

end NUMINAMATH_GPT_jenny_jellybeans_original_l1658_165833


namespace NUMINAMATH_GPT_first_grade_children_count_l1658_165823

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end NUMINAMATH_GPT_first_grade_children_count_l1658_165823


namespace NUMINAMATH_GPT_hearing_aid_cost_l1658_165887

theorem hearing_aid_cost
  (cost : ℝ)
  (insurance_coverage : ℝ)
  (personal_payment : ℝ)
  (total_aid_count : ℕ)
  (h : total_aid_count = 2)
  (h_insurance : insurance_coverage = 0.80)
  (h_personal_payment : personal_payment = 1000)
  (h_equation : personal_payment = (1 - insurance_coverage) * (total_aid_count * cost)) :
  cost = 2500 :=
by
  sorry

end NUMINAMATH_GPT_hearing_aid_cost_l1658_165887


namespace NUMINAMATH_GPT_function_has_two_zeros_for_a_eq_2_l1658_165854

noncomputable def f (a x : ℝ) : ℝ := a ^ x - x - 1

theorem function_has_two_zeros_for_a_eq_2 :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f 2 x1 = 0 ∧ f 2 x2 = 0) := sorry

end NUMINAMATH_GPT_function_has_two_zeros_for_a_eq_2_l1658_165854


namespace NUMINAMATH_GPT_large_cube_painted_blue_l1658_165885

theorem large_cube_painted_blue (n : ℕ) (hp : 1 ≤ n) 
  (hc : (6 * n^2) = (1 / 3) * 6 * n^3) : n = 3 := by
  have hh := hc
  sorry

end NUMINAMATH_GPT_large_cube_painted_blue_l1658_165885


namespace NUMINAMATH_GPT_clock_resale_price_l1658_165868

theorem clock_resale_price
    (C : ℝ)  -- original cost of the clock to the store
    (H1 : 0.40 * C = 100)  -- condition: difference between original cost and buy-back price is $100
    (H2 : ∀ (C : ℝ), resell_price = 1.80 * (0.60 * C))  -- store sold the clock again with a 80% profit on buy-back
    : resell_price = 270 := 
by
  sorry

end NUMINAMATH_GPT_clock_resale_price_l1658_165868


namespace NUMINAMATH_GPT_total_cement_used_l1658_165848

-- Define the amounts of cement used for Lexi's street and Tess's street
def cement_used_lexis_street : ℝ := 10
def cement_used_tess_street : ℝ := 5.1

-- Prove that the total amount of cement used is 15.1 tons
theorem total_cement_used : cement_used_lexis_street + cement_used_tess_street = 15.1 := sorry

end NUMINAMATH_GPT_total_cement_used_l1658_165848


namespace NUMINAMATH_GPT_nat_solutions_l1658_165855

open Nat

theorem nat_solutions (a b c : ℕ) :
  (a ≤ b ∧ b ≤ c ∧ ab + bc + ca = 2 * (a + b + c)) ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 4) :=
by sorry

end NUMINAMATH_GPT_nat_solutions_l1658_165855


namespace NUMINAMATH_GPT_derek_books_ratio_l1658_165831

theorem derek_books_ratio :
  ∃ (T : ℝ), 960 - T - (1/4) * (960 - T) = 360 ∧ T / 960 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_derek_books_ratio_l1658_165831


namespace NUMINAMATH_GPT_digit_in_tens_place_is_nine_l1658_165882

/-
Given:
1. Two numbers represented as 6t5 and 5t6 (where t is a digit).
2. The result of subtracting these two numbers is 9?4, where '?' represents a single digit in the tens place.

Prove:
The digit represented by '?' in the tens place is 9.
-/

theorem digit_in_tens_place_is_nine (t : ℕ) (h1 : 0 ≤ t ∧ t ≤ 9) :
  let a := 600 + t * 10 + 5
  let b := 500 + t * 10 + 6
  let result := a - b
  (result % 100) / 10 = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_digit_in_tens_place_is_nine_l1658_165882


namespace NUMINAMATH_GPT_arrangement_count_l1658_165864

-- Given conditions
def num_basketballs : ℕ := 5
def num_volleyballs : ℕ := 3
def num_footballs : ℕ := 2
def total_balls : ℕ := num_basketballs + num_volleyballs + num_footballs

-- Way to calculate the permutations of multiset
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Proof statement
theorem arrangement_count : 
  factorial total_balls / (factorial num_basketballs * factorial num_volleyballs * factorial num_footballs) = 2520 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_l1658_165864


namespace NUMINAMATH_GPT_min_tiles_to_cover_region_l1658_165883

noncomputable def num_tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area

theorem min_tiles_to_cover_region : num_tiles_needed 6 2 36 72 = 216 :=
by 
  -- This is the format needed to include the assumptions and reach the conclusion
  sorry

end NUMINAMATH_GPT_min_tiles_to_cover_region_l1658_165883


namespace NUMINAMATH_GPT_test_question_count_l1658_165809

def total_test_questions 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (num_4pt_questions : ℕ) : Prop :=
  total_points = points_per_2pt * num_2pt_questions + points_per_4pt * num_4pt_questions 

theorem test_question_count 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (correct_total_questions : ℕ) :
  total_test_questions total_points points_per_2pt points_per_4pt num_2pt_questions (correct_total_questions - num_2pt_questions) → correct_total_questions = 40 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_test_question_count_l1658_165809


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1658_165840

theorem solution_set_of_inequality (x : ℝ) : (x / (x - 1) < 0) ↔ (0 < x ∧ x < 1) := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1658_165840


namespace NUMINAMATH_GPT_value_of_x_l1658_165842

-- Define the custom operation * for the problem
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Define the main problem statement
theorem value_of_x (x : ℝ) (h : star 3 (star 7 x) = 5) : x = 49 / 4 :=
by
  have h7x : star 7 x = 28 - 2 * x := by sorry  -- Derived from the definitions
  have h3star7x : star 3 (28 - 2 * x) = -44 + 4 * x := by sorry  -- Derived from substituting star 7 x
  sorry

end NUMINAMATH_GPT_value_of_x_l1658_165842


namespace NUMINAMATH_GPT_gnuff_tutor_minutes_l1658_165825

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end NUMINAMATH_GPT_gnuff_tutor_minutes_l1658_165825


namespace NUMINAMATH_GPT_payback_time_l1658_165889

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_payback_time_l1658_165889


namespace NUMINAMATH_GPT_nat_ineq_qr_ps_l1658_165888

theorem nat_ineq_qr_ps (a b p q r s : ℕ) (h₀ : q * r - p * s = 1) 
  (h₁ : (p : ℚ) / q < a / b) (h₂ : (a : ℚ) / b < r / s) 
  : b ≥ q + s := sorry

end NUMINAMATH_GPT_nat_ineq_qr_ps_l1658_165888


namespace NUMINAMATH_GPT_trigonometric_identity_l1658_165808

theorem trigonometric_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.tan α = -2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 11 / 5 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1658_165808


namespace NUMINAMATH_GPT_fit_seven_rectangles_l1658_165870

theorem fit_seven_rectangles (s : ℝ) (a : ℝ) : (s > 0) → (a > 0) → (14 * a ^ 2 ≤ s ^ 2 ∧ 2 * a ≤ s) → 
  (∃ (rectangles : Fin 7 → (ℝ × ℝ)), ∀ i, rectangles i = (a, 2 * a) ∧
   ∀ i j, i ≠ j → rectangles i ≠ rectangles j) :=
sorry

end NUMINAMATH_GPT_fit_seven_rectangles_l1658_165870


namespace NUMINAMATH_GPT_expand_array_l1658_165818

theorem expand_array (n : ℕ) (h₁ : n ≥ 3) 
  (matrix : Fin (n-2) → Fin n → Fin n)
  (h₂ : ∀ i : Fin (n-2), ∀ j: Fin n, ∀ k: Fin n, j ≠ k → matrix i j ≠ matrix i k)
  (h₃ : ∀ j : Fin n, ∀ k: Fin (n-2), ∀ l: Fin (n-2), k ≠ l → matrix k j ≠ matrix l j) :
  ∃ (expanded_matrix : Fin n → Fin n → Fin n), 
    (∀ i : Fin n, ∀ j: Fin n, ∀ k: Fin n, j ≠ k → expanded_matrix i j ≠ expanded_matrix i k) ∧
    (∀ j : Fin n, ∀ k: Fin n, ∀ l: Fin n, k ≠ l → expanded_matrix k j ≠ expanded_matrix l j) :=
sorry

end NUMINAMATH_GPT_expand_array_l1658_165818


namespace NUMINAMATH_GPT_min_selling_price_is_400_l1658_165875

-- Definitions for the problem conditions
def total_products := 20
def average_price := 1200
def less_than_1000_count := 10
def price_of_most_expensive := 11000
def total_retail_price := total_products * average_price

-- The theorem to state the problem condition and the expected result
theorem min_selling_price_is_400 (x : ℕ) :
  -- Condition 1: Total retail price
  total_retail_price =
  -- 10 products sell for x dollars
  (10 * x) +
  -- 9 products sell for 1000 dollars
  (9 * 1000) +
  -- 1 product sells for the maximum price 11000
  price_of_most_expensive → 
  -- Conclusion: The minimum price x is 400
  x = 400 :=
by
  sorry

end NUMINAMATH_GPT_min_selling_price_is_400_l1658_165875


namespace NUMINAMATH_GPT_pyramid_volume_correct_l1658_165866

noncomputable def PyramidVolume (base_area : ℝ) (triangle_area_1 : ℝ) (triangle_area_2 : ℝ) : ℝ :=
  let side := Real.sqrt base_area
  let height_1 := (2 * triangle_area_1) / side
  let height_2 := (2 * triangle_area_2) / side
  let h_sq := height_1 ^ 2 - (Real.sqrt (height_1 ^ 2 + height_2 ^ 2 - 512)) ^ 2
  let height := Real.sqrt h_sq
  (1/3) * base_area * height

theorem pyramid_volume_correct :
  PyramidVolume 256 120 112 = 1163 := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_correct_l1658_165866


namespace NUMINAMATH_GPT_transformed_function_equivalence_l1658_165803

-- Define the original function
def original_function (x : ℝ) : ℝ := 2 * x + 1

-- Define the transformation involving shifting 2 units to the right
def transformed_function (x : ℝ) : ℝ := original_function (x - 2)

-- The theorem we want to prove
theorem transformed_function_equivalence : 
  ∀ x : ℝ, transformed_function x = 2 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_transformed_function_equivalence_l1658_165803


namespace NUMINAMATH_GPT_integer_solutions_eq_l1658_165850

theorem integer_solutions_eq :
  { (x, y) : ℤ × ℤ | 2 * x ^ 4 - 4 * y ^ 4 - 7 * x ^ 2 * y ^ 2 - 27 * x ^ 2 + 63 * y ^ 2 + 85 = 0 }
  = { (3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3) } :=
by sorry

end NUMINAMATH_GPT_integer_solutions_eq_l1658_165850


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l1658_165802

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem number_of_zeros_of_f :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l1658_165802


namespace NUMINAMATH_GPT_max_ab_l1658_165856

theorem max_ab (a b : ℝ) (h : 4 * a + b = 1) (ha : a > 0) (hb : b > 0) : ab <= 1 / 16 :=
sorry

end NUMINAMATH_GPT_max_ab_l1658_165856


namespace NUMINAMATH_GPT_minimum_value_of_y_l1658_165891

theorem minimum_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 36 * y) : y ≥ -7 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_y_l1658_165891


namespace NUMINAMATH_GPT_mike_worked_four_hours_l1658_165857

-- Define the time to perform each task in minutes
def time_wash_car : ℕ := 10
def time_change_oil : ℕ := 15
def time_change_tires : ℕ := 30

-- Define the number of tasks Mike performed
def num_wash_cars : ℕ := 9
def num_change_oil : ℕ := 6
def num_change_tires : ℕ := 2

-- Define the total minutes Mike worked
def total_minutes_worked : ℕ :=
  (num_wash_cars * time_wash_car) +
  (num_change_oil * time_change_oil) +
  (num_change_tires * time_change_tires)

-- Define the conversion from minutes to hours
def total_hours_worked : ℕ := total_minutes_worked / 60

-- Formalize the proof statement
theorem mike_worked_four_hours :
  total_hours_worked = 4 :=
by
  sorry

end NUMINAMATH_GPT_mike_worked_four_hours_l1658_165857


namespace NUMINAMATH_GPT_diagonals_in_polygon_of_150_sides_l1658_165800

-- Definition of the number of diagonals formula
def number_of_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Given condition: the polygon has 150 sides
def n : ℕ := 150

-- Statement to prove
theorem diagonals_in_polygon_of_150_sides : number_of_diagonals n = 11025 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_in_polygon_of_150_sides_l1658_165800


namespace NUMINAMATH_GPT_team_total_points_l1658_165852

theorem team_total_points 
  (n : ℕ)
  (best_score actual : ℕ)
  (desired_avg : ℕ)
  (hypothetical_score : ℕ)
  (current_best_score : ℕ)
  (team_size : ℕ)
  (h1 : team_size = 8)
  (h2 : current_best_score = 85)
  (h3 : hypothetical_score = 92)
  (h4 : desired_avg = 84)
  (h5 : hypothetical_score - current_best_score = 7)
  (h6 : team_size * desired_avg = 672) :
  (actual = 665) :=
sorry

end NUMINAMATH_GPT_team_total_points_l1658_165852


namespace NUMINAMATH_GPT_circle_intersection_l1658_165806

theorem circle_intersection : 
  ∀ (O : ℝ × ℝ), ∃ (m n : ℤ), (dist (O.1, O.2) (m, n) ≤ 100 + 1/14) := 
sorry

end NUMINAMATH_GPT_circle_intersection_l1658_165806


namespace NUMINAMATH_GPT_correct_option_is_A_l1658_165830

-- Define the options as terms
def optionA (x : ℝ) := (1/2) * x - 5 * x = 18
def optionB (x : ℝ) := (1/2) * x > 5 * x - 1
def optionC (y : ℝ) := 8 * y - 4
def optionD := 5 - 2 = 3

-- Define a function to check if an option is an equation
def is_equation (option : Prop) : Prop :=
  ∃ (x : ℝ), option = ((1/2) * x - 5 * x = 18)

-- Prove that optionA is the equation
theorem correct_option_is_A : is_equation (optionA x) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_A_l1658_165830


namespace NUMINAMATH_GPT_square_area_l1658_165896

theorem square_area :
  ∃ (s : ℝ), (8 * s - 2 = 30) ∧ (s ^ 2 = 16) :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1658_165896


namespace NUMINAMATH_GPT_at_least_one_fraction_less_than_two_l1658_165835

theorem at_least_one_fraction_less_than_two {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
by
  sorry

end NUMINAMATH_GPT_at_least_one_fraction_less_than_two_l1658_165835


namespace NUMINAMATH_GPT_rectangle_area_perimeter_l1658_165884

-- Defining the problem conditions
def positive_int (n : Int) : Prop := n > 0

-- The main statement of the problem
theorem rectangle_area_perimeter (a b : Int) (h1 : positive_int a) (h2 : positive_int b) : 
  ¬ (a + 2) * (b + 2) - 4 = 146 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_perimeter_l1658_165884


namespace NUMINAMATH_GPT_problem_I_problem_II_l1658_165892

open Real -- To use real number definitions and sin function.
open Set -- To use set constructs like intervals.

noncomputable def f (x : ℝ) : ℝ := sin (4 * x - π / 6) + sqrt 3 * sin (4 * x + π / 3)

-- Proof statement for monotonically decreasing interval of f(x).
theorem problem_I (k : ℤ) : 
  ∃ k : ℤ, ∀ x : ℝ, x ∈ Icc ((π / 12) + (k * π / 2)) ((π / 3) + (k * π / 2)) → 
  (4 * x + π / 6) ∈ Icc ((π / 2) + 2 * k * π) ((3 * π / 2) + 2 * k * π) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 4)

-- Proof statement for the range of g(x) on the interval [-π, 0].
theorem problem_II : 
  ∀ x : ℝ, x ∈ Icc (-π) 0 → g x ∈ Icc (-2) (sqrt 2) := 
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1658_165892


namespace NUMINAMATH_GPT_quadruple_application_of_h_l1658_165814

-- Define the function as specified in the condition
def h (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem quadruple_application_of_h : h (h (h (h 40))) = 9.536 :=
  by
    sorry

end NUMINAMATH_GPT_quadruple_application_of_h_l1658_165814


namespace NUMINAMATH_GPT_distinct_non_zero_real_numbers_l1658_165873

theorem distinct_non_zero_real_numbers (
  a b c : ℝ
) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + 2 * b * x1 + c = 0 ∧ ax^2 + 2 * b * x2 + c = 0) 
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ bx^2 + 2 * c * x1 + a = 0 ∧ bx^2 + 2 * c * x2 + a = 0)
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ cx^2 + 2 * a * x1 + b = 0 ∧ cx^2 + 2 * a * x2 + b = 0) :=
sorry

end NUMINAMATH_GPT_distinct_non_zero_real_numbers_l1658_165873


namespace NUMINAMATH_GPT_smallest_sum_ab_l1658_165867

theorem smallest_sum_ab (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2^10 * 3^6 = a^b) : a + b = 866 :=
sorry

end NUMINAMATH_GPT_smallest_sum_ab_l1658_165867


namespace NUMINAMATH_GPT_cost_difference_is_35_88_usd_l1658_165897

/-
  Mr. Llesis bought 50 kilograms of rice at different prices per kilogram from various suppliers.
  He bought:
  - 15 kilograms at €1.2 per kilogram from Supplier A
  - 10 kilograms at €1.4 per kilogram from Supplier B
  - 12 kilograms at €1.6 per kilogram from Supplier C
  - 8 kilograms at €1.9 per kilogram from Supplier D
  - 5 kilograms at €2.3 per kilogram from Supplier E

  He kept 7/10 of the total rice in storage and gave the rest to Mr. Everest.
  The current conversion rate is €1 = $1.15.
  
  Prove that the difference in cost in US dollars between the rice kept and the rice given away is $35.88.
-/

def euros_to_usd (euros : ℚ) : ℚ :=
  euros * (115 / 100)

def total_cost : ℚ := 
  (15 * 1.2) + (10 * 1.4) + (12 * 1.6) + (8 * 1.9) + (5 * 2.3)

def cost_kept : ℚ := (7/10) * total_cost
def cost_given : ℚ := (3/10) * total_cost

theorem cost_difference_is_35_88_usd :
  euros_to_usd cost_kept - euros_to_usd cost_given = 35.88 := 
sorry

end NUMINAMATH_GPT_cost_difference_is_35_88_usd_l1658_165897


namespace NUMINAMATH_GPT_first_group_persons_l1658_165817

-- Define the conditions as formal variables
variables (P : ℕ) (hours_per_day_1 days_1 hours_per_day_2 days_2 num_persons_2 : ℕ)

-- Define the conditions from the problem
def first_group_work := P * days_1 * hours_per_day_1
def second_group_work := num_persons_2 * days_2 * hours_per_day_2

-- Set the conditions based on the problem statement
axiom conditions : 
  hours_per_day_1 = 5 ∧ 
  days_1 = 12 ∧ 
  hours_per_day_2 = 6 ∧
  days_2 = 26 ∧
  num_persons_2 = 30 ∧
  first_group_work = second_group_work

-- Statement to prove
theorem first_group_persons : P = 78 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_first_group_persons_l1658_165817


namespace NUMINAMATH_GPT_combined_total_circles_squares_l1658_165837

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end NUMINAMATH_GPT_combined_total_circles_squares_l1658_165837
