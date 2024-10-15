import Mathlib

namespace NUMINAMATH_GPT_ratio_of_distances_l2418_241847

theorem ratio_of_distances
  (w x y : ℝ)
  (hw : w > 0)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq_time : y / w = x / w + (x + y) / (5 * w)) :
  x / y = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_distances_l2418_241847


namespace NUMINAMATH_GPT_xy_sum_l2418_241857

variable (x y : ℚ)

theorem xy_sum : (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_xy_sum_l2418_241857


namespace NUMINAMATH_GPT_total_time_l2418_241899

theorem total_time {minutes seconds : ℕ} (hmin : minutes = 3450) (hsec : seconds = 7523) :
  ∃ h m s : ℕ, h = 59 ∧ m = 35 ∧ s = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_time_l2418_241899


namespace NUMINAMATH_GPT_part1_part2_l2418_241893
noncomputable def equation1 (x k : ℝ) := 3 * (2 * x - 1) = k + 2 * x
noncomputable def equation2 (x k : ℝ) := (x - k) / 2 = x + 2 * k

theorem part1 (x k : ℝ) (h1 : equation1 4 k) : equation2 x k ↔ x = -65 := sorry

theorem part2 (x k : ℝ) (h1 : equation1 x k) (h2 : equation2 x k) : k = -1 / 7 := sorry

end NUMINAMATH_GPT_part1_part2_l2418_241893


namespace NUMINAMATH_GPT_factorize_expr_l2418_241811

theorem factorize_expr (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := 
sorry

end NUMINAMATH_GPT_factorize_expr_l2418_241811


namespace NUMINAMATH_GPT_jessa_cupcakes_l2418_241865

-- Define the number of classes and students
def fourth_grade_classes : ℕ := 3
def students_per_fourth_grade_class : ℕ := 30
def pe_classes : ℕ := 1
def students_per_pe_class : ℕ := 50

-- Calculate the total number of cupcakes needed
def total_cupcakes_needed : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) +
  (pe_classes * students_per_pe_class)

-- Statement to prove
theorem jessa_cupcakes : total_cupcakes_needed = 140 :=
by
  sorry

end NUMINAMATH_GPT_jessa_cupcakes_l2418_241865


namespace NUMINAMATH_GPT_major_axis_length_l2418_241815

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def foci_1 : ℝ × ℝ := (3, 5)
def foci_2 : ℝ × ℝ := (23, 40)
def reflected_foci_1 : ℝ × ℝ := (-3, 5)

theorem major_axis_length :
  distance (reflected_foci_1.1) (reflected_foci_1.2) (foci_2.1) (foci_2.2) = Real.sqrt 1921 :=
sorry

end NUMINAMATH_GPT_major_axis_length_l2418_241815


namespace NUMINAMATH_GPT_arithmetic_evaluation_l2418_241863

theorem arithmetic_evaluation : (10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3) = -4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_evaluation_l2418_241863


namespace NUMINAMATH_GPT_new_average_production_l2418_241897

theorem new_average_production (n : ℕ) (average_past today : ℕ) (h₁ : average_past = 70) (h₂ : today = 90) (h₃ : n = 3) : 
  (average_past * n + today) / (n + 1) = 75 := by
  sorry

end NUMINAMATH_GPT_new_average_production_l2418_241897


namespace NUMINAMATH_GPT_find_exact_speed_l2418_241828

variable (d t v : ℝ)

-- Conditions as Lean definitions
def distance_eq1 : d = 50 * (t - 1/12) := sorry
def distance_eq2 : d = 70 * (t + 1/12) := sorry
def travel_time : t = 1/2 := sorry -- deduced travel time from the equations and given conditions
def correct_speed : v = 42 := sorry -- Mr. Bird needs to drive at 42 mph to be exactly on time

-- Lean 4 statement proving the required speed is 42 mph
theorem find_exact_speed : v = d / t :=
  by
    sorry

end NUMINAMATH_GPT_find_exact_speed_l2418_241828


namespace NUMINAMATH_GPT_bars_per_set_correct_l2418_241843

-- Define the total number of metal bars and the number of sets
def total_metal_bars : ℕ := 14
def number_of_sets : ℕ := 2

-- Define the function to compute bars per set
def bars_per_set (total_bars : ℕ) (sets : ℕ) : ℕ :=
  total_bars / sets

-- The proof statement
theorem bars_per_set_correct : bars_per_set total_metal_bars number_of_sets = 7 := by
  sorry

end NUMINAMATH_GPT_bars_per_set_correct_l2418_241843


namespace NUMINAMATH_GPT_scientific_notation_11580000_l2418_241854

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_11580000_l2418_241854


namespace NUMINAMATH_GPT_unique_solutions_l2418_241842

noncomputable def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ∣ (b^4 + 1) ∧ b ∣ (a^4 + 1) ∧ (Nat.floor (Real.sqrt a) = Nat.floor (Real.sqrt b))

theorem unique_solutions :
  ∀ (a b : ℕ), is_solution a b → (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by 
  sorry

end NUMINAMATH_GPT_unique_solutions_l2418_241842


namespace NUMINAMATH_GPT_area_S_inequality_l2418_241804

noncomputable def F (t : ℝ) : ℝ := 2 * (t - ⌊t⌋)

def S (t : ℝ) : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 - F t) * (p.1 - F t) + p.2 * p.2 ≤ (F t) * (F t) }

theorem area_S_inequality (t : ℝ) : 0 ≤ π * (F t) ^ 2 ∧ π * (F t) ^ 2 ≤ 4 * π := 
by sorry

end NUMINAMATH_GPT_area_S_inequality_l2418_241804


namespace NUMINAMATH_GPT_money_conditions_l2418_241814

theorem money_conditions (c d : ℝ) (h1 : 7 * c - d > 80) (h2 : 4 * c + d = 44) (h3 : d < 2 * c) :
  c > 124 / 11 ∧ d < 2 * c ∧ d = 12 :=
by
  sorry

end NUMINAMATH_GPT_money_conditions_l2418_241814


namespace NUMINAMATH_GPT_pool_ratio_l2418_241877

theorem pool_ratio 
  (total_pools : ℕ)
  (ark_athletic_wear_pools : ℕ)
  (total_pools_eq : total_pools = 800)
  (ark_athletic_wear_pools_eq : ark_athletic_wear_pools = 200)
  : ((total_pools - ark_athletic_wear_pools) / ark_athletic_wear_pools) = 3 :=
by
  sorry

end NUMINAMATH_GPT_pool_ratio_l2418_241877


namespace NUMINAMATH_GPT_teena_distance_behind_poe_l2418_241819

theorem teena_distance_behind_poe (D : ℝ)
    (teena_speed : ℝ) (poe_speed : ℝ)
    (time_hours : ℝ) (teena_ahead : ℝ) :
    teena_speed = 55 
    → poe_speed = 40 
    → time_hours = 1.5 
    → teena_ahead = 15 
    → D + teena_ahead = (teena_speed - poe_speed) * time_hours 
    → D = 7.5 := 
by 
    intros 
    sorry

end NUMINAMATH_GPT_teena_distance_behind_poe_l2418_241819


namespace NUMINAMATH_GPT_complex_division_l2418_241880

def i_units := Complex.I

def numerator := (3 : ℂ) + i_units
def denominator := (1 : ℂ) + i_units
def expected_result := (2 : ℂ) - i_units

theorem complex_division :
  numerator / denominator = expected_result :=
by sorry

end NUMINAMATH_GPT_complex_division_l2418_241880


namespace NUMINAMATH_GPT_garden_area_maximal_l2418_241882

/-- Given a garden with sides 20 meters, 16 meters, 12 meters, and 10 meters, 
    prove that the area is approximately 194.4 square meters. -/
theorem garden_area_maximal (a b c d : ℝ) (h1 : a = 20) (h2 : b = 16) (h3 : c = 12) (h4 : d = 10) :
    ∃ A : ℝ, abs (A - 194.4) < 0.1 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_maximal_l2418_241882


namespace NUMINAMATH_GPT_function_increasing_on_interval_l2418_241810

theorem function_increasing_on_interval {x : ℝ} (hx : x < 1) : 
  (-1/2) * x^2 + x + 4 < -1/2 * (x + 1)^2 + (x + 1) + 4 :=
sorry

end NUMINAMATH_GPT_function_increasing_on_interval_l2418_241810


namespace NUMINAMATH_GPT_molecular_weight_compound_l2418_241872

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def num_H : ℝ := 1
def num_Br : ℝ := 1
def num_O : ℝ := 3

def molecular_weight (num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O : ℝ) : ℝ :=
  (num_H * atomic_weight_H) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)

theorem molecular_weight_compound : molecular_weight num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O = 128.91 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_compound_l2418_241872


namespace NUMINAMATH_GPT_opposite_of_neg_three_sevenths_l2418_241826

theorem opposite_of_neg_three_sevenths:
  ∀ x : ℚ, (x = -3 / 7) → (∃ y : ℚ, y + x = 0 ∧ y = 3 / 7) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_sevenths_l2418_241826


namespace NUMINAMATH_GPT_Lauryn_earnings_l2418_241830

variables (L : ℝ)

theorem Lauryn_earnings (h1 : 0.70 * L + L = 3400) : L = 2000 :=
sorry

end NUMINAMATH_GPT_Lauryn_earnings_l2418_241830


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2418_241851

theorem solve_equation_1 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6 := sorry

theorem solve_equation_2 (x : ℝ) : (1 / 2) * (x - 1)^3 = -4 ↔ x = -1 := sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2418_241851


namespace NUMINAMATH_GPT_num_of_lists_is_correct_l2418_241875

theorem num_of_lists_is_correct :
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  total_lists = 50625 :=
by
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  show total_lists = 50625
  sorry

end NUMINAMATH_GPT_num_of_lists_is_correct_l2418_241875


namespace NUMINAMATH_GPT_rice_difference_on_15th_and_first_10_squares_l2418_241896

-- Definitions
def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ := 
  (3 * (3^n - 1)) / (3 - 1)

-- Theorem statement
theorem rice_difference_on_15th_and_first_10_squares :
  grains_on_square 15 - sum_first_n_squares 10 = 14260335 :=
by
  sorry

end NUMINAMATH_GPT_rice_difference_on_15th_and_first_10_squares_l2418_241896


namespace NUMINAMATH_GPT_find_fg_l2418_241822

def f (x : ℕ) : ℕ := 3 * x^2 + 2
def g (x : ℕ) : ℕ := 4 * x + 1

theorem find_fg :
  f (g 3) = 509 :=
by
  sorry

end NUMINAMATH_GPT_find_fg_l2418_241822


namespace NUMINAMATH_GPT_common_root_solutions_l2418_241856

theorem common_root_solutions (a : ℝ) (b : ℝ) :
  (a^2 * b^2 + a * b - 1 = 0) ∧ (b^2 - a * b - a^2 = 0) →
  a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
  a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_common_root_solutions_l2418_241856


namespace NUMINAMATH_GPT_polynomial_at_x_is_minus_80_l2418_241832

def polynomial (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def x_value : ℤ := 2

theorem polynomial_at_x_is_minus_80 : polynomial x_value = -80 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_at_x_is_minus_80_l2418_241832


namespace NUMINAMATH_GPT_count_integers_in_interval_l2418_241870

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end NUMINAMATH_GPT_count_integers_in_interval_l2418_241870


namespace NUMINAMATH_GPT_simplify_product_of_fractions_l2418_241874

theorem simplify_product_of_fractions :
  (252 / 21) * (7 / 168) * (12 / 4) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_product_of_fractions_l2418_241874


namespace NUMINAMATH_GPT_fraction_home_l2418_241867

-- Defining the conditions
def fractionFun := 5 / 13
def fractionYouth := 4 / 13

-- Stating the theorem to be proven
theorem fraction_home : 1 - (fractionFun + fractionYouth) = 4 / 13 := by
  sorry

end NUMINAMATH_GPT_fraction_home_l2418_241867


namespace NUMINAMATH_GPT_problem_statements_l2418_241889

noncomputable def f (x : ℕ) : ℕ := x % 2
noncomputable def g (x : ℕ) : ℕ := x % 3

theorem problem_statements (x : ℕ) : (f (2 * x) = 0) ∧ (f x + f (x + 3) = 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_statements_l2418_241889


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2418_241879

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 > 1) ∧ ¬(x^2 > 1 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2418_241879


namespace NUMINAMATH_GPT_find_number_to_be_multiplied_l2418_241831

-- Define the conditions of the problem
variable (x : ℕ)

-- Condition 1: The correct multiplication would have been 43x
-- Condition 2: The actual multiplication done was 34x
-- Condition 3: The difference between correct and actual result is 1242

theorem find_number_to_be_multiplied (h : 43 * x - 34 * x = 1242) : 
  x = 138 := by
  sorry

end NUMINAMATH_GPT_find_number_to_be_multiplied_l2418_241831


namespace NUMINAMATH_GPT_parallel_lines_a_value_l2418_241873

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ((a + 1) * x + 3 * y + 3 = 0) → (x + (a - 1) * y + 1 = 0)) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_value_l2418_241873


namespace NUMINAMATH_GPT_rectangular_plot_breadth_l2418_241818

theorem rectangular_plot_breadth :
  ∀ (l b : ℝ), (l = 3 * b) → (l * b = 588) → (b = 14) :=
by
  intros l b h1 h2
  sorry

end NUMINAMATH_GPT_rectangular_plot_breadth_l2418_241818


namespace NUMINAMATH_GPT_bus_ride_cost_l2418_241862

/-- The cost of a bus ride from town P to town Q, given that the cost of a train ride is $2.35 more 
    than a bus ride, and the combined cost of one train ride and one bus ride is $9.85. -/
theorem bus_ride_cost (B : ℝ) (h1 : ∃T, T = B + 2.35) (h2 : ∃T, T + B = 9.85) : B = 3.75 :=
by
  obtain ⟨T1, hT1⟩ := h1
  obtain ⟨T2, hT2⟩ := h2
  simp only [hT1, add_right_inj] at hT2
  sorry

end NUMINAMATH_GPT_bus_ride_cost_l2418_241862


namespace NUMINAMATH_GPT_carson_clawed_total_l2418_241838

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end NUMINAMATH_GPT_carson_clawed_total_l2418_241838


namespace NUMINAMATH_GPT_units_digit_7_pow_6_pow_5_l2418_241824

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_6_pow_5_l2418_241824


namespace NUMINAMATH_GPT_number_of_foons_correct_l2418_241834

-- Define the conditions
def area : ℝ := 5  -- Area in cm^2
def thickness : ℝ := 0.5  -- Thickness in cm
def total_volume : ℝ := 50  -- Total volume in cm^3

-- Define the proof problem
theorem number_of_foons_correct :
  (total_volume / (area * thickness) = 20) :=
by
  -- The necessary computation would go here, but for now we'll use sorry to indicate the outcome
  sorry

end NUMINAMATH_GPT_number_of_foons_correct_l2418_241834


namespace NUMINAMATH_GPT_repeating_decimal_fraction_difference_l2418_241888

theorem repeating_decimal_fraction_difference :
  ∀ (F : ℚ),
  F = 817 / 999 → (999 - 817 = 182) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_difference_l2418_241888


namespace NUMINAMATH_GPT_population_after_panic_l2418_241881

noncomputable def original_population : ℕ := 7200
def first_event_loss (population : ℕ) : ℕ := population * 10 / 100
def after_first_event (population : ℕ) : ℕ := population - first_event_loss population
def second_event_loss (population : ℕ) : ℕ := population * 25 / 100
def after_second_event (population : ℕ) : ℕ := population - second_event_loss population

theorem population_after_panic : after_second_event (after_first_event original_population) = 4860 := sorry

end NUMINAMATH_GPT_population_after_panic_l2418_241881


namespace NUMINAMATH_GPT_rowing_upstream_speed_l2418_241809

def speed_in_still_water : ℝ := 31
def speed_downstream : ℝ := 37

def speed_stream : ℝ := speed_downstream - speed_in_still_water

def speed_upstream : ℝ := speed_in_still_water - speed_stream

theorem rowing_upstream_speed :
  speed_upstream = 25 := by
  sorry

end NUMINAMATH_GPT_rowing_upstream_speed_l2418_241809


namespace NUMINAMATH_GPT_f_at_2_f_pos_solution_set_l2418_241895

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - (3 - a) * x + 2 * (1 - a)

-- Question (I)
theorem f_at_2 : f a 2 = 0 := by sorry

-- Question (II)
theorem f_pos_solution_set :
  (∀ x, (a < -1 → (f a x > 0 ↔ (x < 2 ∨ 1 - a < x))) ∧
       (a = -1 → ¬(f a x > 0)) ∧
       (a > -1 → (f a x > 0 ↔ (1 - a < x ∧ x < 2)))) := 
by sorry

end NUMINAMATH_GPT_f_at_2_f_pos_solution_set_l2418_241895


namespace NUMINAMATH_GPT_inverse_composition_has_correct_value_l2418_241800

noncomputable def f (x : ℝ) : ℝ := 5 * x + 7
noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 5

theorem inverse_composition_has_correct_value : 
  f_inv (f_inv 9) = -33 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_inverse_composition_has_correct_value_l2418_241800


namespace NUMINAMATH_GPT_union_A_B_complement_U_A_intersection_B_range_of_a_l2418_241850

-- Define the sets A, B, C, and U
def setA (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 8
def setB (x : ℝ) : Prop := 1 < x ∧ x < 6
def setC (a : ℝ) (x : ℝ) : Prop := x > a
def U (x : ℝ) : Prop := True  -- U being the universal set of all real numbers

-- Define complements and intersections
def complement (A : ℝ → Prop) (x : ℝ) : Prop := ¬ A x
def intersection (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x

-- Proof problems
theorem union_A_B : ∀ x, union setA setB x ↔ (1 < x ∧ x ≤ 8) :=
by 
  intros x
  sorry

theorem complement_U_A_intersection_B : ∀ x, intersection (complement setA) setB x ↔ (1 < x ∧ x < 2) :=
by 
  intros x
  sorry

theorem range_of_a (a : ℝ) : (∃ x, intersection setA (setC a) x) → a < 8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_union_A_B_complement_U_A_intersection_B_range_of_a_l2418_241850


namespace NUMINAMATH_GPT_minimum_value_of_x_y_l2418_241813

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  x + y

theorem minimum_value_of_x_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 - x) * (-y) = x) : minimum_value x y = 4 :=
  sorry

end NUMINAMATH_GPT_minimum_value_of_x_y_l2418_241813


namespace NUMINAMATH_GPT_find_x_l2418_241808

theorem find_x :
  (2 + 3 = 5) →
  (3 + 4 = 7) →
  (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) →
  x = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_x_l2418_241808


namespace NUMINAMATH_GPT_cannot_divide_1980_into_four_groups_l2418_241820

theorem cannot_divide_1980_into_four_groups :
  ¬∃ (S₁ S₂ S₃ S₄ : ℕ),
    S₂ = S₁ + 10 ∧
    S₃ = S₂ + 10 ∧
    S₄ = S₃ + 10 ∧
    (1 + 1980) * 1980 / 2 = S₁ + S₂ + S₃ + S₄ := 
sorry

end NUMINAMATH_GPT_cannot_divide_1980_into_four_groups_l2418_241820


namespace NUMINAMATH_GPT_x_fifth_power_sum_l2418_241885

theorem x_fifth_power_sum (x : ℝ) (h : x + 1 / x = -5) : x^5 + 1 / x^5 = -2525 := by
  sorry

end NUMINAMATH_GPT_x_fifth_power_sum_l2418_241885


namespace NUMINAMATH_GPT_relay_race_total_time_correct_l2418_241823

-- Conditions as definitions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25
def athlete5_time : ℕ := 80
def athlete6_time : ℕ := athlete5_time - 20
def athlete7_time : ℕ := 70
def athlete8_time : ℕ := athlete7_time - 5

-- Sum of all athletes' times
def total_time : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time + athlete5_time +
  athlete6_time + athlete7_time + athlete8_time

-- Statement to prove
theorem relay_race_total_time_correct : total_time = 475 :=
  by
  sorry

end NUMINAMATH_GPT_relay_race_total_time_correct_l2418_241823


namespace NUMINAMATH_GPT_find_b8_l2418_241803

noncomputable section

def increasing_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

axiom b_seq : ℕ → ℕ

axiom seq_inc : increasing_sequence b_seq

axiom b7_eq : b_seq 7 = 198

theorem find_b8 : b_seq 8 = 321 := by
  sorry

end NUMINAMATH_GPT_find_b8_l2418_241803


namespace NUMINAMATH_GPT_area_of_shaded_region_l2418_241821

def radius_of_first_circle : ℝ := 4
def radius_of_second_circle : ℝ := 5
def radius_of_third_circle : ℝ := 2
def radius_of_fourth_circle : ℝ := 9

theorem area_of_shaded_region :
  π * (radius_of_fourth_circle ^ 2) - π * (radius_of_first_circle ^ 2) - π * (radius_of_second_circle ^ 2) - π * (radius_of_third_circle ^ 2) = 36 * π :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_shaded_region_l2418_241821


namespace NUMINAMATH_GPT_sqrt_112_consecutive_integers_product_l2418_241861

theorem sqrt_112_consecutive_integers_product : 
  (∃ (a b : ℕ), a * a < 112 ∧ 112 < b * b ∧ b = a + 1 ∧ a * b = 110) :=
by 
  use 10, 11
  repeat { sorry }

end NUMINAMATH_GPT_sqrt_112_consecutive_integers_product_l2418_241861


namespace NUMINAMATH_GPT_visiting_plans_correct_l2418_241864

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of places to visit
def num_places : ℕ := 3

-- Define the total number of visiting plans without any restrictions
def total_visiting_plans : ℕ := num_places ^ num_students

-- Define the number of visiting plans where no one visits Haxi Station
def no_haxi_visiting_plans : ℕ := (num_places - 1) ^ num_students

-- Define the number of visiting plans where Haxi Station has at least one visitor
def visiting_plans_with_haxi : ℕ := total_visiting_plans - no_haxi_visiting_plans

-- Prove that the number of different visiting plans with at least one student visiting Haxi Station is 65
theorem visiting_plans_correct : visiting_plans_with_haxi = 65 := by
  -- Omitted proof
  sorry

end NUMINAMATH_GPT_visiting_plans_correct_l2418_241864


namespace NUMINAMATH_GPT_problem_solution_l2418_241817
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.foldl (· + ·) 0

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_seq : ℕ → ℕ → ℕ
| 0, n => f n
| (k+1), n => f (f_seq k n)

theorem problem_solution :
  f_seq 2016 9 = 8 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2418_241817


namespace NUMINAMATH_GPT_price_of_each_shirt_is_15_30_l2418_241898

theorem price_of_each_shirt_is_15_30:
  ∀ (shorts_price : ℝ) (num_shorts : ℕ) (shirt_num : ℕ) (total_paid : ℝ) (discount : ℝ),
  shorts_price = 15 →
  num_shorts = 3 →
  shirt_num = 5 →
  total_paid = 117 →
  discount = 0.10 →
  (total_paid - (num_shorts * shorts_price - discount * (num_shorts * shorts_price))) / shirt_num = 15.30 :=
by 
  sorry

end NUMINAMATH_GPT_price_of_each_shirt_is_15_30_l2418_241898


namespace NUMINAMATH_GPT_smallest_lcm_l2418_241825

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end NUMINAMATH_GPT_smallest_lcm_l2418_241825


namespace NUMINAMATH_GPT_first_term_arithmetic_series_l2418_241836

theorem first_term_arithmetic_series 
  (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 240)
  (h2 : 30 * (2 * a + 179 * d) = 3600) : 
  a = -353 / 15 :=
by
  have eq1 : 2 * a + 59 * d = 8 := by sorry
  have eq2 : 2 * a + 179 * d = 120 := by sorry
  sorry

end NUMINAMATH_GPT_first_term_arithmetic_series_l2418_241836


namespace NUMINAMATH_GPT_sphere_volume_given_surface_area_l2418_241860

theorem sphere_volume_given_surface_area (r : ℝ) (V : ℝ) (S : ℝ)
  (hS : S = 36 * Real.pi)
  (h_surface_area : 4 * Real.pi * r^2 = S)
  (h_volume : V = (4/3) * Real.pi * r^3) : V = 36 * Real.pi := by
  sorry

end NUMINAMATH_GPT_sphere_volume_given_surface_area_l2418_241860


namespace NUMINAMATH_GPT_striped_shirts_more_than_shorts_l2418_241839

theorem striped_shirts_more_than_shorts :
  ∀ (total_students striped_students checkered_students short_students : ℕ),
    total_students = 81 →
    striped_students = total_students * 2 / 3 →
    checkered_students = total_students - striped_students →
    short_students = checkered_students + 19 →
    striped_students - short_students = 8 :=
by
  intros total_students striped_students checkered_students short_students
  sorry

end NUMINAMATH_GPT_striped_shirts_more_than_shorts_l2418_241839


namespace NUMINAMATH_GPT_find_a_tangent_line_l2418_241805

theorem find_a_tangent_line (a : ℝ) : 
  (∃ (x0 y0 : ℝ), y0 = a * x0^2 + (15/4 : ℝ) * x0 - 9 ∧ 
                  (y0 = 0 ∨ (x0 = 3/2 ∧ y0 = 27/4)) ∧ 
                  ∃ (m : ℝ), (0 - y0) = m * (1 - x0) ∧ (m = 2 * a * x0 + 15/4)) → 
  (a = -1 ∨ a = -25/64) := 
sorry

end NUMINAMATH_GPT_find_a_tangent_line_l2418_241805


namespace NUMINAMATH_GPT_average_age_of_team_l2418_241845

variable (A : ℕ)
variable (captain_age : ℕ)
variable (wicket_keeper_age : ℕ)
variable (vice_captain_age : ℕ)

-- Conditions
def team_size := 11
def captain := 25
def wicket_keeper := captain + 3
def vice_captain := wicket_keeper - 4
def remaining_players := team_size - 3
def remaining_average := A - 1

-- Prove the average age of the whole team
theorem average_age_of_team :
  captain_age = 25 ∧
  wicket_keeper_age = captain_age + 3 ∧
  vice_captain_age = wicket_keeper_age - 4 ∧
  11 * A = (captain + wicket_keeper + vice_captain) + 8 * (A - 1) → 
  A = 23 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_team_l2418_241845


namespace NUMINAMATH_GPT_factorization_eq_l2418_241892

theorem factorization_eq (x : ℝ) : 
  -3 * x^3 + 12 * x^2 - 12 * x = -3 * x * (x - 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_eq_l2418_241892


namespace NUMINAMATH_GPT_trigonometric_identity_l2418_241890

-- Define the problem conditions and formulas
variables (α : Real) (h : Real.cos (Real.pi / 6 + α) = Real.sqrt 3 / 3)

-- State the theorem
theorem trigonometric_identity : Real.cos (5 * Real.pi / 6 - α) = - (Real.sqrt 3 / 3) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2418_241890


namespace NUMINAMATH_GPT_fifth_group_members_l2418_241846

-- Define the number of members in the choir
def total_members : ℕ := 150 

-- Define the number of members in each group
def group1 : ℕ := 18 
def group2 : ℕ := 29 
def group3 : ℕ := 34 
def group4 : ℕ := 23 

-- Define the fifth group as the remaining members
def group5 : ℕ := total_members - (group1 + group2 + group3 + group4)

theorem fifth_group_members : group5 = 46 := sorry

end NUMINAMATH_GPT_fifth_group_members_l2418_241846


namespace NUMINAMATH_GPT_total_revenue_is_405_l2418_241802

-- Define the cost of rentals
def canoeCost : ℕ := 15
def kayakCost : ℕ := 18

-- Define terms for number of rentals
variables (C K : ℕ)

-- Conditions
axiom ratio_condition : 2 * C = 3 * K
axiom difference_condition : C = K + 5

-- Total revenue
def totalRevenue (C K : ℕ) : ℕ := (canoeCost * C) + (kayakCost * K)

-- Theorem statement
theorem total_revenue_is_405 (C K : ℕ) (H1 : 2 * C = 3 * K) (H2 : C = K + 5) : 
  totalRevenue C K = 405 := by
  sorry

end NUMINAMATH_GPT_total_revenue_is_405_l2418_241802


namespace NUMINAMATH_GPT_smallest_n_for_terminating_fraction_l2418_241884

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end NUMINAMATH_GPT_smallest_n_for_terminating_fraction_l2418_241884


namespace NUMINAMATH_GPT_evaluate_expression_l2418_241866

theorem evaluate_expression (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2418_241866


namespace NUMINAMATH_GPT_solve_inequality_l2418_241891

theorem solve_inequality (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ x ∈ Set.Ioo (-1 / 3 : ℝ) 1 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2418_241891


namespace NUMINAMATH_GPT_tenth_term_is_26_l2418_241894

-- Definitions used from the conditions
def first_term : ℤ := 8
def common_difference : ℤ := 2
def term_number : ℕ := 10

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Proving that the 10th term is 26 given the conditions
theorem tenth_term_is_26 : nth_term first_term common_difference term_number = 26 := by
  sorry

end NUMINAMATH_GPT_tenth_term_is_26_l2418_241894


namespace NUMINAMATH_GPT_boys_passed_percentage_l2418_241868

theorem boys_passed_percentage
  (total_candidates : ℝ)
  (total_girls : ℝ)
  (failed_percentage : ℝ)
  (girls_passed_percentage : ℝ)
  (boys_passed_percentage : ℝ) :
  total_candidates = 2000 →
  total_girls = 900 →
  failed_percentage = 70.2 →
  girls_passed_percentage = 32 →
  boys_passed_percentage = 28 :=
by
  sorry

end NUMINAMATH_GPT_boys_passed_percentage_l2418_241868


namespace NUMINAMATH_GPT_two_times_difference_eq_20_l2418_241829

theorem two_times_difference_eq_20 (x y : ℕ) (hx : x = 30) (hy : y = 20) (hsum : x + y = 50) : 2 * (x - y) = 20 := by
  sorry

end NUMINAMATH_GPT_two_times_difference_eq_20_l2418_241829


namespace NUMINAMATH_GPT_mass_percentage_of_Cl_in_NaOCl_l2418_241859

theorem mass_percentage_of_Cl_in_NaOCl :
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  100 * (Cl_mass / NaOCl_mass) = 47.6 := 
by
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Cl_in_NaOCl_l2418_241859


namespace NUMINAMATH_GPT_chloe_cherries_l2418_241887

noncomputable def cherries_received (x y : ℝ) : Prop :=
  x = y + 8 ∧ y = x / 3

theorem chloe_cherries : ∃ (x : ℝ), ∀ (y : ℝ), cherries_received x y → x = 12 := 
by
  sorry

end NUMINAMATH_GPT_chloe_cherries_l2418_241887


namespace NUMINAMATH_GPT_f_is_odd_f_is_decreasing_range_of_m_l2418_241855

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) : f (-x) = - f x := by
  sorry

-- Prove that f(x) is decreasing on ℝ
theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  sorry

-- Prove the range of m if f(m-1) + f(2m-1) > 0
theorem range_of_m (m : ℝ) (h : f (m - 1) + f (2 * m - 1) > 0) : m < 2 / 3 := by
  sorry

end NUMINAMATH_GPT_f_is_odd_f_is_decreasing_range_of_m_l2418_241855


namespace NUMINAMATH_GPT_time_to_cover_escalator_l2418_241853

variable (v_e v_p L : ℝ)

theorem time_to_cover_escalator
  (h_v_e : v_e = 15)
  (h_v_p : v_p = 5)
  (h_L : L = 180) :
  (L / (v_e + v_p) = 9) :=
by
  -- Set up the given conditions
  rw [h_v_e, h_v_p, h_L]
  -- This will now reduce to proving 180 / (15 + 5) = 9
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l2418_241853


namespace NUMINAMATH_GPT_min_value_fraction_l2418_241858

theorem min_value_fraction (a b : ℝ) (h : x^2 - 3*x + a*b < 0 ∧ 1 < x ∧ x < 2) (h1 : a > b) : 
  (∃ minValue : ℝ, minValue = 4 ∧ ∀ a b : ℝ, a > b → minValue ≤ (a^2 + b^2) / (a - b)) := 
sorry

end NUMINAMATH_GPT_min_value_fraction_l2418_241858


namespace NUMINAMATH_GPT_tan_theta_determined_l2418_241840

theorem tan_theta_determined (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4) (h_zero : Real.tan θ + Real.tan (4 * θ) = 0) :
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_GPT_tan_theta_determined_l2418_241840


namespace NUMINAMATH_GPT_product_equals_sum_only_in_two_cases_l2418_241827

theorem product_equals_sum_only_in_two_cases (x y : ℤ) : 
  x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by 
  sorry

end NUMINAMATH_GPT_product_equals_sum_only_in_two_cases_l2418_241827


namespace NUMINAMATH_GPT_roots_equation_value_l2418_241807

theorem roots_equation_value (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α + β = 1) :
    α^4 + 3 * β = 5 := by
sorry

end NUMINAMATH_GPT_roots_equation_value_l2418_241807


namespace NUMINAMATH_GPT_perfect_squares_l2418_241841

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end NUMINAMATH_GPT_perfect_squares_l2418_241841


namespace NUMINAMATH_GPT_position_after_steps_l2418_241812

def equally_spaced_steps (total_distance num_steps distance_per_step steps_taken : ℕ) : Prop :=
  total_distance = num_steps * distance_per_step ∧ 
  ∀ k : ℕ, k ≤ num_steps → k * distance_per_step = distance_per_step * k

theorem position_after_steps (total_distance num_steps distance_per_step steps_taken : ℕ) 
  (h_eq : equally_spaced_steps total_distance num_steps distance_per_step steps_taken) 
  (h_total : total_distance = 32) (h_num : num_steps = 8) (h_steps : steps_taken = 6) : 
  steps_taken * (total_distance / num_steps) = 24 := 
by 
  sorry

end NUMINAMATH_GPT_position_after_steps_l2418_241812


namespace NUMINAMATH_GPT_projectiles_meet_time_l2418_241848

def distance : ℕ := 2520
def speed1 : ℕ := 432
def speed2 : ℕ := 576
def combined_speed : ℕ := speed1 + speed2

theorem projectiles_meet_time :
  (distance * 60) / combined_speed = 150 := 
by
  sorry

end NUMINAMATH_GPT_projectiles_meet_time_l2418_241848


namespace NUMINAMATH_GPT_trading_cards_initial_total_l2418_241876

theorem trading_cards_initial_total (x : ℕ) 
  (h1 : ∃ d : ℕ, d = (1 / 3 : ℕ) * x)
  (h2 : ∃ n1 : ℕ, n1 = (1 / 5 : ℕ) * (1 / 3 : ℕ) * x)
  (h3 : ∃ n2 : ℕ, n2 = (1 / 3 : ℕ) * ((1 / 5 : ℕ) * (1 / 3 : ℕ) * x))
  (h4 : ∃ n3 : ℕ, n3 = (1 / 2 : ℕ) * (2 / 45 : ℕ) * x)
  (h5 : (1 / 15 : ℕ) * x + (2 / 45 : ℕ) * x + (1 / 45 : ℕ) * x = 850) :
  x = 6375 := 
sorry

end NUMINAMATH_GPT_trading_cards_initial_total_l2418_241876


namespace NUMINAMATH_GPT_fourth_student_number_systematic_sampling_l2418_241801

theorem fourth_student_number_systematic_sampling :
  ∀ (students : Finset ℕ), students = Finset.range 55 →
  ∀ (sample_size : ℕ), sample_size = 4 →
  ∀ (numbers_in_sample : Finset ℕ),
  numbers_in_sample = {3, 29, 42} →
  ∃ (fourth_student : ℕ), fourth_student = 44 :=
  by sorry

end NUMINAMATH_GPT_fourth_student_number_systematic_sampling_l2418_241801


namespace NUMINAMATH_GPT_negation_proof_l2418_241883

theorem negation_proof :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_negation_proof_l2418_241883


namespace NUMINAMATH_GPT_ratio_son_grandson_l2418_241869

-- Define the conditions
variables (Markus_age Son_age Grandson_age : ℕ)
axiom Markus_twice_son : Markus_age = 2 * Son_age
axiom sum_ages : Markus_age + Son_age + Grandson_age = 140
axiom Grandson_age_20 : Grandson_age = 20

-- Define the goal to prove
theorem ratio_son_grandson : (Son_age : ℚ) / Grandson_age = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_son_grandson_l2418_241869


namespace NUMINAMATH_GPT_suraj_avg_after_10th_inning_l2418_241844

theorem suraj_avg_after_10th_inning (A : ℝ) 
  (h1 : ∀ A : ℝ, (9 * A + 200) / 10 = A + 8) :
  ∀ A : ℝ, A = 120 → (A + 8 = 128) :=
by
  sorry

end NUMINAMATH_GPT_suraj_avg_after_10th_inning_l2418_241844


namespace NUMINAMATH_GPT_sum_of_fractions_l2418_241852

theorem sum_of_fractions :
  (1 / (1^2 * 2^2) + 1 / (2^2 * 3^2) + 1 / (3^2 * 4^2) + 1 / (4^2 * 5^2)
  + 1 / (5^2 * 6^2) + 1 / (6^2 * 7^2)) = 48 / 49 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l2418_241852


namespace NUMINAMATH_GPT_base_b_digit_sum_l2418_241833

theorem base_b_digit_sum :
  ∃ (b : ℕ), ((b^2 / 2 + b / 2) % b = 2) ∧ (b = 8) :=
by
  sorry

end NUMINAMATH_GPT_base_b_digit_sum_l2418_241833


namespace NUMINAMATH_GPT_implies_neg_p_and_q_count_l2418_241806

-- Definitions of the logical conditions
variables (p q : Prop)

def cond1 : Prop := p ∧ q
def cond2 : Prop := p ∧ ¬ q
def cond3 : Prop := ¬ p ∧ q
def cond4 : Prop := ¬ p ∧ ¬ q

-- Negative of the statement "p and q are both true"
def neg_p_and_q := ¬ (p ∧ q)

-- The Lean 4 statement to prove
theorem implies_neg_p_and_q_count :
  (cond2 p q → neg_p_and_q p q) ∧ 
  (cond3 p q → neg_p_and_q p q) ∧ 
  (cond4 p q → neg_p_and_q p q) ∧ 
  ¬ (cond1 p q → neg_p_and_q p q) :=
sorry

end NUMINAMATH_GPT_implies_neg_p_and_q_count_l2418_241806


namespace NUMINAMATH_GPT_total_amount_is_correct_l2418_241886

-- Definitions based on the conditions
def share_a (x : ℕ) : ℕ := 2 * x
def share_b (x : ℕ) : ℕ := 4 * x
def share_c (x : ℕ) : ℕ := 5 * x
def share_d (x : ℕ) : ℕ := 4 * x

-- Condition: combined share of a and b is 1800
def combined_share_of_ab (x : ℕ) : Prop := share_a x + share_b x = 1800

-- Theorem we want to prove: Total amount given to all children is $4500
theorem total_amount_is_correct (x : ℕ) (h : combined_share_of_ab x) : 
  share_a x + share_b x + share_c x + share_d x = 4500 := sorry

end NUMINAMATH_GPT_total_amount_is_correct_l2418_241886


namespace NUMINAMATH_GPT_simplify_product_of_fractions_l2418_241878

theorem simplify_product_of_fractions :
  8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_product_of_fractions_l2418_241878


namespace NUMINAMATH_GPT_total_sales_is_10400_l2418_241871

-- Define the conditions
def tough_week_sales : ℝ := 800
def good_week_sales : ℝ := 2 * tough_week_sales
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the total sales function
def total_sales (good_sales : ℝ) (tough_sales : ℝ) (good_weeks : ℕ) (tough_weeks : ℕ) : ℝ :=
  good_weeks * good_sales + tough_weeks * tough_sales

-- Prove that the total sales is $10400
theorem total_sales_is_10400 : total_sales good_week_sales tough_week_sales good_weeks tough_weeks = 10400 := 
by
  sorry

end NUMINAMATH_GPT_total_sales_is_10400_l2418_241871


namespace NUMINAMATH_GPT_distance_from_stream_to_meadow_l2418_241835

noncomputable def distance_from_car_to_stream : ℝ := 0.2
noncomputable def distance_from_meadow_to_campsite : ℝ := 0.1
noncomputable def total_distance_hiked : ℝ := 0.7

theorem distance_from_stream_to_meadow : 
  (total_distance_hiked - distance_from_car_to_stream - distance_from_meadow_to_campsite = 0.4) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_stream_to_meadow_l2418_241835


namespace NUMINAMATH_GPT_ratio_of_heights_l2418_241849

def min_height := 140
def brother_height := 180
def grow_needed := 20

def mary_height := min_height - grow_needed
def height_ratio := mary_height / brother_height

theorem ratio_of_heights : height_ratio = (2 / 3) := 
  sorry

end NUMINAMATH_GPT_ratio_of_heights_l2418_241849


namespace NUMINAMATH_GPT_cost_two_enchiladas_two_tacos_three_burritos_l2418_241816

variables (e t b : ℝ)

theorem cost_two_enchiladas_two_tacos_three_burritos 
  (h1 : 2 * e + 3 * t + b = 5.00)
  (h2 : 3 * e + 2 * t + 2 * b = 7.50) : 
  2 * e + 2 * t + 3 * b = 10.625 :=
sorry

end NUMINAMATH_GPT_cost_two_enchiladas_two_tacos_three_burritos_l2418_241816


namespace NUMINAMATH_GPT_space_shuttle_speed_kmh_l2418_241837

-- Define the given conditions
def speedInKmPerSecond : ℕ := 4
def secondsInAnHour : ℕ := 3600

-- State the proof problem
theorem space_shuttle_speed_kmh : speedInKmPerSecond * secondsInAnHour = 14400 := by
  sorry

end NUMINAMATH_GPT_space_shuttle_speed_kmh_l2418_241837
