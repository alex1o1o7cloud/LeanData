import Mathlib

namespace NUMINAMATH_GPT_simplify_fraction_l145_14587

theorem simplify_fraction (num denom : ℚ) (h_num: num = (3/7 + 5/8)) (h_denom: denom = (5/12 + 2/3)) :
  (num / denom) = (177/182) := 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l145_14587


namespace NUMINAMATH_GPT_solution_correct_l145_14551

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 ≤ |x - 3| ∧ |x - 3| ≤ 5
def condition2 (x : ℝ) : Prop := (x - 3) ^ 2 ≤ 16

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7)

-- Prove that the solution set is correct given the conditions
theorem solution_correct (x : ℝ) : condition1 x ∧ condition2 x ↔ solution_set x :=
by
  sorry

end NUMINAMATH_GPT_solution_correct_l145_14551


namespace NUMINAMATH_GPT_factorize_expr_l145_14572

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l145_14572


namespace NUMINAMATH_GPT_average_class_weight_l145_14541

theorem average_class_weight :
  let students_A := 50
  let weight_A := 60
  let students_B := 60
  let weight_B := 80
  let students_C := 70
  let weight_C := 75
  let students_D := 80
  let weight_D := 85
  let total_students := students_A + students_B + students_C + students_D
  let total_weight := students_A * weight_A + students_B * weight_B + students_C * weight_C + students_D * weight_D
  (total_weight / total_students : ℝ) = 76.35 :=
by
  sorry

end NUMINAMATH_GPT_average_class_weight_l145_14541


namespace NUMINAMATH_GPT_calculate_expr_l145_14556

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end NUMINAMATH_GPT_calculate_expr_l145_14556


namespace NUMINAMATH_GPT_population_increase_time_l145_14555

theorem population_increase_time (persons_added : ℕ) (time_minutes : ℕ) (seconds_per_minute : ℕ) (total_seconds : ℕ) (time_for_one_person : ℕ) :
  persons_added = 160 →
  time_minutes = 40 →
  seconds_per_minute = 60 →
  total_seconds = time_minutes * seconds_per_minute →
  time_for_one_person = total_seconds / persons_added →
  time_for_one_person = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_population_increase_time_l145_14555


namespace NUMINAMATH_GPT_y_explicit_and_range_l145_14522

theorem y_explicit_and_range (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2*(m-1)*x1 + m + 1 = 0) (h2 : x2^2 - 2*(m-1)*x2 + m + 1 = 0) :
  x1 + x2 = 2*(m-1) ∧ x1 * x2 = m + 1 ∧ (x1^2 + x2^2 = 4*m^2 - 10*m + 2) 
  ∧ ∀ (y : ℝ), (∃ m, y = 4*m^2 - 10*m + 2) → y ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_y_explicit_and_range_l145_14522


namespace NUMINAMATH_GPT_time_to_cross_platform_is_correct_l145_14524

noncomputable def speed_of_train := 36 -- speed in km/h
noncomputable def time_to_cross_pole := 12 -- time in seconds
noncomputable def time_to_cross_platform := 49.996960243180546 -- time in seconds

theorem time_to_cross_platform_is_correct : time_to_cross_platform = 49.996960243180546 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_platform_is_correct_l145_14524


namespace NUMINAMATH_GPT_part1_part2_l145_14569

variable (x : ℝ)

def A : Set ℝ := { x | 2 * x + 1 < 5 }
def B : Set ℝ := { x | x^2 - x - 2 < 0 }

theorem part1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem part2 : A ∪ { x | x ≤ -1 ∨ x ≥ 2 } = Set.univ :=
sorry

end NUMINAMATH_GPT_part1_part2_l145_14569


namespace NUMINAMATH_GPT_reduced_price_per_kg_l145_14591

-- Definitions
variables {P R Q : ℝ}

-- Conditions
axiom reduction_price : R = P * 0.82
axiom original_quantity : Q * P = 1080
axiom reduced_quantity : (Q + 8) * R = 1080

-- Proof statement
theorem reduced_price_per_kg : R = 24.30 :=
by {
  sorry
}

end NUMINAMATH_GPT_reduced_price_per_kg_l145_14591


namespace NUMINAMATH_GPT_solve_fraction_equation_l145_14573

theorem solve_fraction_equation :
  ∀ (x : ℚ), (5 * x + 3) / (7 * x - 4) = 4128 / 4386 → x = 115 / 27 := by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l145_14573


namespace NUMINAMATH_GPT_calculation_result_l145_14553

theorem calculation_result : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end NUMINAMATH_GPT_calculation_result_l145_14553


namespace NUMINAMATH_GPT_binomial_square_expression_l145_14540

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end NUMINAMATH_GPT_binomial_square_expression_l145_14540


namespace NUMINAMATH_GPT_bianca_next_day_run_l145_14506

-- Define the conditions
variable (miles_first_day : ℕ) (total_miles : ℕ)

-- Set the conditions for Bianca's run
def conditions := miles_first_day = 8 ∧ total_miles = 12

-- State the proposition we need to prove
def miles_next_day (miles_first_day total_miles : ℕ) : ℕ := total_miles - miles_first_day

-- The theorem stating the problem to prove
theorem bianca_next_day_run (h : conditions 8 12) : miles_next_day 8 12 = 4 := by
  unfold conditions at h
  simp [miles_next_day] at h
  sorry

end NUMINAMATH_GPT_bianca_next_day_run_l145_14506


namespace NUMINAMATH_GPT_fifth_term_of_geometric_sequence_l145_14530

theorem fifth_term_of_geometric_sequence (a r : ℕ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h_a : a = 5) (h_fourth_term : a * r^3 = 405) :
  a * r^4 = 405 := by
  sorry

end NUMINAMATH_GPT_fifth_term_of_geometric_sequence_l145_14530


namespace NUMINAMATH_GPT_ratio_of_radii_l145_14561

namespace CylinderAndSphere

variable (r R : ℝ)
variable (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2)

theorem ratio_of_radii (r R : ℝ) (h_cylinder : 2 * Real.pi * r * (4 * r) = 4 * Real.pi * R ^ 2) :
    R / r = Real.sqrt 2 :=
by
  sorry

end CylinderAndSphere

end NUMINAMATH_GPT_ratio_of_radii_l145_14561


namespace NUMINAMATH_GPT_find_functions_l145_14509

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def domain (f g : ℝ → ℝ) : Prop := ∀ x, x ≠ 1 → x ≠ -1 → true

theorem find_functions
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_domain : domain f g)
  (h_eq : ∀ x, x ≠ 1 → x ≠ -1 → f x + g x = 1 / (x - 1)) :
  (∀ x, x ≠ 1 → x ≠ -1 → f x = x / (x^2 - 1)) ∧ 
  (∀ x, x ≠ 1 → x ≠ -1 → g x = 1 / (x^2 - 1)) := 
by
  sorry

end NUMINAMATH_GPT_find_functions_l145_14509


namespace NUMINAMATH_GPT_average_first_15_even_numbers_l145_14557

theorem average_first_15_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30) / 15 = 16 :=
by 
  sorry

end NUMINAMATH_GPT_average_first_15_even_numbers_l145_14557


namespace NUMINAMATH_GPT_trigonometric_identity_l145_14568

open Real

variable (α : ℝ)
variable (h1 : π < α)
variable (h2 : α < 2 * π)
variable (h3 : cos (α - 7 * π) = -3 / 5)

theorem trigonometric_identity :
  sin (3 * π + α) * tan (α - 7 * π / 2) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l145_14568


namespace NUMINAMATH_GPT_pairings_equal_l145_14531

-- Definitions for City A
def A_girls (n : ℕ) : Type := Fin n
def A_boys (n : ℕ) : Type := Fin n
def A_knows (n : ℕ) (g : A_girls n) (b : A_boys n) : Prop := True

-- Definitions for City B
def B_girls (n : ℕ) : Type := Fin n
def B_boys (n : ℕ) : Type := Fin (2 * n - 1)
def B_knows (n : ℕ) (i : Fin n) (j : Fin (2 * n - 1)) : Prop :=
  j.val < 2 * (i.val + 1)

-- Function to count the number of ways to pair r girls and r boys in city A
noncomputable def A (n r : ℕ) : ℕ := 
  if h : r ≤ n then 
    Nat.choose n r * Nat.choose n r * (r.factorial)
  else 0

-- Recurrence relation for city B
noncomputable def B (n r : ℕ) : ℕ :=
  if r = 0 then 1 else if r > n then 0 else
  if n < 2 then if r = 1 then (2 - 1) * 2 else 0 else
  B (n - 1) r + (2 * n - r) * B (n - 1) (r - 1)

-- We want to prove that number of pairings in city A equals number of pairings in city B for any r <= n
theorem pairings_equal (n r : ℕ) (h : r ≤ n) : A n r = B n r := sorry

end NUMINAMATH_GPT_pairings_equal_l145_14531


namespace NUMINAMATH_GPT_simplify_polynomial_l145_14525

variable (x : ℝ)

theorem simplify_polynomial :
  (6*x^10 + 8*x^9 + 3*x^7) + (2*x^12 + 3*x^10 + x^9 + 5*x^7 + 4*x^4 + 7*x + 6) =
  2*x^12 + 9*x^10 + 9*x^9 + 8*x^7 + 4*x^4 + 7*x + 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l145_14525


namespace NUMINAMATH_GPT_min_max_value_l145_14500

-- Definition of the function to be minimized and maximized
def f (x y : ℝ) : ℝ := |x^3 - x * y^2|

-- Conditions
def x_condition (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def y_condition (y : ℝ) : Prop := true

-- Goal: Prove the minimum of the maximum value
theorem min_max_value :
  ∃ y : ℝ, (∀ x : ℝ, x_condition x → f x y ≤ 8) ∧ (∀ y' : ℝ, (∀ x : ℝ, x_condition x → f x y' ≤ 8) → y' = y) :=
sorry

end NUMINAMATH_GPT_min_max_value_l145_14500


namespace NUMINAMATH_GPT_complete_square_l145_14546

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_l145_14546


namespace NUMINAMATH_GPT_logistics_center_correct_l145_14596

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-6, 9)
def C : ℝ × ℝ := (-3, -8)

def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_correct : 
  ∀ L : ℝ × ℝ, 
  (rectilinear_distance L A = rectilinear_distance L B) ∧ 
  (rectilinear_distance L B = rectilinear_distance L C) ∧
  (rectilinear_distance L A = rectilinear_distance L C) → 
  L = logistics_center := sorry

end NUMINAMATH_GPT_logistics_center_correct_l145_14596


namespace NUMINAMATH_GPT_box_dimensions_sum_l145_14518

theorem box_dimensions_sum (X Y Z : ℝ) (hXY : X * Y = 18) (hXZ : X * Z = 54) (hYZ : Y * Z = 36) (hX_pos : X > 0) (hY_pos : Y > 0) (hZ_pos : Z > 0) :
  X + Y + Z = 11 := 
sorry

end NUMINAMATH_GPT_box_dimensions_sum_l145_14518


namespace NUMINAMATH_GPT_find_total_cost_l145_14519

-- Define the cost per kg for flour
def F : ℕ := 21

-- Conditions in the problem
axiom cost_eq_mangos_rice (M R : ℕ) : 10 * M = 10 * R
axiom cost_eq_flour_rice (R : ℕ) : 6 * F = 2 * R

-- Define the cost calculations
def total_cost (M R F : ℕ) : ℕ := (4 * M) + (3 * R) + (5 * F)

-- Prove the total cost given the conditions
theorem find_total_cost (M R : ℕ) (h1 : 10 * M = 10 * R) (h2 : 6 * F = 2 * R) : total_cost M R F = 546 :=
sorry

end NUMINAMATH_GPT_find_total_cost_l145_14519


namespace NUMINAMATH_GPT_susan_age_indeterminate_l145_14598

-- Definitions and conditions
def james_age_in_15_years : ℕ := 37
def current_james_age : ℕ := james_age_in_15_years - 15
def james_age_8_years_ago : ℕ := current_james_age - 8
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def current_janet_age : ℕ := janet_age_8_years_ago + 8

-- Problem: Prove that without Janet's age when Susan was born, we cannot determine Susan's age in 5 years.
theorem susan_age_indeterminate (susan_current_age : ℕ) : 
  (∃ janet_age_when_susan_born : ℕ, susan_current_age = current_janet_age - janet_age_when_susan_born) → 
  ¬ (∃ susan_age_in_5_years : ℕ, susan_age_in_5_years = susan_current_age + 5) := 
by
  sorry

end NUMINAMATH_GPT_susan_age_indeterminate_l145_14598


namespace NUMINAMATH_GPT_div_result_l145_14571

theorem div_result : 2.4 / 0.06 = 40 := 
sorry

end NUMINAMATH_GPT_div_result_l145_14571


namespace NUMINAMATH_GPT_cost_of_meal_l145_14502

noncomputable def total_cost (hamburger_cost fry_cost drink_cost : ℕ) (num_hamburgers num_fries num_drinks : ℕ) (discount_rate : ℕ) : ℕ :=
  let initial_cost := (hamburger_cost * num_hamburgers) + (fry_cost * num_fries) + (drink_cost * num_drinks)
  let discount := initial_cost * discount_rate / 100
  initial_cost - discount

theorem cost_of_meal :
  total_cost 5 3 2 3 4 6 10 = 35 := by
  sorry

end NUMINAMATH_GPT_cost_of_meal_l145_14502


namespace NUMINAMATH_GPT_find_number_l145_14589

theorem find_number (N : ℝ) (h : 0.6 * (3 / 5) * N = 36) : N = 100 :=
by sorry

end NUMINAMATH_GPT_find_number_l145_14589


namespace NUMINAMATH_GPT_laura_walk_distance_l145_14545

theorem laura_walk_distance 
  (east_blocks : ℕ) 
  (north_blocks : ℕ) 
  (block_length_miles : ℕ → ℝ) 
  (h_east_blocks : east_blocks = 8) 
  (h_north_blocks : north_blocks = 14) 
  (h_block_length_miles : ∀ b : ℕ, b = 1 → block_length_miles b = 1 / 4) 
  : (east_blocks + north_blocks) * block_length_miles 1 = 5.5 := 
by 
  sorry

end NUMINAMATH_GPT_laura_walk_distance_l145_14545


namespace NUMINAMATH_GPT_part2_x_values_part3_no_real_x_for_2000_l145_14577

noncomputable def average_daily_sales (x : ℝ) : ℝ :=
  24 + 4 * x

noncomputable def profit_per_unit (x : ℝ) : ℝ :=
  60 - 5 * x

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  (60 - 5 * x) * (24 + 4 * x)

theorem part2_x_values : 
  {x : ℝ | daily_sales_profit x = 1540} = {1, 5} := sorry

theorem part3_no_real_x_for_2000 : 
  ∀ x : ℝ, daily_sales_profit x ≠ 2000 := sorry

end NUMINAMATH_GPT_part2_x_values_part3_no_real_x_for_2000_l145_14577


namespace NUMINAMATH_GPT_number_of_balls_greater_l145_14597

theorem number_of_balls_greater (n x : ℤ) (h1 : n = 25) (h2 : n - x = 30 - n) : x = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_balls_greater_l145_14597


namespace NUMINAMATH_GPT_Mr_Caiden_payment_l145_14582

-- Defining the conditions as variables and constants
def total_roofing_needed : ℕ := 300
def cost_per_foot : ℕ := 8
def free_roofing : ℕ := 250

-- Define the remaining roofing needed and the total cost
def remaining_roofing : ℕ := total_roofing_needed - free_roofing
def total_cost : ℕ := remaining_roofing * cost_per_foot

-- The proof statement: 
theorem Mr_Caiden_payment : total_cost = 400 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Mr_Caiden_payment_l145_14582


namespace NUMINAMATH_GPT_necessary_not_sufficient_l145_14599

variable (a b : ℝ)

theorem necessary_not_sufficient : 
  (a > b) -> ¬ (a > b+1) ∨ (a > b+1 ∧ a > b) :=
by
  intro h
  have h1 : ¬ (a > b+1) := sorry
  have h2 : (a > b+1 -> a > b) := sorry
  exact Or.inl h1

end NUMINAMATH_GPT_necessary_not_sufficient_l145_14599


namespace NUMINAMATH_GPT_largest_two_digit_integer_l145_14585

theorem largest_two_digit_integer
  (a b : ℕ) (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 3 * (10 * a + b) = 10 * b + a + 5) :
  10 * a + b = 13 :=
by {
  -- Sorry is placed here to indicate that the proof is not provided
  sorry
}

end NUMINAMATH_GPT_largest_two_digit_integer_l145_14585


namespace NUMINAMATH_GPT_general_term_formula_sum_of_2_pow_an_l145_14533

variable {S : ℕ → ℕ}
variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℕ}

axiom S5_eq_30 : S 5 = 30
axiom a1_a6_eq_14 : a 1 + a 6 = 14

theorem general_term_formula : ∀ n, a n = 2 * n :=
sorry

theorem sum_of_2_pow_an (n : ℕ) : T n = (4^(n + 1)) / 3 - 4 / 3 :=
sorry

end NUMINAMATH_GPT_general_term_formula_sum_of_2_pow_an_l145_14533


namespace NUMINAMATH_GPT_proof_n_eq_neg2_l145_14583

theorem proof_n_eq_neg2 (n : ℤ) (h : |n + 6| = 2 - n) : n = -2 := 
by
  sorry

end NUMINAMATH_GPT_proof_n_eq_neg2_l145_14583


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l145_14595

theorem quadratic_inequality_solution (a m : ℝ) (h : a < 0) :
  (∀ x : ℝ, ax^2 + 6*x - a^2 < 0 ↔ (x < 1 ∨ x > m)) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l145_14595


namespace NUMINAMATH_GPT_brenda_skittles_l145_14570

theorem brenda_skittles (initial additional : ℕ) (h1 : initial = 7) (h2 : additional = 8) :
  initial + additional = 15 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_brenda_skittles_l145_14570


namespace NUMINAMATH_GPT_largest_possible_3_digit_sum_l145_14547

theorem largest_possible_3_digit_sum (X Y Z : ℕ) (h_diff : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
(h_digit_X : 0 ≤ X ∧ X ≤ 9) (h_digit_Y : 0 ≤ Y ∧ Y ≤ 9) (h_digit_Z : 0 ≤ Z ∧ Z ≤ 9) :
  (100 * X + 10 * X + X) + (10 * Y + X) + X = 994 → (X, Y, Z) = (8, 9, 0) := by
  sorry

end NUMINAMATH_GPT_largest_possible_3_digit_sum_l145_14547


namespace NUMINAMATH_GPT_abs_a_gt_abs_b_l145_14550

variable (a b : Real)

theorem abs_a_gt_abs_b (h1 : a > 0) (h2 : b < 0) (h3 : a + b > 0) : |a| > |b| :=
by
  sorry

end NUMINAMATH_GPT_abs_a_gt_abs_b_l145_14550


namespace NUMINAMATH_GPT_div_by_64_l145_14526

theorem div_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (5^n - 8*n^2 + 4*n - 1) :=
sorry

end NUMINAMATH_GPT_div_by_64_l145_14526


namespace NUMINAMATH_GPT_min_x_plus_y_l145_14543

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_x_plus_y_l145_14543


namespace NUMINAMATH_GPT_no_n_makes_g_multiple_of_5_and_7_l145_14565

def g (n : ℕ) : ℕ := 4 + 2 * n + 3 * n^2 + n^3 + 4 * n^4 + 3 * n^5

theorem no_n_makes_g_multiple_of_5_and_7 :
  ¬ ∃ n, (2 ≤ n ∧ n ≤ 100) ∧ (g n % 5 = 0 ∧ g n % 7 = 0) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_no_n_makes_g_multiple_of_5_and_7_l145_14565


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l145_14536

-- Define the atomic weights for Hydrogen, Chlorine, and Oxygen
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.45
def atomic_weight_O : ℝ := 15.999

-- Define the molecular weight of the compound
def molecular_weight (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  H_weight + Cl_weight + 2 * O_weight

-- The proof problem statement
theorem molecular_weight_of_compound :
  molecular_weight atomic_weight_H atomic_weight_Cl atomic_weight_O = 68.456 :=
sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l145_14536


namespace NUMINAMATH_GPT_theta_range_l145_14512

noncomputable def f (x θ : ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_range (θ : ℝ) (k : ℤ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x θ > 0) →
  θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
sorry

end NUMINAMATH_GPT_theta_range_l145_14512


namespace NUMINAMATH_GPT_probability_second_year_not_science_l145_14542

def total_students := 2000

def first_year := 600
def first_year_science := 300
def first_year_arts := 200
def first_year_engineering := 100

def second_year := 450
def second_year_science := 250
def second_year_arts := 150
def second_year_engineering := 50

def third_year := 550
def third_year_science := 300
def third_year_arts := 200
def third_year_engineering := 50

def postgraduate := 400
def postgraduate_science := 200
def postgraduate_arts := 100
def postgraduate_engineering := 100

def not_third_year_not_science :=
  (first_year_arts + first_year_engineering) +
  (second_year_arts + second_year_engineering) +
  (postgraduate_arts + postgraduate_engineering)

def second_year_not_science := second_year_arts + second_year_engineering

theorem probability_second_year_not_science :
  (second_year_not_science / not_third_year_not_science : ℚ) = (2 / 7 : ℚ) :=
by
  let total := (first_year_arts + first_year_engineering) + (second_year_arts + second_year_engineering) + (postgraduate_arts + postgraduate_engineering)
  have not_third_year_not_science : total = 300 + 200 + 200 := by sorry
  have second_year_not_science_eq : second_year_not_science = 200 := by sorry
  sorry

end NUMINAMATH_GPT_probability_second_year_not_science_l145_14542


namespace NUMINAMATH_GPT_rhombus_diagonal_solution_l145_14592

variable (d1 : ℝ) (A : ℝ)

def rhombus_other_diagonal (d1 d2 A : ℝ) : Prop :=
  A = (d1 * d2) / 2

theorem rhombus_diagonal_solution (h1 : d1 = 16) (h2 : A = 80) : rhombus_other_diagonal d1 10 A :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_solution_l145_14592


namespace NUMINAMATH_GPT_meanScore_is_91_666_l145_14510

-- Define Jane's quiz scores
def janesScores : List ℕ := [85, 88, 90, 92, 95, 100]

-- Define the total sum of Jane's quiz scores
def sumScores (scores : List ℕ) : ℕ := scores.foldl (· + ·) 0

-- The number of Jane's quiz scores
def numberOfScores (scores : List ℕ) : ℕ := scores.length

-- Define the mean of Jane's quiz scores
def meanScore (scores : List ℕ) : ℚ := sumScores scores / numberOfScores scores

-- The theorem to be proven
theorem meanScore_is_91_666 (h : janesScores = [85, 88, 90, 92, 95, 100]) :
  meanScore janesScores = 91.66666666666667 := by 
  sorry

end NUMINAMATH_GPT_meanScore_is_91_666_l145_14510


namespace NUMINAMATH_GPT_set_intersection_subset_condition_l145_14586

-- Define the sets A and B
def A (x : ℝ) : Prop := 1 < x - 1 ∧ x - 1 ≤ 4
def B (a : ℝ) (x : ℝ) : Prop := x < a

-- First proof problem: A ∩ B = {x | 2 < x < 3}
theorem set_intersection (a : ℝ) (x : ℝ) (h_a : a = 3) :
  A x ∧ B a x ↔ 2 < x ∧ x < 3 :=
by
  sorry

-- Second proof problem: a > 5 given A ⊆ B
theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ a > 5 :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_subset_condition_l145_14586


namespace NUMINAMATH_GPT_modulo_sum_of_99_plus_5_l145_14503

theorem modulo_sum_of_99_plus_5 : let s_n := (99 / 2) * (2 * 1 + (99 - 1) * 1)
                                 let sum_with_5 := s_n + 5
                                 sum_with_5 % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_modulo_sum_of_99_plus_5_l145_14503


namespace NUMINAMATH_GPT_intersection_set_l145_14554

-- Definition of the sets A and B
def setA : Set ℝ := { x | -2 < x ∧ x < 2 }
def setB : Set ℝ := { x | x < 0.5 }

-- The main theorem: Finding the intersection A ∩ B
theorem intersection_set : { x : ℝ | -2 < x ∧ x < 0.5 } = setA ∩ setB := by
  sorry

end NUMINAMATH_GPT_intersection_set_l145_14554


namespace NUMINAMATH_GPT_volume_of_box_l145_14504

variable (width length height : ℝ)
variable (Volume : ℝ)

-- Given conditions
def w : ℝ := 9
def l : ℝ := 4
def h : ℝ := 7

-- The statement to prove
theorem volume_of_box : Volume = l * w * h := by
  sorry

end NUMINAMATH_GPT_volume_of_box_l145_14504


namespace NUMINAMATH_GPT_intersection_M_N_eq_2_4_l145_14579

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℕ := {x | ∃ y, y = Real.log (6 - x) ∧ x < 6}

theorem intersection_M_N_eq_2_4 : M ∩ N = {2, 4} :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_eq_2_4_l145_14579


namespace NUMINAMATH_GPT_perc_freshmen_in_SLA_l145_14574

variables (T : ℕ) (P : ℝ)

-- 60% of students are freshmen
def freshmen (T : ℕ) : ℝ := 0.60 * T

-- 4.8% of students are freshmen psychology majors in the school of liberal arts
def freshmen_psych_majors (T : ℕ) : ℝ := 0.048 * T

-- 20% of freshmen in the school of liberal arts are psychology majors
def perc_fresh_psych (F_LA : ℝ) : ℝ := 0.20 * F_LA

-- Number of freshmen in the school of liberal arts as a percentage P of the total number of freshmen
def fresh_in_SLA_as_perc (T : ℕ) (P : ℝ) : ℝ := P * (0.60 * T)

theorem perc_freshmen_in_SLA (T : ℕ) (P : ℝ) :
  (0.20 * (P * (0.60 * T)) = 0.048 * T) → P = 0.4 :=
sorry

end NUMINAMATH_GPT_perc_freshmen_in_SLA_l145_14574


namespace NUMINAMATH_GPT_unique_solution_l145_14534

theorem unique_solution :
  ∀ (x y z n : ℕ), n ≥ 2 → z ≤ 5 * 2^(2 * n) → (x^ (2 * n + 1) - y^ (2 * n + 1) = x * y * z + 2^(2 * n + 1)) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  intros x y z n hn hzn hxyz
  sorry

end NUMINAMATH_GPT_unique_solution_l145_14534


namespace NUMINAMATH_GPT_quadratic_solution_transform_l145_14529

theorem quadratic_solution_transform (a b c : ℝ) (hA : 0 = a * (-3)^2 + b * (-3) + c) (hB : 0 = a * 4^2 + b * 4 + c) :
  (∃ x1 x2 : ℝ, a * (x1 - 1)^2 + b * (x1 - 1) + c = 0 ∧ a * (x2 - 1)^2 + b * (x2 - 1) + c = 0 ∧ x1 = -2 ∧ x2 = 5) :=
  sorry

end NUMINAMATH_GPT_quadratic_solution_transform_l145_14529


namespace NUMINAMATH_GPT_unattainable_y_l145_14578

theorem unattainable_y (x : ℝ) (h : x ≠ -(5 / 4)) :
    (∀ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3 / 4) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_unattainable_y_l145_14578


namespace NUMINAMATH_GPT_rational_iff_arithmetic_progression_l145_14505

theorem rational_iff_arithmetic_progression (x : ℝ) : 
  (∃ (i j k : ℤ), i < j ∧ j < k ∧ (x + i) + (x + k) = 2 * (x + j)) ↔ 
  (∃ n d : ℤ, d ≠ 0 ∧ x = n / d) := 
sorry

end NUMINAMATH_GPT_rational_iff_arithmetic_progression_l145_14505


namespace NUMINAMATH_GPT_max_y_value_l145_14576

theorem max_y_value (x y : Int) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
by sorry

end NUMINAMATH_GPT_max_y_value_l145_14576


namespace NUMINAMATH_GPT_students_attended_school_l145_14588

-- Definitions based on conditions
def total_students (S : ℕ) : Prop :=
  ∃ (L R : ℕ), 
    (L = S / 2) ∧ 
    (R = L / 4) ∧ 
    (5 = R / 5)

-- Theorem stating the problem
theorem students_attended_school (S : ℕ) : total_students S → S = 200 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_students_attended_school_l145_14588


namespace NUMINAMATH_GPT_square_area_is_correct_l145_14575

-- Define the condition: the side length of the square field
def side_length : ℝ := 7

-- Define the theorem to prove the area of the square field with given side length
theorem square_area_is_correct : side_length * side_length = 49 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_area_is_correct_l145_14575


namespace NUMINAMATH_GPT_tylenol_pill_mg_l145_14517

noncomputable def tylenol_dose_per_pill : ℕ :=
  let mg_per_dose := 1000
  let hours_per_dose := 6
  let days := 14
  let pills := 112
  let doses_per_day := 24 / hours_per_dose
  let total_doses := doses_per_day * days
  let total_mg := total_doses * mg_per_dose
  total_mg / pills

theorem tylenol_pill_mg :
  tylenol_dose_per_pill = 500 := by
  sorry

end NUMINAMATH_GPT_tylenol_pill_mg_l145_14517


namespace NUMINAMATH_GPT_num_fixed_last_two_digits_l145_14527

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end NUMINAMATH_GPT_num_fixed_last_two_digits_l145_14527


namespace NUMINAMATH_GPT_product_of_two_digit_numbers_is_not_five_digits_l145_14511

theorem product_of_two_digit_numbers_is_not_five_digits :
  ∀ (a b c d : ℕ), (10 ≤ 10 * a + b) → (10 * a + b ≤ 99) → (10 ≤ 10 * c + d) → (10 * c + d ≤ 99) → 
    (10 * a + b) * (10 * c + d) < 10000 :=
by
  intros a b c d H1 H2 H3 H4
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_product_of_two_digit_numbers_is_not_five_digits_l145_14511


namespace NUMINAMATH_GPT_calculate_expression_l145_14539

theorem calculate_expression : (-1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0) = Real.sqrt 2 - 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l145_14539


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_eq_l145_14581

theorem no_real_roots_of_quadratic_eq (k : ℝ) (h : k < -1) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - k = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_eq_l145_14581


namespace NUMINAMATH_GPT_part_one_part_two_l145_14584

def discriminant (a b c : ℝ) := b^2 - 4*a*c

theorem part_one (a : ℝ) (h : 0 < a) : 
  (∃ x : ℝ, ax^2 - 3*x + 2 < 0) ↔ 0 < a ∧ a < 9/8 := 
by 
  sorry

theorem part_two (a x : ℝ) : 
  (ax^2 - 3*x + 2 > ax - 1) ↔ 
  (a = 0 ∧ x < 1) ∨ 
  (a < 0 ∧ 3/a < x ∧ x < 1) ∨ 
  (0 < a ∧ (a > 3 ∧ (x < 3/a ∨ x > 1)) ∨ (a = 3 ∧ x ≠ 1) ∨ (0 < a ∧ a < 3 ∧ (x < 1 ∨ x > 3/a))) :=
by 
  sorry

end NUMINAMATH_GPT_part_one_part_two_l145_14584


namespace NUMINAMATH_GPT_max_min_value_of_f_l145_14563

theorem max_min_value_of_f (x y z : ℝ) :
  (-1 ≤ 2 * x + y - z) ∧ (2 * x + y - z ≤ 8) ∧
  (2 ≤ x - y + z) ∧ (x - y + z ≤ 9) ∧
  (-3 ≤ x + 2 * y - z) ∧ (x + 2 * y - z ≤ 7) →
  (-6 ≤ 7 * x + 5 * y - 2 * z) ∧ (7 * x + 5 * y - 2 * z ≤ 47) :=
by
  sorry

end NUMINAMATH_GPT_max_min_value_of_f_l145_14563


namespace NUMINAMATH_GPT_prime_p_satisfies_condition_l145_14507

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_p_satisfies_condition {p : ℕ} (hp : is_prime p) (hp2_8 : is_prime (p^2 + 8)) : p = 3 :=
sorry

end NUMINAMATH_GPT_prime_p_satisfies_condition_l145_14507


namespace NUMINAMATH_GPT_largest_four_digit_number_last_digit_l145_14537

theorem largest_four_digit_number_last_digit (n : ℕ) (n' : ℕ) (m r a b : ℕ) :
  (1000 * m + 100 * r + 10 * a + b = n) →
  (100 * m + 10 * r + a = n') →
  (n % 9 = 0) →
  (n' % 4 = 0) →
  b = 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_four_digit_number_last_digit_l145_14537


namespace NUMINAMATH_GPT_f_1991_eq_1988_l145_14593

def f (n : ℕ) : ℕ := sorry

theorem f_1991_eq_1988 : f 1991 = 1988 :=
by sorry

end NUMINAMATH_GPT_f_1991_eq_1988_l145_14593


namespace NUMINAMATH_GPT_max_quotient_l145_14532

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  ∃ max_val, max_val = 225 ∧ ∀ (x y : ℝ), (100 ≤ x ∧ x ≤ 300) ∧ (500 ≤ y ∧ y ≤ 1500) → (y^2 / x^2) ≤ max_val := 
by
  use 225
  sorry

end NUMINAMATH_GPT_max_quotient_l145_14532


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l145_14521

variable (P Q : Prop)

theorem intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false
  (h : P ∧ Q = False) : (P ∨ Q = False) ↔ (P ∧ Q = False) := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l145_14521


namespace NUMINAMATH_GPT_members_didnt_show_up_l145_14516

theorem members_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  total_members = 14 →
  points_per_member = 5 →
  total_points = 35 →
  total_members - (total_points / points_per_member) = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_members_didnt_show_up_l145_14516


namespace NUMINAMATH_GPT_election_at_least_one_past_officer_l145_14508

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem election_at_least_one_past_officer : 
  let total_candidates := 16
  let past_officers := 7
  let officer_positions := 5
  choose total_candidates officer_positions - choose (total_candidates - past_officers) officer_positions = 4242 :=
by
  sorry

end NUMINAMATH_GPT_election_at_least_one_past_officer_l145_14508


namespace NUMINAMATH_GPT_goods_train_length_is_280_l145_14535

noncomputable def length_of_goods_train (passenger_speed passenger_speed_kmh: ℝ) 
                                       (goods_speed goods_speed_kmh: ℝ) 
                                       (time_to_pass: ℝ) : ℝ :=
  let kmh_to_ms := (1000 : ℝ) / (3600 : ℝ)
  let passenger_speed_ms := passenger_speed * kmh_to_ms
  let goods_speed_ms     := goods_speed * kmh_to_ms
  let relative_speed     := passenger_speed_ms + goods_speed_ms
  relative_speed * time_to_pass

theorem goods_train_length_is_280 :
  length_of_goods_train 70 70 42 42 9 = 280 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_length_is_280_l145_14535


namespace NUMINAMATH_GPT_find_x_l145_14566

theorem find_x :
  ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 :=
by sorry

end NUMINAMATH_GPT_find_x_l145_14566


namespace NUMINAMATH_GPT_perpendicular_vectors_l145_14558

theorem perpendicular_vectors (x y : ℝ) (a : ℝ × ℝ := (1, 2)) (b : ℝ × ℝ := (2 + x, 1 - y)) 
  (hperp : (a.1 * b.1 + a.2 * b.2 = 0)) : 2 * y - x = 4 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l145_14558


namespace NUMINAMATH_GPT_balance_balls_l145_14538

variable (G B Y W R : ℕ)

theorem balance_balls :
  (4 * G = 8 * B) →
  (3 * Y = 7 * B) →
  (8 * B = 5 * W) →
  (2 * R = 6 * B) →
  (5 * G + 3 * Y + 3 * R = 26 * B) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_balance_balls_l145_14538


namespace NUMINAMATH_GPT_largest_base7_three_digit_is_342_l145_14548

-- Definition of the base-7 number 666
def base7_666 : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

-- The largest decimal number represented by a three-digit base-7 number is 342
theorem largest_base7_three_digit_is_342 : base7_666 = 342 := by
  sorry

end NUMINAMATH_GPT_largest_base7_three_digit_is_342_l145_14548


namespace NUMINAMATH_GPT_find_pairs_l145_14552

theorem find_pairs (m n : ℕ) (h : m > 0 ∧ n > 0 ∧ m^2 = n^2 + m + n + 2018) :
  (m, n) = (1010, 1008) ∨ (m, n) = (506, 503) :=
by sorry

end NUMINAMATH_GPT_find_pairs_l145_14552


namespace NUMINAMATH_GPT_flowchart_output_value_l145_14567

theorem flowchart_output_value :
  ∃ n : ℕ, S = n * (n + 1) / 2 ∧ n = 10 → S = 55 :=
by
  sorry

end NUMINAMATH_GPT_flowchart_output_value_l145_14567


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l145_14523

/--
Given an arithmetic sequence $\{a_n\}$ and $S_n$ being the sum of the first $n$ terms, 
with $a_1=1$ and $S_3=9$, prove that the common difference $d$ is equal to $2$.
-/
theorem arithmetic_sequence_common_difference :
  ∃ (d : ℝ), (∀ (n : ℕ), aₙ = 1 + (n - 1) * d) ∧ S₃ = a₁ + (a₁ + d) + (a₁ + 2 * d) ∧ a₁ = 1 ∧ S₃ = 9 → d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l145_14523


namespace NUMINAMATH_GPT_number_of_possible_scenarios_l145_14564

theorem number_of_possible_scenarios 
  (subjects : ℕ) 
  (students : ℕ) 
  (h_subjects : subjects = 4) 
  (h_students : students = 3) : 
  (subjects ^ students) = 64 := 
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_number_of_possible_scenarios_l145_14564


namespace NUMINAMATH_GPT_triangle_ctg_inequality_l145_14514

noncomputable def ctg (x : Real) := Real.cos x / Real.sin x

theorem triangle_ctg_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  ctg α ^ 2 + ctg β ^ 2 + ctg γ ^ 2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_triangle_ctg_inequality_l145_14514


namespace NUMINAMATH_GPT_gcd_is_18_l145_14544

-- Define gcdX that represents the greatest common divisor of X and Y.
noncomputable def gcdX (X Y : ℕ) : ℕ := Nat.gcd X Y

-- Given conditions
def cond_lcm (X Y : ℕ) : Prop := Nat.lcm X Y = 180
def cond_ratio (X Y : ℕ) : Prop := ∃ k : ℕ, X = 2 * k ∧ Y = 5 * k

-- Theorem to prove that the gcd of X and Y is 18
theorem gcd_is_18 {X Y : ℕ} (h1 : cond_lcm X Y) (h2 : cond_ratio X Y) : gcdX X Y = 18 :=
by
  sorry

end NUMINAMATH_GPT_gcd_is_18_l145_14544


namespace NUMINAMATH_GPT_average_growth_rate_l145_14559

theorem average_growth_rate (x : ℝ) (hx : (1 + x)^2 = 1.44) : x < 0.22 :=
sorry

end NUMINAMATH_GPT_average_growth_rate_l145_14559


namespace NUMINAMATH_GPT_larger_number_l145_14594

theorem larger_number (a b : ℤ) (h1 : a - b = 5) (h2 : a + b = 37) : a = 21 :=
sorry

end NUMINAMATH_GPT_larger_number_l145_14594


namespace NUMINAMATH_GPT_avg_speed_train_l145_14580

theorem avg_speed_train {D V : ℝ} (h1 : D = 20 * (90 / 60)) (h2 : 360 = 6 * 60) : 
  V = D / (360 / 60) :=
  by sorry

end NUMINAMATH_GPT_avg_speed_train_l145_14580


namespace NUMINAMATH_GPT_evaporation_amount_l145_14501

variable (E : ℝ)

def initial_koolaid_powder : ℝ := 2
def initial_water : ℝ := 16
def final_percentage : ℝ := 0.04

theorem evaporation_amount :
  (initial_koolaid_powder = 2) →
  (initial_water = 16) →
  (0.04 * (initial_koolaid_powder + 4 * (initial_water - E)) = initial_koolaid_powder) →
  E = 4 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_evaporation_amount_l145_14501


namespace NUMINAMATH_GPT_exists_indices_for_sequences_l145_14520

theorem exists_indices_for_sequences 
  (a b c : ℕ → ℕ) :
  ∃ (p q : ℕ), p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end NUMINAMATH_GPT_exists_indices_for_sequences_l145_14520


namespace NUMINAMATH_GPT_cost_of_each_box_of_cereal_l145_14562

theorem cost_of_each_box_of_cereal
  (total_groceries_cost : ℝ)
  (gallon_of_milk_cost : ℝ)
  (number_of_cereal_boxes : ℕ)
  (banana_cost_each : ℝ)
  (number_of_bananas : ℕ)
  (apple_cost_each : ℝ)
  (number_of_apples : ℕ)
  (cookie_cost_multiplier : ℝ)
  (number_of_cookie_boxes : ℕ) :
  total_groceries_cost = 25 →
  gallon_of_milk_cost = 3 →
  number_of_cereal_boxes = 2 →
  banana_cost_each = 0.25 →
  number_of_bananas = 4 →
  apple_cost_each = 0.5 →
  number_of_apples = 4 →
  cookie_cost_multiplier = 2 →
  number_of_cookie_boxes = 2 →
  (total_groceries_cost - (gallon_of_milk_cost + (banana_cost_each * number_of_bananas) + 
                           (apple_cost_each * number_of_apples) + 
                           (number_of_cookie_boxes * (cookie_cost_multiplier * gallon_of_milk_cost)))) / 
  number_of_cereal_boxes = 3.5 := 
sorry

end NUMINAMATH_GPT_cost_of_each_box_of_cereal_l145_14562


namespace NUMINAMATH_GPT_subtraction_equals_eleven_l145_14590

theorem subtraction_equals_eleven (K A N G R O : ℕ) (h1: K ≠ A) (h2: K ≠ N) (h3: K ≠ G) (h4: K ≠ R) (h5: K ≠ O) (h6: A ≠ N) (h7: A ≠ G) (h8: A ≠ R) (h9: A ≠ O) (h10: N ≠ G) (h11: N ≠ R) (h12: N ≠ O) (h13: G ≠ R) (h14: G ≠ O) (h15: R ≠ O) (sum_eq : 100 * K + 10 * A + N + 10 * G + A = 100 * R + 10 * O + O) : 
  (10 * R + N) - (10 * K + G) = 11 := 
by 
  sorry

end NUMINAMATH_GPT_subtraction_equals_eleven_l145_14590


namespace NUMINAMATH_GPT_correct_sampling_methods_l145_14549

/-- 
Given:
1. A group of 500 senior year students with the following blood type distribution: 200 with blood type O,
125 with blood type A, 125 with blood type B, and 50 with blood type AB.
2. A task to select a sample of 20 students to study the relationship between blood type and color blindness.
3. A high school soccer team consisting of 11 players, and the need to draw 2 players to investigate their study load.
4. Sampling methods: I. Random sampling, II. Systematic sampling, III. Stratified sampling.

Prove:
The correct sampling methods are: Stratified sampling (III) for the blood type-color blindness study and
Random sampling (I) for the soccer team study.
-/ 

theorem correct_sampling_methods (students : Finset ℕ) (blood_type_O blood_type_A blood_type_B blood_type_AB : ℕ)
  (sample_size_students soccer_team_size draw_size_soccer_team : ℕ)
  (sampling_methods : Finset ℕ) : 
  (students.card = 500) →
  (blood_type_O = 200) →
  (blood_type_A = 125) →
  (blood_type_B = 125) →
  (blood_type_AB = 50) →
  (sample_size_students = 20) →
  (soccer_team_size = 11) →
  (draw_size_soccer_team = 2) →
  (sampling_methods = {1, 2, 3}) →
  (s = (3, 1)) :=
by
  sorry

end NUMINAMATH_GPT_correct_sampling_methods_l145_14549


namespace NUMINAMATH_GPT_sum_of_numbers_l145_14560

theorem sum_of_numbers (x y : ℕ) (h1 : x = 18) (h2 : y = 2 * x - 3) : x + y = 51 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l145_14560


namespace NUMINAMATH_GPT_find_g_l145_14513

-- Definitions for functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := sorry -- We will define this later in the statement

theorem find_g :
  (∀ x : ℝ, g (x + 2) = f x) →
  (∀ x : ℝ, g x = 2 * x - 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_g_l145_14513


namespace NUMINAMATH_GPT_xyz_sum_l145_14528

theorem xyz_sum (x y z : ℝ) 
  (h1 : y + z = 17 - 2 * x) 
  (h2 : x + z = 1 - 2 * y) 
  (h3 : x + y = 8 - 2 * z) : 
  x + y + z = 6.5 :=
sorry

end NUMINAMATH_GPT_xyz_sum_l145_14528


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l145_14515

theorem problem_1 (x y z : ℝ) (h : z = (x + y) / 2) : z = (x + y) / 2 :=
sorry

theorem problem_2 (x y w : ℝ) (h1 : w = x + y) : w = x + y :=
sorry

theorem problem_3 (x w y : ℝ) (h1 : w = x + y) (h2 : y = w - x) : y = w - x :=
sorry

theorem problem_4 (x z v : ℝ) (h1 : z = (x + y) / 2) (h2 : v = 2 * z) : v = 2 * (x + (x + y) / 2) :=
sorry

theorem problem_5 (x z u : ℝ) (h : u = - (x + z) / 5) : x + z + 5 * u = 0 :=
sorry

theorem problem_6 (y z t : ℝ) (h : t = (6 + y + z) / 2) : t = (6 + y + z) / 2 :=
sorry

theorem problem_7 (y z s : ℝ) (h : y + z + 4 * s - 10 = 0) : y + z + 4 * s - 10 = 0 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_problem_7_l145_14515
