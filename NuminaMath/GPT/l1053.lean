import Mathlib

namespace NUMINAMATH_GPT_total_birds_on_fence_l1053_105329

theorem total_birds_on_fence (initial_pairs : ℕ) (birds_per_pair : ℕ) 
                             (new_pairs : ℕ) (new_birds_per_pair : ℕ)
                             (initial_birds : initial_pairs * birds_per_pair = 24)
                             (new_birds : new_pairs * new_birds_per_pair = 8) : 
                             ((initial_pairs * birds_per_pair) + (new_pairs * new_birds_per_pair) = 32) :=
sorry

end NUMINAMATH_GPT_total_birds_on_fence_l1053_105329


namespace NUMINAMATH_GPT_mean_age_of_oldest_three_l1053_105309

theorem mean_age_of_oldest_three (x : ℕ) (h : (x + (x + 1) + (x + 2)) / 3 = 6) : 
  (((x + 4) + (x + 5) + (x + 6)) / 3 = 10) := 
by
  sorry

end NUMINAMATH_GPT_mean_age_of_oldest_three_l1053_105309


namespace NUMINAMATH_GPT_total_height_of_sculpture_and_base_l1053_105336

def height_of_sculpture_m : Float := 0.88
def height_of_base_cm : Float := 20
def meter_to_cm : Float := 100

theorem total_height_of_sculpture_and_base :
  (height_of_sculpture_m * meter_to_cm + height_of_base_cm) = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_height_of_sculpture_and_base_l1053_105336


namespace NUMINAMATH_GPT_find_g_seven_l1053_105335

variable {g : ℝ → ℝ}

theorem find_g_seven (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_g_seven_l1053_105335


namespace NUMINAMATH_GPT_divisibility_of_special_number_l1053_105308

theorem divisibility_of_special_number (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
    ∃ d : ℕ, 100100 * a + 10010 * b + 1001 * c = 11 * d := 
sorry

end NUMINAMATH_GPT_divisibility_of_special_number_l1053_105308


namespace NUMINAMATH_GPT_double_root_possible_values_l1053_105381

theorem double_root_possible_values (b_3 b_2 b_1 : ℤ) (s : ℤ)
  (h : (Polynomial.X - Polynomial.C s) ^ 2 ∣
    Polynomial.C 24 + Polynomial.C b_1 * Polynomial.X + Polynomial.C b_2 * Polynomial.X ^ 2 + Polynomial.C b_3 * Polynomial.X ^ 3 + Polynomial.X ^ 4) :
  s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 :=
sorry

end NUMINAMATH_GPT_double_root_possible_values_l1053_105381


namespace NUMINAMATH_GPT_regular_tetrahedron_properties_l1053_105343

-- Definitions
def equilateral (T : Type) : Prop := sorry -- equilateral triangle property
def equal_sides (T : Type) : Prop := sorry -- all sides equal property
def equal_angles (T : Type) : Prop := sorry -- all angles equal property

def regular (H : Type) : Prop := sorry -- regular tetrahedron property
def equal_edges (H : Type) : Prop := sorry -- all edges are equal
def equal_edge_angles (H : Type) : Prop := sorry -- angles between two edges at the same vertex are equal
def congruent_equilateral_faces (H : Type) : Prop := sorry -- faces are congruent equilateral triangles
def equal_dihedral_angles (H : Type) : Prop := sorry -- dihedral angles between adjacent faces are equal

-- Theorem statement
theorem regular_tetrahedron_properties :
  ∀ (T H : Type), 
    (equilateral T → equal_sides T ∧ equal_angles T) →
    (regular H → 
      (equal_edges H ∧ equal_edge_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_dihedral_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_edge_angles H)) :=
by
  intros T H hT hH
  sorry

end NUMINAMATH_GPT_regular_tetrahedron_properties_l1053_105343


namespace NUMINAMATH_GPT_length_of_ribbon_l1053_105396

theorem length_of_ribbon (perimeter : ℝ) (sides : ℕ) (h1 : perimeter = 42) (h2 : sides = 6) : (perimeter / sides) = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_ribbon_l1053_105396


namespace NUMINAMATH_GPT_four_digit_integer_l1053_105375

theorem four_digit_integer (a b c d : ℕ) 
    (h1 : a + b + c + d = 16) 
    (h2 : b + c = 10) 
    (h3 : a - d = 2) 
    (h4 : (a - b + c - d) % 11 = 0) : 
    a = 4 ∧ b = 4 ∧ c = 6 ∧ d = 2 :=
sorry

end NUMINAMATH_GPT_four_digit_integer_l1053_105375


namespace NUMINAMATH_GPT_no_equal_differences_between_products_l1053_105355

theorem no_equal_differences_between_products (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    ¬ (∃ k : ℕ, ac - ab = k ∧ ad - ac = k ∧ bc - ad = k ∧ bd - bc = k ∧ cd - bd = k) :=
by
  sorry

end NUMINAMATH_GPT_no_equal_differences_between_products_l1053_105355


namespace NUMINAMATH_GPT_designated_time_to_B_l1053_105387

theorem designated_time_to_B (s v : ℝ) (x : ℝ) (V' : ℝ)
  (h1 : s / 2 = (x + 2) * V')
  (h2 : s / (2 * V') + 1 + s / (2 * (V' + v)) = x) :
  x = (v + Real.sqrt (9 * v ^ 2 + 6 * v * s)) / v :=
by
  sorry

end NUMINAMATH_GPT_designated_time_to_B_l1053_105387


namespace NUMINAMATH_GPT_solution_set_of_cx_sq_minus_bx_plus_a_l1053_105342

theorem solution_set_of_cx_sq_minus_bx_plus_a (a b c : ℝ) (h1 : a < 0)
(h2 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x : ℝ, cx^2 - bx + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_cx_sq_minus_bx_plus_a_l1053_105342


namespace NUMINAMATH_GPT_smallest_number_satisfies_conditions_l1053_105300

-- Define the number we are looking for
def number : ℕ := 391410

theorem smallest_number_satisfies_conditions :
  (number % 7 = 2) ∧
  (number % 11 = 2) ∧
  (number % 13 = 2) ∧
  (number % 17 = 3) ∧
  (number % 23 = 0) ∧
  (number % 5 = 0) :=
by
  -- We need to prove that 391410 satisfies all the given conditions.
  -- This proof will include detailed steps to verify each condition
  sorry

end NUMINAMATH_GPT_smallest_number_satisfies_conditions_l1053_105300


namespace NUMINAMATH_GPT_turtle_marathon_time_l1053_105334

/-- Given a marathon distance of 42 kilometers and 195 meters and a turtle's speed of 15 meters per minute,
prove that the turtle will reach the finish line in 1 day, 22 hours, and 53 minutes. -/
theorem turtle_marathon_time :
  let speed := 15 -- meters per minute
  let distance_km := 42 -- kilometers
  let distance_m := 195 -- meters
  let total_distance := distance_km * 1000 + distance_m -- total distance in meters
  let time_min := total_distance / speed -- time to complete the marathon in minutes
  let hours := time_min / 60 -- time to complete the marathon in hours (division and modulus)
  let minutes := time_min % 60 -- remaining minutes after converting total minutes to hours
  let days := hours / 24 -- time to complete the marathon in days (division and modulus)
  let remaining_hours := hours % 24 -- remaining hours after converting total hours to days
  (days, remaining_hours, minutes) = (1, 22, 53) -- expected result
:= 
sorry

end NUMINAMATH_GPT_turtle_marathon_time_l1053_105334


namespace NUMINAMATH_GPT_days_C_alone_l1053_105352

theorem days_C_alone (r_A r_B r_C : ℝ) (h1 : r_A + r_B = 1 / 3) (h2 : r_B + r_C = 1 / 6) (h3 : r_A + r_C = 5 / 18) : 
  1 / r_C = 18 := 
  sorry

end NUMINAMATH_GPT_days_C_alone_l1053_105352


namespace NUMINAMATH_GPT_find_g_2022_l1053_105311

def g : ℝ → ℝ := sorry -- This is pre-defined to say there exists such a function

theorem find_g_2022 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y)) :
  g 2022 = 4086462 :=
sorry

end NUMINAMATH_GPT_find_g_2022_l1053_105311


namespace NUMINAMATH_GPT_volleyball_team_selection_l1053_105331

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (n.choose k)

theorem volleyball_team_selection : 
  let quadruplets := ["Bella", "Bianca", "Becca", "Brooke"];
  let total_players := 16;
  let starters := 7;
  let num_quadruplets := quadruplets.length;
  ∃ ways : ℕ, 
    ways = binom num_quadruplets 3 * binom (total_players - num_quadruplets) (starters - 3) 
    ∧ ways = 1980 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_team_selection_l1053_105331


namespace NUMINAMATH_GPT_find_distinct_numbers_l1053_105328

theorem find_distinct_numbers (k l : ℕ) (h : 64 / k = 4 * (64 / l)) : k = 1 ∧ l = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_distinct_numbers_l1053_105328


namespace NUMINAMATH_GPT_max_area_guaranteed_l1053_105378

noncomputable def max_rectangle_area (board_size : ℕ) (removed_cells : ℕ) : ℕ :=
  if board_size = 8 ∧ removed_cells = 8 then 8 else 0

theorem max_area_guaranteed :
  max_rectangle_area 8 8 = 8 :=
by
  -- Proof logic goes here
  sorry

end NUMINAMATH_GPT_max_area_guaranteed_l1053_105378


namespace NUMINAMATH_GPT_remainder_of_N_mod_103_l1053_105363

noncomputable def N : ℕ :=
  sorry -- This will capture the mathematical calculation of N using the conditions stated.

theorem remainder_of_N_mod_103 : (N % 103) = 43 :=
  sorry

end NUMINAMATH_GPT_remainder_of_N_mod_103_l1053_105363


namespace NUMINAMATH_GPT_range_of_a_l1053_105380

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x ≤ 1 then (a - 3) * x - 3 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 3 < a ∧ a ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1053_105380


namespace NUMINAMATH_GPT_calculate_length_of_floor_l1053_105350

-- Define the conditions and the objective to prove
variable (breadth length : ℝ)
variable (cost rate : ℝ)
variable (area : ℝ)

-- Given conditions
def length_more_by_percentage : Prop := length = 2 * breadth
def painting_cost : Prop := cost = 529 ∧ rate = 3

-- Objective
def length_of_floor : ℝ := 2 * breadth

theorem calculate_length_of_floor : 
  (length_more_by_percentage breadth length) →
  (painting_cost cost rate) →
  length_of_floor breadth = 18.78 :=
by
  sorry

end NUMINAMATH_GPT_calculate_length_of_floor_l1053_105350


namespace NUMINAMATH_GPT_Kelly_baking_powder_difference_l1053_105394

theorem Kelly_baking_powder_difference : 0.4 - 0.3 = 0.1 :=
by 
  -- sorry is a placeholder for a proof
  sorry

end NUMINAMATH_GPT_Kelly_baking_powder_difference_l1053_105394


namespace NUMINAMATH_GPT_city_tax_problem_l1053_105322

theorem city_tax_problem :
  ∃ (x y : ℕ), 
    ((x + 3000) * (y - 10) = x * y) ∧
    ((x - 1000) * (y + 10) = x * y) ∧
    (x = 3000) ∧
    (y = 20) ∧
    (x * y = 60000) :=
by
  sorry

end NUMINAMATH_GPT_city_tax_problem_l1053_105322


namespace NUMINAMATH_GPT_find_xy_l1053_105313

variable (x y : ℝ)

theorem find_xy (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end NUMINAMATH_GPT_find_xy_l1053_105313


namespace NUMINAMATH_GPT_cos_double_angle_l1053_105395

open Real

theorem cos_double_angle (α : ℝ) (h : tan α = 3) : cos (2 * α) = -4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1053_105395


namespace NUMINAMATH_GPT_max_value_of_product_l1053_105327

theorem max_value_of_product (x y z w : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) (h_sum : x + y + z + w = 1) : 
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_product_l1053_105327


namespace NUMINAMATH_GPT_northern_village_population_l1053_105362

theorem northern_village_population
    (x : ℕ) -- Northern village population
    (western_village_population : ℕ := 400)
    (southern_village_population : ℕ := 200)
    (total_conscripted : ℕ := 60)
    (northern_village_conscripted : ℕ := 10)
    (h : (northern_village_conscripted : ℚ) / total_conscripted = (x : ℚ) / (x + western_village_population + southern_village_population)) : 
    x = 120 :=
    sorry

end NUMINAMATH_GPT_northern_village_population_l1053_105362


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1053_105353

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic property of the sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (h1 : is_arithmetic_sequence a d)
  (h2 : a 2 + a 4 + a 7 + a 11 = 44) :
  a 3 + a 5 + a 10 = 33 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1053_105353


namespace NUMINAMATH_GPT_IncorrectOption_l1053_105370

namespace Experiment

def OptionA : Prop := 
  ∃ method : String, method = "sampling detection"

def OptionB : Prop := 
  ¬(∃ experiment : String, experiment = "does not need a control group, nor repeated experiments")

def OptionC : Prop := 
  ∃ action : String, action = "test tube should be gently shaken"

def OptionD : Prop := 
  ∃ condition : String, condition = "field of view should not be too bright"

theorem IncorrectOption : OptionB :=
  sorry

end Experiment

end NUMINAMATH_GPT_IncorrectOption_l1053_105370


namespace NUMINAMATH_GPT_cost_of_pants_is_250_l1053_105330

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_cost_of_pants_is_250_l1053_105330


namespace NUMINAMATH_GPT_sin_B_value_triangle_area_l1053_105320

-- Problem 1: sine value of angle B given the conditions
theorem sin_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C) :
  Real.sin B = (4 * Real.sqrt 5) / 9 :=
sorry

-- Problem 2: Area of triangle ABC given the conditions and b = 4
theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C)
  (h3 : b = 4) :
  (1 / 2) * b * c * Real.sin A = (14 * Real.sqrt 5) / 9 :=
sorry

end NUMINAMATH_GPT_sin_B_value_triangle_area_l1053_105320


namespace NUMINAMATH_GPT_base_for_784_as_CDEC_l1053_105348

theorem base_for_784_as_CDEC : 
  ∃ (b : ℕ), 
  (b^3 ≤ 784 ∧ 784 < b^4) ∧ 
  (∃ C D : ℕ, C ≠ D ∧ 784 = (C * b^3 + D * b^2 + C * b + C) ∧ 
  b = 6) :=
sorry

end NUMINAMATH_GPT_base_for_784_as_CDEC_l1053_105348


namespace NUMINAMATH_GPT_tweets_when_hungry_l1053_105384

theorem tweets_when_hungry (H : ℕ) : 
  (18 * 20) + (H * 20) + (45 * 20) = 1340 → H = 4 := by
  sorry

end NUMINAMATH_GPT_tweets_when_hungry_l1053_105384


namespace NUMINAMATH_GPT_correct_mean_after_correction_l1053_105339

theorem correct_mean_after_correction
  (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ)
  (h : n = 30) (h_mean : incorrect_mean = 150) (h_incorrect_value : incorrect_value = 135) (h_correct_value : correct_value = 165) :
  (incorrect_mean * n - incorrect_value + correct_value) / n = 151 :=
  by
  sorry

end NUMINAMATH_GPT_correct_mean_after_correction_l1053_105339


namespace NUMINAMATH_GPT_number_of_common_tangents_of_two_circles_l1053_105305

theorem number_of_common_tangents_of_two_circles 
  (x y : ℝ)
  (circle1 : x^2 + y^2 = 1)
  (circle2 : x^2 + y^2 - 6 * x - 8 * y + 9 = 0) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_common_tangents_of_two_circles_l1053_105305


namespace NUMINAMATH_GPT_integer_roots_of_polynomial_l1053_105358

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 6 * x^2 - 4 * x + 24 = 0} = {2, -2} :=
by
  sorry

end NUMINAMATH_GPT_integer_roots_of_polynomial_l1053_105358


namespace NUMINAMATH_GPT_division_remainder_l1053_105354

theorem division_remainder (dividend quotient divisor remainder : ℕ) 
  (h_dividend : dividend = 12401) 
  (h_quotient : quotient = 76) 
  (h_divisor : divisor = 163) 
  (h_remainder : dividend = quotient * divisor + remainder) : 
  remainder = 13 := 
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1053_105354


namespace NUMINAMATH_GPT_range_of_a_l1053_105314

noncomputable def f (a x : ℝ) :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a) ∧
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  1 / 7 ≤ a ∧ a < 1 / 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1053_105314


namespace NUMINAMATH_GPT_custom_op_12_7_l1053_105346

def custom_op (a b : ℤ) := (a + b) * (a - b)

theorem custom_op_12_7 : custom_op 12 7 = 95 := by
  sorry

end NUMINAMATH_GPT_custom_op_12_7_l1053_105346


namespace NUMINAMATH_GPT_forgotten_angles_sum_l1053_105301

theorem forgotten_angles_sum (n : ℕ) (h : (n-2) * 180 = 3240 + x) : x = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_forgotten_angles_sum_l1053_105301


namespace NUMINAMATH_GPT_sachin_rahul_age_ratio_l1053_105304

theorem sachin_rahul_age_ratio 
(S_age : ℕ) 
(R_age : ℕ) 
(h1 : R_age = S_age + 4) 
(h2 : S_age = 14) : 
S_age / Int.gcd S_age R_age = 7 ∧ R_age / Int.gcd S_age R_age = 9 := 
by 
sorry

end NUMINAMATH_GPT_sachin_rahul_age_ratio_l1053_105304


namespace NUMINAMATH_GPT_kate_money_left_l1053_105359

def kate_savings_march := 27
def kate_savings_april := 13
def kate_savings_may := 28
def kate_expenditure_keyboard := 49
def kate_expenditure_mouse := 5

def total_savings := kate_savings_march + kate_savings_april + kate_savings_may
def total_expenditure := kate_expenditure_keyboard + kate_expenditure_mouse
def money_left := total_savings - total_expenditure

-- Prove that Kate has $14 left
theorem kate_money_left : money_left = 14 := 
by 
  sorry

end NUMINAMATH_GPT_kate_money_left_l1053_105359


namespace NUMINAMATH_GPT_mean_eq_value_of_z_l1053_105361

theorem mean_eq_value_of_z (z : ℤ) : 
  ((6 + 15 + 9 + 20) / 4 : ℚ) = ((13 + z) / 2 : ℚ) → (z = 12) := by
  sorry

end NUMINAMATH_GPT_mean_eq_value_of_z_l1053_105361


namespace NUMINAMATH_GPT_prove_value_l1053_105391

variable (m n : ℤ)

-- Conditions from the problem
def condition1 : Prop := m^2 + 2 * m * n = 384
def condition2 : Prop := 3 * m * n + 2 * n^2 = 560

-- Proposition to be proved
theorem prove_value (h1 : condition1 m n) (h2 : condition2 m n) : 2 * m^2 + 13 * m * n + 6 * n^2 - 444 = 2004 := by
  sorry

end NUMINAMATH_GPT_prove_value_l1053_105391


namespace NUMINAMATH_GPT_correct_statements_l1053_105303

def f (x : ℝ) (b : ℝ) (c : ℝ) := x * (abs x) + b * x + c

theorem correct_statements (b c : ℝ) :
  (∀ x, c = 0 → f (-x) b 0 = - f x b 0) ∧
  (∀ x, b = 0 → c > 0 → (f x 0 c = 0 → x = 0) ∧ ∀ y, f y 0 c ≤ 0) ∧
  (∀ x, ∃ k : ℝ, f (k + x) b c = f (k - x) b c) ∧
  ¬(∀ x, x > 0 → f x b c = c - b^2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1053_105303


namespace NUMINAMATH_GPT_remaining_amount_to_be_paid_l1053_105369

theorem remaining_amount_to_be_paid (part_payment : ℝ) (percentage : ℝ) (h : part_payment = 650 ∧ percentage = 0.15) :
    (part_payment / percentage - part_payment) = 3683.33 := by
  cases h with
  | intro h1 h2 =>
    sorry

end NUMINAMATH_GPT_remaining_amount_to_be_paid_l1053_105369


namespace NUMINAMATH_GPT_fraction_solution_l1053_105383

theorem fraction_solution (a : ℕ) (h : a > 0) (h_eq : (a : ℚ) / (a + 45) = 0.75) : a = 135 :=
sorry

end NUMINAMATH_GPT_fraction_solution_l1053_105383


namespace NUMINAMATH_GPT_penultimate_digit_even_l1053_105323

theorem penultimate_digit_even (n : ℕ) (h : n > 2) : ∃ k : ℕ, ∃ d : ℕ, d % 2 = 0 ∧ 10 * d + k = (3 ^ n) % 100 :=
sorry

end NUMINAMATH_GPT_penultimate_digit_even_l1053_105323


namespace NUMINAMATH_GPT_log_identity_l1053_105398

theorem log_identity
  (x : ℝ)
  (h1 : x < 1)
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^4) / Real.log 10 = 100) :
  (Real.log x / Real.log 10)^3 - Real.log (x^5) / Real.log 10 = -114 + Real.sqrt 104 := 
by
  sorry

end NUMINAMATH_GPT_log_identity_l1053_105398


namespace NUMINAMATH_GPT_buildingC_floors_if_five_times_l1053_105319

-- Defining the number of floors in Building B
def floorsBuildingB : ℕ := 13

-- Theorem to prove the number of floors in Building C if it had five times as many floors as Building B
theorem buildingC_floors_if_five_times (FB : ℕ) (h : FB = floorsBuildingB) : (5 * FB) = 65 :=
by
  rw [h]
  exact rfl

end NUMINAMATH_GPT_buildingC_floors_if_five_times_l1053_105319


namespace NUMINAMATH_GPT_shaded_area_is_10_l1053_105345

-- Definitions based on conditions:
def rectangle_area : ℕ := 12
def unshaded_triangle_area : ℕ := 2

-- Proof statement without the actual proof.
theorem shaded_area_is_10 : rectangle_area - unshaded_triangle_area = 10 := by
  sorry

end NUMINAMATH_GPT_shaded_area_is_10_l1053_105345


namespace NUMINAMATH_GPT_square_side_length_l1053_105337

theorem square_side_length (s : ℝ) (h : s^2 = 3 * 4 * s) : s = 12 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l1053_105337


namespace NUMINAMATH_GPT_common_root_equations_l1053_105372

theorem common_root_equations (a b : ℝ) 
  (h : ∃ x₀ : ℝ, (x₀ ^ 2 + a * x₀ + b = 0) ∧ (x₀ ^ 2 + b * x₀ + a = 0)) 
  (hc : ∀ x₁ x₂ : ℝ, (x₁ ^ 2 + a * x₁ + b = 0 ∧ x₂ ^ 2 + bx₀ + a = 0) → x₁ = x₂) :
  a + b = -1 :=
sorry

end NUMINAMATH_GPT_common_root_equations_l1053_105372


namespace NUMINAMATH_GPT_sinks_per_house_l1053_105360

theorem sinks_per_house (total_sinks : ℕ) (houses : ℕ) (h_total_sinks : total_sinks = 266) (h_houses : houses = 44) :
  total_sinks / houses = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_sinks_per_house_l1053_105360


namespace NUMINAMATH_GPT_find_z_l1053_105315

open Complex

noncomputable def sqrt_five : ℝ := Real.sqrt 5

theorem find_z (z : ℂ) 
  (hz1 : z.re < 0) 
  (hz2 : z.im > 0) 
  (h_modulus : abs z = 3) 
  (h_real_part : z.re = -sqrt_five) : 
  z = -sqrt_five + 2 * I :=
by
  sorry

end NUMINAMATH_GPT_find_z_l1053_105315


namespace NUMINAMATH_GPT_range_of_a_for_decreasing_function_l1053_105389

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * (a - 1) * x + 5

noncomputable def f' (x : ℝ) : ℝ := -2 * x - 2 * (a - 1)

theorem range_of_a_for_decreasing_function :
  (∀ x : ℝ, -1 ≤ x → f' a x ≤ 0) → 2 ≤ a := sorry

end NUMINAMATH_GPT_range_of_a_for_decreasing_function_l1053_105389


namespace NUMINAMATH_GPT_frustum_midsection_area_relation_l1053_105321

theorem frustum_midsection_area_relation 
  (S₁ S₂ S₀ : ℝ) 
  (h₁: 0 ≤ S₁ ∧ 0 ≤ S₂ ∧ 0 ≤ S₀)
  (h₂: ∃ a h, (a / (a + 2 * h))^2 = S₂ / S₁ ∧ (a / (a + h))^2 = S₂ / S₀) :
  2 * Real.sqrt S₀ = Real.sqrt S₁ + Real.sqrt S₂ := 
sorry

end NUMINAMATH_GPT_frustum_midsection_area_relation_l1053_105321


namespace NUMINAMATH_GPT_largest_number_not_sum_of_two_composites_l1053_105374

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end NUMINAMATH_GPT_largest_number_not_sum_of_two_composites_l1053_105374


namespace NUMINAMATH_GPT_grazing_months_l1053_105318

theorem grazing_months :
  ∀ (m : ℕ),
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * b_months
  let c_ox_months := c_oxen * m
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  let c_part := (c_ox_months : ℝ) / (total_ox_months : ℝ) * rent
  (c_part = c_share) → m = 3 :=
by { sorry }

end NUMINAMATH_GPT_grazing_months_l1053_105318


namespace NUMINAMATH_GPT_teams_have_equal_people_l1053_105366

-- Definitions capturing the conditions
def managers : Nat := 3
def employees : Nat := 3
def teams : Nat := 3

-- The total number of people
def total_people : Nat := managers + employees

-- The proof statement
theorem teams_have_equal_people : total_people / teams = 2 := by
  sorry

end NUMINAMATH_GPT_teams_have_equal_people_l1053_105366


namespace NUMINAMATH_GPT_find_number_l1053_105338

theorem find_number (x : ℝ) (h : x / 100 = 31.76 + 0.28) : x = 3204 := 
  sorry

end NUMINAMATH_GPT_find_number_l1053_105338


namespace NUMINAMATH_GPT_batsman_average_after_12th_innings_l1053_105392

noncomputable def batsman_average (runs_in_12th_innings : ℕ) (average_increase : ℕ) (initial_average_after_11_innings : ℕ) : ℕ :=
initial_average_after_11_innings + average_increase

theorem batsman_average_after_12th_innings
(score_in_12th_innings : ℕ)
(average_increase : ℕ)
(initial_average_after_11_innings : ℕ)
(total_runs_after_11_innings := 11 * initial_average_after_11_innings)
(total_runs_after_12_innings := total_runs_after_11_innings + score_in_12th_innings)
(new_average_after_12_innings := total_runs_after_12_innings / 12)
:
score_in_12th_innings = 80 ∧ average_increase = 3 ∧ initial_average_after_11_innings = 44 → 
batsman_average score_in_12th_innings average_increase initial_average_after_11_innings = 47 := 
by
  -- skipping the actual proof for now
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_innings_l1053_105392


namespace NUMINAMATH_GPT_correct_number_of_statements_l1053_105379

-- Definitions based on the problem's conditions
def condition_1 : Prop :=
  ∀ (n : ℕ) (a b c d e : ℚ), n = 5 ∧ ∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (x < 0 ∧ y < 0 ∧ z < 0 ∧ d ≥ 0 ∧ e ≥ 0) →
  (a * b * c * d * e < 0 ∨ a * b * c * d * e = 0)

def condition_2 : Prop := 
  ∀ m : ℝ, |m| + m = 0 → m ≤ 0

def condition_3 : Prop := 
  ∀ a b : ℝ, (1 / a < 1 / b) → ¬ (a < b ∨ b < a)

def condition_4 : Prop := 
  ∀ a : ℝ, ∃ max_val, max_val = 5 ∧ 5 - |a - 5| ≤ max_val

-- Main theorem to state the correct number of true statements
theorem correct_number_of_statements : 
  (condition_2 ∧ condition_4) ∧
  ¬condition_1 ∧ 
  ¬condition_3 :=
by
  sorry

end NUMINAMATH_GPT_correct_number_of_statements_l1053_105379


namespace NUMINAMATH_GPT_math_problem_equivalence_l1053_105364

section

variable (x y z : ℝ) (w : String)

theorem math_problem_equivalence (h₀ : x / 15 = 4 / 5) (h₁ : y = 80) (h₂ : z = 0.8) (h₃ : w = "八折"):
  x = 12 ∧ y = 80 ∧ z = 0.8 ∧ w = "八折" :=
by
  sorry

end

end NUMINAMATH_GPT_math_problem_equivalence_l1053_105364


namespace NUMINAMATH_GPT_yoongi_has_5_carrots_l1053_105382

def yoongis_carrots (initial_carrots sister_gave: ℕ) : ℕ :=
  initial_carrots + sister_gave

theorem yoongi_has_5_carrots : yoongis_carrots 3 2 = 5 := by 
  sorry

end NUMINAMATH_GPT_yoongi_has_5_carrots_l1053_105382


namespace NUMINAMATH_GPT_find_length_PB_l1053_105333

noncomputable def radius (O : Type*) : ℝ := sorry

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*}

def Point (α : Type*) := α

variables (P T A B : Point ℝ) (O : Circle ℝ) (r : ℝ)

def PA := (4 : ℝ)
def PT (AB : ℝ) := AB - 2
def PB (AB : ℝ) := 4 + AB

def power_of_a_point (PA PB PT : ℝ) := PA * PB = PT^2

theorem find_length_PB (AB : ℝ) 
  (h1 : power_of_a_point PA (PB AB) (PT AB)) 
  (h2 : PA < PB AB) : 
  PB AB = 18 := 
by 
  sorry

end NUMINAMATH_GPT_find_length_PB_l1053_105333


namespace NUMINAMATH_GPT_flour_needed_for_one_loaf_l1053_105306

-- Define the conditions
def flour_needed_for_two_loaves : ℚ := 5 -- cups of flour needed for two loaves

-- Define the theorem to prove
theorem flour_needed_for_one_loaf : flour_needed_for_two_loaves / 2 = 2.5 :=
by 
  -- Skip the proof.
  sorry

end NUMINAMATH_GPT_flour_needed_for_one_loaf_l1053_105306


namespace NUMINAMATH_GPT_fill_in_square_l1053_105390

variable {α : Type*} [CommRing α]

theorem fill_in_square (a b : α) (square : α) (h : square * 3 * a * b = 3 * a^2 * b) : square = a :=
sorry

end NUMINAMATH_GPT_fill_in_square_l1053_105390


namespace NUMINAMATH_GPT_kittens_per_bunny_l1053_105397

-- Conditions
def total_initial_bunnies : ℕ := 30
def fraction_given_to_friend : ℚ := 2 / 5
def total_bunnies_after_birth : ℕ := 54

-- Determine the number of kittens each bunny gave birth to
theorem kittens_per_bunny (initial_bunnies given_fraction total_bunnies_after : ℕ) 
  (h1 : initial_bunnies = total_initial_bunnies)
  (h2 : given_fraction = fraction_given_to_friend)
  (h3 : total_bunnies_after = total_bunnies_after_birth) :
  (total_bunnies_after - (total_initial_bunnies - (total_initial_bunnies * fraction_given_to_friend))) / 
    (total_initial_bunnies * (1 - fraction_given_to_friend)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_kittens_per_bunny_l1053_105397


namespace NUMINAMATH_GPT_find_f_15_l1053_105326

theorem find_f_15
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - 2 * y) + 3 * x ^ 2 + 2) :
  f 15 = 1202 := 
sorry

end NUMINAMATH_GPT_find_f_15_l1053_105326


namespace NUMINAMATH_GPT_gcd_1995_228_eval_f_at_2_l1053_105367

-- Euclidean Algorithm Problem
theorem gcd_1995_228 : Nat.gcd 1995 228 = 57 :=
by
  sorry

-- Horner's Method Problem
def f (x : ℝ) : ℝ := 3 * x ^ 5 + 2 * x ^ 3 - 8 * x + 5

theorem eval_f_at_2 : f 2 = 101 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1995_228_eval_f_at_2_l1053_105367


namespace NUMINAMATH_GPT_complement_A_union_B_l1053_105317

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end NUMINAMATH_GPT_complement_A_union_B_l1053_105317


namespace NUMINAMATH_GPT_value_of_S_2016_l1053_105347

variable (a d : ℤ)
variable (S : ℕ → ℤ)

-- Definitions of conditions
def a_1 := -2014
def sum_2012 := S 2012
def sum_10 := S 10
def S_n (n : ℕ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom S_condition : (sum_2012 / 2012) - (sum_10 / 10) = 2002
axiom S_def : ∀ n : ℕ, S n = S_n n

-- The theorem to be proved
theorem value_of_S_2016 : S 2016 = 2016 := by
  sorry

end NUMINAMATH_GPT_value_of_S_2016_l1053_105347


namespace NUMINAMATH_GPT_tangent_line_at_1_extreme_points_range_of_a_l1053_105307

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x ^ 2 - 3 * x + 2)

theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  ∃ m b, ∀ x, f x a = m * x + b ∧ m = 1 ∧ b = -1 := sorry

theorem extreme_points (a : ℝ) :
  (0 < a ∧ a <= 8 / 9 → ∀ x, 0 < x → f x a = 0) ∧
  (a > 8 / 9 → ∃ x1 x2, x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x1 → f x a = 0) ∧
   (∀ x, x1 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) ∧
  (a < 0 → ∃ x1 x2, x1 < 0 ∧ 0 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a >= 0) ↔ 0 ≤ a ∧ a ≤ 1 := sorry

end NUMINAMATH_GPT_tangent_line_at_1_extreme_points_range_of_a_l1053_105307


namespace NUMINAMATH_GPT_parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l1053_105386

noncomputable def parabola (p x : ℝ) : ℝ := (p-1) * x^2 + 2 * p * x + 4

-- 1. Prove that if \( p = 2 \), the parabola \( g_p \) is tangent to the \( x \)-axis.
theorem parabola_tangent_xaxis_at_p2 : ∀ x, parabola 2 x = (x + 2)^2 := 
by 
  intro x
  sorry

-- 2. Prove that if \( p = 0 \), the vertex of the parabola \( g_p \) lies on the \( y \)-axis.
theorem parabola_vertex_yaxis_at_p0 : ∃ x, parabola 0 x = 4 := 
by 
  sorry

-- 3. Prove the parabolas for \( p = 2 \) and \( p = 0 \) are symmetric with respect to \( M(-1, 2) \).
theorem parabolas_symmetric_m_point : ∀ x, 
  (parabola 2 x = (x + 2)^2) → 
  (parabola 0 x = -x^2 + 4) → 
  (-1, 2) = (-1, 2) := 
by 
  sorry

-- 4. Prove that the points \( (0, 4) \) and \( (-2, 0) \) lie on the curve for all \( p \).
theorem parabola_familiy_point_through : ∀ p, 
  parabola p 0 = 4 ∧ 
  parabola p (-2) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l1053_105386


namespace NUMINAMATH_GPT_min_distance_origin_to_intersections_l1053_105399

theorem min_distance_origin_to_intersections (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hline : (1 : ℝ)/a + 4/b = 1) :
  |(0 : ℝ) - a| + |(0 : ℝ) - b| = 9 :=
sorry

end NUMINAMATH_GPT_min_distance_origin_to_intersections_l1053_105399


namespace NUMINAMATH_GPT_division_of_fractions_l1053_105316

theorem division_of_fractions : (2 / 3) / (1 / 4) = (8 / 3) := by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l1053_105316


namespace NUMINAMATH_GPT_temp_interpretation_l1053_105371

theorem temp_interpretation (below_zero : ℤ) (above_zero : ℤ) (h : below_zero = -2):
  above_zero = 3 → 3 = 0 := by
  intro h2
  have : above_zero = 3 := h2
  sorry

end NUMINAMATH_GPT_temp_interpretation_l1053_105371


namespace NUMINAMATH_GPT_cost_of_remaining_shirt_l1053_105340

theorem cost_of_remaining_shirt :
  ∀ (shirts total_cost cost_per_shirt remaining_shirt_cost : ℕ),
  shirts = 5 →
  total_cost = 85 →
  cost_per_shirt = 15 →
  (3 * cost_per_shirt) + (2 * remaining_shirt_cost) = total_cost →
  remaining_shirt_cost = 20 :=
by
  intros shirts total_cost cost_per_shirt remaining_shirt_cost
  intros h_shirts h_total h_cost_per_shirt h_equation
  sorry

end NUMINAMATH_GPT_cost_of_remaining_shirt_l1053_105340


namespace NUMINAMATH_GPT_find_b_l1053_105356

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  2 * x / (x^2 + b * x + 1)

noncomputable def f_inverse (y : ℝ) : ℝ :=
  (1 - y) / y

theorem find_b (b : ℝ) (h : ∀ x, f_inverse (f x b) = x) : b = 4 :=
sorry

end NUMINAMATH_GPT_find_b_l1053_105356


namespace NUMINAMATH_GPT_range_of_a_l1053_105349

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (x^2 - 2*a*x + a) > 0) → (a ≤ 0 ∨ a ≥ 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_range_of_a_l1053_105349


namespace NUMINAMATH_GPT_number_of_kg_of_mangoes_l1053_105357

variable {m : ℕ}
def cost_apples := 8 * 70
def cost_mangoes (m : ℕ) := 75 * m
def total_cost := 1235

theorem number_of_kg_of_mangoes (h : cost_apples + cost_mangoes m = total_cost) : m = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_kg_of_mangoes_l1053_105357


namespace NUMINAMATH_GPT_union_A_B_l1053_105373

def setA : Set ℝ := { x | Real.log x / Real.log (1/2) > -1 }
def setB : Set ℝ := { x | 2^x > Real.sqrt 2 }

theorem union_A_B : setA ∪ setB = { x | 0 < x } := by
  sorry

end NUMINAMATH_GPT_union_A_B_l1053_105373


namespace NUMINAMATH_GPT_probability_of_black_ball_l1053_105325

/-- Let the probability of drawing a red ball be 0.42, and the probability of drawing a white ball be 0.28. Prove that the probability of drawing a black ball is 0.3. -/
theorem probability_of_black_ball (p_red p_white p_black : ℝ) (h1 : p_red = 0.42) (h2 : p_white = 0.28) (h3 : p_red + p_white + p_black = 1) : p_black = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_black_ball_l1053_105325


namespace NUMINAMATH_GPT_gift_equation_l1053_105377

theorem gift_equation (x : ℝ) : 15 * (x + 40) = 900 := 
by
  sorry

end NUMINAMATH_GPT_gift_equation_l1053_105377


namespace NUMINAMATH_GPT_number_of_crowns_l1053_105310

-- Define the conditions
def feathers_per_crown : ℕ := 7
def total_feathers : ℕ := 6538

-- Theorem statement
theorem number_of_crowns : total_feathers / feathers_per_crown = 934 :=
by {
  sorry  -- proof omitted
}

end NUMINAMATH_GPT_number_of_crowns_l1053_105310


namespace NUMINAMATH_GPT_range_of_a_l1053_105365

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x - a

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > 2 - 2 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1053_105365


namespace NUMINAMATH_GPT_original_price_l1053_105332

variable (p q : ℝ)

theorem original_price (x : ℝ)
  (hp : x * (1 + p / 100) * (1 - q / 100) = 1) :
  x = 10000 / (10000 + 100 * (p - q) - p * q) :=
sorry

end NUMINAMATH_GPT_original_price_l1053_105332


namespace NUMINAMATH_GPT_king_middle_school_teachers_l1053_105385

theorem king_middle_school_teachers 
    (students : ℕ)
    (classes_per_student : ℕ)
    (normal_class_size : ℕ)
    (special_classes : ℕ)
    (special_class_size : ℕ)
    (classes_per_teacher : ℕ)
    (H1 : students = 1500)
    (H2 : classes_per_student = 5)
    (H3 : normal_class_size = 30)
    (H4 : special_classes = 10)
    (H5 : special_class_size = 15)
    (H6 : classes_per_teacher = 3) : 
    ∃ teachers : ℕ, teachers = 85 :=
by
  sorry

end NUMINAMATH_GPT_king_middle_school_teachers_l1053_105385


namespace NUMINAMATH_GPT_reflection_proof_l1053_105302

def original_center : (ℝ × ℝ) := (8, -3)
def reflection_line (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)
def reflected_center : (ℝ × ℝ) := reflection_line original_center

theorem reflection_proof : reflected_center = (-3, -8) := by
  sorry

end NUMINAMATH_GPT_reflection_proof_l1053_105302


namespace NUMINAMATH_GPT_proof_allison_brian_noah_l1053_105393

-- Definitions based on the problem conditions

-- Definition for the cubes
def allison_cube := [6, 6, 6, 6, 6, 6]
def brian_cube := [1, 2, 2, 3, 3, 4]
def noah_cube := [3, 3, 3, 3, 5, 5]

-- Helper function to calculate the probability of succeeding conditions
def probability_succeeding (A B C : List ℕ) : ℚ :=
  if (A.all (λ x => x = 6)) ∧ (B.all (λ x => x ≤ 5)) ∧ (C.all (λ x => x ≤ 5)) then 1 else 0

-- Define the proof statement for the given problem
theorem proof_allison_brian_noah :
  probability_succeeding allison_cube brian_cube noah_cube = 1 :=
by
  -- Since all conditions fulfill the requirement, we'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_GPT_proof_allison_brian_noah_l1053_105393


namespace NUMINAMATH_GPT_greatest_remainder_when_dividing_by_10_l1053_105388

theorem greatest_remainder_when_dividing_by_10 (x : ℕ) : 
  ∃ r : ℕ, r < 10 ∧ r = x % 10 ∧ r = 9 :=
by
  sorry

end NUMINAMATH_GPT_greatest_remainder_when_dividing_by_10_l1053_105388


namespace NUMINAMATH_GPT_equal_distribution_l1053_105376

variables (Emani Howard : ℕ)

-- Emani has $30 more than Howard
axiom emani_condition : Emani = Howard + 30

-- Emani has $150
axiom emani_has_money : Emani = 150

theorem equal_distribution : (Emani + Howard) / 2 = 135 :=
by
  sorry

end NUMINAMATH_GPT_equal_distribution_l1053_105376


namespace NUMINAMATH_GPT_solve_inequality_system_l1053_105368

theorem solve_inequality_system (x : ℝ) :
  (x - 1 < 2 * x + 1) ∧ ((2 * x - 5) / 3 ≤ 1) → (-2 < x ∧ x ≤ 4) :=
by
  intro cond
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1053_105368


namespace NUMINAMATH_GPT_dave_winfield_home_runs_l1053_105312

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end NUMINAMATH_GPT_dave_winfield_home_runs_l1053_105312


namespace NUMINAMATH_GPT_inequality_solution_l1053_105341

theorem inequality_solution {x : ℝ} : 5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1053_105341


namespace NUMINAMATH_GPT_weight_distribution_l1053_105324

theorem weight_distribution (x y z : ℕ) 
  (h1 : x + y + z = 100) 
  (h2 : x + 10 * y + 50 * z = 500) : 
  x = 60 ∧ y = 39 ∧ z = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_distribution_l1053_105324


namespace NUMINAMATH_GPT_H_function_is_f_x_abs_x_l1053_105351

-- Definition: A function f is odd if ∀ x ∈ ℝ, f(-x) = -f(x)
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition: A function f is strictly increasing if ∀ x1, x2 ∈ ℝ, x1 < x2 implies f(x1) < f(x2)
def is_strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- Define the function f(x) = x * |x|
def f (x : ℝ) : ℝ := x * abs x

-- The main theorem which states that f(x) = x * |x| is an "H function"
theorem H_function_is_f_x_abs_x : is_odd f ∧ is_strictly_increasing f :=
  sorry

end NUMINAMATH_GPT_H_function_is_f_x_abs_x_l1053_105351


namespace NUMINAMATH_GPT_stacy_grew_more_l1053_105344

variable (initial_height_stacy current_height_stacy brother_growth stacy_growth_more : ℕ)

-- Conditions
def stacy_initial_height : initial_height_stacy = 50 := by sorry
def stacy_current_height : current_height_stacy = 57 := by sorry
def brother_growth_last_year : brother_growth = 1 := by sorry

-- Compute Stacy's growth
def stacy_growth : ℕ := current_height_stacy - initial_height_stacy

-- Prove the difference in growth
theorem stacy_grew_more :
  stacy_growth - brother_growth = stacy_growth_more → stacy_growth_more = 6 := 
by sorry

end NUMINAMATH_GPT_stacy_grew_more_l1053_105344
