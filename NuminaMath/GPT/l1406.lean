import Mathlib

namespace NUMINAMATH_GPT_square_perimeter_l1406_140657

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : perimeter_of_square side_length = 20 := by
  sorry

end NUMINAMATH_GPT_square_perimeter_l1406_140657


namespace NUMINAMATH_GPT_newspaper_price_l1406_140656

-- Define the conditions as variables
variables 
  (P : ℝ)                    -- Price per edition for Wednesday, Thursday, and Friday
  (total_cost : ℝ := 28)     -- Total cost over 8 weeks
  (sunday_cost : ℝ := 2)     -- Cost of Sunday edition
  (weeks : ℕ := 8)           -- Number of weeks
  (wednesday_thursday_friday_editions : ℕ := 3 * weeks) -- Total number of editions for Wednesday, Thursday, and Friday over 8 weeks

-- Math proof problem statement
theorem newspaper_price : 
  (total_cost - weeks * sunday_cost) / wednesday_thursday_friday_editions = 0.5 :=
  sorry

end NUMINAMATH_GPT_newspaper_price_l1406_140656


namespace NUMINAMATH_GPT_factor_polynomial_l1406_140647

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1406_140647


namespace NUMINAMATH_GPT_solve_for_x_l1406_140627

theorem solve_for_x (x : ℝ) (h : 9 / (5 + x / 0.75) = 1) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1406_140627


namespace NUMINAMATH_GPT_solve_xy_l1406_140677

theorem solve_xy : ∃ x y : ℝ, (x - y = 10 ∧ x^2 + y^2 = 100) ↔ ((x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0)) := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_xy_l1406_140677


namespace NUMINAMATH_GPT_find_atomic_weight_of_Na_l1406_140617

def atomic_weight_of_Na_is_correct : Prop :=
  ∃ (atomic_weight_of_Na : ℝ),
    (atomic_weight_of_Na + 35.45 + 16.00 = 74) ∧ (atomic_weight_of_Na = 22.55)

theorem find_atomic_weight_of_Na : atomic_weight_of_Na_is_correct :=
by
  sorry

end NUMINAMATH_GPT_find_atomic_weight_of_Na_l1406_140617


namespace NUMINAMATH_GPT_determine_k_l1406_140633

theorem determine_k (k : ℤ) : (∀ n : ℤ, gcd (4 * n + 1) (k * n + 1) = 1) ↔ 
  (∃ m : ℕ, k = 4 + 2 ^ m ∨ k = 4 - 2 ^ m) :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l1406_140633


namespace NUMINAMATH_GPT_min_distance_to_line_l1406_140641

theorem min_distance_to_line : 
  let A := 5
  let B := -3
  let C := 4
  let d (x₀ y₀ : ℤ) := (abs (A * x₀ + B * y₀ + C) : ℝ) / (Real.sqrt (A ^ 2 + B ^ 2))
  ∃ (x₀ y₀ : ℤ), d x₀ y₀ = Real.sqrt 34 / 85 := 
by 
  sorry

end NUMINAMATH_GPT_min_distance_to_line_l1406_140641


namespace NUMINAMATH_GPT_inequality_proof_l1406_140674

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n) (h2 : (Finset.univ.sum a) ≥ 0) :
  (Finset.univ.sum (λ i => Real.sqrt (a i ^ 2 + 1))) ≥
  Real.sqrt (2 * n * (Finset.univ.sum a)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1406_140674


namespace NUMINAMATH_GPT_problem_1_solution_problem_2_solution_l1406_140662

noncomputable def problem_1 : Real :=
  (-3) + (2 - Real.pi)^0 - (1 / 2)⁻¹

theorem problem_1_solution :
  problem_1 = -4 :=
by
  sorry

noncomputable def problem_2 (a : Real) : Real :=
  (2 * a)^3 - a * a^2 + 3 * a^6 / a^3

theorem problem_2_solution (a : Real) :
  problem_2 a = 10 * a^3 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_solution_problem_2_solution_l1406_140662


namespace NUMINAMATH_GPT_three_digit_cubes_divisible_by_8_and_9_l1406_140673

theorem three_digit_cubes_divisible_by_8_and_9 : 
  ∃! n : ℕ, (216 ≤ n^3 ∧ n^3 ≤ 999) ∧ (n % 6 = 0) :=
sorry

end NUMINAMATH_GPT_three_digit_cubes_divisible_by_8_and_9_l1406_140673


namespace NUMINAMATH_GPT_area_ratio_trapezoid_triangle_l1406_140665

-- Define the geometric elements and given conditions.
variable (AB CD EAB ABCD : ℝ)
variable (trapezoid_ABCD : AB = 10)
variable (trapezoid_ABCD_CD : CD = 25)
variable (ratio_areas_EDC_EAB : (CD / AB)^2 = 25 / 4)
variable (trapezoid_relation : (ABCD + EAB) / EAB = 25 / 4)

-- The goal is to prove the ratio of the areas of triangle EAB to trapezoid ABCD.
theorem area_ratio_trapezoid_triangle :
  (EAB / ABCD) = 4 / 21 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_trapezoid_triangle_l1406_140665


namespace NUMINAMATH_GPT_belle_rawhide_bones_per_evening_l1406_140636

theorem belle_rawhide_bones_per_evening 
  (cost_rawhide_bone : ℝ)
  (cost_dog_biscuit : ℝ)
  (num_dog_biscuits_per_evening : ℕ)
  (total_weekly_cost : ℝ)
  (days_per_week : ℕ)
  (rawhide_bones_per_evening : ℕ)
  (h1 : cost_rawhide_bone = 1)
  (h2 : cost_dog_biscuit = 0.25)
  (h3 : num_dog_biscuits_per_evening = 4)
  (h4 : total_weekly_cost = 21)
  (h5 : days_per_week = 7)
  (h6 : rawhide_bones_per_evening * cost_rawhide_bone * (days_per_week : ℝ) = total_weekly_cost - num_dog_biscuits_per_evening * cost_dog_biscuit * (days_per_week : ℝ)) :
  rawhide_bones_per_evening = 2 := 
sorry

end NUMINAMATH_GPT_belle_rawhide_bones_per_evening_l1406_140636


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1406_140640

-- Defining the conditions
def a : Int := -3
def b : Int := -2

-- Defining the expression
def expr (a b : Int) : Int := (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2)

-- Stating the theorem/proof problem
theorem simplify_and_evaluate : expr a b = -6 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1406_140640


namespace NUMINAMATH_GPT_intersection_points_count_l1406_140658

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ (∀ x, f x = g x → x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_count_l1406_140658


namespace NUMINAMATH_GPT_calculate_expression_l1406_140603

theorem calculate_expression (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2 * a^2 * b^2 + b^4 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1406_140603


namespace NUMINAMATH_GPT_complement_intersection_l1406_140644

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_intersection :
  compl A ∩ B = {-2, -1} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1406_140644


namespace NUMINAMATH_GPT_james_marbles_left_l1406_140615

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end NUMINAMATH_GPT_james_marbles_left_l1406_140615


namespace NUMINAMATH_GPT_triplet_solution_l1406_140631

theorem triplet_solution (x y z : ℝ) 
  (h1 : y = (x^3 + 12 * x) / (3 * x^2 + 4))
  (h2 : z = (y^3 + 12 * y) / (3 * y^2 + 4))
  (h3 : x = (z^3 + 12 * z) / (3 * z^2 + 4)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ 
  (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end NUMINAMATH_GPT_triplet_solution_l1406_140631


namespace NUMINAMATH_GPT_maximize_profit_l1406_140678

noncomputable def profit_function (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

theorem maximize_profit :
  (∀ x : ℝ, 30 ≤ x ∧ x ≤ 54 → profit_function x ≤ 432) ∧ profit_function 42 = 432 := sorry

end NUMINAMATH_GPT_maximize_profit_l1406_140678


namespace NUMINAMATH_GPT_additional_people_needed_l1406_140612

-- Definition of the conditions
def person_hours (people: ℕ) (hours: ℕ) : ℕ := people * hours

-- Assertion that 8 people can paint the fence in 3 hours
def eight_people_three_hours : Prop := person_hours 8 3 = 24

-- Definition of the additional people required
def additional_people (initial_people required_people: ℕ) : ℕ := required_people - initial_people

-- Main theorem stating the problem
theorem additional_people_needed : eight_people_three_hours → additional_people 8 12 = 4 :=
by
  sorry

end NUMINAMATH_GPT_additional_people_needed_l1406_140612


namespace NUMINAMATH_GPT_abc_equal_l1406_140638

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_abc_equal_l1406_140638


namespace NUMINAMATH_GPT_arithmetic_geometric_ratio_l1406_140685

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
1 + 3 = a1 + a2

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
b2 ^ 2 = 4

theorem arithmetic_geometric_ratio (a1 a2 b2 : ℝ) 
  (h1 : arithmetic_sequence a1 a2) 
  (h2 : geometric_sequence b2) : 
  (a1 + a2) / b2 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_arithmetic_geometric_ratio_l1406_140685


namespace NUMINAMATH_GPT_birds_flew_up_l1406_140659

theorem birds_flew_up (original_birds total_birds birds_flew_up : ℕ) 
  (h1 : original_birds = 14)
  (h2 : total_birds = 35)
  (h3 : total_birds = original_birds + birds_flew_up) :
  birds_flew_up = 21 :=
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_birds_flew_up_l1406_140659


namespace NUMINAMATH_GPT_circle_symmetric_line_l1406_140690

theorem circle_symmetric_line (a b : ℝ) (h : a < 2) (hb : b = -2) : a + b < 0 := by
  sorry

end NUMINAMATH_GPT_circle_symmetric_line_l1406_140690


namespace NUMINAMATH_GPT_ways_to_stand_on_staircase_l1406_140618

theorem ways_to_stand_on_staircase (A B C : Type) (steps : Fin 7) : 
  ∃ ways : Nat, ways = 336 := by sorry

end NUMINAMATH_GPT_ways_to_stand_on_staircase_l1406_140618


namespace NUMINAMATH_GPT_line_curve_intersection_l1406_140689

theorem line_curve_intersection (a : ℝ) : 
  (∃! (x y : ℝ), (y = a * (x + 2)) ∧ (x ^ 2 - y * |y| = 1)) ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
by
  sorry

end NUMINAMATH_GPT_line_curve_intersection_l1406_140689


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1406_140648

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  (a + b) / 2 = 11 / 16 :=
by 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  show (a + b) / 2 = 11 / 16
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1406_140648


namespace NUMINAMATH_GPT_unique_function_satisfying_conditions_l1406_140637

noncomputable def f : (ℝ → ℝ) := sorry

axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem unique_function_satisfying_conditions : ∀ x : ℝ, f x = x := sorry

end NUMINAMATH_GPT_unique_function_satisfying_conditions_l1406_140637


namespace NUMINAMATH_GPT_find_number_l1406_140696

theorem find_number (x : ℕ) (h : 112 * x = 70000) : x = 625 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1406_140696


namespace NUMINAMATH_GPT_area_of_given_rhombus_l1406_140634

open Real

noncomputable def area_of_rhombus_with_side_and_angle (side : ℝ) (angle : ℝ) : ℝ :=
  let half_diag1 := side * cos (angle / 2)
  let half_diag2 := side * sin (angle / 2)
  let diag1 := 2 * half_diag1
  let diag2 := 2 * half_diag2
  (diag1 * diag2) / 2

theorem area_of_given_rhombus :
  area_of_rhombus_with_side_and_angle 25 40 = 201.02 :=
by
  sorry

end NUMINAMATH_GPT_area_of_given_rhombus_l1406_140634


namespace NUMINAMATH_GPT_stone_length_l1406_140609

theorem stone_length (hall_length_m : ℕ) (hall_breadth_m : ℕ) (number_of_stones : ℕ) (stone_width_dm : ℕ) 
    (length_in_dm : 10 > 0) :
    hall_length_m = 36 → hall_breadth_m = 15 → number_of_stones = 2700 → stone_width_dm = 5 →
    ∀ L : ℕ, 
    (10 * hall_length_m) * (10 * hall_breadth_m) = number_of_stones * (L * stone_width_dm) → 
    L = 4 :=
by
  intros h1 h2 h3 h4
  simp at *
  sorry

end NUMINAMATH_GPT_stone_length_l1406_140609


namespace NUMINAMATH_GPT_total_copies_in_half_hour_l1406_140651

-- Define the rates of the copy machines
def rate_machine1 : ℕ := 35
def rate_machine2 : ℕ := 65

-- Define the duration of time in minutes
def time_minutes : ℕ := 30

-- Define the total number of copies made by both machines in the given duration
def total_copies_made : ℕ := rate_machine1 * time_minutes + rate_machine2 * time_minutes

-- Prove that the total number of copies made is 3000
theorem total_copies_in_half_hour : total_copies_made = 3000 := by
  -- The proof is skipped with sorry for the demonstration purpose
  sorry

end NUMINAMATH_GPT_total_copies_in_half_hour_l1406_140651


namespace NUMINAMATH_GPT_polynomial_remainder_l1406_140672

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ), (3 * X^5 - 2 * X^3 + 5 * X - 9) = (X - 1) * (X - 2) * q + (92 * X - 95) :=
by
  intro q
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1406_140672


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l1406_140676

-- Define the condition that a quadratic equation has real roots given ac < 0

variable {a b c : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_real_roots (h : a * c < 0) : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l1406_140676


namespace NUMINAMATH_GPT_unique_solution_quadratic_eq_l1406_140682

theorem unique_solution_quadratic_eq (q : ℚ) (hq : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_quadratic_eq_l1406_140682


namespace NUMINAMATH_GPT_proof_x_plus_y_l1406_140652

variables (x y : ℝ)

-- Definitions for the given conditions
def cond1 (x y : ℝ) : Prop := 2 * |x| + x + y = 18
def cond2 (x y : ℝ) : Prop := x + 2 * |y| - y = 14

theorem proof_x_plus_y (x y : ℝ) (h1 : cond1 x y) (h2 : cond2 x y) : x + y = 14 := by
  sorry

end NUMINAMATH_GPT_proof_x_plus_y_l1406_140652


namespace NUMINAMATH_GPT_additional_discount_A_is_8_l1406_140623

-- Define the problem conditions
def full_price_A : ℝ := 125
def full_price_B : ℝ := 130
def discount_B : ℝ := 0.10
def price_difference : ℝ := 2

-- Define the unknown additional discount of store A
def discount_A (x : ℝ) : Prop :=
  full_price_A - (full_price_A * (x / 100)) = (full_price_B - (full_price_B * discount_B)) - price_difference

-- Theorem stating that the additional discount offered by store A is 8%
theorem additional_discount_A_is_8 : discount_A 8 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_additional_discount_A_is_8_l1406_140623


namespace NUMINAMATH_GPT_patrick_purchased_pencils_l1406_140621

theorem patrick_purchased_pencils 
  (S : ℝ) -- selling price of one pencil
  (C : ℝ) -- cost price of one pencil
  (P : ℕ) -- number of pencils purchased
  (h1 : C = 1.3333333333333333 * S) -- condition 1: cost of pencils is 1.3333333 times the selling price
  (h2 : (P : ℝ) * C - (P : ℝ) * S = 20 * S) -- condition 2: loss equals selling price of 20 pencils
  : P = 60 := 
sorry

end NUMINAMATH_GPT_patrick_purchased_pencils_l1406_140621


namespace NUMINAMATH_GPT_range_of_function_l1406_140688

theorem range_of_function : ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2)/(x^2 + x + 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l1406_140688


namespace NUMINAMATH_GPT_arithmetic_progression_20th_term_and_sum_l1406_140681

theorem arithmetic_progression_20th_term_and_sum :
  let a := 3
  let d := 4
  let n := 20
  let a_20 := a + (n - 1) * d
  let S_20 := n / 2 * (a + a_20)
  a_20 = 79 ∧ S_20 = 820 := by
    let a := 3
    let d := 4
    let n := 20
    let a_20 := a + (n - 1) * d
    let S_20 := n / 2 * (a + a_20)
    sorry

end NUMINAMATH_GPT_arithmetic_progression_20th_term_and_sum_l1406_140681


namespace NUMINAMATH_GPT_total_import_value_l1406_140680

-- Define the given conditions
def export_value : ℝ := 8.07
def additional_amount : ℝ := 1.11
def factor : ℝ := 1.5

-- Define the import value to be proven
def import_value : ℝ := 46.4

-- Main theorem statement
theorem total_import_value :
  export_value = factor * import_value + additional_amount → import_value = 46.4 :=
by sorry

end NUMINAMATH_GPT_total_import_value_l1406_140680


namespace NUMINAMATH_GPT_PB_length_l1406_140629

/-- In a square ABCD with area 1989 cm², with the center O, and
a point P inside such that ∠OPB = 45° and PA : PB = 5 : 14,
prove that PB = 42 cm. -/
theorem PB_length (s PA PB : ℝ) (h₁ : s^2 = 1989) 
(h₂ : PA / PB = 5 / 14) 
(h₃ : 25 * (PA / PB)^2 + 196 * (PB / PA)^2 = s^2) :
  PB = 42 := 
by sorry

end NUMINAMATH_GPT_PB_length_l1406_140629


namespace NUMINAMATH_GPT_at_least_one_neg_l1406_140650

theorem at_least_one_neg (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
sorry

end NUMINAMATH_GPT_at_least_one_neg_l1406_140650


namespace NUMINAMATH_GPT_value_of_b_minus_d_squared_l1406_140642

theorem value_of_b_minus_d_squared
  (a b c d : ℤ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 9) :
  (b - d) ^ 2 = 4 :=
sorry

end NUMINAMATH_GPT_value_of_b_minus_d_squared_l1406_140642


namespace NUMINAMATH_GPT_length_of_arc_l1406_140695

variable {O A B : Type}
variable (angle_OAB : Real) (radius_OA : Real)

theorem length_of_arc (h1 : angle_OAB = 45) (h2 : radius_OA = 5) :
  (length_of_arc_AB = 5 * π / 4) :=
sorry

end NUMINAMATH_GPT_length_of_arc_l1406_140695


namespace NUMINAMATH_GPT_cricket_scores_l1406_140649

-- Define the conditions
variable (X : ℝ) (A B C D E average10 average6 : ℝ)
variable (matches10 matches6 : ℕ)

-- Set the given constants
axiom average_runs_10 : average10 = 38.9
axiom matches_10 : matches10 = 10
axiom average_runs_6 : average6 = 42
axiom matches_6 : matches6 = 6

-- Define the equations based on the conditions
axiom eq1 : X = average10 * matches10
axiom eq2 : A + B + C + D = X - (average6 * matches6)
axiom eq3 : E = (A + B + C + D) / 4

-- The target statement
theorem cricket_scores : X = 389 ∧ A + B + C + D = 137 ∧ E = 34.25 :=
  by
    sorry

end NUMINAMATH_GPT_cricket_scores_l1406_140649


namespace NUMINAMATH_GPT_range_of_m_l1406_140605

variable {x m : ℝ}

def condition_p (x : ℝ) : Prop := |x - 3| ≤ 2
def condition_q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, ¬(condition_p x) → ¬(condition_q x m)) ∧ ¬(∀ x, ¬(condition_q x m) → ¬(condition_p x)) →
  2 < m ∧ m < 4 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1406_140605


namespace NUMINAMATH_GPT_largest_divisor_of_five_even_numbers_l1406_140684

theorem largest_divisor_of_five_even_numbers (n : ℕ) (h₁ : n % 2 = 1) : 
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) ∧ 
       (∀ d', (∀ n, n % 2 = 1 → d' ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) → d' ≤ d) ∧ 
       d = 480 := sorry

end NUMINAMATH_GPT_largest_divisor_of_five_even_numbers_l1406_140684


namespace NUMINAMATH_GPT_abs_x_gt_1_iff_x_sq_minus1_gt_0_l1406_140645

theorem abs_x_gt_1_iff_x_sq_minus1_gt_0 (x : ℝ) : (|x| > 1) ↔ (x^2 - 1 > 0) := by
  sorry

end NUMINAMATH_GPT_abs_x_gt_1_iff_x_sq_minus1_gt_0_l1406_140645


namespace NUMINAMATH_GPT_remove_terms_for_desired_sum_l1406_140600

theorem remove_terms_for_desired_sum :
  let series_sum := (1/3) + (1/5) + (1/7) + (1/9) + (1/11) + (1/13)
  series_sum - (1/11 + 1/13) = 11/20 :=
by
  sorry

end NUMINAMATH_GPT_remove_terms_for_desired_sum_l1406_140600


namespace NUMINAMATH_GPT_variance_uniform_l1406_140679

noncomputable def variance_of_uniform (α β : ℝ) (h : α < β) : ℝ :=
  let E := (α + β) / 2
  (β - α)^2 / 12

theorem variance_uniform (α β : ℝ) (h : α < β) :
  variance_of_uniform α β h = (β - α)^2 / 12 :=
by
  -- statement of proof only, actual proof here is sorry
  sorry

end NUMINAMATH_GPT_variance_uniform_l1406_140679


namespace NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_main_l1406_140654

example (x : ℝ) : (x^2 - 3 * x > 0) → (x > 4) ∨ (x < 0 ∧ x > 0) := by
  sorry

theorem necessary_condition (x : ℝ) :
  (x^2 - 3 * x > 0) → (x > 4) :=
by
  sorry

theorem not_sufficient_condition (x : ℝ) :
  ¬ (x > 4) → (x^2 - 3 * x > 0) :=
by
  sorry

theorem main (x : ℝ) :
  (x^2 - 3 * x > 0) ↔ ¬ (x > 4) :=
by
  sorry

end NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_main_l1406_140654


namespace NUMINAMATH_GPT_rectangle_not_sum_110_l1406_140666

noncomputable def not_sum_110 : Prop :=
  ∀ (w : ℕ), (w > 0) → (2 * w^2 + 6 * w ≠ 110)

theorem rectangle_not_sum_110 : not_sum_110 := 
  sorry

end NUMINAMATH_GPT_rectangle_not_sum_110_l1406_140666


namespace NUMINAMATH_GPT_distinguishable_large_triangles_l1406_140660

def num_of_distinguishable_large_eq_triangles : Nat :=
  let colors := 8
  let pairs := 7 + Nat.choose 7 2
  colors * pairs

theorem distinguishable_large_triangles : num_of_distinguishable_large_eq_triangles = 224 := by
  sorry

end NUMINAMATH_GPT_distinguishable_large_triangles_l1406_140660


namespace NUMINAMATH_GPT_no_fraternity_member_is_club_member_l1406_140639

variable {U : Type} -- Domain of discourse, e.g., the set of all people at the school
variables (Club Member Student Honest Fraternity : U → Prop)

theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, Club x → Student x)
  (h2 : ∀ x, Club x → ¬ Honest x)
  (h3 : ∀ x, Fraternity x → Honest x) :
  ∀ x, Fraternity x → ¬ Club x := 
sorry

end NUMINAMATH_GPT_no_fraternity_member_is_club_member_l1406_140639


namespace NUMINAMATH_GPT_optimal_roof_angle_no_friction_l1406_140635

theorem optimal_roof_angle_no_friction {g x : ℝ} (hg : 0 < g) (hx : 0 < x) :
  ∃ α : ℝ, α = 45 :=
by
  sorry

end NUMINAMATH_GPT_optimal_roof_angle_no_friction_l1406_140635


namespace NUMINAMATH_GPT_wickets_before_last_match_l1406_140670

-- Define the conditions
variable (W : ℕ)

-- Initial average
def initial_avg : ℝ := 12.4

-- Runs given in the last match
def runs_last_match : ℝ := 26

-- Wickets taken in the last match
def wickets_last_match : ℕ := 4

-- The new average after the last match
def new_avg : ℝ := initial_avg - 0.4

-- Prove the theorem
theorem wickets_before_last_match :
  (12.4 * W + runs_last_match) / (W + wickets_last_match) = new_avg → W = 55 :=
by
  sorry

end NUMINAMATH_GPT_wickets_before_last_match_l1406_140670


namespace NUMINAMATH_GPT_quiz_minimum_correct_l1406_140686

theorem quiz_minimum_correct (x : ℕ) (hx : 7 * x + 14 ≥ 120) : x ≥ 16 := 
by sorry

end NUMINAMATH_GPT_quiz_minimum_correct_l1406_140686


namespace NUMINAMATH_GPT_division_example_l1406_140694

theorem division_example : 0.45 / 0.005 = 90 := by
  sorry

end NUMINAMATH_GPT_division_example_l1406_140694


namespace NUMINAMATH_GPT_sum_reciprocal_transformation_l1406_140668

theorem sum_reciprocal_transformation 
  (a b c d S : ℝ) 
  (h1 : a + b + c + d = S)
  (h2 : 1 / a + 1 / b + 1 / c + 1 / d = S)
  (h3 : a ≠ 0 ∧ a ≠ 1)
  (h4 : b ≠ 0 ∧ b ≠ 1)
  (h5 : c ≠ 0 ∧ c ≠ 1)
  (h6 : d ≠ 0 ∧ d ≠ 1) :
  S = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocal_transformation_l1406_140668


namespace NUMINAMATH_GPT_ten_percent_of_x_is_17_85_l1406_140607

-- Define the conditions and the proof statement
theorem ten_percent_of_x_is_17_85 :
  ∃ x : ℝ, (3 - (1/4) * 2 - (1/3) * 3 - (1/7) * x = 27) ∧ (0.10 * x = 17.85) := sorry

end NUMINAMATH_GPT_ten_percent_of_x_is_17_85_l1406_140607


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1406_140606

variable {a : ℝ}

theorem sufficient_not_necessary_condition (ha : a > 1 / a^2) :
  a^2 > 1 / a ∧ ∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1406_140606


namespace NUMINAMATH_GPT_solve_for_N_l1406_140610

theorem solve_for_N (N : ℤ) (h1 : N < 0) (h2 : 2 * N * N + N = 15) : N = -3 :=
sorry

end NUMINAMATH_GPT_solve_for_N_l1406_140610


namespace NUMINAMATH_GPT_value_of_p_l1406_140614

theorem value_of_p (x y p : ℝ) 
  (h1 : 3 * x - 2 * y = 4 - p) 
  (h2 : 4 * x - 3 * y = 2 + p) 
  (h3 : x > y) : 
  p < -1 := 
sorry

end NUMINAMATH_GPT_value_of_p_l1406_140614


namespace NUMINAMATH_GPT_painted_cubes_l1406_140697

theorem painted_cubes (n : ℕ) (h1 : 3 < n)
  (h2 : 6 * (n - 2)^2 = 12 * (n - 2)) :
  n = 4 := by
  sorry

end NUMINAMATH_GPT_painted_cubes_l1406_140697


namespace NUMINAMATH_GPT_sum_of_ages_is_26_l1406_140622

def Yoongi_aunt_age := 38
def Yoongi_age := Yoongi_aunt_age - 23
def Hoseok_age := Yoongi_age - 4
def sum_of_ages := Yoongi_age + Hoseok_age

theorem sum_of_ages_is_26 : sum_of_ages = 26 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_26_l1406_140622


namespace NUMINAMATH_GPT_range_of_a_l1406_140602

theorem range_of_a
  (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 6)
  (y : ℝ) (hy : 0 < y)
  (h : (y / 4 - 2 * (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) :
  a ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1406_140602


namespace NUMINAMATH_GPT_inequality_solution_set_impossible_l1406_140691

theorem inequality_solution_set_impossible (a b : ℝ) (h_b : b ≠ 0) : ¬ (a = 0 ∧ ∀ x, ax + b > 0 ∧ x > (b / a)) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_set_impossible_l1406_140691


namespace NUMINAMATH_GPT_problem_statement_l1406_140667

noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 11
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 11

theorem problem_statement : (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 :=
by
  -- conditions
  let a := Real.sqrt 3 - Real.sqrt 11
  let b := Real.sqrt 3 + Real.sqrt 11
  have h1 : a = Real.sqrt 3 - Real.sqrt 11 := rfl
  have h2 : b = Real.sqrt 3 + Real.sqrt 11 := rfl
  -- question statement
  sorry

end NUMINAMATH_GPT_problem_statement_l1406_140667


namespace NUMINAMATH_GPT_group_size_is_eight_l1406_140687

/-- Theorem: The number of people in the group is 8 if the 
average weight increases by 6 kg when a new person replaces 
one weighing 45 kg, and the weight of the new person is 93 kg. -/
theorem group_size_is_eight
    (n : ℕ)
    (H₁ : 6 * n = 48)
    (H₂ : 93 - 45 = 48) :
    n = 8 :=
by
  sorry

end NUMINAMATH_GPT_group_size_is_eight_l1406_140687


namespace NUMINAMATH_GPT_find_surface_area_of_ball_l1406_140604

noncomputable def surface_area_of_ball : ℝ :=
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area

theorem find_surface_area_of_ball :
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area = (2 / 3) * Real.pi :=
by
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  sorry

end NUMINAMATH_GPT_find_surface_area_of_ball_l1406_140604


namespace NUMINAMATH_GPT_max_g_value_on_interval_l1406_140661

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end NUMINAMATH_GPT_max_g_value_on_interval_l1406_140661


namespace NUMINAMATH_GPT_solve_for_x_l1406_140653

theorem solve_for_x (x : ℝ) (h : (1 / 5) + (5 / x) = (12 / x) + (1 / 12)) : x = 60 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1406_140653


namespace NUMINAMATH_GPT_solve_for_x_l1406_140613

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1406_140613


namespace NUMINAMATH_GPT_expand_square_binomial_l1406_140663

variable (m n : ℝ)

theorem expand_square_binomial : (3 * m - n) ^ 2 = 9 * m ^ 2 - 6 * m * n + n ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_expand_square_binomial_l1406_140663


namespace NUMINAMATH_GPT_minimum_cost_for_13_bottles_l1406_140693

def cost_per_bottle_shop_A := 200 -- in cents
def discount_shop_B := 15 / 100 -- discount
def promotion_B_threshold := 4
def promotion_A_threshold := 4

-- Function to calculate the cost in Shop A for given number of bottles
def shop_A_cost (bottles : ℕ) : ℕ :=
  let batches := bottles / 5
  let remainder := bottles % 5
  (batches * 4 + remainder) * cost_per_bottle_shop_A

-- Function to calculate the cost in Shop B for given number of bottles
def shop_B_cost (bottles : ℕ) : ℕ :=
  if bottles >= promotion_B_threshold then
    (bottles * cost_per_bottle_shop_A) * (1 - discount_shop_B)
  else
    bottles * cost_per_bottle_shop_A

-- Function to calculate combined cost for given numbers of bottles from Shops A and B
def combined_cost (bottles_A bottles_B : ℕ) : ℕ :=
  shop_A_cost bottles_A + shop_B_cost bottles_B

theorem minimum_cost_for_13_bottles : ∃ a b, a + b = 13 ∧ combined_cost a b = 2000 := 
sorry

end NUMINAMATH_GPT_minimum_cost_for_13_bottles_l1406_140693


namespace NUMINAMATH_GPT_range_of_a_l1406_140616

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1406_140616


namespace NUMINAMATH_GPT_number_value_l1406_140625

theorem number_value (x : ℝ) (h : x = 3 * (1/x * -x) + 5) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_value_l1406_140625


namespace NUMINAMATH_GPT_johns_speed_l1406_140619

theorem johns_speed :
  ∀ (v : ℝ), 
    (∀ (t : ℝ), 24 = 30 * (t + 4 / 60) → 24 = v * (t - 8 / 60)) → 
    v = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_johns_speed_l1406_140619


namespace NUMINAMATH_GPT_exists_pair_satisfying_system_l1406_140643

theorem exists_pair_satisfying_system (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) ↔ m ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_pair_satisfying_system_l1406_140643


namespace NUMINAMATH_GPT_percentage_of_number_l1406_140608

theorem percentage_of_number (n : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * n = 16) : 0.4 * n = 192 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_of_number_l1406_140608


namespace NUMINAMATH_GPT_smallest_x_value_l1406_140655

theorem smallest_x_value (x : ℝ) (h : 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36)) : x = -3 :=
sorry

end NUMINAMATH_GPT_smallest_x_value_l1406_140655


namespace NUMINAMATH_GPT_smallest_nat_number_l1406_140611

theorem smallest_nat_number (x : ℕ) (h1 : 5 ∣ x) (h2 : 7 ∣ x) (h3 : x % 3 = 1) : x = 70 :=
sorry

end NUMINAMATH_GPT_smallest_nat_number_l1406_140611


namespace NUMINAMATH_GPT_eccentricity_range_l1406_140624

section EllipseEccentricity

variables {F1 F2 : ℝ × ℝ}
variable (M : ℝ × ℝ)

-- Conditions from a)
def is_orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def is_inside_ellipse (F1 F2 M : ℝ × ℝ) : Prop :=
  is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) ∧ 
  -- other conditions to assert M is inside could be defined but this is unspecified
  true

-- Statement from c)
theorem eccentricity_range {a b c e : ℝ}
  (h : ∀ (M: ℝ × ℝ), is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) → is_inside_ellipse F1 F2 M)
  (h1 : c^2 < a^2 - c^2)
  (h2 : e^2 = c^2 / a^2) :
  0 < e ∧ e < (Real.sqrt 2) / 2 := 
sorry

end EllipseEccentricity

end NUMINAMATH_GPT_eccentricity_range_l1406_140624


namespace NUMINAMATH_GPT_solve_for_s_l1406_140630

theorem solve_for_s (s : ℝ) :
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 14) = (s^2 - 3 * s - 18) / (s^2 - 2 * s - 24) →
  s = -5 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_s_l1406_140630


namespace NUMINAMATH_GPT_find_general_term_arithmetic_sequence_l1406_140626

-- Definitions needed
variable {a_n : ℕ → ℚ}
variable {S_n : ℕ → ℚ}

-- The main theorem to prove
theorem find_general_term_arithmetic_sequence 
  (h1 : a_n 4 - a_n 2 = 4)
  (h2 : S_n 3 = 9)
  (h3 : ∀ n : ℕ, S_n n = n / 2 * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) :
  (∀ n : ℕ, a_n n = 2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_general_term_arithmetic_sequence_l1406_140626


namespace NUMINAMATH_GPT_ratio_unit_price_l1406_140646

theorem ratio_unit_price (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let vX := 1.25 * v
  let pX := 0.85 * p
  (pX / vX) / (p / v) = 17 / 25 := by
{
  sorry
}

end NUMINAMATH_GPT_ratio_unit_price_l1406_140646


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l1406_140699

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_prod : (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120) : a + b + c + d + e = 39 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l1406_140699


namespace NUMINAMATH_GPT_original_price_of_coffee_l1406_140664

/-- 
  Define the prices of the cups of coffee as per the conditions.
  Let x be the original price of one cup of coffee.
  Assert the conditions and find the original price.
-/
theorem original_price_of_coffee (x : ℝ) 
  (h1 : x + x / 2 + 3 = 57) 
  (h2 : (x + x / 2 + 3)/3 = 19) : 
  x = 36 := 
by
  sorry

end NUMINAMATH_GPT_original_price_of_coffee_l1406_140664


namespace NUMINAMATH_GPT_number_of_cars_parked_l1406_140601

-- Definitions for the given conditions
def total_area (length width : ℕ) : ℕ := length * width
def usable_area (total : ℕ) : ℕ := (8 * total) / 10
def cars_parked (usable : ℕ) (area_per_car : ℕ) : ℕ := usable / area_per_car

-- Given conditions
def length : ℕ := 400
def width : ℕ := 500
def area_per_car : ℕ := 10
def expected_cars : ℕ := 16000 -- correct answer from solution

-- Define a proof statement
theorem number_of_cars_parked : cars_parked (usable_area (total_area length width)) area_per_car = expected_cars := by
  sorry

end NUMINAMATH_GPT_number_of_cars_parked_l1406_140601


namespace NUMINAMATH_GPT_students_like_apple_and_chocolate_not_carrot_l1406_140675

-- Definitions based on the conditions
def total_students : ℕ := 50
def apple_likers : ℕ := 23
def chocolate_likers : ℕ := 20
def carrot_likers : ℕ := 10
def non_likers : ℕ := 15

-- The main statement we need to prove: 
-- the number of students who liked both apple pie and chocolate cake but not carrot cake
theorem students_like_apple_and_chocolate_not_carrot : 
  ∃ (a b c d : ℕ), a + b + d = apple_likers ∧
                    a + c + d = chocolate_likers ∧
                    b + c + d = carrot_likers ∧
                    a + b + c + (50 - (35) - 15) = 35 ∧ 
                    a = 7 :=
by 
  sorry

end NUMINAMATH_GPT_students_like_apple_and_chocolate_not_carrot_l1406_140675


namespace NUMINAMATH_GPT_find_m_l1406_140698

-- Define the functions f and g
def f (x m : ℝ) := x^2 - 2 * x + m
def g (x m : ℝ) := x^2 - 3 * x + 5 * m

-- The condition to be proved
theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1406_140698


namespace NUMINAMATH_GPT_distance_between_joe_and_gracie_l1406_140628

open Complex

noncomputable def joe_point : ℂ := 2 + 3 * I
noncomputable def gracie_point : ℂ := -2 + 2 * I
noncomputable def distance := abs (joe_point - gracie_point)

theorem distance_between_joe_and_gracie :
  distance = Real.sqrt 17 := by
  sorry

end NUMINAMATH_GPT_distance_between_joe_and_gracie_l1406_140628


namespace NUMINAMATH_GPT_axis_of_symmetry_l1406_140692

-- Define points and the parabola equation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 2 5
def B := Point.mk 4 5

def parabola (b c : ℝ) (p : Point) : Prop :=
  p.y = 2 * p.x^2 + b * p.x + c

theorem axis_of_symmetry (b c : ℝ) (hA : parabola b c A) (hB : parabola b c B) : ∃ x_axis : ℝ, x_axis = 3 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1406_140692


namespace NUMINAMATH_GPT_multiplication_problem_l1406_140683

noncomputable def problem_statement (x : ℂ) : Prop :=
  (x^4 + 30 * x^2 + 225) * (x^2 - 15) = x^6 - 3375

theorem multiplication_problem (x : ℂ) : 
  problem_statement x :=
sorry

end NUMINAMATH_GPT_multiplication_problem_l1406_140683


namespace NUMINAMATH_GPT_x_lt_y_l1406_140669

variable {a b c d x y : ℝ}

theorem x_lt_y 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (cd)^(y/2)) : 
  x < y :=
by 
  sorry

end NUMINAMATH_GPT_x_lt_y_l1406_140669


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_sum_l1406_140671

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_of_sum_l1406_140671


namespace NUMINAMATH_GPT_root_of_equation_l1406_140620

theorem root_of_equation : 
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = (x - 1) / x) →
  f (4 * (1 / 2)) = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_root_of_equation_l1406_140620


namespace NUMINAMATH_GPT_marbles_ratio_l1406_140632

theorem marbles_ratio (miriam_current_marbles miriam_initial_marbles marbles_brother marbles_sister marbles_total_given marbles_savanna : ℕ)
  (h1 : miriam_current_marbles = 30)
  (h2 : marbles_brother = 60)
  (h3 : marbles_sister = 2 * marbles_brother)
  (h4 : miriam_initial_marbles = 300)
  (h5 : marbles_total_given = miriam_initial_marbles - miriam_current_marbles)
  (h6 : marbles_savanna = marbles_total_given - (marbles_brother + marbles_sister)) :
  (marbles_savanna : ℚ) / miriam_current_marbles = 3 :=
by
  sorry

end NUMINAMATH_GPT_marbles_ratio_l1406_140632
