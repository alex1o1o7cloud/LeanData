import Mathlib

namespace NUMINAMATH_GPT_grounded_days_for_lying_l1007_100733

def extra_days_per_grade_below_b : ℕ := 3
def grades_below_b : ℕ := 4
def total_days_grounded : ℕ := 26

theorem grounded_days_for_lying : 
  (total_days_grounded - (grades_below_b * extra_days_per_grade_below_b) = 14) := 
by 
  sorry

end NUMINAMATH_GPT_grounded_days_for_lying_l1007_100733


namespace NUMINAMATH_GPT_sandy_took_310_dollars_l1007_100754

theorem sandy_took_310_dollars (X : ℝ) (h70percent : 0.70 * X = 217) : X = 310 := by
  sorry

end NUMINAMATH_GPT_sandy_took_310_dollars_l1007_100754


namespace NUMINAMATH_GPT_goat_can_circle_around_tree_l1007_100787

/-- 
  Given a goat tied with a rope of length 4.7 meters (L) near an old tree with a cylindrical trunk of radius 0.5 meters (R), 
  with the shortest distance from the stake to the surface of the tree being 1 meter (d), 
  prove that the minimal required rope length to encircle the tree and return to the stake is less than 
  or equal to the given rope length of 4.7 meters (L).
-/ 
theorem goat_can_circle_around_tree (L R d : ℝ) (hR : R = 0.5) (hd : d = 1) (hL : L = 4.7) : 
  ∃ L_min, L_min ≤ L := 
by
  -- Detailed proof steps omitted.
  sorry

end NUMINAMATH_GPT_goat_can_circle_around_tree_l1007_100787


namespace NUMINAMATH_GPT_part_I_part_II_l1007_100772

-- Definition of functions
def f (x a : ℝ) := |3 * x - a|
def g (x : ℝ) := |x + 1|

-- Part (I): Solution set for f(x) < 3 when a = 4
theorem part_I (x : ℝ) : f x 4 < 3 ↔ (1 / 3 < x ∧ x < 7 / 3) :=
by 
  sorry

-- Part (II): Range of a such that f(x) + g(x) > 1 for all x in ℝ
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a + g x > 1) ↔ (a < -6 ∨ a > 0) :=
by 
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1007_100772


namespace NUMINAMATH_GPT_expand_product_l1007_100794

theorem expand_product (x : ℤ) : 
  (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1007_100794


namespace NUMINAMATH_GPT_hexahedron_octahedron_ratio_l1007_100712

open Real

theorem hexahedron_octahedron_ratio (a : ℝ) (h_a_pos : 0 < a) :
  let r1 := (sqrt 6 * a / 9)
  let r2 := (sqrt 6 * a / 6)
  let ratio := r1 / r2
  ∃ m n : ℕ, gcd m n = 1 ∧ (ratio = (m : ℝ) / (n : ℝ)) ∧ (m * n = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_hexahedron_octahedron_ratio_l1007_100712


namespace NUMINAMATH_GPT_problem_1_problem_2_l1007_100753

theorem problem_1 (x y : ℝ) (h1 : x - y = 3) (h2 : 3*x - 8*y = 14) : x = 2 ∧ y = -1 :=
sorry

theorem problem_2 (x y : ℝ) (h1 : 3*x + 4*y = 16) (h2 : 5*x - 6*y = 33) : x = 6 ∧ y = -1/2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1007_100753


namespace NUMINAMATH_GPT_task_completion_days_l1007_100749

theorem task_completion_days (a b c: ℕ) :
  (b = a + 6) → (c = b + 3) → 
  (3 / a + 4 / b = 9 / c) →
  a = 18 ∧ b = 24 ∧ c = 27 :=
  by
  sorry

end NUMINAMATH_GPT_task_completion_days_l1007_100749


namespace NUMINAMATH_GPT_find_angle_l1007_100773

variable (x : ℝ)

theorem find_angle (h1 : x + (180 - x) = 180) (h2 : x + (90 - x) = 90) (h3 : 180 - x = 3 * (90 - x)) : x = 45 := 
by
  sorry

end NUMINAMATH_GPT_find_angle_l1007_100773


namespace NUMINAMATH_GPT_definite_integral_ln_l1007_100707

open Real

theorem definite_integral_ln (a b : ℝ) (h₁ : a = 1) (h₂ : b = exp 1) :
  ∫ x in a..b, (1 + log x) = exp 1 := by
  sorry

end NUMINAMATH_GPT_definite_integral_ln_l1007_100707


namespace NUMINAMATH_GPT_ratio_of_A_to_B_l1007_100769

theorem ratio_of_A_to_B (total_weight compound_A_weight compound_B_weight : ℝ)
  (h1 : total_weight = 108)
  (h2 : compound_B_weight = 90)
  (h3 : compound_A_weight = total_weight - compound_B_weight) :
  compound_A_weight / compound_B_weight = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_A_to_B_l1007_100769


namespace NUMINAMATH_GPT_solve_for_x_l1007_100731

theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)⁻¹) : x = 8 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1007_100731


namespace NUMINAMATH_GPT_investment_schemes_correct_l1007_100776

-- Define the parameters of the problem
def num_projects : Nat := 3
def num_districts : Nat := 4

-- Function to count the number of valid investment schemes
def count_investment_schemes (num_projects num_districts : Nat) : Nat :=
  let total_schemes := num_districts ^ num_projects
  let invalid_schemes := num_districts
  total_schemes - invalid_schemes

-- Theorem statement
theorem investment_schemes_correct :
  count_investment_schemes num_projects num_districts = 60 := by
  sorry

end NUMINAMATH_GPT_investment_schemes_correct_l1007_100776


namespace NUMINAMATH_GPT_worth_of_entire_lot_l1007_100791

theorem worth_of_entire_lot (half_share : ℝ) (amount_per_tenth : ℝ) (total_amount : ℝ) :
  half_share = 0.5 →
  amount_per_tenth = 460 →
  total_amount = (amount_per_tenth * 10) →
  (total_amount * 2) = 9200 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_worth_of_entire_lot_l1007_100791


namespace NUMINAMATH_GPT_intersection_M_N_l1007_100720

open Real

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 2 - abs x > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by
sorry

end NUMINAMATH_GPT_intersection_M_N_l1007_100720


namespace NUMINAMATH_GPT_garrison_provisions_last_initially_l1007_100716

noncomputable def garrison_initial_provisions (x : ℕ) : Prop :=
  ∃ x : ℕ, 2000 * (x - 21) = 3300 * 20 ∧ x = 54

theorem garrison_provisions_last_initially :
  garrison_initial_provisions 54 :=
by
  sorry

end NUMINAMATH_GPT_garrison_provisions_last_initially_l1007_100716


namespace NUMINAMATH_GPT_inequality_problem_l1007_100785

theorem inequality_problem (x : ℝ) (hx : 0 < x) : 
  1 + x ^ 2018 ≥ (2 * x) ^ 2017 / (1 + x) ^ 2016 := 
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1007_100785


namespace NUMINAMATH_GPT_minimum_cost_l1007_100748

theorem minimum_cost (
    x y m w : ℝ) 
    (h1 : 4 * x + 2 * y = 400)
    (h2 : 2 * x + 4 * y = 320)
    (h3 : m ≥ 16)
    (h4 : m + (80 - m) = 80)
    (h5 : w = 80 * m + 40 * (80 - m)) :
    x = 80 ∧ y = 40 ∧ w = 3840 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_cost_l1007_100748


namespace NUMINAMATH_GPT_sum_of_coefficients_eq_minus_36_l1007_100777

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem sum_of_coefficients_eq_minus_36 
  (a b c : ℝ)
  (h_min : ∀ x, quadratic a b c x ≥ -36)
  (h_points : quadratic a b c (-3) = 0 ∧ quadratic a b c 5 = 0)
  : a + b + c = -36 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_eq_minus_36_l1007_100777


namespace NUMINAMATH_GPT_shoveling_hours_l1007_100730

def initial_rate := 25

def rate_decrease := 2

def snow_volume := 6 * 12 * 3

def shoveling_rate (hour : ℕ) : ℕ :=
  if hour = 0 then initial_rate
  else initial_rate - rate_decrease * hour

def cumulative_snow (hour : ℕ) : ℕ :=
  if hour = 0 then snow_volume - shoveling_rate 0
  else cumulative_snow (hour - 1) - shoveling_rate hour

theorem shoveling_hours : cumulative_snow 12 ≠ 0 ∧ cumulative_snow 13 = 47 := by
  sorry

end NUMINAMATH_GPT_shoveling_hours_l1007_100730


namespace NUMINAMATH_GPT_problem1_problem2_l1007_100721

-- Define A and B as given
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y - 5 * x - 1
def B (x y : ℝ) : ℝ := -x^2 + x * y - 1

-- Problem statement 1: Prove 3A + 6B simplifies as expected
theorem problem1 (x y : ℝ) : 3 * A x y + 6 * B x y = -3 * x * y - 15 * x - 9 :=
  by
    sorry

-- Problem statement 2: Prove that if 3A + 6B is independent of x, then y = -5
theorem problem2 (y : ℝ) (h : ∀ x : ℝ, 3 * A x y + 6 * B x y = -9) : y = -5 :=
  by
    sorry

end NUMINAMATH_GPT_problem1_problem2_l1007_100721


namespace NUMINAMATH_GPT_div_expression_l1007_100761

variable {α : Type*} [Field α]

theorem div_expression (a b c : α) : 4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end NUMINAMATH_GPT_div_expression_l1007_100761


namespace NUMINAMATH_GPT_bug_converges_to_final_position_l1007_100758

noncomputable def bug_final_position : ℝ × ℝ := 
  let horizontal_sum := ∑' n, if n % 4 = 0 then (1 / 4) ^ (n / 4) else 0
  let vertical_sum := ∑' n, if n % 4 = 1 then (1 / 4) ^ (n / 4) else 0
  (horizontal_sum, vertical_sum)

theorem bug_converges_to_final_position : bug_final_position = (4 / 5, 2 / 5) := 
  sorry

end NUMINAMATH_GPT_bug_converges_to_final_position_l1007_100758


namespace NUMINAMATH_GPT_solve_equation_in_integers_l1007_100774
-- Import the necessary library for Lean

-- Define the main theorem to solve the equation in integers
theorem solve_equation_in_integers :
  ∃ (xs : List (ℕ × ℕ)), (∀ x y, (3^x - 2^y = 1 → (x, y) ∈ xs)) ∧ xs = [(1, 1), (2, 3)] :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_in_integers_l1007_100774


namespace NUMINAMATH_GPT_solve_equation_l1007_100792

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l1007_100792


namespace NUMINAMATH_GPT_sum_coordinates_point_C_l1007_100740

/-
Let point A = (0,0), point B is on the line y = 6, and the slope of AB is 3/4.
Point C lies on the y-axis with a slope of 1/2 from B to C.
We need to prove that the sum of the coordinates of point C is 2.
-/
theorem sum_coordinates_point_C : 
  ∃ (A B C : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B.2 = 6 ∧ 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 ∧ 
  C.1 = 0 ∧ 
  (C.2 - B.2) / (C.1 - B.1) = 1 / 2 ∧ 
  C.1 + C.2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_coordinates_point_C_l1007_100740


namespace NUMINAMATH_GPT_find_eighth_term_l1007_100734

noncomputable def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + n * d

theorem find_eighth_term (a d : ℕ) :
  (arithmetic_sequence a d 0) + 
  (arithmetic_sequence a d 1) + 
  (arithmetic_sequence a d 2) + 
  (arithmetic_sequence a d 3) + 
  (arithmetic_sequence a d 4) + 
  (arithmetic_sequence a d 5) = 21 ∧
  arithmetic_sequence a d 6 = 7 →
  arithmetic_sequence a d 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_eighth_term_l1007_100734


namespace NUMINAMATH_GPT_inequality_proof_l1007_100726

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1007_100726


namespace NUMINAMATH_GPT_trig_identity_30deg_l1007_100729

theorem trig_identity_30deg :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  let c30 := Real.cos (Real.pi / 6)
  t30 = (Real.sqrt 3) / 3 ∧ s30 = 1 / 2 ∧ c30 = (Real.sqrt 3) / 2 →
  t30 + 4 * s30 + 2 * c30 = (2 * (Real.sqrt 3) + 3) / 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_trig_identity_30deg_l1007_100729


namespace NUMINAMATH_GPT_range_of_k_intersecting_AB_l1007_100757

theorem range_of_k_intersecting_AB 
  (A B : ℝ × ℝ) 
  (hA : A = (2, 7)) 
  (hB : B = (9, 6)) 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (H : ∃ x : ℝ, A.2 = k * A.1 ∧ B.2 = k * B.1):
  (2 / 3) ≤ k ∧ k ≤ 7 / 2 :=
by sorry

end NUMINAMATH_GPT_range_of_k_intersecting_AB_l1007_100757


namespace NUMINAMATH_GPT_apples_not_ripe_l1007_100741

theorem apples_not_ripe (total_apples good_apples : ℕ) (h1 : total_apples = 14) (h2 : good_apples = 8) : total_apples - good_apples = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_apples_not_ripe_l1007_100741


namespace NUMINAMATH_GPT_employee_payments_l1007_100783

theorem employee_payments :
  ∃ (A B C : ℤ), A = 900 ∧ B = 600 ∧ C = 500 ∧
    A + B + C = 2000 ∧
    A = 3 * B / 2 ∧
    C = 400 + 100 := 
by
  sorry

end NUMINAMATH_GPT_employee_payments_l1007_100783


namespace NUMINAMATH_GPT_larger_cross_section_distance_l1007_100765

theorem larger_cross_section_distance
  (h_area1 : ℝ)
  (h_area2 : ℝ)
  (dist_planes : ℝ)
  (h_area1_val : h_area1 = 256 * Real.sqrt 2)
  (h_area2_val : h_area2 = 576 * Real.sqrt 2)
  (dist_planes_val : dist_planes = 10) :
  ∃ h : ℝ, h = 30 :=
by
  sorry

end NUMINAMATH_GPT_larger_cross_section_distance_l1007_100765


namespace NUMINAMATH_GPT_identical_prob_of_painted_cubes_l1007_100755

/-
  Given:
  - Each face of a cube can be painted in one of 3 colors.
  - Each cube has 6 faces.
  - The total possible ways to paint both cubes is 531441.
  - The total ways to paint them such that they are identical after rotation is 66.

  Prove:
  - The probability of two painted cubes being identical after rotation is 2/16101.
-/
theorem identical_prob_of_painted_cubes :
  let total_ways := 531441
  let identical_ways := 66
  (identical_ways : ℚ) / total_ways = 2 / 16101 := by
  sorry

end NUMINAMATH_GPT_identical_prob_of_painted_cubes_l1007_100755


namespace NUMINAMATH_GPT_find_X_l1007_100775

def star (a b : ℤ) : ℤ := 5 * a - 3 * b

theorem find_X (X : ℤ) (h1 : star X (star 3 2) = 18) : X = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_X_l1007_100775


namespace NUMINAMATH_GPT_factorization_ce_sum_eq_25_l1007_100746

theorem factorization_ce_sum_eq_25 {C E : ℤ} (h : (C * x - 13) * (E * x - 7) = 20 * x^2 - 87 * x + 91) : 
  C * E + C = 25 :=
sorry

end NUMINAMATH_GPT_factorization_ce_sum_eq_25_l1007_100746


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1007_100703

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b) ∨ (b = c) ∨ (a = c)

def is_valid_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : is_isosceles_triangle a b b) (h₂ : is_valid_triangle a b b) : a + b + b = 15 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1007_100703


namespace NUMINAMATH_GPT_find_original_polynomial_calculate_correct_result_l1007_100795

variable {P : Polynomial ℝ}
variable (Q : Polynomial ℝ := 2 * X ^ 2 + X - 5)
variable (R : Polynomial ℝ := X ^ 2 + 3 * X - 1)

theorem find_original_polynomial (h : P - Q = R) : P = 3 * X ^ 2 + 4 * X - 6 :=
by
  sorry

theorem calculate_correct_result (h : P = 3 * X ^ 2 + 4 * X - 6) : P - Q = X ^ 2 + X + 9 :=
by
  sorry

end NUMINAMATH_GPT_find_original_polynomial_calculate_correct_result_l1007_100795


namespace NUMINAMATH_GPT_arith_seq_sum_geom_mean_proof_l1007_100789

theorem arith_seq_sum_geom_mean_proof (a_1 : ℝ) (a_n : ℕ → ℝ)
(common_difference : ℝ) (s_n : ℕ → ℝ)
(h_sequence : ∀ n, a_n n = a_1 + (n - 1) * common_difference)
(h_sum : ∀ n, s_n n = n / 2 * (2 * a_1 + (n - 1) * common_difference))
(h_geom_mean : (s_n 2) ^ 2 = s_n 1 * s_n 4)
(h_common_diff : common_difference = -1) :
a_1 = -1 / 2 :=
sorry

end NUMINAMATH_GPT_arith_seq_sum_geom_mean_proof_l1007_100789


namespace NUMINAMATH_GPT_teacher_A_realizes_fish_l1007_100737

variable (Teacher : Type) (has_fish : Teacher → Prop) (is_laughing : Teacher → Prop)
variables (A B C : Teacher)

-- Initial assumptions
axiom all_laughing : is_laughing A ∧ is_laughing B ∧ is_laughing C
axiom each_thinks_others_have_fish : (¬has_fish A ∧ has_fish B ∧ has_fish C) 
                                      ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
                                      ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C)

-- The logical conclusion
theorem teacher_A_realizes_fish : (∃ A B C : Teacher, 
  is_laughing A ∧ is_laughing B ∧ is_laughing C ∧
  ((¬has_fish A ∧ has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C))) →
  (has_fish A ∧ is_laughing B ∧ is_laughing C) :=
sorry -- proof not required.

end NUMINAMATH_GPT_teacher_A_realizes_fish_l1007_100737


namespace NUMINAMATH_GPT_casey_correct_result_l1007_100709

variable (x : ℕ)

def incorrect_divide (x : ℕ) := x / 7
def incorrect_subtract (x : ℕ) := x - 20
def incorrect_result := 19

def reverse_subtract (x : ℕ) := x + 20
def reverse_divide (x : ℕ) := x * 7

def correct_multiply (x : ℕ) := x * 7
def correct_add (x : ℕ) := x + 20

theorem casey_correct_result (x : ℕ) (h : reverse_divide (reverse_subtract incorrect_result) = x) : correct_add (correct_multiply x) = 1931 :=
by
  sorry

end NUMINAMATH_GPT_casey_correct_result_l1007_100709


namespace NUMINAMATH_GPT_floor_sqrt_23_squared_l1007_100732

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_floor_sqrt_23_squared_l1007_100732


namespace NUMINAMATH_GPT_min_distance_l1007_100724

theorem min_distance (W : ℝ) (b : ℝ) (n : ℕ) (H_W : W = 42) (H_b : b = 3) (H_n : n = 8) : 
  ∃ d : ℝ, d = 2 ∧ (W - n * b = 9 * d) := 
by 
  -- Here should go the proof
  sorry

end NUMINAMATH_GPT_min_distance_l1007_100724


namespace NUMINAMATH_GPT_clock_angle_3_to_7_l1007_100762

theorem clock_angle_3_to_7 : 
  let number_of_rays := 12
  let total_degrees := 360
  let degree_per_ray := total_degrees / number_of_rays
  let angle_3_to_7 := 4 * degree_per_ray
  angle_3_to_7 = 120 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_3_to_7_l1007_100762


namespace NUMINAMATH_GPT_crackers_per_person_l1007_100719

variable (darrenA : Nat)
variable (darrenB : Nat)
variable (aCrackersPerBox : Nat)
variable (bCrackersPerBox : Nat)
variable (calvinA : Nat)
variable (calvinB : Nat)
variable (totalPeople : Nat)

-- Definitions based on the conditions
def totalDarrenCrackers := darrenA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCalvinA := 2 * darrenA - 1
def totalCalvinCrackers := totalCalvinA * aCrackersPerBox + darrenB * bCrackersPerBox
def totalCrackers := totalDarrenCrackers + totalCalvinCrackers
def crackersPerPerson := totalCrackers / totalPeople

-- The theorem to prove the question equals the answer given the conditions
theorem crackers_per_person :
  darrenA = 4 →
  darrenB = 2 →
  aCrackersPerBox = 24 →
  bCrackersPerBox = 30 →
  calvinA = 7 →
  calvinB = darrenB →
  totalPeople = 5 →
  crackersPerPerson = 76 :=
by
  intros
  sorry

end NUMINAMATH_GPT_crackers_per_person_l1007_100719


namespace NUMINAMATH_GPT_bobby_pancakes_left_l1007_100714

def total_pancakes : ℕ := 21
def pancakes_eaten_by_bobby : ℕ := 5
def pancakes_eaten_by_dog : ℕ := 7

theorem bobby_pancakes_left : total_pancakes - (pancakes_eaten_by_bobby + pancakes_eaten_by_dog) = 9 :=
  by
  sorry

end NUMINAMATH_GPT_bobby_pancakes_left_l1007_100714


namespace NUMINAMATH_GPT_lawnmower_percentage_drop_l1007_100743

theorem lawnmower_percentage_drop :
  ∀ (initial_value value_after_one_year value_after_six_months : ℝ)
    (percentage_drop_in_year : ℝ),
  initial_value = 100 →
  value_after_one_year = 60 →
  percentage_drop_in_year = 20 →
  value_after_one_year = (1 - percentage_drop_in_year / 100) * value_after_six_months →
  (initial_value - value_after_six_months) / initial_value * 100 = 25 :=
by
  intros initial_value value_after_one_year value_after_six_months percentage_drop_in_year
  intros h_initial h_value_after_one_year h_percentage_drop_in_year h_value_equation
  sorry

end NUMINAMATH_GPT_lawnmower_percentage_drop_l1007_100743


namespace NUMINAMATH_GPT_find_ratio_b_over_a_l1007_100718

theorem find_ratio_b_over_a (a b : ℝ)
  (h1 : ∀ x, deriv (fun x => a * x^2 + b) x = 2 * a * x)
  (h2 : deriv (fun x => a * x^2 + b) 1 = 2)
  (h3 : a * 1^2 + b = 3) : b / a = 2 := 
sorry

end NUMINAMATH_GPT_find_ratio_b_over_a_l1007_100718


namespace NUMINAMATH_GPT_number_of_kids_stay_home_l1007_100735

def total_kids : ℕ := 313473
def kids_at_camp : ℕ := 38608
def kids_stay_home : ℕ := 274865

theorem number_of_kids_stay_home :
  total_kids - kids_at_camp = kids_stay_home := 
by
  -- Subtracting the number of kids who go to camp from the total number of kids
  sorry

end NUMINAMATH_GPT_number_of_kids_stay_home_l1007_100735


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l1007_100742

theorem batsman_average_after_17th_inning (A : ℝ) :
  let total_runs_after_17_innings := 16 * A + 87
  let new_average := total_runs_after_17_innings / 17
  new_average = A + 3 → 
  (A + 3) = 39 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l1007_100742


namespace NUMINAMATH_GPT_solve_equation_l1007_100796

variable (x : ℝ)

def equation := (x / (2 * x - 3)) + (5 / (3 - 2 * x)) = 4
def condition := x ≠ 3 / 2

theorem solve_equation : equation x ∧ condition x → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1007_100796


namespace NUMINAMATH_GPT_even_square_even_square_even_even_l1007_100779

-- Definition for a natural number being even
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Statement 1: If p is even, then p^2 is even
theorem even_square_even (p : ℕ) (hp : is_even p) : is_even (p * p) :=
sorry

-- Statement 2: If p^2 is even, then p is even
theorem square_even_even (p : ℕ) (hp_squared : is_even (p * p)) : is_even p :=
sorry

end NUMINAMATH_GPT_even_square_even_square_even_even_l1007_100779


namespace NUMINAMATH_GPT_verification_equation_3_conjecture_general_equation_l1007_100738

theorem verification_equation_3 : 
  4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15)) :=
sorry

theorem conjecture :
  Real.sqrt (5 * (5 / 24)) = 5 * Real.sqrt (5 / 24) :=
sorry

theorem general_equation (n : ℕ) (h : 2 ≤ n) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
sorry

end NUMINAMATH_GPT_verification_equation_3_conjecture_general_equation_l1007_100738


namespace NUMINAMATH_GPT_cost_of_50_tulips_l1007_100725

theorem cost_of_50_tulips (c : ℕ → ℝ) :
  (∀ n : ℕ, n ≤ 40 → c n = n * (36 / 18)) ∧
  (∀ n : ℕ, n > 40 → c n = (40 * (36 / 18) + (n - 40) * (36 / 18)) * 0.9) ∧
  (c 18 = 36) →
  c 50 = 90 := sorry

end NUMINAMATH_GPT_cost_of_50_tulips_l1007_100725


namespace NUMINAMATH_GPT_first_day_more_than_300_l1007_100723

def paperclips (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_more_than_300 : ∃ n, paperclips n > 300 ∧ n = 4 := by
  sorry

end NUMINAMATH_GPT_first_day_more_than_300_l1007_100723


namespace NUMINAMATH_GPT_certain_number_existence_l1007_100717

theorem certain_number_existence : ∃ x : ℝ, (102 * 102) + (x * x) = 19808 ∧ x = 97 := by
  sorry

end NUMINAMATH_GPT_certain_number_existence_l1007_100717


namespace NUMINAMATH_GPT_blake_spent_60_on_mangoes_l1007_100788

def spent_on_oranges : ℕ := 40
def spent_on_apples : ℕ := 50
def initial_amount : ℕ := 300
def change : ℕ := 150
def total_spent := initial_amount - change
def total_spent_on_fruits := spent_on_oranges + spent_on_apples
def spending_on_mangoes := total_spent - total_spent_on_fruits

theorem blake_spent_60_on_mangoes : spending_on_mangoes = 60 := 
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_blake_spent_60_on_mangoes_l1007_100788


namespace NUMINAMATH_GPT_fraction_eval_l1007_100768

theorem fraction_eval : 
    (1 / (3 - (1 / (3 - (1 / (3 - (1 / 4))))))) = (11 / 29) := 
by
  sorry

end NUMINAMATH_GPT_fraction_eval_l1007_100768


namespace NUMINAMATH_GPT_complex_product_l1007_100702

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_product_l1007_100702


namespace NUMINAMATH_GPT_distance_focus_to_asymptote_l1007_100706

theorem distance_focus_to_asymptote (m : ℝ) (x y : ℝ) (h1 : (x^2) / 9 - (y^2) / m = 1) 
  (h2 : (Real.sqrt 14) / 3 = (Real.sqrt (9 + m)) / 3) : 
  ∃ d : ℝ, d = Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_distance_focus_to_asymptote_l1007_100706


namespace NUMINAMATH_GPT_savings_after_one_year_l1007_100793

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem savings_after_one_year :
  compound_interest 1000 0.10 2 1 = 1102.50 :=
by
  sorry

end NUMINAMATH_GPT_savings_after_one_year_l1007_100793


namespace NUMINAMATH_GPT_remainder_sum_l1007_100766

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7) = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_l1007_100766


namespace NUMINAMATH_GPT_percentage_increase_book_price_l1007_100704

theorem percentage_increase_book_price (OldP NewP : ℕ) (hOldP : OldP = 300) (hNewP : NewP = 330) :
  ((NewP - OldP : ℕ) / OldP : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_book_price_l1007_100704


namespace NUMINAMATH_GPT_cost_effective_combination_l1007_100756

/--
Jackson wants to impress his girlfriend by filling her hot tub with champagne.
The hot tub holds 400 liters of liquid. He has three types of champagne bottles:
1. Small bottle: Holds 0.75 liters with a price of $70 per bottle.
2. Medium bottle: Holds 1.5 liters with a price of $120 per bottle.
3. Large bottle: Holds 3 liters with a price of $220 per bottle.

If he purchases more than 50 bottles of any type, he will get a 10% discount on 
that type. If he purchases over 100 bottles of any type, he will get 20% off 
on that type of bottles. 

Prove that the most cost-effective combination of bottles for 
Jackson to purchase is 134 large bottles for a total cost of $23,584 after the discount.
-/
theorem cost_effective_combination :
  let volume := 400
  let small_bottle_volume := 0.75
  let small_bottle_cost := 70
  let medium_bottle_volume := 1.5
  let medium_bottle_cost := 120
  let large_bottle_volume := 3
  let large_bottle_cost := 220
  let discount_50 := 0.10
  let discount_100 := 0.20
  let cost_134_large_bottles := (134 * large_bottle_cost) * (1 - discount_100)
  cost_134_large_bottles = 23584 :=
sorry

end NUMINAMATH_GPT_cost_effective_combination_l1007_100756


namespace NUMINAMATH_GPT_inequality_holds_for_unit_interval_l1007_100708

theorem inequality_holds_for_unit_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
    5 * (x ^ 2 + y ^ 2) ^ 2 ≤ 4 + (x + y) ^ 4 :=
by
    sorry

end NUMINAMATH_GPT_inequality_holds_for_unit_interval_l1007_100708


namespace NUMINAMATH_GPT_stratified_sampling_females_l1007_100780

theorem stratified_sampling_females :
  let total_employees := 200
  let male_employees := 120
  let female_employees := 80
  let sample_size := 20
  number_of_female_in_sample = (female_employees / total_employees) * sample_size := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_females_l1007_100780


namespace NUMINAMATH_GPT_perimeter_of_one_rectangle_l1007_100727

-- Define the conditions
def is_divided_into_congruent_rectangles (s : ℕ) : Prop :=
  ∃ (height width : ℕ), height = s ∧ width = s / 4

-- Main proof statement
theorem perimeter_of_one_rectangle {s : ℕ} (h₁ : 4 * s = 144)
  (h₂ : is_divided_into_congruent_rectangles s) : 
  ∃ (perimeter : ℕ), perimeter = 90 :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_one_rectangle_l1007_100727


namespace NUMINAMATH_GPT_cost_price_of_article_l1007_100798

theorem cost_price_of_article (SP : ℝ) (profit_percentage : ℝ) (profit_fraction : ℝ) (CP : ℝ) : 
  SP = 120 → profit_percentage = 25 → profit_fraction = profit_percentage / 100 → 
  SP = CP + profit_fraction * CP → CP = 96 :=
by intros hSP hprofit_percentage hprofit_fraction heq
   sorry

end NUMINAMATH_GPT_cost_price_of_article_l1007_100798


namespace NUMINAMATH_GPT_maria_needs_green_beans_l1007_100763

theorem maria_needs_green_beans :
  ∀ (potatoes carrots onions green_beans : ℕ), 
  (carrots = 6 * potatoes) →
  (onions = 2 * carrots) →
  (green_beans = onions / 3) →
  (potatoes = 2) →
  green_beans = 8 :=
by
  intros potatoes carrots onions green_beans h1 h2 h3 h4
  rw [h4, Nat.mul_comm 6 2] at h1
  rw [h1, Nat.mul_comm 2 12] at h2
  rw [h2] at h3
  sorry

end NUMINAMATH_GPT_maria_needs_green_beans_l1007_100763


namespace NUMINAMATH_GPT_is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l1007_100744

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Given that a, b, c are the sides of the triangle
axiom lengths_of_triangle : a > 0 ∧ b > 0 ∧ c > 0

-- Problem 1: Prove that triangle is isosceles if x=1 is a root
theorem is_isosceles_of_x_eq_one_root  : ((a - c) * (1:ℝ)^2 - 2 * b * (1:ℝ) + (a + c) = 0) → a = b ∧ a ≠ c := 
by
  intros h
  sorry

-- Problem 2: Prove that triangle is right-angled if the equation has two equal real roots
theorem is_right_angled_of_equal_roots : (b^2 = a^2 - c^2) → (a^2 = b^2 + c^2) := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_is_isosceles_of_x_eq_one_root_is_right_angled_of_equal_roots_l1007_100744


namespace NUMINAMATH_GPT_number_of_dogs_is_112_l1007_100751

-- Definitions based on the given conditions.
def ratio_dogs_to_cats_to_bunnies (D C B : ℕ) : Prop := 4 * C = 7 * D ∧ 9 * C = 7 * B
def total_dogs_and_bunnies (D B : ℕ) (total : ℕ) : Prop := D + B = total

-- The hypothesis and conclusion of the problem.
theorem number_of_dogs_is_112 (D C B : ℕ) (x : ℕ) (h1: ratio_dogs_to_cats_to_bunnies D C B) (h2: total_dogs_and_bunnies D B 364) : D = 112 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_dogs_is_112_l1007_100751


namespace NUMINAMATH_GPT_jan_clean_car_water_l1007_100750

def jan_water_problem
  (initial_water : ℕ)
  (car_water : ℕ)
  (plant_additional : ℕ)
  (plate_clothes_water : ℕ)
  (remaining_water : ℕ)
  (used_water : ℕ)
  (car_cleaning_water : ℕ) : Prop :=
  initial_water = 65 ∧
  plate_clothes_water = 24 ∧
  plant_additional = 11 ∧
  remaining_water = 2 * plate_clothes_water ∧
  used_water = initial_water - remaining_water ∧
  car_water = used_water + plant_additional ∧
  car_cleaning_water = car_water / 4

theorem jan_clean_car_water : jan_water_problem 65 17 11 24 48 17 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_jan_clean_car_water_l1007_100750


namespace NUMINAMATH_GPT_sum_of_squares_and_products_l1007_100705

theorem sum_of_squares_and_products
  (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_and_products_l1007_100705


namespace NUMINAMATH_GPT_product_sum_divisibility_l1007_100710

theorem product_sum_divisibility (m n : ℕ) (h : (m + n) ∣ (m * n)) (hm : 0 < m) (hn : 0 < n) : m + n ≤ n^2 :=
sorry

end NUMINAMATH_GPT_product_sum_divisibility_l1007_100710


namespace NUMINAMATH_GPT_red_ball_probability_l1007_100770

theorem red_ball_probability 
  (red_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : black_balls = 9)
  (h3 : total_balls = red_balls + black_balls) :
  (red_balls : ℚ) / total_balls = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_red_ball_probability_l1007_100770


namespace NUMINAMATH_GPT_inequality_range_l1007_100781

theorem inequality_range (k : ℝ) : (∀ x : ℝ, abs (x + 1) - abs (x - 2) > k) → k < -3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_range_l1007_100781


namespace NUMINAMATH_GPT_total_juice_drunk_l1007_100701

noncomputable def juiceConsumption (samDrink benDrink : ℕ) (samConsRatio benConsRatio : ℚ) : ℚ :=
  let samConsumed := samConsRatio * samDrink
  let samRemaining := samDrink - samConsumed
  let benConsumed := benConsRatio * benDrink
  let benRemaining := benDrink - benConsumed
  let benToSam := (1 / 2) * benRemaining + 1
  let samTotal := samConsumed + benToSam
  let benTotal := benConsumed - benToSam
  samTotal + benTotal

theorem total_juice_drunk : juiceConsumption 12 20 (2 / 3 : ℚ) (2 / 3 : ℚ) = 32 :=
sorry

end NUMINAMATH_GPT_total_juice_drunk_l1007_100701


namespace NUMINAMATH_GPT_find_k_l1007_100764

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem find_k (k : ℝ) :
  let sum_vector := (vector_a.1 + 2 * (vector_b k).1, vector_a.2 + 2 * (vector_b k).2)
  let diff_vector := (2 * vector_a.1 - (vector_b k).1, 2 * vector_a.2 - (vector_b k).2)
  sum_vector.1 * diff_vector.2 = sum_vector.2 * diff_vector.1
  → k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1007_100764


namespace NUMINAMATH_GPT_bond_face_value_l1007_100799

theorem bond_face_value
  (F : ℝ)
  (S : ℝ)
  (hS : S = 3846.153846153846)
  (hI1 : I = 0.05 * F)
  (hI2 : I = 0.065 * S) :
  F = 5000 :=
by
  sorry

end NUMINAMATH_GPT_bond_face_value_l1007_100799


namespace NUMINAMATH_GPT_number_of_students_taking_statistics_l1007_100778

theorem number_of_students_taking_statistics
  (total_students : ℕ)
  (history_students : ℕ)
  (history_or_statistics : ℕ)
  (history_only : ℕ)
  (history_and_statistics : ℕ := history_students - history_only)
  (statistics_only : ℕ := history_or_statistics - history_and_statistics - history_only)
  (statistics_students : ℕ := history_and_statistics + statistics_only) :
  total_students = 90 → history_students = 36 → history_or_statistics = 59 → history_only = 29 →
    statistics_students = 30 :=
by
  intros
  -- Proof goes here but is omitted.
  sorry

end NUMINAMATH_GPT_number_of_students_taking_statistics_l1007_100778


namespace NUMINAMATH_GPT_men_absent_l1007_100797

theorem men_absent (n : ℕ) (d1 d2 : ℕ) (x : ℕ) 
  (h1 : n = 22) 
  (h2 : d1 = 20) 
  (h3 : d2 = 22) 
  (hc : n * d1 = (n - x) * d2) : 
  x = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_men_absent_l1007_100797


namespace NUMINAMATH_GPT_negation_equiv_l1007_100700

theorem negation_equiv {x : ℝ} : 
  (¬ (x^2 < 1 → -1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_negation_equiv_l1007_100700


namespace NUMINAMATH_GPT_squirrel_can_catch_nut_l1007_100711

-- Define the initial distance between Gabriel and the squirrel.
def initial_distance : ℝ := 3.75

-- Define the speed of the nut.
def nut_speed : ℝ := 5.0

-- Define the jumping distance of the squirrel.
def squirrel_jump_distance : ℝ := 1.8

-- Define the acceleration due to gravity.
def gravity : ℝ := 10.0

-- Define the positions of the nut and the squirrel as functions of time.
def nut_position_x (t : ℝ) : ℝ := nut_speed * t
def squirrel_position_x : ℝ := initial_distance
def nut_position_y (t : ℝ) : ℝ := 0.5 * gravity * t^2

-- Define the squared distance between the nut and the squirrel.
def distance_squared (t : ℝ) : ℝ :=
  (nut_position_x t - squirrel_position_x)^2 + (nut_position_y t)^2

-- Prove that the minimum distance squared is less than or equal to the squirrel's jumping distance squared.
theorem squirrel_can_catch_nut : ∃ t : ℝ, distance_squared t ≤ squirrel_jump_distance^2 := by
  -- Sorry placeholder, as the proof is not required.
  sorry

end NUMINAMATH_GPT_squirrel_can_catch_nut_l1007_100711


namespace NUMINAMATH_GPT_prime_sum_is_prime_l1007_100782

def prime : ℕ → Prop := sorry 

theorem prime_sum_is_prime (A B : ℕ) (hA : prime A) (hB : prime B) (hAB : prime (A - B)) (hABB : prime (A - B - B)) : prime (A + B + (A - B) + (A - B - B)) :=
sorry

end NUMINAMATH_GPT_prime_sum_is_prime_l1007_100782


namespace NUMINAMATH_GPT_intersection_unique_point_l1007_100728

theorem intersection_unique_point
    (h1 : ∀ (x y : ℝ), 2 * x + 3 * y = 6)
    (h2 : ∀ (x y : ℝ), 4 * x - 3 * y = 6)
    (h3 : ∀ y : ℝ, 2 = 2)
    (h4 : ∀ x : ℝ, y = 2 / 3)
    : ∃! (x y : ℝ), (2 * x + 3 * y = 6) ∧ (4 * x - 3 * y = 6) ∧ (x = 2) ∧ (y = 2 / 3) := 
by
    sorry

end NUMINAMATH_GPT_intersection_unique_point_l1007_100728


namespace NUMINAMATH_GPT_vector_odot_not_symmetric_l1007_100790

-- Define the vector operation ⊛
def vector_odot (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

-- Statement: Prove that the operation is not symmetric
theorem vector_odot_not_symmetric (a b : ℝ × ℝ) : vector_odot a b ≠ vector_odot b a := by
  sorry

end NUMINAMATH_GPT_vector_odot_not_symmetric_l1007_100790


namespace NUMINAMATH_GPT_one_non_congruent_triangle_with_perimeter_10_l1007_100715

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end NUMINAMATH_GPT_one_non_congruent_triangle_with_perimeter_10_l1007_100715


namespace NUMINAMATH_GPT_ratio_of_c_to_d_l1007_100759

theorem ratio_of_c_to_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
    (h1 : 9 * x - 6 * y = c) (h2 : 15 * x - 10 * y = d) :
    c / d = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_c_to_d_l1007_100759


namespace NUMINAMATH_GPT_has_four_digits_l1007_100752

def least_number_divisible (n: ℕ) : Prop := 
  n = 9600 ∧ 
  (∃ k1 k2 k3 k4: ℕ, n = 15 * k1 ∧ n = 25 * k2 ∧ n = 40 * k3 ∧ n = 75 * k4)

theorem has_four_digits : ∀ n: ℕ, least_number_divisible n → (Nat.digits 10 n).length = 4 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_has_four_digits_l1007_100752


namespace NUMINAMATH_GPT_minimum_boxes_required_l1007_100722

theorem minimum_boxes_required 
  (total_brochures : ℕ)
  (small_box_capacity : ℕ) (small_boxes_available : ℕ)
  (medium_box_capacity : ℕ) (medium_boxes_available : ℕ)
  (large_box_capacity : ℕ) (large_boxes_available : ℕ)
  (complete_fill : ∀ (box_capacity brochures : ℕ), box_capacity ∣ brochures)
  (min_boxes_required : ℕ) :
  total_brochures = 10000 →
  small_box_capacity = 50 →
  small_boxes_available = 40 →
  medium_box_capacity = 200 →
  medium_boxes_available = 25 →
  large_box_capacity = 500 →
  large_boxes_available = 10 →
  min_boxes_required = 35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_minimum_boxes_required_l1007_100722


namespace NUMINAMATH_GPT_area_enclosed_by_equation_l1007_100760

theorem area_enclosed_by_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 10 * y = -20) → (∃ r : ℝ, r^2 = 9 ∧ ∃ c : ℝ × ℝ, (∃ a b, (x - a)^2 + (y - b)^2 = r^2)) :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_equation_l1007_100760


namespace NUMINAMATH_GPT_intersection_M_N_l1007_100713

-- Definitions of sets M and N
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

-- Lean statement to prove that the intersection of M and N is {1, 2}
theorem intersection_M_N : M ∩ N = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1007_100713


namespace NUMINAMATH_GPT_elastic_collision_inelastic_collision_l1007_100786

-- Given conditions for Case A and Case B
variables (L V : ℝ) (m : ℝ) -- L is length of the rods, V is the speed, m is mass of each sphere

-- Prove Case A: The dumbbells separate maintaining their initial velocities
theorem elastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly elastic collision, the dumbbells separate maintaining their initial velocities
  true := sorry

-- Prove Case B: The dumbbells start rotating around the collision point with angular velocity V / (2 * L)
theorem inelastic_collision (h1 : L > 0) (h2 : V > 0) (h3 : m > 0) :
  -- After a perfectly inelastic collision, the dumbbells start rotating around the collision point with angular velocity V / (2 * L)
  true := sorry

end NUMINAMATH_GPT_elastic_collision_inelastic_collision_l1007_100786


namespace NUMINAMATH_GPT_slope_undefined_iff_vertical_l1007_100767

theorem slope_undefined_iff_vertical (m : ℝ) :
  let M := (2 * m + 3, m)
  let N := (m - 2, 1)
  (2 * m + 3 - (m - 2) = 0 ∧ m - 1 ≠ 0) ↔ m = -5 :=
by
  sorry

end NUMINAMATH_GPT_slope_undefined_iff_vertical_l1007_100767


namespace NUMINAMATH_GPT_probability_age_between_30_and_40_l1007_100747

-- Assume total number of people in the group is 200
def total_people : ℕ := 200

-- Assume 80 people have an age of more than 40 years
def people_age_more_than_40 : ℕ := 80

-- Assume 70 people have an age between 30 and 40 years
def people_age_between_30_and_40 : ℕ := 70

-- Assume 30 people have an age between 20 and 30 years
def people_age_between_20_and_30 : ℕ := 30

-- Assume 20 people have an age of less than 20 years
def people_age_less_than_20 : ℕ := 20

-- The proof problem statement
theorem probability_age_between_30_and_40 :
  (people_age_between_30_and_40 : ℚ) / (total_people : ℚ) = 7 / 20 :=
by
  sorry

end NUMINAMATH_GPT_probability_age_between_30_and_40_l1007_100747


namespace NUMINAMATH_GPT_B_contribution_l1007_100736

-- Define the conditions
def capitalA : ℝ := 3500
def monthsA : ℕ := 12
def monthsB : ℕ := 7
def profit_ratio_A : ℕ := 2
def profit_ratio_B : ℕ := 3

-- Statement: B's contribution to the capital
theorem B_contribution :
  (capitalA * monthsA * profit_ratio_B) / (monthsB * profit_ratio_A) = 4500 := by
  sorry

end NUMINAMATH_GPT_B_contribution_l1007_100736


namespace NUMINAMATH_GPT_total_pencils_correct_l1007_100771

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l1007_100771


namespace NUMINAMATH_GPT_expenditure_representation_l1007_100745

theorem expenditure_representation (income expenditure : ℤ)
  (h_income : income = 60)
  (h_expenditure : expenditure = 40) :
  -expenditure = -40 :=
by {
  sorry
}

end NUMINAMATH_GPT_expenditure_representation_l1007_100745


namespace NUMINAMATH_GPT_polynomial_sum_l1007_100739

variable {R : Type*} [CommRing R] {x y : R}

/-- Given that the sum of a polynomial P and x^2 - y^2 is x^2 + y^2, we want to prove that P is 2y^2. -/
theorem polynomial_sum (P : R) (h : P + (x^2 - y^2) = x^2 + y^2) : P = 2 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l1007_100739


namespace NUMINAMATH_GPT_commodities_price_difference_l1007_100784

theorem commodities_price_difference : 
  ∀ (C1 C2 : ℕ), 
    C1 = 477 → 
    C1 + C2 = 827 → 
    C1 - C2 = 127 :=
by
  intros C1 C2 h1 h2
  sorry

end NUMINAMATH_GPT_commodities_price_difference_l1007_100784
