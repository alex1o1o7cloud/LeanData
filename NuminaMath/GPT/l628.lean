import Mathlib

namespace NUMINAMATH_GPT_smallest_a_l628_62883

theorem smallest_a (a : ℤ) : 
  (112 ∣ (a * 43 * 62 * 1311)) ∧ (33 ∣ (a * 43 * 62 * 1311)) ↔ a = 1848 := 
sorry

end NUMINAMATH_GPT_smallest_a_l628_62883


namespace NUMINAMATH_GPT_right_triangle_leg_square_l628_62848

theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = a + 2) : 
  b^2 = 4 * a + 4 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_leg_square_l628_62848


namespace NUMINAMATH_GPT_lemon_loaf_each_piece_weight_l628_62874

def pan_length := 20  -- cm
def pan_width := 18   -- cm
def pan_height := 5   -- cm
def total_pieces := 25
def density := 2      -- g/cm³

noncomputable def weight_of_each_piece : ℕ := by
  have volume := pan_length * pan_width * pan_height
  have volume_of_each_piece := volume / total_pieces
  have mass_of_each_piece := volume_of_each_piece * density
  exact mass_of_each_piece

theorem lemon_loaf_each_piece_weight :
  weight_of_each_piece = 144 :=
sorry

end NUMINAMATH_GPT_lemon_loaf_each_piece_weight_l628_62874


namespace NUMINAMATH_GPT_find_radioactive_balls_within_7_checks_l628_62827

theorem find_radioactive_balls_within_7_checks :
  ∃ (balls : Finset α), balls.card = 11 ∧ ∃ radioactive_balls ⊆ balls, radioactive_balls.card = 2 ∧
  (∀ (check : Finset α → Prop), (∀ S, check S ↔ (∃ b ∈ S, b ∈ radioactive_balls)) →
  ∃ checks : Finset (Finset α), checks.card ≤ 7 ∧ (∀ b ∈ radioactive_balls, ∃ S ∈ checks, b ∈ S)) :=
sorry

end NUMINAMATH_GPT_find_radioactive_balls_within_7_checks_l628_62827


namespace NUMINAMATH_GPT_total_spending_in_4_years_l628_62870

def trevor_spending_per_year : ℕ := 80
def reed_to_trevor_diff : ℕ := 20
def reed_to_quinn_factor : ℕ := 2

theorem total_spending_in_4_years :
  ∃ (reed_spending quinn_spending : ℕ),
  (reed_spending = trevor_spending_per_year - reed_to_trevor_diff) ∧
  (reed_spending = reed_to_quinn_factor * quinn_spending) ∧
  ((trevor_spending_per_year + reed_spending + quinn_spending) * 4 = 680) :=
sorry

end NUMINAMATH_GPT_total_spending_in_4_years_l628_62870


namespace NUMINAMATH_GPT_Eve_spend_l628_62886

noncomputable def hand_mitts := 14.00
noncomputable def apron := 16.00
noncomputable def utensils_set := 10.00
noncomputable def small_knife := 2 * utensils_set
noncomputable def total_cost_for_one_niece := hand_mitts + apron + utensils_set + small_knife
noncomputable def total_cost_for_three_nieces := 3 * total_cost_for_one_niece
noncomputable def discount := 0.25 * total_cost_for_three_nieces
noncomputable def final_cost := total_cost_for_three_nieces - discount

theorem Eve_spend : final_cost = 135.00 :=
by sorry

end NUMINAMATH_GPT_Eve_spend_l628_62886


namespace NUMINAMATH_GPT_solution_sets_and_range_l628_62845

theorem solution_sets_and_range 
    (x a : ℝ) 
    (A : Set ℝ)
    (M : Set ℝ) :
    (∀ x, x ∈ A ↔ 1 ≤ x ∧ x ≤ 4) ∧
    (M = {x | (x - a) * (x - 2) ≤ 0} ) ∧
    (M ⊆ A) → (1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_sets_and_range_l628_62845


namespace NUMINAMATH_GPT_probability_two_point_distribution_l628_62880

theorem probability_two_point_distribution 
  (P : ℕ → ℚ)
  (two_point_dist : P 0 + P 1 = 1)
  (condition : P 1 = (3 / 2) * P 0) :
  P 1 = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_point_distribution_l628_62880


namespace NUMINAMATH_GPT_cryptarithm_C_value_l628_62826

/--
Given digits A, B, and C where A, B, and C are distinct and non-repeating,
and the following conditions hold:
1. ABC - BC = A0A
Prove that C = 9.
-/
theorem cryptarithm_C_value (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_non_repeating: (0 <= A ∧ A <= 9) ∧ (0 <= B ∧ B <= 9) ∧ (0 <= C ∧ C <= 9))
  (h_subtraction : 100 * A + 10 * B + C - (10 * B + C) = 100 * A + 0 + A) :
  C = 9 := sorry

end NUMINAMATH_GPT_cryptarithm_C_value_l628_62826


namespace NUMINAMATH_GPT_find_a_l628_62811

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

-- Define the derivative of function f with respect to x
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

-- Define the condition for the problem
def condition (a : ℝ) : Prop := f' a 1 = 2

-- The statement to be proved
theorem find_a (a : ℝ) (h : condition a) : a = -3 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_find_a_l628_62811


namespace NUMINAMATH_GPT_pieces_of_gum_per_cousin_l628_62889

theorem pieces_of_gum_per_cousin (total_gum : ℕ) (num_cousins : ℕ) (h1 : total_gum = 20) (h2 : num_cousins = 4) : total_gum / num_cousins = 5 := by
  sorry

end NUMINAMATH_GPT_pieces_of_gum_per_cousin_l628_62889


namespace NUMINAMATH_GPT_find_a33_l628_62813

theorem find_a33 : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → a 2 = 6 → (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) → a 33 = 3 :=
by
  intros a h1 h2 h_rec
  sorry

end NUMINAMATH_GPT_find_a33_l628_62813


namespace NUMINAMATH_GPT_probability_below_8_l628_62839

theorem probability_below_8 
  (P10 P9 P8 : ℝ)
  (P10_eq : P10 = 0.24)
  (P9_eq : P9 = 0.28)
  (P8_eq : P8 = 0.19) :
  1 - (P10 + P9 + P8) = 0.29 := 
by
  sorry

end NUMINAMATH_GPT_probability_below_8_l628_62839


namespace NUMINAMATH_GPT_find_g_3_l628_62812

theorem find_g_3 (p q r : ℝ) (g : ℝ → ℝ) (h1 : g x = p * x^7 + q * x^3 + r * x + 7) (h2 : g (-3) = -11) (h3 : ∀ x, g (x) + g (-x) = 14) : g 3 = 25 :=
by 
  sorry

end NUMINAMATH_GPT_find_g_3_l628_62812


namespace NUMINAMATH_GPT_find_x_l628_62859

-- Define the angles AXB, CYX, and XYB as given in the problem.
def angle_AXB : ℝ := 150
def angle_CYX : ℝ := 130
def angle_XYB : ℝ := 55

-- Define a function that represents the sum of angles in a triangle.
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the angles.
def angle_XYZ : ℝ := angle_AXB - angle_XYB
def angle_YXZ : ℝ := 180 - angle_CYX
def angle_YXZ_proof (x : ℝ) : Prop := sum_of_angles_in_triangle angle_XYZ angle_YXZ x

-- State the theorem to be proved.
theorem find_x : angle_YXZ_proof 35 :=
sorry

end NUMINAMATH_GPT_find_x_l628_62859


namespace NUMINAMATH_GPT_nails_no_three_collinear_l628_62847

-- Let's denote the 8x8 chessboard as an 8x8 grid of cells

-- Define a type for positions on the chessboard
def Position := (ℕ × ℕ)

-- Condition: 16 nails should be placed in such a way that no three are collinear. 
-- Let's create an inductive type to capture these conditions

def no_three_collinear (nails : List Position) : Prop :=
  ∀ (p1 p2 p3 : Position), p1 ∈ nails → p2 ∈ nails → p3 ∈ nails → 
  (p1.1 = p2.1 ∧ p2.1 = p3.1) → False ∧
  (p1.2 = p2.2 ∧ p2.2 = p3.2) → False ∧
  (p1.1 - p1.2 = p2.1 - p2.2 ∧ p2.1 - p2.2 = p3.1 - p3.2) → False

-- The main statement to prove
theorem nails_no_three_collinear :
  ∃ nails : List Position, List.length nails = 16 ∧ no_three_collinear nails :=
sorry

end NUMINAMATH_GPT_nails_no_three_collinear_l628_62847


namespace NUMINAMATH_GPT_quadratic_complete_square_l628_62849

theorem quadratic_complete_square :
  ∀ x : ℝ, x^2 - 4 * x + 5 = (x - 2)^2 + 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l628_62849


namespace NUMINAMATH_GPT_union_sets_l628_62871

-- Define the sets A and B based on the given conditions
def set_A : Set ℝ := {x | abs (x - 1) < 2}
def set_B : Set ℝ := {x | Real.log x / Real.log 2 < 3}

-- Problem statement: Prove that the union of sets A and B is {x | -1 < x < 9}
theorem union_sets : (set_A ∪ set_B) = {x | -1 < x ∧ x < 9} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l628_62871


namespace NUMINAMATH_GPT_length_second_platform_l628_62825

-- Define the conditions
def length_train : ℕ := 100
def time_platform1 : ℕ := 15
def length_platform1 : ℕ := 350
def time_platform2 : ℕ := 20

-- Prove the length of the second platform is 500m
theorem length_second_platform : ∀ (speed_train : ℚ), 
  speed_train = (length_train + length_platform1) / time_platform1 →
  (speed_train = (length_train + L) / time_platform2) → 
  L = 500 :=
by 
  intro speed_train h1 h2
  sorry

end NUMINAMATH_GPT_length_second_platform_l628_62825


namespace NUMINAMATH_GPT_sum_of_first_9_terms_is_27_l628_62892

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Definition for the geometric sequence
noncomputable def b_n (n : ℕ) : ℝ := sorry -- Definition for the arithmetic sequence

axiom a_geo_seq : ∃ r : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n * r
axiom b_ari_seq : ∃ d : ℝ, ∀ n : ℕ, b_n (n + 1) = b_n n + d
axiom a5_eq_3 : 3 * a_n 5 - a_n 3 * a_n 7 = 0
axiom b5_eq_a5 : b_n 5 = a_n 5

noncomputable def S_9 := (1 / 2) * 9 * (b_n 1 + b_n 9)

theorem sum_of_first_9_terms_is_27 : S_9 = 27 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_is_27_l628_62892


namespace NUMINAMATH_GPT_child_tickets_sold_l628_62832

theorem child_tickets_sold
  (A C : ℕ)
  (h1 : A + C = 130)
  (h2 : 12 * A + 4 * C = 840) : C = 90 :=
  by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_child_tickets_sold_l628_62832


namespace NUMINAMATH_GPT_find_ages_of_siblings_l628_62895

-- Define the ages of the older brother and the younger sister as variables x and y
variables (x y : ℕ)

-- Define the conditions as provided in the problem
def condition1 : Prop := x = 4 * y
def condition2 : Prop := x + 3 = 3 * (y + 3)

-- State that the system of equations defined by condition1 and condition2 is consistent
theorem find_ages_of_siblings (x y : ℕ) (h1 : x = 4 * y) (h2 : x + 3 = 3 * (y + 3)) : 
  (x = 4 * y) ∧ (x + 3 = 3 * (y + 3)) :=
by 
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_find_ages_of_siblings_l628_62895


namespace NUMINAMATH_GPT_find_third_number_l628_62852

theorem find_third_number (x : ℕ) : 9548 + 7314 = x + 13500 ↔ x = 3362 :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l628_62852


namespace NUMINAMATH_GPT_arithmetic_sequence_a15_l628_62878

theorem arithmetic_sequence_a15 
  (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 3 + a 13 = 20)
  (h2 : a 2 = -2) :
  a 15 = 24 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a15_l628_62878


namespace NUMINAMATH_GPT_notebooks_last_days_l628_62803

-- Given conditions
def n := 5
def p := 40
def u := 4

-- Derived conditions
def total_pages := n * p
def days := total_pages / u

-- The theorem statement
theorem notebooks_last_days : days = 50 := sorry

end NUMINAMATH_GPT_notebooks_last_days_l628_62803


namespace NUMINAMATH_GPT_paint_for_cube_l628_62893

theorem paint_for_cube (paint_per_unit_area : ℕ → ℕ → ℕ)
  (h2 : paint_per_unit_area 2 1 = 1) :
  paint_per_unit_area 6 1 = 9 :=
by
  sorry

end NUMINAMATH_GPT_paint_for_cube_l628_62893


namespace NUMINAMATH_GPT_sum_of_three_numbers_is_neg_fifteen_l628_62863

theorem sum_of_three_numbers_is_neg_fifteen
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 5)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 10) :
  a + b + c = -15 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_is_neg_fifteen_l628_62863


namespace NUMINAMATH_GPT_find_y_l628_62857

theorem find_y 
  (x y : ℕ) 
  (h1 : x % y = 9) 
  (h2 : x / y = 96) 
  (h3 : (x % y: ℝ) / y = 0.12) 
  : y = 75 := 
  by 
    sorry

end NUMINAMATH_GPT_find_y_l628_62857


namespace NUMINAMATH_GPT_rhombus_perimeter_l628_62884

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) : ∃ p, p = 8 * Real.sqrt 41 := by
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  have h3 : s = 2 * Real.sqrt 41 := by sorry
  let p := 4 * s
  have h4 : p = 8 * Real.sqrt 41 := by sorry
  exact ⟨p, h4⟩

end NUMINAMATH_GPT_rhombus_perimeter_l628_62884


namespace NUMINAMATH_GPT_f_not_factorable_l628_62877

noncomputable def f (n : ℕ) (x : ℕ) : ℕ := x^n + 5 * x^(n - 1) + 3

theorem f_not_factorable (n : ℕ) (hn : n > 1) :
  ¬ ∃ g h : ℕ → ℕ, (∀ a b : ℕ, a ≠ 0 ∧ b ≠ 0 → g a * h b = f n a * f n b) ∧ 
    (∀ a b : ℕ, (g a = 0 ∧ h b = 0) → (a = 0 ∧ b = 0)) ∧ 
    (∃ pg qh : ℕ, pg ≥ 1 ∧ qh ≥ 1 ∧ g 1 = 1 ∧ h 1 = 1 ∧ (pg + qh = n)) := 
sorry

end NUMINAMATH_GPT_f_not_factorable_l628_62877


namespace NUMINAMATH_GPT_simplify_expression_l628_62800

def i : Complex := Complex.I

theorem simplify_expression : 7 * (4 - 2 * i) + 4 * i * (7 - 2 * i) = 36 + 14 * i := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l628_62800


namespace NUMINAMATH_GPT_even_function_increasing_on_negative_half_l628_62833

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

theorem even_function_increasing_on_negative_half (h1 : ∀ x, f (-x) = f x)
                                                  (h2 : ∀ a b : ℝ, a < b → b < 0 → f a < f b)
                                                  (h3 : x1 < 0 ∧ 0 < x2) (h4 : x1 + x2 > 0) 
                                                  : f (- x1) > f (x2) :=
by
  sorry

end NUMINAMATH_GPT_even_function_increasing_on_negative_half_l628_62833


namespace NUMINAMATH_GPT_sum_squares_6_to_14_l628_62808

def sum_of_squares (n : ℕ) := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_squares_6_to_14 :
  (sum_of_squares 14) - (sum_of_squares 5) = 960 :=
by
  sorry

end NUMINAMATH_GPT_sum_squares_6_to_14_l628_62808


namespace NUMINAMATH_GPT_A_eq_three_l628_62897

theorem A_eq_three (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (A : ℤ)
  (h : A = ((a + 1 : ℕ) / (b : ℕ)) + (b : ℕ) / (a : ℕ)) : A = 3 := by
  sorry

end NUMINAMATH_GPT_A_eq_three_l628_62897


namespace NUMINAMATH_GPT_find_points_A_C_find_equation_line_l_l628_62867

variables (A B C : ℝ × ℝ)
variables (l : ℝ → ℝ)

-- Condition: the coordinates of point B are (2, 1)
def B_coord : Prop := B = (2, 1)

-- Condition: the equation of the line containing the altitude on side BC is x - 2y - 1 = 0
def altitude_BC (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Condition: the equation of the angle bisector of angle A is y = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0

-- Statement of the theorems to be proved
theorem find_points_A_C
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0) :
  (A = (1, 0)) ∧ (C = (4, -3)) :=
sorry

theorem find_equation_line_l
    (hB : B_coord B)
    (h_altitude_BC : altitude_BC 1 0)
    (h_angle_bisector_A : angle_bisector_A 0)
    (hA : A = (1, 0)) :
  ((∀ x : ℝ, l x = x - 1)) :=
sorry

end NUMINAMATH_GPT_find_points_A_C_find_equation_line_l_l628_62867


namespace NUMINAMATH_GPT_option_A_sufficient_not_necessary_l628_62817

variable (a b : ℝ)

def A : Set ℝ := { x | x^2 - x + a ≤ 0 }
def B : Set ℝ := { x | x^2 - x + b ≤ 0 }

theorem option_A_sufficient_not_necessary : (A = B → a = b) ∧ (a = b → A = B) :=
by
  sorry

end NUMINAMATH_GPT_option_A_sufficient_not_necessary_l628_62817


namespace NUMINAMATH_GPT_smallest_n_for_inequality_l628_62807

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_inequality_l628_62807


namespace NUMINAMATH_GPT_cistern_fill_time_l628_62860

theorem cistern_fill_time (hA : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = C / 10) 
                          (hB : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = -(C / 15)) :
  ∀ C : ℝ, 0 < C → ∃ t : ℝ, t = 30 := 
by 
  sorry

end NUMINAMATH_GPT_cistern_fill_time_l628_62860


namespace NUMINAMATH_GPT_intersection_M_N_l628_62887

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l628_62887


namespace NUMINAMATH_GPT_janes_score_is_110_l628_62850

-- Definitions and conditions
def sarah_score_condition (x y : ℕ) : Prop := x = y + 50
def average_score_condition (x y : ℕ) : Prop := (x + y) / 2 = 110
def janes_score (x y : ℕ) : ℕ := (x + y) / 2

-- The proof problem statement
theorem janes_score_is_110 (x y : ℕ) 
  (h_sarah : sarah_score_condition x y) 
  (h_avg   : average_score_condition x y) : 
  janes_score x y = 110 := 
by
  sorry

end NUMINAMATH_GPT_janes_score_is_110_l628_62850


namespace NUMINAMATH_GPT_sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l628_62842

-- Define the first proof problem
theorem sqrt_1001_1003_plus_1_eq_1002 : Real.sqrt (1001 * 1003 + 1) = 1002 := 
by sorry

-- Define the second proof problem to verify the identity
theorem verify_identity (n : ℤ) : (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
by sorry

-- Define the third proof problem
theorem sqrt_2014_2017_plus_1_eq_2014_2017 : Real.sqrt (2014 * 2015 * 2016 * 2017 + 1) = 2014 * 2017 :=
by sorry

end NUMINAMATH_GPT_sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l628_62842


namespace NUMINAMATH_GPT_angles_on_y_axis_l628_62818

theorem angles_on_y_axis :
  {θ : ℝ | ∃ k : ℤ, (θ = 2 * k * Real.pi + Real.pi / 2) ∨ (θ = 2 * k * Real.pi + 3 * Real.pi / 2)} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by 
  sorry

end NUMINAMATH_GPT_angles_on_y_axis_l628_62818


namespace NUMINAMATH_GPT_Jimin_addition_l628_62802

theorem Jimin_addition (x : ℕ) (h : 96 / x = 6) : 34 + x = 50 := 
by
  sorry

end NUMINAMATH_GPT_Jimin_addition_l628_62802


namespace NUMINAMATH_GPT_max_point_of_f_l628_62840

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Define the first derivative of the function
def f_prime (x : ℝ) : ℝ := 3 * x^2 - 12

-- Define the second derivative of the function
def f_double_prime (x : ℝ) : ℝ := 6 * x

-- Prove that a = -2 is the maximum value point of f(x)
theorem max_point_of_f : ∃ a : ℝ, (f_prime a = 0) ∧ (f_double_prime a < 0) ∧ (a = -2) :=
sorry

end NUMINAMATH_GPT_max_point_of_f_l628_62840


namespace NUMINAMATH_GPT_sandy_change_from_twenty_dollar_bill_l628_62888

theorem sandy_change_from_twenty_dollar_bill :
  let cappuccino_cost := 2
  let iced_tea_cost := 3
  let cafe_latte_cost := 1.5
  let espresso_cost := 1
  let num_cappuccinos := 3
  let num_iced_teas := 2
  let num_cafe_lattes := 2
  let num_espressos := 2
  let total_cost := num_cappuccinos * cappuccino_cost
                  + num_iced_teas * iced_tea_cost
                  + num_cafe_lattes * cafe_latte_cost
                  + num_espressos * espresso_cost
  20 - total_cost = 3 := 
by
  sorry

end NUMINAMATH_GPT_sandy_change_from_twenty_dollar_bill_l628_62888


namespace NUMINAMATH_GPT_cloud9_total_revenue_after_discounts_and_refunds_l628_62836

theorem cloud9_total_revenue_after_discounts_and_refunds :
  let individual_total := 12000
  let individual_early_total := 3000
  let group_a_total := 6000
  let group_a_participants := 8
  let group_b_total := 9000
  let group_b_participants := 15
  let group_c_total := 15000
  let group_c_participants := 22
  let individual_refund1 := 500
  let individual_refund1_count := 3
  let individual_refund2 := 300
  let individual_refund2_count := 2
  let group_refund := 800
  let group_refund_count := 2

  -- Discounts
  let early_booking_discount := 0.03
  let discount_between_5_and_10 := 0.05
  let discount_between_11_and_20 := 0.1
  let discount_21_and_more := 0.15

  -- Calculating individual bookings
  let individual_early_discount_total := individual_early_total * early_booking_discount
  let individual_total_after_discount := individual_total - individual_early_discount_total

  -- Calculating group bookings
  let group_a_discount := group_a_total * discount_between_5_and_10
  let group_a_early_discount := (group_a_total - group_a_discount) * early_booking_discount
  let group_a_total_after_discount := group_a_total - group_a_discount - group_a_early_discount

  let group_b_discount := group_b_total * discount_between_11_and_20
  let group_b_total_after_discount := group_b_total - group_b_discount

  let group_c_discount := group_c_total * discount_21_and_more
  let group_c_early_discount := (group_c_total - group_c_discount) * early_booking_discount
  let group_c_total_after_discount := group_c_total - group_c_discount - group_c_early_discount

  let total_group_after_discount := group_a_total_after_discount + group_b_total_after_discount + group_c_total_after_discount

  -- Calculating refunds
  let total_individual_refunds := (individual_refund1 * individual_refund1_count) + (individual_refund2 * individual_refund2_count)
  let total_group_refunds := group_refund

  let total_refunds := total_individual_refunds + total_group_refunds

  -- Final total calculation after all discounts and refunds
  let final_total := individual_total_after_discount + total_group_after_discount - total_refunds
  final_total = 35006.50 := by
  -- The rest of the proof would go here, but we use sorry to bypass the proof.
  sorry

end NUMINAMATH_GPT_cloud9_total_revenue_after_discounts_and_refunds_l628_62836


namespace NUMINAMATH_GPT_find_fraction_l628_62835

theorem find_fraction
  (w x y F : ℝ)
  (h1 : 5 / w + F = 5 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  F = 10 := 
sorry

end NUMINAMATH_GPT_find_fraction_l628_62835


namespace NUMINAMATH_GPT_apples_per_case_l628_62806

theorem apples_per_case (total_apples : ℕ) (number_of_cases : ℕ) (h1 : total_apples = 1080) (h2 : number_of_cases = 90) : total_apples / number_of_cases = 12 := by
  sorry

end NUMINAMATH_GPT_apples_per_case_l628_62806


namespace NUMINAMATH_GPT_closest_ratio_adults_children_l628_62816

theorem closest_ratio_adults_children :
  ∃ (a c : ℕ), 25 * a + 15 * c = 1950 ∧ a ≥ 1 ∧ c ≥ 1 ∧ a / c = 24 / 25 := sorry

end NUMINAMATH_GPT_closest_ratio_adults_children_l628_62816


namespace NUMINAMATH_GPT_sarah_shaded_area_l628_62824

theorem sarah_shaded_area (r : ℝ) (A_square : ℝ) (A_circle : ℝ) (A_circles : ℝ) (A_shaded : ℝ) :
  let side_length := 27
  let radius := side_length / (3 * 2)
  let area_square := side_length * side_length
  let area_one_circle := Real.pi * (radius * radius)
  let total_area_circles := 9 * area_one_circle
  let shaded_area := area_square - total_area_circles
  shaded_area = 729 - 182.25 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_sarah_shaded_area_l628_62824


namespace NUMINAMATH_GPT_part_a_l628_62809

theorem part_a : 
  ∃ (x y : ℕ → ℕ), (∀ n : ℕ, (1 + Real.sqrt 33) ^ n = x n + y n * Real.sqrt 33) :=
sorry

end NUMINAMATH_GPT_part_a_l628_62809


namespace NUMINAMATH_GPT_jessica_mark_meet_time_jessica_mark_total_distance_l628_62855

noncomputable def jessica_start_time : ℚ := 7.75 -- 7:45 AM
noncomputable def mark_start_time : ℚ := 8.25 -- 8:15 AM
noncomputable def distance_between_towns : ℚ := 72
noncomputable def jessica_speed : ℚ := 14 -- miles per hour
noncomputable def mark_speed : ℚ := 18 -- miles per hour
noncomputable def t : ℚ := 81 / 32 -- time in hours when they meet

theorem jessica_mark_meet_time :
  7.75 + t = 10.28375 -- 10.17 hours in decimal
  :=
by
  -- Proof omitted.
  sorry

theorem jessica_mark_total_distance :
  jessica_speed * t + mark_speed * (t - (mark_start_time - jessica_start_time)) = distance_between_towns
  :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_jessica_mark_meet_time_jessica_mark_total_distance_l628_62855


namespace NUMINAMATH_GPT_correct_calculation_option_l628_62894

theorem correct_calculation_option :
  (∀ a : ℝ, 3 * a^5 - a^5 ≠ 3) ∧
  (∀ a : ℝ, a^2 + a^5 ≠ a^7) ∧
  (∀ a : ℝ, a^5 + a^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x^2 * y + x * y^2 ≠ 2 * x^3 * y^3) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_option_l628_62894


namespace NUMINAMATH_GPT_inequality_ab5_bc5_ca5_l628_62875

theorem inequality_ab5_bc5_ca5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) :=
sorry

end NUMINAMATH_GPT_inequality_ab5_bc5_ca5_l628_62875


namespace NUMINAMATH_GPT_rita_canoe_distance_l628_62861

theorem rita_canoe_distance 
  (up_speed : ℕ) (down_speed : ℕ)
  (wind_up_decrease : ℕ) (wind_down_increase : ℕ)
  (total_time : ℕ) 
  (effective_up_speed : ℕ := up_speed - wind_up_decrease)
  (effective_down_speed : ℕ := down_speed + wind_down_increase)
  (T_up : ℚ := D / effective_up_speed)
  (T_down : ℚ := D / effective_down_speed) :
  (T_up + T_down = total_time) ->
  (D = 7) := 
by
  sorry

-- Parameters as defined in the problem
def up_speed : ℕ := 3
def down_speed : ℕ := 9
def wind_up_decrease : ℕ := 2
def wind_down_increase : ℕ := 4
def total_time : ℕ := 8

end NUMINAMATH_GPT_rita_canoe_distance_l628_62861


namespace NUMINAMATH_GPT_number_of_cookies_on_the_fifth_plate_l628_62866

theorem number_of_cookies_on_the_fifth_plate
  (c : ℕ → ℕ)
  (h1 : c 1 = 5)
  (h2 : c 2 = 7)
  (h3 : c 3 = 10)
  (h4 : c 4 = 14)
  (h6 : c 6 = 25)
  (h_diff : ∀ n, c (n + 1) - c n = c (n + 2) - c (n + 1) + 1) :
  c 5 = 19 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cookies_on_the_fifth_plate_l628_62866


namespace NUMINAMATH_GPT_cartons_in_a_case_l628_62828

-- Definitions based on problem conditions
def numberOfBoxesInCarton (c : ℕ) (b : ℕ) : ℕ := c * b * 300
def paperClipsInTwoCases (c : ℕ) (b : ℕ) : ℕ := 2 * numberOfBoxesInCarton c b

-- Condition from problem statement: paperClipsInTwoCases c b = 600
theorem cartons_in_a_case 
  (c b : ℕ) 
  (h1 : paperClipsInTwoCases c b = 600) 
  (h2 : b ≥ 1) : 
  c = 1 := 
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_cartons_in_a_case_l628_62828


namespace NUMINAMATH_GPT_angela_finished_9_problems_l628_62896

def martha_problems : Nat := 2

def jenna_problems : Nat := 4 * martha_problems - 2

def mark_problems : Nat := jenna_problems / 2

def total_problems : Nat := 20

def total_friends_problems : Nat := martha_problems + jenna_problems + mark_problems

def angela_problems : Nat := total_problems - total_friends_problems

theorem angela_finished_9_problems : angela_problems = 9 := by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_angela_finished_9_problems_l628_62896


namespace NUMINAMATH_GPT_trick_deck_cost_l628_62882

theorem trick_deck_cost :
  ∀ (x : ℝ), 3 * x + 2 * x = 35 → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_trick_deck_cost_l628_62882


namespace NUMINAMATH_GPT_diametrically_opposite_points_l628_62858

theorem diametrically_opposite_points (n : ℕ) (h : (35 - 7 = n / 2)) : n = 56 := by
  sorry

end NUMINAMATH_GPT_diametrically_opposite_points_l628_62858


namespace NUMINAMATH_GPT_like_terms_monomials_m_n_l628_62834

theorem like_terms_monomials_m_n (m n : ℕ) (h1 : 3 * x ^ m * y = - x ^ 3 * y ^ n) :
  m - n = 2 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_monomials_m_n_l628_62834


namespace NUMINAMATH_GPT_gcd_of_128_144_480_is_16_l628_62851

-- Define the three numbers
def a := 128
def b := 144
def c := 480

-- Define the problem statement in Lean
theorem gcd_of_128_144_480_is_16 : Int.gcd (Int.gcd a b) c = 16 :=
by
  -- Definitions using given conditions
  -- use Int.gcd function to define the problem precisely.
  -- The proof will be left as "sorry" since we don't need to solve it
  sorry

end NUMINAMATH_GPT_gcd_of_128_144_480_is_16_l628_62851


namespace NUMINAMATH_GPT_optimal_garden_dimensions_l628_62838

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), l ≥ 100 ∧ w ≥ 60 ∧ l + w = 180 ∧ l * w = 8000 := by
  sorry

end NUMINAMATH_GPT_optimal_garden_dimensions_l628_62838


namespace NUMINAMATH_GPT_cost_of_blue_pill_l628_62841

variable (cost_total : ℝ) (days : ℕ) (daily_cost : ℝ)
variable (blue_pill : ℝ) (red_pill : ℝ)

-- Conditions
def condition1 (days : ℕ) : Prop := days = 21
def condition2 (blue_pill red_pill : ℝ) : Prop := blue_pill = red_pill + 2
def condition3 (cost_total daily_cost : ℝ) (days : ℕ) : Prop := cost_total = daily_cost * days
def condition4 (daily_cost blue_pill red_pill : ℝ) : Prop := daily_cost = blue_pill + red_pill

-- Target to prove
theorem cost_of_blue_pill
  (h1 : condition1 days)
  (h2 : condition2 blue_pill red_pill)
  (h3 : condition3 cost_total daily_cost days)
  (h4 : condition4 daily_cost blue_pill red_pill)
  (h5 : cost_total = 945) :
  blue_pill = 23.5 :=
by sorry

end NUMINAMATH_GPT_cost_of_blue_pill_l628_62841


namespace NUMINAMATH_GPT_min_weight_of_lightest_l628_62805

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end NUMINAMATH_GPT_min_weight_of_lightest_l628_62805


namespace NUMINAMATH_GPT_volume_between_spheres_l628_62801

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  volume_of_sphere 10 - volume_of_sphere 4 = (3744 / 3) * Real.pi := by
  sorry

end NUMINAMATH_GPT_volume_between_spheres_l628_62801


namespace NUMINAMATH_GPT_distance_at_2_point_5_l628_62862

def distance_data : List (ℝ × ℝ) :=
  [(0, 0), (1, 10), (2, 40), (3, 90), (4, 160), (5, 250)]

def quadratic_relation (t s k : ℝ) : Prop :=
  s = k * t^2

theorem distance_at_2_point_5 :
  ∃ k : ℝ, (∀ (t s : ℝ), (t, s) ∈ distance_data → quadratic_relation t s k) ∧ quadratic_relation 2.5 62.5 k :=
by
  sorry

end NUMINAMATH_GPT_distance_at_2_point_5_l628_62862


namespace NUMINAMATH_GPT_incorrect_statements_l628_62869

open Function

theorem incorrect_statements (a : ℝ) (x y x₁ y₁ x₂ y₂ k : ℝ) : 
  ¬ (a = -1 ↔ (∀ x y, a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0 → (a = -1 ∨ a = 0))) ∧ 
  ¬ (∀ x y (x₁ y₁ x₂ y₂ : ℝ), (∃ (m : ℝ), (y - y₁) = m * (x - x₁) ∧ (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)) → 
    ((y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁))) :=
sorry

end NUMINAMATH_GPT_incorrect_statements_l628_62869


namespace NUMINAMATH_GPT_integer_solutions_count_2009_l628_62873

theorem integer_solutions_count_2009 :
  ∃ s : Finset (ℤ × ℤ × ℤ), (∀ (x y z : ℤ), (x, y, z) ∈ s ↔ x * y * z = 2009) ∧ s.card = 72 :=
  sorry

end NUMINAMATH_GPT_integer_solutions_count_2009_l628_62873


namespace NUMINAMATH_GPT_angelina_speed_from_grocery_to_gym_l628_62814

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end NUMINAMATH_GPT_angelina_speed_from_grocery_to_gym_l628_62814


namespace NUMINAMATH_GPT_equal_constant_difference_l628_62854

theorem equal_constant_difference (x : ℤ) (k : ℤ) :
  x^2 - 6*x + 11 = k ∧ -x^2 + 8*x - 13 = k ∧ 3*x^2 - 16*x + 19 = k → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_equal_constant_difference_l628_62854


namespace NUMINAMATH_GPT_brownie_count_l628_62881

noncomputable def initial_brownies : ℕ := 20
noncomputable def to_school_administrator (n : ℕ) : ℕ := n / 2
noncomputable def remaining_after_administrator (n : ℕ) : ℕ := n - to_school_administrator n
noncomputable def to_best_friend (n : ℕ) : ℕ := remaining_after_administrator n / 2
noncomputable def remaining_after_best_friend (n : ℕ) : ℕ := remaining_after_administrator n - to_best_friend n
noncomputable def to_friend_simon : ℕ := 2
noncomputable def final_brownies : ℕ := remaining_after_best_friend initial_brownies - to_friend_simon

theorem brownie_count : final_brownies = 3 := by
  sorry

end NUMINAMATH_GPT_brownie_count_l628_62881


namespace NUMINAMATH_GPT_sum_groups_is_250_l628_62864

-- Definitions based on the conditions
def group1 := [3, 13, 23, 33, 43]
def group2 := [7, 17, 27, 37, 47]

-- The proof problem
theorem sum_groups_is_250 : (group1.sum + group2.sum) = 250 :=
by
  sorry

end NUMINAMATH_GPT_sum_groups_is_250_l628_62864


namespace NUMINAMATH_GPT_mike_hours_per_day_l628_62899

theorem mike_hours_per_day (total_hours : ℕ) (total_days : ℕ) (h_total_hours : total_hours = 15) (h_total_days : total_days = 5) : (total_hours / total_days) = 3 := by
  sorry

end NUMINAMATH_GPT_mike_hours_per_day_l628_62899


namespace NUMINAMATH_GPT_maddy_credits_to_graduate_l628_62819

theorem maddy_credits_to_graduate (semesters : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ)
  (semesters_eq : semesters = 8)
  (credits_per_class_eq : credits_per_class = 3)
  (classes_per_semester_eq : classes_per_semester = 5) :
  semesters * (classes_per_semester * credits_per_class) = 120 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_maddy_credits_to_graduate_l628_62819


namespace NUMINAMATH_GPT_compute_f_g_at_2_l628_62820

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 4 * x - 1

theorem compute_f_g_at_2 :
  f (g 2) = 49 :=
by
  sorry

end NUMINAMATH_GPT_compute_f_g_at_2_l628_62820


namespace NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l628_62821

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l628_62821


namespace NUMINAMATH_GPT_boxes_total_is_correct_l628_62815

def initial_boxes : ℕ := 7
def additional_boxes_per_box : ℕ := 7
def final_non_empty_boxes : ℕ := 10
def total_boxes := 77

theorem boxes_total_is_correct
  (h1 : initial_boxes = 7)
  (h2 : additional_boxes_per_box = 7)
  (h3 : final_non_empty_boxes = 10)
  : total_boxes = 77 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_boxes_total_is_correct_l628_62815


namespace NUMINAMATH_GPT_cubical_cake_l628_62829

noncomputable def cubical_cake_properties : Prop :=
  let a : ℝ := 3
  let top_area := (1 / 2) * 3 * 1.5
  let height := 3
  let volume := top_area * height
  let vertical_triangles_area := 2 * ((1 / 2) * 1.5 * 3)
  let vertical_rectangular_area := 3 * 3
  let iced_area := top_area + vertical_triangles_area + vertical_rectangular_area
  volume + iced_area = 22.5

theorem cubical_cake : cubical_cake_properties := sorry

end NUMINAMATH_GPT_cubical_cake_l628_62829


namespace NUMINAMATH_GPT_tracy_initial_candies_l628_62823

variable (x : ℕ)
variable (b : ℕ)

theorem tracy_initial_candies : 
  (x % 6 = 0) ∧
  (34 ≤ (1 / 2 * x)) ∧
  ((1 / 2 * x) ≤ 38) ∧
  (1 ≤ b) ∧
  (b ≤ 5) ∧
  (1 / 2 * x - 30 - b = 3) →
  x = 72 := 
sorry

end NUMINAMATH_GPT_tracy_initial_candies_l628_62823


namespace NUMINAMATH_GPT_guilt_of_X_and_Y_l628_62843

-- Definitions
variable (X Y : Prop)

-- Conditions
axiom condition1 : ¬X ∨ Y
axiom condition2 : X

-- Conclusion to prove
theorem guilt_of_X_and_Y : X ∧ Y := by
  sorry

end NUMINAMATH_GPT_guilt_of_X_and_Y_l628_62843


namespace NUMINAMATH_GPT_decreasing_on_neg_l628_62868

variable (f : ℝ → ℝ)

-- Condition 1: f(x) is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Condition 2: f(x) is increasing on (0, +∞)
def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove: f(x) is decreasing on (-∞, 0)
theorem decreasing_on_neg (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : increasing_on_pos f) :
  ∀ x y, x < y → y < 0 → f y < f x :=
by 
  sorry

end NUMINAMATH_GPT_decreasing_on_neg_l628_62868


namespace NUMINAMATH_GPT_find_base_of_triangle_l628_62844

def triangle_base (area : ℝ) (height : ℝ) (base : ℝ) : Prop :=
  area = (base * height) / 2

theorem find_base_of_triangle : triangle_base 24 8 6 :=
by
  -- Simplification and computation steps are omitted as per the instruction
  sorry

end NUMINAMATH_GPT_find_base_of_triangle_l628_62844


namespace NUMINAMATH_GPT_function_properties_l628_62890

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / x

theorem function_properties : 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) + 2 * f x = 3 * x) ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 < x → ∀ y : ℝ, x < y → f x < f y) := by
  -- Proof of the theorem would go here
  sorry

end NUMINAMATH_GPT_function_properties_l628_62890


namespace NUMINAMATH_GPT_length_EQ_l628_62810

-- Define the square EFGH with side length 8
def square_EFGH (a : ℝ) (b : ℝ): Prop := a = 8 ∧ b = 8

-- Define the rectangle IJKL with IL = 12 and JK = 8
def rectangle_IJKL (l : ℝ) (w : ℝ): Prop := l = 12 ∧ w = 8

-- Define the perpendicularity of EH and IJ
def perpendicular_EH_IJ : Prop := true

-- Define the shaded area condition
def shaded_area_condition (area_IJKL : ℝ) (shaded_area : ℝ): Prop :=
  shaded_area = (1/3) * area_IJKL

-- Theorem to prove
theorem length_EQ (a b l w area_IJKL shaded_area EH HG HQ EQ : ℝ):
  square_EFGH a b →
  rectangle_IJKL l w →
  perpendicular_EH_IJ →
  shaded_area_condition area_IJKL shaded_area →
  HQ * HG = shaded_area →
  EQ = EH - HQ →
  EQ = 4 := by
  intros hSquare hRectangle hPerpendicular hShadedArea hHQHG hEQ
  sorry

end NUMINAMATH_GPT_length_EQ_l628_62810


namespace NUMINAMATH_GPT_smaller_of_two_digit_product_4680_l628_62891

theorem smaller_of_two_digit_product_4680 (a b : ℕ) (h1 : a * b = 4680) (h2 : 10 ≤ a) (h3 : a < 100) (h4 : 10 ≤ b) (h5 : b < 100): min a b = 40 :=
sorry

end NUMINAMATH_GPT_smaller_of_two_digit_product_4680_l628_62891


namespace NUMINAMATH_GPT_ball_initial_height_l628_62837

theorem ball_initial_height (c : ℝ) (d : ℝ) (h : ℝ) 
  (H1 : c = 4 / 5) 
  (H2 : d = 1080) 
  (H3 : d = h + 2 * h * c / (1 - c)) : 
  h = 216 :=
sorry

end NUMINAMATH_GPT_ball_initial_height_l628_62837


namespace NUMINAMATH_GPT_center_of_circle_sum_l628_62831

open Real

theorem center_of_circle_sum (x y : ℝ) (h k : ℝ) :
  (x - h)^2 + (y - k)^2 = 2 → (h = 3) → (k = 4) → h + k = 7 :=
by
  intro h_eq k_eq
  sorry

end NUMINAMATH_GPT_center_of_circle_sum_l628_62831


namespace NUMINAMATH_GPT_solve_factorial_equation_in_natural_numbers_l628_62885

theorem solve_factorial_equation_in_natural_numbers :
  ∃ n k : ℕ, n! + 3 * n + 8 = k^2 ↔ n = 2 ∧ k = 4 := by
sorry

end NUMINAMATH_GPT_solve_factorial_equation_in_natural_numbers_l628_62885


namespace NUMINAMATH_GPT_correct_average_after_error_l628_62876

theorem correct_average_after_error (n : ℕ) (a m_wrong m_correct : ℤ) 
  (h_n : n = 30) (h_a : a = 60) (h_m_wrong : m_wrong = 90) (h_m_correct : m_correct = 15) : 
  ((n * a + (m_correct - m_wrong)) / n : ℤ) = 57 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_after_error_l628_62876


namespace NUMINAMATH_GPT_comic_book_issue_pages_l628_62872

theorem comic_book_issue_pages (total_pages: ℕ) 
  (speed_month1 speed_month2 speed_month3: ℕ) 
  (bonus_pages: ℕ) (issue1_2_pages: ℕ) 
  (issue3_pages: ℕ)
  (h1: total_pages = 220)
  (h2: speed_month1 = 5)
  (h3: speed_month2 = 4)
  (h4: speed_month3 = 4)
  (h5: issue3_pages = issue1_2_pages + 4)
  (h6: bonus_pages = 3)
  (h7: (issue1_2_pages + bonus_pages) + 
       (issue1_2_pages + bonus_pages) + 
       (issue3_pages + bonus_pages) = total_pages) : 
  issue1_2_pages = 69 := 
by 
  sorry

end NUMINAMATH_GPT_comic_book_issue_pages_l628_62872


namespace NUMINAMATH_GPT_calculator_sum_l628_62804

theorem calculator_sum :
  let A := 2
  let B := 0
  let C := -1
  let D := 3
  let n := 47
  let A' := if n % 2 = 1 then -A else A
  let B' := B -- B remains 0 after any number of sqrt operations
  let C' := if n % 2 = 1 then -C else C
  let D' := D ^ (3 ^ n)
  A' + B' + C' + D' = 3 ^ (3 ^ 47) - 3
:= by
  sorry

end NUMINAMATH_GPT_calculator_sum_l628_62804


namespace NUMINAMATH_GPT_lesser_fraction_of_sum_and_product_l628_62830

open Real

theorem lesser_fraction_of_sum_and_product (a b : ℚ)
  (h1 : a + b = 11 / 12)
  (h2 : a * b = 1 / 6) :
  min a b = 1 / 4 :=
sorry

end NUMINAMATH_GPT_lesser_fraction_of_sum_and_product_l628_62830


namespace NUMINAMATH_GPT_lcm_of_4_6_10_18_l628_62865

theorem lcm_of_4_6_10_18 : Nat.lcm (Nat.lcm 4 6) (Nat.lcm 10 18) = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_of_4_6_10_18_l628_62865


namespace NUMINAMATH_GPT_solution_set_of_absolute_value_inequality_l628_62879

theorem solution_set_of_absolute_value_inequality {x : ℝ} : 
  (|2 * x - 3| > 1) ↔ (x < 1 ∨ x > 2) := 
sorry

end NUMINAMATH_GPT_solution_set_of_absolute_value_inequality_l628_62879


namespace NUMINAMATH_GPT_coloring_possible_l628_62846

theorem coloring_possible (n k : ℕ) (h1 : 2 ≤ k ∧ k ≤ n) (h2 : n ≥ 2) :
  (n ≥ k ∧ k ≥ 3) ∨ (2 ≤ k ∧ k ≤ n ∧ n ≤ 3) :=
sorry

end NUMINAMATH_GPT_coloring_possible_l628_62846


namespace NUMINAMATH_GPT_f_2002_l628_62898

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (n : ℕ) (h : n > 1) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

axiom f_2001 : f 2001 = 1

theorem f_2002 : f 2002 = 2 :=
  sorry

end NUMINAMATH_GPT_f_2002_l628_62898


namespace NUMINAMATH_GPT_largest_integer_divisor_of_p_squared_minus_3q_squared_l628_62853

theorem largest_integer_divisor_of_p_squared_minus_3q_squared (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) :
  ∃ d : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → d ∣ (p^2 - 3*q^2)) ∧ 
           (∀ k : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → k ∣ (p^2 - 3*q^2)) → k ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_GPT_largest_integer_divisor_of_p_squared_minus_3q_squared_l628_62853


namespace NUMINAMATH_GPT_center_of_circle_l628_62822

-- Defining the equation of the circle as a hypothesis
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y = 0

-- Stating the theorem about the center of the circle
theorem center_of_circle : ∀ x y : ℝ, circle_eq x y → (x = 2 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l628_62822


namespace NUMINAMATH_GPT_exists_root_in_interval_l628_62856

open Real

theorem exists_root_in_interval 
  (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hr : a * r ^ 2 + b * r + c = 0) 
  (hs : -a * s ^ 2 + b * s + c = 0) : 
  ∃ t : ℝ, r < t ∧ t < s ∧ (a / 2) * t ^ 2 + b * t + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_root_in_interval_l628_62856
