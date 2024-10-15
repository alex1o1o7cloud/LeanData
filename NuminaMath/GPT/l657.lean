import Mathlib

namespace NUMINAMATH_GPT_brass_to_band_ratio_l657_65745

theorem brass_to_band_ratio
  (total_students : ℕ)
  (marching_band_fraction brass_saxophone_fraction saxophone_alto_fraction : ℚ)
  (alto_saxophone_students : ℕ)
  (h1 : total_students = 600)
  (h2 : marching_band_fraction = 1 / 5)
  (h3 : brass_saxophone_fraction = 1 / 5)
  (h4 : saxophone_alto_fraction = 1 / 3)
  (h5 : alto_saxophone_students = 4) :
  ((brass_saxophone_fraction * saxophone_alto_fraction) * total_students * marching_band_fraction = 4) →
  ((brass_saxophone_fraction * 3 * marching_band_fraction * total_students) / (marching_band_fraction * total_students) = 1 / 2) :=
by {
  -- Here we state the proof but leave it as a sorry placeholder.
  sorry
}

end NUMINAMATH_GPT_brass_to_band_ratio_l657_65745


namespace NUMINAMATH_GPT_find_angle_B_find_side_b_l657_65706

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {m n : ℝ × ℝ}
variable {dot_product_max : ℝ}

-- Conditions
def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin A + c * Real.sin C - b * Real.sin B = Real.sqrt 2 * a * Real.sin C

def vectors (m n : ℝ × ℝ) := 
  m = (Real.cos A, Real.cos (2 * A)) ∧ n = (12, -5)

def side_length_a (a : ℝ) := 
  a = 4

-- Questions and Proof Problems
theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1 : triangle_condition a b c A B C) : 
  B = π / 4 :=
sorry

theorem find_side_b (A B C : ℝ) (a b c : ℝ) 
  (m n : ℝ × ℝ) (max_dot_product_condition : Real.cos A = 3 / 5) 
  (ha : side_length_a a) (hb : b = a * Real.sin B / Real.sin A) : 
  b = 5 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_find_angle_B_find_side_b_l657_65706


namespace NUMINAMATH_GPT_geom_sequence_common_ratio_l657_65738

variable {α : Type*} [LinearOrderedField α]

theorem geom_sequence_common_ratio (a1 q : α) (h : a1 > 0) (h_eq : a1 + a1 * q + a1 * q^2 + a1 * q = 9 * a1 * q^2) : q = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_geom_sequence_common_ratio_l657_65738


namespace NUMINAMATH_GPT_negative_double_inequality_l657_65777

theorem negative_double_inequality (a : ℝ) (h : a < 0) : 2 * a < a :=
by { sorry }

end NUMINAMATH_GPT_negative_double_inequality_l657_65777


namespace NUMINAMATH_GPT_weekly_allowance_is_8_l657_65715

variable (A : ℝ)

def condition_1 (A : ℝ) : Prop := ∃ A : ℝ, A / 2 + 8 = 12

theorem weekly_allowance_is_8 (A : ℝ) (h : condition_1 A) : A = 8 :=
sorry

end NUMINAMATH_GPT_weekly_allowance_is_8_l657_65715


namespace NUMINAMATH_GPT_probability_of_red_card_l657_65707

theorem probability_of_red_card (successful_attempts not_successful_attempts : ℕ) (h : successful_attempts = 5) (h2 : not_successful_attempts = 8) : (successful_attempts / (successful_attempts + not_successful_attempts) : ℚ) = 5 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_of_red_card_l657_65707


namespace NUMINAMATH_GPT_initial_trees_count_l657_65735

variable (x : ℕ)

-- Conditions of the problem
def initial_rows := 24
def additional_rows := 12
def total_rows := initial_rows + additional_rows
def trees_per_row_initial := x
def trees_per_row_final := 28

-- Total number of trees should remain constant
theorem initial_trees_count :
  initial_rows * trees_per_row_initial = total_rows * trees_per_row_final → 
  trees_per_row_initial = 42 := 
by sorry

end NUMINAMATH_GPT_initial_trees_count_l657_65735


namespace NUMINAMATH_GPT_quadratic_root_identity_l657_65734

theorem quadratic_root_identity (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_identity_l657_65734


namespace NUMINAMATH_GPT_find_k_l657_65723

-- Definitions of the vectors and condition about perpendicularity
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (-2, k)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- The theorem that states if vector_a is perpendicular to (2 * vector_a - vector_b), then k = 14
theorem find_k (k : ℝ) (h : perpendicular vector_a (2 • vector_a - vector_b k)) : k = 14 := sorry

end NUMINAMATH_GPT_find_k_l657_65723


namespace NUMINAMATH_GPT_x_gt_1_sufficient_but_not_necessary_x_gt_0_l657_65746

theorem x_gt_1_sufficient_but_not_necessary_x_gt_0 (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬(x > 0 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_x_gt_1_sufficient_but_not_necessary_x_gt_0_l657_65746


namespace NUMINAMATH_GPT_max_radius_of_inner_spheres_l657_65770

theorem max_radius_of_inner_spheres (R : ℝ) : 
  ∃ r : ℝ, (2 * r ≤ R) ∧ (r ≤ (4 * Real.sqrt 2 - 1) / 4 * R) :=
sorry

end NUMINAMATH_GPT_max_radius_of_inner_spheres_l657_65770


namespace NUMINAMATH_GPT_find_larger_integer_l657_65724

variable (x : ℤ) (smaller larger : ℤ)
variable (ratio_1_to_4 : smaller = 1 * x ∧ larger = 4 * x)
variable (condition : smaller + 12 = larger)

theorem find_larger_integer : larger = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_integer_l657_65724


namespace NUMINAMATH_GPT_tan_x_eq_2_solution_set_l657_65762

theorem tan_x_eq_2_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} = {x : ℝ | Real.tan x = 2} :=
sorry

end NUMINAMATH_GPT_tan_x_eq_2_solution_set_l657_65762


namespace NUMINAMATH_GPT_solution_to_inequality_l657_65704

theorem solution_to_inequality (x : ℝ) :
  (∃ y : ℝ, y = x^(1/3) ∧ y + 3 / (y + 2) ≤ 0) ↔ x < -8 := 
sorry

end NUMINAMATH_GPT_solution_to_inequality_l657_65704


namespace NUMINAMATH_GPT_jake_weight_l657_65790

variable (J K : ℕ)

-- Conditions given in the problem
axiom h1 : J - 8 = 2 * K
axiom h2 : J + K = 293

-- Statement to prove
theorem jake_weight : J = 198 :=
by
  sorry

end NUMINAMATH_GPT_jake_weight_l657_65790


namespace NUMINAMATH_GPT_cards_received_while_in_hospital_l657_65747

theorem cards_received_while_in_hospital (T H C : ℕ) (hT : T = 690) (hC : C = 287) (hH : H = T - C) : H = 403 :=
by
  sorry

end NUMINAMATH_GPT_cards_received_while_in_hospital_l657_65747


namespace NUMINAMATH_GPT_part1_part2_l657_65710

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + (1 + a) * Real.exp (-x)

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = 0 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≥ a + 1) → a ≤ 3 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l657_65710


namespace NUMINAMATH_GPT_sphere_volume_l657_65765

theorem sphere_volume (length width : ℝ) (angle_deg : ℝ) (h_length : length = 4) (h_width : width = 3) (h_angle : angle_deg = 60) :
  ∃ (volume : ℝ), volume = (125 / 6) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l657_65765


namespace NUMINAMATH_GPT_problem_statement_l657_65748

theorem problem_statement (x : ℝ) (h : x^2 + 4 * x - 2 = 0) : 3 * x^2 + 12 * x - 23 = -17 :=
sorry

end NUMINAMATH_GPT_problem_statement_l657_65748


namespace NUMINAMATH_GPT_inradii_sum_l657_65711

theorem inradii_sum (ABCD : Type) (r_a r_b r_c r_d : ℝ) 
  (inscribed_quadrilateral : Prop) 
  (inradius_BCD : Prop) 
  (inradius_ACD : Prop) 
  (inradius_ABD : Prop) 
  (inradius_ABC : Prop) 
  (Tebo_theorem : Prop) :
  r_a + r_c = r_b + r_d := 
by
  sorry

end NUMINAMATH_GPT_inradii_sum_l657_65711


namespace NUMINAMATH_GPT_original_average_rent_is_800_l657_65749

def original_rent (A : ℝ) : Prop :=
  let friends : ℝ := 4
  let old_rent : ℝ := 800
  let increased_rent : ℝ := old_rent * 1.25
  let new_total_rent : ℝ := (850 * friends)
  old_rent * 4 - 800 + increased_rent = new_total_rent

theorem original_average_rent_is_800 (A : ℝ) : original_rent A → A = 800 :=
by 
  sorry

end NUMINAMATH_GPT_original_average_rent_is_800_l657_65749


namespace NUMINAMATH_GPT_cos_theta_eq_neg_2_div_sqrt_13_l657_65754

theorem cos_theta_eq_neg_2_div_sqrt_13 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < π) 
  (h3 : Real.tan θ = -3/2) : 
  Real.cos θ = -2 / Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_cos_theta_eq_neg_2_div_sqrt_13_l657_65754


namespace NUMINAMATH_GPT_log_4_135_eq_half_log_2_45_l657_65720

noncomputable def a : ℝ := Real.log 135 / Real.log 4
noncomputable def b : ℝ := Real.log 45 / Real.log 2

theorem log_4_135_eq_half_log_2_45 : a = b / 2 :=
by
  sorry

end NUMINAMATH_GPT_log_4_135_eq_half_log_2_45_l657_65720


namespace NUMINAMATH_GPT_smallest_number_of_locks_and_keys_l657_65773

open Finset Nat

-- Definitions based on conditions
def committee : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

def can_open_safe (members : Finset ℕ) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 6 → members ⊆ subset

def cannot_open_safe (members : Finset ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset.card = 5 ∧ members ⊆ subset

-- Proof statement
theorem smallest_number_of_locks_and_keys :
  ∃ (locks keys : ℕ), locks = 462 ∧ keys = 2772 ∧
  (∀ (subset : Finset ℕ), subset.card = 6 → can_open_safe subset) ∧
  (∀ (subset : Finset ℕ), subset.card = 5 → ¬can_open_safe subset) :=
sorry

end NUMINAMATH_GPT_smallest_number_of_locks_and_keys_l657_65773


namespace NUMINAMATH_GPT_intersection_A_B_subset_A_B_l657_65783

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def set_B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

-- Problem 1: Prove A ∩ B when a = -1
theorem intersection_A_B (a : ℝ) (h : a = -1) : set_A a ∩ set_B = {x | 1 / 2 < x ∧ x < 2} :=
sorry

-- Problem 2: Find the range of a such that A ⊆ B
theorem subset_A_B (a : ℝ) : (-1 < a ∧ a ≤ 1) ↔ (set_A a ⊆ set_B) :=
sorry

end NUMINAMATH_GPT_intersection_A_B_subset_A_B_l657_65783


namespace NUMINAMATH_GPT_intersection_points_l657_65760

noncomputable def line1 (x y : ℝ) : Prop := 3 * x - 2 * y = 12
noncomputable def line2 (x y : ℝ) : Prop := 2 * x + 4 * y = 8
noncomputable def line3 (x y : ℝ) : Prop := -5 * x + 15 * y = 30
noncomputable def line4 (x : ℝ) : Prop := x = -3

theorem intersection_points : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ 
  (∃ (x y : ℝ), line1 x y ∧ x = -3 ∧ y = -10.5) ∧ 
  ¬(∃ (x y : ℝ), line2 x y ∧ line3 x y) ∧
  ∃ (x y : ℝ), line4 x ∧ y = -10.5 :=
  sorry

end NUMINAMATH_GPT_intersection_points_l657_65760


namespace NUMINAMATH_GPT_minimum_value_l657_65739

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + (1 / (a * b * c)) ≥ 10 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l657_65739


namespace NUMINAMATH_GPT_find_smallest_n_l657_65741

noncomputable def smallest_n (c : ℕ) (n : ℕ) : Prop :=
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ c → n + 2 - 2*k ≥ 0) ∧ c * (n - c + 1) = 2009

theorem find_smallest_n : ∃ n c : ℕ, smallest_n c n ∧ n = 89 :=
sorry

end NUMINAMATH_GPT_find_smallest_n_l657_65741


namespace NUMINAMATH_GPT_vaccine_codes_l657_65767

theorem vaccine_codes (vaccines : List ℕ) :
  vaccines = [785, 567, 199, 507, 175] :=
  by
  sorry

end NUMINAMATH_GPT_vaccine_codes_l657_65767


namespace NUMINAMATH_GPT_ratio_of_remaining_areas_of_squares_l657_65787

/--
  Given:
  - Square C has a side length of 48 cm.
  - Square D has a side length of 60 cm.
  - A smaller square of side length 12 cm is cut out from both squares.

  Show that:
  - The ratio of the remaining area of square C to the remaining area of square D is 5/8.
-/
theorem ratio_of_remaining_areas_of_squares : 
  let sideC := 48
  let sideD := 60
  let sideSmall := 12
  let areaC := sideC * sideC
  let areaD := sideD * sideD
  let areaSmall := sideSmall * sideSmall
  let remainingC := areaC - areaSmall
  let remainingD := areaD - areaSmall
  (remainingC : ℚ) / remainingD = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_remaining_areas_of_squares_l657_65787


namespace NUMINAMATH_GPT_simplify_expression_l657_65761

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_expression_l657_65761


namespace NUMINAMATH_GPT_largest_interesting_number_l657_65797

def is_interesting_number (x : ℝ) : Prop :=
  ∃ y z : ℝ, (0 ≤ y ∧ y < 1) ∧ (0 ≤ z ∧ z < 1) ∧ x = 0 + y * 10⁻¹ + z ∧ 2 * (0 + y * 10⁻¹ + z) = 0 + z

theorem largest_interesting_number : ∀ x, is_interesting_number x → x ≤ 0.375 :=
by
  sorry

end NUMINAMATH_GPT_largest_interesting_number_l657_65797


namespace NUMINAMATH_GPT_fewest_four_dollar_frisbees_l657_65703

theorem fewest_four_dollar_frisbees (x y: ℕ): 
    x + y = 64 ∧ 3 * x + 4 * y = 200 → y = 8 := by sorry

end NUMINAMATH_GPT_fewest_four_dollar_frisbees_l657_65703


namespace NUMINAMATH_GPT_randy_money_left_l657_65737

theorem randy_money_left (initial_money lunch ice_cream_cone remaining : ℝ) 
  (h1 : initial_money = 30)
  (h2 : lunch = 10)
  (h3 : remaining = initial_money - lunch)
  (h4 : ice_cream_cone = remaining * (1/4)) :
  (remaining - ice_cream_cone) = 15 := by
  sorry

end NUMINAMATH_GPT_randy_money_left_l657_65737


namespace NUMINAMATH_GPT_katarina_miles_l657_65743

theorem katarina_miles 
  (total_miles : ℕ) 
  (miles_harriet : ℕ) 
  (miles_tomas : ℕ)
  (miles_tyler : ℕ)
  (miles_katarina : ℕ) 
  (combined_miles : total_miles = 195) 
  (same_miles : miles_tomas = miles_harriet ∧ miles_tyler = miles_harriet)
  (harriet_miles : miles_harriet = 48) :
  miles_katarina = 51 :=
sorry

end NUMINAMATH_GPT_katarina_miles_l657_65743


namespace NUMINAMATH_GPT_geometric_sequence_increasing_condition_l657_65789

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (h_geo : is_geometric a) (h_cond : a 0 < a 1 ∧ a 1 < a 2) :
  ¬(∀ n : ℕ, a n < a (n + 1)) → (a 0 < a 1 ∧ a 1 < a 2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_increasing_condition_l657_65789


namespace NUMINAMATH_GPT_sum_of_three_largest_l657_65742

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_largest_l657_65742


namespace NUMINAMATH_GPT_count_false_propositions_l657_65709

theorem count_false_propositions 
  (P : Prop) 
  (inverse_P : Prop) 
  (negation_P : Prop) 
  (converse_P : Prop) 
  (h1 : ¬P) 
  (h2 : inverse_P) 
  (h3 : negation_P ↔ ¬P) 
  (h4 : converse_P ↔ P) : 
  ∃ n : ℕ, n = 2 ∧ 
  ¬P ∧ ¬converse_P ∧ 
  inverse_P ∧ negation_P := 
sorry

end NUMINAMATH_GPT_count_false_propositions_l657_65709


namespace NUMINAMATH_GPT_number_of_solutions_eq_4_l657_65729

noncomputable def num_solutions := 
  ∃ n : ℕ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → (3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * Real.cos x = 0) → n = 4)

-- To state the above more clearly, we can add an abbreviation function for the equation.
noncomputable def equation (x : ℝ) : ℝ := 3 * (Real.cos x) ^ 3 - 7 * (Real.cos x) ^ 2 + 3 * Real.cos x

theorem number_of_solutions_eq_4 :
  (∃ n, n = 4 ∧ ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → equation x = 0 → true) := sorry

end NUMINAMATH_GPT_number_of_solutions_eq_4_l657_65729


namespace NUMINAMATH_GPT_max_men_with_all_items_l657_65708

theorem max_men_with_all_items (total_men married men_with_TV men_with_radio men_with_AC men_with_car men_with_smartphone : ℕ) 
  (H_married : married = 2300) 
  (H_TV : men_with_TV = 2100) 
  (H_radio : men_with_radio = 2600) 
  (H_AC : men_with_AC = 1800) 
  (H_car : men_with_car = 2500) 
  (H_smartphone : men_with_smartphone = 2200) : 
  ∃ m, m ≤ married ∧ m ≤ men_with_TV ∧ m ≤ men_with_radio ∧ m ≤ men_with_AC ∧ m ≤ men_with_car ∧ m ≤ men_with_smartphone ∧ m = 1800 := 
  sorry

end NUMINAMATH_GPT_max_men_with_all_items_l657_65708


namespace NUMINAMATH_GPT_length_of_train_l657_65786

variable (L V : ℝ)

def platform_crossing (L V : ℝ) := L + 350 = V * 39
def post_crossing (L V : ℝ) := L = V * 18

theorem length_of_train (h1 : platform_crossing L V) (h2 : post_crossing L V) : L = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l657_65786


namespace NUMINAMATH_GPT_product_of_two_numbers_l657_65794

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l657_65794


namespace NUMINAMATH_GPT_sum_of_grid_numbers_l657_65791

theorem sum_of_grid_numbers (A E: ℕ) (S: ℕ) 
    (hA: A = 2) 
    (hE: E = 3)
    (h1: ∃ B : ℕ, 2 + B = S ∧ 3 + B = S)
    (h2: ∃ D : ℕ, 2 + D = S ∧ D + 3 = S)
    (h3: ∃ F : ℕ, 3 + F = S ∧ F + 3 = S)
    (h4: ∃ G H I: ℕ, 
         2 + G = S ∧ G + H = S ∧ H + C = S ∧ 
         3 + H = S ∧ E + I = S ∧ H + I = S):
  A + B + C + D + E + F + G + H + I = 22 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_grid_numbers_l657_65791


namespace NUMINAMATH_GPT_range_of_a_l657_65731

open Real

theorem range_of_a (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |2^x₁ - a| = 1 ∧ |2^x₂ - a| = 1) ↔ 1 < a :=
by 
    sorry

end NUMINAMATH_GPT_range_of_a_l657_65731


namespace NUMINAMATH_GPT_fraction_to_decimal_l657_65758

theorem fraction_to_decimal : (3 / 24 : ℚ) = 0.125 := 
by
  -- proof will be filled here
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l657_65758


namespace NUMINAMATH_GPT_original_children_count_l657_65717

theorem original_children_count (x : ℕ) (h1 : 46800 / x + 1950 = 46800 / (x - 2))
    : x = 8 :=
sorry

end NUMINAMATH_GPT_original_children_count_l657_65717


namespace NUMINAMATH_GPT_stone_10th_image_l657_65756

-- Definition of the recursive sequence
def stones (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 1 => stones n + 3 * (n + 1) + 1

-- The statement we need to prove
theorem stone_10th_image : stones 9 = 145 := 
  sorry

end NUMINAMATH_GPT_stone_10th_image_l657_65756


namespace NUMINAMATH_GPT_total_weight_of_dumbbell_system_l657_65757

-- Definitions from the given conditions
def weight_pair1 : ℕ := 3
def weight_pair2 : ℕ := 5
def weight_pair3 : ℕ := 8

-- Goal: Prove that the total weight of the dumbbell system is 32 lbs
theorem total_weight_of_dumbbell_system :
  2 * weight_pair1 + 2 * weight_pair2 + 2 * weight_pair3 = 32 :=
by sorry

end NUMINAMATH_GPT_total_weight_of_dumbbell_system_l657_65757


namespace NUMINAMATH_GPT_cubic_polynomial_roots_l657_65788

noncomputable def cubic_polynomial (a_3 a_2 a_1 a_0 x : ℝ) : ℝ :=
  a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

theorem cubic_polynomial_roots (a_3 a_2 a_1 a_0 : ℝ) 
    (h_nonzero_a3 : a_3 ≠ 0)
    (r1 r2 r3 : ℝ)
    (h_roots : cubic_polynomial a_3 a_2 a_1 a_0 r1 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r2 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r3 = 0)
    (h_condition : (cubic_polynomial a_3 a_2 a_1 a_0 (1/2) 
                    + cubic_polynomial a_3 a_2 a_1 a_0 (-1/2)) 
                    / (cubic_polynomial a_3 a_2 a_1 a_0 0) = 1003) :
  (1 / (r1 * r2) + 1 / (r2 * r3) + 1 / (r3 * r1)) = 2002 :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_l657_65788


namespace NUMINAMATH_GPT_zoo_revenue_l657_65793

def num_children_mon : ℕ := 7
def num_adults_mon : ℕ := 5
def num_children_tue : ℕ := 4
def num_adults_tue : ℕ := 2
def cost_child : ℕ := 3
def cost_adult : ℕ := 4

theorem zoo_revenue : 
  (num_children_mon * cost_child + num_adults_mon * cost_adult) + 
  (num_children_tue * cost_child + num_adults_tue * cost_adult) 
  = 61 := 
by
  sorry

end NUMINAMATH_GPT_zoo_revenue_l657_65793


namespace NUMINAMATH_GPT_not_true_B_l657_65712

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem not_true_B (x y : ℝ) : 2 * star x y ≠ star (2 * x) (2 * y) := by
  sorry

end NUMINAMATH_GPT_not_true_B_l657_65712


namespace NUMINAMATH_GPT_chord_slope_of_ellipse_bisected_by_point_A_l657_65775

theorem chord_slope_of_ellipse_bisected_by_point_A :
  ∀ (P Q : ℝ × ℝ),
  (P.1^2 / 36 + P.2^2 / 9 = 1) ∧ (Q.1^2 / 36 + Q.2^2 / 9 = 1) ∧ 
  ((P.1 + Q.1) / 2 = 1) ∧ ((P.2 + Q.2) / 2 = 1) →
  (Q.2 - P.2) / (Q.1 - P.1) = -1 / 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chord_slope_of_ellipse_bisected_by_point_A_l657_65775


namespace NUMINAMATH_GPT_area_of_smaller_circle_l657_65780

theorem area_of_smaller_circle
  (PA AB : ℝ)
  (r s : ℝ)
  (tangent_at_T : true) -- placeholder; represents the tangency condition
  (common_tangents : true) -- placeholder; represents the external tangents condition
  (PA_eq_AB : PA = AB) :
  PA = 5 →
  AB = 5 →
  r = 2 * s →
  ∃ (s : ℝ) (area : ℝ), s = 5 / (2 * (Real.sqrt 2)) ∧ area = (Real.pi * s^2) ∧ area = (25 * Real.pi) / 8 := by
  intros hPA hAB h_r_s
  use 5 / (2 * (Real.sqrt 2))
  use (Real.pi * (5 / (2 * (Real.sqrt 2)))^2)
  simp [←hPA,←hAB]
  sorry

end NUMINAMATH_GPT_area_of_smaller_circle_l657_65780


namespace NUMINAMATH_GPT_expr1_eval_expr2_eval_l657_65792

theorem expr1_eval : (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (16 / 3) + 3 * Real.sqrt (25 / 3)) = 115 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

theorem expr2_eval : (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (8 / 3) - 3 * Real.sqrt (5 / 3)) = 3 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_expr1_eval_expr2_eval_l657_65792


namespace NUMINAMATH_GPT_original_solution_is_10_percent_l657_65781

def sugar_percentage_original_solution (x : ℕ) :=
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * 42 = 18

theorem original_solution_is_10_percent : sugar_percentage_original_solution 10 :=
by
  unfold sugar_percentage_original_solution
  norm_num

end NUMINAMATH_GPT_original_solution_is_10_percent_l657_65781


namespace NUMINAMATH_GPT_frame_cover_100x100_l657_65705

theorem frame_cover_100x100 :
  ∃! (cover: (ℕ → ℕ → Prop)), (∀ (n : ℕ) (frame: ℕ → ℕ → Prop),
    (∃ (i j : ℕ), (cover (i + n) j ∧ frame (i + n) j ∧ cover (i - n) j ∧ frame (i - n) j) ∧
                   (∃ (k l : ℕ), (cover k (l + n) ∧ frame k (l + n) ∧ cover k (l - n) ∧ frame k (l - n)))) →
    (∃ (i' j' k' l' : ℕ), cover i' j' ∧ frame i' j' ∧ cover k' l' ∧ frame k' l')) :=
sorry

end NUMINAMATH_GPT_frame_cover_100x100_l657_65705


namespace NUMINAMATH_GPT_one_over_a5_eq_30_l657_65740

noncomputable def S : ℕ → ℝ
| n => n / (n + 1)

noncomputable def a (n : ℕ) := if n = 0 then S 0 else S n - S (n - 1)

theorem one_over_a5_eq_30 :
  (1 / a 5) = 30 :=
by
  sorry

end NUMINAMATH_GPT_one_over_a5_eq_30_l657_65740


namespace NUMINAMATH_GPT_inequality_neg_multiplication_l657_65713

theorem inequality_neg_multiplication (m n : ℝ) (h : m > n) : -2 * m < -2 * n :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_neg_multiplication_l657_65713


namespace NUMINAMATH_GPT_factorization_implies_k_l657_65701

theorem factorization_implies_k (x y k : ℝ) (h : ∃ (a b c d e f : ℝ), 
                            x^3 + 3 * x^2 - 2 * x * y - k * x - 4 * y = (a * x + b * y + c) * (d * x^2 + e * xy + f)) :
  k = -2 :=
sorry

end NUMINAMATH_GPT_factorization_implies_k_l657_65701


namespace NUMINAMATH_GPT_prism_volume_l657_65732

/-- The volume of a rectangular prism given the areas of three of its faces. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l657_65732


namespace NUMINAMATH_GPT_find_other_root_l657_65722

theorem find_other_root (k r : ℝ) (h1 : ∀ x : ℝ, 3 * x^2 + k * x + 6 = 0) (h2 : ∃ x : ℝ, 3 * x^2 + k * x + 6 = 0 ∧ x = 3) :
  r = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_other_root_l657_65722


namespace NUMINAMATH_GPT_simplify_expression_l657_65752

theorem simplify_expression (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l657_65752


namespace NUMINAMATH_GPT_total_dots_is_78_l657_65736

-- Define the conditions as Lean definitions
def ladybugs_monday : ℕ := 8
def ladybugs_tuesday : ℕ := 5
def dots_per_ladybug : ℕ := 6

-- Define the total number of ladybugs
def total_ladybugs : ℕ := ladybugs_monday + ladybugs_tuesday

-- Define the total number of dots
def total_dots : ℕ := total_ladybugs * dots_per_ladybug

-- Theorem stating the problem to solve
theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end NUMINAMATH_GPT_total_dots_is_78_l657_65736


namespace NUMINAMATH_GPT_ratio_of_sums_l657_65776

theorem ratio_of_sums (a b c u v w : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
    (h1 : a^2 + b^2 + c^2 = 9) (h2 : u^2 + v^2 + w^2 = 49) (h3 : a * u + b * v + c * w = 21) : 
    (a + b + c) / (u + v + w) = 3 / 7 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l657_65776


namespace NUMINAMATH_GPT_min_even_integers_among_eight_l657_65700

theorem min_even_integers_among_eight :
  ∃ (x y z a b m n o : ℤ), 
    x + y + z = 30 ∧
    x + y + z + a + b = 49 ∧
    x + y + z + a + b + m + n + o = 78 ∧
    (∀ e : ℕ, (∀ x y z a b m n o : ℤ, x + y + z = 30 ∧ x + y + z + a + b = 49 ∧ x + y + z + a + b + m + n + o = 78 → 
    e = 2)) := sorry

end NUMINAMATH_GPT_min_even_integers_among_eight_l657_65700


namespace NUMINAMATH_GPT_largest_circle_area_in_region_S_l657_65744

-- Define the region S
def region_S (x y : ℝ) : Prop :=
  |x + (1 / 2) * y| ≤ 10 ∧ |x| ≤ 10 ∧ |y| ≤ 10

-- The question is to determine the value of k such that the area of the largest circle 
-- centered at (0, 0) fitting inside region S is k * π.
theorem largest_circle_area_in_region_S :
  ∃ k : ℝ, k = 80 :=
sorry

end NUMINAMATH_GPT_largest_circle_area_in_region_S_l657_65744


namespace NUMINAMATH_GPT_walk_to_Lake_Park_restaurant_time_l657_65726

-- Define the problem parameters
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_time_gone : ℕ := 32

-- Define the goal to prove
theorem walk_to_Lake_Park_restaurant_time :
  total_time_gone - (time_to_hidden_lake + time_from_hidden_lake) = 10 :=
by
  -- skipping the proof here
  sorry

end NUMINAMATH_GPT_walk_to_Lake_Park_restaurant_time_l657_65726


namespace NUMINAMATH_GPT_total_packing_peanuts_used_l657_65721

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def large_orders_sent : ℕ := 3
def small_orders_sent : ℕ := 4

theorem total_packing_peanuts_used :
  (large_orders_sent * large_order_weight) + (small_orders_sent * small_order_weight) = 800 := 
by
  sorry

end NUMINAMATH_GPT_total_packing_peanuts_used_l657_65721


namespace NUMINAMATH_GPT_Xiaolong_dad_age_correct_l657_65728
noncomputable def Xiaolong_age (x : ℕ) : ℕ := x
noncomputable def mom_age (x : ℕ) : ℕ := 9 * x
noncomputable def dad_age (x : ℕ) : ℕ := 9 * x + 3
noncomputable def dad_age_next_year (x : ℕ) : ℕ := 9 * x + 4
noncomputable def Xiaolong_age_next_year (x : ℕ) : ℕ := x + 1
noncomputable def dad_age_predicated_next_year (x : ℕ) : ℕ := 8 * (x + 1)

theorem Xiaolong_dad_age_correct (x : ℕ) (h : 9 * x + 4 = 8 * (x + 1)) : dad_age x = 39 := by
  sorry

end NUMINAMATH_GPT_Xiaolong_dad_age_correct_l657_65728


namespace NUMINAMATH_GPT_g_x_even_l657_65772

theorem g_x_even (a b c : ℝ) (g : ℝ → ℝ):
  (∀ x, g x = a * x^6 + b * x^4 - c * x^2 + 5)
  → g 32 = 3
  → g 32 + g (-32) = 6 :=
by
  sorry

end NUMINAMATH_GPT_g_x_even_l657_65772


namespace NUMINAMATH_GPT_phone_extension_permutations_l657_65718

theorem phone_extension_permutations : 
  (∃ (l : List ℕ), l = [5, 7, 8, 9, 0] ∧ Nat.factorial l.length = 120) :=
sorry

end NUMINAMATH_GPT_phone_extension_permutations_l657_65718


namespace NUMINAMATH_GPT_Vasya_mushrooms_l657_65768

def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digitsSum (n : ℕ) : ℕ := (n / 100) + ((n % 100) / 10) + (n % 10)

theorem Vasya_mushrooms :
  ∃ n : ℕ, isThreeDigit n ∧ digitsSum n = 14 ∧ n = 950 := 
by
  sorry

end NUMINAMATH_GPT_Vasya_mushrooms_l657_65768


namespace NUMINAMATH_GPT_Jane_saves_five_dollars_l657_65725

noncomputable def first_pair_cost : ℝ := 50
noncomputable def second_pair_cost_A : ℝ := first_pair_cost * 0.6
noncomputable def second_pair_cost_B : ℝ := first_pair_cost - 15
noncomputable def promotion_A_total_cost : ℝ := first_pair_cost + second_pair_cost_A
noncomputable def promotion_B_total_cost : ℝ := first_pair_cost + second_pair_cost_B
noncomputable def Jane_savings : ℝ := promotion_B_total_cost - promotion_A_total_cost

theorem Jane_saves_five_dollars : Jane_savings = 5 := by
  sorry

end NUMINAMATH_GPT_Jane_saves_five_dollars_l657_65725


namespace NUMINAMATH_GPT_triangle_reciprocal_sum_l657_65774

variables {A B C D L M N : Type} -- Points are types
variables {t_1 t_2 t_3 t_4 t_5 t_6 : ℝ} -- Areas are real numbers

-- Assume conditions as hypotheses
variable (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
variable (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
variable (h3 : ∀ (t1 t5 t3 t4 : ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5))

theorem triangle_reciprocal_sum 
  (h1 : ∀ (t1 t4 t5 t6: ℝ), t_1 = t1 ∧ t_4 = t4 ∧ t_5 = t5 ∧ t_6 = t6 -> (t1 + t4) = (t5 + t6))
  (h2 : ∀ (t2 t4 t3 t6: ℝ), t_2 = t2 ∧ t_4 = t4 ∧ t_3 = t3 ∧ t_6 = t6 -> (t2 + t4) = (t3 + t6))
  (h3 : ∀ (t1 t5 t3 t4: ℝ), t_1 = t1 ∧ t_5 = t5 ∧ t_3 = t3 ∧ t_4 = t4 -> (t1 + t3) = (t4 + t5)) :
  (1 / t_1 + 1 / t_3 + 1 / t_5) = (1 / t_2 + 1 / t_4 + 1 / t_6) :=
sorry

end NUMINAMATH_GPT_triangle_reciprocal_sum_l657_65774


namespace NUMINAMATH_GPT_person_speed_approx_l657_65719

noncomputable def convertDistance (meters : ℝ) : ℝ := meters * 0.000621371
noncomputable def convertTime (minutes : ℝ) (seconds : ℝ) : ℝ := (minutes + (seconds / 60)) / 60
noncomputable def calculateSpeed (distance_miles : ℝ) (time_hours : ℝ) : ℝ := distance_miles / time_hours

theorem person_speed_approx (street_length_meters : ℝ) (time_min : ℝ) (time_sec : ℝ) :
  street_length_meters = 900 →
  time_min = 3 →
  time_sec = 20 →
  abs ((calculateSpeed (convertDistance street_length_meters) (convertTime time_min time_sec)) - 10.07) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_person_speed_approx_l657_65719


namespace NUMINAMATH_GPT_Andy_late_minutes_l657_65785

theorem Andy_late_minutes 
  (school_start : Nat := 8*60) -- 8:00 AM in minutes since midnight
  (normal_travel_time : Nat := 30) -- 30 minutes
  (red_light_stops : Nat := 3 * 4) -- 3 minutes each at 4 lights
  (construction_wait : Nat := 10) -- 10 minutes
  (detour_time : Nat := 7) -- 7 minutes
  (store_stop_time : Nat := 5) -- 5 minutes
  (traffic_delay : Nat := 15) -- 15 minutes
  (departure_time : Nat := 7*60 + 15) -- 7:15 AM in minutes since midnight
  : 34 = departure_time + normal_travel_time + red_light_stops + construction_wait + detour_time + store_stop_time + traffic_delay - school_start := 
by sorry

end NUMINAMATH_GPT_Andy_late_minutes_l657_65785


namespace NUMINAMATH_GPT_barker_high_school_team_count_l657_65764

theorem barker_high_school_team_count (students_total : ℕ) (baseball_team : ℕ) (hockey_team : ℕ) 
  (both_sports : ℕ) : 
  students_total = 36 → baseball_team = 25 → hockey_team = 19 → both_sports = (baseball_team + hockey_team - students_total) → both_sports = 8 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_barker_high_school_team_count_l657_65764


namespace NUMINAMATH_GPT_elizabeth_spendings_elizabeth_savings_l657_65782

section WeddingGift

def steak_knife_set_cost : ℝ := 80
def steak_knife_sets : ℕ := 2
def dinnerware_set_cost : ℝ := 200
def fancy_napkins_sets : ℕ := 3
def fancy_napkins_total_cost : ℝ := 45
def wine_glasses_cost : ℝ := 100
def discount_steak_dinnerware : ℝ := 0.10
def discount_napkins : ℝ := 0.20
def sales_tax : ℝ := 0.05

def total_cost_before_discounts : ℝ :=
  (steak_knife_sets * steak_knife_set_cost) + dinnerware_set_cost + fancy_napkins_total_cost + wine_glasses_cost

def total_discount : ℝ :=
  ((steak_knife_sets * steak_knife_set_cost) * discount_steak_dinnerware) + (dinnerware_set_cost * discount_steak_dinnerware) + (fancy_napkins_total_cost * discount_napkins)

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discount

def total_cost_with_tax : ℝ :=
  total_cost_after_discounts + (total_cost_after_discounts * sales_tax)

def savings : ℝ :=
  total_cost_before_discounts - total_cost_after_discounts

theorem elizabeth_spendings :
  total_cost_with_tax = 558.60 :=
by sorry

theorem elizabeth_savings :
  savings = 63 :=
by sorry

end WeddingGift

end NUMINAMATH_GPT_elizabeth_spendings_elizabeth_savings_l657_65782


namespace NUMINAMATH_GPT_number_of_math_players_l657_65702

theorem number_of_math_players (total_players physics_players both_players : ℕ)
    (h1 : total_players = 25)
    (h2 : physics_players = 15)
    (h3 : both_players = 6)
    (h4 : total_players = physics_players + (total_players - physics_players - (total_players - physics_players - both_players)) + both_players ) :
  total_players - (physics_players - both_players) = 16 :=
sorry

end NUMINAMATH_GPT_number_of_math_players_l657_65702


namespace NUMINAMATH_GPT_line_equation_parallel_to_x_axis_through_point_l657_65769

-- Define the point (3, -2)
def point : ℝ × ℝ := (3, -2)

-- Define a predicate for a line being parallel to the X-axis
def is_parallel_to_x_axis (line : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, line x k

-- Define the equation of the line passing through the given point
def equation_of_line_through_point (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- State the theorem to be proved
theorem line_equation_parallel_to_x_axis_through_point :
  ∀ (line : ℝ → ℝ → Prop), 
    (equation_of_line_through_point point line) → (is_parallel_to_x_axis line) → (∀ x, line x (-2)) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_parallel_to_x_axis_through_point_l657_65769


namespace NUMINAMATH_GPT_partner_profit_share_correct_l657_65798

-- Definitions based on conditions
def total_profit : ℝ := 280000
def profit_share_shekhar : ℝ := 0.28
def profit_share_rajeev : ℝ := 0.22
def profit_share_jatin : ℝ := 0.20
def profit_share_simran : ℝ := 0.18
def profit_share_ramesh : ℝ := 0.12

-- Each partner's share in the profit
def shekhar_share : ℝ := profit_share_shekhar * total_profit
def rajeev_share : ℝ := profit_share_rajeev * total_profit
def jatin_share : ℝ := profit_share_jatin * total_profit
def simran_share : ℝ := profit_share_simran * total_profit
def ramesh_share : ℝ := profit_share_ramesh * total_profit

-- Statement to be proved
theorem partner_profit_share_correct :
    shekhar_share = 78400 ∧ 
    rajeev_share = 61600 ∧ 
    jatin_share = 56000 ∧ 
    simran_share = 50400 ∧ 
    ramesh_share = 33600 ∧ 
    (shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit) :=
by sorry

end NUMINAMATH_GPT_partner_profit_share_correct_l657_65798


namespace NUMINAMATH_GPT_mickys_sticks_more_l657_65750

theorem mickys_sticks_more 
  (simons_sticks : ℕ := 36)
  (gerrys_sticks : ℕ := (2 * simons_sticks) / 3)
  (total_sticks_needed : ℕ := 129)
  (total_simons_and_gerrys_sticks : ℕ := simons_sticks + gerrys_sticks)
  (mickys_sticks : ℕ := total_sticks_needed - total_simons_and_gerrys_sticks) :
  mickys_sticks - total_simons_and_gerrys_sticks = 9 :=
by
  sorry

end NUMINAMATH_GPT_mickys_sticks_more_l657_65750


namespace NUMINAMATH_GPT_train_speed_l657_65796

variable (length : ℕ) (time : ℕ)
variable (h_length : length = 120)
variable (h_time : time = 6)

theorem train_speed (length time : ℕ) (h_length : length = 120) (h_time : time = 6) :
  length / time = 20 := by
  sorry

end NUMINAMATH_GPT_train_speed_l657_65796


namespace NUMINAMATH_GPT_sum_of_three_numbers_l657_65784

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35) 
  (h2 : b + c = 57) 
  (h3 : c + a = 62) : 
  a + b + c = 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l657_65784


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l657_65716

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : a > 0) :
  ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) :=
by
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l657_65716


namespace NUMINAMATH_GPT_range_of_a_l657_65795

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l657_65795


namespace NUMINAMATH_GPT_find_picture_area_l657_65799

variable (x y : ℕ)
    (h1 : x > 1)
    (h2 : y > 1)
    (h3 : (3 * x + 2) * (y + 4) - x * y = 62)

theorem find_picture_area : x * y = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_picture_area_l657_65799


namespace NUMINAMATH_GPT_sum_of_17th_roots_of_unity_except_1_l657_65733

theorem sum_of_17th_roots_of_unity_except_1 :
  Complex.exp (2 * Real.pi * Complex.I / 17) +
  Complex.exp (4 * Real.pi * Complex.I / 17) +
  Complex.exp (6 * Real.pi * Complex.I / 17) +
  Complex.exp (8 * Real.pi * Complex.I / 17) +
  Complex.exp (10 * Real.pi * Complex.I / 17) +
  Complex.exp (12 * Real.pi * Complex.I / 17) +
  Complex.exp (14 * Real.pi * Complex.I / 17) +
  Complex.exp (16 * Real.pi * Complex.I / 17) +
  Complex.exp (18 * Real.pi * Complex.I / 17) +
  Complex.exp (20 * Real.pi * Complex.I / 17) +
  Complex.exp (22 * Real.pi * Complex.I / 17) +
  Complex.exp (24 * Real.pi * Complex.I / 17) +
  Complex.exp (26 * Real.pi * Complex.I / 17) +
  Complex.exp (28 * Real.pi * Complex.I / 17) +
  Complex.exp (30 * Real.pi * Complex.I / 17) +
  Complex.exp (32 * Real.pi * Complex.I / 17) = 0 := sorry

end NUMINAMATH_GPT_sum_of_17th_roots_of_unity_except_1_l657_65733


namespace NUMINAMATH_GPT_mean_noon_temperature_l657_65771

def temperatures : List ℕ := [82, 80, 83, 88, 90, 92, 90, 95]

def mean_temperature (temps : List ℕ) : ℚ :=
  (temps.foldr (λ a b => a + b) 0 : ℚ) / temps.length

theorem mean_noon_temperature :
  mean_temperature temperatures = 87.5 := by
  sorry

end NUMINAMATH_GPT_mean_noon_temperature_l657_65771


namespace NUMINAMATH_GPT_total_students_l657_65763

-- Define the conditions
def students_in_front : Nat := 7
def position_from_back : Nat := 6

-- Define the proof problem
theorem total_students : (students_in_front + 1 + (position_from_back - 1)) = 13 := by
  -- Proof steps will go here (use sorry to skip for now)
  sorry

end NUMINAMATH_GPT_total_students_l657_65763


namespace NUMINAMATH_GPT_find_a_l657_65714

theorem find_a (a : ℝ) (h1 : 1 < a) (h2 : 1 + a = 3) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l657_65714


namespace NUMINAMATH_GPT_problem_solution_l657_65766

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * n - 1

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_c (n : ℕ) : ℕ :=
  sequence_a n * 2 ^ (sequence_a n)

def sum_T (n : ℕ) : ℕ :=
  (6 * n - 5) * 2 ^ (2 * n + 1) + 10

theorem problem_solution (n : ℕ) (hn : n ≥ 1) :
  ∀ n, (sum_S 1 = 1) ∧ (sequence_a 1 = 1) ∧ 
          (∀ n ≥ 2, sequence_a n = 2 * n - 1) ∧
          (sum_T n = (6 * n - 5) * 2 ^ (2 * n + 1) + 10 / 9) :=
by sorry

end NUMINAMATH_GPT_problem_solution_l657_65766


namespace NUMINAMATH_GPT_find_Q_over_P_l657_65730

theorem find_Q_over_P (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -7 → x ≠ 0 → x ≠ 5 →
    (P / (x + 7 : ℝ) + Q / (x^2 - 6 * x) = (x^2 - 6 * x + 14) / (x^3 + x^2 - 30 * x))) :
  Q / P = 12 :=
  sorry

end NUMINAMATH_GPT_find_Q_over_P_l657_65730


namespace NUMINAMATH_GPT_focus_of_parabola_l657_65751

theorem focus_of_parabola (h : ∀ y x, y^2 = 8 * x ↔ ∃ p, y^2 = 4 * p * x ∧ p = 2): (2, 0) ∈ {f | ∃ x y, y^2 = 8 * x ∧ f = (p, 0)} :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l657_65751


namespace NUMINAMATH_GPT_bisection_contains_root_l657_65778

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem bisection_contains_root : (1 < 1.5) ∧ f 1 < 0 ∧ f 1.5 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_bisection_contains_root_l657_65778


namespace NUMINAMATH_GPT_length_of_train_l657_65755

-- Define the conditions as variables
def speed : ℝ := 39.27272727272727
def time : ℝ := 55
def length_bridge : ℝ := 480

-- Calculate the total distance using the given conditions
def total_distance : ℝ := speed * time

-- Prove that the length of the train is 1680 meters
theorem length_of_train :
  (total_distance - length_bridge) = 1680 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l657_65755


namespace NUMINAMATH_GPT_complete_square_eq_l657_65779

theorem complete_square_eq (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_eq_l657_65779


namespace NUMINAMATH_GPT_solve_inequality_l657_65759

theorem solve_inequality (a x : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) :
  ((0 ≤ a ∧ a < 1/2 → a < x ∧ x < 1 - a) ∧ 
   (a = 1/2 → false) ∧ 
   (1/2 < a ∧ a ≤ 1 → 1 - a < x ∧ x < a)) ↔ (x - a) * (x + a - 1) < 0 := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l657_65759


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l657_65753

noncomputable def algebraic_expression (x : ℝ) : ℝ :=
  (x / (x^2 + 2*x + 1) - 1 / (2*x + 2)) / ((x - 1) / (4*x + 4))

noncomputable def substitution_value : ℝ :=
  2 * Real.cos (Real.pi / 4) - 1

theorem algebraic_expression_evaluation :
  algebraic_expression substitution_value = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l657_65753


namespace NUMINAMATH_GPT_find_n_solution_l657_65727

theorem find_n_solution (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_solution_l657_65727
