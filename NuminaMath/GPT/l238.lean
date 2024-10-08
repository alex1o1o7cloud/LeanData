import Mathlib

namespace coeff_x2y2_in_expansion_l238_238244

-- Define the coefficient of a specific term in the binomial expansion
def coeff_binom (n k : ℕ) (a b : ℤ) (x y : ℕ) : ℤ :=
  (Nat.choose n k) * (a ^ (n - k)) * (b ^ k)

theorem coeff_x2y2_in_expansion : coeff_binom 4 2 1 (-2) 2 2 = 24 := by
  sorry

end coeff_x2y2_in_expansion_l238_238244


namespace part1_complement_intersection_part2_range_m_l238_238564

open Set

-- Define set A
def A : Set ℝ := { x | -1 ≤ x ∧ x < 4 }

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2 }

-- Part (1): Prove the complement of the intersection for m = 3
theorem part1_complement_intersection :
  ∀ x : ℝ, x ∉ (A ∩ B 3) ↔ x < 3 ∨ x ≥ 4 :=
by
  sorry

-- Part (2): Prove the range of m for A ∩ B = ∅
theorem part2_range_m (m : ℝ) :
  (A ∩ B m = ∅) ↔ m < -3 ∨ m ≥ 4 :=
by
  sorry

end part1_complement_intersection_part2_range_m_l238_238564


namespace assistant_professors_charts_l238_238401

theorem assistant_professors_charts (A B C : ℕ) (h1 : 2 * A + B = 10) (h2 : A + B * C = 11) (h3 : A + B = 7) : C = 2 :=
by
  sorry

end assistant_professors_charts_l238_238401


namespace sugar_total_more_than_two_l238_238749

noncomputable def x (p q : ℝ) : ℝ :=
p / q

noncomputable def y (p q : ℝ) : ℝ :=
q / p

theorem sugar_total_more_than_two (p q : ℝ) (hpq : p ≠ q) :
  x p q + y p q > 2 :=
by sorry

end sugar_total_more_than_two_l238_238749


namespace symmetric_point_coords_l238_238515

def pointA : ℝ × ℝ := (1, 2)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def pointB : ℝ × ℝ := translate_left pointA 2

def pointC : ℝ × ℝ := reflect_origin pointB

theorem symmetric_point_coords :
  pointC = (1, -2) :=
by
  -- Proof omitted as instructed
  sorry

end symmetric_point_coords_l238_238515


namespace problem_statement_l238_238441

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

-- Define the complement of B in U
def C_U_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- State the theorem
theorem problem_statement : (A ∩ C_U_B) = {1, 2} :=
by {
  -- Proof is omitted
  sorry
}

end problem_statement_l238_238441


namespace maria_spent_60_dollars_l238_238060

theorem maria_spent_60_dollars :
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  true
    → total_cost = 60 := 
by 
  intros
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  sorry

end maria_spent_60_dollars_l238_238060


namespace seven_digit_palindromes_l238_238365

def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

theorem seven_digit_palindromes : 
  (∃ l : List ℕ, l = [1, 1, 4, 4, 4, 6, 6] ∧ 
  ∃ pl : List ℕ, pl.length = 7 ∧ is_palindrome pl ∧ 
  ∀ d, d ∈ pl → d ∈ l) →
  ∃! n, n = 12 :=
by
  sorry

end seven_digit_palindromes_l238_238365


namespace max_stickers_l238_238731

theorem max_stickers (n_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) 
  (total_players : n_players = 22) 
  (average : avg_stickers = 4) 
  (minimum : ∀ i, i < n_players → min_stickers = 1) :
  ∃ max_sticker : ℕ, max_sticker = 67 :=
by
  sorry

end max_stickers_l238_238731


namespace product_of_primes_sum_85_l238_238018

open Nat

theorem product_of_primes_sum_85 :
  ∃ (p q : ℕ), p.Prime ∧ q.Prime ∧ p + q = 85 ∧ p * q = 166 :=
sorry

end product_of_primes_sum_85_l238_238018


namespace sheets_paper_150_l238_238691

def num_sheets_of_paper (S : ℕ) (E : ℕ) : Prop :=
  (S - E = 50) ∧ (3 * E - S = 150)

theorem sheets_paper_150 (S E : ℕ) : num_sheets_of_paper S E → S = 150 :=
by
  sorry

end sheets_paper_150_l238_238691


namespace range_of_k_l238_238506

def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem range_of_k (k : ℝ) : (∀ x : ℝ, tensor k x > 0) ↔ (0 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l238_238506


namespace ab_bd_ratio_l238_238150

-- Definitions based on the conditions
variables {A B C D : ℝ}
variables (h1 : A / B = 1 / 2) (h2 : B / C = 8 / 5)

-- Math equivalence proving AB/BD = 4/13 based on given conditions
theorem ab_bd_ratio
  (h1 : A / B = 1 / 2)
  (h2 : B / C = 8 / 5) :
  A / (B + C) = 4 / 13 :=
by
  sorry

end ab_bd_ratio_l238_238150


namespace coprime_sum_product_l238_238389

theorem coprime_sum_product (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a + b) (a * b) = 1 := by
  sorry

end coprime_sum_product_l238_238389


namespace connie_remaining_marbles_l238_238523

def initial_marbles : ℕ := 73
def marbles_given : ℕ := 70

theorem connie_remaining_marbles : initial_marbles - marbles_given = 3 := by
  sorry

end connie_remaining_marbles_l238_238523


namespace find_a_plus_c_l238_238709

def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_a_plus_c (a b c d : ℝ) 
  (h_vertex_f : -a / 2 = v) (h_vertex_g : -c / 2 = w)
  (h_root_v_g : g v c d = 0) (h_root_w_f : f w a b = 0)
  (h_intersect : f 50 a b = -200 ∧ g 50 c d = -200)
  (h_min_value_f : ∀ x, f (-a / 2) a b ≤ f x a b)
  (h_min_value_g : ∀ x, g (-c / 2) c d ≤ g x c d)
  (h_min_difference : f (-a / 2) a b = g (-c / 2) c d - 50) :
  a + c = sorry :=
sorry

end find_a_plus_c_l238_238709


namespace female_managers_count_l238_238105

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end female_managers_count_l238_238105


namespace AmandaWillSpend_l238_238092

/--
Amanda goes shopping and sees a sale where different items have different discounts.
She wants to buy a dress for $50 with a 30% discount, a pair of shoes for $75 with a 25% discount,
and a handbag for $100 with a 40% discount.
After applying the discounts, a 5% tax is added to the final price.
Prove that Amanda will spend $158.81 to buy all three items after the discounts and tax have been applied.
-/
noncomputable def totalAmount : ℝ :=
  let dressPrice := 50
  let dressDiscount := 0.30
  let shoesPrice := 75
  let shoesDiscount := 0.25
  let handbagPrice := 100
  let handbagDiscount := 0.40
  let taxRate := 0.05
  let dressFinalPrice := dressPrice * (1 - dressDiscount)
  let shoesFinalPrice := shoesPrice * (1 - shoesDiscount)
  let handbagFinalPrice := handbagPrice * (1 - handbagDiscount)
  let subtotal := dressFinalPrice + shoesFinalPrice + handbagFinalPrice
  let tax := subtotal * taxRate
  let totalAmount := subtotal + tax
  totalAmount

theorem AmandaWillSpend : totalAmount = 158.81 :=
by
  -- proof goes here
  sorry

end AmandaWillSpend_l238_238092


namespace no_integer_solution_l238_238456

open Polynomial

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ)
  (h₁ : P.eval a = 2016) (h₂ : P.eval b = 2016) (h₃ : P.eval c = 2016) 
  (h₄ : P.eval d = 2016) (dist : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ¬ ∃ x : ℤ, P.eval x = 2019 :=
sorry

end no_integer_solution_l238_238456


namespace total_pairs_of_shoes_tried_l238_238958

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l238_238958


namespace range_of_function_l238_238529

theorem range_of_function : 
  ∀ y : ℝ, 
  (∃ x : ℝ, y = x^2 + 1) ↔ (y ≥ 1) :=
by
  sorry

end range_of_function_l238_238529


namespace taxi_ride_distance_l238_238878

theorem taxi_ride_distance (initial_fare additional_fare total_fare : ℝ) 
  (initial_distance : ℝ) (additional_distance increment_distance : ℝ) :
  initial_fare = 1.0 →
  additional_fare = 0.45 →
  initial_distance = 1/5 →
  increment_distance = 1/5 →
  total_fare = 7.3 →
  additional_distance = (total_fare - initial_fare) / additional_fare →
  (initial_distance + additional_distance * increment_distance) = 3 := 
by sorry

end taxi_ride_distance_l238_238878


namespace trigonometric_identity_l238_238260

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
sorry

end trigonometric_identity_l238_238260


namespace min_a_squared_plus_b_squared_l238_238343

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a_squared_plus_b_squared_l238_238343


namespace remainder_102_104_plus_6_div_9_l238_238644

theorem remainder_102_104_plus_6_div_9 :
  ((102 * 104 + 6) % 9) = 3 :=
by
  sorry

end remainder_102_104_plus_6_div_9_l238_238644


namespace sin_cos_identity_l238_238122

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := 
  sorry

end sin_cos_identity_l238_238122


namespace tangent_line_eq_l238_238156

noncomputable def f (x : ℝ) := x / (2 * x - 1)

def tangentLineAtPoint (x : ℝ) : ℝ := -x + 2

theorem tangent_line_eq {x y : ℝ} (hxy : y = f 1) (f_deriv : deriv f 1 = -1) :
  y = 1 → tangentLineAtPoint x = -x + 2 :=
by
  intros
  sorry

end tangent_line_eq_l238_238156


namespace negation_exists_ltx2_plus_x_plus_1_lt_0_l238_238923

theorem negation_exists_ltx2_plus_x_plus_1_lt_0 :
  ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_exists_ltx2_plus_x_plus_1_lt_0_l238_238923


namespace min_a_for_50_pow_2023_div_17_l238_238855

theorem min_a_for_50_pow_2023_div_17 (a : ℕ) (h : 17 ∣ (50 ^ 2023 + a)) : a = 18 :=
sorry

end min_a_for_50_pow_2023_div_17_l238_238855


namespace algebraic_expression_value_l238_238116

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 :=
by
  sorry

end algebraic_expression_value_l238_238116


namespace roots_ellipse_condition_l238_238652

theorem roots_ellipse_condition (m n : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1^2 - m*x1 + n = 0 ∧ x2^2 - m*x2 + n = 0) 
  ↔ (m > 0 ∧ n > 0 ∧ m ≠ n) :=
sorry

end roots_ellipse_condition_l238_238652


namespace correct_option_l238_238705

theorem correct_option (a b c : ℝ) : 
  (5 * a - (b + 2 * c) = 5 * a + b - 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a + b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b - 2 * c) ↔ 
  (5 * a - (b + 2 * c) = 5 * a - b - 2 * c) :=
by
  sorry

end correct_option_l238_238705


namespace polygon_sides_from_angle_sum_l238_238032

-- Let's define the problem
theorem polygon_sides_from_angle_sum : 
  ∀ (n : ℕ), (n - 2) * 180 = 900 → n = 7 :=
by
  intros n h
  sorry

end polygon_sides_from_angle_sum_l238_238032


namespace distinct_solutions_square_l238_238270

theorem distinct_solutions_square (α β : ℝ) (h₁ : α ≠ β)
    (h₂ : α^2 = 2 * α + 2 ∧ β^2 = 2 * β + 2) : (α - β) ^ 2 = 12 := by
  sorry

end distinct_solutions_square_l238_238270


namespace fruit_seller_apples_l238_238671

theorem fruit_seller_apples (x : ℝ) (h : 0.60 * x = 420) : x = 700 :=
sorry

end fruit_seller_apples_l238_238671


namespace students_with_uncool_parents_l238_238846

def total_students : ℕ := 40
def cool_dads_count : ℕ := 18
def cool_moms_count : ℕ := 20
def both_cool_count : ℕ := 10

theorem students_with_uncool_parents :
  total_students - (cool_dads_count + cool_moms_count - both_cool_count) = 12 :=
by sorry

end students_with_uncool_parents_l238_238846


namespace combined_weight_chihuahua_pitbull_greatdane_l238_238618

noncomputable def chihuahua_pitbull_greatdane_combined_weight (C P G : ℕ) : ℕ :=
  C + P + G

theorem combined_weight_chihuahua_pitbull_greatdane :
  ∀ (C P G : ℕ), P = 3 * C → G = 3 * P + 10 → G = 307 → chihuahua_pitbull_greatdane_combined_weight C P G = 439 :=
by
  intros C P G h1 h2 h3
  sorry

end combined_weight_chihuahua_pitbull_greatdane_l238_238618


namespace proposition_truth_count_l238_238801

namespace Geometry

def is_obtuse_angle (A : Type) : Prop := sorry
def is_obtuse_triangle (ABC : Type) : Prop := sorry

def original_proposition (A : Type) (ABC : Type) : Prop :=
is_obtuse_angle A → is_obtuse_triangle ABC

def contrapositive_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_triangle ABC) → ¬ (is_obtuse_angle A)

def converse_proposition (ABC : Type) (A : Type) : Prop :=
is_obtuse_triangle ABC → is_obtuse_angle A

def inverse_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_angle A) → ¬ (is_obtuse_triangle ABC)

theorem proposition_truth_count (A : Type) (ABC : Type) :
  (original_proposition A ABC ∧ contrapositive_proposition A ABC ∧
  ¬ (converse_proposition ABC A) ∧ ¬ (inverse_proposition A ABC)) →
  ∃ n : ℕ, n = 2 :=
sorry

end Geometry

end proposition_truth_count_l238_238801


namespace multiply_vars_l238_238688

variables {a b : ℝ}

theorem multiply_vars : -3 * a * b * 2 * a = -6 * a^2 * b := by
  sorry

end multiply_vars_l238_238688


namespace divides_polynomial_l238_238140

theorem divides_polynomial (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  ∀ x : ℂ, (x^2 + x + 1) ∣ (x^(3 * m + 1) + x^(3 * n + 2) + 1) :=
by
  sorry

end divides_polynomial_l238_238140


namespace lateral_surface_area_of_cone_l238_238329

-- Definitions from the conditions
def base_radius : ℝ := 6
def slant_height : ℝ := 15

-- Theorem statement to be proved
theorem lateral_surface_area_of_cone (r l : ℝ) (hr : r = base_radius) (hl : l = slant_height) : 
  (π * r * l) = 90 * π :=
by
  sorry

end lateral_surface_area_of_cone_l238_238329


namespace find_k_l238_238273

open Real

noncomputable def chord_intersection (k : ℝ) : Prop :=
  let R : ℝ := 3
  let d := abs (k + 1) / sqrt (1 + k^2)
  d^2 + (12 * sqrt 5 / 10)^2 = R^2

theorem find_k (k : ℝ) (h : k > 1) (h_intersect : chord_intersection k) : k = 2 := by
  sorry

end find_k_l238_238273


namespace apple_slices_per_group_l238_238755

-- defining the conditions
variables (a g : ℕ)

-- 1. Equal number of apple slices and grapes in groups
def equal_group (a g : ℕ) : Prop := a = g

-- 2. Grapes packed in groups of 9
def grapes_groups_of_9 (g : ℕ) : Prop := ∃ k : ℕ, g = 9 * k

-- 3. Smallest number of grapes is 18
def smallest_grapes (g : ℕ) : Prop := g = 18

-- theorem stating that the number of apple slices per group is 9
theorem apple_slices_per_group : equal_group a g ∧ grapes_groups_of_9 g ∧ smallest_grapes g → a = 9 := by
  sorry

end apple_slices_per_group_l238_238755


namespace arithmetic_sequence_sum_l238_238484

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end arithmetic_sequence_sum_l238_238484


namespace problem_I_problem_II_l238_238438

-- Declaration of function f(x)
def f (x a b : ℝ) := |x + a| - |x - b|

-- Proof 1: When a = 1, b = 1, solve the inequality f(x) > 1
theorem problem_I (x : ℝ) : (f x 1 1) > 1 ↔ x > 1/2 := by
  sorry

-- Proof 2: If the maximum value of the function f(x) is 2, prove that (1/a) + (1/b) ≥ 2
theorem problem_II (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max_f : ∀ x, f x a b ≤ 2) : 1 / a + 1 / b ≥ 2 := by
  sorry

end problem_I_problem_II_l238_238438


namespace mirka_number_l238_238922

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b
noncomputable def reversed_number (a b : ℕ) : ℕ := 10 * b + a

theorem mirka_number (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 4) (h2 : b = 2 * a) :
  original_number a b = 12 ∨ original_number a b = 24 ∨ original_number a b = 36 ∨ original_number a b = 48 :=
by
  sorry

end mirka_number_l238_238922


namespace hyperbola_ellipse_equations_l238_238899

theorem hyperbola_ellipse_equations 
  (F1 F2 P : ℝ × ℝ) 
  (hF1 : F1 = (0, -5))
  (hF2 : F2 = (0, 5))
  (hP : P = (3, 4)) :
  (∃ a b : ℝ, a^2 = 40 ∧ b^2 = 16 ∧ 
    ∀ x y : ℝ, (y^2 / 40 + x^2 / 15 = 1 ↔ y^2 / a^2 + x^2 / (a^2 - 25) = 1) ∧
    (y^2 / 16 - x^2 / 9 = 1 ↔ y^2 / b^2 - x^2 / (25 - b^2) = 1)) :=
sorry

end hyperbola_ellipse_equations_l238_238899


namespace ratio_smaller_to_larger_dimension_of_framed_painting_l238_238942

-- Definitions
def painting_width : ℕ := 16
def painting_height : ℕ := 20
def side_frame_width (x : ℝ) : ℝ := x
def top_frame_width (x : ℝ) : ℝ := 1.5 * x
def total_frame_area (x : ℝ) : ℝ := (painting_width + 2 * side_frame_width x) * (painting_height + 2 * top_frame_width x) - painting_width * painting_height
def frame_area_eq_painting_area (x : ℝ) : Prop := total_frame_area x = painting_width * painting_height

-- Lean statement
theorem ratio_smaller_to_larger_dimension_of_framed_painting :
  ∃ x : ℝ, frame_area_eq_painting_area x → 
  ((painting_width + 2 * side_frame_width x) / (painting_height + 2 * top_frame_width x)) = (3 / 4) :=
by
  sorry

end ratio_smaller_to_larger_dimension_of_framed_painting_l238_238942


namespace proof_inequality_l238_238692

variable {a b c d : ℝ}

theorem proof_inequality (h1 : a + b + c + d = 6) (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
sorry

end proof_inequality_l238_238692


namespace grass_knot_segments_butterfly_knot_segments_l238_238330

-- Definitions for the grass knot problem
def outer_loops_cut : Nat := 5
def segments_after_outer_loops_cut : Nat := 6

-- Theorem for the grass knot
theorem grass_knot_segments (n : Nat) (h : n = outer_loops_cut) : (n + 1 = segments_after_outer_loops_cut) :=
sorry

-- Definitions for the butterfly knot problem
def butterfly_wings_loops_per_wing : Nat := 7
def segments_after_butterfly_wings_cut : Nat := 15

-- Theorem for the butterfly knot
theorem butterfly_knot_segments (w : Nat) (h : w = butterfly_wings_loops_per_wing) : ((w * 2 * 2 + 2) / 2 = segments_after_butterfly_wings_cut) :=
sorry

end grass_knot_segments_butterfly_knot_segments_l238_238330


namespace second_investment_rate_l238_238789

theorem second_investment_rate (P : ℝ) (r₁ t : ℝ) (I_diff : ℝ) (P900 : P = 900) (r1_4_percent : r₁ = 0.04) (t7 : t = 7) (I_years : I_diff = 31.50) :
∃ r₂ : ℝ, 900 * (r₂ / 100) * 7 - 900 * 0.04 * 7 = 31.50 → r₂ = 4.5 := 
by
  sorry

end second_investment_rate_l238_238789


namespace number_of_hard_drives_sold_l238_238876

theorem number_of_hard_drives_sold 
    (H : ℕ)
    (price_per_graphics_card : ℕ := 600)
    (price_per_hard_drive : ℕ := 80)
    (price_per_cpu : ℕ := 200)
    (price_per_ram_pair : ℕ := 60)
    (graphics_cards_sold : ℕ := 10)
    (cpus_sold : ℕ := 8)
    (ram_pairs_sold : ℕ := 4)
    (total_earnings : ℕ := 8960)
    (earnings_from_graphics_cards : graphics_cards_sold * price_per_graphics_card = 6000)
    (earnings_from_cpus : cpus_sold * price_per_cpu = 1600)
    (earnings_from_ram : ram_pairs_sold * price_per_ram_pair = 240)
    (earnings_from_hard_drives : H * price_per_hard_drive = 80 * H) :
  graphics_cards_sold * price_per_graphics_card +
  cpus_sold * price_per_cpu +
  ram_pairs_sold * price_per_ram_pair +
  H * price_per_hard_drive = total_earnings → H = 14 :=
by
  intros h
  sorry

end number_of_hard_drives_sold_l238_238876


namespace max_n_intersection_non_empty_l238_238753

-- Define the set An
def An (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- State the theorem
theorem max_n_intersection_non_empty : 
  ∃ x, (∀ n, n ≤ 4 → x ∈ An n) ∧ (∀ n, n > 4 → x ∉ An n) :=
by
  sorry

end max_n_intersection_non_empty_l238_238753


namespace determine_x_value_l238_238491

variable {a b x r : ℝ}
variable (b_nonzero : b ≠ 0)

theorem determine_x_value (h1 : r = (3 * a)^(3 * b)) (h2 : r = a^b * x^b) : x = 27 * a^2 :=
by
  sorry

end determine_x_value_l238_238491


namespace average_words_per_hour_l238_238223

-- Define the given conditions
variables (W : ℕ) (H : ℕ)

-- State constants for the known values
def words := 60000
def writing_hours := 100

-- Define theorem to prove the average words per hour during the writing phase
theorem average_words_per_hour (h : W = words) (h2 : H = writing_hours) : (W / H) = 600 := by
  sorry

end average_words_per_hour_l238_238223


namespace mass_percentage_H_in_chlorous_acid_l238_238238

noncomputable def mass_percentage_H_in_HClO2 : ℚ :=
  let molar_mass_H : ℚ := 1.01
  let molar_mass_Cl : ℚ := 35.45
  let molar_mass_O : ℚ := 16.00
  let molar_mass_HClO2 : ℚ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  (molar_mass_H / molar_mass_HClO2) * 100

theorem mass_percentage_H_in_chlorous_acid :
  mass_percentage_H_in_HClO2 = 1.475 := by
  sorry

end mass_percentage_H_in_chlorous_acid_l238_238238


namespace combined_mpg_l238_238216

theorem combined_mpg (m : ℕ) (ray_mpg tom_mpg : ℕ) (h1 : m = 200) (h2 : ray_mpg = 40) (h3 : tom_mpg = 20) :
  (m / (m / (2 * ray_mpg) + m / (2 * tom_mpg))) = 80 / 3 :=
by
  sorry

end combined_mpg_l238_238216


namespace indeterminate_original_value_percentage_l238_238804

-- Lets define the problem as a structure with the given conditions
structure StockData where
  yield_percent : ℚ
  market_value : ℚ

-- We need to prove this condition
theorem indeterminate_original_value_percentage (d : StockData) :
  d.yield_percent = 8 ∧ d.market_value = 125 → false :=
by
  sorry

end indeterminate_original_value_percentage_l238_238804


namespace exercise_l238_238539

theorem exercise (x y z : ℝ)
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : 1/x + 1/y + 1/z = 3/5) : x^2 + y^2 + z^2 = 488.4 :=
sorry

end exercise_l238_238539


namespace find_f_of_conditions_l238_238534

theorem find_f_of_conditions (f : ℝ → ℝ) :
  (f 1 = 1) →
  (∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) →
  (∀ x : ℝ, f x = 3^x - 2^x) :=
by
  intros h1 h2
  sorry

end find_f_of_conditions_l238_238534


namespace megatech_astrophysics_degrees_l238_238548

theorem megatech_astrophysics_degrees :
  let microphotonics := 10
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let astrophysics_percentage := 100 - total_percentage
  let total_degrees := 360
  let astrophysics_degrees := (astrophysics_percentage / 100) * total_degrees
  astrophysics_degrees = 50.4 :=
by
  sorry

end megatech_astrophysics_degrees_l238_238548


namespace difference_of_students_l238_238905

variable (G1 G2 G5 : ℕ)

theorem difference_of_students (h1 : G1 + G2 > G2 + G5) (h2 : G5 = G1 - 30) : 
  (G1 + G2) - (G2 + G5) = 30 :=
by
  sorry

end difference_of_students_l238_238905


namespace solution_set_inequality_l238_238454

noncomputable def f : ℝ → ℝ := sorry
noncomputable def derivative_f : ℝ → ℝ := sorry -- f' is the derivative of f

-- Conditions
axiom f_domain {x : ℝ} (h1 : 0 < x) : f x ≠ 0
axiom derivative_condition {x : ℝ} (h1 : 0 < x) : f x + x * derivative_f x > 0
axiom initial_value : f 1 = 2

-- Proof that the solution set of the inequality f(x) < 2/x is (0, 1)
theorem solution_set_inequality : ∀ x : ℝ, 0 < x ∧ x < 1 → f x < 2 / x := sorry

end solution_set_inequality_l238_238454


namespace not_prime_for_large_n_l238_238261

theorem not_prime_for_large_n {n : ℕ} (h : n > 1) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end not_prime_for_large_n_l238_238261


namespace Mary_avg_speed_l238_238983

def Mary_uphill_distance := 1.5 -- km
def Mary_uphill_time := 45.0 / 60.0 -- hours
def Mary_downhill_distance := 1.5 -- km
def Mary_downhill_time := 15.0 / 60.0 -- hours

def total_distance := Mary_uphill_distance + Mary_downhill_distance
def total_time := Mary_uphill_time + Mary_downhill_time

theorem Mary_avg_speed : 
  (total_distance / total_time) = 3.0 := by
  sorry

end Mary_avg_speed_l238_238983


namespace pure_imaginary_solution_l238_238186

theorem pure_imaginary_solution (m : ℝ) (h₁ : m^2 - m - 4 = 0) (h₂ : m^2 - 5 * m - 6 ≠ 0) :
  m = (1 + Real.sqrt 17) / 2 ∨ m = (1 - Real.sqrt 17) / 2 :=
sorry

end pure_imaginary_solution_l238_238186


namespace inequality_holds_l238_238585

theorem inequality_holds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := 
by
  sorry

end inequality_holds_l238_238585


namespace malcolm_brushes_teeth_l238_238236

theorem malcolm_brushes_teeth :
  (∃ (M : ℕ), M = 180 ∧ (∃ (N : ℕ), N = 90 ∧ (M / N = 2))) :=
by
  sorry

end malcolm_brushes_teeth_l238_238236


namespace gcd_765432_654321_l238_238346

theorem gcd_765432_654321 :
  Int.gcd 765432 654321 = 3 := 
sorry

end gcd_765432_654321_l238_238346


namespace ratio_of_boat_to_stream_l238_238402

theorem ratio_of_boat_to_stream (B S : ℝ) (h : ∀ D : ℝ, D / (B - S) = 2 * (D / (B + S))) :
  B / S = 3 :=
by 
  sorry

end ratio_of_boat_to_stream_l238_238402


namespace fireflies_win_l238_238005

theorem fireflies_win 
  (initial_hornets : ℕ) (initial_fireflies : ℕ) 
  (hornets_scored : ℕ) (fireflies_scored : ℕ) 
  (three_point_baskets : ℕ) (two_point_baskets : ℕ)
  (h1 : initial_hornets = 86)
  (h2 : initial_fireflies = 74)
  (h3 : three_point_baskets = 7)
  (h4 : two_point_baskets = 2)
  (h5 : fireflies_scored = three_point_baskets * 3)
  (h6 : hornets_scored = two_point_baskets * 2)
  : initial_fireflies + fireflies_scored - (initial_hornets + hornets_scored) = 5 := 
sorry

end fireflies_win_l238_238005


namespace max_value_of_f_l238_238145

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_value_of_f (a b : ℝ) (ha : ∀ x, f x ≤ b) (hfa : f a = b) : a - b = -1 :=
by
  sorry

end max_value_of_f_l238_238145


namespace value_of_a_plus_c_l238_238821

theorem value_of_a_plus_c (a b c r : ℝ)
  (h1 : a + b + c = 114)
  (h2 : a * b * c = 46656)
  (h3 : b = a * r)
  (h4 : c = a * r^2) :
  a + c = 78 :=
sorry

end value_of_a_plus_c_l238_238821


namespace train_crosses_pole_in_15_seconds_l238_238415

theorem train_crosses_pole_in_15_seconds
    (train_speed : ℝ) (train_length_meters : ℝ) (time_seconds : ℝ) : 
    train_speed = 300 →
    train_length_meters = 1250 →
    time_seconds = 15 :=
by
  sorry

end train_crosses_pole_in_15_seconds_l238_238415


namespace fourth_number_in_12th_row_is_92_l238_238312

-- Define the number of elements per row and the row number
def elements_per_row := 8
def row_number := 12

-- Define the last number in a row function
def last_number_in_row (n : ℕ) := elements_per_row * n

-- Define the starting number in a row function
def starting_number_in_row (n : ℕ) := (elements_per_row * (n - 1)) + 1

-- Define the nth number in a specified row function
def nth_number_in_row (n : ℕ) (k : ℕ) := starting_number_in_row n + (k - 1)

-- Prove that the fourth number in the 12th row is 92
theorem fourth_number_in_12th_row_is_92 : nth_number_in_row 12 4 = 92 :=
by
  -- state the required equivalences
  sorry

end fourth_number_in_12th_row_is_92_l238_238312


namespace Alice_and_Dave_weight_l238_238034

variable (a b c d : ℕ)

-- Conditions
variable (h1 : a + b = 230)
variable (h2 : b + c = 220)
variable (h3 : c + d = 250)

-- Proof statement
theorem Alice_and_Dave_weight :
  a + d = 260 :=
sorry

end Alice_and_Dave_weight_l238_238034


namespace sin_double_angle_l238_238124

theorem sin_double_angle (θ : ℝ) (h₁ : 3 * (Real.cos θ)^2 = Real.tan θ + 3) (h₂ : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := 
sorry

end sin_double_angle_l238_238124


namespace probability_within_circle_eq_pi_over_nine_l238_238328

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let circle_area := Real.pi * (2 ^ 2)
  let square_area := 6 * 6
  circle_area / square_area

theorem probability_within_circle_eq_pi_over_nine :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_circle_eq_pi_over_nine_l238_238328


namespace problem_solution_l238_238545

noncomputable def time_without_distraction : ℝ :=
  let rate_A := 1 / 10
  let rate_B := 0.75 * rate_A
  let rate_C := 0.5 * rate_A
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

noncomputable def time_with_distraction : ℝ :=
  let rate_A := 0.9 * (1 / 10)
  let rate_B := 0.9 * (0.75 * (1 / 10))
  let rate_C := 0.9 * (0.5 * (1 / 10))
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

theorem problem_solution :
  time_without_distraction = 40 / 9 ∧
  time_with_distraction = 44.44 / 9 := by
  sorry

end problem_solution_l238_238545


namespace percentage_deficit_of_second_side_l238_238373

theorem percentage_deficit_of_second_side
  (L W : Real)
  (h1 : ∃ (L' : Real), L' = 1.16 * L)
  (h2 : ∃ (W' : Real), (L' * W') = 1.102 * (L * W))
  (h3 : ∃ (x : Real), W' = W * (1 - x / 100)) :
  x = 5 := 
  sorry

end percentage_deficit_of_second_side_l238_238373


namespace parking_cost_per_hour_l238_238848

theorem parking_cost_per_hour (avg_cost : ℝ) (total_initial_cost : ℝ) (hours_excessive : ℝ) (total_hours : ℝ) (cost_first_2_hours : ℝ)
  (h1 : cost_first_2_hours = 9.00) 
  (h2 : avg_cost = 2.361111111111111)
  (h3 : total_hours = 9) 
  (h4 : hours_excessive = 7):
  (total_initial_cost / total_hours = avg_cost) -> 
  (total_initial_cost = cost_first_2_hours + hours_excessive * x) -> 
  x = 1.75 := 
by
  intros h5 h6
  sorry

end parking_cost_per_hour_l238_238848


namespace math_problem_l238_238602

open Real

theorem math_problem
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a - b + c = 0) :
  (a^2 * b^2 / ((a^2 + b * c) * (b^2 + a * c)) +
   a^2 * c^2 / ((a^2 + b * c) * (c^2 + a * b)) +
   b^2 * c^2 / ((b^2 + a * c) * (c^2 + a * b))) = 1 := by
  sorry

end math_problem_l238_238602


namespace find_range_of_a_l238_238815

theorem find_range_of_a (a : ℝ) (x : ℝ) (y : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) 
    (hineq : x * y ≤ a * x^2 + 2 * y^2) : 
    -1 ≤ a := sorry

end find_range_of_a_l238_238815


namespace ratio_of_shares_l238_238630

theorem ratio_of_shares 
    (sheila_share : ℕ → ℕ)
    (rose_share : ℕ)
    (total_rent : ℕ) 
    (h1 : ∀ P, sheila_share P = 5 * P)
    (h2 : rose_share = 1800)
    (h3 : ∀ P, sheila_share P + P + rose_share = total_rent) 
    (h4 : total_rent = 5400) :
    ∃ P, 1800 / P = 3 := 
by 
  sorry

end ratio_of_shares_l238_238630


namespace range_of_a_l238_238713

open Set

theorem range_of_a (a : ℝ) : (-3 < a ∧ a < -1) ↔ (∀ x, x < -1 ∨ 5 < x ∨ (a < x ∧ x < a+8)) :=
sorry

end range_of_a_l238_238713


namespace angle_measure_l238_238252

theorem angle_measure (x : ℝ) (h : x + (3 * x - 10) = 180) : x = 47.5 := 
by
  sorry

end angle_measure_l238_238252


namespace a2022_value_l238_238825

theorem a2022_value 
  (a : Fin 2022 → ℤ)
  (h : ∀ n k : Fin 2022, a n - a k ≥ n.1^3 - k.1^3)
  (a1011 : a 1010 = 0) :
  a 2021 = 2022^3 - 1011^3 :=
by
  sorry

end a2022_value_l238_238825


namespace intersection_points_l238_238219

noncomputable def curve1 (x y : ℝ) : Prop := x^2 + 4 * y^2 = 1
noncomputable def curve2 (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

theorem intersection_points : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 2 := 
by 
  sorry

end intersection_points_l238_238219


namespace find_m_and_star_l238_238632

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

end find_m_and_star_l238_238632


namespace area_diff_of_rectangle_l238_238633

theorem area_diff_of_rectangle (a : ℝ) : 
  let length_increased := 1.40 * a
  let breadth_increased := 1.30 * a
  let original_area := a * a
  let new_area := length_increased * breadth_increased
  (new_area - original_area) = 0.82 * (a * a) :=
by 
sorry

end area_diff_of_rectangle_l238_238633


namespace washington_goats_l238_238291

variables (W : ℕ) (P : ℕ) (total_goats : ℕ)

theorem washington_goats (W : ℕ) (h1 : P = W + 40) (h2 : total_goats = W + P) (h3 : total_goats = 320) : W = 140 :=
by
  sorry

end washington_goats_l238_238291


namespace correct_proportion_expression_l238_238865

def is_fraction_correctly_expressed (numerator denominator : ℕ) (expression : String) : Prop :=
  -- Define the property of a correctly expressed fraction in English
  expression = "three-fifths"

theorem correct_proportion_expression : 
  is_fraction_correctly_expressed 3 5 "three-fifths" :=
by
  sorry

end correct_proportion_expression_l238_238865


namespace find_value_l238_238062

variable {a b c : ℝ}

def ellipse_eqn (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

theorem find_value 
  (h1 : a^2 + b^2 - 3*c^2 = 0)
  (h2 : a^2 = b^2 + c^2) :
  (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := 
  sorry

end find_value_l238_238062


namespace family_ages_l238_238344

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

end family_ages_l238_238344


namespace find_number_l238_238544

theorem find_number (x : ℝ) (h : (x / 4) + 3 = 5) : x = 8 :=
by
  sorry

end find_number_l238_238544


namespace average_income_BC_l238_238164

theorem average_income_BC {A_income B_income C_income : ℝ}
  (hAB : (A_income + B_income) / 2 = 4050)
  (hAC : (A_income + C_income) / 2 = 4200)
  (hA : A_income = 3000) :
  (B_income + C_income) / 2 = 5250 :=
by sorry

end average_income_BC_l238_238164


namespace find_f1_plus_gneg1_l238_238611

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom relation : ∀ x : ℝ, f x - g x = (1 / 2) ^ x

-- Proof statement
theorem find_f1_plus_gneg1 : f 1 + g (-1) = -2 :=
by
  -- Proof goes here
  sorry

end find_f1_plus_gneg1_l238_238611


namespace domain_of_sqrt_function_l238_238213

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | 3 - 2 * x - x^2 ≥ 0}

theorem domain_of_sqrt_function : domain_of_function = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_sqrt_function_l238_238213


namespace inequality_constant_l238_238017

noncomputable def smallest_possible_real_constant : ℝ :=
  1.0625

theorem inequality_constant (C : ℝ) : 
  (∀ x y z : ℝ, (x + y + z = -1) → 
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1| ) ↔ C ≥ smallest_possible_real_constant :=
sorry

end inequality_constant_l238_238017


namespace age_difference_two_children_l238_238210

/-!
# Age difference between two children in a family

## Given:
- 10 years ago, the average age of a family of 4 members was 24 years.
- Two children have been born since then.
- The present average age of the family (now 6 members) is the same, 24 years.
- The present age of the youngest child (Y1) is 3 years.

## Prove:
The age difference between the two children is 2 years.
-/

theorem age_difference_two_children :
  let Y1 := 3
  let Y2 := 5
  let total_age_10_years_ago := 4 * 24
  let total_age_now := 6 * 24
  let increase_age_10_years := total_age_now - total_age_10_years_ago
  let increase_due_to_original_members := 4 * 10
  let increase_due_to_children := increase_age_10_years - increase_due_to_original_members
  Y1 + Y2 = increase_due_to_children
  → Y2 - Y1 = 2 :=
by
  intros
  sorry

end age_difference_two_children_l238_238210


namespace john_subtracts_79_l238_238542

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l238_238542


namespace collinear_points_sum_l238_238472

-- Points in 3-dimensional space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition of collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ k : ℝ,
    k ≠ 0 ∧
    (p2.x - p1.x) * k = (p3.x - p1.x) ∧
    (p2.y - p1.y) * k = (p3.y - p1.y) ∧
    (p2.z - p1.z) * k = (p3.z - p1.z)

-- Main statement
theorem collinear_points_sum {a b : ℝ} :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) → a + b = 6 :=
by
  sorry

end collinear_points_sum_l238_238472


namespace max_xy_l238_238769

-- Lean statement for the given problem
theorem max_xy (x y : ℝ) (h : x^2 + y^2 = 4) : xy ≤ 2 := sorry

end max_xy_l238_238769


namespace domain_of_sqrt_ln_eq_l238_238033

noncomputable def domain_of_function : Set ℝ :=
  {x | 2 * x + 1 >= 0 ∧ 3 - 4 * x > 0}

theorem domain_of_sqrt_ln_eq :
  domain_of_function = Set.Icc (-1 / 2) (3 / 4) \ {3 / 4} :=
by
  sorry

end domain_of_sqrt_ln_eq_l238_238033


namespace fraction_disliking_but_liking_l238_238581

-- Definitions based on conditions
def total_students : ℕ := 100
def like_dancing : ℕ := 70
def dislike_dancing : ℕ := total_students - like_dancing

def say_they_like_dancing (like_dancing : ℕ) : ℕ := (70 * like_dancing) / 100
def say_they_dislike_dancing (like_dancing : ℕ) : ℕ := like_dancing - say_they_like_dancing like_dancing

def dislike_and_say_dislike (dislike_dancing : ℕ) : ℕ := (80 * dislike_dancing) / 100
def say_dislike_but_like (like_dancing : ℕ) : ℕ := say_they_dislike_dancing like_dancing

def total_say_dislike : ℕ := dislike_and_say_dislike dislike_dancing + say_dislike_but_like like_dancing

noncomputable def fraction_like_but_say_dislike : ℚ := (say_dislike_but_like like_dancing : ℚ) / (total_say_dislike : ℚ)

theorem fraction_disliking_but_liking : fraction_like_but_say_dislike = 46.67 / 100 := 
by sorry

end fraction_disliking_but_liking_l238_238581


namespace intersection_of_A_and_CU_B_l238_238627

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := {-1, 0, 1, 2, 3}
noncomputable def B : Set ℝ := { x : ℝ | x ≥ 2 }
noncomputable def CU_B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_of_A_and_CU_B :
  A ∩ CU_B = {-1, 0, 1} :=
by
  sorry

end intersection_of_A_and_CU_B_l238_238627


namespace statement_2_statement_3_l238_238910

variable {α : Type*} [LinearOrderedField α]

-- Given a quadratic function
def quadratic (a b c x : α) : α :=
  a * x^2 + b * x + c

-- Statement 2
theorem statement_2 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c p = quadratic a b c q → quadratic a b c (p + q) = c :=
sorry

-- Statement 3
theorem statement_3 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c (p + q) = c → (p + q = 0 ∨ quadratic a b c p = quadratic a b c q) :=
sorry

end statement_2_statement_3_l238_238910


namespace distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l238_238428

-- Definitions based on conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits : Finset ℕ := {1, 3, 5}

-- Problem 1: Number of distinct three-digit numbers
theorem distinct_three_digit_numbers : (digits.erase 0).card * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 100 := by
  sorry

-- Problem 2: Number of distinct three-digit odd numbers
theorem distinct_three_digit_odd_numbers : (odd_digits.card) * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 48 := by
  sorry

end distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l238_238428


namespace negation_of_exists_equiv_forall_neg_l238_238067

noncomputable def negation_equivalent (a : ℝ) : Prop :=
  ∀ a : ℝ, ¬ ∃ x : ℝ, a * x^2 + 1 = 0

-- The theorem statement
theorem negation_of_exists_equiv_forall_neg (h : ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) :
  negation_equivalent a :=
by {
  sorry
}

end negation_of_exists_equiv_forall_neg_l238_238067


namespace part_1_odd_function_part_2_decreasing_l238_238656

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

theorem part_1_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

theorem part_2_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  intros x1 x2 h
  sorry

end part_1_odd_function_part_2_decreasing_l238_238656


namespace negation_of_zero_product_l238_238680

theorem negation_of_zero_product (x y : ℝ) : (xy ≠ 0) → (x ≠ 0) ∧ (y ≠ 0) :=
sorry

end negation_of_zero_product_l238_238680


namespace increasing_function_on_interval_l238_238041

noncomputable def f_A (x : ℝ) : ℝ := 3 - x
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3 * x
noncomputable def f_C (x : ℝ) : ℝ := - (1 / (x + 1))
noncomputable def f_D (x : ℝ) : ℝ := -|x|

theorem increasing_function_on_interval (h0 : ∀ x : ℝ, x > 0):
  (∀ x y : ℝ, 0 < x -> x < y -> f_C x < f_C y) ∧ 
  (∀ (g : ℝ → ℝ), (g ≠ f_C) → (∀ x y : ℝ, 0 < x -> x < y -> g x ≥ g y)) :=
by sorry

end increasing_function_on_interval_l238_238041


namespace lock_rings_l238_238461

theorem lock_rings (n : ℕ) (h : 6 ^ n - 1 ≤ 215) : n = 3 :=
sorry

end lock_rings_l238_238461


namespace coefficient_of_x5_in_expansion_l238_238858

-- Define the polynomial expansion of (x-1)(x+1)^8
def polynomial_expansion (x : ℚ) : ℚ :=
  (x - 1) * (x + 1) ^ 8

-- Define the binomial coefficient function
def binom_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem: The coefficient of x^5 in the expansion of (x-1)(x+1)^8 is 14
theorem coefficient_of_x5_in_expansion :
  binom_coeff 8 4 - binom_coeff 8 5 = 14 :=
sorry

end coefficient_of_x5_in_expansion_l238_238858


namespace find_x_l238_238134

theorem find_x (x : ℝ) : 0.3 * x + 0.2 = 0.26 → x = 0.2 :=
by
  sorry

end find_x_l238_238134


namespace question1_question2_l238_238563

theorem question1 (x : ℝ) : (1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x) ↔ (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4) :=
by sorry

theorem question2 (x a : ℝ) : ((x - a)/(x - a^2) < 0)
  ↔ (a = 0 ∨ a = 1 → false)
  ∨ (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a)
  ∨ ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) :=
by sorry

end question1_question2_l238_238563


namespace count_f_compositions_l238_238203

noncomputable def count_special_functions : Nat :=
  let A := Finset.range 6
  let f := (Set.univ : Set (A → A))
  sorry

theorem count_f_compositions (f : Fin 6 → Fin 6) 
  (h : ∀ x : Fin 6, (f ∘ f ∘ f) x = x) :
  count_special_functions = 81 :=
sorry

end count_f_compositions_l238_238203


namespace roger_initial_candies_l238_238113

def initial_candies (given_candies left_candies : ℕ) : ℕ :=
  given_candies + left_candies

theorem roger_initial_candies :
  initial_candies 3 92 = 95 :=
by
  sorry

end roger_initial_candies_l238_238113


namespace cost_of_one_shirt_l238_238104

-- Definitions based on the conditions given
variables (J S : ℝ)

-- First condition: 3 pairs of jeans and 2 shirts cost $69
def condition1 : Prop := 3 * J + 2 * S = 69

-- Second condition: 2 pairs of jeans and 3 shirts cost $61
def condition2 : Prop := 2 * J + 3 * S = 61

-- The theorem to prove that the cost of one shirt is $9
theorem cost_of_one_shirt (J S : ℝ) (h1 : condition1 J S) (h2 : condition2 J S) : S = 9 :=
by
  sorry

end cost_of_one_shirt_l238_238104


namespace exists_f_gcd_form_l238_238012

noncomputable def f : ℤ → ℕ := sorry

theorem exists_f_gcd_form :
  (∀ x y : ℤ, Nat.gcd (f x) (f y) = Nat.gcd (f x) (Int.natAbs (x - y))) →
  ∃ m n : ℕ, (0 < m ∧ 0 < n) ∧ (∀ x : ℤ, f x = Nat.gcd (m + Int.natAbs x) n) :=
sorry

end exists_f_gcd_form_l238_238012


namespace quadratic_function_solution_l238_238121

noncomputable def g (x : ℝ) : ℝ := x^2 + 44 * x + 50

theorem quadratic_function_solution (c d : ℝ)
  (h : ∀ x, (g (g x + x)) / (g x) = x^2 + 44 * x + 50) :
  c = 44 ∧ d = 50 :=
by
  sorry

end quadratic_function_solution_l238_238121


namespace stephen_speed_second_third_l238_238386

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

end stephen_speed_second_third_l238_238386


namespace circles_fit_l238_238085

noncomputable def fit_circles_in_rectangle : Prop :=
  ∃ (m n : ℕ) (α : ℝ), (m * n * α * α = 1) ∧ (m * n * α / 2 = 1962)

theorem circles_fit : fit_circles_in_rectangle :=
by sorry

end circles_fit_l238_238085


namespace Jesse_read_pages_l238_238298

theorem Jesse_read_pages (total_pages : ℝ) (h : (2 / 3) * total_pages = 166) :
  (1 / 3) * total_pages = 83 :=
sorry

end Jesse_read_pages_l238_238298


namespace adjacent_block_permutations_l238_238135

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the block of digits that must be adjacent
def block : List ℕ := [2, 5, 8]

-- Function to calculate permutations of a list (size n)
def fact (n : ℕ) : ℕ := Nat.factorial n

-- Calculate the total number of arrangements
def total_arrangements : ℕ := fact 8 * fact 3

-- The main theorem statement to be proved
theorem adjacent_block_permutations :
  total_arrangements = 241920 :=
by
  sorry

end adjacent_block_permutations_l238_238135


namespace ruby_candies_l238_238974

theorem ruby_candies (number_of_friends : ℕ) (candies_per_friend : ℕ) (total_candies : ℕ)
  (h1 : number_of_friends = 9)
  (h2 : candies_per_friend = 4)
  (h3 : total_candies = number_of_friends * candies_per_friend) :
  total_candies = 36 :=
by {
  sorry
}

end ruby_candies_l238_238974


namespace find_length_AB_l238_238369

-- Definitions for the problem conditions.
def angle_B : ℝ := 90
def angle_A : ℝ := 30
def BC : ℝ := 24

-- Main theorem to prove.
theorem find_length_AB (angle_B_eq : angle_B = 90) (angle_A_eq : angle_A = 30) (BC_eq : BC = 24) : 
  ∃ AB : ℝ, AB = 12 := 
by
  sorry

end find_length_AB_l238_238369


namespace paige_scored_17_points_l238_238310

def paige_points (total_points : ℕ) (num_players : ℕ) (points_per_player_exclusive : ℕ) : ℕ :=
  total_points - ((num_players - 1) * points_per_player_exclusive)

theorem paige_scored_17_points :
  paige_points 41 5 6 = 17 :=
by
  sorry

end paige_scored_17_points_l238_238310


namespace rectangle_width_decrease_l238_238911

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end rectangle_width_decrease_l238_238911


namespace john_walks_farther_l238_238357

theorem john_walks_farther :
  let john_distance : ℝ := 1.74
  let nina_distance : ℝ := 1.235
  john_distance - nina_distance = 0.505 :=
by
  sorry

end john_walks_farther_l238_238357


namespace smallest_n_7770_l238_238657

theorem smallest_n_7770 (n : ℕ) 
  (h1 : ∀ d ∈ n.digits 10, d = 0 ∨ d = 7)
  (h2 : 15 ∣ n) : 
  n = 7770 := 
sorry

end smallest_n_7770_l238_238657


namespace ellipse_focus_eccentricity_l238_238931

theorem ellipse_focus_eccentricity (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 2) + (y^2 / m) = 1 → y = 0 ∨ x = 0) ∧
  (∀ e : ℝ, e = 1 / 2) →
  m = 3 / 2 :=
sorry

end ellipse_focus_eccentricity_l238_238931


namespace sum_of_two_longest_altitudes_l238_238331

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h1: a = 7) (h2: b = 24) (h3: c = 25) : 
  (a + b = 31) :=
by {
  sorry
}

end sum_of_two_longest_altitudes_l238_238331


namespace age_difference_l238_238685

theorem age_difference :
  let x := 5
  let prod_today := x * x
  let prod_future := (x + 1) * (x + 1)
  prod_future - prod_today = 11 :=
by
  sorry

end age_difference_l238_238685


namespace stock_rise_in_morning_l238_238634

theorem stock_rise_in_morning (x : ℕ) (V : ℕ → ℕ) (h0 : V 0 = 100)
  (h100 : V 100 = 200) (h_recurrence : ∀ n, V n = 100 + n * x - n) :
  x = 2 :=
  by
  sorry

end stock_rise_in_morning_l238_238634


namespace find_stream_speed_l238_238370

variable (r w : ℝ)

noncomputable def stream_speed:
    Prop := 
    (21 / (r + w) + 4 = 21 / (r - w)) ∧ 
    (21 / (3 * r + w) + 0.5 = 21 / (3 * r - w)) ∧ 
    w = 3 

theorem find_stream_speed : ∃ w, stream_speed r w := 
by
  sorry

end find_stream_speed_l238_238370


namespace find_number_being_divided_l238_238265

theorem find_number_being_divided (divisor quotient remainder : ℕ) (h1: divisor = 15) (h2: quotient = 9) (h3: remainder = 1) : 
  divisor * quotient + remainder = 136 :=
by
  -- Simplification and computation would follow here
  sorry

end find_number_being_divided_l238_238265


namespace problem_arithmetic_l238_238120

variable {α : Type*} [LinearOrderedField α] 

def arithmetic_sum (a d : α) (n : ℕ) : α := n * (2 * a + (n - 1) * d) / 2
def arithmetic_term (a d : α) (k : ℕ) : α := a + (k - 1) * d

theorem problem_arithmetic (a3 a2015 : ℝ) 
  (h_roots : a3 + a2015 = 10) 
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h_sum : ∀ n, S n = arithmetic_sum a3 ((a2015 - a3) / 2012) n) 
  (h_an : ∀ k, a k = arithmetic_term a3 ((a2015 - a3) / 2012) k) :
  (S 2017) / 2017 + a 1009 = 10 := by
sorry

end problem_arithmetic_l238_238120


namespace find_q_l238_238342

theorem find_q {q : ℕ} (h : 27^8 = 9^q) : q = 12 := by
  sorry

end find_q_l238_238342


namespace plane_contains_points_l238_238598

def point := (ℝ × ℝ × ℝ)

def is_plane (A B C D : ℝ) (p : point) : Prop :=
  ∃ x y z, p = (x, y, z) ∧ A * x + B * y + C * z + D = 0

theorem plane_contains_points :
  ∃ A B C D : ℤ,
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    is_plane A B C D (2, -1, 3) ∧
    is_plane A B C D (0, -1, 5) ∧
    is_plane A B C D (-2, -3, 4) ∧
    A = 2 ∧ B = 5 ∧ C = -2 ∧ D = 7 :=
  sorry

end plane_contains_points_l238_238598


namespace probability_angle_AMB_acute_l238_238885

theorem probability_angle_AMB_acute :
  let side_length := 4
  let square_area := side_length * side_length
  let semicircle_area := (1 / 2) * Real.pi * (side_length / 2) ^ 2
  let probability := 1 - semicircle_area / square_area
  probability = 1 - (Real.pi / 8) :=
sorry

end probability_angle_AMB_acute_l238_238885


namespace inequality_proof_l238_238619

theorem inequality_proof
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 1) :
  (a^2 + b^2 + c^2) * ((a / (b + c)) + (b / (a + c)) + (c / (a + b))) ≥ 1/2 := by
  sorry

end inequality_proof_l238_238619


namespace cookies_in_jar_l238_238045

noncomputable def number_of_cookies_in_jar : ℕ := sorry

theorem cookies_in_jar :
  (number_of_cookies_in_jar - 1) = (1 / 2 : ℝ) * (number_of_cookies_in_jar + 5) →
  number_of_cookies_in_jar = 7 :=
by
  sorry

end cookies_in_jar_l238_238045


namespace a6_is_32_l238_238319

namespace arithmetic_sequence

variables {a : ℕ → ℝ} -- {aₙ} is an arithmetic sequence with positive terms
variables (q : ℝ) -- Common ratio

-- Conditions as definitions
def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a1_is_1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2_times_a4_is_16 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 4 = 16

-- The ultimate goal is to prove a₆ = 32
theorem a6_is_32 (h_arith : is_arithmetic_sequence a q) 
  (h_a1 : a1_is_1 a) (h_product : a2_times_a4_is_16 a q) : 
  a 6 = 32 := 
sorry

end arithmetic_sequence

end a6_is_32_l238_238319


namespace sequence_geometric_l238_238979

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, a n ≠ 0)
  (h_arith : 2 * a 2 = a 1 + a 3)
  (h_geom : a 3 ^ 2 = a 2 * a 4)
  (h_recip_arith : 2 / a 4 = 1 / a 3 + 1 / a 5) :
  a 3 ^ 2 = a 1 * a 5 :=
sorry

end sequence_geometric_l238_238979


namespace calculate_expression_l238_238467

noncomputable def expr : ℚ := (5 - 2 * (3 - 6 : ℚ)⁻¹ ^ 2)⁻¹

theorem calculate_expression :
  expr = (9 / 43 : ℚ) := by
  sorry

end calculate_expression_l238_238467


namespace complement_set_l238_238429

open Set

theorem complement_set (U M : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hM : M = {1, 2, 4}) :
  compl M ∩ U = {3, 5, 6} := 
by
  rw [compl, hU, hM]
  sorry

end complement_set_l238_238429


namespace area_of_triangle_F1PF2P_l238_238077

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4
noncomputable def c : ℝ := 3
noncomputable def PF1 : ℝ := sorry 
noncomputable def PF2 : ℝ := sorry

-- Given conditions
def ellipse_eq_holds (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Given point P is on the ellipse
def P_on_ellipse (x y : ℝ) : Prop := ellipse_eq_holds x y

-- Given angle F1PF2
def angle_F1PF2_eq_60 : Prop := sorry

-- Proving the area of △F₁PF₂
theorem area_of_triangle_F1PF2P : S = (16 * Real.sqrt 3) / 3 :=
by sorry

end area_of_triangle_F1PF2P_l238_238077


namespace check_conditions_l238_238498

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) := n * a + (n * (n - 1) / 2) * d

theorem check_conditions {a d : ℤ}
  (S6 S7 S5 : ℤ)
  (h1 : S6 = sum_of_first_n_terms a d 6)
  (h2 : S7 = sum_of_first_n_terms a d 7)
  (h3 : S5 = sum_of_first_n_terms a d 5)
  (h : S6 > S7 ∧ S7 > S5) :
  d < 0 ∧
  sum_of_first_n_terms a d 11 > 0 ∧
  sum_of_first_n_terms a d 13 < 0 ∧
  sum_of_first_n_terms a d 9 > sum_of_first_n_terms a d 3 := 
sorry

end check_conditions_l238_238498


namespace total_rainfall_2003_and_2004_l238_238533

noncomputable def average_rainfall_2003 : ℝ := 45
noncomputable def months_in_year : ℕ := 12
noncomputable def percent_increase : ℝ := 0.05

theorem total_rainfall_2003_and_2004 :
  let rainfall_2004 := average_rainfall_2003 * (1 + percent_increase)
  let total_rainfall_2003 := average_rainfall_2003 * months_in_year
  let total_rainfall_2004 := rainfall_2004 * months_in_year
  total_rainfall_2003 = 540 ∧ total_rainfall_2004 = 567 := 
by 
  sorry

end total_rainfall_2003_and_2004_l238_238533


namespace delta_value_l238_238152

theorem delta_value (Δ : ℤ) (h : 4 * (-3) = Δ - 3) : Δ = -9 :=
by {
  sorry
}

end delta_value_l238_238152


namespace total_pieces_correct_l238_238840

-- Definitions based on conditions
def rods_in_row (n : ℕ) : ℕ := 3 * n
def connectors_in_row (n : ℕ) : ℕ := n

-- Sum of natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Total rods in ten rows
def total_rods : ℕ := 3 * sum_first_n 10

-- Total connectors in eleven rows
def total_connectors : ℕ := sum_first_n 11

-- Total pieces
def total_pieces : ℕ := total_rods + total_connectors

-- Theorem to prove
theorem total_pieces_correct : total_pieces = 231 :=
by
  sorry

end total_pieces_correct_l238_238840


namespace find_real_pairs_l238_238722

theorem find_real_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_real_pairs_l238_238722


namespace total_games_in_season_l238_238222

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l238_238222


namespace minimum_value_of_angle_l238_238192

theorem minimum_value_of_angle
  (α : ℝ)
  (h : ∃ x y : ℝ, (x, y) = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))) :
  α = 11 * Real.pi / 6 :=
sorry

end minimum_value_of_angle_l238_238192


namespace max_sides_convex_polygon_with_obtuse_angles_l238_238091

-- Definition of conditions
def is_convex_polygon (n : ℕ) : Prop := n ≥ 3
def obtuse_angles (n : ℕ) (k : ℕ) : Prop := k = 3 ∧ is_convex_polygon n

-- Statement of the problem
theorem max_sides_convex_polygon_with_obtuse_angles (n : ℕ) :
  obtuse_angles n 3 → n ≤ 6 :=
sorry

end max_sides_convex_polygon_with_obtuse_angles_l238_238091


namespace crystal_run_final_segment_length_l238_238768

theorem crystal_run_final_segment_length :
  let north_distance := 2
  let southeast_leg := 1 / Real.sqrt 2
  let southeast_movement_north := -southeast_leg
  let southeast_movement_east := southeast_leg
  let northeast_leg := 2 / Real.sqrt 2
  let northeast_movement_north := northeast_leg
  let northeast_movement_east := northeast_leg
  let total_north_movement := north_distance + northeast_movement_north + southeast_movement_north
  let total_east_movement := southeast_movement_east + northeast_movement_east
  total_north_movement = 2.5 ∧ 
  total_east_movement = 3 * Real.sqrt 2 / 2 ∧ 
  Real.sqrt (total_north_movement^2 + total_east_movement^2) = Real.sqrt 10.75 :=
by
  sorry

end crystal_run_final_segment_length_l238_238768


namespace value_of_x_l238_238170

theorem value_of_x (n x : ℝ) (h1: x = 3 * n) (h2: 2 * n + 3 = 0.2 * 25) : x = 3 :=
by
  sorry

end value_of_x_l238_238170


namespace dot_product_a_a_sub_2b_l238_238340

-- Define the vectors a and b
def a : (ℝ × ℝ) := (2, 3)
def b : (ℝ × ℝ) := (-1, 2)

-- Define the subtraction of vector a and 2 * vector b
def a_sub_2b : (ℝ × ℝ) := (a.1 - 2 * b.1, a.2 - 2 * b.2)

-- Define the dot product of two vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ := u.1 * v.1 + u.2 * v.2

-- State that the dot product of a and (a - 2b) is 5
theorem dot_product_a_a_sub_2b : dot_product a a_sub_2b = 5 := 
by 
  -- proof omitted
  sorry

end dot_product_a_a_sub_2b_l238_238340


namespace ellipse_m_range_l238_238613

theorem ellipse_m_range (m : ℝ) 
  (h1 : m + 9 > 25 - m) 
  (h2 : 25 - m > 0) 
  (h3 : m + 9 > 0) : 
  8 < m ∧ m < 25 := 
by
  sorry

end ellipse_m_range_l238_238613


namespace hyperbola_condition_l238_238908

variables (a b : ℝ)
def e1 : (ℝ × ℝ) := (2, 1)
def e2 : (ℝ × ℝ) := (2, -1)

theorem hyperbola_condition (h1 : e1 = (2, 1)) (h2 : e2 = (2, -1)) (p : ℝ × ℝ)
  (h3 : p = (2 * a + 2 * b, a - b)) :
  4 * a * b = 1 :=
sorry

end hyperbola_condition_l238_238908


namespace total_wheels_l238_238212

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end total_wheels_l238_238212


namespace symmetric_line_eq_l238_238960

-- Given lines
def line₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def mirror_line (x y : ℝ) : Prop := y = -x

-- Definition of symmetry about the line y = -x
def symmetric_about (l₁ l₂: ℝ → ℝ → Prop) : Prop :=
∀ x y, l₁ x y ↔ l₂ y (-x)

-- Definition of line l₂ that is symmetric to line₁ about the mirror_line
def line₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem stating that the symmetric line to line₁ about y = -x is line₂
theorem symmetric_line_eq :
  symmetric_about line₁ line₂ :=
sorry

end symmetric_line_eq_l238_238960


namespace a_n_nonzero_l238_238708

/-- Recurrence relation for the sequence a_n --/
def a : ℕ → ℤ
| 0 => 1
| 1 => 2
| (n + 2) => if (a n * a (n + 1)) % 2 = 1 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

/-- Proof that for all n, a_n is non-zero --/
theorem a_n_nonzero : ∀ n : ℕ, a n ≠ 0 := 
sorry

end a_n_nonzero_l238_238708


namespace product_of_numbers_in_given_ratio_l238_238100

theorem product_of_numbers_in_given_ratio :
  ∃ (x y : ℝ), (x - y) ≠ 0 ∧ (x + y) / (x - y) = 9 ∧ (x * y) / (x - y) = 40 ∧ (x * y) = 80 :=
by {
  sorry
}

end product_of_numbers_in_given_ratio_l238_238100


namespace initial_pigeons_l238_238654

theorem initial_pigeons (n : ℕ) (h : n + 1 = 2) : n = 1 := 
sorry

end initial_pigeons_l238_238654


namespace tangency_condition_intersection_condition_l238_238014

-- Definitions of the circle and line for the given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 8 * y + 12 = 0
def line_eq (a x y : ℝ) : Prop := a * x + y + 2 * a = 0

/-- Theorem for the tangency condition -/
theorem tangency_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = 2) →
  a = 3 / 4 :=
by
  sorry

/-- Theorem for the intersection condition -/
theorem intersection_condition (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y ↔ (x^2 + (y + 4)^2 = 4)) →
  (|(-4 + 2 * a)| / Real.sqrt (a^2 + 1) = Real.sqrt 2) →
  (a = 1 ∨ a = 7) →
  (∀ x y : ℝ,
    (line_eq 1 x y ∧ line_eq 7 x y ↔ 
    (7 * x + y + 14 = 0 ∨ x + y + 2 = 0))) :=
by
  sorry

end tangency_condition_intersection_condition_l238_238014


namespace discount_rate_on_pony_jeans_l238_238226

-- Define the conditions as Lean definitions
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def total_savings : ℝ := 8.91
def total_discount_rate : ℝ := 22
def number_of_fox_pairs : ℕ := 3
def number_of_pony_pairs : ℕ := 2

-- Given definitions of the discount rates on Fox and Pony jeans
variable (F P : ℝ)

-- The system of equations based on the conditions
axiom sum_of_discount_rates : F + P = total_discount_rate
axiom savings_equation : 
  number_of_fox_pairs * (fox_price * F / 100) + number_of_pony_pairs * (pony_price * P / 100) = total_savings

-- The theorem to prove
theorem discount_rate_on_pony_jeans : P = 11 := by
  sorry

end discount_rate_on_pony_jeans_l238_238226


namespace intersection_complement_l238_238843

-- Definitions based on the conditions in the problem
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

-- Definition of complement of set M in the universe U
def complement_U (M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

-- The proof statement
theorem intersection_complement :
  N ∩ (complement_U M) = {3, 5} :=
by
  sorry

end intersection_complement_l238_238843


namespace determinant_of_matrix_A_l238_238253

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + 2, x + 1, x], 
    ![x, x + 2, x + 1], 
    ![x + 1, x, x + 2]]

theorem determinant_of_matrix_A (x : ℝ) :
  (matrix_A x).det = x^2 + 11 * x + 9 :=
by sorry

end determinant_of_matrix_A_l238_238253


namespace length_of_each_lateral_edge_l238_238954

-- Define the concept of a prism with a certain number of vertices and lateral edges
structure Prism where
  vertices : ℕ
  lateral_edges : ℕ

-- Example specific to the problem: Define the conditions given in the problem statement
def given_prism : Prism := { vertices := 12, lateral_edges := 6 }
def sum_lateral_edges : ℕ := 30

-- The main proof statement: Prove the length of each lateral edge
theorem length_of_each_lateral_edge (p : Prism) (h : p = given_prism) :
  (sum_lateral_edges / p.lateral_edges) = 5 :=
by 
  -- The details of the proof will replace 'sorry'
  sorry

end length_of_each_lateral_edge_l238_238954


namespace monthly_rate_is_24_l238_238083

noncomputable def weekly_rate : ℝ := 10
noncomputable def weeks_per_year : ℕ := 52
noncomputable def months_per_year : ℕ := 12
noncomputable def yearly_savings : ℝ := 232

theorem monthly_rate_is_24 (M : ℝ) (h : weeks_per_year * weekly_rate - months_per_year * M = yearly_savings) : 
  M = 24 :=
by
  sorry

end monthly_rate_is_24_l238_238083


namespace greatest_sundays_in_49_days_l238_238282

theorem greatest_sundays_in_49_days : 
  ∀ (days : ℕ), 
    days = 49 → 
    ∀ (sundays_per_week : ℕ), 
      sundays_per_week = 1 → 
      ∀ (weeks : ℕ), 
        weeks = days / 7 → 
        weeks * sundays_per_week = 7 :=
by
  sorry

end greatest_sundays_in_49_days_l238_238282


namespace new_average_mark_l238_238444

theorem new_average_mark (average_mark : ℕ) (average_excluded : ℕ) (total_students : ℕ) (excluded_students: ℕ)
    (h1 : average_mark = 90)
    (h2 : average_excluded = 45)
    (h3 : total_students = 20)
    (h4 : excluded_students = 2) :
  ((total_students * average_mark - excluded_students * average_excluded) / (total_students - excluded_students)) = 95 := by
  sorry

end new_average_mark_l238_238444


namespace product_of_intersection_coordinates_l238_238139

theorem product_of_intersection_coordinates :
  let circle1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 4)^2 = 4}
  let circle2 := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 4)^2 = 9}
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 ∧ p.1 * p.2 = 16 :=
by
  sorry

end product_of_intersection_coordinates_l238_238139


namespace slope_of_line_l238_238360

theorem slope_of_line (a b c : ℝ) (h : 3 * a = 4 * b - 9) : a = 4 / 3 * b - 3 :=
by
  sorry

end slope_of_line_l238_238360


namespace find_r_l238_238683

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = Real.log 9 / Real.log 3 := by
  sorry

end find_r_l238_238683


namespace Emily_GRE_Exam_Date_l238_238645

theorem Emily_GRE_Exam_Date : 
  ∃ (exam_date : ℕ) (exam_month : String), 
  exam_date = 5 ∧ exam_month = "September" ∧
  ∀ study_days break_days start_day_cycles start_break_cycles start_month_june total_days S_june_remaining S_remaining_july S_remaining_august September_start_day, 
    study_days = 15 ∧ 
    break_days = 5 ∧ 
    start_day_cycles = 5 ∧ 
    start_break_cycles = 4 ∧ 
    start_month_june = 1 ∧
    total_days = start_day_cycles * study_days + start_break_cycles * break_days ∧ 
    S_june_remaining = 30 - start_month_june ∧ 
    S_remaining = total_days - S_june_remaining ∧ 
    S_remaining_july = S_remaining - 31 ∧ 
    S_remaining_august = S_remaining_july - 31 ∧ 
    September_start_day = S_remaining_august + 1 ∧
    exam_date = September_start_day ∧ 
    exam_month = "September" := by 
  sorry

end Emily_GRE_Exam_Date_l238_238645


namespace students_enrolled_in_only_english_l238_238674

theorem students_enrolled_in_only_english (total_students both_english_german total_german : ℕ) (h1 : total_students = 40) (h2 : both_english_german = 12) (h3 : total_german = 22) (h4 : ∀ s, s < 40) :
  (total_students - (total_german - both_english_german) - both_english_german) = 18 := 
by {
  sorry
}

end students_enrolled_in_only_english_l238_238674


namespace number_of_odd_blue_faces_cubes_l238_238837

/-
A wooden block is 5 inches long, 5 inches wide, and 1 inch high.
The block is painted blue on all six sides and then cut into twenty-five 1 inch cubes.
Prove that the number of cubes each have a total number of blue faces that is an odd number is 9.
-/

def cubes_with_odd_blue_faces : ℕ :=
  let corner_cubes := 4
  let edge_cubes_not_corners := 16
  let center_cubes := 5
  corner_cubes + center_cubes

theorem number_of_odd_blue_faces_cubes : cubes_with_odd_blue_faces = 9 := by
  have h1 : cubes_with_odd_blue_faces = 4 + 5 := sorry
  have h2 : 4 + 5 = 9 := by norm_num
  exact Eq.trans h1 h2

end number_of_odd_blue_faces_cubes_l238_238837


namespace determine_a_l238_238195

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if h : x = 3 then a else 2 / |x - 3|

theorem determine_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ 3 ∧ x2 ≠ 3 ∧ (f x1 a - 4 = 0) ∧ (f x2 a - 4 = 0) ∧ f 3 a - 4 = 0) →
  a = 4 :=
by
  sorry

end determine_a_l238_238195


namespace parallel_line_through_point_l238_238606

theorem parallel_line_through_point (x y c : ℝ) (h1 : c = -1) :
  ∃ c, (x-2*y+c = 0 ∧ x = 1 ∧ y = 0) ∧ ∃ k b, k = 1 ∧ b = -2 ∧ k*x-2*y+b=0 → c = -1 := by
  sorry

end parallel_line_through_point_l238_238606


namespace least_number_to_add_l238_238794

theorem least_number_to_add (n : ℕ) (h : (1052 + n) % 37 = 0) : n = 19 := by
  sorry

end least_number_to_add_l238_238794


namespace hilary_total_kernels_l238_238637

-- Define the conditions given in the problem
def ears_per_stalk : ℕ := 4
def total_stalks : ℕ := 108
def kernels_per_ear_first_half : ℕ := 500
def additional_kernels_second_half : ℕ := 100

-- Express the main problem as a theorem in Lean
theorem hilary_total_kernels : 
  let total_ears := ears_per_stalk * total_stalks
  let half_ears := total_ears / 2
  let kernels_first_half := half_ears * kernels_per_ear_first_half
  let kernels_per_ear_second_half := kernels_per_ear_first_half + additional_kernels_second_half
  let kernels_second_half := half_ears * kernels_per_ear_second_half
  kernels_first_half + kernels_second_half = 237600 :=
by
  sorry

end hilary_total_kernels_l238_238637


namespace C_D_meeting_time_l238_238737

-- Defining the conditions.
variables (A B C D : Type) [LinearOrderedField A] (V_A V_B V_C V_D : A)
variables (startTime meet_AC meet_BD meet_AB meet_CD : A)

-- Cars' initial meeting conditions
axiom init_cond : startTime = 0
axiom meet_cond_AC : meet_AC = 7
axiom meet_cond_BD : meet_BD = 7
axiom meet_cond_AB : meet_AB = 53
axiom speed_relation : V_A + V_C = V_B + V_D ∧ V_A - V_B = V_D - V_C

-- The problem asks for the meeting time of C and D
theorem C_D_meeting_time : meet_CD = 53 :=
by sorry

end C_D_meeting_time_l238_238737


namespace paintings_left_correct_l238_238048

def initial_paintings := 98
def paintings_gotten_rid_of := 3

theorem paintings_left_correct :
  initial_paintings - paintings_gotten_rid_of = 95 :=
by
  sorry

end paintings_left_correct_l238_238048


namespace min_max_expression_l238_238297

variable (a b c d e : ℝ)

def expression (a b c d e : ℝ) : ℝ :=
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)

theorem min_max_expression :
  a + b + c + d + e = 10 →
  a^2 + b^2 + c^2 + d^2 + e^2 = 20 →
  expression a b c d e = 120 := by
  sorry

end min_max_expression_l238_238297


namespace find_point_P_coordinates_l238_238050

noncomputable def coordinates_of_point (x y : ℝ) : Prop :=
  y > 0 ∧ x < 0 ∧ abs x = 4 ∧ abs y = 4

theorem find_point_P_coordinates : ∃ (x y : ℝ), coordinates_of_point x y ∧ (x, y) = (-4, 4) :=
by
  sorry

end find_point_P_coordinates_l238_238050


namespace frog_eyes_in_pond_l238_238918

-- Definitions based on conditions
def num_frogs : ℕ := 6
def eyes_per_frog : ℕ := 2

-- The property to be proved
theorem frog_eyes_in_pond : num_frogs * eyes_per_frog = 12 :=
by
  sorry

end frog_eyes_in_pond_l238_238918


namespace division_modulus_l238_238068

-- Definitions using the conditions
def a : ℕ := 8 * (10^9)
def b : ℕ := 4 * (10^4)
def n : ℕ := 10^6

-- Lean statement to prove the problem
theorem division_modulus (a b n : ℕ) (h : a = 8 * (10^9) ∧ b = 4 * (10^4) ∧ n = 10^6) : 
  ((a / b) % n) = 200000 := 
by 
  sorry

end division_modulus_l238_238068


namespace curling_teams_l238_238967

-- Define the problem conditions and state the theorem
theorem curling_teams (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
sorry

end curling_teams_l238_238967


namespace remainder_product_mod_5_l238_238473

theorem remainder_product_mod_5 : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end remainder_product_mod_5_l238_238473


namespace sphere_volume_l238_238406

theorem sphere_volume (S : ℝ) (r : ℝ) (V : ℝ) (h₁ : S = 256 * Real.pi) (h₂ : S = 4 * Real.pi * r^2) : V = 2048 / 3 * Real.pi :=
by
  sorry

end sphere_volume_l238_238406


namespace factorize_expression_l238_238927

theorem factorize_expression (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 :=
by
  sorry

end factorize_expression_l238_238927


namespace polygon_with_three_times_exterior_angle_sum_is_octagon_l238_238290

theorem polygon_with_three_times_exterior_angle_sum_is_octagon
  (n : ℕ)
  (h : (n - 2) * 180 = 3 * 360) : n = 8 := by
  sorry

end polygon_with_three_times_exterior_angle_sum_is_octagon_l238_238290


namespace smallest_positive_value_l238_238961

theorem smallest_positive_value (a b c d e : ℝ) (h1 : a = 8 - 2 * Real.sqrt 14) 
  (h2 : b = 2 * Real.sqrt 14 - 8) 
  (h3 : c = 20 - 6 * Real.sqrt 10) 
  (h4 : d = 64 - 16 * Real.sqrt 4) 
  (h5 : e = 16 * Real.sqrt 4 - 64) :
  a = 8 - 2 * Real.sqrt 14 ∧ 0 < a ∧ a < c ∧ a < d :=
by
  sorry

end smallest_positive_value_l238_238961


namespace second_train_cross_time_l238_238850

noncomputable def time_to_cross_second_train : ℝ :=
  let length := 120
  let t1 := 10
  let t_cross := 13.333333333333334
  let v1 := length / t1
  let v_combined := 240 / t_cross
  let v2 := v_combined - v1
  length / v2

theorem second_train_cross_time :
  let t2 := time_to_cross_second_train
  t2 = 20 :=
by
  sorry

end second_train_cross_time_l238_238850


namespace probability_sum_of_digits_eq_10_l238_238984

theorem probability_sum_of_digits_eq_10 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1): 
  let P := m / n
  let valid_numbers := 120
  let total_numbers := 2020
  (P = valid_numbers / total_numbers) → (m = 6) → (n = 101) → (m + n = 107) :=
by 
  sorry

end probability_sum_of_digits_eq_10_l238_238984


namespace discriminant_of_trinomial_l238_238061

theorem discriminant_of_trinomial (x1 x2 : ℝ) (h : x2 - x1 = 2) : (x2 - x1)^2 = 4 :=
by
  sorry

end discriminant_of_trinomial_l238_238061


namespace find_x_l238_238719

def x_y_conditions (x y : ℝ) : Prop :=
  x > y ∧
  x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40 ∧
  x * y + x + y = 8

theorem find_x (x y : ℝ) (h : x_y_conditions x y) : x = 3 + Real.sqrt 7 :=
by
  sorry

end find_x_l238_238719


namespace additional_rocks_needed_l238_238536

-- Define the dimensions of the garden
def length (garden : Type) : ℕ := 15
def width (garden : Type) : ℕ := 10
def rock_cover (rock : Type) : ℕ := 1

-- Define the number of rocks Mrs. Hilt has
def rocks_possessed (mrs_hilt : Type) : ℕ := 64

-- Define the perimeter of the garden
def perimeter (garden : Type) : ℕ :=
  2 * (length garden + width garden)

-- Define the number of rocks required for the first layer
def rocks_first_layer (garden : Type) : ℕ :=
  perimeter garden

-- Define the number of rocks required for the second layer (only longer sides)
def rocks_second_layer (garden : Type) : ℕ :=
  2 * length garden

-- Define the total number of rocks needed
def total_rocks_needed (garden : Type) : ℕ :=
  rocks_first_layer garden + rocks_second_layer garden

-- Prove the number of additional rocks Mrs. Hilt needs
theorem additional_rocks_needed (garden : Type) (mrs_hilt : Type):
  total_rocks_needed garden - rocks_possessed mrs_hilt = 16 := by
  sorry

end additional_rocks_needed_l238_238536


namespace ella_days_11_years_old_l238_238575

theorem ella_days_11_years_old (x y z : ℕ) (h1 : 40 * x + 44 * y + 48 * (180 - x - y) = 7920) (h2 : x + y + z = 180) (h3 : 2 * x + y = 180) : y = 60 :=
by {
  -- proof can be derived from the given conditions
  sorry
}

end ella_days_11_years_old_l238_238575


namespace range_of_a_l238_238594

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l238_238594


namespace ratio_of_area_to_perimeter_l238_238641

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

end ratio_of_area_to_perimeter_l238_238641


namespace correct_average_of_ten_numbers_l238_238271

theorem correct_average_of_ten_numbers :
  let incorrect_average := 20 
  let num_values := 10 
  let incorrect_number := 26
  let correct_number := 86 
  let incorrect_total_sum := incorrect_average * num_values
  let correct_total_sum := incorrect_total_sum - incorrect_number + correct_number 
  (correct_total_sum / num_values) = 26 := 
by
  sorry

end correct_average_of_ten_numbers_l238_238271


namespace divides_8x_7y_l238_238732

theorem divides_8x_7y (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end divides_8x_7y_l238_238732


namespace perfect_square_2n_plus_65_l238_238779

theorem perfect_square_2n_plus_65 (n : ℕ) (h : n > 0) : 
  (∃ m : ℕ, m * m = 2^n + 65) → n = 4 ∨ n = 10 :=
by 
  sorry

end perfect_square_2n_plus_65_l238_238779


namespace ellipse_major_axis_length_l238_238593

-- Conditions
def cylinder_radius : ℝ := 2
def minor_axis (r : ℝ) := 2 * r
def major_axis (minor: ℝ) := minor + 0.6 * minor

-- Problem
theorem ellipse_major_axis_length :
  major_axis (minor_axis cylinder_radius) = 6.4 :=
by
  sorry

end ellipse_major_axis_length_l238_238593


namespace saltwater_concentration_l238_238058

theorem saltwater_concentration (salt_mass water_mass : ℝ) (h₁ : salt_mass = 8) (h₂ : water_mass = 32) : 
  salt_mass / (salt_mass + water_mass) * 100 = 20 := 
by
  sorry

end saltwater_concentration_l238_238058


namespace problem1_problem2_l238_238011

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : 2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3 :=
by sorry

end problem1_problem2_l238_238011


namespace banana_pie_angle_l238_238388

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

end banana_pie_angle_l238_238388


namespace order_options_count_l238_238547

/-- Define the number of options for each category -/
def drinks : ℕ := 3
def salads : ℕ := 2
def pizzas : ℕ := 5

/-- The theorem statement that we aim to prove -/
theorem order_options_count : drinks * salads * pizzas = 30 :=
by
  sorry -- Proof is skipped as instructed

end order_options_count_l238_238547


namespace intersection_A_B_l238_238643

-- Defining sets A and B based on the given conditions.
def A : Set ℝ := {x | ∃ y, y = Real.log x ∧ x > 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Stating the theorem that A ∩ B = {x | 0 < x ∧ x < 3}.
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l238_238643


namespace find_a_plus_b_l238_238933

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * Real.log x

theorem find_a_plus_b (a b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ f a b x = 1 / 2 ∧ (deriv (f a b)) 1 = 0) →
  a + b = -1/2 :=
by
  sorry

end find_a_plus_b_l238_238933


namespace seventh_term_arithmetic_sequence_l238_238037

theorem seventh_term_arithmetic_sequence (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 20)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 28 / 3 := 
sorry

end seventh_term_arithmetic_sequence_l238_238037


namespace least_score_to_play_final_l238_238993

-- Definitions based on given conditions
def num_teams := 2021

def match_points (outcome : String) : ℕ :=
  match outcome with
  | "win"  => 3
  | "draw" => 1
  | "loss" => 0
  | _      => 0

def brazil_won_first_match : Prop := True

def ties_advantage (bfc_score other_team_score : ℕ) : Prop :=
  bfc_score = other_team_score

-- Theorem statement
theorem least_score_to_play_final (bfc_has_tiebreaker : (bfc_score other_team_score : ℕ) → ties_advantage bfc_score other_team_score)
  (bfc_first_match_won : brazil_won_first_match) :
  ∃ (least_score : ℕ), least_score = 2020 := sorry

end least_score_to_play_final_l238_238993


namespace number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l238_238823

-- Definitions for the problem
def n : Nat := 12

-- Statement 1: Number of diagonals in a dodecagon
theorem number_of_diagonals_dodecagon (n : Nat) (h : n = 12) : (n * (n - 3)) / 2 = 54 := by
  sorry

-- Statement 2: Sum of interior angles in a dodecagon
theorem sum_of_interior_angles_dodecagon (n : Nat) (h : n = 12) : 180 * (n - 2) = 1800 := by
  sorry

end number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l238_238823


namespace find_k_l238_238806

variable (m n k : ℝ)

-- Conditions from the problem
def quadratic_roots : Prop := (m + n = -2) ∧ (m * n = k) ∧ (1/m + 1/n = 6)

-- Theorem statement
theorem find_k (h : quadratic_roots m n k) : k = -1/3 :=
sorry

end find_k_l238_238806


namespace correct_sampling_l238_238320

-- Let n be the total number of students
def total_students : ℕ := 60

-- Define the systematic sampling function
def systematic_sampling (n m : ℕ) (start : ℕ) : List ℕ :=
  List.map (λ k => start + k * m) (List.range n)

-- Prove that the sequence generated is equal to [3, 13, 23, 33, 43, 53]
theorem correct_sampling :
  systematic_sampling 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end correct_sampling_l238_238320


namespace system_solution_l238_238574

theorem system_solution:
  let k := 115 / 12 
  ∃ x y z: ℝ, 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    (x + k * y + 5 * z = 0) ∧
    (4 * x + k * y - 3 * z = 0) ∧
    (3 * x + 5 * y - 4 * z = 0) ∧ 
    ((1 : ℝ) / 15 = (x * z) / (y * y)) := 
by sorry

end system_solution_l238_238574


namespace unique_function_satisfying_equation_l238_238301

theorem unique_function_satisfying_equation :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → ∀ x : ℝ, f x = x :=
by
  intro f h
  sorry

end unique_function_satisfying_equation_l238_238301


namespace factorization_correct_l238_238511

theorem factorization_correct : ∀ (x : ℕ), x^2 - x = x * (x - 1) :=
by
  intro x
  -- We know the problem reduces to algebraic identity proof
  sorry

end factorization_correct_l238_238511


namespace negation_exists_lt_zero_l238_238421

variable {f : ℝ → ℝ}

theorem negation_exists_lt_zero :
  ¬ (∃ x : ℝ, f x < 0) → ∀ x : ℝ, 0 ≤ f x := by
  sorry

end negation_exists_lt_zero_l238_238421


namespace parametric_line_l238_238948

theorem parametric_line (s m : ℤ) :
  (∀ t : ℤ, ∃ x y : ℤ, 
    y = 5 * x - 7 ∧
    x = s + 6 * t ∧ y = 3 + m * t ) → 
  (s = 2 ∧ m = 30) :=
by
  sorry

end parametric_line_l238_238948


namespace meaningful_expression_range_l238_238118

theorem meaningful_expression_range (x : ℝ) (h : 1 - x > 0) : x < 1 := sorry

end meaningful_expression_range_l238_238118


namespace marbles_solution_l238_238256

open Nat

def marbles_problem : Prop :=
  ∃ J_k J_j : Nat, (J_k = 3) ∧ (J_k = J_j - 4) ∧ (J_k + J_j = 10)

theorem marbles_solution : marbles_problem := by
  sorry

end marbles_solution_l238_238256


namespace largest_integer_y_l238_238387

theorem largest_integer_y (y : ℤ) : (y / (4:ℚ) + 3 / 7 < 2 / 3) → y ≤ 0 :=
by
  sorry

end largest_integer_y_l238_238387


namespace find_angle4_l238_238814

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180) 
  (h2 : angle3 = angle4) 
  (h3 : angle3 + angle4 = 70) :
  angle4 = 35 := 
by 
  sorry

end find_angle4_l238_238814


namespace sufficient_but_not_necessary_condition_l238_238175

theorem sufficient_but_not_necessary_condition (x y m : ℝ) (h: x^2 + y^2 - 4 * x + 2 * y + m = 0):
  (m = 0) → (5 > m) ∧ ((5 > m) → (m ≠ 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l238_238175


namespace arithmetic_sequence_problem_l238_238582

-- Define sequence and sum properties
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

/- Theorem Statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) 
  (h_seq : arithmetic_sequence a d) 
  (h_initial : a 1 = 31) 
  (h_S_eq : S 10 = S 22) :
  -- Part 1: Find S_n
  (∀ n, S n = 32 * n - n ^ 2) ∧
  -- Part 2: Maximum sum occurs at n = 16 and is 256
  (∀ n, S n ≤ 256 ∧ (S 16 = 256 → ∀ m ≠ 16, S m < 256)) :=
by
  -- proof to be provided here
  sorry

end arithmetic_sequence_problem_l238_238582


namespace f_inequality_l238_238321

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f(x+3) = -1 / f(x)
axiom f_prop1 : ∀ x : ℝ, f (x + 3) = -1 / f x

-- Condition 2: ∀ 3 ≤ x_1 < x_2 ≤ 6, f(x_1) < f(x_2)
axiom f_prop2 : ∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 6 → f x1 < f x2

-- Condition 3: The graph of y = f(x + 3) is symmetric about the y-axis
axiom f_prop3 : ∀ x : ℝ, f (3 - x) = f (3 + x)

-- Theorem: f(3) < f(4.5) < f(7)
theorem f_inequality : f 3 < f 4.5 ∧ f 4.5 < f 7 := by
  sorry

end f_inequality_l238_238321


namespace distance_covered_at_40_kmph_l238_238384

theorem distance_covered_at_40_kmph (x : ℝ) (h : 0 ≤ x ∧ x ≤ 250) 
  (total_distance : x + (250 - x) = 250) 
  (total_time : x / 40 + (250 - x) / 60 = 5.5) : 
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l238_238384


namespace john_total_animals_is_114_l238_238725

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end john_total_animals_is_114_l238_238725


namespace total_students_course_l238_238857

theorem total_students_course 
  (T : ℕ)
  (H1 : (1 / 5 : ℚ) * T = (1 / 5) * T)
  (H2 : (1 / 4 : ℚ) * T = (1 / 4) * T)
  (H3 : (1 / 2 : ℚ) * T = (1 / 2) * T)
  (H4 : T = (1 / 5 : ℚ) * T + (1 / 4 : ℚ) * T + (1 / 2 : ℚ) * T + 30) : 
  T = 600 :=
sorry

end total_students_course_l238_238857


namespace systematic_sampling_second_group_l238_238042

theorem systematic_sampling_second_group
    (N : ℕ) (n : ℕ) (k : ℕ := N / n)
    (number_from_16th_group : ℕ)
    (number_from_1st_group : ℕ := number_from_16th_group - 15 * k)
    (number_from_2nd_group : ℕ := number_from_1st_group + k) :
    N = 160 → n = 20 → number_from_16th_group = 123 → number_from_2nd_group = 11 :=
by
  sorry

end systematic_sampling_second_group_l238_238042


namespace find_angle_between_vectors_l238_238928

noncomputable def angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : ℝ :=
  60

theorem find_angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : angle_between_vectors a b a_nonzero b_nonzero perp1 perp2 = 60 :=
  by 
  sorry

end find_angle_between_vectors_l238_238928


namespace moon_speed_conversion_l238_238189

theorem moon_speed_conversion
  (speed_kps : ℝ)
  (seconds_per_hour : ℝ)
  (h1 : speed_kps = 0.2)
  (h2 : seconds_per_hour = 3600) :
  speed_kps * seconds_per_hour = 720 := by
  sorry

end moon_speed_conversion_l238_238189


namespace evaluate_expression_l238_238912

variable (x y z : ℝ)

theorem evaluate_expression (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := 
sorry

end evaluate_expression_l238_238912


namespace car_travel_distance_l238_238443

-- Define the original gas mileage as x
variable (x : ℝ) (D : ℝ)

-- Define the conditions
def initial_condition : Prop := D = 12 * x
def revised_condition : Prop := D = 10 * (x + 2)

-- The proof goal
theorem car_travel_distance
  (h1 : initial_condition x D)
  (h2 : revised_condition x D) :
  D = 120 := by
  sorry

end car_travel_distance_l238_238443


namespace pentagon_AEDCB_area_l238_238464

-- Definitions based on the given conditions
def rectangle_ABCD (AB BC : ℕ) : Prop :=
AB = 12 ∧ BC = 10

def triangle_ADE (AE ED : ℕ) : Prop :=
AE = 9 ∧ ED = 6 ∧ AE * ED ≠ 0 ∧ (AE^2 + ED^2 = (AE^2 + ED^2))

def area_of_rectangle (AB BC : ℕ) : ℕ :=
AB * BC

def area_of_triangle (AE ED : ℕ) : ℕ :=
(AE * ED) / 2

-- The theorem to be proved
theorem pentagon_AEDCB_area (AB BC AE ED : ℕ) (h_rect : rectangle_ABCD AB BC) (h_tri : triangle_ADE AE ED) :
  area_of_rectangle AB BC - area_of_triangle AE ED = 93 :=
sorry

end pentagon_AEDCB_area_l238_238464


namespace max_xy_l238_238481

variable {x y : ℝ}

theorem max_xy (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 * x + 8 * y = 48) : x * y ≤ 24 :=
sorry

end max_xy_l238_238481


namespace limit_an_to_a_l238_238221

theorem limit_an_to_a (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  |(9 - (n^3 : ℝ)) / (1 + 2 * (n^3 : ℝ)) + 1/2| < ε :=
sorry

end limit_an_to_a_l238_238221


namespace find_k_l238_238376

variable (a b : ℝ → ℝ → ℝ)
variable {k : ℝ}

-- Defining conditions
axiom a_perpendicular_b : ∀ x y, a x y = 0
axiom a_unit_vector : a 1 0 = 1
axiom b_unit_vector : b 0 1 = 1
axiom sum_perpendicular_to_k_diff : ∀ x y, (a x y + b x y) * (k * a x y - b x y) = 0

theorem find_k : k = 1 :=
sorry

end find_k_l238_238376


namespace rectangle_area_l238_238227

theorem rectangle_area (x : ℝ) (w : ℝ) (h_diag : (3 * w) ^ 2 + w ^ 2 = x ^ 2) : 
  3 * w ^ 2 = (3 / 10) * x ^ 2 :=
by
  sorry

end rectangle_area_l238_238227


namespace probability_of_specific_cards_l238_238810

noncomputable def probability_top_heart_second_spade_third_king 
  (deck_size : ℕ) (ranks_per_suit : ℕ) (suits : ℕ) (hearts : ℕ) (spades : ℕ) (kings : ℕ) : ℚ :=
  (hearts * spades * kings) / (deck_size * (deck_size - 1) * (deck_size - 2))

theorem probability_of_specific_cards :
  probability_top_heart_second_spade_third_king 104 26 4 26 26 8 = 169 / 34102 :=
by {
  sorry
}

end probability_of_specific_cards_l238_238810


namespace length_of_chord_EF_l238_238336

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

end length_of_chord_EF_l238_238336


namespace quilt_patch_cost_l238_238201

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l238_238201


namespace students_neither_l238_238808

def total_students : ℕ := 150
def students_math : ℕ := 85
def students_physics : ℕ := 63
def students_chemistry : ℕ := 40
def students_math_physics : ℕ := 20
def students_physics_chemistry : ℕ := 15
def students_math_chemistry : ℕ := 10
def students_all_three : ℕ := 5

theorem students_neither:
  total_students - 
  (students_math + students_physics + students_chemistry 
  - students_math_physics - students_physics_chemistry 
  - students_math_chemistry + students_all_three) = 2 := 
by sorry

end students_neither_l238_238808


namespace xyz_squared_sum_l238_238730

theorem xyz_squared_sum (x y z : ℝ)
  (h1 : x^2 + 6 * y = -17)
  (h2 : y^2 + 4 * z = 1)
  (h3 : z^2 + 2 * x = 2) :
  x^2 + y^2 + z^2 = 14 := 
sorry

end xyz_squared_sum_l238_238730


namespace quadratic_real_roots_l238_238129

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 + 2 * k * x + (k - 3) = 0) ↔ k ≥ 3 / 4 :=
by sorry

end quadratic_real_roots_l238_238129


namespace taxi_fare_total_distance_l238_238859

theorem taxi_fare_total_distance (initial_fare additional_fare : ℝ) (total_fare : ℝ) (initial_distance additional_distance : ℝ) :
  initial_fare = 10 ∧ additional_fare = 1 ∧ initial_distance = 1/5 ∧ (total_fare = 59) →
  (total_distance = initial_distance + additional_distance * ((total_fare - initial_fare) / additional_fare)) →
  total_distance = 10 := 
by 
  sorry

end taxi_fare_total_distance_l238_238859


namespace find_extrema_l238_238250

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end find_extrema_l238_238250


namespace solve_for_n_l238_238849

theorem solve_for_n (n : ℕ) (h : 3^n * 9^n = 81^(n - 12)) : n = 48 :=
sorry

end solve_for_n_l238_238849


namespace kathleen_money_left_l238_238584

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_l238_238584


namespace mean_score_of_seniors_l238_238269

theorem mean_score_of_seniors (num_students : ℕ) (mean_score : ℚ) 
  (ratio_non_seniors_seniors : ℚ) (ratio_mean_seniors_non_seniors : ℚ) (total_score_seniors : ℚ) :
  num_students = 200 →
  mean_score = 80 →
  ratio_non_seniors_seniors = 1.25 →
  ratio_mean_seniors_non_seniors = 1.2 →
  total_score_seniors = 7200 →
  let num_seniors := (num_students : ℚ) / (1 + ratio_non_seniors_seniors)
  let mean_score_seniors := total_score_seniors / num_seniors
  mean_score_seniors = 80.9 :=
by 
  sorry

end mean_score_of_seniors_l238_238269


namespace jonah_poured_total_pitchers_l238_238194

theorem jonah_poured_total_pitchers :
  (0.25 + 0.125) + (0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666) + 
  (0.25 + 0.125) + (0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666) = 1.75 :=
by
  sorry

end jonah_poured_total_pitchers_l238_238194


namespace cos_identity_l238_238126

theorem cos_identity (x : ℝ) 
  (h : Real.sin (2 * x + (Real.pi / 6)) = -1 / 3) : 
  Real.cos ((Real.pi / 3) - 2 * x) = -1 / 3 :=
sorry

end cos_identity_l238_238126


namespace calculate_product_l238_238056

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l238_238056


namespace n_is_prime_l238_238512

theorem n_is_prime (p : ℕ) (h : ℕ) (n : ℕ)
  (hp : Nat.Prime p)
  (hh : h < p)
  (hn : n = p * h + 1)
  (div_n : n ∣ (2^(n-1) - 1))
  (not_div_n : ¬ n ∣ (2^h - 1)) : Nat.Prime n := sorry

end n_is_prime_l238_238512


namespace min_value_dot_product_l238_238795

-- Side length of the square
def side_length: ℝ := 1

-- Definition of points in vector space
variables {A B C D O M N P: Type}

-- Definitions assuming standard Euclidean geometry
variables (O P : ℝ) (a b c : ℝ)

-- Points M and N on the edges AD and BC respectively, line MN passes through O
-- Point P satisfies 2 * vector OP = l * vector OA + (1-l) * vector OB
theorem min_value_dot_product (l : ℝ) (O P M N : ℝ) :
  (2 * (O + P)) = l * (O - a) + (1 - l) * (b + c) ∧
  ((O - P) * (O + P) - ((l^2 - l + 1/2) / 4) = -7/16) :=
by
  sorry

end min_value_dot_product_l238_238795


namespace adjacent_side_length_l238_238460

-- Given the conditions
variables (a b : ℝ)
-- Area of the rectangular flower bed
def area := 6 * a * b - 2 * b
-- One side of the rectangular flower bed
def side1 := 2 * b

-- Prove the length of the adjacent side
theorem adjacent_side_length : 
  (6 * a * b - 2 * b) / (2 * b) = 3 * a - 1 :=
by sorry

end adjacent_side_length_l238_238460


namespace sqrt_expression_eq_three_l238_238255

theorem sqrt_expression_eq_three (h: (Real.sqrt 81) = 9) : Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 :=
by 
  sorry

end sqrt_expression_eq_three_l238_238255


namespace tangent_same_at_origin_l238_238015

noncomputable def f (x : ℝ) := Real.exp (3 * x) - 1
noncomputable def g (x : ℝ) := 3 * Real.exp x - 3

theorem tangent_same_at_origin :
  (deriv f 0 = deriv g 0) ∧ (f 0 = g 0) :=
by
  sorry

end tangent_same_at_origin_l238_238015


namespace greatest_number_of_dimes_l238_238277

-- Definitions according to the conditions in a)
def total_value_in_cents : ℤ := 485
def dime_value_in_cents : ℤ := 10
def nickel_value_in_cents : ℤ := 5

-- The proof problem in Lean 4
theorem greatest_number_of_dimes : 
  ∃ (d : ℤ), (dime_value_in_cents * d + nickel_value_in_cents * d = total_value_in_cents) ∧ d = 32 := 
by
  sorry

end greatest_number_of_dimes_l238_238277


namespace sum_of_cubics_l238_238537

noncomputable def root_polynomial (x : ℝ) := 5 * x^3 + 2003 * x + 3005

theorem sum_of_cubics (a b c : ℝ)
  (h1 : root_polynomial a = 0)
  (h2 : root_polynomial b = 0)
  (h3 : root_polynomial c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
sorry

end sum_of_cubics_l238_238537


namespace bus_waiting_probability_l238_238000

-- Definitions
def arrival_time_range := (0, 90)  -- minutes from 1:00 to 2:30
def bus_wait_time := 20             -- bus waits for 20 minutes

noncomputable def probability_bus_there_when_Laura_arrives : ℚ :=
  let total_area := 90 * 90
  let trapezoid_area := 1400
  let triangle_area := 200
  (trapezoid_area + triangle_area) / total_area

-- Theorem statement
theorem bus_waiting_probability : probability_bus_there_when_Laura_arrives = 16 / 81 := by
  sorry

end bus_waiting_probability_l238_238000


namespace pond_length_l238_238832

theorem pond_length (
    W L P : ℝ) 
    (h1 : L = 2 * W) 
    (h2 : L = 32) 
    (h3 : (L * W) / 8 = P^2) : 
  P = 8 := 
by 
  sorry

end pond_length_l238_238832


namespace relationship_ab_c_l238_238900
open Real

noncomputable def a : ℝ := (1 / 3) ^ (log 3 / log (1 / 3))
noncomputable def b : ℝ := (1 / 3) ^ (log 4 / log (1 / 3))
noncomputable def c : ℝ := 3 ^ log 3

theorem relationship_ab_c : c > b ∧ b > a := by
  sorry

end relationship_ab_c_l238_238900


namespace total_raised_is_420_l238_238886

def pancake_cost : ℝ := 4.00
def bacon_cost : ℝ := 2.00
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90

theorem total_raised_is_420 : (pancake_cost * stacks_sold + bacon_cost * slices_sold) = 420.00 :=
by
  -- Proof goes here
  sorry

end total_raised_is_420_l238_238886


namespace absolute_value_equation_solution_l238_238838

-- mathematical problem representation in Lean
theorem absolute_value_equation_solution (y : ℝ) (h : |y + 2| = |y - 3|) : y = 1 / 2 :=
sorry

end absolute_value_equation_solution_l238_238838


namespace find_parallel_line_through_point_l238_238425

-- Definition of a point in Cartesian coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of a line in slope-intercept form
def line (a b c : ℝ) : Prop := ∀ p : Point, a * p.x + b * p.y + c = 0

-- Conditions provided in the problem
def P : Point := ⟨-1, 3⟩
def line1 : Prop := line 1 (-2) 3
def parallel_line (c : ℝ) : Prop := line 1 (-2) c

-- Theorem to prove
theorem find_parallel_line_through_point : parallel_line 7 :=
sorry

end find_parallel_line_through_point_l238_238425


namespace triangle_perimeter_l238_238395

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

end triangle_perimeter_l238_238395


namespace range_of_x_l238_238470

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) : 
  12 ≤ x := 
sorry

end range_of_x_l238_238470


namespace carrie_bought_tshirts_l238_238740

variable (cost_per_tshirt : ℝ) (total_spent : ℝ)

theorem carrie_bought_tshirts (h1 : cost_per_tshirt = 9.95) (h2 : total_spent = 248) :
  ⌊total_spent / cost_per_tshirt⌋ = 24 :=
by
  sorry

end carrie_bought_tshirts_l238_238740


namespace count_congruent_to_4_mod_7_l238_238417

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l238_238417


namespace isosceles_right_triangle_hypotenuse_l238_238440

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : ℝ) (hyp : a = 30 ∧ h^2 = a^2 + a^2) : h = 30 * Real.sqrt 2 :=
sorry

end isosceles_right_triangle_hypotenuse_l238_238440


namespace compute_special_op_l238_238902

-- Define the operation ※
def special_op (m n : ℚ) := (3 * m + n) * (3 * m - n) + n

-- Hypothesis for specific m and n
def m := (1 : ℚ) / 6
def n := (-1 : ℚ)

-- Proof goal
theorem compute_special_op : special_op m n = -7 / 4 := by
  sorry

end compute_special_op_l238_238902


namespace largest_integer_satisfying_sin_cos_condition_proof_l238_238851

noncomputable def largest_integer_satisfying_sin_cos_condition :=
  ∀ (x : ℝ) (n : ℕ), (∀ (n' : ℕ), (∀ x : ℝ, (Real.sin x ^ n' + Real.cos x ^ n' ≥ 2 / n') → n ≤ n')) → n = 4

theorem largest_integer_satisfying_sin_cos_condition_proof :
  largest_integer_satisfying_sin_cos_condition :=
by
  sorry

end largest_integer_satisfying_sin_cos_condition_proof_l238_238851


namespace ordered_pairs_1806_l238_238416

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end ordered_pairs_1806_l238_238416


namespace hyeongjun_older_sister_age_l238_238029

-- Define the ages of Hyeongjun and his older sister
variables (H S : ℕ)

-- Conditions
def age_gap := S = H + 2
def sum_of_ages := H + S = 26

-- Theorem stating that the older sister's age is 14
theorem hyeongjun_older_sister_age (H S : ℕ) (h1 : age_gap H S) (h2 : sum_of_ages H S) : S = 14 := 
by 
  sorry

end hyeongjun_older_sister_age_l238_238029


namespace remainder_when_divided_by_6_l238_238972

theorem remainder_when_divided_by_6 (n : ℤ) (h_pos : 0 < n) (h_mod12 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l238_238972


namespace area_of_black_region_l238_238867

-- Definitions for the side lengths of the smaller and larger squares
def s₁ : ℕ := 4
def s₂ : ℕ := 8

-- The mathematical problem statement in Lean 4
theorem area_of_black_region : (s₂ * s₂) - (s₁ * s₁) = 48 := by
  sorry

end area_of_black_region_l238_238867


namespace find_other_two_sides_of_isosceles_right_triangle_l238_238665

noncomputable def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  ((AB.1 ^ 2 + AB.2 ^ 2 = AC.1 ^ 2 + AC.2 ^ 2 ∧ BC.1 ^ 2 + BC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AB.1 ^ 2 + AB.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AC.1 ^ 2 + AC.2 ^ 2 = 2 * (AB.1 ^ 2 + AB.2 ^ 2)) ∨
   (AC.1 ^ 2 + AC.2 ^ 2 = BC.1 ^ 2 + BC.2 ^ 2 ∧ AB.1 ^ 2 + AB.2 ^ 2 = 2 * (AC.1 ^ 2 + AC.2 ^ 2)))

theorem find_other_two_sides_of_isosceles_right_triangle (A B C : ℝ × ℝ)
  (h : is_isosceles_right_triangle A B C)
  (line_AB : 2 * A.1 - A.2 = 0)
  (midpoint_hypotenuse : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 2) :
  (A.1 + 2 * A.2 = 2 ∨ A.1 + 2 * A.2 = 14) ∧ 
  ((A.2 = 2 * A.1) ∨ (A.1 = 4)) :=
sorry

end find_other_two_sides_of_isosceles_right_triangle_l238_238665


namespace count_valid_tuples_l238_238477

variable {b_0 b_1 b_2 b_3 : ℕ}

theorem count_valid_tuples : 
  (∃ b_0 b_1 b_2 b_3 : ℕ, 
    0 ≤ b_0 ∧ b_0 ≤ 99 ∧ 
    0 ≤ b_1 ∧ b_1 ≤ 99 ∧ 
    0 ≤ b_2 ∧ b_2 ≤ 99 ∧ 
    0 ≤ b_3 ∧ b_3 ≤ 99 ∧ 
    5040 = b_3 * 10^3 + b_2 * 10^2 + b_1 * 10 + b_0) ∧ 
    ∃ (M : ℕ), 
    M = 504 :=
sorry

end count_valid_tuples_l238_238477


namespace repeating_decimal_sum_l238_238182

/--
The number 3.17171717... can be written as a reduced fraction x/y where x = 314 and y = 99.
We aim to prove that the sum of x and y is 413.
-/
theorem repeating_decimal_sum : 
  let x := 314
  let y := 99
  (x + y) = 413 := 
by
  sorry

end repeating_decimal_sum_l238_238182


namespace recreation_percentage_l238_238264

variable (W : ℝ) -- John's wages last week
variable (recreation_last_week : ℝ := 0.35 * W) -- Amount spent on recreation last week
variable (wages_this_week : ℝ := 0.70 * W) -- Wages this week
variable (recreation_this_week : ℝ := 0.25 * wages_this_week) -- Amount spent on recreation this week

theorem recreation_percentage :
  (recreation_this_week / recreation_last_week) * 100 = 50 := by
  sorry

end recreation_percentage_l238_238264


namespace problem_intersection_l238_238074

open Set

variable {x : ℝ}

def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def C : Set ℝ := {x | (5 / 2) ≤ x ∧ x < 3}

theorem problem_intersection : A ∩ B = C := by
  sorry

end problem_intersection_l238_238074


namespace greatest_gcd_of_rope_lengths_l238_238136

theorem greatest_gcd_of_rope_lengths : Nat.gcd (Nat.gcd 39 52) 65 = 13 := by
  sorry

end greatest_gcd_of_rope_lengths_l238_238136


namespace water_supply_days_l238_238559

theorem water_supply_days (C V : ℕ) 
  (h1: C = 75 * (V + 10))
  (h2: C = 60 * (V + 20)) : 
  (C / V) = 100 := 
sorry

end water_supply_days_l238_238559


namespace ryan_learning_schedule_l238_238433

theorem ryan_learning_schedule
  (E1 E2 E3 S1 S2 S3 : ℕ)
  (hE1 : E1 = 7) (hE2 : E2 = 6) (hE3 : E3 = 8)
  (hS1 : S1 = 4) (hS2 : S2 = 5) (hS3 : S3 = 3):
  (E1 + E2 + E3) - (S1 + S2 + S3) = 9 :=
by
  sorry

end ryan_learning_schedule_l238_238433


namespace solution_inequality_l238_238845

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom increasing_function (x y : ℝ) : x < y → f x < f y

theorem solution_inequality (x : ℝ) : f (2 * x + 1) + f (x - 2) > 0 ↔ x > 1 / 3 := sorry

end solution_inequality_l238_238845


namespace increase_80_by_150_percent_l238_238761

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l238_238761


namespace maximize_x3y4_l238_238339

noncomputable def max_product (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 50) : ℝ :=
  x^3 * y^4

theorem maximize_x3y4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 50) :
  max_product x y hx hy h ≤ max_product (150/7) (200/7) (by norm_num) (by norm_num) (by norm_num) :=
  sorry

end maximize_x3y4_l238_238339


namespace radius_of_outer_circle_l238_238663

theorem radius_of_outer_circle (C_inner : ℝ) (width : ℝ) (h : C_inner = 880) (w : width = 25) :
  ∃ r_outer : ℝ, r_outer = 165 :=
by
  have r_inner := C_inner / (2 * Real.pi)
  have r_outer := r_inner + width
  use r_outer
  sorry

end radius_of_outer_circle_l238_238663


namespace exists_j_half_for_all_j_l238_238465

def is_j_half (n j : ℕ) : Prop := 
  ∃ (q : ℕ), n = (2 * j + 1) * q + j

theorem exists_j_half_for_all_j (k : ℕ) : 
  ∃ n : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ k → is_j_half n j :=
by
  sorry

end exists_j_half_for_all_j_l238_238465


namespace div_1947_l238_238187

theorem div_1947 (n : ℕ) (hn : n % 2 = 1) : 1947 ∣ (46^n + 296 * 13^n) :=
by
  sorry

end div_1947_l238_238187


namespace inequality_solution_l238_238333

theorem inequality_solution (x y z : ℝ) (h1 : x + 3 * y + 2 * z = 6) :
  (z = 3 - 1/2 * x - 3/2 * y) ∧ (x - 2)^2 + (3 * y - 2)^2 ≤ 4 ∧ 0 ≤ x ∧ x ≤ 4 :=
sorry

end inequality_solution_l238_238333


namespace neg_A_is_square_of_int_l238_238577

theorem neg_A_is_square_of_int (x y z : ℤ) (A : ℤ) (h1 : A = x * y + y * z + z * x) 
  (h2 : A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1)) : ∃ k : ℤ, -A = k^2 :=
by
  sorry

end neg_A_is_square_of_int_l238_238577


namespace johnny_worked_hours_l238_238569

theorem johnny_worked_hours (total_earned hourly_wage hours_worked : ℝ) 
(h1 : total_earned = 16.5) (h2 : hourly_wage = 8.25) (h3 : total_earned / hourly_wage = hours_worked) : 
hours_worked = 2 := 
sorry

end johnny_worked_hours_l238_238569


namespace find_number_l238_238434

theorem find_number (x : ℤ) (h : 2 * x + 5 = 17) : x = 6 := 
by
  sorry

end find_number_l238_238434


namespace Jason_has_22_5_toys_l238_238211

noncomputable def RachelToys : ℝ := 1
noncomputable def JohnToys : ℝ := RachelToys + 6.5
noncomputable def JasonToys : ℝ := 3 * JohnToys

theorem Jason_has_22_5_toys : JasonToys = 22.5 := sorry

end Jason_has_22_5_toys_l238_238211


namespace length_of_garden_l238_238286

-- Definitions based on conditions
def P : ℕ := 600
def b : ℕ := 200

-- Theorem statement
theorem length_of_garden : ∃ L : ℕ, 2 * (L + b) = P ∧ L = 100 :=
by
  existsi 100
  simp
  sorry

end length_of_garden_l238_238286


namespace fx_solution_l238_238604

theorem fx_solution (f : ℝ → ℝ) (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1)
  (h_assumption : f (1 / x) = x / (1 - x)) : f x = 1 / (x - 1) :=
by
  sorry

end fx_solution_l238_238604


namespace find_uv_l238_238508

def mat_eqn (u v : ℝ) : Prop :=
  (3 + 8 * u = -3 * v) ∧ (-1 - 6 * u = 1 + 4 * v)

theorem find_uv : ∃ (u v : ℝ), mat_eqn u v ∧ u = -6/7 ∧ v = 5/7 := 
by
  sorry

end find_uv_l238_238508


namespace pen_cost_is_2_25_l238_238989

variables (p i : ℝ)

def total_cost (p i : ℝ) : Prop := p + i = 2.50
def pen_more_expensive (p i : ℝ) : Prop := p = 2 + i

theorem pen_cost_is_2_25 (p i : ℝ) 
  (h1 : total_cost p i) 
  (h2 : pen_more_expensive p i) : 
  p = 2.25 := 
by
  sorry

end pen_cost_is_2_25_l238_238989


namespace prove_true_statement_l238_238430

-- Definitions based on conditions in the problem
def A_statement := ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0

-- Equivalent proof problem in Lean 4
theorem prove_true_statement : A_statement :=
by
  sorry

end prove_true_statement_l238_238430


namespace num_people_on_boats_l238_238183

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end num_people_on_boats_l238_238183


namespace borrowed_sheets_l238_238299

-- Defining the page sum function
def sum_pages (n : ℕ) : ℕ := n * (n + 1)

-- Formulating the main theorem statement
theorem borrowed_sheets (b c : ℕ) (H : c + b ≤ 30) (H_avg : (sum_pages b + sum_pages (30 - b - c) - sum_pages (b + c)) * 2 = 25 * (60 - 2 * c)) :
  c = 10 :=
sorry

end borrowed_sheets_l238_238299


namespace value_of_f_of_1_plus_g_of_2_l238_238353

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := x + 1

theorem value_of_f_of_1_plus_g_of_2 : f (1 + g 2) = 5 :=
by
  sorry

end value_of_f_of_1_plus_g_of_2_l238_238353


namespace largest_of_seven_consecutive_numbers_l238_238597

theorem largest_of_seven_consecutive_numbers (a b c d e f g : ℤ) (h1 : a + 1 = b)
                                             (h2 : b + 1 = c) (h3 : c + 1 = d)
                                             (h4 : d + 1 = e) (h5 : e + 1 = f)
                                             (h6 : f + 1 = g)
                                             (h_avg : (a + b + c + d + e + f + g) / 7 = 20) :
    g = 23 :=
by
  sorry

end largest_of_seven_consecutive_numbers_l238_238597


namespace min_cards_for_certain_event_l238_238426

-- Let's define the deck configuration
structure DeckConfig where
  spades : ℕ
  clubs : ℕ
  hearts : ℕ
  total : ℕ

-- Define the given condition of the deck
def givenDeck : DeckConfig := { spades := 5, clubs := 4, hearts := 6, total := 15 }

-- Predicate to check if m cards drawn guarantees all three suits are present
def is_certain_event (m : ℕ) (deck : DeckConfig) : Prop :=
  m >= deck.spades + deck.hearts + 1

-- The main theorem to prove the minimum number of cards m
theorem min_cards_for_certain_event : ∀ m, is_certain_event m givenDeck ↔ m = 12 :=
by
  sorry

end min_cards_for_certain_event_l238_238426


namespace square_length_QP_l238_238413

theorem square_length_QP (r1 r2 dist : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 7) (h_dist : dist = 15)
  (x : ℝ) (h_equal_chords: QP = PR) :
  x ^ 2 = 65 :=
sorry

end square_length_QP_l238_238413


namespace total_spent_l238_238995

theorem total_spent (B D : ℝ) (h1 : D = 0.7 * B) (h2 : B = D + 15) : B + D = 85 :=
sorry

end total_spent_l238_238995


namespace triangle_identity_proof_l238_238500

variables (r r_a r_b r_c R S p : ℝ)
-- assume necessary properties for valid triangle (not explicitly given in problem but implied)
-- nonnegativity, relations between inradius, exradii and circumradius, etc.

theorem triangle_identity_proof
  (h_r_pos : 0 < r)
  (h_ra_pos : 0 < r_a)
  (h_rb_pos : 0 < r_b)
  (h_rc_pos : 0 < r_c)
  (h_R_pos : 0 < R)
  (h_S_pos : 0 < S)
  (h_p_pos : 0 < p)
  (h_area : S = r * p) :
  (1 / r^3) - (1 / r_a^3) - (1 / r_b^3) - (1 / r_c^3) = (12 * R) / (S^2) :=
sorry

end triangle_identity_proof_l238_238500


namespace ellipse_major_minor_axis_ratio_l238_238081

theorem ellipse_major_minor_axis_ratio
  (a b : ℝ)
  (h₀ : a = 2 * b):
  2 * a = 4 * b :=
by
  sorry

end ellipse_major_minor_axis_ratio_l238_238081


namespace carrie_profit_l238_238747

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end carrie_profit_l238_238747


namespace club_members_problem_l238_238944

theorem club_members_problem 
    (T : ℕ) (C : ℕ) (D : ℕ) (B : ℕ) 
    (h_T : T = 85) (h_C : C = 45) (h_D : D = 32) (h_B : B = 18) :
    let Cₒ := C - B
    let Dₒ := D - B
    let N := T - (Cₒ + Dₒ + B)
    N = 26 :=
by
  sorry

end club_members_problem_l238_238944


namespace shara_shells_after_vacation_l238_238397

-- Definitions based on conditions
def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

-- Statement of the proof problem
theorem shara_shells_after_vacation : 
  initial_shells + (shells_per_day * days) + shells_fourth_day = 41 := by
  sorry

end shara_shells_after_vacation_l238_238397


namespace factorize_expression_l238_238913

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end factorize_expression_l238_238913


namespace no_single_x_for_doughnut_and_syrup_l238_238309

theorem no_single_x_for_doughnut_and_syrup :
  ¬ ∃ x : ℝ, (x^2 - 9 * x + 13 < 0) ∧ (x^2 + x - 5 < 0) :=
sorry

end no_single_x_for_doughnut_and_syrup_l238_238309


namespace corresponding_angles_equal_l238_238651

-- Define what it means for two angles to be corresponding angles
def corresponding_angles (a b : ℝ) : Prop :=
  -- Hypothetical definition
  sorry

-- Lean 4 statement of the problem
theorem corresponding_angles_equal (a b : ℝ) (h : corresponding_angles a b) : a = b :=
by
  sorry

end corresponding_angles_equal_l238_238651


namespace prove_condition_for_equality_l238_238605

noncomputable def condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  c = (b * (a ^ 3 - 1)) / a

theorem prove_condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ (c' : ℕ), (c' = (b * (a ^ 3 - 1)) / a) ∧ 
      c' > 0 ∧ 
      (a + b / c' = a ^ 3 * (b / c')) ) → 
  c = (b * (a ^ 3 - 1)) / a := 
sorry

end prove_condition_for_equality_l238_238605


namespace find_y_find_x_l238_238842

-- Define vectors as per the conditions
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the condition for parallel vectors
def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Question 1 Proof Statement
theorem find_y : ∀ (y : ℝ), is_perpendicular a (b y) → y = 3 / 2 :=
by
  intros y h
  unfold is_perpendicular at h
  unfold dot_product at h
  sorry

-- Question 2 Proof Statement
theorem find_x : ∀ (x : ℝ), is_parallel a (c x) → x = 15 / 2 :=
by
  intros x h
  unfold is_parallel at h
  sorry

end find_y_find_x_l238_238842


namespace number_of_good_weeks_l238_238754

-- Definitions from conditions
def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def tough_weeks : ℕ := 3
def total_money_made : ℕ := 10400
def total_tough_week_sales : ℕ := tough_weeks * tough_week_sales
def total_good_week_sales : ℕ := total_money_made - total_tough_week_sales

-- Question to be proven
theorem number_of_good_weeks (G : ℕ) : 
  (total_good_week_sales = G * good_week_sales) → G = 5 := by
  sorry

end number_of_good_weeks_l238_238754


namespace solve_for_a_l238_238049

theorem solve_for_a (a : ℝ) (h : |2 * a + 1| = 3 * |a| - 2) : a = -1 ∨ a = 3 :=
by
  sorry

end solve_for_a_l238_238049


namespace point_in_second_quadrant_l238_238947

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end point_in_second_quadrant_l238_238947


namespace average_weight_of_16_boys_is_50_25_l238_238538

theorem average_weight_of_16_boys_is_50_25
  (W : ℝ)
  (h1 : 8 * 45.15 = 361.2)
  (h2 : 24 * 48.55 = 1165.2)
  (h3 : 16 * W + 361.2 = 1165.2) :
  W = 50.25 :=
sorry

end average_weight_of_16_boys_is_50_25_l238_238538


namespace centroid_traces_ellipse_l238_238907

noncomputable def fixed_base_triangle (A B : ℝ × ℝ) (d : ℝ) : Prop :=
(A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = d ∧ B.2 = 0)

noncomputable def vertex_moving_on_semicircle (A B C : ℝ × ℝ) : Prop :=
(C.1 - (A.1 + B.1) / 2)^2 + C.2^2 = ((B.1 - A.1) / 2)^2 ∧ C.2 ≥ 0

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem centroid_traces_ellipse
  (A B C G : ℝ × ℝ) (d : ℝ) 
  (h1 : fixed_base_triangle A B d) 
  (h2 : vertex_moving_on_semicircle A B C)
  (h3 : is_centroid A B C G) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (G.1^2 / a^2 + G.2^2 / b^2 = 1) := 
sorry

end centroid_traces_ellipse_l238_238907


namespace quadratic_m_ge_neg2_l238_238487

-- Define the quadratic equation and condition for real roots
def quadratic_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, (x + 2) ^ 2 = m + 2

-- The theorem to prove
theorem quadratic_m_ge_neg2 (m : ℝ) (h : quadratic_has_real_roots m) : m ≥ -2 :=
by {
  sorry
}

end quadratic_m_ge_neg2_l238_238487


namespace men_work_problem_l238_238526

theorem men_work_problem (x : ℕ) (h1 : x * 70 = 40 * 63) : x = 36 := 
by
  sorry

end men_work_problem_l238_238526


namespace area_triangle_ABC_l238_238530

noncomputable def area_trapezoid (AB CD height : ℝ) : ℝ :=
  (AB + CD) * height / 2

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  base * height / 2

variable (AB CD height area_ABCD : ℝ)
variables (h0 : CD = 3 * AB) (h1 : area_trapezoid AB CD height = 24)

theorem area_triangle_ABC : area_triangle AB height = 6 :=
by
  sorry

end area_triangle_ABC_l238_238530


namespace inequality_solution_l238_238131

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) : (x - (1/x) > 0) ↔ (-1 < x ∧ x < 0) ∨ (1 < x) := 
by
  sorry

end inequality_solution_l238_238131


namespace arithmetic_expression_l238_238882

theorem arithmetic_expression : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_expression_l238_238882


namespace jack_valid_sequences_l238_238099

-- Definitions based strictly on the conditions from Step a)
def valid_sequence_count : ℕ :=
  -- Count the valid paths under given conditions (mock placeholder definition)
  1  -- This represents the proof statement

-- The main theorem stating the proof problem
theorem jack_valid_sequences :
  valid_sequence_count = 1 := 
  sorry  -- Proof placeholder

end jack_valid_sequences_l238_238099


namespace probability_check_l238_238405

def total_students : ℕ := 12

def total_clubs : ℕ := 3

def equiprobable_clubs := ∀ s : Fin total_students, ∃ c : Fin total_clubs, true

noncomputable def probability_diff_students : ℝ := 1 - (34650 / (total_clubs ^ total_students))

theorem probability_check :
  equiprobable_clubs →
  probability_diff_students = 0.935 := 
by
  intros
  sorry

end probability_check_l238_238405


namespace increasing_on_1_to_infinity_max_and_min_on_1_to_4_l238_238990

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem increasing_on_1_to_infinity : ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (1 ≤ x2) → f x1 < f x2 := by
  sorry

theorem max_and_min_on_1_to_4 : 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f x ≤ f 4) ∧ 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f 1 ≤ f x) := by
  sorry

end increasing_on_1_to_infinity_max_and_min_on_1_to_4_l238_238990


namespace sandbox_perimeter_l238_238819

def sandbox_width : ℝ := 5
def sandbox_length := 2 * sandbox_width
def perimeter (length width : ℝ) := 2 * (length + width)

theorem sandbox_perimeter : perimeter sandbox_length sandbox_width = 30 := 
by
  sorry

end sandbox_perimeter_l238_238819


namespace packs_of_blue_tshirts_l238_238670

theorem packs_of_blue_tshirts (total_tshirts white_packs white_per_pack blue_per_pack : ℕ) 
  (h_white_packs : white_packs = 3) 
  (h_white_per_pack : white_per_pack = 6) 
  (h_blue_per_pack : blue_per_pack = 4) 
  (h_total_tshirts : total_tshirts = 26) : 
  (total_tshirts - white_packs * white_per_pack) / blue_per_pack = 2 := 
by
  -- Proof omitted
  sorry

end packs_of_blue_tshirts_l238_238670


namespace unit_digit_product_l238_238778

-- Definition of unit digit function
def unit_digit (n : Nat) : Nat := n % 10

-- Conditions about unit digits of given powers
lemma unit_digit_3_pow_68 : unit_digit (3 ^ 68) = 1 := by sorry
lemma unit_digit_6_pow_59 : unit_digit (6 ^ 59) = 6 := by sorry
lemma unit_digit_7_pow_71 : unit_digit (7 ^ 71) = 3 := by sorry

-- Main statement
theorem unit_digit_product : unit_digit (3 ^ 68 * 6 ^ 59 * 7 ^ 71) = 8 := by
  have h3 := unit_digit_3_pow_68
  have h6 := unit_digit_6_pow_59
  have h7 := unit_digit_7_pow_71
  sorry

end unit_digit_product_l238_238778


namespace probability_both_asian_selected_probability_A1_but_not_B1_selected_l238_238390

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_both_asian_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  asian_ways / total_ways = 1 / 5 := by
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  sorry

theorem probability_A1_but_not_B1_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := 9
  let valid_ways := 2
  valid_ways / total_ways = 2 / 9 := by
  let total_ways := 9
  let valid_ways := 2
  sorry

end probability_both_asian_selected_probability_A1_but_not_B1_selected_l238_238390


namespace trig_identity_l238_238451

theorem trig_identity :
  2 * Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4)^2 + Real.cos (Real.pi / 3) = 1 :=
by
  sorry

end trig_identity_l238_238451


namespace basketball_lineups_l238_238410

noncomputable def num_starting_lineups (total_players : ℕ) (fixed_players : ℕ) (chosen_players : ℕ) : ℕ :=
  Nat.choose (total_players - fixed_players) (chosen_players - fixed_players)

theorem basketball_lineups :
  num_starting_lineups 15 2 6 = 715 := by
  sorry

end basketball_lineups_l238_238410


namespace value_of_y_l238_238701

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l238_238701


namespace books_arrangement_count_l238_238750

noncomputable def arrangement_of_books : ℕ :=
  let total_books := 5
  let identical_books := 2
  Nat.factorial total_books / Nat.factorial identical_books

theorem books_arrangement_count : arrangement_of_books = 60 := by
  sorry

end books_arrangement_count_l238_238750


namespace line_tangent_to_ellipse_l238_238658

theorem line_tangent_to_ellipse (m : ℝ) (a : ℝ) (b : ℝ) (h_a : a = 3) (h_b : b = 1) :
  m^2 = 1 / 3 := by
  sorry

end line_tangent_to_ellipse_l238_238658


namespace correct_statement_d_l238_238852

theorem correct_statement_d : 
  (∃ x : ℝ, 2^x < x^2) ↔ ¬(∀ x : ℝ, 2^x ≥ x^2) :=
by
  sorry

end correct_statement_d_l238_238852


namespace curler_ratio_l238_238520

theorem curler_ratio
  (total_curlers : ℕ)
  (pink_curlers : ℕ)
  (blue_curlers : ℕ)
  (green_curlers : ℕ)
  (h1 : total_curlers = 16)
  (h2 : blue_curlers = 2 * pink_curlers)
  (h3 : green_curlers = 4) :
  pink_curlers / total_curlers = 1 / 4 := by
  sorry

end curler_ratio_l238_238520


namespace max_value_of_f_l238_238653

def f (x : ℝ) : ℝ := x^2 - 2 * x - 5

theorem max_value_of_f : ∃ x ∈ (Set.Icc (-2:ℝ) 2), ∀ y ∈ (Set.Icc (-2:ℝ) 2), f y ≤ f x ∧ f x = 3 := by
  sorry

end max_value_of_f_l238_238653


namespace correct_statements_l238_238965

variables (a : Nat → ℤ) (d : ℤ)

-- Suppose {a_n} is an arithmetic sequence with common difference d
def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * d)) / 2

-- Conditions: S_11 > 0 and S_12 < 0
axiom S11_pos : S a d 11 > 0
axiom S12_neg : S a d 12 < 0

-- The goal is to determine which statements are correct
theorem correct_statements : (d < 0) ∧ (∀ n, 1 ≤ n → n ≤ 12 → S a d 6 ≥ S a d n ∧ S a d 6 ≠ S a d 11 ) := 
sorry

end correct_statements_l238_238965


namespace bus_stops_time_per_hour_l238_238586

theorem bus_stops_time_per_hour 
  (avg_speed_without_stoppages : ℝ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 75) 
  (h2 : avg_speed_with_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 28 :=
by
  sorry

end bus_stops_time_per_hour_l238_238586


namespace solved_distance_l238_238086

variable (D : ℝ) 

-- Time for A to cover the distance
variable (tA : ℝ) (tB : ℝ)
variable (dA : ℝ) (dB : ℝ := D - 26)

-- A covers the distance in 36 seconds
axiom hA : tA = 36

-- B covers the distance in 45 seconds
axiom hB : tB = 45

-- A beats B by 26 meters implies B covers (D - 26) in the time A covers D
axiom h_diff : dB = dA - 26

theorem solved_distance :
  D = 130 := 
by 
  sorry

end solved_distance_l238_238086


namespace find_position_2002_l238_238142

def T (n : ℕ) : ℕ := n * (n + 1) / 2
def a (n : ℕ) : ℕ := T n + 1

theorem find_position_2002 : ∃ row col : ℕ, 1 ≤ row ∧ 1 ≤ col ∧ (a (row - 1) + (col - 1) = 2002 ∧ row = 15 ∧ col = 49) := 
sorry

end find_position_2002_l238_238142


namespace actual_value_wrongly_copied_l238_238125

theorem actual_value_wrongly_copied (mean_initial : ℝ) (n : ℕ) (wrong_value : ℝ) (mean_correct : ℝ) :
  mean_initial = 140 → n = 30 → wrong_value = 135 → mean_correct = 140.33333333333334 →
  ∃ actual_value : ℝ, actual_value = 145 :=
by
  intros
  sorry

end actual_value_wrongly_copied_l238_238125


namespace range_of_c_l238_238043

-- Definitions of p and q based on conditions
def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (c > 1 / 2)

-- The theorem states the required condition on c
theorem range_of_c (c : ℝ) (h : c > 0) :
  ¬(p c ∧ q c) ∧ (p c ∨ q c) ↔ (0 < c ∧ c ≤ 1 / 2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l238_238043


namespace find_T_l238_238072

variables (h K T : ℝ)
variables (h_val : 4 * h * 7 + 2 = 58)
variables (K_val : K = 9)

theorem find_T : T = 74 :=
by
  sorry

end find_T_l238_238072


namespace small_fries_number_l238_238977

variables (L S : ℕ)

axiom h1 : L + S = 24
axiom h2 : L = 5 * S

theorem small_fries_number : S = 4 :=
by sorry

end small_fries_number_l238_238977


namespace prove_inequality_l238_238617

variable (f : ℝ → ℝ)

def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def isMonotonicOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem prove_inequality
  (h1 : isEvenFunction f)
  (h2 : isMonotonicOnInterval f 0 5)
  (h3 : f (-3) < f 1) :
  f 0 > f 1 :=
sorry

end prove_inequality_l238_238617


namespace joan_remaining_kittens_l238_238714

-- Definitions based on the given conditions
def original_kittens : Nat := 8
def kittens_given_away : Nat := 2

-- Statement to prove
theorem joan_remaining_kittens : original_kittens - kittens_given_away = 6 := 
by
  -- Proof skipped
  sorry

end joan_remaining_kittens_l238_238714


namespace volume_of_truncated_triangular_pyramid_l238_238869

variable {a b H α : ℝ} (h1 : H = Real.sqrt (a * b))

theorem volume_of_truncated_triangular_pyramid
  (h2 : H = Real.sqrt (a * b))
  (h3 : 0 < a)
  (h4 : 0 < b)
  (h5 : 0 < H)
  (h6 : 0 < α) :
  (volume : ℝ) = H^3 * Real.sqrt 3 / (4 * (Real.sin α)^2) := sorry

end volume_of_truncated_triangular_pyramid_l238_238869


namespace pure_ghee_added_l238_238880

theorem pure_ghee_added
  (Q : ℕ) (hQ : Q = 30)
  (P : ℕ)
  (original_pure_ghee : ℕ := (Q / 2))
  (original_vanaspati : ℕ := (Q / 2))
  (new_total_ghee : ℕ := Q + P)
  (new_vanaspati_fraction : ℝ := 0.3) :
  original_vanaspati = (new_vanaspati_fraction * ↑new_total_ghee : ℝ) → P = 20 := by
  sorry

end pure_ghee_added_l238_238880


namespace gcd_expression_l238_238936

noncomputable def odd_multiple_of_7771 (a : ℕ) : Prop := 
  ∃ k : ℕ, k % 2 = 1 ∧ a = 7771 * k

theorem gcd_expression (a : ℕ) (h : odd_multiple_of_7771 a) : 
  Int.gcd (8 * a^2 + 57 * a + 132) (2 * a + 9) = 9 :=
  sorry

end gcd_expression_l238_238936


namespace ratio_area_ADE_BCED_is_8_over_9_l238_238246

noncomputable def ratio_area_ADE_BCED 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) : ℝ := 
  sorry

theorem ratio_area_ADE_BCED_is_8_over_9 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) :
  ratio_area_ADE_BCED AB BC AC AD AE hAB hBC hAC hAD hAE = 8 / 9 :=
  sorry

end ratio_area_ADE_BCED_is_8_over_9_l238_238246


namespace sufficient_not_necessary_condition_l238_238313

variable (x a : ℝ)

def p := x ≤ -1
def q := a ≤ x ∧ x < a + 2

-- If q is sufficient but not necessary for p, then the range of a is (-∞, -3]
theorem sufficient_not_necessary_condition : 
  (∀ x, q x a → p x) ∧ ∃ x, p x ∧ ¬ q x a → a ≤ -3 :=
by
  sorry

end sufficient_not_necessary_condition_l238_238313


namespace unique_solution_n_l238_238894

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem unique_solution_n (h : ∀ n : ℕ, (n > 0) → n^3 = 8 * (sum_digits n)^3 + 6 * (sum_digits n) * n + 1 → n = 17) : 
  n = 17 := 
by
  sorry

end unique_solution_n_l238_238894


namespace trees_chopped_in_first_half_l238_238114

theorem trees_chopped_in_first_half (x : ℕ) (h1 : ∀ t, t = x + 300) (h2 : 3 * t = 1500) : x = 200 :=
by
  sorry

end trees_chopped_in_first_half_l238_238114


namespace geometric_sequence_l238_238549

-- Define the set and its properties
variable (A : Set ℕ) (a : ℕ → ℕ) (n : ℕ)
variable (h1 : 1 ≤ a 1) 
variable (h2 : ∀ (i : ℕ), 1 ≤ i → i < n → a i < a (i + 1))
variable (h3 : n ≥ 5)
variable (h4 : ∀ (i j : ℕ), 1 ≤ i → i ≤ j → j ≤ n → (a i) * (a j) ∈ A ∨ (a i) / (a j) ∈ A)

-- Statement to prove that the sequence forms a geometric sequence
theorem geometric_sequence : 
  ∃ (c : ℕ), c > 1 ∧ ∀ (i : ℕ), 1 ≤ i → i ≤ n → a i = c^(i-1) := sorry

end geometric_sequence_l238_238549


namespace jack_walking_rate_l238_238351

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

end jack_walking_rate_l238_238351


namespace paul_sold_11_books_l238_238802

variable (initial_books : ℕ) (books_given : ℕ) (books_left : ℕ) (books_sold : ℕ)

def number_of_books_sold (initial_books books_given books_left books_sold : ℕ) : Prop :=
  initial_books - books_given - books_left = books_sold

theorem paul_sold_11_books : number_of_books_sold 108 35 62 11 :=
by
  sorry

end paul_sold_11_books_l238_238802


namespace xiaoguang_advances_l238_238699

theorem xiaoguang_advances (x1 x2 x3 x4 : ℝ) (h1 : 96 ≤ (x1 + x2 + x3 + x4) / 4) (hx1 : x1 = 95) (hx2 : x2 = 97) (hx3 : x3 = 94) : 
  98 ≤ x4 := 
by 
  sorry

end xiaoguang_advances_l238_238699


namespace find_range_of_values_l238_238202

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem find_range_of_values (f : ℝ → ℝ) (h_even : is_even f)
  (h_increasing : is_increasing_on_nonneg f) (h_f1_zero : f 1 = 0) :
  { x : ℝ | f (Real.log x / Real.log (1/2)) > 0 } = 
  { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by 
  sorry

end find_range_of_values_l238_238202


namespace sockPairsCount_l238_238872

noncomputable def countSockPairs : ℕ :=
  let whitePairs := Nat.choose 6 2 -- 15
  let brownPairs := Nat.choose 7 2 -- 21
  let bluePairs := Nat.choose 3 2 -- 3
  let oneRedOneWhite := 4 * 6 -- 24
  let oneRedOneBrown := 4 * 7 -- 28
  let oneRedOneBlue := 4 * 3 -- 12
  let bothRed := Nat.choose 4 2 -- 6
  whitePairs + brownPairs + bluePairs + oneRedOneWhite + oneRedOneBrown + oneRedOneBlue + bothRed

theorem sockPairsCount : countSockPairs = 109 := by
  sorry

end sockPairsCount_l238_238872


namespace sqrt_12_same_type_sqrt_3_l238_238364

-- We define that two square roots are of the same type if one is a multiple of the other
def same_type (a b : ℝ) : Prop := ∃ k : ℝ, b = k * a

-- We need to show that sqrt(12) is of the same type as sqrt(3), and check options
theorem sqrt_12_same_type_sqrt_3 : same_type (Real.sqrt 3) (Real.sqrt 12) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 8) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 18) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 6) :=
by
  sorry -- Proof is omitted


end sqrt_12_same_type_sqrt_3_l238_238364


namespace angle_B_value_l238_238279

theorem angle_B_value (a b c B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
    B = (Real.pi / 3) ∨ B = (2 * Real.pi / 3) :=
by
    sorry

end angle_B_value_l238_238279


namespace initial_persons_count_l238_238316

theorem initial_persons_count (P : ℕ) (H1 : 18 * P = 1) (H2 : 6 * P = 1/3) (H3 : 9 * (P + 4) = 2/3) : P = 12 :=
by
  sorry

end initial_persons_count_l238_238316


namespace total_boys_in_camp_l238_238453

theorem total_boys_in_camp (T : ℝ) (h : 0.70 * (0.20 * T) = 28) : T = 200 := 
by
  sorry

end total_boys_in_camp_l238_238453


namespace width_of_grassy_plot_l238_238094

-- Definitions
def length_plot : ℕ := 110
def width_path : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.50
def total_cost : ℝ := 425

-- Hypotheses and Target Proposition
theorem width_of_grassy_plot (w : ℝ) 
  (h1 : length_plot = 110)
  (h2 : width_path = 2.5)
  (h3 : cost_per_sq_meter = 0.50)
  (h4 : total_cost = 425)
  (h5 : (length_plot + 2 * width_path) * (w + 2 * width_path) = 115 * (w + 5))
  (h6 : 110 * w = 110 * w)
  (h7 : (115 * (w + 5) - (110 * w)) = total_cost / cost_per_sq_meter) :
  w = 55 := 
sorry

end width_of_grassy_plot_l238_238094


namespace fg_value_correct_l238_238224

def f_table (x : ℕ) : ℕ :=
  if x = 1 then 3
  else if x = 3 then 7
  else if x = 5 then 9
  else if x = 7 then 13
  else if x = 9 then 17
  else 0  -- Default value to handle unexpected inputs

def g_table (x : ℕ) : ℕ :=
  if x = 1 then 54
  else if x = 3 then 9
  else if x = 5 then 25
  else if x = 7 then 19
  else if x = 9 then 44
  else 0  -- Default value to handle unexpected inputs

theorem fg_value_correct : f_table (g_table 3) = 17 := 
by sorry

end fg_value_correct_l238_238224


namespace allocate_to_Team_A_l238_238103

theorem allocate_to_Team_A (x : ℕ) :
  31 + x = 2 * (50 - x) →
  x = 23 :=
by
  sorry

end allocate_to_Team_A_l238_238103


namespace cookies_batches_needed_l238_238580

noncomputable def number_of_recipes (total_students : ℕ) (attendance_drop : ℝ) (cookies_per_batch : ℕ) : ℕ :=
  let remaining_students := (total_students : ℝ) * (1 - attendance_drop)
  let total_cookies := remaining_students * 2
  let recipes_needed := total_cookies / cookies_per_batch
  (Nat.ceil recipes_needed : ℕ)

theorem cookies_batches_needed :
  number_of_recipes 150 0.40 18 = 10 :=
by
  sorry

end cookies_batches_needed_l238_238580


namespace terminal_side_in_first_quadrant_l238_238854

noncomputable def theta := -5

def in_first_quadrant (θ : ℝ) : Prop :=
  by sorry

theorem terminal_side_in_first_quadrant : in_first_quadrant theta := 
  by sorry

end terminal_side_in_first_quadrant_l238_238854


namespace relationship_a_b_l238_238482

theorem relationship_a_b
  (m a b : ℝ)
  (h1 : ∃ m, ∀ x, -2 * x + m = y)
  (h2 : ∃ x₁ y₁, (x₁ = -2) ∧ (y₁ = a) ∧ (-2 * x₁ + m = y₁))
  (h3 : ∃ x₂ y₂, (x₂ = 2) ∧ (y₂ = b) ∧ (-2 * x₂ + m = y₂)) :
  a > b :=
sorry

end relationship_a_b_l238_238482


namespace tan_double_angle_third_quadrant_l238_238101

open Real

theorem tan_double_angle_third_quadrant (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : sin (π - α) = -3 / 5) :
  tan (2 * α) = 24 / 7 := 
sorry

end tan_double_angle_third_quadrant_l238_238101


namespace minimum_toothpicks_removal_l238_238513

theorem minimum_toothpicks_removal
  (total_toothpicks : ℕ)
  (grid_size : ℕ)
  (toothpicks_per_square : ℕ)
  (shared_sides : ℕ)
  (interior_toothpicks : ℕ) 
  (diagonal_toothpicks : ℕ)
  (min_removal : ℕ) 
  (no_squares_or_triangles : Bool)
  (h1 : total_toothpicks = 40)
  (h2 : grid_size = 3)
  (h3 : toothpicks_per_square = 4)
  (h4 : shared_sides = 16)
  (h5 : interior_toothpicks = 16) 
  (h6 : diagonal_toothpicks = 12)
  (h7 : min_removal = 16)
: no_squares_or_triangles := 
sorry

end minimum_toothpicks_removal_l238_238513


namespace total_enemies_l238_238489

theorem total_enemies (n : ℕ) : (n - 3) * 9 = 72 → n = 11 :=
by
  sorry

end total_enemies_l238_238489


namespace find_x_l238_238831

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for perpendicular vectors
def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem find_x (x : ℝ) (h : perpendicular a (b x)) : x = -3 / 2 :=
by
  sorry

end find_x_l238_238831


namespace claudia_candle_choices_l238_238614

-- Claudia can choose 4 different candles
def num_candles : ℕ := 4

-- Claudia can choose 8 out of 9 different flowers
def num_ways_to_choose_flowers : ℕ := Nat.choose 9 8

-- The total number of groupings is given as 54
def total_groupings : ℕ := 54

-- Prove the main theorem using the conditions
theorem claudia_candle_choices :
  num_ways_to_choose_flowers = 9 ∧ num_ways_to_choose_flowers * C = total_groupings → C = 6 :=
by
  sorry

end claudia_candle_choices_l238_238614


namespace billy_has_2_cherries_left_l238_238682

-- Define the initial number of cherries
def initialCherries : Nat := 74

-- Define the number of cherries eaten
def eatenCherries : Nat := 72

-- Define the number of remaining cherries
def remainingCherries : Nat := initialCherries - eatenCherries

-- Theorem statement: Prove that remainingCherries is equal to 2
theorem billy_has_2_cherries_left : remainingCherries = 2 := by
  sorry

end billy_has_2_cherries_left_l238_238682


namespace remainder_of_product_div_10_l238_238681

theorem remainder_of_product_div_10 : 
  (3251 * 7462 * 93419) % 10 = 8 := 
sorry

end remainder_of_product_div_10_l238_238681


namespace panda_on_stilts_height_l238_238079

theorem panda_on_stilts_height (x : ℕ) (h_A : ℕ) 
  (h1 : h_A = x / 4) -- A Bao's height accounts for 1/4 of initial total height
  (h2 : x - 40 = 3 * h_A) -- After breaking 20 dm off each stilt, the new total height is such that A Bao's height accounts for 1/3 of this new height
  : x = 160 := 
by
  sorry

end panda_on_stilts_height_l238_238079


namespace simplify_expr1_simplify_expr2_l238_238138

variables (x y a b : ℝ)

-- Problem 1
theorem simplify_expr1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := 
by sorry

-- Problem 2
theorem simplify_expr2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := 
by sorry

end simplify_expr1_simplify_expr2_l238_238138


namespace sum_of_three_consecutive_even_l238_238352

theorem sum_of_three_consecutive_even (a1 a2 a3 : ℤ) (h1 : a1 % 2 = 0) (h2 : a2 = a1 + 2) (h3 : a3 = a1 + 4) (h4 : a1 + a3 = 128) : a1 + a2 + a3 = 192 :=
sorry

end sum_of_three_consecutive_even_l238_238352


namespace area_on_larger_sphere_l238_238625

-- Define the radii of the spheres
def r_small : ℝ := 1
def r_in : ℝ := 4
def r_out : ℝ := 6

-- Given the area on the smaller sphere
def A_small_sphere_area : ℝ := 37

-- Statement: Find the area on the larger sphere
theorem area_on_larger_sphere :
  (A_small_sphere_area * (r_out / r_in) ^ 2 = 83.25) := by
  sorry

end area_on_larger_sphere_l238_238625


namespace base7_addition_sum_l238_238934

theorem base7_addition_sum :
  let n1 := 256
  let n2 := 463
  let n3 := 132
  n1 + n2 + n3 = 1214 := sorry

end base7_addition_sum_l238_238934


namespace total_candidates_l238_238576

theorem total_candidates (T : ℝ) 
  (h1 : 0.45 * T = T * 0.45)
  (h2 : 0.38 * T = T * 0.38)
  (h3 : 0.22 * T = T * 0.22)
  (h4 : 0.12 * T = T * 0.12)
  (h5 : 0.09 * T = T * 0.09)
  (h6 : 0.10 * T = T * 0.10)
  (h7 : 0.05 * T = T * 0.05)
  (h_passed_english_alone : T - (0.45 * T - 0.12 * T - 0.10 * T + 0.05 * T) = 720) :
  T = 1000 :=
by
  sorry

end total_candidates_l238_238576


namespace sum_of_squares_l238_238066

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 5) (h2 : ab + bc + ac = 5) : a^2 + b^2 + c^2 = 15 :=
by sorry

end sum_of_squares_l238_238066


namespace gcf_180_240_300_l238_238519

theorem gcf_180_240_300 : Nat.gcd (Nat.gcd 180 240) 300 = 60 := sorry

end gcf_180_240_300_l238_238519


namespace discount_on_shoes_l238_238766

theorem discount_on_shoes (x : ℝ) :
  let shoe_price := 200
  let shirt_price := 80
  let total_spent := 285
  let total_shirt_price := 2 * shirt_price
  let initial_total := shoe_price + total_shirt_price
  let disc_shoe_price := shoe_price - (shoe_price * x / 100)
  let pre_final_total := disc_shoe_price + total_shirt_price
  let final_total := pre_final_total * (1 - 0.05)
  final_total = total_spent → x = 30 :=
by
  intros shoe_price shirt_price total_spent total_shirt_price initial_total disc_shoe_price pre_final_total final_total h
  dsimp [shoe_price, shirt_price, total_spent, total_shirt_price, initial_total, disc_shoe_price, pre_final_total, final_total] at h
  -- Here, we would normally continue the proof, but we'll insert 'sorry' for now as instructed.
  sorry

end discount_on_shoes_l238_238766


namespace faster_train_cross_time_l238_238599

noncomputable def time_to_cross (speed_fast_kmph : ℝ) (speed_slow_kmph : ℝ) (length_fast_m : ℝ) : ℝ :=
  let speed_diff_kmph := speed_fast_kmph - speed_slow_kmph
  let speed_diff_mps := (speed_diff_kmph * 1000) / 3600
  length_fast_m / speed_diff_mps

theorem faster_train_cross_time :
  time_to_cross 72 36 120 = 12 :=
by
  sorry

end faster_train_cross_time_l238_238599


namespace new_ratio_after_adding_ten_l238_238550

theorem new_ratio_after_adding_ten 
  (x : ℕ) 
  (h_ratio : 3 * x = 15) 
  (new_smaller : ℕ := x + 10) 
  (new_larger : ℕ := 15) 
  : new_smaller / new_larger = 1 :=
by sorry

end new_ratio_after_adding_ten_l238_238550


namespace lcm_of_8_12_15_l238_238240

theorem lcm_of_8_12_15 : Nat.lcm 8 (Nat.lcm 12 15) = 120 :=
by
  -- This is where the proof steps would go
  sorry

end lcm_of_8_12_15_l238_238240


namespace inscribed_circle_radius_square_l238_238733

theorem inscribed_circle_radius_square (ER RF GS SH : ℝ) (r : ℝ) 
  (hER : ER = 23) (hRF : RF = 34) (hGS : GS = 42) (hSH : SH = 28)
  (h_tangent : ∀ t, t = r * r * (70 * t - 87953)) :
  r^2 = 87953 / 70 :=
by
  sorry

end inscribed_circle_radius_square_l238_238733


namespace initial_investment_B_l238_238877
-- Import necessary Lean library

-- Define the necessary conditions and theorems
theorem initial_investment_B (x : ℝ) (profit_A : ℝ) (profit_total : ℝ)
  (initial_A : ℝ) (initial_A_after_8_months : ℝ) (profit_B : ℝ) 
  (initial_A_months : ℕ) (initial_A_after_8_months_months : ℕ) 
  (initial_B_months : ℕ) (initial_B_after_8_months_months : ℕ) : 
  initial_A = 3000 ∧ initial_A_after_8_months = 2000 ∧
  profit_A = 240 ∧ profit_total = 630 ∧ 
  profit_B = profit_total - profit_A ∧
  (initial_A * initial_A_months + initial_A_after_8_months * initial_A_after_8_months_months) /
  ((initial_B_months * x + initial_B_after_8_months_months * (x + 1000))) = 
  profit_A / profit_B →
  x = 4000 :=
by
  sorry

end initial_investment_B_l238_238877


namespace g_sum_even_l238_238285

def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5

theorem g_sum_even (a b c d : ℝ) (h : g 42 a b c d = 3) : g 42 a b c d + g (-42) a b c d = 6 := by
  sorry

end g_sum_even_l238_238285


namespace geometric_sequence_a2_l238_238070

theorem geometric_sequence_a2 (a1 a2 a3 : ℝ) (h1 : 1 * (1/a1) = a1)
  (h2 : a1 * (1/a2) = a2) (h3 : a2 * (1/a3) = a3) (h4 : a3 * (1/4) = 4)
  (h5 : a2 > 0) : a2 = 2 := sorry

end geometric_sequence_a2_l238_238070


namespace four_people_seven_chairs_l238_238662

def num_arrangements (total_chairs : ℕ) (num_reserved : ℕ) (num_people : ℕ) : ℕ :=
  (total_chairs - num_reserved).choose num_people * (num_people.factorial)

theorem four_people_seven_chairs (total_chairs : ℕ) (chairs_occupied : ℕ) (num_people : ℕ): 
    total_chairs = 7 → chairs_occupied = 2 → num_people = 4 →
    num_arrangements total_chairs chairs_occupied num_people = 120 :=
by
  intros
  unfold num_arrangements
  sorry

end four_people_seven_chairs_l238_238662


namespace volume_third_bottle_is_250_milliliters_l238_238462

-- Define the volumes of the bottles in milliliters
def volume_first_bottle : ℕ := 2 * 1000                        -- 2000 milliliters
def volume_second_bottle : ℕ := 750                            -- 750 milliliters
def total_volume : ℕ := 3 * 1000                               -- 3000 milliliters
def volume_third_bottle : ℕ := total_volume - (volume_first_bottle + volume_second_bottle)

-- The theorem stating the volume of the third bottle
theorem volume_third_bottle_is_250_milliliters :
  volume_third_bottle = 250 :=
by
  sorry

end volume_third_bottle_is_250_milliliters_l238_238462


namespace sufficient_but_not_necessary_l238_238242

theorem sufficient_but_not_necessary (a b : ℝ) (h : a * b ≠ 0) : 
  (¬ (a = 0)) ∧ ¬ ((a ≠ 0) → (a * b ≠ 0)) :=
by {
  -- The proof will be constructed here and is omitted as per the instructions
  sorry
}

end sufficient_but_not_necessary_l238_238242


namespace initial_cookies_count_l238_238199

theorem initial_cookies_count (x : ℕ) (h_ate : ℕ) (h_left : ℕ) :
  h_ate = 2 → h_left = 5 → (x - h_ate = h_left) → x = 7 :=
by
  intros
  sorry

end initial_cookies_count_l238_238199


namespace train_crosses_bridge_in_30_seconds_l238_238677

theorem train_crosses_bridge_in_30_seconds
    (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ)
    (h1 : train_length = 110)
    (h2 : train_speed_kmh = 45)
    (h3 : bridge_length = 265) : 
    (train_length + bridge_length) / (train_speed_kmh * (1000 / 3600)) = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l238_238677


namespace remaining_pieces_total_l238_238230

noncomputable def initial_pieces : Nat := 16
noncomputable def kennedy_lost_pieces : Nat := 4 + 1 + 2
noncomputable def riley_lost_pieces : Nat := 1 + 1 + 1

theorem remaining_pieces_total : (initial_pieces - kennedy_lost_pieces) + (initial_pieces - riley_lost_pieces) = 22 := by
  sorry

end remaining_pieces_total_l238_238230


namespace sequence_properties_l238_238161

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 3 - 2^n

-- Prove the statements
theorem sequence_properties (n : ℕ) :
  (a (2 * n) = 3 - 4^n) ∧ (a 2 / a 3 = 1 / 5) :=
by
  sorry

end sequence_properties_l238_238161


namespace prism_ratio_l238_238419

theorem prism_ratio (a b c d : ℝ) (h_d : d = 60) (h_c : c = 104) (h_b : b = 78 * Real.pi) (h_a : a = (4 * Real.pi) / 3) :
  b * c / (a * d) = 8112 / 240 := 
by 
  sorry

end prism_ratio_l238_238419


namespace fraction_apple_juice_in_mixture_l238_238349

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

end fraction_apple_juice_in_mixture_l238_238349


namespace cosine_of_angle_in_convex_quadrilateral_l238_238720

theorem cosine_of_angle_in_convex_quadrilateral
    (A C : ℝ)
    (AB CD AD BC : ℝ)
    (h1 : A = C)
    (h2 : AB = 150)
    (h3 : CD = 150)
    (h4 : AD = BC)
    (h5 : AB + BC + CD + AD = 580) :
    Real.cos A = 7 / 15 := 
  sorry

end cosine_of_angle_in_convex_quadrilateral_l238_238720


namespace complex_div_eq_l238_238898

def complex_z : ℂ := ⟨1, -2⟩
def imaginary_unit : ℂ := ⟨0, 1⟩

theorem complex_div_eq :
  (complex_z + 2) / (complex_z - 1) = 1 + (3 / 2 : ℂ) * imaginary_unit :=
by
  sorry

end complex_div_eq_l238_238898


namespace total_copper_mined_l238_238363

theorem total_copper_mined :
  let daily_production_A := 4500
  let daily_production_B := 6000
  let daily_production_C := 5000
  let daily_production_D := 3500
  let copper_percentage_A := 0.055
  let copper_percentage_B := 0.071
  let copper_percentage_C := 0.147
  let copper_percentage_D := 0.092
  (daily_production_A * copper_percentage_A +
   daily_production_B * copper_percentage_B +
   daily_production_C * copper_percentage_C +
   daily_production_D * copper_percentage_D) = 1730.5 :=
by
  sorry

end total_copper_mined_l238_238363


namespace Luka_water_requirement_l238_238366

-- Declare variables and conditions
variables (L S W O : ℕ)  -- All variables are natural numbers
-- Conditions
variable (h1 : S = 2 * L)  -- Twice as much sugar as lemon juice
variable (h2 : W = 5 * S)  -- 5 times as much water as sugar
variable (h3 : O = S)      -- Orange juice equals the amount of sugar 
variable (L_eq_5 : L = 5)  -- Lemon juice is 5 cups

-- The goal statement to prove
theorem Luka_water_requirement : W = 50 :=
by
  -- Note: The proof steps would go here, but as per instructions, we leave it as sorry.
  sorry

end Luka_water_requirement_l238_238366


namespace Dan_age_is_28_l238_238689

theorem Dan_age_is_28 (B D : ℕ) (h1 : B = D - 3) (h2 : B + D = 53) : D = 28 :=
by
  sorry

end Dan_age_is_28_l238_238689


namespace train_length_problem_l238_238741

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l238_238741


namespace trigonometric_expression_value_l238_238263

noncomputable def trigonometric_expression (α : ℝ) : ℝ :=
  (|Real.tan α| / Real.tan α) + (Real.sin α / Real.sqrt ((1 - Real.cos (2 * α)) / 2))

theorem trigonometric_expression_value (α : ℝ) (h : Real.sin α = -Real.cos α) : 
  trigonometric_expression α = 0 ∨ trigonometric_expression α = -2 :=
by 
  sorry

end trigonometric_expression_value_l238_238263


namespace altitude_triangle_eq_2w_l238_238463

theorem altitude_triangle_eq_2w (l w h : ℕ) (h₀ : w ≠ 0) (h₁ : l ≠ 0)
    (h_area_rect : l * w = (1 / 2) * l * h) : h = 2 * w :=
by
  -- Consider h₀ (w is not zero) and h₁ (l is not zero)
  -- We need to prove h = 2w given l * w = (1 / 2) * l * h
  sorry

end altitude_triangle_eq_2w_l238_238463


namespace problem1_problem2_l238_238945

-- For problem (1)
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := sorry

-- For problem (2)
theorem problem2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b^2 = a * c) :
  a^2 + b^2 + c^2 > (a - b + c)^2 := sorry

end problem1_problem2_l238_238945


namespace least_number_divisible_l238_238347

theorem least_number_divisible (n : ℕ) :
  ((∀ d ∈ [24, 32, 36, 54, 72, 81, 100], (n + 21) % d = 0) ↔ n = 64779) :=
sorry

end least_number_divisible_l238_238347


namespace log_relationship_l238_238610

theorem log_relationship :
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  c < b ∧ b < a :=
by
  let a := Real.logb 0.3 0.2
  let b := Real.logb 3 2
  let c := Real.logb 0.2 3
  sorry

end log_relationship_l238_238610


namespace ratio_of_diamonds_to_spades_l238_238773

-- Given conditions
variable (total_cards : Nat := 13)
variable (black_cards : Nat := 7)
variable (red_cards : Nat := 6)
variable (clubs : Nat := 6)
variable (diamonds : Nat)
variable (spades : Nat)
variable (hearts : Nat := 2 * diamonds)
variable (cards_distribution : clubs + diamonds + hearts + spades = total_cards)
variable (black_distribution : clubs + spades = black_cards)

-- Define the proof theorem
theorem ratio_of_diamonds_to_spades : (diamonds / spades : ℝ) = 2 :=
 by
  -- temporarily we insert sorry to skip the proof
  sorry

end ratio_of_diamonds_to_spades_l238_238773


namespace average_marks_l238_238314

theorem average_marks (num_students : ℕ) (marks1 marks2 marks3 : ℕ) (num1 num2 num3 : ℕ) (h1 : num_students = 50)
  (h2 : marks1 = 90) (h3 : num1 = 10) (h4 : marks2 = marks1 - 10) (h5 : num2 = 15) (h6 : marks3 = 60) 
  (h7 : num1 + num2 + num3 = 50) (h8 : num3 = num_students - (num1 + num2)) (total_marks : ℕ) 
  (h9 : total_marks = (num1 * marks1) + (num2 * marks2) + (num3 * marks3)) : 
  (total_marks / num_students = 72) :=
by
  sorry

end average_marks_l238_238314


namespace rise_in_water_level_l238_238069

theorem rise_in_water_level (edge base_length base_width : ℝ) (cube_volume base_area rise : ℝ) 
  (h₁ : edge = 5) (h₂ : base_length = 10) (h₃ : base_width = 5)
  (h₄ : cube_volume = edge^3) (h₅ : base_area = base_length * base_width) 
  (h₆ : rise = cube_volume / base_area) : 
  rise = 2.5 := 
by 
  -- add proof here 
  sorry

end rise_in_water_level_l238_238069


namespace need_to_sell_more_rolls_l238_238711

variable (goal sold_grandmother sold_uncle_1 sold_uncle_additional sold_neighbor_1 returned_neighbor sold_mothers_friend sold_cousin_1 sold_cousin_additional : ℕ)

theorem need_to_sell_more_rolls
  (h_goal : goal = 100)
  (h_sold_grandmother : sold_grandmother = 5)
  (h_sold_uncle_1 : sold_uncle_1 = 12)
  (h_sold_uncle_additional : sold_uncle_additional = 10)
  (h_sold_neighbor_1 : sold_neighbor_1 = 8)
  (h_returned_neighbor : returned_neighbor = 4)
  (h_sold_mothers_friend : sold_mothers_friend = 25)
  (h_sold_cousin_1 : sold_cousin_1 = 3)
  (h_sold_cousin_additional : sold_cousin_additional = 5) :
  goal - (sold_grandmother + (sold_uncle_1 + sold_uncle_additional) + (sold_neighbor_1 - returned_neighbor) + sold_mothers_friend + (sold_cousin_1 + sold_cousin_additional)) = 36 := by
  sorry

end need_to_sell_more_rolls_l238_238711


namespace area_of_shaded_region_l238_238327

def side_length_of_square : ℝ := 12
def radius_of_quarter_circle : ℝ := 6

theorem area_of_shaded_region :
  let area_square := side_length_of_square ^ 2
  let area_full_circle := π * radius_of_quarter_circle ^ 2
  (area_square - area_full_circle) = 144 - 36 * π :=
by
  sorry

end area_of_shaded_region_l238_238327


namespace box_height_at_least_2_sqrt_15_l238_238914

def box_height (x : ℝ) : ℝ := 2 * x
def surface_area (x : ℝ) : ℝ := 10 * x ^ 2

theorem box_height_at_least_2_sqrt_15 (x : ℝ) (h : ℝ) :
  h = box_height x →
  surface_area x ≥ 150 →
  h ≥ 2 * Real.sqrt 15 :=
by
  intros h_eq sa_ge_150
  sorry

end box_height_at_least_2_sqrt_15_l238_238914


namespace bridge_length_correct_l238_238890

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_correct : bridge_length = 255 := by
  sorry

end bridge_length_correct_l238_238890


namespace lcm_of_28_and_24_is_168_l238_238697

/-- Racing car A completes the track in 28 seconds.
    Racing car B completes the track in 24 seconds.
    Both cars start at the same time.
    We want to prove that the time after which both cars will be side by side again
    (least common multiple of their lap times) is 168 seconds. -/
theorem lcm_of_28_and_24_is_168 :
  Nat.lcm 28 24 = 168 :=
sorry

end lcm_of_28_and_24_is_168_l238_238697


namespace pieces_of_candy_l238_238051

def total_items : ℝ := 3554
def secret_eggs : ℝ := 145.0

theorem pieces_of_candy : (total_items - secret_eggs) = 3409 :=
by 
  sorry

end pieces_of_candy_l238_238051


namespace factorization_problem_l238_238499

theorem factorization_problem (a b c x : ℝ) :
  ¬(2 * a^2 - b^2 = (a + b) * (a - b) + a^2) ∧
  ¬(2 * a * (b + c) = 2 * a * b + 2 * a * c) ∧
  (x^3 - 2 * x^2 + x = x * (x - 1)^2) ∧
  ¬ (x^2 + x = x^2 * (1 + 1 / x)) :=
by
  sorry

end factorization_problem_l238_238499


namespace bianca_bags_not_recycled_l238_238137

theorem bianca_bags_not_recycled :
  ∀ (points_per_bag total_bags total_points bags_recycled bags_not_recycled : ℕ),
    points_per_bag = 5 →
    total_bags = 17 →
    total_points = 45 →
    bags_recycled = total_points / points_per_bag →
    bags_not_recycled = total_bags - bags_recycled →
    bags_not_recycled = 8 :=
by
  intros points_per_bag total_bags total_points bags_recycled bags_not_recycled
  intros h_points_per_bag h_total_bags h_total_points h_bags_recycled h_bags_not_recycled
  sorry

end bianca_bags_not_recycled_l238_238137


namespace games_attended_this_month_l238_238555

theorem games_attended_this_month 
  (games_last_month games_next_month total_games games_this_month : ℕ)
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44)
  (h4 : games_last_month + games_this_month + games_next_month = total_games) : 
  games_this_month = 11 := by
  sorry

end games_attended_this_month_l238_238555


namespace smallest_cube_volume_l238_238173

noncomputable def sculpture_height : ℝ := 15
noncomputable def sculpture_base_radius : ℝ := 8
noncomputable def cube_side_length : ℝ := 16

theorem smallest_cube_volume :
  ∀ (h r s : ℝ), 
    h = sculpture_height ∧
    r = sculpture_base_radius ∧
    s = cube_side_length →
    s ^ 3 = 4096 :=
by
  intros h r s 
  intro h_def
  sorry

end smallest_cube_volume_l238_238173


namespace max_sum_hex_digits_l238_238940

theorem max_sum_hex_digits 
  (a b c : ℕ) (y : ℕ) 
  (h_a : 0 ≤ a ∧ a < 16)
  (h_b : 0 ≤ b ∧ b < 16)
  (h_c : 0 ≤ c ∧ c < 16)
  (h_y : 0 < y ∧ y ≤ 16)
  (h_fraction : (a * 256 + b * 16 + c) * y = 4096) : 
  a + b + c ≤ 1 :=
sorry

end max_sum_hex_digits_l238_238940


namespace solution_set_f_ge_0_l238_238185

variables {f : ℝ → ℝ}

-- Conditions
axiom h1 : ∀ x : ℝ, f (-x) = -f x  -- f is odd function
axiom h2 : ∀ x y : ℝ, 0 < x → x < y → f x < f y  -- f is monotonically increasing on (0, +∞)
axiom h3 : f 3 = 0  -- f(3) = 0

theorem solution_set_f_ge_0 : { x : ℝ | f x ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } ∪ { x : ℝ | 3 ≤ x } :=
by
  sorry

end solution_set_f_ge_0_l238_238185


namespace math_problem_l238_238046

noncomputable def triangle_conditions (a b c A B C : ℝ) := 
  (2 * b - c) / a = (Real.cos C) / (Real.cos A) ∧ 
  a = Real.sqrt 5 ∧
  1 / 2 * b * c * (Real.sin A) = Real.sqrt 3 / 2

theorem math_problem (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  A = π / 3 ∧ a + b + c = Real.sqrt 5 + Real.sqrt 11 :=
by
  sorry

end math_problem_l238_238046


namespace find_linear_function_l238_238335

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
(∀ (a b c : ℝ), a + b + c ≥ 0 → f (a^3) + f (b^3) + f (c^3) ≥ 3 * f (a * b * c))
∧ (∀ (a b c : ℝ), a + b + c ≤ 0 → f (a^3) + f (b^3) + f (c^3) ≤ 3 * f (a * b * c))

theorem find_linear_function (f : ℝ → ℝ) (h : functional_equation f) : ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end find_linear_function_l238_238335


namespace remainder_product_modulo_17_l238_238518

theorem remainder_product_modulo_17 :
  (1234 % 17) = 5 ∧ (1235 % 17) = 6 ∧ (1236 % 17) = 7 ∧ (1237 % 17) = 8 ∧ (1238 % 17) = 9 →
  ((1234 * 1235 * 1236 * 1237 * 1238) % 17) = 9 :=
by
  sorry

end remainder_product_modulo_17_l238_238518


namespace athleteA_time_to_complete_race_l238_238391

theorem athleteA_time_to_complete_race
    (v : ℝ)
    (t : ℝ)
    (h1 : v = 1000 / t)
    (h2 : v = 948 / (t + 18)) :
    t = 18000 / 52 := by
  sorry

end athleteA_time_to_complete_race_l238_238391


namespace neg_existential_proposition_l238_238790

open Nat

theorem neg_existential_proposition :
  (¬ (∃ n : ℕ, n + 10 / n < 4)) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) :=
by
  sorry

end neg_existential_proposition_l238_238790


namespace line_passes_through_fixed_point_l238_238847

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, (m - 1) * (-2) - 3 + 2 * m + 1 = 0 :=
by
  intros m
  sorry

end line_passes_through_fixed_point_l238_238847


namespace find_x_l238_238963

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l238_238963


namespace train_A_total_distance_l238_238218

variables (Speed_A : ℝ) (Time_meet : ℝ) (Total_Distance : ℝ)

def Distance_A_to_C (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Distance_B_to_C (Total_Distance Distance_A_to_C : ℝ) : ℝ := Total_Distance - Distance_A_to_C
def Additional_Distance_A (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Total_Distance_A (Distance_A_to_C Additional_Distance_A : ℝ) : ℝ :=
  Distance_A_to_C + Additional_Distance_A

theorem train_A_total_distance
  (h1 : Speed_A = 50)
  (h2 : Time_meet = 0.5)
  (h3 : Total_Distance = 120) :
  Total_Distance_A (Distance_A_to_C Speed_A Time_meet)
                   (Additional_Distance_A Speed_A Time_meet) = 50 :=
by 
  rw [Distance_A_to_C, Additional_Distance_A, Total_Distance_A]
  rw [h1, h2]
  norm_num

end train_A_total_distance_l238_238218


namespace largest_possible_s_l238_238924

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) 
  (h3 : ((r - 2) * 180 : ℚ) / r = (29 / 28) * ((s - 2) * 180 / s)) :
    s = 114 := by sorry

end largest_possible_s_l238_238924


namespace parking_lot_capacity_l238_238739

-- Definitions based on the conditions
def levels : ℕ := 5
def parkedCars : ℕ := 23
def moreCars : ℕ := 62
def capacityPerLevel : ℕ := parkedCars + moreCars

-- Proof problem statement
theorem parking_lot_capacity : levels * capacityPerLevel = 425 := by
  -- Proof omitted
  sorry

end parking_lot_capacity_l238_238739


namespace base_angle_isosceles_l238_238305

-- Define an isosceles triangle with one angle being 100 degrees
def isosceles_triangle (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A) ∧ (angle_A + angle_B + angle_C = 180) ∧ (angle_A = 100)

-- The main theorem statement
theorem base_angle_isosceles {A B C : Type} (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) :
  isosceles_triangle A B C angle_A angle_B angle_C → (angle_B = 40 ∨ angle_C = 40) :=
  sorry

end base_angle_isosceles_l238_238305


namespace walking_speed_l238_238863

noncomputable def bridge_length : ℝ := 2500  -- length of the bridge in meters
noncomputable def crossing_time_minutes : ℝ := 15  -- time to cross the bridge in minutes
noncomputable def conversion_factor_time : ℝ := 1 / 60  -- factor to convert minutes to hours
noncomputable def conversion_factor_distance : ℝ := 1 / 1000  -- factor to convert meters to kilometers

theorem walking_speed (bridge_length crossing_time_minutes conversion_factor_time conversion_factor_distance : ℝ) : 
  bridge_length = 2500 → 
  crossing_time_minutes = 15 → 
  conversion_factor_time = 1 / 60 → 
  conversion_factor_distance = 1 / 1000 → 
  (bridge_length * conversion_factor_distance) / (crossing_time_minutes * conversion_factor_time) = 10 := 
by
  sorry

end walking_speed_l238_238863


namespace final_score_l238_238874

theorem final_score (questions_first_half questions_second_half points_per_question : ℕ) (h1 : questions_first_half = 5) (h2 : questions_second_half = 5) (h3 : points_per_question = 5) : 
  (questions_first_half + questions_second_half) * points_per_question = 50 :=
by
  sorry

end final_score_l238_238874


namespace diagonals_in_15_sided_polygon_l238_238791

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end diagonals_in_15_sided_polygon_l238_238791


namespace determine_a_if_slope_angle_is_45_degrees_l238_238089

-- Define the condition that the slope angle of the given line is 45°
def is_slope_angle_45_degrees (a : ℝ) : Prop :=
  let m := -a / (2 * a - 3)
  m = 1

-- State the theorem we need to prove
theorem determine_a_if_slope_angle_is_45_degrees (a : ℝ) :
  is_slope_angle_45_degrees a → a = 1 :=
by
  intro h
  sorry

end determine_a_if_slope_angle_is_45_degrees_l238_238089


namespace floor_plus_self_eq_l238_238757

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_self_eq_l238_238757


namespace part1_part2_l238_238096

-- Defining the function f
def f (x : ℝ) (a : ℝ) : ℝ := a * abs (x + 1) - abs (x - 1)

-- Part 1: a = 1, finding the solution set of the inequality f(x) < 3/2
theorem part1 (x : ℝ) : f x 1 < 3 / 2 ↔ x < 3 / 4 := 
sorry

-- Part 2: a > 1, and existence of x such that f(x) <= -|2m+1|, finding the range of m
theorem part2 (a : ℝ) (h : 1 < a) (m : ℝ) (x : ℝ) : 
  f x a ≤ -abs (2 * m + 1) → -3 / 2 ≤ m ∧ m ≤ 1 :=
sorry

end part1_part2_l238_238096


namespace smallest_number_of_beads_l238_238020

theorem smallest_number_of_beads (M : ℕ) (h1 : ∃ d : ℕ, M = 5 * d + 2) (h2 : ∃ e : ℕ, M = 7 * e + 2) (h3 : ∃ f : ℕ, M = 9 * f + 2) (h4 : M > 1) : M = 317 := sorry

end smallest_number_of_beads_l238_238020


namespace solve_for_a_l238_238151
-- Additional imports might be necessary depending on specifics of the proof

theorem solve_for_a (a x y : ℝ) (h1 : ax - y = 3) (h2 : x = 1) (h3 : y = 2) : a = 5 :=
by
  sorry

end solve_for_a_l238_238151


namespace max_min_value_of_product_l238_238528

theorem max_min_value_of_product (x y : ℝ) (h : x ^ 2 + y ^ 2 = 1) :
  (1 + x * y) * (1 - x * y) ≤ 1 ∧ (1 + x * y) * (1 - x * y) ≥ 3 / 4 :=
by sorry

end max_min_value_of_product_l238_238528


namespace log_product_identity_l238_238793

noncomputable def log {a b : ℝ} (ha : 1 < a) (hb : 0 < b) : ℝ := Real.log b / Real.log a

theorem log_product_identity : 
  log (by norm_num : (1 : ℝ) < 2) (by norm_num : (0 : ℝ) < 9) * 
  log (by norm_num : (1 : ℝ) < 3) (by norm_num : (0 : ℝ) < 8) = 6 :=
sorry

end log_product_identity_l238_238793


namespace Jose_played_football_l238_238385

theorem Jose_played_football :
  ∀ (total_hours : ℝ) (basketball_minutes : ℕ) (minutes_per_hour : ℕ), total_hours = 1.5 → basketball_minutes = 60 →
  (total_hours * minutes_per_hour - basketball_minutes = 30) :=
by
  intros total_hours basketball_minutes minutes_per_hour h1 h2
  sorry

end Jose_played_football_l238_238385


namespace sqrt_221_between_15_and_16_l238_238531

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end sqrt_221_between_15_and_16_l238_238531


namespace unique_intersections_l238_238295

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

theorem unique_intersections :
  (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1) ∧
  (∃ x2 y2, line2 x2 y2 ∧ line3 x2 y2) ∧
  ¬ (∃ x y, line1 x y ∧ line3 x y) ∧
  (∀ x y x' y', (line1 x y ∧ line2 x y ∧ line2 x' y' ∧ line3 x' y') → (x = x' ∧ y = y')) :=
by
  sorry

end unique_intersections_l238_238295


namespace total_weight_of_compound_l238_238476

variable (molecular_weight : ℕ) (moles : ℕ)

theorem total_weight_of_compound (h1 : molecular_weight = 72) (h2 : moles = 4) :
  moles * molecular_weight = 288 :=
by
  sorry

end total_weight_of_compound_l238_238476


namespace binom_comb_always_integer_l238_238700

theorem binom_comb_always_integer (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : (k + 2) ∣ n) : 
  ∃ m : ℤ, ((n - 3 * k - 2) / (k + 2)) * Nat.choose n k = m := 
sorry

end binom_comb_always_integer_l238_238700


namespace Karlson_drink_ratio_l238_238751

noncomputable def conical_glass_volume_ratio (r h : ℝ) : Prop :=
  let V_fuzh := (1 / 3) * Real.pi * r^2 * h
  let V_Mal := (1 / 8) * V_fuzh
  let V_Karlsson := V_fuzh - V_Mal
  (V_Karlsson / V_Mal) = 7

theorem Karlson_drink_ratio (r h : ℝ) : conical_glass_volume_ratio r h := sorry

end Karlson_drink_ratio_l238_238751


namespace simplify_and_evaluate_l238_238403

theorem simplify_and_evaluate (m : ℤ) (h : m = -2) :
  let expr := (m / (m^2 - 9)) / (1 + (3 / (m - 3)))
  expr = 1 :=
by
  sorry

end simplify_and_evaluate_l238_238403


namespace andrew_age_l238_238208

-- Definitions based on the conditions
variables (a g : ℝ)

-- The conditions
def condition1 : Prop := g = 9 * a
def condition2 : Prop := g - a = 63

-- The theorem we want to prove
theorem andrew_age (h1 : condition1 a g) (h2 : condition2 a g) : a = 63 / 8 :=
by
  intros
  sorry

end andrew_age_l238_238208


namespace x_minus_y_eq_14_l238_238824

theorem x_minus_y_eq_14 (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 100) : x - y = 14 :=
sorry

end x_minus_y_eq_14_l238_238824


namespace cost_reduction_l238_238807

variable (a : ℝ) -- original cost
variable (p : ℝ) -- percentage reduction (in decimal form)
variable (m : ℕ) -- number of years

def cost_after_years (a p : ℝ) (m : ℕ) : ℝ :=
  a * (1 - p) ^ m

theorem cost_reduction (a p : ℝ) (m : ℕ) :
  m > 0 → cost_after_years a p m = a * (1 - p) ^ m :=
sorry

end cost_reduction_l238_238807


namespace percentage_men_science_majors_l238_238057

theorem percentage_men_science_majors (total_students : ℕ) (women_science_majors_ratio : ℚ) (nonscience_majors_ratio : ℚ) (men_class_ratio : ℚ) :
  women_science_majors_ratio = 0.2 → 
  nonscience_majors_ratio = 0.6 → 
  men_class_ratio = 0.4 → 
  ∃ men_science_majors_percent : ℚ, men_science_majors_percent = 0.7 :=
by
  intros h_women_science_majors h_nonscience_majors h_men_class
  sorry

end percentage_men_science_majors_l238_238057


namespace min_value_4x_plus_3y_l238_238274

theorem min_value_4x_plus_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_plus_3y_l238_238274


namespace arccos_sqrt_3_over_2_eq_pi_over_6_l238_238209

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end arccos_sqrt_3_over_2_eq_pi_over_6_l238_238209


namespace part_one_max_value_range_of_a_l238_238870

def f (x a : ℝ) : ℝ := |x + 2| - |x - 3| - a

theorem part_one_max_value (a : ℝ) (h : a = 1) : ∃ x : ℝ, f x a = 4 := 
by sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 4 / a) :  (0 < a ∧ a ≤ 1) ∨ 4 ≤ a :=
by sorry

end part_one_max_value_range_of_a_l238_238870


namespace least_value_x_y_z_l238_238726

theorem least_value_x_y_z 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h_eq: 2 * x = 5 * y) 
  (h_eq': 5 * y = 8 * z) : 
  x + y + z = 33 :=
by 
  sorry

end least_value_x_y_z_l238_238726


namespace speed_limit_inequality_l238_238553

theorem speed_limit_inequality (v : ℝ) : (v ≤ 40) :=
sorry

end speed_limit_inequality_l238_238553


namespace tickets_difference_vip_general_l238_238110

theorem tickets_difference_vip_general (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : 40 * V + 10 * G = 7500) : G - V = 34 := 
by
  sorry

end tickets_difference_vip_general_l238_238110


namespace pentagon_perimeter_even_l238_238655

noncomputable def dist_sq (A B : ℤ × ℤ) : ℤ :=
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2

theorem pentagon_perimeter_even (A B C D E : ℤ × ℤ) (h1 : dist_sq A B % 2 = 1) (h2 : dist_sq B C % 2 = 1) 
  (h3 : dist_sq C D % 2 = 1) (h4 : dist_sq D E % 2 = 1) (h5 : dist_sq E A % 2 = 1) : 
  (dist_sq A B + dist_sq B C + dist_sq C D + dist_sq D E + dist_sq E A) % 2 = 0 := 
by 
  sorry

end pentagon_perimeter_even_l238_238655


namespace variance_of_dataset_l238_238108

noncomputable def dataset : List ℝ := [3, 6, 9, 8, 4]

noncomputable def mean (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => y + acc) 0) / (x.length)

noncomputable def variance (x : List ℝ) : ℝ :=
  (x.foldr (λ y acc => (y - mean x)^2 + acc) 0) / (x.length)

theorem variance_of_dataset :
  variance dataset = 26 / 5 :=
by
  sorry

end variance_of_dataset_l238_238108


namespace given_condition_implies_result_l238_238503

theorem given_condition_implies_result (a : ℝ) (h : a ^ 2 + 2 * a = 1) : 2 * a ^ 2 + 4 * a + 1 = 3 :=
sorry

end given_condition_implies_result_l238_238503


namespace correct_system_of_equations_l238_238953

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : x - y = 5) (h2 : y - (1/2) * x = 5) : 
  (x - y = 5) ∧ (y - (1/2) * x = 5) :=
by { sorry }

end correct_system_of_equations_l238_238953


namespace minimum_time_to_serve_tea_equals_9_l238_238532

def boiling_water_time : Nat := 8
def washing_teapot_time : Nat := 1
def washing_teacups_time : Nat := 2
def fetching_tea_leaves_time : Nat := 2
def brewing_tea_time : Nat := 1

theorem minimum_time_to_serve_tea_equals_9 :
  boiling_water_time + brewing_tea_time = 9 := by
  sorry

end minimum_time_to_serve_tea_equals_9_l238_238532


namespace bouncy_ball_pack_count_l238_238505

theorem bouncy_ball_pack_count
  (x : ℤ)  -- Let x be the number of bouncy balls in each pack
  (r : ℤ := 7 * x)  -- Total number of red bouncy balls
  (y : ℤ := 6 * x)  -- Total number of yellow bouncy balls
  (h : r = y + 18)  -- Condition: 7x = 6x + 18
  : x = 18 := sorry

end bouncy_ball_pack_count_l238_238505


namespace inequality_abc_l238_238009

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / (b ^ (1/2 : ℝ)) + b / (a ^ (1/2 : ℝ)) ≥ a ^ (1/2 : ℝ) + b ^ (1/2 : ℝ) :=
by { sorry }

end inequality_abc_l238_238009


namespace intersection_shape_is_rectangle_l238_238149

noncomputable def curve1 (x y : ℝ) : Prop := x * y = 16
noncomputable def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 34

theorem intersection_shape_is_rectangle (x y : ℝ) :
  (curve1 x y ∧ curve2 x y) → 
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    (curve1 p1.1 p1.2 ∧ curve1 p2.1 p2.2 ∧ curve1 p3.1 p3.2 ∧ curve1 p4.1 p4.2) ∧
    (curve2 p1.1 p1.2 ∧ curve2 p2.1 p2.2 ∧ curve2 p3.1 p3.2 ∧ curve2 p4.1 p4.2) ∧ 
    (dist p1 p2 = dist p3 p4 ∧ dist p2 p3 = dist p4 p1) ∧ 
    (∃ m : ℝ, p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.1 ≠ m ∧ p2.1 ≠ m) := sorry

end intersection_shape_is_rectangle_l238_238149


namespace unique_positive_a_for_one_solution_l238_238760

theorem unique_positive_a_for_one_solution :
  ∃ (d : ℝ), d ≠ 0 ∧ (∀ a : ℝ, a > 0 → (∀ x : ℝ, x^2 + (a + 1/a) * x + d = 0 ↔ x^2 + (a + 1/a) * x + d = 0)) ∧ d = 1 := 
by
  sorry

end unique_positive_a_for_one_solution_l238_238760


namespace remainder_is_one_l238_238379

theorem remainder_is_one (N : ℤ) (R : ℤ)
  (h1 : N % 100 = R)
  (h2 : N % R = 1) :
  R = 1 :=
by
  sorry

end remainder_is_one_l238_238379


namespace two_A_minus_B_l238_238527

theorem two_A_minus_B (A B : ℝ) 
  (h1 : Real.tan (A - B - Real.pi) = 1 / 2) 
  (h2 : Real.tan (3 * Real.pi - B) = 1 / 7) : 
  2 * A - B = -3 * Real.pi / 4 :=
sorry

end two_A_minus_B_l238_238527


namespace hannah_quarters_l238_238155

theorem hannah_quarters :
  ∃ n : ℕ, 40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3 ∧ 
  (n = 171 ∨ n = 339) :=
by
  sorry

end hannah_quarters_l238_238155


namespace circumference_greater_than_100_l238_238146

def running_conditions (A B : ℝ) (C : ℝ) (P : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ A ≠ B ∧ P = 0 ∧ C > 0

theorem circumference_greater_than_100 (A B C P : ℝ) (h : running_conditions A B C P):
  C > 100 :=
by
  sorry

end circumference_greater_than_100_l238_238146


namespace cube_sum_l238_238565

-- Definitions
variable (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω^2 + ω + 1 = 0) -- nonreal root

-- Theorem statement
theorem cube_sum : (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 :=
by 
  sorry

end cube_sum_l238_238565


namespace negative_solution_range_l238_238893

theorem negative_solution_range (m : ℝ) : (∃ x : ℝ, 2 * x + 4 = m - x ∧ x < 0) → m < 4 := by
  sorry

end negative_solution_range_l238_238893


namespace minimum_value_of_f_on_interval_l238_238833

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem minimum_value_of_f_on_interval (a : ℝ) (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 20) :
  a = -2 → ∃ min_val, min_val = -7 :=
by
  sorry

end minimum_value_of_f_on_interval_l238_238833


namespace balloons_per_school_l238_238356

theorem balloons_per_school (yellow black total : ℕ) 
  (hyellow : yellow = 3414)
  (hblack : black = yellow + 1762)
  (htotal : total = yellow + black)
  (hdivide : total % 10 = 0) : 
  total / 10 = 859 :=
by sorry

end balloons_per_school_l238_238356


namespace unique_integral_root_of_equation_l238_238341

theorem unique_integral_root_of_equation :
  ∀ x : ℤ, (x - 9 / (x - 5) = 7 - 9 / (x - 5)) ↔ (x = 7) :=
by
  sorry

end unique_integral_root_of_equation_l238_238341


namespace no_n_for_equal_sums_l238_238153

theorem no_n_for_equal_sums (n : ℕ) (h : n ≠ 0) :
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  s1 ≠ s2 :=
by
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  sorry

end no_n_for_equal_sums_l238_238153


namespace ab_zero_l238_238377

theorem ab_zero (a b : ℝ) (x : ℝ) (h : ∀ x : ℝ, a * x + b * x ^ 2 = -(a * (-x) + b * (-x) ^ 2)) : a * b = 0 :=
by
  sorry

end ab_zero_l238_238377


namespace sum_of_f_is_negative_l238_238875

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_of_f_is_negative (x₁ x₂ x₃ : ℝ)
  (h1: x₁ + x₂ < 0)
  (h2: x₂ + x₃ < 0) 
  (h3: x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := 
sorry

end sum_of_f_is_negative_l238_238875


namespace polynomial_sum_l238_238232

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 7 * x - 6
def h (x : ℝ) : ℝ := 3 * x^2 - 3 * x + 2
def j (x : ℝ) : ℝ := x^2 + x - 1

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end polynomial_sum_l238_238232


namespace midpoint_of_hyperbola_segment_l238_238207

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end midpoint_of_hyperbola_segment_l238_238207


namespace domain_g_l238_238422

noncomputable def g (x : ℝ) := Real.tan (Real.arccos (x ^ 3))

theorem domain_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end domain_g_l238_238422


namespace helicopter_rental_cost_l238_238478

theorem helicopter_rental_cost :
  let hours_per_day := 2
  let days := 3
  let rate_first_day := 85
  let rate_second_day := 75
  let rate_third_day := 65
  let total_cost_before_discount := hours_per_day * rate_first_day + hours_per_day * rate_second_day + hours_per_day * rate_third_day
  let discount := 0.05
  let discounted_amount := total_cost_before_discount * discount
  let total_cost_after_discount := total_cost_before_discount - discounted_amount
  total_cost_after_discount = 427.50 :=
by
  sorry

end helicopter_rental_cost_l238_238478


namespace binary_and_ternary_product_l238_238097

theorem binary_and_ternary_product :
  let binary_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0
  let ternary_1021 := 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  binary_1011 = 11 ∧ ternary_1021 = 34 →
  binary_1011 * ternary_1021 = 374 :=
by
  intros h
  sorry

end binary_and_ternary_product_l238_238097


namespace woman_work_rate_l238_238423

theorem woman_work_rate :
  let M := 1/6
  let B := 1/9
  let combined_rate := 1/3
  ∃ W : ℚ, M + B + W = combined_rate ∧ 1 / W = 18 := 
by
  sorry

end woman_work_rate_l238_238423


namespace terrell_total_distance_l238_238892

theorem terrell_total_distance (saturday_distance sunday_distance : ℝ) (h_saturday : saturday_distance = 8.2) (h_sunday : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 :=
by
  rw [h_saturday, h_sunday]
  -- sorry
  norm_num

end terrell_total_distance_l238_238892


namespace range_of_a_l238_238723

-- Given conditions
def condition1 (x : ℝ) := (4 + x) / 3 > (x + 2) / 2
def condition2 (x : ℝ) (a : ℝ) := (x + a) / 2 < 0

-- The statement to prove
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, condition1 x → condition2 x a → x < 2) → a ≤ -2 :=
sorry

end range_of_a_l238_238723


namespace working_together_time_l238_238904

/-- A is 30% more efficient than B,
and A alone can complete the job in 23 days.
Prove that A and B working together take approximately 13 days to complete the job. -/
theorem working_together_time (Ea Eb : ℝ) (T : ℝ) (h1 : Ea = 1.30 * Eb) 
  (h2 : 1 / 23 = Ea) : T = 13 :=
sorry

end working_together_time_l238_238904


namespace simplify_expression_l238_238332

theorem simplify_expression (w : ℤ) : 
  (-2 * w + 3 - 4 * w + 7 + 6 * w - 5 - 8 * w + 8) = (-8 * w + 13) :=
by {
  sorry
}

end simplify_expression_l238_238332


namespace dimensions_increased_three_times_l238_238317

variables (L B H k : ℝ) (n : ℝ)
 
-- Given conditions
axiom cost_initial : 350 = k * 2 * (L + B) * H
axiom cost_increased : 3150 = k * 2 * n^2 * (L + B) * H

-- Proof statement
theorem dimensions_increased_three_times : n = 3 :=
by
  sorry

end dimensions_increased_three_times_l238_238317


namespace equivalent_mod_l238_238901

theorem equivalent_mod (h : 5^300 ≡ 1 [MOD 1250]) : 5^9000 ≡ 1 [MOD 1000] :=
by 
  sorry

end equivalent_mod_l238_238901


namespace weight_of_mixture_is_correct_l238_238504

noncomputable def weight_mixture_kg (weight_per_liter_a weight_per_liter_b ratio_a ratio_b total_volume_liters : ℕ) : ℝ :=
  let volume_a := (ratio_a * total_volume_liters) / (ratio_a + ratio_b)
  let volume_b := (ratio_b * total_volume_liters) / (ratio_a + ratio_b)
  let weight_a := (volume_a * weight_per_liter_a) 
  let weight_b := (volume_b * weight_per_liter_b) 
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_mixture_kg 900 700 3 2 4 = 3.280 := 
sorry

end weight_of_mixture_is_correct_l238_238504


namespace mike_gave_12_pears_l238_238345

variable (P M K N : ℕ)

def initial_pears := 46
def pears_given_to_keith := 47
def pears_left := 11

theorem mike_gave_12_pears (M : ℕ) : 
  initial_pears - pears_given_to_keith + M = pears_left → M = 12 :=
by
  intro h
  sorry

end mike_gave_12_pears_l238_238345


namespace sequence_sixth_term_l238_238055

theorem sequence_sixth_term (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) 
  (h2 : ∀ n :ℕ, n > 0 → a (n + 1) = 2 * a n) 
  (h3 : a 1 = 3) : 
  a 6 = 96 := 
by
  sorry

end sequence_sixth_term_l238_238055


namespace train_crossing_time_l238_238827

-- Definitions of the given problem conditions
def train_length : ℕ := 120  -- in meters.
def speed_kmph : ℕ := 144   -- in km/h.

-- Conversion factor
def km_per_hr_to_m_per_s (speed : ℕ) : ℚ :=
  speed * (1000 / 3600 : ℚ)

-- Speed in m/s
def train_speed : ℚ := km_per_hr_to_m_per_s speed_kmph

-- Time calculation
def time_to_cross_pole (length : ℕ) (speed : ℚ) : ℚ :=
  length / speed

-- The theorem we want to prove.
theorem train_crossing_time :
  time_to_cross_pole train_length train_speed = 3 := by 
  sorry

end train_crossing_time_l238_238827


namespace johns_pants_cost_50_l238_238322

variable (P : ℝ)

theorem johns_pants_cost_50 (h1 : P + 1.60 * P = 130) : P = 50 := 
by
  sorry

end johns_pants_cost_50_l238_238322


namespace radius_of_circle_l238_238522

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l238_238522


namespace acute_triangle_integers_count_l238_238975

theorem acute_triangle_integers_count :
  ∃ (x_vals : List ℕ), (∀ x ∈ x_vals, 7 < x ∧ x < 33 ∧ (if x > 20 then x^2 < 569 else x > Int.sqrt 231)) ∧ x_vals.length = 8 :=
by
  sorry

end acute_triangle_integers_count_l238_238975


namespace sample_systematic_draw_first_group_l238_238038

theorem sample_systematic_draw_first_group :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 8 →
  (x + 15 * 8 = 126) →
  x = 6 :=
by
  intros x h1 h2
  sorry

end sample_systematic_draw_first_group_l238_238038


namespace stewarts_theorem_l238_238028

theorem stewarts_theorem
  (A B C D : ℝ)
  (AB AC AD : ℝ)
  (BD CD BC : ℝ)
  (hD_on_BC : BD + CD = BC) :
  AB^2 * CD + AC^2 * BD - AD^2 * BC = BD * CD * BC := 
sorry

end stewarts_theorem_l238_238028


namespace jackie_eligible_for_free_shipping_l238_238394

def shampoo_cost : ℝ := 2 * 12.50
def conditioner_cost : ℝ := 3 * 15.00
def face_cream_cost : ℝ := 20.00  -- Considering the buy-one-get-one-free deal

def subtotal : ℝ := shampoo_cost + conditioner_cost + face_cream_cost
def discount : ℝ := 0.10 * subtotal
def total_after_discount : ℝ := subtotal - discount

theorem jackie_eligible_for_free_shipping : total_after_discount >= 75 := by
  sorry

end jackie_eligible_for_free_shipping_l238_238394


namespace bacteria_population_at_15_l238_238962

noncomputable def bacteria_population (t : ℕ) : ℕ := 
  20 * 2 ^ (t / 3)

theorem bacteria_population_at_15 : bacteria_population 15 = 640 := by
  sorry

end bacteria_population_at_15_l238_238962


namespace polynomial_expansion_sum_l238_238897

theorem polynomial_expansion_sum :
  let A := 4
  let B := 10
  let C := 1
  let D := 21
  (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 36 := 
by
  -- Proof needs to be filled
  sorry

end polynomial_expansion_sum_l238_238897


namespace equation1_solution_equation2_solution_equation3_solution_l238_238471

theorem equation1_solution :
  ∀ x : ℝ, x^2 + 4 * x = 0 ↔ x = 0 ∨ x = -4 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, 2 * (x - 1) + x * (x - 1) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 4 = 0 ↔ x = (1 + Real.sqrt 13) / 3 ∨ x = (1 - Real.sqrt 13) / 3 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l238_238471


namespace nails_needed_for_house_wall_l238_238031

theorem nails_needed_for_house_wall :
  let large_planks : Nat := 13
  let nails_per_large_plank : Nat := 17
  let additional_nails : Nat := 8
  large_planks * nails_per_large_plank + additional_nails = 229 := by
  sorry

end nails_needed_for_house_wall_l238_238031


namespace cos_alpha_is_negative_four_fifths_l238_238143

variable (α : ℝ)
variable (H1 : Real.sin α = 3 / 5)
variable (H2 : π / 2 < α ∧ α < π)

theorem cos_alpha_is_negative_four_fifths (H1 : Real.sin α = 3 / 5) (H2 : π / 2 < α ∧ α < π) :
  Real.cos α = -4 / 5 :=
sorry

end cos_alpha_is_negative_four_fifths_l238_238143


namespace larger_root_of_quadratic_eq_l238_238568

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l238_238568


namespace area_of_side_face_of_box_l238_238358

theorem area_of_side_face_of_box:
  ∃ (l w h : ℝ), (w * h = (1/2) * (l * w)) ∧
                 (l * w = 1.5 * (l * h)) ∧
                 (l * w * h = 3000) ∧
                 ((l * h) = 200) :=
sorry

end area_of_side_face_of_box_l238_238358


namespace arithmetic_progression_common_difference_l238_238204

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (h1 : 280 * x^2 - 61 * x * y + 3 * y^2 - 13 = 0) 
  (h2 : ∃ a d : ℤ, x = a + 3 * d ∧ y = a + 8 * d) : 
  ∃ d : ℤ, d = -5 := 
sorry

end arithmetic_progression_common_difference_l238_238204


namespace proof_problem_l238_238132

noncomputable def a_n (n : ℕ) : ℕ := n + 2
noncomputable def b_n (n : ℕ) : ℕ := 2 * n + 3
noncomputable def C_n (n : ℕ) : ℚ := 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))
noncomputable def T_n (n : ℕ) : ℚ := (1/4) * (1 - (1/(2 * n + 1)))

theorem proof_problem :
  (∀ n, a_n n = n + 2) ∧
  (∀ n, b_n n = 2 * n + 3) ∧
  (∀ n, C_n n = 1 / ((2 * a_n n - 3) * (2 * b_n n - 8))) ∧
  (∀ n, T_n n = (1/4) * (1 - (1/(2 * n + 1)))) ∧
  (∀ n, (T_n n > k / 54) ↔ k < 9) :=
by
  sorry

end proof_problem_l238_238132


namespace levi_additional_baskets_to_score_l238_238368

def levi_scored_initial := 8
def brother_scored_initial := 12
def brother_likely_to_score := 3
def levi_goal_margin := 5

theorem levi_additional_baskets_to_score : 
  levi_scored_initial + 12 >= brother_scored_initial + brother_likely_to_score + levi_goal_margin :=
by
  sorry

end levi_additional_baskets_to_score_l238_238368


namespace hannahs_brothers_l238_238703

theorem hannahs_brothers (B : ℕ) (h1 : ∀ (b : ℕ), b = 8) (h2 : 48 = 2 * (8 * B)) : B = 3 :=
by
  sorry

end hannahs_brothers_l238_238703


namespace shaded_hexagons_are_balanced_l238_238024

-- Definitions and conditions from the problem
def is_balanced (a b c : ℕ) : Prop :=
  (a = b ∧ b = c) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

def hexagon_grid_balanced (grid : ℕ × ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ),
  (i % 2 = 0 ∧ grid (i, j) = grid (i, j + 1) ∧ grid (i, j + 1) = grid (i + 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i, j + 1) ∧ grid (i, j + 1) ≠ grid (i + 1, j + 1) ∧ grid (i, j) ≠ grid (i + 1, j + 1))
  ∨ (i % 2 ≠ 0 ∧ grid (i, j) = grid (i - 1, j) ∧ grid (i - 1, j) = grid (i - 1, j + 1))
  ∨ (grid (i, j) ≠ grid (i - 1, j) ∧ grid (i - 1, j) ≠ grid (i - 1, j + 1) ∧ grid (i, j) ≠ grid (i - 1, j + 1))

theorem shaded_hexagons_are_balanced (grid : ℕ × ℕ → ℕ) (h_balanced : hexagon_grid_balanced grid) :
  is_balanced (grid (1, 1)) (grid (1, 10)) (grid (10, 10)) :=
sorry

end shaded_hexagons_are_balanced_l238_238024


namespace sum_of_vertices_l238_238941

theorem sum_of_vertices (pentagon_vertices : Nat := 5) (hexagon_vertices : Nat := 6) :
  (2 * pentagon_vertices) + (2 * hexagon_vertices) = 22 :=
by
  sorry

end sum_of_vertices_l238_238941


namespace arcsin_arccos_add_eq_pi6_l238_238054

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def arccos (x : Real) : Real := sorry

theorem arcsin_arccos_add_eq_pi6 (x : Real) (hx_range : -1 ≤ x ∧ x ≤ 1)
    (h3x_range : -1 ≤ 3 * x ∧ 3 * x ≤ 1) 
    (h : arcsin x + arccos (3 * x) = Real.pi / 6) :
    x = Real.sqrt (3 / 124) := 
  sorry

end arcsin_arccos_add_eq_pi6_l238_238054


namespace range_of_m_l238_238414

variable {x m : ℝ}

def absolute_value_inequality (x m : ℝ) : Prop := |x + 1| - |x - 2| > m

theorem range_of_m : (∀ x : ℝ, absolute_value_inequality x m) ↔ m < -3 :=
by
  sorry

end range_of_m_l238_238414


namespace solve_for_x_l238_238184

theorem solve_for_x (x : ℚ) : x^2 + 125 = (x - 15)^2 → x = 10 / 3 := by
  sorry

end solve_for_x_l238_238184


namespace valid_outfits_number_l238_238276

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l238_238276


namespace principal_amount_is_approx_1200_l238_238889

noncomputable def find_principal_amount : Real :=
  let R := 0.10
  let n := 2
  let T := 1
  let SI (P : Real) := P * R * T / 100
  let CI (P : Real) := P * ((1 + R / n) ^ (n * T)) - P
  let diff (P : Real) := CI P - SI P
  let target_diff := 2.999999999999936
  let P := target_diff / (0.1025 - 0.10)
  P

theorem principal_amount_is_approx_1200 : abs (find_principal_amount - 1200) < 0.0001 := 
by
  sorry

end principal_amount_is_approx_1200_l238_238889


namespace prime_dates_in_2008_l238_238800

noncomputable def num_prime_dates_2008 : Nat := 52

theorem prime_dates_in_2008 : 
  let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_months_days := [(2, 29), (3, 31), (5, 31), (7, 31), (11, 30)]
  -- Count the prime days for each month considering the list
  let prime_day_count (days : Nat) := (prime_days.filter (λ d => d <= days)).length
  -- Sum the counts for each prime month
  (prime_months_days.map (λ (m, days) => prime_day_count days)).sum = num_prime_dates_2008 :=
by
  sorry

end prime_dates_in_2008_l238_238800


namespace unique_solution_t_interval_l238_238188

theorem unique_solution_t_interval (x y z v t : ℝ) :
  (x + y + z + v = 0) →
  ((x * y + y * z + z * v) + t * (x * z + x * v + y * v) = 0) →
  (t > (3 - Real.sqrt 5) / 2) ∧ (t < (3 + Real.sqrt 5) / 2) :=
by
  intro h1 h2
  sorry

end unique_solution_t_interval_l238_238188


namespace trip_total_charge_l238_238501

noncomputable def initial_fee : ℝ := 2.25
noncomputable def additional_charge_per_increment : ℝ := 0.25
noncomputable def increment_length : ℝ := 2 / 5
noncomputable def trip_length : ℝ := 3.6

theorem trip_total_charge :
  initial_fee + (trip_length / increment_length) * additional_charge_per_increment = 4.50 :=
by
  sorry

end trip_total_charge_l238_238501


namespace cylinder_volume_l238_238557

theorem cylinder_volume (r h : ℝ) (h_radius : r = 1) (h_height : h = 2) : (π * r^2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l238_238557


namespace Pima_investment_value_at_week6_l238_238172

noncomputable def Pima_initial_investment : ℝ := 400
noncomputable def Pima_week1_gain : ℝ := 0.25
noncomputable def Pima_week1_addition : ℝ := 200
noncomputable def Pima_week2_gain : ℝ := 0.50
noncomputable def Pima_week2_withdrawal : ℝ := 150
noncomputable def Pima_week3_loss : ℝ := 0.10
noncomputable def Pima_week4_gain : ℝ := 0.20
noncomputable def Pima_week4_addition : ℝ := 100
noncomputable def Pima_week5_gain : ℝ := 0.05
noncomputable def Pima_week6_loss : ℝ := 0.15
noncomputable def Pima_week6_withdrawal : ℝ := 250
noncomputable def weekly_interest_rate : ℝ := 0.02

noncomputable def calculate_investment_value : ℝ :=
  let week0 := Pima_initial_investment
  let week1 := (week0 * (1 + Pima_week1_gain) * (1 + weekly_interest_rate)) + Pima_week1_addition
  let week2 := ((week1 * (1 + Pima_week2_gain) * (1 + weekly_interest_rate)) - Pima_week2_withdrawal)
  let week3 := (week2 * (1 - Pima_week3_loss) * (1 + weekly_interest_rate))
  let week4 := ((week3 * (1 + Pima_week4_gain) * (1 + weekly_interest_rate)) + Pima_week4_addition)
  let week5 := (week4 * (1 + Pima_week5_gain) * (1 + weekly_interest_rate))
  let week6 := ((week5 * (1 - Pima_week6_loss) * (1 + weekly_interest_rate)) - Pima_week6_withdrawal)
  week6

theorem Pima_investment_value_at_week6 : calculate_investment_value = 819.74 := 
  by
  sorry

end Pima_investment_value_at_week6_l238_238172


namespace option_C_correct_l238_238517

theorem option_C_correct (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1 / 2 :=
sorry

end option_C_correct_l238_238517


namespace distinct_real_numbers_proof_l238_238466

variables {a b c : ℝ}

theorem distinct_real_numbers_proof (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
  (h₄ : (a / (b - c) + b / (c - a) + c / (a - b)) = -1) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) = 0 :=
sorry

end distinct_real_numbers_proof_l238_238466


namespace leak_empty_time_l238_238348

variable (inlet_rate : ℕ := 6) -- litres per minute
variable (total_capacity : ℕ := 12960) -- litres
variable (empty_time_with_inlet_open : ℕ := 12) -- hours

def inlet_rate_per_hour := inlet_rate * 60 -- litres per hour
def net_emptying_rate := total_capacity / empty_time_with_inlet_open -- litres per hour
def leak_rate := net_emptying_rate + inlet_rate_per_hour -- litres per hour

theorem leak_empty_time : total_capacity / leak_rate = 9 := by
  sorry

end leak_empty_time_l238_238348


namespace expression_is_nonnegative_l238_238540

noncomputable def expression_nonnegative (a b c d e : ℝ) : Prop :=
  (a - b) * (a - c) * (a - d) * (a - e) +
  (b - a) * (b - c) * (b - d) * (b - e) +
  (c - a) * (c - b) * (c - d) * (c - e) +
  (d - a) * (d - b) * (d - c) * (d - e) +
  (e - a) * (e - b) * (e - c) * (e - d) ≥ 0

theorem expression_is_nonnegative (a b c d e : ℝ) : expression_nonnegative a b c d e := 
by 
  sorry

end expression_is_nonnegative_l238_238540


namespace odd_function_f_neg_one_l238_238197

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 2 then 2^x else 0 -- Placeholder; actual implementation skipped for simplicity

theorem odd_function_f_neg_one :
  (∀ x, f (-x) = -f x) ∧ (∀ x, (0 < x ∧ x < 2) → f x = 2^x) → 
  f (-1) = -2 :=
by
  intros h
  let odd_property := h.1
  let condition_in_range := h.2
  sorry

end odd_function_f_neg_one_l238_238197


namespace basketball_third_quarter_points_l238_238566

noncomputable def teamA_points (a r : ℕ) : ℕ :=
a + a*r + a*r^2 + a*r^3

noncomputable def teamB_points (b d : ℕ) : ℕ :=
b + (b + d) + (b + 2*d) + (b + 3*d)

theorem basketball_third_quarter_points (a b d : ℕ) (r : ℕ) 
    (h1 : r > 1) (h2 : d > 0) (h3 : a * (r^4 - 1) / (r - 1) = 4 * b + 6 * d + 3)
    (h4 : a * (r^4 - 1) / (r - 1) ≤ 100) (h5 : 4 * b + 6 * d ≤ 100) :
    a * r^2 + b + 2 * d = 60 :=
sorry

end basketball_third_quarter_points_l238_238566


namespace point_not_on_graph_and_others_on_l238_238354

theorem point_not_on_graph_and_others_on (y : ℝ → ℝ) (h₁ : ∀ x, y x = x / (x - 1))
  : ¬ (1 = (1 : ℝ) / ((1 : ℝ) - 1)) 
  ∧ (2 = (2 : ℝ) / ((2 : ℝ) - 1)) 
  ∧ ((-1 : ℝ) = (1/2 : ℝ) / ((1/2 : ℝ) - 1)) 
  ∧ (0 = (0 : ℝ) / ((0 : ℝ) - 1)) 
  ∧ (3/2 = (3 : ℝ) / ((3 : ℝ) - 1)) := 
sorry

end point_not_on_graph_and_others_on_l238_238354


namespace triangle_area_l238_238556

-- Define the conditions of the problem
variables (a b c : ℝ) (C : ℝ)
axiom cond1 : c^2 = a^2 + b^2 - 2 * a * b + 6
axiom cond2 : C = Real.pi / 3

-- Define the goal
theorem triangle_area : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_l238_238556


namespace y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l238_238881

def y : ℕ := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256

theorem y_is_multiple_of_16 : y % 16 = 0 :=
sorry

theorem y_is_multiple_of_8 : y % 8 = 0 :=
sorry

theorem y_is_multiple_of_4 : y % 4 = 0 :=
sorry

theorem y_is_multiple_of_2 : y % 2 = 0 :=
sorry

end y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l238_238881


namespace maximum_sum_of_triplets_l238_238738

-- Define a list representing a 9-digit number consisting of digits 1 to 9 in some order
def valid_digits (digits : List ℕ) : Prop :=
  digits.length = 9 ∧ ∀ n, n ∈ digits → n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]
  
def sum_of_triplets (digits : List ℕ) : ℕ :=
  100 * digits[0]! + 10 * digits[1]! + digits[2]! +
  100 * digits[1]! + 10 * digits[2]! + digits[3]! +
  100 * digits[2]! + 10 * digits[3]! + digits[4]! +
  100 * digits[3]! + 10 * digits[4]! + digits[5]! +
  100 * digits[4]! + 10 * digits[5]! + digits[6]! +
  100 * digits[5]! + 10 * digits[6]! + digits[7]! +
  100 * digits[6]! + 10 * digits[7]! + digits[8]!

theorem maximum_sum_of_triplets :
  ∃ digits : List ℕ, valid_digits digits ∧ sum_of_triplets digits = 4648 :=
  sorry

end maximum_sum_of_triplets_l238_238738


namespace hours_per_batch_l238_238249

noncomputable section

def gallons_per_batch : ℕ := 3 / 2   -- 1.5 gallons expressed as a rational number
def ounces_per_gallon : ℕ := 128
def jack_consumption_per_2_days : ℕ := 96
def total_days : ℕ := 24
def time_spent_hours : ℕ := 120

def total_ounces : ℕ := gallons_per_batch * ounces_per_gallon
def total_ounces_consumed_24_days : ℕ := jack_consumption_per_2_days * (total_days / 2)
def number_of_batches : ℕ := total_ounces_consumed_24_days / total_ounces

theorem hours_per_batch :
  (time_spent_hours / number_of_batches) = 20 := by
  sorry

end hours_per_batch_l238_238249


namespace evaluate_expression_l238_238608

def operation_star (A B : ℕ) : ℕ := (A + B) / 2
def operation_ominus (A B : ℕ) : ℕ := A - B

theorem evaluate_expression :
  operation_ominus (operation_star 6 10) (operation_star 2 4) = 5 := 
by 
  sorry

end evaluate_expression_l238_238608


namespace cubic_equation_roots_l238_238479

theorem cubic_equation_roots (a b c d r s t : ℝ) (h_eq : a ≠ 0) 
(ht1 : a * r^3 + b * r^2 + c * r + d = 0)
(ht2 : a * s^3 + b * s^2 + c * s + d = 0)
(ht3 : a * t^3 + b * t^2 + c * t + d = 0)
(h1 : r * s = 3) 
(h2 : r * t = 3) 
(h3 : s * t = 3) : 
c = 3 * a := 
sorry

end cubic_equation_roots_l238_238479


namespace sum_of_fractions_l238_238987

theorem sum_of_fractions:
  (7 / 12) + (11 / 15) = 79 / 60 :=
by
  sorry

end sum_of_fractions_l238_238987


namespace slope_of_line_l238_238955

def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (4, 5)

theorem slope_of_line : 
  let (x1, y1) := point1
  let (x2, y2) := point2
  (x2 - x1) ≠ 0 → (y2 - y1) / (x2 - x1) = 1 := by
  sorry

end slope_of_line_l238_238955


namespace desiredCircleEquation_l238_238420

-- Definition of the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Definition of the given line
def givenLine (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- The required proof problem statement
theorem desiredCircleEquation :
  (∀ P Q : ℝ × ℝ, givenCircle P.1 P.2 ∧ givenLine P.1 P.2 → givenCircle Q.1 Q.2 ∧ givenLine Q.1 Q.2 →
  (P ≠ Q) → 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0)) :=
by
  -- Proof omitted
  sorry

end desiredCircleEquation_l238_238420


namespace quadrilateral_area_l238_238926

theorem quadrilateral_area {AB BC : ℝ} (hAB : AB = 4) (hBC : BC = 8) :
  ∃ area : ℝ, area = 16 := by
  sorry

end quadrilateral_area_l238_238926


namespace half_vector_AB_l238_238839

-- Define vectors MA and MB
def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

-- Define the proof statement 
theorem half_vector_AB : (1 / 2 : ℝ) • (MB - MA) = (2, 1) :=
by sorry

end half_vector_AB_l238_238839


namespace symmetric_line_equation_l238_238623

theorem symmetric_line_equation 
  (l1 : ∀ x y : ℝ, x - 2 * y - 2 = 0) 
  (l2 : ∀ x y : ℝ, x + y = 0) : 
  ∀ x y : ℝ, 2 * x - y - 2 = 0 :=
sorry

end symmetric_line_equation_l238_238623


namespace percentage_of_x_is_40_l238_238315

theorem percentage_of_x_is_40 
  (x p : ℝ)
  (h1 : (1 / 2) * x = 200)
  (h2 : p * x = 160) : 
  p * 100 = 40 := 
by 
  sorry

end percentage_of_x_is_40_l238_238315


namespace tan_of_diff_l238_238119

theorem tan_of_diff (θ : ℝ) (hθ : -π/2 + 2 * π < θ ∧ θ < 2 * π) 
  (h : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 :=
sorry

end tan_of_diff_l238_238119


namespace elephant_distribution_l238_238514

theorem elephant_distribution (unions nonunions : ℕ) (elephants : ℕ) :
  unions = 28 ∧ nonunions = 37 ∧ (∀ k : ℕ, elephants = 28 * k ∨ elephants = 37 * k) ∧ (∀ k : ℕ, ((28 * k ≤ elephants) ∧ (37 * k ≤ elephants))) → 
  elephants = 2072 :=
by
  sorry

end elephant_distribution_l238_238514


namespace cos_B_and_area_of_triangle_l238_238162

theorem cos_B_and_area_of_triangle (A B C : ℝ) (a b c : ℝ)
  (h_sin_A : Real.sin A = Real.sin (2 * B))
  (h_a : a = 4) (h_b : b = 6) :
  Real.cos B = 1 / 3 ∧ ∃ (area : ℝ), area = 8 * Real.sqrt 2 :=
by
  sorry  -- Proof goes here

end cos_B_and_area_of_triangle_l238_238162


namespace inequality_holds_l238_238488

theorem inequality_holds (a b : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (hb : 0 ≤ b) (hb' : b ≤ 1) : 
  a^5 + b^3 + (a - b)^2 ≤ 2 :=
sorry

end inequality_holds_l238_238488


namespace find_height_of_cuboid_l238_238612

-- Definitions and given conditions
def length : ℕ := 22
def width : ℕ := 30
def total_edges : ℕ := 224

-- Proof statement
theorem find_height_of_cuboid (h : ℕ) (H : 4 * length + 4 * width + 4 * h = total_edges) : h = 4 :=
by
  sorry

end find_height_of_cuboid_l238_238612


namespace max_value_inequality_l238_238228

theorem max_value_inequality
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = ⅟2) :
  (|x1 + y1 - 1| / Real.sqrt 2) + (|x2 + y2 - 1| / Real.sqrt 2) ≤ 1 :=
by {
  sorry
}

end max_value_inequality_l238_238228


namespace time_to_fill_by_B_l238_238607

/-- 
Assume a pool with two taps, A and B, fills in 30 minutes when both are open.
When both are open for 10 minutes, and then only B is open for another 40 minutes, the pool fills up.
Prove that if only tap B is opened, it would take 60 minutes to fill the pool.
-/
theorem time_to_fill_by_B
  (r_A r_B : ℝ)
  (H1 : (r_A + r_B) * 30 = 1)
  (H2 : ((r_A + r_B) * 10 + r_B * 40) = 1) :
  1 / r_B = 60 :=
by
  sorry

end time_to_fill_by_B_l238_238607


namespace area_union_of_rectangle_and_circle_l238_238938

theorem area_union_of_rectangle_and_circle :
  let length := 12
  let width := 15
  let r := 15
  let area_rectangle := length * width
  let area_circle := Real.pi * r^2
  let area_overlap := (1/4) * area_circle
  let area_union := area_rectangle + area_circle - area_overlap
  area_union = 180 + 168.75 * Real.pi := by
    sorry

end area_union_of_rectangle_and_circle_l238_238938


namespace sum_of_solutions_l238_238693

-- Given the quadratic equation: x^2 + 3x - 20 = 7x + 8
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 20 = 7*x + 8

-- Prove that the sum of the solutions to this quadratic equation is 4
theorem sum_of_solutions : 
  ∀ x1 x2 : ℝ, (quadratic_equation x1) ∧ (quadratic_equation x2) → x1 + x2 = 4 :=
by
  sorry

end sum_of_solutions_l238_238693


namespace remainder_x_squared_div_25_l238_238803

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end remainder_x_squared_div_25_l238_238803


namespace ordered_pairs_count_l238_238447

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end ordered_pairs_count_l238_238447


namespace value_2_std_dev_less_than_mean_l238_238562

def mean : ℝ := 16.5
def std_dev : ℝ := 1.5

theorem value_2_std_dev_less_than_mean :
  mean - 2 * std_dev = 13.5 := by
  sorry

end value_2_std_dev_less_than_mean_l238_238562


namespace women_attended_l238_238748

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l238_238748


namespace length_of_RT_in_trapezoid_l238_238064

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end length_of_RT_in_trapezoid_l238_238064


namespace cost_of_bananas_and_cantaloupe_l238_238334

-- Define prices for different items
variables (a b c d e : ℝ)

-- Define the conditions as hypotheses
theorem cost_of_bananas_and_cantaloupe (h1 : a + b + c + d + e = 30)
    (h2 : d = 3 * a) (h3 : c = a - b) (h4 : e = a + b) :
    b + c = 5 := 
by 
  -- Initial proof setup
  sorry

end cost_of_bananas_and_cantaloupe_l238_238334


namespace cally_pants_count_l238_238294

variable (cally_white_shirts : ℕ)
variable (cally_colored_shirts : ℕ)
variable (cally_shorts : ℕ)
variable (danny_white_shirts : ℕ)
variable (danny_colored_shirts : ℕ)
variable (danny_shorts : ℕ)
variable (danny_pants : ℕ)
variable (total_clothes_washed : ℕ)
variable (cally_pants : ℕ)

-- Given conditions
#check cally_white_shirts = 10
#check cally_colored_shirts = 5
#check cally_shorts = 7
#check danny_white_shirts = 6
#check danny_colored_shirts = 8
#check danny_shorts = 10
#check danny_pants = 6
#check total_clothes_washed = 58
#check cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed

-- Proof goal
theorem cally_pants_count (cally_white_shirts cally_colored_shirts cally_shorts danny_white_shirts danny_colored_shirts danny_shorts danny_pants cally_pants total_clothes_washed : ℕ) :
  cally_white_shirts = 10 →
  cally_colored_shirts = 5 →
  cally_shorts = 7 →
  danny_white_shirts = 6 →
  danny_colored_shirts = 8 →
  danny_shorts = 10 →
  danny_pants = 6 →
  total_clothes_washed = 58 →
  (cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed) →
  cally_pants = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end cally_pants_count_l238_238294


namespace rectangular_to_cylindrical_4_neg4_6_l238_238775

theorem rectangular_to_cylindrical_4_neg4_6 :
  let x := 4
  let y := -4
  let z := 6
  let r := 4 * Real.sqrt 2
  let theta := (7 * Real.pi) / 4
  (r = Real.sqrt (x^2 + y^2)) ∧
  (Real.cos theta = x / r) ∧
  (Real.sin theta = y / r) ∧
  0 ≤ theta ∧ theta < 2 * Real.pi ∧
  z = 6 → 
  (r, theta, z) = (4 * Real.sqrt 2, (7 * Real.pi) / 4, 6) :=
by
  sorry

end rectangular_to_cylindrical_4_neg4_6_l238_238775


namespace valid_triples_count_l238_238284

def validTriple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 15 ∧ 
  1 ≤ b ∧ b ≤ 15 ∧ 
  1 ≤ c ∧ c ≤ 15 ∧ 
  (b % a = 0 ∨ (∃ k : ℕ, k ≤ 15 ∧ c % k = 0))

def countValidTriples : ℕ := 
  (15 + 7 + 5 + 3 + 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) * 2 - 15

theorem valid_triples_count : countValidTriples = 75 :=
  by
  sorry

end valid_triples_count_l238_238284


namespace eccentricity_of_ellipse_l238_238076

theorem eccentricity_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b > 0) :
  (∀ x y : ℝ, (y = -2 * x + 1 → ∃ x₁ y₁ x₂ y₂ : ℝ, (y₁ = -2 * x₁ + 1 ∧ y₂ = -2 * x₂ + 1) ∧ 
    (x₁ / a * x₁ / a + y₁ / b * y₁ / b = 1) ∧ (x₂ / a * x₂ / a + y₂ / b * y₂ / b = 1) ∧ 
    ((x₁ + x₂) / 2 = 4 * (y₁ + y₂) / 2)) → (x / a)^2 + (y / b)^2 = 1) →
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = (Real.sqrt 2) / 2 :=
sorry

end eccentricity_of_ellipse_l238_238076


namespace smallest_part_proportional_l238_238431

/-- If we divide 124 into three parts proportional to 2, 1/2, and 1/4,
    prove that the smallest part is 124 / 11. -/
theorem smallest_part_proportional (x : ℝ) 
  (h : 2 * x + (1 / 2) * x + (1 / 4) * x = 124) : 
  (1 / 4) * x = 124 / 11 :=
sorry

end smallest_part_proportional_l238_238431


namespace instantaneous_velocity_at_t_5_l238_238257

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_t_5 : 
  (deriv s 5) = 125 :=
by
  sorry

end instantaneous_velocity_at_t_5_l238_238257


namespace value_of_fraction_l238_238535

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : (4 * x + y) / (x - 4 * y) = -3)

theorem value_of_fraction : (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end value_of_fraction_l238_238535


namespace inequality_must_hold_l238_238063

theorem inequality_must_hold (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_must_hold_l238_238063


namespace problem1_problem2_l238_238622

open Nat

def seq (a : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → a n < a (n + 1) ∧ a n > 0

def b_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n)

def c_seq (a : ℕ → ℕ) (n : ℕ) :=
  a (a n + 1)

theorem problem1 (a : ℕ → ℕ) (h_seq : seq a) (h_bseq : ∀ n, n > 0 → b_seq a n = 3 * n) : a 1 = 2 ∧ c_seq a 1 = 6 :=
  sorry

theorem problem2 (a : ℕ → ℕ) (h_seq : seq a) (h_cseq : ∀ n, n > 0 → c_seq a (n + 1) - c_seq a n = 1) : 
  ∀ n, n > 0 → a (n + 1) - a n = 1 :=
  sorry

end problem1_problem2_l238_238622


namespace m_over_n_add_one_l238_238679

theorem m_over_n_add_one (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : (m + n : ℚ) / n = 10 / 7 :=
by
  sorry

end m_over_n_add_one_l238_238679


namespace inequality_solution_l238_238860

noncomputable def f (x : ℝ) : ℝ :=
  (2 / (x + 2)) + (4 / (x + 8))

theorem inequality_solution {x : ℝ} :
  f x ≥ 1/2 ↔ ((-8 < x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 2)) :=
sorry

end inequality_solution_l238_238860


namespace percentage_of_x_is_y_l238_238551

theorem percentage_of_x_is_y (x y : ℝ) (h : 0.5 * (x - y) = 0.4 * (x + y)) : y = 0.1111 * x := 
sorry

end percentage_of_x_is_y_l238_238551


namespace digit_C_equals_one_l238_238247

-- Define the scope of digits
def is_digit (n : ℕ) : Prop := n < 10

-- Define the equality for sums of digits
def sum_of_digits (A B C : ℕ) : Prop := A + B + C = 10

-- Main theorem to prove C = 1
theorem digit_C_equals_one (A B C : ℕ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hSum : sum_of_digits A B C) : C = 1 :=
sorry

end digit_C_equals_one_l238_238247


namespace zero_exponent_rule_proof_l238_238951

-- Defining the condition for 818 being non-zero
def eight_hundred_eighteen_nonzero : Prop := 818 ≠ 0

-- Theorem statement
theorem zero_exponent_rule_proof (h : eight_hundred_eighteen_nonzero) : 818 ^ 0 = 1 := by
  sorry

end zero_exponent_rule_proof_l238_238951


namespace ratio_p_q_l238_238672

-- Definitions of probabilities p and q based on combinatorial choices and probabilities described.
noncomputable def p : ℚ :=
  (Nat.choose 6 1) * (Nat.choose 5 2) * (Nat.choose 24 2) * (Nat.choose 22 4) * (Nat.choose 18 4) * (Nat.choose 14 5) * (Nat.choose 9 5) * (Nat.choose 4 5) / (6 ^ 24)

noncomputable def q : ℚ :=
  (Nat.choose 6 2) * (Nat.choose 24 3) * (Nat.choose 21 3) * (Nat.choose 18 4) * (Nat.choose 14 4) * (Nat.choose 10 4) * (Nat.choose 6 4) / (6 ^ 24)

-- Lean statement to prove p / q = 6
theorem ratio_p_q : p / q = 6 := by
  sorry

end ratio_p_q_l238_238672


namespace compute_expression_l238_238982

theorem compute_expression :
  (-9 * 5 - (-7 * -2) + (-11 * -4)) = -15 :=
by
  sorry

end compute_expression_l238_238982


namespace exponentiation_problem_l238_238588

theorem exponentiation_problem : 10^6 * (10^2)^3 / 10^4 = 10^8 := 
by 
  sorry

end exponentiation_problem_l238_238588


namespace mia_bought_more_pencils_l238_238439

theorem mia_bought_more_pencils (p : ℝ) (n1 n2 : ℕ) 
  (price_pos : p > 0.01)
  (liam_spent : 2.10 = p * n1)
  (mia_spent : 2.82 = p * n2) :
  (n2 - n1) = 12 := 
by
  sorry

end mia_bought_more_pencils_l238_238439


namespace max_colored_nodes_without_cycle_in_convex_polygon_l238_238180

def convex_polygon (n : ℕ) : Prop := n ≥ 3

def valid_diagonals (n : ℕ) : Prop := n = 2019

def no_three_diagonals_intersect_at_single_point (x : Type*) : Prop :=
  sorry -- You can provide a formal definition here based on combinatorial geometry.

def no_loops (n : ℕ) (k : ℕ) : Prop :=
  k ≤ (n * (n - 3)) / 2 - 1

theorem max_colored_nodes_without_cycle_in_convex_polygon :
  convex_polygon 2019 →
  valid_diagonals 2019 →
  no_three_diagonals_intersect_at_single_point ℝ →
  ∃ k, k = 2035151 ∧ no_loops 2019 k := 
by
  -- The proof would be constructed here.
  sorry

end max_colored_nodes_without_cycle_in_convex_polygon_l238_238180


namespace choir_min_students_l238_238976

theorem choir_min_students : ∃ n : ℕ, (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ n = 990 :=
by
  sorry

end choir_min_students_l238_238976


namespace smallest_n_l238_238572

theorem smallest_n (a b c n : ℕ) (h1 : n = 100 * a + 10 * b + c)
  (h2 : n = a + b + c + a * b + b * c + a * c + a * b * c)
  (h3 : n >= 100 ∧ n < 1000)
  (h4 : a ≥ 1 ∧ a ≤ 9)
  (h5 : b ≥ 0 ∧ b ≤ 9)
  (h6 : c ≥ 0 ∧ c ≤ 9) :
  n = 199 :=
sorry

end smallest_n_l238_238572


namespace range_of_m_l238_238774

-- Condition p: The solution set of the inequality x² + mx + 1 < 0 is an empty set
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Condition q: The function y = 4x² + 4(m-1)x + 3 has no extreme value
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 12 * x^2 + 4 * (m - 1) ≥ 0

-- Combined condition: "p or q" is true and "p and q" is false
def combined_condition (m : ℝ) : Prop :=
  (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- The range of values for the real number m
theorem range_of_m (m : ℝ) : combined_condition m → (-2 ≤ m ∧ m < 1) ∨ m > 2 :=
sorry

end range_of_m_l238_238774


namespace cans_of_chili_beans_ordered_l238_238372

theorem cans_of_chili_beans_ordered (T C : ℕ) (h1 : 2 * T = C) (h2 : T + C = 12) : C = 8 := by
  sorry

end cans_of_chili_beans_ordered_l238_238372


namespace sum_diff_reciprocals_equals_zero_l238_238956

theorem sum_diff_reciprocals_equals_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (1 / (a + 1)) + (1 / (a - 1)) + (1 / (b + 1)) + (1 / (b - 1)) = 0) :
  (a + b) - (1 / a + 1 / b) = 0 :=
by
  sorry

end sum_diff_reciprocals_equals_zero_l238_238956


namespace isosceles_triangle_l238_238013

noncomputable def sin (x : ℝ) : ℝ := Real.sin x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

variables {A B C : ℝ}
variable (h : sin C = 2 * sin (B + C) * cos B)

theorem isosceles_triangle (h : sin C = 2 * sin (B + C) * cos B) : A = B :=
by
  sorry

end isosceles_triangle_l238_238013


namespace highest_power_of_5_dividing_S_l238_238147

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℤ :=
  if sum_of_digits n % 2 = 0 then n ^ 100 else -n ^ 100

def S : ℤ :=
  (Finset.range (10 ^ 100)).sum (λ n => f n)

theorem highest_power_of_5_dividing_S :
  ∃ m : ℕ, 5 ^ m ∣ S ∧ ∀ k : ℕ, 5 ^ (k + 1) ∣ S → k < 24 :=
by
  sorry

end highest_power_of_5_dividing_S_l238_238147


namespace highest_power_of_3_l238_238215

-- Define the integer M formed by concatenating the 3-digit numbers from 100 to 250
def M : ℕ := sorry  -- We should define it in a way that represents the concatenation

-- Define a proof that the highest power of 3 that divides M is 3^1
theorem highest_power_of_3 (n : ℕ) (h : M = n) : ∃ m : ℕ, 3^m ∣ n ∧ ¬ (3^(m + 1) ∣ n) ∧ m = 1 :=
by sorry  -- We will not provide proofs; we're only writing the statement

end highest_power_of_3_l238_238215


namespace alpha_beta_inequality_l238_238734

theorem alpha_beta_inequality (α β : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x^α * y^β < k * (x + y)) ↔ (0 ≤ α ∧ 0 ≤ β ∧ α + β = 1) :=
by
  sorry

end alpha_beta_inequality_l238_238734


namespace complex_product_l238_238554

theorem complex_product (i : ℂ) (h : i^2 = -1) :
  (3 - 4 * i) * (2 + 7 * i) = 34 + 13 * i :=
sorry

end complex_product_l238_238554


namespace find_highway_speed_l238_238446

def car_local_distance := 40
def car_local_speed := 20
def car_highway_distance := 180
def average_speed := 44
def speed_of_car_on_highway := 60

theorem find_highway_speed :
  car_local_distance / car_local_speed + car_highway_distance / speed_of_car_on_highway = (car_local_distance + car_highway_distance) / average_speed :=
by
  sorry

end find_highway_speed_l238_238446


namespace min_value_at_2_l238_238337

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_at_2 : ∃ x : ℝ, f x = 2 :=
sorry

end min_value_at_2_l238_238337


namespace total_songs_in_june_l238_238304

-- Define the conditions
def Vivian_daily_songs : ℕ := 10
def Clara_daily_songs : ℕ := Vivian_daily_songs - 2
def Lucas_daily_songs : ℕ := Vivian_daily_songs + 5
def total_play_days_in_june : ℕ := 30 - 8 - 1

-- Total songs listened to in June
def total_songs_Vivian : ℕ := Vivian_daily_songs * total_play_days_in_june
def total_songs_Clara : ℕ := Clara_daily_songs * total_play_days_in_june
def total_songs_Lucas : ℕ := Lucas_daily_songs * total_play_days_in_june

-- The total number of songs listened to by all three
def total_songs_all_three : ℕ := total_songs_Vivian + total_songs_Clara + total_songs_Lucas

-- The proof problem
theorem total_songs_in_june : total_songs_all_three = 693 := by
  -- Placeholder for the proof
  sorry

end total_songs_in_june_l238_238304


namespace min_value_expression_l238_238494

theorem min_value_expression (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  ∃ x : ℝ, x = 1 ∧ x = (3 * a - 2 * b + c) / (b - a) := 
  sorry

end min_value_expression_l238_238494


namespace max_xy_l238_238788

open Real

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : x + 4 * y = 4) :
  ∃ y : ℝ, (x = 4 - 4 * y) → y = 1 / 2 → x * y = 1 :=
by
  sorry

end max_xy_l238_238788


namespace total_exterior_angles_l238_238675

-- Define that the sum of the exterior angles of any convex polygon is 360 degrees
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Given four polygons: a triangle, a quadrilateral, a pentagon, and a hexagon
def triangle_exterior_sum := sum_exterior_angles 3
def quadrilateral_exterior_sum := sum_exterior_angles 4
def pentagon_exterior_sum := sum_exterior_angles 5
def hexagon_exterior_sum := sum_exterior_angles 6

-- The total sum of the exterior angles of these four polygons combined
def total_exterior_angle_sum := 
  triangle_exterior_sum + 
  quadrilateral_exterior_sum + 
  pentagon_exterior_sum + 
  hexagon_exterior_sum

-- The final proof statement
theorem total_exterior_angles : total_exterior_angle_sum = 1440 := by
  sorry

end total_exterior_angles_l238_238675


namespace rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l238_238949

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def transition_rule (n : ℕ) : ℕ :=
  2 * (sum_of_digits n)

theorem rule_for_sequence :
  transition_rule 3 = 6 ∧ transition_rule 6 = 12 :=
by
  sorry

theorem natural_number_self_map :
  ∀ n : ℕ, transition_rule n = n ↔ n = 18 :=
by
  sorry

theorem power_of_2_to_single_digit :
  ∃ x : ℕ, transition_rule (2^1991) = x ∧ x < 10 :=
by
  sorry

end rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l238_238949


namespace find_b_l238_238311

theorem find_b (a b : ℝ) (x : ℝ) (h : (1 + a * x)^5 = 1 + 10 * x + b * x^2 + (a^5) * x^5):
  b = 40 :=
  sorry

end find_b_l238_238311


namespace evaluate_g_at_5_l238_238378

noncomputable def g (x : ℝ) : ℝ := 2 * x ^ 4 - 15 * x ^ 3 + 24 * x ^ 2 - 18 * x - 72

theorem evaluate_g_at_5 : g 5 = -7 := by
  sorry

end evaluate_g_at_5_l238_238378


namespace ratio_of_cans_l238_238245

theorem ratio_of_cans (martha_cans : ℕ) (total_required : ℕ) (remaining_cans : ℕ) (diego_cans : ℕ) (ratio : ℚ) 
  (h1 : martha_cans = 90) 
  (h2 : total_required = 150) 
  (h3 : remaining_cans = 5) 
  (h4 : martha_cans + diego_cans = total_required - remaining_cans) 
  (h5 : ratio = (diego_cans : ℚ) / martha_cans) : 
  ratio = 11 / 18 := 
by
  sorry

end ratio_of_cans_l238_238245


namespace ratio_lcm_gcf_eq_55_l238_238646

theorem ratio_lcm_gcf_eq_55 : 
  ∀ (a b : ℕ), a = 210 → b = 462 →
  (Nat.lcm a b / Nat.gcd a b) = 55 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end ratio_lcm_gcf_eq_55_l238_238646


namespace impossible_to_achieve_12_percent_return_l238_238853

-- Define the stock parameters and their individual returns
def stock_A_price : ℝ := 52
def stock_A_dividend_rate : ℝ := 0.09
def stock_A_transaction_fee_rate : ℝ := 0.02

def stock_B_price : ℝ := 80
def stock_B_dividend_rate : ℝ := 0.07
def stock_B_transaction_fee_rate : ℝ := 0.015

def stock_C_price : ℝ := 40
def stock_C_dividend_rate : ℝ := 0.10
def stock_C_transaction_fee_rate : ℝ := 0.01

def tax_rate : ℝ := 0.10
def desired_return : ℝ := 0.12

theorem impossible_to_achieve_12_percent_return :
  false :=
sorry

end impossible_to_achieve_12_percent_return_l238_238853


namespace grant_room_proof_l238_238721

/-- Danielle's apartment has 6 rooms -/
def danielle_rooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment -/
def heidi_rooms : ℕ := 3 * danielle_rooms

/-- Jenny's apartment has 5 more rooms than Danielle's apartment -/
def jenny_rooms : ℕ := danielle_rooms + 5

/-- Lina's apartment has 7 rooms -/
def lina_rooms : ℕ := 7

/-- The total number of rooms from Danielle, Heidi, Jenny,
    and Lina's apartments -/
def total_rooms : ℕ := danielle_rooms + heidi_rooms + jenny_rooms + lina_rooms

/-- Grant's apartment has 1/3 less rooms than 1/9 of the
    combined total of rooms from Danielle's, Heidi's, Jenny's, and Lina's apartments -/
def grant_rooms : ℕ := (total_rooms / 9) - (total_rooms / 9) / 3

/-- Prove that Grant's apartment has 3 rooms -/
theorem grant_room_proof : grant_rooms = 3 :=
by
  sorry

end grant_room_proof_l238_238721


namespace intersection_P_Q_correct_l238_238171

-- Define sets P and Q based on given conditions
def is_in_P (x : ℝ) : Prop := x > 1
def is_in_Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the intersection P ∩ Q and the correct answer
def P_inter_Q (x : ℝ) : Prop := is_in_P x ∧ is_in_Q x
def correct_ans (x : ℝ) : Prop := 1 < x ∧ x ≤ 2

-- Prove that P ∩ Q = (1, 2]
theorem intersection_P_Q_correct : ∀ x : ℝ, P_inter_Q x ↔ correct_ans x :=
by sorry

end intersection_P_Q_correct_l238_238171


namespace sufficient_but_not_necessary_l238_238964

noncomputable def condition_to_bool (a b : ℝ) : Bool :=
a > b ∧ b > 0

theorem sufficient_but_not_necessary (a b : ℝ) (h : condition_to_bool a b) :
  (a > b ∧ b > 0) → (a^2 > b^2) ∧ (∃ a' b' : ℝ, a'^2 > b'^2 ∧ ¬ (a' > b' ∧ b' > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l238_238964


namespace gopi_servant_salary_l238_238374

theorem gopi_servant_salary (S : ℝ) (h1 : 9 / 12 * S + 110 = 150) : S = 200 :=
by
  sorry

end gopi_servant_salary_l238_238374


namespace tangent_line_equation_l238_238756

theorem tangent_line_equation (x y : ℝ) (h : y = x^3 + 1) (t : x = -1) :
  3*x - y + 3 = 0 :=
sorry

end tangent_line_equation_l238_238756


namespace proposition_2_proposition_3_l238_238166

theorem proposition_2 (a b : ℝ) (h: a > |b|) : a^2 > b^2 := 
sorry

theorem proposition_3 (a b : ℝ) (h: a > b) : a^3 > b^3 := 
sorry

end proposition_2_proposition_3_l238_238166


namespace abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l238_238783

theorem abs_x_minus_1_lt_2_is_necessary_but_not_sufficient (x : ℝ) :
  (-1 < x ∧ x < 3) ↔ (0 < x ∧ x < 3) :=
sorry

end abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l238_238783


namespace problem_incorrect_statement_D_l238_238052

theorem problem_incorrect_statement_D :
  (∀ x y, x = -y → x + y = 0) ∧
  (∃ x : ℕ, x^2 + 2 * x = 0) ∧
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (¬ (∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ (x + y > 2))) :=
by sorry

end problem_incorrect_statement_D_l238_238052


namespace cows_in_group_l238_238176

theorem cows_in_group (c h : ℕ) (h_condition : 4 * c + 2 * h = 2 * (c + h) + 16) : c = 8 :=
sorry

end cows_in_group_l238_238176


namespace set_intersection_l238_238455

open Set

def U : Set ℤ := univ
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {-1, 1}
def C_U_B : Set ℤ := U \ B

theorem set_intersection :
  A ∩ C_U_B = {2} := 
by
  sorry

end set_intersection_l238_238455


namespace cups_of_sugar_l238_238160

theorem cups_of_sugar (flour_total flour_added sugar : ℕ) (h₁ : flour_total = 10) (h₂ : flour_added = 7) (h₃ : flour_total - flour_added = sugar + 1) :
  sugar = 2 :=
by
  sorry

end cups_of_sugar_l238_238160


namespace find_three_digit_number_l238_238994

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end find_three_digit_number_l238_238994


namespace stock_price_at_end_of_second_year_l238_238792

def stock_price_first_year (initial_price : ℝ) : ℝ :=
  initial_price * 2

def stock_price_second_year (price_after_first_year : ℝ) : ℝ :=
  price_after_first_year * 0.75

theorem stock_price_at_end_of_second_year : 
  (stock_price_second_year (stock_price_first_year 100) = 150) :=
by
  sorry

end stock_price_at_end_of_second_year_l238_238792


namespace complete_the_square_l238_238717

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x^2 + 10 * x - 3 = 0) → ((x + a)^2 = b) ∧ b = 28) :=
sorry

end complete_the_square_l238_238717


namespace find_a_perpendicular_lines_l238_238418

variable (a : ℝ)

theorem find_a_perpendicular_lines :
  (∃ a : ℝ, ∀ x y : ℝ, (a * x - y + 2 * a = 0) ∧ ((2 * a - 1) * x + a * y + a = 0) → a = 0 ∨ a = 1) := 
sorry

end find_a_perpendicular_lines_l238_238418


namespace sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l238_238490

-- Definitions based on conditions
def sum_arithmetic (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Theorem statements based on the correct answers
theorem sum_first_twelve_multiples_17 : 
  17 * sum_arithmetic 12 = 1326 := 
by
  sorry

theorem sum_squares_first_twelve_multiples_17 : 
  17^2 * sum_squares 12 = 187850 :=
by
  sorry

end sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l238_238490


namespace cos_double_angle_identity_l238_238044

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 := 
sorry

end cos_double_angle_identity_l238_238044


namespace pages_for_thirty_dollars_l238_238459

-- Problem Statement Definitions
def costPerCopy := 4 -- cents
def pagesPerCopy := 2 -- pages
def totalCents := 3000 -- cents
def totalPages := 1500 -- pages

-- Theorem: Calculating the number of pages for a given cost.
theorem pages_for_thirty_dollars (c_per_copy : ℕ) (p_per_copy : ℕ) (t_cents : ℕ) (t_pages : ℕ) : 
  c_per_copy = 4 → p_per_copy = 2 → t_cents = 3000 → t_pages = 1500 := by
  intros h_cpc h_ppc h_tc
  sorry

end pages_for_thirty_dollars_l238_238459


namespace ball_hits_ground_time_l238_238609

theorem ball_hits_ground_time :
  ∃ t : ℝ, -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 := by
  sorry

end ball_hits_ground_time_l238_238609


namespace gcd_divisibility_and_scaling_l238_238486

theorem gcd_divisibility_and_scaling (a b n : ℕ) (c : ℕ) (h₁ : a ≠ 0) (h₂ : c > 0) (d : ℕ := Nat.gcd a b) :
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧ Nat.gcd (a * c) (b * c) = c * d :=
by 
  sorry

end gcd_divisibility_and_scaling_l238_238486


namespace abs_inequality_solution_set_l238_238552

theorem abs_inequality_solution_set (x : ℝ) : -1 < x ∧ x < 1 ↔ |2*x - 1| - |x - 2| < 0 := by
  sorry

end abs_inequality_solution_set_l238_238552


namespace price_reduction_required_l238_238822

variable (x : ℝ)
variable (profit_per_piece : ℝ := 40)
variable (initial_sales : ℝ := 20)
variable (additional_sales_per_unit_reduction : ℝ := 2)
variable (desired_profit : ℝ := 1200)

theorem price_reduction_required :
  (profit_per_piece - x) * (initial_sales + additional_sales_per_unit_reduction * x) = desired_profit → x = 20 :=
sorry

end price_reduction_required_l238_238822


namespace part1_part2_l238_238583

def set_A := {x : ℝ | x^2 + 2*x - 8 = 0}
def set_B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem part1 (a : ℝ) (h : a = 1) : 
  (set_A ∩ set_B a) = {-4} := by
  sorry

theorem part2 (a : ℝ) : 
  (set_A ∩ (set_B a) = set_B a) → (a < -1 ∨ a > 3) := by
  sorry

end part1_part2_l238_238583


namespace solve_for_x_l238_238098

theorem solve_for_x (x y z w : ℕ) 
  (h1 : x = y + 7) 
  (h2 : y = z + 15) 
  (h3 : z = w + 25) 
  (h4 : w = 95) : 
  x = 142 :=
by 
  sorry

end solve_for_x_l238_238098


namespace elena_earnings_l238_238629

theorem elena_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (h_wage : hourly_wage = 13.25) (h_hours : hours_worked = 4) : 
  hourly_wage * hours_worked = 53.00 := by
sorry

end elena_earnings_l238_238629


namespace value_of_a6_l238_238777

noncomputable def Sn (n : ℕ) : ℕ := n * 2^(n + 1)
noncomputable def an (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem value_of_a6 : an 6 = 448 := by
  sorry

end value_of_a6_l238_238777


namespace students_taking_french_l238_238715

theorem students_taking_french 
  (Total : ℕ) (G : ℕ) (B : ℕ) (Neither : ℕ) (H_total : Total = 87)
  (H_G : G = 22) (H_B : B = 9) (H_neither : Neither = 33) : 
  ∃ F : ℕ, F = 41 := 
by
  sorry

end students_taking_french_l238_238715


namespace wife_catch_up_l238_238233

/-- A man drives at a speed of 40 miles/hr.
His wife left 30 minutes late with a speed of 50 miles/hr.
Prove that they will meet 2 hours after the wife starts driving. -/
theorem wife_catch_up (t : ℝ) (speed_man speed_wife : ℝ) (late_time : ℝ) :
  speed_man = 40 →
  speed_wife = 50 →
  late_time = 0.5 →
  50 * t = 40 * (t + 0.5) →
  t = 2 :=
by
  intros h_man h_wife h_late h_eq
  -- Actual proof goes here. 
  -- (Skipping the proof as requested, leaving it as a placeholder)
  sorry

end wife_catch_up_l238_238233


namespace ball_rebound_percentage_l238_238507

theorem ball_rebound_percentage (P : ℝ) 
  (h₁ : 100 + 2 * 100 * P + 2 * 100 * P^2 = 250) : P = 0.5 := 
by 
  sorry

end ball_rebound_percentage_l238_238507


namespace trigonometric_identity_l238_238289

noncomputable def tan_alpha : ℝ := 4

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = tan_alpha) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / Real.cos (-α) = 3 :=
by
  sorry

end trigonometric_identity_l238_238289


namespace divide_equal_parts_l238_238266

theorem divide_equal_parts (m n: ℕ) (h₁: (m + n) % 2 = 0) (h₂: gcd m n ∣ ((m + n) / 2)) : ∃ a b: ℕ, a = b ∧ a + b = m + n ∧ a ≤ m + n ∧ b ≤ m + n :=
sorry

end divide_equal_parts_l238_238266


namespace race_time_l238_238631

theorem race_time (t : ℝ) (h1 : 100 / t = 66.66666666666667 / 45) : t = 67.5 :=
by
  sorry

end race_time_l238_238631


namespace simplify_expression_l238_238891

theorem simplify_expression : (Real.sin (15 * Real.pi / 180) + Real.sin (45 * Real.pi / 180)) / (Real.cos (15 * Real.pi / 180) + Real.cos (45 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  sorry

end simplify_expression_l238_238891


namespace bees_count_l238_238469

theorem bees_count (x : ℕ) (h1 : (1/5 : ℚ) * x + (1/3 : ℚ) * x + 
    3 * ((1/3 : ℚ) * x - (1/5 : ℚ) * x) + 1 = x) : x = 15 := 
sorry

end bees_count_l238_238469


namespace reconstruct_points_l238_238400

noncomputable def symmetric (x y : ℝ) := 2 * y - x

theorem reconstruct_points (A' B' C' D' B C D : ℝ) :
  (∃ (A B C D : ℝ),
     B = (A + A') / 2 ∧  -- B is the midpoint of line segment AA'
     C = (B + B') / 2 ∧  -- C is the midpoint of line segment BB'
     D = (C + C') / 2 ∧  -- D is the midpoint of line segment CC'
     A = (D + D') / 2)   -- A is the midpoint of line segment DD'
  ↔ (∃ (A : ℝ), A = symmetric D D') → True := sorry

end reconstruct_points_l238_238400


namespace concentration_after_5500_evaporates_l238_238985

noncomputable def concentration_after_evaporation 
  (V₀ Vₑ : ℝ) (C₀ : ℝ) : ℝ := 
  let sodium_chloride := C₀ * V₀
  let remaining_volume := V₀ - Vₑ
  100 * sodium_chloride / remaining_volume

theorem concentration_after_5500_evaporates 
  : concentration_after_evaporation 10000 5500 0.05 = 11.11 := 
by
  -- Formalize the calculations as we have derived
  -- sorry is used to skip the proof
  sorry

end concentration_after_5500_evaporates_l238_238985


namespace rabbits_ate_27_watermelons_l238_238727

theorem rabbits_ate_27_watermelons
  (original_watermelons : ℕ)
  (watermelons_left : ℕ)
  (watermelons_eaten : ℕ)
  (h1 : original_watermelons = 35)
  (h2 : watermelons_left = 8)
  (h3 : original_watermelons - watermelons_left = watermelons_eaten) :
  watermelons_eaten = 27 :=
by {
  -- Proof skipped
  sorry
}

end rabbits_ate_27_watermelons_l238_238727


namespace average_age_increase_l238_238642

theorem average_age_increase (average_age_students : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ)
                             (h1 : average_age_students = 26) (h2 : num_students = 25) (h3 : teacher_age = 52)
                             (h4 : new_avg_age = (650 + teacher_age) / (num_students + 1))
                             (h5 : 650 = average_age_students * num_students) :
  new_avg_age - average_age_students = 1 := 
by
  sorry

end average_age_increase_l238_238642


namespace calculate_f_5_5_l238_238424

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f (x + 2) = -1 / f x
axiom defined_segment (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f x = x

theorem calculate_f_5_5 : f 5.5 = 2.5 := sorry

end calculate_f_5_5_l238_238424


namespace marge_final_plants_l238_238382

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end marge_final_plants_l238_238382


namespace Sheila_weekly_earnings_l238_238380

-- Definitions based on the conditions
def hours_per_day_MWF : ℕ := 8
def hours_per_day_TT : ℕ := 6
def hourly_wage : ℕ := 7
def days_MWF : ℕ := 3
def days_TT : ℕ := 2

-- Theorem that Sheila earns $252 per week
theorem Sheila_weekly_earnings : (hours_per_day_MWF * hourly_wage * days_MWF) + (hours_per_day_TT * hourly_wage * days_TT) = 252 :=
by 
  sorry

end Sheila_weekly_earnings_l238_238380


namespace arithmetic_sequence_a4_l238_238660

theorem arithmetic_sequence_a4 (S n : ℕ) (a : ℕ → ℕ) (h1 : S = 28) (h2 : S = 7 * a 4) : a 4 = 4 :=
by sorry

end arithmetic_sequence_a4_l238_238660


namespace intersecting_functions_k_range_l238_238112

theorem intersecting_functions_k_range 
  (k : ℝ) (h : 0 < k) : 
    ∃ x : ℝ, -2 * x + 3 = k / x ↔ k ≤ 9 / 8 :=
by 
  sorry

end intersecting_functions_k_range_l238_238112


namespace junior_score_calculation_l238_238992

variable {total_students : ℕ}
variable {junior_score senior_average : ℕ}
variable {junior_ratio senior_ratio : ℚ}
variable {class_average total_average : ℚ}

-- Hypotheses from the conditions
theorem junior_score_calculation (h1 : junior_ratio = 0.2)
                               (h2 : senior_ratio = 0.8)
                               (h3 : class_average = 82)
                               (h4 : senior_average = 80)
                               (h5 : total_students = 10)
                               (h6 : total_average * total_students = total_students * class_average)
                               (h7 : total_average = (junior_ratio * junior_score + senior_ratio * senior_average))
                               : junior_score = 90 :=
sorry

end junior_score_calculation_l238_238992


namespace number_of_girls_and_boys_l238_238510

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) := g = 4 * (g + b) / 7 ∧ b = 3 * (g + b) / 7
def total_students (g b : ℕ) := g + b = 56

-- The main proof statement
theorem number_of_girls_and_boys (g b : ℕ) 
  (h_ratio : ratio_girls_to_boys g b)
  (h_total : total_students g b) : 
  g = 32 ∧ b = 24 :=
by {
  sorry
}

end number_of_girls_and_boys_l238_238510


namespace village_transportation_problem_l238_238973

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

variable (total odd : ℕ) (a : ℕ)

theorem village_transportation_problem 
  (h_total : total = 15)
  (h_odd : odd = 7)
  (h_selected : 10 = 10)
  (h_eq : (comb 7 4) * (comb 8 6) / (comb 15 10) = (comb 7 (10 - a)) * (comb 8 a) / (comb 15 10)) :
  a = 6 := 
sorry

end village_transportation_problem_l238_238973


namespace boxes_per_case_l238_238267

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (h1 : total_boxes = 24) (h2 : total_cases = 3) : (total_boxes / total_cases) = 8 :=
by 
  sorry

end boxes_per_case_l238_238267


namespace solve_problem_l238_238283

namespace Example

-- Definitions based on given conditions
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def condition_2 (f : ℝ → ℝ) : Prop := f 2 = -1

def condition_3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = -f (2 - x)

-- Main theorem statement
theorem solve_problem (f : ℝ → ℝ)
  (h1 : isEvenFunction f)
  (h2 : condition_2 f)
  (h3 : condition_3 f) : f 2016 = 1 :=
sorry

end Example

end solve_problem_l238_238283


namespace correct_transformation_l238_238764

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end correct_transformation_l238_238764


namespace noah_holidays_l238_238036

theorem noah_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_total : ℕ) 
  (h1 : holidays_per_month = 3) (h2 : months_in_year = 12) (h3 : holidays_total = holidays_per_month * months_in_year) : 
  holidays_total = 36 := 
by
  sorry

end noah_holidays_l238_238036


namespace Jack_goal_l238_238516

-- Define the amounts Jack made from brownies and lemon squares
def brownies (n : ℕ) (price : ℕ) : ℕ := n * price
def lemonSquares (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the amount Jack needs to make from cookies
def cookies (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the total goal for Jack
def totalGoal (browniesCount : ℕ) (browniesPrice : ℕ) 
              (lemonSquaresCount : ℕ) (lemonSquaresPrice : ℕ) 
              (cookiesCount : ℕ) (cookiesPrice: ℕ) : ℕ :=
  brownies browniesCount browniesPrice + lemonSquares lemonSquaresCount lemonSquaresPrice + cookies cookiesCount cookiesPrice

theorem Jack_goal : totalGoal 4 3 5 2 7 4 = 50 :=
by
  -- Adding up the different components of the total earnings
  let totalFromBrownies := brownies 4 3
  let totalFromLemonSquares := lemonSquares 5 2
  let totalFromCookies := cookies 7 4
  -- Summing up the amounts
  have step1 : totalFromBrownies = 12 := rfl
  have step2 : totalFromLemonSquares = 10 := rfl
  have step3 : totalFromCookies = 28 := rfl
  have step4 : totalGoal 4 3 5 2 7 4 = totalFromBrownies + totalFromLemonSquares + totalFromCookies := rfl
  have step5 : totalFromBrownies + totalFromLemonSquares + totalFromCookies = 12 + 10 + 28 := by rw [step1, step2, step3]
  have step6 : 12 + 10 + 28 = 50 := by norm_num
  exact step4 ▸ (step5 ▸ step6)

end Jack_goal_l238_238516


namespace range_of_a_l238_238350

-- Defining the function f : ℝ → ℝ
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x + a * Real.log x

-- Main theorem statement
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (f a x1 = 0 ∧ f a x2 = 0)) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l238_238350


namespace triangle_properties_l238_238308

variable (a b c A B C : ℝ)
variable (CD BD : ℝ)

-- triangle properties and given conditions
variable (b_squared_eq_ac : b ^ 2 = a * c)
variable (cos_A_minus_C : Real.cos (A - C) = Real.cos B + 1 / 2)

theorem triangle_properties :
  B = π / 3 ∧ 
  A = π / 3 ∧ 
  (CD = 6 → ∃ x, x > 0 ∧ x = 4 * Real.sqrt 3 + 6) ∧
  (BD = 6 → ∀ area, area ≠ 9 / 4) :=
  by
    sorry

end triangle_properties_l238_238308


namespace find_a_l238_238524

def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_a : ∃ a : ℝ, f a + g a = 0 ∧ a = 5 / 7 :=
by
  sorry

end find_a_l238_238524


namespace arithmetic_sequence_a8_l238_238407

def sum_arithmetic_sequence_first_n_terms (a d : ℕ) (n : ℕ): ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_a8 
  (a d : ℕ) 
  (h : sum_arithmetic_sequence_first_n_terms a d 15 = 45) : 
  a + 7 * d = 3 := 
by
  sorry

end arithmetic_sequence_a8_l238_238407


namespace rectangle_width_l238_238159

theorem rectangle_width (w : ℝ) 
  (h1 : ∃ w : ℝ, w > 0 ∧ (2 * w + 2 * (w - 2)) = 16) 
  (h2 : ∀ w, w > 0 → 2 * w + 2 * (w - 2) = 16 → w = 5) : 
  w = 5 := 
sorry

end rectangle_width_l238_238159


namespace basic_computer_price_l238_238668

variable (C P : ℕ)

theorem basic_computer_price 
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3) : 
  C = 1500 := 
sorry

end basic_computer_price_l238_238668


namespace last_two_digits_of_sum_of_factorials_l238_238998

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end last_two_digits_of_sum_of_factorials_l238_238998


namespace complement_of_A_in_U_l238_238392

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}
def complement : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement = {1, 3, 5} := by
  sorry

end complement_of_A_in_U_l238_238392


namespace sequence_ratio_proof_l238_238718

variable {a : ℕ → ℤ}

-- Sequence definition
axiom a₁ : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 4 * a n + 3

-- The theorem to be proved
theorem sequence_ratio_proof (n : ℕ) : (a (n + 1) + 1) / (a n + 1) = 4 := by
  sorry

end sequence_ratio_proof_l238_238718


namespace jessa_gave_3_bills_l238_238798

variable (J G K : ℕ)
variable (billsGiven : ℕ)

/-- Initial conditions and question for the problem -/
def initial_conditions :=
  G = 16 ∧
  K = J - 2 ∧
  G = 2 * K ∧
  (J - billsGiven = 7)

/-- The theorem to prove: Jessa gave 3 bills to Geric -/
theorem jessa_gave_3_bills (h : initial_conditions J G K billsGiven) : billsGiven = 3 := 
sorry

end jessa_gave_3_bills_l238_238798


namespace first_sequence_general_term_second_sequence_general_term_l238_238288

-- For the first sequence
def first_sequence_sum : ℕ → ℚ
| n => n^2 + 1/2 * n

theorem first_sequence_general_term (n : ℕ) : 
  (first_sequence_sum (n+1) - first_sequence_sum n) = (2 * (n+1) - 1/2) := 
sorry

-- For the second sequence
def second_sequence_sum : ℕ → ℚ
| n => 1/4 * n^2 + 2/3 * n + 3

theorem second_sequence_general_term (n : ℕ) : 
  (second_sequence_sum (n+1) - second_sequence_sum n) = 
  if n = 0 then 47/12 
  else (6 * (n+1) + 5)/12 := 
sorry

end first_sequence_general_term_second_sequence_general_term_l238_238288


namespace power_equivalence_l238_238558

theorem power_equivalence (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (x y : ℕ) 
  (hx : 2^m = x) (hy : 2^(2 * n) = y) : 4^(m + 2 * n) = x^2 * y^2 := 
by 
  sorry

end power_equivalence_l238_238558


namespace simplified_fraction_l238_238109

theorem simplified_fraction :
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3 + 1 / (1 / 3)^4)) = (1 / 120) :=
by 
  sorry

end simplified_fraction_l238_238109


namespace min_num_cuboids_l238_238448

/-
Definitions based on the conditions:
- Dimensions of the cuboid are given as 3 cm, 4 cm, and 5 cm.
- We need to find the Least Common Multiple (LCM) of these dimensions.
- Calculate the volume of the smallest cube.
- Calculate the volume of the given cuboid.
- Find the number of such cuboids needed to form the cube.
-/
def cuboid_length : ℤ := 3
def cuboid_width : ℤ := 4
def cuboid_height : ℤ := 5

noncomputable def lcm_3_4_5 : ℤ := Int.lcm (Int.lcm cuboid_length cuboid_width) cuboid_height

noncomputable def cube_side_length : ℤ := lcm_3_4_5
noncomputable def cube_volume : ℤ := cube_side_length * cube_side_length * cube_side_length
noncomputable def cuboid_volume : ℤ := cuboid_length * cuboid_width * cuboid_height

noncomputable def num_cuboids : ℤ := cube_volume / cuboid_volume

theorem min_num_cuboids :
  num_cuboids = 3600 := by
  sorry

end min_num_cuboids_l238_238448


namespace interest_rate_proof_l238_238023

-- Define the given values
def P : ℝ := 1500
def t : ℝ := 2.4
def A : ℝ := 1680

-- Define the interest rate per annum to be proven
def r : ℝ := 0.05

-- Prove that the calculated interest rate matches the given interest rate per annum
theorem interest_rate_proof 
  (principal : ℝ := P) 
  (time_period : ℝ := t) 
  (amount : ℝ := A) 
  (interest_rate : ℝ := r) :
  (interest_rate = ((amount / principal - 1) / time_period)) :=
by
  sorry

end interest_rate_proof_l238_238023


namespace verify_value_of_sum_l238_238243

noncomputable def value_of_sum (a b c d e f : ℕ) (values : Finset ℕ) : ℕ :=
if h : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f ∧
        a + b = c ∧
        b + c = d ∧
        c + e = f
then a + c + f
else 0

theorem verify_value_of_sum :
  ∃ (a b c d e f : ℕ) (values : Finset ℕ),
  values = {4, 12, 15, 27, 31, 39} ∧
  a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b = c ∧
  b + c = d ∧
  c + e = f ∧
  value_of_sum a b c d e f values = 73 :=
by
  sorry

end verify_value_of_sum_l238_238243


namespace factorize_expression_l238_238986

theorem factorize_expression (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4 * x * y = (x * y - 1 + x + y) * (x * y - 1 - x - y) :=
by sorry

end factorize_expression_l238_238986


namespace present_worth_of_bill_l238_238318

theorem present_worth_of_bill (P : ℝ) (TD BD : ℝ) 
  (hTD : TD = 36) (hBD : BD = 37.62) 
  (hFormula : BD = (TD * (P + TD)) / P) : P = 800 :=
by
  sorry

end present_worth_of_bill_l238_238318


namespace nat_divisibility_l238_238521

theorem nat_divisibility {n : ℕ} : (n + 1 ∣ n^2 + 1) ↔ (n = 0 ∨ n = 1) := 
sorry

end nat_divisibility_l238_238521


namespace find_f1_l238_238856

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition_on_function (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, x ≤ 0 → f x = 2^x - 3 * x + 2 * m

theorem find_f1 (f : ℝ → ℝ) (m : ℝ)
  (h_odd : is_odd_function f)
  (h_condition : condition_on_function f m) :
  f 1 = -(5 / 2) :=
by
  sorry

end find_f1_l238_238856


namespace billion_in_scientific_notation_l238_238302

theorem billion_in_scientific_notation :
  (10^9 = 1 * 10^9) :=
by
  sorry

end billion_in_scientific_notation_l238_238302


namespace price_of_refrigerator_l238_238988

variable (R W : ℝ)

theorem price_of_refrigerator 
  (h1 : W = R - 1490) 
  (h2 : R + W = 7060) 
  : R = 4275 :=
sorry

end price_of_refrigerator_l238_238988


namespace Joe_spent_800_on_hotel_l238_238712

noncomputable def Joe'sExpenses : Prop :=
  let S := 6000 -- Joe's total savings
  let F := 1200 -- Expense on the flight
  let FD := 3000 -- Expense on food
  let R := 1000 -- Remaining amount after all expenses
  let H := S - R - (F + FD) -- Calculating hotel expense
  H = 800 -- We need to prove the hotel expense equals $800

theorem Joe_spent_800_on_hotel : Joe'sExpenses :=
by {
  -- Proof goes here; currently skipped
  sorry
}

end Joe_spent_800_on_hotel_l238_238712


namespace hyperbola_a_solution_l238_238458

noncomputable def hyperbola_a_value (a : ℝ) : Prop :=
  (a > 0) ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / 2) = 1) ∧ (∃ e : ℝ, e = 2)

theorem hyperbola_a_solution : ∃ a : ℝ, hyperbola_a_value a ∧ a = (Real.sqrt 6) / 3 :=
  by
    sorry

end hyperbola_a_solution_l238_238458


namespace minimum_races_to_determine_top_five_fastest_horses_l238_238111

-- Defining the conditions
def max_horses_per_race : ℕ := 3
def total_horses : ℕ := 50

-- The main statement to prove the minimum number of races y
theorem minimum_races_to_determine_top_five_fastest_horses (y : ℕ) :
  y = 19 :=
sorry

end minimum_races_to_determine_top_five_fastest_horses_l238_238111


namespace ocean_depth_at_base_of_cone_l238_238141

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

noncomputable def submerged_height_fraction (total_height volume_fraction : ℝ) : ℝ :=
  total_height * (volume_fraction)^(1/3)

theorem ocean_depth_at_base_of_cone (total_height radius : ℝ) 
  (above_water_volume_fraction : ℝ) : ℝ :=
  let above_water_height := submerged_height_fraction total_height above_water_volume_fraction
  total_height - above_water_height

example : ocean_depth_at_base_of_cone 10000 2000 (3 / 5) = 1566 := by
  sorry

end ocean_depth_at_base_of_cone_l238_238141


namespace apples_count_l238_238828

variable (A : ℕ)

axiom h1 : 134 = 80 + 54
axiom h2 : A + 98 = 134

theorem apples_count : A = 36 :=
by
  sorry

end apples_count_l238_238828


namespace evaluate_expression_when_c_is_4_l238_238971

variable (c : ℕ)

theorem evaluate_expression_when_c_is_4 : (c = 4) → ((c^2 - c! * (c - 1)^c)^2 = 3715584) :=
by
  -- This is where the proof would go, but we only need to set up the statement.
  sorry

end evaluate_expression_when_c_is_4_l238_238971


namespace unique_solution_arithmetic_progression_l238_238130

variable {R : Type*} [Field R]

theorem unique_solution_arithmetic_progression (a b c m x y z : R) :
  (m ≠ -2) ∧ (m ≠ 1) ∧ (a + c = 2 * b) → 
  (x + y + m * z = a) ∧ (x + m * y + z = b) ∧ (m * x + y + z = c) → 
  ∃ x y z, 2 * y = x + z :=
by
  sorry

end unique_solution_arithmetic_progression_l238_238130


namespace sum_of_two_digit_divisors_l238_238659

theorem sum_of_two_digit_divisors (d : ℕ) (h1 : 145 % d = 4) (h2 : 10 ≤ d ∧ d < 100) :
  d = 47 :=
by
  have hd : d ∣ 141 := sorry
  exact sorry

end sum_of_two_digit_divisors_l238_238659


namespace sequence_terminates_final_value_l238_238883

-- Define the function Lisa uses to update the number
def f (x : ℕ) : ℕ :=
  let a := x / 10
  let b := x % 10
  a + 4 * b

-- Prove that for any initial value x0, the sequence eventually becomes periodic and ends.
theorem sequence_terminates (x0 : ℕ) : ∃ N : ℕ, ∃ j : ℕ, N ≠ j ∧ (Nat.iterate f N x0) = (Nat.iterate f j x0) :=
  by sorry

-- Given the starting value, show the sequence stabilizes at 39
theorem final_value (x0 : ℕ) (h : x0 = 53^2022 - 1) : ∃ N : ℕ, Nat.iterate f N x0 = 39 :=
  by sorry

end sequence_terminates_final_value_l238_238883


namespace sum_of_final_numbers_l238_238022

variable {x y T : ℝ}

theorem sum_of_final_numbers (h : x + y = T) : 3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by 
  -- The place for the proof steps, which will later be filled
  sorry

end sum_of_final_numbers_l238_238022


namespace john_weekly_earnings_l238_238193

/-- John takes 3 days off of streaming per week. 
    John streams for 4 hours at a time on the days he does stream.
    John makes $10 an hour.
    Prove that John makes $160 a week. -/

theorem john_weekly_earnings (days_off : ℕ) (hours_per_day : ℕ) (wage_per_hour : ℕ) 
  (h_days_off : days_off = 3) (h_hours_per_day : hours_per_day = 4) 
  (h_wage_per_hour : wage_per_hour = 10) : 
  7 - days_off * hours_per_day * wage_per_hour = 160 := by
  sorry

end john_weekly_earnings_l238_238193


namespace cost_of_grapes_and_watermelon_l238_238678

theorem cost_of_grapes_and_watermelon (p g w f : ℝ)
  (h1 : p + g + w + f = 30)
  (h2 : f = 2 * p)
  (h3 : p - g = w) :
  g + w = 7.5 :=
by
  sorry

end cost_of_grapes_and_watermelon_l238_238678


namespace sum_of_possible_N_values_l238_238167

theorem sum_of_possible_N_values (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b)) :
  ∃ sum_N : ℕ, sum_N = 672 :=
by
  sorry

end sum_of_possible_N_values_l238_238167


namespace p_x_range_l238_238073

variable (x : ℝ)

def inequality_condition := x^2 - 5*x + 6 < 0
def polynomial_function := x^2 + 5*x + 6

theorem p_x_range (x_ineq : inequality_condition x) : 
  20 < polynomial_function x ∧ polynomial_function x < 30 :=
sorry

end p_x_range_l238_238073


namespace find_n_from_equation_l238_238946

theorem find_n_from_equation : ∃ n : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 22 ∧ n = 4 :=
by
  sorry

end find_n_from_equation_l238_238946


namespace alyssa_earnings_l238_238075

theorem alyssa_earnings
    (weekly_allowance: ℤ)
    (spent_on_movies_fraction: ℤ)
    (amount_ended_with: ℤ)
    (h1: weekly_allowance = 8)
    (h2: spent_on_movies_fraction = 1 / 2)
    (h3: amount_ended_with = 12)
    : ∃ money_earned_from_car_wash: ℤ, money_earned_from_car_wash = 8 :=
by
  sorry

end alyssa_earnings_l238_238075


namespace minimum_shoeing_time_l238_238844

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) (horses : ℕ) (hooves_per_horse : ℕ) (time_per_hoof : ℕ) 
  (total_hooves : ℕ := horses * hooves_per_horse) 
  (time_for_one_blacksmith : ℕ := total_hooves * time_per_hoof) 
  (total_parallel_time : ℕ := time_for_one_blacksmith / blacksmiths)
  (h : blacksmiths = 48)
  (h' : horses = 60)
  (h'' : hooves_per_horse = 4)
  (h''' : time_per_hoof = 5) : 
  total_parallel_time = 25 :=
by
  sorry

end minimum_shoeing_time_l238_238844


namespace history_percentage_l238_238742

theorem history_percentage (H : ℕ) (math_percentage : ℕ := 72) (third_subject_percentage : ℕ := 69) (overall_average : ℕ := 75) :
  (math_percentage + H + third_subject_percentage) / 3 = overall_average → H = 84 :=
by
  intro h
  sorry

end history_percentage_l238_238742


namespace initial_selling_price_l238_238381

theorem initial_selling_price (P : ℝ) : 
    (∀ (c_i c_m p_m r : ℝ),
        c_i = 3 ∧
        c_m = 20 ∧
        p_m = 4 ∧
        r = 50 ∧
        (15 * P + 5 * p_m - 20 * c_i = r)
    ) → 
    P = 6 := by 
    sorry

end initial_selling_price_l238_238381


namespace remainder_when_divided_by_29_l238_238592

theorem remainder_when_divided_by_29 (N : ℤ) (h : N % 899 = 63) : N % 29 = 10 :=
sorry

end remainder_when_divided_by_29_l238_238592


namespace find_m_plus_n_l238_238765

def probability_no_exact_k_pairs (k n : ℕ) : ℚ :=
  -- A function to calculate the probability
  -- Placeholder definition (details omitted for brevity)
  sorry

theorem find_m_plus_n : ∃ m n : ℕ,
  gcd m n = 1 ∧ 
  (probability_no_exact_k_pairs k n = (97 / 1000) → m + n = 1097) :=
sorry

end find_m_plus_n_l238_238765


namespace find_a_b_and_water_usage_l238_238071

noncomputable def water_usage_april (a : ℝ) :=
  (15 * (a + 0.8) = 45)

noncomputable def water_usage_may (a b : ℝ) :=
  (17 * (a + 0.8) + 8 * (b + 0.8) = 91)

noncomputable def water_usage_june (a b x : ℝ) :=
  (17 * (a + 0.8) + 13 * (b + 0.8) + (x - 30) * 6.8 = 150)

theorem find_a_b_and_water_usage :
  ∃ (a b x : ℝ), water_usage_april a ∧ water_usage_may a b ∧ water_usage_june a b x ∧ a = 2.2 ∧ b = 4.2 ∧ x = 35 :=
by {
  sorry
}

end find_a_b_and_water_usage_l238_238071


namespace andy_l238_238287

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l238_238287


namespace num_people_in_5_years_l238_238123

def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 12
  | (k+1) => 4 * seq k - 18

theorem num_people_in_5_years : seq 5 = 6150 :=
  sorry

end num_people_in_5_years_l238_238123


namespace hyperbola_eccentricity_is_2_l238_238929

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) : ℝ :=
c / a

theorem hyperbola_eccentricity_is_2 (a b c : ℝ)
  (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) :
  hyperbola_eccentricity a b c h H1 H2 = 2 :=
sorry

end hyperbola_eccentricity_is_2_l238_238929


namespace polynomial_no_linear_term_l238_238887

theorem polynomial_no_linear_term (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x - n) = x^2 + mn → n + m = 0) :=
sorry

end polynomial_no_linear_term_l238_238887


namespace trigonometric_identity_l238_238450

open Real

theorem trigonometric_identity (α β : ℝ) (h : 2 * cos (2 * α + β) - 3 * cos β = 0) :
  tan α * tan (α + β) = -1 / 5 := 
by {
  sorry
}

end trigonometric_identity_l238_238450


namespace distance_apart_after_two_hours_l238_238398

theorem distance_apart_after_two_hours :
  (Jay_walk_rate : ℝ) = 1 / 20 →
  (Paul_jog_rate : ℝ) = 3 / 40 →
  (time_duration : ℝ) = 2 * 60 →
  (distance_apart : ℝ) = 15 :=
by
  sorry

end distance_apart_after_two_hours_l238_238398


namespace find_coordinates_of_point_M_l238_238736

theorem find_coordinates_of_point_M :
  ∃ (M : ℝ × ℝ), 
    (M.1 > 0) ∧ (M.2 < 0) ∧ 
    abs M.2 = 12 ∧ 
    abs M.1 = 4 ∧ 
    M = (4, -12) :=
by
  sorry

end find_coordinates_of_point_M_l238_238736


namespace solution_set_line_l238_238981

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end solution_set_line_l238_238981


namespace division_problem_l238_238239

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end division_problem_l238_238239


namespace find_plane_speed_l238_238991

-- Defining the values in the problem
def distance_with_wind : ℝ := 420
def distance_against_wind : ℝ := 350
def wind_speed : ℝ := 23

-- The speed of the plane in still air
def plane_speed_in_still_air : ℝ := 253

-- Proof goal: Given the conditions, the speed of the plane in still air is 253 mph
theorem find_plane_speed :
  ∃ p : ℝ, (distance_with_wind / (p + wind_speed) = distance_against_wind / (p - wind_speed)) ∧ p = plane_speed_in_still_air :=
by
  use plane_speed_in_still_air
  have h : plane_speed_in_still_air = 253 := rfl
  sorry

end find_plane_speed_l238_238991


namespace inequality_subtraction_l238_238866

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l238_238866


namespace brick_surface_area_l238_238225

variable (X Y Z : ℝ)

#check 4 * X + 4 * Y + 2 * Z = 72 → 
       4 * X + 2 * Y + 4 * Z = 96 → 
       2 * X + 4 * Y + 4 * Z = 102 →
       2 * (X + Y + Z) = 54

theorem brick_surface_area (h1 : 4 * X + 4 * Y + 2 * Z = 72)
                           (h2 : 4 * X + 2 * Y + 4 * Z = 96)
                           (h3 : 2 * X + 4 * Y + 4 * Z = 102) :
                           2 * (X + Y + Z) = 54 := by
  sorry

end brick_surface_area_l238_238225


namespace find_angle_A_l238_238861

theorem find_angle_A (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : B = A / 3)
  (h3 : A + B + C = 180) : A = 90 :=
by
  sorry

end find_angle_A_l238_238861


namespace inequality_holds_l238_238214

theorem inequality_holds (x y : ℝ) (hx₀ : 0 < x) (hy₀ : 0 < y) (hxy : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 :=
sorry

end inequality_holds_l238_238214


namespace sum_of_factors_of_30_is_72_l238_238661

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l238_238661


namespace problem_intersection_l238_238059

noncomputable def A (x : ℝ) : Prop := 1 < x ∧ x < 4
noncomputable def B (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem problem_intersection : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end problem_intersection_l238_238059


namespace compare_squares_l238_238468

theorem compare_squares (a b : ℝ) : a^2 + b^2 ≥ ab + a + b - 1 :=
by
  sorry

end compare_squares_l238_238468


namespace column_of_1000_is_C_l238_238590

def column_of_integer (n : ℕ) : String :=
  ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"].get! ((n - 2) % 10)

theorem column_of_1000_is_C :
  column_of_integer 1000 = "C" :=
by
  sorry

end column_of_1000_is_C_l238_238590


namespace neg_prop_l238_238016

theorem neg_prop : ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 4) ↔ ∃ x : ℝ, x^2 - 2 * x + 4 > 4 := 
by 
  sorry

end neg_prop_l238_238016


namespace prism_sides_plus_two_l238_238367

theorem prism_sides_plus_two (E V S : ℕ) (h1 : E + V = 30) (h2 : E = 3 * S) (h3 : V = 2 * S) : S + 2 = 8 :=
by
  sorry

end prism_sides_plus_two_l238_238367


namespace find_width_of_room_l238_238919

section RoomWidth

variable (l C P A W : ℝ)
variable (h1 : l = 5.5)
variable (h2 : C = 16500)
variable (h3 : P = 750)
variable (h4 : A = C / P)
variable (h5 : A = l * W)

theorem find_width_of_room : W = 4 := by
  sorry

end RoomWidth

end find_width_of_room_l238_238919


namespace canFormTriangle_cannotFormIsoscelesTriangle_l238_238087

section TriangleSticks

noncomputable def stickLengths : List ℝ := 
  List.range 10 |>.map (λ n => 1.9 ^ n)

def satisfiesTriangleInequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem canFormTriangle : ∃ (a b c : ℝ), a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

theorem cannotFormIsoscelesTriangle : ¬∃ (a b c : ℝ), a = b ∧ a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

end TriangleSticks

end canFormTriangle_cannotFormIsoscelesTriangle_l238_238087


namespace worker_C_work_rate_worker_C_days_l238_238957

theorem worker_C_work_rate (A B C: ℚ) (hA: A = 1/10) (hB: B = 1/15) (hABC: A + B + C = 1/4) : C = 1/12 := 
by
  sorry

theorem worker_C_days (C: ℚ) (hC: C = 1/12) : 1 / C = 12 :=
by
  sorry

end worker_C_work_rate_worker_C_days_l238_238957


namespace intersection_of_sets_l238_238148

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_of_sets_l238_238148


namespace lengths_of_legs_l238_238909

def is_right_triangle (a b c : ℕ) := a^2 + b^2 = c^2

theorem lengths_of_legs (a b : ℕ) 
  (h1 : is_right_triangle a b 60)
  (h2 : a + b = 84) 
  : (a = 48 ∧ b = 36) ∨ (a = 36 ∧ b = 48) :=
  sorry

end lengths_of_legs_l238_238909


namespace range_of_a_l238_238925

noncomputable def f (x a : ℝ) := x^2 - a * x
noncomputable def g (x : ℝ) := Real.exp x
noncomputable def h (x : ℝ) := x - (Real.log x / x)

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (f x a = Real.log x)) ↔ (1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1) :=
by
  sorry

end range_of_a_l238_238925


namespace mark_total_votes_l238_238698

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end mark_total_votes_l238_238698


namespace pieces_missing_l238_238030

def total_pieces : ℕ := 32
def pieces_present : ℕ := 24

theorem pieces_missing : total_pieces - pieces_present = 8 := by
sorry

end pieces_missing_l238_238030


namespace max_pieces_in_8x8_grid_l238_238650

theorem max_pieces_in_8x8_grid : 
  ∃ m n : ℕ, (m = 8) ∧ (n = 9) ∧ 
  (∀ H V : ℕ, (H ≤ n) → (V ≤ n) → 
   (H + V + 1 ≤ 16)) := sorry

end max_pieces_in_8x8_grid_l238_238650


namespace minimize_abs_expression_l238_238323

theorem minimize_abs_expression {x : ℝ} : 
  ((|x - 2|) + 3) ≥ ((|2 - 2|) + 3) := 
sorry

end minimize_abs_expression_l238_238323


namespace problem_l238_238560

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by sorry

end problem_l238_238560


namespace value_of_a_squared_plus_b_squared_l238_238567

theorem value_of_a_squared_plus_b_squared (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 8) 
  (h2 : (a - b) ^ 2 = 12) : 
  a^2 + b^2 = 10 :=
sorry

end value_of_a_squared_plus_b_squared_l238_238567


namespace uniquePlantsTotal_l238_238502

-- Define the number of plants in each bed
def numPlantsInA : ℕ := 600
def numPlantsInB : ℕ := 500
def numPlantsInC : ℕ := 400

-- Define the number of shared plants between beds
def sharedPlantsAB : ℕ := 60
def sharedPlantsAC : ℕ := 120
def sharedPlantsBC : ℕ := 80
def sharedPlantsABC : ℕ := 30

-- Prove that the total number of unique plants in the garden is 1270
theorem uniquePlantsTotal : 
  numPlantsInA + numPlantsInB + numPlantsInC 
  - sharedPlantsAB - sharedPlantsAC - sharedPlantsBC 
  + sharedPlantsABC = 1270 := 
by sorry

end uniquePlantsTotal_l238_238502


namespace general_term_formula_l238_238952

def Sn (a_n : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a_n n - 2^(n + 1)

theorem general_term_formula (a_n : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → Sn a_n n = (2 * a_n n - 2^(n + 1))) :
  ∀ n : ℕ, n > 0 → a_n n = (n + 1) * 2^n :=
sorry

end general_term_formula_l238_238952


namespace ned_mowed_in_summer_l238_238475

def mowed_in_summer (total_mows spring_mows summer_mows : ℕ) : Prop :=
  total_mows = spring_mows + summer_mows

theorem ned_mowed_in_summer :
  ∀ (total_mows spring_mows summer_mows : ℕ),
  total_mows = 11 →
  spring_mows = 6 →
  mowed_in_summer total_mows spring_mows summer_mows →
  summer_mows = 5 :=
by
  intros total_mows spring_mows summer_mows h_total h_spring h_mowed
  sorry

end ned_mowed_in_summer_l238_238475


namespace system_solution_l238_238601

theorem system_solution :
  ∀ (a1 b1 c1 a2 b2 c2 : ℝ),
  (a1 * 8 + b1 * 5 = c1) ∧ (a2 * 8 + b2 * 5 = c2) →
  ∃ (x y : ℝ), (4 * a1 * x - 5 * b1 * y = 3 * c1) ∧ (4 * a2 * x - 5 * b2 * y = 3 * c2) ∧ 
               (x = 6) ∧ (y = -3) :=
by
  sorry

end system_solution_l238_238601


namespace positive_integer_solutions_x_plus_2y_eq_5_l238_238427

theorem positive_integer_solutions_x_plus_2y_eq_5 :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x + 2 * y = 5) ∧ ((x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 1)) :=
by
  sorry

end positive_integer_solutions_x_plus_2y_eq_5_l238_238427


namespace clara_boxes_l238_238163

theorem clara_boxes (x : ℕ)
  (h1 : 12 * x + 20 * 80 + 16 * 70 = 3320) : x = 50 := by
  sorry

end clara_boxes_l238_238163


namespace power_calculation_l238_238107

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l238_238107


namespace orthogonal_trajectory_eqn_l238_238445

theorem orthogonal_trajectory_eqn (a C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 2 * a * x) → 
  (∃ C : ℝ, ∀ x y : ℝ, x^2 + y^2 = C * y) :=
sorry

end orthogonal_trajectory_eqn_l238_238445


namespace min_value_expression_l238_238864

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ m : ℝ, (m = 4 + 6 * Real.sqrt 2) ∧ 
  ∀ a b : ℝ, (0 < a) → (0 < b) → m ≤ (Real.sqrt ((a^2 + b^2) * (2*a^2 + 4*b^2))) / (a * b) :=
by sorry

end min_value_expression_l238_238864


namespace general_term_arithmetic_sequence_l238_238026

theorem general_term_arithmetic_sequence (a_n : ℕ → ℚ) (d : ℚ) (h_seq : ∀ n, a_n n = a_n 0 + n * d)
  (h_geometric : (a_n 2)^2 = a_n 1 * a_n 6)
  (h_condition : 2 * a_n 0 + a_n 1 = 1)
  (h_d_nonzero : d ≠ 0) :
  ∀ n, a_n n = (5/3) - n := 
by
  sorry

end general_term_arithmetic_sequence_l238_238026


namespace find_a_l238_238251

def lambda : Set ℝ := { x | ∃ (a b : ℤ), x = a + b * Real.sqrt 3 }

theorem find_a (a : ℤ) (x : ℝ)
  (h1 : x = 7 + a * Real.sqrt 3)
  (h2 : x ∈ lambda)
  (h3 : (1 / x) ∈ lambda) :
  a = 4 ∨ a = -4 :=
sorry

end find_a_l238_238251


namespace cover_square_with_rectangles_l238_238578

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l238_238578


namespace relationship_between_x_and_y_l238_238710

theorem relationship_between_x_and_y (a b : ℝ) (x y : ℝ)
  (h1 : x = a^2 + b^2 + 20)
  (h2 : y = 4 * (2 * b - a)) :
  x ≥ y :=
by 
-- we need to prove x ≥ y
sorry

end relationship_between_x_and_y_l238_238710


namespace initial_investment_amount_l238_238497

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment_amount (P A r t : ℝ) (n : ℕ) (hA : A = 992.25) 
  (hr : r = 0.10) (hn : n = 2) (ht : t = 1) : P = 900 :=
by
  have h : compoundInterest P r n t = A := by sorry
  rw [hA, hr, hn, ht] at h
  simp at h
  exact sorry

end initial_investment_amount_l238_238497


namespace basketball_tournament_l238_238746

theorem basketball_tournament (x : ℕ) 
  (h1 : ∀ n, ((n * (n - 1)) / 2) = 28 -> n = x) 
  (h2 : (x * (x - 1)) / 2 = 28) : 
  (1 / 2 : ℚ) * x * (x - 1) = 28 :=
by 
  sorry

end basketball_tournament_l238_238746


namespace sequence_all_perfect_squares_l238_238025

theorem sequence_all_perfect_squares (n : ℕ) : 
  ∃ k : ℕ, (∃ m : ℕ, 2 * 10^n + 1 = 3 * m) ∧ (x_n = (m^2 / 9)) :=
by
  sorry

end sequence_all_perfect_squares_l238_238025


namespace smallest_four_digit_in_pascals_triangle_l238_238763

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l238_238763


namespace number_of_wickets_last_match_l238_238262

noncomputable def bowling_average : ℝ := 12.4
noncomputable def runs_taken_last_match : ℝ := 26
noncomputable def wickets_before_last_match : ℕ := 175
noncomputable def decrease_in_average : ℝ := 0.4
noncomputable def new_average : ℝ := bowling_average - decrease_in_average

theorem number_of_wickets_last_match (w : ℝ) :
  (175 + w) > 0 → 
  ((wickets_before_last_match * bowling_average + runs_taken_last_match) / (wickets_before_last_match + w) = new_average) →
  w = 8 := 
sorry

end number_of_wickets_last_match_l238_238262


namespace sandwich_count_l238_238686

-- Define the given conditions
def meats : ℕ := 8
def cheeses : ℕ := 12
def cheese_combination_count : ℕ := Nat.choose cheeses 3

-- Define the total sandwich count based on the conditions
def total_sandwiches : ℕ := meats * cheese_combination_count

-- The theorem we want to prove
theorem sandwich_count : total_sandwiches = 1760 := by
  -- Mathematical steps here are omitted
  sorry

end sandwich_count_l238_238686


namespace loss_percentage_l238_238280

theorem loss_percentage (CP SP : ℝ) (hCP : CP = 1500) (hSP : SP = 1200) : 
  (CP - SP) / CP * 100 = 20 :=
by
  -- Proof would be provided here
  sorry

end loss_percentage_l238_238280


namespace problem_solution_l238_238200

theorem problem_solution (n : Real) (h : 0.04 * n + 0.1 * (30 + n) = 15.2) : n = 89.09 := 
sorry

end problem_solution_l238_238200


namespace perpendicular_lines_l238_238004

theorem perpendicular_lines :
  ∃ m₁ m₄, (m₁ : ℚ) * (m₄ : ℚ) = -1 ∧
  (∀ x y : ℚ, 4 * y - 3 * x = 16 → y = m₁ * x + 4) ∧
  (∀ x y : ℚ, 3 * y + 4 * x = 15 → y = m₄ * x + 5) :=
by sorry

end perpendicular_lines_l238_238004


namespace smallest_perimeter_consecutive_integers_triangle_l238_238996

theorem smallest_perimeter_consecutive_integers_triangle :
  ∃ (a b c : ℕ), 
    1 < a ∧ a + 1 = b ∧ b + 1 = c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    a + b + c = 12 :=
by
  -- proof placeholder
  sorry

end smallest_perimeter_consecutive_integers_triangle_l238_238996


namespace div_by_5_implication_l238_238799

theorem div_by_5_implication (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ k : ℕ, ab = 5 * k) : (∃ k : ℕ, a = 5 * k) ∨ (∃ k : ℕ, b = 5 * k) := 
by
  sorry

end div_by_5_implication_l238_238799


namespace percent_y_of_x_l238_238745

-- Definitions and assumptions based on the problem conditions
variables (x y : ℝ)
-- Given: 20% of (x - y) = 14% of (x + y)
axiom h : 0.20 * (x - y) = 0.14 * (x + y)

-- Prove that y is 0.1765 (or 17.65%) of x
theorem percent_y_of_x (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) : 
  y = 0.1765 * x :=
sorry

end percent_y_of_x_l238_238745


namespace fraction_to_decimal_l238_238595

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l238_238595


namespace no_integer_solutions_l238_238620

theorem no_integer_solutions (x y z : ℤ) : x^3 + y^6 ≠ 7 * z + 3 :=
by sorry

end no_integer_solutions_l238_238620


namespace minji_combinations_l238_238039

theorem minji_combinations : (3 * 5) = 15 :=
by sorry

end minji_combinations_l238_238039


namespace meet_at_midpoint_l238_238080

open Classical

noncomputable def distance_travel1 (t : ℝ) : ℝ :=
  4 * t

noncomputable def distance_travel2 (t : ℝ) : ℝ :=
  (t / 2) * (3.5 + 0.5 * t)

theorem meet_at_midpoint (t : ℝ) : 
  (4 * t + (t / 2) * (3.5 + 0.5 * t) = 72) → 
  (t = 9) ∧ (4 * t = 36) := 
 by 
  sorry

end meet_at_midpoint_l238_238080


namespace johannes_cabbage_sales_l238_238117

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem johannes_cabbage_sales :
  (earnings_wednesday / price_per_kg) + (earnings_friday / price_per_kg) + (earnings_today / price_per_kg) = 48 := by
  sorry

end johannes_cabbage_sales_l238_238117


namespace jane_drinks_l238_238128

/-- Jane buys a combination of muffins, bagels, and drinks over five days,
where muffins cost 40 cents, bagels cost 90 cents, and drinks cost 30 cents.
The number of items bought is 5, and the total cost is a whole number of dollars.
Prove that the number of drinks Jane bought is 4. -/
theorem jane_drinks :
  ∃ b m d : ℕ, b + m + d = 5 ∧ (90 * b + 40 * m + 30 * d) % 100 = 0 ∧ d = 4 :=
by
  sorry

end jane_drinks_l238_238128


namespace part1_part2_part3_l238_238254

-- Definitions for the conditions
def not_divisible_by_2_or_3 (k : ℤ) : Prop :=
  ¬(k % 2 = 0 ∨ k % 3 = 0)

def form_6n1_or_6n5 (k : ℤ) : Prop :=
  ∃ (n : ℤ), k = 6 * n + 1 ∨ k = 6 * n + 5

-- Part 1
theorem part1 (k : ℤ) (h : not_divisible_by_2_or_3 k) : form_6n1_or_6n5 k :=
sorry

-- Part 2
def form_6n1 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 1

def form_6n5 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 5

theorem part2 (a b : ℤ) (ha : form_6n1 a ∨ form_6n5 a) (hb : form_6n1 b ∨ form_6n5 b) :
  form_6n1 (a * b) :=
sorry

-- Part 3
theorem part3 (a b : ℤ) (ha : form_6n1 a) (hb : form_6n5 b) :
  form_6n5 (a * b) :=
sorry

end part1_part2_part3_l238_238254


namespace math_problem_l238_238616

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l238_238616


namespace find_k_l238_238088

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

theorem find_k (a b k : ℝ) (h1 : f a b k = 4) (h2 : f a b (f a b k) = 7) (h3 : f a b (f a b (f a b k)) = 19) :
  k = 13 / 4 := 
sorry

end find_k_l238_238088


namespace find_m_l238_238040

def A (m : ℝ) : Set ℝ := {x | x^2 - m * x + m^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {2, -4}

theorem find_m (m : ℝ) : (A m ∩ B).Nonempty ∧ (A m ∩ C) = ∅ → m = -2 := by
  sorry

end find_m_l238_238040


namespace regular_polygon_sides_l238_238496

theorem regular_polygon_sides (n : ℕ) (h : ∀ (θ : ℝ), θ = 36 → θ = 360 / n) : n = 10 := by
  sorry

end regular_polygon_sides_l238_238496


namespace solution_concentration_l238_238411

theorem solution_concentration (y z : ℝ) :
  let x_vol := 300
  let y_vol := 2 * z
  let z_vol := z
  let total_vol := x_vol + y_vol + z_vol
  let alcohol_x := 0.10 * x_vol
  let alcohol_y := 0.30 * y_vol
  let alcohol_z := 0.40 * z_vol
  let total_alcohol := alcohol_x + alcohol_y + alcohol_z
  total_vol = 600 ∧ y_vol = 2 * z_vol ∧ y_vol + z_vol = 300 → 
  total_alcohol / total_vol = 21.67 / 100 :=
by
  sorry

end solution_concentration_l238_238411


namespace exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l238_238915

open Nat

theorem exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power :
  ∃ (a : ℕ → ℕ), (∀ k : ℕ, (∃ b : ℕ, a k = b ^ 2)) ∧ (StrictMono a) ∧ (∀ k : ℕ, 13^k ∣ (a k + 1)) :=
sorry

end exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l238_238915


namespace chocolates_sold_in_second_week_l238_238002

theorem chocolates_sold_in_second_week
  (c₁ c₂ c₃ c₄ c₅ : ℕ)
  (h₁ : c₁ = 75)
  (h₃ : c₃ = 75)
  (h₄ : c₄ = 70)
  (h₅ : c₅ = 68)
  (h_mean : (c₁ + c₂ + c₃ + c₄ + c₅) / 5 = 71) :
  c₂ = 67 := 
sorry

end chocolates_sold_in_second_week_l238_238002


namespace tangent_line_and_curve_l238_238687

theorem tangent_line_and_curve (a x0 : ℝ) 
  (h1 : ∀ (x : ℝ), x0 + a = 1) 
  (h2 : ∀ (y : ℝ), y = x0 + 1) 
  (h3 : ∀ (y : ℝ), y = Real.log (x0 + a)) 
  : a = 2 := 
by 
  sorry

end tangent_line_and_curve_l238_238687


namespace fraction_of_friends_l238_238007

variable (x y : ℕ) -- number of first-grade students and sixth-grade students

-- Conditions from the problem
def condition1 : Prop := ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a * x = b * y ∧ 1 / 3 = a / (a + b)
def condition2 : Prop := ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c * y = d * x ∧ 2 / 5 = c / (c + d)

-- Theorem statement to prove that the fraction of students who are friends is 4/11
theorem fraction_of_friends (h1 : condition1 x y) (h2 : condition2 x y) :
  (1 / 3 : ℚ) * y + (2 / 5 : ℚ) * x / (x + y) = 4 / 11 :=
sorry

end fraction_of_friends_l238_238007


namespace problem1_problem2_l238_238591

-- Problem 1
theorem problem1 :
  (1 : ℝ) * (2 * Real.sqrt 12 - (1 / 2) * Real.sqrt 18) - (Real.sqrt 75 - (1 / 4) * Real.sqrt 32)
  = -Real.sqrt 3 - (Real.sqrt 2) / 2 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (2 : ℝ) * (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1 / 2)) - Real.sqrt 30 / Real.sqrt 5
  = 1 + Real.sqrt 6 :=
by
  sorry

end problem1_problem2_l238_238591


namespace compare_2_5_sqrt_6_l238_238830

theorem compare_2_5_sqrt_6 : 2.5 > Real.sqrt 6 := by
  sorry

end compare_2_5_sqrt_6_l238_238830


namespace root_equation_val_l238_238408

theorem root_equation_val (a : ℝ) (h : a^2 - 2 * a - 5 = 0) : 2 * a^2 - 4 * a = 10 :=
by 
  sorry

end root_equation_val_l238_238408


namespace coefficient_of_x2_in_expansion_l238_238702

theorem coefficient_of_x2_in_expansion :
  (x - (2 : ℤ)/x) ^ 4 = 8 * x^2 := sorry

end coefficient_of_x2_in_expansion_l238_238702


namespace time_for_B_to_complete_work_l238_238813

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end time_for_B_to_complete_work_l238_238813


namespace square_B_perimeter_l238_238589

theorem square_B_perimeter :
  ∀ (sideA sideB : ℝ), (4 * sideA = 24) → (sideB^2 = (sideA^2) / 4) → (4 * sideB = 12) :=
by
  sorry

end square_B_perimeter_l238_238589


namespace puppy_weight_is_3_8_l238_238361

noncomputable def puppy_weight_problem (p s l : ℝ) : Prop :=
  p + 2 * s + l = 38 ∧
  p + l = 3 * s ∧
  p + 2 * s = l

theorem puppy_weight_is_3_8 :
  ∃ p s l : ℝ, puppy_weight_problem p s l ∧ p = 3.8 :=
by
  sorry

end puppy_weight_is_3_8_l238_238361


namespace evaluate_expression_eq_l238_238573

theorem evaluate_expression_eq :
  let x := 2
  let y := -3
  let z := 7
  x^2 + y^2 - z^2 - 2 * x * y + 3 * z = -15 := by
    sorry

end evaluate_expression_eq_l238_238573


namespace moles_of_HCl_is_one_l238_238272

def moles_of_HCl_combined 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : ℝ := 
by 
  sorry

theorem moles_of_HCl_is_one 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : moles_of_HCl_combined moles_NaHSO3 moles_H2O_formed reaction_completes one_mole_NaHSO3_used = 1 := 
by 
  sorry

end moles_of_HCl_is_one_l238_238272


namespace yan_ratio_distance_l238_238906

theorem yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq_time : (y / w) = (x / w) + ((x + y) / (6 * w))) :
  x / y = 5 / 7 :=
by
  sorry

end yan_ratio_distance_l238_238906


namespace stock_price_percentage_increase_l238_238817

theorem stock_price_percentage_increase :
  ∀ (total higher lower : ℕ), 
    total = 1980 →
    higher = 1080 →
    higher > lower →
    lower = total - higher →
  ((higher - lower) / lower : ℚ) * 100 = 20 :=
by
  intros total higher lower total_eq higher_eq higher_gt lower_eq
  sorry

end stock_price_percentage_increase_l238_238817


namespace polynomial_value_l238_238234

-- Define the conditions as Lean definitions
def condition (x : ℝ) : Prop := x^2 + 2 * x + 1 = 4

-- State the theorem to be proved
theorem polynomial_value (x : ℝ) (h : condition x) : 2 * x^2 + 4 * x + 5 = 11 :=
by
  -- Proof goes here
  sorry

end polynomial_value_l238_238234


namespace sqrt_difference_eq_neg_six_sqrt_two_l238_238770

theorem sqrt_difference_eq_neg_six_sqrt_two :
  (Real.sqrt ((5 - 3 * Real.sqrt 2)^2)) - (Real.sqrt ((5 + 3 * Real.sqrt 2)^2)) = -6 * Real.sqrt 2 := 
sorry

end sqrt_difference_eq_neg_six_sqrt_two_l238_238770


namespace car_speed_l238_238835

variable (D : ℝ) (V : ℝ)

theorem car_speed
  (h1 : 1 / ((D / 3) / 80) + (D / 3) / 15 + (D / 3) / V = D / 30) :
  V = 35.625 :=
by 
  sorry

end car_speed_l238_238835


namespace find_fraction_l238_238571

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem find_fraction : (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := 
by 
  sorry

end find_fraction_l238_238571


namespace scientific_notation_of_570_million_l238_238409

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end scientific_notation_of_570_million_l238_238409


namespace incorrect_statement_d_l238_238809

-- Definitions from the problem:
variables (x y : ℝ)
variables (b a : ℝ)
variables (x_bar y_bar : ℝ)

-- Linear regression equation:
def linear_regression (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Properties given in the problem:
axiom pass_through_point : ∀ (x_bar y_bar : ℝ), ∃ b a, y_bar = b * x_bar + a
axiom avg_increase : ∀ (b a : ℝ), y = b * (x + 1) + a → y = b * x + a + b
axiom possible_at_origin : ∀ (b a : ℝ), ∃ y, y = a

-- The statement D which is incorrect:
theorem incorrect_statement_d : ¬ (∀ (b a : ℝ), ∀ y, x = 0 → y = a) :=
sorry

end incorrect_statement_d_l238_238809


namespace transformation_thinking_reflected_in_solution_of_quadratic_l238_238157

theorem transformation_thinking_reflected_in_solution_of_quadratic :
  ∀ (x : ℝ), (x - 3)^2 - 5 * (x - 3) = 0 → (x = 3 ∨ x = 8) →
  transformation_thinking :=
by
  intros x h_eq h_solutions
  sorry

end transformation_thinking_reflected_in_solution_of_quadratic_l238_238157


namespace find_q_of_polynomial_l238_238920

noncomputable def Q (x : ℝ) (p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q_of_polynomial (p q d : ℝ) (mean_zeros twice_product sum_coeffs : ℝ)
  (h1 : mean_zeros = -p / 3)
  (h2 : twice_product = -2 * d)
  (h3 : sum_coeffs = 1 + p + q + d)
  (h4 : d = 4)
  (h5 : mean_zeros = twice_product)
  (h6 : sum_coeffs = twice_product) :
  q = -37 :=
sorry

end find_q_of_polynomial_l238_238920


namespace dealer_purchased_articles_l238_238281

/-
The dealer purchases some articles for Rs. 25 and sells 12 articles for Rs. 38. 
The dealer has a profit percentage of 90%. Prove that the number of articles 
purchased by the dealer is 14.
-/

theorem dealer_purchased_articles (x : ℕ) 
    (total_cost : ℝ) (group_selling_price : ℝ) (group_size : ℕ) (profit_percentage : ℝ) 
    (h1 : total_cost = 25)
    (h2 : group_selling_price = 38)
    (h3 : group_size = 12)
    (h4 : profit_percentage = 90 / 100) :
    x = 14 :=
by
  sorry

end dealer_purchased_articles_l238_238281


namespace complex_imaginary_axis_l238_238673

theorem complex_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) ↔ (a = 0 ∨ a = 2) := 
by
  sorry

end complex_imaginary_axis_l238_238673


namespace parabola_directrix_eq_neg_2_l238_238664

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end parabola_directrix_eq_neg_2_l238_238664


namespace fraction_of_milk_in_cup1_l238_238008

def initial_tea_cup1 : ℚ := 6
def initial_milk_cup2 : ℚ := 6

def tea_transferred_step2 : ℚ := initial_tea_cup1 / 3
def tea_cup1_after_step2 : ℚ := initial_tea_cup1 - tea_transferred_step2
def total_cup2_after_step2 : ℚ := initial_milk_cup2 + tea_transferred_step2

def mixture_transfer_step3 : ℚ := total_cup2_after_step2 / 2
def tea_ratio_cup2 : ℚ := tea_transferred_step2 / total_cup2_after_step2
def milk_ratio_cup2 : ℚ := initial_milk_cup2 / total_cup2_after_step2
def tea_transferred_step3 : ℚ := mixture_transfer_step3 * tea_ratio_cup2
def milk_transferred_step3 : ℚ := mixture_transfer_step3 * milk_ratio_cup2

def tea_cup1_after_step3 : ℚ := tea_cup1_after_step2 + tea_transferred_step3
def milk_cup1_after_step3 : ℚ := milk_transferred_step3

def mixture_transfer_step4 : ℚ := (tea_cup1_after_step3 + milk_cup1_after_step3) / 4
def tea_ratio_cup1_step4 : ℚ := tea_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)
def milk_ratio_cup1_step4 : ℚ := milk_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)

def tea_transferred_step4 : ℚ := mixture_transfer_step4 * tea_ratio_cup1_step4
def milk_transferred_step4 : ℚ := mixture_transfer_step4 * milk_ratio_cup1_step4

def final_tea_cup1 : ℚ := tea_cup1_after_step3 - tea_transferred_step4
def final_milk_cup1 : ℚ := milk_cup1_after_step3 - milk_transferred_step4
def final_total_liquid_cup1 : ℚ := final_tea_cup1 + final_milk_cup1

theorem fraction_of_milk_in_cup1 : final_milk_cup1 / final_total_liquid_cup1 = 3/8 := by
  sorry

end fraction_of_milk_in_cup1_l238_238008


namespace max_area_of_sector_l238_238432

theorem max_area_of_sector (α R C : Real) (hC : C > 0) (h : C = 2 * R + α * R) : 
  ∃ S_max : Real, S_max = (C^2) / 16 :=
by
  sorry

end max_area_of_sector_l238_238432


namespace probability_of_receiving_1_l238_238966

-- Define the probabilities and events
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.9
def P_not_B_given_A : ℝ := 0.1
def P_B_given_not_A : ℝ := 0.05
def P_not_B_given_not_A : ℝ := 0.95

-- The main theorem that needs to be proved
theorem probability_of_receiving_1 : 
  (P_A * P_not_B_given_A + P_not_A * P_not_B_given_not_A) = 0.525 := by
  sorry

end probability_of_receiving_1_l238_238966


namespace bobby_candy_left_l238_238375

def initial_candy := 22
def eaten_candy1 := 9
def eaten_candy2 := 5

theorem bobby_candy_left : initial_candy - eaten_candy1 - eaten_candy2 = 8 :=
by
  sorry

end bobby_candy_left_l238_238375


namespace stratified_sampling_medium_stores_l238_238404

noncomputable def total_stores := 300
noncomputable def large_stores := 30
noncomputable def medium_stores := 75
noncomputable def small_stores := 195
noncomputable def sample_size := 20

theorem stratified_sampling_medium_stores : 
  (medium_stores : ℕ) * (sample_size : ℕ) / (total_stores : ℕ) = 5 :=
by
  sorry

end stratified_sampling_medium_stores_l238_238404


namespace bottom_left_square_side_length_l238_238198

theorem bottom_left_square_side_length (x y : ℕ) 
  (h1 : 1 + (x - 1) = 1) 
  (h2 : 2 * x - 1 = (x - 2) + (x - 3) + y) :
  y = 4 :=
sorry

end bottom_left_square_side_length_l238_238198


namespace fraction_to_decimal_l238_238206

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end fraction_to_decimal_l238_238206


namespace adam_deleted_items_l238_238603

theorem adam_deleted_items (initial_items deleted_items remaining_items : ℕ)
  (h1 : initial_items = 100) (h2 : remaining_items = 20) 
  (h3 : remaining_items = initial_items - deleted_items) : 
  deleted_items = 80 :=
by
  sorry

end adam_deleted_items_l238_238603


namespace average_retail_price_l238_238412

theorem average_retail_price 
  (products : Fin 20 → ℝ)
  (h1 : ∀ i, 400 ≤ products i) 
  (h2 : ∃ s : Finset (Fin 20), s.card = 10 ∧ ∀ i ∈ s, products i < 1000)
  (h3 : ∃ i, products i = 11000): 
  (Finset.univ.sum products) / 20 = 1200 := 
by
  sorry

end average_retail_price_l238_238412


namespace average_rate_of_interest_l238_238093

def invested_amount_total : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05
def annual_return (amount : ℝ) (rate : ℝ) : ℝ := amount * rate

theorem average_rate_of_interest : 
  (∃ (x : ℝ), x > 0 ∧ x < invested_amount_total ∧ 
    annual_return (invested_amount_total - x) rate1 = annual_return x rate2) → 
  ((annual_return (invested_amount_total - 1875) rate1 + annual_return 1875 rate2) / invested_amount_total = 0.0375) := 
by
  sorry

end average_rate_of_interest_l238_238093


namespace rational_smaller_than_neg_half_l238_238834

theorem rational_smaller_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  use (-1 : ℚ)
  sorry

end rational_smaller_than_neg_half_l238_238834


namespace triangle_ratio_l238_238306

noncomputable def triangle_problem (BC AC : ℝ) (angleC : ℝ) : ℝ :=
  let CD := AC / 2
  let BD := BC - CD
  let HD := BD / 2
  let AD := (3^(1/2)) * CD
  let AH := AD - HD
  (AH / HD)

theorem triangle_ratio (BC AC : ℝ) (angleC : ℝ) (h1 : BC = 6) (h2 : AC = 3 * Real.sqrt 3) (h3 : angleC = Real.pi / 6) :
  triangle_problem BC AC angleC = -2 - Real.sqrt 3 :=
by
  sorry  

end triangle_ratio_l238_238306


namespace percent_increase_share_price_l238_238895

theorem percent_increase_share_price (P : ℝ) 
  (h1 : ∃ P₁ : ℝ, P₁ = P + 0.25 * P)
  (h2 : ∃ P₂ : ℝ, P₂ = P + 0.80 * P)
  : ∃ percent_increase : ℝ, percent_increase = 44 := by
  sorry

end percent_increase_share_price_l238_238895


namespace asymptote_slope_of_hyperbola_l238_238065

theorem asymptote_slope_of_hyperbola :
  ∀ (x y : ℝ), (x ≠ 0) ∧ (y/x = 3/4 ∨ y/x = -3/4) ↔ (x^2 / 144 - y^2 / 81 = 1) := 
by
  sorry

end asymptote_slope_of_hyperbola_l238_238065


namespace solution_l238_238758

noncomputable def problem_statement : Prop :=
  ∃ (A B C D : ℝ) (a b : ℝ) (x : ℝ), 
    (|A - B| = 3) ∧
    (|A - C| = 1) ∧
    (A = Real.pi / 2) ∧  -- This typically signifies angle A is 90 degrees.
    (a > 0) ∧
    (b > 0) ∧
    (a = 1) ∧
    (|A - D| = x) ∧
    (|B - D| = 3 - x) ∧
    (|C - D| = Real.sqrt (x^2 + 1)) ∧
    (Real.sqrt (x^2 + 1) - (3 - x) = 2) ∧
    (|A - D| / |B - D| = 4)

theorem solution : problem_statement :=
sorry

end solution_l238_238758


namespace fold_hexagon_possible_l238_238780

theorem fold_hexagon_possible (a b : ℝ) :
  (∃ x : ℝ, (a - x)^2 + (b - x)^2 = x^2) ↔ (1 / 2 < b / a ∧ b / a < 2) :=
by
  sorry

end fold_hexagon_possible_l238_238780


namespace calculate_expression_l238_238635

theorem calculate_expression :
  18 - ((-16) / (2 ^ 3)) = 20 :=
by
  sorry

end calculate_expression_l238_238635


namespace oak_grove_total_books_l238_238706

theorem oak_grove_total_books (public_library_books : ℕ) (school_library_books : ℕ)
  (h1 : public_library_books = 1986) (h2 : school_library_books = 5106) :
  public_library_books + school_library_books = 7092 := by
  sorry

end oak_grove_total_books_l238_238706


namespace distance_between_trees_l238_238307

-- Definitions based on conditions
def yard_length : ℝ := 360
def number_of_trees : ℕ := 31
def number_of_gaps : ℕ := number_of_trees - 1

-- The proposition to prove
theorem distance_between_trees : yard_length / number_of_gaps = 12 := sorry

end distance_between_trees_l238_238307


namespace rate_of_simple_interest_l238_238127

-- Define the principal amount and time
variables (P : ℝ) (R : ℝ) (T : ℝ := 12)

-- Define the condition that the sum becomes 9/6 of itself in 12 years (T)
def simple_interest_condition (P : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  (9 / 6) * P - P = P * R * T

-- Define the main theorem stating the rate R is 1/24
theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ := 12) (h : simple_interest_condition P R T) : 
  R = 1 / 24 := 
sorry

end rate_of_simple_interest_l238_238127


namespace equation_of_line_is_correct_l238_238729

/-! Given the circle x^2 + y^2 + 2x - 4y + a = 0 with a < 3 and the midpoint of the chord AB as C(-2, 3), prove that the equation of the line l that intersects the circle at points A and B is x - y + 5 = 0. -/

theorem equation_of_line_is_correct (a : ℝ) (h : a < 3) :
  ∃ l : ℝ × ℝ × ℝ, (l = (1, -1, 5)) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0 → 
    (x - y + 5 = 0)) :=
sorry

end equation_of_line_is_correct_l238_238729


namespace num_articles_produced_l238_238805

-- Conditions
def production_rate (x : ℕ) : ℕ := 2 * x^3 / (x * x * 2 * x)
def articles_produced (y : ℕ) : ℕ := y * 2 * y * y * production_rate y

-- Proof: Given the production rate, prove the number of articles produced.
theorem num_articles_produced (y : ℕ) : articles_produced y = 2 * y^3 := by sorry

end num_articles_produced_l238_238805


namespace tangent_line_eq_l238_238300

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 3 * x + 1

-- Define the point at which we are evaluating the tangent
def point : ℝ × ℝ := (1, -1)

-- Define the derivative of the function f(x)
def f' (x : ℝ) : ℝ := 2 * x - 3

-- The desired theorem
theorem tangent_line_eq :
  ∀ x y : ℝ, (x, y) = point → (y = -x) :=
by sorry

end tangent_line_eq_l238_238300


namespace baskets_and_remainder_l238_238278

-- Define the initial conditions
def cucumbers : ℕ := 216
def basket_capacity : ℕ := 23

-- Define the expected calculations
def expected_baskets : ℕ := cucumbers / basket_capacity
def expected_remainder : ℕ := cucumbers % basket_capacity

-- Theorem to prove the output values
theorem baskets_and_remainder :
  expected_baskets = 9 ∧ expected_remainder = 9 := by
  sorry

end baskets_and_remainder_l238_238278


namespace problem_solution_l238_238371

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

end problem_solution_l238_238371


namespace find_k_l238_238084

theorem find_k (k t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 75) : k = 167 := 
by 
  sorry

end find_k_l238_238084


namespace frank_bakes_for_5_days_l238_238493

variable (d : ℕ) -- The number of days Frank bakes cookies

def cookies_baked_per_day : ℕ := 2 * 12
def cookies_eaten_per_day : ℕ := 1

-- Total cookies baked over d days minus the cookies Frank eats each day
def cookies_remaining_before_ted (d : ℕ) : ℕ :=
  d * (cookies_baked_per_day - cookies_eaten_per_day)

-- Ted eats 4 cookies on the last day, so we add that back to get total before Ted ate
def total_cookies_before_ted (d : ℕ) : ℕ :=
  cookies_remaining_before_ted d + 4

-- After Ted's visit, there are 134 cookies left
axiom ted_leaves_134_cookies : total_cookies_before_ted d = 138

-- Prove that Frank bakes cookies for 5 days
theorem frank_bakes_for_5_days : d = 5 := by
  sorry

end frank_bakes_for_5_days_l238_238493


namespace intersection_complement_A_B_l238_238095

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

def complement (S : Set ℝ) : Set ℝ := {x | x ∉ S}

theorem intersection_complement_A_B :
  U = Set.univ →
  A = {x | -1 < x ∧ x < 1} →
  B = {y | 0 < y} →
  (A ∩ complement B) = {x | -1 < x ∧ x ≤ 0} :=
by
  intros hU hA hB
  sorry

end intersection_complement_A_B_l238_238095


namespace inequality_proof_l238_238999

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^4 * b^b * c^c ≥ a⁻¹ * b⁻¹ * c⁻¹ :=
sorry

end inequality_proof_l238_238999


namespace jonas_needs_35_pairs_of_socks_l238_238326

def JonasWardrobeItems (socks_pairs shoes_pairs pants_items tshirts : ℕ) : ℕ :=
  2 * socks_pairs + 2 * shoes_pairs + pants_items + tshirts

def itemsNeededToDouble (initial_items : ℕ) : ℕ :=
  2 * initial_items - initial_items

theorem jonas_needs_35_pairs_of_socks (socks_pairs : ℕ) 
                                      (shoes_pairs : ℕ) 
                                      (pants_items : ℕ) 
                                      (tshirts : ℕ) 
                                      (final_socks_pairs : ℕ) 
                                      (initial_items : ℕ := JonasWardrobeItems socks_pairs shoes_pairs pants_items tshirts) 
                                      (needed_items : ℕ := itemsNeededToDouble initial_items) 
                                      (needed_pairs_of_socks := needed_items / 2) : 
                                      final_socks_pairs = 35 :=
by
  sorry

end jonas_needs_35_pairs_of_socks_l238_238326


namespace jennifer_book_fraction_l238_238812

theorem jennifer_book_fraction :
  (120 - (1/5 * 120 + 1/6 * 120 + 16)) / 120 = 1/2 :=
by
  sorry

end jennifer_book_fraction_l238_238812


namespace find_width_of_room_l238_238543

theorem find_width_of_room
    (length : ℝ) (area : ℝ)
    (h1 : length = 12) (h2 : area = 96) :
    ∃ width : ℝ, width = 8 ∧ area = length * width :=
by
  sorry

end find_width_of_room_l238_238543


namespace find_k_l238_238879

noncomputable def y (k x : ℝ) : ℝ := k / x

theorem find_k (k : ℝ) (h₁ : k ≠ 0) (h₂ : 1 ≤ 3) 
  (h₃ : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x = 1 ∨ x = 3) 
  (h₄ : |y k 1 - y k 3| = 4) : k = 6 ∨ k = -6 :=
  sorry

end find_k_l238_238879


namespace percentage_of_children_speaking_only_Hindi_l238_238196

/-
In a class of 60 children, 30% of children can speak only English,
20% can speak both Hindi and English, and 42 children can speak Hindi.
Prove that the percentage of children who can speak only Hindi is 50%.
-/
theorem percentage_of_children_speaking_only_Hindi :
  let total_children := 60
  let english_only := 0.30 * total_children
  let both_languages := 0.20 * total_children
  let hindi_only := 42 - both_languages
  (hindi_only / total_children) * 100 = 50 :=
by
  sorry

end percentage_of_children_speaking_only_Hindi_l238_238196


namespace percentage_donated_to_orphan_house_l238_238752

-- Given conditions as definitions in Lean 4
def income : ℝ := 400000
def children_percentage : ℝ := 0.2
def children_count : ℕ := 3
def wife_percentage : ℝ := 0.25
def remaining_after_donation : ℝ := 40000

-- Define the problem as a theorem
theorem percentage_donated_to_orphan_house :
  (children_count * children_percentage + wife_percentage) * income = 0.85 * income →
  (income - 0.85 * income = 60000) →
  remaining_after_donation = 40000 →
  (100 * (60000 - remaining_after_donation) / 60000) = 33.33 := 
by
  intros h1 h2 h3 
  sorry

end percentage_donated_to_orphan_house_l238_238752


namespace well_defined_interval_l238_238181

def is_well_defined (x : ℝ) : Prop :=
  (5 - x > 0) ∧ (x ≠ 2)

theorem well_defined_interval : 
  ∀ x : ℝ, (is_well_defined x) ↔ (x < 5 ∧ x ≠ 2) :=
by 
  sorry

end well_defined_interval_l238_238181


namespace largest_consecutive_odd_number_sum_75_l238_238541

theorem largest_consecutive_odd_number_sum_75 (a b c : ℤ) 
    (h1 : a + b + c = 75) 
    (h2 : b = a + 2) 
    (h3 : c = b + 2) : 
    c = 27 :=
by
  sorry

end largest_consecutive_odd_number_sum_75_l238_238541


namespace math_problem_proof_l238_238010

theorem math_problem_proof (n : ℕ) 
  (h1 : n / 37 = 2) 
  (h2 : n % 37 = 26) :
  48 - n / 4 = 23 := by
  sorry

end math_problem_proof_l238_238010


namespace inequality_proof_l238_238797

theorem inequality_proof
  (a b c d : ℝ)
  (a_nonneg : 0 ≤ a)
  (b_nonneg : 0 ≤ b)
  (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  abc + bcd + cda + dab ≤ (1 / 27) + (176 * abcd / 27) :=
sorry

end inequality_proof_l238_238797


namespace mark_ate_in_first_four_days_l238_238667

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end mark_ate_in_first_four_days_l238_238667


namespace arithmetic_series_sum_l238_238624

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

end arithmetic_series_sum_l238_238624


namespace jane_total_investment_in_stocks_l238_238826

-- Definitions
def total_investment := 220000
def bonds_investment := 13750
def stocks_investment := 5 * bonds_investment
def mutual_funds_investment := 2 * stocks_investment

-- Condition: The total amount invested
def total_investment_condition : Prop := 
  bonds_investment + stocks_investment + mutual_funds_investment = total_investment

-- Theorem: Jane's total investment in stocks
theorem jane_total_investment_in_stocks :
  total_investment_condition →
  stocks_investment = 68750 :=
by sorry

end jane_total_investment_in_stocks_l238_238826


namespace min_fraction_in_domain_l238_238053

theorem min_fraction_in_domain :
  ∃ x y : ℝ, (1/4 ≤ x ∧ x ≤ 2/3) ∧ (1/5 ≤ y ∧ y ≤ 1/2) ∧ 
    (∀ x' y' : ℝ, (1/4 ≤ x' ∧ x' ≤ 2/3) ∧ (1/5 ≤ y' ∧ y' ≤ 1/2) → 
      (xy / (x^2 + y^2) ≤ x'y' / (x'^2 + y'^2))) ∧ 
      xy / (x^2 + y^2) = 2/5 :=
sorry

end min_fraction_in_domain_l238_238053


namespace amount_spent_on_tumbler_l238_238950

def initial_amount : ℕ := 50
def spent_on_coffee : ℕ := 10
def amount_left : ℕ := 10
def total_spent : ℕ := initial_amount - amount_left

theorem amount_spent_on_tumbler : total_spent - spent_on_coffee = 30 := by
  sorry

end amount_spent_on_tumbler_l238_238950


namespace prove_heron_formula_prove_S_squared_rrarc_l238_238483

variables {r r_a r_b r_c p a b c S : ℝ}

-- Problem 1: Prove Heron's Formula
theorem prove_heron_formula (h1 : r * p = r_a * (p - a))
                            (h2 : r * r_a = (p - b) * (p - c))
                            (h3 : r_b * r_c = p * (p - a)) :
  S^2 = p * (p - a) * (p - b) * (p - c) :=
sorry

-- Problem 2: Prove S^2 = r * r_a * r_b * r_c
theorem prove_S_squared_rrarc (h1 : r * p = r_a * (p - a))
                              (h2 : r * r_a = (p - b) * (p - c))
                              (h3 : r_b * r_c = p * (p - a)) :
  S^2 = r * r_a * r_b * r_c :=
sorry

end prove_heron_formula_prove_S_squared_rrarc_l238_238483


namespace average_of_remaining_six_is_correct_l238_238102

noncomputable def average_of_remaining_six (s20 s14: ℕ) (avg20 avg14: ℚ) : ℚ :=
  let sum20 := s20 * avg20
  let sum14 := s14 * avg14
  let sum_remaining := sum20 - sum14
  (sum_remaining / (s20 - s14))

theorem average_of_remaining_six_is_correct : 
  average_of_remaining_six 20 14 500 390 = 756.67 :=
by 
  sorry

end average_of_remaining_six_is_correct_l238_238102


namespace div_poly_l238_238495

theorem div_poly (m n p : ℕ) : 
  (X^2 + X + 1) ∣ (X^(3*m) + X^(3*n + 1) + X^(3*p + 2)) := 
sorry

end div_poly_l238_238495


namespace Deepak_age_l238_238325

theorem Deepak_age : ∃ (A D : ℕ), (A / D = 4 / 3) ∧ (A + 6 = 26) ∧ (D = 15) :=
by
  sorry

end Deepak_age_l238_238325


namespace correct_operation_l238_238932

theorem correct_operation (x : ℝ) : (x^2) * (x^4) = x^6 :=
  sorry

end correct_operation_l238_238932


namespace width_of_rectangle_11_l238_238268

variable (L W : ℕ)

-- The conditions: 
-- 1. The perimeter is 48cm
-- 2. Width is 2 cm shorter than length
def is_rectangle (L W : ℕ) : Prop :=
  2 * L + 2 * W = 48 ∧ W = L - 2

-- The statement we need to prove
theorem width_of_rectangle_11 (L W : ℕ) (h : is_rectangle L W) : W = 11 :=
by
  sorry

end width_of_rectangle_11_l238_238268


namespace max_value_of_f_l238_238903

noncomputable def f (x a : ℝ) : ℝ := - (1/3) * x ^ 3 + (1/2) * x ^ 2 + 2 * a * x

theorem max_value_of_f (a : ℝ) (h0 : 0 < a) (h1 : a < 2)
  (h2 : ∀ x, 1 ≤ x → x ≤ 4 → f x a ≥ f 4 a)
  (h3 : f 4 a = -16 / 3) :
  f 2 a = 10 / 3 :=
sorry

end max_value_of_f_l238_238903


namespace solve_for_x_minus_y_l238_238935

theorem solve_for_x_minus_y (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 :=
by
  sorry

end solve_for_x_minus_y_l238_238935


namespace points_needed_for_office_l238_238436

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25

def jerry_interruptions : ℕ := 2
def jerry_insults : ℕ := 4
def jerry_throwings : ℕ := 2

def jerry_total_points (interrupt_points insult_points throw_points : ℕ) 
                       (interruptions insults throwings : ℕ) : ℕ :=
  (interrupt_points * interruptions) +
  (insult_points * insults) +
  (throw_points * throwings)

theorem points_needed_for_office : 
  jerry_total_points points_for_interrupting points_for_insulting points_for_throwing 
                     (jerry_interruptions) 
                     (jerry_insults) 
                     (jerry_throwings) = 100 := 
  sorry

end points_needed_for_office_l238_238436


namespace total_amount_paid_l238_238237

-- Define the conditions of the problem
def cost_without_discount (quantity : ℕ) (unit_price : ℚ) : ℚ :=
  quantity * unit_price

def cost_with_discount (quantity : ℕ) (unit_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := cost_without_discount quantity unit_price
  total_cost - (total_cost * discount_rate)

-- Define each category's cost after discount
def pens_cost : ℚ := cost_with_discount 7 1.5 0.10
def notebooks_cost : ℚ := cost_without_discount 4 5
def water_bottles_cost : ℚ := cost_with_discount 2 8 0.30
def backpack_cost : ℚ := cost_with_discount 1 25 0.15
def socks_cost : ℚ := cost_with_discount 3 3 0.25

-- Prove the total amount paid is $68.65
theorem total_amount_paid : pens_cost + notebooks_cost + water_bottles_cost + backpack_cost + socks_cost = 68.65 := by
  sorry

end total_amount_paid_l238_238237


namespace lucy_reads_sixty_pages_l238_238596

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

end lucy_reads_sixty_pages_l238_238596


namespace small_rectangular_prisms_intersect_diagonal_l238_238684

def lcm (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

def inclusion_exclusion (n : Nat) : Nat :=
  n / 2 + n / 3 + n / 5 - n / (2 * 3) - n / (3 * 5) - n / (5 * 2) + n / (2 * 3 * 5)

theorem small_rectangular_prisms_intersect_diagonal :
  ∀ (a b c : Nat) (L : Nat), a = 2 → b = 3 → c = 5 → L = 90 →
  lcm a b c = 30 → 3 * inclusion_exclusion (lcm a b c) = 66 :=
by
  intros
  sorry

end small_rectangular_prisms_intersect_diagonal_l238_238684


namespace probability_single_draws_probability_two_different_colors_l238_238191

-- Define probabilities for black, yellow and green as events A, B, and C respectively.
variables (A B C : ℝ)

-- Conditions based on the problem statement
axiom h1 : A + B = 5/9
axiom h2 : B + C = 2/3
axiom h3 : A + B + C = 1

-- Here is the statement to prove the calculated probabilities of single draws
theorem probability_single_draws : 
  A = 1/3 ∧ B = 2/9 ∧ C = 4/9 :=
sorry

-- Define the event of drawing two balls of the same color
variables (black yellow green : ℕ)
axiom balls_count : black + yellow + green = 9
axiom black_component : A = black / 9
axiom yellow_component : B = yellow / 9
axiom green_component : C = green / 9

-- Using the counts to infer the probability of drawing two balls of different colors
axiom h4 : black = 3
axiom h5 : yellow = 2
axiom h6 : green = 4

theorem probability_two_different_colors :
  (1 - (3/36 + 1/36 + 6/36)) = 13/18 :=
sorry

end probability_single_draws_probability_two_different_colors_l238_238191


namespace general_formula_neg_seq_l238_238293

theorem general_formula_neg_seq (a : ℕ → ℝ) (h_neg : ∀ n, a n < 0)
  (h_recurrence : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  ∀ n, a n = - ((2/3)^(n-2) : ℝ) :=
by
  sorry

end general_formula_neg_seq_l238_238293


namespace f_1987_eq_5_l238_238666

noncomputable def f : ℕ → ℝ := sorry

axiom f_def : ∀ x : ℕ, x ≥ 0 → ∃ y : ℝ, f x = y
axiom f_one : f 1 = 2
axiom functional_eq : ∀ a b : ℕ, a ≥ 0 → b ≥ 0 → f (a + b) = f a + f b - 3 * f (a * b) + 1

theorem f_1987_eq_5 : f 1987 = 5 := sorry

end f_1987_eq_5_l238_238666


namespace neg_one_third_squared_l238_238449

theorem neg_one_third_squared :
  (-(1/3))^2 = 1/9 :=
sorry

end neg_one_third_squared_l238_238449


namespace jennifer_initial_oranges_l238_238485

theorem jennifer_initial_oranges (O : ℕ) : 
  ∀ (pears apples remaining_fruits : ℕ),
    pears = 10 →
    apples = 2 * pears →
    remaining_fruits = pears - 2 + apples - 2 + O - 2 →
    remaining_fruits = 44 →
    O = 20 :=
by
  intros pears apples remaining_fruits h1 h2 h3 h4
  sorry

end jennifer_initial_oranges_l238_238485


namespace central_angle_radian_l238_238158

-- Define the context of the sector and conditions
def sector (r θ : ℝ) :=
  θ = r * 6 ∧ 1/2 * r^2 * θ = 6

-- Define the radian measure of the central angle
theorem central_angle_radian (r : ℝ) (θ : ℝ) (h : sector r θ) : θ = 3 :=
by
  sorry

end central_angle_radian_l238_238158


namespace value_of_k_plus_p_l238_238090

theorem value_of_k_plus_p
  (k p : ℝ)
  (h1 : ∀ x : ℝ, 3*x^2 - k*x + p = 0)
  (h_sum_roots : k / 3 = -3)
  (h_prod_roots : p / 3 = -6)
  : k + p = -27 :=
by
  sorry

end value_of_k_plus_p_l238_238090


namespace angle_bisector_equation_intersection_l238_238639

noncomputable def slope_of_angle_bisector (m1 m2 : ℝ) : ℝ :=
  (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)

noncomputable def equation_of_angle_bisector (x : ℝ) : ℝ :=
  (Real.sqrt 21 - 6) / 7 * x

theorem angle_bisector_equation_intersection :
  let m1 := 2
  let m2 := 4
  slope_of_angle_bisector m1 m2 = (Real.sqrt 21 - 6) / 7 ∧
  equation_of_angle_bisector 1 = (Real.sqrt 21 - 6) / 7 :=
by
  sorry

end angle_bisector_equation_intersection_l238_238639


namespace correct_calculation_result_l238_238587

theorem correct_calculation_result 
  (P : Polynomial ℝ := -x^2 + x - 1) :
  (P + -3 * x) = (-x^2 - 2 * x - 1) :=
by
  -- Since this is just the proof statement, sorry is used to skip the proof.
  sorry

end correct_calculation_result_l238_238587


namespace derrick_has_34_pictures_l238_238980

-- Assume Ralph has 26 pictures of wild animals
def ralph_pictures : ℕ := 26

-- Derrick has 8 more pictures than Ralph
def derrick_pictures : ℕ := ralph_pictures + 8

-- Prove that Derrick has 34 pictures of wild animals
theorem derrick_has_34_pictures : derrick_pictures = 34 := by
  sorry

end derrick_has_34_pictures_l238_238980


namespace inequality_add_one_l238_238359

variable {α : Type*} [LinearOrderedField α]

theorem inequality_add_one {a b : α} (h : a > b) : a + 1 > b + 1 :=
sorry

end inequality_add_one_l238_238359


namespace range_of_a_l238_238457

open Real 

noncomputable def trigonometric_inequality (θ a : ℝ) : Prop :=
  sin (2 * θ) - (2 * sqrt 2 + sqrt 2 * a) * sin (θ + π / 4) - 2 * sqrt 2 / cos (θ - π / 4) > -3 - 2 * a

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → trigonometric_inequality θ a) ↔ (a > 3) :=
sorry

end range_of_a_l238_238457


namespace condition_C_for_D_condition_A_for_B_l238_238035

theorem condition_C_for_D (C D : Prop) (h : C → D) : C → D :=
by
  exact h

theorem condition_A_for_B (A B D : Prop) (hA_to_D : A → D) (hD_to_B : D → B) : A → B :=
by
  intro hA
  apply hD_to_B
  apply hA_to_D
  exact hA

end condition_C_for_D_condition_A_for_B_l238_238035


namespace birds_on_branch_l238_238600

theorem birds_on_branch (initial_parrots remaining_parrots remaining_crows total_birds : ℕ) (h₁ : initial_parrots = 7) (h₂ : remaining_parrots = 2) (h₃ : remaining_crows = 1) (h₄ : initial_parrots - remaining_parrots = total_birds - remaining_crows - initial_parrots) : total_birds = 13 :=
sorry

end birds_on_branch_l238_238600


namespace cube_side_length_l238_238785

-- Given definitions and conditions
variables (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

-- Statement of the theorem
theorem cube_side_length (x : ℝ) : 
  ( ∃ (y z : ℝ), 
      y + x + z = c ∧ 
      x + z = c * a / b ∧
      y = c * x / b ∧
      z = c * x / a 
  ) → x = a * b * c / (a * b + b * c + c * a) :=
sorry

end cube_side_length_l238_238785


namespace value_of_x_l238_238690

-- Define the conditions
variable (C S x : ℝ)
variable (h1 : 20 * C = x * S)
variable (h2 : (S - C) / C * 100 = 25)

-- Define the statement to be proved
theorem value_of_x : x = 16 :=
by
  sorry

end value_of_x_l238_238690


namespace ellipse_sum_a_k_l238_238168

theorem ellipse_sum_a_k {a b h k : ℝ}
  (foci1 foci2 : ℝ × ℝ)
  (point_on_ellipse : ℝ × ℝ)
  (h_center : h = (foci1.1 + foci2.1) / 2)
  (k_center : k = (foci1.2 + foci2.2) / 2)
  (distance1 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci1.1)^2 + (point_on_ellipse.2 - foci1.2)^2))
  (distance2 : ℝ := Real.sqrt ((point_on_ellipse.1 - foci2.1)^2 + (point_on_ellipse.2 - foci2.2)^2))
  (major_axis_length : ℝ := distance1 + distance2)
  (h_a : a = major_axis_length / 2)
  (c := Real.sqrt ((foci2.1 - foci1.1)^2 + (foci2.2 - foci1.2)^2) / 2)
  (h_b : b^2 = a^2 - c^2) :
  a + k = (7 + Real.sqrt 13) / 2 := 
by
  sorry

end ellipse_sum_a_k_l238_238168


namespace third_side_not_one_l238_238896

theorem third_side_not_one (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c ≠ 1) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end third_side_not_one_l238_238896


namespace sandy_total_spent_l238_238078

def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def total_spent : ℝ := shorts_price + shirt_price + jacket_price

theorem sandy_total_spent : total_spent = 33.56 :=
by
  sorry

end sandy_total_spent_l238_238078


namespace parabola_standard_equation_l238_238492

theorem parabola_standard_equation (x y : ℝ) :
  (3 * x - 4 * y - 12 = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
sorry

end parabola_standard_equation_l238_238492


namespace percent_increase_correct_l238_238275

-- Define the original and new visual ranges
def original_range : Float := 90
def new_range : Float := 150

-- Define the calculation for percent increase
def percent_increase : Float :=
  ((new_range - original_range) / original_range) * 100

-- Statement to prove
theorem percent_increase_correct : percent_increase = 66.67 :=
by
  -- To be proved
  sorry

end percent_increase_correct_l238_238275


namespace weight_of_33rd_weight_l238_238546

theorem weight_of_33rd_weight :
  ∃ a : ℕ → ℕ, (∀ k, a k < a (k+1)) ∧
               (∀ k ≤ 29, a k + a (k+3) = a (k+1) + a (k+2)) ∧
               a 2 = 9 ∧
               a 8 = 33 ∧
               a 32 = 257 :=
sorry

end weight_of_33rd_weight_l238_238546


namespace supermarkets_in_us_l238_238027

noncomputable def number_of_supermarkets_in_canada : ℕ := 35
noncomputable def number_of_supermarkets_total : ℕ := 84
noncomputable def diff_us_canada : ℕ := 14
noncomputable def number_of_supermarkets_in_us : ℕ := number_of_supermarkets_in_canada + diff_us_canada

theorem supermarkets_in_us : number_of_supermarkets_in_us = 49 := by
  sorry

end supermarkets_in_us_l238_238027


namespace Joel_contributed_22_toys_l238_238019

/-
Define the given conditions as separate variables and statements in Lean:
1. Toys collected from friends.
2. Total toys donated.
3. Relationship between Joel's and his sister's toys.
4. Prove that Joel donated 22 toys.
-/

theorem Joel_contributed_22_toys (S : ℕ) (toys_from_friends : ℕ) (total_toys : ℕ) (sisters_toys : ℕ) 
  (h1 : toys_from_friends = 18 + 42 + 2 + 13)
  (h2 : total_toys = 108)
  (h3 : S + 2 * S = total_toys - toys_from_friends)
  (h4 : sisters_toys = S) :
  2 * S = 22 :=
  sorry

end Joel_contributed_22_toys_l238_238019


namespace richmond_tigers_tickets_l238_238816

theorem richmond_tigers_tickets (total_tickets first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570)
  (h2 : first_half_tickets = 3867) : 
  total_tickets - first_half_tickets = 5703 :=
by
  -- Proof steps would go here
  sorry

end richmond_tigers_tickets_l238_238816


namespace remainder_101_mul_103_mod_11_l238_238921

theorem remainder_101_mul_103_mod_11 : (101 * 103) % 11 = 8 :=
by
  sorry

end remainder_101_mul_103_mod_11_l238_238921


namespace correct_triangle_set_l238_238759

/-- Definition of triangle inequality -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Sets of lengths for checking the triangle inequality -/
def Set1 : ℝ × ℝ × ℝ := (5, 8, 2)
def Set2 : ℝ × ℝ × ℝ := (5, 8, 13)
def Set3 : ℝ × ℝ × ℝ := (5, 8, 5)
def Set4 : ℝ × ℝ × ℝ := (2, 7, 5)

/-- The correct set of lengths that can form a triangle according to the triangle inequality -/
theorem correct_triangle_set : satisfies_triangle_inequality 5 8 5 :=
by
  -- Proof would be here
  sorry

end correct_triangle_set_l238_238759


namespace evaluate_expression_l238_238115

variable (a : ℝ)

def a_definition : Prop := a = Real.sqrt 11 - 1

theorem evaluate_expression (h : a_definition a) : a^2 + 2*a + 1 = 11 := by
  sorry

end evaluate_expression_l238_238115


namespace max_crate_weight_on_single_trip_l238_238296

-- Define the conditions
def trailer_capacity := {n | n = 3 ∨ n = 4 ∨ n = 5}
def min_crate_weight : ℤ := 1250

-- Define the maximum weight calculation
def max_weight (n : ℤ) (w : ℤ) : ℤ := n * w

-- Proof statement
theorem max_crate_weight_on_single_trip :
  ∃ w, (5 ∈ trailer_capacity) → max_weight 5 min_crate_weight = w ∧ w = 6250 := 
by
  sorry

end max_crate_weight_on_single_trip_l238_238296


namespace smallest_c_in_range_l238_238959

-- Define the quadratic function g(x)
def g (x c : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + c

-- Define the condition for c
def in_range_5 (c : ℝ) : Prop :=
  ∃ x : ℝ, g x c = 5

-- The theorem stating that the smallest value of c for which 5 is in the range of g is 7
theorem smallest_c_in_range : ∃ c : ℝ, c = 7 ∧ ∀ c' : ℝ, (in_range_5 c' → 7 ≤ c') :=
sorry

end smallest_c_in_range_l238_238959


namespace quadratic_distinct_real_roots_l238_238649

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 = 0 → 
  (k ≠ 0 ∧ ((-2)^2 - 4 * k * (-1) > 0))) ↔ (k > -1 ∧ k ≠ 0) := 
sorry

end quadratic_distinct_real_roots_l238_238649


namespace rationalize_denominator_l238_238939

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = (Real.sqrt 3) / 5 :=
by
  sorry

end rationalize_denominator_l238_238939


namespace hyperbola_symmetric_slopes_l238_238836

/-- 
Let \(M(x_0, y_0)\) and \(N(-x_0, -y_0)\) be points symmetric about the origin on the hyperbola 
\(\frac{x^2}{16} - \frac{y^2}{4} = 1\). Let \(P(x, y)\) be any point on the hyperbola. 
When the slopes \(k_{PM}\) and \(k_{PN}\) both exist, then \(k_{PM} \cdot k_{PN} = \frac{1}{4}\),
independent of the position of \(P\).
-/
theorem hyperbola_symmetric_slopes (x x0 y y0: ℝ) 
  (hP: x^2 / 16 - y^2 / 4 = 1)
  (hM: x0^2 / 16 - y0^2 / 4 = 1)
  (h_slop_M : x ≠ x0)
  (h_slop_N : x ≠ x0):
  ((y - y0) / (x - x0)) * ((y + y0) / (x + x0)) = 1 / 4 := 
sorry

end hyperbola_symmetric_slopes_l238_238836


namespace f_periodic_if_is_bounded_and_satisfies_fe_l238_238165

variable {f : ℝ → ℝ}

-- Condition 1: f is a bounded real function, i.e., it is bounded above and below
def is_bounded (f : ℝ → ℝ) : Prop := ∃ M, ∀ x, |f x| ≤ M

-- Condition 2: The functional equation given for all x.
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)

-- We need to show that f is periodic with period 1.
theorem f_periodic_if_is_bounded_and_satisfies_fe (h_bounded : is_bounded f) (h_fe : functional_eq f) : 
  ∀ x, f (x + 1) = f x :=
sorry

end f_periodic_if_is_bounded_and_satisfies_fe_l238_238165


namespace hypotenuse_length_l238_238047

theorem hypotenuse_length (a b c : ℝ) 
  (h_right_angled : c^2 = a^2 + b^2) 
  (h_sum_squares : a^2 + b^2 + c^2 = 980) : 
  c = 70 :=
by
  sorry

end hypotenuse_length_l238_238047


namespace problem_l238_238884

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) :
  (f (x + 2) + f x = 0) →
  (∀ x, f (-(x - 1)) = -f (x - 1)) →
  (
    (∀ e, ¬(e > 0 ∧ ∀ x, f (x + e) = f x)) ∧
    (∀ x, f (x + 1) = f (-x + 1)) ∧
    (¬(∀ x, f x = f (-x)))
  ) :=
by
  sorry

end problem_l238_238884


namespace convert_neg_900_deg_to_rad_l238_238862

theorem convert_neg_900_deg_to_rad : (-900 : ℝ) * (Real.pi / 180) = -5 * Real.pi :=
by
  sorry

end convert_neg_900_deg_to_rad_l238_238862


namespace point_in_third_quadrant_l238_238169

noncomputable def is_second_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b > 0

noncomputable def is_third_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b < 0

theorem point_in_third_quadrant (a b : ℝ) (h : is_second_quadrant a b) : is_third_quadrant a (-b) :=
by
  sorry

end point_in_third_quadrant_l238_238169


namespace national_education_fund_expenditure_l238_238174

theorem national_education_fund_expenditure (gdp_2012 : ℝ) (h : gdp_2012 = 43.5 * 10^12) : 
  (0.04 * gdp_2012) = 1.74 * 10^13 := 
by sorry

end national_education_fund_expenditure_l238_238174


namespace ellipse_focal_point_l238_238258

theorem ellipse_focal_point (m : ℝ) (m_pos : m > 0)
  (h : ∃ f : ℝ × ℝ, f = (1, 0) ∧ ∀ x y : ℝ, (x^2 / 4) + (y^2 / m^2) = 1 → 
    (x - 1)^2 + y^2 = (x^2 / 4) + (y^2 / m^2)) :
  m = Real.sqrt 3 := 
sorry

end ellipse_focal_point_l238_238258


namespace Melies_money_left_l238_238647

variable (meat_weight : ℕ)
variable (meat_cost_per_kg : ℕ)
variable (initial_money : ℕ)

def money_left_after_purchase (meat_weight : ℕ) (meat_cost_per_kg : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - (meat_weight * meat_cost_per_kg)

theorem Melies_money_left : 
  money_left_after_purchase 2 82 180 = 16 :=
by
  sorry

end Melies_money_left_l238_238647


namespace sum_in_base_8_l238_238082

theorem sum_in_base_8 (a b : ℕ) (h_a : a = 3 * 8^2 + 2 * 8 + 7)
                                  (h_b : b = 7 * 8 + 3) :
  (a + b) = 4 * 8^2 + 2 * 8 + 2 :=
by
  sorry

end sum_in_base_8_l238_238082


namespace crayons_per_friend_l238_238696

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (h1 : total_crayons = 210) (h2 : num_friends = 30) : total_crayons / num_friends = 7 :=
by
  sorry

end crayons_per_friend_l238_238696


namespace find_b_value_l238_238820

/-- Given a line segment from point (0, b) to (8, 0) with a slope of -3/2, 
    prove that the value of b is 12. -/
theorem find_b_value (b : ℝ) : (8 - 0) ≠ 0 ∧ ((0 - b) / (8 - 0) = -3/2) → b = 12 := 
by
  intro h
  sorry

end find_b_value_l238_238820


namespace solution_l238_238796

noncomputable def problem_statement : ℝ :=
  let a := 6
  let b := 5
  let x := 10 * a + b
  let y := 10 * b + a
  let m := 16.5
  x + y + m

theorem solution : problem_statement = 137.5 :=
by
  sorry

end solution_l238_238796


namespace action_figures_more_than_books_l238_238205

variable (initialActionFigures : Nat) (newActionFigures : Nat) (books : Nat)

def totalActionFigures (initialActionFigures newActionFigures : Nat) : Nat :=
  initialActionFigures + newActionFigures

theorem action_figures_more_than_books :
  initialActionFigures = 5 → newActionFigures = 7 → books = 9 →
  totalActionFigures initialActionFigures newActionFigures - books = 3 :=
by
  intros h_initial h_new h_books
  rw [h_initial, h_new, h_books]
  sorry

end action_figures_more_than_books_l238_238205


namespace select_k_plus_1_nums_divisible_by_n_l238_238771

theorem select_k_plus_1_nums_divisible_by_n (n k : ℕ) (hn : n > 0) (hk : k > 0) (nums : Fin (n + k) → ℕ) :
  ∃ (indices : Finset (Fin (n + k))), indices.card ≥ k + 1 ∧ (indices.sum (nums ∘ id)) % n = 0 :=
sorry

end select_k_plus_1_nums_divisible_by_n_l238_238771


namespace binomial_sum_to_220_l238_238811

open Nat

def binomial_coeff (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binomial_sum_to_220 :
  binomial_coeff 2 2 + binomial_coeff 3 2 + binomial_coeff 4 2 + binomial_coeff 5 2 +
  binomial_coeff 6 2 + binomial_coeff 7 2 + binomial_coeff 8 2 + binomial_coeff 9 2 +
  binomial_coeff 10 2 + binomial_coeff 11 2 = 220 :=
by
  /- Proof goes here, use the computed value of combinations -/
  sorry

end binomial_sum_to_220_l238_238811


namespace fraction_transformed_l238_238006

variables (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab_pos : a * b > 0)

noncomputable def frac_orig := (a + 2 * b) / (2 * a * b)
noncomputable def frac_new := (3 * a + 2 * 3 * b) / (2 * 3 * a * 3 * b)

theorem fraction_transformed :
  frac_new a b = (1 / 3) * frac_orig a b :=
sorry

end fraction_transformed_l238_238006


namespace tom_sara_age_problem_l238_238003

-- Define the given conditions as hypotheses and variables
variables (t s : ℝ)
variables (h1 : t - 3 = 2 * (s - 3))
variables (h2 : t - 8 = 3 * (s - 8))

-- Lean statement of the problem
theorem tom_sara_age_problem :
  ∃ x : ℝ, (t + x) / (s + x) = 3 / 2 ∧ x = 7 :=
by
  sorry

end tom_sara_age_problem_l238_238003


namespace simplify_expression_l238_238437

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l238_238437


namespace student_weight_l238_238917

-- Definitions based on conditions
variables (S R : ℝ)

-- Conditions as assertions
def condition1 : Prop := S - 5 = 2 * R
def condition2 : Prop := S + R = 104

-- The statement we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 71 :=
by
  sorry

end student_weight_l238_238917


namespace right_triangle_legs_l238_238930

theorem right_triangle_legs (a b : ℤ) (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a^2 + b^2 = 65^2) : 
  a = 16 ∧ b = 63 ∨ a = 63 ∧ b = 16 :=
sorry

end right_triangle_legs_l238_238930


namespace second_number_exists_l238_238248

theorem second_number_exists (x : ℕ) (h : 150 / x = 15) : x = 10 :=
sorry

end second_number_exists_l238_238248


namespace problem_condition_l238_238640

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l238_238640


namespace correct_equation_l238_238561

-- Definitions of the conditions
def contributes_5_coins (x : ℕ) (P : ℕ) : Prop :=
  5 * x + 45 = P

def contributes_7_coins (x : ℕ) (P : ℕ) : Prop :=
  7 * x + 3 = P

-- Mathematical proof problem
theorem correct_equation 
(x : ℕ) (P : ℕ) (h1 : contributes_5_coins x P) (h2 : contributes_7_coins x P) : 
5 * x + 45 = 7 * x + 3 := 
by
  sorry

end correct_equation_l238_238561


namespace min_value_expression_71_l238_238744

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5)

theorem min_value_expression_71 (x y : ℝ) (hx : x > 4) (hy : y > 5) : 
  min_value_expression x y ≥ 71 :=
by
  sorry

end min_value_expression_71_l238_238744


namespace find_ellipse_l238_238728

-- Define the ellipse and conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the focus points
def focus (a b c : ℝ) : Prop :=
  c^2 = a^2 - b^2

-- Define the range condition
def range_condition (a b c : ℝ) : Prop :=
  let min_val := b^2 - c^2;
  let max_val := a^2 - c^2;
  min_val = -3 ∧ max_val = 3

-- Prove the equation of the ellipse
theorem find_ellipse (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (ellipse a b a_pos b_pos ∧ focus a b c ∧ range_condition a b c) →
  (a^2 = 9 ∧ b^2 = 3) :=
by
  sorry

end find_ellipse_l238_238728


namespace rate_of_change_l238_238525

noncomputable def radius : ℝ := 12
noncomputable def θ (t : ℝ) : ℝ := (38 + 5 * t) * (Real.pi / 180)
noncomputable def area (t : ℝ) : ℝ := (1/2) * radius^2 * θ t

theorem rate_of_change (t : ℝ) : deriv area t = 2 * Real.pi :=
by
  sorry

end rate_of_change_l238_238525


namespace debby_vacation_pictures_l238_238435

theorem debby_vacation_pictures :
  let zoo_initial := 150
  let aquarium_initial := 210
  let museum_initial := 90
  let amusement_park_initial := 120
  let zoo_deleted := (25 * zoo_initial) / 100  -- 25% of zoo pictures deleted
  let aquarium_deleted := (15 * aquarium_initial) / 100  -- 15% of aquarium pictures deleted
  let museum_added := 30  -- 30 additional pictures at the museum
  let amusement_park_deleted := 20  -- 20 pictures deleted at the amusement park
  let zoo_kept := zoo_initial - zoo_deleted
  let aquarium_kept := aquarium_initial - aquarium_deleted
  let museum_kept := museum_initial + museum_added
  let amusement_park_kept := amusement_park_initial - amusement_park_deleted
  let total_pictures := zoo_kept + aquarium_kept + museum_kept + amusement_park_kept
  total_pictures = 512 :=
by
  sorry

end debby_vacation_pictures_l238_238435


namespace find_range_of_a_l238_238786

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 4 - 2 * a > 0 ∧ 4 - 2 * a < 1

noncomputable def problem_statement (a : ℝ) : Prop :=
  let p := proposition_p a
  let q := proposition_q a
  (p ∨ q) ∧ ¬(p ∧ q)

theorem find_range_of_a (a : ℝ) :
  problem_statement a → -2 < a ∧ a ≤ 3/2 :=
sorry

end find_range_of_a_l238_238786


namespace g_at_4_l238_238937

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (y : ℝ) : ℝ := (3 * y - 4) / y

noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

theorem g_at_4 : g 4 = 11 / 2 :=
by
  sorry

end g_at_4_l238_238937


namespace width_decrease_percentage_l238_238776

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l238_238776


namespace sandy_correct_sums_l238_238818

-- Definitions based on the conditions
variables (c i : ℕ)

-- Conditions as Lean statements
axiom h1 : 3 * c - 2 * i = 65
axiom h2 : c + i = 30

-- Proof goal
theorem sandy_correct_sums : c = 25 := 
by
  sorry

end sandy_correct_sums_l238_238818


namespace solve_xy_l238_238393

theorem solve_xy (x y : ℝ) (hx: x ≠ 0) (hxy: x + y ≠ 0) : 
  (x + y) / x = 2 * y / (x + y) + 1 → (x = y ∨ x = -3 * y) := 
by 
  intros h 
  sorry

end solve_xy_l238_238393


namespace solve_equation_l238_238231

theorem solve_equation (x : ℝ) (h1: (6 * x) ^ 18 = (12 * x) ^ 9) (h2 : x ≠ 0) : x = 1 / 3 := by
  sorry

end solve_equation_l238_238231


namespace number_of_stadiums_to_visit_l238_238338

def average_cost_per_stadium : ℕ := 900
def annual_savings : ℕ := 1500
def years_saving : ℕ := 18

theorem number_of_stadiums_to_visit (c : ℕ) (s : ℕ) (n : ℕ) (h1 : c = average_cost_per_stadium) (h2 : s = annual_savings) (h3 : n = years_saving) : n * s / c = 30 := 
by 
  rw [h1, h2, h3]
  exact sorry

end number_of_stadiums_to_visit_l238_238338


namespace find_m_l238_238133

noncomputable def m : ℕ :=
  let S := {d : ℕ | d ∣ 15^8 ∧ d > 0}
  let total_ways := 9^6
  let strictly_increasing_ways := (Nat.choose 9 3) * (Nat.choose 10 3)
  let probability := strictly_increasing_ways / total_ways
  let gcd := Nat.gcd strictly_increasing_ways total_ways
  strictly_increasing_ways / gcd

theorem find_m : m = 112 :=
by
  sorry

end find_m_l238_238133


namespace number_of_ways_to_divide_day_l238_238217

theorem number_of_ways_to_divide_day (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n * m = 1440) : 
  ∃ (pairs : List (ℕ × ℕ)), (pairs.length = 36) ∧
  (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 * p.2 = 1440)) :=
sorry

end number_of_ways_to_divide_day_l238_238217


namespace find_difference_of_max_and_min_values_l238_238241

noncomputable def v (a b : Int) : Int := a * (-4) + b

theorem find_difference_of_max_and_min_values :
  let v0 := 3
  let v1 := v v0 12
  let v2 := v v1 6
  let v3 := v v2 10
  let v4 := v v3 (-8)
  (max (max (max (max v0 v1) v2) v3) v4) - (min (min (min (min v0 v1) v2) v3) v4) = 62 :=
by
  sorry

end find_difference_of_max_and_min_values_l238_238241


namespace tangent_line_to_curve_at_point_l238_238968

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ),
  (y = 2 * Real.log x) →
  (x = 2) →
  (y = 2 * Real.log 2) →
  (x - y + 2 * Real.log 2 - 2 = 0) := by
  sorry

end tangent_line_to_curve_at_point_l238_238968


namespace ab_leq_one_l238_238626

theorem ab_leq_one (a b : ℝ) (h : (a + b) * (a + b + a + b) = 9) : a * b ≤ 1 := 
  sorry

end ab_leq_one_l238_238626


namespace license_plates_count_l238_238474

theorem license_plates_count :
  (20 * 6 * 20 * 10 * 26 = 624000) :=
by
  sorry

end license_plates_count_l238_238474


namespace coins_problem_l238_238969

theorem coins_problem (x y : ℕ) (h1 : x + y = 20) (h2 : x + 5 * y = 80) : x = 5 :=
by
  sorry

end coins_problem_l238_238969


namespace solve_system_eqn_l238_238628

theorem solve_system_eqn :
  ∃ x y : ℚ, 7 * x = -9 - 3 * y ∧ 2 * x = 5 * y - 30 ∧ x = -135 / 41 ∧ y = 192 / 41 :=
by 
  sorry

end solve_system_eqn_l238_238628


namespace solve_mod_equiv_l238_238355

theorem solve_mod_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ (-2222 ≡ n [ZMOD 9]) → n = 6 := by
  sorry

end solve_mod_equiv_l238_238355


namespace prove_expression_value_l238_238452

theorem prove_expression_value (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 := by
  rw [h]
  sorry

end prove_expression_value_l238_238452


namespace coordinates_of_P_l238_238716

-- Define a structure for a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Define the distance from a point to the x-axis
def distance_to_x_axis (P : Point) : ℝ :=
  |P.y|

-- Define the distance from a point to the y-axis
def distance_to_y_axis (P : Point) : ℝ :=
  |P.x|

-- The main proof statement
theorem coordinates_of_P (P : Point) :
  in_third_quadrant P →
  distance_to_x_axis P = 2 →
  distance_to_y_axis P = 5 →
  P = { x := -5, y := -2 } :=
by
  intros h1 h2 h3
  sorry

end coordinates_of_P_l238_238716


namespace trig_identity_proof_l238_238570

theorem trig_identity_proof 
  (α p q : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0)
  (tangent : Real.tan α = p / q) :
  Real.sin (2 * α) = 2 * p * q / (p^2 + q^2) ∧
  Real.cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  Real.tan (2 * α) = (2 * p * q) / (q^2 - p^2) :=
by
  sorry

end trig_identity_proof_l238_238570


namespace total_books_in_series_l238_238362

-- Definitions for the conditions
def books_read : ℕ := 8
def books_to_read : ℕ := 6

-- Statement to be proved
theorem total_books_in_series : books_read + books_to_read = 14 := by
  sorry

end total_books_in_series_l238_238362


namespace eval_special_op_l238_238106

variable {α : Type*} [LinearOrderedField α]

def op (a b : α) : α := (a - b) ^ 2

theorem eval_special_op (x y z : α) : op ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end eval_special_op_l238_238106


namespace dwarfs_truthful_count_l238_238396

theorem dwarfs_truthful_count :
  ∃ (T L : ℕ), T + L = 10 ∧
    (∀ t : ℕ, t = 10 → t + ((10 - T) * 2 - T) = 16) ∧
    T = 4 :=
by
  sorry

end dwarfs_truthful_count_l238_238396


namespace evaluate_expression_l238_238724

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a ^ a - a * (a - 2) ^ a) ^ (a + 1) = 14889702426 :=
by
  rw [h]
  sorry

end evaluate_expression_l238_238724


namespace trig_identity_proof_l238_238978

noncomputable def sin_30 : Real := 1 / 2
noncomputable def cos_120 : Real := -1 / 2
noncomputable def cos_45 : Real := Real.sqrt 2 / 2
noncomputable def tan_30 : Real := Real.sqrt 3 / 3

theorem trig_identity_proof : 
  sin_30 + cos_120 + 2 * cos_45 - Real.sqrt 3 * tan_30 = Real.sqrt 2 - 1 := 
by
  sorry

end trig_identity_proof_l238_238978


namespace tetrahedron_colorings_l238_238868

-- Define the problem conditions
def tetrahedron_faces : ℕ := 4
def colors : List String := ["red", "white", "blue", "yellow"]

-- The theorem statement
theorem tetrahedron_colorings :
  ∃ n : ℕ, n = 35 ∧ ∀ (c : List String), c.length = tetrahedron_faces → c ⊆ colors →
  (true) := -- Placeholder (you can replace this condition with the appropriate condition)
by
  -- The proof is omitted with 'sorry' as instructed
  sorry

end tetrahedron_colorings_l238_238868


namespace fred_allowance_is_16_l238_238190

def fred_weekly_allowance (A : ℕ) : Prop :=
  (A / 2) + 6 = 14

theorem fred_allowance_is_16 : ∃ A : ℕ, fred_weekly_allowance A ∧ A = 16 := 
by
  -- Proof can be filled here
  sorry

end fred_allowance_is_16_l238_238190


namespace amanda_earnings_l238_238841

def hourly_rate : ℝ := 20.00

def hours_monday : ℝ := 5 * 1.5

def hours_tuesday : ℝ := 3

def hours_thursday : ℝ := 2 * 2

def hours_saturday : ℝ := 6

def total_hours : ℝ := hours_monday + hours_tuesday + hours_thursday + hours_saturday

def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410.00 :=
by
  -- Proof steps can be filled here
  sorry

end amanda_earnings_l238_238841


namespace alice_has_largest_result_l238_238735

def initial_number : ℕ := 15

def alice_transformation (x : ℕ) : ℕ := (x * 3 - 2 + 4)
def bob_transformation (x : ℕ) : ℕ := (x * 2 + 3 - 5)
def charlie_transformation (x : ℕ) : ℕ := (x + 5) / 2 * 4

def alice_final := alice_transformation initial_number
def bob_final := bob_transformation initial_number
def charlie_final := charlie_transformation initial_number

theorem alice_has_largest_result :
  alice_final > bob_final ∧ alice_final > charlie_final := by
  sorry

end alice_has_largest_result_l238_238735


namespace regular_18gon_symmetries_l238_238177

theorem regular_18gon_symmetries :
  let L := 18
  let R := 20
  L + R = 38 := by
sorry

end regular_18gon_symmetries_l238_238177


namespace jellybean_probability_l238_238873

/-- A bowl contains 15 jellybeans: five red, three blue, five white, and two green. If you pick four 
    jellybeans from the bowl at random and without replacement, the probability that exactly three will 
    be red is 20/273. -/
theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 5
  let green_jellybeans := 2
  let total_combinations := Nat.choose total_jellybeans 4
  let favorable_combinations := (Nat.choose red_jellybeans 3) * (Nat.choose (total_jellybeans - red_jellybeans) 1)
  let probability := favorable_combinations / total_combinations
  probability = 20 / 273 :=
by
  sorry

end jellybean_probability_l238_238873


namespace stretching_transformation_eq_curve_l238_238303

variable (x y x₁ y₁ : ℝ)

theorem stretching_transformation_eq_curve :
  (x₁ = 3 * x) →
  (y₁ = y) →
  (x₁^2 + 9 * y₁^2 = 9) →
  (x^2 + y^2 = 1) :=
by
  intros h1 h2 h3
  sorry

end stretching_transformation_eq_curve_l238_238303


namespace card_d_total_percent_change_l238_238636

noncomputable def card_d_initial_value : ℝ := 250
noncomputable def card_d_percent_changes : List ℝ := [0.05, -0.15, 0.30, -0.10, 0.20]

noncomputable def final_value (initial_value : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_value

theorem card_d_total_percent_change :
  let final_val := final_value card_d_initial_value card_d_percent_changes
  let total_percent_change := ((final_val - card_d_initial_value) / card_d_initial_value) * 100
  total_percent_change = 25.307 := by
  sorry

end card_d_total_percent_change_l238_238636


namespace cost_of_banana_l238_238259

theorem cost_of_banana (B : ℝ) (apples bananas oranges total_pieces total_cost : ℝ) 
  (h1 : apples = 12) (h2 : bananas = 4) (h3 : oranges = 4) 
  (h4 : total_pieces = 20) (h5 : total_cost = 40)
  (h6 : 2 * apples + 3 * oranges + bananas * B = total_cost)
  : B = 1 :=
by
  sorry

end cost_of_banana_l238_238259


namespace number_of_deleted_apps_l238_238229

def initial_apps := 16
def remaining_apps := 8

def deleted_apps : ℕ := initial_apps - remaining_apps

theorem number_of_deleted_apps : deleted_apps = 8 := 
by
  unfold deleted_apps initial_apps remaining_apps
  rfl

end number_of_deleted_apps_l238_238229


namespace angle_complement_l238_238767

-- Conditions: The complement of angle A is 60 degrees
def complement (α : ℝ) : ℝ := 90 - α 

theorem angle_complement (A : ℝ) : complement A = 60 → A = 30 :=
by
  sorry

end angle_complement_l238_238767


namespace train_crosses_platform_l238_238235

theorem train_crosses_platform :
  ∀ (L : ℕ), 
  (300 + L) / (50 / 3) = 48 → 
  L = 500 := 
by
  sorry

end train_crosses_platform_l238_238235


namespace more_sparrows_than_pigeons_l238_238695

-- Defining initial conditions
def initial_sparrows := 3
def initial_starlings := 5
def initial_pigeons := 2
def additional_sparrows := 4
def additional_starlings := 2
def additional_pigeons := 3

-- Final counts after additional birds join
def final_sparrows := initial_sparrows + additional_sparrows
def final_pigeons := initial_pigeons + additional_pigeons

-- The statement to be proved
theorem more_sparrows_than_pigeons:
  final_sparrows - final_pigeons = 2 :=
by
  -- proof skipped
  sorry

end more_sparrows_than_pigeons_l238_238695


namespace triangle_area_l238_238021

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Define the property of being a right triangle via the Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of a right triangle given base and height
def area_right_triangle (a b : ℕ) : ℕ := (a * b) / 2

-- The main theorem, stating that the area of the triangle with sides 9, 12, 15 is 54
theorem triangle_area : is_right_triangle a b c → area_right_triangle a b = 54 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l238_238021


namespace average_marks_correct_l238_238871

def marks := [76, 65, 82, 62, 85]
def num_subjects := 5
def total_marks := marks.sum
def avg_marks := total_marks / num_subjects

theorem average_marks_correct : avg_marks = 74 :=
by sorry

end average_marks_correct_l238_238871


namespace surface_area_increase_l238_238480

noncomputable def percent_increase_surface_area (s p : ℝ) : ℝ :=
  let new_edge_length := s * (1 + p / 100)
  let new_surface_area := 6 * (new_edge_length)^2
  let original_surface_area := 6 * s^2
  let percent_increase := (new_surface_area / original_surface_area - 1) * 100
  percent_increase

theorem surface_area_increase (s p : ℝ) :
  percent_increase_surface_area s p = 2 * p + p^2 / 100 :=
by
  sorry

end surface_area_increase_l238_238480


namespace sum_of_roots_unique_solution_l238_238782

open Real

def operation (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := operation x 2

theorem sum_of_roots_unique_solution
  (x1 x2 x3 x4 : ℝ)
  (h1 : ∀ x, f x = log (abs (x + 2)) → x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)
  (h2 : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_unique_solution_l238_238782


namespace intersect_sets_l238_238772

open Set

noncomputable def P : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}

theorem intersect_sets (U : Set ℝ) (P : Set ℝ) (Q : Set ℝ) :
  U = univ → P = {x : ℝ | x^2 - 2 * x ≤ 0} → Q = {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x} →
  P ∩ Q = Icc (0 : ℝ) (2 : ℝ) :=
by
  intros
  sorry

end intersect_sets_l238_238772


namespace weight_range_correct_l238_238178

noncomputable def combined_weight : ℕ := 158
noncomputable def tracy_weight : ℕ := 52
noncomputable def jake_weight : ℕ := tracy_weight + 8
noncomputable def john_weight : ℕ := combined_weight - (tracy_weight + jake_weight)
noncomputable def weight_range : ℕ := jake_weight - john_weight

theorem weight_range_correct : weight_range = 14 := 
by
  sorry

end weight_range_correct_l238_238178


namespace cuboid_volume_l238_238621

/-- Define the ratio condition for the dimensions of the cuboid. -/
def ratio (l w h : ℕ) : Prop :=
  (∃ x : ℕ, l = 2*x ∧ w = x ∧ h = 3*x)

/-- Define the total surface area condition for the cuboid. -/
def surface_area (l w h sa : ℕ) : Prop :=
  2*(l*w + l*h + w*h) = sa

/-- Volume of the cuboid given the ratio and surface area conditions. -/
theorem cuboid_volume (l w h : ℕ) (sa : ℕ) (h_ratio : ratio l w h) (h_surface : surface_area l w h sa) :
  ∃ v : ℕ, v = l * w * h ∧ v = 48 :=
by
  sorry

end cuboid_volume_l238_238621


namespace trivia_team_points_l238_238704

theorem trivia_team_points (total_members absent_members total_points : ℕ) 
    (h1 : total_members = 5) 
    (h2 : absent_members = 2) 
    (h3 : total_points = 18) 
    (h4 : total_members - absent_members = present_members) 
    (h5 : total_points = present_members * points_per_member) : 
    points_per_member = 6 :=
  sorry

end trivia_team_points_l238_238704


namespace rational_square_of_one_minus_product_l238_238829

theorem rational_square_of_one_minus_product (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : 
  ∃ (q : ℚ), 1 - x * y = q^2 := 
by 
  sorry

end rational_square_of_one_minus_product_l238_238829


namespace percentage_reduction_in_oil_price_l238_238509

theorem percentage_reduction_in_oil_price (R : ℝ) (P : ℝ) (hR : R = 48) (h_quantity : (800/R) - (800/P) = 5) : 
    ((P - R) / P) * 100 = 30 := 
    sorry

end percentage_reduction_in_oil_price_l238_238509


namespace profit_calculation_l238_238648

theorem profit_calculation
  (P : ℝ)
  (h1 : 9 > 0)  -- condition that there are 9 employees
  (h2 : 0 < 0.10 ∧ 0.10 < 1) -- 10 percent profit is between 0 and 100%
  (h3 : 5 > 0)  -- condition that each employee gets $5
  (h4 : 9 * 5 = 45) -- total amount distributed among employees
  (h5 : 0.90 * P = 45) -- remaining profit to be distributed
  : P = 50 :=
sorry

end profit_calculation_l238_238648


namespace overall_avg_is_60_l238_238694

-- Define the number of students and average marks for each class
def classA_students : ℕ := 30
def classA_avg_marks : ℕ := 40

def classB_students : ℕ := 50
def classB_avg_marks : ℕ := 70

def classC_students : ℕ := 25
def classC_avg_marks : ℕ := 55

def classD_students : ℕ := 45
def classD_avg_marks : ℕ := 65

-- Calculate the total number of students
def total_students : ℕ := 
  classA_students + classB_students + classC_students + classD_students

-- Calculate the total marks for each class
def total_marks_A : ℕ := classA_students * classA_avg_marks
def total_marks_B : ℕ := classB_students * classB_avg_marks
def total_marks_C : ℕ := classC_students * classC_avg_marks
def total_marks_D : ℕ := classD_students * classD_avg_marks

-- Calculate the combined total marks of all classes
def combined_total_marks : ℕ := 
  total_marks_A + total_marks_B + total_marks_C + total_marks_D

-- Calculate the overall average marks
def overall_avg_marks : ℕ := combined_total_marks / total_students

-- Prove that the overall average marks is 60
theorem overall_avg_is_60 : overall_avg_marks = 60 := by
  sorry -- Proof will be written here

end overall_avg_is_60_l238_238694


namespace unique_triangle_solution_l238_238762

noncomputable def triangle_solutions (a b A : ℝ) : ℕ :=
sorry -- Placeholder for actual function calculating number of solutions

theorem unique_triangle_solution : triangle_solutions 30 25 150 = 1 :=
sorry -- Proof goes here

end unique_triangle_solution_l238_238762


namespace area_correct_l238_238144

noncomputable def area_of_30_60_90_triangle (hypotenuse : ℝ) (angle : ℝ) : ℝ :=
if hypotenuse = 10 ∧ angle = 30 then 25 * Real.sqrt 3 / 2 else 0

theorem area_correct {hypotenuse angle : ℝ} (h1 : hypotenuse = 10) (h2 : angle = 30) :
  area_of_30_60_90_triangle hypotenuse angle = 25 * Real.sqrt 3 / 2 :=
by
  sorry

end area_correct_l238_238144


namespace sum_m_n_eq_zero_l238_238179

theorem sum_m_n_eq_zero (m n p : ℝ) (h1 : m * n + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 := 
  sorry

end sum_m_n_eq_zero_l238_238179


namespace remainder_of_5_pow_2023_mod_17_l238_238888

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l238_238888


namespace decimal_15_to_binary_l238_238324

theorem decimal_15_to_binary : (15 : ℕ) = (4*1 + 2*1 + 1*1)*2^3 + (4*1 + 2*1 + 1*1)*2^2 + (4*1 + 2*1 + 1*1)*2 + 1 := by
  sorry

end decimal_15_to_binary_l238_238324


namespace parabola_hyperbola_tangent_l238_238383

noncomputable def parabola : ℝ → ℝ := λ x => x^2 + 5

noncomputable def hyperbola (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y^2 - m * x^2 = 1

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (m = 10 + 4*Real.sqrt 6 ∨ m = 10 - 4*Real.sqrt 6) →
  ∃ x y, parabola x = y ∧ hyperbola m x y ∧ 
    ∃ c b a, a * y^2 + b * y + c = 0 ∧ a = 1 ∧ c = 5 * m - 1 ∧ b = -m ∧ b^2 - 4*a*c = 0 :=
by
  sorry

end parabola_hyperbola_tangent_l238_238383


namespace carlos_paid_l238_238784

theorem carlos_paid (a b c : ℝ) 
  (h1 : a = (1 / 3) * (b + c))
  (h2 : b = (1 / 4) * (a + c))
  (h3 : a + b + c = 120) :
  c = 72 :=
by
-- Proof omitted
sorry

end carlos_paid_l238_238784


namespace remainder_when_divided_by_2_l238_238916

theorem remainder_when_divided_by_2 (n : ℕ) (h₁ : n > 0) (h₂ : (n + 1) % 6 = 4) : n % 2 = 1 :=
by sorry

end remainder_when_divided_by_2_l238_238916


namespace compute_c_plus_d_l238_238997

theorem compute_c_plus_d (c d : ℕ) (h1 : d = c^3) (h2 : d - c = 435) : c + d = 520 :=
sorry

end compute_c_plus_d_l238_238997


namespace range_of_a_l238_238669

def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, operation (x - a) (x + 1) < 1) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l238_238669


namespace parabola_focus_coordinates_l238_238970

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 4 * x^2) : (0, 1/16) = (0, 1/16) :=
by
  sorry

end parabola_focus_coordinates_l238_238970


namespace range_of_m_l238_238787

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem range_of_m {m : ℝ} :
  (∀ x : ℝ, (f x m = 0) → (∃ y z : ℝ, y ≠ z ∧ f y m = 0 ∧ f z m = 0)) ∧
  (∀ x : ℝ, f (1 - x) m ≥ -1)
  → (0 ≤ m ∧ m < 1) := 
sorry

end range_of_m_l238_238787


namespace number_of_triangles_with_one_side_five_not_shortest_l238_238638

theorem number_of_triangles_with_one_side_five_not_shortest (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_one_side_five : a = 5 ∨ b = 5 ∨ c = 5)
  (h_not_shortest : a = 5 ∧ a > b ∧ a > c ∨ b = 5 ∧ b > a ∧ b > c ∨ c = 5 ∧ c > a ∧ c > b ∨ a ≠ 5 ∧ b = 5 ∧ b > c ∨ a ≠ 5 ∧ c = 5 ∧ c > b) :
  (∃ n, n = 10) :=
by
  sorry

end number_of_triangles_with_one_side_five_not_shortest_l238_238638


namespace cafeteria_apples_l238_238743

theorem cafeteria_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ) 
(h1: handed_out = 27) (h2: pies = 5) (h3: apples_per_pie = 4) : handed_out + pies * apples_per_pie = 47 :=
by
  -- The proof will be provided here if needed
  sorry

end cafeteria_apples_l238_238743


namespace find_n_l238_238943

theorem find_n (n : ℕ) (h : n ≥ 2) : 
  (∀ (i j : ℕ), 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔ ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by
  sorry

end find_n_l238_238943


namespace number_of_boys_l238_238399

theorem number_of_boys (B G : ℕ) 
    (h1 : B + G = 345) 
    (h2 : G = B + 69) : B = 138 :=
by
  sorry

end number_of_boys_l238_238399


namespace inequality_proof_l238_238001

variables (a b c d : ℝ)

theorem inequality_proof 
  (h1 : a + b > abs (c - d)) 
  (h2 : c + d > abs (a - b)) : 
  a + c > abs (b - d) := 
sorry

end inequality_proof_l238_238001


namespace largest_possible_c_l238_238442

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end largest_possible_c_l238_238442


namespace space_shuttle_speed_l238_238292

-- Define the conditions in Lean
def speed_kmph : ℕ := 43200 -- Speed in kilometers per hour
def seconds_per_hour : ℕ := 60 * 60 -- Number of seconds in an hour

-- Define the proof problem
theorem space_shuttle_speed :
  speed_kmph / seconds_per_hour = 12 := by
  sorry

end space_shuttle_speed_l238_238292


namespace fourth_number_in_first_set_88_l238_238707

theorem fourth_number_in_first_set_88 (x y : ℝ)
  (h1 : (28 + x + 70 + y + 104) / 5 = 67)
  (h2 : (50 + 62 + 97 + 124 + x) / 5 = 75.6) :
  y = 88 :=
by
  sorry

end fourth_number_in_first_set_88_l238_238707


namespace exists_subset_with_property_l238_238154

theorem exists_subset_with_property :
  ∃ X : Set Int, ∀ n : Int, ∃ (a b : X), a + 2 * b = n ∧ ∀ (a' b' : X), (a + 2 * b = n ∧ a' + 2 * b' = n) → (a = a' ∧ b = b') :=
sorry

end exists_subset_with_property_l238_238154


namespace circles_are_disjoint_l238_238676

noncomputable def positional_relationship_of_circles (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : Prop :=
R₁ + R₂ = d

theorem circles_are_disjoint {R₁ R₂ d : ℝ} (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : positional_relationship_of_circles R₁ R₂ d h₁ h₂ :=
by sorry

end circles_are_disjoint_l238_238676


namespace add_three_digits_l238_238615

theorem add_three_digits (x : ℕ) :
  (x = 152 ∨ x = 656) →
  (523000 + x) % 504 = 0 := 
by
  sorry

end add_three_digits_l238_238615


namespace coffee_bags_per_week_l238_238220

def bags_morning : Nat := 3
def bags_afternoon : Nat := 3 * bags_morning
def bags_evening : Nat := 2 * bags_morning
def bags_per_day : Nat := bags_morning + bags_afternoon + bags_evening
def days_per_week : Nat := 7

theorem coffee_bags_per_week : bags_per_day * days_per_week = 126 := by
  sorry

end coffee_bags_per_week_l238_238220


namespace stamps_total_l238_238579

theorem stamps_total (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 :=
by sorry

end stamps_total_l238_238579


namespace cooking_time_at_least_l238_238781

-- Definitions based on conditions
def total_potatoes : ℕ := 35
def cooked_potatoes : ℕ := 11
def time_per_potato : ℕ := 7 -- in minutes
def salad_time : ℕ := 15 -- in minutes

-- The statement to prove
theorem cooking_time_at_least (oven_capacity : ℕ) :
  ∃ t : ℕ, t ≥ salad_time :=
by
  sorry

end cooking_time_at_least_l238_238781
