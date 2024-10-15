import Mathlib

namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l2346_234635

theorem algebraic_expression_evaluation (x y : ℝ) (h : 2 * x - y + 1 = 3) : 4 * x - 2 * y + 5 = 9 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l2346_234635


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_l2346_234663

noncomputable def geometric_sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence (a₁ q : ℝ) (n : ℕ) 
  (h1 : a₁ + a₁ * q^3 = 10) 
  (h2 : a₁ * q + a₁ * q^4 = 20) : 
  geometric_sequence_sum a₁ q n = (10 / 9) * (2^n - 1) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_geometric_sequence_l2346_234663


namespace NUMINAMATH_GPT_value_of_a_l2346_234607

theorem value_of_a (M : Set ℝ) (N : Set ℝ) (a : ℝ) 
  (hM : M = {-1, 0, 1, 2}) (hN : N = {x | x^2 - a * x < 0}) 
  (hIntersect : M ∩ N = {1, 2}) : 
  a = 3 := 
sorry

end NUMINAMATH_GPT_value_of_a_l2346_234607


namespace NUMINAMATH_GPT_remaining_tickets_l2346_234669

def initial_tickets : ℝ := 49.0
def lost_tickets : ℝ := 6.0
def spent_tickets : ℝ := 25.0

theorem remaining_tickets : initial_tickets - lost_tickets - spent_tickets = 18.0 := by
  sorry

end NUMINAMATH_GPT_remaining_tickets_l2346_234669


namespace NUMINAMATH_GPT_find_local_min_l2346_234694

def z (x y : ℝ) : ℝ := x^2 + 2 * y^2 - 2 * x * y - x - 2 * y

theorem find_local_min: ∃ (x y : ℝ), x = 2 ∧ y = 3/2 ∧ ∀ ⦃h : ℝ⦄, h ≠ 0 → z (2 + h) (3/2 + h) > z 2 (3/2) :=
by
  sorry

end NUMINAMATH_GPT_find_local_min_l2346_234694


namespace NUMINAMATH_GPT_not_odd_function_l2346_234677

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x ^ 2 + 1)

theorem not_odd_function : ¬ is_odd_function f := by
  sorry

end NUMINAMATH_GPT_not_odd_function_l2346_234677


namespace NUMINAMATH_GPT_suki_bags_l2346_234636

theorem suki_bags (bag_weight_suki : ℕ) (bag_weight_jimmy : ℕ) (containers : ℕ) 
  (container_weight : ℕ) (num_bags_jimmy : ℝ) (num_containers : ℕ)
  (h1 : bag_weight_suki = 22) 
  (h2 : bag_weight_jimmy = 18) 
  (h3 : container_weight = 8) 
  (h4 : num_bags_jimmy = 4.5)
  (h5 : num_containers = 28) : 
  6 = ⌊(num_containers * container_weight - num_bags_jimmy * bag_weight_jimmy) / bag_weight_suki⌋ :=
by
  sorry

end NUMINAMATH_GPT_suki_bags_l2346_234636


namespace NUMINAMATH_GPT_range_of_m_if_real_roots_specific_m_given_conditions_l2346_234668

open Real

-- Define the quadratic equation and its conditions
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x ^ 2 - x + 2 * m - 4 = 0
def has_real_roots (m : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2

-- Proof that m ≤ 17/8 if the quadratic equation has real roots
theorem range_of_m_if_real_roots (m : ℝ) : has_real_roots m → m ≤ 17 / 8 := 
sorry

-- Define a condition on the roots
def roots_condition (x1 x2 m : ℝ) : Prop := (x1 - 3) * (x2 - 3) = m ^ 2 - 1

-- Proof of specific m when roots condition is given
theorem specific_m_given_conditions (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ roots_condition x1 x2 m) → m = -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_if_real_roots_specific_m_given_conditions_l2346_234668


namespace NUMINAMATH_GPT_roots_transformation_l2346_234648

-- Given polynomial
def poly1 (x : ℝ) : ℝ := x^3 - 3*x^2 + 8

-- Polynomial with roots 3*r1, 3*r2, 3*r3
def poly2 (x : ℝ) : ℝ := x^3 - 9*x^2 + 216

-- Theorem stating the equivalence
theorem roots_transformation (r1 r2 r3 : ℝ) 
  (h : ∀ x, poly1 x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x, poly2 x = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
sorry

end NUMINAMATH_GPT_roots_transformation_l2346_234648


namespace NUMINAMATH_GPT_germination_probability_l2346_234686

open Nat

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_of_success (p : ℚ) (k : ℕ) (n : ℕ) : ℚ :=
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem germination_probability :
  probability_of_success 0.9 5 7 = 0.124 := by
  sorry

end NUMINAMATH_GPT_germination_probability_l2346_234686


namespace NUMINAMATH_GPT_inequality_solution_l2346_234690

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ (x > 8)) ↔
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2346_234690


namespace NUMINAMATH_GPT_sum_of_a_b_either_1_or_neg1_l2346_234689

theorem sum_of_a_b_either_1_or_neg1 (a b : ℝ) (h1 : a + a = 0) (h2 : b * b = 1) : a + b = 1 ∨ a + b = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_a_b_either_1_or_neg1_l2346_234689


namespace NUMINAMATH_GPT_bill_toys_l2346_234640

variable (B H : ℕ)

theorem bill_toys (h1 : H = B / 2 + 9) (h2 : B + H = 99) : B = 60 := by
  sorry

end NUMINAMATH_GPT_bill_toys_l2346_234640


namespace NUMINAMATH_GPT_correct_answer_l2346_234650

-- Statement of the problem
theorem correct_answer :
  ∃ (answer : String),
    (answer = "long before" ∨ answer = "before long" ∨ answer = "soon after" ∨ answer = "shortly after") ∧
    answer = "long before" :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l2346_234650


namespace NUMINAMATH_GPT_maximum_diagonal_intersections_l2346_234653

theorem maximum_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
  ∃ k, k = (n * (n - 1) * (n - 2) * (n - 3)) / 24 :=
by sorry

end NUMINAMATH_GPT_maximum_diagonal_intersections_l2346_234653


namespace NUMINAMATH_GPT_cost_50_jasmines_discounted_l2346_234643

variable (cost_per_8_jasmines : ℝ) (num_jasmines : ℕ) (discount : ℝ)
variable (proportional : Prop) (c_50_jasmines : ℝ)

-- Given the cost of a bouquet with 8 jasmines
def cost_of_8_jasmines : ℝ := 24

-- Given the price is directly proportional to the number of jasmines
def price_proportional := ∀ (n : ℕ), num_jasmines = 8 → proportional

-- Given the bouquet with 50 jasmines
def num_jasmines_50 : ℕ := 50

-- Applying a 10% discount
def ten_percent_discount : ℝ := 0.9

-- Prove the cost of the bouquet with 50 jasmines after a 10% discount
theorem cost_50_jasmines_discounted :
  proportional ∧ (c_50_jasmines = (cost_of_8_jasmines / 8) * num_jasmines_50) →
  (c_50_jasmines * ten_percent_discount) = 135 :=
by
  sorry

end NUMINAMATH_GPT_cost_50_jasmines_discounted_l2346_234643


namespace NUMINAMATH_GPT_find_a_and_max_value_l2346_234612

noncomputable def f (x a : ℝ) := 2 * x^3 - 6 * x^2 + a

theorem find_a_and_max_value :
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≥ 0) ∧ (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≤ 3)) :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_max_value_l2346_234612


namespace NUMINAMATH_GPT_distance_between_intersections_l2346_234692

open Classical
open Real

noncomputable def curve1 (x y : ℝ) : Prop := y^2 = x
noncomputable def curve2 (x y : ℝ) : Prop := x + 2 * y = 10

theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ),
    (curve1 p1.1 p1.2) ∧ (curve2 p1.1 p1.2) ∧
    (curve1 p2.1 p2.2) ∧ (curve2 p2.1 p2.2) ∧
    (dist p1 p2 = 2 * sqrt 55) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_intersections_l2346_234692


namespace NUMINAMATH_GPT_total_price_l2346_234695

theorem total_price (r w : ℕ) (hr : r = 4275) (hw : w = r - 1490) : r + w = 7060 :=
by
  sorry

end NUMINAMATH_GPT_total_price_l2346_234695


namespace NUMINAMATH_GPT_cube_volume_is_8_l2346_234683

theorem cube_volume_is_8 (a : ℕ) 
  (h_cond : (a+2) * (a-2) * a = a^3 - 8) : 
  a^3 = 8 := 
by
  sorry

end NUMINAMATH_GPT_cube_volume_is_8_l2346_234683


namespace NUMINAMATH_GPT_part1_part2_l2346_234622

-- Definition of sets A, B, and Proposition p for Part 1
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a = 0}
def p (a : ℝ) : Prop := ∀ x ∈ B a, x ∈ A

-- Part 1: Prove the range of a
theorem part1 (a : ℝ) : (p a) → 0 < a ∧ a ≤ 1 :=
  by sorry

-- Definition of sets A and C for Part 2
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 3 > 0}
def necessary_condition (m : ℝ) : Prop := ∀ x ∈ A, x ∈ C m

-- Part 2: Prove the range of m
theorem part2 (m : ℝ) : necessary_condition m → m ≤ 7 / 2 :=
  by sorry

end NUMINAMATH_GPT_part1_part2_l2346_234622


namespace NUMINAMATH_GPT_absents_probability_is_correct_l2346_234649

-- Conditions
def probability_absent := 1 / 10
def probability_present := 9 / 10

-- Calculation of combined probability
def combined_probability : ℚ :=
  3 * (probability_absent * probability_absent * probability_present)

-- Conversion to percentage
def percentage_probability : ℚ :=
  combined_probability * 100

-- Theorem statement
theorem absents_probability_is_correct :
  percentage_probability = 2.7 := 
sorry

end NUMINAMATH_GPT_absents_probability_is_correct_l2346_234649


namespace NUMINAMATH_GPT_bus_stop_l2346_234674

theorem bus_stop (M H : ℕ) 
  (h1 : H = 2 * (M - 15))
  (h2 : M - 15 = 5 * (H - 45)) :
  M = 40 ∧ H = 50 := 
sorry

end NUMINAMATH_GPT_bus_stop_l2346_234674


namespace NUMINAMATH_GPT_minimum_value_fraction_l2346_234628

theorem minimum_value_fraction (m n : ℝ) (h_line : 2 * m * 2 + n * 2 - 4 = 0) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (m + n / 2 = 1) -> ∃ (m n : ℝ), (m > 0 ∧ n > 0) ∧ (3 + 2 * Real.sqrt 2 ≤ (1 / m + 4 / n)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_fraction_l2346_234628


namespace NUMINAMATH_GPT_fraction_zero_iff_l2346_234621

theorem fraction_zero_iff (x : ℝ) (h₁ : (x - 1) / (2 * x - 4) = 0) (h₂ : 2 * x - 4 ≠ 0) : x = 1 := sorry

end NUMINAMATH_GPT_fraction_zero_iff_l2346_234621


namespace NUMINAMATH_GPT_probability_of_all_girls_chosen_is_1_over_11_l2346_234623

-- Defining parameters and conditions
def total_members : ℕ := 12
def boys : ℕ := 6
def girls : ℕ := 6
def chosen_members : ℕ := 3

-- Number of combinations to choose 3 members from 12
def total_combinations : ℕ := Nat.choose total_members chosen_members

-- Number of combinations to choose 3 girls from 6
def girl_combinations : ℕ := Nat.choose girls chosen_members

-- Probability is defined as the ratio of these combinations
def probability_all_girls_chosen : ℚ := girl_combinations / total_combinations

-- Proof Statement
theorem probability_of_all_girls_chosen_is_1_over_11 : probability_all_girls_chosen = 1 / 11 := by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_probability_of_all_girls_chosen_is_1_over_11_l2346_234623


namespace NUMINAMATH_GPT_triangle_property_proof_l2346_234675

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end NUMINAMATH_GPT_triangle_property_proof_l2346_234675


namespace NUMINAMATH_GPT_arithmetic_geometric_inequality_l2346_234652

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + n * d

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ := b1 * r^n

theorem arithmetic_geometric_inequality
  (a1 b1 : ℝ) (d r : ℝ) (n : ℕ)
  (h_pos : 0 < a1) 
  (ha1_eq_b1 : a1 = b1) 
  (h_eq_2np1 : arithmetic_sequence a1 d (2*n+1) = geometric_sequence b1 r (2*n+1)) :
  arithmetic_sequence a1 d (n+1) ≥ geometric_sequence b1 r (n+1) :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_inequality_l2346_234652


namespace NUMINAMATH_GPT_colten_chickens_l2346_234609

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end NUMINAMATH_GPT_colten_chickens_l2346_234609


namespace NUMINAMATH_GPT_triangle_area_base_10_height_10_l2346_234631

theorem triangle_area_base_10_height_10 :
  let base := 10
  let height := 10
  (base * height) / 2 = 50 := by
  sorry

end NUMINAMATH_GPT_triangle_area_base_10_height_10_l2346_234631


namespace NUMINAMATH_GPT_two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l2346_234666

theorem two_pow_1000_mod_3 : 2^1000 % 3 = 1 := sorry
theorem two_pow_1000_mod_5 : 2^1000 % 5 = 1 := sorry
theorem two_pow_1000_mod_11 : 2^1000 % 11 = 1 := sorry
theorem two_pow_1000_mod_13 : 2^1000 % 13 = 3 := sorry

end NUMINAMATH_GPT_two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l2346_234666


namespace NUMINAMATH_GPT_subtraction_to_nearest_thousandth_l2346_234602

theorem subtraction_to_nearest_thousandth : 
  (456.789 : ℝ) - (234.567 : ℝ) = 222.222 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_to_nearest_thousandth_l2346_234602


namespace NUMINAMATH_GPT_proof_mn_squared_l2346_234696

theorem proof_mn_squared (m n : ℤ) (h1 : |m| = 3) (h2 : |n| = 2) (h3 : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end NUMINAMATH_GPT_proof_mn_squared_l2346_234696


namespace NUMINAMATH_GPT_farmers_acres_to_clean_l2346_234673

-- Definitions of the main quantities
variables (A D : ℕ)

-- Conditions
axiom condition1 : A = 80 * D
axiom condition2 : 90 * (D - 1) + 30 = A

-- Theorem asserting the total number of acres to be cleaned
theorem farmers_acres_to_clean : A = 480 :=
by
  -- The proof would go here, but is omitted as per instructions
  sorry

end NUMINAMATH_GPT_farmers_acres_to_clean_l2346_234673


namespace NUMINAMATH_GPT_fair_share_of_bill_l2346_234680

noncomputable def total_bill : Real := 139.00
noncomputable def tip_percent : Real := 0.10
noncomputable def num_people : Real := 6
noncomputable def expected_amount_per_person : Real := 25.48

theorem fair_share_of_bill :
  (total_bill + (tip_percent * total_bill)) / num_people = expected_amount_per_person :=
by
  sorry

end NUMINAMATH_GPT_fair_share_of_bill_l2346_234680


namespace NUMINAMATH_GPT_find_n_l2346_234632

theorem find_n (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) = 14) : n = 2 :=
sorry

end NUMINAMATH_GPT_find_n_l2346_234632


namespace NUMINAMATH_GPT_height_after_16_minutes_l2346_234665

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  8 * Real.sin ((Real.pi / 6) * t - Real.pi / 2) + 10

theorem height_after_16_minutes : ferris_wheel_height 16 = 6 := by
  sorry

end NUMINAMATH_GPT_height_after_16_minutes_l2346_234665


namespace NUMINAMATH_GPT_change_from_15_dollars_l2346_234629

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end NUMINAMATH_GPT_change_from_15_dollars_l2346_234629


namespace NUMINAMATH_GPT_circle_rational_points_l2346_234685

theorem circle_rational_points :
  ( ∃ B : ℚ × ℚ, ∀ k : ℚ, B ∈ {p | p.1 ^ 2 + 2 * p.1 + p.2 ^ 2 = 1992} ) ∧ 
  ( (42 : ℤ)^2 + 2 * 42 + 12^2 = 1992 ) :=
by
  sorry

end NUMINAMATH_GPT_circle_rational_points_l2346_234685


namespace NUMINAMATH_GPT_tan_of_angle_l2346_234626

open Real

-- Given conditions in the problem
variables {α : ℝ}

-- Define the given conditions
def sinα_condition (α : ℝ) : Prop := sin α = 3 / 5
def α_in_quadrant_2 (α : ℝ) : Prop := π / 2 < α ∧ α < π

-- Define the Lean statement
theorem tan_of_angle {α : ℝ} (h1 : sinα_condition α) (h2 : α_in_quadrant_2 α) :
  tan α = -3 / 4 :=
sorry

end NUMINAMATH_GPT_tan_of_angle_l2346_234626


namespace NUMINAMATH_GPT_min_rectilinear_distance_to_parabola_l2346_234625

theorem min_rectilinear_distance_to_parabola :
  ∃ t : ℝ, ∀ t', (|t' + 1| + t'^2) ≥ (|t + 1| + t^2) ∧ (|t + 1| + t^2) = 3 / 4 := sorry

end NUMINAMATH_GPT_min_rectilinear_distance_to_parabola_l2346_234625


namespace NUMINAMATH_GPT_percentage_25_of_200_l2346_234604

def percentage_of (percent : ℝ) (amount : ℝ) : ℝ := percent * amount

theorem percentage_25_of_200 :
  percentage_of 0.25 200 = 50 :=
by sorry

end NUMINAMATH_GPT_percentage_25_of_200_l2346_234604


namespace NUMINAMATH_GPT_find_num_boys_l2346_234682

-- Definitions for conditions
def num_children : ℕ := 13
def num_girls (num_boys : ℕ) : ℕ := num_children - num_boys

-- We will assume we have a predicate representing the truthfulness of statements.
-- boys tell the truth to boys and lie to girls
-- girls tell the truth to girls and lie to boys

theorem find_num_boys (boys_truth_to_boys : Prop) 
                      (boys_lie_to_girls : Prop) 
                      (girls_truth_to_girls : Prop) 
                      (girls_lie_to_boys : Prop)
                      (alternating_statements : Prop) : 
  ∃ (num_boys : ℕ), num_boys = 7 := 
  sorry

end NUMINAMATH_GPT_find_num_boys_l2346_234682


namespace NUMINAMATH_GPT_complement_union_l2346_234627

open Finset

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 4}
def N : Finset ℕ := {2, 4}

theorem complement_union :
  U \ (M ∪ N) = {1, 3} := by
  sorry

end NUMINAMATH_GPT_complement_union_l2346_234627


namespace NUMINAMATH_GPT_solve_system_l2346_234645

def system_of_equations : Prop :=
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ x + 2 * y = -2 ∧ x = 2 ∧ y = -2

theorem solve_system : system_of_equations := by
  sorry

end NUMINAMATH_GPT_solve_system_l2346_234645


namespace NUMINAMATH_GPT_systematic_sampling_starts_with_srs_l2346_234603

-- Define the concept of systematic sampling
def systematically_sampled (initial_sampled: Bool) : Bool :=
  initial_sampled

-- Initial sample is determined by simple random sampling
def simple_random_sampling : Bool :=
  True

-- We need to prove that systematic sampling uses simple random sampling at the start
theorem systematic_sampling_starts_with_srs : systematically_sampled simple_random_sampling = True :=
by 
  sorry

end NUMINAMATH_GPT_systematic_sampling_starts_with_srs_l2346_234603


namespace NUMINAMATH_GPT_expectation_of_transformed_binomial_l2346_234661

def binomial_expectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

def linear_property_of_expectation (a b : ℚ) (E_ξ : ℚ) : ℚ :=
  a * E_ξ + b

theorem expectation_of_transformed_binomial (ξ : ℚ) :
  ξ = binomial_expectation 5 (2/5) →
  linear_property_of_expectation 5 2 ξ = 12 :=
by
  intros h
  rw [h]
  unfold linear_property_of_expectation binomial_expectation
  sorry

end NUMINAMATH_GPT_expectation_of_transformed_binomial_l2346_234661


namespace NUMINAMATH_GPT_water_volume_per_minute_l2346_234699

theorem water_volume_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ) 
  (H_depth : depth = 5) 
  (H_width : width = 35) 
  (H_flow_rate_kmph : flow_rate_kmph = 2) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 5832.75 :=
by
  sorry

end NUMINAMATH_GPT_water_volume_per_minute_l2346_234699


namespace NUMINAMATH_GPT_minimum_students_in_class_l2346_234611

def min_number_of_students (b g : ℕ) : ℕ :=
  b + g

theorem minimum_students_in_class
  (b g : ℕ)
  (h1 : b = 2 * g / 3)
  (h2 : ∃ k : ℕ, g = 3 * k)
  (h3 : ∃ k : ℕ, 1 / 2 < (2 / 3) * g / b) :
  min_number_of_students b g = 5 :=
sorry

end NUMINAMATH_GPT_minimum_students_in_class_l2346_234611


namespace NUMINAMATH_GPT_find_f_2013_l2346_234654

noncomputable def f : ℝ → ℝ := sorry
axiom functional_eq : ∀ (m n : ℝ), f (m + n^2) = f m + 2 * (f n)^2
axiom f_1_ne_0 : f 1 ≠ 0

theorem find_f_2013 : f 2013 = 4024 * (f 1)^2 + f 1 :=
sorry

end NUMINAMATH_GPT_find_f_2013_l2346_234654


namespace NUMINAMATH_GPT_total_selling_price_correct_l2346_234656

def original_price : ℝ := 100
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.08

theorem total_selling_price_correct :
  let discount := original_price * discount_percent
  let sale_price := original_price - discount
  let tax := sale_price * tax_percent
  let total_selling_price := sale_price + tax
  total_selling_price = 75.6 := by
sorry

end NUMINAMATH_GPT_total_selling_price_correct_l2346_234656


namespace NUMINAMATH_GPT_initial_paper_count_l2346_234693

theorem initial_paper_count (used left initial : ℕ) (h_used : used = 156) (h_left : left = 744) :
  initial = used + left :=
sorry

end NUMINAMATH_GPT_initial_paper_count_l2346_234693


namespace NUMINAMATH_GPT_odd_factors_of_360_l2346_234671

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end NUMINAMATH_GPT_odd_factors_of_360_l2346_234671


namespace NUMINAMATH_GPT_prob_even_sum_is_one_third_l2346_234637

def is_even_sum_first_last (d1 d2 d3 d4 : Nat) : Prop :=
  (d1 + d4) % 2 = 0

def num_unique_arrangements : Nat := 12

def num_favorable_arrangements : Nat := 4

def prob_even_sum_first_last : Rat :=
  num_favorable_arrangements / num_unique_arrangements

theorem prob_even_sum_is_one_third :
  prob_even_sum_first_last = 1 / 3 := 
  sorry

end NUMINAMATH_GPT_prob_even_sum_is_one_third_l2346_234637


namespace NUMINAMATH_GPT_range_of_m_l2346_234676

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + m + 8 ≥ 0) ↔ (-8 / 9 ≤ m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2346_234676


namespace NUMINAMATH_GPT_students_opted_both_math_science_l2346_234678

def total_students : ℕ := 40
def not_opted_math : ℕ := 10
def not_opted_science : ℕ := 15
def not_opted_either : ℕ := 2

theorem students_opted_both_math_science :
  let T := total_students
  let M' := not_opted_math
  let S' := not_opted_science
  let E := not_opted_either
  let B := (T - M') + (T - S') - (T - E)
  B = 17 :=
by
  sorry

end NUMINAMATH_GPT_students_opted_both_math_science_l2346_234678


namespace NUMINAMATH_GPT_impossible_to_color_25_cells_l2346_234633

theorem impossible_to_color_25_cells :
  ¬ ∃ (n : ℕ) (n_k : ℕ → ℕ), n = 25 ∧ (∀ k, k > 0 → k < 5 → (k % 2 = 1 → ∃ c : ℕ, n_k c = k)) :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_color_25_cells_l2346_234633


namespace NUMINAMATH_GPT_points_per_bag_l2346_234698

/-
Wendy had 11 bags but didn't recycle 2 of them. She would have earned 
45 points for recycling all 11 bags. Prove that Wendy earns 5 points 
per bag of cans she recycles.
-/

def total_bags : Nat := 11
def unrecycled_bags : Nat := 2
def recycled_bags : Nat := total_bags - unrecycled_bags
def total_points : Nat := 45

theorem points_per_bag : total_points / recycled_bags = 5 := by
  sorry

end NUMINAMATH_GPT_points_per_bag_l2346_234698


namespace NUMINAMATH_GPT_find_m_l2346_234614

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A based on the condition in the problem
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5 * x + m = 0}

-- Define the complement of A in the universal set U
def complementA (m : ℕ) : Set ℕ := U \ A m

-- Given condition that the complement of A in U is {2, 3}
def complementA_condition : Set ℕ := {2, 3}

-- The proof problem statement: Prove that m = 4 given the conditions
theorem find_m (m : ℕ) (h : complementA m = complementA_condition) : m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_l2346_234614


namespace NUMINAMATH_GPT_complete_the_square_l2346_234659

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end NUMINAMATH_GPT_complete_the_square_l2346_234659


namespace NUMINAMATH_GPT_mabel_counts_sharks_l2346_234616

theorem mabel_counts_sharks 
    (fish_day1 : ℕ) 
    (fish_day2 : ℕ) 
    (shark_percentage : ℚ) 
    (total_fish : ℕ) 
    (total_sharks : ℕ) 
    (h1 : fish_day1 = 15) 
    (h2 : fish_day2 = 3 * fish_day1) 
    (h3 : shark_percentage = 0.25) 
    (h4 : total_fish = fish_day1 + fish_day2) 
    (h5 : total_sharks = total_fish * shark_percentage) : 
    total_sharks = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_mabel_counts_sharks_l2346_234616


namespace NUMINAMATH_GPT_Xiaoli_estimate_is_larger_l2346_234644

variables {x y x' y' : ℝ}

theorem Xiaoli_estimate_is_larger (h1 : x > y) (h2 : y > 0) (h3 : x' = 1.01 * x) (h4 : y' = 0.99 * y) : x' - y' > x - y :=
by sorry

end NUMINAMATH_GPT_Xiaoli_estimate_is_larger_l2346_234644


namespace NUMINAMATH_GPT_test_scores_order_l2346_234624

def kaleana_score : ℕ := 75

variable (M Q S : ℕ)

-- Assuming conditions from the problem
axiom h1 : Q = kaleana_score
axiom h2 : M < max Q S
axiom h3 : S > min Q M
axiom h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S

-- Theorem statement
theorem test_scores_order (M Q S : ℕ) (h1 : Q = kaleana_score) (h2 : M < max Q S) (h3 : S > min Q M) (h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S) :
  M < Q ∧ Q < S :=
sorry

end NUMINAMATH_GPT_test_scores_order_l2346_234624


namespace NUMINAMATH_GPT_solve_for_x_l2346_234613

theorem solve_for_x (x : ℝ) (h : (5 * x - 3) / (6 * x - 6) = (4 / 3)) : x = 5 / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2346_234613


namespace NUMINAMATH_GPT_find_y_l2346_234608

theorem find_y (y : ℕ) (h : (2 * y) / 5 = 10) : y = 25 :=
sorry

end NUMINAMATH_GPT_find_y_l2346_234608


namespace NUMINAMATH_GPT_find_cos_squared_y_l2346_234610

noncomputable def α : ℝ := Real.arccos (-3 / 7)

def arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

def transformed_arithmetic_progression (a b c : ℝ) : Prop :=
  14 / Real.cos b = 1 / Real.cos a + 1 / Real.cos c

theorem find_cos_squared_y (x y z : ℝ)
  (h1 : arithmetic_progression x y z)
  (h2 : transformed_arithmetic_progression x y z)
  (hα : 2 * α = z - x) : Real.cos y ^ 2 = 10 / 13 :=
by
  sorry

end NUMINAMATH_GPT_find_cos_squared_y_l2346_234610


namespace NUMINAMATH_GPT_linear_function_graph_not_in_second_quadrant_l2346_234601

open Real

theorem linear_function_graph_not_in_second_quadrant 
  (k b : ℝ) (h1 : k > 0) (h2 : b < 0) :
  ¬ ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b := 
sorry

end NUMINAMATH_GPT_linear_function_graph_not_in_second_quadrant_l2346_234601


namespace NUMINAMATH_GPT_jane_crayon_count_l2346_234657

def billy_crayons : ℝ := 62.0
def total_crayons : ℝ := 114
def jane_crayons : ℝ := total_crayons - billy_crayons

theorem jane_crayon_count : jane_crayons = 52 := by
  unfold jane_crayons
  show total_crayons - billy_crayons = 52
  sorry

end NUMINAMATH_GPT_jane_crayon_count_l2346_234657


namespace NUMINAMATH_GPT_grayson_time_per_answer_l2346_234620

variable (totalQuestions : ℕ) (unansweredQuestions : ℕ) (totalTimeHours : ℕ)

def timePerAnswer (totalQuestions : ℕ) (unansweredQuestions : ℕ) (totalTimeHours : ℕ) : ℕ :=
  let answeredQuestions := totalQuestions - unansweredQuestions
  let totalTimeMinutes := totalTimeHours * 60
  totalTimeMinutes / answeredQuestions

theorem grayson_time_per_answer :
  totalQuestions = 100 →
  unansweredQuestions = 40 →
  totalTimeHours = 2 →
  timePerAnswer totalQuestions unansweredQuestions totalTimeHours = 2 :=
by
  intros hTotal hUnanswered hTime
  rw [hTotal, hUnanswered, hTime]
  sorry

end NUMINAMATH_GPT_grayson_time_per_answer_l2346_234620


namespace NUMINAMATH_GPT_carlos_improved_lap_time_l2346_234615

-- Define the initial condition using a function to denote time per lap initially
def initial_lap_time : ℕ := (45 * 60) / 15

-- Define the later condition using a function to denote time per lap later on
def current_lap_time : ℕ := (42 * 60) / 18

-- Define the proof that calculates the improvement in seconds
theorem carlos_improved_lap_time : initial_lap_time - current_lap_time = 40 := by
  sorry

end NUMINAMATH_GPT_carlos_improved_lap_time_l2346_234615


namespace NUMINAMATH_GPT_g_is_even_l2346_234646

noncomputable def g (x : ℝ) : ℝ := 4^(x^2 - 3) - 2 * |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end NUMINAMATH_GPT_g_is_even_l2346_234646


namespace NUMINAMATH_GPT_total_chocolates_distributed_l2346_234660

theorem total_chocolates_distributed 
  (boys girls : ℕ)
  (chocolates_per_boy chocolates_per_girl : ℕ)
  (h_boys : boys = 60)
  (h_girls : girls = 60)
  (h_chocolates_per_boy : chocolates_per_boy = 2)
  (h_chocolates_per_girl : chocolates_per_girl = 3) : 
  boys * chocolates_per_boy + girls * chocolates_per_girl = 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_chocolates_distributed_l2346_234660


namespace NUMINAMATH_GPT_part_one_part_two_l2346_234697

-- Part (1)
theorem part_one (x : ℝ) : x - (3 * x - 1) ≤ 2 * x + 3 → x ≥ -1 / 2 :=
by sorry

-- Part (2)
theorem part_two (x : ℝ) : 
  (3 * (x - 1) < 4 * x - 2) ∧ ((1 + 4 * x) / 3 > x - 1) → x > -1 :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l2346_234697


namespace NUMINAMATH_GPT_hypotenuse_length_l2346_234634

variable (a b c : ℝ)

-- Given conditions
theorem hypotenuse_length (h1 : b = 3 * a) 
                          (h2 : a^2 + b^2 + c^2 = 500) 
                          (h3 : c^2 = a^2 + b^2) : 
                          c = 5 * Real.sqrt 10 := 
by 
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2346_234634


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l2346_234641

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l2346_234641


namespace NUMINAMATH_GPT_proportionality_cube_and_fourth_root_l2346_234662

variables (x y z : ℝ) (k j m n : ℝ)

theorem proportionality_cube_and_fourth_root (h1 : x = k * y^3) (h2 : y = j * z^(1/4)) : 
  ∃ m : ℝ, ∃ n : ℝ, x = m * z^n ∧ n = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_proportionality_cube_and_fourth_root_l2346_234662


namespace NUMINAMATH_GPT_total_length_proof_l2346_234630

def length_of_first_tape : ℝ := 25
def overlap : ℝ := 3
def number_of_tapes : ℝ := 64

def total_tape_length : ℝ :=
  let effective_length_per_subsequent_tape := length_of_first_tape - overlap
  let length_of_remaining_tapes := effective_length_per_subsequent_tape * (number_of_tapes - 1)
  length_of_first_tape + length_of_remaining_tapes

theorem total_length_proof : total_tape_length = 1411 := by
  sorry

end NUMINAMATH_GPT_total_length_proof_l2346_234630


namespace NUMINAMATH_GPT_swimming_speed_eq_l2346_234672

theorem swimming_speed_eq (S R H : ℝ) (h1 : R = 9) (h2 : H = 5) (h3 : H = (2 * S * R) / (S + R)) :
  S = 45 / 13 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_eq_l2346_234672


namespace NUMINAMATH_GPT_trig_identity_one_trig_identity_two_l2346_234642

theorem trig_identity_one :
  2 * (Real.cos (45 * Real.pi / 180)) - (3 / 2) * (Real.tan (30 * Real.pi / 180)) * (Real.cos (30 * Real.pi / 180)) + (Real.sin (60 * Real.pi / 180))^2 = Real.sqrt 2 :=
sorry

theorem trig_identity_two :
  (Real.sin (30 * Real.pi / 180))⁻¹ * (Real.sin (60 * Real.pi / 180) - Real.cos (45 * Real.pi / 180)) - Real.sqrt ((1 - Real.tan (60 * Real.pi / 180))^2) = 1 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_trig_identity_one_trig_identity_two_l2346_234642


namespace NUMINAMATH_GPT_derek_initial_lunch_cost_l2346_234651

-- Definitions based on conditions
def derek_initial_money : ℕ := 40
def derek_dad_lunch_cost : ℕ := 11
def derek_more_lunch_cost : ℕ := 5
def dave_initial_money : ℕ := 50
def dave_mom_lunch_cost : ℕ := 7
def dave_difference : ℕ := 33

-- Variable X to represent Derek's initial lunch cost
variable (X : ℕ)

-- Definitions based on conditions
def derek_total_spending (X : ℕ) := X + derek_dad_lunch_cost + derek_more_lunch_cost
def derek_remaining_money (X : ℕ) := derek_initial_money - derek_total_spending X
def dave_remaining_money := dave_initial_money - dave_mom_lunch_cost

-- The main theorem to prove Derek spent $14 initially
theorem derek_initial_lunch_cost (h : dave_remaining_money = derek_remaining_money X + dave_difference) : X = 14 := by
  sorry

end NUMINAMATH_GPT_derek_initial_lunch_cost_l2346_234651


namespace NUMINAMATH_GPT_k_value_and_set_exists_l2346_234658

theorem k_value_and_set_exists
  (x1 x2 x3 x4 : ℚ)
  (h1 : (x1 + x2) / (x3 + x4) = -1)
  (h2 : (x1 + x3) / (x2 + x4) = -1)
  (h3 : (x1 + x4) / (x2 + x3) = -1)
  (hne : x1 ≠ x2 ∨ x1 ≠ x3 ∨ x1 ≠ x4 ∨ x2 ≠ x3 ∨ x2 ≠ x4 ∨ x3 ≠ x4) :
  ∃ (A B C : ℚ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ x1 = A ∧ x2 = B ∧ x3 = C ∧ x4 = -A - B - C := 
sorry

end NUMINAMATH_GPT_k_value_and_set_exists_l2346_234658


namespace NUMINAMATH_GPT_min_value_a_plus_3b_plus_9c_l2346_234691

theorem min_value_a_plus_3b_plus_9c {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a + 3*b + 9*c ≥ 27 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_3b_plus_9c_l2346_234691


namespace NUMINAMATH_GPT_relationship_between_abc_l2346_234600

theorem relationship_between_abc 
  (a b c : ℝ) 
  (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) 
  (ha : Real.exp a = 9 * a * Real.log 11)
  (hb : Real.exp b = 10 * b * Real.log 10)
  (hc : Real.exp c = 11 * c * Real.log 9) : 
  a < b ∧ b < c :=
sorry

end NUMINAMATH_GPT_relationship_between_abc_l2346_234600


namespace NUMINAMATH_GPT_lateral_surface_area_eq_total_surface_area_eq_l2346_234688

def r := 3
def h := 10

theorem lateral_surface_area_eq : 2 * Real.pi * r * h = 60 * Real.pi := by
  sorry

theorem total_surface_area_eq : 2 * Real.pi * r * h + 2 * Real.pi * r^2 = 78 * Real.pi := by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_eq_total_surface_area_eq_l2346_234688


namespace NUMINAMATH_GPT_perfect_square_n_l2346_234684

open Nat

theorem perfect_square_n (n : ℕ) : 
  (∃ k : ℕ, 2 ^ (n + 1) * n = k ^ 2) ↔ 
  (∃ m : ℕ, n = 2 * m ^ 2) ∨ (∃ odd_k : ℕ, n = odd_k ^ 2 ∧ odd_k % 2 = 1) := 
sorry

end NUMINAMATH_GPT_perfect_square_n_l2346_234684


namespace NUMINAMATH_GPT_problem_l2346_234638

theorem problem (x : ℝ) (h : 15 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 17 := 
by sorry

end NUMINAMATH_GPT_problem_l2346_234638


namespace NUMINAMATH_GPT_alice_unanswered_questions_l2346_234618

theorem alice_unanswered_questions 
    (c w u : ℕ)
    (h1 : 6 * c - 2 * w + 3 * u = 120)
    (h2 : 3 * c - w = 70)
    (h3 : c + w + u = 40) :
    u = 10 :=
sorry

end NUMINAMATH_GPT_alice_unanswered_questions_l2346_234618


namespace NUMINAMATH_GPT_line_eq_l2346_234647

-- Conditions
def circle_eq (x y : ℝ) (a : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 5 - a

def midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  2*xm = x1 + x2 ∧ 2*ym = y1 + y2

-- Theorem statement
theorem line_eq (a : ℝ) (h : a < 3) :
  circle_eq 0 1 a →
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

end NUMINAMATH_GPT_line_eq_l2346_234647


namespace NUMINAMATH_GPT_find_x_l2346_234670

theorem find_x (x : ℝ) (h : x^29 * 4^15 = 2 * 10^29) : x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l2346_234670


namespace NUMINAMATH_GPT_camila_weeks_needed_l2346_234667

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end NUMINAMATH_GPT_camila_weeks_needed_l2346_234667


namespace NUMINAMATH_GPT_sam_drove_200_miles_l2346_234605

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end NUMINAMATH_GPT_sam_drove_200_miles_l2346_234605


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2346_234679

theorem simplify_and_evaluate_expression (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
    (x ≠ 1) → (x ≠ -1) → (x ≠ 0) → 
    ((x / (x + 1) - (3 * x) / (x - 1)) / (x / (x^2 - 1))) = -8 :=
by 
  intro h3 h4 h5
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2346_234679


namespace NUMINAMATH_GPT_foreman_can_establish_corr_foreman_cannot_with_less_l2346_234619

-- Define the given conditions:
def num_rooms (n : ℕ) := 2^n
def num_checks (n : ℕ) := 2 * n

-- Part (a)
theorem foreman_can_establish_corr (n : ℕ) : 
  ∃ (c : ℕ), c = num_checks n ∧ (c ≥ 2 * n) :=
by
  sorry

-- Part (b)
theorem foreman_cannot_with_less (n : ℕ) : 
  ¬ (∃ (c : ℕ), c = 2 * n - 1 ∧ (c < 2 * n)) :=
by
  sorry

end NUMINAMATH_GPT_foreman_can_establish_corr_foreman_cannot_with_less_l2346_234619


namespace NUMINAMATH_GPT_tree_height_by_time_boy_is_36_inches_l2346_234617

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end NUMINAMATH_GPT_tree_height_by_time_boy_is_36_inches_l2346_234617


namespace NUMINAMATH_GPT_arrange_numbers_in_ascending_order_l2346_234664

noncomputable def S := 222 ^ 2
noncomputable def T := 22 ^ 22
noncomputable def U := 2 ^ 222
noncomputable def V := 22 ^ (2 ^ 2)
noncomputable def W := 2 ^ (22 ^ 2)
noncomputable def X := 2 ^ (2 ^ 22)
noncomputable def Y := 2 ^ (2 ^ (2 ^ 2))

theorem arrange_numbers_in_ascending_order :
  S < Y ∧ Y < V ∧ V < T ∧ T < U ∧ U < W ∧ W < X :=
sorry

end NUMINAMATH_GPT_arrange_numbers_in_ascending_order_l2346_234664


namespace NUMINAMATH_GPT_M_sufficient_not_necessary_for_N_l2346_234639

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem M_sufficient_not_necessary_for_N (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → ¬ (a ∈ M)) :=
sorry

end NUMINAMATH_GPT_M_sufficient_not_necessary_for_N_l2346_234639


namespace NUMINAMATH_GPT_coefficients_sum_l2346_234681

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) ^ 4

theorem coefficients_sum : 
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  ((2 * x - 1) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) 
  ∧ (a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ = 8) :=
sorry

end NUMINAMATH_GPT_coefficients_sum_l2346_234681


namespace NUMINAMATH_GPT_largest_y_coordinate_l2346_234655

theorem largest_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y = 2 := 
by 
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_l2346_234655


namespace NUMINAMATH_GPT_homework_problems_l2346_234687

noncomputable def problems_solved (p t : ℕ) : ℕ := p * t

theorem homework_problems (p t : ℕ) (h_eq: p * t = (3 * p - 5) * (t - 3))
  (h_pos_p: p > 0) (h_pos_t: t > 0) (h_p_ge_15: p ≥ 15) 
  (h_friend_did_20: (3 * p - 5) * (t - 3) ≥ 20) : 
  problems_solved p t = 100 :=
by
  sorry

end NUMINAMATH_GPT_homework_problems_l2346_234687


namespace NUMINAMATH_GPT_remainder_of_83_div_9_l2346_234606

theorem remainder_of_83_div_9 : ∃ r : ℕ, 83 = 9 * 9 + r ∧ r = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_83_div_9_l2346_234606
